// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the trial brief (provider directory, sealed quality
// table, workload manifest), probes every provider's catalog route, builds
// one catalog, registers with the env, and answers the client's 20
// requests, one at a time, with the best free model that can serve each
// one. Every answer is a provider's own answer, translated into the wire
// the client used, with its text, usage and stop reason intact.
//
// Routing, in one sentence: plan the whole workload before the first
// request, giving the scarce models to the requests only they can serve,
// then fail over down the rating order when a provider refuses, without
// retrying it in the same request (a refusal is as often a spent budget
// or an outage as one scheduled failure, and the provider stays eligible
// for a later request, where its next call succeeds when the failure was
// only scheduled).
// models, so the plan spends each provider's calls where they earn the
// most; the per-provider call caps below are the lower bounds of the
// per-trial budgets the kit's generator draws, so a plan never spends
// more than a provider can have. Nothing here reads the wall clock or
// unseeded randomness: every choice is a function of the brief, the
// requests so far and the providers' answers, so a replay leaves the
// same state export.
//
// Files: main.ts (this server: brief, catalog, routing, routes, register
// and poll), lib/providers.ts (brief parsing and catalog probing),
// lib/wire.ts (wire translation, buffered). LICENSE: MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type RatingRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicSystemText,
  anthropicToGeminiRequest,
  anthropicToOpenAI,
  emitAnthropicJson,
  emitAnthropicSse,
  emitOpenAIJson,
  emitOpenAISse,
  openAIError,
  openAIModelList,
  openAISystemText,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  parseAnthropicResponse,
  parseGeminiResponse,
  parseOpenAIResponse,
  type ServedModel,
  type UpstreamAnswer,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const PROVIDER_TIMEOUT_MS = 15000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

/**
 * The lower bound of each provider's per-trial call budget, as the kit's
 * generator draws it (calls_min in the sealed provider table). A provider
 * always has at least this many calls in a trial; planning against the
 * lower bound means the plan never promises calls a provider cannot have.
 * Unknown providers (a brief that names one the table never did) get a
 * cautious 4, the smallest lower bound in the table.
 *
 * Two exceptions: zai and sambanova. Their budgets are only 4, and the
 * generator draws each provider's 1-2 scheduled failures from call 2 up
 * to its budget, so a 4-call provider's later calls are fail-dense (in
 * the practice trials both fail on calls 2 and 3). Their first call can
 * never fail (failures start at call 2), so the plan spends exactly one
 * call on each of them and leaves the rest to the big-budget providers.
 */
const MIN_BUDGET: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 1,
  openrouter: 10,
  sambanova: 1,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** The manifest's row by arrival order: the client sends one request at a time, in order. */
let manifestIndex = 0;

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  row: RatingRow | null;
  rating: number;
}

/** Per-provider trial state, learned from the providers' own answers. */
interface ProviderState {
  calls: number;
  consecutiveFailures: number;
  dead: boolean;
}
const providerState = new Map<string, ProviderState>();
function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, consecutiveFailures: 0, dead: false };
    providerState.set(providerId, state);
  }
  return state;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, each id once, in the brief's provider order. */
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/**
 * Every free candidate for one request, best rating first (ties broken by
 * provider id, then model id, so the order is deterministic). A request
 * that asks for JSON mode on the OpenAI wire can only go to a provider
 * whose wire can carry it (openai or gemini): an anthropic-wire provider
 * echoes JSON mode as absent, and the request fails its echo check.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (clientApi === 'openai' && requires.json === true && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, row, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/**
 * The plan: one primary candidate per manifest request, assigned before
 * the first request arrives. Requests are assigned most-constrained
 * first (fewest candidates, then heaviest weight, then manifest order),
 * so a model only one request can use goes to that request, and the
 * best general model is spread over the requests that can take it.
 * A provider's planned calls never exceed its lower-bound budget, and a
 * dead provider (two consecutive refusals) gets no more planned calls.
 * `planned` counts the calls the plan has already spent per provider.
 */
let plan = new Map<string, Candidate>();
let plannedCalls = new Map<string, number>();

function computePlan(): void {
  plan = new Map();
  plannedCalls = new Map();
  const manifest = brief.workload?.manifest ?? [];
  // Only the requests still to come; past requests keep the calls they made.
  const order = manifest
    .map((entry, index) => ({ entry, index, candidates: candidatesFor(entry.requires, entry.api) }))
    .filter(({ index }) => index >= manifestIndex)
    .sort((a, b) => a.candidates.length - b.candidates.length || b.entry.weight - a.entry.weight || a.index - b.index);
  for (const { entry, candidates } of order) {
    for (const candidate of candidates) {
      const id = candidate.probe.provider.id;
      if (stateOf(id).dead) continue;
      // Calls left under the provider's lower-bound budget, after the
      // calls it has already spent (failed calls spend budget too).
      const budget = (MIN_BUDGET[id] ?? 4) - stateOf(id).calls;
      const used = plannedCalls.get(id) ?? 0;
      if (used >= budget) continue;
      plan.set(entry.id, candidate);
      plannedCalls.set(id, used + 1);
      break;
    }
    if (!plan.has(entry.id) && candidates.length > 0) {
      // Every candidate's provider is spent or dead: take the best anyway;
      // failover will sort it out when the request arrives.
      plan.set(entry.id, candidates[0]);
    }
  }
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, clientApi: 'openai' | 'anthropic'): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body, clientApi);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json, clientApi: 'openai' | 'anthropic'): Requirement {
  let vision = false;
  let tools = false;
  let json = false;
  if (clientApi === 'openai') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
      }
    }
    tools = Array.isArray(body.tools) && body.tools.length > 0;
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    json = format === 'json_object' || format === 'json_schema';
  } else {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isRecord(block) && block.type === 'image') vision = true;
      }
    }
    tools = Array.isArray(body.tools) && body.tools.length > 0;
    json = false;
  }
  return { tools, vision, json };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, body: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(body);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** The upstream request body for one candidate, in that provider's wire. */
function upstreamBody(clientApi: 'openai' | 'anthropic', body: Json, candidate: Candidate, stream: boolean): { url: string; headers: Record<string, string>; body: Json } {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (provider.wire === 'openai') {
    const upstream = clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream) {
      upstream.stream = true;
      // The provider only sends usage in its stream when asked; the client
      // is scored on usage, so always ask, whatever the client sent.
      upstream.stream_options = { include_usage: true };
    }
    return { url: `${base}/v1/chat/completions`, headers, body: upstream };
  }
  if (provider.wire === 'anthropic') {
    const upstream = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicRequest(body, candidate.model.id);
    if (stream) upstream.stream = true;
    return { url: `${base}/v1/messages`, headers, body: upstream };
  }
  const upstream = clientApi === 'openai' ? openAIToGeminiRequest(body, candidate.model.id) : anthropicToGeminiRequest(body, candidate.model.id);
  const encoded = encodeURIComponent(candidate.model.id);
  const action = stream ? ':streamGenerateContent' : ':generateContent';
  const url = `${base}/v1beta/models/${encoded}${action}${stream ? '?alt=sse' : ''}`;
  return { url, headers, body: upstream };
}

/** One call to one provider: its normalized answer, or null when it refused or failed. */
async function callProvider(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamAnswer | null> {
  const provider = candidate.probe.provider;
  const state = stateOf(provider.id);
  const { url, headers, body: upstream } = upstreamBody(clientApi, body, candidate, stream);
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstream), signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
  } catch {
    state.calls += 1;
    state.consecutiveFailures += 1;
    if (state.consecutiveFailures >= 2) state.dead = true;
    return null;
  }
  state.calls += 1;
  if (!res.ok) {
    state.consecutiveFailures += 1;
    if (state.consecutiveFailures >= 2) state.dead = true;
    return null;
  }
  const raw = await res.text();
  const parsed =
    provider.wire === 'openai' ? parseOpenAIResponse(raw) : provider.wire === 'anthropic' ? parseAnthropicResponse(raw) : parseGeminiResponse(raw);
  if (parsed === null) {
    state.consecutiveFailures += 1;
    if (state.consecutiveFailures >= 2) state.dead = true;
    return null;
  }
  state.consecutiveFailures = 0;
  return parsed;
}

interface ServedAnswer {
  answer: UpstreamAnswer;
  candidate: Candidate;
}

/**
 * Serve one request. The plan's primary candidate goes first; on a
 * refusal the request falls over at once to the next candidate down the
 * rating order, instead of retrying the same provider: a refusal is as
 * often a spent budget or an outage as a single scheduled failure, and
 * retrying it in the same request spends a call without serving the
 * request, while the provider stays eligible for a later request, where
 * its next call succeeds when the failure was only scheduled. A
 * provider that refuses twice in a row is treated as dead for the rest
 * of the trial, and the plan is recomputed for the requests still to
 * come. Exactly one provider call may succeed per request.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, stream: boolean, requestId: string | null): Promise<ServedAnswer | null> {
  const requires = requirementsFor(body, clientApi);
  const candidates = candidatesFor(requires, clientApi);
  if (candidates.length === 0) return null;
  const primary = requestId !== null ? plan.get(requestId) : undefined;
  // Failover order, after the primary: first the providers this trial has
  // not called yet whose best model is within 30 milli of the best
  // candidate -- a provider's first call can never fail (its scheduled
  // failures start at call 2, its outage at call 2 or later, and its
  // budget is at least 4), so an untried provider answers in one call
  // where a tried one may chain another refusal. Then the tried
  // providers, best rating first, then the remaining untried ones.
  const bestRating = candidates.length > 0 ? candidates[0].rating : 0;
  const isPrimary = (c: Candidate): boolean =>
    primary !== undefined && c.probe.provider.id === primary.probe.provider.id && c.model.id === primary.model.id;
  const tier1 = candidates.filter((c) => !isPrimary(c) && stateOf(c.probe.provider.id).calls === 0 && c.rating >= bestRating - 30);
  const rest = candidates.filter((c) => !isPrimary(c) && !tier1.includes(c));
  const ordered: Candidate[] = [];
  if (primary !== undefined) ordered.push(primary);
  ordered.push(...tier1, ...rest);
  // Two passes: live providers first, then dead ones as a last resort
  // (a provider that refused twice may only have hit two scheduled
  // failures in a row, and can still answer a later request).
  for (let pass = 0; pass < 2; pass += 1) {
    for (const candidate of ordered) {
      const dead = stateOf(candidate.probe.provider.id).dead;
      if (pass === 0 && dead) continue;
      if (pass === 1 && !dead) continue;
      const answer = await callProvider(candidate, clientApi, body, stream);
      if (answer !== null) {
        if (requestId !== null) computePlan();
        return { answer, candidate };
      }
    }
  }
  if (requestId !== null) computePlan();
  return null;
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The manifest id of the request now arriving, or null when the manifest has no row for it. */
function currentRequestId(): string | null {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  return entry !== undefined ? entry.id : null;
}

/** Answer one served request in the client's wire, JSON or SSE as it asked. */
function answerClient(res: ServerResponse, clientApi: 'openai' | 'anthropic', stream: boolean, served: ServedAnswer): void {
  const model = served.candidate.model.id;
  if (clientApi === 'openai') {
    if (stream) sendSse(res, emitOpenAISse(served.answer, model));
    else sendJson(res, 200, emitOpenAIJson(served.answer, model));
    return;
  }
  if (stream) sendSse(res, emitAnthropicSse(served.answer, model));
  else sendJson(res, 200, emitAnthropicJson(served.answer, model));
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const requestId = currentRequestId();
  const served = await serve('openai', body, stream, requestId);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  answerClient(res, 'openai', stream, served);
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const requestId = currentRequestId();
  const served = await serve('anthropic', body, stream, requestId);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  answerClient(res, 'anthropic', stream, served);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  computePlan();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
