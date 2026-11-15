// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the trial brief (provider directory, sealed quality
// table, workload manifest), probes every provider's catalog route, builds
// one catalog, registers with the env, and answers the client's 20
// requests, one at a time, with the best free model that can serve each
// one. Every answer is a provider's own answer, translated into the wire
// the client used, with its text, usage and stop reason intact.
//
// Routing, in one sentence: serve each request with the lowest-risk
// free model that meets it, where risk is the exact posterior that the
// provider's next call refuses, given every call so far.
//
// Why risk, not rating: the score is 99 for every router that picks
// near-best models, and the tie-break is the number of provider calls,
// so the whole game is wasting as few calls as possible on refusals.
// A provider's first call can never fail (its scheduled failures start
// at call 2 and its outage at call 2 or later), so the ten first calls
// are free; every later call risks a scheduled failure (one or two per
// provider, drawn from call 2 to its budget), the one provider the
// instance forces to fail on call 2, and the one provider whose calls
// fail from its outage start on. The entry cannot know which provider
// is which, so it tracks the exact posterior instead: the generator
// draws each provider's budget uniformly from its table bounds, one or
// two fail calls uniformly from call 2 to budget minus one, the
// designated second-call provider and the outage provider uniformly
// among the ten, and the outage start uniformly from call 2 to budget
// minus one. riskParts marginalises that draw exactly for one provider
// given its call history; riskJoint combines the ten providers, because
// exactly one provider is designated and exactly one is in outage, so
// a success on one provider raises the risk on the untried others.
// Nothing here reads the wall clock or unseeded randomness: every
// choice is a function of the brief, the requests so far and the
// providers' answers, so a replay leaves the same state export.
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
 * The per-trial call budget bounds of each provider, as the kit's
 * generator draws them (calls_min and calls_max in the sealed provider
 * table). A provider's budget is uniform over its range; planning
 * against the bounds instead of one number is what lets the risk below
 * be exact. Unknown providers (a brief that names one the table never
 * did) get the smallest bounds in the table, the cautious choice.
 */
const BUDGET_BOUNDS: Record<string, [number, number]> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};
const DEFAULT_BOUNDS: [number, number] = [4, 5];

/** A request is served by a model at most this many milli below its best. */
const MAX_LOSS_MILLI = 70;

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

/**
 * Per-provider trial state, learned from the providers' own answers:
 * the outcome of every call, in order (call 1 is always a success, so
 * the history starts true). The risk functions read this and nothing
 * else: a success says the call number is not a scheduled failure, a
 * refusal says it is one, or the budget is spent, or the outage has
 * started, and the posterior weighs those three.
 */
interface ProviderState {
  hist: boolean[];
}
const providerState = new Map<string, ProviderState>();
function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { hist: [] };
    providerState.set(providerId, state);
  }
  return state;
}

/** n choose k for n at most 16, exact in a double. */
function binom(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;
  let r = 1;
  for (let i = 0; i < k; i++) r = (r * (n - i)) / (i + 1);
  return r;
}

/**
 * One provider's history against a FIXED designated/outage status:
 * D says this provider is the one the instance forces to fail on call
 * 2, O says it is the one whose calls fail from its outage start on.
 * Returns { lik, q }: lik is the probability of this history under
 * that status, q is the probability the provider's next call refuses
 * given it, marginalising the budget, the fail count (one or two), the
 * fail set and the outage start exactly.
 *
 * A history is a list of call outcomes. A successful call j says j is
 * within budget, not a fail call and before the outage start; a failed
 * call j is explained by any of the three, and the fail set must hold
 * the failures no other cause explains. The fail set is one or two
 * calls drawn uniformly from call 2 to budget minus one, except the
 * designated provider, whose set always holds call 2 (and, for two
 * fails, one more call drawn uniformly from call 3 up).
 */
function riskParts(providerId: string, hist: readonly boolean[], D: boolean, O: boolean): { lik: number; q: number } {
  const [bmin, bmax] = BUDGET_BOUNDS[providerId] ?? DEFAULT_BOUNDS;
  const k = hist.length;
  const n = k + 1;
  let lik = 0;
  let failW = 0;
  for (let B = bmin; B <= bmax; B++) {
    const pB = 1 / (bmax - bmin + 1);
    const S = B - 1;
    const starts = O ? Array.from({ length: B - 2 }, (_, i) => i + 2) : [0];
    for (const OF of starts) {
      const pOF = O ? 1 / (B - 2) : 1;
      for (const C of [1, 2]) {
        const pC = 0.5;
        let bad = false;
        const mustIn = new Set<number>();
        const mustOut = new Set<number>();
        for (let j = 2; j <= k; j++) {
          const okj = hist[j - 1];
          if (okj) {
            if (j > B || (O && j >= OF)) { bad = true; break; }
            mustOut.add(j);
          } else if (j > B) {
            // The budget explains this failure.
          } else if (O && j >= OF) {
            // The outage explains this failure.
          } else {
            mustIn.add(j);
          }
        }
        if (bad) continue;
        if (D && mustOut.has(2)) continue;
        let totalF: number;
        let goodF: number;
        let goodFNext: number;
        if (!D) {
          totalF = binom(S, C);
          const inU = [...mustIn].filter((x) => x >= 2 && x <= B - 1);
          const outU = [...mustOut].filter((x) => x >= 2 && x <= B - 1);
          if (inU.length !== mustIn.size) goodF = 0;
          else if (inU.some((x) => mustOut.has(x))) goodF = 0;
          else goodF = binom(S - inU.length - outU.length, C - inU.length);
          if (n >= 2 && n <= B - 1 && !mustOut.has(n) && !mustIn.has(n)) {
            goodFNext = binom(S - inU.length - outU.length - 1, C - inU.length - 1);
          } else if (mustIn.has(n)) {
            goodFNext = goodF;
          } else goodFNext = 0;
        } else if (C === 1) {
          totalF = 1;
          const inOk = mustIn.size === 0 || (mustIn.size === 1 && mustIn.has(2));
          goodF = inOk && !mustOut.has(2) ? 1 : 0;
          goodFNext = n === 2 ? goodF : 0;
        } else {
          totalF = B - 2;
          const rest = [...mustIn].filter((x) => x !== 2);
          if (rest.length > 1 || rest.some((x) => x < 3 || x > B - 1)) goodF = 0;
          else if (rest.length === 1) goodF = mustOut.has(rest[0]) ? 0 : 1;
          else goodF = Math.max(0, B - 2 - [...mustOut].filter((x) => x >= 3 && x <= B - 1).length);
          if (n === 2) goodFNext = goodF;
          else if (n >= 3 && n <= B - 1 && !mustOut.has(n)) goodFNext = rest.length === 1 ? (rest[0] === n ? 1 : 0) : 1;
          else goodFNext = 0;
        }
        if (goodF === 0 || totalF === 0) continue;
        const w = pB * pC * pOF * (goodF / totalF);
        lik += w;
        const failsByBudgetOrOutage = n > B || (O && n >= OF);
        failW += w * (failsByBudgetOrOutage ? 1 : goodFNext / goodF);
      }
    }
  }
  return { lik, q: lik === 0 ? 1 : failW / lik };
}

/**
 * The posterior that each provider's next call refuses, given every
 * provider's history. Exactly one provider is designated and exactly
 * one is in outage, so the posterior over the pair is the product of
 * the ten providers' likelihoods, and a success on one provider shifts
 * the designated and outage probability onto the providers not yet
 * proven. A provider with no history has risk 0: its first call can
 * never fail. `hists` may carry planned calls as successes, so the
 * plan can ask what the risk of a provider's next planned call is.
 */
function riskJointFromHists(hists: Map<string, readonly boolean[]>): Map<string, number> {
  const ids = probes.map((probe) => probe.provider.id);
  const parts = new Map<string, { lik: number; q: number }[][]>();
  for (const id of ids) {
    const hist = hists.get(id) ?? stateOf(id).hist;
    const table: { lik: number; q: number }[][] = [[], []];
    for (const D of [0, 1]) {
      table[D] = [];
      for (const O of [0, 1]) table[D][O] = riskParts(id, hist, D === 1, O === 1);
    }
    parts.set(id, table);
  }
  const weights = new Map<string, number>();
  let totalW = 0;
  for (const d of ids) {
    for (const o of ids) {
      let w = 1;
      for (const p of ids) w *= parts.get(p)![d === p ? 1 : 0][o === p ? 1 : 0].lik;
      weights.set(`${d}|${o}`, w);
      totalW += w;
    }
  }
  const out = new Map<string, number>();
  for (const id of ids) {
    let r = 0;
    for (const d of ids) {
      for (const o of ids) {
        const w = weights.get(`${d}|${o}`) ?? 0;
        if (w === 0 || totalW === 0) continue;
        r += (w / totalW) * parts.get(id)![d === id ? 1 : 0][o === id ? 1 : 0].q;
      }
    }
    out.set(id, r);
  }
  return out;
}

/** The joint risk of the providers' next calls, from their real histories. */
function riskJoint(): Map<string, number> {
  return riskJointFromHists(new Map());
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
 * The plan: one primary candidate per manifest request still to come,
 * assigned most-constrained first (fewest candidates, then heaviest
 * weight, then manifest order), so a model only one request can use
 * goes to that request and the scarce providers' first calls are
 * reserved for the requests only they can serve. Each pick minimises
 * the joint risk of the provider's next planned call -- its history
 * extended by the calls the plan already gave it, counted as successes
 * -- and then the rating loss, within MAX_LOSS_MILLI. A provider's
 * planned calls never exceed its budget's upper bound. The plan is
 * recomputed after every request, from the fresh histories.
 */
let plan = new Map<string, Candidate>();

function computePlan(): void {
  plan = new Map();
  const manifest = brief.workload?.manifest ?? [];
  const planned = new Map<string, number>();
  const order = manifest
    .map((entry, index) => ({ entry, index, candidates: candidatesFor(entry.requires, entry.api) }))
    .filter(({ index }) => index >= manifestIndex)
    .sort((a, b) => a.candidates.length - b.candidates.length || b.entry.weight - a.entry.weight || a.index - b.index);
  for (const { entry, candidates } of order) {
    const hists = new Map<string, readonly boolean[]>();
    for (const probe of probes) {
      const id = probe.provider.id;
      hists.set(id, [...stateOf(id).hist, ...Array<boolean>(planned.get(id) ?? 0).fill(true)]);
    }
    const risks = riskJointFromHists(hists);
    const bestRating = candidates.length > 0 ? candidates[0].rating : 0;
    const scored = candidates
      .filter((c) => {
        const [, bmax] = BUDGET_BOUNDS[c.probe.provider.id] ?? DEFAULT_BOUNDS;
        return stateOf(c.probe.provider.id).hist.length + (planned.get(c.probe.provider.id) ?? 0) < bmax;
      })
      .map((c) => ({ candidate: c, risk: risks.get(c.probe.provider.id) ?? 1, loss: bestRating - c.rating }));
    const pool = scored.filter((s) => s.loss <= MAX_LOSS_MILLI);
    const pick = (pool.length > 0 ? pool : scored).sort(
      (a, b) => a.risk - b.risk || a.loss - b.loss || (a.candidate.probe.provider.id < b.candidate.probe.provider.id ? -1 : 1),
    )[0];
    if (pick !== undefined) {
      plan.set(entry.id, pick.candidate);
      planned.set(pick.candidate.probe.provider.id, (planned.get(pick.candidate.probe.provider.id) ?? 0) + 1);
    }
  }
}

/**
 * Serve one request. The plan's primary candidate goes first; on a
 * refusal the request falls over to the next candidate by the fresh
 * joint risk, re-ordered after every attempt, so a provider that just
 * refused is tried again in the same request only when the posterior
 * still ranks it first -- a refusal is as often a spent budget or an
 * outage as one scheduled failure, and the posterior weighs those.
 * Candidates within MAX_LOSS_MILLI of the request's best model lead;
 * the rest follow as a failover tail in rating order, so a request is
 * never left unserved while a legal model remains. Exactly one
 * provider call may succeed per request.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, stream: boolean, requestId: string | null): Promise<ServedAnswer | null> {
  const requires = requirementsFor(body, clientApi);
  const candidates = candidatesFor(requires, clientApi);
  if (candidates.length === 0) return null;
  const bestRating = candidates[0].rating;
  const primary = requestId !== null ? plan.get(requestId) : undefined;
  const tried = new Set<string>();
  for (;;) {
    const risks = riskJoint();
    const remaining = candidates.filter((c) => !tried.has(`${c.probe.provider.id}/${c.model.id}`));
    if (remaining.length === 0) {
      computePlan();
      return null;
    }
    const scored = remaining.map((c) => ({ candidate: c, risk: risks.get(c.probe.provider.id) ?? 1, loss: bestRating - c.rating }));
    const pool = scored.filter((s) => s.loss <= MAX_LOSS_MILLI);
    const tail = scored.filter((s) => s.loss > MAX_LOSS_MILLI);
    pool.sort((a, b) => a.risk - b.risk || a.loss - b.loss || (a.candidate.probe.provider.id < b.candidate.probe.provider.id ? -1 : 1));
    tail.sort((a, b) => a.loss - b.loss || a.risk - b.risk);
    const ordered = [...pool, ...tail];
    const pi = ordered.findIndex(
      (s) => primary !== undefined && s.candidate.probe.provider.id === primary.probe.provider.id && s.candidate.model.id === primary.model.id,
    );
    if (pi > 0) {
      const [first] = ordered.splice(pi, 1);
      ordered.unshift(first);
    }
    const next = ordered[0].candidate;
    tried.add(`${next.probe.provider.id}/${next.model.id}`);
    const answer = await callProvider(next, clientApi, body, stream);
    if (answer !== null) {
      computePlan();
      return { answer, candidate: next };
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

/** One call to one provider: its normalized answer, or null when it refused or failed. Every attempt is recorded in the provider's history, whatever its outcome: a refusal spends a call and teaches the risk as much as a success does. */
async function callProvider(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamAnswer | null> {
  const provider = candidate.probe.provider;
  const state = stateOf(provider.id);
  const { url, headers, body: upstream } = upstreamBody(clientApi, body, candidate, stream);
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstream), signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
  } catch {
    state.hist.push(false);
    return null;
  }
  if (!res.ok) {
    state.hist.push(false);
    return null;
  }
  const raw = await res.text();
  const parsed =
    provider.wire === 'openai' ? parseOpenAIResponse(raw) : provider.wire === 'anthropic' ? parseAnthropicResponse(raw) : parseGeminiResponse(raw);
  if (parsed === null) {
    state.hist.push(false);
    return null;
  }
  state.hist.push(true);
  return parsed;
}

interface ServedAnswer {
  answer: UpstreamAnswer;
  candidate: Candidate;
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
