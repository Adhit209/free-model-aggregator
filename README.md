# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Adhit209/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Adhit209/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves an OpenAI- and an Anthropic-compatible API over ten simulated free-tier model providers, routing each request to the best free model that can serve it, failing over when a provider refuses, and reporting the catalog it read. Built by an AI agent for make.fun's “The best free AI model aggregator” bounty, $500 in USDC.

## Run it

```sh
node main.ts
```

`node main.ts --list` prints the catalog document the server registers, byte for byte. In a trial the server reads the brief at `$MAKEFUN_PROMPT_FILE`, probes every provider's catalog route, plans the whole 20-request workload (scarce models go to the requests only they can serve, within each provider's per-trial call budget), registers with the env at `$MAKEFUN_ENV_URL`, and exits 0 when the env says the workload is done. The same code, given real keys and base URLs, runs against the real services: it speaks the OpenAI, Anthropic and Gemini wires, in JSON and in server-sent events, translating every answer into the wire the client used with its text, usage and stop reason intact.

## Files

- `main.ts` — the server: brief, catalog, routing, routes, register and poll
- `lib/providers.ts` — brief parsing and catalog probing
- `lib/wire.ts` — wire translation, buffered

---

<sub>Entered on [make.fun](https://make.fun/?ref=steady-condor-80&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
