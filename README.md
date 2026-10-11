# Policy-Grounded Billing Dispute Agent

A prototype billing-dispute agent for Cresta Agentic Assist. It answers a customer only when a knowledge-base article explicitly supports the answer, and escalates to Tier 2 otherwise.
It is a LangGraph multi-agent workflow. Policy selection is always deterministic, and LLM steps are optional (`USE_LLM=1`, with `LLM_PROVIDER=anthropic` or `groq`).

**Requirements:** Python 3.10+. Install with `pip install -r requirements.txt`, then `cp .env.example .env`. With `USE_LLM=0` (the default) it needs no keys.

**Run:** `python run.py` prints which provider/model each agent uses, then a trace for each turn. It writes `traces/trace_deterministic.json` (`USE_LLM=0`) or `traces/trace_<provider>.json` (`USE_LLM=1`).
**Ask:** `python ask.py` is an interactive loop. It replays the transcript to load the case memory (with no LLM calls), then for each question prints the route, governing rule, cited articles, decision and reply. `--fresh` starts with empty case memory, so only the account header is known. `--no-llm` forces deterministic mode. Type `quit` to exit.
**Tests:** `python -m unittest discover tests` (no keys needed; LLMs are faked).
**Paraphrases:** `python scripts/run_paraphrases.py` runs the paraphrase cases and writes `traces/paraphrases_<provider>.json`.
**LangSmith eval:** `python scripts/eval_langsmith.py` (runs only if `LANGSMITH_API_KEY` is set; use `--recreate` after changing `data/eval_cases.json`).

## Agents (graph nodes)
`policy_agent` → `router_agent` → `intent_agent` → `retrieval_agent` → `gate_agent` → `responder_agent` → `verifier_agent`. Out-of-scope or unroutable turns, and `unknown` intents, go to `escalate_agent` → `verifier_agent`.
- **policy_agent**: the deterministic resolver, with no LLM. Checks R5 first, and R3 over R2 via R0. It runs first because it needs only case facts, not the KB.
- **router_agent**: scores KB categories from frontmatter (`category`, `scope`, `governs`, `tags`). Out of scope escalates with no LLM call. Only an ambiguous match calls the small model (`ROUTER_MODEL`). An in-scope route also gets the category governing the case's current rule. Later agents see only the routed articles and the rules they govern.
- **intent_agent**: the LLM picks from the fixed intent list or `unknown` (a JSON-schema enum). Falls back to keyword rules.
- **retrieval_agent**: BM25 over the routed articles only.
- **gate_agent**: grounds each claim in a retrieved article and builds the template reply as claim-tagged segments.
- **responder_agent**: the LLM rephrases the segments using only the cited evidence and computed facts.
- **verifier_agent**: checks every number, date and $ amount against the evidence or computed facts. It also checks that every segment maps to a claim, keeps any Tier 2 escalation, and cites its article in the sentence that states the claim. That sentence is the one sharing the most words with the evidence quote; a citation moved elsewhere fails. On failure the template reply is used instead.

**LLM providers:** `LLM_PROVIDER` chooses between Anthropic (JSON-schema structured outputs) and Groq (JSON mode, with the schema in the prompt). `RESPONDER_MODEL` runs intent and responder, and `ROUTER_MODEL` runs ambiguous routing. Every provider's output goes through the same enum/schema validation and the same verifier. **Circuit breaker:** the first LLM failure (API error, bad JSON, invented enum value) disables every LLM for the rest of the run, logs one line, and the run continues deterministically. A verifier rejection is not an LLM failure, so it doesn't trip the breaker.

`CaseMemory` sits in the graph state and carries facts, checks and per-turn decisions (with routed categories) across turns.

## Layout
```
graph.py       LangGraph workflow (nodes, routing)     memory.py     CaseMemory
llm.py         LLM providers, validation, breaker      verifier.py   reply verification
facts.py       transcript facts                        retriever.py  scope filter + BM25
resolver.py    rule selection, credit, escalation      gate.py       intents, grounding, templates
loader.py      frontmatter parser                      config.py     .env settings
router.py      frontmatter category routing
run.py         runs the transcript, prints the trace   eval_cases.py shared eval cases
ask.py         interactive questions on the case       traces/       saved runs
data/kb, data/policy   one article / rule per file     data/raw      original sources
data/eval_cases.json   expected decision, rule, articles   scripts/  LangSmith eval, paraphrase run
```

## Reading the trace
Each turn shows the route (categories, keywords or LLM, any category added from the case rule), the token count (routed context vs the full KB, plus LLM input/output tokens), the intent (and whether rules or the LLM picked it), retrieved articles with scores, the governing rule, the checks (in full once, then only what changed), the claims, the verifier result and the reply with its source (`template`, `llm` or `template (verifier fallback)`).

| Decision | Meaning |
|---|---|
| `ANSWER` | Every required claim is grounded in a retrieved article. |
| `PARTIAL` | The grounded claims are answered and the rest are escalated to Tier 2. |
| `ESCALATE` | Nothing is grounded, the intent is unknown, or R5 applies. Goes to Tier 2. |
| `NO_QUESTION` | The turn only provides details (account, dates), so there is nothing to answer. |

## Eval cases and results
`data/eval_cases.json` holds 26 cases. Each has an expected decision, governing rule and cited KB articles.
- **Transcript turns (6) and unseen questions (3):** all pass in deterministic mode.
- **Scenarios (8):** each overrides case facts for the 14:09 credit question. All pass in deterministic mode.

| Scenario | Decision | Rule | Articles |
|---|---|---|---|
| No NOC ticket | PARTIAL | R2 | KB-1002 |
| Outage exactly 4 hours | PARTIAL | R2 | KB-1002, KB-1004 |
| Request on day 31 after resolution | PARTIAL | R2 | KB-1002 |
| Credit over $300 ($301.03) | PARTIAL | R3 | KB-1004, KB-1006 |
| Outage of 31 days | PARTIAL | R3 | KB-1004, KB-1006 |
| Ticket exists but can't be verified | PARTIAL | R3 | KB-1004, KB-1006 |
| Prior resolved dispute within 90 days | ESCALATE | R5 | KB-1007 |
| Fraud flag | ESCALATE | R5 | KB-1007 |

Under R2 no amount is quoted: the $75 ceiling is cumulative and the credit history isn't in the transcript, so the amount goes to Tier 2. Under R3, R4(b) escalations quote the computed credit and say why it needs approval.

**Paraphrases (9):** casual, terse and typo rewordings of the 14:09, 14:11 and 14:13 questions, each expecting the original's outcome.
- **Deterministic: 7 of 9.** The keyword rules miss "75 bucks max" (no `$75` or `limit`) and "how lnog" (typo of `how long`). Both are marked as expected failures in `tests/test_paraphrases.py` and escalate rather than answer wrongly.
- **Groq (`gpt-oss-120b`), `traces/paraphrases_groq.json`: 9 of 9** reach the right decision, rule and articles. On all three 14:13 rewordings the LLM moved the deadline's citation to the end of the paragraph, so the verifier rejected the reply and the template was used.
