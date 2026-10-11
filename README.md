# Policy-Grounded Billing Dispute Agent

A prototype billing-dispute agent for Cresta Agentic Assist. It answers a customer only when a knowledge-base article explicitly supports the answer, and escalates to Tier 2 otherwise.
It is a LangGraph multi-agent workflow. Policy selection is always deterministic, and LLM steps are optional (`USE_LLM=1`).

**Requirements:** Python 3.10+. Install with `pip install -r requirements.txt`, then `cp .env.example .env`. With `USE_LLM=0` (the default) it needs no keys.

**Run:** `python run.py` prints a trace for each turn and writes `trace.json`.
**Tests:** `python -m unittest discover tests` (no keys needed; LLMs are faked).
**LangSmith eval:** `python scripts/eval_langsmith.py` (runs only if `LANGSMITH_API_KEY` is set).

## Agents (graph nodes)
`policy_agent` → `router_agent` → `intent_agent` → `retrieval_agent` → `gate_agent` → `responder_agent` → `verifier_agent`. Out-of-scope or unroutable turns, and `unknown` intents, go to `escalate_agent` → `verifier_agent`.
- **policy_agent**: the deterministic resolver, with no LLM. Checks R5 first, and R3 over R2 via R0. It runs first because it needs only case facts, not the KB.
- **router_agent**: scores KB categories from frontmatter (`category`, `scope`, `governs`, `tags`). Out of scope escalates with no LLM call. Only an ambiguous match calls the small model (`ROUTER_MODEL`, Haiku 5.5). An in-scope route also gets the category governing the case's current rule. Later agents see only the routed articles and the rules they govern.
- **intent_agent**: Claude picks from the fixed intent list or `unknown` (a JSON-schema enum). Falls back to keyword rules.
- **retrieval_agent**: BM25 over the routed articles only.
- **gate_agent**: grounds each claim in a retrieved article and builds the template reply as claim-tagged segments.
- **responder_agent**: Claude rephrases the segments using only the cited evidence and computed facts.
- **verifier_agent**: checks every number, date and $ amount against the evidence or computed facts, and checks that every segment maps to a claim, keeps its citation and keeps any Tier 2 escalation. On failure the template reply is used instead.

`CaseMemory` sits in the graph state and carries facts, checks and per-turn decisions (with routed categories) across turns.

## Layout
```
graph.py       LangGraph workflow (nodes, routing)     memory.py     CaseMemory
llm.py         Claude calls (intent, responder)        verifier.py   reply verification
facts.py       transcript facts                        retriever.py  scope filter + BM25
resolver.py    rule selection, credit, escalation      gate.py       intents, grounding, templates
loader.py      frontmatter parser                      config.py     .env settings
router.py      frontmatter category routing
run.py         runs the transcript, prints the trace   eval_cases.py shared eval cases
data/kb, data/policy   one article / rule per file     data/raw      original sources
data/eval_cases.json   expected decision + rule        scripts/      LangSmith eval
```

## Reading the trace
Each turn shows the route (categories, keywords or LLM, any category added from the case rule), the token count (routed context vs the full KB, plus LLM input/output tokens), the intent (and whether rules or the LLM picked it), retrieved articles with scores, the governing rule, the checks (in full once, then only what changed), the claims, the verifier result and the reply with its source (`template`, `llm` or `template (verifier fallback)`).

| Decision | Meaning |
|---|---|
| `ANSWER` | Every required claim is grounded in a retrieved article. |
| `PARTIAL` | The grounded claims are answered and the rest are escalated to Tier 2. |
| `ESCALATE` | Nothing is grounded, the intent is unknown, or R5 applies. Goes to Tier 2. |
| `NO_QUESTION` | The turn only provides details (account, dates), so there is nothing to answer. |
