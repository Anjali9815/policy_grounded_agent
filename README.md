# Policy-Grounded Billing Dispute Agent

A prototype billing-dispute agent for Cresta Agentic Assist. It answers a customer only when a knowledge-base article explicitly supports the answer, and escalates to Tier 2 otherwise.
It picks the governing policy rule deterministically and does not need an LLM to run.

**Requirements:** Python 3.9+, no dependencies.

**Run:** `python run.py` prints a trace for each turn and writes `trace.json`.
**Tests:** `python -m unittest discover tests`

## Pipeline
1. **facts**: extracts case facts from the transcript, noting each fact's source and the time it became known. Facts the transcript doesn't contain (dispute history, fraud flags) are marked `unverified` and never assumed.
2. **retriever**: drops out-of-scope articles (`scope: not_billing_dispute`), then ranks the rest with BM25.
3. **resolver**: checks R5 (repeat dispute / fraud) first, and R5 overrides everything. If every R3 outage condition holds, R3 governs instead of R2 (citing R0). An R3 credit escalates only under R4(b) and is never compared to the $75 cap.
4. **gate**: each question maps to an intent with required claims. A claim is answered only if a retrieved article states it. Questions that match no intent escalate.
5. **reply**: templates filled only from quoted evidence and computed values, with citations such as `[KB-1004]` and `[R0, R1]`.

## Layout
```
loader.py      frontmatter parser for data/kb and data/policy
facts.py       transcript parsing and fact extraction
retriever.py   scope filter + BM25
resolver.py    rule selection, checks, credit, escalation
gate.py        intents, claim grounding, reply templates
run.py         walks customer turns, prints trace, writes trace.json
data/kb/       KB-1001..KB-1010 (one article per file)
data/policy/   R0_precedence, R1..R5 (one rule per file)
data/raw/      original policy, KB export and chat transcript
tests/         expected outcomes and unseen-question tests
```

## Reading the trace
Each customer turn shows the question, intent, retrieved articles with scores, the governing rule, the checks and the claims. The checks block prints in full once; later turns show only the checks that changed. Each check is `passed`, `failed` or `unverified`, and `not applied` lines explain rules that were deliberately set aside.

| Decision | Meaning |
|---|---|
| `ANSWER` | Every required claim is grounded in a retrieved article. |
| `PARTIAL` | The grounded claims are answered and the rest are escalated to Tier 2. |
| `ESCALATE` | Nothing is grounded, the intent is unknown, or R5 applies. Goes to Tier 2. |
| `NO_QUESTION` | The turn only provides details (account, dates), so there is nothing to answer. |
