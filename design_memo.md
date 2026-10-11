# Design Memo: Policy Grounded Agent Workflow Prototype

## 1. Which replies are grounded

| Turn | Decision | Rule | Evidence | Grounded? |
|---|---|---|---|---|
| 14:02 open dispute | ANSWER | R1 | KB-1001 | Fully: next steps (verify identity, identify line item). |
| 14:04, 14:06 details | NO_QUESTION | none | none | Not applicable; facts stored in case memory. |
| 14:09 "$75 limit?" | ANSWER | R3 over R2 (R0) | KB-1004 | Fully: $225 ÷ 30 × 11 = $82.50, not capped by $75, no R4(b) escalation. R5 is **unverified** (account history not in transcript), so the credit is stated as pending that check. |
| 14:11 taxes | PARTIAL | R3 | KB-1004 | Partly: "plan charge only" is grounded; tax and surcharge treatment is **not covered** by any article, so that part escalates to Tier 2. |
| 14:13 deadline | ANSWER | R3 | KB-1004, KB-1005 | Fully: 30 days from resolution = Oct 21, 2026; the R1 60 day window is rejected. |

Out of scope questions (retention credit, installments) escalate. Unrecognized intent always escalates.

## 2. Hardest point in the policy hierarchy

The R3 / R2 / R4(a) boundary. An $82.50 outage credit *looks* like it breaches the $75 cap. The policy requires choosing the governing rule **before** checking thresholds, and forbids blending rules. Policy selection is deterministic code: R5 first, R3 over R2 only when ticket, duration and timing all hold, and an R3 credit is never compared to $75. Even so, scenario tests caught the agent blending rules: when R3 failed, it still sized the R2 credit with the outage formula. Fixed: R2 quotes no amount, since its $75 ceiling is cumulative over 12 months and credit history is unverified.

A related difficulty was **coverage versus relevance**: KB-1003 is the top hit for the tax question but never answers it. The gate checks whether an article states the required claim, not its score.

## 3. Architecture and evidence

LangGraph agents: policy (deterministic) → router (OKF frontmatter) → intent (LLM) → retrieval (BM25 on routed articles) → gate (claim grounding) → responder (LLM) → verifier (deterministic). Routing cuts context from ~2,807 to ~295 to ~1,390 tokens per turn. The verifier rejects unsupported numbers, dates or amounts, dropped escalations, and citations outside the sentence stating the claim, then falls back to the template. A circuit breaker drops to deterministic mode on any LLM failure.



- **Live Groq** (gpt-oss-120b): LangSmith 9 of 9 on decision, rule and verifier pass. A rule override corrected an LLM misclassification at 14:06.
- **8 rule boundary scenarios** pass (no ticket, exactly 4 hours, day 31, $301 credit, 31 day outage, unverifiable ticket, repeat dispute, fraud).
- **Paraphrases:** rules 7 of 9 (misses escalate safely); LLM 9 of 9.
- **Verifier catches:** Unicode spacing false positives (fixed by normalisation); gpt-oss moving citations away from claims on all three 14:13 rewordings (correctly rejected, template used).
- **LangSmith, live Groq, 26 cases** (transcript, out of scope, 8 scenarios, 9 paraphrases):
  all outputs match expected decision, rule and cited articles (24 scored, 2 unscored due to an
  evaluator error). The verifier rejected 3 of 21 LLM replies for misplaced citations and served
  the template instead.

## 4. Next step: measuring adherence and hallucination

1. Grow the eval set to ~100 labeled turns from real conversations: paraphrases, rule boundaries, multi turn cases, uncovered questions.
2. Track per release: rule accuracy, decision accuracy, evidence accuracy, unsupported claim rate, escalation precision and recall, verifier precision, and LLM fallback rate.
3. Run in LangSmith on every policy, KB or prompt change; block deploys on regression.
4. Add an LLM judge for semantic faithfulness (the verifier checks values and citation placement, not meaning), calibrated on human labels.
5. Known gaps: cross check customer dates against NOC ticket dates; tighten the responder prompt to keep citations in place.

*Data note: the KB export says 9 articles but contains 10.*