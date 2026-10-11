# Process Notes

Built with Claude (planning, policy analysis, review) and Codex (implementation). I directed the design, reviewed every change against the policy map, and tested the agent manually.

## How I worked
1. Mapped the policy hierarchy first: R3 vs R2 vs R4(a) for the credit, the uncovered tax question, R3's 30 day deadline vs R1's 60 days, and R5 as an unverified override.
2. Made policy selection and the refusal gate deterministic, using LLMs only where language understanding adds value.
3. Built a deterministic baseline (tagged v1-baseline), then layered on agents, LLMs, routing and evaluation, keeping it runnable at every step.
4. Tested beyond the transcript: rule boundary scenarios, paraphrases, out of scope questions, and manual questions through ask.py.

## Design decisions I made
- **OKF style knowledge:** one file per KB article and policy rule, with YAML frontmatter (scope, governs, tags), so routing filters by metadata before retrieval or any LLM call.
- **Metadata router to cut tokens:** only routed articles and their governing rule reach the LLM.
- **Multi LLM with guardrails:** separate models per agent, a provider switch (Groq / Anthropic) and a circuit breaker. LLMs propose; code decides.
- **Case memory:** facts, checks and decisions carried across turns.
- **Rule override:** when a turn has no question, rules override the LLM's intent.
- **Honest test failures:** paraphrase misses are marked as known failures rather than patched with keywords.

## Where the AI was corrected or caught something
- Codex flagged that the policy preamble ("the more specific rule governs; never blend rules") was dropped during the file split. Kept as R0.
- The first 14:02 reply quoted the 60 day window, which she never asked about. Changed to KB-1001's next steps.
- "Plan charge only" was answered even when not asked. Codex scoped that claim to questions that mention it.
- Routing on the question alone dropped KB-1004 for the 14:11 tax question. Fixed by running policy first and adding the case rule's category.
- Live Groq misclassified 14:06 as open_dispute. A rule override now corrects it.
- The verifier rejected a correct Groq reply over Unicode hyphens and spaces. Added normalisation.
- **Scenario tests caught rule blending:** once R3 failed, the agent still sized the R2 credit with the outage formula ($82.50, or $0.00 at exactly 4 hours). Fixed: R2 quotes no amount and escalates.
- For an unverifiable ticket, the amount is labeled an estimate pending Tier 2 verification.
- Added a sentence level citation check; it correctly rejected gpt-oss replies that moved citations away from their claims.
- Manual testing: "Can I get $100 back?" got $82.50 but never addressed $100. **[IF DONE]** Added a requested amount slot. **[IF NOT]** Logged as a known gap.

## Not built (deliberately)
- Per dispute tracks for customers with two disputes under different rules; documented as the next design step.
- A paid classifier for routing (cost) and response caching; listed as next steps.