"""Run the paraphrase eval cases through the agent and save the results.

Writes traces/paraphrases_<provider>.json (or paraphrases_deterministic.json with
USE_LLM=0): per case, the expected and actual decision, governing rule and cited
articles, the intent and its source, the reply and its source, the verifier
result (with any rejected LLM reply) and LLM token usage.

Usage: python scripts/run_paraphrases.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval_cases import load_cases  # noqa: E402
from run import TRACES, Agent, header, trace_path  # noqa: E402


def main():
    agent = Agent()
    print("\n".join(header(agent)))
    rows = []
    for c in (c for c in load_cases() if c["paraphrase_of"]):
        t = agent.answer(**c["inputs"])
        got = {"decision": t["decision"], "governing_rule": t["governing_rule"],
               "cited_articles": t["cited_articles"]}
        rows.append({
            "id": c["id"], "paraphrase_of": c["paraphrase_of"], "question": c["inputs"]["question"],
            "pass": got == c["outputs"], "expected": c["outputs"], **got,
            "intent": t["intent"], "intent_source": t["intent_source"],
            "reply": t["reply"], "reply_source": t["reply_source"],
            "verifier_passed": t["verifier"]["passed"], "verifier_errors": t["verifier"]["errors"],
            "rejected_llm_reply": (" ".join(seg["text"] for seg in t["verifier"]["llm_attempt"]["segments"])
                                   if "llm_attempt" in t["verifier"] else None),
            "llm_calls": len(t["tokens"]["llm_calls"]),
            "llm_tokens": t["tokens"]["llm_input"] + t["tokens"]["llm_output"],
            "llm_breaker": t["llm_breaker"],
        })
        r = rows[-1]
        print(f"{'PASS' if r['pass'] else 'FAIL'} {r['id']:<24} intent={r['intent']} ({r['intent_source']}) "
              f"{r['decision']} {r['governing_rule']} {r['cited_articles']} reply={r['reply_source']} "
              f"verifier={'PASS' if r['verifier_passed'] else 'FAIL'}")
        for e in r["verifier_errors"]:
            print(f"    ! {e}")
    out = TRACES / trace_path(agent).name.replace("trace_", "paraphrases_")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"run": header(agent), "passed": sum(r["pass"] for r in rows),
                               "total": len(rows), "cases": rows},
                              indent=2, default=str, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{sum(r['pass'] for r in rows)}/{len(rows)} match the original outcome; wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
