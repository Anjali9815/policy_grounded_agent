"""Walk the transcript's customer turns through the agent graph; print and trace each turn."""
import json
from pathlib import Path

import facts as facts_mod
import gate
import graph
import llm
from loader import load_kb, load_policy
from memory import CaseMemory
from router import OUT_OF_SCOPE_VALUE

TRACE = Path(__file__).resolve().parent / "trace.json"


class Agent:
    def __init__(self, transcript=None, use_llm=None, llm_client=None, router_llm_client=None):
        self.kb, self.policy = load_kb(), load_policy()
        self.excluded = sorted(i for i, d in self.kb.items() if d.get("scope") == OUT_OF_SCOPE_VALUE)
        self.text = transcript or facts_mod.read_transcript()
        self.facts = facts_mod.extract(self.text)
        self.llm = llm_client if llm_client is not None else llm.make(use_llm)
        if router_llm_client is None and llm_client is None:
            router_llm_client = llm.make_router(use_llm)
        self.router_llm = router_llm_client
        self.graph = graph.build(self.kb, self.policy, self.facts, self.llm, self.router_llm)
        self.memory = CaseMemory()
        self.full_context_tokens = graph.context_tokens(self.kb) + graph.context_tokens(self.policy)

    def _usage_logs(self):
        clients = {id(c): c for c in (self.llm, self.router_llm) if c is not None}
        return [getattr(c, "usage_log", []) for c in clients.values()]

    def answer(self, question, time="23:59"):
        """Run one customer message through the graph using the facts known at chat time 'HH:MM'."""
        marks = [len(log) for log in self._usage_logs()]
        s = self.graph.invoke({"question": question, "time": time, "memory": self.memory})
        calls = [c for log, m in zip(self._usage_logs(), marks) for c in log[m:]]
        resolution, result, route = s["resolution"], s["gate"], s["route"]
        governing = graph.governing_for(s["intent"], resolution)
        return {
            "time": time, "question": question, "intent": s["intent"], "intent_source": s["intent_source"],
            "route": {"status": route["status"], "categories": route["categories"],
                      "source": route["source"], "reason": route["reason"],
                      "case_context": route.get("case_context"), "scores": route["scores"],
                      "articles": sorted(s["routed_kb"]), "rules": sorted(s["routed_rules"])},
            "tokens": {
                "routed_context": graph.context_tokens(s["routed_kb"]) + graph.context_tokens(s["routed_rules"]),
                "full_context": self.full_context_tokens,
                "llm_input": sum(c["input_tokens"] for c in calls),
                "llm_output": sum(c["output_tokens"] for c in calls),
                "llm_calls": calls,
            },
            "retrieved": s["retrieved"], "excluded_by_scope": self.excluded,
            "governing_rule": governing,
            "governing_rule_title": self.policy[governing]["title"] if governing else None,
            "citations": resolution["citations"] if governing == resolution["governing_rule"] else [governing] if governing else [],
            "checks": resolution["checks"], "not_applied": resolution["not_applied"],
            "checks_view": self.memory.check_view(resolution) if s["intent"] in gate.INTENTS else None,
            "credit": resolution["credit"], "formula": resolution["formula"],
            "request_deadline": resolution["request_deadline"],
            "pending_verification": resolution["pending_verification"],
            "escalation_reasons": resolution["escalation_reasons"],
            "claims": result["claims"], "decision": result["decision"], "escalate": result["escalate"],
            "segments": s["segments"], "reply": " ".join(seg["text"] for seg in s["segments"]),
            "reply_source": s["reply_source"], "verifier": s["verifier"],
        }

    def run(self):
        return [self.answer(t["text"], t["time"]) for t in facts_mod.customer_turns(self.text)]


def run(transcript=None, use_llm=None):
    return Agent(transcript, use_llm).run()


def show(trace):
    for t in trace:
        print(f"\n=== [{t['time']}] {t['question']}")
        r, tok = t["route"], t["tokens"]
        added = r["case_context"]["added"] if r["case_context"] else []
        matched = [c for c in r["categories"] if c not in added]
        print(f"route:     {r['status']} -> {', '.join(matched) or '-'} ({r['source']})"
              + (f"; + {', '.join(added)} (case rule {r['case_context']['rule']})" if added else ""))
        print(f"tokens:    routed context ~{tok['routed_context']} of ~{tok['full_context']} "
              f"(articles {', '.join(r['articles']) or '-'}; rules {', '.join(r['rules']) or '-'}); "
              f"llm {tok['llm_input']} in / {tok['llm_output']} out over {len(tok['llm_calls'])} calls")
        print(f"intent:    {t['intent']} ({t['intent_source']})")
        print("retrieved: " + (", ".join(f"{r['id']} ({r['score']})" for r in t["retrieved"]) or "-"))
        print(f"governing: {t['governing_rule'] or '-'}"
              + (f" — {t['governing_rule_title']}  cites {t['citations']}" if t["governing_rule"] else ""))
        view = t["checks_view"]
        if view:
            print("checks:" if view["full"] else
                  "checks:    " + ("changed since last shown:" if view["lines"] else "no changes"))
            for line in view["lines"]:
                print(f"  {line}")
            for c in t["claims"]:
                where = ", ".join(c["evidence"]["docs"]) if c["status"] == "grounded" else \
                    f"{c['reason']}; related: {c['related_not_answering'] or '-'}"
                print(f"  claim {c['status']:>8}: {c['id']} ({where})")
        print(f"decision:  {t['decision']}")
        v = t["verifier"]
        print(f"verifier:  {'PASS' if v['passed'] else 'FAIL'} on {v['checked']} reply"
              + (f"; fallback {'PASS' if v['fallback']['passed'] else 'FAIL'}" if "fallback" in v else ""))
        for err in v["errors"]:
            print(f"  ! {err}")
        print(f"reply:     ({t['reply_source']}) {t['reply']}")


if __name__ == "__main__":
    trace = run()
    show(trace)
    TRACE.write_text(json.dumps(trace, indent=2, default=str, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {TRACE.name}")
