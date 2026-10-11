"""Walk the transcript's customer turns through the agent graph; print and trace each turn."""
import argparse
import json
import sys
from pathlib import Path

import facts as facts_mod
import gate
import graph
import llm
from loader import load_kb, load_policy
from memory import CaseMemory
from router import OUT_OF_SCOPE_VALUE

TRACES = Path(__file__).resolve().parent / "traces"


def cited_articles(gate_result):
    """KB ids the gate grounded claims on (every retrieved article that stated a grounded claim)."""
    return sorted({d for c in gate_result["claims"] if c["status"] == "grounded"
                   for d in c["evidence"]["docs"]})


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
        self.breaker = llm.CircuitBreaker()
        self.graph = graph.build(self.kb, self.policy, self.facts, self.llm, self.router_llm, self.breaker)
        self.memory = CaseMemory()
        self.full_context_tokens = graph.context_tokens(self.kb) + graph.context_tokens(self.policy)

    def _usage_logs(self):
        clients = {id(c): c for c in (self.llm, self.router_llm) if c is not None}
        return [getattr(c, "usage_log", []) for c in clients.values()]

    def answer(self, question, time="23:59", fact_overrides=None):
        """Run one customer message through the graph using the facts known at chat time 'HH:MM'.

        fact_overrides replaces extracted case facts for this turn (scenario tests)."""
        marks = [len(log) for log in self._usage_logs()]
        s = self.graph.invoke({"question": question, "time": time, "memory": self.memory,
                               "fact_overrides": fact_overrides or {}})
        calls = [c for log, m in zip(self._usage_logs(), marks) for c in log[m:]]
        resolution, result, route = s["resolution"], s["gate"], s["route"]
        governing = graph.governing_for(s["intent"], resolution)
        return {
            "time": time, "question": question, "intent": s["intent"], "intent_source": s["intent_source"],
            "llm_intent": s.get("llm_intent"),
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
            "fact_overrides": fact_overrides or {},
            "request_deadline": resolution["request_deadline"],
            "pending_verification": resolution["pending_verification"],
            "escalation_reasons": resolution["escalation_reasons"],
            "requested_amount": result.get("requested_amount"),
            "claims": result["claims"], "cited_articles": cited_articles(result),
            "decision": result["decision"], "escalate": result["escalate"],
            "segments": s["segments"], "reply": " ".join(seg["text"] for seg in s["segments"]),
            "reply_source": s["reply_source"], "verifier": s["verifier"],
            "llm_breaker": {"open": self.breaker.open, "reason": self.breaker.reason},
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
        print(f"intent:    {t['intent']} ({t['intent_source']}"
              + (f"; llm said {t['llm_intent']})" if t["llm_intent"] else ")"))
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


def _llm_label(client):
    return f"{getattr(client, 'provider', 'custom')} / {getattr(client, 'model', type(client).__name__)}"


def header(agent):
    """Provider and model used by each agent in this run."""
    main = _llm_label(agent.llm) if agent.llm else None
    small = _llm_label(agent.router_llm) if agent.router_llm else None
    rows = [
        ("policy_agent", "deterministic resolver (no LLM)"),
        ("router_agent", "frontmatter keywords" + (f"; ambiguous only: {small}" if small else " (no LLM)")),
        ("intent_agent", f"{main}; rules fallback" if main else "keyword rules (no LLM)"),
        ("retrieval_agent", "BM25 over routed articles (no LLM)"),
        ("gate_agent", "deterministic claim grounding (no LLM)"),
        ("responder_agent", f"{main}; template fallback" if main else "templates (no LLM)"),
        ("verifier_agent", "deterministic checks (no LLM)"),
    ]
    mode = "LLM" if (main or small) else "deterministic"
    return [f"Run mode: {mode}"] + [f"  {name:<16} {desc}" for name, desc in rows]


def trace_path(agent):
    if agent.llm is None and agent.router_llm is None:
        return TRACES / "trace_deterministic.json"
    return TRACES / f"trace_{getattr(agent.llm or agent.router_llm, 'provider', 'llm')}.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the agent over a transcript's customer turns.")
    parser.add_argument("--transcript", type=Path, default=facts_mod.TRANSCRIPT,
                        help="chat transcript to run (default: Dana Whitfield's)")
    args = parser.parse_args(argv)

    agent = Agent(transcript=facts_mod.read_transcript(args.transcript))
    print(f"Transcript: {args.transcript}")
    print("\n".join(header(agent)))
    trace = agent.run()
    show(trace)
    path = trace_path(agent)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(trace, indent=2, default=str, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {path.relative_to(TRACES.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
