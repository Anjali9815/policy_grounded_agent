"""Walk the transcript's customer turns and print/trace a grounded decision for each."""
import json
from pathlib import Path

import facts as facts_mod
import gate
import resolver
from loader import load_kb, load_policy
from retriever import Retriever

TRACE = Path(__file__).resolve().parent / "trace.json"


def governing_for(intent_name, resolution):
    if intent_name == gate.INFO:
        return None
    rule = gate.INTENTS[intent_name]["governing"]
    return resolution["governing_rule"] if rule == "case" else rule


def run(transcript=None):
    kb, policy = load_kb(), load_policy()
    retriever = Retriever(kb)
    text = transcript or facts_mod.read_transcript()
    all_facts = facts_mod.extract(text)
    trace = []
    for turn in facts_mod.customer_turns(text):
        known = facts_mod.as_of(all_facts, turn["time"])
        intent = gate.classify(turn["text"])
        query = turn["text"] + " " + gate.INTENTS.get(intent, {}).get("expand", "")
        retrieved = retriever.search(query)
        resolution = resolver.resolve(known)
        result = gate.evaluate(intent, retrieved, kb, resolution, known)
        governing = governing_for(intent, resolution)
        trace.append({
            "time": turn["time"], "question": turn["text"], "intent": intent,
            "retrieved": retrieved, "excluded_by_scope": retriever.excluded,
            "governing_rule": governing,
            "governing_rule_title": policy[governing]["title"] if governing else None,
            "citations": resolution["citations"] if governing == resolution["governing_rule"] else [governing] if governing else [],
            "checks": resolution["checks"], "not_applied": resolution["not_applied"],
            "credit": resolution["credit"], "formula": resolution["formula"],
            "request_deadline": resolution["request_deadline"],
            "pending_verification": resolution["pending_verification"],
            "escalation_reasons": resolution["escalation_reasons"],
            **result,
        })
    return trace


def show(trace):
    for t in trace:
        print(f"\n=== [{t['time']}] {t['question']}")
        print(f"intent:    {t['intent']}")
        print("retrieved: " + (", ".join(f"{r['id']} ({r['score']})" for r in t["retrieved"]) or "-"))
        print(f"governing: {t['governing_rule'] or '-'}"
              + (f" — {t['governing_rule_title']}  cites {t['citations']}" if t["governing_rule"] else ""))
        if t["intent"] != gate.INFO:
            for c in t["checks"]:
                print(f"  [{c['status']:>10}] {c['id']}: {c['detail']}")
            for n in t["not_applied"]:
                print(f"  [not applied] {n['rule']}: {n['reason']}")
            for c in t["claims"]:
                where = ", ".join(c["evidence"]["docs"]) if c["status"] == "grounded" else \
                    f"{c['reason']}; related: {c['related_not_answering'] or '-'}"
                print(f"  claim {c['status']:>8}: {c['id']} ({where})")
        print(f"decision:  {t['decision']}")
        print(f"reply:     {t['reply']}")


if __name__ == "__main__":
    trace = run()
    show(trace)
    TRACE.write_text(json.dumps(trace, indent=2, default=str, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {TRACE.name}")
