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
    if intent_name not in gate.INTENTS:
        return None
    rule = gate.INTENTS[intent_name]["governing"]
    return resolution["governing_rule"] if rule == "case" else rule


class Agent:
    def __init__(self, transcript=None):
        self.kb, self.policy = load_kb(), load_policy()
        self.retriever = Retriever(self.kb)
        self.text = transcript or facts_mod.read_transcript()
        self.facts = facts_mod.extract(self.text)

    def answer(self, question, time="23:59"):
        """Evaluate one customer message using the facts known at chat time 'HH:MM'."""
        known = facts_mod.as_of(self.facts, time)
        intent = gate.classify(question)
        query = question + " " + gate.INTENTS.get(intent, {}).get("expand", "")
        retrieved = self.retriever.search(query)
        resolution = resolver.resolve(known)
        result = gate.evaluate(intent, question, retrieved, self.kb, resolution, known)
        governing = governing_for(intent, resolution)
        return {
            "time": time, "question": question, "intent": intent,
            "retrieved": retrieved, "excluded_by_scope": self.retriever.excluded,
            "governing_rule": governing,
            "governing_rule_title": self.policy[governing]["title"] if governing else None,
            "citations": resolution["citations"] if governing == resolution["governing_rule"] else [governing] if governing else [],
            "checks": resolution["checks"], "not_applied": resolution["not_applied"],
            "credit": resolution["credit"], "formula": resolution["formula"],
            "request_deadline": resolution["request_deadline"],
            "pending_verification": resolution["pending_verification"],
            "escalation_reasons": resolution["escalation_reasons"],
            **result,
        }

    def run(self):
        return [self.answer(t["text"], t["time"]) for t in facts_mod.customer_turns(self.text)]


def run(transcript=None):
    return Agent(transcript).run()


def _check_lines(turn):
    lines = {c["id"]: f"[{c['status']:>10}] {c['id']}: {c['detail']}" for c in turn["checks"]}
    lines.update({f"not_applied.{n['rule']}": f"[not applied] {n['rule']}: {n['reason']}"
                  for n in turn["not_applied"]})
    return lines


def show(trace):
    shown = None  # check lines as last printed; the first block is printed in full
    for t in trace:
        print(f"\n=== [{t['time']}] {t['question']}")
        print(f"intent:    {t['intent']}")
        print("retrieved: " + (", ".join(f"{r['id']} ({r['score']})" for r in t["retrieved"]) or "-"))
        print(f"governing: {t['governing_rule'] or '-'}"
              + (f" — {t['governing_rule_title']}  cites {t['citations']}" if t["governing_rule"] else ""))
        if t["intent"] in gate.INTENTS:
            lines = _check_lines(t)
            if shown is None:
                print("checks:")
                changed = list(lines.values())
            else:
                changed = [v for k, v in lines.items() if shown.get(k) != v]
                changed += [f"[   removed] {k}" for k in shown if k not in lines]
                print("checks:    " + ("changed since last shown:" if changed else "no changes"))
            for line in changed:
                print(f"  {line}")
            shown = lines
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
