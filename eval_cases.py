"""Shared eval cases (data/eval_cases.json) resolved to concrete questions and chat times."""
import json

import facts as facts_mod
from loader import DATA

UNSEEN_TIME = "23:59"  # unseen questions are asked after the whole transcript is known


def load_cases(transcript=None):
    turns = {t["time"]: t["text"] for t in facts_mod.customer_turns(transcript or facts_mod.read_transcript())}
    cases = []
    for c in json.loads((DATA / "eval_cases.json").read_text(encoding="utf-8")):
        time = c.get("turn", UNSEEN_TIME)
        question = turns[c["turn"]] if "turn" in c else c["question"]
        cases.append({"id": c["id"], "inputs": {"question": question, "time": time},
                      "outputs": {"decision": c["expected_decision"], "governing_rule": c["expected_rule"]}})
    return cases
