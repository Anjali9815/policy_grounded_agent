"""Shared eval cases (data/eval_cases.json) resolved to concrete questions and chat times."""
import json

import facts as facts_mod
from loader import DATA

UNSEEN_TIME = "23:59"  # unseen questions are asked after the whole transcript is known


def load_cases(transcript=None):
    turns = {t["time"]: t["text"] for t in facts_mod.customer_turns(transcript or facts_mod.read_transcript())}
    cases = []
    for c in json.loads((DATA / "eval_cases.json").read_text(encoding="utf-8")):
        # A case is a transcript turn ("turn"), or its own "question" asked at "time" (default: after the chat).
        time = c.get("time") or c.get("turn") or UNSEEN_TIME
        question = c["question"] if "question" in c else turns[c["turn"]]
        inputs = {"question": question, "time": time}
        if c.get("fact_overrides"):
            inputs["fact_overrides"] = c["fact_overrides"]
        cases.append({"id": c["id"], "scenario": c.get("scenario"), "paraphrase_of": c.get("paraphrase_of"),
                      "inputs": inputs,
                      "outputs": {"decision": c["expected_decision"], "governing_rule": c["expected_rule"],
                                  "cited_articles": c["expected_articles"]}})
    return cases
