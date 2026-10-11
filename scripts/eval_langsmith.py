"""Evaluate the agent on LangSmith (runs only when LANGSMITH_API_KEY is set).

Creates (or reuses) a dataset from the transcript turns, unseen questions,
fact-override scenarios and paraphrases in data/eval_cases.json, runs the agent graph on each example, and scores:
  decision_correct  decision matches the expected ANSWER/PARTIAL/ESCALATE/NO_QUESTION
  rule_correct      governing rule matches the expected rule (or none)
  verifier_pass     the responder's reply passed the verifier (no unsupported numbers)
  evidence_correct  KB articles the gate grounded claims on match the expected articles

Usage: python scripts/eval_langsmith.py [--dataset NAME] [--recreate]
USE_LLM in .env decides whether the LLM agents run.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402  (loads .env)
from eval_cases import load_cases  # noqa: E402
from run import Agent  # noqa: E402

DEFAULT_DATASET = "policy-grounded-billing-dispute"


def decision_correct(outputs, reference_outputs):
    return {"key": "decision_correct", "score": int(outputs["decision"] == reference_outputs["decision"])}


def rule_correct(outputs, reference_outputs):
    return {"key": "rule_correct",
            "score": int(outputs["governing_rule"] == reference_outputs["governing_rule"])}


def verifier_pass(outputs):
    return {"key": "verifier_pass", "score": int(outputs["verifier_passed"]),
            "comment": "; ".join(outputs["verifier_errors"]) or None}


def evidence_correct(outputs, reference_outputs):
    if "cited_articles" not in reference_outputs:
        return {"key": "evidence_correct", "score": None,
                "comment": "dataset has no expected articles; rerun with --recreate"}
    expected, cited = sorted(reference_outputs["cited_articles"]), sorted(outputs["cited_articles"])
    return {"key": "evidence_correct", "score": int(cited == expected),
            "comment": None if cited == expected else f"cited {cited}, expected {expected}"}


EVALUATORS = [decision_correct, rule_correct, verifier_pass, evidence_correct]


def make_target(agent):
    def target(inputs):
        t = agent.answer(inputs["question"], inputs["time"], inputs.get("fact_overrides"))
        return {"decision": t["decision"], "governing_rule": t["governing_rule"], "reply": t["reply"],
                "cited_articles": t["cited_articles"],
                "intent": t["intent"], "reply_source": t["reply_source"],
                "route_status": t["route"]["status"], "categories": t["route"]["categories"],
                "routed_context_tokens": t["tokens"]["routed_context"],
                "llm_tokens": t["tokens"]["llm_input"] + t["tokens"]["llm_output"],
                "verifier_passed": t["verifier"]["passed"], "verifier_errors": t["verifier"]["errors"]}
    return target


def ensure_dataset(client, name, recreate):
    if client.has_dataset(dataset_name=name):
        if not recreate:
            return
        client.delete_dataset(dataset_name=name)
    ds = client.create_dataset(name, description="Billing-dispute transcript turns and unseen questions "
                                                  "with expected decision, governing rule and "
                                                  "cited articles.")
    client.create_examples(dataset_id=ds.id, examples=[
        {"inputs": c["inputs"], "outputs": c["outputs"],
         "metadata": {"case_id": c["id"], "kind": c["id"].split("-")[0],
                      "paraphrase_of": c["paraphrase_of"], "scenario": c["scenario"]}}
        for c in load_cases()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--recreate", action="store_true", help="delete and rebuild the dataset")
    args = parser.parse_args()

    if not config.langsmith_enabled():
        print("LANGSMITH_API_KEY is not set; skipping LangSmith evaluation.")
        return 0

    from langsmith import Client, evaluate

    client = Client()
    ensure_dataset(client, args.dataset, args.recreate)
    agent = Agent()
    evaluate(make_target(agent), data=args.dataset, evaluators=EVALUATORS,
             experiment_prefix=f"agent-{'llm' if agent.llm else 'deterministic'}", client=client)
    return 0


if __name__ == "__main__":
    sys.exit(main())
