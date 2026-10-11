"""ask.py interactive loop and the LangSmith eval outputs/evaluators (no network, no LLM)."""
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ask  # noqa: E402
import eval_langsmith  # noqa: E402
from eval_cases import load_cases  # noqa: E402
from run import Agent  # noqa: E402

CREDIT_Q = "Is my refund limit $75?"


def session(argv, questions):
    feed = iter(questions)

    def fake_input(prompt):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError
    out = io.StringIO()
    code = ask.main(argv, input_fn=fake_input, out=out)
    return code, out.getvalue()


class AskLoop(unittest.TestCase):
    def test_loaded_memory_answers_and_quit_exits(self):
        code, out = session(["--no-llm"], [CREDIT_Q, "", "quit", "never asked"])
        self.assertEqual(code, 0)
        self.assertIn("Case memory: loaded 6 transcript turns", out)
        self.assertIn("governing:  R3", out)
        self.assertIn("cited:      KB-1004", out)
        self.assertIn("decision:   ANSWER", out)
        self.assertIn("$82.50", out)
        self.assertIn("route:      routed -> ", out)
        self.assertEqual(out.count("decision:"), 1)  # blank line skipped, nothing after quit

    def test_fresh_starts_with_empty_memory(self):
        code, out = session(["--no-llm", "--fresh"], [CREDIT_Q, "quit"])
        self.assertIn("Case memory: empty (--fresh)", out)
        # Without the chat's facts the outage ticket is unknown, so R3 can't be applied.
        self.assertIn("governing:  -", out)
        self.assertIn("cited:      -", out)
        self.assertIn("decision:   ESCALATE", out)
        self.assertNotIn("$82.50", out)

    def test_memory_carries_transcript_then_new_question(self):
        agent, time, loaded = ask.load_agent(fresh=False, use_llm=False)
        self.assertEqual((loaded, time), (6, "23:59"))
        agent.answer("Can I pay this bill in installments?", time)
        self.assertEqual(len(agent.memory.decisions), 7)
        self.assertEqual(agent.memory.decisions[-1]["decision"], "ESCALATE")

    def test_out_of_scope_and_eof(self):
        code, out = session(["--no-llm"], ["Can I get a retention credit instead?"])  # then EOF
        self.assertEqual(code, 0)
        self.assertIn("route:      out_of_scope -> -", out)
        self.assertIn("decision:   ESCALATE", out)


class LangSmithEval(unittest.TestCase):
    def test_cited_articles_and_evaluators_on_all_cases(self):
        target = eval_langsmith.make_target(Agent(use_llm=False))
        for case in load_cases():
            if case["paraphrase_of"]:
                continue  # see tests/test_paraphrases.py
            outputs = target(case["inputs"])
            for ev in eval_langsmith.EVALUATORS:
                kwargs = {"outputs": outputs}
                if "reference_outputs" in ev.__code__.co_varnames:
                    kwargs["reference_outputs"] = case["outputs"]
                result = ev(**kwargs)
                self.assertEqual(result["score"], 1, (case["id"], result))

    def test_expected_articles(self):
        expected = {c["id"]: c["outputs"]["cited_articles"] for c in load_cases()}
        self.assertEqual(expected["turn-1402-open-dispute"], ["KB-1001"])
        self.assertEqual(expected["turn-1409-credit-limit"], ["KB-1004"])
        self.assertEqual(expected["turn-1411-taxes"], ["KB-1004"])
        self.assertEqual(expected["turn-1413-deadline"], ["KB-1004", "KB-1005"])
        others = [v for k, v in expected.items() if not k.startswith(("scenario-", "paraphrase-")) and k not in
                  ("turn-1402-open-dispute", "turn-1409-credit-limit", "turn-1411-taxes", "turn-1413-deadline")]
        self.assertTrue(others and all(v == [] for v in others))

    def test_evidence_correct_scoring(self):
        ev = eval_langsmith.evidence_correct
        self.assertEqual(ev({"cited_articles": ["KB-1005", "KB-1004"]},
                            {"cited_articles": ["KB-1004", "KB-1005"]})["score"], 1)
        wrong = ev({"cited_articles": ["KB-1002"]}, {"cited_articles": ["KB-1004"]})
        self.assertEqual(wrong["score"], 0)
        self.assertIn("expected ['KB-1004']", wrong["comment"])
        self.assertIsNone(ev({"cited_articles": []}, {"decision": "ANSWER"})["score"])

    def test_dataset_examples_include_expected_articles(self):
        client = mock.Mock()
        client.has_dataset.return_value = False
        eval_langsmith.ensure_dataset(client, "ds", recreate=False)
        examples = client.create_examples.call_args.kwargs["examples"]
        self.assertEqual(len(examples), len(load_cases()))
        self.assertTrue(all("cited_articles" in e["outputs"] for e in examples))


if __name__ == "__main__":
    unittest.main()
