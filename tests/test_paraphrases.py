"""Rewordings of the 14:09, 14:11 and 14:13 questions, run deterministically.

Each paraphrase must reach the same decision, governing rule and cited articles
as the original turn. Failures are reported per case (subTest) and are not fixed
by tuning keywords to these exact phrasings.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_cases import load_cases  # noqa: E402
from run import Agent  # noqa: E402


# Deterministic keyword gaps, kept as expected failures. Do not fix these by adding the
# exact phrasings to gate.INTENTS; they are the cases the LLM intent agent is meant to cover.
KNOWN_GAPS = {
    # credit_limit matches only "$75", "refund limit" or "limit"; "75 bucks max" has none of them.
    "paraphrase-1409-casual",
    # request_deadline matches only "how long" or "deadline"; the typo "how lnog" has neither.
    "paraphrase-1413-typo",
}


class Paraphrases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {c["id"]: c for c in load_cases() if c["paraphrase_of"]}
        cls.originals = {c["id"]: c for c in load_cases()}
        cls.agent = Agent(use_llm=False)

    def outcome(self, case_id):
        c = self.cases[case_id]
        t = self.agent.answer(**c["inputs"])
        got = {"decision": t["decision"], "governing_rule": t["governing_rule"],
               "cited_articles": t["cited_articles"]}
        return got, c["outputs"], t["intent"]

    def test_three_styles_per_question(self):
        by_original = {}
        for c in self.cases.values():
            by_original.setdefault(c["paraphrase_of"], []).append(c["id"].rsplit("-", 1)[1])
        self.assertEqual(by_original, {o: ["casual", "terse", "typo"] for o in
                                       ("turn-1409-credit-limit", "turn-1411-taxes", "turn-1413-deadline")})
        for c in self.cases.values():  # same expectations and chat time as the original turn
            orig = self.originals[c["paraphrase_of"]]
            self.assertEqual(c["outputs"], orig["outputs"], c["id"])
            self.assertEqual(c["inputs"]["time"], orig["inputs"]["time"], c["id"])
        self.assertTrue(KNOWN_GAPS <= set(self.cases))

    def test_paraphrases_match_original_outcome(self):
        for case_id in sorted(set(self.cases) - KNOWN_GAPS):
            with self.subTest(case=case_id, question=self.cases[case_id]["inputs"]["question"]):
                got, exp, intent = self.outcome(case_id)
                self.assertEqual(got, exp, f"intent={intent}")

    @unittest.expectedFailure
    def test_known_gap_1409_casual(self):
        # "75 bucks max": no "$75" / "refund limit" / "limit" keyword, so intent is unknown -> ESCALATE.
        got, exp, _ = self.outcome("paraphrase-1409-casual")
        self.assertEqual(got, exp)

    @unittest.expectedFailure
    def test_known_gap_1413_typo(self):
        # "how lnog": the typo breaks the "how long" keyword and there is no "deadline" -> ESCALATE.
        got, exp, _ = self.outcome("paraphrase-1413-typo")
        self.assertEqual(got, exp)

    def test_known_gaps_fail_safe(self):
        # Even when missed, the gaps escalate rather than answer wrongly.
        for case_id in sorted(KNOWN_GAPS):
            got, _, intent = self.outcome(case_id)
            self.assertEqual((got["decision"], intent), ("ESCALATE", "unknown"), case_id)


if __name__ == "__main__":
    unittest.main()
