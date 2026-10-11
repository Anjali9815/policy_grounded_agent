"""requested_amount slot: extraction, reply wording per governing rule, and verifier enforcement."""
import sys
import unittest
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate  # noqa: E402
from eval_cases import load_cases  # noqa: E402
from run import Agent  # noqa: E402
from test_ask_eval import session  # noqa: E402
from test_graph import FakeLLM, with_text  # noqa: E402

Q100 = "Can you refund me $100 for the outage, or is there a limit?"
NO_TICKET = {"noc_ticket": False, "outage_start": None, "outage_end": None}


class Extraction(unittest.TestCase):
    def test_forms(self):
        cases = {
            "Can you refund me $100?": Decimal("100"),
            "I want 100 dollars back, what's the limit": Decimal("100"),
            "give me 100 bucks, is there a limit": Decimal("100"),
            "Is $1,250.50 over the limit?": Decimal("1250.50"),
            "isn't your refund limit only $75?": None,                       # the limit, not a request
            "wait so you guys can only refund me like 75 bucks max for this?": None,
            "Is the limit $75 or can I get $100?": Decimal("100"),           # $75 ignored, $100 requested
            "Can I get $100 or $120 under the limit?": None,                 # ambiguous
            "Is the limit the same for everyone?": None,
        }
        for q, expected in cases.items():
            self.assertEqual(gate.extract_requested_amount(q, "credit_limit"), expected, q)

    def test_only_credit_intents(self):
        self.assertIsNone(gate.extract_requested_amount("Does the $100 credit reduce taxes?", "tax_treatment"))
        self.assertIsNone(gate.extract_requested_amount("Can I pay $100 in installments?", "unknown"))


class Replies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = Agent(use_llm=False)

    def ask(self, q=Q100, overrides=None):
        return self.agent.answer(q, "14:09", overrides)

    def test_dana_case_100_dollars(self):
        t = self.ask()
        self.assertEqual((t["decision"], t["governing_rule"], t["cited_articles"]), ("ANSWER", "R3", ["KB-1004"]))
        self.assertEqual(t["requested_amount"], Decimal("100"))
        self.assertTrue(t["reply"].startswith("The credit for your outage comes to $82.50, not $100.00."))
        self.assertIn("still need to be checked [R5]", t["reply"])  # R5 pending notice kept
        self.assertEqual(t["reply"].count("$100.00"), 1)
        self.assertTrue(t["verifier"]["passed"], t["verifier"]["errors"])

    def test_request_below_credit_states_the_credit(self):
        t = self.ask("Can you refund me $60 for the outage, or is there a limit?")
        self.assertTrue(t["reply"].startswith("The credit for your outage comes to $82.50."))
        self.assertNotIn("$60", t["reply"])
        self.assertEqual(t["decision"], "ANSWER")

    def test_r2_declines_and_quotes_ceiling(self):
        t = self.ask(overrides=NO_TICKET)
        self.assertEqual((t["decision"], t["governing_rule"]), ("PARTIAL", "R2"))
        self.assertTrue(t["reply"].startswith("I can't confirm $100.00."))
        self.assertIn("up to $75.00 per account per rolling 12-month period", t["reply"])
        self.assertNotIn("comes to", t["reply"])
        self.assertTrue(t["verifier"]["passed"], t["verifier"]["errors"])

    def test_r4b_and_r5_never_confirm(self):
        for overrides, rule, decision in (({"monthly_recurring_charge": "821.00"}, "R3", "PARTIAL"),
                                          ({"fraud_indicators": True}, "R5", "ESCALATE")):
            t = self.ask(overrides=overrides)
            self.assertEqual((t["governing_rule"], t["decision"]), (rule, decision))
            self.assertTrue(t["reply"].startswith("I can't confirm $100.00."), rule)
            self.assertNotIn("comes to", t["reply"])
            self.assertTrue(t["verifier"]["passed"], t["verifier"]["errors"])

    def test_original_1409_unchanged(self):
        orig = next(c for c in load_cases() if c["id"] == "turn-1409-credit-limit")
        t = Agent(use_llm=False).answer(**orig["inputs"])
        self.assertIsNone(t["requested_amount"])
        self.assertTrue(t["reply"].startswith("The $75.00 limit is not the one that applies"))
        self.assertEqual((t["decision"], t["governing_rule"], t["cited_articles"]), ("ANSWER", "R3", ["KB-1004"]))

    def test_fresh_memory_escalates(self):
        _, out = session(["--no-llm", "--fresh"], [Q100, "quit"])
        self.assertIn("decision:   ESCALATE", out)
        self.assertIn("I can't confirm $100.00.", out)
        self.assertNotIn("comes to", out)


class VerifierEnforcement(unittest.TestCase):
    def answer_with(self, text):
        rewrite = lambda segs: with_text(segs, "notice.requested_amount", text)
        return Agent(llm_client=FakeLLM(rewrite=rewrite)).answer(Q100, "14:09")

    def test_requested_amount_as_approval_rejected(self):
        t = self.answer_with("Good news, you'll get $100.00 back for the outage.")
        self.assertFalse(t["verifier"]["passed"])
        self.assertIn("'notice.requested_amount' presents the requested amount $100.00 as approved",
                      t["verifier"]["errors"])
        self.assertEqual(t["reply_source"], "template (verifier fallback)")
        self.assertIn("$82.50, not $100.00", t["reply"])

    def test_denied_mentions_allowed(self):
        for text in ("Your outage credit is $82.50, not $100.00.",
                     "That works out to $82.50 rather than the $100 you asked for.",
                     "I can’t confirm 100 dollars; the outage credit is $82.50."):
            t = self.answer_with(text)
            self.assertTrue(t["verifier"]["passed"], (text, t["verifier"]["errors"]))

    def test_requested_amount_elsewhere_rejected(self):
        rewrite = lambda segs: with_text(segs, "outage_credit_formula",
                                         "For a confirmed outage the credit is $100.00 [KB-1004].")
        t = Agent(llm_client=FakeLLM(rewrite=rewrite)).answer(Q100, "14:09")
        self.assertFalse(t["verifier"]["passed"])
        self.assertNotIn("credit is $100.00", t["reply"])


if __name__ == "__main__":
    unittest.main()
