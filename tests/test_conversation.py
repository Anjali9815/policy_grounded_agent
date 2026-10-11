"""Expected outcomes for the Dana Whitfield transcript. Run: python3 -m unittest"""
import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import facts  # noqa: E402
from loader import load_kb, load_policy  # noqa: E402
from retriever import Retriever  # noqa: E402
from run import Agent, run  # noqa: E402


def checks(turn):
    return {c["id"]: c["status"] for c in turn["checks"]}


def claims(turn):
    return {c["id"]: c for c in turn["claims"]}


class LoaderAndFacts(unittest.TestCase):
    def test_loads_all_documents(self):
        self.assertEqual(len(load_kb()), 10)
        self.assertEqual(sorted(load_policy()), ["R0", "R1", "R2", "R3", "R4", "R5"])
        self.assertEqual(load_kb()["KB-1004"]["governs"], "R3")

    def test_retriever_excludes_out_of_scope_articles(self):
        r = Retriever(load_kb())
        self.assertEqual(r.excluded, ["KB-1008", "KB-1009"])
        hits = r.search("retention credit device protection replacement deductible", k=10)
        self.assertNotIn("KB-1008", [h["id"] for h in hits])
        self.assertNotIn("KB-1009", [h["id"] for h in hits])

    def test_extracted_facts(self):
        f = facts.extract(facts.read_transcript())
        self.assertEqual(f["monthly_recurring_charge"]["value"], Decimal("225.00"))
        self.assertEqual(f["noc_ticket"]["value"], "OUT-58823")
        self.assertEqual(str(f["outage_start"]["value"]), "2026-09-10 06:00:00")
        self.assertEqual(str(f["outage_end"]["value"]), "2026-09-21 06:00:00")
        self.assertEqual(f["statement_date"]["value"], date(2026, 10, 1))
        self.assertEqual(f["chat_date"]["value"], date(2026, 10, 5))

    def test_history_and_fraud_never_assumed(self):
        f = facts.extract(facts.read_transcript())
        for key in ("prior_dispute_within_90_days", "fraud_indicators"):
            self.assertEqual(f[key]["status"], "unverified")
            self.assertIsNone(f[key]["value"])


class Conversation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.turns = {t["time"]: t for t in run()}

    def test_1409_credit_question(self):
        t = self.turns["14:09"]
        self.assertEqual(t["intent"], "credit_limit")
        self.assertEqual(t["governing_rule"], "R3")
        self.assertIn("R0", t["citations"])
        self.assertEqual(t["credit"], Decimal("82.50"))
        self.assertIn("$82.50", t["reply"])
        # Not capped by the $75 goodwill ceiling, and no R4(a) comparison.
        self.assertEqual({n["rule"] for n in t["not_applied"]} & {"R2", "R4a"}, {"R2", "R4a"})
        self.assertTrue(claims(t)["outage_credit_not_capped"]["status"] == "grounded")
        # No R4(b) escalation.
        c = checks(t)
        for k in ("R4b.credit_not_over_300", "R4b.outage_not_over_30_days", "R4b.ticket_verified"):
            self.assertEqual(c[k], "passed")
        self.assertEqual(t["escalation_reasons"], [])
        # R5 history flagged as unverified before applying.
        self.assertEqual(c["R5.prior_dispute_within_90_days"], "unverified")
        self.assertEqual(c["R5.fraud_indicators"], "unverified")
        self.assertTrue(t["pending_verification"])
        self.assertIn("[R5]", t["reply"])
        self.assertEqual(t["decision"], "ANSWER")

    def test_1411_taxes_partial(self):
        t = self.turns["14:11"]
        self.assertEqual(t["intent"], "tax_treatment")
        self.assertEqual(t["decision"], "PARTIAL")
        cl = claims(t)
        self.assertEqual(cl["credit_plan_charge_only"]["status"], "grounded")
        self.assertIn("KB-1004", cl["credit_plan_charge_only"]["evidence"]["docs"])
        self.assertEqual(cl["credit_tax_treatment"]["status"], "declined")
        self.assertIn("KB-1003", [r["id"] for r in t["retrieved"]])
        self.assertIn("KB-1003", cl["credit_tax_treatment"]["related_not_answering"])
        self.assertTrue(t["escalate"])
        self.assertIn("Tier 2", t["reply"])

    def test_1413_deadline(self):
        t = self.turns["14:13"]
        self.assertEqual(t["intent"], "request_deadline")
        self.assertEqual(t["governing_rule"], "R3")
        self.assertEqual(t["request_deadline"], date(2026, 10, 21))
        self.assertIn("KB-1004", claims(t)["outage_request_window"]["evidence"]["docs"])
        self.assertIn("2026-10-21", t["reply"])
        self.assertIn("R1", {n["rule"] for n in t["not_applied"]})
        self.assertEqual(t["decision"], "ANSWER")

    def test_1402_open_dispute_gives_next_steps(self):
        t = self.turns["14:02"]
        self.assertEqual(t["intent"], "open_dispute")
        self.assertEqual(t["decision"], "ANSWER")
        self.assertEqual({c["evidence"]["doc"] for c in t["claims"]}, {"KB-1001"})
        self.assertIn("Confirm the account holder's identity", t["reply"])
        self.assertIn("Identify the exact line item", t["reply"])
        self.assertNotIn("60 days", t["reply"])

    def test_information_turns_are_not_questions(self):
        for time in ("14:04", "14:06"):
            self.assertEqual(self.turns[time]["decision"], "NO_QUESTION")

    def test_facts_not_used_before_known(self):
        t = self.turns["14:02"]
        self.assertEqual(checks(t)["R3.noc_ticket"], "unverified")


class WrappedTranscript(unittest.TestCase):
    def test_continuation_lines_joined_with_space(self):
        text = ("[14:13] Customer (Dana Whitfield): how long do I have to submit a\n"
                "request like this?\n\n[14:14] Agent (Jordan R.): Let me check.\n")
        turns = facts.turns(text)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["text"], "how long do I have to submit a request like this?")
        self.assertEqual(turns[1]["text"], "Let me check.")


class UnseenQuestions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = Agent()

    def test_retention_credit_not_answered_from_kb_1008(self):
        t = self.agent.answer("Can I get a retention credit instead?")
        self.assertEqual(t["decision"], "ESCALATE")
        self.assertNotIn("KB-1008", [r["id"] for r in t["retrieved"]])
        self.assertNotIn("KB-1008", t["reply"])
        self.assertNotIn("$50", t["reply"])

    def test_installments_unknown_intent_escalates(self):
        t = self.agent.answer("Can I pay this bill in installments?")
        self.assertEqual(t["intent"], "unknown")
        self.assertEqual(t["decision"], "ESCALATE")
        self.assertIn("Tier 2", t["reply"])

    def test_outage_credit_taxed_declines(self):
        t = self.agent.answer("Is the outage credit taxed?")
        self.assertEqual(t["intent"], "tax_treatment")
        self.assertEqual(t["decision"], "ESCALATE")
        self.assertEqual([c["status"] for c in t["claims"]], ["declined"])
        self.assertIn("Tier 2", t["reply"])

    def test_unrecognized_intents_never_answer(self):
        for q in ("Can I get a retention credit instead?", "Can I pay this bill in installments?",
                  "Do you price-match other carriers?", "What's my data usage this month?",
                  "Can you waive my late fee?", ""):
            t = self.agent.answer(q)
            if t["intent"] == "unknown":
                self.assertEqual(t["decision"], "ESCALATE", q)
            self.assertNotEqual(t["decision"], "ANSWER", q)


if __name__ == "__main__":
    unittest.main()
