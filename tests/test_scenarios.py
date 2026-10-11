"""Fact-override scenarios on the 14:09 credit question: which rule governs, escalation, evidence."""
import sys
import unittest
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import facts  # noqa: E402
from eval_cases import load_cases  # noqa: E402
from run import Agent  # noqa: E402

R2_CASES = ("scenario-no-noc-ticket", "scenario-outage-exactly-4h", "scenario-request-day-31")
R4B_CASES = ("scenario-credit-301", "scenario-outage-31-days", "scenario-ticket-unverifiable")


def checks(t):
    return {c["id"]: c["status"] for c in t["checks"]}


class Scenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {c["id"]: c for c in load_cases() if c["id"].startswith("scenario-")}
        agent = Agent(use_llm=False)
        cls.turns = {cid: agent.answer(**c["inputs"]) for cid, c in cls.cases.items()}

    def test_all_scenarios_match_expected(self):
        self.assertEqual(len(self.cases), 8)
        for cid, t in self.turns.items():
            exp = self.cases[cid]["outputs"]
            self.assertEqual(t["decision"], exp["decision"], cid)
            self.assertEqual(t["governing_rule"], exp["governing_rule"], cid)
            self.assertEqual(t["cited_articles"], exp["cited_articles"], cid)
            self.assertTrue(t["verifier"]["passed"], (cid, t["verifier"]["errors"]))

    def test_r2_never_blends_in_the_outage_formula(self):
        for cid in R2_CASES:
            t = self.turns[cid]
            c = checks(t)
            self.assertIsNone(t["credit"], cid)
            self.assertIsNone(t["formula"], cid)
            self.assertNotIn("$", t["reply"].replace("$75.00", ""), cid)  # no amount quoted
            self.assertNotIn("÷", t["reply"], cid)
            self.assertNotIn("R4a.within_goodwill_cap", c, cid)
            self.assertEqual(c["R2.rolling_12_month_total"], "unverified", cid)
            self.assertIn("R2: rolling 12-month goodwill total not verified", t["escalation_reasons"][0], cid)
            self.assertIn("up to $75.00 per account per rolling 12-month period", t["reply"], cid)
            self.assertIn("check the account's credit history before quoting an amount", t["reply"], cid)
            self.assertIn("R3", {n["rule"] for n in t["not_applied"]}, cid)

    def test_r3_failures_that_lead_to_r2(self):
        self.assertEqual(checks(self.turns["scenario-no-noc-ticket"])["R3.noc_ticket"], "failed")
        self.assertEqual(checks(self.turns["scenario-outage-exactly-4h"])["R3.outage_over_4_hours"], "failed")
        c = checks(self.turns["scenario-request-day-31"])
        self.assertEqual(c["R3.within_30_days_of_resolution"], "failed")
        self.assertEqual((c["R3.noc_ticket"], c["R3.outage_over_4_hours"]), ("passed", "passed"))

    def test_exactly_four_hours_explains_outage_credit_and_never_zero(self):
        t = self.turns["scenario-outage-exactly-4h"]
        self.assertIn("doesn't apply", t["reply"])
        self.assertIn("more than 4 consecutive hours", t["reply"])
        self.assertIn("[KB-1004]", t["reply"])
        self.assertIn("general goodwill limit governs", t["reply"])
        self.assertNotIn("$0.00", t["reply"])
        self.assertNotIn("$0.00", str(t))

    def test_known_credit_history_under_cap_answers(self):
        q = self.cases["scenario-no-noc-ticket"]["inputs"]
        t = Agent(use_llm=False).answer(q["question"], q["time"],
                                        {**q["fact_overrides"], "rolling_12mo_goodwill_total": "20.00"})
        self.assertEqual(checks(t)["R2.rolling_12_month_total"], "passed")
        self.assertEqual((t["decision"], t["escalation_reasons"]), ("ANSWER", []))
        t = Agent(use_llm=False).answer(q["question"], q["time"],
                                        {**q["fact_overrides"], "rolling_12mo_goodwill_total": "75.00"})
        self.assertEqual(checks(t)["R2.rolling_12_month_total"], "failed")
        self.assertIn("R4(a)", t["escalation_reasons"][0])

    def test_r4b_escalations(self):
        expected = {"scenario-credit-301": ("R4b.credit_not_over_300", Decimal("301.03"), "over $300.00"),
                    "scenario-outage-31-days": ("R4b.outage_not_over_30_days", Decimal("232.50"), "31 full days"),
                    "scenario-ticket-unverifiable": ("R4b.ticket_verified", Decimal("82.50"), "can't be verified")}
        for cid, (failed, credit, reason) in expected.items():
            t = self.turns[cid]
            c = checks(t)
            self.assertEqual(t["credit"], credit, cid)
            self.assertEqual([k for k, v in c.items() if k.startswith("R4b.") and v == "failed"], [failed], cid)
            self.assertNotIn("R4a.within_goodwill_cap", c, cid)  # an R3 credit is never compared to $75
            self.assertNotIn("R2.rolling_12_month_total", c, cid)
            self.assertIn(reason, t["reply"], cid)
            self.assertIn("[KB-1006]", t["reply"], cid)
            self.assertIn("Escalation Rules", t["route"]["categories"], cid)

    def test_unverifiable_ticket_keeps_r3(self):
        t = self.turns["scenario-ticket-unverifiable"]
        self.assertEqual(checks(t)["R3.noc_ticket"], "unverified")
        self.assertIn("Based on the outage dates you gave, the estimated credit is "
                      "($225.00 ÷ 30) × 11 full days = $82.50.", t["reply"])
        self.assertIn("This is an estimate, pending verification of the outage ticket by a Tier 2 "
                      "Billing Specialist.", t["reply"])
        self.assertNotIn("under Rule 3", t["reply"])  # not stated as a confirmed R3 amount
        self.assertNotIn("None", t["reply"])

    def test_r5_overrides_and_issues_no_credit(self):
        for cid, key in (("scenario-prior-dispute-90d", "R5.prior_dispute_within_90_days"),
                         ("scenario-fraud-flag", "R5.fraud_indicators")):
            t = self.turns[cid]
            self.assertEqual(checks(t)[key], "failed", cid)
            self.assertIsNone(t["credit"], cid)
            self.assertNotIn("$", t["reply"], cid)
            self.assertIn("should not issue a credit themselves", t["reply"], cid)
            self.assertTrue(all(s == "passed" for k, s in checks(t).items() if k.startswith("R3.")), cid)
        self.assertIn("within 90 days", self.turns["scenario-prior-dispute-90d"]["reply"])
        self.assertIn("fraud indicators", self.turns["scenario-fraud-flag"]["reply"])

    def test_governing_article_always_retrieved(self):
        added = [r["id"] for r in self.turns["scenario-fraud-flag"]["retrieved"] if r.get("added")]
        self.assertEqual(added, ["KB-1007"])

    def test_overrides_do_not_leak_between_turns(self):
        agent = Agent(use_llm=False)
        agent.answer(**self.cases["scenario-fraud-flag"]["inputs"])
        t = agent.answer(self.cases["scenario-fraud-flag"]["inputs"]["question"], "14:09")
        self.assertEqual((t["governing_rule"], t["credit"], t["decision"]), ("R3", Decimal("82.50"), "ANSWER"))

    def test_override_validation(self):
        f = facts.extract(facts.read_transcript())
        with self.assertRaises(KeyError):
            facts.apply_overrides(f, {"not_a_fact": 1})
        o = facts.apply_overrides(f, {"noc_ticket": None, "monthly_recurring_charge": "821.00"})
        self.assertEqual(o["noc_ticket"]["status"], "unverified")
        self.assertEqual(o["monthly_recurring_charge"]["value"], Decimal("821.00"))
        self.assertEqual(f["noc_ticket"]["value"], "OUT-58823")


if __name__ == "__main__":
    unittest.main()
