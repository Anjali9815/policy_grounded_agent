"""Graph, memory, verifier and LLM-fallback behaviour. No API key needed: LLMs are faked."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gate  # noqa: E402
from eval_cases import load_cases  # noqa: E402
from llm import ClaudeLLM, LLMError  # noqa: E402
from run import Agent  # noqa: E402

CREDIT_Q = "Okay, but isn't your refund limit only $75? So you can only give me $75 back for this, right?"
TAX_Q = "Does the credit also reduce the taxes and surcharges, or is it only applied against the plan charge?"


class FakeLLM:
    """Stands in for ClaudeLLM. intent: fixed intent or None for rules; rewrite: segments -> segments;
    route_to: categories the router fallback returns. Every call is logged with fake token usage."""

    def __init__(self, intent=None, rewrite=None, fail=False, route_to=None):
        self.intent, self.rewrite, self.fail, self.route_to = intent, rewrite, fail, route_to
        self.usage_log = []

    def route(self, message, categories):
        self.usage_log.append({"purpose": "router", "model": "fake", "input_tokens": 50, "output_tokens": 5})
        if self.fail:
            raise LLMError("fake outage")
        return self.route_to

    def classify(self, message, intents):
        self.usage_log.append({"purpose": "intent", "model": "fake", "input_tokens": 100, "output_tokens": 10})
        if self.fail:
            raise LLMError("fake outage")
        return self.intent or gate.classify(message)

    def phrase(self, question, segments, evidence, computed):
        if self.fail:
            raise LLMError("fake outage")
        return self.rewrite([dict(s) for s in segments]) if self.rewrite else segments


def with_text(segments, claim_id, text):
    for s in segments:
        if s["claim_id"] == claim_id:
            s["text"] = text
    return segments


class DeterministicGraph(unittest.TestCase):
    def test_eval_cases_match_expected(self):
        agent = Agent(use_llm=False)
        for case in load_cases():
            t = agent.answer(**case["inputs"])
            self.assertEqual(t["decision"], case["outputs"]["decision"], case["id"])
            self.assertEqual(t["governing_rule"], case["outputs"]["governing_rule"], case["id"])
            self.assertTrue(t["verifier"]["passed"], (case["id"], t["verifier"]["errors"]))

    def test_case_memory_across_turns(self):
        agent = Agent(use_llm=False)
        trace = agent.run()
        m = agent.memory
        self.assertEqual([d["time"] for d in m.decisions], [t["time"] for t in trace])
        self.assertEqual([d["decision"] for d in m.decisions],
                         ["ANSWER", "NO_QUESTION", "NO_QUESTION", "ANSWER", "PARTIAL", "ANSWER"])
        self.assertEqual(m.facts["noc_ticket"]["value"], "OUT-58823")
        views = [t["checks_view"] for t in trace if t["checks_view"]]
        self.assertTrue(views[0]["full"])
        self.assertTrue(all(not v["full"] for v in views[1:]))
        self.assertEqual(views[-1]["lines"], [])  # 14:13: nothing changed since 14:11

    def test_unknown_routes_to_escalate(self):
        t = Agent(use_llm=False).answer("Can I pay this bill in installments?")
        self.assertEqual((t["intent"], t["decision"]), ("unknown", "ESCALATE"))
        self.assertEqual(t["retrieved"], [])


class LLMPaths(unittest.TestCase):
    def test_llm_unknown_intent_escalates(self):
        t = Agent(llm_client=FakeLLM(intent="unknown")).answer(CREDIT_Q)
        self.assertEqual((t["intent_source"], t["decision"]), ("llm", "ESCALATE"))

    def test_invented_intent_rejected(self):
        llm = ClaudeLLM.__new__(ClaudeLLM)
        llm._json = lambda *a: {"intent": "refund_request"}
        with self.assertRaises(LLMError):
            llm.classify("hi", {"credit_limit": "..."})

    def test_llm_failure_falls_back_to_rules_and_template(self):
        t = Agent(llm_client=FakeLLM(fail=True)).answer(CREDIT_Q)
        self.assertTrue(t["intent_source"].startswith("rules (llm failed"))
        self.assertTrue(t["reply_source"].startswith("template (llm failed"))
        self.assertIn("$82.50", t["reply"])

    def test_faithful_rephrase_passes(self):
        rewrite = lambda segs: with_text(segs, "outage_credit_formula",
                                         "Your credit is ($225.00 ÷ 30) × 11 full days = $82.50 [KB-1004].")
        t = Agent(llm_client=FakeLLM(rewrite=rewrite)).answer(CREDIT_Q)
        self.assertTrue(t["verifier"]["passed"], t["verifier"]["errors"])
        self.assertEqual(t["reply_source"], "llm")

    def test_unsupported_number_falls_back(self):
        rewrite = lambda segs: with_text(segs, "outage_credit_formula",
                                         "You'll get $100.00 back by October 30 [KB-1004].")
        t = Agent(llm_client=FakeLLM(rewrite=rewrite)).answer(CREDIT_Q)
        v = t["verifier"]
        self.assertFalse(v["passed"])
        self.assertEqual(sorted(v["unsupported_values"]), ["100", "md:10-30"])
        self.assertEqual(t["reply_source"], "template (verifier fallback)")
        self.assertTrue(v["fallback"]["passed"])
        self.assertIn("$82.50", t["reply"])
        self.assertNotIn("$100", t["reply"])

    def test_dropped_escalation_falls_back(self):
        rewrite = lambda segs: [s for s in segs if s["claim_id"] != "credit_tax_treatment"]
        t = Agent(llm_client=FakeLLM(rewrite=rewrite)).answer(TAX_Q)
        self.assertFalse(t["verifier"]["passed"])
        self.assertIn("Tier 2", t["reply"])

    def test_unmapped_claim_falls_back(self):
        rewrite = lambda segs: segs + [{"claim_id": "made_up", "text": "Taxes are refunded too."}]
        t = Agent(llm_client=FakeLLM(rewrite=rewrite)).answer(TAX_Q)
        self.assertFalse(t["verifier"]["passed"])
        self.assertNotIn("Taxes are refunded too", t["reply"])

    def test_dropped_citation_falls_back(self):
        rewrite = lambda segs: with_text(segs, "credit_plan_charge_only", "It applies to the plan charge only.")
        t = Agent(llm_client=FakeLLM(rewrite=rewrite)).answer(TAX_Q)
        self.assertFalse(t["verifier"]["passed"])
        self.assertEqual(t["reply_source"], "template (verifier fallback)")


class Routing(unittest.TestCase):
    def test_out_of_scope_escalates_without_any_llm_call(self):
        main, small = FakeLLM(), FakeLLM()
        agent = Agent(llm_client=main, router_llm_client=small)
        for q in ("Can I get a retention credit instead?", "My phone was stolen, can I get a replacement credit?"):
            t = agent.answer(q)
            self.assertEqual(t["route"]["status"], "out_of_scope", q)
            self.assertEqual((t["decision"], t["intent_source"]), ("ESCALATE", "router (out_of_scope)"))
            self.assertEqual(t["route"]["articles"], [])
        self.assertEqual(main.usage_log + small.usage_log, [])

    def test_downstream_sees_only_routed_articles(self):
        trace = Agent(use_llm=False).run()
        for t in trace:
            self.assertTrue({r["id"] for r in t["retrieved"]} <= set(t["route"]["articles"]), t["time"])
            for c in t["claims"]:
                if c["status"] == "grounded":
                    self.assertTrue(set(c["evidence"]["docs"]) <= set(t["route"]["articles"]))
            self.assertLess(t["tokens"]["routed_context"], t["tokens"]["full_context"])
            self.assertEqual(t["tokens"]["llm_input"], 0)
        by_time = {t["time"]: t for t in trace}
        self.assertEqual(by_time["14:13"]["route"]["categories"], ["Service-Outage Credit Policy"])
        self.assertEqual(by_time["14:13"]["route"]["rules"], ["R3"])

    def test_case_context_adds_governing_category(self):
        t = {t["time"]: t for t in Agent(use_llm=False).run()}["14:11"]
        self.assertEqual(t["route"]["case_context"]["rule"], "R3")
        self.assertIn("Service-Outage Credit Policy", t["route"]["case_context"]["added"])

    def test_ambiguous_without_llm_escalates(self):
        t = Agent(use_llm=False).answer("Do you price-match other carriers?")
        self.assertEqual((t["route"]["status"], t["decision"]), ("ambiguous", "ESCALATE"))

    def test_ambiguous_uses_small_llm_once(self):
        small = FakeLLM(route_to=["Service-Outage Credit Policy"])
        t = Agent(llm_client=FakeLLM(), router_llm_client=small).answer("Do you price-match other carriers?")
        self.assertEqual((t["route"]["status"], t["route"]["source"]), ("routed", "llm"))
        self.assertEqual([c["purpose"] for c in small.usage_log], ["router"])
        self.assertEqual(t["decision"], "ESCALATE")  # routed, but no intent fits
        self.assertEqual(t["tokens"]["llm_input"], 150)  # router 50 + intent 100
        self.assertEqual(len(t["tokens"]["llm_calls"]), 2)

    def test_small_llm_out_of_scope_and_failure(self):
        t = Agent(llm_client=FakeLLM(), router_llm_client=FakeLLM(route_to=["out_of_scope"])).answer(
            "Do you price-match other carriers?")
        self.assertEqual((t["route"]["status"], t["decision"]), ("out_of_scope", "ESCALATE"))
        t = Agent(llm_client=FakeLLM(), router_llm_client=FakeLLM(fail=True)).answer(
            "Do you price-match other carriers?")
        self.assertEqual((t["route"]["status"], t["decision"]), ("ambiguous", "ESCALATE"))
        self.assertIn("llm failed", t["route"]["source"])

    def test_clear_routes_never_call_small_llm(self):
        small = FakeLLM(route_to=["Refund Eligibility"])
        Agent(llm_client=FakeLLM(), router_llm_client=small).run()
        self.assertEqual(small.usage_log, [])


if __name__ == "__main__":
    unittest.main()
