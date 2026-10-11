"""Graph, memory, verifier and LLM-fallback behaviour. No API key needed: LLMs are faked."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gate  # noqa: E402
from eval_cases import load_cases  # noqa: E402
import os  # noqa: E402
from unittest import mock  # noqa: E402

import llm  # noqa: E402
from llm import BaseLLM, ClaudeLLM, GroqLLM, LLMError  # noqa: E402
from run import Agent  # noqa: E402

CREDIT_Q = "Okay, but isn't your refund limit only $75? So you can only give me $75 back for this, right?"
DEADLINE_Q = "How long do I have to submit a request like this after an outage gets fixed?"
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
            if case["paraphrase_of"]:
                continue  # covered (and reported one by one) in tests/test_paraphrases.py
            t = agent.answer(**case["inputs"])
            self.assertEqual(t["cited_articles"], sorted(case["outputs"]["cited_articles"]), case["id"])
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

    def test_rule_overrides_llm_intent_on_information_only_turns(self):
        trace = {t["time"]: t for t in Agent(llm_client=FakeLLM(intent="open_dispute")).run()}
        for time in ("14:04", "14:06"):
            t = trace[time]
            self.assertEqual(t["intent"], "provide_information", time)
            self.assertEqual(t["intent_source"], "llm intent overridden by rule", time)
            self.assertEqual(t["llm_intent"], "open_dispute", time)
            self.assertEqual(t["decision"], "NO_QUESTION", time)
        # 14:02 has no provide-information pattern, so the LLM intent stands.
        self.assertEqual((trace["14:02"]["intent_source"], trace["14:02"]["llm_intent"]), ("llm", None))

    def test_no_override_when_turn_asks_something(self):
        agent = Agent(llm_client=FakeLLM(intent="request_deadline"))
        for q in ("My invoice dated October 1st is wrong, can you fix it?",
                  "How long do I have now that service went down on September 10th",
                  "Service went down on September 10th. Is that covered"):
            t = agent.answer(q)
            self.assertEqual((t["intent"], t["intent_source"]), ("request_deadline", "llm"), q)

    def test_invented_intent_rejected(self):
        llm = ClaudeLLM.__new__(ClaudeLLM)
        llm._json = lambda *a: {"intent": "refund_request"}
        with self.assertRaises(LLMError):
            llm.classify("hi", {"credit_limit": "..."})

    def test_llm_failure_falls_back_to_rules_and_template(self):
        t = Agent(llm_client=FakeLLM(fail=True)).answer(CREDIT_Q)
        self.assertTrue(t["intent_source"].startswith("rules (llm failed"))
        self.assertEqual(t["reply_source"], "template (llm disabled)")  # breaker already open
        self.assertIn("$82.50", t["reply"])

    def test_responder_failure_trips_breaker(self):
        class ResponderDown(FakeLLM):
            def phrase(self, *a):
                raise LLMError("fake 500")
        t = Agent(llm_client=ResponderDown()).answer(CREDIT_Q)
        self.assertEqual(t["intent_source"], "llm")
        self.assertTrue(t["reply_source"].startswith("template (llm failed"))
        self.assertEqual(t["llm_breaker"], {"open": True, "reason": "responder_agent: fake 500"})

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

    def test_unicode_spacing_and_dashes_accepted(self):
        rewrite = lambda segs: with_text(with_text(
            segs, "credit_plan_charge_only",
            "It applies to the plan charge only and isn\u2019t limited by the $75 ceiling in KB\u20111002 [KB-1004]."),
            "credit_tax_treatment", "That part is escalated to a Tier\u202f2 Billing Specialist.")
        t = Agent(llm_client=FakeLLM(rewrite=rewrite)).answer(TAX_Q)
        self.assertTrue(t["verifier"]["passed"], t["verifier"]["errors"])

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

    def test_citation_moved_to_end_of_paragraph_falls_back(self):
        moved = ("You need to request it within 30 days of the ticket's resolution date. Your outage was "
                 "resolved on 2026-09-21, so the deadline is 2026-10-21. The 60-day billing-dispute window "
                 "from your statement date is not the deadline for an outage credit [KB-1005, KB-1004] [R0, R1].")
        rewrite = lambda segs: with_text(segs, "outage_request_window", moved)
        t = Agent(llm_client=FakeLLM(rewrite=rewrite)).answer(DEADLINE_Q, "14:13")
        v = t["verifier"]
        self.assertFalse(v["passed"])
        self.assertIn("'outage_request_window' cites ['KB-1005', 'KB-1004'] outside the sentence that "
                      "states the claim", v["errors"])
        self.assertEqual(t["reply_source"], "template (verifier fallback)")
        self.assertTrue(v["fallback"]["passed"])
        self.assertIn("resolution date [KB-1005, KB-1004].", t["reply"])

    def test_citation_after_the_period_still_counts(self):
        ok = ("You need to request it within 30 days of the ticket's resolution date. [KB-1005, KB-1004] "
              "Your outage was resolved on 2026-09-21, so the deadline is 2026-10-21. The 60-day "
              "billing-dispute window from your statement date is not the deadline for an outage credit [R0, R1].")
        rewrite = lambda segs: with_text(segs, "outage_request_window", ok)
        t = Agent(llm_client=FakeLLM(rewrite=rewrite)).answer(DEADLINE_Q, "14:13")
        self.assertTrue(t["verifier"]["passed"], t["verifier"]["errors"])
        self.assertEqual(t["reply_source"], "llm")

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


class RunOutput(unittest.TestCase):
    def test_header_and_trace_path(self):
        import run
        det = Agent(use_llm=False)
        self.assertIn("Run mode: deterministic", run.header(det))
        self.assertEqual(run.trace_path(det).name, "trace_deterministic.json")
        fake = FakeLLM()
        fake.provider, fake.model = "groq", "resp-model"
        small = FakeLLM()
        small.provider, small.model = "groq", "route-model"
        lines = "\n".join(run.header(Agent(llm_client=fake, router_llm_client=small)))
        self.assertIn("intent_agent     groq / resp-model; rules fallback", lines)
        self.assertIn("responder_agent  groq / resp-model; template fallback", lines)
        self.assertIn("router_agent     frontmatter keywords; ambiguous only: groq / route-model", lines)
        self.assertEqual(run.trace_path(Agent(llm_client=fake)).name, "trace_groq.json")


class Breaker(unittest.TestCase):
    def test_first_failure_disables_llm_for_rest_of_run_with_one_log_line(self):
        main = FakeLLM(fail=True)
        agent = Agent(llm_client=main, router_llm_client=FakeLLM())
        with self.assertLogs("policy_agent", level="WARNING") as logs:
            trace = agent.run()
        self.assertEqual(len(logs.output), 1)
        self.assertIn("LLM disabled for this run after intent_agent failed", logs.output[0])
        self.assertEqual(len(main.usage_log), 1)  # only the first, failing call was made
        self.assertTrue(trace[0]["intent_source"].startswith("rules (llm failed"))
        self.assertTrue(all(t["intent_source"] == "rules (llm disabled)" for t in trace[1:]))
        self.assertTrue(all(t["llm_breaker"]["open"] for t in trace))
        # Decisions are unchanged from the deterministic run.
        self.assertEqual([t["decision"] for t in trace],
                         [t["decision"] for t in Agent(use_llm=False).run()])

    def test_verifier_failure_does_not_trip_breaker(self):
        rewrite = lambda segs: with_text(segs, "outage_credit_formula", "You get $100.00 [KB-1004].")
        t = Agent(llm_client=FakeLLM(rewrite=rewrite)).answer(CREDIT_Q)
        self.assertEqual(t["reply_source"], "template (verifier fallback)")
        self.assertFalse(t["llm_breaker"]["open"])


class StubLLM(BaseLLM):
    provider = "stub"

    def __init__(self, reply):
        super().__init__("stub-model")
        self.reply = reply

    def _call(self, system, user, schema):
        return self.reply, 10, 2


class ProviderValidation(unittest.TestCase):
    """Enum and schema checks run on every provider's raw output, including Groq's JSON mode."""

    def test_invented_values_rejected(self):
        with self.assertRaises(LLMError):
            StubLLM('{"categories": ["Refund Eligibility", "Loyalty Perks"]}').route(
                "hi", {"Refund Eligibility": "refund"})
        with self.assertRaises(LLMError):
            StubLLM('{"segments": [{"claim_id": "made_up", "text": "x"}]}').phrase(
                "q", [{"claim_id": "a", "text": "y"}], [], {})
        with self.assertRaises(LLMError):
            StubLLM("not json").classify("hi", {"credit_limit": "..."})
        with self.assertRaises(LLMError):
            StubLLM('["unknown"]').classify("hi", {"credit_limit": "..."})

    def test_valid_output_and_usage_logged(self):
        stub = StubLLM('{"intent": "credit_limit"}')
        self.assertEqual(stub.classify("hi", {"credit_limit": "..."}), "credit_limit")
        self.assertEqual(stub.usage_log[0]["input_tokens"], 10)

    def test_groq_json_mode_call(self):
        g = GroqLLM.__new__(GroqLLM)
        BaseLLM.__init__(g, "groq-model")
        import groq
        g._groq = groq
        completion = mock.Mock()
        completion.choices = [mock.Mock(finish_reason="stop", message=mock.Mock(content='{"intent": "unknown"}'))]
        completion.usage = mock.Mock(prompt_tokens=40, completion_tokens=4)
        g.client = mock.Mock()
        g.client.chat.completions.create.return_value = completion
        self.assertEqual(g.classify("hi", {"credit_limit": "..."}), "unknown")
        kwargs = g.client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["model"], "groq-model")
        self.assertIn('"enum"', kwargs["messages"][0]["content"])
        self.assertEqual(g.usage_log[0]["provider"], "groq")

    def test_provider_config(self):
        env = {"USE_LLM": "1", "LLM_PROVIDER": "groq", "GROQ_API_KEY": "test",
               "RESPONDER_MODEL": "resp-model", "ROUTER_MODEL": "route-model"}
        with mock.patch.dict(os.environ, env):
            self.assertIsInstance(llm.make(), GroqLLM)
            self.assertEqual((llm.make().model, llm.make_router().model), ("resp-model", "route-model"))
        with mock.patch.dict(os.environ, {**env, "ROUTER_MODEL": ""}):
            with self.assertRaises(ValueError):
                llm.make_router()
        with mock.patch.dict(os.environ, {**env, "LLM_PROVIDER": "openai"}):
            with self.assertRaises(ValueError):
                llm.make()
        with mock.patch.dict(os.environ, {"USE_LLM": "1", "LLM_PROVIDER": "anthropic",
                                          "ANTHROPIC_API_KEY": "test", "RESPONDER_MODEL": "",
                                          "ROUTER_MODEL": "", "ANTHROPIC_MODEL": ""}):
            self.assertIsInstance(llm.make(), ClaudeLLM)
            self.assertEqual(llm.make().model, "claude-opus-5-5")
            self.assertFalse(llm.make_router().server_fallback)


if __name__ == "__main__":
    unittest.main()
