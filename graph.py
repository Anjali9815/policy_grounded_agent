"""LangGraph workflow: one invocation per customer turn.

policy_agent -> router_agent -> (out of scope / unrouted) ------------------> escalate_agent
                             -> intent_agent -> (unknown) -------------------> escalate_agent
                                             -> retrieval_agent -> gate_agent -> responder_agent
escalate_agent, responder_agent -> verifier_agent

policy_agent runs first: it needs only the case facts (no KB) and gives the
router the case's governing rule. router_agent narrows the KB to the routed
categories; retrieval, gate and responder see only those articles and the rules
they govern. LLMs (router fallback, intent, responder) are optional and every
one falls back to the deterministic path. The first LLM failure trips a circuit
breaker that disables every LLM for the rest of the run. verifier_agent replaces
an LLM reply that fails verification with the template reply (a verification
failure is not an LLM failure and does not trip the breaker).
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import facts as facts_mod
import gate
import resolver
from llm import OUT_OF_SCOPE, CircuitBreaker, LLMError
from memory import CaseMemory
from retriever import Retriever
from router import Router, approx_tokens
from verifier import verify


class TurnState(TypedDict, total=False):
    question: str
    time: str
    fact_overrides: dict
    memory: CaseMemory
    known: dict
    resolution: dict
    route: dict
    routed_kb: dict
    routed_rules: dict
    intent: str
    intent_source: str
    llm_intent: str
    retrieved: list
    gate: dict
    segments: list
    reply_source: str
    verifier: dict


RETRIEVE_K = 4


def governing_for(intent_name, resolution):
    if intent_name not in gate.INTENTS:
        return None
    rule = gate.INTENTS[intent_name]["governing"]
    return resolution["governing_rule"] if rule == "case" else rule


def _computed(resolution, known):
    return {
        "monthly_recurring_charge": known["monthly_recurring_charge"]["value"],
        "noc_ticket": known["noc_ticket"]["value"],
        "outage_resolved": known["outage_end"]["value"],
        "full_days": resolution["full_days"],
        "credit": resolution["credit"],
        "formula": resolution["formula"],
        "request_deadline": resolution["request_deadline"],
        "governing_rule": resolution["governing_rule"],
    }


def _computed_for(state):
    out = _computed(state["resolution"], state["known"])
    if state["gate"].get("requested_amount") is not None:
        out["customer_asked_for"] = f"${state['gate']['requested_amount']:,.2f} (requested, never approved)"
    return out


def context_tokens(docs):
    return sum(approx_tokens(d["body"]) for d in docs.values())


def build(kb, policy, all_facts, llm=None, router_llm=None, breaker=None):
    router = Router(kb)
    breaker = breaker or CircuitBreaker()

    def disabled(client):
        return " (llm disabled)" if client is not None and breaker.open else ""
    intents = {name: i["description"] for name, i in gate.INTENTS.items()}
    intents[gate.INFO] = gate.INFO_DESCRIPTION

    def policy_agent(state):
        scenario = facts_mod.apply_overrides(all_facts, state.get("fact_overrides"))
        known = facts_mod.as_of(scenario, state["time"])
        state["memory"].update_facts(known)
        return {"known": known, "resolution": resolver.resolve(known)}

    def router_agent(state):
        route = router.route(state["question"])
        small = breaker.allow(router_llm)
        if route["status"] == "ambiguous" and small is not None:
            try:
                chosen = small.route(state["question"], router.describe())
                route = {**route, "source": "llm", "categories": [] if chosen == [OUT_OF_SCOPE] else chosen,
                         "status": "out_of_scope" if chosen == [OUT_OF_SCOPE] else "routed"}
            except LLMError as e:
                breaker.trip("router_agent", e)
                route = {**route, "source": f"keywords (llm failed: {e})"}
        elif route["status"] == "ambiguous":
            route = {**route, "source": "keywords" + disabled(router_llm)}
        res = state["resolution"]
        route = router.with_case_context(route, [res["governing_rule"]] + res["escalation_rules"])
        routed_kb, routed_rules = router.select(route["categories"], kb, policy)
        return {"route": route, "routed_kb": routed_kb, "routed_rules": routed_rules}

    def after_router(state):
        return "intent_agent" if state["route"]["status"] == "routed" else "escalate_agent"

    def intent_agent(state):
        rules = gate.classify(state["question"])
        client = breaker.allow(llm)
        if client is None:
            return {"intent": rules, "intent_source": "rules" + disabled(llm)}
        try:
            intent = client.classify(state["question"], intents)
        except LLMError as e:
            breaker.trip("intent_agent", e)
            return {"intent": rules, "intent_source": f"rules (llm failed: {e})"}
        if intent != gate.INFO and gate.is_information_only(state["question"]):
            return {"intent": gate.INFO, "intent_source": "llm intent overridden by rule",
                    "llm_intent": intent}
        return {"intent": intent, "intent_source": "llm"}

    def after_intent(state):
        return "retrieval_agent" if state["intent"] in intents else "escalate_agent"

    def retrieval_agent(state):
        routed = state["routed_kb"]
        query = state["question"] + " " + gate.INTENTS.get(state["intent"], {}).get("expand", "")
        hits = Retriever(routed).search(query, k=len(routed))
        top = hits[:RETRIEVE_K]
        # Articles governing the case's rule (or its escalation rule) are always passed on.
        res = state["resolution"]
        rules = {r for r in [res["governing_rule"]] + res["escalation_rules"] if r}
        have, scores = {h["id"] for h in top}, {h["id"]: h["score"] for h in hits}
        top += [{"id": a, "title": d["title"], "score": scores.get(a, 0.0), "added": f"governs {d['governs']}"}
                for a, d in routed.items() if d.get("governs") in rules and a not in have]
        return {"retrieved": top}

    def gate_agent(state):
        return {"gate": gate.evaluate(state["intent"], state["question"], state["retrieved"],
                                      state["routed_kb"], state["resolution"], state["known"])}

    def escalate_agent(state):
        status = state["route"]["status"]
        update = {"retrieved": []}
        if status != "routed":  # never reached intent_agent: no classification, no LLM call
            update.update(intent=gate.UNKNOWN, intent_source=f"router ({status})")
        result = gate.evaluate(gate.UNKNOWN, state["question"], [], {}, state["resolution"], state["known"])
        return {**update, "gate": result, "segments": result["segments"], "reply_source": "template"}

    def responder_agent(state):
        result = state["gate"]
        grounded = [c for c in result["claims"] if c["status"] == "grounded"]
        client = breaker.allow(llm)
        if client is None or not grounded:
            return {"segments": result["segments"],
                    "reply_source": "template" + (disabled(llm) if grounded else "")}
        evidence = [{"claim_id": c["id"], "docs": c["evidence"]["docs"],
                     "quotes": [e["quote"] for e in c["evidence"]["all"]]} for c in grounded]
        try:
            segments = client.phrase(state["question"], result["segments"], evidence, _computed_for(state))
            return {"segments": segments, "reply_source": "llm"}
        except LLMError as e:
            breaker.trip("responder_agent", e)
            return {"segments": result["segments"], "reply_source": f"template (llm failed: {e})"}

    def verifier_agent(state):
        result, template = state["gate"], state["gate"]["segments"]
        check = verify(state["segments"], template, result, state["resolution"], state["known"])
        log = {"checked": state["reply_source"], **check}
        segments, source = state["segments"], state["reply_source"]
        if not check["passed"] and source == "llm":
            log["llm_attempt"] = {"segments": segments}
            segments, source = template, "template (verifier fallback)"
            log["fallback"] = verify(template, template, result, state["resolution"], state["known"])
        final_ok = check["passed"] or log.get("fallback", {}).get("passed", False)
        state["memory"].record(state["time"], state["route"]["categories"], state["intent"],
                               result["decision"], governing_for(state["intent"], state["resolution"]),
                               final_ok)
        return {"segments": segments, "reply_source": source, "verifier": log}

    g = StateGraph(TurnState)
    for name, fn in [("policy_agent", policy_agent), ("router_agent", router_agent),
                     ("intent_agent", intent_agent), ("retrieval_agent", retrieval_agent),
                     ("gate_agent", gate_agent), ("escalate_agent", escalate_agent),
                     ("responder_agent", responder_agent), ("verifier_agent", verifier_agent)]:
        g.add_node(name, fn)
    g.add_edge(START, "policy_agent")
    g.add_edge("policy_agent", "router_agent")
    g.add_conditional_edges("router_agent", after_router, ["intent_agent", "escalate_agent"])
    g.add_conditional_edges("intent_agent", after_intent, ["retrieval_agent", "escalate_agent"])
    g.add_edge("retrieval_agent", "gate_agent")
    g.add_edge("gate_agent", "responder_agent")
    g.add_edge("responder_agent", "verifier_agent")
    g.add_edge("escalate_agent", "verifier_agent")
    g.add_edge("verifier_agent", END)
    return g.compile()
