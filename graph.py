"""LangGraph workflow: one invocation per customer turn.

policy_agent -> router_agent -> (out of scope / unrouted) ------------------> escalate_agent
                             -> intent_agent -> (unknown) -------------------> escalate_agent
                                             -> retrieval_agent -> gate_agent -> responder_agent
escalate_agent, responder_agent -> verifier_agent

policy_agent runs first: it needs only the case facts (no KB) and gives the
router the case's governing rule. router_agent narrows the KB to the routed
categories; retrieval, gate and responder see only those articles and the rules
they govern. LLMs (router fallback, intent, responder) are optional and every
one falls back to the deterministic path; verifier_agent replaces an LLM reply
that fails verification with the template reply.
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import facts as facts_mod
import gate
import resolver
from llm import OUT_OF_SCOPE, LLMError
from memory import CaseMemory
from retriever import Retriever
from router import Router, approx_tokens
from verifier import verify


class TurnState(TypedDict, total=False):
    question: str
    time: str
    memory: CaseMemory
    known: dict
    resolution: dict
    route: dict
    routed_kb: dict
    routed_rules: dict
    intent: str
    intent_source: str
    retrieved: list
    gate: dict
    segments: list
    reply_source: str
    verifier: dict


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


def context_tokens(docs):
    return sum(approx_tokens(d["body"]) for d in docs.values())


def build(kb, policy, all_facts, llm=None, router_llm=None):
    router = Router(kb)
    intents = {name: i["description"] for name, i in gate.INTENTS.items()}
    intents[gate.INFO] = gate.INFO_DESCRIPTION

    def policy_agent(state):
        known = facts_mod.as_of(all_facts, state["time"])
        state["memory"].update_facts(known)
        return {"known": known, "resolution": resolver.resolve(known)}

    def router_agent(state):
        route = router.route(state["question"])
        if route["status"] == "ambiguous" and router_llm is not None:
            try:
                chosen = router_llm.route(state["question"], router.describe())
                route = {**route, "source": "llm", "categories": [] if chosen == [OUT_OF_SCOPE] else chosen,
                         "status": "out_of_scope" if chosen == [OUT_OF_SCOPE] else "routed"}
            except LLMError as e:
                route = {**route, "source": f"keywords (llm failed: {e})"}
        route = router.with_case_context(route, state["resolution"]["governing_rule"])
        routed_kb, routed_rules = router.select(route["categories"], kb, policy)
        return {"route": route, "routed_kb": routed_kb, "routed_rules": routed_rules}

    def after_router(state):
        return "intent_agent" if state["route"]["status"] == "routed" else "escalate_agent"

    def intent_agent(state):
        rules = gate.classify(state["question"])
        if llm is None:
            return {"intent": rules, "intent_source": "rules"}
        try:
            return {"intent": llm.classify(state["question"], intents), "intent_source": "llm"}
        except LLMError as e:
            return {"intent": rules, "intent_source": f"rules (llm failed: {e})"}

    def after_intent(state):
        return "retrieval_agent" if state["intent"] in intents else "escalate_agent"

    def retrieval_agent(state):
        query = state["question"] + " " + gate.INTENTS.get(state["intent"], {}).get("expand", "")
        return {"retrieved": Retriever(state["routed_kb"]).search(query)}

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
        if llm is None or not grounded:
            return {"segments": result["segments"], "reply_source": "template"}
        evidence = [{"claim_id": c["id"], "docs": c["evidence"]["docs"],
                     "quotes": [e["quote"] for e in c["evidence"]["all"]]} for c in grounded]
        try:
            segments = llm.phrase(state["question"], result["segments"], evidence,
                                  _computed(state["resolution"], state["known"]))
            return {"segments": segments, "reply_source": "llm"}
        except LLMError as e:
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
