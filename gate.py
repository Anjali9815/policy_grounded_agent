"""Grounding gate: answer a claim only if a retrieved article explicitly states it.

Each intent lists the claims an answer would have to make. A claim is grounded
when its pattern matches the text of a retrieved article; the matching sentence
becomes the quoted evidence. Ungrounded claims are declined and escalated to
Tier 2, so one question can be partly answered and partly escalated. A claim
with a "when" pattern is only required if the question asks about it. Any
question that matches no intent is escalated, never answered.
"""
import re

INTENTS = {
    "credit_limit": {
        "description": "Asks whether a refund/credit limit (e.g. $75) caps the credit, or how much the credit is.",
        "match": r"\$75|refund limit|\blimit\b",
        "expand": "service outage credit ceiling goodwill limit",
        "governing": "case",
        "claims": [
            {"id": "outage_credit_not_capped",
             "claim": "A service-outage credit is not limited by the $75.00 goodwill ceiling.",
             "pattern": r"not limited by, the standard \$75\.00 goodwill credit ceiling",
             "topic": ["75"],
             "reply": 'The $75.00 limit is not the one that applies to an outage credit. Our policy says: '
                      '"{quote}" [{doc}].'},
            {"id": "outage_credit_formula",
             "claim": "The outage credit is (monthly recurring charge ÷ 30) × full days without service.",
             "pattern": r"\(current monthly recurring charge ÷ 30\) × number of full days",
             "topic": ["formula"],
             "reply": "For a confirmed outage the credit is {quote} [{doc}]. With ticket {ticket} that is "
                      "{formula}, under Rule 3."},
        ],
    },
    "tax_treatment": {
        "description": "Asks whether the credit affects taxes, surcharges or which charges it applies to.",
        "match": r"\btax|surcharge",
        "expand": "outage credit plan charge taxes surcharges",
        "governing": "case",
        "claims": [
            {"id": "credit_plan_charge_only",
             "claim": "The outage credit applies to the recurring plan charge.",
             "when": r"plan charge|only applied|applied against",
             "pattern": r"applies to the recurring plan charge only",
             "topic": ["plan charge"],
             "reply": 'The outage credit is calculated on your plan charge: "{quote}" [{doc}].'},
            {"id": "credit_tax_treatment",
             "claim": "Whether an outage credit also reduces taxes and regulatory surcharges.",
             "pattern": r"outage credit[^.\n]*\b(tax|surcharge)|\b(tax|surcharge)[^.\n]*outage credit",
             "topic": ["tax", "surcharge"],
             "decline": "Our guidance doesn't say whether the credit also reduces the taxes and "
                        "regulatory surcharges, so I've escalated that question to a Tier 2 "
                        "Billing Specialist rather than guess."},
        ],
    },
    "request_deadline": {
        "description": "Asks how long they have, or the deadline, to request the credit or dispute.",
        "match": r"how long|deadline",
        "expand": "outage credit request within days resolved resolution",
        "governing": "case",
        "claims": [
            {"id": "outage_request_window",
             "claim": "An outage credit must be requested within 30 days of the outage's resolution.",
             "pattern": r"within 30 days of (the outage being resolved|the ticket's resolution date)",
             "topic": ["30 days"],
             "reply": 'You need to request it {quote_match} [{doc}]. Your outage was resolved on '
                      '{resolved}, so the deadline is {deadline}. The 60-day billing-dispute window '
                      'from your statement date is not the deadline for an outage credit [R0, R1].'},
        ],
    },
    "open_dispute": {
        "description": "Wants to start or file a dispute about a charge on their bill.",
        "match": r"\bdispute (a|this|the) charge|\bneed to dispute",
        "expand": "file billing dispute identity line item",
        "governing": "R1",
        "claims": [
            {"id": "verify_identity",
             "claim": "To open a dispute, the account holder's identity is confirmed first.",
             "pattern": r"Confirm the account holder's identity",
             "topic": ["identity"],
             "reply": "I can open a dispute for you. First: {quote} [{doc}]."},
            {"id": "identify_line_item",
             "claim": "The dispute must name the specific line item being disputed.",
             "pattern": r"Identify the exact line item being disputed",
             "topic": ["line item"],
             "reply": "Then: {quote} [{doc}]."},
        ],
    },
}
INFO = "provide_information"
UNKNOWN = "unknown"
INFO_DESCRIPTION = "Only supplies account or case details (account number, dates, invoice), no question."

# A statement with no question that only supplies account or case details.
INFO_MATCH = r"\bNMB-\d+|card ending|went down on|wasn't back until|invoice dated"


def classify(text):
    for name, intent in INTENTS.items():
        if re.search(intent["match"], text, re.I):
            return name
    if "?" not in text and re.search(INFO_MATCH, text, re.I):
        return INFO
    return UNKNOWN


def _plain(body):
    """Body text as an article states it: markdown emphasis and code ticks removed."""
    return re.sub(r"\*\*|`", "", body)


def _sentence(text, start, end):
    left = max(text.rfind(". ", 0, start) + 2, text.rfind("\n", 0, start) + 1)
    stops = [i for i in (text.find(". ", end), text.find("\n", end)) if i != -1]
    right = min(stops) if stops else len(text)
    return text[left:right].strip().rstrip(".")


def ground(claim, retrieved, kb):
    """Evidence from every retrieved article that states the claim, best-ranked first."""
    found = []
    for hit in retrieved:
        text = _plain(kb[hit["id"]]["body"])
        m = re.search(claim["pattern"], text, re.I)
        if m:
            found.append({"doc": hit["id"], "quote": _sentence(text, m.start(), m.end()), "match": m[0]})
    if not found:
        return None
    return {**found[0], "docs": [f["doc"] for f in found], "all": found}


def related(claim, retrieved, kb):
    """Retrieved articles that touch the claim's topic but do not state the claim."""
    return [h["id"] for h in retrieved
            if any(re.search(rf"\b{t}", _plain(kb[h["id"]]["body"]), re.I) for t in claim["topic"])]


def _seg(claim_id, text):
    return {"claim_id": claim_id, "text": text}


def _result(claims, decision, segments, escalate):
    return {"claims": claims, "decision": decision, "segments": segments,
            "reply": " ".join(s["text"] for s in segments), "escalate": escalate}


def evaluate(intent_name, question, retrieved, kb, resolution, facts):
    """Return grounded/declined claims, a decision, and the template reply.

    The reply is built from segments, each tagged with the claim (or notice) it
    states, so a verifier can check every sentence against its source.
    """
    if intent_name == INFO:
        return _result([], "NO_QUESTION", [_seg("notice.info", "Thanks, noted.")], False)
    if intent_name not in INTENTS:
        return _result([], "ESCALATE", [_seg(
            "notice.unknown", "I don't have guidance I can rely on to answer that, so I've passed it to "
                              "a Tier 2 Billing Specialist rather than guess.")], True)

    intent = INTENTS[intent_name]
    ctx = {
        "ticket": facts["noc_ticket"]["value"],
        "formula": resolution["formula"],
        "deadline": resolution["request_deadline"],
        "resolved": facts["outage_end"]["value"].date() if facts["outage_end"]["value"] else None,
    }
    claims, segments = [], []
    required = [c for c in intent["claims"] if not c.get("when") or re.search(c["when"], question, re.I)]
    for claim in required:
        ev = ground(claim, retrieved, kb)
        needs_case = intent["governing"] == "case" and resolution["governing_rule"] != "R3"
        if ev and not needs_case:
            claims.append({"id": claim["id"], "claim": claim["claim"], "status": "grounded", "evidence": ev})
            segments.append(_seg(claim["id"], claim["reply"].format(
                quote=ev["quote"], quote_match=ev["match"], doc=", ".join(ev["docs"]), **ctx)))
        else:
            reason = ("no retrieved article states this claim" if not ev
                      else "Rule 3 is not established for this case yet")
            claims.append({"id": claim["id"], "claim": claim["claim"], "status": "declined",
                           "reason": reason, "related_not_answering": related(claim, retrieved, kb)})
            segments.append(_seg(claim["id"], claim.get(
                "decline", f"I can't confirm that from our guidance yet ({claim['claim']}), so I've "
                           f"escalated it to a Tier 2 Billing Specialist.")))

    grounded = sum(c["status"] == "grounded" for c in claims)
    if not claims:
        decision = "ESCALATE"
    elif resolution["governing_rule"] == "R5" and intent["governing"] == "case":
        decision = "ESCALATE"
        segments = [_seg("notice.r5_escalation", "Because of your account's dispute history or a fraud "
                                                 "review, a Tier 2 Billing Specialist has to make this "
                                                 "decision [R5].")]
    elif grounded == len(claims):
        decision = "ANSWER"
    elif grounded:
        decision = "PARTIAL"
    else:
        decision = "ESCALATE"

    if intent_name == "credit_limit" and resolution["pending_verification"] and decision != "ESCALATE":
        segments.append(_seg("notice.r5_pending",
                             "Before the credit is applied, your account's dispute history and fraud "
                             "review still need to be checked [R5]; that isn't shown in this "
                             "conversation, so I'm not assuming it's clear."))
    if resolution["escalate"] and intent["governing"] == "case" and decision != "ESCALATE":
        segments.append(_seg("notice.r4_approval",
                             "This credit needs Tier 2 Billing Specialist approval before it is issued [R4]."))

    return _result(claims, decision, segments,
                   decision in ("PARTIAL", "ESCALATE") or resolution["escalate"])
