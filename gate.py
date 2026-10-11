"""Grounding gate: answer a claim only if a retrieved article explicitly states it.

Each intent lists the claims an answer would have to make. A claim is grounded
when its pattern matches the text of a retrieved article; the matching sentence
becomes the quoted evidence. Ungrounded claims are declined and escalated to
Tier 2, so one question can be partly answered and partly escalated. A claim
with a "when" pattern is only required if the question asks about it; a claim
with "rules" applies only when the case falls under one of those rules (and
"if" adds a condition on the resolution). Any question that matches no intent
is escalated, never answered. When R5 governs, the R5 claims replace the
intent's claims and the turn always escalates with no credit issued.
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
             "rules": ["R3"],
             "reply": 'The $75.00 limit is not the one that applies to an outage credit. Our policy says: '
                      '"{quote}" [{doc}].'},
            {"id": "outage_credit_formula",
             "claim": "The outage credit is (monthly recurring charge ÷ 30) × full days without service.",
             "pattern": r"\(current monthly recurring charge ÷ 30\) × number of full days",
             "topic": ["formula"],
             "rules": ["R3"],
             "reply": "For a confirmed outage the credit is {quote} [{doc}]. {credit_sentence}"},
            {"id": "r4b_escalation",
             "claim": "An outage credit over $300, an outage over 30 days or an unverifiable ticket goes to Tier 2.",
             "pattern": r"the NOC ticket can't be verified",
             "topic": ["300"],
             "rules": ["R3"],
             "if": lambda r: "R4" in r["escalation_rules"],
             "reply": "Because {r4b_reason}, this credit has to be approved by a Tier 2 Billing Specialist "
                      "before it is issued. Our policy escalates when: \"{quote}\" [{doc}]."},
            {"id": "outage_needs_over_4_hours",
             "claim": "The service-outage credit requires an outage of more than 4 consecutive hours.",
             "pattern": r"service outage lasting more than 4 consecutive hours",
             "topic": ["4 consecutive hours"],
             "rules": ["R2"],
             "if": lambda r: _status(r, "R3.outage_over_4_hours") == "failed",
             "reply": "The service-outage credit doesn't apply here: it needs an outage of more than 4 "
                      "consecutive hours, and this one lasted {outage_hours} hours. Our policy: "
                      "\"{quote}\" [{doc}]."},
            {"id": "goodwill_cap",
             "claim": "General goodwill credits are capped at $75.00 per account per rolling 12 months.",
             "pattern": r"up to \$75\.00 per account per rolling 12-month period",
             "topic": ["75"],
             "rules": ["R2"],
             "reply": "Without a qualifying service-outage credit, the general goodwill limit governs: a "
                      "refund or credit of {quote_match} [{doc}]."},
            {"id": "goodwill_history_check",
             "claim": "The $75.00 ceiling is cumulative, so credit history is checked before quoting an amount.",
             "pattern": r"check the account's credit history before quoting an amount",
             "topic": ["credit history"],
             "rules": ["R2"],
             "reply": 'That limit is cumulative: "{quote}" [{doc}].'},
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
             "rules": ["R3"],
             "reply": 'The outage credit is calculated on your plan charge: "{quote}" [{doc}].'},
            {"id": "credit_tax_treatment",
             "claim": "Whether an outage credit also reduces taxes and regulatory surcharges.",
             "pattern": r"outage credit[^.\n]*\b(tax|surcharge)|\b(tax|surcharge)[^.\n]*outage credit",
             "topic": ["tax", "surcharge"],
             "rules": ["R3"],
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
             "rules": ["R3"],
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
# Replace an intent's claims when R5 governs; "fact" limits a claim to cases where that fact is set.
R5_CLAIMS = [
    {"id": "r5_repeat_dispute", "fact": "prior_dispute_within_90_days",
     "claim": "A second dispute within 90 days of a resolved one must be handled by Tier 2.",
     "pattern": r"the new dispute must be handled by a Tier 2 Billing Specialist",
     "topic": ["90 days"],
     "reply": 'This account has another billing dispute within 90 days of one that was resolved, and '
              'our policy says: "{quote}" [{doc}].'},
    {"id": "r5_fraud", "fact": "fraud_indicators",
     "claim": "Disputes with fraud indicators must be handled by Tier 2.",
     "pattern": r"same applies if fraud indicators are present",
     "topic": ["fraud"],
     "reply": 'This account has a fraud review flag, and our policy says: "{quote}" [{doc}].'},
    {"id": "r5_no_frontline_credit",
     "claim": "Front-line agents do not issue the credit in these cases.",
     "pattern": r"should not issue a credit themselves",
     "topic": ["credit"],
     "reply": 'So no credit is issued by me: "{quote}" [{doc}]. A Tier 2 Billing Specialist will make '
              'the decision.'},
]

def _status(resolution, check_id):
    return next((c["status"] for c in resolution["checks"] if c["id"] == check_id), None)


def _r4b_reason(resolution):
    failed = {c["id"] for c in resolution["checks"] if c["rule"] == "R4" and c["status"] == "failed"}
    reasons = []
    if "R4b.credit_not_over_300" in failed:
        reasons.append(f"the credit of ${resolution['credit']} is over $300.00")
    if "R4b.outage_not_over_30_days" in failed:
        reasons.append(f"the outage ran {resolution['full_days']} full days")
    if "R4b.ticket_verified" in failed:
        reasons.append("the outage ticket can't be verified right now")
    return " and ".join(reasons) or "an escalation threshold applies"


INFO = "provide_information"
UNKNOWN = "unknown"
INFO_DESCRIPTION = "Only supplies account or case details (account number, dates, invoice), no question."

# A statement with no question that only supplies account or case details.
INFO_MATCH = r"\bNMB-\d+|card ending|went down on|wasn't back until|invoice dated"
# Wh-words anywhere, or an auxiliary opening a sentence ("Can you...", "Is it..."); a
# mid-sentence "is" ("Account is NMB-...") is a statement, not a question.
QUESTION_WORDS = (r"\b(what|when|where|why|how|who|whom|whose|which)\b"
                  r"|(^|[.!;]\s+)(can|could|do|does|did|is|are|was|were|will|would|should|may|am)\b")


def is_information_only(text):
    """No question (no '?' and no question words) and matches a provide-information pattern."""
    return ("?" not in text and not re.search(QUESTION_WORDS, text.strip(), re.I)
            and bool(re.search(INFO_MATCH, text, re.I)))


def classify(text):
    for name, intent in INTENTS.items():
        if re.search(intent["match"], text, re.I):
            return name
    if is_information_only(text):
        return INFO
    return UNKNOWN


def _plain(body):
    """Body text as an article states it: markdown emphasis and code ticks removed."""
    return re.sub(r"\*+|`", "", body)


def _sentence(text, start, end):
    left = max(text.rfind(". ", 0, start) + 2, text.rfind("\n", 0, start) + 1)
    stops = [i for i in (text.find(". ", end), text.find("\n", end)) if i != -1]
    right = min(stops) if stops else len(text)
    return re.sub(r"^[-*]\s+", "", text[left:right].strip()).rstrip(".:")


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
    gov, case = resolution["governing_rule"], intent["governing"] == "case"
    ctx = {
        "ticket": facts["noc_ticket"]["value"],
        # Without a verified ticket the amount rests on the customer's dates, so it is only an estimate.
        "credit_sentence": (f"With ticket {facts['noc_ticket']['value']} that is {resolution['formula']}, "
                            f"under Rule 3." if facts["noc_ticket"]["value"] else
                            f"Based on the outage dates you gave, the estimated credit is {resolution['formula']}. "
                            f"This is an estimate, pending verification of the outage ticket by a Tier 2 "
                            f"Billing Specialist."),
        "formula": resolution["formula"],
        "deadline": resolution["request_deadline"],
        "resolved": facts["outage_end"]["value"].date() if facts["outage_end"]["value"] else None,
        "outage_hours": f"{resolution['outage_hours']:g}" if resolution["outage_hours"] is not None else None,
        "r4b_reason": _r4b_reason(resolution),
    }
    if case and gov == "R5":
        return _evaluate_r5(retrieved, kb, facts, ctx)

    required = [c for c in intent["claims"]
                if (not c.get("when") or re.search(c["when"], question, re.I))
                and (not case or gov is None or gov in c.get("rules", [gov]))
                and c.get("if", lambda r: True)(resolution)]
    claims, segments = _ground_all(required, retrieved, kb, ctx,
                                   None if not case or gov else "the governing rule is not established for this case yet")

    grounded = sum(c["status"] == "grounded" for c in claims)
    if not claims:
        decision = "ESCALATE"
        segments = [_seg("notice.no_claims", "I can't confirm that for your case from our guidance, so I've "
                                             "passed it to a Tier 2 Billing Specialist rather than guess.")]
    elif grounded == len(claims):
        # Everything asked is grounded, but if the credit itself must go to Tier 2 the turn is partial.
        decision = "PARTIAL" if case and resolution["escalate"] else "ANSWER"
    elif grounded:
        decision = "PARTIAL"
    else:
        decision = "ESCALATE"

    if intent_name == "credit_limit" and decision != "ESCALATE":
        pending = {c["rule"] for c in resolution["checks"] if c["status"] == "unverified"
                   and c["rule"] in ("R5", "R4")}
        if "R5" in pending and resolution["credit"]:
            segments.append(_seg("notice.r5_pending",
                                 "Before the credit is applied, your account's dispute history and fraud "
                                 "review still need to be checked [R5]; that isn't shown in this "
                                 "conversation, so I'm not assuming it's clear."))
        if "R4" in pending:
            segments.append(_seg("notice.r4a_pending",
                                 "Your account's goodwill credits over the last 12 months also need to be "
                                 "checked against the cumulative limit before anything is issued [R4]."))
    if case and gov == "R2" and _status(resolution, "R2.rolling_12_month_total") == "unverified" \
            and decision != "ESCALATE":
        segments.append(_seg("notice.r2_history",
                             "Your account's credit history isn't available in this conversation, so I can't "
                             "quote an amount; I've passed that to a Tier 2 Billing Specialist [R2]."))
    grounded_ids = {c["id"] for c in claims if c["status"] == "grounded"}
    if case and resolution["escalate"] and decision != "ESCALATE" and gov != "R2" \
            and "r4b_escalation" not in grounded_ids:
        segments.append(_seg("notice.r4_approval",
                             "This credit needs Tier 2 Billing Specialist approval before it is issued [R4]."))

    return _result(claims, decision, segments,
                   decision in ("PARTIAL", "ESCALATE") or resolution["escalate"])


def _ground_all(required, retrieved, kb, ctx, blocked_reason=None):
    claims, segments = [], []
    for claim in required:
        ev = None if blocked_reason else ground(claim, retrieved, kb)
        if ev:
            claims.append({"id": claim["id"], "claim": claim["claim"], "status": "grounded", "evidence": ev})
            segments.append(_seg(claim["id"], claim["reply"].format(
                quote=ev["quote"], quote_match=ev["match"], doc=", ".join(ev["docs"]), **ctx)))
        else:
            claims.append({"id": claim["id"], "claim": claim["claim"], "status": "declined",
                           "reason": blocked_reason or "no retrieved article states this claim",
                           "related_not_answering": related(claim, retrieved, kb)})
            segments.append(_seg(claim["id"], claim.get(
                "decline", f"I can't confirm that from our guidance yet ({claim['claim']}), so I've "
                           f"escalated it to a Tier 2 Billing Specialist.")))
    return claims, segments


def _evaluate_r5(retrieved, kb, facts, ctx):
    """R5 overrides every other rule: explain why from the KB, issue nothing, escalate."""
    required = [c for c in R5_CLAIMS if not c.get("fact") or facts[c["fact"]]["value"]]
    claims, segments = _ground_all(required, retrieved, kb, ctx)
    if not any(c["status"] == "grounded" for c in claims):
        segments = [_seg("notice.r5_escalation", "Because of your account's dispute history or a fraud "
                                                 "review, a Tier 2 Billing Specialist has to make this "
                                                 "decision [R5].")]
    return _result(claims, "ESCALATE", segments, True)
