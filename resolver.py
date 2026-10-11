"""Deterministic policy selection and credit calculation.

Order: R5 (repeat dispute / fraud) first and overrides everything; then R3 if all
of its conditions hold (R3 is the more specific rule, so it governs instead of R2
per R0); otherwise R2. Escalation for an R3 credit is checked only against R4(b);
an R3 credit is never compared to the R2/R4(a) $75.00 ceiling. A ticket that
exists but cannot be verified keeps R3 and fails R4(b). Under R2 no amount is
computed: the outage formula belongs to R3, and R0 forbids blending two rules
into one calculation, so R2 only checks the rolling 12-month goodwill total.
"""
from datetime import timedelta
from decimal import ROUND_HALF_UP, Decimal

GOODWILL_CAP = Decimal("75.00")      # R2 / R4(a)
OUTAGE_CREDIT_LIMIT = Decimal("300.00")  # R4(b)
OUTAGE_MAX_DAYS = 30                 # R4(b)
OUTAGE_MIN_HOURS = 4                 # R3
OUTAGE_REQUEST_DAYS = 30             # R3
DISPUTE_WINDOW_DAYS = 60             # R1


def _known(f):
    return f["status"] != "unverified"


def _prorate(charge, days):
    return (charge / 30 * days).quantize(Decimal("0.01"), ROUND_HALF_UP)


def resolve(facts):
    checks = []

    def check(check_id, rule, status, detail):
        checks.append({"id": check_id, "rule": rule, "status": status, "detail": detail})
        return status

    # R5: repeat dispute / fraud. A True value is a failure; missing data is unverified.
    r5 = []
    for key, label in (("prior_dispute_within_90_days", "no second dispute within 90 days of a resolved one"),
                       ("fraud_indicators", "no fraud indicators")):
        f = facts[key]
        if not _known(f):
            r5.append(check(f"R5.{key}", "R5", "unverified", f"{label}: {f['source']}"))
        else:
            r5.append(check(f"R5.{key}", "R5", "failed" if f["value"] else "passed", label))

    # R1: dispute filed within 60 days of the statement date.
    window_end = None
    stmt, chat = facts["statement_date"], facts["chat_date"]["value"]
    if _known(stmt):
        window_end = stmt["value"] + timedelta(days=DISPUTE_WINDOW_DAYS)
        check("R1.within_60_days_of_statement", "R1", "passed" if chat <= window_end else "failed",
              f"statement {stmt['value']}, window ends {window_end}, chat {chat}")
    else:
        check("R1.within_60_days_of_statement", "R1", "unverified", stmt["source"])

    # R3 conditions.
    ticket, start, end = facts["noc_ticket"], facts["outage_start"], facts["outage_end"]
    r3, hours, full_days, deadline = [], None, None, None
    if not _known(ticket):
        r3.append(check("R3.noc_ticket", "R3", "unverified", ticket["source"]))
    elif not ticket["value"]:  # looked up and none exists
        r3.append(check("R3.noc_ticket", "R3", "failed", f"no NOC outage ticket found ({ticket['source']})"))
    else:
        r3.append(check("R3.noc_ticket", "R3", "passed", f"{ticket['value']} ({ticket['source']})"))
    if _known(start) and _known(end):
        hours = (end["value"] - start["value"]).total_seconds() / 3600
        full_days = int(hours // 24)
        deadline = end["value"].date() + timedelta(days=OUTAGE_REQUEST_DAYS)
        r3.append(check("R3.outage_over_4_hours", "R3", "passed" if hours > OUTAGE_MIN_HOURS else "failed",
                        f"{start['value']} to {end['value']} = {hours:g} hours"))
        r3.append(check("R3.within_30_days_of_resolution", "R3", "passed" if chat <= deadline else "failed",
                        f"resolved {end['value'].date()}, deadline {deadline}, requested {chat}"))
    else:
        r3.append(check("R3.outage_over_4_hours", "R3", "unverified", "outage dates not established"))
        r3.append(check("R3.within_30_days_of_resolution", "R3", "unverified", "outage dates not established"))

    result = {
        "governing_rule": None, "citations": [], "checks": checks, "not_applied": [],
        "credit": None, "formula": None, "full_days": full_days, "outage_hours": hours,
        "request_deadline": deadline, "dispute_window_end": window_end,
        "escalate": False, "escalation_reasons": [], "escalation_rules": [], "pending_verification": [],
        "may_apply": False,
    }

    if "failed" in r5:
        result.update(governing_rule="R5", citations=["R5"], escalate=True,
                      escalation_reasons=["R5: repeat dispute or fraud indicator; credit decision is Tier 2's"])
        return result

    charge = facts["monthly_recurring_charge"]["value"]
    ticket_status = r3[0]
    # R3 governs when every condition holds; a ticket that exists but can't be verified is R4(b)'s case.
    if ticket_status in ("passed", "unverified") and all(s == "passed" for s in r3[1:]):
        credit = _prorate(charge, full_days)
        result.update(governing_rule="R3", citations=["R3", "R0"], credit=credit,
                      formula=f"(${charge} ÷ 30) × {full_days} full days = ${credit}")
        result["not_applied"] = [
            {"rule": "R2", "reason": "R3 is the more specific rule for a NOC-confirmed outage (R0); "
                                     "the R2 $75.00 goodwill ceiling does not cap this credit"},
            {"rule": "R4a", "reason": "R4(a) applies only to R2 credits; comparing an R3 credit "
                                      "to $75.00 is a misapplication (R4 note)"},
        ]
        if deadline and window_end:
            result["not_applied"].append(
                {"rule": "R1", "reason": f"the R1 60-day window ({window_end}) is not the governing "
                                         f"deadline for an outage credit; R3 sets {deadline}"})
        r4b = [
            check("R4b.credit_not_over_300", "R4", "passed" if credit <= OUTAGE_CREDIT_LIMIT else "failed",
                  f"${credit} vs ${OUTAGE_CREDIT_LIMIT}"),
            check("R4b.outage_not_over_30_days", "R4", "passed" if full_days <= OUTAGE_MAX_DAYS else "failed",
                  f"{full_days} days vs {OUTAGE_MAX_DAYS}"),
            check("R4b.ticket_verified", "R4", "passed", f"{ticket['value']} confirmed by agent lookup")
            if ticket_status == "passed" else
            check("R4b.ticket_verified", "R4", "failed", "NOC ticket number cannot be verified at the time of the request"),
        ]
        if "failed" in r4b:
            failed = [c["id"] for c in checks if c["rule"] == "R4" and c["status"] == "failed"]
            result.update(escalate=True, escalation_rules=["R4"])
            result["escalation_reasons"].append(f"R4(b): {', '.join(failed)}")
    elif "failed" in r3:
        # R3 conditions not met: a general goodwill request under R2, capped at $75.00 (R3, last paragraph).
        failed = [c["id"] for c in checks if c["rule"] == "R3" and c["status"] == "failed"]
        result.update(governing_rule="R2", citations=["R2", "R3"])
        result["not_applied"] = [{"rule": "R3", "reason": f"R3 conditions not met ({', '.join(failed)}); "
                                                          f"evaluated under R2 as general goodwill"}]
        prior = facts["rolling_12mo_goodwill_total"]
        if not _known(prior):
            check("R2.rolling_12_month_total", "R2", "unverified",
                  f"account credit history: {prior['source']}")
            result["escalate"] = True
            result["escalation_reasons"].append(
                "R2: rolling 12-month goodwill total not verified; the credit amount goes to Tier 2")
        elif prior["value"] >= GOODWILL_CAP:
            check("R2.rolling_12_month_total", "R2", "failed",
                  f"${prior['value']} already issued in the last 12 months vs ${GOODWILL_CAP}")
            result.update(escalate=True, escalation_rules=["R4"])
            result["escalation_reasons"].append("R4(a): any further goodwill credit would exceed $75.00")
        else:
            check("R2.rolling_12_month_total", "R2", "passed",
                  f"${prior['value']} issued in the last 12 months; ${GOODWILL_CAP - prior['value']} left")
    else:
        return result  # R3 conditions not yet established; no governing rule.

    result["pending_verification"] = [c["detail"] for c in checks if c["rule"] == "R5" and c["status"] == "unverified"]
    result["may_apply"] = not result["escalate"] and not result["pending_verification"]
    return result
