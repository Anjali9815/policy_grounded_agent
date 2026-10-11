"""Extract case facts from the raw chat transcript.

Every fact records where it came from and when in the chat it became known, so a
turn can only rely on facts established at or before that turn. Facts the
transcript does not contain are marked "unverified" and never assumed.
"""
import re
from datetime import date, datetime
from decimal import Decimal

from loader import DATA

TRANSCRIPT = DATA / "raw" / "NimbusCare Chat Transcript — Dana Whitfield, October 5, 2026.txt"

TURN_RE = re.compile(r"^\[(\d\d:\d\d)\] (Customer|Agent) \(([^)]+)\): (.*)$")
MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], 1)}


def fact(value, status, source, as_of=None):
    """status: header | customer_stated | agent_confirmed | unverified."""
    return {"value": value, "status": status, "source": source, "as_of": as_of}


def unverified(reason):
    return fact(None, "unverified", reason)


def read_transcript(path=TRANSCRIPT):
    return path.read_text(encoding="utf-8")


def turns(text):
    """Chat turns; a hard-wrapped turn's continuation lines are joined with a space."""
    out, current = [], None
    for line in text.splitlines():
        m = TURN_RE.match(line)
        if m:
            current = {"time": m[1], "role": m[2], "speaker": m[3], "text": m[4].strip()}
            out.append(current)
        elif not line.strip():
            current = None
        elif current is not None:
            current["text"] += " " + line.strip()
    return out


def customer_turns(text):
    return [t for t in turns(text) if t["role"] == "Customer"]


def _date(month_day, year):
    month, day = month_day.split()
    return date(year, MONTHS[month], int(re.sub(r"\D", "", day)))


def _datetime(month_day, clock, meridiem, year):
    hour, minute = map(int, clock.split(":"))
    hour = hour % 12 + (12 if meridiem == "PM" else 0)
    d = _date(month_day, year)
    return datetime(d.year, d.month, d.day, hour, minute)


def extract(text):
    facts = {
        # Not present anywhere in the transcript: never assume them clear.
        "prior_dispute_within_90_days": unverified("account dispute history not in transcript"),
        "fraud_indicators": unverified("fraud review flags not in transcript"),
        "rolling_12mo_goodwill_total": unverified("account credit history not in transcript"),
    }

    m = re.search(r"^Date: (\w+ \d+), (\d{4})$", text, re.M)
    chat_date = _date(m[1], int(m[2]))
    year = chat_date.year
    facts["chat_date"] = fact(chat_date, "header", "transcript header 'Date:'")

    m = re.search(r"^Account Number: (\S+)$", text, re.M)
    facts["account_number"] = fact(m[1], "header", "transcript header 'Account Number:'")

    m = re.search(r"^Plan: (.+?) — \$([\d,]+\.\d{2})/month recurring charge$", text, re.M)
    facts["plan_name"] = fact(m[1], "header", "transcript header 'Plan:'")
    facts["monthly_recurring_charge"] = fact(
        Decimal(m[2].replace(",", "")), "header", "transcript header 'Plan:'")

    for t in turns(text):
        src = f"[{t['time']}] {t['role']}"
        if t["role"] == "Agent":
            if re.search(r"you're verified", t["text"]):
                facts["identity_verified"] = fact(True, "agent_confirmed", src, t["time"])
            m = re.search(r"confirmed outage ticket, (OUT-\d{5})", t["text"])
            if m:
                facts["noc_ticket"] = fact(m[1], "agent_confirmed", src, t["time"])
                c = re.search(r"covering (\w+ \d+), (\d+:\d\d) (AM|PM) through "
                              r"(\w+ \d+), (\d+:\d\d) (AM|PM)", t["text"])
                facts["outage_start"] = fact(_datetime(c[1], c[2], c[3], year),
                                             "agent_confirmed", src + " (NOC ticket)", t["time"])
                facts["outage_end"] = fact(_datetime(c[4], c[5], c[6], year),
                                           "agent_confirmed", src + " (NOC ticket)", t["time"])
        else:
            m = re.search(r"invoice dated (\w+ \d+)(?:st|nd|rd|th)?", t["text"])
            if m:
                facts["statement_date"] = fact(_date(m[1], year), "customer_stated", src, t["time"])
            m = re.search(r"disputing the full \$([\d,]+\.\d{2}) recurring charge", t["text"])
            if m:
                facts["disputed_line_item"] = fact(
                    f"${m[1]} recurring charge", "customer_stated", src, t["time"])

    for key in ("noc_ticket", "outage_start", "outage_end", "statement_date", "identity_verified"):
        facts.setdefault(key, unverified("not found in transcript"))
    return facts


def as_of(facts, time):
    """Facts as known at chat time 'HH:MM'; later facts become unverified."""
    out = {}
    for key, f in facts.items():
        if f["as_of"] and f["as_of"] > time:
            out[key] = unverified(f"not yet established at {time} (known from {f['as_of']})")
        else:
            out[key] = f
    return out
