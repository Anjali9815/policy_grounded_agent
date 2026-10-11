"""Check a reply against its evidence before it is sent.

A reply passes only if every number, date and dollar amount in it appears in the
cited evidence or the computed case facts, every segment maps to a claim or
notice from the template, no escalation or notice was dropped, and grounded
segments still cite their article.
"""
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

import resolver

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
# "one" is left out: it is almost always a pronoun ("the one that applies").
NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero _ two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split()) if w != "_"}
NUMBER_WORDS.update({"thirty": "30", "sixty": "60", "ninety": "90", "hundred": "100"})

# Citations and rule/tier names carry digits that are labels, not values.
LABELS = re.compile(r"\[[^\]]*\]|\b(?:Rule|Tier)\s+\d+(?:\([a-z]\))?|\bR\d(?:\([a-z]\))?|\bKB-\d+", re.I)
ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
NAMED_DATE = re.compile(rf"\b({'|'.join(MONTHS)})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b", re.I)
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _num(token):
    try:
        return format(Decimal(token.replace(",", "")).normalize(), "f")
    except InvalidOperation:
        return token


def values(text):
    """Normalized values in text: 'date:YYYY-MM-DD', 'md:MM-DD' (no year) and numbers."""
    out = set()
    text = LABELS.sub(" ", text)

    def iso(m):
        out.add(f"date:{m[1]}-{m[2]}-{m[3]}")
        return " "

    def named(m):
        md = f"{MONTHS.index(m[1].lower()) + 1:02d}-{int(m[2]):02d}"
        out.add(f"date:{m[3]}-{md}" if m[3] else f"md:{md}")
        return " "

    text = NAMED_DATE.sub(named, ISO_DATE.sub(iso, text))
    out.update(_num(t) for t in NUMBER.findall(text))
    out.update(NUMBER_WORDS[w] for w in re.findall(r"[a-z]+", text.lower()) if w in NUMBER_WORDS)
    return out


def _fact_text(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat(sep=" ") if isinstance(value, datetime) else value.isoformat()
    return str(value)


def allowed_values(gate_result, resolution, facts):
    """Values a reply may state: cited evidence quotes plus computed case facts."""
    sources = [e["quote"] for c in gate_result["claims"] if c["status"] == "grounded"
               for e in c["evidence"]["all"]]
    sources += [_fact_text(f["value"]) for f in facts.values() if f["value"] is not None]
    sources += [_fact_text(resolution[k]) for k in
                ("credit", "formula", "full_days", "request_deadline", "dispute_window_end")
                if resolution.get(k) is not None]
    sources += [str(resolver.DISPUTE_WINDOW_DAYS), str(resolver.OUTAGE_REQUEST_DAYS)]
    allowed = set()
    for s in sources:
        allowed |= values(s)
    allowed |= {"md:" + v[10:] for v in allowed if v.startswith("date:")}
    return allowed


def verify(segments, template_segments, gate_result, resolution, facts):
    errors, unsupported = [], []
    template_ids = [s["claim_id"] for s in template_segments]
    claims = {c["id"]: c for c in gate_result["claims"]}
    allowed = allowed_values(gate_result, resolution, facts)

    seen = {s["claim_id"] for s in segments}
    for cid in template_ids:
        if cid not in seen:
            errors.append(f"segment for '{cid}' is missing")
    for s in segments:
        cid, text = s["claim_id"], s["text"]
        if cid not in template_ids:
            errors.append(f"'{cid}' does not map to a grounded claim or notice")
            continue
        claim = claims.get(cid)
        if claim and claim["status"] == "grounded":
            if not any(d in text for d in claim["evidence"]["docs"]):
                errors.append(f"'{cid}' no longer cites {claim['evidence']['docs']}")
        elif claim and "Tier 2" not in text:
            errors.append(f"declined '{cid}' does not say it was escalated to Tier 2")
        bad = sorted(values(text) - allowed)
        if bad:
            unsupported += bad
            errors.append(f"'{cid}' states values not in evidence or computed facts: {bad}")
    return {"passed": not errors, "errors": errors, "unsupported_values": unsupported}
