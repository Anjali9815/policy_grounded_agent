"""Check a reply against its evidence before it is sent.

A reply passes only if every number, date and dollar amount in it appears in the
cited evidence or the computed case facts, every segment maps to a claim or
notice from the template, no escalation or notice was dropped, and each grounded
claim's citation sits in the sentence that states the claim (the sentence sharing
the most words with the evidence quote), not elsewhere in the paragraph.
"""
import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

import resolver
from retriever import tokenize
from router import stem

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


DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")
DASHES.update(dict.fromkeys(map(ord, "\u2018\u2019"), "'"))
# The customer's requested amount may only be repeated to deny it ("not $100.00", "can't confirm $100.00").
DENIAL = re.compile(r"(\bnot|rather than|instead of|can't confirm|cannot confirm|unable to confirm)"
                    r"\s*(the\s+)?$", re.I)


def normalize(text):
    """Fold Unicode spaces and dashes (e.g. 'KB\u20111002', 'Tier\u202f2') to ASCII before checking."""
    return unicodedata.normalize("NFKC", text).translate(DASHES)


def _num(token):
    try:
        return format(Decimal(token.replace(",", "")).normalize(), "f")
    except InvalidOperation:
        return token


def values(text):
    """Normalized values in text: 'date:YYYY-MM-DD', 'md:MM-DD' (no year) and numbers."""
    out = set()
    text = LABELS.sub(" ", normalize(text))

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


SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def sentences(text):
    """Split into sentences; a citation written after the period ('... date. [KB-1004]') stays attached."""
    out = []
    for part in SENTENCE_END.split(text.strip()):
        lead = re.match(r"((?:\[[^\]]*\]\s*)+[.!?]?)\s*", part)
        if out and lead:  # citations opening a chunk belong to the sentence before it
            out[-1] += " " + lead[1].strip()
            part = part[lead.end():]
        if part:
            out.append(part)
    return out


def _words(text):
    return {stem(t) for t in tokenize(LABELS.sub(" ", text))}


def claim_sentences(text, quotes):
    """Sentences of a segment that best restate the evidence quotes (ties all count)."""
    evidence = set().union(*(_words(q) for q in quotes))
    scored = [(len(_words(s) & evidence), s) for s in sentences(text)]
    best = max((n for n, _ in scored), default=0)
    return [s for n, s in scored if n == best and n > 0]


def _cites(sentence, docs):
    return any(re.search(rf"\[[^\]]*\b{re.escape(d)}\b[^\]]*\]", sentence) for d in docs)


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
                ("credit", "formula", "full_days", "outage_hours", "request_deadline", "dispute_window_end")
                if resolution.get(k) is not None]
    sources += [str(resolver.DISPUTE_WINDOW_DAYS), str(resolver.OUTAGE_REQUEST_DAYS)]
    allowed = set()
    for s in sources:
        allowed |= values(s)
    allowed |= {"md:" + v[10:] for v in allowed if v.startswith("date:")}
    return allowed


def _amount_pattern(amount):
    whole, _, cents = format(amount, "f").partition(".")
    whole = f"{int(whole):,}".replace(",", ",?")
    tail = rf"(?:\.{cents})" if cents and cents.strip("0") else r"(?:\.0{1,2})?"
    return re.compile(rf"\$?\s?{whole}{tail}(?!\d)(?:\s*(?:dollars|bucks))?", re.I)


def _requested_only_denied(text, amount):
    """True if every mention of the requested amount is a denial of it, not an approval."""
    for m in _amount_pattern(amount).finditer(text):
        if not DENIAL.search(text[max(0, m.start() - 30):m.start()]):
            return False
    return True


def verify(segments, template_segments, gate_result, resolution, facts):
    errors, unsupported = [], []
    requested = gate_result.get("requested_amount")
    requested_value = format(requested.normalize(), "f") if requested is not None else None
    template_ids = [s["claim_id"] for s in template_segments]
    claims = {c["id"]: c for c in gate_result["claims"]}
    allowed = allowed_values(gate_result, resolution, facts)

    seen = {s["claim_id"] for s in segments}
    for cid in template_ids:
        if cid not in seen:
            errors.append(f"segment for '{cid}' is missing")
    for s in segments:
        cid, text = s["claim_id"], normalize(s["text"])
        if cid not in template_ids:
            errors.append(f"'{cid}' does not map to a grounded claim or notice")
            continue
        claim = claims.get(cid)
        if claim and claim["status"] == "grounded":
            docs = claim["evidence"]["docs"]
            if not any(d in text for d in docs):
                errors.append(f"'{cid}' no longer cites {docs}")
            elif not any(_cites(sen, docs) for sen in
                         claim_sentences(text, [e["quote"] for e in claim["evidence"]["all"]])):
                errors.append(f"'{cid}' cites {docs} outside the sentence that states the claim")
        elif claim and "Tier 2" not in text:
            errors.append(f"declined '{cid}' does not say it was escalated to Tier 2")
        bad = values(text) - allowed
        if requested_value in bad:
            # Known only because the customer said it; allowed solely as a denied amount.
            if _requested_only_denied(text, requested):
                bad.discard(requested_value)
            else:
                errors.append(f"'{cid}' presents the requested amount ${requested:,.2f} as approved")
                bad.discard(requested_value)
        bad = sorted(bad)
        if bad:
            unsupported += bad
            errors.append(f"'{cid}' states values not in evidence or computed facts: {bad}")
    return {"passed": not errors, "errors": errors, "unsupported_values": unsupported}
