"""Route a turn to KB categories using only frontmatter (category, scope, governs, tags).

Each category's vocabulary is its name plus its articles' tags. A question is
scored against every category with IDF weighting across categories, so a word
shared by many categories ("credit") counts less than a distinctive one
("retention"). Outcomes:
  routed        in-scope categories scoring at least RELATIVE_CUTOFF of the best
  out_of_scope  an out-of-scope category clearly beats every in-scope one
  ambiguous     nothing matches, or in- and out-of-scope categories score alike;
                only this case may use the small LLM, otherwise it escalates
Scope is decided from the question alone. Once a turn is in scope, the category
governing the case's current rule is added (case context), so a follow-up like
"does the credit reduce taxes?" still reaches the outage-credit articles.
"""
import math
import re

from retriever import tokenize

OUT_OF_SCOPE_VALUE = "not_billing_dispute"
RELATIVE_CUTOFF = 0.5
AMBIGUITY_MARGIN = 0.2
STOPWORDS = {"no", "not"}  # appear in tags like no-frontline-credit; not topics


def stem(word):
    for suffix in ("ing", "ed", "es", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def terms(text):
    return {stem(t) for t in tokenize(text.replace("-", " ")) if t not in STOPWORDS}


def approx_tokens(text):
    """Rough token count (words + punctuation); no tokenizer dependency."""
    return len(re.findall(r"\w+|[^\w\s]", text))


def _label(category):
    return re.sub(r"\s*\(.*?\)", "", category).strip()


class Router:
    def __init__(self, kb):
        cats = {}
        for doc in kb.values():
            c = cats.setdefault(doc["category"], {
                "category": doc["category"], "scope": doc["scope"],
                "articles": [], "governs": [], "terms": terms(_label(doc["category"]))})
            c["articles"].append(doc["id"])
            if doc.get("governs") and doc["governs"] not in c["governs"]:
                c["governs"].append(doc["governs"])
            for tag in doc.get("tags") or []:
                if tag.replace("-", "_") != doc["scope"]:  # a tag restating the scope is not a topic
                    c["terms"] |= terms(tag)
        self.categories = cats
        df = {}
        for c in cats.values():
            for t in c["terms"]:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + len(cats) / n) for t, n in df.items()}

    def in_scope(self, name):
        return self.categories[name]["scope"] != OUT_OF_SCOPE_VALUE

    def scores(self, question):
        q = terms(question)
        out = []
        for name, c in self.categories.items():
            matched = sorted(q & c["terms"])
            out.append({"category": name, "score": round(sum(self.idf[t] for t in matched), 3),
                        "matched": matched, "in_scope": self.in_scope(name)})
        return sorted(out, key=lambda s: (-s["score"], s["category"]))

    def route(self, question):
        scored = self.scores(question)
        best_in = max((s["score"] for s in scored if s["in_scope"]), default=0)
        best_out = max((s["score"] for s in scored if not s["in_scope"]), default=0)
        result = {"scores": [s for s in scored if s["score"] > 0], "source": "keywords", "categories": []}
        if best_in == 0 and best_out == 0:
            return {**result, "status": "ambiguous", "reason": "no category terms matched"}
        if best_out > best_in * (1 + AMBIGUITY_MARGIN):
            return {**result, "status": "out_of_scope",
                    "reason": f"out-of-scope category scores {best_out} vs in-scope {best_in}"}
        if best_out and best_out >= best_in * (1 - AMBIGUITY_MARGIN):
            return {**result, "status": "ambiguous",
                    "reason": f"in-scope {best_in} and out-of-scope {best_out} score alike"}
        chosen = [s["category"] for s in scored
                  if s["in_scope"] and s["score"] >= best_in * RELATIVE_CUTOFF]
        return {**result, "status": "routed", "categories": chosen, "reason": "keyword match"}

    def with_case_context(self, route, case_rules):
        """Add the categories governing the case's rules (governing rule plus any escalation rule)."""
        case_rules = [r for r in case_rules if r]
        if route["status"] != "routed" or not case_rules:
            return route
        extra = [n for n, c in self.categories.items()
                 if set(case_rules) & set(c["governs"]) and n not in route["categories"]]
        if not extra:
            return route
        return {**route, "categories": route["categories"] + extra,
                "case_context": {"rule": ", ".join(case_rules), "added": extra}}

    def select(self, categories, kb, policy):
        """Only the routed categories' articles and the rules they govern."""
        articles = [a for name in categories for a in self.categories[name]["articles"]]
        rules = sorted({r for name in categories for r in self.categories[name]["governs"]})
        return {a: kb[a] for a in articles}, {r: policy[r] for r in rules}

    def describe(self):
        """In-scope categories for the LLM fallback: name -> topic terms from tags."""
        return {name: ", ".join(sorted(c["terms"])) for name, c in self.categories.items()
                if self.in_scope(name)}
