"""BM25 keyword retrieval over KB articles in billing-dispute scope."""
import math
import re
from collections import Counter

STOPWORDS = set("""
a an and are as at be but by can do does for from get getting have i if in is it its
itself like me my of on only or so that the this those to under was what whatever
when while will with you your just also up end
""".split())


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def _doc_text(doc):
    tags = " ".join(t.replace("-", " ") for t in doc.get("tags") or [])
    return " ".join([doc["title"], doc.get("category") or "", tags, doc["body"]])


class Retriever:
    def __init__(self, kb, k1=1.5, b=0.75):
        self.excluded = sorted(i for i, d in kb.items() if d.get("scope") == "not_billing_dispute")
        self.docs = {i: d for i, d in kb.items() if i not in self.excluded}
        self.k1, self.b = k1, b
        self.tf = {i: Counter(tokenize(_doc_text(d))) for i, d in self.docs.items()}
        self.len = {i: sum(tf.values()) for i, tf in self.tf.items()}
        self.avglen = sum(self.len.values()) / len(self.len)
        df = Counter(term for tf in self.tf.values() for term in tf)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query_terms, doc_id):
        tf, norm = self.tf[doc_id], self.k1 * (1 - self.b + self.b * self.len[doc_id] / self.avglen)
        return sum(self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (tf[t] + norm)
                   for t in query_terms if t in tf)

    def search(self, query, k=4):
        terms = tokenize(query)
        scored = [(round(self.score(terms, i), 3), i) for i in self.docs]
        scored = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))[:k]
        return [{"id": i, "title": self.docs[i]["title"], "score": s} for s, i in scored]
