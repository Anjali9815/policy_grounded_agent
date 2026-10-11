"""Load KB articles and policy rules from markdown files with frontmatter (parsed by hand)."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def _scalar(raw):
    raw = raw.strip()
    if raw in ("", "null", "~"):
        return None
    if raw.startswith('"'):
        return json.loads(raw)
    return raw


def parse(text):
    """Split a document into its frontmatter fields plus a 'body' key."""
    if not text.startswith("---\n"):
        raise ValueError("document has no frontmatter")
    front, body = text[4:].split("\n---\n", 1)
    meta, key = {}, None
    for line in front.splitlines():
        if not line.strip():
            continue
        if line.startswith("  - "):
            meta[key].append(_scalar(line[4:]))
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        meta[key] = [] if not value.strip() else _scalar(value)
    meta["body"] = body.lstrip("\n")
    return meta


def load_dir(subdir):
    docs = {}
    for path in sorted((DATA / subdir).glob("*.md")):
        doc = parse(path.read_text(encoding="utf-8"))
        doc["path"] = str(path.relative_to(DATA.parent))
        docs[doc["id"]] = doc
    return docs


def load_kb():
    return load_dir("kb")


def load_policy():
    return load_dir("policy")
