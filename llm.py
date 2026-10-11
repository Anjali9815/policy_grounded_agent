"""Claude calls for the router, intent and responder agents (only used when USE_LLM=1).

All calls use JSON-schema structured outputs, so categories and intents are
constrained to fixed enums and the reply comes back as claim-tagged segments the
verifier can check. Any failure raises LLMError and the caller falls back to
rules. Token usage of every call is appended to usage_log.
"""
import json

import config

SYSTEM_INTENT = (
    "You classify one customer message from a telecom billing-support chat into exactly one "
    "intent from the given list. Choose \"unknown\" when no listed intent clearly fits; never "
    "stretch an intent to cover a different request."
)

SYSTEM_RESPONDER = (
    "You rewrite a billing-support agent's draft reply so it reads naturally to the customer. "
    "The draft is a list of segments, each tagged with a claim_id. Return one rewritten segment "
    "per draft segment with the same claim_id. Use only the facts in that segment's draft text, "
    "its evidence quotes and the computed facts provided; do not add any number, date, amount, "
    "policy or promise that is not there. Keep every bracketed citation such as [KB-1004] or [R5]. "
    "Keep any statement that something was escalated to a Tier 2 Billing Specialist."
)


SYSTEM_ROUTER = (
    "You route one customer message from a telecom billing-dispute chat to the knowledge-base "
    "categories that could answer it. Return every category that fits. Return only "
    "\"out_of_scope\" when the message is not about a billing dispute, refund, outage credit or "
    "the bill itself."
)
OUT_OF_SCOPE = "out_of_scope"


class LLMError(Exception):
    pass


class ClaudeLLM:
    def __init__(self, model=None, server_fallback=True):
        import anthropic
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model or config.model()
        self.server_fallback = server_fallback  # Haiku has no server-side refusal fallback
        self.usage_log = []

    def _json(self, system, payload, schema, purpose):
        a = self._anthropic
        extra = ({"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
                 if self.server_fallback else {})
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
                **extra,
            )
        except a.APIConnectionError as e:
            raise LLMError(f"connection error: {e}") from e
        except a.RateLimitError as e:
            raise LLMError("rate limited") from e
        except a.APIStatusError as e:
            raise LLMError(f"API error {e.status_code}: {e.message}") from e
        self.usage_log.append({"purpose": purpose, "model": self.model,
                               "input_tokens": response.usage.input_tokens,
                               "output_tokens": response.usage.output_tokens})
        if response.stop_reason != "end_turn":
            raise LLMError(f"stop_reason {response.stop_reason}")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("no text block in response")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"invalid JSON: {e}") from e

    def classify(self, message, intents):
        """intents: {name: description}. Returns one of the names or 'unknown'."""
        names = list(intents) + ["unknown"]
        schema = {"type": "object", "properties": {"intent": {"type": "string", "enum": names}},
                  "required": ["intent"], "additionalProperties": False}
        out = self._json(SYSTEM_INTENT, {"intents": intents, "message": message}, schema, "intent")
        if out.get("intent") not in names:
            raise LLMError(f"intent outside the fixed list: {out.get('intent')!r}")
        return out["intent"]

    def phrase(self, question, segments, evidence, computed):
        ids = [s["claim_id"] for s in segments]
        schema = {
            "type": "object",
            "properties": {"segments": {"type": "array", "items": {
                "type": "object",
                "properties": {"claim_id": {"type": "string", "enum": ids}, "text": {"type": "string"}},
                "required": ["claim_id", "text"], "additionalProperties": False}}},
            "required": ["segments"], "additionalProperties": False,
        }
        payload = {"customer_message": question, "draft_segments": segments,
                   "evidence": evidence, "computed_facts": computed}
        out = self._json(SYSTEM_RESPONDER, payload, schema, "responder")
        return [{"claim_id": s["claim_id"], "text": s["text"].strip()} for s in out["segments"]]


    def route(self, message, categories):
        """categories: {name: topic terms}. Returns a list of names, or ['out_of_scope']."""
        names = list(categories) + [OUT_OF_SCOPE]
        schema = {"type": "object", "properties": {"categories": {
            "type": "array", "items": {"type": "string", "enum": names}}},
            "required": ["categories"], "additionalProperties": False}
        out = self._json(SYSTEM_ROUTER, {"categories": categories, "message": message}, schema, "router")
        chosen = [c for c in dict.fromkeys(out.get("categories") or []) if c in names]
        if not chosen:
            raise LLMError("router returned no category")
        return [OUT_OF_SCOPE] if OUT_OF_SCOPE in chosen else chosen


def make(use_llm=None):
    """A ClaudeLLM when enabled, else None (fully deterministic)."""
    enabled = config.use_llm() if use_llm is None else use_llm
    return ClaudeLLM() if enabled else None


def make_router(use_llm=None):
    """The small, fast model used only for ambiguous routing."""
    enabled = config.use_llm() if use_llm is None else use_llm
    return ClaudeLLM(model=config.router_model(), server_fallback=False) if enabled else None
