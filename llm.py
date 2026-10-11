"""LLM calls for the router, intent and responder agents (only used when USE_LLM=1).

LLM_PROVIDER picks the backend: "anthropic" (Claude, JSON-schema structured
outputs) or "groq" (JSON mode with the schema in the prompt). Either way every
response is validated here against the same schema and fixed enums, so an
invented intent, category or claim id raises LLMError and the caller falls back
to rules. Token usage of every call is appended to usage_log.
"""
import json
import logging

import config

log = logging.getLogger("policy_agent")

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
PROVIDERS = ("anthropic", "groq")


class LLMError(Exception):
    pass


class CircuitBreaker:
    """After the first LLM failure, every LLM is disabled for the rest of the run."""

    def __init__(self):
        self.open = False
        self.reason = None

    def allow(self, client):
        """The client if it may be called, else None (use the deterministic path)."""
        return None if client is None or self.open else client

    def trip(self, where, error):
        if not self.open:
            self.open, self.reason = True, f"{where}: {error}"
            short = " ".join(str(error).split())
            short = short if len(short) <= 100 else short[:97] + "..."
            log.warning("LLM disabled for this run after %s failed: %s", where, short)


class BaseLLM:
    """Prompts, schemas and validation shared by every provider; subclasses implement _call."""

    provider = None

    def __init__(self, model):
        self.model = model
        self.usage_log = []

    def _call(self, system, user, schema):
        """Return (response text, input tokens, output tokens); raise LLMError on any failure."""
        raise NotImplementedError

    def _json(self, system, payload, schema, purpose):
        text, tokens_in, tokens_out = self._call(
            system, json.dumps(payload, ensure_ascii=False, default=str), schema)
        self.usage_log.append({"purpose": purpose, "provider": self.provider, "model": self.model,
                               "input_tokens": tokens_in, "output_tokens": tokens_out})
        try:
            out = json.loads(text)
        except (TypeError, json.JSONDecodeError) as e:
            raise LLMError(f"invalid JSON: {e}") from e
        if not isinstance(out, dict):
            raise LLMError("response is not a JSON object")
        return out

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
        items = out.get("segments")
        if not isinstance(items, list) or not all(
                isinstance(s, dict) and s.get("claim_id") in ids and isinstance(s.get("text"), str)
                for s in items):
            raise LLMError("segments do not match the schema or use an unknown claim_id")
        return [{"claim_id": s["claim_id"], "text": s["text"].strip()} for s in items]

    def route(self, message, categories):
        """categories: {name: topic terms}. Returns a list of names, or ['out_of_scope']."""
        names = list(categories) + [OUT_OF_SCOPE]
        schema = {"type": "object", "properties": {"categories": {
            "type": "array", "items": {"type": "string", "enum": names}}},
            "required": ["categories"], "additionalProperties": False}
        out = self._json(SYSTEM_ROUTER, {"categories": categories, "message": message}, schema, "router")
        chosen = out.get("categories")
        if not isinstance(chosen, list) or not chosen:
            raise LLMError("router returned no category")
        invented = [c for c in chosen if c not in names]
        if invented:
            raise LLMError(f"category outside the fixed list: {invented}")
        chosen = list(dict.fromkeys(chosen))
        return [OUT_OF_SCOPE] if OUT_OF_SCOPE in chosen else chosen


class ClaudeLLM(BaseLLM):
    provider = "anthropic"

    def __init__(self, model, server_fallback=True):
        import anthropic
        super().__init__(model)
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.server_fallback = server_fallback

    def _call(self, system, user, schema):
        a = self._anthropic
        extra = ({"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
                 if self.server_fallback else {})
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
                **extra,
            )
        except a.APIConnectionError as e:
            raise LLMError(f"connection error: {e}") from e
        except a.RateLimitError as e:
            raise LLMError("rate limited") from e
        except a.APIStatusError as e:
            raise LLMError(f"API error {e.status_code}: {e.message}") from e
        except TypeError as e:  # the SDK raises this at call time when no credentials resolve
            if "authentication" not in str(e):
                raise
            raise LLMError("no Anthropic credentials configured") from e
        if response.stop_reason != "end_turn":
            raise LLMError(f"stop_reason {response.stop_reason}")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("no text block in response")
        return text, response.usage.input_tokens, response.usage.output_tokens


class GroqLLM(BaseLLM):
    """Groq chat completions in JSON mode; the schema goes in the prompt and is enforced on return."""

    provider = "groq"

    def __init__(self, model):
        import groq
        super().__init__(model)
        self._groq = groq
        self.client = groq.Groq()  # reads GROQ_API_KEY

    def _call(self, system, user, schema):
        g = self._groq
        prompt = (f"{system}\n\nRespond with a single JSON object that matches this JSON schema "
                  f"exactly, and nothing else:\n{json.dumps(schema)}")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=0,
                max_completion_tokens=2048,
            )
        except g.APIConnectionError as e:
            raise LLMError(f"connection error: {e}") from e
        except g.RateLimitError as e:
            raise LLMError("rate limited") from e
        except g.APIStatusError as e:
            raise LLMError(f"API error {e.status_code}: {e.message}") from e
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise LLMError(f"finish_reason {choice.finish_reason}")
        usage = response.usage
        return (choice.message.content, usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0)


def _build(model):
    provider = config.provider()
    if provider not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {PROVIDERS}, got {provider!r}")
    if not model:
        raise ValueError(f"set RESPONDER_MODEL and ROUTER_MODEL for LLM_PROVIDER={provider}")
    if provider == "groq":
        return GroqLLM(model)
    # Haiku has no server-side refusal fallback; the other current Claude models do.
    return ClaudeLLM(model, server_fallback=not model.startswith("claude-haiku"))


def make(use_llm=None):
    """The intent/responder model when enabled, else None (fully deterministic)."""
    enabled = config.use_llm() if use_llm is None else use_llm
    return _build(config.responder_model()) if enabled else None


def make_router(use_llm=None):
    """The small, fast model used only for ambiguous routing."""
    enabled = config.use_llm() if use_llm is None else use_llm
    return _build(config.router_model()) if enabled else None
