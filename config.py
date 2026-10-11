"""Settings from the environment, with .env loaded if python-dotenv is installed."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
except ImportError:  # deterministic mode needs no packages
    load_dotenv = None

if load_dotenv:
    load_dotenv(ROOT / ".env")


def use_llm():
    return os.getenv("USE_LLM", "0").strip() == "1"


def provider():
    return os.getenv("LLM_PROVIDER", "anthropic").strip().lower()


# Defaults exist only for Anthropic; Groq models must be named in the environment.
DEFAULTS = {"anthropic": {"responder": "claude-opus-5-5", "router": "claude-haiku-5-5"}}


def responder_model():
    """Model for the intent and responder agents."""
    return (os.getenv("RESPONDER_MODEL") or
            (os.getenv("ANTHROPIC_MODEL") if provider() == "anthropic" else None) or
            DEFAULTS.get(provider(), {}).get("responder"))


def router_model():
    """Small, fast model used only when keyword routing is ambiguous."""
    return os.getenv("ROUTER_MODEL") or DEFAULTS.get(provider(), {}).get("router")


def langsmith_enabled():
    return bool(os.getenv("LANGSMITH_API_KEY"))
