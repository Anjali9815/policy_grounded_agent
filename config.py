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


def model():
    return os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")


def router_model():
    return os.getenv("ROUTER_MODEL", "claude-haiku-5-5")


def langsmith_enabled():
    return bool(os.getenv("LANGSMITH_API_KEY"))
