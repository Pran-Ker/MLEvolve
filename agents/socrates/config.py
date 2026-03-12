"""Configuration for the Socrates system."""

import os
from pathlib import Path
from typing import Literal

# Challenge directory (can be overridden via environment variable)
CHALLENGE_PATH = Path(
    os.environ.get(
        "SOCRATES_CHALLENGE_PATH",
        "/workspace/ventilator-pressure-prediction-sample-10"
    )
).absolute()

# Python binary path for the scientist agent
SOCRATES_PYTHON_BIN = os.environ.get(
    "SOCRATES_PYTHON_BIN",
    "/workspace/.venv/bin"
)

# PI mode loop structure:
#   Outer loop (sessions): fresh scientist each time, reads artifacts from previous sessions
#   Inner loop (experiments): scientist iterates within a single session with persistent context
MAX_SESSIONS = int(os.environ.get("SOCRATES_MAX_SESSIONS", "10"))
EXPERIMENTS_PER_SESSION = int(os.environ.get("SOCRATES_EXPERIMENTS_PER_SESSION", "15"))

# PI mode: if True, scientist works with Socrates PIs in session/experiment loops
# If False, scientist works alone in a single session until it signals [FINISHED]
PI_MODE = os.environ.get("PI_MODE", "true").lower() in ("true", "1", "yes")

# Max discussion rounds before forcing approval
MAX_DISCUSSION_ROUNDS = int(os.environ.get("SOCRATES_MAX_DISCUSSION_ROUNDS", "5"))

# Respect [FINISHED] signal in PI mode: if True, stop the current session when the
# scientist outputs [FINISHED], rather than forcing all EXPERIMENTS_PER_SESSION runs.
# In non-PI mode, [FINISHED] is always respected regardless of this setting.
RESPECT_FINISHED = os.environ.get("SOCRATES_RESPECT_FINISHED", "false").lower() in ("true", "1", "yes")

# Max turns for Scientist agent (controls how many model iterations/work cycles per experiment)
MAX_TURNS = int(os.environ.get("SCIENTIST_MAX_TURNS", "80"))

# Enable/disable PIs (for benchmarking different configurations)
ENABLE_PI_A = os.environ.get("SOCRATES_ENABLE_PI_A", "true").lower() in ("true", "1", "yes")

# Enforce GPU usage in scientist prompts
ENFORCE_GPU_USAGE = os.environ.get("SOCRATES_ENFORCE_GPU", "true").lower() in ("true", "1", "yes")

# Provider selection: "claude", "pydantic", or "openhands"
ProviderType = Literal["claude", "pydantic", "openhands"]

# Global defaults (used as fallback if agent-specific not set)
DEFAULT_PROVIDER: ProviderType = os.environ.get("SOCRATES_PROVIDER", "claude")  # type: ignore
DEFAULT_MODEL: str | None = os.environ.get("SOCRATES_MODEL", None)

# Scientist agent configuration
SCIENTIST_PROVIDER: ProviderType = os.environ.get("SCIENTIST_PROVIDER", DEFAULT_PROVIDER)  # type: ignore
SCIENTIST_MODEL: str | None = os.environ.get("SCIENTIST_MODEL", DEFAULT_MODEL)

# Socrates A (methodology-focused PI) configuration
SOCRATES_A_PROVIDER: ProviderType = os.environ.get("SOCRATES_A_PROVIDER", DEFAULT_PROVIDER)  # type: ignore
SOCRATES_A_MODEL: str | None = os.environ.get("SOCRATES_A_MODEL", DEFAULT_MODEL)

# Model examples:
# Claude: "claude-sonnet-4-20250514", "claude-opus-4-20250514"
# Pydantic: "anthropic:claude-sonnet-4-20250514", "openai:gpt-4o"
# OpenHands (LiteLLM): "xai/grok-4-1-fast-reasoning", "google/gemini-3-flash-preview"

