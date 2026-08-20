"""Stable filesystem locations shared by configuration-backed adapters."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = Path(os.getenv("DATA_AGENT_CONFIG_DIR", PROJECT_ROOT / "config")).expanduser().resolve()
