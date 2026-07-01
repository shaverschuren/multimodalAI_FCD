"""
util/config.py

Project-wide configuration loader.

Scripts that need filesystem paths read them from ``config.json`` at the
repository root (the ``src/`` directory).  Copy ``config.json.example`` to
``config.json`` and set ``data_root`` to your local data directory.

Usage
-----
>>> from util.config import get_data_root
>>> data_root = get_data_root()          # returns Path or None
>>> k_fold_path = get_data_root() / "preprocessing" / "k_fold_splits.json"

The ``config.json`` file is gitignored and never committed; only
``config.json.example`` is version-controlled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

# The config file sits next to this util/ package, i.e. at the repo root (src/).
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


def load_config() -> Dict[str, Any]:
    """Return the parsed ``config.json``, or an empty dict if it does not exist."""
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            import warnings
            warnings.warn(f"Could not read {_CONFIG_PATH}: {exc}")
    return {}


def get_data_root() -> Optional[Path]:
    """
    Return the ``data_root`` value from ``config.json`` as a :class:`~pathlib.Path`.

    Returns ``None`` when the key is absent or the config file does not exist.
    Callers should fall back to a required argparse argument when this returns ``None``.
    """
    root = load_config().get("data_root")
    if root:
        return Path(root)
    return None
