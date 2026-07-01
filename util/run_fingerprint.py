"""
util/run_fingerprint.py

Run fingerprint utilities for reproducibility logging.

Emits a stable run-configuration fingerprint so that experiment setup differences
are easy to spot across runs. Saves a JSON summary of training config and
environment at run start.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional


def _to_jsonable(value: Any) -> Any:
    """Convert nested structures to JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_run_fingerprint(
    script_name: str,
    train_config: Dict[str, Any],
    model_kwargs: Dict[str, Any],
    effective_model_config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build fingerprint payload and stable hash for a training run."""
    payload = {
        "script": script_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "train_config": _to_jsonable(train_config),
        "model_kwargs_overrides": _to_jsonable(model_kwargs),
        "effective_model_config": _to_jsonable(effective_model_config),
        "extra": _to_jsonable(extra or {}),
    }

    stable_repr = json.dumps(
        {
            "script": payload["script"],
            "train_config": payload["train_config"],
            "model_kwargs_overrides": payload["model_kwargs_overrides"],
            "effective_model_config": payload["effective_model_config"],
            "extra": payload["extra"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    payload["fingerprint_sha256"] = hashlib.sha256(stable_repr.encode("utf-8")).hexdigest()
    return payload


def emit_run_fingerprint(
    script_name: str,
    train_config: Dict[str, Any],
    model_kwargs: Dict[str, Any],
    effective_model_config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
    print_fn: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Build and print a run fingerprint block. Returns the full payload."""
    payload = build_run_fingerprint(
        script_name=script_name,
        train_config=train_config,
        model_kwargs=model_kwargs,
        effective_model_config=effective_model_config,
        extra=extra,
    )

    print_fn("\n" + "=" * 80)
    print_fn(f"Run Fingerprint ({script_name})")
    print_fn(f"Fingerprint SHA256: {payload['fingerprint_sha256']}")
    print_fn(json.dumps(payload, indent=2, sort_keys=True))
    print_fn("=" * 80)

    return payload
