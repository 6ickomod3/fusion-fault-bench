"""Canonical JSON serialization for immutable experiment intent."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel


def canonical_json(value: BaseModel | Mapping[str, Any]) -> str:
    """Return compact, key-sorted JSON with non-finite values rejected."""

    payload: Mapping[str, Any]
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", by_alias=True)
    else:
        payload = value
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_digest(value: BaseModel | Mapping[str, Any]) -> str:
    """Fingerprint canonical experimental intent."""

    encoded = canonical_json(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
