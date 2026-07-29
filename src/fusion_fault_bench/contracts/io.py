"""Strict contract loading helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    EXPERIMENT_MANIFEST_ADAPTER,
    ExperimentManifestV1Alpha1,
)


def _reject_nonstandard_number(value: str) -> None:
    raise ValueError(f"non-standard JSON number is forbidden: {value}")


def _parse_json_int(value: str) -> int:
    if value == "-0":
        raise ValueError("negative zero is forbidden in canonical JSON")
    return int(value)


def _parse_json_float(value: str) -> float:
    parsed = float(value)
    if parsed == 0.0 and value.startswith("-"):
        raise ValueError("negative zero is forbidden in canonical JSON")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        value[key] = item
    return value


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a standards-compliant JSON object from disk."""

    with path.open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonstandard_number,
            parse_float=_parse_json_float,
            parse_int=_parse_json_int,
        )
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return value


def load_manifest(path: Path) -> ExperimentManifestV1Alpha1:
    """Load and strictly validate a v1alpha1 experiment manifest."""

    raw = path.read_text(encoding="utf-8")
    value = json.loads(
        raw,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_nonstandard_number,
        parse_float=_parse_json_float,
        parse_int=_parse_json_int,
    )
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return EXPERIMENT_MANIFEST_ADAPTER.validate_json(raw)


def validate_manifest_mapping(
    value: Mapping[str, Any],
) -> ExperimentManifestV1Alpha1:
    """Validate a Python mapping through the normative JSON boundary."""

    raw = json.dumps(value, allow_nan=False, separators=(",", ":"))
    return EXPERIMENT_MANIFEST_ADAPTER.validate_json(raw)
