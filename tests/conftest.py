from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fusion_fault_bench.contracts.io import validate_manifest_mapping
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    ExperimentManifestV1Alpha1,
)


@pytest.fixture
def example_path() -> Path:
    return Path("examples/manifests/analytic-bias-v1alpha1.json")


@pytest.fixture
def manifest_data(example_path: Path) -> dict[str, Any]:
    value = json.loads(example_path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture
def validate_manifest() -> Callable[[dict[str, Any]], ExperimentManifestV1Alpha1]:
    def validate(value: dict[str, Any]) -> ExperimentManifestV1Alpha1:
        return validate_manifest_mapping(value)

    return validate
