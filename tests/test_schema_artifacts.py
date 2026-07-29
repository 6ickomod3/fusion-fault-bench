from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fusion_fault_bench.contracts.artifact_v1alpha1 import (
    AnalyticValidationV1Alpha1,
    PayloadIndexV1Alpha1,
    SuccessMarkerV1Alpha1,
)
from fusion_fault_bench.contracts.manifest_v1alpha1 import manifest_json_schema
from fusion_fault_bench.contracts.result_v1alpha1 import (
    AggregateMetricRecordV1Alpha1,
    CrossoverRecordV1Alpha1,
    RunRecordV1Alpha1,
    metric_record_json_schema,
)

SchemaBuilder = Callable[[], dict[str, Any]]


@pytest.mark.parametrize(
    ("path", "build_schema"),
    [
        (
            Path("schemas/manifest-v1alpha1.schema.json"),
            manifest_json_schema,
        ),
        (
            Path("schemas/run-record-v1alpha1.schema.json"),
            lambda: RunRecordV1Alpha1.model_json_schema(by_alias=True),
        ),
        (
            Path("schemas/metric-record-v1alpha1.schema.json"),
            metric_record_json_schema,
        ),
        (
            Path("schemas/aggregate-record-v1alpha1.schema.json"),
            lambda: AggregateMetricRecordV1Alpha1.model_json_schema(by_alias=True),
        ),
        (
            Path("schemas/crossover-record-v1alpha1.schema.json"),
            lambda: CrossoverRecordV1Alpha1.model_json_schema(by_alias=True),
        ),
        (
            Path("schemas/analytic-validation-v1alpha1.schema.json"),
            lambda: AnalyticValidationV1Alpha1.model_json_schema(by_alias=True),
        ),
        (
            Path("schemas/payload-index-v1alpha1.schema.json"),
            lambda: PayloadIndexV1Alpha1.model_json_schema(by_alias=True),
        ),
        (
            Path("schemas/success-marker-v1alpha1.schema.json"),
            lambda: SuccessMarkerV1Alpha1.model_json_schema(by_alias=True),
        ),
    ],
)
def test_committed_schema_matches_contract(path: Path, build_schema: SchemaBuilder) -> None:
    committed = json.loads(path.read_text(encoding="utf-8"))

    assert committed == build_schema()
