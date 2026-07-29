from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from fusion_fault_bench.contracts.health_artifact_v1 import (
    HEALTH_EVAL_ARTIFACT_CONTRACT,
    HEALTH_EVAL_INDEXED_PATHS,
    HEALTH_FIT_ARTIFACT_CONTRACT,
    HEALTH_FIT_INDEXED_PATHS,
    HealthFitReferenceV1,
    HealthPayloadFileEntryV1,
    HealthPayloadIndexV1,
    health_payload_index_json_schema,
)

_DIGEST = "a" * 64


def _entry(path: str) -> HealthPayloadFileEntryV1:
    return HealthPayloadFileEntryV1.model_validate(
        {
            "path": path,
            "byte_length": 10,
            "sha256": _DIGEST,
            "record_count": 1 if path.endswith(".ndjson") else None,
        }
    )


def _index(*, evaluation: bool = False, **updates: Any) -> HealthPayloadIndexV1:
    paths = HEALTH_EVAL_INDEXED_PATHS if evaluation else HEALTH_FIT_INDEXED_PATHS
    payload: dict[str, Any] = {
        "schema": "ffb.health-payload-index/v1",
        "artifact_contract": (
            HEALTH_EVAL_ARTIFACT_CONTRACT if evaluation else HEALTH_FIT_ARTIFACT_CONTRACT
        ),
        "run_id": "run:0",
        "intent_sha256": _DIGEST,
        "main_profile_sha256": _DIGEST,
        "edge_profile_sha256": _DIGEST,
        "files": tuple(_entry(path) for path in paths),
    }
    payload.update(updates)
    return HealthPayloadIndexV1.model_validate(payload)


def test_fit_and_evaluation_indexes_require_exact_member_order() -> None:
    assert tuple(entry.path for entry in _index().files) == HEALTH_FIT_INDEXED_PATHS
    assert tuple(entry.path for entry in _index(evaluation=True).files) == (
        HEALTH_EVAL_INDEXED_PATHS
    )
    with pytest.raises(ValidationError):
        _index(files=tuple(reversed(_index().files)))
    with pytest.raises(ValidationError):
        _index(evaluation=True, files=_index().files)


def test_record_count_is_present_exactly_for_ndjson() -> None:
    with pytest.raises(ValidationError):
        HealthPayloadFileEntryV1(
            path="intent.json",
            byte_length=1,
            sha256=_DIGEST,
            record_count=1,
        )
    with pytest.raises(ValidationError):
        HealthPayloadFileEntryV1(
            path="aggregate-metrics.ndjson",
            byte_length=1,
            sha256=_DIGEST,
            record_count=None,
        )


def test_fit_reference_is_strict_and_bounded() -> None:
    record = HealthFitReferenceV1(
        schema="ffb.health-fit-reference/v1",
        fit_artifact_sha256=_DIGEST,
        fit_run_sha256=_DIGEST,
        intent_sha256=_DIGEST,
        selected_candidate_index=35,
        selected_self_threshold=1.0,
        selected_cross_threshold=1.0,
    )
    assert record.selected_candidate_index == 35
    with pytest.raises(ValidationError):
        HealthFitReferenceV1.model_validate(
            record.model_dump(by_alias=True) | {"selected_candidate_index": 36}
        )


def test_schema_exposes_both_artifact_contracts() -> None:
    schema = health_payload_index_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["$defs"]["HealthArtifactContract"]["enum"]) == {
        HEALTH_FIT_ARTIFACT_CONTRACT,
        HEALTH_EVAL_ARTIFACT_CONTRACT,
    }
