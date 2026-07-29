"""Fail-closed file envelopes for M4 health fit and evaluation artifacts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fusion_fault_bench.contracts.common import ContractModel
from fusion_fault_bench.contracts.result_v1alpha1 import Digest, Identifier

HEALTH_FIT_ARTIFACT_CONTRACT = "ffb.health-fit-payload/v1"
HEALTH_EVAL_ARTIFACT_CONTRACT = "ffb.health-eval-payload/v1"

HEALTH_INTENT_FILE = "intent.json"
HEALTH_MAIN_PROFILE_FILE = "main-profile.json"
HEALTH_EDGE_PROFILE_FILE = "edge-profile.json"
HEALTH_ECDF_FILE = "ecdf-arrays.json"
HEALTH_CANDIDATES_FILE = "threshold-candidates.ndjson"
HEALTH_FIT_SUMMARY_FILE = "fit-summary.json"
HEALTH_FIT_VALIDATION_FILE = "fit-validation.json"
HEALTH_FIT_REFERENCE_FILE = "fit-reference.json"
HEALTH_SEQUENCE_LOSSES_FILE = "sequence-losses.ndjson"
HEALTH_SEQUENCE_CONTRASTS_FILE = "sequence-contrasts.ndjson"
HEALTH_SEQUENCE_EVENTS_FILE = "sequence-events.ndjson"
HEALTH_AGGREGATES_FILE = "aggregate-metrics.ndjson"
HEALTH_EVAL_VALIDATION_FILE = "evaluation-validation.json"
HEALTH_PAYLOAD_INDEX_FILE = "payload-index.json"
HEALTH_RUN_FILE = "run.json"
HEALTH_SUCCESS_FILE = "_SUCCESS"

HEALTH_FIT_INDEXED_PATHS = (
    HEALTH_INTENT_FILE,
    HEALTH_MAIN_PROFILE_FILE,
    HEALTH_EDGE_PROFILE_FILE,
    HEALTH_ECDF_FILE,
    HEALTH_CANDIDATES_FILE,
    HEALTH_FIT_SUMMARY_FILE,
    HEALTH_FIT_VALIDATION_FILE,
)
HEALTH_EVAL_INDEXED_PATHS = (
    HEALTH_INTENT_FILE,
    HEALTH_MAIN_PROFILE_FILE,
    HEALTH_EDGE_PROFILE_FILE,
    HEALTH_FIT_REFERENCE_FILE,
    HEALTH_SEQUENCE_LOSSES_FILE,
    HEALTH_SEQUENCE_CONTRASTS_FILE,
    HEALTH_SEQUENCE_EVENTS_FILE,
    HEALTH_AGGREGATES_FILE,
    HEALTH_EVAL_VALIDATION_FILE,
)
HEALTH_MAX_MEMBER_BYTES = 256 * 1024 * 1024
HEALTH_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
HEALTH_MAX_RECORD_BYTES = 1024 * 1024

type HealthArtifactContract = Literal[
    "ffb.health-fit-payload/v1",
    "ffb.health-eval-payload/v1",
]
type HealthIndexedPath = Literal[
    "intent.json",
    "main-profile.json",
    "edge-profile.json",
    "ecdf-arrays.json",
    "threshold-candidates.ndjson",
    "fit-summary.json",
    "fit-validation.json",
    "fit-reference.json",
    "sequence-losses.ndjson",
    "sequence-contrasts.ndjson",
    "sequence-events.ndjson",
    "aggregate-metrics.ndjson",
    "evaluation-validation.json",
]


class HealthPayloadFileEntryV1(ContractModel):
    """One exact scientific member committed by an M4 payload index."""

    path: HealthIndexedPath
    byte_length: Annotated[int, Field(ge=1, le=HEALTH_MAX_MEMBER_BYTES)]
    sha256: Digest
    record_count: Annotated[int, Field(ge=1)] | None

    @model_validator(mode="after")
    def require_record_count_only_for_ndjson(self) -> Self:
        is_ndjson = self.path.endswith(".ndjson")
        if is_ndjson != (self.record_count is not None):
            raise ValueError("record_count must be present exactly for NDJSON members")
        return self


class HealthPayloadIndexV1(ContractModel):
    """Exact ordered scientific-member envelope for one M4 artifact."""

    schema_id: Literal["ffb.health-payload-index/v1"] = Field(alias="schema")
    artifact_contract: HealthArtifactContract
    run_id: Identifier
    intent_sha256: Digest
    main_profile_sha256: Digest
    edge_profile_sha256: Digest
    files: Annotated[
        tuple[HealthPayloadFileEntryV1, ...],
        Field(min_length=7, max_length=9),
    ]

    @model_validator(mode="after")
    def require_contract_member_order(self) -> Self:
        expected = (
            HEALTH_FIT_INDEXED_PATHS
            if self.artifact_contract == HEALTH_FIT_ARTIFACT_CONTRACT
            else HEALTH_EVAL_INDEXED_PATHS
        )
        if tuple(entry.path for entry in self.files) != expected:
            raise ValueError("health payload files do not match the contract order")
        return self


class HealthFitReferenceV1(ContractModel):
    """Evaluation binding to one immutable, already-published M4 fit."""

    schema_id: Literal["ffb.health-fit-reference/v1"] = Field(alias="schema")
    fit_artifact_sha256: Digest
    fit_run_sha256: Digest
    intent_sha256: Digest
    selected_candidate_index: Annotated[int, Field(ge=0, le=35)]
    selected_self_threshold: Annotated[float, Field(ge=0.95, le=1.0)]
    selected_cross_threshold: Annotated[float, Field(ge=0.95, le=1.0)]


def health_payload_index_json_schema() -> dict[str, object]:
    """Return the public strict schema for both M4 payload envelopes."""

    return HealthPayloadIndexV1.model_json_schema(by_alias=True)
