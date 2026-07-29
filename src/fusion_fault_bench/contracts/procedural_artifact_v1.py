"""Strict file-envelope contracts for M3 procedural artifacts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fusion_fault_bench.contracts.common import ContractModel
from fusion_fault_bench.contracts.result_v1alpha1 import Digest, Identifier

PROCEDURAL_ARTIFACT_CONTRACT = "ffb.procedural-payload/v1"

PROCEDURAL_MANIFEST_FILE = "manifest.json"
PROCEDURAL_PROFILE_FILE = "procedural-profile.json"
PROCEDURAL_SEQUENCE_METRICS_FILE = "sequence-metrics.ndjson"
PROCEDURAL_AGGREGATE_METRICS_FILE = "aggregate-metrics.ndjson"
PROCEDURAL_CROSSOVERS_FILE = "crossovers.ndjson"
PROCEDURAL_VALIDATION_FILE = "procedural-validation.json"
PROCEDURAL_PAYLOAD_INDEX_FILE = "payload-index.json"
PROCEDURAL_RUN_FILE = "run.json"
PROCEDURAL_SUCCESS_FILE = "_SUCCESS"

PROCEDURAL_INDEXED_PAYLOAD_PATHS = (
    PROCEDURAL_MANIFEST_FILE,
    PROCEDURAL_PROFILE_FILE,
    PROCEDURAL_SEQUENCE_METRICS_FILE,
    PROCEDURAL_AGGREGATE_METRICS_FILE,
    PROCEDURAL_CROSSOVERS_FILE,
    PROCEDURAL_VALIDATION_FILE,
)
PROCEDURAL_ARTIFACT_PATHS = (
    *PROCEDURAL_INDEXED_PAYLOAD_PATHS,
    PROCEDURAL_PAYLOAD_INDEX_FILE,
    PROCEDURAL_RUN_FILE,
    PROCEDURAL_SUCCESS_FILE,
)

PROCEDURAL_MAX_SEQUENCE_COUNT = 10_000
PROCEDURAL_MAX_BOOTSTRAP_REPLICATES = 20_000
PROCEDURAL_MAX_BOOTSTRAP_CELLS = 20_000_000
PROCEDURAL_MAX_SEQUENCE_ROWS = 2_000_000
PROCEDURAL_MAX_MEMBER_BYTES = 512 * 1024 * 1024
PROCEDURAL_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
PROCEDURAL_MAX_RECORD_BYTES = 1024 * 1024

type ProceduralIndexedPayloadPath = Literal[
    "manifest.json",
    "procedural-profile.json",
    "sequence-metrics.ndjson",
    "aggregate-metrics.ndjson",
    "crossovers.ndjson",
    "procedural-validation.json",
]


class ProceduralPayloadFileEntryV1Alpha2(ContractModel):
    """One exact member committed by the M3 procedural payload index."""

    path: ProceduralIndexedPayloadPath
    byte_length: Annotated[int, Field(ge=0, le=PROCEDURAL_MAX_MEMBER_BYTES)]
    sha256: Digest

    @model_validator(mode="after")
    def require_nonempty_except_crossover(self) -> Self:
        if self.path != PROCEDURAL_CROSSOVERS_FILE and self.byte_length == 0:
            raise ValueError("only crossovers.ndjson may be empty")
        return self


class ProceduralPayloadIndexV1Alpha2(ContractModel):
    """Deterministic six-member envelope for M3 scientific payloads."""

    schema_id: Literal["ffb.payload-index/v1alpha2"] = Field(alias="schema")
    artifact_contract: Literal["ffb.procedural-payload/v1"]
    run_id: Identifier
    manifest_sha256: Digest
    profile_sha256: Digest
    files: Annotated[
        tuple[ProceduralPayloadFileEntryV1Alpha2, ...],
        Field(min_length=6, max_length=6),
    ]

    @model_validator(mode="after")
    def require_exact_member_order(self) -> Self:
        if tuple(entry.path for entry in self.files) != PROCEDURAL_INDEXED_PAYLOAD_PATHS:
            raise ValueError("procedural payload files must use the fixed six-member order")
        return self


def procedural_payload_index_json_schema() -> dict[str, object]:
    """Return the strict public schema for the M3 payload envelope."""

    return ProceduralPayloadIndexV1Alpha2.model_json_schema(by_alias=True)
