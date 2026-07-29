"""Strict cross-artifact release-evidence contracts for M3."""

from __future__ import annotations

import hashlib
from typing import Annotated, Literal, Self

from pydantic import Field, FiniteFloat, PositiveFloat, model_validator

from fusion_fault_bench.contracts.common import ContractModel
from fusion_fault_bench.contracts.matrix_v1 import (
    M3_CI_SMOKE_MATRIX_SHA256,
    M3_PROCEDURAL_MATRIX_SHA256,
    M3_RELEASE_MANIFEST_SHA256S,
    M3_SMOKE_MANIFEST_SHA256S,
    MatrixId,
)
from fusion_fault_bench.contracts.procedural_artifact_v1 import (
    PROCEDURAL_INDEXED_PAYLOAD_PATHS,
    ProceduralIndexedPayloadPath,
)
from fusion_fault_bench.contracts.procedural_profile_v1 import (
    EDGE_PROFILE_SHA256,
    MAIN_PROFILE_SHA256,
    SMOKE_PROFILE_SHA256,
)
from fusion_fault_bench.contracts.result_v1alpha1 import Digest, Identifier

type IdentityRemovedField = Literal[
    "run_id",
    "manifest_sha256",
    "fault_family",
    "fault_axis",
    "severity",
    "experiment",
]

IDENTITY_ALLOWED_REMOVED_FIELDS: tuple[IdentityRemovedField, ...] = (
    "run_id",
    "manifest_sha256",
    "fault_family",
    "fault_axis",
    "severity",
    "experiment",
)
_RELEASE_EXPERIMENTS = (
    "procedural-lidar-y-bias",
    "procedural-camera-noise-correctly-reported",
    "procedural-camera-noise-underreported",
    "procedural-camera-calibration-x",
    "procedural-camera-calibration-yaw",
    "procedural-camera-timestamp-offset",
    "procedural-camera-dropout",
    "procedural-common-mode-x-fov-edge",
)
_SMOKE_EXPERIMENTS = ("procedural-ci-smoke",)
_MAIN_PROFILE_ID = "constant-velocity-front-roi-v1"
_EDGE_PROFILE_ID = "constant-velocity-fov-edge-v1"
_SMOKE_PROFILE_ID = "constant-velocity-ci-smoke-v1"
_ARTIFACT_SET_DOMAIN = b"fusion-fault-bench/m3-artifact-set/v1\x00"


def _frame4(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(4, "big") + encoded


def _expected_matrix_sha256(matrix_id: MatrixId) -> str:
    return (
        M3_PROCEDURAL_MATRIX_SHA256
        if matrix_id == "m3-procedural-v1"
        else M3_CI_SMOKE_MATRIX_SHA256
    )


class M3MatrixArtifactEvidenceV1(ContractModel):
    """One strictly loaded artifact in frozen matrix execution order."""

    execution_index: Annotated[int, Field(ge=0, le=7)]
    experiment: Identifier
    manifest_sha256: Digest
    profile_id: Identifier
    profile_sha256: Digest
    artifact_sha256: Digest


def _artifact_set_digest(
    matrix_sha256: str,
    artifacts: tuple[M3MatrixArtifactEvidenceV1, ...],
) -> str:
    payload = bytearray(_ARTIFACT_SET_DOMAIN)
    payload.extend(_frame4(matrix_sha256))
    payload.extend(len(artifacts).to_bytes(4, "big"))
    for artifact in artifacts:
        payload.extend(_frame4(artifact.experiment))
        payload.extend(_frame4(artifact.manifest_sha256))
        payload.extend(_frame4(artifact.profile_sha256))
        payload.extend(_frame4(artifact.artifact_sha256))
    return hashlib.sha256(payload).hexdigest()


class NormalizedIdentityArtifactV1(ContractModel):
    """Commitment to one artifact's ordered normalized identity rows."""

    execution_index: Annotated[int, Field(ge=0, le=6)]
    experiment: Identifier
    manifest_sha256: Digest
    normalized_row_count: Annotated[int, Field(ge=1)]
    normalized_rows_sha256: Digest


class ExcludedIdentityArtifactV1(ContractModel):
    """An artifact excluded only because it uses the separate edge profile."""

    execution_index: Literal[7]
    experiment: Literal["procedural-common-mode-x-fov-edge"]
    manifest_sha256: Digest
    profile_id: Literal["constant-velocity-fov-edge-v1"]
    reason: Literal["different-edge-profile-excluded-from-cross-profile-identity"]


class DropoutIdentityMetricMappingV1(ContractModel):
    """The sole metric-name rewrite permitted for identity comparison."""

    source_metric_name: Literal["conditional-matched-center-mse"]
    destination_metric_name: Literal["matched-center-mse"]
    scope: Literal["dropout-identity-only"]


class ReleaseIdentityComparisonV1(ContractModel):
    """Anchor-peer equality evidence for the seven comparable main artifacts."""

    status: Literal["applicable"]
    profile_id: Literal["constant-velocity-front-roi-v1"]
    comparison_mode: Literal["first-compatible-manifest-anchor-per-method-and-normalized-row"]
    allowed_removed_fields: Annotated[
        tuple[IdentityRemovedField, ...],
        Field(min_length=6, max_length=6),
    ]
    dropout_metric_mapping: DropoutIdentityMetricMappingV1
    included_artifacts: Annotated[
        tuple[NormalizedIdentityArtifactV1, ...],
        Field(min_length=7, max_length=7),
    ]
    excluded_artifacts: Annotated[
        tuple[ExcludedIdentityArtifactV1, ...],
        Field(min_length=1, max_length=1),
    ]
    normalized_row_count: Literal[6_800]
    distinct_normalized_key_count: Literal[1_000]
    comparison_count: Literal[5_800]
    mismatch_count: Annotated[int, Field(ge=0, le=5_800)]
    maximum_absolute_value_discrepancy_m2: Annotated[FiniteFloat, Field(ge=0.0)]
    all_equal: bool

    @model_validator(mode="after")
    def validate_release_identity(self) -> Self:
        if self.allowed_removed_fields != IDENTITY_ALLOWED_REMOVED_FIELDS:
            raise ValueError("identity comparison uses a non-preregistered removed field")
        execution_indices = tuple(item.execution_index for item in self.included_artifacts)
        if execution_indices != tuple(range(7)):
            raise ValueError("identity artifacts must follow the first seven matrix entries")
        experiments = tuple(item.experiment for item in self.included_artifacts)
        if len(set(experiments)) != len(experiments):
            raise ValueError("identity artifacts must have unique experiment identifiers")
        row_counts = tuple(item.normalized_row_count for item in self.included_artifacts)
        if experiments != _RELEASE_EXPERIMENTS[:7] or row_counts != (
            1_000,
            1_000,
            1_000,
            1_000,
            1_000,
            1_000,
            800,
        ):
            raise ValueError("identity artifacts have invalid normalized row counts")
        expected_equal = (
            self.mismatch_count == 0 and self.maximum_absolute_value_discrepancy_m2 == 0.0
        )
        if self.all_equal != expected_equal:
            raise ValueError("identity all_equal disagrees with comparison evidence")
        if (
            self.all_equal
            and len(
                {
                    item.normalized_rows_sha256
                    for item in self.included_artifacts
                    if item.normalized_row_count == 1_000
                }
            )
            != 1
        ):
            raise ValueError("equal identity evidence requires equal full-method row commitments")
        if self.maximum_absolute_value_discrepancy_m2 > 0.0 and self.mismatch_count == 0:
            raise ValueError("positive identity discrepancy requires a mismatch")
        return self


class SmokeIdentityComparisonV1(ContractModel):
    """Explicit non-applicability for the one-manifest CI smoke matrix."""

    status: Literal["not-applicable-single-manifest-smoke"]
    profile_id: Literal["constant-velocity-ci-smoke-v1"]
    experiment: Literal["procedural-ci-smoke"]
    manifest_sha256: Digest
    artifact_count: Literal[1]
    comparison_count: Literal[0]
    reason: Literal["single-manifest-smoke-has-no-cross-manifest-peer"]


type M3IdentityComparisonV1 = Annotated[
    ReleaseIdentityComparisonV1 | SmokeIdentityComparisonV1,
    Field(discriminator="status"),
]


class M3MatrixValidationV1(ContractModel):
    """Matrix-wide identity evidence linked to every strict M3 artifact."""

    schema_id: Literal["ffb.m3-matrix-validation/v1"] = Field(alias="schema")
    matrix_id: MatrixId
    matrix_sha256: Digest
    artifact_count: Annotated[int, Field(ge=1, le=8)]
    artifact_set_sha256: Digest
    ordered_artifacts: Annotated[
        tuple[M3MatrixArtifactEvidenceV1, ...],
        Field(min_length=1, max_length=8),
    ]
    identity_comparison: M3IdentityComparisonV1
    all_checks_passed: bool

    @model_validator(mode="after")
    def validate_matrix_evidence(self) -> Self:
        if self.matrix_sha256 != _expected_matrix_sha256(self.matrix_id):
            raise ValueError("matrix digest disagrees with the frozen matrix")
        if self.artifact_count != len(self.ordered_artifacts):
            raise ValueError("matrix artifact_count disagrees with ordered artifacts")
        expected_count = 8 if self.matrix_id == "m3-procedural-v1" else 1
        if self.artifact_count != expected_count:
            raise ValueError("matrix artifact count disagrees with the frozen matrix")
        if tuple(item.execution_index for item in self.ordered_artifacts) != tuple(
            range(expected_count)
        ):
            raise ValueError("matrix artifacts must use contiguous execution indices")
        experiments = tuple(item.experiment for item in self.ordered_artifacts)
        manifest_digests = tuple(item.manifest_sha256 for item in self.ordered_artifacts)
        if len(set(experiments)) != expected_count or len(set(manifest_digests)) != expected_count:
            raise ValueError("matrix artifacts must have unique experiment and manifest identities")
        if self.matrix_id == "m3-procedural-v1":
            if experiments != _RELEASE_EXPERIMENTS:
                raise ValueError("ordered artifacts disagree with the release matrix")
            if manifest_digests != M3_RELEASE_MANIFEST_SHA256S:
                raise ValueError("ordered manifest digests disagree with the release matrix")
            if tuple(item.profile_id for item in self.ordered_artifacts) != (
                *(_MAIN_PROFILE_ID for _ in range(7)),
                _EDGE_PROFILE_ID,
            ):
                raise ValueError("ordered artifact profiles disagree with the release matrix")
            if tuple(item.profile_sha256 for item in self.ordered_artifacts) != (
                *(MAIN_PROFILE_SHA256 for _ in range(7)),
                EDGE_PROFILE_SHA256,
            ):
                raise ValueError("ordered profile digests disagree with the release matrix")
            if self.identity_comparison.status != "applicable":
                raise ValueError("release matrix requires applicable identity evidence")
            included = self.identity_comparison.included_artifacts
            excluded = self.identity_comparison.excluded_artifacts
            if tuple(
                (item.execution_index, item.experiment, item.manifest_sha256) for item in included
            ) != tuple(
                (item.execution_index, item.experiment, item.manifest_sha256)
                for item in self.ordered_artifacts[:7]
            ):
                raise ValueError("identity inclusions disagree with ordered matrix artifacts")
            if (
                excluded[0].execution_index,
                excluded[0].experiment,
                excluded[0].manifest_sha256,
            ) != (
                self.ordered_artifacts[7].execution_index,
                self.ordered_artifacts[7].experiment,
                self.ordered_artifacts[7].manifest_sha256,
            ):
                raise ValueError("identity exclusion disagrees with the edge artifact")
            expected_pass = self.identity_comparison.all_equal
        else:
            if (
                experiments != _SMOKE_EXPERIMENTS
                or self.ordered_artifacts[0].profile_id != _SMOKE_PROFILE_ID
            ):
                raise ValueError("ordered artifact disagrees with the smoke matrix")
            if (
                manifest_digests != M3_SMOKE_MANIFEST_SHA256S
                or self.ordered_artifacts[0].profile_sha256 != SMOKE_PROFILE_SHA256
            ):
                raise ValueError("ordered digests disagree with the smoke matrix")
            if self.identity_comparison.status != "not-applicable-single-manifest-smoke":
                raise ValueError("smoke matrix requires explicit single-manifest status")
            artifact = self.ordered_artifacts[0]
            if (
                self.identity_comparison.experiment,
                self.identity_comparison.manifest_sha256,
            ) != (artifact.experiment, artifact.manifest_sha256):
                raise ValueError("smoke identity status disagrees with its artifact")
            expected_pass = True
        if self.all_checks_passed != expected_pass:
            raise ValueError("matrix all_checks_passed disagrees with identity evidence")
        if self.artifact_set_sha256 != _artifact_set_digest(
            self.matrix_sha256,
            self.ordered_artifacts,
        ):
            raise ValueError("artifact-set digest disagrees with ordered artifacts")
        return self


class IndexedMemberDigestPairV1(ContractModel):
    """One matrix-ordered indexed-member digest comparison."""

    execution_index: Annotated[int, Field(ge=0, le=7)]
    experiment: Identifier
    manifest_sha256: Digest
    path: ProceduralIndexedPayloadPath
    first_sha256: Digest
    second_sha256: Digest
    equal: bool

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        if self.equal != (self.first_sha256 == self.second_sha256):
            raise ValueError("indexed-member equality flag disagrees with its digests")
        return self


class RepeatRunMeasurementV1(ContractModel):
    """Measured resources and artifact-set identity for one clean matrix run."""

    artifact_set_sha256: Digest
    run_record_sha256s: Annotated[
        tuple[Digest, ...],
        Field(min_length=1, max_length=8),
    ]
    cpu_model: Annotated[str, Field(min_length=1, max_length=256)]
    wall_time_seconds: PositiveFloat
    peak_memory_bytes: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def validate_cpu_model(self) -> Self:
        if not self.cpu_model.strip():
            raise ValueError("CPU model must contain a non-whitespace name")
        return self


class RepeatVerificationV1(ContractModel):
    """Exact indexed-payload repeat comparison for two complete matrix runs."""

    schema_id: Literal["ffb.repeat-verification/v1"] = Field(alias="schema")
    matrix_id: MatrixId
    matrix_sha256: Digest
    artifact_count: Annotated[int, Field(ge=1, le=8)]
    first_run: RepeatRunMeasurementV1
    second_run: RepeatRunMeasurementV1
    indexed_member_pairs: Annotated[
        tuple[IndexedMemberDigestPairV1, ...],
        Field(min_length=6, max_length=48),
    ]
    comparison_count: Annotated[int, Field(ge=6, le=48)]
    mismatch_count: Annotated[int, Field(ge=0, le=48)]
    resource_measurement_scope: Literal[
        "self-reported-by-tracked-wait4-driver-not-independently-recomputable"
    ]
    execution_evidence_scope: Literal[
        "distinct-path-inode-and-run-record-consistency-not-cryptographic-proof"
    ]
    same_named_cpu: bool
    all_equal: bool
    all_checks_passed: bool

    @model_validator(mode="after")
    def validate_repeat_evidence(self) -> Self:
        if self.matrix_sha256 != _expected_matrix_sha256(self.matrix_id):
            raise ValueError("matrix digest disagrees with the frozen matrix")
        expected_artifact_count = 8 if self.matrix_id == "m3-procedural-v1" else 1
        expected_comparison_count = expected_artifact_count * len(PROCEDURAL_INDEXED_PAYLOAD_PATHS)
        if self.artifact_count != expected_artifact_count:
            raise ValueError("repeat artifact count disagrees with the frozen matrix")
        if (
            self.comparison_count != expected_comparison_count
            or len(self.indexed_member_pairs) != expected_comparison_count
        ):
            raise ValueError("repeat comparison count disagrees with indexed members")
        if (
            len(self.first_run.run_record_sha256s) != expected_artifact_count
            or len(self.second_run.run_record_sha256s) != expected_artifact_count
        ):
            raise ValueError("repeat run-record count disagrees with the frozen matrix")
        if any(
            first == second
            for first, second in zip(
                self.first_run.run_record_sha256s,
                self.second_run.run_record_sha256s,
                strict=True,
            )
        ):
            raise ValueError("repeat evidence requires distinct volatile run records")
        expected_order = tuple(
            (execution_index, path)
            for execution_index in range(expected_artifact_count)
            for path in PROCEDURAL_INDEXED_PAYLOAD_PATHS
        )
        if (
            tuple((pair.execution_index, pair.path) for pair in self.indexed_member_pairs)
            != expected_order
        ):
            raise ValueError("repeat member pairs are not in matrix/member order")
        expected_experiments = (
            _RELEASE_EXPERIMENTS if self.matrix_id == "m3-procedural-v1" else _SMOKE_EXPERIMENTS
        )
        expected_manifest_digests = (
            M3_RELEASE_MANIFEST_SHA256S
            if self.matrix_id == "m3-procedural-v1"
            else M3_SMOKE_MANIFEST_SHA256S
        )
        block_manifest_digests: list[Digest] = []
        for execution_index in range(expected_artifact_count):
            block = self.indexed_member_pairs[
                execution_index * len(PROCEDURAL_INDEXED_PAYLOAD_PATHS) : (execution_index + 1)
                * len(PROCEDURAL_INDEXED_PAYLOAD_PATHS)
            ]
            if {pair.experiment for pair in block} != {
                expected_experiments[execution_index]
            } or len({pair.manifest_sha256 for pair in block}) != 1:
                raise ValueError("repeat member block has inconsistent artifact identity")
            block_manifest_digests.append(block[0].manifest_sha256)
        if len(set(block_manifest_digests)) != expected_artifact_count:
            raise ValueError("repeat artifact blocks must have unique manifest identities")
        if tuple(block_manifest_digests) != expected_manifest_digests:
            raise ValueError("repeat artifact blocks disagree with frozen manifest identities")
        expected_mismatches = sum(not pair.equal for pair in self.indexed_member_pairs)
        if self.mismatch_count != expected_mismatches:
            raise ValueError("repeat mismatch count disagrees with member pairs")
        expected_same_cpu = self.first_run.cpu_model == self.second_run.cpu_model
        if self.same_named_cpu != expected_same_cpu:
            raise ValueError("same_named_cpu disagrees with the recorded CPU models")
        set_digests_equal = (
            self.first_run.artifact_set_sha256 == self.second_run.artifact_set_sha256
        )
        expected_all_equal = expected_mismatches == 0 and set_digests_equal
        if expected_mismatches == 0 and not set_digests_equal:
            raise ValueError("equal indexed members require equal artifact-set digests")
        if expected_mismatches > 0 and set_digests_equal:
            raise ValueError("mismatched indexed members require distinct artifact-set digests")
        if self.all_equal != expected_all_equal:
            raise ValueError("repeat all_equal disagrees with digest evidence")
        if self.all_checks_passed != (self.all_equal and self.same_named_cpu):
            raise ValueError("repeat all_checks_passed disagrees with repeat evidence")
        return self


def m3_matrix_validation_json_schema() -> dict[str, object]:
    """Return the strict public schema for matrix-wide M3 evidence."""

    return M3MatrixValidationV1.model_json_schema(by_alias=True)


def repeat_verification_json_schema() -> dict[str, object]:
    """Return the strict public schema for M3 repeat evidence."""

    return RepeatVerificationV1.model_json_schema(by_alias=True)
