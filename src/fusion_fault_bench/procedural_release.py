"""Independent matrix-identity and repeat evidence for M3 releases."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.bundle_v1alpha1 import expected_sequence_ids
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AvailabilityControlManifest,
    CommonModeControlManifest,
    GeometryCrossoverManifest,
    ProceduralSource,
)
from fusion_fault_bench.contracts.matrix_v1 import LoadedExperimentMatrix
from fusion_fault_bench.contracts.procedural_artifact_v1 import (
    PROCEDURAL_ARTIFACT_PATHS,
    PROCEDURAL_INDEXED_PAYLOAD_PATHS,
)
from fusion_fault_bench.contracts.procedural_release_v1 import (
    IDENTITY_ALLOWED_REMOVED_FIELDS,
    DropoutIdentityMetricMappingV1,
    ExcludedIdentityArtifactV1,
    IndexedMemberDigestPairV1,
    M3MatrixArtifactEvidenceV1,
    M3MatrixValidationV1,
    NormalizedIdentityArtifactV1,
    ReleaseIdentityComparisonV1,
    RepeatRunMeasurementV1,
    RepeatVerificationV1,
    SmokeIdentityComparisonV1,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    LocalizationMetricRecord,
)
from fusion_fault_bench.procedural_artifacts import (
    LoadedProceduralArtifact,
    load_procedural_artifact,
)

type ProceduralManifest = (
    GeometryCrossoverManifest | AvailabilityControlManifest | CommonModeControlManifest
)
type IdentityKey = tuple[str, str, str]

_ARTIFACT_SET_DOMAIN = b"fusion-fault-bench/m3-artifact-set/v1\x00"
_IDENTITY_ROWS_DOMAIN = b"fusion-fault-bench/m3-identity-rows/v1\x00"
_MAIN_PROFILE_ID = "constant-velocity-front-roi-v1"
_EDGE_PROFILE_ID = "constant-velocity-fov-edge-v1"
_SMOKE_PROFILE_ID = "constant-velocity-ci-smoke-v1"
_IDENTITY_METHOD_ORDER = (
    "camera-only",
    "lidar-only",
    "fixed-fusion",
    "fault-target-drop-policy",
    "performance-oracle",
)
_NORMALIZED_IDENTITY_FIELDS = frozenset(
    {
        "schema",
        "record_level",
        "sequence_id",
        "method_id",
        "eligible_object_frame_count",
        "valid_object_frame_count",
        "metric_name",
        "status",
        "value",
        "unit",
    }
)
_IDENTITY_SOURCE_FIELDS = _NORMALIZED_IDENTITY_FIELDS | frozenset(IDENTITY_ALLOWED_REMOVED_FIELDS)


class ProceduralReleaseValidationError(ValueError):
    """M3 release evidence disagrees with its strict artifact inputs."""


@dataclass(frozen=True, slots=True)
class RepeatRunResources:
    """Externally measured matrix-run resources attached to repeat evidence."""

    wall_time_seconds: float
    peak_memory_bytes: int


@dataclass(frozen=True, slots=True)
class _NormalizedIdentityRows:
    rows: Mapping[IdentityKey, Mapping[str, Any]]
    sha256: str


def _frame4(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) >= 2**32:
        raise ProceduralReleaseValidationError("release identity field is too long")
    return len(encoded).to_bytes(4, "big") + encoded


def _canonical_mapping_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _identity_rows_digest(
    ordered_rows: Sequence[Mapping[str, Any]],
) -> str:
    payload = bytearray(_IDENTITY_ROWS_DOMAIN)
    payload.extend(len(ordered_rows).to_bytes(8, "big"))
    for row in ordered_rows:
        encoded = _canonical_mapping_bytes(row)
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
    return hashlib.sha256(payload).hexdigest()


def _procedural_manifest(value: object) -> ProceduralManifest:
    if isinstance(
        value,
        (GeometryCrossoverManifest, AvailabilityControlManifest, CommonModeControlManifest),
    ) and isinstance(value.source, ProceduralSource):
        return value
    raise ProceduralReleaseValidationError("matrix contains a non-procedural artifact")


def _ordered_artifacts(
    matrix: LoadedExperimentMatrix,
    artifacts: Sequence[LoadedProceduralArtifact],
) -> tuple[LoadedProceduralArtifact, ...]:
    if sha256_digest(matrix.matrix) != matrix.matrix_sha256:
        raise ProceduralReleaseValidationError("matrix digest is inconsistent")
    if len(artifacts) != len(matrix.manifests):
        raise ProceduralReleaseValidationError("artifact collection is incomplete")

    by_manifest: dict[str, LoadedProceduralArtifact] = {}
    for artifact in artifacts:
        manifest = _procedural_manifest(artifact.manifest)
        manifest_sha256 = sha256_digest(manifest)
        if manifest_sha256 in by_manifest:
            raise ProceduralReleaseValidationError("artifact collection has a duplicate manifest")
        source = cast(ProceduralSource, manifest.source)
        profile_sha256 = sha256_digest(artifact.profile)
        if (
            artifact.profile.profile_id != source.profile_id
            or profile_sha256 != source.profile_sha256
            or artifact.payload_index.manifest_sha256 != manifest_sha256
            or artifact.payload_index.profile_sha256 != profile_sha256
            or artifact.payload_index.run_id != artifact.run.run_id
            or artifact.run.manifest_sha256 != manifest_sha256
            or artifact.run.artifact_sha256 != artifact.artifact_sha256
        ):
            raise ProceduralReleaseValidationError("artifact identity graph is inconsistent")
        by_manifest[manifest_sha256] = artifact

    ordered: list[LoadedProceduralArtifact] = []
    for expected_manifest in matrix.manifests:
        manifest = _procedural_manifest(expected_manifest)
        digest = sha256_digest(manifest)
        artifact = by_manifest.get(digest)
        if artifact is None or artifact.manifest != manifest:
            raise ProceduralReleaseValidationError(
                "artifact collection disagrees with matrix execution order"
            )
        ordered.append(artifact)
    return tuple(ordered)


def compute_m3_artifact_set_digest(
    matrix: LoadedExperimentMatrix,
    artifacts: Sequence[LoadedProceduralArtifact],
) -> str:
    """Hash matrix-ordered scientific artifact identities with explicit framing.

    The preimage is the domain, framed matrix digest, a uint32 artifact count,
    then framed experiment, manifest digest, profile digest, and procedural
    artifact digest for each entry in matrix execution order.
    """

    ordered = _ordered_artifacts(matrix, artifacts)
    payload = bytearray(_ARTIFACT_SET_DOMAIN)
    payload.extend(_frame4(matrix.matrix_sha256))
    payload.extend(len(ordered).to_bytes(4, "big"))
    for artifact in ordered:
        payload.extend(_frame4(artifact.manifest.experiment))
        payload.extend(_frame4(sha256_digest(artifact.manifest)))
        payload.extend(_frame4(sha256_digest(artifact.profile)))
        payload.extend(_frame4(artifact.artifact_sha256))
    return hashlib.sha256(payload).hexdigest()


def _matrix_artifact_evidence(
    artifacts: Sequence[LoadedProceduralArtifact],
) -> tuple[M3MatrixArtifactEvidenceV1, ...]:
    return tuple(
        M3MatrixArtifactEvidenceV1(
            execution_index=index,
            experiment=artifact.manifest.experiment,
            manifest_sha256=sha256_digest(artifact.manifest),
            profile_id=artifact.profile.profile_id,
            profile_sha256=sha256_digest(artifact.profile),
            artifact_sha256=artifact.artifact_sha256,
        )
        for index, artifact in enumerate(artifacts)
    )


def _normalized_identity_rows(
    artifact: LoadedProceduralArtifact,
) -> _NormalizedIdentityRows:
    manifest = _procedural_manifest(artifact.manifest)
    source = cast(ProceduralSource, manifest.source)
    if source.profile_id != _MAIN_PROFILE_ID:
        raise ProceduralReleaseValidationError(
            "only main-profile artifacts enter release identity comparison"
        )
    availability = isinstance(manifest, AvailabilityControlManifest)
    expected_methods = _IDENTITY_METHOD_ORDER[:4] if availability else _IDENTITY_METHOD_ORDER
    rows: dict[IdentityKey, Mapping[str, Any]] = {}
    for record in artifact.metrics:
        if record.severity.index != 0 or not isinstance(record, LocalizationMetricRecord):
            continue
        if availability:
            if record.metric_name != "conditional-matched-center-mse":
                continue
        elif record.metric_name != "matched-center-mse":
            continue
        if record.method_id not in expected_methods:
            raise ProceduralReleaseValidationError("identity row uses a non-comparable method")

        normalized = record.model_dump(mode="json", by_alias=True)
        normalized["experiment"] = manifest.experiment
        if frozenset(normalized) != _IDENTITY_SOURCE_FIELDS:
            raise ProceduralReleaseValidationError("identity source row has an undeclared field")
        for field in IDENTITY_ALLOWED_REMOVED_FIELDS:
            normalized.pop(field)
        if availability:
            normalized["metric_name"] = "matched-center-mse"
        if frozenset(normalized) != _NORMALIZED_IDENTITY_FIELDS:
            raise ProceduralReleaseValidationError(
                "identity normalization removed an undeclared field"
            )
        key = (
            cast(str, normalized["sequence_id"]),
            cast(str, normalized["method_id"]),
            cast(str, normalized["metric_name"]),
        )
        if key in rows:
            raise ProceduralReleaseValidationError("identity normalization produced duplicate keys")
        rows[key] = normalized

    sequence_ids = expected_sequence_ids(manifest)
    expected_keys = {
        (sequence_id, method, "matched-center-mse")
        for sequence_id in sequence_ids
        for method in expected_methods
    }
    if set(rows) != expected_keys:
        raise ProceduralReleaseValidationError("artifact has incomplete normalized identity rows")
    sequence_rank = {sequence_id: index for index, sequence_id in enumerate(sequence_ids)}
    method_rank = {method: index for index, method in enumerate(_IDENTITY_METHOD_ORDER)}
    ordered_keys = sorted(
        rows,
        key=lambda key: (sequence_rank[key[0]], method_rank[key[1]], key[2]),
    )
    ordered_rows = tuple(rows[key] for key in ordered_keys)
    return _NormalizedIdentityRows(
        rows=rows,
        sha256=_identity_rows_digest(ordered_rows),
    )


def _release_identity_comparison(
    artifacts: Sequence[LoadedProceduralArtifact],
) -> ReleaseIdentityComparisonV1:
    included = tuple(
        (index, artifact)
        for index, artifact in enumerate(artifacts)
        if artifact.profile.profile_id == _MAIN_PROFILE_ID
    )
    excluded = tuple(
        (index, artifact)
        for index, artifact in enumerate(artifacts)
        if artifact.profile.profile_id == _EDGE_PROFILE_ID
    )
    if len(included) != 7 or len(excluded) != 1 or excluded[0][0] != 7:
        raise ProceduralReleaseValidationError(
            "release matrix has invalid identity profile grouping"
        )
    normalized = tuple(
        (index, artifact, _normalized_identity_rows(artifact)) for index, artifact in included
    )

    comparison_count = 0
    mismatch_count = 0
    maximum_value_discrepancy = 0.0
    for method in _IDENTITY_METHOD_ORDER:
        compatible = tuple(
            item for item in normalized if method in _procedural_manifest(item[1].manifest).methods
        )
        if len(compatible) < 2:
            raise ProceduralReleaseValidationError(
                "identity method has fewer than two compatible artifacts"
            )
        _, anchor_artifact, anchor = compatible[0]
        sequence_ids = expected_sequence_ids(_procedural_manifest(anchor_artifact.manifest))
        for _, peer_artifact, peer in compatible[1:]:
            if sequence_ids != expected_sequence_ids(_procedural_manifest(peer_artifact.manifest)):
                raise ProceduralReleaseValidationError(
                    "identity peers have different sequence identifiers"
                )
            for sequence_id in sequence_ids:
                key = (sequence_id, method, "matched-center-mse")
                left_row = anchor.rows[key]
                right_row = peer.rows[key]
                comparison_count += 1
                if left_row != right_row:
                    mismatch_count += 1
                left_value = left_row["value"]
                right_value = right_row["value"]
                if isinstance(left_value, (int, float)) and isinstance(
                    right_value,
                    (int, float),
                ):
                    maximum_value_discrepancy = max(
                        maximum_value_discrepancy,
                        abs(float(left_value) - float(right_value)),
                    )

    if comparison_count != 5_800:
        raise ProceduralReleaseValidationError(
            "identity comparison count disagrees with anchor-peer semantics"
        )
    normalized_evidence = tuple(
        NormalizedIdentityArtifactV1(
            execution_index=index,
            experiment=artifact.manifest.experiment,
            manifest_sha256=sha256_digest(artifact.manifest),
            normalized_row_count=len(rows.rows),
            normalized_rows_sha256=rows.sha256,
        )
        for index, artifact, rows in normalized
    )
    normalized_row_count = sum(item.normalized_row_count for item in normalized_evidence)
    if normalized_row_count != 6_800:
        raise ProceduralReleaseValidationError(
            "identity normalized row count disagrees with the frozen matrix"
        )
    edge_index, edge_artifact = excluded[0]
    return ReleaseIdentityComparisonV1(
        status="applicable",
        profile_id=_MAIN_PROFILE_ID,
        comparison_mode=("first-compatible-manifest-anchor-per-method-and-normalized-row"),
        allowed_removed_fields=IDENTITY_ALLOWED_REMOVED_FIELDS,
        dropout_metric_mapping=DropoutIdentityMetricMappingV1(
            source_metric_name="conditional-matched-center-mse",
            destination_metric_name="matched-center-mse",
            scope="dropout-identity-only",
        ),
        included_artifacts=normalized_evidence,
        excluded_artifacts=(
            ExcludedIdentityArtifactV1(
                execution_index=cast(Any, edge_index),
                experiment=cast(Any, edge_artifact.manifest.experiment),
                manifest_sha256=sha256_digest(edge_artifact.manifest),
                profile_id=cast(Any, edge_artifact.profile.profile_id),
                reason=("different-edge-profile-excluded-from-cross-profile-identity"),
            ),
        ),
        normalized_row_count=6_800,
        distinct_normalized_key_count=1_000,
        comparison_count=comparison_count,
        mismatch_count=mismatch_count,
        maximum_absolute_value_discrepancy_m2=maximum_value_discrepancy,
        all_equal=mismatch_count == 0 and maximum_value_discrepancy == 0.0,
    )


def build_m3_matrix_validation(
    matrix: LoadedExperimentMatrix,
    artifacts: Sequence[LoadedProceduralArtifact],
) -> M3MatrixValidationV1:
    """Build matrix-wide identity evidence from strict artifact objects."""

    ordered = _ordered_artifacts(matrix, artifacts)
    artifact_evidence = _matrix_artifact_evidence(ordered)
    if matrix.matrix.matrix_id == "m3-procedural-v1":
        identity = _release_identity_comparison(ordered)
        all_checks_passed = identity.all_equal
    else:
        artifact = ordered[0]
        if artifact.profile.profile_id != _SMOKE_PROFILE_ID:
            raise ProceduralReleaseValidationError(
                "smoke artifact uses the wrong procedural profile"
            )
        identity = SmokeIdentityComparisonV1(
            status="not-applicable-single-manifest-smoke",
            profile_id=_SMOKE_PROFILE_ID,
            experiment=cast(Any, artifact.manifest.experiment),
            manifest_sha256=sha256_digest(artifact.manifest),
            artifact_count=1,
            comparison_count=0,
            reason="single-manifest-smoke-has-no-cross-manifest-peer",
        )
        all_checks_passed = True
    return M3MatrixValidationV1(
        schema="ffb.m3-matrix-validation/v1",
        matrix_id=matrix.matrix.matrix_id,
        matrix_sha256=matrix.matrix_sha256,
        artifact_count=len(ordered),
        artifact_set_sha256=compute_m3_artifact_set_digest(matrix, ordered),
        ordered_artifacts=artifact_evidence,
        identity_comparison=identity,
        all_checks_passed=all_checks_passed,
    )


def validate_m3_matrix_validation(
    evidence: M3MatrixValidationV1,
    *,
    matrix: LoadedExperimentMatrix,
    artifacts: Sequence[LoadedProceduralArtifact],
) -> None:
    """Independently rebuild and compare every matrix-validation field."""

    expected = build_m3_matrix_validation(matrix, artifacts)
    if evidence != expected:
        raise ProceduralReleaseValidationError(
            "M3 matrix validation disagrees with strict artifacts"
        )


def _absolute_real_run_root(path: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(path)))
    current = Path(root.anchor)
    try:
        for component in root.parts[1:]:
            current /= component
            metadata = os.lstat(current)
            if stat.S_ISLNK(metadata.st_mode):
                raise ProceduralReleaseValidationError("repeat run root contains a symlink")
    except ProceduralReleaseValidationError:
        raise
    except OSError as error:
        raise ProceduralReleaseValidationError("repeat run root cannot be inspected") from error
    if not root.is_dir():
        raise ProceduralReleaseValidationError("repeat run root must be a directory")
    return root


def load_m3_artifact_set(
    matrix: LoadedExperimentMatrix,
    run_root: Path,
) -> tuple[LoadedProceduralArtifact, ...]:
    """Strictly load one exact artifact directory per matrix entry."""

    root = _absolute_real_run_root(run_root)
    expected_names = tuple(
        _procedural_manifest(manifest).experiment for manifest in matrix.manifests
    )
    try:
        entries = tuple(root.iterdir())
    except OSError as error:
        raise ProceduralReleaseValidationError("repeat run root cannot be enumerated") from error
    if {entry.name for entry in entries} != set(expected_names):
        raise ProceduralReleaseValidationError(
            "repeat run root artifact allowlist disagrees with the matrix"
        )
    for entry in entries:
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ProceduralReleaseValidationError(
                "repeat run root entries must be real artifact directories"
            )
    loaded = tuple(load_procedural_artifact(root / experiment) for experiment in expected_names)
    return _ordered_artifacts(matrix, loaded)


def _require_independent_run_root_inodes(
    matrix: LoadedExperimentMatrix,
    first_root: Path,
    second_root: Path,
) -> None:
    """Reject hard-linked artifact trees masquerading as independent reruns."""

    for raw_manifest in matrix.manifests:
        experiment = _procedural_manifest(raw_manifest).experiment
        first_artifact = first_root / experiment
        second_artifact = second_root / experiment
        try:
            if os.path.samestat(first_artifact.stat(), second_artifact.stat()):
                raise ProceduralReleaseValidationError(
                    "repeat artifact directories must have independent inodes"
                )
            for member in PROCEDURAL_ARTIFACT_PATHS:
                if os.path.samestat(
                    (first_artifact / member).stat(),
                    (second_artifact / member).stat(),
                ):
                    raise ProceduralReleaseValidationError(
                        "repeat artifact members must have independent inodes"
                    )
        except ProceduralReleaseValidationError:
            raise
        except OSError as error:
            raise ProceduralReleaseValidationError(
                "repeat artifact inode independence cannot be inspected"
            ) from error


def _validated_resources(resources: RepeatRunResources) -> RepeatRunResources:
    if (
        not math.isfinite(resources.wall_time_seconds)
        or resources.wall_time_seconds <= 0.0
        or type(resources.peak_memory_bytes) is not int
        or resources.peak_memory_bytes <= 0
    ):
        raise ProceduralReleaseValidationError(
            "repeat resource measurements must be finite and positive"
        )
    return resources


def _cpu_model(artifacts: Sequence[LoadedProceduralArtifact]) -> str:
    models = {artifact.run.environment.cpu_model for artifact in artifacts}
    if len(models) != 1:
        raise ProceduralReleaseValidationError(
            "one repeat artifact set contains multiple CPU models"
        )
    return next(iter(models))


def _repeat_verification_from_artifacts(
    matrix: LoadedExperimentMatrix,
    first: Sequence[LoadedProceduralArtifact],
    second: Sequence[LoadedProceduralArtifact],
    *,
    first_resources: RepeatRunResources,
    second_resources: RepeatRunResources,
) -> RepeatVerificationV1:
    first_ordered = _ordered_artifacts(matrix, first)
    second_ordered = _ordered_artifacts(matrix, second)
    first_measurements = _validated_resources(first_resources)
    second_measurements = _validated_resources(second_resources)
    pairs: list[IndexedMemberDigestPairV1] = []
    for execution_index, (first_artifact, second_artifact) in enumerate(
        zip(first_ordered, second_ordered, strict=True)
    ):
        first_manifest_sha256 = sha256_digest(first_artifact.manifest)
        second_manifest_sha256 = sha256_digest(second_artifact.manifest)
        if first_manifest_sha256 != second_manifest_sha256:
            raise ProceduralReleaseValidationError("repeat artifact sets use different manifests")
        for expected_path, first_entry, second_entry in zip(
            PROCEDURAL_INDEXED_PAYLOAD_PATHS,
            first_artifact.payload_index.files,
            second_artifact.payload_index.files,
            strict=True,
        ):
            if first_entry.path != expected_path or second_entry.path != expected_path:
                raise ProceduralReleaseValidationError(
                    "repeat payload indexes use a noncanonical member order"
                )
            pairs.append(
                IndexedMemberDigestPairV1(
                    execution_index=execution_index,
                    experiment=first_artifact.manifest.experiment,
                    manifest_sha256=first_manifest_sha256,
                    path=expected_path,
                    first_sha256=first_entry.sha256,
                    second_sha256=second_entry.sha256,
                    equal=first_entry.sha256 == second_entry.sha256,
                )
            )
    mismatch_count = sum(not pair.equal for pair in pairs)
    first_set_sha256 = compute_m3_artifact_set_digest(matrix, first_ordered)
    second_set_sha256 = compute_m3_artifact_set_digest(matrix, second_ordered)
    first_run = RepeatRunMeasurementV1(
        artifact_set_sha256=first_set_sha256,
        run_record_sha256s=tuple(artifact.run_sha256 for artifact in first_ordered),
        cpu_model=_cpu_model(first_ordered),
        wall_time_seconds=first_measurements.wall_time_seconds,
        peak_memory_bytes=first_measurements.peak_memory_bytes,
    )
    second_run = RepeatRunMeasurementV1(
        artifact_set_sha256=second_set_sha256,
        run_record_sha256s=tuple(artifact.run_sha256 for artifact in second_ordered),
        cpu_model=_cpu_model(second_ordered),
        wall_time_seconds=second_measurements.wall_time_seconds,
        peak_memory_bytes=second_measurements.peak_memory_bytes,
    )
    all_equal = mismatch_count == 0 and first_set_sha256 == second_set_sha256
    same_named_cpu = first_run.cpu_model == second_run.cpu_model
    return RepeatVerificationV1(
        schema="ffb.repeat-verification/v1",
        matrix_id=matrix.matrix.matrix_id,
        matrix_sha256=matrix.matrix_sha256,
        artifact_count=len(first_ordered),
        first_run=first_run,
        second_run=second_run,
        indexed_member_pairs=tuple(pairs),
        comparison_count=len(pairs),
        mismatch_count=mismatch_count,
        resource_measurement_scope=(
            "self-reported-by-tracked-wait4-driver-not-independently-recomputable"
        ),
        execution_evidence_scope=(
            "distinct-path-inode-and-run-record-consistency-not-cryptographic-proof"
        ),
        same_named_cpu=same_named_cpu,
        all_equal=all_equal,
        all_checks_passed=all_equal and same_named_cpu,
    )


def build_repeat_verification(
    matrix: LoadedExperimentMatrix,
    first_run_root: Path,
    second_run_root: Path,
    *,
    first_resources: RepeatRunResources,
    second_resources: RepeatRunResources,
) -> RepeatVerificationV1:
    """Strictly load two run roots and build exact indexed-member evidence."""

    first_root = _absolute_real_run_root(first_run_root)
    second_root = _absolute_real_run_root(second_run_root)
    if os.path.samestat(first_root.stat(), second_root.stat()):
        raise ProceduralReleaseValidationError(
            "repeat verification requires two distinct run roots"
        )
    _require_independent_run_root_inodes(matrix, first_root, second_root)
    first = load_m3_artifact_set(matrix, first_root)
    second = load_m3_artifact_set(matrix, second_root)
    return _repeat_verification_from_artifacts(
        matrix,
        first,
        second,
        first_resources=first_resources,
        second_resources=second_resources,
    )


def validate_repeat_verification(
    evidence: RepeatVerificationV1,
    *,
    matrix: LoadedExperimentMatrix,
    first_run_root: Path,
    second_run_root: Path,
    first_resources: RepeatRunResources,
    second_resources: RepeatRunResources,
) -> None:
    """Independently reload both roots and compare every repeat-evidence field."""

    expected = build_repeat_verification(
        matrix,
        first_run_root,
        second_run_root,
        first_resources=first_resources,
        second_resources=second_resources,
    )
    if evidence != expected:
        raise ProceduralReleaseValidationError(
            "repeat verification disagrees with strict artifact roots"
        )


def build_m3_repeat_evidence(
    matrix: LoadedExperimentMatrix,
    first_run_root: Path,
    second_run_root: Path,
    *,
    first_resources: RepeatRunResources,
    second_resources: RepeatRunResources,
) -> tuple[M3MatrixValidationV1, RepeatVerificationV1]:
    """Strictly rebuild and gate one complete two-run matrix evidence envelope.

    This is the canonical evidence producer. For the release matrix it is also
    the release-eligibility gate; for smoke it enforces the same repeat and
    artifact-integrity rules while retaining the schema's explicit CI-only
    identity status.
    """

    first_root = _absolute_real_run_root(first_run_root)
    second_root = _absolute_real_run_root(second_run_root)
    if os.path.samestat(first_root.stat(), second_root.stat()):
        raise ProceduralReleaseValidationError("M3 repeat evidence requires two distinct run roots")
    _require_independent_run_root_inodes(matrix, first_root, second_root)
    first_artifacts = load_m3_artifact_set(matrix, first_root)
    second_artifacts = load_m3_artifact_set(matrix, second_root)
    matrix_evidence = build_m3_matrix_validation(matrix, first_artifacts)
    repeat_evidence = _repeat_verification_from_artifacts(
        matrix,
        first_artifacts,
        second_artifacts,
        first_resources=first_resources,
        second_resources=second_resources,
    )
    if (
        matrix_evidence.matrix_id != repeat_evidence.matrix_id
        or matrix_evidence.matrix_sha256 != repeat_evidence.matrix_sha256
        or matrix_evidence.artifact_set_sha256 != repeat_evidence.first_run.artifact_set_sha256
    ):
        raise ProceduralReleaseValidationError(
            "matrix and repeat evidence do not identify the same first artifact set"
        )
    if not matrix_evidence.all_checks_passed:
        raise ProceduralReleaseValidationError("M3 matrix-wide identity gate did not pass")
    if not repeat_evidence.all_checks_passed:
        raise ProceduralReleaseValidationError("M3 deterministic repeat gate did not pass")
    return matrix_evidence, repeat_evidence


def validate_m3_release_eligibility(
    matrix_evidence: M3MatrixValidationV1,
    repeat_evidence: RepeatVerificationV1,
    *,
    matrix: LoadedExperimentMatrix,
    first_run_root: Path,
    second_run_root: Path,
    first_resources: RepeatRunResources,
    second_resources: RepeatRunResources,
) -> tuple[M3MatrixValidationV1, RepeatVerificationV1]:
    """Strictly rebuild and enforce every matrix-wide M3 release gate.

    The lower-level builders intentionally permit truthful failing records for
    diagnosis. This validator requires the frozen release matrix, delegates to
    the canonical strict producer, and exact-compares submitted records.
    """

    if matrix.matrix.matrix_id != "m3-procedural-v1":
        raise ProceduralReleaseValidationError(
            "M3 release eligibility requires the frozen release matrix"
        )
    rebuilt_matrix, rebuilt_repeat = build_m3_repeat_evidence(
        matrix,
        first_run_root,
        second_run_root,
        first_resources=first_resources,
        second_resources=second_resources,
    )
    if matrix_evidence != rebuilt_matrix:
        raise ProceduralReleaseValidationError(
            "submitted matrix evidence disagrees with strict first-run artifacts"
        )
    if repeat_evidence != rebuilt_repeat:
        raise ProceduralReleaseValidationError(
            "submitted repeat evidence disagrees with strict artifact roots"
        )
    return rebuilt_matrix, rebuilt_repeat
