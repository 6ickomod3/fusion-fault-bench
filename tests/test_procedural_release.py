from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

import fusion_fault_bench.procedural_release as release_module
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.bundle_v1alpha1 import (
    expected_conditions,
    expected_sequence_ids,
)
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AvailabilityControlManifest,
    ProceduralSource,
)
from fusion_fault_bench.contracts.matrix_v1 import (
    LoadedExperimentMatrix,
    load_experiment_matrix,
)
from fusion_fault_bench.contracts.procedural_artifact_v1 import (
    PROCEDURAL_ARTIFACT_CONTRACT,
    PROCEDURAL_ARTIFACT_PATHS,
    PROCEDURAL_CROSSOVERS_FILE,
    PROCEDURAL_INDEXED_PAYLOAD_PATHS,
    ProceduralPayloadFileEntryV1Alpha2,
    ProceduralPayloadIndexV1Alpha2,
)
from fusion_fault_bench.contracts.procedural_profile_v1 import ProceduralProfileV1
from fusion_fault_bench.contracts.procedural_release_v1 import (
    IDENTITY_ALLOWED_REMOVED_FIELDS,
    M3MatrixValidationV1,
    RepeatVerificationV1,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    LocalizationMetricRecord,
    MetricRecordV1Alpha1,
    RunRecordV1Alpha1,
    RuntimeEnvironment,
    SeverityCoordinate,
)
from fusion_fault_bench.procedural_artifacts import LoadedProceduralArtifact
from fusion_fault_bench.procedural_release import (
    ProceduralReleaseValidationError,
    RepeatRunResources,
    build_m3_matrix_validation,
    build_repeat_verification,
    compute_m3_artifact_set_digest,
    validate_m3_matrix_validation,
    validate_m3_release_eligibility,
    validate_repeat_verification,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
RELEASE_MATRIX_PATH = Path("examples/matrices/m3-procedural-v1.json")
SMOKE_MATRIX_PATH = Path("examples/matrices/m3-ci-smoke-v1.json")
_CPU_MODEL = "Test M3 CPU"


@pytest.fixture(scope="module")
def release_matrix() -> LoadedExperimentMatrix:
    return load_experiment_matrix(
        RELEASE_MATRIX_PATH,
        source_root=REPOSITORY_ROOT,
    )


@pytest.fixture(scope="module")
def smoke_matrix() -> LoadedExperimentMatrix:
    return load_experiment_matrix(
        SMOKE_MATRIX_PATH,
        source_root=REPOSITORY_ROOT,
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _profile_for(
    matrix: LoadedExperimentMatrix,
    profile_id: str,
) -> ProceduralProfileV1:
    return next(profile for profile in matrix.profiles if profile.profile_id == profile_id)


def _identity_rows(
    manifest: Any,
    *,
    run_id: str,
    eligible_count: int,
) -> tuple[MetricRecordV1Alpha1, ...]:
    if manifest.source.profile_id != "constant-velocity-front-roi-v1":
        return ()
    condition = expected_conditions(manifest)[0]
    severity = SeverityCoordinate(
        index=condition.severity_index,
        magnitude=condition.magnitude,
        direction=cast(Any, condition.direction),
        unit=cast(Any, condition.unit),
    )
    availability = isinstance(manifest, AvailabilityControlManifest)
    metric_name = "conditional-matched-center-mse" if availability else "matched-center-mse"
    method_values = {
        "camera-only": 1.0,
        "lidar-only": 0.09,
        "fixed-fusion": 0.0826,
        "fault-target-drop-policy": 0.0826,
        "performance-oracle": 0.0826,
    }
    return tuple(
        LocalizationMetricRecord(
            schema="ffb.sequence-metric/v1alpha1",
            record_level="sequence",
            run_id=run_id,
            manifest_sha256=sha256_digest(manifest),
            sequence_id=sequence_id,
            fault_family=cast(Any, condition.fault_family),
            fault_axis=cast(Any, condition.fault_axis),
            severity=severity,
            method_id=method,
            eligible_object_frame_count=eligible_count,
            valid_object_frame_count=eligible_count,
            metric_name=metric_name,
            status="ok",
            value=method_values[method] + sequence_index / 1_000_000.0,
            unit="m^2",
        )
        for sequence_index, sequence_id in enumerate(expected_sequence_ids(manifest))
        for method in manifest.methods
        if availability or method in method_values
    )


def _fake_artifact(
    matrix: LoadedExperimentMatrix,
    execution_index: int,
    *,
    digest_salt: str = "shared",
) -> LoadedProceduralArtifact:
    manifest = matrix.manifests[execution_index]
    source = cast(ProceduralSource, manifest.source)
    profile = _profile_for(matrix, source.profile_id)
    manifest_sha256 = sha256_digest(manifest)
    profile_sha256 = sha256_digest(profile)
    run_id = f"run:{_digest(f'{manifest.experiment}:run')}"
    artifact_sha256 = _digest(
        f"{matrix.matrix.matrix_id}:{manifest.experiment}:artifact:{digest_salt}"
    )
    now = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)
    run = RunRecordV1Alpha1(
        schema="ffb.run/v1alpha1",
        run_id=run_id,
        manifest_sha256=manifest_sha256,
        package_version="0.1.0",
        git_revision="a" * 40,
        source_dirty=False,
        lockfile_sha256="b" * 64,
        command=("ffb", "procedural", "matrix", "run"),
        environment=RuntimeEnvironment(
            python_version="3.12.13",
            os_name="Darwin",
            os_release="25.0",
            machine="arm64",
            cpu_model=_CPU_MODEL,
            logical_cpu_count=4,
            memory_bytes=16 * 1024**3,
        ),
        started_at=now,
        ended_at=now + timedelta(seconds=1),
        status="succeeded",
        artifact_sha256=artifact_sha256,
    )
    files = tuple(
        ProceduralPayloadFileEntryV1Alpha2(
            path=path,
            byte_length=(
                0
                if path == PROCEDURAL_CROSSOVERS_FILE
                and isinstance(manifest, AvailabilityControlManifest)
                else 1
            ),
            sha256=_digest(f"{manifest.experiment}:{path}:{digest_salt}"),
        )
        for path in PROCEDURAL_INDEXED_PAYLOAD_PATHS
    )
    payload_index = ProceduralPayloadIndexV1Alpha2(
        schema="ffb.payload-index/v1alpha2",
        artifact_contract=PROCEDURAL_ARTIFACT_CONTRACT,
        run_id=run_id,
        manifest_sha256=manifest_sha256,
        profile_sha256=profile_sha256,
        files=files,
    )
    eligible_count = profile.source.frame_count * profile.source.object_count
    return LoadedProceduralArtifact(
        path=Path("unused") / manifest.experiment,
        manifest=cast(Any, manifest),
        profile=profile,
        metrics=_identity_rows(
            manifest,
            run_id=run_id,
            eligible_count=eligible_count,
        ),
        aggregates=(),
        crossovers=(),
        validation=cast(Any, None),
        payload_index=payload_index,
        run=run,
        artifact_sha256=artifact_sha256,
        run_sha256=_digest(f"{manifest.experiment}:run-record"),
    )


def _fake_artifact_set(
    matrix: LoadedExperimentMatrix,
    *,
    digest_salt: str = "shared",
) -> tuple[LoadedProceduralArtifact, ...]:
    return tuple(
        _fake_artifact(matrix, index, digest_salt=digest_salt)
        for index in range(len(matrix.manifests))
    )


def _mutate_identity_value(
    artifact: LoadedProceduralArtifact,
) -> LoadedProceduralArtifact:
    rows = list(artifact.metrics)
    row = cast(LocalizationMetricRecord, rows[0])
    assert row.value is not None
    rows[0] = row.model_copy(update={"value": row.value + 0.5})
    return replace(artifact, metrics=tuple(rows))


def _mutate_indexed_member(
    artifact: LoadedProceduralArtifact,
) -> LoadedProceduralArtifact:
    files = list(artifact.payload_index.files)
    files[2] = files[2].model_copy(update={"sha256": _digest("one-indexed-member-mismatch")})
    payload_index = artifact.payload_index.model_copy(update={"files": tuple(files)})
    artifact_sha256 = _digest("second-artifact-set-differs")
    run = artifact.run.model_copy(update={"artifact_sha256": artifact_sha256})
    return replace(
        artifact,
        payload_index=payload_index,
        run=run,
        artifact_sha256=artifact_sha256,
    )


def _second_run_artifacts(
    artifacts: tuple[LoadedProceduralArtifact, ...],
) -> tuple[LoadedProceduralArtifact, ...]:
    return tuple(
        replace(
            artifact,
            path=Path("unused") / "second" / artifact.manifest.experiment,
            run_sha256=_digest(f"{artifact.manifest.experiment}:second-run-record"),
        )
        for artifact in artifacts
    )


def test_release_identity_evidence_normalizes_every_main_peer_and_excludes_edge(
    release_matrix: LoadedExperimentMatrix,
) -> None:
    artifacts = _fake_artifact_set(release_matrix)

    evidence = build_m3_matrix_validation(release_matrix, artifacts)

    assert isinstance(evidence, M3MatrixValidationV1)
    assert evidence.schema_id == "ffb.m3-matrix-validation/v1"
    assert evidence.artifact_count == 8
    assert evidence.all_checks_passed
    identity = evidence.identity_comparison
    assert identity.status == "applicable"
    assert identity.allowed_removed_fields == IDENTITY_ALLOWED_REMOVED_FIELDS
    assert identity.dropout_metric_mapping.source_metric_name == ("conditional-matched-center-mse")
    assert identity.dropout_metric_mapping.destination_metric_name == ("matched-center-mse")
    assert identity.normalized_row_count == 6_800
    assert identity.distinct_normalized_key_count == 1_000
    assert identity.comparison_count == 5_800
    assert identity.mismatch_count == 0
    assert identity.all_equal
    assert identity.excluded_artifacts[0].execution_index == 7
    validate_m3_matrix_validation(
        evidence,
        matrix=release_matrix,
        artifacts=artifacts,
    )


def test_retained_one_field_identity_mutation_is_detected(
    release_matrix: LoadedExperimentMatrix,
) -> None:
    artifacts = _fake_artifact_set(release_matrix)
    passing = build_m3_matrix_validation(release_matrix, artifacts)
    mutated = list(artifacts)
    mutated[1] = _mutate_identity_value(mutated[1])

    failing = build_m3_matrix_validation(release_matrix, mutated)

    assert failing.identity_comparison.status == "applicable"
    assert failing.identity_comparison.mismatch_count == 1
    assert failing.identity_comparison.maximum_absolute_value_discrepancy_m2 == 0.5
    assert not failing.identity_comparison.all_equal
    assert not failing.all_checks_passed
    with pytest.raises(
        ProceduralReleaseValidationError,
        match="matrix validation disagrees",
    ):
        validate_m3_matrix_validation(
            passing,
            matrix=release_matrix,
            artifacts=mutated,
        )


def test_release_contract_rejects_contradictory_identity_commitments(
    release_matrix: LoadedExperimentMatrix,
) -> None:
    evidence = build_m3_matrix_validation(
        release_matrix,
        _fake_artifact_set(release_matrix),
    )

    wrong_profile = evidence.model_dump(mode="python", by_alias=True)
    ordered_artifacts = cast(
        tuple[dict[str, Any], ...],
        wrong_profile["ordered_artifacts"],
    )
    ordered_artifacts[0]["profile_id"] = "bogus-profile"
    with pytest.raises(ValidationError, match="profiles disagree"):
        M3MatrixValidationV1.model_validate(wrong_profile)

    wrong_row_commitment = evidence.model_dump(mode="python", by_alias=True)
    identity = cast(dict[str, Any], wrong_row_commitment["identity_comparison"])
    included = cast(
        tuple[dict[str, Any], ...],
        identity["included_artifacts"],
    )
    included[1]["normalized_rows_sha256"] = _digest("contradictory-row-commitment")
    with pytest.raises(ValidationError, match="row commitments"):
        M3MatrixValidationV1.model_validate(wrong_row_commitment)

    wrong_removed_fields = evidence.model_dump(mode="python", by_alias=True)
    identity = cast(dict[str, Any], wrong_removed_fields["identity_comparison"])
    removed_fields = cast(tuple[str, ...], identity["allowed_removed_fields"])
    identity["allowed_removed_fields"] = tuple(reversed(removed_fields))
    with pytest.raises(ValidationError, match="non-preregistered removed field"):
        M3MatrixValidationV1.model_validate(wrong_removed_fields)

    wrong_artifact_set = evidence.model_dump(mode="python", by_alias=True)
    wrong_artifact_set["artifact_set_sha256"] = _digest("wrong-artifact-set")
    with pytest.raises(ValidationError, match="artifact-set digest"):
        M3MatrixValidationV1.model_validate(wrong_artifact_set)

    wrong_profile_digest = evidence.model_dump(mode="python", by_alias=True)
    ordered_artifacts = cast(
        tuple[dict[str, Any], ...],
        wrong_profile_digest["ordered_artifacts"],
    )
    ordered_artifacts[0]["profile_sha256"] = _digest("wrong-main-profile")
    with pytest.raises(ValidationError, match="profile digests disagree"):
        M3MatrixValidationV1.model_validate(wrong_profile_digest)


def test_smoke_matrix_has_explicit_single_manifest_no_peer_status(
    smoke_matrix: LoadedExperimentMatrix,
) -> None:
    artifacts = _fake_artifact_set(smoke_matrix)

    evidence = build_m3_matrix_validation(smoke_matrix, artifacts)

    assert evidence.identity_comparison.status == ("not-applicable-single-manifest-smoke")
    assert evidence.identity_comparison.comparison_count == 0
    assert evidence.all_checks_passed


def _make_run_roots(
    tmp_path: Path,
    matrix: LoadedExperimentMatrix,
) -> tuple[Path, Path]:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for manifest in matrix.manifests:
        first_artifact = first / manifest.experiment
        second_artifact = second / manifest.experiment
        first_artifact.mkdir()
        second_artifact.mkdir()
        for member in PROCEDURAL_ARTIFACT_PATHS:
            (first_artifact / member).write_bytes(b"first")
            (second_artifact / member).write_bytes(b"second")
    return first, second


def test_release_repeat_verification_has_all_48_ordered_member_pairs(
    release_matrix: LoadedExperimentMatrix,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root, second_root = _make_run_roots(tmp_path, release_matrix)
    first_artifacts = _fake_artifact_set(release_matrix)
    second_artifacts = _second_run_artifacts(first_artifacts)
    artifacts_by_root = {
        first_root: {artifact.manifest.experiment: artifact for artifact in first_artifacts},
        second_root: {artifact.manifest.experiment: artifact for artifact in second_artifacts},
    }

    def fake_load(path: Path) -> LoadedProceduralArtifact:
        return artifacts_by_root[path.parent][path.name]

    monkeypatch.setattr(release_module, "load_procedural_artifact", fake_load)
    first_resources = RepeatRunResources(
        wall_time_seconds=100.25,
        peak_memory_bytes=1_000_000,
    )
    second_resources = RepeatRunResources(
        wall_time_seconds=101.5,
        peak_memory_bytes=1_100_000,
    )

    evidence = build_repeat_verification(
        release_matrix,
        first_root,
        second_root,
        first_resources=first_resources,
        second_resources=second_resources,
    )

    assert evidence.comparison_count == 48
    assert evidence.mismatch_count == 0
    assert evidence.all_equal
    assert evidence.same_named_cpu
    assert evidence.all_checks_passed
    assert evidence.first_run.cpu_model == _CPU_MODEL
    assert evidence.first_run.wall_time_seconds == first_resources.wall_time_seconds
    assert evidence.first_run.peak_memory_bytes == first_resources.peak_memory_bytes
    assert evidence.second_run.wall_time_seconds == second_resources.wall_time_seconds
    assert evidence.second_run.peak_memory_bytes == second_resources.peak_memory_bytes
    assert tuple(
        (
            pair.execution_index,
            pair.experiment,
            pair.manifest_sha256,
            pair.path,
        )
        for pair in evidence.indexed_member_pairs
    ) == tuple(
        (
            execution_index,
            manifest.experiment,
            sha256_digest(manifest),
            path,
        )
        for execution_index, manifest in enumerate(release_matrix.manifests)
        for path in PROCEDURAL_INDEXED_PAYLOAD_PATHS
    )

    wrong_experiment = evidence.model_dump(mode="python", by_alias=True)
    pairs = cast(
        tuple[dict[str, Any], ...],
        wrong_experiment["indexed_member_pairs"],
    )
    for pair in pairs[: len(PROCEDURAL_INDEXED_PAYLOAD_PATHS)]:
        pair["experiment"] = "wrong-experiment"
    with pytest.raises(ValidationError, match="artifact identity"):
        RepeatVerificationV1.model_validate(wrong_experiment)

    copied_run_records = evidence.model_dump(mode="python", by_alias=True)
    first_run = cast(dict[str, Any], copied_run_records["first_run"])
    second_run = cast(dict[str, Any], copied_run_records["second_run"])
    second_run["run_record_sha256s"] = first_run["run_record_sha256s"]
    with pytest.raises(ValidationError, match="distinct volatile run records"):
        RepeatVerificationV1.model_validate(copied_run_records)


def test_repeat_verification_rejects_same_or_incomplete_run_roots(
    smoke_matrix: LoadedExperimentMatrix,
    tmp_path: Path,
) -> None:
    same_root = tmp_path / "same"
    same_root.mkdir()
    resources = RepeatRunResources(
        wall_time_seconds=1.0,
        peak_memory_bytes=1,
    )
    with pytest.raises(
        ProceduralReleaseValidationError,
        match="two distinct run roots",
    ):
        build_repeat_verification(
            smoke_matrix,
            same_root,
            same_root,
            first_resources=resources,
            second_resources=resources,
        )

    first_root, second_root = _make_run_roots(tmp_path, smoke_matrix)
    (first_root / "unexpected-artifact").mkdir()
    with pytest.raises(
        ProceduralReleaseValidationError,
        match="artifact allowlist",
    ):
        build_repeat_verification(
            smoke_matrix,
            first_root,
            second_root,
            first_resources=resources,
            second_resources=resources,
        )


def test_repeat_verification_rejects_hard_linked_members(
    smoke_matrix: LoadedExperimentMatrix,
    tmp_path: Path,
) -> None:
    first_root, second_root = _make_run_roots(tmp_path, smoke_matrix)
    experiment = smoke_matrix.manifests[0].experiment
    member = PROCEDURAL_INDEXED_PAYLOAD_PATHS[0]
    second_member = second_root / experiment / member
    second_member.unlink()
    os.link(first_root / experiment / member, second_member)
    resources = RepeatRunResources(wall_time_seconds=1.0, peak_memory_bytes=1)

    with pytest.raises(
        ProceduralReleaseValidationError,
        match="members must have independent inodes",
    ):
        build_repeat_verification(
            smoke_matrix,
            first_root,
            second_root,
            first_resources=resources,
            second_resources=resources,
        )


def test_repeat_verification_reports_one_indexed_member_mismatch(
    smoke_matrix: LoadedExperimentMatrix,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root, second_root = _make_run_roots(tmp_path, smoke_matrix)
    first_artifact = _fake_artifact_set(smoke_matrix)[0]
    equal_second_artifact = replace(
        first_artifact,
        path=Path("unused") / "second" / first_artifact.manifest.experiment,
        run_sha256=_digest(f"{first_artifact.manifest.experiment}:second-run-record"),
    )
    artifacts_by_root = {
        first_root: first_artifact,
        second_root: equal_second_artifact,
    }

    def fake_load(path: Path) -> LoadedProceduralArtifact:
        return artifacts_by_root[path.parent]

    monkeypatch.setattr(release_module, "load_procedural_artifact", fake_load)
    first_resources = RepeatRunResources(
        wall_time_seconds=10.0,
        peak_memory_bytes=100_000,
    )
    second_resources = RepeatRunResources(
        wall_time_seconds=11.0,
        peak_memory_bytes=110_000,
    )
    passing = build_repeat_verification(
        smoke_matrix,
        first_root,
        second_root,
        first_resources=first_resources,
        second_resources=second_resources,
    )
    assert passing.all_equal
    assert passing.comparison_count == 6
    assert passing.mismatch_count == 0
    assert passing.first_run.artifact_set_sha256 == (
        compute_m3_artifact_set_digest(smoke_matrix, (first_artifact,))
    )
    with pytest.raises(
        ProceduralReleaseValidationError,
        match="finite and positive",
    ):
        build_repeat_verification(
            smoke_matrix,
            first_root,
            second_root,
            first_resources=RepeatRunResources(
                wall_time_seconds=float("nan"),
                peak_memory_bytes=100_000,
            ),
            second_resources=second_resources,
        )

    artifacts_by_root[second_root] = _mutate_indexed_member(equal_second_artifact)
    failing = build_repeat_verification(
        smoke_matrix,
        first_root,
        second_root,
        first_resources=first_resources,
        second_resources=second_resources,
    )

    assert isinstance(failing, RepeatVerificationV1)
    assert failing.comparison_count == 6
    assert failing.mismatch_count == 1
    assert sum(not pair.equal for pair in failing.indexed_member_pairs) == 1
    assert failing.indexed_member_pairs[2].path == "sequence-metrics.ndjson"
    assert not failing.all_equal
    assert failing.same_named_cpu
    assert not failing.all_checks_passed
    validate_repeat_verification(
        failing,
        matrix=smoke_matrix,
        first_run_root=first_root,
        second_run_root=second_root,
        first_resources=first_resources,
        second_resources=second_resources,
    )
    with pytest.raises(
        ProceduralReleaseValidationError,
        match="repeat verification disagrees",
    ):
        validate_repeat_verification(
            passing,
            matrix=smoke_matrix,
            first_run_root=first_root,
            second_run_root=second_root,
            first_resources=first_resources,
            second_resources=second_resources,
        )


def _patch_release_roots(
    monkeypatch: pytest.MonkeyPatch,
    *,
    first_root: Path,
    second_root: Path,
    first_artifacts: tuple[LoadedProceduralArtifact, ...],
    second_artifacts: tuple[LoadedProceduralArtifact, ...],
) -> None:
    artifacts_by_root = {
        first_root: {artifact.manifest.experiment: artifact for artifact in first_artifacts},
        second_root: {artifact.manifest.experiment: artifact for artifact in second_artifacts},
    }

    def fake_load(path: Path) -> LoadedProceduralArtifact:
        return artifacts_by_root[path.parent][path.name]

    monkeypatch.setattr(release_module, "load_procedural_artifact", fake_load)


def test_release_eligibility_strictly_rebuilds_both_release_roots(
    release_matrix: LoadedExperimentMatrix,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root, second_root = _make_run_roots(tmp_path, release_matrix)
    first_artifacts = _fake_artifact_set(release_matrix)
    second_artifacts = _second_run_artifacts(first_artifacts)
    _patch_release_roots(
        monkeypatch,
        first_root=first_root,
        second_root=second_root,
        first_artifacts=first_artifacts,
        second_artifacts=second_artifacts,
    )
    resources = RepeatRunResources(wall_time_seconds=1.0, peak_memory_bytes=1)
    matrix_evidence = build_m3_matrix_validation(release_matrix, first_artifacts)
    repeat_evidence = build_repeat_verification(
        release_matrix,
        first_root,
        second_root,
        first_resources=resources,
        second_resources=resources,
    )

    rebuilt_matrix, rebuilt_repeat = validate_m3_release_eligibility(
        matrix_evidence,
        repeat_evidence,
        matrix=release_matrix,
        first_run_root=first_root,
        second_run_root=second_root,
        first_resources=resources,
        second_resources=resources,
    )

    assert rebuilt_matrix == matrix_evidence
    assert rebuilt_repeat == repeat_evidence


def test_release_eligibility_rejects_smoke_and_truthful_failed_gates(
    release_matrix: LoadedExperimentMatrix,
    smoke_matrix: LoadedExperimentMatrix,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = RepeatRunResources(wall_time_seconds=1.0, peak_memory_bytes=1)
    smoke_artifacts = _fake_artifact_set(smoke_matrix)
    smoke_matrix_evidence = build_m3_matrix_validation(smoke_matrix, smoke_artifacts)
    with pytest.raises(
        ProceduralReleaseValidationError,
        match="requires the frozen release matrix",
    ):
        validate_m3_release_eligibility(
            smoke_matrix_evidence,
            cast(Any, None),
            matrix=smoke_matrix,
            first_run_root=tmp_path / "missing-first",
            second_run_root=tmp_path / "missing-second",
            first_resources=resources,
            second_resources=resources,
        )

    first_root, second_root = _make_run_roots(tmp_path, release_matrix)
    passing_artifacts = _fake_artifact_set(release_matrix)
    matrix_failing_artifacts = list(passing_artifacts)
    matrix_failing_artifacts[1] = _mutate_identity_value(matrix_failing_artifacts[1])
    matrix_failing_tuple = tuple(matrix_failing_artifacts)
    matrix_failing_second = _second_run_artifacts(matrix_failing_tuple)
    _patch_release_roots(
        monkeypatch,
        first_root=first_root,
        second_root=second_root,
        first_artifacts=matrix_failing_tuple,
        second_artifacts=matrix_failing_second,
    )
    failing_matrix = build_m3_matrix_validation(release_matrix, matrix_failing_tuple)
    passing_repeat = build_repeat_verification(
        release_matrix,
        first_root,
        second_root,
        first_resources=resources,
        second_resources=resources,
    )
    with pytest.raises(ProceduralReleaseValidationError, match="identity gate did not pass"):
        validate_m3_release_eligibility(
            failing_matrix,
            passing_repeat,
            matrix=release_matrix,
            first_run_root=first_root,
            second_run_root=second_root,
            first_resources=resources,
            second_resources=resources,
        )

    second_failing = list(_second_run_artifacts(passing_artifacts))
    second_failing[0] = _mutate_indexed_member(second_failing[0])
    second_failing_tuple = tuple(second_failing)
    _patch_release_roots(
        monkeypatch,
        first_root=first_root,
        second_root=second_root,
        first_artifacts=passing_artifacts,
        second_artifacts=second_failing_tuple,
    )
    passing_matrix = build_m3_matrix_validation(release_matrix, passing_artifacts)
    failing_repeat = build_repeat_verification(
        release_matrix,
        first_root,
        second_root,
        first_resources=resources,
        second_resources=resources,
    )
    with pytest.raises(ProceduralReleaseValidationError, match="repeat gate did not pass"):
        validate_m3_release_eligibility(
            passing_matrix,
            failing_repeat,
            matrix=release_matrix,
            first_run_root=first_root,
            second_run_root=second_root,
            first_resources=resources,
            second_resources=resources,
        )


def test_release_eligibility_rejects_cross_link_and_whitespace_cpu(
    release_matrix: LoadedExperimentMatrix,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root, second_root = _make_run_roots(tmp_path, release_matrix)
    artifacts = _fake_artifact_set(release_matrix)
    second_artifacts = _second_run_artifacts(artifacts)
    _patch_release_roots(
        monkeypatch,
        first_root=first_root,
        second_root=second_root,
        first_artifacts=artifacts,
        second_artifacts=second_artifacts,
    )
    resources = RepeatRunResources(wall_time_seconds=1.0, peak_memory_bytes=1)
    matrix_evidence = build_m3_matrix_validation(release_matrix, artifacts)
    repeat_evidence = build_repeat_verification(
        release_matrix,
        first_root,
        second_root,
        first_resources=resources,
        second_resources=resources,
    )
    wrong_set_digest = _digest("independently-valid-but-wrong-set")
    wrong_repeat = repeat_evidence.model_copy(
        update={
            "first_run": repeat_evidence.first_run.model_copy(
                update={"artifact_set_sha256": wrong_set_digest}
            ),
            "second_run": repeat_evidence.second_run.model_copy(
                update={"artifact_set_sha256": wrong_set_digest}
            ),
        }
    )
    monkeypatch.setattr(
        release_module,
        "_repeat_verification_from_artifacts",
        lambda *args, **kwargs: wrong_repeat,
    )
    with pytest.raises(
        ProceduralReleaseValidationError,
        match="same first artifact set",
    ):
        validate_m3_release_eligibility(
            matrix_evidence,
            wrong_repeat,
            matrix=release_matrix,
            first_run_root=first_root,
            second_run_root=second_root,
            first_resources=resources,
            second_resources=resources,
        )

    payload = repeat_evidence.model_dump(mode="python", by_alias=True)
    cast(dict[str, Any], payload["first_run"])["cpu_model"] = "   "
    with pytest.raises(ValidationError, match="non-whitespace"):
        RepeatVerificationV1.model_validate(payload)
