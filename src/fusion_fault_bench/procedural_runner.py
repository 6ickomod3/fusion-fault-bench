"""Clean-source orchestration for the frozen M3 procedural matrices."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from fusion_fault_bench.artifacts import ArtifactValidationError, derive_run_id
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AvailabilityControlManifest,
    CommonModeControlManifest,
    GeometryCrossoverManifest,
    ProceduralSource,
)
from fusion_fault_bench.contracts.matrix_v1 import load_experiment_matrix
from fusion_fault_bench.contracts.result_v1alpha1 import RunRecordV1Alpha1
from fusion_fault_bench.experiments.procedural import (
    generate_procedural_sequence_metrics,
)
from fusion_fault_bench.procedural_artifacts import (
    PROCEDURAL_ARTIFACT_CONTRACT,
    LoadedProceduralArtifact,
    ProceduralArtifactWriteRequest,
    write_procedural_artifact,
)
from fusion_fault_bench.procedural_evaluation import (
    evaluate_procedural_records,
    validate_evaluated_procedural_records,
)
from fusion_fault_bench.procedural_release import build_m3_matrix_validation
from fusion_fault_bench.procedural_validation import build_procedural_validation
from fusion_fault_bench.provenance import (
    CleanSourceSnapshot,
    collect_runtime_environment,
    discover_clean_source,
    verify_locked_execution,
)
from fusion_fault_bench.scenarios.procedural import generate_procedural_sequences

type ProceduralManifest = (
    GeometryCrossoverManifest | AvailabilityControlManifest | CommonModeControlManifest
)

_GENERATED_ROOT = Path("reports/generated")


class ProceduralRunnerError(ValueError):
    """M3 execution failed without weakening its frozen-source contract."""


def _initial_snapshot(matrix_path: Path) -> CleanSourceSnapshot:
    try:
        snapshot = discover_clean_source(matrix_path)
        verify_locked_execution(snapshot)
        return snapshot
    except (OSError, ValueError) as error:
        raise ProceduralRunnerError("M3 clean-source validation failed") from error


def _verify_unchanged_source(
    matrix_path: Path,
    *,
    initial: CleanSourceSnapshot,
) -> None:
    try:
        final = discover_clean_source(matrix_path)
        verify_locked_execution(final)
    except (OSError, ValueError) as error:
        raise ProceduralRunnerError("M3 final clean-source validation failed") from error
    if final != initial:
        raise ProceduralRunnerError("M3 source provenance changed during evaluation")


def _validated_output_root(
    output_dir: Path,
    *,
    source_root: Path,
) -> tuple[Path, str]:
    if output_dir.is_absolute() or any(part in {".", ".."} for part in output_dir.parts):
        raise ProceduralRunnerError("M3 output must use a normalized repository-relative path")
    try:
        output_dir.relative_to(_GENERATED_ROOT)
    except ValueError:
        raise ProceduralRunnerError("M3 output must remain under reports/generated") from None
    if output_dir == _GENERATED_ROOT:
        raise ProceduralRunnerError("M3 output requires a matrix-specific directory")
    absolute = source_root / output_dir
    if os.path.lexists(absolute) and absolute.is_symlink():
        raise ProceduralRunnerError("M3 output root must not be a symlink")
    return absolute, output_dir.as_posix()


def _procedural_manifest(value: object) -> ProceduralManifest:
    if isinstance(
        value,
        (GeometryCrossoverManifest, AvailabilityControlManifest, CommonModeControlManifest),
    ):
        if not isinstance(value.source, ProceduralSource):
            raise ProceduralRunnerError("M3 matrix manifest is not procedural")
        return value
    raise ProceduralRunnerError("M3 matrix contains an unsupported manifest kind")


def run_procedural_matrix(
    matrix_path: Path,
    *,
    output_dir: Path,
) -> tuple[LoadedProceduralArtifact, ...]:
    """Execute, validate, and publish every entry in one frozen M3 matrix."""

    snapshot = _initial_snapshot(matrix_path)
    loaded_matrix = load_experiment_matrix(matrix_path, source_root=snapshot.source_root)
    expected_matrix_path = loaded_matrix.path.relative_to(snapshot.source_root).as_posix()
    if snapshot.manifest_relative_path != expected_matrix_path:
        raise ProceduralRunnerError("M3 source snapshot does not identify the loaded matrix")
    output_root, output_argument = _validated_output_root(
        output_dir,
        source_root=snapshot.source_root,
    )
    profiles = {profile.profile_id: profile for profile in loaded_matrix.profiles}
    destinations = tuple(
        output_root / _procedural_manifest(manifest).experiment
        for manifest in loaded_matrix.manifests
    )
    if len(set(destinations)) != len(destinations):
        raise ProceduralRunnerError("M3 matrix resolves duplicate artifact destinations")
    if any(os.path.lexists(destination) for destination in destinations):
        raise FileExistsError("M3 artifact destination already exists")

    environment = collect_runtime_environment()
    command = (
        "ffb",
        "procedural",
        "matrix",
        "run",
        snapshot.manifest_relative_path,
        "--output-dir",
        output_argument,
    )
    requests: list[tuple[ProceduralArtifactWriteRequest, Path]] = []
    for raw_manifest, destination in zip(
        loaded_matrix.manifests,
        destinations,
        strict=True,
    ):
        manifest = _procedural_manifest(raw_manifest)
        source = manifest.source
        if not isinstance(source, ProceduralSource):
            raise ProceduralRunnerError("M3 matrix manifest is not procedural")
        profile = profiles[source.profile_id]
        manifest_sha256 = sha256_digest(manifest)
        run_id = derive_run_id(
            manifest_sha256=manifest_sha256,
            git_revision=snapshot.git_revision,
            lockfile_sha256=snapshot.lockfile_sha256,
            package_version=snapshot.package_version,
            artifact_contract=PROCEDURAL_ARTIFACT_CONTRACT,
        )
        started_at = datetime.now(UTC)
        sequences = generate_procedural_sequences(
            profile,
            split=source.split,
            sequence_count=source.sequence_count,
            data_master_seed=manifest.rng.data_master_seed,
        )
        metrics = generate_procedural_sequence_metrics(
            manifest,
            profile=profile,
            run_id=run_id,
        )
        evaluated = evaluate_procedural_records(
            manifest,
            run_id=run_id,
            metrics=metrics,
        )
        validation = build_procedural_validation(
            manifest,
            profile=profile,
            run_id=run_id,
            sequences=sequences,
            metrics=evaluated.metrics,
        )
        if not validation.all_checks_passed:
            raise ArtifactValidationError("M3 procedural validation failed its release gates")
        ended_at = datetime.now(UTC)
        run = RunRecordV1Alpha1(
            schema="ffb.run/v1alpha1",
            run_id=run_id,
            manifest_sha256=manifest_sha256,
            package_version=snapshot.package_version,
            git_revision=snapshot.git_revision,
            source_dirty=False,
            lockfile_sha256=snapshot.lockfile_sha256,
            command=command,
            environment=environment,
            started_at=started_at,
            ended_at=ended_at,
            status="succeeded",
            artifact_sha256="0" * 64,
        )
        validate_evaluated_procedural_records(
            manifest,
            run=run,
            records=evaluated,
        )
        requests.append(
            (
                ProceduralArtifactWriteRequest(
                    manifest=manifest,
                    profile=profile,
                    metrics=evaluated.metrics,
                    aggregates=evaluated.aggregates,
                    crossovers=evaluated.crossovers,
                    validation=validation,
                    run=run,
                ),
                destination,
            )
        )

    _verify_unchanged_source(matrix_path, initial=snapshot)
    artifacts = tuple(
        write_procedural_artifact(
            request,
            destination,
            source_root=snapshot.source_root,
            git_metadata_dirs=(snapshot.git_dir, snapshot.git_common_dir),
        )
        for request, destination in requests
    )
    matrix_validation = build_m3_matrix_validation(loaded_matrix, artifacts)
    if not matrix_validation.all_checks_passed:
        raise ArtifactValidationError("M3 matrix-level validation failed its release gates")
    _verify_unchanged_source(matrix_path, initial=snapshot)
    return artifacts
