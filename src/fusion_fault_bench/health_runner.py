"""Clean-source orchestration for the frozen M4 health benchmark."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from fusion_fault_bench.artifacts import ArtifactValidationError, derive_run_id
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.health_artifact_v1 import (
    HEALTH_EVAL_ARTIFACT_CONTRACT,
    HEALTH_FIT_ARTIFACT_CONTRACT,
)
from fusion_fault_bench.contracts.health_v1 import M4_HEALTH_INTENT_PATH
from fusion_fault_bench.contracts.result_v1alpha1 import (
    RunRecordV1Alpha1,
    RuntimeEnvironment,
)
from fusion_fault_bench.health_artifacts import (
    HealthEvaluationArtifactTransaction,
    HealthFitArtifactWriteRequest,
    LoadedHealthEvaluationArtifact,
    LoadedHealthFitArtifact,
    load_health_fit_artifact,
    write_health_fit_artifact,
)
from fusion_fault_bench.health_benchmark import fit_health_benchmark
from fusion_fault_bench.health_test_benchmark import stream_health_benchmark_test
from fusion_fault_bench.provenance import (
    CleanSourceSnapshot,
    collect_runtime_environment,
    discover_clean_source,
    verify_locked_execution,
)

_GENERATED_ROOT = Path("reports/generated")


class HealthRunnerError(ValueError):
    """M4 execution failed without weakening its frozen-source contract."""


def _initial_snapshot(intent_path: Path) -> CleanSourceSnapshot:
    try:
        snapshot = discover_clean_source(intent_path)
        verify_locked_execution(snapshot)
        return snapshot
    except (OSError, ValueError) as error:
        raise HealthRunnerError("M4 clean-source validation failed") from error


def _verify_unchanged_source(
    intent_path: Path,
    *,
    initial: CleanSourceSnapshot,
) -> None:
    try:
        final = discover_clean_source(intent_path)
        verify_locked_execution(final)
    except (OSError, ValueError) as error:
        raise HealthRunnerError("M4 final clean-source validation failed") from error
    if final != initial:
        raise HealthRunnerError("M4 source provenance changed during evaluation")


def _reject_symlink_components(path: Path, *, source_root: Path, label: str) -> None:
    current = source_root
    for part in path.parts:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise HealthRunnerError(f"M4 {label} must not use symlinks")


def _validated_generated_path(
    path: Path,
    *,
    source_root: Path,
    label: str,
) -> tuple[Path, str]:
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise HealthRunnerError(f"M4 {label} must use a normalized repository-relative path")
    try:
        path.relative_to(_GENERATED_ROOT)
    except ValueError:
        raise HealthRunnerError(f"M4 {label} must remain under reports/generated") from None
    if path == _GENERATED_ROOT:
        raise HealthRunnerError(f"M4 {label} requires an artifact-specific directory")
    _reject_symlink_components(path, source_root=source_root, label=label)
    return source_root / path, path.as_posix()


def _require_frozen_intent_snapshot(snapshot: CleanSourceSnapshot) -> None:
    if snapshot.manifest_relative_path != M4_HEALTH_INTENT_PATH.as_posix():
        raise HealthRunnerError("M4 source snapshot does not identify the frozen intent")


def _run_record(
    *,
    artifact_contract: str,
    command: tuple[str, ...],
    environment: RuntimeEnvironment,
    intent_sha256: str,
    snapshot: CleanSourceSnapshot,
    started_at: datetime,
    ended_at: datetime,
) -> RunRecordV1Alpha1:
    run_id = derive_run_id(
        manifest_sha256=intent_sha256,
        git_revision=snapshot.git_revision,
        lockfile_sha256=snapshot.lockfile_sha256,
        package_version=snapshot.package_version,
        artifact_contract=artifact_contract,
    )
    return RunRecordV1Alpha1(
        schema="ffb.run/v1alpha1",
        run_id=run_id,
        manifest_sha256=intent_sha256,
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


def fit_health_benchmark_artifact(
    *,
    output_dir: Path,
) -> LoadedHealthFitArtifact:
    """Fit the exact frozen M4 intent and publish one no-overwrite artifact."""

    snapshot = _initial_snapshot(M4_HEALTH_INTENT_PATH)
    _require_frozen_intent_snapshot(snapshot)
    destination, output_argument = _validated_generated_path(
        output_dir,
        source_root=snapshot.source_root,
        label="fit output",
    )
    if os.path.lexists(destination):
        raise FileExistsError("M4 health fit artifact destination already exists")

    environment = collect_runtime_environment()
    started_at = datetime.now(UTC)
    fit = fit_health_benchmark(source_root=snapshot.source_root)
    if not fit.validation.all_checks_passed:
        raise ArtifactValidationError("M4 health fit failed its release gates")
    ended_at = datetime.now(UTC)
    intent_sha256 = sha256_digest(fit.intent)
    run = _run_record(
        artifact_contract=HEALTH_FIT_ARTIFACT_CONTRACT,
        command=("ffb", "health", "fit", "--output-dir", output_argument),
        environment=environment,
        intent_sha256=intent_sha256,
        snapshot=snapshot,
        started_at=started_at,
        ended_at=ended_at,
    )
    request = HealthFitArtifactWriteRequest(
        intent=fit.intent,
        main_profile=fit.profiles.main_profile,
        edge_profile=fit.profiles.edge_profile,
        calibration=fit.calibration,
        candidates=fit.candidates,
        summary=fit.summary,
        validation=fit.validation,
        run=run,
    )
    _verify_unchanged_source(M4_HEALTH_INTENT_PATH, initial=snapshot)
    artifact = write_health_fit_artifact(
        request,
        destination,
        source_root=snapshot.source_root,
        git_metadata_dirs=(snapshot.git_dir, snapshot.git_common_dir),
    )
    _verify_unchanged_source(M4_HEALTH_INTENT_PATH, initial=snapshot)
    return artifact


def evaluate_health_benchmark_artifact(
    fit_artifact_path: Path,
    *,
    output_dir: Path,
) -> LoadedHealthEvaluationArtifact:
    """Apply an authenticated M4 fit and publish its exact frozen test matrix."""

    snapshot = _initial_snapshot(M4_HEALTH_INTENT_PATH)
    _require_frozen_intent_snapshot(snapshot)
    fit_path, fit_argument = _validated_generated_path(
        fit_artifact_path,
        source_root=snapshot.source_root,
        label="fit artifact",
    )
    destination, output_argument = _validated_generated_path(
        output_dir,
        source_root=snapshot.source_root,
        label="evaluation output",
    )
    if (
        destination == fit_path
        or destination.is_relative_to(fit_path)
        or fit_path.is_relative_to(destination)
    ):
        raise HealthRunnerError("M4 fit and evaluation artifact paths must be disjoint")
    if os.path.lexists(destination):
        raise FileExistsError("M4 health evaluation artifact destination already exists")

    fit_artifact = load_health_fit_artifact(fit_path)
    environment = collect_runtime_environment()
    started_at = datetime.now(UTC)
    with HealthEvaluationArtifactTransaction(
        destination,
        fit_artifact=fit_artifact,
        source_root=snapshot.source_root,
        git_metadata_dirs=(snapshot.git_dir, snapshot.git_common_dir),
    ) as transaction:
        evaluation = stream_health_benchmark_test(
            fit_artifact=fit_artifact,
            condition_sink=transaction.append_condition,
        )
        if not evaluation.validation.all_checks_passed:
            raise ArtifactValidationError("M4 health evaluation failed its release gates")
        ended_at = datetime.now(UTC)
        intent_sha256 = sha256_digest(fit_artifact.intent)
        run = _run_record(
            artifact_contract=HEALTH_EVAL_ARTIFACT_CONTRACT,
            command=(
                "ffb",
                "health",
                "evaluate",
                fit_argument,
                "--output-dir",
                output_argument,
            ),
            environment=environment,
            intent_sha256=intent_sha256,
            snapshot=snapshot,
            started_at=started_at,
            ended_at=ended_at,
        )
        _verify_unchanged_source(M4_HEALTH_INTENT_PATH, initial=snapshot)
        artifact = transaction.finalize(
            validation=evaluation.validation,
            run=run,
        )
    _verify_unchanged_source(M4_HEALTH_INTENT_PATH, initial=snapshot)
    return artifact
