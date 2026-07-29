"""Clean-source orchestration for the CPU-only M1 analytic experiment."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fusion_fault_bench.artifacts import (
    MAX_BOOTSTRAP_CELLS,
    MAX_BOOTSTRAP_REPLICATES,
    MAX_SEQUENCE_COUNT,
    MAX_SEQUENCE_ROWS,
    ArtifactValidationError,
    ArtifactWriteRequest,
    LoadedArtifact,
    derive_run_id,
    write_artifact,
)
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.bundle_v1alpha1 import expected_conditions
from fusion_fault_bench.contracts.io import load_manifest
from fusion_fault_bench.contracts.manifest_v1alpha1 import AnalyticCrossoverManifest
from fusion_fault_bench.contracts.result_v1alpha1 import RunRecordV1Alpha1
from fusion_fault_bench.evaluation import evaluate_analytic_records
from fusion_fault_bench.experiments.analytic import generate_analytic_sequence_metrics
from fusion_fault_bench.provenance import (
    collect_runtime_environment,
    discover_clean_source,
    logical_reproduction_command,
    verify_locked_execution,
)
from fusion_fault_bench.validation import build_analytic_validation


def preflight_analytic_manifest(manifest: AnalyticCrossoverManifest) -> None:
    """Reject an analytic experiment that exceeds the frozen CPU resource caps."""

    sequence_count = manifest.source.sequence_count
    bootstrap_replicates = manifest.evaluation.bootstrap.replicates
    sequence_rows = sequence_count * len(expected_conditions(manifest)) * len(manifest.methods)
    if sequence_count > MAX_SEQUENCE_COUNT:
        raise ArtifactValidationError("sequence_count exceeds the M1 execution cap")
    if bootstrap_replicates > MAX_BOOTSTRAP_REPLICATES:
        raise ArtifactValidationError("bootstrap replicates exceed the M1 execution cap")
    if sequence_count * bootstrap_replicates > MAX_BOOTSTRAP_CELLS:
        raise ArtifactValidationError("bootstrap matrix exceeds the M1 execution cap")
    if sequence_rows > MAX_SEQUENCE_ROWS:
        raise ArtifactValidationError("sequence records exceed the M1 execution cap")


def run_analytic_experiment(
    manifest_path: Path,
    *,
    output_dir: Path,
) -> LoadedArtifact:
    """Execute, validate, and atomically publish one analytic artifact."""

    snapshot = discover_clean_source(manifest_path)
    verify_locked_execution(snapshot)
    manifest_union = load_manifest(manifest_path)
    if not isinstance(manifest_union, AnalyticCrossoverManifest):
        raise ValueError("M1 run accepts only an analytic-crossover manifest")
    manifest = manifest_union
    preflight_analytic_manifest(manifest)

    manifest_digest = sha256_digest(manifest)
    run_id = derive_run_id(
        manifest_sha256=manifest_digest,
        git_revision=snapshot.git_revision,
        lockfile_sha256=snapshot.lockfile_sha256,
        package_version=snapshot.package_version,
    )
    command = logical_reproduction_command(
        snapshot=snapshot,
        experiment=manifest.experiment,
        manifest_sha256=manifest_digest,
    )
    started_at = datetime.now(UTC)
    environment = collect_runtime_environment()
    metrics = generate_analytic_sequence_metrics(manifest, run_id=run_id)
    evaluated = evaluate_analytic_records(manifest, run_id=run_id, metrics=metrics)
    analytic_validation = build_analytic_validation(
        manifest,
        run_id=run_id,
        metrics=evaluated.metrics,
    )
    if not analytic_validation.all_monte_carlo_checks_passed:
        raise ArtifactValidationError(
            "analytic Monte Carlo validation failed the preregistered 6-SE acceptance gate"
        )
    final_snapshot = discover_clean_source(manifest_path)
    if final_snapshot != snapshot:
        raise ValueError("clean source provenance changed during analytic evaluation")
    verify_locked_execution(final_snapshot)
    ended_at = datetime.now(UTC)
    run = RunRecordV1Alpha1(
        schema="ffb.run/v1alpha1",
        run_id=run_id,
        manifest_sha256=manifest_digest,
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
    return write_artifact(
        ArtifactWriteRequest(
            manifest=manifest,
            run=run,
            metrics=evaluated.metrics,
            aggregates=evaluated.aggregates,
            crossovers=evaluated.crossovers,
            analytic_validation=analytic_validation,
        ),
        output_dir,
        source_root=snapshot.source_root,
        git_metadata_dirs=(snapshot.git_dir, snapshot.git_common_dir),
    )
