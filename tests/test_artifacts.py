from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import fusion_fault_bench.artifacts as artifact_module
from fusion_fault_bench.artifacts import (
    ArtifactValidationError,
    ArtifactWriteRequest,
    canonical_json_bytes,
    canonical_ndjson_bytes,
    compute_artifact_digest,
    compute_run_record_digest,
    derive_run_id,
    discover_git_metadata_dirs,
    load_artifact,
    write_artifact,
)
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.artifact_v1alpha1 import (
    ARTIFACT_PATHS,
    INDEXED_PAYLOAD_PATHS,
    MANIFEST_FILE,
    PAYLOAD_INDEX_FILE,
    RUN_FILE,
    SEQUENCE_METRICS_FILE,
    SUCCESS_FILE,
    AnalyticCrossoverReferenceV1Alpha1,
    AnalyticPopulationPointV1Alpha1,
    AnalyticValidationV1Alpha1,
    PayloadFileEntryV1Alpha1,
    PayloadIndexV1Alpha1,
    SuccessMarkerV1Alpha1,
)
from fusion_fault_bench.contracts.bundle_v1alpha1 import (
    expected_conditions,
    expected_sequence_ids,
)
from fusion_fault_bench.contracts.io import validate_manifest_mapping
from fusion_fault_bench.contracts.manifest_v1alpha1 import AnalyticCrossoverManifest
from fusion_fault_bench.contracts.result_v1alpha1 import (
    AggregateMetricRecordV1Alpha1,
    CrossoverRecordV1Alpha1,
    LocalizationMetricRecord,
    MetricRecordV1Alpha1,
    RunRecordV1Alpha1,
    RuntimeEnvironment,
    SeverityCoordinate,
)
from fusion_fault_bench.validation import build_analytic_validation

GIT_REVISION = "b" * 40
LOCK_DIGEST = "c" * 64
PLACEHOLDER_DIGEST = "0" * 64


def _small_manifest(data: dict[str, Any]) -> AnalyticCrossoverManifest:
    value = copy.deepcopy(data)
    value["source"]["sequence_count"] = 2
    value["fault_sweep"]["magnitude_values_m"] = [0.0, 1.0]
    manifest = validate_manifest_mapping(value)
    assert isinstance(manifest, AnalyticCrossoverManifest)
    return manifest


def _severity(condition: Any) -> SeverityCoordinate:
    return SeverityCoordinate(
        index=condition.severity_index,
        magnitude=condition.magnitude,
        direction=condition.direction,
        unit=condition.unit,
    )


def _run(manifest: AnalyticCrossoverManifest) -> RunRecordV1Alpha1:
    manifest_digest = sha256_digest(manifest)
    run_id = derive_run_id(
        manifest_sha256=manifest_digest,
        git_revision=GIT_REVISION,
        lockfile_sha256=LOCK_DIGEST,
        package_version="0.1.0",
    )
    started = datetime(2026, 7, 28, 12, tzinfo=UTC)
    return RunRecordV1Alpha1(
        schema="ffb.run/v1alpha1",
        run_id=run_id,
        manifest_sha256=manifest_digest,
        package_version="0.1.0",
        git_revision=GIT_REVISION,
        source_dirty=False,
        lockfile_sha256=LOCK_DIGEST,
        command=(
            "ffb",
            "run",
            "examples/manifests/analytic-bias-v1alpha1.json",
            "--output-dir",
            f"reports/generated/{manifest.experiment}-{manifest_digest[:12]}",
        ),
        environment=RuntimeEnvironment(
            python_version="3.12.13",
            os_name="macOS",
            os_release="15.5",
            machine="arm64",
            cpu_model="Test CPU",
            logical_cpu_count=8,
            memory_bytes=16_000_000_000,
        ),
        started_at=started,
        ended_at=started + timedelta(seconds=1),
        status="succeeded",
        artifact_sha256=PLACEHOLDER_DIGEST,
    )


def _records(
    manifest: AnalyticCrossoverManifest,
    run: RunRecordV1Alpha1,
) -> tuple[
    list[MetricRecordV1Alpha1],
    list[AggregateMetricRecordV1Alpha1],
    list[CrossoverRecordV1Alpha1],
]:
    metrics: list[MetricRecordV1Alpha1] = []
    aggregates: list[AggregateMetricRecordV1Alpha1] = []
    for condition in expected_conditions(manifest):
        severity = _severity(condition)
        for method in manifest.methods:
            for sequence_id in expected_sequence_ids(manifest):
                metrics.append(
                    LocalizationMetricRecord(
                        schema="ffb.sequence-metric/v1alpha1",
                        record_level="sequence",
                        run_id=run.run_id,
                        manifest_sha256=run.manifest_sha256,
                        sequence_id=sequence_id,
                        fault_family=condition.fault_family,
                        fault_axis=condition.fault_axis,
                        severity=severity,
                        method_id=method,
                        eligible_object_frame_count=1,
                        valid_object_frame_count=1,
                        metric_name="matched-center-mse",
                        status="ok",
                        value=0.25,
                        unit="m^2",
                    )
                )
            aggregates.append(
                AggregateMetricRecordV1Alpha1(
                    schema="ffb.aggregate-metric/v1alpha1",
                    record_level="aggregate",
                    run_id=run.run_id,
                    manifest_sha256=run.manifest_sha256,
                    fault_family=condition.fault_family,
                    fault_axis=condition.fault_axis,
                    severity=severity,
                    method_id=method,
                    metric_name="matched-center-mse",
                    status="ok",
                    estimate=0.25,
                    interval_lower=0.25,
                    interval_upper=0.25,
                    unit="m^2",
                    sequence_count=2,
                    contributing_sequence_count=2,
                    bootstrap_replicates=2000,
                    defined_bootstrap_replicates=2000,
                    confidence_level=0.95,
                    interval_method="paired-sequence-percentile-pointwise",
                    aggregation="object-frame-mean-then-sequence-mean",
                )
            )
        aggregates.append(
            AggregateMetricRecordV1Alpha1(
                schema="ffb.aggregate-metric/v1alpha1",
                record_level="aggregate",
                run_id=run.run_id,
                manifest_sha256=run.manifest_sha256,
                fault_family=condition.fault_family,
                fault_axis=condition.fault_axis,
                severity=severity,
                method_id="fixed-fusion",
                metric_name="fused-minus-healthy",
                status="ok",
                estimate=0.0,
                interval_lower=0.0,
                interval_upper=0.0,
                unit="m^2",
                sequence_count=2,
                contributing_sequence_count=2,
                bootstrap_replicates=2000,
                defined_bootstrap_replicates=2000,
                confidence_level=0.95,
                interval_method="paired-sequence-percentile-pointwise",
                aggregation="object-frame-mean-then-sequence-mean",
            )
        )
    crossovers = [
        CrossoverRecordV1Alpha1(
            schema="ffb.crossover/v1alpha1",
            run_id=run.run_id,
            manifest_sha256=run.manifest_sha256,
            fault_family="additive-position-bias",
            fault_axis="x",
            direction=direction,
            severity_unit="m",
            status="observed",
            point_curve_crossed=True,
            point_estimate=0.0,
            interval_lower=0.0,
            interval_upper=0.0,
            tested_maximum=1.0,
            censoring="none",
            bootstrap_crossing_fraction=1.0,
            sequence_count=2,
            bootstrap_replicates=2000,
            confidence_level=0.95,
            interval_method="right-censored-percentile",
        )
        for direction in ("negative", "positive")
    ]
    return metrics, aggregates, crossovers


def _analytic(
    manifest: AnalyticCrossoverManifest,
    run: RunRecordV1Alpha1,
    metrics: list[MetricRecordV1Alpha1],
) -> AnalyticValidationV1Alpha1:
    return build_analytic_validation(
        manifest,
        run_id=run.run_id,
        metrics=tuple(metrics),
    )


@pytest.fixture
def artifact_request(manifest_data: dict[str, Any]) -> ArtifactWriteRequest:
    manifest = _small_manifest(manifest_data)
    run = _run(manifest)
    metrics, aggregates, crossovers = _records(manifest, run)
    return ArtifactWriteRequest(
        manifest=manifest,
        run=run,
        metrics=list(reversed(metrics)),
        aggregates=list(reversed(aggregates)),
        crossovers=list(reversed(crossovers)),
        analytic_validation=_analytic(manifest, run, metrics),
    )


@pytest.fixture
def written_artifact(
    tmp_path: Path,
    artifact_request: ArtifactWriteRequest,
):
    return write_artifact(
        artifact_request,
        tmp_path / "artifact",
        git_metadata_dirs=(),
    )


def _copy_artifact(source: Path, tmp_path: Path, name: str) -> Path:
    destination = tmp_path / name
    shutil.copytree(source, destination)
    return destination


def test_hash_framing_and_positive_zero_are_exact() -> None:
    index_bytes = b'{"schema":"ffb.payload-index/v1alpha1"}\n'
    run_bytes = b'{"schema":"ffb.run/v1alpha1"}\n'
    expected_artifact = hashlib.sha256(
        b"fusion-fault-bench/artifact/v1\x00" + len(index_bytes).to_bytes(8, "big") + index_bytes
    ).hexdigest()
    expected_run = hashlib.sha256(
        b"fusion-fault-bench/run-record/v1\x00" + len(run_bytes).to_bytes(8, "big") + run_bytes
    ).hexdigest()
    fields = ("a" * 64, "b" * 40, "c" * 64, "0.1.0")
    preimage = b"fusion-fault-bench/run-id/v1\x00" + b"".join(
        len(value).to_bytes(4, "big") + value
        for value in (
            fields[0].encode(),
            fields[1].encode(),
            fields[2].encode(),
            fields[3].encode(),
            b"ffb.scientific-payload/v1",
        )
    )

    assert compute_artifact_digest(index_bytes) == expected_artifact
    assert compute_run_record_digest(run_bytes) == expected_run
    assert (
        derive_run_id(
            manifest_sha256=fields[0],
            git_revision=fields[1],
            lockfile_sha256=fields[2],
            package_version=fields[3],
        )
        == f"run:{hashlib.sha256(preimage).hexdigest()}"
    )
    assert canonical_json_bytes({"zero": -0.0}) == b'{"zero":0.0}\n'


def test_payload_index_requires_the_exact_allowlist_order() -> None:
    entries = tuple(
        PayloadFileEntryV1Alpha1(path=path, byte_length=1, sha256="a" * 64)
        for path in INDEXED_PAYLOAD_PATHS
    )
    value = {
        "schema": "ffb.payload-index/v1alpha1",
        "artifact_contract": "ffb.scientific-payload/v1",
        "run_id": "run:test",
        "manifest_sha256": "b" * 64,
        "files": tuple(reversed(entries)),
    }

    with pytest.raises(ValidationError, match="fixed five-member order"):
        PayloadIndexV1Alpha1.model_validate(value)


def test_analytic_contract_rejects_contradictory_flags(
    artifact_request: ArtifactWriteRequest,
) -> None:
    value = artifact_request.analytic_validation.model_dump(mode="python", by_alias=True)
    value["all_monte_carlo_checks_passed"] = False

    with pytest.raises(ValidationError, match="conjunction"):
        AnalyticValidationV1Alpha1.model_validate(value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "severity",
            SeverityCoordinate(index=0, magnitude=0.0, direction="identity", unit="rad"),
            "severity unit",
        ),
        ("expected_actual_variance_xy_m2", (-1.0, 0.25), "actual variances"),
        ("expected_reported_variance_xy_m2", (0.0, 0.25), "reported variances"),
        ("absolute_standardized_error", 1.0, "standardized error disagrees"),
    ],
)
def test_population_point_contract_rejects_contradictions(
    artifact_request: ArtifactWriteRequest,
    field: str,
    value: object,
    message: str,
) -> None:
    point = artifact_request.analytic_validation.population_points[0]
    raw = point.model_dump(mode="python")
    raw[field] = value

    with pytest.raises(ValidationError, match=message):
        AnalyticPopulationPointV1Alpha1.model_validate(raw)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"tested_maximum": 0.0}, "exceed the identity"),
        ({"grid_point_estimate": None}, "uncensored root"),
        ({"grid_point_estimate": 2.0}, "tested interval"),
        (
            {"grid_status": "not-crossed", "grid_point_estimate": 0.5},
            "right-censored",
        ),
        ({"continuous_point_estimate": None}, "requires a root"),
        ({"continuous_point_estimate": -0.5}, "precede identity"),
        (
            {"continuous_status": "no-finite-root"},
            "no-finite-root status requires a null root",
        ),
    ],
)
def test_crossover_reference_contract_rejects_contradictions(
    updates: dict[str, object],
    message: str,
) -> None:
    reference = AnalyticCrossoverReferenceV1Alpha1(
        direction="negative",
        severity_unit="m",
        tested_maximum=1.0,
        grid_status="crossed",
        grid_point_estimate=0.5,
        grid_censoring="none",
        continuous_status="finite",
        continuous_point_estimate=0.5,
    )
    raw = reference.model_dump(mode="python")
    raw.update(updates)

    with pytest.raises(ValidationError, match=message):
        AnalyticCrossoverReferenceV1Alpha1.model_validate(raw)


def test_analytic_contract_rejects_incomplete_and_misordered_points(
    artifact_request: ArtifactWriteRequest,
) -> None:
    analytic = artifact_request.analytic_validation
    raw = analytic.model_dump(mode="python", by_alias=True)
    raw["population_points"] = analytic.population_points[:4]
    with pytest.raises(ValidationError, match="complete method triples"):
        AnalyticValidationV1Alpha1.model_validate(raw)

    raw["population_points"] = (
        analytic.population_points[1],
        analytic.population_points[0],
        *analytic.population_points[2:],
    )
    with pytest.raises(ValidationError, match="fixed order"):
        AnalyticValidationV1Alpha1.model_validate(raw)

    raw["population_points"] = (
        analytic.population_points[0],
        analytic.population_points[4],
        *analytic.population_points[2:],
    )
    with pytest.raises(ValidationError, match="share one severity"):
        AnalyticValidationV1Alpha1.model_validate(raw)


def test_analytic_contract_rejects_direction_and_point_pass_mismatch(
    artifact_request: ArtifactWriteRequest,
) -> None:
    analytic = artifact_request.analytic_validation
    raw = analytic.model_dump(mode="python", by_alias=True)
    raw["crossover_references"] = tuple(reversed(analytic.crossover_references))
    with pytest.raises(ValidationError, match="direction order"):
        AnalyticValidationV1Alpha1.model_validate(raw)

    changed_point = analytic.population_points[0].model_copy(update={"monte_carlo_passed": False})
    raw["crossover_references"] = analytic.crossover_references
    raw["population_points"] = (changed_point, *analytic.population_points[1:])
    with pytest.raises(ValidationError, match="pass flag"):
        AnalyticValidationV1Alpha1.model_validate(raw)


def test_canonical_ndjson_rejects_empty_and_bounded_output(
    artifact_request: ArtifactWriteRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ArtifactValidationError, match="at least one"):
        canonical_ndjson_bytes(())

    monkeypatch.setattr(artifact_module, "MAX_LINE_BYTES", 10)
    with pytest.raises(ArtifactValidationError, match="line cap"):
        canonical_ndjson_bytes((artifact_request.metrics[0],))

    monkeypatch.setattr(artifact_module, "MAX_LINE_BYTES", 1024 * 1024)
    monkeypatch.setattr(artifact_module, "MAX_SCIENTIFIC_MEMBER_BYTES", 10)
    with pytest.raises(ArtifactValidationError, match="member exceeds"):
        canonical_ndjson_bytes((artifact_request.metrics[0],))


def test_writer_rejects_wrong_run_identity_and_forged_analytic_evidence(
    tmp_path: Path,
    artifact_request: ArtifactWriteRequest,
) -> None:
    wrong_run = RunRecordV1Alpha1.model_validate(
        {
            **artifact_request.run.model_dump(mode="python", by_alias=True),
            "run_id": "run:wrong",
        }
    )
    with pytest.raises(ArtifactValidationError, match="run_id disagrees"):
        write_artifact(
            ArtifactWriteRequest(
                manifest=artifact_request.manifest,
                run=wrong_run,
                metrics=artifact_request.metrics,
                aggregates=artifact_request.aggregates,
                crossovers=artifact_request.crossovers,
                analytic_validation=artifact_request.analytic_validation,
            ),
            tmp_path / "wrong-run",
            git_metadata_dirs=(),
        )

    analytic = artifact_request.analytic_validation
    reference = analytic.crossover_references[0].model_copy(
        update={"continuous_point_estimate": 3.5}
    )
    forged = AnalyticValidationV1Alpha1.model_validate(
        {
            **analytic.model_dump(mode="python", by_alias=True),
            "crossover_references": (reference, *analytic.crossover_references[1:]),
        }
    )
    with pytest.raises(ArtifactValidationError, match="independent population"):
        write_artifact(
            ArtifactWriteRequest(
                manifest=artifact_request.manifest,
                run=artifact_request.run,
                metrics=artifact_request.metrics,
                aggregates=artifact_request.aggregates,
                crossovers=artifact_request.crossovers,
                analytic_validation=forged,
            ),
            tmp_path / "forged",
            git_metadata_dirs=(),
        )


@pytest.mark.parametrize(
    "manifest_path",
    [
        "manifest.json",
        "../manifest.json",
        r"examples\manifest.json",
        "./examples/manifest.json",
        "examples//manifest.json",
        "examples/manifest.txt",
        "~/manifest.json",
        "examples/-manifest.json",
        "examples/manifest name.json",
        "examples/manifest;rm.json",
    ],
)
def test_writer_rejects_unsafe_logical_manifest_path(
    tmp_path: Path,
    artifact_request: ArtifactWriteRequest,
    manifest_path: str,
) -> None:
    command = list(artifact_request.run.command)
    command[2] = manifest_path
    wrong_run = RunRecordV1Alpha1.model_validate(
        {
            **artifact_request.run.model_dump(mode="python", by_alias=True),
            "command": tuple(command),
        }
    )
    with pytest.raises(ArtifactValidationError, match="safe tracked"):
        write_artifact(
            ArtifactWriteRequest(
                manifest=artifact_request.manifest,
                run=wrong_run,
                metrics=artifact_request.metrics,
                aggregates=artifact_request.aggregates,
                crossovers=artifact_request.crossovers,
                analytic_validation=artifact_request.analytic_validation,
            ),
            tmp_path / "wrong-command",
            git_metadata_dirs=(),
        )


def test_writer_rejects_wrong_logical_command_structure(
    tmp_path: Path,
    artifact_request: ArtifactWriteRequest,
) -> None:
    wrong_run = RunRecordV1Alpha1.model_validate(
        {
            **artifact_request.run.model_dump(mode="python", by_alias=True),
            "command": ("ffb",),
        }
    )
    with pytest.raises(ArtifactValidationError, match="frozen logical"):
        write_artifact(
            ArtifactWriteRequest(
                manifest=artifact_request.manifest,
                run=wrong_run,
                metrics=artifact_request.metrics,
                aggregates=artifact_request.aggregates,
                crossovers=artifact_request.crossovers,
                analytic_validation=artifact_request.analytic_validation,
            ),
            tmp_path / "wrong-command",
            git_metadata_dirs=(),
        )


def test_writer_finalizes_digest_sorts_records_and_round_trips(
    written_artifact,
) -> None:
    assert set(path.name for path in written_artifact.path.iterdir()) == set(ARTIFACT_PATHS)
    assert written_artifact.run.artifact_sha256 == written_artifact.artifact_sha256
    assert written_artifact.artifact_sha256 != PLACEHOLDER_DIGEST
    marker_bytes = (written_artifact.path / SUCCESS_FILE).read_bytes()
    marker = SuccessMarkerV1Alpha1.model_validate_json(marker_bytes)
    assert marker_bytes == canonical_json_bytes(marker)
    assert marker.artifact_sha256 == written_artifact.artifact_sha256
    assert marker.run_sha256 == written_artifact.run_sha256
    assert written_artifact.run_sha256 == compute_run_record_digest(
        (written_artifact.path / RUN_FILE).read_bytes()
    )
    sequence_ids = [record.sequence_id for record in written_artifact.metrics]
    assert sequence_ids == sorted(sequence_ids)
    assert load_artifact(written_artifact.path) == written_artifact


def test_two_writes_have_identical_scientific_payload(
    tmp_path: Path,
    artifact_request: ArtifactWriteRequest,
) -> None:
    first = write_artifact(artifact_request, tmp_path / "first", git_metadata_dirs=())
    second = write_artifact(artifact_request, tmp_path / "second", git_metadata_dirs=())

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.run_sha256 == second.run_sha256
    for path in (*INDEXED_PAYLOAD_PATHS, PAYLOAD_INDEX_FILE):
        assert (first.path / path).read_bytes() == (second.path / path).read_bytes()


def test_timestamp_only_rerun_preserves_payload_digest_but_changes_run_integrity(
    tmp_path: Path,
    artifact_request: ArtifactWriteRequest,
) -> None:
    first = write_artifact(artifact_request, tmp_path / "first", git_metadata_dirs=())
    later_run_data = artifact_request.run.model_dump(mode="python", by_alias=True)
    later_run_data["started_at"] = artifact_request.run.started_at + timedelta(minutes=1)
    assert artifact_request.run.ended_at is not None
    later_run_data["ended_at"] = artifact_request.run.ended_at + timedelta(minutes=1)
    later_run = RunRecordV1Alpha1.model_validate(later_run_data)
    second = write_artifact(
        ArtifactWriteRequest(
            manifest=artifact_request.manifest,
            run=later_run,
            metrics=artifact_request.metrics,
            aggregates=artifact_request.aggregates,
            crossovers=artifact_request.crossovers,
            analytic_validation=artifact_request.analytic_validation,
        ),
        tmp_path / "second",
        git_metadata_dirs=(),
    )

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.run_sha256 != second.run_sha256
    for path in (*INDEXED_PAYLOAD_PATHS, PAYLOAD_INDEX_FILE):
        assert (first.path / path).read_bytes() == (second.path / path).read_bytes()
    assert (first.path / RUN_FILE).read_bytes() != (second.path / RUN_FILE).read_bytes()
    assert (first.path / SUCCESS_FILE).read_bytes() != (second.path / SUCCESS_FILE).read_bytes()


def test_writer_refuses_existing_and_dangling_destinations(
    tmp_path: Path,
    artifact_request: ArtifactWriteRequest,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        write_artifact(artifact_request, existing, git_metadata_dirs=())

    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        write_artifact(artifact_request, dangling, git_metadata_dirs=())


def test_atomic_publish_never_replaces_an_existing_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    marker = destination / "keep"
    marker.write_text("keep", encoding="utf-8")

    parent_fd = os.open(
        tmp_path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        with pytest.raises(FileExistsError):
            artifact_module._atomic_rename_no_replace_at(
                parent_fd,
                source.name,
                parent_fd,
                destination.name,
            )
    finally:
        os.close(parent_fd)

    assert source.is_dir()
    assert marker.read_text(encoding="utf-8") == "keep"


def test_writer_rejects_symlink_parent_and_git_metadata(
    tmp_path: Path,
    artifact_request: ArtifactWriteRequest,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ArtifactValidationError, match="not a real directory"):
        write_artifact(
            artifact_request,
            linked_parent / "artifact",
            git_metadata_dirs=(),
        )

    git_dir = tmp_path / "git-metadata"
    git_dir.mkdir()
    with pytest.raises(ArtifactValidationError, match="inside Git metadata"):
        write_artifact(
            artifact_request,
            git_dir / "artifact",
            git_metadata_dirs=(git_dir,),
        )


def test_writer_cleans_only_its_staging_directory_on_failure(
    tmp_path: Path,
    artifact_request: ArtifactWriteRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    keep = parent / "keep"
    parent.mkdir()
    keep.mkdir()

    def fail_write(_directory_fd: int, _name: str, _value: bytes) -> None:
        raise OSError("injected write failure")

    monkeypatch.setattr(artifact_module, "_write_exclusive_at", fail_write)
    with pytest.raises(OSError, match="injected"):
        write_artifact(
            artifact_request,
            parent / "artifact",
            git_metadata_dirs=(),
        )

    assert keep.is_dir()
    assert list(parent.iterdir()) == [keep]


def test_writer_parent_swap_cannot_redirect_publication_into_git_metadata(
    tmp_path: Path,
    artifact_request: ArtifactWriteRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "checked-parent"
    moved_parent = tmp_path / "pinned-parent"
    git_dir = tmp_path / "git-metadata"
    parent.mkdir()
    git_dir.mkdir()
    original_publish = artifact_module._atomic_rename_no_replace_at

    def swap_parent_then_publish(
        source_dir_fd: int,
        source_name: str,
        destination_dir_fd: int,
        destination_name: str,
    ) -> None:
        parent.rename(moved_parent)
        parent.symlink_to(git_dir, target_is_directory=True)
        original_publish(
            source_dir_fd,
            source_name,
            destination_dir_fd,
            destination_name,
        )

    monkeypatch.setattr(
        artifact_module,
        "_atomic_rename_no_replace_at",
        swap_parent_then_publish,
    )

    with pytest.raises(
        ArtifactValidationError,
        match="destination parent changed during artifact publication",
    ):
        write_artifact(
            artifact_request,
            parent / "artifact",
            git_metadata_dirs=(git_dir,),
        )

    assert list(git_dir.iterdir()) == []
    assert (moved_parent / "artifact").is_dir()
    assert {path.name for path in (moved_parent / "artifact").iterdir()} == set(ARTIFACT_PATHS)


def test_fd_relative_writer_guards_reject_unsafe_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ArtifactValidationError, match="must be absolute"):
        artifact_module._open_or_create_real_directory(Path("relative"))

    parent_fd = artifact_module._open_or_create_real_directory(tmp_path)
    try:
        with pytest.raises(ArtifactValidationError, match="changed during"):
            artifact_module._assert_directory_fd_matches_path(
                parent_fd,
                tmp_path / "missing",
                label="test parent",
            )
        with pytest.raises(ArtifactValidationError, match="inside Git metadata"):
            artifact_module._reject_directory_fd_in_git_metadata(
                parent_fd,
                (tmp_path,),
            )
        for name in ("", ".", "..", "nested/member", r"nested\member"):
            with pytest.raises(ArtifactValidationError, match="single path segment"):
                artifact_module._validate_relative_member_name(name)

        (tmp_path / "directory-member").mkdir()
        with pytest.raises(ArtifactValidationError, match="not a regular file"):
            artifact_module._read_at(parent_fd, "directory-member", byte_cap=1)
        (tmp_path / "oversized-member").write_bytes(b"xx")
        with pytest.raises(ArtifactValidationError, match="exceeds its byte cap"):
            artifact_module._read_at(parent_fd, "oversized-member", byte_cap=1)

        monkeypatch.setattr(artifact_module.os, "write", lambda _fd, _value: 0)
        with pytest.raises(OSError, match="short write"):
            artifact_module._write_exclusive_at(parent_fd, "short-write", b"x")
    finally:
        os.close(parent_fd)


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    [
        (
            "bom",
            lambda root: (root / MANIFEST_FILE).write_bytes(
                b"\xef\xbb\xbf" + (root / MANIFEST_FILE).read_bytes()
            ),
            "BOM",
        ),
        (
            "negative-zero",
            lambda root: (root / MANIFEST_FILE).write_bytes(
                (root / MANIFEST_FILE).read_bytes().replace(b"0.0", b"-0.0", 1)
            ),
            "negative zero",
        ),
        (
            "cr",
            lambda root: (root / RUN_FILE).write_bytes(
                (root / RUN_FILE).read_bytes().replace(b"\n", b"\r\n")
            ),
            "CR byte",
        ),
        (
            "extra-lf",
            lambda root: (root / RUN_FILE).write_bytes((root / RUN_FILE).read_bytes() + b"\n"),
            "extra LF",
        ),
        (
            "blank-ndjson",
            lambda root: (root / SEQUENCE_METRICS_FILE).write_bytes(
                (root / SEQUENCE_METRICS_FILE).read_bytes() + b"\n"
            ),
            "blank",
        ),
        (
            "noncanonical",
            lambda root: (root / RUN_FILE).write_bytes(
                (root / RUN_FILE).read_bytes().replace(b'{"', b'{ "', 1)
            ),
            "not canonical",
        ),
    ],
)
def test_loader_rejects_noncanonical_encodings(
    written_artifact,
    tmp_path: Path,
    name: str,
    mutate,
    message: str,
) -> None:
    copied = _copy_artifact(written_artifact.path, tmp_path, name)
    mutate(copied)

    with pytest.raises(ArtifactValidationError, match=message):
        load_artifact(copied)


def test_loader_rejects_duplicate_keys_and_nonstandard_numbers(
    written_artifact,
    tmp_path: Path,
) -> None:
    duplicate = _copy_artifact(written_artifact.path, tmp_path, "duplicate")
    run_path = duplicate / RUN_FILE
    run_path.write_bytes(run_path.read_bytes().replace(b"{", b'{"run_id":"duplicate",', 1))
    with pytest.raises(ArtifactValidationError, match="duplicate JSON object key"):
        load_artifact(duplicate)

    nan = _copy_artifact(written_artifact.path, tmp_path, "nan")
    manifest_path = nan / MANIFEST_FILE
    manifest_path.write_bytes(manifest_path.read_bytes().replace(b"0.0", b"NaN", 1))
    with pytest.raises(ArtifactValidationError, match="non-standard JSON number"):
        load_artifact(nan)


@pytest.mark.parametrize(
    ("name", "replacement", "message"),
    [
        ("missing-lf", b"{}", "missing its terminal LF"),
        ("non-utf8", b'{"value":"\xff"}\n', "not valid UTF-8"),
        ("invalid-json", b"{\n", "invalid JSON"),
        ("top-level-list", b"[]\n", "top-level JSON object"),
        ("invalid-schema", b'{"schema":"ffb.run/v1alpha1"}\n', "violates its schema"),
    ],
)
def test_loader_rejects_invalid_single_json_members(
    written_artifact,
    tmp_path: Path,
    name: str,
    replacement: bytes,
    message: str,
) -> None:
    copied = _copy_artifact(written_artifact.path, tmp_path, name)
    (copied / RUN_FILE).write_bytes(replacement)

    with pytest.raises(ArtifactValidationError, match=message):
        load_artifact(copied)


def test_loader_rejects_empty_ndjson_and_nonregular_member(
    written_artifact,
    tmp_path: Path,
) -> None:
    empty = _copy_artifact(written_artifact.path, tmp_path, "empty-ndjson")
    (empty / SEQUENCE_METRICS_FILE).write_bytes(b"")
    with pytest.raises(ArtifactValidationError, match="at least one record"):
        load_artifact(empty)

    directory_member = _copy_artifact(written_artifact.path, tmp_path, "directory-member")
    marker = directory_member / SUCCESS_FILE
    marker.unlink()
    marker.mkdir()
    with pytest.raises(ArtifactValidationError, match="not a regular file"):
        load_artifact(directory_member)


def test_loader_rejects_mutation_wrong_marker_and_dirty_run(
    written_artifact,
    tmp_path: Path,
) -> None:
    mutated = _copy_artifact(written_artifact.path, tmp_path, "mutated")
    metric_path = mutated / SEQUENCE_METRICS_FILE
    metric_path.write_bytes(metric_path.read_bytes().replace(b"0.25", b"0.26", 1))
    with pytest.raises(ArtifactValidationError, match="disagrees"):
        load_artifact(mutated)

    marker = _copy_artifact(written_artifact.path, tmp_path, "marker")
    (marker / SUCCESS_FILE).write_bytes(b"f" * 64 + b"\n")
    with pytest.raises(ArtifactValidationError, match="_SUCCESS"):
        load_artifact(marker)

    wrong_run_marker = _copy_artifact(
        written_artifact.path,
        tmp_path,
        "wrong-run-marker",
    )
    marker_path = wrong_run_marker / SUCCESS_FILE
    marker_model = SuccessMarkerV1Alpha1.model_validate_json(marker_path.read_bytes())
    marker_path.write_bytes(
        canonical_json_bytes(marker_model.model_copy(update={"run_sha256": "f" * 64}))
    )
    with pytest.raises(ArtifactValidationError, match="_SUCCESS run digest"):
        load_artifact(wrong_run_marker)

    changed_provenance = _copy_artifact(
        written_artifact.path,
        tmp_path,
        "changed-provenance",
    )
    run_path = changed_provenance / RUN_FILE
    run_value = json.loads(run_path.read_bytes())
    run_value["environment"]["cpu_model"] = "Mutated CPU"
    run_path.write_bytes(canonical_json_bytes(run_value))
    with pytest.raises(ArtifactValidationError, match="_SUCCESS run digest"):
        load_artifact(changed_provenance)

    dirty = _copy_artifact(written_artifact.path, tmp_path, "dirty")
    run_path = dirty / RUN_FILE
    run_path.write_bytes(
        run_path.read_bytes().replace(b'"source_dirty":false', b'"source_dirty":true')
    )
    marker_path = dirty / SUCCESS_FILE
    marker_model = SuccessMarkerV1Alpha1.model_validate_json(marker_path.read_bytes())
    marker_path.write_bytes(
        canonical_json_bytes(
            marker_model.model_copy(
                update={"run_sha256": compute_run_record_digest(run_path.read_bytes())}
            )
        )
    )
    with pytest.raises(ArtifactValidationError, match="dirty source"):
        load_artifact(dirty)


def test_loader_rejects_wrong_record_order(
    written_artifact,
    tmp_path: Path,
) -> None:
    copied = _copy_artifact(written_artifact.path, tmp_path, "wrong-order")
    metric_path = copied / SEQUENCE_METRICS_FILE
    lines = metric_path.read_bytes().splitlines(keepends=True)
    lines[0], lines[1] = lines[1], lines[0]
    metric_path.write_bytes(b"".join(lines))

    with pytest.raises(ArtifactValidationError, match="wrong canonical order"):
        load_artifact(copied)


def test_loader_rejects_file_allowlist_and_symlink_members(
    written_artifact,
    tmp_path: Path,
) -> None:
    extra = _copy_artifact(written_artifact.path, tmp_path, "extra")
    (extra / "extra.json").write_bytes(b"{}\n")
    with pytest.raises(ArtifactValidationError, match="allowlist mismatch"):
        load_artifact(extra)

    missing = _copy_artifact(written_artifact.path, tmp_path, "missing")
    (missing / SUCCESS_FILE).unlink()
    with pytest.raises(ArtifactValidationError, match="allowlist mismatch"):
        load_artifact(missing)

    linked = _copy_artifact(written_artifact.path, tmp_path, "linked")
    run_path = linked / RUN_FILE
    run_path.unlink()
    run_path.symlink_to(written_artifact.path / RUN_FILE)
    with pytest.raises(ArtifactValidationError, match="symlink"):
        load_artifact(linked)

    root_link = tmp_path / "root-link"
    root_link.symlink_to(written_artifact.path, target_is_directory=True)
    with pytest.raises(ArtifactValidationError, match="artifact root"):
        load_artifact(root_link)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    _copy_artifact(written_artifact.path, real_parent, "artifact")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ArtifactValidationError, match="symlink component"):
        load_artifact(linked_parent / "artifact")


def test_loader_binds_parsing_to_the_scanned_inode(
    written_artifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied = _copy_artifact(written_artifact.path, tmp_path, "inode-swap")
    manifest_bytes = (copied / MANIFEST_FILE).read_bytes()
    replacement = tmp_path / "replacement-manifest.json"
    original_read = artifact_module._read_small_file
    swapped = False

    def swap_before_read(
        path: Path,
        *,
        label: str,
        expected_stat=None,
    ) -> bytes:
        nonlocal swapped
        if path.name == MANIFEST_FILE and not swapped:
            replacement.write_bytes(manifest_bytes)
            replacement.replace(path)
            swapped = True
        return original_read(path, label=label, expected_stat=expected_stat)

    monkeypatch.setattr(artifact_module, "_read_small_file", swap_before_read)
    with pytest.raises(ArtifactValidationError, match="changed after the artifact tree scan"):
        load_artifact(copied)


def test_loader_enforces_size_and_writer_execution_caps(
    written_artifact,
    artifact_request: ArtifactWriteRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(artifact_module, "MAX_LINE_BYTES", 10)
    with pytest.raises(ArtifactValidationError, match="line cap"):
        load_artifact(written_artifact.path)
    monkeypatch.setattr(artifact_module, "MAX_LINE_BYTES", 1024 * 1024)
    monkeypatch.setattr(artifact_module, "MAX_SEQUENCE_ROWS", 1)
    with pytest.raises(ArtifactValidationError, match="sequence records"):
        write_artifact(
            artifact_request,
            tmp_path / "too-many",
            git_metadata_dirs=(),
        )

    monkeypatch.setattr(artifact_module, "MAX_SEQUENCE_ROWS", 2_000_000)
    monkeypatch.setattr(artifact_module, "MAX_SCIENTIFIC_MEMBER_BYTES", 10)
    with pytest.raises(ArtifactValidationError, match="member cap"):
        load_artifact(written_artifact.path)

    monkeypatch.setattr(
        artifact_module,
        "MAX_SCIENTIFIC_MEMBER_BYTES",
        512 * 1024 * 1024,
    )
    monkeypatch.setattr(artifact_module, "MAX_ARTIFACT_BYTES", 10)
    with pytest.raises(ArtifactValidationError, match="complete artifact"):
        load_artifact(written_artifact.path)


def test_manifest_implied_row_cap_precedes_supplied_or_parsed_rows(
    written_artifact,
    artifact_request: ArtifactWriteRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    implied_rows = (
        artifact_request.manifest.source.sequence_count
        * len(expected_conditions(artifact_request.manifest))
        * len(artifact_request.manifest.methods)
    )
    monkeypatch.setattr(artifact_module, "MAX_SEQUENCE_ROWS", implied_rows - 1)
    truncated_request = ArtifactWriteRequest(
        manifest=artifact_request.manifest,
        run=artifact_request.run,
        metrics=artifact_request.metrics[:1],
        aggregates=artifact_request.aggregates,
        crossovers=artifact_request.crossovers,
        analytic_validation=artifact_request.analytic_validation,
    )
    with pytest.raises(ArtifactValidationError, match="manifest-implied"):
        write_artifact(
            truncated_request,
            tmp_path / "implied-too-large",
            git_metadata_dirs=(),
        )

    def fail_if_parsed(*_args: object, **_kwargs: object) -> tuple[()]:
        raise AssertionError("NDJSON parsing must not precede manifest-implied caps")

    monkeypatch.setattr(artifact_module, "_load_ndjson", fail_if_parsed)
    with pytest.raises(ArtifactValidationError, match="manifest-implied"):
        load_artifact(written_artifact.path)


def test_discover_git_metadata_and_unavailable_repository(tmp_path: Path) -> None:
    discovered = discover_git_metadata_dirs(Path.cwd())
    assert discovered
    assert all(path.is_dir() for path in discovered)

    with pytest.raises(ArtifactValidationError, match="unavailable"):
        discover_git_metadata_dirs(tmp_path)
