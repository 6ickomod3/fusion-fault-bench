# pyright: reportPrivateUsage=false

from __future__ import annotations

import builtins
import copy
import hashlib
import io
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import fusion_fault_bench.replay_runner as replay_runner
from fusion_fault_bench.artifacts import (
    ArtifactValidationError,
    canonical_json_bytes,
    derive_run_id,
)
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_REPLAY_IDENTITY_SET_SHA256,
    REPLAY_CURATED_ARTIFACT_CONTRACT,
)
from fusion_fault_bench.contracts.replay_v1 import (
    M5_REPLAY_INTENT_PATH,
    M5_REPLAY_INTENT_SHA256,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    RunRecordV1Alpha1,
    RuntimeEnvironment,
)
from fusion_fault_bench.provenance import CleanSourceSnapshot
from fusion_fault_bench.replay_artifacts import canonical_replay_ndjson_bytes


@dataclass(frozen=True, slots=True)
class _Row:
    row_id: str
    value: float


def _snapshot(source_root: Path) -> CleanSourceSnapshot:
    git_dir = source_root / ".git"
    git_dir.mkdir(exist_ok=True)
    return CleanSourceSnapshot(
        source_root=source_root.resolve(),
        git_revision="1" * 40,
        git_dir=git_dir.resolve(),
        git_common_dir=git_dir.resolve(),
        lockfile_sha256="2" * 64,
        package_version="0.1.0",
        manifest_relative_path=M5_REPLAY_INTENT_PATH.as_posix(),
    )


def _environment() -> RuntimeEnvironment:
    return RuntimeEnvironment(
        python_version="3.12.13",
        os_name="Darwin",
        os_release="1.0",
        machine="test-machine",
        cpu_model="Test CPU",
        logical_cpu_count=4,
        memory_bytes=8 * 1024**3,
    )


def _run(
    snapshot: CleanSourceSnapshot,
    *,
    label: str,
    seconds: int,
) -> RunRecordV1Alpha1:
    started = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    return RunRecordV1Alpha1(
        schema="ffb.run/v1alpha1",
        run_id=derive_run_id(
            manifest_sha256=M5_REPLAY_INTENT_SHA256,
            git_revision=snapshot.git_revision,
            lockfile_sha256=snapshot.lockfile_sha256,
            package_version=snapshot.package_version,
            artifact_contract=REPLAY_CURATED_ARTIFACT_CONTRACT,
        ),
        manifest_sha256=M5_REPLAY_INTENT_SHA256,
        package_version=snapshot.package_version,
        git_revision=snapshot.git_revision,
        source_dirty=False,
        lockfile_sha256=snapshot.lockfile_sha256,
        command=("ffb", "replay", "run", "--output-dir", f"reports/generated/{label}"),
        environment=_environment(),
        started_at=started,
        ended_at=started + timedelta(seconds=1),
        status="succeeded",
        artifact_sha256="0" * 64,
    )


def _resources(
    *,
    elapsed_seconds: float = 12.5,
    peak_rss_bytes: int = 128 * 1024**2,
) -> replay_runner.ReplayRunResources:
    return replay_runner.ReplayRunResources(
        elapsed_seconds=elapsed_seconds,
        peak_rss_bytes=peak_rss_bytes,
        measurement_scope=("metadata-through-canonical-scientific-members-before-publication"),
    )


def _external_resource_evidence(
    *,
    run_id: str,
    run_label: str,
    elapsed_seconds: float,
    peak_rss_bytes: int,
    local_artifact_sha256: str = "a" * 64,
    local_run_sha256: str | None = None,
    persisted_internal_elapsed_seconds: float = 1.0,
    persisted_internal_peak_rss_bytes: int = 1024,
) -> replay_runner.ReplayExecutionResourceEvidenceV1:
    return replay_runner.ReplayExecutionResourceEvidenceV1(
        schema="ffb.replay-execution-resource-evidence/v1",
        run_id=run_id,
        replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
        replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
        run_label=run_label,
        local_artifact_sha256=local_artifact_sha256,
        local_run_sha256=local_run_sha256 or ("b" if run_label == "primary" else "c") * 64,
        environment_sha256="d" * 64,
        logical_command_sha256="e" * 64,
        persisted_internal_elapsed_seconds=persisted_internal_elapsed_seconds,
        persisted_internal_peak_rss_bytes=persisted_internal_peak_rss_bytes,
        persisted_internal_measurement_scope=(
            "metadata-through-canonical-scientific-members-before-publication"
        ),
        tool_path="/usr/bin/time",
        tool_options=("-l",),
        parser_contract="ffb.darwin-time-l-strict/v1",
        raw_log_sha256=("f" if run_label == "primary" else "1") * 64,
        raw_log_byte_length=100,
        elapsed_seconds=elapsed_seconds,
        peak_rss_bytes=peak_rss_bytes,
        exit_status=0,
        scientific_replay_worker_count=1,
        cpu_process_scope="one-scientific-replay-worker-no-benchmark-multiprocessing",
        helper_process_policy="sequential-provenance-and-resource-measurement-helpers-only",
        accelerator_requested=False,
        wall_time_cap_seconds=1800.0,
        peak_rss_cap_bytes=1_073_741_824,
        wall_time_within_cap=True,
        peak_rss_within_cap=True,
        measurement_scope=(
            "operator-recorded-darwin-time-l-for-complete-replay-cli-lifetime;"
            "self-reported-not-independent-attestation"
        ),
    )


def _fake_benchmark() -> SimpleNamespace:
    return SimpleNamespace(
        plan=SimpleNamespace(),
        fit_calibration_sha256=replay_runner.M4_FROZEN_CALIBRATION_SHA256,
        data_master_seed=replay_runner.M5_DATA_MASTER_SEED,
        bootstrap_replicates=replay_runner._BOOTSTRAP_REPLICATES,
        scene_frame_counts=(16,) * 10,
        log_group_ordinals=("log-group:00", "log-group:01") * 5,
    )


def _install_fake_local_decoder(
    monkeypatch: pytest.MonkeyPatch,
    benchmark: SimpleNamespace,
) -> None:
    monkeypatch.setattr(
        replay_runner,
        "_decode_local_benchmark",
        lambda *_args, **_kwargs: benchmark,
    )


_SCHEMA_BY_PATH = {
    "descriptor-aggregates.ndjson": "ffb.replay-local-descriptor-aggregate/v1",
    "persistent-scene-evaluations.ndjson": ("ffb.replay-local-persistent-scene-evaluation/v1"),
    "persistent-population-metrics.ndjson": ("ffb.replay-local-persistent-population-metric/v1"),
    "persistent-crossovers.ndjson": "ffb.replay-local-persistent-crossover/v1",
    "health-sequence-results.ndjson": "ffb.replay-health-result/v1",
    "health-sequence-contrasts.ndjson": ("ffb.replay-local-health-sequence-contrast/v1"),
    "health-sequence-events.ndjson": "ffb.replay-health-sequence-event/v1",
    "health-population-metrics.ndjson": "ffb.replay-local-health-population-metric/v1",
}


def _members(
    *,
    changed_path: str | None = None,
) -> tuple[dict[str, bytes], dict[str, int]]:
    members: dict[str, bytes] = {}
    counts: dict[str, int] = {}
    for path in replay_runner.REPLAY_LOCAL_SCIENTIFIC_PATHS:
        count = 710 if path == "persistent-scene-evaluations.ndjson" else 1
        records = tuple(
            canonical_json_bytes(
                {
                    "schema": _SCHEMA_BY_PATH[path],
                    "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
                    "row_id": f"{path}:{index}",
                    "value": float(index + (1 if path == changed_path else 0)),
                }
            )
            for index in range(count)
        )
        members[path] = b"".join(records)
        counts[path] = count
    return members, counts


def _write(
    source_root: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    seconds: int,
    changed_path: str | None = None,
    resources: replay_runner.ReplayRunResources | None = None,
) -> replay_runner.LoadedReplayLocalArtifact:
    snapshot = _snapshot(source_root)
    members, counts = _members(changed_path=changed_path)
    benchmark = _fake_benchmark()
    _install_fake_local_decoder(monkeypatch, benchmark)
    pending = replay_runner._write_local_artifact(
        destination=source_root / "reports" / "generated" / label,
        snapshot=snapshot,
        benchmark=benchmark,  # type: ignore[arg-type]
        run=_run(snapshot, label=label, seconds=seconds),
        resources=resources if resources is not None else _resources(),
        members=members,
        record_counts=counts,
    )
    return replay_runner._finalize_local_artifact(
        pending,
        snapshot=snapshot,
    )


def test_dataset_root_is_fixed_absolute_real_and_checkout_disjoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()

    monkeypatch.delenv("NUSCENES_ROOT", raising=False)
    with pytest.raises(
        replay_runner.ReplayRunnerError,
        match="dataset environment is unavailable",
    ):
        replay_runner._resolve_dataset_root(source_root=source_root)

    monkeypatch.setenv("NUSCENES_ROOT", "relative")
    with pytest.raises(replay_runner.ReplayRunnerError, match="must be absolute"):
        replay_runner._resolve_dataset_root(source_root=source_root)

    inside = source_root / "dataset"
    inside.mkdir()
    monkeypatch.setenv("NUSCENES_ROOT", str(inside))
    with pytest.raises(replay_runner.ReplayRunnerError, match="disjoint"):
        replay_runner._resolve_dataset_root(source_root=source_root)

    external = tmp_path / "external"
    external.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(external, target_is_directory=True)
    monkeypatch.setenv("NUSCENES_ROOT", str(linked))
    with pytest.raises(replay_runner.ReplayRunnerError, match="validation failed"):
        replay_runner._resolve_dataset_root(source_root=source_root)

    monkeypatch.setenv("NUSCENES_ROOT", str(external))
    assert replay_runner._resolve_dataset_root(source_root=source_root) == external.resolve()


def test_metadata_guard_is_scoped_and_blocks_payload_open(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    version = dataset / "v1.0-mini"
    payload = dataset / "samples" / "CAM_FRONT" / "private.jpg"
    version.mkdir(parents=True)
    payload.parent.mkdir(parents=True)
    (version / "scene.json").write_text("[]", encoding="utf-8")
    payload.write_bytes(b"private")
    ordinary = tmp_path / "ordinary.txt"
    ordinary.write_text("ok", encoding="utf-8")
    dataset_fd = os.open(dataset, os.O_RDONLY)

    try:
        with replay_runner._metadata_read_guard(dataset) as evidence:
            assert (version / "scene.json").read_text(encoding="utf-8") == "[]"
            assert ordinary.read_text(encoding="utf-8") == "ok"
            with pytest.raises(OSError, match="blocked"):
                builtins.open(payload, "rb")  # noqa: SIM115
            with pytest.raises(OSError, match="blocked"):
                io.open(payload, "rb")  # noqa: SIM115, UP020
            with pytest.raises(OSError, match="blocked"):
                os.open(payload, os.O_RDONLY)
            with pytest.raises(OSError, match="blocked"):
                os.open(
                    Path("samples/CAM_FRONT/private.jpg"),
                    os.O_RDONLY,
                    dir_fd=dataset_fd,
                )
    finally:
        os.close(dataset_fd)

    assert evidence.metadata_table_reads == 1
    assert evidence.blocked_dataset_reads == 4
    assert evidence.raw_sensor_payload_reads == 4
    assert payload.read_bytes() == b"private"


def test_metadata_guard_rejects_allowlisted_symlink_and_hardlink_aliases(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    version = dataset / "v1.0-mini"
    version.mkdir(parents=True)
    external = tmp_path / "external.json"
    external.write_text("[]", encoding="utf-8")
    scene = version / "scene.json"

    scene.symlink_to(external)
    with (
        replay_runner._metadata_read_guard(dataset) as symlink_evidence,
        pytest.raises(OSError, match="blocked"),
    ):
        scene.read_text(encoding="utf-8")
    assert symlink_evidence.metadata_table_reads == 0
    assert symlink_evidence.blocked_dataset_reads == 1

    scene.unlink()
    os.link(external, scene)
    with (
        replay_runner._metadata_read_guard(dataset) as hardlink_evidence,
        pytest.raises(OSError, match="blocked"),
    ):
        scene.read_text(encoding="utf-8")
    assert external.stat().st_nlink == 2
    assert hardlink_evidence.metadata_table_reads == 0
    assert hardlink_evidence.blocked_dataset_reads == 1


def test_metadata_guard_rejects_an_outside_symlink_into_dataset(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    version = dataset / "v1.0-mini"
    version.mkdir(parents=True)
    scene = version / "scene.json"
    scene.write_text("[]", encoding="utf-8")
    outside_alias = tmp_path / "outside-scene.json"
    outside_alias.symlink_to(scene)

    with (
        replay_runner._metadata_read_guard(dataset) as evidence,
        pytest.raises(OSError, match="blocked"),
    ):
        outside_alias.read_text(encoding="utf-8")
    assert evidence.metadata_table_reads == 0
    assert evidence.blocked_dataset_reads == 1


def test_payload_attempt_is_reported_without_private_exception_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "private-dataset"
    dataset.mkdir()
    payload = dataset / "samples" / "secret.jpg"
    payload.parent.mkdir()
    payload.write_bytes(b"private")

    def malicious_loader(root: Path) -> Any:
        return (root / "samples" / "secret.jpg").read_bytes()

    monkeypatch.setattr(replay_runner, "load_nuscenes_mini", malicious_loader)
    with pytest.raises(replay_runner.ReplayRunnerError) as captured:
        replay_runner._load_population(dataset)

    assert str(dataset) not in str(captured.value)
    assert "secret.jpg" not in str(captured.value)
    assert captured.value.__cause__ is None


def test_scientific_serialization_keeps_only_aggregate_sufficient_rows() -> None:
    benchmark = SimpleNamespace(
        descriptor_aggregates=(_Row("descriptor", 1.0),),
        persistent_scene_evaluations=tuple(
            _Row(f"persistent:{index}", float(index)) for index in range(710)
        ),
        persistent_metrics=(_Row("persistent-metric", 2.0),),
        persistent_crossovers=(_Row("crossover", 3.0),),
        health_results=(_Row("health-result", 4.0),),
        health_contrasts=(_Row("health-contrast", 5.0),),
        health_events=(_Row("health-event", 6.0),),
        health_metrics=(_Row("health-metric", 7.0),),
    )

    members, counts = replay_runner._scientific_members(
        benchmark,  # type: ignore[arg-type]
        forbidden_paths=("/private/dataset",),
    )

    assert tuple(members) == replay_runner.REPLAY_LOCAL_SCIENTIFIC_PATHS
    assert counts["persistent-scene-evaluations.ndjson"] == 710
    assert all(value.endswith(b"\n") for value in members.values())
    assert b"track:" not in b"".join(members.values())
    assert b"/private/dataset" not in b"".join(members.values())


@pytest.mark.parametrize(
    "record",
    (
        {"object_id": "track:0001"},
        {"safe": "track:0001"},
        {"safe": "/private/dataset"},
        {"safe": "frame.jpg"},
        {"safe": "api_key=abcdefgh"},
    ),
)
def test_local_scientific_serializer_rejects_private_material(
    record: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match=r"private|payload|forbidden"):
        replay_runner._local_ndjson_bytes(
            (record,),
            schema="ffb.replay-local-descriptor-aggregate/v1",
            forbidden_paths=(),
        )


def test_local_artifact_is_exclusive_strict_and_symlink_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    artifact = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="primary",
        seconds=0,
    )

    assert artifact.path == (source_root / "reports/generated/primary").resolve()
    assert artifact.member_record_counts["persistent-scene-evaluations.ndjson"] == 710
    assert repr(artifact) == "LoadedReplayLocalArtifact()"
    reloaded = replay_runner.load_replay_local_artifact(artifact.path)
    assert reloaded.artifact_sha256 == artifact.artifact_sha256
    assert reloaded.run_sha256 == artifact.run_sha256

    snapshot = _snapshot(source_root)
    members, counts = _members()
    with pytest.raises(FileExistsError):
        replay_runner._write_local_artifact(
            destination=artifact.path,
            snapshot=snapshot,
            benchmark=_fake_benchmark(),  # type: ignore[arg-type]
            run=_run(snapshot, label="primary", seconds=2),
            resources=_resources(),
            members=members,
            record_counts=counts,
        )

    linked = source_root / "reports/generated/linked"
    linked.symlink_to(artifact.path, target_is_directory=True)
    with pytest.raises(ArtifactValidationError, match="real directory"):
        replay_runner.load_replay_local_artifact(linked)


def test_local_loader_rejects_member_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    artifact = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="primary",
        seconds=0,
    )
    target = artifact.path / "descriptor-aggregates.ndjson"
    target.write_bytes(target.read_bytes().replace(b'"value":0.0', b'"value":1.0'))

    with pytest.raises(ArtifactValidationError, match="commitment"):
        replay_runner.load_replay_local_artifact(artifact.path)


def test_local_loader_rejects_valid_resource_mutation_against_success_commitment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    artifact = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="primary",
        seconds=0,
    )
    target = artifact.path / "resources.json"
    mutated = replay_runner._strict_json_mapping(
        target.read_bytes(),
        label="resources.json",
    )
    mutated["elapsed_seconds"] = 13.5
    target.write_bytes(canonical_json_bytes(mutated))

    assert (artifact.path / "_SUCCESS").is_file()
    with pytest.raises(ArtifactValidationError, match="resource commitment"):
        replay_runner.load_replay_local_artifact(artifact.path)


def test_repeat_verification_compares_all_members_and_independent_inodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    primary = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="primary",
        seconds=0,
    )
    repeat = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="repeat",
        seconds=5,
        resources=_resources(
            elapsed_seconds=13.5,
            peak_rss_bytes=160 * 1024**2,
        ),
    )
    commitments, verification = replay_runner.build_replay_repeat_verification(
        primary,
        repeat,
    )

    assert tuple(row.relative_role for row in commitments) == tuple(
        sorted(row.relative_role for row in commitments)
    )
    assert len(commitments) == len(replay_runner.REPLAY_LOCAL_SCIENTIFIC_PATHS)
    assert all(row.equal for row in commitments)
    assert all(
        row.replay_identity_set_sha256 == M5_REPLAY_IDENTITY_SET_SHA256 for row in commitments
    )
    assert verification.scientific_member_count == len(commitments)
    assert (
        verification.source_member_commitments_sha256
        == hashlib.sha256(canonical_replay_ndjson_bytes(commitments)).hexdigest()
    )
    assert verification.primary_local_artifact_sha256 == primary.artifact_sha256
    assert verification.repeat_local_artifact_sha256 == repeat.artifact_sha256
    assert primary.artifact_sha256 != repeat.artifact_sha256
    assert verification.mismatch_count == 0
    assert verification.run_records_distinct
    assert verification.source_paths_and_inodes_independent
    assert verification.same_named_cpu_environment
    assert verification.all_checks_passed


def test_repeat_verification_preserves_a_scientific_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    primary = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="primary",
        seconds=0,
    )
    repeat = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="repeat",
        seconds=5,
        changed_path="health-population-metrics.ndjson",
    )

    commitments, verification = replay_runner.build_replay_repeat_verification(
        primary,
        repeat,
    )

    assert sum(not row.equal for row in commitments) == 1
    assert verification.mismatch_count == 1
    assert not verification.scientific_members_all_equal
    assert not verification.all_checks_passed


def test_resource_caps_output_boundary_and_pending_release_evidence(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    with pytest.raises(ValueError, match="frozen cap"):
        replay_runner.ReplayRunResources(
            elapsed_seconds=1800.0,
            peak_rss_bytes=128 * 1024**2,
            measurement_scope=(
                "metadata-through-publication-and-final-source-verification-before-profile-binding"
            ),
        )
    with pytest.raises(ValueError, match="frozen cap"):
        replay_runner.ReplayRunResources(
            elapsed_seconds=1.0,
            peak_rss_bytes=1024**3,
            measurement_scope=(
                "metadata-through-publication-and-final-source-verification-before-profile-binding"
            ),
        )
    with pytest.raises(replay_runner.ReplayRunnerError, match="reports/generated"):
        replay_runner._validated_output_path(
            Path("reports/releases/m5"),
            source_root=source_root,
        )

    pending = replay_runner.replay_runner_pending_validation_checks()
    assert "implementation-review" in pending
    assert "results-and-claims-review" in pending
    assert "software-verification" in pending
    assert "repeat-scientific-members" not in pending


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("elapsed_seconds", float("nan")),
        ("elapsed_seconds", -0.01),
        ("elapsed_seconds", 1),
        ("elapsed_seconds", "1.0"),
        ("peak_rss_bytes", 0),
        ("peak_rss_bytes", True),
        ("raw_sensor_payload_reads", 1),
        ("raw_sensor_payload_reads", False),
        ("scientific_replay_worker_count", 2),
        ("scientific_replay_worker_count", True),
        ("gpu_used", True),
        ("gpu_used", 0),
        ("torch_imported", True),
        ("torch_imported", 0),
        ("cuda_used", True),
        ("cuda_used", 0),
        ("measurement_scope", "process-lifetime"),
    ),
)
def test_resource_contract_rejects_each_non_cpu_or_unbounded_variant(
    field: str,
    value: object,
) -> None:
    arguments: dict[str, object] = {
        "elapsed_seconds": 1.0,
        "peak_rss_bytes": 128 * 1024**2,
        "measurement_scope": (
            "metadata-through-publication-and-final-source-verification-before-profile-binding"
        ),
        "raw_sensor_payload_reads": 0,
        "scientific_replay_worker_count": 1,
        "gpu_used": False,
        "torch_imported": False,
        "cuda_used": False,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match="frozen cap"):
        replay_runner.ReplayRunResources(**arguments)  # type: ignore[arg-type]


def test_execution_repr_and_failed_repeat_cannot_be_represented(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    artifact = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="primary",
        seconds=0,
    )
    local = replay_runner.ReplayLocalExecution(
        benchmark=cast(Any, SimpleNamespace()),
        artifact=artifact,
        run=artifact.run,
        resources=artifact.resources,
    )
    failed_verification = SimpleNamespace(all_checks_passed=False)

    assert repr(local) == "ReplayLocalExecution()"
    with pytest.raises(ValueError, match="failed release gate"):
        replay_runner.ReplayRepeatExecution(
            primary=local,
            repeat=local,
            source_commitments=(),
            repeat_verification=cast(Any, failed_verification),
        )

    passed_verification = SimpleNamespace(all_checks_passed=True)
    repeat = replay_runner.ReplayRepeatExecution(
        primary=local,
        repeat=local,
        source_commitments=(),
        repeat_verification=cast(Any, passed_verification),
    )
    assert repr(repeat) == "ReplayRepeatExecution()"


def test_output_path_normalization_and_exclusivity(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    valid = Path("reports/generated/m5-primary")

    target, argument = replay_runner._validated_output_path(
        valid,
        source_root=source_root,
    )
    assert target == source_root / valid
    assert argument == valid.as_posix()

    for invalid in (
        source_root / "absolute",
        Path("reports/generated/../private"),
        Path("reports/generated"),
    ):
        with pytest.raises(replay_runner.ReplayRunnerError):
            replay_runner._validated_output_path(invalid, source_root=source_root)

    target.mkdir(parents=True)
    with pytest.raises(replay_runner.ReplayRunnerError, match="already exists"):
        replay_runner._validated_output_path(valid, source_root=source_root)


def test_cpu_import_boundary_and_platform_rss_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch.testing-private", SimpleNamespace())
    with pytest.raises(replay_runner.ReplayRunnerError, match="CPU-only"):
        replay_runner._ensure_cpu_only_import_boundary()
    monkeypatch.delitem(sys.modules, "torch.testing-private")
    replay_runner._ensure_cpu_only_import_boundary()

    usage = SimpleNamespace(ru_maxrss=123)
    monkeypatch.setattr(replay_runner.resource, "getrusage", lambda _: usage)
    monkeypatch.setattr(replay_runner.sys, "platform", "linux")
    assert replay_runner._peak_rss_bytes() == 123 * 1024
    monkeypatch.setattr(replay_runner.sys, "platform", "darwin")
    assert replay_runner._peak_rss_bytes() == 123


def test_json_mapping_and_recursive_privacy_scanner_cover_supported_shapes() -> None:
    assert replay_runner._json_mapping(_Row("row", 1.5)) == {
        "row_id": "row",
        "value": 1.5,
    }
    assert replay_runner._json_mapping({"key": "value"}) == {"key": "value"}
    assert replay_runner._json_mapping(_environment())["cpu_model"] == "Test CPU"
    with pytest.raises(TypeError, match="not serializable"):
        replay_runner._json_mapping(object())

    replay_runner._scan_local_value(
        {"safe": [1, 2.0, None, {"nested": "public-label"}]},
        forbidden_paths=(),
    )
    with pytest.raises(ValueError, match="private"):
        replay_runner._scan_local_value(
            {"safe": [{"nested": "contains-/private/fixed-root-here"}]},
            forbidden_paths=("/private/fixed-root",),
        )


def test_local_row_binding_and_ndjson_byte_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = "ffb.replay-local-descriptor-aggregate/v1"
    bound = replay_runner._local_row_mapping(
        {
            "schema": schema,
            "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
            "value": 1.0,
        },
        schema=schema,
        forbidden_paths=(),
    )
    assert bound["schema"] == schema
    assert bound["replay_intent_sha256"] == M5_REPLAY_INTENT_SHA256

    with pytest.raises(ValueError, match="schema"):
        replay_runner._local_row_mapping(
            {"schema": "wrong"},
            schema=schema,
            forbidden_paths=(),
        )
    with pytest.raises(ValueError, match="frozen intent"):
        replay_runner._local_row_mapping(
            {"replay_intent_sha256": "0" * 64},
            schema=schema,
            forbidden_paths=(),
        )
    with pytest.raises(ValueError, match="nonempty"):
        replay_runner._local_ndjson_bytes(
            (),
            schema=schema,
            forbidden_paths=(),
        )

    monkeypatch.setattr(replay_runner, "_LOCAL_MAX_RECORD_BYTES", 8)
    with pytest.raises(ValueError, match="record exceeds"):
        replay_runner._local_ndjson_bytes(
            ({"value": "bounded"},),
            schema=schema,
            forbidden_paths=(),
        )
    monkeypatch.setattr(replay_runner, "_LOCAL_MAX_RECORD_BYTES", 1024 * 1024)
    monkeypatch.setattr(replay_runner, "_LOCAL_MAX_MEMBER_BYTES", 1)
    with pytest.raises(ValueError, match="member exceeds"):
        replay_runner._local_ndjson_bytes(
            ({"value": 1},),
            schema=schema,
            forbidden_paths=(),
        )


def test_scientific_members_reject_incomplete_scene_grid_and_total_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = SimpleNamespace(
        descriptor_aggregates=(_Row("descriptor", 1.0),),
        persistent_scene_evaluations=tuple(
            _Row(f"persistent:{index}", float(index)) for index in range(709)
        ),
        persistent_metrics=(_Row("persistent-metric", 2.0),),
        persistent_crossovers=(_Row("crossover", 3.0),),
        health_results=(_Row("health-result", 4.0),),
        health_contrasts=(_Row("health-contrast", 5.0),),
        health_events=(_Row("health-event", 6.0),),
        health_metrics=(_Row("health-metric", 7.0),),
    )
    with pytest.raises(ValueError, match="710"):
        replay_runner._scientific_members(
            benchmark,  # type: ignore[arg-type]
            forbidden_paths=(),
        )

    benchmark.persistent_scene_evaluations = tuple(
        _Row(f"persistent:{index}", float(index)) for index in range(710)
    )
    monkeypatch.setattr(replay_runner, "_LOCAL_MAX_ARTIFACT_BYTES", 1)
    with pytest.raises(ValueError, match="artifact cap"):
        replay_runner._scientific_members(
            benchmark,  # type: ignore[arg-type]
            forbidden_paths=(),
        )


@pytest.mark.parametrize(
    "value",
    (
        b"",
        b'{"a":1}',
        b'{"a":1}\r\n',
        b'{"a":1}\n{"b":2}\n',
        b'{"a":1,"a":2}\n',
        b'{"a":NaN}\n',
        b"[1]\n",
        b'{"a": 1}\n',
    ),
)
def test_strict_json_rejects_noncanonical_or_ambiguous_records(value: bytes) -> None:
    with pytest.raises(ArtifactValidationError):
        replay_runner._strict_json_mapping(value, label="member")


def test_strict_json_and_ndjson_accept_canonical_records_and_apply_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = canonical_json_bytes({"a": 1})
    second = canonical_json_bytes({"b": 2})
    assert replay_runner._strict_json_mapping(first, label="member") == {"a": 1}
    records, count = replay_runner._strict_ndjson(first + second, label="member")
    assert records == ({"a": 1}, {"b": 2})
    assert count == 2

    for invalid in (b"", b'{"a":1}', b'{"a":1}\r\n'):
        with pytest.raises(ArtifactValidationError, match="NDJSON"):
            replay_runner._strict_ndjson(invalid, label="member")
    monkeypatch.setattr(replay_runner, "_LOCAL_MAX_RECORD_BYTES", len(first) - 1)
    with pytest.raises(ArtifactValidationError, match="oversized"):
        replay_runner._strict_ndjson(first, label="member")


def test_source_index_and_resource_mapping_are_exact_contracts() -> None:
    members, counts = _members()
    run_id = "run:" + "1" * 64
    benchmark = _fake_benchmark()
    resources = _resources()
    resource_mapping = replay_runner._resources_mapping(run_id, resources)
    resources_bytes = canonical_json_bytes(resource_mapping)
    index = replay_runner._source_index_mapping(
        run_id=run_id,
        benchmark=benchmark,  # type: ignore[arg-type]
        members=members,
        record_counts=counts,
        resources_bytes=resources_bytes,
    )
    context = replay_runner._validate_source_index(
        index,
        members=members,
        record_counts=counts,
        resources_bytes=resources_bytes,
    )
    assert context.run_id == run_id
    assert context.scene_frame_counts == (16,) * 10
    assert context.log_group_ordinals == ("log-group:00", "log-group:01") * 5

    bad_index = copy.deepcopy(index)
    bad_index["extra"] = True
    with pytest.raises(ArtifactValidationError, match="index is invalid"):
        replay_runner._validate_source_index(
            bad_index,
            members=members,
            record_counts=counts,
            resources_bytes=resources_bytes,
        )

    bad_index = copy.deepcopy(index)
    cast(list[object], bad_index["members"]).pop()
    with pytest.raises(ArtifactValidationError, match="incomplete"):
        replay_runner._validate_source_index(
            bad_index,
            members=members,
            record_counts=counts,
            resources_bytes=resources_bytes,
        )

    bad_index = copy.deepcopy(index)
    cast(list[object], bad_index["members"])[0] = "not-an-object"
    with pytest.raises(ArtifactValidationError, match="entry is invalid"):
        replay_runner._validate_source_index(
            bad_index,
            members=members,
            record_counts=counts,
            resources_bytes=resources_bytes,
        )

    bad_index = copy.deepcopy(index)
    first_entry = cast(dict[str, object], cast(list[object], bad_index["members"])[0])
    first_entry["record_count"] = 999
    with pytest.raises(ArtifactValidationError, match="commitment"):
        replay_runner._validate_source_index(
            bad_index,
            members=members,
            record_counts=counts,
            resources_bytes=resources_bytes,
        )

    assert index["resources_member"] == {
        "path": "resources.json",
        "relative_role": "execution-resource-diagnostics",
        "schema": "ffb.replay-local-resources/v1",
        "byte_length": len(resources_bytes),
        "record_count": 1,
        "sha256": hashlib.sha256(resources_bytes).hexdigest(),
    }
    bad_index = copy.deepcopy(index)
    cast(dict[str, object], bad_index["resources_member"])["sha256"] = "0" * 64
    with pytest.raises(ArtifactValidationError, match="resource commitment"):
        replay_runner._validate_source_index(
            bad_index,
            members=members,
            record_counts=counts,
            resources_bytes=resources_bytes,
        )

    assert (
        replay_runner._resources_from_mapping(
            resource_mapping,
            run_id=run_id,
        )
        == resources
    )
    for key, value in (
        ("run_id", "run:" + "2" * 64),
        ("wall_time_cap_seconds", 1.0),
        ("wall_time_cap_seconds", 1800),
        ("peak_rss_cap_bytes", 1),
        ("peak_rss_cap_bytes", float(1024**3)),
        ("elapsed_seconds", True),
        ("elapsed_seconds", "12.5"),
        ("peak_rss_bytes", float(128 * 1024**2)),
        ("peak_rss_bytes", "not-an-integer"),
        ("raw_sensor_payload_reads", False),
        ("scientific_replay_worker_count", True),
        ("gpu_used", 0),
        ("torch_imported", 0),
        ("cuda_used", 0),
    ):
        mutated = dict(resource_mapping)
        mutated[key] = value
        with pytest.raises(ArtifactValidationError, match="resource evidence"):
            replay_runner._resources_from_mapping(mutated, run_id=run_id)


def test_local_tree_rejects_extra_symlink_and_hardlinked_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    artifact = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="primary",
        seconds=0,
    )

    extra = artifact.path / "extra.json"
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="allowlist"):
        replay_runner.load_replay_local_artifact(artifact.path)
    extra.unlink()

    target = artifact.path / "descriptor-aggregates.ndjson"
    original = target.read_bytes()
    target.unlink()
    target.symlink_to(artifact.path / "health-population-metrics.ndjson")
    with pytest.raises(ArtifactValidationError, match="regular file"):
        replay_runner.load_replay_local_artifact(artifact.path)
    target.unlink()
    target.write_bytes(original)

    backup = artifact.path.parent / "descriptor-backup.ndjson"
    backup.write_bytes(original)
    target.unlink()
    os.link(backup, target)
    with pytest.raises(ArtifactValidationError, match="hard-linked"):
        replay_runner.load_replay_local_artifact(artifact.path)


@pytest.mark.parametrize("mutation", ["hardlink", "replace", "rewrite"])
def test_local_member_read_reauthenticates_after_concurrent_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    member = tmp_path / "private-member.ndjson"
    member.write_bytes(b"original")
    expected = member.lstat()
    alias = tmp_path / "private-member-alias.ndjson"
    displaced = tmp_path / "private-member-displaced.ndjson"
    replacement = tmp_path / "private-member-replacement.ndjson"
    replacement.write_bytes(b"replaced")
    original_read = os.read
    mutated = False

    def mutating_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        value = original_read(descriptor, count)
        if value and not mutated:
            mutated = True
            if mutation == "hardlink":
                os.link(member, alias)
            elif mutation == "replace":
                member.replace(displaced)
                replacement.replace(member)
            else:
                member.write_bytes(b"rewritten")
        return value

    monkeypatch.setattr(replay_runner.os, "read", mutating_read)

    with pytest.raises(ArtifactValidationError, match="changed during load"):
        replay_runner._read_local_member(member, expected=expected)


def test_local_loader_rejects_row_count_run_resource_and_success_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()

    row_artifact = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="row",
        seconds=0,
    )
    row_path = row_artifact.path / "descriptor-aggregates.ndjson"
    row_payload = replay_runner._strict_json_mapping(
        row_path.read_bytes(),
        label="descriptor",
    )
    row_payload["schema"] = "ffb.wrong/v1"
    row_path.write_bytes(canonical_json_bytes(row_payload))
    with pytest.raises(ArtifactValidationError, match="row binding"):
        replay_runner.load_replay_local_artifact(row_artifact.path)

    count_artifact = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="count",
        seconds=1,
    )
    count_path = count_artifact.path / "persistent-scene-evaluations.ndjson"
    count_path.write_bytes(b"".join(count_path.read_bytes().splitlines(keepends=True)[:-1]))
    with pytest.raises(ArtifactValidationError, match="incomplete"):
        replay_runner.load_replay_local_artifact(count_artifact.path)

    run_artifact = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="run",
        seconds=2,
    )
    invalid_run = run_artifact.run.model_copy(update={"artifact_sha256": "f" * 64})
    (run_artifact.path / "run.json").write_bytes(canonical_json_bytes(invalid_run))
    with pytest.raises(ArtifactValidationError, match="run identity"):
        replay_runner.load_replay_local_artifact(run_artifact.path)

    resource_artifact = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="resources",
        seconds=3,
    )
    resources_path = resource_artifact.path / "resources.json"
    resources_payload = replay_runner._strict_json_mapping(
        resources_path.read_bytes(),
        label="resources",
    )
    resources_payload["gpu_used"] = True
    resources_path.write_bytes(canonical_json_bytes(resources_payload))
    with pytest.raises(ArtifactValidationError, match="resource"):
        replay_runner.load_replay_local_artifact(resource_artifact.path)

    success_artifact = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="success",
        seconds=4,
    )
    success_path = success_artifact.path / "_SUCCESS"
    success_payload = replay_runner._strict_json_mapping(
        success_path.read_bytes(),
        label="success",
    )
    success_payload["artifact_sha256"] = "f" * 64
    success_path.write_bytes(canonical_json_bytes(success_payload))
    with pytest.raises(ArtifactValidationError, match="success marker"):
        replay_runner.load_replay_local_artifact(success_artifact.path)


def test_writer_cleans_staging_after_verification_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    snapshot = _snapshot(source_root)
    members, counts = _members()
    destination = source_root / "reports/generated/primary"
    benchmark = _fake_benchmark()
    _install_fake_local_decoder(monkeypatch, benchmark)
    monkeypatch.setattr(replay_runner, "read_file_at", lambda *args, **kwargs: b"wrong")

    with pytest.raises(ArtifactValidationError, match="staging verification"):
        replay_runner._write_local_artifact(
            destination=destination,
            snapshot=snapshot,
            benchmark=benchmark,  # type: ignore[arg-type]
            run=_run(snapshot, label="primary", seconds=0),
            resources=_resources(),
            members=members,
            record_counts=counts,
        )

    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []


def test_writer_removes_published_artifact_when_final_reload_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    snapshot = _snapshot(source_root)
    members, counts = _members()
    destination = source_root / "reports/generated/primary"
    benchmark = _fake_benchmark()
    _install_fake_local_decoder(monkeypatch, benchmark)
    real_loader = replay_runner._load_replay_local_payload
    load_count = 0

    def fail_second_load(
        path: Path,
        **kwargs: object,
    ) -> replay_runner._PendingReplayLocalArtifact:
        nonlocal load_count
        load_count += 1
        if load_count == 2:
            raise ArtifactValidationError("simulated published reload failure")
        return real_loader(path, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(replay_runner, "_load_replay_local_payload", fail_second_load)
    with pytest.raises(ArtifactValidationError, match="published reload"):
        replay_runner._write_local_artifact(
            destination=destination,
            snapshot=snapshot,
            benchmark=benchmark,  # type: ignore[arg-type]
            run=_run(snapshot, label="primary", seconds=0),
            resources=_resources(),
            members=members,
            record_counts=counts,
        )

    assert load_count == 2
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []


def test_source_snapshot_gates_are_exact_and_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    snapshot = _snapshot(source_root)
    verified: list[CleanSourceSnapshot] = []
    monkeypatch.setattr(replay_runner, "discover_clean_source", lambda _: snapshot)
    monkeypatch.setattr(replay_runner, "verify_locked_execution", verified.append)
    assert replay_runner._initial_snapshot() == snapshot
    replay_runner._verify_unchanged_source(snapshot)
    assert verified == [snapshot, snapshot]

    wrong_manifest = replace(snapshot, manifest_relative_path="examples/wrong.json")
    monkeypatch.setattr(replay_runner, "discover_clean_source", lambda _: wrong_manifest)
    with pytest.raises(replay_runner.ReplayRunnerError, match="frozen replay intent"):
        replay_runner._initial_snapshot()

    changed = replace(snapshot, git_revision="3" * 40)
    monkeypatch.setattr(replay_runner, "discover_clean_source", lambda _: changed)
    with pytest.raises(replay_runner.ReplayRunnerError, match="changed"):
        replay_runner._verify_unchanged_source(snapshot)

    def fail_discovery(_: Path) -> CleanSourceSnapshot:
        raise OSError("/private/secret/checkout")

    monkeypatch.setattr(replay_runner, "discover_clean_source", fail_discovery)
    with pytest.raises(replay_runner.ReplayRunnerError) as captured:
        replay_runner._initial_snapshot()
    assert "/private/secret" not in str(captured.value)
    assert captured.value.__cause__ is None


def test_population_loader_requires_all_metadata_attestations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    validation = SimpleNamespace(
        headline_profile_passed_attested=True,
        structural_integrity_passed_attested=True,
        keyframe_blob_check_count=replay_runner.EXPECTED_KEYFRAME_BLOB_CHECK_COUNT,
        keyframe_blob_validation_passed_attested=True,
    )
    metadata = SimpleNamespace(validation=validation)
    population = SimpleNamespace(population_id="sentinel")

    @contextmanager
    def accepted_guard(_: Path) -> Any:
        yield replay_runner._MetadataReadEvidence(
            metadata_table_reads=len(replay_runner._NUSCENES_TABLES)
        )

    monkeypatch.setattr(replay_runner, "_metadata_read_guard", accepted_guard)
    monkeypatch.setattr(replay_runner, "load_nuscenes_mini", lambda _: metadata)
    monkeypatch.setattr(replay_runner, "extract_m5_replay_source", lambda _: population)
    assert replay_runner._load_population(dataset) is population

    validation.structural_integrity_passed_attested = False
    with pytest.raises(replay_runner.ReplayRunnerError, match="acceptance gate"):
        replay_runner._load_population(dataset)


def test_metadata_guard_blocks_writes_but_allows_existing_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    version = dataset / "v1.0-mini"
    version.mkdir(parents=True)
    scene = version / "scene.json"
    scene.write_text("[]", encoding="utf-8")
    descriptor = os.open(scene, os.O_RDONLY)
    dataset_fd = os.open(dataset, os.O_RDONLY)
    try:
        with replay_runner._metadata_read_guard(dataset) as evidence:
            with builtins.open(descriptor, "rb", closefd=False) as stream:
                assert stream.read() == b"[]"
            with pytest.raises(OSError, match="blocked"):
                builtins.open(scene, "w")  # noqa: SIM115
            with pytest.raises(OSError, match="blocked"):
                io.open(scene, "a")  # noqa: SIM115, UP020
            with pytest.raises(OSError, match="blocked"):
                os.open(scene, os.O_RDWR)

            monkeypatch.setattr(replay_runner.sys, "platform", "linux")
            monkeypatch.setattr(replay_runner.os, "readlink", lambda _: os.fspath(dataset))
            allowed_fd = os.open(
                Path("v1.0-mini/scene.json"),
                os.O_RDONLY,
                dir_fd=dataset_fd,
            )
            try:
                assert os.read(allowed_fd, 2) == b"[]"
            finally:
                os.close(allowed_fd)
    finally:
        os.close(descriptor)
        os.close(dataset_fd)

    assert evidence.metadata_table_reads == 1
    assert evidence.blocked_dataset_reads == 3


def _descriptor_row() -> SimpleNamespace:
    return SimpleNamespace(
        descriptor_id="frame-count",
        population="nuscenes-mini-replay",
        population_count=10,
        statistic="median",
        category_label=None,
        status="ok",
        value=40.0,
        unit="frames",
    )


def test_run_descriptor_and_profile_contract_binding(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    snapshot = _snapshot(source_root)
    started = datetime(2026, 1, 1, tzinfo=UTC)
    run = replay_runner._run_record(
        snapshot=snapshot,
        output_argument="reports/generated/primary",
        environment=_environment(),
        started_at=started,
        ended_at=started + timedelta(seconds=1),
    )
    benchmark = SimpleNamespace(
        descriptor_aggregates=(_descriptor_row(),),
        log_group_ordinals=(0, 0, 1, 1, 2, 2, 3, 3, 3, 3),
    )
    descriptors = replay_runner._descriptor_contracts(
        benchmark,  # type: ignore[arg-type]
        run_id=run.run_id,
    )
    assert len(descriptors) == 1
    assert descriptors[0].run_id == run.run_id
    assert run.command[-1] == "reports/generated/primary"

    resource_evidence = (
        _external_resource_evidence(
            run_id=run.run_id,
            run_label="primary",
            elapsed_seconds=2.0,
            peak_rss_bytes=128 * 1024**2,
        ),
        _external_resource_evidence(
            run_id=run.run_id,
            run_label="repeat",
            elapsed_seconds=3.0,
            peak_rss_bytes=192 * 1024**2,
        ),
    )
    profile = replay_runner._profile_summary(
        benchmark,  # type: ignore[arg-type]
        run_id=run.run_id,
        resource_evidence=resource_evidence,
    )
    assert profile.distinct_log_group_count == 4
    assert profile.elapsed_seconds == 3.0
    assert profile.peak_rss_bytes == 192 * 1024**2
    with pytest.raises(ValueError, match="primary/repeat order"):
        replay_runner._profile_summary(
            benchmark,  # type: ignore[arg-type]
            run_id=run.run_id,
            resource_evidence=tuple(reversed(resource_evidence)),  # type: ignore[arg-type]
        )


def _mock_execute_dependencies(
    *,
    source_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[CleanSourceSnapshot, SimpleNamespace, SimpleNamespace, dict[str, int]]:
    snapshot = _snapshot(source_root)
    benchmark = SimpleNamespace(
        plan=SimpleNamespace(),
        fit_calibration_sha256=replay_runner.M4_FROZEN_CALIBRATION_SHA256,
        data_master_seed=replay_runner.M5_DATA_MASTER_SEED,
        bootstrap_replicates=replay_runner._BOOTSTRAP_REPLICATES,
        scene_frame_counts=(16,) * 10,
        descriptor_aggregates=(_descriptor_row(),),
        log_group_ordinals=("log-group:00", "log-group:01") * 5,
    )
    population = SimpleNamespace(population_id="population")
    artifact = SimpleNamespace(
        artifact_sha256="a" * 64,
        run_sha256="b" * 64,
        benchmark=benchmark,
    )
    pending = SimpleNamespace(benchmark=benchmark)
    calls = {
        "source_verifications": 0,
        "finalizations": 0,
        "rollbacks": 0,
        "cpu_boundary_checks": 0,
    }
    monkeypatch.setattr(
        replay_runner,
        "_validated_output_path",
        lambda *_args, **_kwargs: (
            source_root / "reports/generated/primary",
            "reports/generated/primary",
        ),
    )
    monkeypatch.setattr(
        replay_runner,
        "_resolve_dataset_root",
        lambda **_kwargs: source_root.parent / "dataset",
    )
    monkeypatch.setattr(
        replay_runner,
        "_ensure_cpu_only_import_boundary",
        lambda: calls.__setitem__(
            "cpu_boundary_checks",
            calls["cpu_boundary_checks"] + 1,
        ),
    )
    monkeypatch.setattr(replay_runner, "_load_population", lambda _: population)
    monkeypatch.setattr(replay_runner, "run_replay_benchmark", lambda *args, **kwargs: benchmark)
    monkeypatch.setattr(replay_runner, "collect_runtime_environment", _environment)
    monkeypatch.setattr(
        replay_runner,
        "_scientific_members",
        lambda *args, **kwargs: ({"member": b"{}\n"}, {"member": 1}),
    )
    monkeypatch.setattr(replay_runner, "_peak_rss_bytes", lambda: 128 * 1024**2)
    monkeypatch.setattr(
        replay_runner,
        "_verify_unchanged_source",
        lambda _: calls.__setitem__(
            "source_verifications",
            calls["source_verifications"] + 1,
        ),
    )
    monkeypatch.setattr(replay_runner, "_write_local_artifact", lambda **kwargs: pending)
    monkeypatch.setattr(
        replay_runner,
        "_finalize_local_artifact",
        lambda *_args, **_kwargs: (
            calls.__setitem__("finalizations", calls["finalizations"] + 1) or artifact
        ),
    )
    monkeypatch.setattr(
        replay_runner,
        "_rollback_pending_local_artifact",
        lambda *_args, **_kwargs: calls.__setitem__("rollbacks", calls["rollbacks"] + 1),
    )
    return snapshot, benchmark, artifact, calls


def test_execute_replay_local_orchestrates_two_source_checks_and_resource_scopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    snapshot, benchmark, artifact, calls = _mock_execute_dependencies(
        source_root=source_root,
        monkeypatch=monkeypatch,
    )

    execution = replay_runner._execute_replay_local(
        Path("reports/generated/primary"),
        snapshot=snapshot,
    )

    assert execution.benchmark is benchmark
    assert execution.artifact is artifact
    assert calls["source_verifications"] == 2
    assert calls["finalizations"] == 1
    assert calls["rollbacks"] == 0
    assert calls["cpu_boundary_checks"] == 2
    assert (
        execution.resources.measurement_scope
        == "metadata-through-publication-and-final-source-verification-before-profile-binding"
    )


@pytest.mark.parametrize(
    ("stage", "expected"),
    (
        ("benchmark", "benchmark computation"),
        ("evidence", "evidence construction"),
        ("publication", "artifact publication"),
        ("post-publication", "post-publication resource gate"),
    ),
)
def test_execute_replay_local_sanitizes_stage_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    expected: str,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    snapshot, _, _, _ = _mock_execute_dependencies(
        source_root=source_root,
        monkeypatch=monkeypatch,
    )

    def fail(*args: object, **kwargs: object) -> Any:
        raise ValueError("/private/dataset/secret.json")

    if stage == "benchmark":
        monkeypatch.setattr(replay_runner, "run_replay_benchmark", fail)
    elif stage == "evidence":
        monkeypatch.setattr(replay_runner, "_scientific_members", fail)
    elif stage == "publication":
        monkeypatch.setattr(replay_runner, "_write_local_artifact", fail)
    else:
        rss_values = iter((128 * 1024**2, 1024**3))
        monkeypatch.setattr(replay_runner, "_peak_rss_bytes", lambda: next(rss_values))

    with pytest.raises(replay_runner.ReplayRunnerError, match=expected) as captured:
        replay_runner._execute_replay_local(
            Path("reports/generated/primary"),
            snapshot=snapshot,
        )
    assert "/private/dataset" not in str(captured.value)
    assert captured.value.__cause__ is None


def test_run_replay_local_delegates_one_frozen_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    snapshot = _snapshot(source_root)
    sentinel = SimpleNamespace(execution="sentinel")
    monkeypatch.setattr(replay_runner, "_initial_snapshot", lambda: snapshot)
    monkeypatch.setattr(
        replay_runner,
        "_execute_replay_local",
        lambda output, *, snapshot: (output, snapshot, sentinel),
    )
    output = Path("reports/generated/primary")
    assert replay_runner.run_replay_local(output) == (output, snapshot, sentinel)


def test_repeat_verification_rejects_identity_and_records_independence_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    primary = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="primary",
        seconds=0,
    )
    repeat = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="repeat",
        seconds=5,
    )

    wrong_run = repeat.run.model_copy(update={"run_id": "run:" + "f" * 64})
    with pytest.raises(ValueError, match="run identity"):
        replay_runner.build_replay_repeat_verification(
            primary,
            replace(repeat, run=wrong_run),
        )

    _, reused = replay_runner.build_replay_repeat_verification(primary, primary)
    assert not reused.run_records_distinct
    assert not reused.source_paths_and_inodes_independent
    assert not reused.all_checks_passed

    other_environment = _environment().model_copy(update={"cpu_model": "Other CPU"})
    _, different_environment = replay_runner.build_replay_repeat_verification(
        primary,
        replace(repeat, run=repeat.run.model_copy(update={"environment": other_environment})),
    )
    assert not different_environment.same_named_cpu_environment
    assert not different_environment.all_checks_passed

    _, missing_path = replay_runner.build_replay_repeat_verification(
        primary,
        replace(repeat, path=source_root / "missing"),
    )
    assert not missing_path.source_paths_and_inodes_independent
    assert not missing_path.all_checks_passed


def test_run_replay_repeat_success_disjointness_and_failure_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    snapshot = _snapshot(source_root)
    primary_artifact = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="artifact-primary",
        seconds=0,
    )
    repeat_artifact = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="artifact-repeat",
        seconds=5,
    )
    executions = iter(
        (
            SimpleNamespace(artifact=primary_artifact),
            SimpleNamespace(artifact=repeat_artifact),
        )
    )
    verification = SimpleNamespace(all_checks_passed=True)
    monkeypatch.setattr(replay_runner, "_initial_snapshot", lambda: snapshot)
    monkeypatch.setattr(
        replay_runner,
        "_execute_replay_local",
        lambda *_args, **_kwargs: next(executions),
    )
    monkeypatch.setattr(
        replay_runner,
        "build_replay_repeat_verification",
        lambda *_args: ((), verification),
    )
    result = replay_runner.run_replay_repeat(
        primary_output_dir=Path("reports/generated/run-primary"),
        repeat_output_dir=Path("reports/generated/run-repeat"),
    )
    assert result.repeat_verification is verification

    with pytest.raises(replay_runner.ReplayRunnerError, match="disjoint"):
        replay_runner.run_replay_repeat(
            primary_output_dir=Path("reports/generated/nested"),
            repeat_output_dir=Path("reports/generated/nested/repeat"),
        )


def test_run_replay_repeat_sanitizes_verification_and_gate_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    snapshot = _snapshot(source_root)
    execution = SimpleNamespace(artifact=SimpleNamespace())
    monkeypatch.setattr(replay_runner, "_initial_snapshot", lambda: snapshot)
    monkeypatch.setattr(replay_runner, "_execute_replay_local", lambda *_args, **_kwargs: execution)

    def fail_verification(*_args: object) -> Any:
        raise ValueError("/private/repeat/path")

    monkeypatch.setattr(
        replay_runner,
        "build_replay_repeat_verification",
        fail_verification,
    )
    with pytest.raises(replay_runner.ReplayRunnerError, match="verification failed") as captured:
        replay_runner.run_replay_repeat(
            primary_output_dir=Path("reports/generated/run-primary"),
            repeat_output_dir=Path("reports/generated/run-repeat"),
        )
    assert "/private/repeat" not in str(captured.value)
    assert captured.value.__cause__ is None

    monkeypatch.setattr(
        replay_runner,
        "build_replay_repeat_verification",
        lambda *_args: ((), SimpleNamespace(all_checks_passed=False)),
    )
    with pytest.raises(replay_runner.ReplayRunnerError, match="repeat gate failed"):
        replay_runner.run_replay_repeat(
            primary_output_dir=Path("reports/generated/run-primary"),
            repeat_output_dir=Path("reports/generated/run-repeat"),
        )


def test_curation_handoff_includes_health_contrasts_and_exact_scene_log_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = SimpleNamespace(
        plan=SimpleNamespace(plan="frozen"),
        descriptor_aggregates=(_descriptor_row(),),
        persistent_scene_evaluations=("persistent-scenes",),
        persistent_metrics=("persistent-metrics",),
        persistent_crossovers=("crossovers",),
        health_results=("health-results",),
        health_contrasts=("health-contrasts",),
        health_events=("health-events",),
        log_group_ordinals=tuple(f"log-group:{index:02d}" for index in range(10)),
    )
    run = SimpleNamespace(run_id="run:" + "1" * 64)
    artifact = SimpleNamespace(benchmark=benchmark, run=run)
    captured: dict[str, object] = {}
    sentinel = SimpleNamespace(curated=True)
    cpu_checks = 0

    def curate(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    def check_cpu_boundary() -> None:
        nonlocal cpu_checks
        cpu_checks += 1

    monkeypatch.setattr(replay_runner, "curate_replay_evidence", curate)
    monkeypatch.setattr(
        replay_runner,
        "_ensure_cpu_only_import_boundary",
        check_cpu_boundary,
    )
    assert (
        replay_runner._curate_reloaded_local_artifact(
            cast(Any, artifact),
            profile_summary=cast(Any, SimpleNamespace(profile="summary")),
        )
        is sentinel
    )
    assert captured["health_contrasts"] == ("health-contrasts",)
    bindings = cast(tuple[Any, ...], captured["log_group_bindings"])
    assert tuple(row.sequence_id for row in bindings) == tuple(
        f"nuscenes:{scene_name}" for scene_name in replay_runner.M5_SCENE_NAMES
    )
    assert tuple(row.log_group_ordinal for row in bindings) == tuple(
        f"log-group:{index:02d}" for index in range(10)
    )
    assert cpu_checks == 1


def test_pending_publication_has_no_success_until_explicit_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    snapshot = _snapshot(source_root)
    benchmark = _fake_benchmark()
    _install_fake_local_decoder(monkeypatch, benchmark)
    members, counts = _members()
    destination = source_root / "reports/generated/pending"

    pending = replay_runner._write_local_artifact(
        destination=destination,
        snapshot=snapshot,
        benchmark=benchmark,  # type: ignore[arg-type]
        run=_run(snapshot, label="pending", seconds=0),
        resources=_resources(),
        members=members,
        record_counts=counts,
    )

    assert repr(pending) == "_PendingReplayLocalArtifact()"
    assert destination.is_dir()
    assert not (destination / "_SUCCESS").exists()
    with pytest.raises(ArtifactValidationError, match="allowlist"):
        replay_runner.load_replay_local_artifact(destination)

    loaded = replay_runner._finalize_local_artifact(
        pending,
        snapshot=snapshot,
    )
    assert (destination / "_SUCCESS").is_file()
    assert loaded.artifact_sha256 == pending.artifact_sha256
    assert loaded.run_sha256 == pending.run_sha256


@pytest.mark.parametrize("failed_gate", ("source", "resource"))
def test_post_publication_gate_failure_rolls_back_without_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_gate: str,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    real_writer = replay_runner._write_local_artifact
    real_rollback = replay_runner._rollback_pending_local_artifact
    snapshot, benchmark, _, _ = _mock_execute_dependencies(
        source_root=source_root,
        monkeypatch=monkeypatch,
    )
    _install_fake_local_decoder(monkeypatch, benchmark)
    monkeypatch.setattr(replay_runner, "_write_local_artifact", real_writer)
    monkeypatch.setattr(
        replay_runner,
        "_rollback_pending_local_artifact",
        real_rollback,
    )
    monkeypatch.setattr(replay_runner, "_scientific_members", lambda *args, **kwargs: _members())
    destination = source_root / "reports/generated/primary"

    if failed_gate == "source":
        source_checks = 0

        def fail_second_source_check(_: CleanSourceSnapshot) -> None:
            nonlocal source_checks
            source_checks += 1
            if source_checks == 2:
                raise replay_runner.ReplayRunnerError("simulated final source gate failure")

        monkeypatch.setattr(
            replay_runner,
            "_verify_unchanged_source",
            fail_second_source_check,
        )
        expected = "source gate failure"
    else:
        rss_values = iter((128 * 1024**2, 1024**3))
        monkeypatch.setattr(replay_runner, "_peak_rss_bytes", lambda: next(rss_values))
        expected = "post-publication resource gate"

    with pytest.raises(replay_runner.ReplayRunnerError, match=expected):
        replay_runner._execute_replay_local(
            Path("reports/generated/primary"),
            snapshot=snapshot,
        )

    assert not destination.exists()
    assert not (destination / "_SUCCESS").exists()


def test_finalization_failure_removes_success_before_pending_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    snapshot = _snapshot(source_root)
    benchmark = _fake_benchmark()
    _install_fake_local_decoder(monkeypatch, benchmark)
    members, counts = _members()
    destination = source_root / "reports/generated/pending"
    pending = replay_runner._write_local_artifact(
        destination=destination,
        snapshot=snapshot,
        benchmark=benchmark,  # type: ignore[arg-type]
        run=_run(snapshot, label="pending", seconds=0),
        resources=_resources(),
        members=members,
        record_counts=counts,
    )

    def fail_final_reload(*args: object, **kwargs: object) -> Any:
        raise ArtifactValidationError("simulated final reload failure")

    monkeypatch.setattr(
        replay_runner,
        "_load_replay_local_artifact",
        fail_final_reload,
    )
    with pytest.raises(ArtifactValidationError, match="final reload"):
        replay_runner._finalize_local_artifact(
            pending,
            snapshot=snapshot,
        )

    assert destination.is_dir()
    assert not (destination / "_SUCCESS").exists()
    replay_runner._rollback_pending_local_artifact(
        pending,
        snapshot=snapshot,
    )
    assert not destination.exists()


def test_real_typed_row_decoders_reject_non_roundtripping_payloads() -> None:
    descriptor_record = {
        "schema": "ffb.replay-local-descriptor-aggregate/v1",
        "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
        "descriptor_id": "frame-count",
        "population": "nuscenes-mini-replay",
        "population_count": 10,
        "statistic": "median",
        "category_label": None,
        "status": "ok",
        "value": 16.0,
        "unit": "frames",
    }
    descriptor = replay_runner._decode_dataclass_row(
        descriptor_record,
        adapter=replay_runner._DESCRIPTOR_ADAPTER,
        label="descriptor",
    )
    assert descriptor.descriptor_id == "frame-count"

    with pytest.raises(ArtifactValidationError, match="round trip"):
        replay_runner._decode_dataclass_row(
            {**descriptor_record, "ignored_extra": True},
            adapter=replay_runner._DESCRIPTOR_ADAPTER,
            label="descriptor",
        )

    result_record = {
        "schema": "ffb.replay-health-result/v1",
        "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
        "replay_experiment_identity_sha256": "1" * 64,
        "sequence_id": f"nuscenes:{replay_runner.M5_SCENE_NAMES[0]}",
        "condition_id": "replay-clean",
        "condition_selector": "replay-clean:0",
        "method": "camera-only",
        "window": "score",
        "loss_sum_m2": 1.0,
        "valid_object_frame_count": 1,
        "eligible_object_frame_count": 1,
    }
    result = replay_runner._decode_contract_row(
        result_record,
        model=replay_runner.ReplayHealthResultV1,
        label="health-result",
    )
    assert result.condition_selector == "replay-clean:0"
    with pytest.raises(ArtifactValidationError, match="invalid typed row"):
        replay_runner._decode_contract_row(
            {**result_record, "valid_object_frame_count": "1"},
            model=replay_runner.ReplayHealthResultV1,
            label="health-result",
        )


def test_local_benchmark_decoder_dispatches_all_roles_to_authoritative_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = {path: ({},) for path in replay_runner.REPLAY_LOCAL_SCIENTIFIC_PATHS}
    context = replay_runner._ReplayLocalSourceContext(
        run_id="run:" + "1" * 64,
        fit_calibration_sha256=replay_runner.M4_FROZEN_CALIBRATION_SHA256,
        data_master_seed=replay_runner.M5_DATA_MASTER_SEED,
        bootstrap_replicates=replay_runner._BOOTSTRAP_REPLICATES,
        scene_frame_counts=(16,) * 10,
        log_group_ordinals=("log-group:00", "log-group:01") * 5,
    )
    decoded_labels: list[str] = []
    sentinel = SimpleNamespace(authoritative=True)

    def decode_dataclass(
        _row: object,
        *,
        adapter: object,
        label: str,
    ) -> object:
        del adapter
        decoded_labels.append(label)
        return label

    def decode_contract(
        _row: object,
        *,
        model: object,
        label: str,
    ) -> object:
        del model
        decoded_labels.append(label)
        return label

    captured: dict[str, object] = {}

    def build_authoritative(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(replay_runner, "_decode_dataclass_row", decode_dataclass)
    monkeypatch.setattr(replay_runner, "_decode_contract_row", decode_contract)
    monkeypatch.setattr(replay_runner, "ReplayBenchmarkEvidence", build_authoritative)
    assert (
        replay_runner._decode_local_benchmark(
            records,
            context=context,
            plan=cast(Any, "authenticated-plan"),
        )
        is sentinel
    )
    assert set(decoded_labels) == set(replay_runner.REPLAY_LOCAL_SCIENTIFIC_PATHS)
    assert captured["scene_frame_counts"] == (16,) * 10
    assert captured["log_group_ordinals"] == ("log-group:00", "log-group:01") * 5

    def reject_grid(**kwargs: object) -> object:
        del kwargs
        raise ValueError("incomplete authoritative grid")

    monkeypatch.setattr(replay_runner, "ReplayBenchmarkEvidence", reject_grid)
    with pytest.raises(ArtifactValidationError, match="authoritative validation"):
        replay_runner._decode_local_benchmark(
            records,
            context=context,
            plan=cast(Any, "authenticated-plan"),
        )


def test_strict_local_reload_cannot_bypass_typed_member_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    real_decoder = replay_runner._decode_local_benchmark
    artifact = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="primary",
        seconds=0,
    )
    monkeypatch.setattr(replay_runner, "_decode_local_benchmark", real_decoder)

    with pytest.raises(ArtifactValidationError, match="invalid typed row"):
        replay_runner.load_replay_local_artifact(artifact.path)


def test_curation_reloads_bytes_and_rejects_post_execution_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    artifact = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="primary",
        seconds=0,
    )
    monkeypatch.setattr(
        replay_runner,
        "load_replay_plan",
        lambda **_kwargs: artifact.benchmark.plan,
    )
    target = artifact.path / "descriptor-aggregates.ndjson"
    target.write_bytes(target.read_bytes().replace(b'"value":0.0', b'"value":9.0'))
    with pytest.raises(replay_runner.ReplayRunnerError, match="source validation"):
        replay_runner.curate_replay_local_artifact(
            artifact,
            profile_summary=cast(Any, SimpleNamespace(untrusted=True)),
        )


def test_existing_artifact_repeat_verification_uses_two_strict_reloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    snapshot = _snapshot(source_root)
    monkeypatch.setattr(replay_runner, "_initial_snapshot", lambda: snapshot)
    monkeypatch.setattr(replay_runner, "collect_runtime_environment", _environment)
    source_checks: list[CleanSourceSnapshot] = []
    monkeypatch.setattr(
        replay_runner,
        "_verify_unchanged_source",
        source_checks.append,
    )
    primary = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="primary",
        seconds=0,
    )
    repeat = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="repeat",
        seconds=5,
    )
    monkeypatch.setattr(
        replay_runner,
        "load_replay_plan",
        lambda **_kwargs: primary.benchmark.plan,
    )

    verified = replay_runner.verify_replay_repeat_artifacts(
        primary_path=primary.path,
        repeat_path=repeat.path,
    )
    assert repr(verified) == "ReplayLoadedRepeatEvidence()"
    assert verified.repeat_verification.all_checks_passed
    assert verified.primary.path == primary.path
    assert verified.repeat.path == repeat.path
    assert source_checks == [snapshot]

    mismatch = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="mismatch",
        seconds=10,
        changed_path="health-population-metrics.ndjson",
    )
    with pytest.raises(replay_runner.ReplayRunnerError, match="repeat gate failed"):
        replay_runner.verify_replay_repeat_artifacts(
            primary_path=primary.path,
            repeat_path=mismatch.path,
        )


def test_existing_artifact_repeat_verification_rejects_current_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    primary = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="primary",
        seconds=0,
    )
    repeat_artifact = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="repeat",
        seconds=10,
    )
    monkeypatch.setattr(
        replay_runner,
        "load_replay_plan",
        lambda **_kwargs: primary.benchmark.plan,
    )
    drifted = replace(_snapshot(source_root), git_revision="9" * 40)
    monkeypatch.setattr(replay_runner, "_initial_snapshot", lambda: drifted)
    monkeypatch.setattr(replay_runner, "collect_runtime_environment", _environment)

    with pytest.raises(
        replay_runner.ReplayRunnerError,
        match="existing replay artifact verification failed",
    ):
        replay_runner.verify_replay_repeat_artifacts(
            primary_path=primary.path,
            repeat_path=repeat_artifact.path,
        )


def test_current_replay_authority_requires_exact_canonical_output_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    artifact = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="primary",
        seconds=0,
    )
    wrong_command = artifact.run.model_copy(
        update={
            "command": (
                "ffb",
                "replay",
                "run",
                "--output-dir",
                "reports/generated/not-primary",
            )
        }
    )

    with pytest.raises(ValueError, match="current curation authority"):
        replay_runner._require_current_replay_authority(
            replace(artifact, run=wrong_command),
            snapshot=_snapshot(source_root),
            environment=_environment(),
        )


def test_verified_repeat_curation_binds_external_resources_to_loaded_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    snapshot = _snapshot(source_root)
    monkeypatch.setattr(replay_runner, "_initial_snapshot", lambda: snapshot)
    monkeypatch.setattr(replay_runner, "collect_runtime_environment", _environment)
    final_source_checks: list[CleanSourceSnapshot] = []
    monkeypatch.setattr(
        replay_runner,
        "_verify_unchanged_source",
        final_source_checks.append,
    )
    primary = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="primary",
        seconds=0,
    )
    repeat_artifact = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="repeat",
        seconds=10,
    )
    commitments, verification = replay_runner.build_replay_repeat_verification(
        primary,
        repeat_artifact,
    )
    repeat_evidence = replay_runner.ReplayLoadedRepeatEvidence(
        primary=primary,
        repeat=repeat_artifact,
        source_commitments=commitments,
        repeat_verification=verification,
    )
    captured: dict[str, object] = {}
    sentinel = SimpleNamespace(curated=True)

    def verify_existing(**kwargs: object) -> replay_runner.ReplayLoadedRepeatEvidence:
        captured["verify_paths"] = kwargs
        return repeat_evidence

    def import_resources(
        **kwargs: object,
    ) -> tuple[
        replay_runner.ReplayExecutionResourceEvidenceV1,
        replay_runner.ReplayExecutionResourceEvidenceV1,
    ]:
        captured["resource_import"] = kwargs
        primary_binding = cast(replay_runner.ReplayResourceRunBinding, kwargs["primary"])
        repeat_binding = cast(replay_runner.ReplayResourceRunBinding, kwargs["repeat"])
        return (
            _external_resource_evidence(
                run_id=primary.run.run_id,
                run_label="primary",
                elapsed_seconds=20.0,
                peak_rss_bytes=192 * 1024**2,
                local_artifact_sha256=primary.artifact_sha256,
                local_run_sha256=primary.run_sha256,
                persisted_internal_elapsed_seconds=(
                    primary_binding.persisted_internal_elapsed_seconds
                ),
                persisted_internal_peak_rss_bytes=(
                    primary_binding.persisted_internal_peak_rss_bytes
                ),
            ),
            _external_resource_evidence(
                run_id=primary.run.run_id,
                run_label="repeat",
                elapsed_seconds=25.0,
                peak_rss_bytes=256 * 1024**2,
                local_artifact_sha256=repeat_artifact.artifact_sha256,
                local_run_sha256=repeat_artifact.run_sha256,
                persisted_internal_elapsed_seconds=(
                    repeat_binding.persisted_internal_elapsed_seconds
                ),
                persisted_internal_peak_rss_bytes=(
                    repeat_binding.persisted_internal_peak_rss_bytes
                ),
            ),
        )

    def curate_loaded(
        artifact: replay_runner.LoadedReplayLocalArtifact,
        *,
        profile_summary: object,
    ) -> object:
        captured["curation_artifact"] = artifact
        captured["profile"] = profile_summary
        return sentinel

    monkeypatch.setattr(replay_runner, "verify_replay_repeat_artifacts", verify_existing)
    monkeypatch.setattr(
        replay_runner,
        "import_replay_execution_resource_evidence",
        import_resources,
    )
    monkeypatch.setattr(replay_runner, "_curate_reloaded_local_artifact", curate_loaded)

    result = replay_runner.curate_replay_verified_repeat(
        repeat_evidence,
        primary_log_path=Path("primary.time.log"),
        repeat_log_path=Path("repeat.time.log"),
    )

    assert result is sentinel
    resource_import = cast(dict[str, object], captured["resource_import"])
    assert (
        cast(
            replay_runner.ReplayResourceRunBinding,
            resource_import["primary"],
        ).local_artifact_sha256
        == primary.artifact_sha256
    )
    assert (
        cast(
            replay_runner.ReplayResourceRunBinding,
            resource_import["repeat"],
        ).local_run_sha256
        == repeat_artifact.run_sha256
    )
    profile = cast(replay_runner.ReplayProfileSummaryV1, captured["profile"])
    assert profile.elapsed_seconds == 25.0
    assert profile.peak_rss_bytes == 256 * 1024**2
    assert tuple(row.run_label for row in profile.resource_evidence) == (
        "primary",
        "repeat",
    )
    assert captured["curation_artifact"] is primary
    assert final_source_checks == [snapshot]


def test_verified_repeat_curation_rejects_checkout_change_in_verification_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    primary = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="primary",
        seconds=0,
    )
    repeat_artifact = _write(
        source_root,
        monkeypatch=monkeypatch,
        label="repeat",
        seconds=10,
    )
    commitments, verification = replay_runner.build_replay_repeat_verification(
        primary,
        repeat_artifact,
    )
    repeat_evidence = replay_runner.ReplayLoadedRepeatEvidence(
        primary=primary,
        repeat=repeat_artifact,
        source_commitments=commitments,
        repeat_verification=verification,
    )
    checkout_after_mutation = replace(_snapshot(source_root), git_revision="9" * 40)
    monkeypatch.setattr(
        replay_runner,
        "_initial_snapshot",
        lambda: checkout_after_mutation,
    )
    monkeypatch.setattr(replay_runner, "collect_runtime_environment", _environment)
    monkeypatch.setattr(
        replay_runner,
        "verify_replay_repeat_artifacts",
        lambda **_kwargs: repeat_evidence,
    )

    with pytest.raises(
        replay_runner.ReplayRunnerError,
        match="curation repeat authority changed",
    ):
        replay_runner.curate_replay_verified_repeat(
            repeat_evidence,
            primary_log_path=Path("primary.time.log"),
            repeat_log_path=Path("repeat.time.log"),
        )


def test_single_process_guard_blocks_python_spawn_apis_and_restores_them() -> None:
    original_popen = subprocess.Popen
    with replay_runner._single_process_guard():
        assert subprocess.Popen is not original_popen
        with pytest.raises(replay_runner.ReplayRunnerError, match="single-process"):
            subprocess.run(("true",), check=True)
    assert subprocess.Popen is original_popen


def test_scientific_benchmark_spawn_attempt_fails_inside_runner_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    snapshot, _, _, _ = _mock_execute_dependencies(
        source_root=source_root,
        monkeypatch=monkeypatch,
    )

    def spawn_from_benchmark(*args: object, **kwargs: object) -> object:
        del args, kwargs
        subprocess.run(("true",), check=True)
        raise AssertionError("spawn unexpectedly succeeded")

    monkeypatch.setattr(
        replay_runner,
        "run_replay_benchmark",
        spawn_from_benchmark,
    )
    with pytest.raises(replay_runner.ReplayRunnerError, match="single-process"):
        replay_runner._execute_replay_local(
            Path("reports/generated/primary"),
            snapshot=snapshot,
        )


def test_scientific_curation_spawn_attempt_fails_inside_runner_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = SimpleNamespace(
        plan=SimpleNamespace(plan="frozen"),
        descriptor_aggregates=(_descriptor_row(),),
        persistent_scene_evaluations=("persistent-scenes",),
        persistent_metrics=("persistent-metrics",),
        persistent_crossovers=("crossovers",),
        health_results=("health-results",),
        health_contrasts=("health-contrasts",),
        health_events=("health-events",),
        log_group_ordinals=("log-group:00", "log-group:01") * 5,
    )
    artifact = SimpleNamespace(
        benchmark=benchmark,
        run=SimpleNamespace(run_id="run:" + "1" * 64),
    )

    def spawn_from_curation(**kwargs: object) -> object:
        del kwargs
        subprocess.run(("true",), check=True)
        raise AssertionError("spawn unexpectedly succeeded")

    monkeypatch.setattr(replay_runner, "curate_replay_evidence", spawn_from_curation)
    with pytest.raises(replay_runner.ReplayRunnerError, match="single-process"):
        replay_runner._curate_reloaded_local_artifact(
            cast(Any, artifact),
            profile_summary=cast(Any, SimpleNamespace(profile="summary")),
        )


def test_raw_payload_hardlink_alias_attempt_is_counted_and_blocked(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    payload = dataset / "samples/CAM_FRONT/private.jpg"
    alias = dataset / "samples/CAM_FRONT/hardlink-alias"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"private")
    os.link(payload, alias)

    with (
        replay_runner._metadata_read_guard(dataset) as evidence,
        pytest.raises(OSError, match="blocked"),
    ):
        alias.read_bytes()

    assert payload.stat().st_nlink == 2
    assert evidence.blocked_dataset_reads == 1
    assert evidence.raw_sensor_payload_reads == 1
