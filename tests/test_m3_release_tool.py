from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import fusion_fault_bench.contracts.procedural_release_v1 as release_contract
from fusion_fault_bench.contracts.matrix_v1 import (
    M3_CI_SMOKE_MATRIX_SHA256,
    M3_SMOKE_MANIFEST_SHA256S,
    load_experiment_matrix,
)
from fusion_fault_bench.contracts.procedural_artifact_v1 import (
    PROCEDURAL_INDEXED_PAYLOAD_PATHS,
)
from fusion_fault_bench.contracts.procedural_profile_v1 import SMOKE_PROFILE_SHA256
from fusion_fault_bench.contracts.procedural_release_v1 import (
    IndexedMemberDigestPairV1,
    M3MatrixArtifactEvidenceV1,
    M3MatrixValidationV1,
    RepeatRunMeasurementV1,
    RepeatVerificationV1,
    SmokeIdentityComparisonV1,
)
from fusion_fault_bench.procedural_release import RepeatRunResources
from fusion_fault_bench.provenance import discover_clean_source, verify_locked_execution

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
m3_release = importlib.import_module("tools.m3_release")
SMOKE_MATRIX_PATH = Path("examples/matrices/m3-ci-smoke-v1.json")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _smoke_evidence() -> tuple[M3MatrixValidationV1, RepeatVerificationV1]:
    artifact = M3MatrixArtifactEvidenceV1(
        execution_index=0,
        experiment="procedural-ci-smoke",
        manifest_sha256=M3_SMOKE_MANIFEST_SHA256S[0],
        profile_id="constant-velocity-ci-smoke-v1",
        profile_sha256=SMOKE_PROFILE_SHA256,
        artifact_sha256=_digest("smoke-artifact"),
    )
    artifact_set_sha256 = release_contract._artifact_set_digest(
        M3_CI_SMOKE_MATRIX_SHA256,
        (artifact,),
    )
    matrix = M3MatrixValidationV1(
        schema="ffb.m3-matrix-validation/v1",
        matrix_id="m3-ci-smoke-v1",
        matrix_sha256=M3_CI_SMOKE_MATRIX_SHA256,
        artifact_count=1,
        artifact_set_sha256=artifact_set_sha256,
        ordered_artifacts=(artifact,),
        identity_comparison=SmokeIdentityComparisonV1(
            status="not-applicable-single-manifest-smoke",
            profile_id="constant-velocity-ci-smoke-v1",
            experiment="procedural-ci-smoke",
            manifest_sha256=M3_SMOKE_MANIFEST_SHA256S[0],
            artifact_count=1,
            comparison_count=0,
            reason="single-manifest-smoke-has-no-cross-manifest-peer",
        ),
        all_checks_passed=True,
    )
    pairs = tuple(
        IndexedMemberDigestPairV1(
            execution_index=0,
            experiment="procedural-ci-smoke",
            manifest_sha256=M3_SMOKE_MANIFEST_SHA256S[0],
            path=path,
            first_sha256=_digest(f"member:{path}"),
            second_sha256=_digest(f"member:{path}"),
            equal=True,
        )
        for path in PROCEDURAL_INDEXED_PAYLOAD_PATHS
    )
    repeat = RepeatVerificationV1(
        schema="ffb.repeat-verification/v1",
        matrix_id="m3-ci-smoke-v1",
        matrix_sha256=M3_CI_SMOKE_MATRIX_SHA256,
        artifact_count=1,
        first_run=RepeatRunMeasurementV1(
            artifact_set_sha256=artifact_set_sha256,
            run_record_sha256s=(_digest("first-run-record"),),
            cpu_model="Test CPU",
            wall_time_seconds=1.0,
            peak_memory_bytes=1_000,
        ),
        second_run=RepeatRunMeasurementV1(
            artifact_set_sha256=artifact_set_sha256,
            run_record_sha256s=(_digest("second-run-record"),),
            cpu_model="Test CPU",
            wall_time_seconds=1.1,
            peak_memory_bytes=1_100,
        ),
        indexed_member_pairs=pairs,
        comparison_count=6,
        mismatch_count=0,
        resource_measurement_scope=(
            "self-reported-by-tracked-wait4-driver-not-independently-recomputable"
        ),
        execution_evidence_scope=(
            "distinct-path-inode-and-run-record-consistency-not-cryptographic-proof"
        ),
        same_named_cpu=True,
        all_equal=True,
        all_checks_passed=True,
    )
    return matrix, repeat


def test_repeat_evidence_writer_loader_and_allowlist(tmp_path: Path) -> None:
    matrix, repeat = _smoke_evidence()
    destination = tmp_path / "evidence"

    written = m3_release._write_m3_repeat_evidence(
        destination,
        matrix_validation=matrix,
        repeat_verification=repeat,
    )

    assert written.matrix_validation == matrix
    assert written.repeat_verification == repeat
    assert m3_release.load_m3_repeat_evidence(destination) == written
    linked = tmp_path / "linked-evidence"
    linked.symlink_to(destination, target_is_directory=True)
    with pytest.raises(
        m3_release.ProceduralReleaseDriverError,
        match="symlink components",
    ):
        m3_release.load_m3_repeat_evidence(linked)
    with pytest.raises(FileExistsError):
        m3_release._write_m3_repeat_evidence(
            destination,
            matrix_validation=matrix,
            repeat_verification=repeat,
        )

    unexpected = destination / "unexpected.json"
    unexpected.write_bytes(b"{}")
    with pytest.raises(
        m3_release.ProceduralReleaseDriverError,
        match="allowlist",
    ):
        m3_release.load_m3_repeat_evidence(destination)
    unexpected.unlink()

    matrix_path = destination / m3_release.M3_MATRIX_VALIDATION_FILE
    matrix_path.write_bytes(matrix_path.read_bytes() + b"\n")
    with pytest.raises(
        m3_release.ProceduralReleaseDriverError,
        match="canonical JSON",
    ):
        m3_release.load_m3_repeat_evidence(destination)


def test_repeat_execute_and_validate_orchestration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix, repeat = _smoke_evidence()
    loaded_matrix = load_experiment_matrix(
        SMOKE_MATRIX_PATH,
        source_root=REPOSITORY_ROOT,
    )
    loaded_matrix = replace(
        loaded_matrix,
        path=tmp_path / SMOKE_MATRIX_PATH,
    )
    snapshot = SimpleNamespace(
        source_root=tmp_path,
        manifest_relative_path=SMOKE_MATRIX_PATH.as_posix(),
    )
    monkeypatch.setattr(m3_release, "_initial_snapshot", lambda _path: snapshot)
    monkeypatch.setattr(
        m3_release,
        "_load_matrix_for_snapshot",
        lambda *_args, **_kwargs: loaded_matrix,
    )
    monkeypatch.setattr(
        m3_release,
        "_verify_unchanged_source",
        lambda *_args, **_kwargs: None,
    )
    measured_calls: list[str] = []

    def fake_measured_child(**kwargs: Any) -> RepeatRunResources:
        measured_calls.append(cast(str, kwargs["output_relative_path"]))
        return RepeatRunResources(
            wall_time_seconds=float(len(measured_calls)),
            peak_memory_bytes=1_000 * len(measured_calls),
        )

    monkeypatch.setattr(m3_release, "_run_measured_matrix_child", fake_measured_child)
    monkeypatch.setattr(
        m3_release,
        "build_m3_repeat_evidence",
        lambda *_args, **_kwargs: (matrix, repeat),
    )
    first = Path("reports/generated/test-first")
    second = Path("reports/generated/test-second")
    evidence_path = Path("reports/generated/test-evidence")

    executed = m3_release.execute_procedural_repeat(
        SMOKE_MATRIX_PATH,
        first_output_dir=first,
        second_output_dir=second,
        evidence_dir=evidence_path,
    )
    validated = m3_release.validate_procedural_repeat(
        SMOKE_MATRIX_PATH,
        first_output_dir=first,
        second_output_dir=second,
        evidence_dir=evidence_path,
    )

    assert executed == validated
    assert measured_calls == [first.as_posix(), second.as_posix()]

    monkeypatch.setattr(
        m3_release,
        "build_m3_repeat_evidence",
        lambda *_args, **_kwargs: (
            matrix.model_copy(update={"all_checks_passed": False}),
            repeat,
        ),
    )
    with pytest.raises(
        m3_release.ProceduralReleaseDriverError,
        match="disagrees",
    ):
        m3_release.validate_procedural_repeat(
            SMOKE_MATRIX_PATH,
            first_output_dir=first,
            second_output_dir=second,
            evidence_dir=evidence_path,
        )


def test_measured_child_resources_and_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = SimpleNamespace(pid=123, returncode=None)
    monkeypatch.setattr(
        m3_release.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    ticks = iter((10.0, 12.5))
    monkeypatch.setattr(m3_release.time, "perf_counter", lambda: next(ticks))
    usage = SimpleNamespace(ru_maxrss=2_000)
    monkeypatch.setattr(
        m3_release.os,
        "wait4",
        lambda _pid, _options: (123, 0, usage),
    )

    measured = m3_release._run_measured_matrix_child(
        source_root=tmp_path,
        matrix_relative_path=SMOKE_MATRIX_PATH.as_posix(),
        output_relative_path="reports/generated/test",
    )

    assert measured.wall_time_seconds == 2.5
    assert measured.peak_memory_bytes > 0
    assert process.returncode == 0

    ticks = iter((20.0, 21.0))
    monkeypatch.setattr(m3_release.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(
        m3_release.os,
        "wait4",
        lambda _pid, _options: (123, 1 << 8, usage),
    )
    with pytest.raises(
        m3_release.ProceduralReleaseDriverError,
        match="did not complete",
    ):
        m3_release._run_measured_matrix_child(
            source_root=tmp_path,
            matrix_relative_path=SMOKE_MATRIX_PATH.as_posix(),
            output_relative_path="reports/generated/test",
        )


def test_real_two_run_smoke_when_checkout_is_clean() -> None:
    try:
        snapshot = discover_clean_source(REPOSITORY_ROOT / SMOKE_MATRIX_PATH)
        verify_locked_execution(snapshot)
    except (OSError, ValueError):
        pytest.skip("real repeat integration requires a clean locked checkout")

    relative_base = Path(f"reports/generated/pytest-m3-repeat-{os.getpid()}")
    absolute_base = REPOSITORY_ROOT / relative_base
    first = relative_base / "first"
    second = relative_base / "second"
    evidence = relative_base / "evidence"
    if os.path.lexists(absolute_base):
        pytest.fail("real repeat integration destination unexpectedly exists")
    try:
        executed = m3_release.execute_procedural_repeat(
            SMOKE_MATRIX_PATH,
            first_output_dir=first,
            second_output_dir=second,
            evidence_dir=evidence,
        )
        validated = m3_release.validate_procedural_repeat(
            SMOKE_MATRIX_PATH,
            first_output_dir=first,
            second_output_dir=second,
            evidence_dir=evidence,
        )
        assert executed == validated
        assert validated.repeat_verification.comparison_count == 6
        assert validated.repeat_verification.all_checks_passed
    finally:
        shutil.rmtree(absolute_base, ignore_errors=True)
