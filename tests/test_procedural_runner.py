from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import fusion_fault_bench.procedural_runner as procedural_runner
from fusion_fault_bench.contracts.io import load_manifest
from fusion_fault_bench.contracts.procedural_profile_v1 import (
    load_procedural_profile,
)
from fusion_fault_bench.contracts.result_v1alpha1 import RuntimeEnvironment
from fusion_fault_bench.provenance import CleanSourceSnapshot

MATRIX_PATH = Path("examples/matrices/m3-ci-smoke-v1.json")
MANIFEST_PATH = Path("examples/manifests/procedural-ci-smoke-v1alpha1.json")
PROFILE_PATH = Path("examples/profiles/constant-velocity-ci-smoke-v1.json")
OUTPUT_PATH = Path("reports/generated/m3-ci-smoke-test")


def _snapshot(source_root: Path) -> CleanSourceSnapshot:
    git_dir = source_root / ".git"
    return CleanSourceSnapshot(
        source_root=source_root,
        git_revision="a" * 40,
        git_dir=git_dir,
        git_common_dir=git_dir,
        lockfile_sha256=hashlib.sha256(b"lock").hexdigest(),
        package_version="0.1.0",
        manifest_relative_path=MATRIX_PATH.as_posix(),
    )


def _environment() -> RuntimeEnvironment:
    return RuntimeEnvironment(
        python_version="3.12.13",
        os_name="TestOS",
        os_release="1.0",
        machine="test-machine",
        cpu_model="test CPU",
        logical_cpu_count=4,
        memory_bytes=8_000_000_000,
    )


def _patch_matrix_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_root: Path,
    validation_passed: bool = True,
) -> dict[str, Any]:
    manifest = load_manifest(MANIFEST_PATH)
    profile = load_procedural_profile(PROFILE_PATH)
    snapshot = _snapshot(source_root)
    loaded = SimpleNamespace(
        path=source_root / MATRIX_PATH,
        manifests=(manifest,),
        profiles=(profile,),
    )
    captured: dict[str, Any] = {"snapshot": snapshot}
    monkeypatch.setattr(procedural_runner, "_initial_snapshot", lambda _path: snapshot)
    monkeypatch.setattr(
        procedural_runner,
        "_verify_unchanged_source",
        lambda *_args, **_kwargs: (
            captured.setdefault("source_checks", 0) or captured.update(source_checks=1)
        ),
    )
    monkeypatch.setattr(
        procedural_runner,
        "load_experiment_matrix",
        lambda *_args, **_kwargs: loaded,
    )
    monkeypatch.setattr(procedural_runner, "collect_runtime_environment", _environment)
    sequences = (object(),)
    metrics = (object(),)
    evaluated = SimpleNamespace(metrics=metrics, aggregates=(object(),), crossovers=())
    validation = SimpleNamespace(all_checks_passed=validation_passed)
    monkeypatch.setattr(
        procedural_runner,
        "generate_procedural_sequences",
        lambda *_args, **_kwargs: sequences,
    )
    monkeypatch.setattr(
        procedural_runner,
        "generate_procedural_sequence_metrics",
        lambda *_args, **_kwargs: metrics,
    )
    monkeypatch.setattr(
        procedural_runner,
        "evaluate_procedural_records",
        lambda *_args, **_kwargs: evaluated,
    )
    monkeypatch.setattr(
        procedural_runner,
        "build_procedural_validation",
        lambda *_args, **_kwargs: validation,
    )
    monkeypatch.setattr(
        procedural_runner,
        "validate_evaluated_procedural_records",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        procedural_runner,
        "build_m3_matrix_validation",
        lambda *_args, **_kwargs: SimpleNamespace(all_checks_passed=True),
    )
    artifact = object()

    def capture_write(request, destination, **kwargs):
        captured["request"] = request
        captured["destination"] = destination
        captured["write_kwargs"] = kwargs
        return artifact

    monkeypatch.setattr(
        procedural_runner,
        "write_procedural_artifact",
        capture_write,
    )
    captured["artifact"] = artifact
    captured["manifest"] = manifest
    captured["profile"] = profile
    captured["sequences"] = sequences
    captured["metrics"] = metrics
    captured["validation"] = validation
    return captured


def test_matrix_runner_builds_one_validated_no_overwrite_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_matrix_run(monkeypatch, source_root=tmp_path)

    artifacts = procedural_runner.run_procedural_matrix(
        MATRIX_PATH,
        output_dir=OUTPUT_PATH,
    )

    assert artifacts == (captured["artifact"],)
    request = captured["request"]
    assert request.manifest is captured["manifest"]
    assert request.profile is captured["profile"]
    assert request.metrics is captured["metrics"]
    assert request.validation is captured["validation"]
    assert request.run.command == (
        "ffb",
        "procedural",
        "matrix",
        "run",
        MATRIX_PATH.as_posix(),
        "--output-dir",
        OUTPUT_PATH.as_posix(),
    )
    assert request.run.source_dirty is False
    assert captured["destination"] == (tmp_path / OUTPUT_PATH / "procedural-ci-smoke")
    assert captured["write_kwargs"]["source_root"] == tmp_path


@pytest.mark.parametrize(
    "destination",
    (
        Path("/tmp/m3"),
        Path("reports/generated"),
        Path("outputs/m3"),
        Path("reports/generated/m3/.."),
    ),
)
def test_matrix_runner_rejects_noncanonical_output_roots(
    destination: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(procedural_runner.ProceduralRunnerError):
        procedural_runner._validated_output_root(
            destination,
            source_root=tmp_path,
        )


def test_matrix_runner_rejects_existing_destination_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_matrix_run(monkeypatch, source_root=tmp_path)
    destination = tmp_path / OUTPUT_PATH / "procedural-ci-smoke"
    destination.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        procedural_runner.run_procedural_matrix(
            MATRIX_PATH,
            output_dir=OUTPUT_PATH,
        )

    assert "request" not in captured


def test_matrix_runner_refuses_failed_scientific_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_matrix_run(
        monkeypatch,
        source_root=tmp_path,
        validation_passed=False,
    )

    with pytest.raises(
        procedural_runner.ArtifactValidationError,
        match="failed its release gates",
    ):
        procedural_runner.run_procedural_matrix(
            MATRIX_PATH,
            output_dir=OUTPUT_PATH,
        )

    assert "request" not in captured


def test_matrix_runner_rejects_symlink_output_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    output = tmp_path / OUTPUT_PATH
    output.parent.mkdir(parents=True)
    output.symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        procedural_runner.ProceduralRunnerError,
        match="must not be a symlink",
    ):
        procedural_runner._validated_output_root(
            OUTPUT_PATH,
            source_root=tmp_path,
        )


def test_snapshot_helpers_sanitize_discovery_and_change_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        procedural_runner,
        "discover_clean_source",
        lambda _path: (_ for _ in ()).throw(ValueError("private detail")),
    )
    with pytest.raises(
        procedural_runner.ProceduralRunnerError,
        match="clean-source validation failed",
    ):
        procedural_runner._initial_snapshot(MATRIX_PATH)

    initial = _snapshot(tmp_path)
    changed = replace(initial, git_revision="b" * 40)
    monkeypatch.setattr(
        procedural_runner,
        "discover_clean_source",
        lambda _path: changed,
    )
    monkeypatch.setattr(
        procedural_runner,
        "verify_locked_execution",
        lambda _snapshot: None,
    )
    with pytest.raises(
        procedural_runner.ProceduralRunnerError,
        match="source provenance changed",
    ):
        procedural_runner._verify_unchanged_source(MATRIX_PATH, initial=initial)


def test_snapshot_final_check_sanitizes_locked_environment_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _snapshot(tmp_path)
    monkeypatch.setattr(
        procedural_runner,
        "discover_clean_source",
        lambda _path: initial,
    )
    monkeypatch.setattr(
        procedural_runner,
        "verify_locked_execution",
        lambda _snapshot: (_ for _ in ()).throw(ValueError("private detail")),
    )

    with pytest.raises(
        procedural_runner.ProceduralRunnerError,
        match="final clean-source validation failed",
    ):
        procedural_runner._verify_unchanged_source(MATRIX_PATH, initial=initial)


def test_manifest_and_matrix_identity_guards_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analytic = load_manifest(Path("examples/manifests/analytic-bias-v1alpha1.json"))
    with pytest.raises(
        procedural_runner.ProceduralRunnerError,
        match="unsupported manifest",
    ):
        procedural_runner._procedural_manifest(analytic)

    captured = _patch_matrix_run(monkeypatch, source_root=tmp_path)
    snapshot = captured["snapshot"]
    monkeypatch.setattr(
        procedural_runner,
        "_initial_snapshot",
        lambda _path: replace(
            snapshot,
            manifest_relative_path="examples/matrices/wrong.json",
        ),
    )
    with pytest.raises(
        procedural_runner.ProceduralRunnerError,
        match="does not identify",
    ):
        procedural_runner.run_procedural_matrix(
            MATRIX_PATH,
            output_dir=OUTPUT_PATH,
        )


def test_matrix_level_failure_is_not_reported_as_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_matrix_run(monkeypatch, source_root=tmp_path)
    monkeypatch.setattr(
        procedural_runner,
        "build_m3_matrix_validation",
        lambda *_args, **_kwargs: SimpleNamespace(all_checks_passed=False),
    )

    with pytest.raises(
        procedural_runner.ArtifactValidationError,
        match="matrix-level validation failed",
    ):
        procedural_runner.run_procedural_matrix(
            MATRIX_PATH,
            output_dir=OUTPUT_PATH,
        )
