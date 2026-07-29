from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import fusion_fault_bench.health_runner as health_runner
from fusion_fault_bench.artifacts import derive_run_id
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.health_artifact_v1 import (
    HEALTH_EVAL_ARTIFACT_CONTRACT,
    HEALTH_FIT_ARTIFACT_CONTRACT,
)
from fusion_fault_bench.contracts.health_v1 import (
    M4_HEALTH_INTENT_PATH,
    load_health_benchmark_intent,
)
from fusion_fault_bench.contracts.result_v1alpha1 import RuntimeEnvironment
from fusion_fault_bench.provenance import CleanSourceSnapshot

FIT_OUTPUT = Path("reports/generated/m4-health-fit-test")
EVALUATION_OUTPUT = Path("reports/generated/m4-health-evaluation-test")


def _snapshot(source_root: Path) -> CleanSourceSnapshot:
    git_dir = source_root / ".git"
    return CleanSourceSnapshot(
        source_root=source_root,
        git_revision="a" * 40,
        git_dir=git_dir,
        git_common_dir=git_dir,
        lockfile_sha256=hashlib.sha256(b"lock").hexdigest(),
        package_version="0.1.0",
        manifest_relative_path=M4_HEALTH_INTENT_PATH.as_posix(),
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


def _intent() -> object:
    return load_health_benchmark_intent(source_root=Path.cwd()).intent


def _patch_common(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_root: Path,
) -> tuple[dict[str, Any], object]:
    captured: dict[str, Any] = {"source_checks": 0}
    snapshot = _snapshot(source_root)
    intent = _intent()
    monkeypatch.setattr(health_runner, "_initial_snapshot", lambda _path: snapshot)

    def capture_source_check(*_args: object, **_kwargs: object) -> None:
        captured["source_checks"] += 1

    monkeypatch.setattr(
        health_runner,
        "_verify_unchanged_source",
        capture_source_check,
    )
    monkeypatch.setattr(health_runner, "collect_runtime_environment", _environment)
    return captured, intent


def test_fit_runner_builds_content_addressed_no_overwrite_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured, intent = _patch_common(monkeypatch, source_root=tmp_path)
    validation = SimpleNamespace(all_checks_passed=True)
    fit = SimpleNamespace(
        intent=intent,
        profiles=SimpleNamespace(main_profile=object(), edge_profile=object()),
        calibration=object(),
        candidates=(object(),),
        summary=object(),
        validation=validation,
    )
    monkeypatch.setattr(
        health_runner,
        "fit_health_benchmark",
        lambda **_kwargs: fit,
    )
    artifact = object()

    def capture_write(request: object, destination: Path, **kwargs: object) -> object:
        captured["request"] = request
        captured["destination"] = destination
        captured["write_kwargs"] = kwargs
        return artifact

    monkeypatch.setattr(health_runner, "write_health_fit_artifact", capture_write)

    result = health_runner.fit_health_benchmark_artifact(output_dir=FIT_OUTPUT)

    assert result is artifact
    request = captured["request"]
    assert request.intent is intent
    assert request.calibration is fit.calibration
    assert request.validation is validation
    assert request.run.command == (
        "ffb",
        "health",
        "fit",
        "--output-dir",
        FIT_OUTPUT.as_posix(),
    )
    assert request.run.run_id == derive_run_id(
        manifest_sha256=sha256_digest(intent),
        git_revision="a" * 40,
        lockfile_sha256=_snapshot(tmp_path).lockfile_sha256,
        package_version="0.1.0",
        artifact_contract=HEALTH_FIT_ARTIFACT_CONTRACT,
    )
    assert captured["destination"] == tmp_path / FIT_OUTPUT
    assert captured["write_kwargs"] == {
        "source_root": tmp_path,
        "git_metadata_dirs": (tmp_path / ".git", tmp_path / ".git"),
    }
    assert captured["source_checks"] == 2


def test_evaluation_runner_uses_only_authenticated_fit_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured, intent = _patch_common(monkeypatch, source_root=tmp_path)
    fit = SimpleNamespace(
        path=tmp_path / FIT_OUTPUT,
        intent=intent,
        main_profile=object(),
        edge_profile=object(),
        summary=SimpleNamespace(
            selected_candidate_index=0,
            selected_self_threshold=0.95,
            selected_cross_threshold=0.95,
        ),
        artifact_sha256="b" * 64,
        run_sha256="c" * 64,
    )
    monkeypatch.setattr(
        health_runner,
        "load_health_fit_artifact",
        lambda path: captured.setdefault("loaded_fit_path", path) and fit,
    )
    validation = SimpleNamespace(all_checks_passed=True)
    evaluation = SimpleNamespace(
        validation=validation,
    )
    batches = (object(), object())

    def capture_evaluate(
        *,
        fit_artifact: object,
        condition_sink: Any,
    ) -> object:
        captured["evaluated_fit"] = fit_artifact
        for batch in batches:
            condition_sink(batch)
        return evaluation

    monkeypatch.setattr(
        health_runner,
        "stream_health_benchmark_test",
        capture_evaluate,
    )
    artifact = object()

    class FakeTransaction:
        def __init__(
            self,
            destination: Path,
            **kwargs: object,
        ) -> None:
            captured["destination"] = destination
            captured["transaction_kwargs"] = kwargs
            captured["batches"] = []

        def __enter__(self) -> FakeTransaction:
            captured["transaction_entered"] = True
            return self

        def __exit__(self, *_args: object) -> None:
            captured["transaction_exited"] = True

        def append_condition(self, batch: object) -> None:
            captured["batches"].append(batch)

        def finalize(self, *, validation: object, run: object) -> object:
            captured["validation"] = validation
            captured["run"] = run
            return artifact

    monkeypatch.setattr(health_runner, "HealthEvaluationArtifactTransaction", FakeTransaction)

    result = health_runner.evaluate_health_benchmark_artifact(
        FIT_OUTPUT,
        output_dir=EVALUATION_OUTPUT,
    )

    assert result is artifact
    assert captured["loaded_fit_path"] == tmp_path / FIT_OUTPUT
    assert captured["evaluated_fit"] is fit
    assert captured["batches"] == list(batches)
    assert captured["validation"] is validation
    run = captured["run"]
    assert run.command == (
        "ffb",
        "health",
        "evaluate",
        FIT_OUTPUT.as_posix(),
        "--output-dir",
        EVALUATION_OUTPUT.as_posix(),
    )
    assert run.run_id == derive_run_id(
        manifest_sha256=sha256_digest(intent),
        git_revision="a" * 40,
        lockfile_sha256=_snapshot(tmp_path).lockfile_sha256,
        package_version="0.1.0",
        artifact_contract=HEALTH_EVAL_ARTIFACT_CONTRACT,
    )
    assert captured["destination"] == tmp_path / EVALUATION_OUTPUT
    assert captured["transaction_kwargs"] == {
        "fit_artifact": fit,
        "source_root": tmp_path,
        "git_metadata_dirs": (tmp_path / ".git", tmp_path / ".git"),
    }
    assert captured["transaction_entered"] is True
    assert captured["transaction_exited"] is True
    assert captured["source_checks"] == 2


@pytest.mark.parametrize(
    "destination",
    (
        Path("/tmp/m4"),
        Path("reports/generated"),
        Path("outputs/m4"),
        Path("reports/generated/m4/.."),
    ),
)
def test_runner_rejects_noncanonical_generated_paths(
    destination: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(health_runner.HealthRunnerError):
        health_runner._validated_generated_path(
            destination,
            source_root=tmp_path,
            label="output",
        )


def test_runner_rejects_symlinked_generated_path_component(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "generated").symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        health_runner.HealthRunnerError,
        match="must not use symlinks",
    ):
        health_runner._validated_generated_path(
            FIT_OUTPUT,
            source_root=tmp_path,
            label="fit output",
        )


def test_fit_runner_rejects_existing_destination_before_computation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common(monkeypatch, source_root=tmp_path)
    (tmp_path / FIT_OUTPUT).mkdir(parents=True)
    called = False

    def unexpected_fit(**_kwargs: object) -> object:
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(health_runner, "fit_health_benchmark", unexpected_fit)

    with pytest.raises(FileExistsError, match="already exists"):
        health_runner.fit_health_benchmark_artifact(output_dir=FIT_OUTPUT)
    assert not called


def test_evaluation_runner_rejects_nested_or_existing_destinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common(monkeypatch, source_root=tmp_path)
    with pytest.raises(health_runner.HealthRunnerError, match="disjoint"):
        health_runner.evaluate_health_benchmark_artifact(
            FIT_OUTPUT,
            output_dir=FIT_OUTPUT / "evaluation",
        )

    (tmp_path / EVALUATION_OUTPUT).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exists"):
        health_runner.evaluate_health_benchmark_artifact(
            FIT_OUTPUT,
            output_dir=EVALUATION_OUTPUT,
        )


def test_runners_refuse_failed_scientific_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _captured, intent = _patch_common(monkeypatch, source_root=tmp_path)
    failed = SimpleNamespace(all_checks_passed=False)
    fit = SimpleNamespace(
        intent=intent,
        profiles=SimpleNamespace(main_profile=object(), edge_profile=object()),
        calibration=object(),
        candidates=(object(),),
        summary=object(),
        validation=failed,
    )
    monkeypatch.setattr(health_runner, "fit_health_benchmark", lambda **_kwargs: fit)
    with pytest.raises(
        health_runner.ArtifactValidationError,
        match="fit failed its release gates",
    ):
        health_runner.fit_health_benchmark_artifact(output_dir=FIT_OUTPUT)

    loaded_fit = SimpleNamespace(intent=intent)
    monkeypatch.setattr(
        health_runner,
        "load_health_fit_artifact",
        lambda _path: loaded_fit,
    )
    monkeypatch.setattr(
        health_runner,
        "stream_health_benchmark_test",
        lambda **_kwargs: SimpleNamespace(validation=failed),
    )

    class FailedTransaction:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> FailedTransaction:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def append_condition(self, _batch: object) -> None:
            pass

        def finalize(self, **_kwargs: object) -> object:
            raise AssertionError("failed validation must not publish")

    monkeypatch.setattr(
        health_runner,
        "HealthEvaluationArtifactTransaction",
        FailedTransaction,
    )
    with pytest.raises(
        health_runner.ArtifactValidationError,
        match="evaluation failed its release gates",
    ):
        health_runner.evaluate_health_benchmark_artifact(
            FIT_OUTPUT,
            output_dir=EVALUATION_OUTPUT,
        )


def test_snapshot_helpers_fail_closed_and_sanitize_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        health_runner,
        "discover_clean_source",
        lambda _path: (_ for _ in ()).throw(ValueError("private detail")),
    )
    with pytest.raises(
        health_runner.HealthRunnerError,
        match="clean-source validation failed",
    ):
        health_runner._initial_snapshot(M4_HEALTH_INTENT_PATH)

    initial = _snapshot(tmp_path)
    verified: list[CleanSourceSnapshot] = []
    monkeypatch.setattr(health_runner, "discover_clean_source", lambda _path: initial)
    monkeypatch.setattr(
        health_runner,
        "verify_locked_execution",
        lambda snapshot: verified.append(snapshot),
    )
    assert health_runner._initial_snapshot(M4_HEALTH_INTENT_PATH) == initial
    health_runner._verify_unchanged_source(
        M4_HEALTH_INTENT_PATH,
        initial=initial,
    )
    assert verified == [initial, initial]

    changed = replace(initial, git_revision="b" * 40)
    monkeypatch.setattr(health_runner, "discover_clean_source", lambda _path: changed)
    monkeypatch.setattr(health_runner, "verify_locked_execution", lambda _snapshot: None)
    with pytest.raises(
        health_runner.HealthRunnerError,
        match="source provenance changed",
    ):
        health_runner._verify_unchanged_source(
            M4_HEALTH_INTENT_PATH,
            initial=initial,
        )

    monkeypatch.setattr(health_runner, "discover_clean_source", lambda _path: initial)
    monkeypatch.setattr(
        health_runner,
        "verify_locked_execution",
        lambda _snapshot: (_ for _ in ()).throw(ValueError("private detail")),
    )
    with pytest.raises(
        health_runner.HealthRunnerError,
        match="final clean-source validation failed",
    ):
        health_runner._verify_unchanged_source(
            M4_HEALTH_INTENT_PATH,
            initial=initial,
        )


def test_runner_rejects_snapshot_for_any_other_source(
    tmp_path: Path,
) -> None:
    with pytest.raises(health_runner.HealthRunnerError, match="frozen intent"):
        health_runner._require_frozen_intent_snapshot(
            replace(
                _snapshot(tmp_path),
                manifest_relative_path="examples/health/not-frozen.json",
            )
        )
