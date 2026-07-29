# pyright: reportPrivateUsage=false

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import fusion_fault_bench.health_test_benchmark as benchmark
from fusion_fault_bench.artifacts import ArtifactValidationError
from fusion_fault_bench.contracts.health_v1 import load_health_benchmark_intent
from fusion_fault_bench.contracts.procedural_profile_v1 import load_procedural_profile
from fusion_fault_bench.health import HealthCalibration, HealthThresholds
from fusion_fault_bench.health_artifacts import LoadedHealthFitArtifact
from fusion_fault_bench.health_benchmark import expand_test_cases
from fusion_fault_bench.health_test_benchmark import (
    HealthBenchmarkEvaluation,
    HealthBenchmarkStreamSummary,
    HealthConditionEvaluation,
)
from fusion_fault_bench.scenarios.health import (
    HealthBaseSequence,
    HealthFaultSpec,
    generate_health_base_sequences,
)


@dataclass(frozen=True)
class _Case:
    condition_id: str
    population: str
    fault: HealthFaultSpec


def _calibration() -> HealthCalibration:
    values = np.asarray([0.0, 1e12], dtype=np.float64)
    return HealthCalibration(
        camera_self_mean=values,
        camera_self_maximum=values,
        lidar_self_mean=values,
        lidar_self_maximum=values,
        camera_from_lidar_cross_mean=values,
        camera_from_lidar_cross_maximum=values,
        lidar_from_camera_cross_mean=values,
        lidar_from_camera_cross_maximum=values,
    )


def _cases() -> tuple[_Case, ...]:
    return (
        _Case(
            "identity",
            "main-test",
            HealthFaultSpec(
                family="identity",
                target="none",
                axis="none",
                unit="identity",
                value=0.0,
            ),
        ),
        _Case(
            "dropout",
            "main-test",
            HealthFaultSpec(
                family="dropout",
                target="camera",
                axis="availability",
                unit="probability",
                value=1.0,
            ),
        ),
        _Case(
            "yaw",
            "main-test",
            HealthFaultSpec(
                family="calibration-yaw",
                target="camera",
                axis="yaw",
                unit="rad",
                value=0.06,
            ),
        ),
        _Case(
            "motion",
            "main-test",
            HealthFaultSpec(
                family="clean-predictor-mismatch",
                target="none",
                axis="motion",
                unit="m/s^2",
                value=8.0,
            ),
        ),
        _Case(
            "common",
            "edge-test",
            HealthFaultSpec(
                family="common-mode-position-bias",
                target="both",
                axis="x",
                unit="m",
                value=1.0,
            ),
        ),
    )


def _bases():
    main = load_procedural_profile(Path("examples/profiles/constant-velocity-front-roi-v1.json"))
    edge = load_procedural_profile(Path("examples/profiles/constant-velocity-fov-edge-v1.json"))
    return (
        generate_health_base_sequences(
            main,
            split="test",
            sequence_count=2,
            data_master_seed=1729,
        ),
        generate_health_base_sequences(
            edge,
            split="test",
            sequence_count=2,
            data_master_seed=1729,
        ),
    )


def test_apply_only_orchestration_streams_cases_and_builds_validation() -> None:
    main_bases, edge_bases = _bases()
    result = benchmark._evaluate_health_cases(
        cases=_cases(),
        main_bases=main_bases,
        edge_bases=edge_bases,
        calibration=_calibration(),
        thresholds=HealthThresholds(self_score=1.0, cross_score=1.0),
    )
    assert result.evaluated_sequence_condition_count == 10
    assert len(result.condition_ids) == 5
    assert result.validation.all_checks_passed
    assert len(result.sequence_events) == 40
    assert len(result.sequence_contrasts) == 150
    assert not any(
        row.condition_id == "common"
        and row.metric_name in {"attribution-fraction", "event-outcome-correct-fraction"}
        for row in result.aggregates
    )


def test_condition_sink_matches_materialized_rows_without_global_row_retention() -> None:
    main_bases, edge_bases = _bases()
    batches: list[HealthConditionEvaluation] = []
    summary = benchmark._evaluate_health_cases(
        cases=tuple(reversed(_cases())),
        main_bases=main_bases,
        edge_bases=edge_bases,
        calibration=_calibration(),
        thresholds=HealthThresholds(self_score=1.0, cross_score=1.0),
        condition_sink=batches.append,
        materialize_rows=False,
    )
    assert isinstance(summary, HealthBenchmarkStreamSummary)
    assert summary.validation.all_checks_passed
    assert summary.condition_ids == tuple(sorted(case.condition_id for case in _cases()))
    assert tuple(batch.condition_id for batch in batches) == summary.condition_ids
    assert summary.evaluated_sequence_condition_count == 10
    for batch in batches:
        assert batch.sequence_ids == tuple(sorted(batch.sequence_ids))
        assert all(row.condition_id == batch.condition_id for row in batch.sequence_losses)
        assert all(row.condition_id == batch.condition_id for row in batch.sequence_contrasts)
        assert all(row.condition_id == batch.condition_id for row in batch.sequence_events)
        assert all(row.condition_id == batch.condition_id for row in batch.aggregates)

    materialized = benchmark._evaluate_health_cases(
        cases=_cases(),
        main_bases=main_bases,
        edge_bases=edge_bases,
        calibration=_calibration(),
        thresholds=HealthThresholds(self_score=1.0, cross_score=1.0),
    )
    assert isinstance(materialized, HealthBenchmarkEvaluation)
    assert tuple(row for batch in batches for row in batch.sequence_losses) == (
        materialized.sequence_losses
    )
    assert tuple(row for batch in batches for row in batch.sequence_contrasts) == (
        materialized.sequence_contrasts
    )
    assert tuple(row for batch in batches for row in batch.sequence_events) == (
        materialized.sequence_events
    )
    assert tuple(row for batch in batches for row in batch.aggregates) == materialized.aggregates


def test_orchestration_rejects_duplicate_cases_and_exact_count_mismatch() -> None:
    main_bases, edge_bases = _bases()
    duplicate = (_cases()[0], _cases()[0])
    with pytest.raises(ValueError):
        benchmark._evaluate_health_cases(
            cases=duplicate,
            main_bases=main_bases,
            edge_bases=edge_bases,
            calibration=_calibration(),
            thresholds=HealthThresholds(self_score=1.0, cross_score=1.0),
        )
    with pytest.raises(ValueError):
        benchmark._evaluate_health_cases(
            cases=_cases(),
            main_bases=main_bases,
            edge_bases=edge_bases,
            calibration=_calibration(),
            thresholds=HealthThresholds(self_score=1.0, cross_score=1.0),
            require_exact_matrix=True,
        )


def _minimal_fit_handle(
    *,
    path: Path,
    artifact_sha256: str,
    run_sha256: str,
) -> LoadedHealthFitArtifact:
    handle = object.__new__(LoadedHealthFitArtifact)
    object.__setattr__(handle, "path", path)
    object.__setattr__(handle, "artifact_sha256", artifact_sha256)
    object.__setattr__(handle, "run_sha256", run_sha256)
    return handle


def test_public_apply_only_entrypoint_reloads_and_derives_every_scientific_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    intent = load_health_benchmark_intent(source_root=Path.cwd()).intent
    main = load_procedural_profile(Path("examples/profiles/constant-velocity-front-roi-v1.json"))
    edge = load_procedural_profile(Path("examples/profiles/constant-velocity-fov-edge-v1.json"))
    artifact_sha256 = "a" * 64
    run_sha256 = "b" * 64
    handle = _minimal_fit_handle(
        path=tmp_path,
        artifact_sha256=artifact_sha256,
        run_sha256=run_sha256,
    )
    authenticated = cast(
        LoadedHealthFitArtifact,
        SimpleNamespace(
            path=tmp_path,
            artifact_sha256=artifact_sha256,
            run_sha256=run_sha256,
            intent=intent,
            main_profile=main,
            edge_profile=edge,
            calibration=_calibration(),
            summary=SimpleNamespace(
                selected_self_threshold=0.999,
                selected_cross_threshold=0.995,
            ),
        ),
    )
    loaded_paths: list[Path] = []

    def fake_load(path: Path) -> LoadedHealthFitArtifact:
        loaded_paths.append(path)
        return authenticated

    population_calls: list[tuple[str, int, int, object]] = []

    def fake_population(
        profile: object,
        *,
        split: str,
        sequence_count: int,
        data_master_seed: int,
    ) -> tuple[HealthBaseSequence, ...]:
        population_calls.append((split, sequence_count, data_master_seed, profile))
        return tuple(cast(HealthBaseSequence, object()) for _ in range(sequence_count))

    captured: dict[str, object] = {}
    expected_result = cast(HealthBenchmarkEvaluation, object())

    def fake_cases(**kwargs: object) -> HealthBenchmarkEvaluation:
        captured.update(kwargs)
        return expected_result

    monkeypatch.setattr(benchmark, "load_health_fit_artifact", fake_load)
    monkeypatch.setattr(benchmark, "generate_health_base_sequences", fake_population)
    monkeypatch.setattr(benchmark, "_evaluate_health_cases", fake_cases)

    result = benchmark.evaluate_health_benchmark_test(fit_artifact=handle)

    assert result is expected_result
    assert loaded_paths == [tmp_path]
    assert population_calls == [
        ("test", 200, 1729, main),
        ("test", 100, 1729, edge),
    ]
    cases = cast(tuple[benchmark.HealthTestCase, ...], captured["cases"])
    assert tuple(case.condition_id for case in cases) == tuple(
        case.condition_id for case in expand_test_cases(intent)
    )
    assert captured["calibration"] is authenticated.calibration
    thresholds = cast(HealthThresholds, captured["thresholds"])
    assert thresholds == HealthThresholds(self_score=0.999, cross_score=0.995)
    assert captured["require_exact_matrix"] is True


def test_public_stream_entrypoint_reloads_and_derives_every_scientific_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    intent = load_health_benchmark_intent(source_root=Path.cwd()).intent
    main = load_procedural_profile(Path("examples/profiles/constant-velocity-front-roi-v1.json"))
    edge = load_procedural_profile(Path("examples/profiles/constant-velocity-fov-edge-v1.json"))
    artifact_sha256 = "a" * 64
    run_sha256 = "b" * 64
    handle = _minimal_fit_handle(
        path=tmp_path,
        artifact_sha256=artifact_sha256,
        run_sha256=run_sha256,
    )
    authenticated = cast(
        LoadedHealthFitArtifact,
        SimpleNamespace(
            path=tmp_path,
            artifact_sha256=artifact_sha256,
            run_sha256=run_sha256,
            intent=intent,
            main_profile=main,
            edge_profile=edge,
            calibration=_calibration(),
            summary=SimpleNamespace(
                selected_self_threshold=0.999,
                selected_cross_threshold=0.995,
            ),
        ),
    )
    monkeypatch.setattr(benchmark, "load_health_fit_artifact", lambda _path: authenticated)

    population_calls: list[tuple[str, int, int, object]] = []

    def fake_population(
        profile: object,
        *,
        split: str,
        sequence_count: int,
        data_master_seed: int,
    ) -> tuple[HealthBaseSequence, ...]:
        population_calls.append((split, sequence_count, data_master_seed, profile))
        return tuple(cast(HealthBaseSequence, object()) for _ in range(sequence_count))

    captured: dict[str, object] = {}
    expected_result = cast(HealthBenchmarkStreamSummary, object())

    def fake_cases(**kwargs: object) -> HealthBenchmarkStreamSummary:
        captured.update(kwargs)
        return expected_result

    sink = cast(Callable[[HealthConditionEvaluation], None], object())
    monkeypatch.setattr(benchmark, "generate_health_base_sequences", fake_population)
    monkeypatch.setattr(benchmark, "_evaluate_health_cases", fake_cases)

    result = benchmark.stream_health_benchmark_test(
        fit_artifact=handle,
        condition_sink=sink,
    )

    assert result is expected_result
    assert population_calls == [
        ("test", 200, 1729, main),
        ("test", 100, 1729, edge),
    ]
    cases = cast(tuple[benchmark.HealthTestCase, ...], captured["cases"])
    assert tuple(case.condition_id for case in cases) == tuple(
        case.condition_id for case in expand_test_cases(intent)
    )
    assert captured["calibration"] is authenticated.calibration
    assert captured["thresholds"] == HealthThresholds(
        self_score=0.999,
        cross_score=0.995,
    )
    assert captured["require_exact_matrix"] is True
    assert captured["condition_sink"] is sink
    assert captured["materialize_rows"] is False


def test_public_stream_entrypoint_rejects_noncanonical_fit_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    intent = load_health_benchmark_intent(source_root=Path.cwd()).intent
    artifact_sha256 = "a" * 64
    run_sha256 = "b" * 64
    handle = _minimal_fit_handle(
        path=tmp_path,
        artifact_sha256=artifact_sha256,
        run_sha256=run_sha256,
    )
    authenticated = cast(
        LoadedHealthFitArtifact,
        SimpleNamespace(
            path=tmp_path,
            artifact_sha256=artifact_sha256,
            run_sha256=run_sha256,
            intent=intent,
        ),
    )
    monkeypatch.setattr(benchmark, "load_health_fit_artifact", lambda _path: authenticated)
    monkeypatch.setattr(benchmark, "expand_test_cases", lambda _intent: ())

    with pytest.raises(ArtifactValidationError, match="frozen M4 matrix"):
        benchmark.stream_health_benchmark_test(
            fit_artifact=handle,
            condition_sink=lambda _condition: None,
        )


def test_public_apply_only_entrypoint_rejects_fabricated_handle_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    authentic = _minimal_fit_handle(
        path=tmp_path,
        artifact_sha256="a" * 64,
        run_sha256="b" * 64,
    )
    fabricated = _minimal_fit_handle(
        path=tmp_path,
        artifact_sha256="c" * 64,
        run_sha256="b" * 64,
    )

    def fake_load(_path: Path) -> LoadedHealthFitArtifact:
        return authentic

    monkeypatch.setattr(benchmark, "load_health_fit_artifact", fake_load)
    with pytest.raises(ArtifactValidationError, match="not authentic"):
        benchmark.evaluate_health_benchmark_test(fit_artifact=fabricated)
    with pytest.raises(TypeError, match="LoadedHealthFitArtifact"):
        benchmark.evaluate_health_benchmark_test(
            fit_artifact=cast(LoadedHealthFitArtifact, object())
        )

    assert tuple(signature(benchmark.evaluate_health_benchmark_test).parameters) == (
        "fit_artifact",
    )
    call = cast(Callable[..., object], benchmark.evaluate_health_benchmark_test)
    for injected in (
        {"thresholds": HealthThresholds(self_score=1.0, cross_score=1.0)},
        {"cases": _cases()},
        {"main_profile": object()},
        {"calibration": _calibration()},
    ):
        with pytest.raises(TypeError, match="unexpected keyword"):
            call(fit_artifact=authentic, **injected)
