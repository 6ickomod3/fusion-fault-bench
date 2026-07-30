from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import fusion_fault_bench.replay_benchmark as replay_benchmark
from fusion_fault_bench.contracts.replay_v1 import M5_SCENE_NAMES
from fusion_fault_bench.geometry.camera import PinholeCamera
from fusion_fault_bench.replay_geometry import NominalEligibility, RigidTransform3
from fusion_fault_bench.replay_source import (
    ReplayFrame,
    ReplayObjectFrame,
    ReplayPopulation,
    ReplayScene,
    ReplaySensorSnapshot,
)

ROOT = Path(__file__).resolve().parents[1]


def _transform() -> RigidTransform3:
    return RigidTransform3(
        rotation=np.eye(3, dtype=np.float64),
        translation_m=np.zeros(3, dtype=np.float64),
    )


def _synthetic_population() -> ReplayPopulation:
    pose = _transform()
    camera_model = PinholeCamera(
        intrinsic=np.asarray(
            ((100.0, 0.0, 50.0), (0.0, 100.0, 50.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        width_px=100,
        height_px=100,
    )
    scenes: list[ReplayScene] = []
    for scene_index, scene_name in enumerate(M5_SCENE_NAMES):
        frames: list[ReplayFrame] = []
        for frame_index in range(16):
            timestamp_us = 1_000_000 + scene_index * 10_000_000 + frame_index * 500_000
            reference_time_s = frame_index * 0.5
            center_global = np.asarray(
                (
                    12.0 + scene_index + 0.5 * reference_time_s,
                    1.0 + 0.1 * scene_index,
                    1.5,
                ),
                dtype=np.float64,
            )
            support = NominalEligibility(
                center_reference_ego_m=center_global,
                center_camera_m=center_global,
                roi_pass=True,
                camera_center_pass=True,
                lidar_points_pass=True,
                camera_estimator_available=True,
                lidar_estimator_available=True,
                eligible=True,
            )
            frames.append(
                ReplayFrame(
                    frame_index=frame_index,
                    sample_timestamp_us=timestamp_us,
                    reference_time_s=reference_time_s,
                    lidar=ReplaySensorSnapshot(
                        timestamp_us=timestamp_us,
                        global_from_ego=pose,
                        ego_from_sensor=pose,
                        camera=None,
                    ),
                    camera=ReplaySensorSnapshot(
                        timestamp_us=timestamp_us + 20_000,
                        global_from_ego=pose,
                        ego_from_sensor=pose,
                        camera=camera_model,
                    ),
                    objects=(
                        ReplayObjectFrame(
                            object_id="track:0000",
                            center_global_m=center_global,
                            size_width_length_height_m=np.asarray((2.0, 4.0, 1.5)),
                            orientation_global_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0)),
                            velocity_global_mps=np.asarray((0.5, 0.0, 0.0)),
                            velocity_method="centered",
                            acceleration_global_mps2=np.zeros(3, dtype=np.float64),
                            acceleration_method="centered",
                            category_name="vehicle.car",
                            visibility_level="v80-100",
                            num_lidar_points=5,
                            support=support,
                        ),
                    ),
                )
            )
        scenes.append(
            ReplayScene(
                scene_name=scene_name,
                sequence_id=f"nuscenes:{scene_name}",
                log_group_id=f"log-group:{scene_index % 2:02d}",
                frames=tuple(frames),
            )
        )
    return ReplayPopulation(scenes=tuple(scenes))


def test_all_ten_scene_orchestration_covers_both_frozen_panels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draw_calls = 0
    bootstrap_calls = 0
    generated_conditions: list[str] = []
    draw_ids_by_scene: dict[str, set[int]] = {}
    bootstrap_object_ids: set[int] = set()
    descriptors_complete = False
    original_draw = replay_benchmark.draw_replay_scene_randomness
    original_bootstrap = replay_benchmark.replay_bootstrap_indices
    original_generate = replay_benchmark.generate_replay_condition_sequence
    original_descriptors = replay_benchmark.build_replay_descriptor_aggregates
    original_persistent_aggregate = replay_benchmark.aggregate_replay_persistent_case
    original_health_aggregate = replay_benchmark.aggregate_replay_health_case
    original_crossovers = replay_benchmark.evaluate_replay_persistent_crossovers

    def counted_draw(*args: Any, **kwargs: Any) -> Any:
        nonlocal draw_calls
        draw_calls += 1
        return original_draw(*args, **kwargs)

    def counted_bootstrap(*args: Any, **kwargs: Any) -> Any:
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        return original_bootstrap(*args, **kwargs)

    def counted_generate(*args: Any, **kwargs: Any) -> Any:
        assert descriptors_complete
        scene = args[0]
        condition = kwargs["condition"]
        draws = kwargs["draws"]
        generated_conditions.append(condition.selector)
        draw_ids_by_scene.setdefault(scene.sequence_id, set()).add(id(draws))
        return original_generate(*args, **kwargs)

    def counted_descriptors(*args: Any, **kwargs: Any) -> Any:
        nonlocal descriptors_complete
        result = original_descriptors(*args, **kwargs)
        descriptors_complete = True
        return result

    def counted_persistent_aggregate(*args: Any, **kwargs: Any) -> Any:
        bootstrap_object_ids.add(id(kwargs["bootstrap_indices"]))
        return original_persistent_aggregate(*args, **kwargs)

    def counted_health_aggregate(*args: Any, **kwargs: Any) -> Any:
        bootstrap_object_ids.add(id(kwargs["bootstrap_indices"]))
        return original_health_aggregate(*args, **kwargs)

    def counted_crossovers(*args: Any, **kwargs: Any) -> Any:
        bootstrap_object_ids.add(id(kwargs["bootstrap_indices"]))
        return original_crossovers(*args, **kwargs)

    monkeypatch.setattr(
        replay_benchmark,
        "draw_replay_scene_randomness",
        counted_draw,
    )
    monkeypatch.setattr(replay_benchmark, "replay_bootstrap_indices", counted_bootstrap)
    monkeypatch.setattr(
        replay_benchmark,
        "generate_replay_condition_sequence",
        counted_generate,
    )
    monkeypatch.setattr(
        replay_benchmark,
        "build_replay_descriptor_aggregates",
        counted_descriptors,
    )
    monkeypatch.setattr(
        replay_benchmark,
        "aggregate_replay_persistent_case",
        counted_persistent_aggregate,
    )
    monkeypatch.setattr(
        replay_benchmark,
        "aggregate_replay_health_case",
        counted_health_aggregate,
    )
    monkeypatch.setattr(
        replay_benchmark,
        "evaluate_replay_persistent_crossovers",
        counted_crossovers,
    )

    result = replay_benchmark.run_replay_benchmark(
        _synthetic_population(),
        source_root=ROOT,
    )

    assert len(result.plan.persistent_cases) == 71
    assert len(result.plan.health_cases) == 43
    assert len(result.persistent_scene_evaluations) == 710
    assert len(result.persistent_metrics) == 464
    assert len(result.persistent_crossovers) == 10
    assert len(result.health_results) == 12_660
    assert len(result.health_contrasts) == 6_450
    assert len(result.health_events) == 1_720
    assert len(result.health_metrics) == 14_988
    assert sum(row.status == "not-applicable" for row in result.health_metrics) == 240
    assert result.bootstrap_replicates == 2_000
    assert result.log_group_ordinals == ("log-group:00", "log-group:01") * 5
    assert result.scene_frame_counts == (16,) * 10
    assert {row.population for row in result.descriptor_aggregates} == {
        "nuscenes-mini-replay",
        "m3-main-test-comparator",
    }
    assert {row.condition_selector for row in result.persistent_metrics} == {
        case.fault_condition.selector for case in result.plan.persistent_cases
    }
    assert {row.condition_selector for row in result.health_metrics} == {
        case.selector for case in result.plan.health_cases
    }
    assert draw_calls == 10
    assert bootstrap_calls == 1
    assert generated_conditions == [
        *(
            case.fault_condition.selector
            for case in result.plan.persistent_cases
            for _ in M5_SCENE_NAMES
        ),
        *(case.selector for case in result.plan.health_cases for _ in M5_SCENE_NAMES),
    ]
    assert len(draw_ids_by_scene) == 10
    assert all(len(draw_ids) == 1 for draw_ids in draw_ids_by_scene.values())
    assert len({next(iter(draw_ids)) for draw_ids in draw_ids_by_scene.values()}) == 10
    assert len(bootstrap_object_ids) == 1
    assert repr(result) == "ReplayBenchmarkEvidence()"

    with pytest.raises(ValueError, match="health result"):
        replace(result, health_results=())
    with pytest.raises(ValueError, match="health metric"):
        replace(result, health_metrics=(*result.health_metrics, result.health_metrics[0]))
    wrong_identity = result.plan.persistent_cases[-1].identity_sha256
    with pytest.raises(ValueError, match="persistent scene grid"):
        replace(
            result,
            persistent_scene_evaluations=(
                replace(
                    result.persistent_scene_evaluations[0],
                    replay_experiment_identity_sha256=wrong_identity,
                ),
                *result.persistent_scene_evaluations[1:],
            ),
        )
    with pytest.raises(ValueError, match="persistent metric grid"):
        replace(
            result,
            persistent_metrics=(
                replace(
                    result.persistent_metrics[0],
                    replay_experiment_identity_sha256=wrong_identity,
                ),
                *result.persistent_metrics[1:],
            ),
        )
    with pytest.raises(ValueError, match="crossover grid"):
        replace(
            result,
            persistent_crossovers=tuple(reversed(result.persistent_crossovers)),
        )
    with pytest.raises(ValueError, match="crossover grid"):
        replace(
            result,
            persistent_crossovers=(
                replace(
                    result.persistent_crossovers[0],
                    replay_experiment_identity_sha256=wrong_identity,
                ),
                *result.persistent_crossovers[1:],
            ),
        )
    with pytest.raises(ValueError, match="health result grid"):
        replace(
            result,
            health_results=(
                result.health_results[0].model_copy(
                    update={"replay_experiment_identity_sha256": "0" * 64}
                ),
                *result.health_results[1:],
            ),
        )
    with pytest.raises(ValueError, match="health contrast grid"):
        replace(
            result,
            health_contrasts=(
                replace(
                    result.health_contrasts[0],
                    replay_experiment_identity_sha256="0" * 64,
                ),
                *result.health_contrasts[1:],
            ),
        )
    with pytest.raises(ValueError, match="health event semantics"):
        replace(
            result,
            health_events=(
                result.health_events[0].model_copy(
                    update={"schedule": replay_benchmark.replay_health_schedule(17)}
                ),
                *result.health_events[1:],
            ),
        )
    with pytest.raises(ValueError, match="descriptor comparison"):
        replace(
            result,
            descriptor_aggregates=tuple(
                row
                for row in result.descriptor_aggregates
                if not (
                    row.population == "nuscenes-mini-replay" and row.descriptor_id == "sample-count"
                )
            ),
        )
    sample_descriptor_index = next(
        index
        for index, row in enumerate(result.descriptor_aggregates)
        if row.population == "nuscenes-mini-replay"
        and row.descriptor_id == "sample-count"
        and row.statistic == "minimum"
    )
    descriptors = list(result.descriptor_aggregates)
    descriptors[sample_descriptor_index] = replace(
        descriptors[sample_descriptor_index],
        unit="fraction",
    )
    with pytest.raises(ValueError, match="descriptor statistic grid"):
        replace(result, descriptor_aggregates=tuple(descriptors))
    descriptors[sample_descriptor_index] = replace(
        result.descriptor_aggregates[sample_descriptor_index],
        value=17.0,
    )
    with pytest.raises(ValueError, match="descriptor provenance"):
        replace(result, descriptor_aggregates=tuple(descriptors))
    with pytest.raises(ValueError, match="log-group ordinals"):
        replace(
            result,
            log_group_ordinals=("log-group:00", "log-group:02") * 5,
        )


def test_orchestration_rejects_invalid_local_population_before_faults() -> None:
    population = _synthetic_population()
    noncontiguous_groups = ReplayPopulation(
        scenes=tuple(
            replace(
                scene,
                log_group_id=("log-group:00" if index % 2 == 0 else "log-group:02"),
            )
            for index, scene in enumerate(population.scenes)
        )
    )
    with pytest.raises(ValueError, match="log-group ordinals"):
        replay_benchmark.run_replay_benchmark(
            noncontiguous_groups,
            source_root=ROOT,
        )

    short_first_scene = replace(
        population.scenes[0],
        frames=population.scenes[0].frames[:15],
    )
    short_population = ReplayPopulation(scenes=(short_first_scene, *population.scenes[1:]))
    with pytest.raises(ValueError, match="at least 16"):
        replay_benchmark.run_replay_benchmark(short_population, source_root=ROOT)

    empty_frames = tuple(replace(frame, objects=()) for frame in population.scenes[0].frames)
    empty_first_scene = replace(population.scenes[0], frames=empty_frames)
    empty_population = ReplayPopulation(scenes=(empty_first_scene, *population.scenes[1:]))
    with pytest.raises(ValueError, match="nonempty frozen base support"):
        replay_benchmark.run_replay_benchmark(empty_population, source_root=ROOT)
