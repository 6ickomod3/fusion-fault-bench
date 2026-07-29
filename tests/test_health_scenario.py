from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.io import load_json_object
from fusion_fault_bench.contracts.procedural_profile_v1 import (
    load_procedural_profile,
)
from fusion_fault_bench.reference.procedural import reference_eligibility_mask
from fusion_fault_bench.scenarios.health import (
    M4_HEALTH_INTENT_SHA256,
    HealthEventSchedule,
    HealthFaultSpec,
    adapt_procedural_sequence,
    build_bounded_acceleration_control,
    generate_health_base_sequences,
    health_event_schedule,
)
from fusion_fault_bench.scenarios.procedural import generate_procedural_sequences

_ROOT = Path(__file__).resolve().parents[1]
_MAIN_PROFILE = _ROOT / "examples/profiles/constant-velocity-front-roi-v1.json"
_EDGE_PROFILE = _ROOT / "examples/profiles/constant-velocity-fov-edge-v1.json"


def test_scenario_is_bound_to_the_frozen_m4_intent() -> None:
    intent = load_json_object(_ROOT / "examples/health/m4-health-v1.json")

    assert sha256_digest(intent) == M4_HEALTH_INTENT_SHA256


def test_event_schedules_use_exact_half_open_windows() -> None:
    standard = health_event_schedule("standard")
    cold = health_event_schedule("cold_start")

    assert standard.score_frames == (2, 48)
    assert standard.fault_active_frames == (12, 36)
    assert standard.recovery_frames == (36, 48)
    assert standard.clean_prefix_frames == (0, 12)
    assert standard.predictor_initialization_frames == (0, 2)
    np.testing.assert_array_equal(
        np.flatnonzero(standard.active_mask()),
        np.arange(12, 36),
    )
    assert cold.score_frames == (0, 48)
    assert cold.fault_active_frames == (0, 24)
    assert cold.recovery_frames == (24, 48)
    assert cold.clean_prefix_frames is None
    assert cold.predictor_initialization_frames is None
    np.testing.assert_array_equal(
        np.flatnonzero(cold.active_mask()),
        np.arange(24),
    )
    with pytest.raises(ValueError, match="unknown"):
        health_event_schedule("invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"schedule_id": "invalid"}, "unknown"),
        ({"frame_count": 47}, "exactly 48"),
        ({"score_frames": (2, 49)}, "score_frames"),
        ({"clean_prefix_frames": (12, 12)}, "clean_prefix_frames"),
    ],
)
def test_event_schedule_contract_rejects_invalid_intervals(
    updates: dict[str, object],
    match: str,
) -> None:
    values: dict[str, object] = {
        "schedule_id": "standard",
        "frame_count": 48,
        "score_frames": (2, 48),
        "fault_active_frames": (12, 36),
        "recovery_frames": (36, 48),
        "clean_prefix_frames": (0, 12),
        "predictor_initialization_frames": (0, 2),
    }
    values.update(updates)
    with pytest.raises(ValueError, match=match):
        HealthEventSchedule(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("target", ["camera", "lidar"])
def test_mirrored_generic_fault_coordinates_are_accepted(target: str) -> None:
    for family, axis, unit, value in (
        ("additive-position-bias", "y", "m", -0.5),
        ("increased-noise-underreported", "xy", "std-scale", 1.5),
        ("increased-noise-correctly-reported", "xy", "std-scale", 1.5),
        ("timestamp-offset", "time", "s", 0.1),
        ("dropout", "availability", "probability", 0.25),
    ):
        spec = HealthFaultSpec(
            family=family,  # type: ignore[arg-type]
            target=target,  # type: ignore[arg-type]
            axis=axis,  # type: ignore[arg-type]
            unit=unit,  # type: ignore[arg-type]
            value=value,
        )
        assert spec.target == target


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "family": "unknown",
                "target": "none",
                "axis": "none",
                "unit": "identity",
                "value": 0.0,
            },
            "unknown M4 fault family",
        ),
        (
            {
                "family": "identity",
                "target": "none",
                "axis": "none",
                "unit": "identity",
                "value": 1.0,
            },
            "identity requires value zero",
        ),
        (
            {
                "family": "dropout",
                "target": "camera",
                "axis": "availability",
                "unit": "probability",
                "value": 0.0,
            },
            "dropout probability",
        ),
        (
            {
                "family": "calibration-yaw",
                "target": "lidar",
                "axis": "yaw",
                "unit": "rad",
                "value": 0.06,
            },
            "invalid M4 coordinate",
        ),
        (
            {
                "family": "timestamp-offset",
                "target": "camera",
                "axis": "time",
                "unit": "s",
                "value": 0.1,
                "schedule": "cold_start",
            },
            "cold-start",
        ),
        (
            {
                "family": "increased-noise-underreported",
                "target": "camera",
                "axis": "xy",
                "unit": "std-scale",
                "value": 1.0,
            },
            "noise scale",
        ),
        (
            {
                "family": "additive-position-bias",
                "target": "camera",
                "axis": "y",
                "unit": "m",
                "value": 0.0,
            },
            "nonzero",
        ),
        (
            {
                "family": "clean-predictor-mismatch",
                "target": "none",
                "axis": "motion",
                "unit": "m/s^2",
                "value": 7.0,
            },
            "exactly 8",
        ),
        (
            {
                "family": "identity",
                "target": "none",
                "axis": "none",
                "unit": "identity",
                "value": -0.0,
            },
            "positive zero",
        ),
        (
            {
                "family": "identity",
                "target": "none",
                "axis": "none",
                "unit": "identity",
                "value": float("nan"),
            },
            "finite",
        ),
    ],
)
def test_fault_spec_rejects_noncontractual_coordinates(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        HealthFaultSpec(**kwargs)  # type: ignore[arg-type]


def test_base_adapter_preserves_m3_draw_order_and_stationary_scene_frame() -> None:
    profile = load_procedural_profile(_MAIN_PROFILE)
    source = generate_procedural_sequences(
        profile,
        split="test",
        sequence_count=2,
        data_master_seed=1729,
    )[0]
    adapted = adapt_procedural_sequence(source, profile=profile)

    assert adapted.health_frame == "persistent-scene-bev"
    assert adapted.motion_model == "profile-constant-velocity"
    assert adapted.sequence_id == source.sequence_id
    assert adapted.object_ids == source.object_ids
    assert adapted.eligible_object_frame_count == 48 * 6
    np.testing.assert_array_equal(adapted.truth_xy_m, source.truth_xy_m)
    np.testing.assert_array_equal(
        adapted.camera_standard_normal_xy.reshape(-1, 2),
        source.camera_standard_normal_xy,
    )
    np.testing.assert_array_equal(
        adapted.lidar_standard_normal_xy.reshape(-1, 2),
        source.lidar_standard_normal_xy,
    )
    np.testing.assert_array_equal(
        adapted.dropout_uniform_by_frame,
        source.fault_uniform_by_frame,
    )
    np.testing.assert_array_equal(
        adapted.velocity_xy_mps,
        np.broadcast_to(source.velocity_xy_mps, (48, 6, 2)),
    )
    assert not adapted.truth_xy_m.flags.writeable
    assert not adapted.eligibility_mask.flags.writeable
    assert not adapted.camera_standard_normal_xy.flags.writeable


def test_base_population_generation_is_repeatable_and_keeps_profile_splits() -> None:
    profile = load_procedural_profile(_EDGE_PROFILE)

    first = generate_health_base_sequences(
        profile,
        split="test",
        sequence_count=2,
        data_master_seed=1729,
    )
    second = generate_health_base_sequences(
        profile,
        split="test",
        sequence_count=2,
        data_master_seed=1729,
    )

    assert tuple(item.sequence_id for item in first) == tuple(item.sequence_id for item in second)
    assert all(item.object_count == 4 for item in first)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.truth_xy_m, right.truth_xy_m)
        np.testing.assert_array_equal(
            left.camera_standard_normal_xy,
            right.camera_standard_normal_xy,
        )
        assert left.eligibility_sha256 == right.eligibility_sha256


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"sequence_id": ""}, "nonempty"),
        ({"profile_id": "invalid"}, "main and edge"),
        ({"motion_model": "invalid"}, "motion model"),
        ({"health_frame": "ego-bev"}, "persistent scene"),
        (
            {
                "object_ids": (
                    "object:01",
                    "object:00",
                    "object:02",
                    "object:03",
                    "object:04",
                    "object:05",
                )
            },
            "UTF-8",
        ),
        ({"sequence_index": -1}, "nonnegative"),
        ({"frame_indices": np.arange(48, dtype=np.float64)}, "integer vector"),
        ({"frame_indices": np.arange(48, dtype=np.int64) + 1}, "zero-based"),
        ({"reference_times_s": np.arange(48, dtype=np.float64)}, "0.1 s"),
        ({"truth_xy_m": np.zeros((47, 6, 2))}, "shape"),
        ({"truth_xy_m": np.full((48, 6, 2), np.nan)}, "finite"),
        ({"eligibility_mask": np.ones((47, 6), dtype=np.bool_)}, "invalid shape"),
        (
            {
                "eligibility_mask": np.vstack(
                    (
                        np.zeros((1, 6), dtype=np.bool_),
                        np.ones((47, 6), dtype=np.bool_),
                    )
                )
            },
            "at least one",
        ),
        ({"dropout_uniform_by_frame": np.ones(48)}, r"\[0, 1\)"),
        ({"camera_true_translation_m": np.zeros(2)}, "shape"),
        ({"camera_true_rotation": np.diag((1.0, 1.0, 2.0))}, "orthonormal"),
        ({"camera_true_rotation": np.diag((1.0, 1.0, -1.0))}, "right-handed"),
        ({"roi_x_max_m": 1.0}, "ROI"),
        ({"eligibility_sha256": "0" * 64}, "does not match"),
    ],
)
def test_health_base_sequence_validates_every_immutable_boundary(
    updates: dict[str, object],
    match: str,
) -> None:
    profile = load_procedural_profile(_MAIN_PROFILE)
    base = generate_health_base_sequences(
        profile,
        split="test",
        sequence_count=2,
        data_master_seed=1729,
    )[0]

    with pytest.raises(ValueError, match=match):
        replace(base, **updates)  # type: ignore[arg-type]


def test_bounded_acceleration_uses_literal_recurrence_and_recomputed_eligibility() -> None:
    profile = load_procedural_profile(_MAIN_PROFILE)
    base = generate_health_base_sequences(
        profile,
        split="test",
        sequence_count=2,
        data_master_seed=1729,
    )[0]

    maneuver = build_bounded_acceleration_control(base)

    expected_position = np.empty_like(base.truth_xy_m)
    expected_velocity = np.empty_like(base.velocity_xy_mps)
    expected_position[0] = base.truth_xy_m[0]
    expected_velocity[0] = base.velocity_xy_mps[0]
    side = np.asarray((-1.0, -1.0, -1.0, 1.0, 1.0, 1.0))
    for transition in range(47):
        acceleration = np.zeros((6, 2), dtype=np.float64)
        if 18 <= transition < 24:
            acceleration[:, 1] = side * 8.0
        elif 24 <= transition < 30:
            acceleration[:, 1] = -side * 8.0
        expected_position[transition + 1] = (
            expected_position[transition]
            + expected_velocity[transition] * 0.1
            + 0.5 * acceleration * 0.1**2
        )
        expected_velocity[transition + 1] = expected_velocity[transition] + acceleration * 0.1

    independent_eligibility = reference_eligibility_mask(
        expected_position,
        x_min_m=5.0,
        x_max_m=60.0,
        abs_y_max_m=40.0,
        camera_half_fov_rad=0.7,
    )
    assert maneuver.motion_model == "bounded-acceleration-control"
    np.testing.assert_array_equal(maneuver.truth_xy_m, expected_position)
    np.testing.assert_array_equal(maneuver.velocity_xy_mps, expected_velocity)
    np.testing.assert_array_equal(maneuver.eligibility_mask, independent_eligibility)
    assert np.all(maneuver.eligibility_mask)
    np.testing.assert_array_equal(
        maneuver.camera_standard_normal_xy,
        base.camera_standard_normal_xy,
    )
    np.testing.assert_array_equal(
        maneuver.lidar_standard_normal_xy,
        base.lidar_standard_normal_xy,
    )
    np.testing.assert_array_equal(
        maneuver.dropout_uniform_by_frame,
        base.dropout_uniform_by_frame,
    )


def test_bounded_acceleration_rejects_wrong_population_and_recursion() -> None:
    main = load_procedural_profile(_MAIN_PROFILE)
    validation = generate_health_base_sequences(
        main,
        split="validation",
        sequence_count=2,
        data_master_seed=1729,
    )[0]
    with pytest.raises(ValueError, match="main test"):
        build_bounded_acceleration_control(validation)

    test = generate_health_base_sequences(
        main,
        split="test",
        sequence_count=2,
        data_master_seed=1729,
    )[0]
    maneuver = build_bounded_acceleration_control(test)
    with pytest.raises(ValueError, match="recursively"):
        build_bounded_acceleration_control(maneuver)

    incomplete = np.array(test.eligibility_mask, copy=True)
    incomplete[0, 0] = False
    incomplete_base = replace(test)
    object.__setattr__(incomplete_base, "eligibility_mask", incomplete)
    with pytest.raises(ValueError, match="full base draw support"):
        build_bounded_acceleration_control(incomplete_base)
