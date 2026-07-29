from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from fusion_fault_bench.contracts.procedural_profile_v1 import (
    load_procedural_profile,
)
from fusion_fault_bench.experiments.health import (
    HealthObservationSequence,
    generate_health_observations,
)
from fusion_fault_bench.health import (
    HealthFrameInput,
    ModalityMeasurement,
    ObjectHealthInput,
)
from fusion_fault_bench.scenarios.health import (
    HealthBaseSequence,
    HealthFaultSpec,
    build_bounded_acceleration_control,
    generate_health_base_sequences,
    health_event_schedule,
)

_ROOT = Path(__file__).resolve().parents[1]
_MAIN_PROFILE = _ROOT / "examples/profiles/constant-velocity-front-roi-v1.json"
_EDGE_PROFILE = _ROOT / "examples/profiles/constant-velocity-fov-edge-v1.json"


@pytest.fixture(scope="module")
def main_test_base() -> HealthBaseSequence:
    profile = load_procedural_profile(_MAIN_PROFILE)
    return generate_health_base_sequences(
        profile,
        split="test",
        sequence_count=2,
        data_master_seed=1729,
    )[0]


@pytest.fixture(scope="module")
def edge_test_base() -> HealthBaseSequence:
    profile = load_procedural_profile(_EDGE_PROFILE)
    return generate_health_base_sequences(
        profile,
        split="test",
        sequence_count=2,
        data_master_seed=1729,
    )[0]


def _identity(base: HealthBaseSequence) -> HealthObservationSequence:
    return generate_health_observations(
        base,
        fault=HealthFaultSpec(
            family="identity",
            target="none",
            axis="none",
            unit="identity",
            value=0.0,
        ),
    )


def test_identity_uses_true_extrinsic_and_keeps_covariance_roles_separate(
    main_test_base: HealthBaseSequence,
) -> None:
    observed = _identity(main_test_base)

    expected_camera = main_test_base.truth_xy_m + main_test_base.camera_standard_normal_xy
    expected_lidar = main_test_base.truth_xy_m + 0.3 * main_test_base.lidar_standard_normal_xy
    np.testing.assert_allclose(
        observed.camera_value_xy_m,
        expected_camera,
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        observed.lidar_value_xy_m,
        expected_lidar,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        observed.camera_actual_covariance_xy_m2[..., 0, 0],
        1.0,
    )
    np.testing.assert_array_equal(
        observed.camera_reported_covariance_xy_m2,
        observed.camera_actual_covariance_xy_m2,
    )
    np.testing.assert_allclose(
        observed.lidar_actual_covariance_xy_m2[..., 0, 0],
        0.09,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        observed.lidar_reported_covariance_xy_m2,
        observed.lidar_actual_covariance_xy_m2,
    )
    np.testing.assert_array_equal(
        observed.camera_reported_times_s,
        observed.reference_times_s,
    )
    np.testing.assert_array_equal(
        observed.lidar_reported_times_s,
        observed.reference_times_s,
    )
    assert np.all(observed.camera_available)
    assert np.all(observed.lidar_available)
    for array in (
        observed.truth_xy_m,
        observed.camera_value_xy_m,
        observed.lidar_value_xy_m,
        observed.camera_actual_covariance_xy_m2,
        observed.camera_reported_covariance_xy_m2,
        observed.camera_available,
    ):
        assert not array.flags.writeable


@pytest.mark.parametrize(
    "fault",
    [
        HealthFaultSpec("additive-position-bias", "camera", "y", "m", -2.0),
        HealthFaultSpec("additive-position-bias", "lidar", "y", "m", 2.0),
        HealthFaultSpec(
            "increased-noise-underreported",
            "camera",
            "xy",
            "std-scale",
            4.0,
        ),
        HealthFaultSpec(
            "increased-noise-underreported",
            "lidar",
            "xy",
            "std-scale",
            4.0,
        ),
        HealthFaultSpec(
            "increased-noise-correctly-reported",
            "camera",
            "xy",
            "std-scale",
            4.0,
        ),
        HealthFaultSpec(
            "increased-noise-correctly-reported",
            "lidar",
            "xy",
            "std-scale",
            4.0,
        ),
        HealthFaultSpec("timestamp-offset", "camera", "time", "s", -0.4),
        HealthFaultSpec("timestamp-offset", "lidar", "time", "s", 0.4),
        HealthFaultSpec(
            "dropout",
            "camera",
            "availability",
            "probability",
            0.75,
        ),
        HealthFaultSpec(
            "dropout",
            "lidar",
            "availability",
            "probability",
            0.75,
        ),
        HealthFaultSpec("calibration-translation", "camera", "x", "m", 2.0),
        HealthFaultSpec("calibration-yaw", "camera", "yaw", "rad", 0.06),
    ],
)
def test_every_transient_operator_is_exact_identity_outside_event(
    main_test_base: HealthBaseSequence,
    fault: HealthFaultSpec,
) -> None:
    clean = _identity(main_test_base)
    faulted = generate_health_observations(main_test_base, fault=fault)
    outside = ~health_event_schedule(fault.schedule).active_mask()

    for field_name in (
        "camera_value_xy_m",
        "lidar_value_xy_m",
        "camera_actual_covariance_xy_m2",
        "lidar_actual_covariance_xy_m2",
        "camera_reported_covariance_xy_m2",
        "lidar_reported_covariance_xy_m2",
        "camera_reported_times_s",
        "lidar_reported_times_s",
        "camera_available",
        "lidar_available",
    ):
        clean_value = getattr(clean, field_name)
        faulted_value = getattr(faulted, field_name)
        np.testing.assert_array_equal(faulted_value[outside], clean_value[outside])


@pytest.mark.parametrize(
    ("target", "value"),
    [("camera", -3.0), ("lidar", 3.0)],
)
def test_additive_bias_changes_only_target_axis_during_event(
    main_test_base: HealthBaseSequence,
    target: str,
    value: float,
) -> None:
    clean = _identity(main_test_base)
    faulted = generate_health_observations(
        main_test_base,
        fault=HealthFaultSpec(
            "additive-position-bias",
            target,  # type: ignore[arg-type]
            "y",
            "m",
            value,
        ),
    )
    active = health_event_schedule("standard").active_mask()
    target_name = f"{target}_value_xy_m"
    other_name = "lidar_value_xy_m" if target == "camera" else "camera_value_xy_m"
    difference = getattr(faulted, target_name) - getattr(clean, target_name)

    np.testing.assert_array_equal(difference[active, :, 0], 0.0)
    np.testing.assert_allclose(
        difference[active, :, 1],
        value,
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_array_equal(getattr(faulted, other_name), getattr(clean, other_name))


@pytest.mark.parametrize("target", ["camera", "lidar"])
def test_noise_scaling_distinguishes_actual_and_reported_covariance(
    main_test_base: HealthBaseSequence,
    target: str,
) -> None:
    under = generate_health_observations(
        main_test_base,
        fault=HealthFaultSpec(
            "increased-noise-underreported",
            target,  # type: ignore[arg-type]
            "xy",
            "std-scale",
            3.0,
        ),
    )
    correct = generate_health_observations(
        main_test_base,
        fault=HealthFaultSpec(
            "increased-noise-correctly-reported",
            target,  # type: ignore[arg-type]
            "xy",
            "std-scale",
            3.0,
        ),
    )
    active = health_event_schedule("standard").active_mask()
    base_variance = 1.0 if target == "camera" else 0.09
    actual_name = f"{target}_actual_covariance_xy_m2"
    reported_name = f"{target}_reported_covariance_xy_m2"
    value_name = f"{target}_value_xy_m"

    np.testing.assert_array_equal(
        getattr(under, actual_name)[active, :, 0, 0],
        base_variance * 9.0,
    )
    np.testing.assert_array_equal(
        getattr(under, reported_name)[active, :, 0, 0],
        base_variance,
    )
    np.testing.assert_array_equal(
        getattr(correct, actual_name),
        getattr(under, actual_name),
    )
    np.testing.assert_array_equal(
        getattr(correct, reported_name)[active, :, 0, 0],
        base_variance * 9.0,
    )
    np.testing.assert_array_equal(
        getattr(correct, value_name),
        getattr(under, value_name),
    )


@pytest.mark.parametrize("translation_m", [-3.0, 3.0])
def test_calibration_translation_corrupts_reconstruction_metadata_only(
    main_test_base: HealthBaseSequence,
    translation_m: float,
) -> None:
    clean = _identity(main_test_base)
    faulted = generate_health_observations(
        main_test_base,
        fault=HealthFaultSpec(
            "calibration-translation",
            "camera",
            "x",
            "m",
            translation_m,
        ),
    )
    active = health_event_schedule("standard").active_mask()
    difference = faulted.camera_value_xy_m - clean.camera_value_xy_m

    np.testing.assert_allclose(
        difference[active, :, 0],
        translation_m,
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        difference[active, :, 1],
        0.0,
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_array_equal(
        faulted.camera_actual_covariance_xy_m2,
        clean.camera_actual_covariance_xy_m2,
    )
    np.testing.assert_array_equal(
        faulted.camera_reported_covariance_xy_m2,
        clean.camera_reported_covariance_xy_m2,
    )


@pytest.mark.parametrize("yaw_rad", [-0.06, 0.06])
def test_calibration_yaw_rotates_true_proxy_only_at_reported_reconstruction(
    main_test_base: HealthBaseSequence,
    yaw_rad: float,
) -> None:
    clean = _identity(main_test_base)
    faulted = generate_health_observations(
        main_test_base,
        fault=HealthFaultSpec(
            "calibration-yaw",
            "camera",
            "yaw",
            "rad",
            yaw_rad,
        ),
    )
    active = health_event_schedule("standard").active_mask()
    cosine = np.cos(yaw_rad)
    sine = np.sin(yaw_rad)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)))
    expected = clean.camera_value_xy_m[active] @ rotation.T

    np.testing.assert_allclose(
        faulted.camera_value_xy_m[active],
        expected,
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_array_equal(
        faulted.camera_reported_covariance_xy_m2,
        clean.camera_reported_covariance_xy_m2,
    )


@pytest.mark.parametrize(
    ("target", "offset_s"),
    [("camera", -0.6), ("lidar", 0.6)],
)
def test_timestamp_fault_uses_reference_clock_for_alignment_and_reports_residual(
    main_test_base: HealthBaseSequence,
    target: str,
    offset_s: float,
) -> None:
    clean = _identity(main_test_base)
    faulted = generate_health_observations(
        main_test_base,
        fault=HealthFaultSpec(
            "timestamp-offset",
            target,  # type: ignore[arg-type]
            "time",
            "s",
            offset_s,
        ),
    )
    active = health_event_schedule("standard").active_mask()
    value_name = f"{target}_value_xy_m"
    time_name = f"{target}_reported_times_s"

    np.testing.assert_allclose(
        getattr(faulted, value_name)[active] - getattr(clean, value_name)[active],
        -offset_s * main_test_base.velocity_xy_mps[active],
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        getattr(faulted, time_name)[active] - clean.reference_times_s[active],
        offset_s,
        rtol=0.0,
        atol=5e-16,
    )
    other = "lidar" if target == "camera" else "camera"
    np.testing.assert_array_equal(
        getattr(faulted, f"{other}_reported_times_s"),
        clean.reference_times_s,
    )


@pytest.mark.parametrize("target", ["camera", "lidar"])
def test_dropout_is_frame_shared_nested_and_strict_u_less_than_probability(
    main_test_base: HealthBaseSequence,
    target: str,
) -> None:
    low = generate_health_observations(
        main_test_base,
        fault=HealthFaultSpec(
            "dropout",
            target,  # type: ignore[arg-type]
            "availability",
            "probability",
            0.25,
        ),
    )
    high = generate_health_observations(
        main_test_base,
        fault=HealthFaultSpec(
            "dropout",
            target,  # type: ignore[arg-type]
            "availability",
            "probability",
            0.75,
        ),
    )
    full = generate_health_observations(
        main_test_base,
        fault=HealthFaultSpec(
            "dropout",
            target,  # type: ignore[arg-type]
            "availability",
            "probability",
            1.0,
        ),
    )
    active = health_event_schedule("standard").active_mask()
    expected_low_drop = active & (main_test_base.dropout_uniform_by_frame < 0.25)
    expected_high_drop = active & (main_test_base.dropout_uniform_by_frame < 0.75)
    low_available = getattr(low, f"{target}_available")
    high_available = getattr(high, f"{target}_available")
    full_available = getattr(full, f"{target}_available")

    np.testing.assert_array_equal(~low_available, expected_low_drop)
    np.testing.assert_array_equal(~high_available, expected_high_drop)
    assert np.all((~low_available) <= (~high_available))
    assert np.all(~full_available[active])
    assert np.all(full_available[~active])
    other = "lidar" if target == "camera" else "camera"
    assert np.all(getattr(high, f"{other}_available"))

    frames = high.health_frame_inputs()
    for frame_index, frame in enumerate(frames):
        target_missing = not bool(high_available[frame_index])
        for item in frame.objects:
            measurement = item.camera if target == "camera" else item.lidar
            assert (measurement is None) == target_missing


def test_common_mode_shifts_both_modalities_without_creating_a_target(
    edge_test_base: HealthBaseSequence,
) -> None:
    clean = _identity(edge_test_base)
    faulted = generate_health_observations(
        edge_test_base,
        fault=HealthFaultSpec(
            "common-mode-position-bias",
            "both",
            "x",
            "m",
            -4.0,
        ),
    )
    active = health_event_schedule("standard").active_mask()

    for field_name in ("camera_value_xy_m", "lidar_value_xy_m"):
        difference = getattr(faulted, field_name) - getattr(clean, field_name)
        np.testing.assert_allclose(
            difference[active, :, 0],
            -4.0,
            rtol=0.0,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            difference[active, :, 1],
            0.0,
            rtol=0.0,
            atol=2e-14,
        )


def test_cold_start_fault_uses_its_own_event_and_recovery_windows(
    main_test_base: HealthBaseSequence,
) -> None:
    clean = _identity(main_test_base)
    faulted = generate_health_observations(
        main_test_base,
        fault=HealthFaultSpec(
            "additive-position-bias",
            "lidar",
            "y",
            "m",
            3.0,
            schedule="cold_start",
        ),
    )

    np.testing.assert_allclose(
        faulted.lidar_value_xy_m[:24, :, 1] - clean.lidar_value_xy_m[:24, :, 1],
        3.0,
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_array_equal(
        faulted.lidar_value_xy_m[24:],
        clean.lidar_value_xy_m[24:],
    )


def test_clean_bounded_acceleration_changes_truth_not_observation_metadata(
    main_test_base: HealthBaseSequence,
) -> None:
    maneuver = build_bounded_acceleration_control(main_test_base)
    observed = generate_health_observations(
        maneuver,
        fault=HealthFaultSpec(
            "clean-predictor-mismatch",
            "none",
            "motion",
            "m/s^2",
            8.0,
        ),
    )

    assert observed.sequence_id == main_test_base.sequence_id
    np.testing.assert_array_equal(observed.truth_xy_m, maneuver.truth_xy_m)
    np.testing.assert_array_equal(
        observed.camera_reported_times_s,
        maneuver.reference_times_s,
    )
    np.testing.assert_array_equal(
        observed.lidar_reported_times_s,
        maneuver.reference_times_s,
    )
    assert np.all(observed.camera_available)
    assert np.all(observed.lidar_available)


def test_scorer_adapter_structurally_excludes_truth_and_condition_metadata(
    main_test_base: HealthBaseSequence,
) -> None:
    observed = generate_health_observations(
        main_test_base,
        fault=HealthFaultSpec(
            "calibration-translation",
            "camera",
            "x",
            "m",
            3.0,
        ),
    )
    frame = observed.health_frame_inputs()[12]

    assert isinstance(frame, HealthFrameInput)
    assert {item.name for item in fields(frame)} == {
        "reference_time_s",
        "camera_available",
        "lidar_available",
        "objects",
    }
    assert isinstance(frame.objects[0], ObjectHealthInput)
    assert {item.name for item in fields(frame.objects[0])} == {
        "object_id",
        "camera",
        "lidar",
    }
    assert isinstance(frame.objects[0].camera, ModalityMeasurement)
    assert {item.name for item in fields(frame.objects[0].camera)} == {
        "value_xy_m",
        "reported_covariance_xy_m2",
        "reported_time_s",
    }
    forbidden = {
        "truth",
        "actual_covariance",
        "velocity",
        "fault",
        "target",
        "severity",
        "split",
        "seed",
        "sequence_id",
        "frame_index",
    }
    assert forbidden.isdisjoint(item.name for item in fields(frame))

    eligibility = np.array(observed.eligibility_mask, copy=True)
    eligibility[0, 0] = False
    reduced = replace(observed, eligibility_mask=eligibility)
    assert len(reduced.health_frame_inputs()[0].objects) == 5


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"sequence_id": ""}, "nonempty"),
        ({"health_frame": "ego-bev"}, "persistent scene"),
        ({"object_ids": ()}, "nonempty"),
        (
            {
                "object_ids": (
                    "object:00",
                    "object:00",
                    "object:02",
                    "object:03",
                    "object:04",
                    "object:05",
                )
            },
            "UTF-8",
        ),
        ({"frame_indices": np.arange(48, dtype=np.float64)}, "integer vector"),
        ({"frame_indices": np.arange(48, dtype=np.int64) + 1}, "zero-based"),
        ({"reference_times_s": np.zeros(47)}, "shape"),
        ({"reference_times_s": np.full(48, np.inf)}, "finite"),
        ({"reference_times_s": np.zeros(48)}, "strictly increasing"),
        ({"truth_xy_m": np.zeros((47, 6, 2))}, "shape"),
        ({"truth_xy_m": np.full((48, 6, 2), np.nan)}, "finite"),
        (
            {"camera_actual_covariance_xy_m2": np.zeros((47, 6, 2, 2))},
            "shape",
        ),
        (
            {
                "camera_actual_covariance_xy_m2": np.full(
                    (48, 6, 2, 2),
                    np.inf,
                )
            },
            "finite",
        ),
        (
            {
                "camera_actual_covariance_xy_m2": np.broadcast_to(
                    np.asarray(((1.0, 1.0), (0.0, 1.0))),
                    (48, 6, 2, 2),
                )
            },
            "symmetric",
        ),
        (
            {
                "camera_actual_covariance_xy_m2": np.broadcast_to(
                    np.asarray(((1.0, 0.0), (0.0, 0.0))),
                    (48, 6, 2, 2),
                )
            },
            "positive-definite",
        ),
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
            "eligible object",
        ),
        ({"camera_available": np.ones(47, dtype=np.bool_)}, "shape"),
    ],
)
def test_observation_sequence_validates_every_role_and_shape(
    main_test_base: HealthBaseSequence,
    updates: dict[str, object],
    match: str,
) -> None:
    observed = _identity(main_test_base)

    with pytest.raises(ValueError, match=match):
        replace(observed, **updates)  # type: ignore[arg-type]


def test_observation_generator_rejects_population_operator_mismatch(
    main_test_base: HealthBaseSequence,
    edge_test_base: HealthBaseSequence,
) -> None:
    with pytest.raises(ValueError, match="edge test"):
        generate_health_observations(
            main_test_base,
            fault=HealthFaultSpec(
                "common-mode-position-bias",
                "both",
                "x",
                "m",
                1.0,
            ),
        )
    with pytest.raises(ValueError, match="main population"):
        generate_health_observations(
            edge_test_base,
            fault=HealthFaultSpec(
                "calibration-yaw",
                "camera",
                "yaw",
                "rad",
                0.06,
            ),
        )
    maneuver = build_bounded_acceleration_control(main_test_base)
    with pytest.raises(ValueError, match="reserved"):
        generate_health_observations(
            maneuver,
            fault=HealthFaultSpec(
                "identity",
                "none",
                "none",
                "identity",
                0.0,
            ),
        )
    with pytest.raises(ValueError, match="requires the maneuver"):
        generate_health_observations(
            main_test_base,
            fault=HealthFaultSpec(
                "clean-predictor-mismatch",
                "none",
                "motion",
                "m/s^2",
                8.0,
            ),
        )

    profile = load_procedural_profile(_MAIN_PROFILE)
    validation = generate_health_base_sequences(
        profile,
        split="validation",
        sequence_count=2,
        data_master_seed=1729,
    )[0]
    with pytest.raises(ValueError, match="test split"):
        generate_health_observations(
            validation,
            fault=HealthFaultSpec(
                "additive-position-bias",
                "lidar",
                "y",
                "m",
                3.0,
                schedule="cold_start",
            ),
        )
