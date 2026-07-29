from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.bundle_v1alpha1 import (
    ConditionKey,
    expected_conditions,
)
from fusion_fault_bench.contracts.io import validate_manifest_mapping
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AvailabilityControlManifest,
    CommonModeControlManifest,
    GeometryCrossoverManifest,
)
from fusion_fault_bench.contracts.procedural_profile_v1 import (
    load_procedural_profile,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    LocalizationMetricRecord,
    RateMetricRecord,
)
from fusion_fault_bench.experiments.procedural import (
    ProceduralConditionOutputs,
    generate_procedural_condition_outputs,
    generate_procedural_sequence_metric_rows,
    generate_procedural_sequence_metrics,
)
from fusion_fault_bench.procedural_evaluation import evaluate_procedural_records
from fusion_fault_bench.reference.procedural import (
    timestamp_displacement_xy,
    yaw_displacement_xy,
)

_TRANSLATION = (1.5, 0.0, 1.5)
_QUATERNION = (0.5, -0.5, 0.5, -0.5)
_TRUTH = np.asarray(
    (
        (10.0, 1.0),
        (12.0, -2.0),
        (11.0, 1.5),
        (13.0, -1.0),
    ),
    dtype=np.float64,
)
_VELOCITY = np.asarray(
    (
        (2.0, 0.5),
        (-1.0, 0.2),
        (2.0, 0.5),
        (-1.0, 0.2),
    ),
    dtype=np.float64,
)
_FRAME_INDICES = np.asarray((0, 0, 1, 1), dtype=np.int64)
_ZEROS = np.zeros((4, 2), dtype=np.float64)
_UNIFORMS = np.asarray((0.2, 0.6), dtype=np.float64)


def _base_data(manifest_data: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(manifest_data)
    data.pop("analytic_validation")
    data["kind"] = "geometry-crossover"
    data["source"] = {
        "kind": "procedural",
        "split": "test",
        "profile_id": "constant-velocity-ci-smoke-v1",
        "profile_sha256": "a" * 64,
        "sequence_count": 2,
    }
    data["roi"] = {
        "frame": "ego-bev",
        "x_min_m": 5.0,
        "x_max_m": 60.0,
        "abs_y_max_m": 40.0,
        "camera_half_fov_rad": 0.7,
    }
    data["observations"]["camera"]["actual_std_xy_m"] = [1.0, 1.0]
    data["observations"]["camera"]["reported_std_xy_m"] = [1.0, 1.0]
    data["observations"]["lidar"]["actual_std_xy_m"] = [0.3, 0.3]
    data["observations"]["lidar"]["reported_std_xy_m"] = [0.3, 0.3]
    data["evaluation"]["bootstrap"]["replicates"] = 200
    return data


def _geometry(
    manifest_data: dict[str, Any],
    fault: dict[str, Any],
) -> GeometryCrossoverManifest:
    data = _base_data(manifest_data)
    data["fault_sweep"] = fault
    manifest = validate_manifest_mapping(data)
    assert isinstance(manifest, GeometryCrossoverManifest)
    return manifest


def _availability(manifest_data: dict[str, Any]) -> AvailabilityControlManifest:
    data = _base_data(manifest_data)
    data["kind"] = "availability-control"
    data["fault_sweep"] = {
        "kind": "dropout",
        "target": "camera",
        "axis": "availability",
        "unit": "probability",
        "injection_site": "availability",
        "process": "shared-target-modality-frame-bernoulli",
        "probability_values": [0.0, 0.25, 0.75, 1.0],
    }
    data["methods"] = [
        "camera-only",
        "lidar-only",
        "fixed-fusion",
        "fault-target-drop-policy",
    ]
    data["evaluation"] = {
        "mode": "availability-control",
        "metrics": [
            "coverage",
            "conditional-matched-center-mse",
            "undefined-output-rate",
        ],
        "missing_output_policy": "undefined-no-localization-penalty",
        "rate_aggregation": "count-ratio-with-sequence-bootstrap",
        "conditional_loss_aggregation": ("valid-object-frame-ratio-with-sequence-bootstrap"),
        "undefined_bootstrap_replicate_action": ("exclude-and-require-two-sided-support"),
        "unimodal_missing_input_action": "undefined",
        "fixed_fusion_missing_input_action": "undefined",
        "target_drop_identity_action": "fixed-fusion",
        "target_drop_nonidentity_action": "use-nontarget-modality",
        "bootstrap": {
            "method": "percentile",
            "unit": "sequence",
            "resampling": "paired-indices-across-severities-and-methods",
            "interval_scope": "pointwise",
            "replicates": 200,
            "confidence_level": 0.95,
        },
        "crossover": "not-applicable",
    }
    manifest = validate_manifest_mapping(data)
    assert isinstance(manifest, AvailabilityControlManifest)
    return manifest


def _common_mode(manifest_data: dict[str, Any]) -> CommonModeControlManifest:
    data = _base_data(manifest_data)
    data["kind"] = "common-mode-control"
    data["fault_sweep"] = {
        "kind": "common-mode-position-bias",
        "target": "both",
        "axis": "x",
        "unit": "m",
        "injection_site": "estimator-output",
        "direction_policy": "symmetric-paired",
        "persistence": "sequence",
        "magnitude_values_m": [0.0, 1.0],
    }
    data["methods"] = ["camera-only", "lidar-only", "fixed-fusion"]
    data["evaluation"] = {
        "mode": "common-mode-blind-spot-control",
        "primary_loss": "matched-center-mse",
        "loss_unit": "m^2",
        "aggregation": "object-frame-mean-then-sequence-mean",
        "primary_contrast": "none",
        "bootstrap": {
            "method": "percentile",
            "unit": "sequence",
            "resampling": "paired-indices-across-severities-and-methods",
            "interval_scope": "pointwise",
            "replicates": 200,
            "confidence_level": 0.95,
        },
        "crossover": "not-applicable",
    }
    manifest = validate_manifest_mapping(data)
    assert isinstance(manifest, CommonModeControlManifest)
    return manifest


def _condition(
    manifest: GeometryCrossoverManifest | AvailabilityControlManifest | CommonModeControlManifest,
    *,
    magnitude: float,
    direction: str,
) -> ConditionKey:
    return next(
        condition
        for condition in expected_conditions(manifest)
        if condition.magnitude == magnitude and condition.direction == direction
    )


def _outputs(
    manifest: GeometryCrossoverManifest | AvailabilityControlManifest | CommonModeControlManifest,
    condition: ConditionKey,
    *,
    camera_normal: np.ndarray = _ZEROS,
    lidar_normal: np.ndarray = _ZEROS,
) -> ProceduralConditionOutputs:
    return generate_procedural_condition_outputs(
        manifest,
        condition=condition,
        truth_xy_m=_TRUTH,
        velocity_xy_mps=_VELOCITY,
        eligible_frame_indices=_FRAME_INDICES,
        camera_standard_normal_xy=camera_normal,
        lidar_standard_normal_xy=lidar_normal,
        fault_uniform_by_frame=_UNIFORMS,
        camera_true_translation_m=_TRANSLATION,
        camera_true_quaternion_wxyz=_QUATERNION,
    )


def test_calibration_identity_translation_and_healthy_invariance(
    manifest_data: dict[str, Any],
) -> None:
    manifest = _geometry(
        manifest_data,
        {
            "kind": "calibration-translation",
            "target": "camera",
            "axis": "x",
            "unit": "m",
            "injection_site": "calibration-metadata",
            "perturbation_frame": "ego",
            "direction_policy": "symmetric-paired",
            "persistence": "sequence",
            "magnitude_values_m": [0.0, 0.5],
        },
    )
    identity = _outputs(manifest, _condition(manifest, magnitude=0.0, direction="identity"))
    positive = _outputs(manifest, _condition(manifest, magnitude=0.5, direction="positive"))
    negative = _outputs(manifest, _condition(manifest, magnitude=0.5, direction="negative"))

    np.testing.assert_allclose(identity.camera_value_xy_m, _TRUTH, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(identity.lidar_value_xy_m, _TRUTH)
    np.testing.assert_array_equal(positive.lidar_value_xy_m, identity.lidar_value_xy_m)
    np.testing.assert_array_equal(negative.lidar_value_xy_m, identity.lidar_value_xy_m)
    np.testing.assert_allclose(
        positive.camera_value_xy_m - identity.camera_value_xy_m,
        np.tile((0.5, 0.0), (4, 1)),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        negative.camera_value_xy_m - identity.camera_value_xy_m,
        np.tile((-0.5, 0.0), (4, 1)),
        rtol=0.0,
        atol=1e-12,
    )


def test_camera_translation_matches_additive_bias_and_lidar_bias_is_exact(
    manifest_data: dict[str, Any],
) -> None:
    calibration = _geometry(
        manifest_data,
        {
            "kind": "calibration-translation",
            "target": "camera",
            "axis": "x",
            "unit": "m",
            "injection_site": "calibration-metadata",
            "perturbation_frame": "ego",
            "direction_policy": "symmetric-paired",
            "persistence": "sequence",
            "magnitude_values_m": [0.0, 1.0],
        },
    )
    camera_bias = _geometry(
        manifest_data,
        {
            "kind": "additive-position-bias",
            "target": "camera",
            "axis": "x",
            "unit": "m",
            "injection_site": "estimator-output",
            "direction_policy": "symmetric-paired",
            "persistence": "sequence",
            "magnitude_values_m": [0.0, 1.0],
        },
    )
    lidar_bias = _geometry(
        manifest_data,
        {
            "kind": "additive-position-bias",
            "target": "lidar",
            "axis": "y",
            "unit": "m",
            "injection_site": "estimator-output",
            "direction_policy": "symmetric-paired",
            "persistence": "sequence",
            "magnitude_values_m": [0.0, 1.0],
        },
    )
    calibration_output = _outputs(
        calibration,
        _condition(calibration, magnitude=1.0, direction="positive"),
    )
    camera_bias_output = _outputs(
        camera_bias,
        _condition(camera_bias, magnitude=1.0, direction="positive"),
    )
    lidar_bias_output = _outputs(
        lidar_bias,
        _condition(lidar_bias, magnitude=1.0, direction="negative"),
    )

    np.testing.assert_allclose(
        calibration_output.camera_value_xy_m,
        camera_bias_output.camera_value_xy_m,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        lidar_bias_output.lidar_value_xy_m - _TRUTH,
        np.tile((0.0, -1.0), (4, 1)),
    )
    np.testing.assert_allclose(
        lidar_bias_output.camera_value_xy_m,
        _TRUTH,
        rtol=0.0,
        atol=1e-12,
    )


def test_yaw_and_timestamp_match_independent_center_oracles(
    manifest_data: dict[str, Any],
) -> None:
    yaw = _geometry(
        manifest_data,
        {
            "kind": "calibration-yaw",
            "target": "camera",
            "axis": "yaw",
            "unit": "rad",
            "injection_site": "calibration-metadata",
            "perturbation_frame": "ego",
            "direction_policy": "symmetric-paired",
            "persistence": "sequence",
            "magnitude_values_rad": [0.0, 0.08],
        },
    )
    timing = _geometry(
        manifest_data,
        {
            "kind": "timestamp-offset",
            "target": "camera",
            "axis": "time",
            "unit": "s",
            "injection_site": "timestamp-metadata",
            "timestamp_convention": "reported-minus-true",
            "direction_policy": "symmetric-paired",
            "persistence": "sequence",
            "magnitude_values_s": [0.0, 0.4],
        },
    )
    yaw_output = _outputs(yaw, _condition(yaw, magnitude=0.08, direction="positive"))
    timing_output = _outputs(
        timing,
        _condition(timing, magnitude=0.4, direction="positive"),
    )
    expected_yaw = np.asarray(
        [yaw_displacement_xy(point, 0.08) for point in _TRUTH],
        dtype=np.float64,
    )
    expected_timing = np.asarray(
        [timestamp_displacement_xy(velocity, 0.4) for velocity in _VELOCITY],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        yaw_output.camera_value_xy_m - _TRUTH,
        expected_yaw,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        timing_output.camera_value_xy_m - _TRUTH,
        expected_timing,
        rtol=0.0,
        atol=1e-12,
    )


def test_yaw_rotates_the_physical_proxy_base_error(
    manifest_data: dict[str, Any],
) -> None:
    manifest = _geometry(
        manifest_data,
        {
            "kind": "calibration-yaw",
            "target": "camera",
            "axis": "yaw",
            "unit": "rad",
            "injection_site": "calibration-metadata",
            "perturbation_frame": "ego",
            "direction_policy": "symmetric-paired",
            "persistence": "sequence",
            "magnitude_values_rad": [0.0, 0.08],
        },
    )
    camera_normal = np.asarray(
        ((0.1, -0.2), (0.3, 0.4), (-0.5, 0.6), (0.7, -0.8)),
        dtype=np.float64,
    )
    positive = _outputs(
        manifest,
        _condition(manifest, magnitude=0.08, direction="positive"),
        camera_normal=camera_normal,
    )
    noisy_truth = _TRUTH + camera_normal
    expected = noisy_truth + np.asarray(
        [yaw_displacement_xy(point, 0.08) for point in noisy_truth],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        positive.camera_value_xy_m,
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_noise_scaling_separates_actual_and_reported_covariance(
    manifest_data: dict[str, Any],
) -> None:
    correct = _geometry(
        manifest_data,
        {
            "kind": "increased-noise-correctly-reported",
            "target": "camera",
            "axis": "xy",
            "unit": "std-scale",
            "injection_site": "true-error-model",
            "reported_uncertainty": "tracks-actual",
            "persistence": "sequence-configuration",
            "std_scale_values": [1.0, 2.0],
        },
    )
    underreported = _geometry(
        manifest_data,
        {
            "kind": "increased-noise-underreported",
            "target": "camera",
            "axis": "xy",
            "unit": "std-scale",
            "injection_site": "true-error-model",
            "reported_uncertainty": "nominal",
            "persistence": "sequence-configuration",
            "std_scale_values": [1.0, 2.0],
        },
    )
    generator = np.random.Generator(np.random.PCG64DXSM(8103))
    normals = generator.standard_normal((20_000, 2), dtype=np.float64)
    truth = np.tile((20.0, 0.0), (20_000, 1))
    velocity = np.zeros_like(truth)
    frame_indices = np.zeros(20_000, dtype=np.int64)
    common = {
        "truth_xy_m": truth,
        "velocity_xy_mps": velocity,
        "eligible_frame_indices": frame_indices,
        "camera_standard_normal_xy": normals,
        "lidar_standard_normal_xy": np.zeros_like(normals),
        "fault_uniform_by_frame": np.asarray((0.5,)),
        "camera_true_translation_m": _TRANSLATION,
        "camera_true_quaternion_wxyz": _QUATERNION,
    }
    correct_output = generate_procedural_condition_outputs(
        correct,
        condition=_condition(correct, magnitude=2.0, direction="increase"),
        **common,
    )
    underreported_output = generate_procedural_condition_outputs(
        underreported,
        condition=_condition(underreported, magnitude=2.0, direction="increase"),
        **common,
    )
    residual = correct_output.camera_value_xy_m - truth

    np.testing.assert_array_equal(
        correct_output.camera_value_xy_m,
        underreported_output.camera_value_xy_m,
    )
    np.testing.assert_array_equal(
        correct_output.camera_reported_variance_xy_m2,
        np.asarray((4.0, 4.0)),
    )
    np.testing.assert_array_equal(
        underreported_output.camera_reported_variance_xy_m2,
        np.asarray((1.0, 1.0)),
    )
    np.testing.assert_allclose(
        np.cov(residual, rowvar=False, ddof=1),
        np.eye(2) * 4.0,
        rtol=0.04,
        atol=0.18,
    )


def test_performance_oracle_selects_one_complete_sequence_candidate(
    manifest_data: dict[str, Any],
) -> None:
    manifest = _geometry(
        manifest_data,
        {
            "kind": "additive-position-bias",
            "target": "camera",
            "axis": "x",
            "unit": "m",
            "injection_site": "estimator-output",
            "direction_policy": "symmetric-paired",
            "persistence": "sequence",
            "magnitude_values_m": [0.0, 1.0],
        },
    )
    manifest = manifest.model_copy(
        update={
            "observations": manifest.observations.model_copy(
                update={
                    "lidar": manifest.observations.lidar.model_copy(
                        update={
                            "actual_std_xy_m": (1.0, 1.0),
                            "reported_std_xy_m": (1.0, 1.0),
                        }
                    )
                }
            )
        }
    )
    truth = np.zeros((2, 2), dtype=np.float64)
    rows = generate_procedural_sequence_metric_rows(
        manifest,
        run_id="oracle-test",
        sequence_id="procedural:constant-velocity-ci-smoke-v1:test:000000",
        truth_xy_m=truth,
        velocity_xy_mps=np.zeros_like(truth),
        eligible_frame_indices=np.asarray((0, 1), dtype=np.int64),
        camera_standard_normal_xy=np.asarray(((0.0, 0.0), (10.0, 0.0))),
        lidar_standard_normal_xy=np.asarray(((10.0, 0.0), (0.0, 0.0))),
        fault_uniform_by_frame=np.asarray((0.2, 0.8)),
        camera_true_translation_m=_TRANSLATION,
        camera_true_quaternion_wxyz=_QUATERNION,
    )
    oracle = next(
        row for row in rows if row.severity.index == 0 and row.method_id == "performance-oracle"
    )
    assert isinstance(oracle, LocalizationMetricRecord)
    assert oracle.value == 25.0


def test_dropout_masks_are_frame_shared_nested_and_explicit_at_endpoints(
    manifest_data: dict[str, Any],
) -> None:
    manifest = _availability(manifest_data)
    identity = _outputs(manifest, _condition(manifest, magnitude=0.0, direction="identity"))
    mild = _outputs(manifest, _condition(manifest, magnitude=0.25, direction="increase"))
    severe = _outputs(manifest, _condition(manifest, magnitude=0.75, direction="increase"))
    full = _outputs(manifest, _condition(manifest, magnitude=1.0, direction="increase"))

    np.testing.assert_array_equal(identity.camera_available, np.ones(4, dtype=np.bool_))
    np.testing.assert_array_equal(
        mild.camera_available,
        np.asarray((False, False, True, True)),
    )
    np.testing.assert_array_equal(severe.camera_available, np.zeros(4, dtype=np.bool_))
    np.testing.assert_array_equal(full.camera_available, np.zeros(4, dtype=np.bool_))
    assert np.all(severe.camera_available <= mild.camera_available)
    assert np.all(mild.camera_available <= identity.camera_available)
    np.testing.assert_array_equal(full.lidar_available, np.ones(4, dtype=np.bool_))

    rows = generate_procedural_sequence_metric_rows(
        manifest,
        run_id="dropout-test",
        sequence_id="procedural:constant-velocity-ci-smoke-v1:test:000000",
        truth_xy_m=_TRUTH,
        velocity_xy_mps=_VELOCITY,
        eligible_frame_indices=_FRAME_INDICES,
        camera_standard_normal_xy=_ZEROS,
        lidar_standard_normal_xy=_ZEROS,
        fault_uniform_by_frame=_UNIFORMS,
        camera_true_translation_m=_TRANSLATION,
        camera_true_quaternion_wxyz=_QUATERNION,
    )
    full_camera_loss = next(
        row
        for row in rows
        if row.severity.magnitude == 1.0
        and row.method_id == "camera-only"
        and row.metric_name == "conditional-matched-center-mse"
    )
    full_target_drop_coverage = next(
        row
        for row in rows
        if row.severity.magnitude == 1.0
        and row.method_id == "fault-target-drop-policy"
        and row.metric_name == "coverage"
    )
    assert isinstance(full_camera_loss, LocalizationMetricRecord)
    assert full_camera_loss.status == "undefined"
    assert full_camera_loss.value is None
    assert isinstance(full_target_drop_coverage, RateMetricRecord)
    assert full_target_drop_coverage.value == 1.0


def test_common_mode_preserves_cross_modal_disagreement(
    manifest_data: dict[str, Any],
) -> None:
    manifest = _common_mode(manifest_data)
    camera_normal = np.asarray(
        ((0.1, -0.2), (0.3, 0.4), (-0.5, 0.6), (0.7, -0.8)),
        dtype=np.float64,
    )
    lidar_normal = np.asarray(
        ((-0.2, 0.1), (0.2, -0.3), (0.4, 0.5), (-0.6, 0.7)),
        dtype=np.float64,
    )
    identity = _outputs(
        manifest,
        _condition(manifest, magnitude=0.0, direction="identity"),
        camera_normal=camera_normal,
        lidar_normal=lidar_normal,
    )
    positive = _outputs(
        manifest,
        _condition(manifest, magnitude=1.0, direction="positive"),
        camera_normal=camera_normal,
        lidar_normal=lidar_normal,
    )

    np.testing.assert_allclose(
        positive.camera_value_xy_m - positive.lidar_value_xy_m,
        identity.camera_value_xy_m - identity.lidar_value_xy_m,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        positive.fixed_fusion_value_xy_m - identity.fixed_fusion_value_xy_m,
        np.tile((1.0, 0.0), (4, 1)),
        rtol=0.0,
        atol=1e-12,
    )


def test_profile_wrapper_consumes_scenario_rows_in_contract_order(
    manifest_data: dict[str, Any],
) -> None:
    profile = load_procedural_profile(Path("examples/profiles/constant-velocity-ci-smoke-v1.json"))
    manifest = _geometry(
        manifest_data,
        {
            "kind": "additive-position-bias",
            "target": "lidar",
            "axis": "y",
            "unit": "m",
            "injection_site": "estimator-output",
            "direction_policy": "symmetric-paired",
            "persistence": "sequence",
            "magnitude_values_m": [0.0, 1.0],
        },
    )
    manifest = manifest.model_copy(
        update={
            "source": manifest.source.model_copy(update={"profile_sha256": sha256_digest(profile)})
        }
    )

    rows = generate_procedural_sequence_metrics(
        manifest,
        profile=profile,
        run_id="wrapper-test",
    )

    assert len(rows) == 2 * len(expected_conditions(manifest)) * len(manifest.methods)
    assert rows[0].sequence_id.endswith(":000000")
    assert rows[-1].sequence_id.endswith(":000001")
    evaluated = evaluate_procedural_records(
        manifest,
        run_id="wrapper-test",
        metrics=rows,
    )
    assert evaluated.metrics == rows
    assert len(evaluated.crossovers) == 2
    wrong_source = manifest.source.model_copy(
        update={"profile_id": "constant-velocity-front-roi-v1"}
    )
    with pytest.raises(ValueError, match="profile ID"):
        generate_procedural_sequence_metrics(
            manifest.model_copy(update={"source": wrong_source}),
            profile=profile,
            run_id="wrapper-test",
        )
