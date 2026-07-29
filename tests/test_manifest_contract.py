from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AnalyticCrossoverManifest,
    AvailabilityControlManifest,
    CommonModeControlManifest,
    ExperimentManifestV1Alpha1,
    GeometryCrossoverManifest,
)

Validator = Callable[[dict[str, Any]], ExperimentManifestV1Alpha1]
DIGEST = "a" * 64


def _procedural_source() -> dict[str, Any]:
    return {
        "kind": "procedural",
        "split": "test",
        "profile_id": "constant-velocity-front-roi-v1",
        "profile_sha256": DIGEST,
        "sequence_count": 100,
    }


def _roi() -> dict[str, Any]:
    return {
        "frame": "ego-bev",
        "x_min_m": 5.0,
        "x_max_m": 60.0,
        "abs_y_max_m": 30.0,
        "camera_half_fov_rad": 0.6108652381980153,
    }


def _bootstrap() -> dict[str, Any]:
    return {
        "method": "percentile",
        "unit": "sequence",
        "resampling": "paired-indices-across-severities-and-methods",
        "interval_scope": "pointwise",
        "replicates": 1000,
        "confidence_level": 0.95,
    }


def test_example_manifest_round_trips_with_public_aliases(
    manifest_data: dict[str, Any], validate_manifest: Validator
) -> None:
    manifest = validate_manifest(manifest_data)

    assert isinstance(manifest, AnalyticCrossoverManifest)
    assert manifest.schema_id == "ffb.manifest/v1alpha1"
    assert manifest.healthy_modality == "lidar"
    assert manifest.model_dump(mode="json") == manifest_data


def test_healthy_modality_is_derived_from_fault_target(
    manifest_data: dict[str, Any], validate_manifest: Validator
) -> None:
    manifest_data["fault_sweep"]["target"] = "lidar"
    manifest = validate_manifest(manifest_data)

    assert isinstance(manifest, AnalyticCrossoverManifest)
    assert manifest.healthy_modality == "camera"


def test_identity_observation_uncertainty_must_be_calibrated(
    manifest_data: dict[str, Any], validate_manifest: Validator
) -> None:
    manifest_data["observations"]["camera"]["reported_std_xy_m"] = [1.4, 0.6]

    with pytest.raises(ValidationError, match="must match at identity"):
        validate_manifest(manifest_data)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("evaluation", "crossover", "zero_tolerance_m2"), 1e300),
        (("evaluation", "oracle_recovery_denominator_tolerance_m2"), 1e300),
        (("analytic_validation", "monte_carlo_standard_error_multiplier"), 7.0),
        (("rng", "data_master_seed"), 2**128),
        (("evaluation", "bootstrap", "replicates"), 201),
        (("evaluation", "bootstrap", "confidence_level"), 0.9499),
    ],
)
def test_scientific_tolerances_and_seed_width_are_fixed(
    manifest_data: dict[str, Any],
    validate_manifest: Validator,
    path: tuple[str, ...],
    value: float | int,
) -> None:
    target: dict[str, Any] = manifest_data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValidationError):
        validate_manifest(manifest_data)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda data: data.update({"unexpected": "value"}),
            "Extra inputs are not permitted",
        ),
        (
            lambda data: data["rng"].update({"bootstrap_seed": 1729}),
            "must differ",
        ),
        (
            lambda data: data["source"].update({"sequence_count": 1}),
            "greater than or equal to 2",
        ),
        (
            lambda data: data["fault_sweep"].update({"magnitude_values_m": [0.1, 0.2]}),
            "identity magnitude 0",
        ),
        (
            lambda data: data["fault_sweep"].update({"magnitude_values_m": [0.0, -0.2]}),
            "non-negative",
        ),
        (
            lambda data: data["fault_sweep"].update({"magnitude_values_m": [0.0, 0.5, 0.25]}),
            "strictly increasing",
        ),
        (
            lambda data: data["fault_sweep"].update({"magnitude_values_m": [-0.0, 0.2]}),
            "canonical positive zero",
        ),
    ],
)
def test_manifest_rejects_invalid_core_semantics(
    manifest_data: dict[str, Any],
    validate_manifest: Validator,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    mutate(manifest_data)

    with pytest.raises(ValidationError, match=message):
        validate_manifest(manifest_data)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"axis": "time"}, "Input should be 'x' or 'y'"),
        ({"unit": "s"}, "Input should be 'm'"),
        ({"injection_site": "timestamp-metadata"}, "estimator-output"),
        ({"direction_policy": "positive"}, "symmetric-paired"),
    ],
)
def test_additive_bias_has_no_invalid_axis_unit_or_site_combinations(
    manifest_data: dict[str, Any],
    validate_manifest: Validator,
    update: dict[str, Any],
    message: str,
) -> None:
    manifest_data["fault_sweep"].update(update)

    with pytest.raises(ValidationError, match=message):
        validate_manifest(manifest_data)


def test_correctly_reported_noise_contract_is_fixed(
    manifest_data: dict[str, Any], validate_manifest: Validator
) -> None:
    manifest_data["fault_sweep"] = {
        "kind": "increased-noise-correctly-reported",
        "target": "camera",
        "axis": "xy",
        "unit": "std-scale",
        "injection_site": "true-error-model",
        "reported_uncertainty": "tracks-actual",
        "persistence": "sequence-configuration",
        "std_scale_values": [1.0, 1.5, 2.0],
    }
    validate_manifest(manifest_data)

    manifest_data["fault_sweep"]["reported_uncertainty"] = "nominal"
    with pytest.raises(ValidationError):
        validate_manifest(manifest_data)

    manifest_data["fault_sweep"]["reported_uncertainty"] = "tracks-actual"
    manifest_data["fault_sweep"]["std_scale_values"] = [0.0, 1.5]
    with pytest.raises(ValidationError, match="identity scale 1"):
        validate_manifest(manifest_data)


def test_underreported_noise_contract_is_fixed(
    manifest_data: dict[str, Any], validate_manifest: Validator
) -> None:
    manifest_data["fault_sweep"] = {
        "kind": "increased-noise-underreported",
        "target": "lidar",
        "axis": "xy",
        "unit": "std-scale",
        "injection_site": "true-error-model",
        "reported_uncertainty": "nominal",
        "persistence": "sequence-configuration",
        "std_scale_values": [1.0, 2.0],
    }

    manifest = validate_manifest(manifest_data)
    assert isinstance(manifest, AnalyticCrossoverManifest)
    assert manifest.healthy_modality == "camera"


@pytest.mark.parametrize(
    "fault_kind",
    ["calibration-yaw", "calibration-translation", "timestamp-offset", "dropout"],
)
def test_analytic_crossover_rejects_geometry_and_availability_faults(
    manifest_data: dict[str, Any],
    validate_manifest: Validator,
    fault_kind: str,
) -> None:
    manifest_data["fault_sweep"]["kind"] = fault_kind

    with pytest.raises(ValidationError, match="does not match any of the expected tags"):
        validate_manifest(manifest_data)


def test_primary_methods_are_exact_and_health_gate_is_not_configurable(
    manifest_data: dict[str, Any], validate_manifest: Validator
) -> None:
    manifest_data["methods"][-1] = "health-gated"

    with pytest.raises(ValidationError):
        validate_manifest(manifest_data)


def test_geometry_manifest_accepts_only_geometry_compatible_faults(
    manifest_data: dict[str, Any], validate_manifest: Validator
) -> None:
    manifest_data.pop("analytic_validation")
    manifest_data.update(
        {
            "kind": "geometry-crossover",
            "source": _procedural_source(),
            "roi": _roi(),
            "fault_sweep": {
                "kind": "calibration-yaw",
                "target": "camera",
                "axis": "yaw",
                "unit": "rad",
                "injection_site": "calibration-metadata",
                "perturbation_frame": "ego",
                "direction_policy": "symmetric-paired",
                "persistence": "sequence",
                "magnitude_values_rad": [0.0, 0.01, 0.02],
            },
        }
    )

    manifest = validate_manifest(manifest_data)
    assert isinstance(manifest, GeometryCrossoverManifest)
    assert manifest.healthy_modality == "lidar"

    manifest_data["fault_sweep"]["axis"] = "time"
    with pytest.raises(ValidationError):
        validate_manifest(manifest_data)


def test_timestamp_fault_requires_geometry_temporal_manifest(
    manifest_data: dict[str, Any], validate_manifest: Validator
) -> None:
    manifest_data.pop("analytic_validation")
    manifest_data.update(
        {
            "kind": "geometry-crossover",
            "source": _procedural_source(),
            "roi": _roi(),
            "fault_sweep": {
                "kind": "timestamp-offset",
                "target": "lidar",
                "axis": "time",
                "unit": "s",
                "injection_site": "timestamp-metadata",
                "timestamp_convention": "reported-minus-true",
                "direction_policy": "symmetric-paired",
                "persistence": "sequence",
                "magnitude_values_s": [0.0, 0.05, 0.1],
            },
        }
    )

    assert isinstance(validate_manifest(manifest_data), GeometryCrossoverManifest)


def test_common_mode_has_no_healthy_reference_or_oracles(
    manifest_data: dict[str, Any], validate_manifest: Validator
) -> None:
    manifest_data.pop("analytic_validation")
    manifest_data.update(
        {
            "kind": "common-mode-control",
            "source": _procedural_source(),
            "roi": _roi(),
            "fault_sweep": {
                "kind": "common-mode-position-bias",
                "target": "both",
                "axis": "x",
                "unit": "m",
                "injection_site": "estimator-output",
                "direction_policy": "symmetric-paired",
                "persistence": "sequence",
                "magnitude_values_m": [0.0, 0.5, 1.0],
            },
            "methods": ["camera-only", "lidar-only", "fixed-fusion"],
            "evaluation": {
                "mode": "common-mode-blind-spot-control",
                "primary_loss": "matched-center-mse",
                "loss_unit": "m^2",
                "aggregation": "object-frame-mean-then-sequence-mean",
                "primary_contrast": "none",
                "bootstrap": _bootstrap(),
                "crossover": "not-applicable",
            },
        }
    )

    manifest = validate_manifest(manifest_data)
    assert isinstance(manifest, CommonModeControlManifest)

    manifest_data["evaluation"]["primary_contrast"] = "fused-minus-healthy"
    with pytest.raises(ValidationError):
        validate_manifest(manifest_data)


def test_dropout_uses_availability_metrics_not_crossover(
    manifest_data: dict[str, Any], validate_manifest: Validator
) -> None:
    manifest_data.pop("analytic_validation")
    manifest_data.update(
        {
            "kind": "availability-control",
            "source": _procedural_source(),
            "roi": _roi(),
            "fault_sweep": {
                "kind": "dropout",
                "target": "camera",
                "axis": "availability",
                "unit": "probability",
                "injection_site": "availability",
                "process": "shared-target-modality-frame-bernoulli",
                "probability_values": [0.0, 0.5, 1.0],
            },
            "methods": [
                "camera-only",
                "lidar-only",
                "fixed-fusion",
                "fault-target-drop-policy",
            ],
            "evaluation": {
                "mode": "availability-control",
                "metrics": [
                    "coverage",
                    "conditional-matched-center-mse",
                    "undefined-output-rate",
                ],
                "missing_output_policy": "undefined-no-localization-penalty",
                "rate_aggregation": "count-ratio-with-sequence-bootstrap",
                "conditional_loss_aggregation": (
                    "valid-object-frame-ratio-with-sequence-bootstrap"
                ),
                "undefined_bootstrap_replicate_action": ("exclude-and-require-two-sided-support"),
                "unimodal_missing_input_action": "undefined",
                "fixed_fusion_missing_input_action": "undefined",
                "target_drop_identity_action": "fixed-fusion",
                "target_drop_nonidentity_action": "use-nontarget-modality",
                "bootstrap": _bootstrap(),
                "crossover": "not-applicable",
            },
        }
    )

    manifest = validate_manifest(manifest_data)
    assert isinstance(manifest, AvailabilityControlManifest)

    manifest_data["fault_sweep"]["probability_values"] = [0.0, 1.1]
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        validate_manifest(manifest_data)


def test_bootstrap_manifests_require_at_least_two_sequences(
    manifest_data: dict[str, Any], validate_manifest: Validator
) -> None:
    manifest_data.pop("analytic_validation")
    manifest_data.update(
        {
            "kind": "common-mode-control",
            "source": {**_procedural_source(), "sequence_count": 1},
            "roi": _roi(),
            "fault_sweep": {
                "kind": "common-mode-position-bias",
                "target": "both",
                "axis": "x",
                "unit": "m",
                "injection_site": "estimator-output",
                "direction_policy": "symmetric-paired",
                "persistence": "sequence",
                "magnitude_values_m": [0.0, 0.5],
            },
            "methods": ["camera-only", "lidar-only", "fixed-fusion"],
            "evaluation": {
                "mode": "common-mode-blind-spot-control",
                "primary_loss": "matched-center-mse",
                "loss_unit": "m^2",
                "aggregation": "object-frame-mean-then-sequence-mean",
                "primary_contrast": "none",
                "bootstrap": _bootstrap(),
                "crossover": "not-applicable",
            },
        }
    )

    with pytest.raises(ValidationError, match="greater than or equal to 2"):
        validate_manifest(manifest_data)


def test_nuscenes_scene_selection_is_nonempty_and_unique(
    manifest_data: dict[str, Any], validate_manifest: Validator
) -> None:
    geometry = copy.deepcopy(manifest_data)
    geometry.pop("analytic_validation")
    geometry.update(
        {
            "kind": "geometry-crossover",
            "source": {
                "kind": "nuscenes-mini",
                "adapter_profile": "nuscenes-mini-matched-centers-v1",
                "scene_names": ["scene-0061", "scene-0103"],
                "camera_channel": "CAM_FRONT",
            },
            "roi": _roi(),
        }
    )
    assert isinstance(validate_manifest(geometry), GeometryCrossoverManifest)

    geometry["source"]["scene_names"] = []
    with pytest.raises(ValidationError):
        validate_manifest(geometry)

    geometry["source"]["scene_names"] = ["scene-0061", "scene-0061"]
    with pytest.raises(ValidationError, match="must be unique"):
        validate_manifest(geometry)
