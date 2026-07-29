from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from fusion_fault_bench.contracts.procedural_validation_v1 import ProceduralValidationV1


def _oracle(check_id: str, unit: str = "m") -> dict[str, Any]:
    return {
        "check_id": check_id,
        "unit": unit,
        "maximum_absolute_discrepancy": 0.0,
        "tolerance": 1e-12,
        "passed": True,
    }


def _moment_checks() -> tuple[dict[str, Any], ...]:
    checks: list[dict[str, Any]] = []
    for sensor in ("camera", "lidar"):
        for coordinate in ("x", "y"):
            for statistic in ("mean", "variance"):
                checks.append(
                    {
                        "check_id": f"{sensor}-{coordinate}-{statistic}",
                        "statistic": statistic,
                        "sensor_a": sensor,
                        "coordinate_a": coordinate,
                        "sensor_b": None,
                        "coordinate_b": None,
                        "sample_count": 96,
                        "ddof": 1,
                        "expectation": 0.0,
                        "observed_value": 0.0,
                        "six_standard_error_bound": 1.0,
                        "absolute_discrepancy": 0.0,
                        "unit": "m" if statistic == "mean" else "m^2",
                        "passed": True,
                    }
                )
        checks.append(
            {
                "check_id": f"{sensor}-xy-covariance",
                "statistic": "within-sensor-covariance",
                "sensor_a": sensor,
                "coordinate_a": "x",
                "sensor_b": sensor,
                "coordinate_b": "y",
                "sample_count": 96,
                "ddof": 1,
                "expectation": 0.0,
                "observed_value": 0.0,
                "six_standard_error_bound": 1.0,
                "absolute_discrepancy": 0.0,
                "unit": "m^2",
                "passed": True,
            }
        )
    for camera_coordinate in ("x", "y"):
        for lidar_coordinate in ("x", "y"):
            checks.append(
                {
                    "check_id": (f"camera-{camera_coordinate}-lidar-{lidar_coordinate}-covariance"),
                    "statistic": "camera-lidar-cross-covariance",
                    "sensor_a": "camera",
                    "coordinate_a": camera_coordinate,
                    "sensor_b": "lidar",
                    "coordinate_b": lidar_coordinate,
                    "sample_count": 96,
                    "ddof": 1,
                    "expectation": 0.0,
                    "observed_value": 0.0,
                    "six_standard_error_bound": 1.0,
                    "absolute_discrepancy": 0.0,
                    "unit": "m^2",
                    "passed": True,
                }
            )
    return tuple(checks)


def _expected_loss_checks() -> tuple[dict[str, Any], ...]:
    severity = {
        "index": 0,
        "magnitude": 0.0,
        "direction": "identity",
        "unit": "m",
    }
    checks = [
        {
            "check_id": f"identity-{method}-loss",
            "fault_family": "calibration-translation",
            "fault_axis": "x",
            "severity": severity,
            "method_id": method,
            "metric_name": "matched-center-mse",
            "expected_value_m2": 2.0,
            "empirical_value_m2": 2.0,
            "analytic_standard_error_m2": 0.2,
            "absolute_standardized_error": 0.0,
            "standard_error_multiplier": 6.0,
            "passed": True,
        }
        for method in (
            "camera-only",
            "lidar-only",
            "fixed-fusion",
            "fault-target-drop-policy",
        )
    ]
    checks.append(
        {
            "check_id": "identity-fixed-fusion-contrast",
            "fault_family": "calibration-translation",
            "fault_axis": "x",
            "severity": severity,
            "method_id": "fixed-fusion",
            "metric_name": "fused-minus-healthy",
            "expected_value_m2": -0.1,
            "empirical_value_m2": -0.1,
            "analytic_standard_error_m2": 0.1,
            "absolute_standardized_error": 0.0,
            "standard_error_multiplier": 6.0,
            "passed": True,
        }
    )
    return tuple(checks)


def _validation_mapping() -> dict[str, Any]:
    return {
        "schema": "ffb.procedural-validation/v1",
        "run_id": "run:test",
        "manifest_sha256": "a" * 64,
        "profile_id": "constant-velocity-ci-smoke-v1",
        "profile_sha256": "b" * 64,
        "split": "test",
        "sequence_count": 4,
        "frame_count": 8,
        "object_count": 3,
        "total_eligible_object_frame_count": 96,
        "profile_checks": {
            "schema_valid": True,
            "profile_id_valid": True,
            "profile_digest_valid": True,
            "split_count_valid": True,
            "roi_valid": True,
            "isotropic_yaw_compatible": True,
            "split_family_support_valid": True,
            "canonical_ordering_valid": True,
            "all_checks_passed": True,
        },
        "eligibility": {
            "ordered_sequence_commitments_sha256": "c" * 64,
            "minimum_eligible_object_frame_count": 24,
            "maximum_eligible_object_frame_count": 24,
            "total_eligible_object_frame_count": 96,
            "eligibility_invariant": True,
        },
        "oracle_checks": {
            "identity_center": _oracle("identity-center"),
            "calibration_translation_center": _oracle("calibration-translation-center"),
            "translation_bias_equivalence_center": _oracle("translation-bias-equivalence-center"),
            "translation_bias_equivalence_sequence_loss": _oracle(
                "translation-bias-equivalence-sequence-loss",
                "m^2",
            ),
            "calibration_yaw_center": _oracle("calibration-yaw-center"),
            "timestamp_alignment_center": _oracle("timestamp-alignment-center"),
            "static_timestamp_center": _oracle("static-timestamp-center"),
            "fault_cancellation_mutation_rejected": True,
            "all_checks_passed": True,
        },
        "moment_checks": _moment_checks(),
        "expected_loss_checks": _expected_loss_checks(),
        "dropout_validation": {"status": "not-applicable"},
        "identity_comparison": {
            "status": "deferred-to-matrix",
            "reason": "cross-manifest-identity-requires-complete-matrix",
        },
        "common_mode_validation": {"status": "not-applicable"},
        "deterministic_model_checks": {
            "reported_covariance_behavior": "passed",
            "expected_curve_response": "passed",
            "complete_sequence_performance_oracle": "passed",
            "identity_row_reconstruction": "passed",
            "all_checks_passed": True,
        },
        "resources": {
            "implied_sequence_row_count": 60,
            "sequence_row_cap": 2_000_000,
            "implied_bootstrap_cell_count": 800,
            "bootstrap_cell_cap": 20_000_000,
            "sequence_count": 4,
            "sequence_count_cap": 10_000,
            "bootstrap_replicates": 200,
            "bootstrap_replicate_cap": 20_000,
            "sequence_rows_within_cap": True,
            "bootstrap_cells_within_cap": True,
            "sequence_count_within_cap": True,
            "bootstrap_replicates_within_cap": True,
            "all_checks_passed": True,
        },
        "all_checks_passed": True,
    }


def test_complete_validation_contract_accepts_exact_consistent_evidence() -> None:
    validation = ProceduralValidationV1.model_validate(_validation_mapping())

    assert validation.all_checks_passed
    assert validation.resources.implied_sequence_row_count == 60


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (("profile_checks", "all_checks_passed"), "conjunction"),
        (("oracle_checks", "identity_center", "passed"), "pass flag"),
        (("moment_checks", 0, "absolute_discrepancy"), "moment discrepancy"),
        (
            ("deterministic_model_checks", "all_checks_passed"),
            "deterministic model gates",
        ),
        (("resources", "bootstrap_cells_within_cap"), "resource pass flags"),
        (("all_checks_passed",), "procedural all_checks_passed"),
    ],
)
def test_validation_contract_rejects_contradictory_pass_evidence(
    path: tuple[str | int, ...],
    message: str,
) -> None:
    value = copy.deepcopy(_validation_mapping())
    cursor: Any = value
    for component in path[:-1]:
        cursor = cursor[component]
    terminal = path[-1]
    if terminal == "absolute_discrepancy":
        cursor[terminal] = 1.0
    else:
        cursor[terminal] = False

    with pytest.raises(ValidationError, match=message):
        ProceduralValidationV1.model_validate(value)


def test_validation_contract_rejects_incomplete_semantic_evidence_sets() -> None:
    missing_moment = _validation_mapping()
    missing_moment["moment_checks"] = missing_moment["moment_checks"][:-1]
    with pytest.raises(ValidationError, match="exact frozen 14-row suite"):
        ProceduralValidationV1.model_validate(missing_moment)

    missing_affine_method = _validation_mapping()
    missing_affine_method["expected_loss_checks"] = missing_affine_method["expected_loss_checks"][
        :-1
    ]
    with pytest.raises(ValidationError, match="exact affine method set"):
        ProceduralValidationV1.model_validate(missing_affine_method)


def test_validation_contract_rejects_wrong_named_or_counted_evidence() -> None:
    wrong_eligibility = _validation_mapping()
    wrong_eligibility["eligibility"]["minimum_eligible_object_frame_count"] = 25
    with pytest.raises(ValidationError, match="eligibility maximum"):
        ProceduralValidationV1.model_validate(wrong_eligibility)

    wrong_oracle = _validation_mapping()
    wrong_oracle["oracle_checks"]["calibration_yaw_center"]["check_id"] = "wrong-yaw"
    with pytest.raises(ValidationError, match="frozen check IDs"):
        ProceduralValidationV1.model_validate(wrong_oracle)

    duplicate_loss = _validation_mapping()
    duplicate_loss["expected_loss_checks"] = (
        *duplicate_loss["expected_loss_checks"],
        duplicate_loss["expected_loss_checks"][0],
    )
    with pytest.raises(ValidationError, match="check IDs must be unique"):
        ProceduralValidationV1.model_validate(duplicate_loss)

    wrong_sample_count = _validation_mapping()
    wrong_sample_count["moment_checks"][0]["sample_count"] = 95
    with pytest.raises(ValidationError, match="complete eligible population"):
        ProceduralValidationV1.model_validate(wrong_sample_count)


def test_edge_common_mode_can_explicitly_exclude_cross_manifest_identity() -> None:
    value = _validation_mapping()
    value["identity_comparison"] = {
        "status": "not-applicable",
        "reason": "edge-common-mode-has-no-comparable-profile-peer",
    }
    value["common_mode_validation"] = {
        "status": "applicable",
        "maximum_disagreement_discrepancy_m": 0.0,
        "tolerance_m": 1e-12,
        "passed": True,
    }
    value["expected_loss_checks"] = tuple(
        check
        for check in value["expected_loss_checks"]
        if check["metric_name"] == "matched-center-mse"
        and check["method_id"] != "fault-target-drop-policy"
    )
    value["deterministic_model_checks"]["complete_sequence_performance_oracle"] = "not-applicable"
    value["deterministic_model_checks"]["identity_row_reconstruction"] = "not-applicable"

    validation = ProceduralValidationV1.model_validate(value)

    assert validation.identity_comparison.status == "not-applicable"
