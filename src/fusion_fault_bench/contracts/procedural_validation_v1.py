"""Typed, internally consistent validation evidence for M3 procedural runs."""

from __future__ import annotations

import math
from typing import Annotated, Literal, Self

from pydantic import Field, FiniteFloat, PositiveFloat, model_validator

from fusion_fault_bench.contracts.common import ContractModel
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    FaultAxis,
    FaultFamily,
    MethodId,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    Digest,
    Identifier,
    SeverityCoordinate,
)

type Coordinate = Literal["x", "y"]
type Sensor = Literal["camera", "lidar"]
type ValidationUnit = Literal["m", "m^2", "fraction"]

_ORACLE_RECONCILIATION_TOLERANCE = 1e-12
_SE_MULTIPLIER = 6.0


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=_ORACLE_RECONCILIATION_TOLERANCE,
    )


class ProfileValidationChecksV1(ContractModel):
    """Fail-closed checks linking one manifest to one immutable profile."""

    schema_valid: bool
    profile_id_valid: bool
    profile_digest_valid: bool
    split_count_valid: bool
    roi_valid: bool
    isotropic_yaw_compatible: bool
    split_family_support_valid: bool
    canonical_ordering_valid: bool
    all_checks_passed: bool

    @model_validator(mode="after")
    def validate_conjunction(self) -> Self:
        expected = all(
            (
                self.schema_valid,
                self.profile_id_valid,
                self.profile_digest_valid,
                self.split_count_valid,
                self.roi_valid,
                self.isotropic_yaw_compatible,
                self.split_family_support_valid,
                self.canonical_ordering_valid,
            )
        )
        if self.all_checks_passed != expected:
            raise ValueError("profile all_checks_passed must be the conjunction")
        return self


class EligibilityValidationV1(ContractModel):
    """Ordered eligibility commitments and invariance evidence."""

    ordered_sequence_commitments_sha256: Digest
    minimum_eligible_object_frame_count: Annotated[int, Field(ge=1)]
    maximum_eligible_object_frame_count: Annotated[int, Field(ge=1)]
    total_eligible_object_frame_count: Annotated[int, Field(ge=1)]
    eligibility_invariant: bool

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.maximum_eligible_object_frame_count < self.minimum_eligible_object_frame_count:
            raise ValueError("eligibility maximum must not precede the minimum")
        return self


class OracleDiscrepancyV1(ContractModel):
    """One exact implementation-oracle discrepancy and fixed tolerance."""

    check_id: Identifier
    unit: Literal["m", "m^2"]
    maximum_absolute_discrepancy: Annotated[FiniteFloat, Field(ge=0.0)]
    tolerance: Annotated[FiniteFloat, Field(ge=1e-12, le=1e-12)]
    passed: bool

    @model_validator(mode="after")
    def validate_pass(self) -> Self:
        if self.passed != (self.maximum_absolute_discrepancy <= self.tolerance):
            raise ValueError("oracle pass flag disagrees with discrepancy and tolerance")
        return self


class ProceduralOracleChecksV1(ContractModel):
    """The fixed lower-level identity, geometry, and timing oracle suite."""

    identity_center: OracleDiscrepancyV1
    calibration_translation_center: OracleDiscrepancyV1
    translation_bias_equivalence_center: OracleDiscrepancyV1
    translation_bias_equivalence_sequence_loss: OracleDiscrepancyV1
    calibration_yaw_center: OracleDiscrepancyV1
    timestamp_alignment_center: OracleDiscrepancyV1
    static_timestamp_center: OracleDiscrepancyV1
    fault_cancellation_mutation_rejected: bool
    all_checks_passed: bool

    @model_validator(mode="after")
    def validate_suite(self) -> Self:
        expected_identities = (
            (self.identity_center, "identity-center", "m"),
            (
                self.calibration_translation_center,
                "calibration-translation-center",
                "m",
            ),
            (
                self.translation_bias_equivalence_center,
                "translation-bias-equivalence-center",
                "m",
            ),
            (
                self.translation_bias_equivalence_sequence_loss,
                "translation-bias-equivalence-sequence-loss",
                "m^2",
            ),
            (self.calibration_yaw_center, "calibration-yaw-center", "m"),
            (self.timestamp_alignment_center, "timestamp-alignment-center", "m"),
            (self.static_timestamp_center, "static-timestamp-center", "m"),
        )
        if any(
            record.check_id != check_id or record.unit != unit
            for record, check_id, unit in expected_identities
        ):
            raise ValueError("oracle fields must use their frozen check IDs and units")
        expected = self.fault_cancellation_mutation_rejected and all(
            (
                self.identity_center.passed,
                self.calibration_translation_center.passed,
                self.translation_bias_equivalence_center.passed,
                self.translation_bias_equivalence_sequence_loss.passed,
                self.calibration_yaw_center.passed,
                self.timestamp_alignment_center.passed,
                self.static_timestamp_center.passed,
            )
        )
        if self.all_checks_passed != expected:
            raise ValueError("oracle all_checks_passed must be the conjunction")
        return self


class MomentCheckV1(ContractModel):
    """One preregistered Gaussian mean, variance, or covariance check."""

    check_id: Identifier
    statistic: Literal[
        "mean",
        "variance",
        "within-sensor-covariance",
        "camera-lidar-cross-covariance",
    ]
    sensor_a: Sensor
    coordinate_a: Coordinate
    sensor_b: Sensor | None
    coordinate_b: Coordinate | None
    sample_count: Annotated[int, Field(ge=2)]
    ddof: Literal[1]
    expectation: FiniteFloat
    observed_value: FiniteFloat
    six_standard_error_bound: Annotated[FiniteFloat, Field(ge=0.0)]
    absolute_discrepancy: Annotated[FiniteFloat, Field(ge=0.0)]
    unit: Literal["m", "m^2"]
    passed: bool

    @model_validator(mode="after")
    def validate_statistic(self) -> Self:
        paired = self.statistic in {
            "within-sensor-covariance",
            "camera-lidar-cross-covariance",
        }
        if paired != (self.sensor_b is not None and self.coordinate_b is not None):
            raise ValueError("covariance checks require exactly two sensor coordinates")
        if self.statistic == "within-sensor-covariance" and self.sensor_b != self.sensor_a:
            raise ValueError("within-sensor covariance must use one sensor")
        if self.statistic == "within-sensor-covariance" and self.coordinate_b == self.coordinate_a:
            raise ValueError("within-sensor covariance must use distinct coordinates")
        if self.statistic == "camera-lidar-cross-covariance" and {
            self.sensor_a,
            self.sensor_b,
        } != {"camera", "lidar"}:
            raise ValueError("cross-modal covariance must use camera and lidar")
        expected_unit = "m" if self.statistic == "mean" else "m^2"
        if self.unit != expected_unit:
            raise ValueError(f"{self.statistic} requires unit {expected_unit}")
        discrepancy = abs(self.observed_value - self.expectation)
        if not _close(self.absolute_discrepancy, discrepancy):
            raise ValueError("moment discrepancy disagrees with observed and expected values")
        if self.passed != (discrepancy <= self.six_standard_error_bound):
            raise ValueError("moment pass flag disagrees with the six-SE bound")
        return self


class ExpectedLossCheckV1(ContractModel):
    """One affine-Gaussian expected loss or signed-contrast check."""

    check_id: Identifier
    fault_family: FaultFamily
    fault_axis: FaultAxis
    severity: SeverityCoordinate
    method_id: MethodId
    metric_name: Literal["matched-center-mse", "fused-minus-healthy"]
    expected_value_m2: FiniteFloat
    empirical_value_m2: FiniteFloat
    analytic_standard_error_m2: PositiveFloat
    absolute_standardized_error: Annotated[FiniteFloat, Field(ge=0.0)]
    standard_error_multiplier: Annotated[FiniteFloat, Field(ge=6.0, le=6.0)]
    passed: bool

    @model_validator(mode="after")
    def validate_expected_loss(self) -> Self:
        expected_standardized = (
            abs(self.empirical_value_m2 - self.expected_value_m2) / self.analytic_standard_error_m2
        )
        if not _close(self.absolute_standardized_error, expected_standardized):
            raise ValueError("expected-loss standardized error is inconsistent")
        if self.metric_name == "fused-minus-healthy" and self.method_id != "fixed-fusion":
            raise ValueError("fused-minus-healthy must use fixed-fusion method_id")
        if self.metric_name == "matched-center-mse" and (
            self.expected_value_m2 < 0.0 or self.empirical_value_m2 < 0.0
        ):
            raise ValueError("matched-center MSE values must be non-negative")
        expected_pass = expected_standardized <= self.standard_error_multiplier
        if self.passed != expected_pass:
            raise ValueError("expected-loss pass flag disagrees with the six-SE threshold")
        return self


class DropoutValidationNotApplicableV1(ContractModel):
    """Explicit absence of dropout checks for non-availability manifests."""

    status: Literal["not-applicable"]


class DropoutValidationApplicableV1(ContractModel):
    """Independent exact-mask validation for one availability artifact."""

    status: Literal["applicable"]
    uniform_vectors_sha256: Digest
    exact_mask_comparison_count: Annotated[int, Field(ge=1)]
    frame_sharing_passed: bool
    nesting_passed: bool
    endpoint_behavior_passed: bool
    maximum_mask_discrepancy: Annotated[int, Field(ge=0, le=1)]
    all_checks_passed: bool

    @model_validator(mode="after")
    def validate_dropout(self) -> Self:
        expected = (
            self.maximum_mask_discrepancy == 0
            and self.frame_sharing_passed
            and self.nesting_passed
            and self.endpoint_behavior_passed
        )
        if self.all_checks_passed != expected:
            raise ValueError("dropout all_checks_passed must be the conjunction")
        return self


type DropoutValidationV1 = Annotated[
    DropoutValidationNotApplicableV1 | DropoutValidationApplicableV1,
    Field(discriminator="status"),
]


class IdentityComparisonApplicableV1(ContractModel):
    """Exact same-profile identity comparison after fixed field removal."""

    status: Literal["applicable"]
    scope: Literal["same-profile-split-observations-seeds-and-comparable-methods"]
    comparison_count: Annotated[int, Field(ge=1)]
    maximum_absolute_value_discrepancy_m2: Annotated[FiniteFloat, Field(ge=0.0)]
    tolerance_m2: Annotated[FiniteFloat, Field(ge=1e-12, le=1e-12)]
    passed: bool

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        expected = self.maximum_absolute_value_discrepancy_m2 <= self.tolerance_m2
        if self.passed != expected:
            raise ValueError("identity pass flag disagrees with discrepancy and tolerance")
        return self


class IdentityComparisonNotApplicableV1(ContractModel):
    """Explicit absence of a comparable same-profile manifest."""

    status: Literal["not-applicable"]
    reason: Literal["edge-common-mode-has-no-comparable-profile-peer"]


class IdentityComparisonDeferredV1(ContractModel):
    """Cross-manifest equality requires the complete ordered M3 matrix."""

    status: Literal["deferred-to-matrix"]
    reason: Literal["cross-manifest-identity-requires-complete-matrix"]


type IdentityComparisonV1 = Annotated[
    IdentityComparisonApplicableV1
    | IdentityComparisonNotApplicableV1
    | IdentityComparisonDeferredV1,
    Field(discriminator="status"),
]


class CommonModeValidationNotApplicableV1(ContractModel):
    """Explicit absence of the common-mode disagreement check."""

    status: Literal["not-applicable"]


class CommonModeValidationApplicableV1(ContractModel):
    """Exact common-mode cross-modal disagreement invariance."""

    status: Literal["applicable"]
    maximum_disagreement_discrepancy_m: Annotated[FiniteFloat, Field(ge=0.0)]
    tolerance_m: Annotated[FiniteFloat, Field(ge=1e-12, le=1e-12)]
    passed: bool

    @model_validator(mode="after")
    def validate_common_mode(self) -> Self:
        expected = self.maximum_disagreement_discrepancy_m <= self.tolerance_m
        if self.passed != expected:
            raise ValueError("common-mode pass flag disagrees with discrepancy and tolerance")
        return self


type CommonModeValidationV1 = Annotated[
    CommonModeValidationNotApplicableV1 | CommonModeValidationApplicableV1,
    Field(discriminator="status"),
]


class ResourceValidationV1(ContractModel):
    """Manifest-implied resource counts and every inherited M3 cap."""

    implied_sequence_row_count: Annotated[int, Field(ge=1)]
    sequence_row_cap: Literal[2_000_000]
    implied_bootstrap_cell_count: Annotated[int, Field(ge=1)]
    bootstrap_cell_cap: Literal[20_000_000]
    sequence_count: Annotated[int, Field(ge=1)]
    sequence_count_cap: Literal[10_000]
    bootstrap_replicates: Annotated[int, Field(ge=200)]
    bootstrap_replicate_cap: Literal[20_000]
    sequence_rows_within_cap: bool
    bootstrap_cells_within_cap: bool
    sequence_count_within_cap: bool
    bootstrap_replicates_within_cap: bool
    all_checks_passed: bool

    @model_validator(mode="after")
    def validate_resources(self) -> Self:
        expected_flags = (
            self.implied_sequence_row_count <= self.sequence_row_cap,
            self.implied_bootstrap_cell_count <= self.bootstrap_cell_cap,
            self.sequence_count <= self.sequence_count_cap,
            self.bootstrap_replicates <= self.bootstrap_replicate_cap,
        )
        actual_flags = (
            self.sequence_rows_within_cap,
            self.bootstrap_cells_within_cap,
            self.sequence_count_within_cap,
            self.bootstrap_replicates_within_cap,
        )
        if actual_flags != expected_flags:
            raise ValueError("resource pass flags disagree with counts and caps")
        if self.all_checks_passed != all(expected_flags):
            raise ValueError("resource all_checks_passed must be the conjunction")
        return self


class DeterministicModelChecksV1(ContractModel):
    """Named gates enforced while rebuilding deterministic M3 evidence."""

    reported_covariance_behavior: Literal["passed"]
    expected_curve_response: Literal["passed", "not-applicable"]
    complete_sequence_performance_oracle: Literal["passed", "not-applicable"]
    identity_row_reconstruction: Literal["passed", "not-applicable"]
    all_checks_passed: bool

    @model_validator(mode="after")
    def validate_named_gates(self) -> Self:
        if not self.all_checks_passed:
            raise ValueError("deterministic model gates must pass before publication")
        return self


class ProceduralValidationV1(ContractModel):
    """Complete indexed M3 validation record for one procedural artifact."""

    schema_id: Literal["ffb.procedural-validation/v1"] = Field(alias="schema")
    run_id: Identifier
    manifest_sha256: Digest
    profile_id: Identifier
    profile_sha256: Digest
    split: Literal["train", "validation", "test"]
    sequence_count: Annotated[int, Field(ge=2)]
    frame_count: Annotated[int, Field(ge=2)]
    object_count: Annotated[int, Field(ge=1)]
    total_eligible_object_frame_count: Annotated[int, Field(ge=1)]
    profile_checks: ProfileValidationChecksV1
    eligibility: EligibilityValidationV1
    oracle_checks: ProceduralOracleChecksV1
    moment_checks: Annotated[tuple[MomentCheckV1, ...], Field(min_length=1)]
    expected_loss_checks: tuple[ExpectedLossCheckV1, ...]
    dropout_validation: DropoutValidationV1
    identity_comparison: IdentityComparisonV1
    common_mode_validation: CommonModeValidationV1
    deterministic_model_checks: DeterministicModelChecksV1
    resources: ResourceValidationV1
    all_checks_passed: bool

    @model_validator(mode="after")
    def validate_complete_record(self) -> Self:
        if (
            self.total_eligible_object_frame_count
            != self.eligibility.total_eligible_object_frame_count
        ):
            raise ValueError("top-level and eligibility total counts must agree")
        if not (
            self.eligibility.minimum_eligible_object_frame_count * self.sequence_count
            <= self.total_eligible_object_frame_count
            <= self.eligibility.maximum_eligible_object_frame_count * self.sequence_count
        ):
            raise ValueError("eligible total is outside the per-sequence count bounds")
        if self.resources.sequence_count != self.sequence_count:
            raise ValueError("resource and validation sequence counts must agree")
        expected_moment_keys = {
            ("camera-x-mean", "mean", "camera", "x", None, None),
            ("camera-y-mean", "mean", "camera", "y", None, None),
            ("lidar-x-mean", "mean", "lidar", "x", None, None),
            ("lidar-y-mean", "mean", "lidar", "y", None, None),
            ("camera-x-variance", "variance", "camera", "x", None, None),
            ("camera-y-variance", "variance", "camera", "y", None, None),
            ("lidar-x-variance", "variance", "lidar", "x", None, None),
            ("lidar-y-variance", "variance", "lidar", "y", None, None),
            (
                "camera-xy-covariance",
                "within-sensor-covariance",
                "camera",
                "x",
                "camera",
                "y",
            ),
            (
                "lidar-xy-covariance",
                "within-sensor-covariance",
                "lidar",
                "x",
                "lidar",
                "y",
            ),
            (
                "camera-x-lidar-x-covariance",
                "camera-lidar-cross-covariance",
                "camera",
                "x",
                "lidar",
                "x",
            ),
            (
                "camera-x-lidar-y-covariance",
                "camera-lidar-cross-covariance",
                "camera",
                "x",
                "lidar",
                "y",
            ),
            (
                "camera-y-lidar-x-covariance",
                "camera-lidar-cross-covariance",
                "camera",
                "y",
                "lidar",
                "x",
            ),
            (
                "camera-y-lidar-y-covariance",
                "camera-lidar-cross-covariance",
                "camera",
                "y",
                "lidar",
                "y",
            ),
        }
        actual_moment_keys = {
            (
                check.check_id,
                check.statistic,
                check.sensor_a,
                check.coordinate_a,
                check.sensor_b,
                check.coordinate_b,
            )
            for check in self.moment_checks
        }
        if actual_moment_keys != expected_moment_keys or len(self.moment_checks) != 14:
            raise ValueError("moment checks must contain the exact frozen 14-row suite")
        if any(
            check.sample_count != self.total_eligible_object_frame_count
            for check in self.moment_checks
        ):
            raise ValueError("moment checks must use the complete eligible population")
        expected_loss_ids = tuple(check.check_id for check in self.expected_loss_checks)
        if len(set(expected_loss_ids)) != len(expected_loss_ids):
            raise ValueError("expected-loss check IDs must be unique")
        expected_loss_keys = tuple(
            (
                check.fault_family,
                check.fault_axis,
                check.severity.index,
                check.severity.magnitude,
                check.severity.direction,
                check.severity.unit,
                check.method_id,
                check.metric_name,
            )
            for check in self.expected_loss_checks
        )
        if len(set(expected_loss_keys)) != len(expected_loss_keys):
            raise ValueError("expected-loss semantic keys must be unique")
        grouped_loss_pairs: dict[
            tuple[str, str, int, float, str, str],
            set[tuple[str, str]],
        ] = {}
        for key in expected_loss_keys:
            grouped_loss_pairs.setdefault(key[:6], set()).add((key[6], key[7]))
        if self.dropout_validation.status == "applicable":
            if grouped_loss_pairs:
                raise ValueError("dropout validation cannot contain affine loss checks")
        else:
            required_pairs = (
                {
                    ("camera-only", "matched-center-mse"),
                    ("lidar-only", "matched-center-mse"),
                    ("fixed-fusion", "matched-center-mse"),
                }
                if self.common_mode_validation.status == "applicable"
                else {
                    ("camera-only", "matched-center-mse"),
                    ("lidar-only", "matched-center-mse"),
                    ("fixed-fusion", "matched-center-mse"),
                    ("fault-target-drop-policy", "matched-center-mse"),
                    ("fixed-fusion", "fused-minus-healthy"),
                }
            )
            if not grouped_loss_pairs or any(
                pairs != required_pairs for pairs in grouped_loss_pairs.values()
            ):
                raise ValueError(
                    "expected-loss checks must contain the exact affine method set per condition"
                )
        dropout_passed = (
            self.dropout_validation.status == "not-applicable"
            or self.dropout_validation.all_checks_passed
        )
        common_mode_passed = (
            self.common_mode_validation.status == "not-applicable"
            or self.common_mode_validation.passed
        )
        expected = all(
            (
                self.profile_checks.all_checks_passed,
                self.eligibility.eligibility_invariant,
                self.oracle_checks.all_checks_passed,
                all(check.passed for check in self.moment_checks),
                all(check.passed for check in self.expected_loss_checks),
                dropout_passed,
                (
                    self.identity_comparison.status != "applicable"
                    or self.identity_comparison.passed
                ),
                common_mode_passed,
                self.deterministic_model_checks.all_checks_passed,
                self.resources.all_checks_passed,
            )
        )
        if self.all_checks_passed != expected:
            raise ValueError("procedural all_checks_passed must be the conjunction")
        return self


def procedural_validation_json_schema() -> dict[str, object]:
    """Return the strict public schema for M3 procedural validation records."""

    return ProceduralValidationV1.model_json_schema(by_alias=True)
