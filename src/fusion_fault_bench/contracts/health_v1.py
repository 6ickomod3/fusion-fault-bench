"""Strict, content-addressed contract for the frozen M4 health benchmark intent."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, FiniteFloat, TypeAdapter, model_validator

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.common import ContractModel
from fusion_fault_bench.contracts.io import load_json_object

M4_HEALTH_INTENT_SHA256 = "c19573b72a6ef58ad5efdd7039110da79beacd3b398007a508eb7ecfc4c81357"
M4_HEALTH_INTENT_PATH = Path("examples/health/m4-health-v1.json")

_INTENT_FILE_CAP_BYTES = 256 * 1024
_MAIN_PROFILE_SHA256 = "4771a6e69d75b9af41f99ab794c0af1b51e6103e43474c8e0f07df3e6f3ca68c"
_EDGE_PROFILE_SHA256 = "ca1544f69023847af7bdad9f1306ae3885f2e5d067d6afc026038f87ae36448d"

type Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
type Identifier = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")]
type NonEmptyText = Annotated[str, Field(min_length=1)]
type FrameWindow = tuple[
    Annotated[int, Field(ge=0, le=48)],
    Annotated[int, Field(ge=1, le=48)],
]
type ModalityTarget = Literal["camera", "lidar", "both", "none"]
type ScheduleId = Literal["standard", "cold_start"]
type FaultFamily = Literal[
    "identity",
    "additive-position-bias",
    "increased-noise-underreported",
    "increased-noise-correctly-reported",
    "timestamp-offset",
    "dropout",
    "calibration-translation",
    "calibration-yaw",
    "clean-predictor-mismatch",
    "common-mode-position-bias",
]
type FaultAxis = Literal["none", "x", "y", "xy", "time", "availability", "yaw", "motion"]
type SeverityUnit = Literal[
    "identity",
    "m",
    "std-scale",
    "s",
    "probability",
    "rad",
    "m/s^2",
]
type HealthMethodId = Literal[
    "camera-only",
    "lidar-only",
    "fixed-fusion",
    "self-nis-gate",
    "cross-nis-gate",
    "direct-telemetry-gate",
    "combined-health-gate",
    "combined-health-gate-abstain",
    "fault-target-drop-policy",
    "frame-action-performance-oracle",
]

EXPECTED_M4_METHODS: tuple[HealthMethodId, ...] = (
    "camera-only",
    "lidar-only",
    "fixed-fusion",
    "self-nis-gate",
    "cross-nis-gate",
    "direct-telemetry-gate",
    "combined-health-gate",
    "combined-health-gate-abstain",
    "fault-target-drop-policy",
    "frame-action-performance-oracle",
)
EXPECTED_M4_THRESHOLDS = (0.95, 0.975, 0.99, 0.995, 0.999, 1.0)
EXPECTED_M4_SELECTION_CONDITIONS = (
    "validation-camera-output-y-bias",
    "validation-lidar-output-y-bias",
    "validation-camera-noise-underreported",
    "validation-lidar-noise-underreported",
    "validation-camera-noise-correctly-reported",
    "validation-lidar-noise-correctly-reported",
    "validation-camera-calibration-x",
)


class CoordinateContractV1(ContractModel):
    """Exact geometry and numeric conventions shared with the benchmark contract."""

    handedness: Literal["right-handed"]
    point_representation: Literal["column-vector"]
    transform_notation: Literal["T_{target<-source}"]
    health_frame: Literal["persistent-scene-bev"]
    length_unit: Literal["m"]
    time_unit: Literal["s"]
    angle_unit: Literal["rad"]
    floating_point: Literal["float64"]


class SplitSequenceCountsV1(ContractModel):
    """Frozen sequence populations."""

    train: Literal[200]
    validation: Literal[200]
    test: Literal[200]
    edge_test: Literal[100]


class SplitUsesV1(ContractModel):
    """One-way train/validation/test use boundary."""

    train: Literal["clean-ecdf-fit-only"]
    validation: Literal["threshold-selection-only"]
    test: Literal["frozen-fit-evaluation-only"]
    edge_test: Literal["held-out-support-and-common-mode-controls-only"]


class SourcePopulationV1(ContractModel):
    """Content-addressed populations and their permitted roles."""

    profile_id: Literal["constant-velocity-front-roi-v1"]
    profile_sha256: Digest
    edge_profile_id: Literal["constant-velocity-fov-edge-v1"]
    edge_profile_sha256: Digest
    data_master_seed: Literal[1729]
    split_sequence_counts: SplitSequenceCountsV1
    split_uses: SplitUsesV1
    test_once: Literal[True]
    variant_clustering_key: Literal["base-sequence-index-within-split"]

    @model_validator(mode="after")
    def require_frozen_profile_digests(self) -> Self:
        if (
            self.profile_sha256 != _MAIN_PROFILE_SHA256
            or self.edge_profile_sha256 != _EDGE_PROFILE_SHA256
        ):
            raise ValueError("M4 source profile digest changed")
        return self


class ObservationModelV1(ContractModel):
    """Actual and reported observation uncertainty remain distinct."""

    camera_actual_std_xy_m: tuple[FiniteFloat, FiniteFloat]
    camera_reported_std_xy_m: tuple[FiniteFloat, FiniteFloat]
    lidar_actual_std_xy_m: tuple[FiniteFloat, FiniteFloat]
    lidar_reported_std_xy_m: tuple[FiniteFloat, FiniteFloat]
    base_error_dependence: Literal["temporally-iid-independent-between-modalities"]
    paired_base_draws_across_conditions_methods_and_policies: Literal[True]
    actual_error_separate_from_reported_covariance: Literal[True]
    fault_generation_uses_true_state_pose_and_time: Literal[True]
    metadata_faults_apply_only_to_reconstruction_or_alignment: Literal[True]

    @model_validator(mode="after")
    def require_nominal_uncertainty(self) -> Self:
        if (
            self.camera_actual_std_xy_m != (1.0, 1.0)
            or self.camera_reported_std_xy_m != (1.0, 1.0)
            or self.lidar_actual_std_xy_m != (0.3, 0.3)
            or self.lidar_reported_std_xy_m != (0.3, 0.3)
        ):
            raise ValueError("M4 nominal observation uncertainty changed")
        return self


class StandardEventScheduleV1(ContractModel):
    """Standard clean-prefix, active-event, and recovery windows."""

    frame_count: Literal[48]
    predictor_initialization_frames: FrameWindow
    clean_prefix_frames: FrameWindow
    score_frames: FrameWindow
    fault_active_frames: FrameWindow
    recovery_frames: FrameWindow

    @model_validator(mode="after")
    def require_frozen_standard_windows(self) -> Self:
        if (
            self.predictor_initialization_frames != (0, 2)
            or self.clean_prefix_frames != (0, 12)
            or self.score_frames != (2, 48)
            or self.fault_active_frames != (12, 36)
            or self.recovery_frames != (36, 48)
        ):
            raise ValueError("standard M4 event schedule changed")
        return self


class ColdStartEventScheduleV1(ContractModel):
    """Cold-start event windows with no inherited standard prefix."""

    frame_count: Literal[48]
    score_frames: FrameWindow
    fault_active_frames: FrameWindow
    recovery_frames: FrameWindow

    @model_validator(mode="after")
    def require_frozen_cold_start_windows(self) -> Self:
        if (
            self.score_frames != (0, 48)
            or self.fault_active_frames != (0, 24)
            or self.recovery_frames != (24, 48)
        ):
            raise ValueError("cold-start M4 event schedule changed")
        return self


class EventSchedulesV1(ContractModel):
    """Both versioned event schedules and their corruption boundary."""

    standard: StandardEventScheduleV1
    cold_start: ColdStartEventScheduleV1
    interval_semantics: Literal["zero-based-half-open"]
    outside_active_event: Literal["exact-identity-observation-and-metadata-configuration"]
    fault_accumulation_across_frames: Literal[False]


class ValidationConditionV1(ContractModel):
    """One validation-only threshold-selection or direct-rule condition."""

    condition_id: Identifier
    family: FaultFamily
    target: ModalityTarget
    axis: FaultAxis
    unit: SeverityUnit
    values: Annotated[tuple[FiniteFloat, ...], Field(min_length=1, max_length=4)]
    schedule: Literal["standard"]
    selection_role: Literal[
        "numeric-threshold-utility",
        "numeric-threshold-utility-negative-control",
        "direct-telemetry-validation-only",
    ]


class ValidationControlV1(ContractModel):
    """The sole clean validation feasibility condition."""

    condition_id: Literal["validation-main-clean"]
    family: Literal["identity"]
    target: Literal["none"]
    axis: Literal["none"]
    unit: Literal["identity"]
    values: tuple[FiniteFloat]
    schedule: Literal["standard"]
    population: Literal["main-validation"]
    selection_role: Literal["clean-feasibility-only"]


class TestConditionV1(ContractModel):
    """One apply-only test condition."""

    condition_id: Identifier
    family: FaultFamily
    target: ModalityTarget
    axis: FaultAxis
    unit: SeverityUnit
    values: Annotated[tuple[FiniteFloat, ...], Field(min_length=1, max_length=4)]
    schedule: Literal["standard"]
    test_role: Literal[
        "unseen-severity",
        "unseen-severity-negative-control",
        "unseen-severity-direct-telemetry",
        "held-out-fault-family",
    ]


class MotionRuleV1(ContractModel):
    """Exact bounded-acceleration negative-control recurrence."""

    time_step_s: FiniteFloat
    transition_indices: NonEmptyText
    object_side: NonEmptyText
    acceleration_x_mps2: FiniteFloat
    acceleration_y_mps2: NonEmptyText
    position_recurrence: NonEmptyText
    velocity_recurrence: NonEmptyText
    initial_state: NonEmptyText
    eligibility: NonEmptyText


class OrdinaryTestControlV1(ContractModel):
    """A test-only identity, common-mode, or cold-start control."""

    condition_id: Literal[
        "test-main-clean",
        "test-edge-clean",
        "test-common-mode-x-edge",
        "test-cold-start-camera-calibration-x",
        "test-cold-start-lidar-y-bias",
    ]
    family: FaultFamily
    target: ModalityTarget
    axis: FaultAxis
    unit: SeverityUnit
    values: Annotated[tuple[FiniteFloat, ...], Field(min_length=1, max_length=4)]
    schedule: ScheduleId
    population: Literal["main-test", "edge-test"]
    test_role: Literal[
        "clean-regression",
        "held-out-support-shift",
        "agreement-blind-spot",
        "cold-start-diagnostic",
    ]


class MotionTestControlV1(ContractModel):
    """The clean predictor-mismatch control with an explicit motion program."""

    condition_id: Literal["test-clean-bounded-acceleration"]
    family: Literal["clean-predictor-mismatch"]
    target: Literal["none"]
    axis: Literal["motion"]
    unit: Literal["m/s^2"]
    values: tuple[FiniteFloat]
    schedule: Literal["standard"]
    population: Literal["main-test"]
    motion_rule: MotionRuleV1
    test_role: Literal["difficult-clean-predictor-mismatch"]


type TestControlV1 = Annotated[
    OrdinaryTestControlV1 | MotionTestControlV1,
    Field(discriminator="condition_id"),
]


class PredictorV1(ContractModel):
    """Frozen two-prior-observation constant-velocity predictor."""

    history_scope: Literal["independent-per-modality-per-object"]
    history_length: Literal[2]
    history_members: tuple[NonEmptyText, ...]
    requires_strict_times: Literal["t_a<t_b<t_k"]
    prediction_clock: Literal["frame-reference-time-only"]
    reported_timestamp_use: Literal["direct-residual-only"]
    h_equation: Literal["h=(t_k-t_b)/(t_b-t_a)"]
    mean_equation: Literal["z_pred=(1+h)*z_b-h*z_a"]
    covariance_equation: Literal["P_pred=(1+h)^2*R_b+h^2*R_a"]
    process_noise: FiniteFloat
    score_before_current_update: Literal[True]
    unavailable_measurements_update_history: Literal[False]
    policy_action_changes_monitoring_history: Literal[False]

    @model_validator(mode="after")
    def require_frozen_history_members(self) -> Self:
        if (
            self.history_members
            != (
                "estimate_xy_m",
                "reported_covariance_xy_m2",
                "reference_time_s",
            )
            or self.process_noise != 0.0
        ):
            raise ValueError("M4 predictor history contract changed")
        return self


class FeaturesV1(ContractModel):
    """Observable feature boundary and explicit prohibited inputs."""

    minimum_mature_object_count: Literal[2]
    numeric_score_statuses: tuple[NonEmptyText, ...]
    insufficient_support_condition: NonEmptyText
    numeric_object_aggregation: tuple[NonEmptyText, ...]
    self_nis_equation: NonEmptyText
    camera_from_lidar_cross_equation: NonEmptyText
    lidar_from_camera_cross_equation: NonEmptyText
    clean_ecdf_arrays: tuple[NonEmptyText, ...]
    direct_features: tuple[NonEmptyText, ...]
    diagnostic_only_features: tuple[NonEmptyText, ...]
    prohibited_inputs: tuple[NonEmptyText, ...]

    @model_validator(mode="after")
    def require_feature_sets(self) -> Self:
        if (
            self.numeric_score_statuses != ("defined", "insufficient-support")
            or self.numeric_object_aggregation != ("arithmetic-mean", "maximum")
            or self.clean_ecdf_arrays
            != (
                "camera-self-mean",
                "camera-self-maximum",
                "lidar-self-mean",
                "lidar-self-maximum",
                "camera-from-lidar-cross-mean",
                "camera-from-lidar-cross-maximum",
                "lidar-from-camera-cross-mean",
                "lidar-from-camera-cross-maximum",
            )
            or len(set(self.prohibited_inputs)) != len(self.prohibited_inputs)
        ):
            raise ValueError("M4 feature channel set changed")
        required_prohibited = {
            "truth",
            "latent-velocity",
            "fault-family",
            "fault-target",
            "severity",
            "direction",
            "event-phase",
            "event-boundary",
            "seed",
            "split",
            "sequence-id",
            "manifest",
            "absolute-frame-index-as-feature",
        }
        if set(self.prohibited_inputs) != required_prohibited:
            raise ValueError("M4 prohibited policy input set changed")
        return self


class CalibrationV1(ContractModel):
    """Clean-only ECDF fitting semantics."""

    fit_split: Literal["train"]
    fit_conditions: tuple[Literal["clean-only"]]
    fit_frames: FrameWindow
    expected_values_per_ecdf_array: Literal[9200]
    ecdf_rank_equation: Literal["rank(x)=count(clean_value<x)/clean_value_count"]
    tie_rule: Literal["strict-less-than"]
    channel_score: Literal["maximum-of-mean-statistic-rank-and-maximum-statistic-rank"]
    alarm_comparison: Literal["score>threshold"]
    threshold_one_disables_numeric_alarm: Literal[True] = Field(
        alias="threshold_one_disables_numeric-alarm"
    )

    @model_validator(mode="after")
    def require_fit_window(self) -> Self:
        if self.fit_frames != (2, 48):
            raise ValueError("M4 ECDF fit window changed")
        return self


class RawDecisionRulesV1(ContractModel):
    """Exact observable raw-label algorithms."""

    direct_telemetry_gate: NonEmptyText = Field(alias="direct-telemetry-gate")
    self_nis_gate: NonEmptyText = Field(alias="self-nis-gate")
    cross_nis_gate: NonEmptyText = Field(alias="cross-nis-gate")
    combined_health_gate: NonEmptyText = Field(alias="combined-health-gate")


class ActionRulesV1(ContractModel):
    """Action selection from availability and the latched health state."""

    neither_modality_available: Literal["undefined"] = Field(alias="neither-modality-available")
    only_camera_available: Literal["camera-only-immediately-regardless-of-latch"] = Field(
        alias="only-camera-available"
    )
    only_lidar_available: Literal["lidar-only-immediately-regardless-of-latch"] = Field(
        alias="only-lidar-available"
    )
    both_available_and_healthy: Literal["fixed-fusion"] = Field(alias="both-available-and-healthy")
    both_available_and_camera_fault: Literal["lidar-only"] = Field(
        alias="both-available-and-camera-fault"
    )
    both_available_and_lidar_fault: Literal["camera-only"] = Field(
        alias="both-available-and-lidar-fault"
    )
    both_available_and_ambiguous_nonabstaining: Literal["fixed-fusion"] = Field(
        alias="both-available-and-ambiguous-nonabstaining"
    )
    both_available_and_ambiguous_abstaining: Literal["undefined"] = Field(
        alias="both-available-and-ambiguous-abstaining"
    )


class PolicyV1(ContractModel):
    """Raw evidence, latch counters, and executed action remain distinct."""

    raw_labels: tuple[NonEmptyText, ...]
    raw_evidence_statuses: tuple[NonEmptyText, ...]
    latched_states: tuple[NonEmptyText, ...]
    executed_actions: tuple[NonEmptyText, ...]
    direct_timestamp_tolerance_s: FiniteFloat
    raw_decision_rules: RawDecisionRulesV1
    insufficient_support_rule: NonEmptyText
    insufficient_support_counter_behavior: NonEmptyText
    activation_streak: Literal[2]
    activation_rule: NonEmptyText
    activation_recurrence: NonEmptyText
    recovery_streak: Literal[3]
    recovery_rule: NonEmptyText
    counter_reset_rule: NonEmptyText
    latched_attribution_switching: Literal["never-switch-until-recovered-healthy"]
    action_rules: ActionRulesV1
    detection_uses: Literal["latched-state-not-raw-label"]
    missing_input_action_override_is_not_detection: Literal[True] = Field(
        alias="missing-input-action-override_is_not_detection"
    )

    @model_validator(mode="after")
    def require_label_and_action_sets(self) -> Self:
        labels = ("healthy", "camera-fault", "lidar-fault", "ambiguous")
        if (
            self.raw_labels != labels
            or self.latched_states != labels
            or self.raw_evidence_statuses != ("update-eligible", "insufficient-support")
            or self.executed_actions != ("camera-only", "lidar-only", "fixed-fusion", "undefined")
            or self.direct_timestamp_tolerance_s != 1e-12
        ):
            raise ValueError("M4 health label or action set changed")
        return self


class FaultTargetDropPolicyV1(ContractModel):
    """Nondeployable target-aware diagnostic semantics."""

    outside_active_event: Literal["fixed-fusion"]
    inside_active_single_target_event: Literal["non-target-modality-only"] = Field(
        alias="inside_active_single-target_event"
    )
    selected_modality_unavailable: Literal["undefined"] = Field(
        alias="selected-modality-unavailable"
    )
    availability_fallback_to_target: Literal[False]
    identity_condition: Literal["fixed-fusion"]
    common_mode_applicable: Literal[False]
    deployable: Literal[False]


class ThresholdFeasibilityV1(ContractModel):
    """Literal clean feasibility gates."""

    condition_id: Literal["validation-main-clean"]
    mean_clean_regression_m2_max: FiniteFloat
    upper_pointwise_95pct_clean_regression_m2_max: FiniteFloat
    false_alert_episode_starts_per_clean_sequence_max: FiniteFloat
    clean_coverage_must_equal_fixed: Literal[True]

    @model_validator(mode="after")
    def require_literal_feasibility_limits(self) -> Self:
        if (
            self.mean_clean_regression_m2_max,
            self.upper_pointwise_95pct_clean_regression_m2_max,
            self.false_alert_episode_starts_per_clean_sequence_max,
        ) != (0.002, 0.005, 0.05):
            raise ValueError("M4 clean feasibility limits changed")
        return self


class ThresholdSelectionV1(ContractModel):
    """Frozen 36-candidate global threshold selection."""

    self_thresholds: tuple[FiniteFloat, ...]
    cross_thresholds: tuple[FiniteFloat, ...]
    candidate_count: Literal[36]
    iteration_order: Literal["self-threshold-ascending-outer-then-cross-threshold-ascending-inner"]
    selected_policy: Literal["combined-health-gate"]
    all_deployable_ablations_share_selected_thresholds: Literal[True]
    feasibility: ThresholdFeasibilityV1
    selection_conditions: tuple[Identifier, ...]
    objective: NonEmptyText
    tie_tolerance_m2: FiniteFloat
    tie_break_order: tuple[NonEmptyText, ...]
    no_feasible_candidate: Literal["hard-preregistration-failure-no-gate-relaxation"]
    retain_all_candidate_records: Literal[True]
    fit_artifact_must_exist_before_test: Literal[True]
    test_threshold_override: Literal[False]
    test_refit: Literal[False]

    @model_validator(mode="after")
    def require_frozen_grid(self) -> Self:
        if (
            self.self_thresholds != EXPECTED_M4_THRESHOLDS
            or self.cross_thresholds != EXPECTED_M4_THRESHOLDS
            or self.candidate_count != len(self.self_thresholds) * len(self.cross_thresholds)
            or len(set(self.self_thresholds)) != len(self.self_thresholds)
            or len(set(self.cross_thresholds)) != len(self.cross_thresholds)
            or len(set(self.selection_conditions)) != len(self.selection_conditions)
            or self.tie_tolerance_m2 != 1e-12
            or self.tie_break_order
            != (
                "lower-false-alert-episode-starts",
                "lower-clean-regression",
                "larger-self-threshold",
                "larger-cross-threshold",
            )
        ):
            raise ValueError("M4 threshold grid or selection order changed")
        return self


class FrameActionOracleV1(ContractModel):
    """Frame-granularity hindsight ceiling with deterministic exact-loss ties."""

    id: Literal["frame-action-performance-oracle"]
    action_granularity: Literal["one-action-per-frame-shared-by-all-eligible-objects"]
    candidate_actions: tuple[HealthMethodId, ...]
    selection_rule: NonEmptyText
    tie_break_order: tuple[HealthMethodId, ...]
    sequence_aggregation: NonEmptyText
    hindsight: Literal[True]
    deployable: Literal[False]
    dominance_gate: NonEmptyText
    common_mode_applicable: Literal[False]
    unequal_coverage_recovery_fraction: Literal[False]

    @model_validator(mode="after")
    def require_oracle_action_order(self) -> Self:
        expected = ("camera-only", "lidar-only", "fixed-fusion")
        if self.candidate_actions != expected or self.tie_break_order != self.candidate_actions:
            raise ValueError("frame-action oracle candidate or tie order changed")
        return self


class LossWindowsV1(ContractModel):
    """Loss windows for one event schedule."""

    score: FrameWindow
    event: FrameWindow
    recovery: FrameWindow


class LossWindowsByScheduleV1(ContractModel):
    """Schedule-specific estimand windows."""

    standard: LossWindowsV1
    cold_start: LossWindowsV1


class EvaluationV1(ContractModel):
    """Loss contrasts and support rules."""

    loss_windows_by_schedule: LossWindowsByScheduleV1
    primary_contrast: NonEmptyText
    clean_regression: NonEmptyText
    target_drop_gap: NonEmptyText
    frame_oracle_gap: NonEmptyText
    frame_oracle_recovery_fraction: NonEmptyText
    recovery_fraction_denominator_tolerance_m2: FiniteFloat
    recovery_fraction_clipped: Literal[False]
    abstaining_comparison: NonEmptyText
    dropout_comparison: NonEmptyText
    dropout_zero_loss_imputation: Literal[False]
    common_mode_has_healthy_target_drop_or_oracle: Literal[False]
    attribution_and_action_utility_are_separate_estimands: Literal[True]

    @model_validator(mode="after")
    def require_schedule_specific_windows(self) -> Self:
        windows = self.loss_windows_by_schedule
        if (
            windows.standard.score != (2, 48)
            or windows.standard.event != (12, 36)
            or windows.standard.recovery != (36, 48)
            or windows.cold_start.score != (0, 48)
            or windows.cold_start.event != (0, 24)
            or windows.cold_start.recovery != (24, 48)
            or self.recovery_fraction_denominator_tolerance_m2 != 1e-12
        ):
            raise ValueError("M4 evaluation windows changed")
        return self


class HealthMetricsV1(ContractModel):
    """Coverage-first event and recovery reporting semantics."""

    event_outcomes: tuple[NonEmptyText, ...]
    event_outcome_reduction: NonEmptyText
    common_mode_outcomes: NonEmptyText
    detection_fraction_reported_before_latency: Literal[True]
    detection_latency: NonEmptyText
    attribution_latency: NonEmptyText
    multiple_latch_episodes: NonEmptyText
    dropout_first_realized_missing_latency_also_reported: Literal[True] = Field(
        alias="dropout_first-realized-missing_latency_also_reported"
    )
    dropout_realized_fraction_reported_first: Literal[True]
    dropout_no_realization_outcome: NonEmptyText
    dropout_first_missing_latency_denominator: NonEmptyText
    early_clear: NonEmptyText
    state_at_final_fault_frame_reported: Literal[True]
    recovery_denominator: NonEmptyText
    recovery_latency: NonEmptyText
    false_alert_definition: NonEmptyText
    state_occupancy_reported: Literal[True]
    action_occupancy_reported: Literal[True]
    conditional_metric_defined_fraction_reported: Literal[True]
    frame_auroc_auprc: Literal["secondary-or-not-applicable"]
    brier_and_ece: Literal["not-applicable-nonprobabilistic-scores"]

    @model_validator(mode="after")
    def require_event_outcomes(self) -> Self:
        if self.event_outcomes != (
            "correct",
            "ambiguous",
            "wrong-sensor",
            "missed",
        ):
            raise ValueError("M4 event outcome partition changed")
        return self


class InferenceV1(ContractModel):
    """Complete-sequence paired bootstrap contract."""

    bootstrap_engine: Literal["numpy-pcg64dxsm-v1"]
    validation_bootstrap_seed: Literal[2718]
    test_bootstrap_seed: Literal[314159]
    bootstrap_replicates: Literal[2000]
    confidence_level: FiniteFloat
    interval: Literal["pointwise-percentile-linear-quantile"]
    resampling_unit: Literal["complete-base-sequence"]
    one_index_matrix_reused_across_all_variants_methods_windows_and_contrasts_per_split: Literal[
        True
    ]
    family_weighted_metrics_recomputed_inside_replicate: Literal[True]
    conditional_ratio_minimum_defined_replicate_fraction: FiniteFloat
    missed_events_remain_in_event_denominators: Literal[True]
    selection_uncertainty_in_test_intervals: Literal[False]
    multiplicity: Literal["pointwise-not-simultaneous"]

    @model_validator(mode="after")
    def require_frozen_inference_fractions(self) -> Self:
        if (
            self.confidence_level != 0.95
            or self.conditional_ratio_minimum_defined_replicate_fraction != 0.975
        ):
            raise ValueError("M4 inference fractions changed")
        return self


class ArtifactsV1(ContractModel):
    """Two-phase fit/evaluation artifact boundary."""

    fit_schema: Literal["ffb.health-fit-payload/v1"]
    evaluation_schema: Literal["ffb.health-eval-payload/v1"]
    fit_contents: tuple[NonEmptyText, ...]
    evaluation_contents: tuple[NonEmptyText, ...]
    no_overwrite: Literal[True]
    two_execution_scientific_byte_repeat: Literal[True]
    public_release_excludes_frame_features_and_sequence_rows: Literal[True]
    public_release_commits_omitted_rows_by_hash_length_and_count: Literal[True]


class ResourceCapsV1(ContractModel):
    """Hard CPU and evidence-size limits."""

    candidate_frame_evaluations_max: Literal[50_000_000]
    bootstrap_cells_max: Literal[100_000_000]
    peak_rss_bytes_max: Literal[1_073_741_824]
    wall_time_seconds_per_full_run_max: Literal[1800]
    curated_release_bytes_max: Literal[52_428_800]
    single_process_cpu_only: Literal[True]
    feature_computation_once_then_immutable_reuse: Literal[True]


class AcceptanceV1(ContractModel):
    """Evidence-completeness gates without a favorable-result requirement."""

    minimum_policy_gain: None
    minimum_detection_rate: None
    minimum_attribution_rate: None
    negative_result_releasable: Literal[True]
    required_controls: tuple[NonEmptyText, ...]
    full_local_verification: tuple[NonEmptyText, ...]


_EXPECTED_VALIDATION_MATRIX = (
    (
        "validation-camera-output-y-bias",
        "additive-position-bias",
        "camera",
        "y",
        "m",
        (-2.0, -0.5, 0.5, 2.0),
        "numeric-threshold-utility",
    ),
    (
        "validation-lidar-output-y-bias",
        "additive-position-bias",
        "lidar",
        "y",
        "m",
        (-2.0, -0.5, 0.5, 2.0),
        "numeric-threshold-utility",
    ),
    (
        "validation-camera-noise-underreported",
        "increased-noise-underreported",
        "camera",
        "xy",
        "std-scale",
        (1.5, 4.0),
        "numeric-threshold-utility",
    ),
    (
        "validation-lidar-noise-underreported",
        "increased-noise-underreported",
        "lidar",
        "xy",
        "std-scale",
        (1.5, 4.0),
        "numeric-threshold-utility",
    ),
    (
        "validation-camera-noise-correctly-reported",
        "increased-noise-correctly-reported",
        "camera",
        "xy",
        "std-scale",
        (1.5, 4.0),
        "numeric-threshold-utility-negative-control",
    ),
    (
        "validation-lidar-noise-correctly-reported",
        "increased-noise-correctly-reported",
        "lidar",
        "xy",
        "std-scale",
        (1.5, 4.0),
        "numeric-threshold-utility-negative-control",
    ),
    (
        "validation-camera-timestamp-offset",
        "timestamp-offset",
        "camera",
        "time",
        "s",
        (-0.4, -0.1, 0.1, 0.4),
        "direct-telemetry-validation-only",
    ),
    (
        "validation-lidar-timestamp-offset",
        "timestamp-offset",
        "lidar",
        "time",
        "s",
        (-0.4, -0.1, 0.1, 0.4),
        "direct-telemetry-validation-only",
    ),
    (
        "validation-camera-dropout",
        "dropout",
        "camera",
        "availability",
        "probability",
        (0.25, 0.75),
        "direct-telemetry-validation-only",
    ),
    (
        "validation-lidar-dropout",
        "dropout",
        "lidar",
        "availability",
        "probability",
        (0.25, 0.75),
        "direct-telemetry-validation-only",
    ),
    (
        "validation-camera-calibration-x",
        "calibration-translation",
        "camera",
        "x",
        "m",
        (-2.0, -0.5, 0.5, 2.0),
        "numeric-threshold-utility",
    ),
)

_EXPECTED_TEST_MATRIX = (
    (
        "test-camera-output-y-bias",
        "additive-position-bias",
        "camera",
        "y",
        "m",
        (-3.0, -0.75, 0.75, 3.0),
        "unseen-severity",
    ),
    (
        "test-lidar-output-y-bias",
        "additive-position-bias",
        "lidar",
        "y",
        "m",
        (-3.0, -0.75, 0.75, 3.0),
        "unseen-severity",
    ),
    (
        "test-camera-noise-underreported",
        "increased-noise-underreported",
        "camera",
        "xy",
        "std-scale",
        (1.25, 3.0),
        "unseen-severity",
    ),
    (
        "test-lidar-noise-underreported",
        "increased-noise-underreported",
        "lidar",
        "xy",
        "std-scale",
        (1.25, 3.0),
        "unseen-severity",
    ),
    (
        "test-camera-noise-correctly-reported",
        "increased-noise-correctly-reported",
        "camera",
        "xy",
        "std-scale",
        (1.25, 3.0),
        "unseen-severity-negative-control",
    ),
    (
        "test-lidar-noise-correctly-reported",
        "increased-noise-correctly-reported",
        "lidar",
        "xy",
        "std-scale",
        (1.25, 3.0),
        "unseen-severity-negative-control",
    ),
    (
        "test-camera-timestamp-offset",
        "timestamp-offset",
        "camera",
        "time",
        "s",
        (-0.6, -0.15, 0.15, 0.6),
        "unseen-severity-direct-telemetry",
    ),
    (
        "test-lidar-timestamp-offset",
        "timestamp-offset",
        "lidar",
        "time",
        "s",
        (-0.6, -0.15, 0.15, 0.6),
        "unseen-severity-direct-telemetry",
    ),
    (
        "test-camera-dropout",
        "dropout",
        "camera",
        "availability",
        "probability",
        (0.1, 0.5, 1.0),
        "unseen-severity-direct-telemetry",
    ),
    (
        "test-lidar-dropout",
        "dropout",
        "lidar",
        "availability",
        "probability",
        (0.1, 0.5, 1.0),
        "unseen-severity-direct-telemetry",
    ),
    (
        "test-camera-calibration-x",
        "calibration-translation",
        "camera",
        "x",
        "m",
        (-3.0, -0.75, 0.75, 3.0),
        "unseen-severity",
    ),
    (
        "test-camera-calibration-yaw",
        "calibration-yaw",
        "camera",
        "yaw",
        "rad",
        (-0.06, -0.015, 0.015, 0.06),
        "held-out-fault-family",
    ),
)

_EXPECTED_TEST_CONTROLS = (
    (
        "test-main-clean",
        "identity",
        "none",
        "none",
        "identity",
        (0.0,),
        "standard",
        "main-test",
        "clean-regression",
    ),
    (
        "test-edge-clean",
        "identity",
        "none",
        "none",
        "identity",
        (0.0,),
        "standard",
        "edge-test",
        "held-out-support-shift",
    ),
    (
        "test-clean-bounded-acceleration",
        "clean-predictor-mismatch",
        "none",
        "motion",
        "m/s^2",
        (8.0,),
        "standard",
        "main-test",
        "difficult-clean-predictor-mismatch",
    ),
    (
        "test-common-mode-x-edge",
        "common-mode-position-bias",
        "both",
        "x",
        "m",
        (-4.0, -1.0, 1.0, 4.0),
        "standard",
        "edge-test",
        "agreement-blind-spot",
    ),
    (
        "test-cold-start-camera-calibration-x",
        "calibration-translation",
        "camera",
        "x",
        "m",
        (3.0,),
        "cold_start",
        "main-test",
        "cold-start-diagnostic",
    ),
    (
        "test-cold-start-lidar-y-bias",
        "additive-position-bias",
        "lidar",
        "y",
        "m",
        (3.0,),
        "cold_start",
        "main-test",
        "cold-start-diagnostic",
    ),
)


def _condition_signature(
    condition: ValidationConditionV1 | TestConditionV1,
) -> tuple[str, str, str, str, str, tuple[float, ...], str]:
    return (
        condition.condition_id,
        condition.family,
        condition.target,
        condition.axis,
        condition.unit,
        tuple(condition.values),
        (
            condition.selection_role
            if isinstance(condition, ValidationConditionV1)
            else condition.test_role
        ),
    )


def _test_control_signature(
    control: TestControlV1,
) -> tuple[str, str, str, str, str, tuple[float, ...], str, str, str]:
    return (
        control.condition_id,
        control.family,
        control.target,
        control.axis,
        control.unit,
        tuple(control.values),
        control.schedule,
        control.population,
        control.test_role,
    )


class HealthBenchmarkIntentV1(ContractModel):
    """Complete immutable M4 scientific intent."""

    schema_id: Literal["ffb.health-benchmark-intent/v1"] = Field(alias="schema")
    benchmark_id: Literal["m4-health-v1"]
    release_id: Literal["m4-health-v0.1.0"]
    coordinate_contract: CoordinateContractV1
    source_population: SourcePopulationV1
    observation_model: ObservationModelV1
    event_schedules: EventSchedulesV1
    validation_matrix: Annotated[
        tuple[ValidationConditionV1, ...],
        Field(min_length=11, max_length=11),
    ]
    validation_controls: tuple[ValidationControlV1]
    test_matrix: Annotated[
        tuple[TestConditionV1, ...],
        Field(min_length=12, max_length=12),
    ]
    test_controls: Annotated[
        tuple[TestControlV1, ...],
        Field(min_length=6, max_length=6),
    ]
    predictor: PredictorV1
    features: FeaturesV1
    calibration: CalibrationV1
    policy: PolicyV1
    fault_target_drop_policy: FaultTargetDropPolicyV1
    methods: Annotated[
        tuple[HealthMethodId, ...],
        Field(min_length=10, max_length=10),
    ]
    threshold_selection: ThresholdSelectionV1
    oracle: FrameActionOracleV1
    evaluation: EvaluationV1
    health_metrics: HealthMetricsV1
    inference: InferenceV1
    artifacts: ArtifactsV1
    resource_caps: ResourceCapsV1
    acceptance: AcceptanceV1
    non_goals: Annotated[tuple[NonEmptyText, ...], Field(min_length=14, max_length=14)]

    @model_validator(mode="after")
    def require_frozen_health_intent(self) -> Self:
        all_conditions = (
            *self.validation_matrix,
            *self.validation_controls,
            *self.test_matrix,
            *self.test_controls,
        )
        condition_ids = tuple(item.condition_id for item in all_conditions)
        if len(set(condition_ids)) != len(condition_ids):
            raise ValueError("M4 condition IDs must be globally unique")

        clean = self.validation_controls
        if (
            len(clean) != 1
            or clean[0].condition_id != "validation-main-clean"
            or clean[0].values != (0.0,)
        ):
            raise ValueError("M4 requires the literal sole validation clean condition")

        validation_ids = {item.condition_id for item in self.validation_matrix}
        selection_ids = self.threshold_selection.selection_conditions
        if (
            not set(selection_ids).issubset(validation_ids)
            or any(
                next(
                    item for item in self.validation_matrix if item.condition_id == condition_id
                ).selection_role
                == "direct-telemetry-validation-only"
                for condition_id in selection_ids
            )
            or self.threshold_selection.feasibility.condition_id != clean[0].condition_id
        ):
            raise ValueError("M4 threshold selection references an ineligible condition")
        if selection_ids != EXPECTED_M4_SELECTION_CONDITIONS:
            raise ValueError("M4 threshold selection condition set or order changed")

        validation_yaw = [
            item for item in self.validation_matrix if item.family == "calibration-yaw"
        ]
        test_yaw = [item for item in self.test_matrix if item.family == "calibration-yaw"]
        if (
            validation_yaw
            or len(test_yaw) != 1
            or test_yaw[0].condition_id != "test-camera-calibration-yaw"
            or test_yaw[0].test_role != "held-out-fault-family"
        ):
            raise ValueError("camera calibration yaw must remain a held-out test family")

        cold_start_ids = tuple(
            item.condition_id for item in self.test_controls if item.schedule == "cold_start"
        )
        if cold_start_ids != (
            "test-cold-start-camera-calibration-x",
            "test-cold-start-lidar-y-bias",
        ):
            raise ValueError("M4 cold-start diagnostic set changed")
        if tuple(_test_control_signature(item) for item in self.test_controls) != (
            _EXPECTED_TEST_CONTROLS
        ):
            raise ValueError("M4 test control fault-operator matrix changed")
        if any(item.schedule != "standard" for item in self.test_matrix):
            raise ValueError("M4 primary test matrix must use the standard schedule")

        if self.methods != EXPECTED_M4_METHODS or len(set(self.methods)) != len(self.methods):
            raise ValueError("M4 method set or order changed")

        required_caps = (
            self.resource_caps.candidate_frame_evaluations_max,
            self.resource_caps.bootstrap_cells_max,
            self.resource_caps.peak_rss_bytes_max,
            self.resource_caps.wall_time_seconds_per_full_run_max,
            self.resource_caps.curated_release_bytes_max,
        )
        if required_caps != (
            50_000_000,
            100_000_000,
            1_073_741_824,
            1800,
            52_428_800,
        ):
            raise ValueError("M4 resource caps changed")

        if tuple(_condition_signature(item) for item in self.validation_matrix) != (
            _EXPECTED_VALIDATION_MATRIX
        ):
            raise ValueError("M4 validation fault-operator matrix changed")
        if tuple(_condition_signature(item) for item in self.test_matrix) != (
            _EXPECTED_TEST_MATRIX
        ):
            raise ValueError("M4 test fault-operator matrix changed")

        if len(set(self.non_goals)) != len(self.non_goals):
            raise ValueError("M4 non-goals must be unique")
        if sha256_digest(self) != M4_HEALTH_INTENT_SHA256:
            raise ValueError("M4 health intent canonical digest is not preregistered")
        return self


HEALTH_BENCHMARK_INTENT_ADAPTER = TypeAdapter(HealthBenchmarkIntentV1)


@dataclass(frozen=True, slots=True)
class LoadedHealthBenchmarkIntent:
    """One exact frozen health intent and its canonical digest."""

    path: Path
    intent: HealthBenchmarkIntentV1
    intent_sha256: str


def _safe_intent_path(path: Path, *, source_root: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(source_root)))
    if not root.is_dir() or root.is_symlink() or root.resolve(strict=True) != root:
        raise ValueError("source_root must be one existing real directory")
    candidate = root / path if not path.is_absolute() else path
    absolute = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise ValueError("health intent path must remain inside source_root") from error
    if relative != M4_HEALTH_INTENT_PATH:
        raise ValueError(f"health intent must be {M4_HEALTH_INTENT_PATH.as_posix()}")
    current = root
    try:
        for part in relative.parts:
            current /= part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("health intent path must not use symlinks")
    except ValueError:
        raise
    except OSError as error:
        raise ValueError("health intent path is unavailable") from error
    metadata = absolute.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _INTENT_FILE_CAP_BYTES:
        raise ValueError("health intent must be one bounded regular file")
    return absolute


def load_health_benchmark_intent(
    path: Path = M4_HEALTH_INTENT_PATH,
    *,
    source_root: Path,
) -> LoadedHealthBenchmarkIntent:
    """Strictly load the exact preregistered M4 intent."""

    intent_path = _safe_intent_path(path, source_root=source_root)
    load_json_object(intent_path)
    raw = intent_path.read_text(encoding="utf-8")
    intent = HEALTH_BENCHMARK_INTENT_ADAPTER.validate_json(raw)
    digest = sha256_digest(intent)
    if digest != M4_HEALTH_INTENT_SHA256:
        raise ValueError("M4 health intent canonical digest is not preregistered")
    return LoadedHealthBenchmarkIntent(
        path=intent_path,
        intent=intent,
        intent_sha256=digest,
    )


def health_benchmark_intent_json_schema() -> dict[str, Any]:
    """Return the strict public schema for the M4 benchmark intent."""

    return HEALTH_BENCHMARK_INTENT_ADAPTER.json_schema(by_alias=True)
