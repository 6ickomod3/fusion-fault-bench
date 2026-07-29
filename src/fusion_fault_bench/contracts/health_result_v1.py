"""Strict M4 fit, sequence, aggregate, and validation records."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, FiniteFloat, model_validator

from fusion_fault_bench.contracts.common import ContractModel
from fusion_fault_bench.contracts.result_v1alpha1 import Digest, Identifier

type HealthMethod = Literal[
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
type HealthPolicy = Literal[
    "self-nis-gate",
    "cross-nis-gate",
    "direct-telemetry-gate",
    "combined-health-gate",
]
type HealthPolicyMethod = Literal[
    "self-nis-gate",
    "cross-nis-gate",
    "direct-telemetry-gate",
    "combined-health-gate",
    "combined-health-gate-abstain",
]
type HealthState = Literal["healthy", "camera-fault", "lidar-fault", "ambiguous"]
type EventOutcome = Literal["correct", "ambiguous", "wrong-sensor", "missed"]
type HealthWindow = Literal["score", "event", "recovery"]
type MetricStatus = Literal["ok", "undefined", "not-applicable"]

SELF_THRESHOLDS = (0.95, 0.975, 0.99, 0.995, 0.999, 1.0)
CROSS_THRESHOLDS = SELF_THRESHOLDS


class HealthThresholdCandidateV1(ContractModel):
    """One of the 36 validation candidates in exact iteration order."""

    schema_id: Literal["ffb.health-threshold-candidate/v1"] = Field(alias="schema")
    candidate_index: Annotated[int, Field(ge=0, le=35)]
    self_threshold: Annotated[FiniteFloat, Field(ge=0.95, le=1.0)]
    cross_threshold: Annotated[FiniteFloat, Field(ge=0.95, le=1.0)]
    mean_clean_regression_m2: FiniteFloat
    upper_95pct_clean_regression_m2: FiniteFloat
    false_alert_episode_starts_per_sequence: Annotated[FiniteFloat, Field(ge=0.0)]
    clean_coverage: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    fixed_clean_coverage: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    feasible: bool
    validation_regret_m2: Annotated[FiniteFloat, Field(ge=0.0)]

    @model_validator(mode="after")
    def require_grid_order_and_feasibility(self) -> Self:
        expected_self = SELF_THRESHOLDS[self.candidate_index // len(CROSS_THRESHOLDS)]
        expected_cross = CROSS_THRESHOLDS[self.candidate_index % len(CROSS_THRESHOLDS)]
        if self.self_threshold != expected_self or self.cross_threshold != expected_cross:
            raise ValueError("threshold candidate does not follow the frozen Cartesian order")
        expected_feasible = (
            self.mean_clean_regression_m2 <= 0.002
            and self.upper_95pct_clean_regression_m2 <= 0.005
            and self.false_alert_episode_starts_per_sequence <= 0.05
            and self.clean_coverage == self.fixed_clean_coverage
        )
        if self.feasible != expected_feasible:
            raise ValueError("threshold feasibility disagrees with frozen clean gates")
        return self


class HealthFitSummaryV1(ContractModel):
    """Content-bound clean fit and selected validation policy."""

    schema_id: Literal["ffb.health-fit-summary/v1"] = Field(alias="schema")
    intent_sha256: Digest
    main_profile_sha256: Digest
    edge_profile_sha256: Digest
    train_sequence_count: Literal[200]
    validation_sequence_count: Literal[200]
    ecdf_channel_count: Literal[8]
    ecdf_values_per_channel: Literal[9200]
    candidate_count: Literal[36]
    selected_candidate_index: Annotated[int, Field(ge=0, le=35)]
    selected_self_threshold: Annotated[FiniteFloat, Field(ge=0.95, le=1.0)]
    selected_cross_threshold: Annotated[FiniteFloat, Field(ge=0.95, le=1.0)]
    selection_status: Literal["selected"]

    @model_validator(mode="after")
    def require_selected_grid_coordinate(self) -> Self:
        expected_self = SELF_THRESHOLDS[self.selected_candidate_index // 6]
        expected_cross = CROSS_THRESHOLDS[self.selected_candidate_index % 6]
        if (
            self.selected_self_threshold != expected_self
            or self.selected_cross_threshold != expected_cross
        ):
            raise ValueError("selected thresholds disagree with candidate index")
        return self


class HealthSequenceLossV1(ContractModel):
    """One method/window sequence statistic before population inference."""

    schema_id: Literal["ffb.health-sequence-loss/v1"] = Field(alias="schema")
    sequence_id: Identifier
    condition_id: Identifier
    method: HealthMethod
    window: HealthWindow
    loss_sum_m2: Annotated[FiniteFloat, Field(ge=0.0)]
    valid_object_frame_count: Annotated[int, Field(ge=0)]
    eligible_object_frame_count: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def require_valid_subset(self) -> Self:
        if self.valid_object_frame_count > self.eligible_object_frame_count:
            raise ValueError("valid count cannot exceed eligible count")
        if self.valid_object_frame_count == 0 and self.loss_sum_m2 != 0.0:
            raise ValueError("undefined sequence/window loss must have zero accumulated loss")
        return self


class HealthSequenceContrastV1(ContractModel):
    """Common-support sufficient statistics for one policy/window contrast."""

    schema_id: Literal["ffb.health-sequence-contrast/v1"] = Field(alias="schema")
    sequence_id: Identifier
    condition_id: Identifier
    policy: HealthPolicyMethod
    window: HealthWindow
    fixed_support_sha256: Digest
    policy_support_sha256: Digest
    fixed_policy_common_count: Annotated[int, Field(ge=0)]
    fixed_on_common_loss_sum_m2: Annotated[FiniteFloat, Field(ge=0.0)]
    policy_on_fixed_common_loss_sum_m2: Annotated[FiniteFloat, Field(ge=0.0)]
    target_drop_applicable: bool
    policy_target_drop_common_count: Annotated[int, Field(ge=0)] | None
    policy_on_target_common_loss_sum_m2: Annotated[FiniteFloat, Field(ge=0.0)] | None
    target_drop_on_common_loss_sum_m2: Annotated[FiniteFloat, Field(ge=0.0)] | None
    frame_oracle_applicable: bool
    policy_frame_oracle_common_count: Annotated[int, Field(ge=0)] | None
    policy_on_oracle_common_loss_sum_m2: Annotated[FiniteFloat, Field(ge=0.0)] | None
    frame_oracle_on_common_loss_sum_m2: Annotated[FiniteFloat, Field(ge=0.0)] | None
    frame_oracle_support_sha256: Digest | None

    @staticmethod
    def _require_zero_sums_for_empty_support(
        count: int,
        sums: tuple[FiniteFloat, FiniteFloat],
        *,
        label: str,
    ) -> None:
        if count == 0 and any(value != 0.0 for value in sums):
            raise ValueError(f"{label} zero support requires zero loss sums")

    @model_validator(mode="after")
    def require_applicability_and_support_consistency(self) -> Self:
        self._require_zero_sums_for_empty_support(
            self.fixed_policy_common_count,
            (
                self.fixed_on_common_loss_sum_m2,
                self.policy_on_fixed_common_loss_sum_m2,
            ),
            label="fixed-policy",
        )
        target_values = (
            self.policy_target_drop_common_count,
            self.policy_on_target_common_loss_sum_m2,
            self.target_drop_on_common_loss_sum_m2,
        )
        if self.target_drop_applicable and not all(value is not None for value in target_values):
            raise ValueError("target-drop applicability must match its sufficient statistics")
        if not self.target_drop_applicable and any(value is not None for value in target_values):
            raise ValueError("inapplicable target-drop statistics must be null")
        if self.target_drop_applicable:
            assert self.policy_target_drop_common_count is not None
            assert self.policy_on_target_common_loss_sum_m2 is not None
            assert self.target_drop_on_common_loss_sum_m2 is not None
            self._require_zero_sums_for_empty_support(
                self.policy_target_drop_common_count,
                (
                    self.policy_on_target_common_loss_sum_m2,
                    self.target_drop_on_common_loss_sum_m2,
                ),
                label="policy-target-drop",
            )

        oracle_values = (
            self.policy_frame_oracle_common_count,
            self.policy_on_oracle_common_loss_sum_m2,
            self.frame_oracle_on_common_loss_sum_m2,
            self.frame_oracle_support_sha256,
        )
        if self.frame_oracle_applicable and not all(value is not None for value in oracle_values):
            raise ValueError("frame-oracle applicability must match its sufficient statistics")
        if not self.frame_oracle_applicable and any(value is not None for value in oracle_values):
            raise ValueError("inapplicable frame-oracle statistics must be null")
        if self.frame_oracle_applicable:
            assert self.policy_frame_oracle_common_count is not None
            assert self.policy_on_oracle_common_loss_sum_m2 is not None
            assert self.frame_oracle_on_common_loss_sum_m2 is not None
            self._require_zero_sums_for_empty_support(
                self.policy_frame_oracle_common_count,
                (
                    self.policy_on_oracle_common_loss_sum_m2,
                    self.frame_oracle_on_common_loss_sum_m2,
                ),
                label="policy-frame-oracle",
            )

        if self.identical_support_recovery_applicable and (
            self.fixed_policy_common_count != self.policy_frame_oracle_common_count
            or self.policy_on_fixed_common_loss_sum_m2 != self.policy_on_oracle_common_loss_sum_m2
        ):
            raise ValueError("recovery applicability requires identical nonempty three-way support")
        return self

    @property
    def identical_support_recovery_applicable(self) -> bool:
        """Whether fixed, policy, and oracle commit to one nonempty support."""

        return (
            self.frame_oracle_applicable
            and self.fixed_policy_common_count > 0
            and self.fixed_support_sha256
            == self.policy_support_sha256
            == self.frame_oracle_support_sha256
        )


class HealthSequenceEventV1(ContractModel):
    """One sequence-level policy event and censoring record."""

    schema_id: Literal["ffb.health-sequence-event/v1"] = Field(alias="schema")
    sequence_id: Identifier
    condition_id: Identifier
    policy: HealthPolicy
    detected: bool
    detection_latency_frames: Annotated[int, Field(ge=0, le=23)] | None
    first_latch_label: HealthState | None
    outcome: EventOutcome
    correctly_attributed: bool
    attribution_latency_frames: Annotated[int, Field(ge=0, le=23)] | None
    realized_dropout: bool | None
    first_missing_frame_minus_event_start: Annotated[int, Field(ge=0, le=23)] | None
    detection_minus_first_missing_frames: Annotated[int, Field(ge=-23, le=23)] | None
    latch_episode_count: Annotated[int, Field(ge=0, le=24)]
    false_alert_episode_count: Annotated[int, Field(ge=0, le=24)]
    early_clear: bool
    final_active_state: HealthState
    active_frame_count: Literal[24]
    active_healthy_frames: Annotated[int, Field(ge=0)]
    active_camera_fault_frames: Annotated[int, Field(ge=0)]
    active_lidar_fault_frames: Annotated[int, Field(ge=0)]
    active_ambiguous_frames: Annotated[int, Field(ge=0)]
    active_camera_action_frames: Annotated[int, Field(ge=0)]
    active_lidar_action_frames: Annotated[int, Field(ge=0)]
    active_fixed_action_frames: Annotated[int, Field(ge=0)]
    active_undefined_action_frames: Annotated[int, Field(ge=0)]
    recovery_eligible: bool
    recovered: bool
    recovery_latency_frames: Annotated[int, Field(ge=0, le=23)] | None

    @model_validator(mode="after")
    def require_censoring_consistency(self) -> Self:
        if self.detected != (self.detection_latency_frames is not None):
            raise ValueError("detection latency must be defined exactly when detected")
        if self.detected != (self.first_latch_label is not None):
            raise ValueError("first latch label must be defined exactly when detected")
        if self.first_latch_label == "healthy":
            raise ValueError("a detected event cannot first latch healthy")
        if self.correctly_attributed != (self.attribution_latency_frames is not None):
            raise ValueError("attribution latency must be defined exactly when correct")
        if self.correctly_attributed and not self.detected:
            raise ValueError("correct attribution requires a detected event")
        if self.correctly_attributed:
            assert self.attribution_latency_frames is not None
            assert self.detection_latency_frames is not None
            if self.attribution_latency_frames < self.detection_latency_frames:
                raise ValueError("attribution cannot precede detection")
        if self.outcome == "correct" and (
            not self.correctly_attributed
            or self.attribution_latency_frames != self.detection_latency_frames
        ):
            raise ValueError("correct first attribution must occur at detection")
        if self.correctly_attributed and self.outcome != "correct":
            assert self.attribution_latency_frames is not None
            assert self.detection_latency_frames is not None
            if self.attribution_latency_frames <= self.detection_latency_frames:
                raise ValueError("later correct attribution must follow first detection")
            if self.latch_episode_count < 2:
                raise ValueError("later correct attribution requires a later latch episode")
        if self.detected and self.latch_episode_count < 1:
            raise ValueError("a detected event requires a latch episode")
        if self.early_clear and not self.detected:
            raise ValueError("early clear requires a detected event")
        if self.recovery_eligible != (self.final_active_state != "healthy"):
            raise ValueError("recovery eligibility must follow the final active state")
        if self.recovered and not self.recovery_eligible:
            raise ValueError("recovery requires an eligible nonhealthy final active state")
        if self.recovered != (self.recovery_latency_frames is not None):
            raise ValueError("recovery latency must be defined exactly when recovered")
        if not self.detected and self.outcome != "missed":
            raise ValueError("undetected events must have missed outcome")
        realized = self.realized_dropout is True
        if realized != (self.first_missing_frame_minus_event_start is not None):
            raise ValueError("first-missing onset must be defined exactly for realized dropout")
        if (realized and self.detected) != (self.detection_minus_first_missing_frames is not None):
            raise ValueError("dropout response latency requires realized dropout and detection")
        if self.detection_minus_first_missing_frames is not None:
            assert self.detection_latency_frames is not None
            assert self.first_missing_frame_minus_event_start is not None
            if (
                self.detection_minus_first_missing_frames
                != self.detection_latency_frames - self.first_missing_frame_minus_event_start
            ):
                raise ValueError(
                    "dropout response latency must equal detection minus missing onset"
                )
        if (
            self.active_healthy_frames
            + self.active_camera_fault_frames
            + self.active_lidar_fault_frames
            + self.active_ambiguous_frames
            != self.active_frame_count
        ):
            raise ValueError("active state occupancy must sum to active_frame_count")
        if (
            self.active_camera_action_frames
            + self.active_lidar_action_frames
            + self.active_fixed_action_frames
            + self.active_undefined_action_frames
            != self.active_frame_count
        ):
            raise ValueError("active action occupancy must sum to active_frame_count")
        final_state_count = {
            "healthy": self.active_healthy_frames,
            "camera-fault": self.active_camera_fault_frames,
            "lidar-fault": self.active_lidar_fault_frames,
            "ambiguous": self.active_ambiguous_frames,
        }[self.final_active_state]
        if final_state_count == 0:
            raise ValueError("final active state must occur in the active window")
        return self


class HealthAggregateMetricV1(ContractModel):
    """One sequence-bootstrap population estimate."""

    schema_id: Literal["ffb.health-aggregate/v1"] = Field(alias="schema")
    condition_id: Identifier
    method: HealthMethod | None
    metric_name: Identifier
    window: HealthWindow | None
    unit: Literal["m^2", "fraction", "frames", "count-per-sequence"]
    status: MetricStatus
    estimate: FiniteFloat | None
    interval_lower: FiniteFloat | None
    interval_upper: FiniteFloat | None
    sequence_count: Annotated[int, Field(ge=1)]
    bootstrap_replicates: Literal[2000]
    defined_bootstrap_replicates: Annotated[int, Field(ge=0, le=2000)]

    @model_validator(mode="after")
    def require_interval_status(self) -> Self:
        values = (self.estimate, self.interval_lower, self.interval_upper)
        if self.status == "ok":
            if any(value is None for value in values):
                raise ValueError("ok aggregate metrics require an estimate and interval")
            if self.interval_lower > self.interval_upper:  # type: ignore[operator]
                raise ValueError("aggregate interval is reversed")
        elif any(value is not None for value in values):
            raise ValueError("non-ok aggregate metrics cannot carry numeric estimates")
        return self


class HealthValidationCheckV1(ContractModel):
    """One named release-blocking M4 validation check."""

    check_id: Identifier
    passed: bool
    observed: FiniteFloat | int | bool | str
    expected: FiniteFloat | int | bool | str


class HealthValidationV1(ContractModel):
    """Conjunctive M4 scientific and implementation validation evidence."""

    schema_id: Literal["ffb.health-validation/v1"] = Field(alias="schema")
    intent_sha256: Digest
    checks: Annotated[tuple[HealthValidationCheckV1, ...], Field(min_length=1)]
    all_checks_passed: bool

    @model_validator(mode="after")
    def require_unique_conjunction(self) -> Self:
        check_ids = tuple(check.check_id for check in self.checks)
        if len(set(check_ids)) != len(check_ids):
            raise ValueError("health validation check IDs must be unique")
        if self.all_checks_passed != all(check.passed for check in self.checks):
            raise ValueError("all_checks_passed must be the check conjunction")
        return self
