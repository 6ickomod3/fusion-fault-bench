"""Strict replay-only health contracts for variable-length M5 sequences."""

from __future__ import annotations

import math
from typing import Annotated, Literal, Self

from pydantic import Field, FiniteFloat, model_validator

from fusion_fault_bench.contracts.common import ContractModel
from fusion_fault_bench.contracts.health_result_v1 import (
    EventOutcome,
    HealthMethod,
    HealthPolicy,
    HealthState,
    HealthWindow,
)
from fusion_fault_bench.contracts.replay_v1 import (
    M5_HEALTH_EXPERIMENT_IDS,
    M5_SCENE_NAMES,
)
from fusion_fault_bench.contracts.result_v1alpha1 import Digest, Identifier

type ReplayFaultFamily = Literal[
    "identity",
    "additive-position-bias",
    "increased-noise-underreported",
    "increased-noise-correctly-reported",
    "timestamp-offset",
    "dropout",
    "calibration-translation",
    "calibration-yaw",
    "common-mode-position-bias",
]
type ReplayFaultTarget = Literal["camera", "lidar", "both", "none"]
ReplayConditionSelector = Annotated[
    str,
    Field(
        max_length=128,
        pattern=(
            r"^replay-[a-z0-9][a-z0-9-]*:"
            r"(?:0|[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))$"
        ),
    ),
]
ReplayObjectIdentifier = Annotated[
    str,
    Field(pattern=r"^track:[0-9]{4,}$", max_length=127),
]
_REPLAY_SEQUENCE_IDS = frozenset(f"nuscenes:{name}" for name in M5_SCENE_NAMES)


class ReplayHealthScheduleV1(ContractModel):
    """Variable-length replay schedule derived only from the frame count."""

    schema_id: Literal["ffb.replay-health-schedule/v1"] = Field(alias="schema")
    frame_count: Annotated[int, Field(ge=16)]
    predictor_initialization_frames: tuple[int, int]
    clean_prefix_frames: tuple[int, int]
    score_frames: tuple[int, int]
    fault_active_frames: tuple[int, int]
    recovery_frames: tuple[int, int]

    @model_validator(mode="after")
    def require_frozen_schedule(self) -> Self:
        a = self.frame_count // 4
        b = (3 * self.frame_count) // 4
        expected = (
            (self.predictor_initialization_frames, (0, 2)),
            (self.clean_prefix_frames, (0, a)),
            (self.score_frames, (2, self.frame_count)),
            (self.fault_active_frames, (a, b)),
            (self.recovery_frames, (b, self.frame_count)),
        )
        if any(observed != required for observed, required in expected):
            raise ValueError("replay health windows do not match the frozen frame-count equations")
        if a < 4 or b - a < 2 or self.frame_count - b < 3:
            raise ValueError("replay health schedule violates a frozen phase constraint")
        return self

    @property
    def event_start(self) -> int:
        """Return the inclusive active-event start."""

        return self.fault_active_frames[0]

    @property
    def event_end(self) -> int:
        """Return the exclusive active-event end."""

        return self.fault_active_frames[1]

    @property
    def active_frame_count(self) -> int:
        """Return the number of active observation steps."""

        return self.event_end - self.event_start

    @property
    def recovery_frame_count(self) -> int:
        """Return the number of recovery observation steps."""

        return self.recovery_frames[1] - self.recovery_frames[0]


class ReplayModalityMeasurementV1(ContractModel):
    """One aligned estimator value and estimator-reported covariance."""

    value_xy_m: tuple[FiniteFloat, FiniteFloat]
    reported_covariance_xy_m2: tuple[
        tuple[FiniteFloat, FiniteFloat],
        tuple[FiniteFloat, FiniteFloat],
    ]
    reported_time_s: FiniteFloat

    @model_validator(mode="after")
    def require_spd_covariance(self) -> Self:
        (a, b), (c, d) = self.reported_covariance_xy_m2
        if not math.isclose(float(b), float(c), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("reported covariance must be symmetric")
        if float(a) <= 0.0 or float(a) * float(d) - float(b) * float(c) <= 0.0:
            raise ValueError("reported covariance must be positive definite")
        return self


class ReplayObjectHealthInputV1(ContractModel):
    """One opaque known-object input in a replay frame."""

    object_id: ReplayObjectIdentifier
    camera: ReplayModalityMeasurementV1 | None
    lidar: ReplayModalityMeasurementV1 | None


class ReplayHealthFrameInputV1(ContractModel):
    """Replay frame input; unlike M4, the object tuple may be empty."""

    schema_id: Literal["ffb.replay-health-frame-input/v1"] = Field(alias="schema")
    sequence_id: Identifier
    frame_index: Annotated[int, Field(ge=0)]
    reference_time_s: FiniteFloat
    camera_available: bool
    lidar_available: bool
    objects: tuple[ReplayObjectHealthInputV1, ...]

    @model_validator(mode="after")
    def require_order_and_availability(self) -> Self:
        if self.sequence_id not in _REPLAY_SEQUENCE_IDS:
            raise ValueError("replay health frame lies outside the frozen scene population")
        identifiers = tuple(item.object_id for item in self.objects)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("object_id values must be unique within a replay frame")
        if identifiers != tuple(sorted(identifiers, key=lambda value: value.encode("utf-8"))):
            raise ValueError("replay frame objects must use frozen UTF-8 object order")
        for item in self.objects:
            if not self.camera_available and item.camera is not None:
                raise ValueError("camera measurement cannot exist when camera is unavailable")
            if not self.lidar_available and item.lidar is not None:
                raise ValueError("lidar measurement cannot exist when lidar is unavailable")
        return self


class ReplayNumericChannelEvidenceV1(ContractModel):
    """One replay NIS channel, including exact zero-object support."""

    status: Literal["defined", "insufficient-support"]
    mature_object_count: Annotated[int, Field(ge=0)]
    current_object_count: Annotated[int, Field(ge=0)]
    mature_fraction: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    mean_nis: Annotated[FiniteFloat, Field(ge=0.0)] | None
    maximum_nis: Annotated[FiniteFloat, Field(ge=0.0)] | None
    score: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)] | None

    @model_validator(mode="after")
    def require_support_consistency(self) -> Self:
        if self.mature_object_count > self.current_object_count:
            raise ValueError("mature support cannot exceed current support")
        expected_fraction = (
            0.0
            if self.current_object_count == 0
            else self.mature_object_count / self.current_object_count
        )
        if not math.isclose(
            float(self.mature_fraction),
            expected_fraction,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("mature fraction does not match replay support counts")
        statistics = (self.mean_nis, self.maximum_nis, self.score)
        if self.status == "defined":
            if self.mature_object_count < 2 or any(value is None for value in statistics):
                raise ValueError(
                    "defined replay evidence requires two mature objects and statistics"
                )
        elif self.mature_object_count >= 2 or any(value is not None for value in statistics):
            raise ValueError("insufficient replay evidence requires null statistics")
        if self.current_object_count == 0 and (
            self.status != "insufficient-support"
            or self.mature_object_count != 0
            or self.mature_fraction != 0.0
        ):
            raise ValueError("zero-object evidence must use the frozen empty-frame representation")
        return self


class ReplayHealthEvidenceV1(ContractModel):
    """Serializable replay health evidence produced before current updates."""

    schema_id: Literal["ffb.replay-health-evidence/v1"] = Field(alias="schema")
    sequence_id: Identifier
    frame_index: Annotated[int, Field(ge=0)]
    reference_time_s: FiniteFloat
    camera_available: bool
    lidar_available: bool
    camera_timestamp_suspicious: bool
    lidar_timestamp_suspicious: bool
    camera_missing_fraction_last_four: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    lidar_missing_fraction_last_four: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    camera_self: ReplayNumericChannelEvidenceV1
    lidar_self: ReplayNumericChannelEvidenceV1
    camera_from_lidar_cross: ReplayNumericChannelEvidenceV1
    lidar_from_camera_cross: ReplayNumericChannelEvidenceV1

    @model_validator(mode="after")
    def require_aligned_channel_support(self) -> Self:
        if self.sequence_id not in _REPLAY_SEQUENCE_IDS:
            raise ValueError("replay health evidence lies outside the frozen scene population")
        counts = {
            channel.current_object_count
            for channel in (
                self.camera_self,
                self.lidar_self,
                self.camera_from_lidar_cross,
                self.lidar_from_camera_cross,
            )
        }
        if len(counts) != 1:
            raise ValueError("replay evidence channels must share current object support")
        if counts == {0} and (self.camera_timestamp_suspicious or self.lidar_timestamp_suspicious):
            raise ValueError("an empty replay frame cannot contain timestamp evidence")
        return self


class ReplayHealthSequenceEventV1(ContractModel):
    """Dynamic replay event, occupancy, censoring, and elapsed-time record."""

    schema_id: Literal["ffb.replay-health-sequence-event/v1"] = Field(alias="schema")
    replay_experiment_identity_sha256: Digest
    sequence_id: Identifier
    condition_id: Identifier
    condition_selector: ReplayConditionSelector
    policy: HealthPolicy
    fault_family: ReplayFaultFamily
    fault_target: ReplayFaultTarget
    schedule: ReplayHealthScheduleV1
    detected: bool
    detection_latency_steps: Annotated[int, Field(ge=0)] | None
    detection_latency_s: Annotated[FiniteFloat, Field(ge=0.0)] | None
    detection_censor_bound_steps: Annotated[int, Field(ge=1)]
    first_latch_label: HealthState | None
    outcome: EventOutcome
    correctly_attributed: bool
    attribution_latency_steps: Annotated[int, Field(ge=0)] | None
    attribution_latency_s: Annotated[FiniteFloat, Field(ge=0.0)] | None
    attribution_censor_bound_steps: Annotated[int, Field(ge=1)]
    realized_dropout: bool | None
    first_missing_step: Annotated[int, Field(ge=0)] | None
    first_missing_latency_s: Annotated[FiniteFloat, Field(ge=0.0)] | None
    detection_minus_first_missing_steps: int | None
    detection_minus_first_missing_s: FiniteFloat | None
    latch_episode_count: Annotated[int, Field(ge=0)]
    false_alert_episode_count: Annotated[int, Field(ge=0)]
    early_clear: bool
    final_active_state: HealthState
    active_healthy_steps: Annotated[int, Field(ge=0)]
    active_camera_fault_steps: Annotated[int, Field(ge=0)]
    active_lidar_fault_steps: Annotated[int, Field(ge=0)]
    active_ambiguous_steps: Annotated[int, Field(ge=0)]
    active_camera_action_steps: Annotated[int, Field(ge=0)]
    active_lidar_action_steps: Annotated[int, Field(ge=0)]
    active_fixed_action_steps: Annotated[int, Field(ge=0)]
    active_undefined_action_steps: Annotated[int, Field(ge=0)]
    recovery_eligible: bool
    recovered: bool
    recovery_latency_steps: Annotated[int, Field(ge=0)] | None
    recovery_latency_s: Annotated[FiniteFloat, Field(ge=0.0)] | None
    recovery_censor_bound_steps: Annotated[int, Field(ge=3)]

    @model_validator(mode="after")
    def require_dynamic_event_consistency(self) -> Self:
        if self.sequence_id not in _REPLAY_SEQUENCE_IDS:
            raise ValueError("replay event lies outside the frozen scene population")
        if self.condition_id not in M5_HEALTH_EXPERIMENT_IDS:
            raise ValueError("replay event condition is not in the frozen M5-B matrix")
        if not self.condition_selector.startswith(f"{self.condition_id}:"):
            raise ValueError("condition selector must bind the base condition_id")
        if self.fault_family == "identity" and self.fault_target != "none":
            raise ValueError("identity replay events require target none")
        if self.fault_family == "common-mode-position-bias" and self.fault_target != "both":
            raise ValueError("common-mode replay events require target both")
        if self.fault_family in {"calibration-translation", "calibration-yaw"} and (
            self.fault_target != "camera"
        ):
            raise ValueError("replay calibration faults require target camera")
        if self.fault_family not in {
            "identity",
            "common-mode-position-bias",
            "calibration-translation",
            "calibration-yaw",
        } and self.fault_target not in {"camera", "lidar"}:
            raise ValueError("single-target replay faults require camera or lidar target")

        active_count = self.schedule.active_frame_count
        recovery_count = self.schedule.recovery_frame_count
        if (
            self.detection_censor_bound_steps != active_count
            or self.attribution_censor_bound_steps != active_count
            or self.recovery_censor_bound_steps != recovery_count
        ):
            raise ValueError("event censor bounds must equal the dynamic window lengths")
        observed_active_steps = (
            self.detection_latency_steps,
            self.attribution_latency_steps,
            self.first_missing_step,
        )
        if any(value is not None and value >= active_count for value in observed_active_steps):
            raise ValueError("observed active latency lies outside the dynamic active window")
        if (
            self.recovery_latency_steps is not None
            and self.recovery_latency_steps >= recovery_count
        ):
            raise ValueError("observed recovery latency lies outside the recovery window")
        if self.detected != (self.detection_latency_steps is not None):
            raise ValueError("detection step must be defined exactly when detected")
        if self.detected != (self.detection_latency_s is not None):
            raise ValueError("detection seconds must be defined exactly when detected")
        if self.detected != (self.first_latch_label is not None):
            raise ValueError("first latch label must be defined exactly when detected")
        if self.first_latch_label == "healthy":
            raise ValueError("a detected event cannot first latch healthy")
        if self.correctly_attributed != (self.attribution_latency_steps is not None):
            raise ValueError("attribution step must be defined exactly when attributed")
        if self.correctly_attributed != (self.attribution_latency_s is not None):
            raise ValueError("attribution seconds must be defined exactly when attributed")
        if self.correctly_attributed and not self.detected:
            raise ValueError("correct attribution requires detection")
        if self.correctly_attributed:
            assert self.attribution_latency_steps is not None
            assert self.detection_latency_steps is not None
            if self.attribution_latency_steps < self.detection_latency_steps:
                raise ValueError("correct attribution cannot precede detection")
        if self.outcome == "correct" and (
            not self.correctly_attributed
            or self.attribution_latency_steps != self.detection_latency_steps
        ):
            raise ValueError("correct first attribution must occur at detection")
        if self.correctly_attributed and self.outcome != "correct":
            assert self.attribution_latency_steps is not None
            assert self.detection_latency_steps is not None
            if (
                self.attribution_latency_steps <= self.detection_latency_steps
                or self.latch_episode_count < 2
            ):
                raise ValueError("later correct attribution requires a later latch episode")
        if not self.detected and self.outcome != "missed":
            raise ValueError("undetected replay events must be missed")
        if self.detected and self.latch_episode_count < 1:
            raise ValueError("detected replay events require a latch episode")
        if self.early_clear and not self.detected:
            raise ValueError("early clear requires detection")

        is_dropout = self.fault_family == "dropout"
        if is_dropout != (self.realized_dropout is not None):
            raise ValueError("realized_dropout must be present exactly for dropout conditions")
        realized = self.realized_dropout is True
        if realized != (self.first_missing_step is not None):
            raise ValueError("first-missing step must be defined exactly for realized dropout")
        if realized != (self.first_missing_latency_s is not None):
            raise ValueError("first-missing seconds must be defined exactly for realized dropout")
        response_defined = realized and self.detected
        if response_defined != (self.detection_minus_first_missing_steps is not None):
            raise ValueError("dropout response step requires realized dropout and detection")
        if response_defined != (self.detection_minus_first_missing_s is not None):
            raise ValueError("dropout response seconds require realized dropout and detection")
        if self.detection_minus_first_missing_steps is not None:
            assert self.detection_latency_steps is not None
            assert self.first_missing_step is not None
            if (
                self.detection_minus_first_missing_steps
                != self.detection_latency_steps - self.first_missing_step
            ):
                raise ValueError("dropout response must equal detection minus first missing")
            bound = active_count - 1
            if not -bound <= self.detection_minus_first_missing_steps <= bound:
                raise ValueError("dropout response lies outside the signed dynamic bound")
        if (
            self.fault_family == "identity" or (is_dropout and self.realized_dropout is False)
        ) and (self.detected or self.outcome != "missed"):
            raise ValueError("identity and unrealized dropout must be forced missed")
        targetless = self.fault_target in {"none", "both"}
        if targetless and self.correctly_attributed:
            raise ValueError("targetless replay events cannot be correctly attributed")
        if targetless and self.detected and self.outcome != "ambiguous":
            raise ValueError("a targetless detected event must be ambiguous")
        if self.detected and not targetless:
            expected_outcome = (
                "ambiguous"
                if self.first_latch_label == "ambiguous"
                else "correct"
                if self.first_latch_label == f"{self.fault_target}-fault"
                else "wrong-sensor"
            )
            if self.outcome != expected_outcome:
                raise ValueError("single-target outcome must match the first latch label")

        if (
            self.active_healthy_steps
            + self.active_camera_fault_steps
            + self.active_lidar_fault_steps
            + self.active_ambiguous_steps
            != active_count
        ):
            raise ValueError("active state occupancy must sum to active_frame_count")
        if (
            self.active_camera_action_steps
            + self.active_lidar_action_steps
            + self.active_fixed_action_steps
            + self.active_undefined_action_steps
            != active_count
        ):
            raise ValueError("active action occupancy must sum to active_frame_count")
        final_state_count = {
            "healthy": self.active_healthy_steps,
            "camera-fault": self.active_camera_fault_steps,
            "lidar-fault": self.active_lidar_fault_steps,
            "ambiguous": self.active_ambiguous_steps,
        }[self.final_active_state]
        if final_state_count == 0:
            raise ValueError("final active state must occur in active occupancy")
        if self.recovery_eligible != (self.final_active_state != "healthy"):
            raise ValueError("recovery eligibility must follow final active state")
        if self.recovered and not self.recovery_eligible:
            raise ValueError("recovery requires a nonhealthy final active state")
        if self.recovered != (self.recovery_latency_steps is not None):
            raise ValueError("recovery step must be defined exactly when recovered")
        if self.recovered != (self.recovery_latency_s is not None):
            raise ValueError("recovery seconds must be defined exactly when recovered")
        if self.latch_episode_count > active_count:
            raise ValueError("latch episode count exceeds active-frame bound")
        false_alert_bound = (
            self.schedule.frame_count - 2 if self.fault_family == "identity" else active_count
        )
        if self.false_alert_episode_count > false_alert_bound:
            raise ValueError("false-alert count exceeds its dynamic bound")
        false_alert_applicable = self.fault_family == "identity" or (
            is_dropout and self.realized_dropout is False
        )
        if not false_alert_applicable and self.false_alert_episode_count != 0:
            raise ValueError("non-control fault rows must record zero false alerts")
        return self


class ReplayHealthResultV1(ContractModel):
    """One replay sequence/method/window localization sufficient-statistic row."""

    schema_id: Literal["ffb.replay-health-result/v1"] = Field(alias="schema")
    replay_experiment_identity_sha256: Digest
    sequence_id: Identifier
    condition_id: Identifier
    condition_selector: ReplayConditionSelector
    method: HealthMethod
    window: HealthWindow
    loss_sum_m2: Annotated[FiniteFloat, Field(ge=0.0)]
    valid_object_frame_count: Annotated[int, Field(ge=0)]
    eligible_object_frame_count: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def require_support_consistency(self) -> Self:
        if self.sequence_id not in _REPLAY_SEQUENCE_IDS:
            raise ValueError("replay result lies outside the frozen scene population")
        if self.condition_id not in M5_HEALTH_EXPERIMENT_IDS:
            raise ValueError("replay result condition is not in the frozen M5-B matrix")
        if not self.condition_selector.startswith(f"{self.condition_id}:"):
            raise ValueError("condition selector must bind the base condition_id")
        if self.valid_object_frame_count > self.eligible_object_frame_count:
            raise ValueError("valid replay support cannot exceed eligible replay support")
        if self.valid_object_frame_count == 0 and self.loss_sum_m2 != 0.0:
            raise ValueError("zero valid replay support requires zero accumulated loss")
        return self
