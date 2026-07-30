"""Replay-only variable-length health scoring without changing M4 classes."""

from __future__ import annotations

import math
from dataclasses import dataclass

from fusion_fault_bench.contracts.replay_health_v1 import (
    ReplayFaultFamily,
    ReplayFaultTarget,
    ReplayHealthScheduleV1,
    ReplayHealthSequenceEventV1,
)
from fusion_fault_bench.health import (
    ExecutedAction,
    HealthCalibration,
    HealthFrameEvidence,
    HealthFrameInput,
    HealthLabel,
    HealthLatchState,
    HealthMethodId,
    HealthScorer,
    HealthThresholds,
    NumericChannelEvidence,
    NumericScoreStatus,
    ObjectHealthInput,
    RawEvidenceStatus,
    RawHealthDecision,
    advance_latch,
    choose_executed_action,
    decide_raw,
)

_EMPTY_OBJECT_ID = "__ffb_replay_empty_frame__"


def replay_health_schedule(frame_count: int) -> ReplayHealthScheduleV1:
    """Construct the one frozen variable-length schedule from ``frame_count``."""

    if isinstance(frame_count, bool):
        raise TypeError("frame_count must be an integer")
    a = frame_count // 4
    b = (3 * frame_count) // 4
    return ReplayHealthScheduleV1(
        schema="ffb.replay-health-schedule/v1",
        frame_count=frame_count,
        predictor_initialization_frames=(0, 2),
        clean_prefix_frames=(0, a),
        score_frames=(2, frame_count),
        fault_active_frames=(a, b),
        recovery_frames=(b, frame_count),
    )


@dataclass(frozen=True, slots=True)
class ReplayHealthFrameInput:
    """Leakage-bounded replay input that permits an exact empty object tuple."""

    reference_time_s: float
    camera_available: bool
    lidar_available: bool
    objects: tuple[ObjectHealthInput, ...]

    def __post_init__(self) -> None:
        reference_time = float(self.reference_time_s)
        if not math.isfinite(reference_time):
            raise ValueError("reference_time_s must be finite")
        object.__setattr__(self, "reference_time_s", reference_time)
        identifiers = tuple(item.object_id for item in self.objects)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("object_id values must be unique within a replay frame")
        if identifiers != tuple(sorted(identifiers, key=lambda value: value.encode("utf-8"))):
            raise ValueError("replay objects must use frozen UTF-8 order")
        for item in self.objects:
            if not self.camera_available and item.camera is not None:
                raise ValueError("camera measurement cannot exist when camera is unavailable")
            if not self.lidar_available and item.lidar is not None:
                raise ValueError("lidar measurement cannot exist when lidar is unavailable")


@dataclass(frozen=True, slots=True)
class ReplayNumericChannelEvidence:
    """One numeric channel with zero-object support represented explicitly."""

    status: NumericScoreStatus
    mature_object_count: int
    current_object_count: int
    mature_fraction: float
    mean_nis: float | None
    maximum_nis: float | None
    score: float | None

    def __post_init__(self) -> None:
        if self.status not in {"defined", "insufficient-support"}:
            raise ValueError("unknown replay numeric status")
        if not 0 <= self.mature_object_count <= self.current_object_count:
            raise ValueError("mature support must lie within current support")
        expected_fraction = (
            0.0
            if self.current_object_count == 0
            else self.mature_object_count / self.current_object_count
        )
        if not math.isclose(
            self.mature_fraction,
            expected_fraction,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("mature_fraction does not match replay support")
        statistics = (self.mean_nis, self.maximum_nis, self.score)
        if self.status == "defined":
            if self.mature_object_count < 2 or any(value is None for value in statistics):
                raise ValueError("defined replay evidence requires mature numeric statistics")
        elif self.mature_object_count >= 2 or any(value is not None for value in statistics):
            raise ValueError("insufficient replay evidence must have null statistics")


@dataclass(frozen=True, slots=True)
class ReplayHealthFrameEvidence:
    """Observable replay evidence formed before current measurement updates."""

    reference_time_s: float
    camera_available: bool
    lidar_available: bool
    camera_timestamp_suspicious: bool
    lidar_timestamp_suspicious: bool
    camera_missing_fraction_last_four: float
    lidar_missing_fraction_last_four: float
    camera_self: ReplayNumericChannelEvidence
    lidar_self: ReplayNumericChannelEvidence
    camera_from_lidar_cross: ReplayNumericChannelEvidence
    lidar_from_camera_cross: ReplayNumericChannelEvidence

    def __post_init__(self) -> None:
        reference_time = float(self.reference_time_s)
        if not math.isfinite(reference_time):
            raise ValueError("reference_time_s must be finite")
        object.__setattr__(self, "reference_time_s", reference_time)
        for field_name in (
            "camera_missing_fraction_last_four",
            "lidar_missing_fraction_last_four",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must lie in [0, 1]")
            object.__setattr__(self, field_name, value)
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
            raise ValueError("replay numeric channels must share current support")
        if counts == {0} and (self.camera_timestamp_suspicious or self.lidar_timestamp_suspicious):
            raise ValueError("empty replay frames cannot contain timestamp evidence")

    @property
    def object_count(self) -> int:
        """Return common current object support."""

        return self.camera_self.current_object_count


def _from_m4_numeric(
    value: NumericChannelEvidence,
    *,
    empty: bool,
) -> ReplayNumericChannelEvidence:
    if empty:
        return ReplayNumericChannelEvidence(
            status="insufficient-support",
            mature_object_count=0,
            current_object_count=0,
            mature_fraction=0.0,
            mean_nis=None,
            maximum_nis=None,
            score=None,
        )
    return ReplayNumericChannelEvidence(
        status=value.status,
        mature_object_count=value.mature_object_count,
        current_object_count=value.current_object_count,
        mature_fraction=value.mature_fraction,
        mean_nis=value.mean_nis,
        maximum_nis=value.maximum_nis,
        score=value.score,
    )


def _to_m4_numeric(value: ReplayNumericChannelEvidence) -> NumericChannelEvidence:
    if value.current_object_count == 0:
        return NumericChannelEvidence(
            status="insufficient-support",
            mature_object_count=0,
            current_object_count=1,
            mature_fraction=0.0,
            mean_nis=None,
            maximum_nis=None,
            score=None,
        )
    return NumericChannelEvidence(
        status=value.status,
        mature_object_count=value.mature_object_count,
        current_object_count=value.current_object_count,
        mature_fraction=value.mature_fraction,
        mean_nis=value.mean_nis,
        maximum_nis=value.maximum_nis,
        score=value.score,
    )


def _to_m4_evidence(value: ReplayHealthFrameEvidence) -> HealthFrameEvidence:
    return HealthFrameEvidence(
        reference_time_s=value.reference_time_s,
        camera_available=value.camera_available,
        lidar_available=value.lidar_available,
        camera_timestamp_suspicious=value.camera_timestamp_suspicious,
        lidar_timestamp_suspicious=value.lidar_timestamp_suspicious,
        camera_missing_fraction_last_four=value.camera_missing_fraction_last_four,
        lidar_missing_fraction_last_four=value.lidar_missing_fraction_last_four,
        camera_self=_to_m4_numeric(value.camera_self),
        lidar_self=_to_m4_numeric(value.lidar_self),
        camera_from_lidar_cross=_to_m4_numeric(value.camera_from_lidar_cross),
        lidar_from_camera_cross=_to_m4_numeric(value.lidar_from_camera_cross),
    )


class ReplayHealthScorer:
    """M4-compatible scorer extended only at the zero-object frame boundary."""

    def __init__(self, calibration: HealthCalibration) -> None:
        self._m4 = HealthScorer(calibration)

    def process_frame(self, frame: ReplayHealthFrameInput) -> ReplayHealthFrameEvidence:
        """Score before update; an internal sentinel advances empty-frame clocks only."""

        empty = not frame.objects
        m4_objects = (
            (
                ObjectHealthInput(
                    object_id=_EMPTY_OBJECT_ID,
                    camera=None,
                    lidar=None,
                ),
            )
            if empty
            else frame.objects
        )
        evidence = self._m4.process_frame(
            HealthFrameInput(
                reference_time_s=frame.reference_time_s,
                camera_available=frame.camera_available,
                lidar_available=frame.lidar_available,
                objects=m4_objects,
            )
        )
        return ReplayHealthFrameEvidence(
            reference_time_s=evidence.reference_time_s,
            camera_available=evidence.camera_available,
            lidar_available=evidence.lidar_available,
            camera_timestamp_suspicious=evidence.camera_timestamp_suspicious,
            lidar_timestamp_suspicious=evidence.lidar_timestamp_suspicious,
            camera_missing_fraction_last_four=evidence.camera_missing_fraction_last_four,
            lidar_missing_fraction_last_four=evidence.lidar_missing_fraction_last_four,
            camera_self=_from_m4_numeric(evidence.camera_self, empty=empty),
            lidar_self=_from_m4_numeric(evidence.lidar_self, empty=empty),
            camera_from_lidar_cross=_from_m4_numeric(
                evidence.camera_from_lidar_cross,
                empty=empty,
            ),
            lidar_from_camera_cross=_from_m4_numeric(
                evidence.lidar_from_camera_cross,
                empty=empty,
            ),
        )


def decide_replay_raw(
    *,
    method: HealthMethodId,
    evidence: ReplayHealthFrameEvidence,
    thresholds: HealthThresholds,
) -> RawHealthDecision:
    """Apply the frozen M4 decision priority to replay evidence."""

    return decide_raw(
        method=method,
        evidence=_to_m4_evidence(evidence),
        thresholds=thresholds,
    )


@dataclass(frozen=True, slots=True)
class ReplayHealthPolicyOutput:
    """One replay raw decision, post-transition latch, and current action."""

    evidence: ReplayHealthFrameEvidence
    raw_decision: RawHealthDecision
    latched_state: HealthLatchState
    executed_action: ExecutedAction


class ReplayHealthPolicy:
    """Stateful replay wrapper around the unchanged M4 latch and action rules."""

    def __init__(
        self,
        *,
        method: HealthMethodId,
        thresholds: HealthThresholds,
        abstain_on_ambiguous: bool = False,
    ) -> None:
        if method not in {
            "self-nis-gate",
            "cross-nis-gate",
            "direct-telemetry-gate",
            "combined-health-gate",
        }:
            raise ValueError("unknown health method")
        if abstain_on_ambiguous and method != "combined-health-gate":
            raise ValueError("ambiguous abstention is defined only for the combined gate")
        self._method: HealthMethodId = method
        self._thresholds = thresholds
        self._abstain_on_ambiguous = abstain_on_ambiguous
        self._state = HealthLatchState()

    @property
    def state(self) -> HealthLatchState:
        """Return current immutable latch state."""

        return self._state

    def apply(self, evidence: ReplayHealthFrameEvidence) -> ReplayHealthPolicyOutput:
        """Apply the current decision before choosing the current executed action."""

        decision = decide_replay_raw(
            method=self._method,
            evidence=evidence,
            thresholds=self._thresholds,
        )
        state = advance_latch(self._state, decision)
        self._state = state
        return ReplayHealthPolicyOutput(
            evidence=evidence,
            raw_decision=decision,
            latched_state=state,
            executed_action=choose_executed_action(
                camera_available=evidence.camera_available,
                lidar_available=evidence.lidar_available,
                latched_label=state.label,
                abstain_on_ambiguous=self._abstain_on_ambiguous,
            ),
        )


@dataclass(frozen=True, slots=True)
class ReplayHealthPolicyTrace:
    """Aligned policy streams for every replay observation step."""

    policy: HealthMethodId
    reference_times_s: tuple[float, ...]
    camera_available: tuple[bool, ...]
    lidar_available: tuple[bool, ...]
    raw_labels: tuple[HealthLabel, ...]
    evidence_statuses: tuple[RawEvidenceStatus, ...]
    latched_labels: tuple[HealthLabel, ...]
    actions: tuple[ExecutedAction, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.reference_times_s),
            len(self.camera_available),
            len(self.lidar_available),
            len(self.raw_labels),
            len(self.evidence_statuses),
            len(self.latched_labels),
            len(self.actions),
        }
        if len(lengths) != 1 or not self.reference_times_s:
            raise ValueError("replay policy trace streams must be nonempty and aligned")
        if any(not math.isfinite(float(value)) for value in self.reference_times_s):
            raise ValueError("replay reference times must be finite")
        if any(
            later <= earlier
            for earlier, later in zip(
                self.reference_times_s,
                self.reference_times_s[1:],
                strict=False,
            )
        ):
            raise ValueError("replay reference times must be strictly increasing")


def evaluate_replay_policy_trace(
    evidence: tuple[ReplayHealthFrameEvidence, ...],
    *,
    method: HealthMethodId,
    thresholds: HealthThresholds,
    abstain_on_ambiguous: bool = False,
) -> ReplayHealthPolicyTrace:
    """Evaluate one policy over already-computed immutable replay evidence."""

    if not evidence:
        raise ValueError("replay evidence trace must be nonempty")
    policy = ReplayHealthPolicy(
        method=method,
        thresholds=thresholds,
        abstain_on_ambiguous=abstain_on_ambiguous,
    )
    outputs = tuple(policy.apply(frame) for frame in evidence)
    return ReplayHealthPolicyTrace(
        policy=method,
        reference_times_s=tuple(frame.reference_time_s for frame in evidence),
        camera_available=tuple(frame.camera_available for frame in evidence),
        lidar_available=tuple(frame.lidar_available for frame in evidence),
        raw_labels=tuple(output.raw_decision.label for output in outputs),
        evidence_statuses=tuple(output.raw_decision.evidence_status for output in outputs),
        latched_labels=tuple(output.latched_state.label for output in outputs),
        actions=tuple(output.executed_action for output in outputs),
    )


def _healthy_to_nonhealthy(
    labels: tuple[HealthLabel, ...],
    *,
    start: int,
    end: int,
) -> tuple[tuple[int, HealthLabel], ...]:
    previous: HealthLabel = "healthy" if start == 0 else labels[start - 1]
    transitions: list[tuple[int, HealthLabel]] = []
    for frame_index in range(start, end):
        current = labels[frame_index]
        if previous == "healthy" and current != "healthy":
            transitions.append((frame_index, current))
        previous = current
    return tuple(transitions)


def _nonhealthy_to_healthy(
    labels: tuple[HealthLabel, ...],
    *,
    start: int,
    end: int,
) -> tuple[int, ...]:
    previous: HealthLabel = "healthy" if start == 0 else labels[start - 1]
    transitions: list[int] = []
    for frame_index in range(start, end):
        current = labels[frame_index]
        if previous != "healthy" and current == "healthy":
            transitions.append(frame_index)
        previous = current
    return tuple(transitions)


def replay_sequence_event_record(
    trace: ReplayHealthPolicyTrace,
    *,
    schedule: ReplayHealthScheduleV1,
    replay_experiment_identity_sha256: str,
    sequence_id: str,
    condition_id: str,
    condition_selector: str,
    fault_family: ReplayFaultFamily,
    fault_target: ReplayFaultTarget,
) -> ReplayHealthSequenceEventV1:
    """Reduce a trace using the frozen dynamic event and censoring semantics."""

    if len(trace.latched_labels) != schedule.frame_count:
        raise ValueError("replay trace and dynamic schedule must align")
    if fault_family == "identity" and fault_target != "none":
        raise ValueError("identity replay events require target none")
    if fault_family == "common-mode-position-bias" and fault_target != "both":
        raise ValueError("common-mode replay events require target both")
    if fault_family not in {"identity", "common-mode-position-bias"} and fault_target not in {
        "camera",
        "lidar",
    }:
        raise ValueError("single-target replay faults require camera or lidar target")

    event_start, event_end = schedule.fault_active_frames
    recovery_start, recovery_end = schedule.recovery_frames
    transitions = _healthy_to_nonhealthy(
        trace.latched_labels,
        start=event_start,
        end=event_end,
    )
    first_transition = transitions[0] if transitions else None

    realized_dropout: bool | None = None
    first_missing_frame: int | None = None
    if fault_family == "dropout":
        availability = trace.camera_available if fault_target == "camera" else trace.lidar_available
        missing_frames = tuple(
            frame_index
            for frame_index in range(event_start, event_end)
            if not availability[frame_index]
        )
        realized_dropout = bool(missing_frames)
        first_missing_frame = missing_frames[0] if missing_frames else None

    force_missed = fault_family == "identity" or (
        fault_family == "dropout" and realized_dropout is False
    )
    detected = first_transition is not None and not force_missed
    detection_frame = first_transition[0] if detected and first_transition is not None else None
    first_latch_label = first_transition[1] if detected and first_transition is not None else None
    targetless = fault_target in {"none", "both"}
    if not detected:
        outcome = "missed"
    elif targetless or first_latch_label == "ambiguous":
        outcome = "ambiguous"
    elif first_latch_label == f"{fault_target}-fault":
        outcome = "correct"
    else:
        outcome = "wrong-sensor"

    correct_transitions = (
        tuple(frame_index for frame_index, label in transitions if label == f"{fault_target}-fault")
        if not targetless and not force_missed
        else ()
    )
    correctly_attributed = bool(correct_transitions)
    active_clear_frames = _nonhealthy_to_healthy(
        trace.latched_labels,
        start=event_start,
        end=event_end,
    )
    early_clear = detected and any(
        detection_frame is not None and frame_index > detection_frame
        for frame_index in active_clear_frames
    )
    final_active_state = trace.latched_labels[event_end - 1]
    recovery_eligible = final_active_state != "healthy"
    recovery_transitions = _nonhealthy_to_healthy(
        trace.latched_labels,
        start=recovery_start,
        end=recovery_end,
    )
    recovered = recovery_eligible and bool(recovery_transitions)
    recovery_frame = recovery_transitions[0] if recovered else None

    score_start, score_end = schedule.score_frames
    score_transitions = _healthy_to_nonhealthy(
        trace.latched_labels,
        start=score_start,
        end=score_end,
    )
    if fault_family == "identity":
        false_alert_count = len(score_transitions)
    elif fault_family == "dropout" and realized_dropout is False:
        false_alert_count = len(transitions)
    else:
        false_alert_count = 0

    active_labels = trace.latched_labels[event_start:event_end]
    active_actions = trace.actions[event_start:event_end]
    attribution_frame = correct_transitions[0] if correct_transitions else None
    times = trace.reference_times_s

    def elapsed(frame_index: int | None, origin: int) -> float | None:
        return None if frame_index is None else float(times[frame_index] - times[origin])

    detection_step = None if detection_frame is None else detection_frame - event_start
    attribution_step = None if attribution_frame is None else attribution_frame - event_start
    first_missing_step = None if first_missing_frame is None else first_missing_frame - event_start
    recovery_step = None if recovery_frame is None else recovery_frame - recovery_start
    response_step = (
        None
        if detection_frame is None or first_missing_frame is None
        else detection_frame - first_missing_frame
    )
    response_seconds = (
        None
        if detection_frame is None or first_missing_frame is None
        else float(times[detection_frame] - times[first_missing_frame])
    )

    return ReplayHealthSequenceEventV1(
        schema="ffb.replay-health-sequence-event/v1",
        replay_experiment_identity_sha256=replay_experiment_identity_sha256,
        sequence_id=sequence_id,
        condition_id=condition_id,
        condition_selector=condition_selector,
        policy=trace.policy,
        fault_family=fault_family,
        fault_target=fault_target,
        schedule=schedule,
        detected=detected,
        detection_latency_steps=detection_step,
        detection_latency_s=elapsed(detection_frame, event_start),
        detection_censor_bound_steps=schedule.active_frame_count,
        first_latch_label=first_latch_label,
        outcome=outcome,
        correctly_attributed=correctly_attributed,
        attribution_latency_steps=attribution_step,
        attribution_latency_s=elapsed(attribution_frame, event_start),
        attribution_censor_bound_steps=schedule.active_frame_count,
        realized_dropout=realized_dropout,
        first_missing_step=first_missing_step,
        first_missing_latency_s=elapsed(first_missing_frame, event_start),
        detection_minus_first_missing_steps=response_step,
        detection_minus_first_missing_s=response_seconds,
        latch_episode_count=len(transitions),
        false_alert_episode_count=false_alert_count,
        early_clear=early_clear,
        final_active_state=final_active_state,
        active_healthy_steps=active_labels.count("healthy"),
        active_camera_fault_steps=active_labels.count("camera-fault"),
        active_lidar_fault_steps=active_labels.count("lidar-fault"),
        active_ambiguous_steps=active_labels.count("ambiguous"),
        active_camera_action_steps=active_actions.count("camera-only"),
        active_lidar_action_steps=active_actions.count("lidar-only"),
        active_fixed_action_steps=active_actions.count("fixed-fusion"),
        active_undefined_action_steps=active_actions.count("undefined"),
        recovery_eligible=recovery_eligible,
        recovered=recovered,
        recovery_latency_steps=recovery_step,
        recovery_latency_s=elapsed(recovery_frame, recovery_start),
        recovery_censor_bound_steps=schedule.recovery_frame_count,
    )
