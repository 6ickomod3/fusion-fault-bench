"""Per-sequence M4 actions, losses, event outcomes, and support accounting."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.contracts.health_result_v1 import (
    HealthMethod,
    HealthPolicyMethod,
    HealthSequenceContrastV1,
    HealthSequenceEventV1,
    HealthSequenceLossV1,
    HealthWindow,
)
from fusion_fault_bench.contracts.health_result_v1 import (
    HealthPolicy as HealthPolicyId,
)
from fusion_fault_bench.experiments.health import HealthObservationSequence
from fusion_fault_bench.geometry._arrays import immutable_float64_copy
from fusion_fault_bench.health import (
    ExecutedAction,
    HealthFrameEvidence,
    HealthLabel,
    HealthMethodId,
    HealthPolicy,
    HealthThresholds,
    RawEvidenceStatus,
)
from fusion_fault_bench.scenarios.health import HealthFaultSpec, health_event_schedule

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]

_HEALTH_POLICY_METHODS: tuple[HealthMethodId, ...] = (
    "self-nis-gate",
    "cross-nis-gate",
    "direct-telemetry-gate",
    "combined-health-gate",
)
_CONTRAST_POLICY_METHODS: tuple[HealthPolicyMethod, ...] = (
    "self-nis-gate",
    "cross-nis-gate",
    "direct-telemetry-gate",
    "combined-health-gate",
    "combined-health-gate-abstain",
)
_ALL_METHODS: tuple[HealthMethod, ...] = (
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
_SUPPORT_MASK_DOMAIN = b"fusion-fault-bench/health-support-mask/v1\x00"


def _immutable_bool(value: npt.ArrayLike) -> BoolArray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.bool_))
    return np.frombuffer(array.tobytes(order="C"), dtype=np.bool_).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class HealthPolicyTrace:
    """One policy's distinct raw, latched, and executed frame streams."""

    policy: HealthPolicyId
    raw_labels: tuple[HealthLabel, ...]
    evidence_statuses: tuple[RawEvidenceStatus, ...]
    latched_labels: tuple[HealthLabel, ...]
    actions: tuple[ExecutedAction, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.raw_labels),
            len(self.evidence_statuses),
            len(self.latched_labels),
            len(self.actions),
        }
        if len(lengths) != 1 or not self.raw_labels:
            raise ValueError("health policy trace streams must be nonempty and aligned")


@dataclass(frozen=True, slots=True)
class MethodLossTrace:
    """Object-frame squared losses and the exact method-specific support."""

    method: HealthMethod
    loss_m2: FloatArray
    valid_mask: BoolArray

    def __post_init__(self) -> None:
        losses = np.asarray(self.loss_m2, dtype=np.float64)
        valid = np.asarray(self.valid_mask, dtype=np.bool_)
        if losses.ndim != 2 or valid.shape != losses.shape or losses.size == 0:
            raise ValueError("method loss and support grids must be aligned and nonempty")
        if not np.all(np.isfinite(losses)) or np.any(losses < 0.0):
            raise ValueError("method loss grid must be finite and nonnegative")
        if np.any(losses[~valid] != 0.0):
            raise ValueError("loss must be zero outside method support")
        object.__setattr__(self, "loss_m2", immutable_float64_copy(losses))
        object.__setattr__(self, "valid_mask", _immutable_bool(valid))


@dataclass(frozen=True, slots=True)
class HealthSequenceEvaluation:
    """All method losses and deployable policy traces for one sequence/case."""

    sequence_id: str
    condition_id: str
    losses: tuple[MethodLossTrace, ...]
    policy_traces: tuple[HealthPolicyTrace, ...]

    def __post_init__(self) -> None:
        methods = tuple(item.method for item in self.losses)
        if len(set(methods)) != len(methods):
            raise ValueError("sequence evaluation method loss IDs must be unique")
        policies = tuple(item.policy for item in self.policy_traces)
        if policies != _HEALTH_POLICY_METHODS:
            raise ValueError("sequence evaluation policy traces use the wrong order")

    def loss(self, method: HealthMethod) -> MethodLossTrace:
        """Return one method loss trace by its exact ID."""

        for trace in self.losses:
            if trace.method == method:
                return trace
        raise KeyError(method)

    def policy_trace(self, policy: HealthPolicyId) -> HealthPolicyTrace:
        """Return one deployable policy trace by its exact ID."""

        for trace in self.policy_traces:
            if trace.policy == policy:
                return trace
        raise KeyError(policy)


def fixed_fusion_values(
    observations: HealthObservationSequence,
) -> tuple[FloatArray, BoolArray]:
    """Fuse current camera/LiDAR estimates using their reported covariance."""

    frame_count = observations.frame_count
    object_count = observations.object_count
    values = np.zeros((frame_count, object_count, 2), dtype=np.float64)
    available = np.asarray(
        observations.camera_available & observations.lidar_available,
        dtype=np.bool_,
    )
    for frame_index in np.flatnonzero(available):
        for object_index in range(object_count):
            camera_covariance = observations.camera_reported_covariance_xy_m2[
                frame_index,
                object_index,
            ]
            lidar_covariance = observations.lidar_reported_covariance_xy_m2[
                frame_index,
                object_index,
            ]
            camera_information = np.linalg.inv(camera_covariance)
            lidar_information = np.linalg.inv(lidar_covariance)
            fused_covariance = np.linalg.inv(camera_information + lidar_information)
            values[frame_index, object_index] = fused_covariance @ (
                camera_information @ observations.camera_value_xy_m[frame_index, object_index]
                + lidar_information @ observations.lidar_value_xy_m[frame_index, object_index]
            )
    return immutable_float64_copy(values), _immutable_bool(available)


def evaluate_policy_trace(
    evidence: tuple[HealthFrameEvidence, ...],
    *,
    method: HealthMethodId,
    thresholds: HealthThresholds,
    abstain_on_ambiguous: bool = False,
) -> HealthPolicyTrace:
    """Apply a policy recurrence to an immutable, already-computed feature trace."""

    policy = HealthPolicy(
        method=method,
        thresholds=thresholds,
        abstain_on_ambiguous=abstain_on_ambiguous,
    )
    raw_labels: list[HealthLabel] = []
    statuses: list[RawEvidenceStatus] = []
    latched: list[HealthLabel] = []
    actions: list[ExecutedAction] = []
    for frame in evidence:
        output = policy.apply(frame)
        raw_labels.append(output.raw_decision.label)
        statuses.append(output.raw_decision.evidence_status)
        latched.append(output.latched_state.label)
        actions.append(output.executed_action)
    return HealthPolicyTrace(
        policy=method,
        raw_labels=tuple(raw_labels),
        evidence_statuses=tuple(statuses),
        latched_labels=tuple(latched),
        actions=tuple(actions),
    )


def target_drop_actions(
    observations: HealthObservationSequence,
    *,
    fault: HealthFaultSpec,
) -> tuple[ExecutedAction, ...] | None:
    if fault.family == "common-mode-position-bias":
        return None
    start, end = health_event_schedule(fault.schedule).fault_active_frames
    fixed_action: ExecutedAction = "fixed-fusion"
    camera_action: ExecutedAction = "camera-only"
    lidar_action: ExecutedAction = "lidar-only"
    actions: list[ExecutedAction] = [fixed_action] * observations.frame_count
    if fault.target == "camera":
        for frame_index in range(start, end):
            actions[frame_index] = lidar_action
    elif fault.target == "lidar":
        for frame_index in range(start, end):
            actions[frame_index] = camera_action
    return tuple(actions)


def _action_defined(
    action: ExecutedAction,
    *,
    camera_available: bool,
    lidar_available: bool,
) -> bool:
    if action == "camera-only":
        return camera_available
    if action == "lidar-only":
        return lidar_available
    if action == "fixed-fusion":
        return camera_available and lidar_available
    return False


def _repeat_action(action: ExecutedAction, count: int) -> tuple[ExecutedAction, ...]:
    return tuple(action for _ in range(count))


def frame_oracle_actions(
    observations: HealthObservationSequence,
    *,
    fused: FloatArray,
    fused_available: BoolArray,
) -> tuple[ExecutedAction, ...]:
    actions: list[ExecutedAction] = []
    for frame_index in range(observations.frame_count):
        eligible = observations.eligibility_mask[frame_index]
        candidates: list[tuple[float, ExecutedAction]] = []
        if observations.camera_available[frame_index]:
            error = (
                observations.camera_value_xy_m[frame_index, eligible]
                - observations.truth_xy_m[frame_index, eligible]
            )
            candidates.append((float(np.square(error).sum(axis=1).mean()), "camera-only"))
        if observations.lidar_available[frame_index]:
            error = (
                observations.lidar_value_xy_m[frame_index, eligible]
                - observations.truth_xy_m[frame_index, eligible]
            )
            candidates.append((float(np.square(error).sum(axis=1).mean()), "lidar-only"))
        if fused_available[frame_index]:
            error = fused[frame_index, eligible] - observations.truth_xy_m[frame_index, eligible]
            candidates.append((float(np.square(error).sum(axis=1).mean()), "fixed-fusion"))
        if not candidates:
            actions.append("undefined")
        else:
            # Stable minimum retains the frozen camera, LiDAR, fixed tie order.
            actions.append(min(candidates, key=lambda item: item[0])[1])
    return tuple(actions)


def loss_trace_for_actions(
    observations: HealthObservationSequence,
    *,
    method: HealthMethod,
    actions: tuple[ExecutedAction, ...],
    fused: FloatArray,
) -> MethodLossTrace:
    if len(actions) != observations.frame_count:
        raise ValueError("method action trace does not match the observation frames")
    losses = np.zeros(
        (observations.frame_count, observations.object_count),
        dtype=np.float64,
    )
    valid = np.zeros_like(observations.eligibility_mask)
    for frame_index, action in enumerate(actions):
        if not _action_defined(
            action,
            camera_available=bool(observations.camera_available[frame_index]),
            lidar_available=bool(observations.lidar_available[frame_index]),
        ):
            continue
        if action == "camera-only":
            value = observations.camera_value_xy_m[frame_index]
        elif action == "lidar-only":
            value = observations.lidar_value_xy_m[frame_index]
        elif action == "fixed-fusion":
            value = fused[frame_index]
        else:
            raise AssertionError("a defined action cannot be undefined")
        eligible = observations.eligibility_mask[frame_index]
        error = value[eligible] - observations.truth_xy_m[frame_index, eligible]
        losses[frame_index, eligible] = np.square(error).sum(axis=1)
        valid[frame_index, eligible] = True
    return MethodLossTrace(method=method, loss_m2=losses, valid_mask=valid)


def evaluate_health_sequence(
    observations: HealthObservationSequence,
    *,
    condition_id: str,
    fault: HealthFaultSpec,
    evidence: tuple[HealthFrameEvidence, ...],
    thresholds: HealthThresholds,
) -> HealthSequenceEvaluation:
    """Evaluate every applicable method without recomputing observable features."""

    if len(evidence) != observations.frame_count:
        raise ValueError("feature trace and observations must align")
    fused, fused_available = fixed_fusion_values(observations)
    policy_traces = tuple(
        evaluate_policy_trace(
            evidence,
            method=method,
            thresholds=thresholds,
        )
        for method in _HEALTH_POLICY_METHODS
    )
    combined_abstain = evaluate_policy_trace(
        evidence,
        method="combined-health-gate",
        thresholds=thresholds,
        abstain_on_ambiguous=True,
    )
    combined = policy_traces[-1]
    if (
        combined.raw_labels != combined_abstain.raw_labels
        or combined.evidence_statuses != combined_abstain.evidence_statuses
        or combined.latched_labels != combined_abstain.latched_labels
    ):
        raise AssertionError("combined abstention changed the health-state trace")

    camera_action: ExecutedAction = "camera-only"
    lidar_action: ExecutedAction = "lidar-only"
    fixed_action: ExecutedAction = "fixed-fusion"
    action_by_method: list[tuple[HealthMethod, tuple[ExecutedAction, ...]]] = [
        ("camera-only", _repeat_action(camera_action, observations.frame_count)),
        ("lidar-only", _repeat_action(lidar_action, observations.frame_count)),
        ("fixed-fusion", _repeat_action(fixed_action, observations.frame_count)),
        *((cast(HealthMethod, trace.policy), trace.actions) for trace in policy_traces),
        ("combined-health-gate-abstain", combined_abstain.actions),
    ]
    target_drop = target_drop_actions(observations, fault=fault)
    if target_drop is not None:
        action_by_method.append(("fault-target-drop-policy", target_drop))
    if fault.family != "common-mode-position-bias":
        action_by_method.append(
            (
                "frame-action-performance-oracle",
                frame_oracle_actions(
                    observations,
                    fused=fused,
                    fused_available=fused_available,
                ),
            )
        )
    losses = tuple(
        loss_trace_for_actions(
            observations,
            method=method,
            actions=actions,
            fused=fused,
        )
        for method, actions in action_by_method
    )
    return HealthSequenceEvaluation(
        sequence_id=observations.sequence_id,
        condition_id=condition_id,
        losses=losses,
        policy_traces=policy_traces,
    )


def sequence_loss_records(
    evaluation: HealthSequenceEvaluation,
    *,
    observations: HealthObservationSequence,
    fault: HealthFaultSpec,
) -> tuple[HealthSequenceLossV1, ...]:
    """Reduce object/frame losses within each sequence before inference."""

    if evaluation.sequence_id != observations.sequence_id:
        raise ValueError("sequence evaluation and observations disagree")
    schedule = health_event_schedule(fault.schedule)
    windows: tuple[tuple[HealthWindow, tuple[int, int]], ...] = (
        ("score", schedule.score_frames),
        ("event", schedule.fault_active_frames),
        ("recovery", schedule.recovery_frames),
    )
    records: list[HealthSequenceLossV1] = []
    for trace in evaluation.losses:
        for window, (start, end) in windows:
            eligible = observations.eligibility_mask[start:end]
            valid = trace.valid_mask[start:end]
            if np.any(valid & ~eligible):
                raise AssertionError("method support escaped the frozen eligibility mask")
            records.append(
                HealthSequenceLossV1(
                    schema="ffb.health-sequence-loss/v1",
                    sequence_id=evaluation.sequence_id,
                    condition_id=evaluation.condition_id,
                    method=trace.method,
                    window=window,
                    loss_sum_m2=float(trace.loss_m2[start:end].sum(dtype=np.float64)),
                    valid_object_frame_count=int(np.count_nonzero(valid)),
                    eligible_object_frame_count=int(np.count_nonzero(eligible)),
                )
            )
    return tuple(records)


def paired_common_support_sequence_contrast(
    left: MethodLossTrace,
    right: MethodLossTrace,
    *,
    frame_window: tuple[int, int],
) -> tuple[float | None, int]:
    """Compute left-minus-right only on their exact paired object/frame support."""

    left_sum, right_sum, count = _common_support_loss_sums(
        left,
        right,
        frame_window=frame_window,
    )
    return (None if count == 0 else (left_sum - right_sum) / count), count


def _common_support_loss_sums(
    left: MethodLossTrace,
    right: MethodLossTrace,
    *,
    frame_window: tuple[int, int],
) -> tuple[float, float, int]:
    """Retain both loss sums on exact paired object/frame support."""

    if left.loss_m2.shape != right.loss_m2.shape:
        raise ValueError("common-support method traces must align")
    start, end = frame_window
    if not 0 <= start < end <= left.loss_m2.shape[0]:
        raise ValueError("common-support frame window is invalid")
    support = left.valid_mask[start:end] & right.valid_mask[start:end]
    count = int(np.count_nonzero(support))
    if count == 0:
        return 0.0, 0.0, 0
    return (
        math.fsum(float(value) for value in left.loss_m2[start:end][support]),
        math.fsum(float(value) for value in right.loss_m2[start:end][support]),
        count,
    )


def health_support_mask_sha256(
    mask: npt.ArrayLike,
    *,
    frame_window: tuple[int, int],
) -> str:
    """Commit exact windowed object/frame support without local identifiers."""

    array = np.asarray(mask)
    if array.ndim != 2 or array.dtype.kind != "b":
        raise ValueError("health support mask must be one two-dimensional boolean array")
    start, end = frame_window
    if not 0 <= start < end <= array.shape[0]:
        raise ValueError("health support frame window is invalid")
    windowed = np.ascontiguousarray(array[start:end], dtype=np.bool_)
    packed = np.packbits(windowed.reshape(-1), bitorder="big").tobytes()
    preimage = b"".join(
        (
            _SUPPORT_MASK_DOMAIN,
            start.to_bytes(4, "big"),
            end.to_bytes(4, "big"),
            windowed.shape[0].to_bytes(4, "big"),
            windowed.shape[1].to_bytes(4, "big"),
            packed,
        )
    )
    return hashlib.sha256(preimage).hexdigest()


def sequence_contrast_records(
    evaluation: HealthSequenceEvaluation,
    *,
    fault: HealthFaultSpec,
) -> tuple[HealthSequenceContrastV1, ...]:
    """Retain every sequence-level common-support statistic used by inference."""

    traces = {trace.method: trace for trace in evaluation.losses}
    required = {"fixed-fusion", *_CONTRAST_POLICY_METHODS}
    if not required.issubset(traces):
        raise ValueError("sequence evaluation is missing a contrast method")
    target_drop = traces.get("fault-target-drop-policy")
    frame_oracle = traces.get("frame-action-performance-oracle")
    if (target_drop is None) != (frame_oracle is None):
        raise ValueError("target-drop and frame-oracle applicability must align")

    schedule = health_event_schedule(fault.schedule)
    windows: tuple[tuple[HealthWindow, tuple[int, int]], ...] = (
        ("score", schedule.score_frames),
        ("event", schedule.fault_active_frames),
        ("recovery", schedule.recovery_frames),
    )
    fixed = traces["fixed-fusion"]
    records: list[HealthSequenceContrastV1] = []
    for policy_id in _CONTRAST_POLICY_METHODS:
        policy = traces[policy_id]
        for window, bounds in windows:
            fixed_support_sha256 = health_support_mask_sha256(
                fixed.valid_mask,
                frame_window=bounds,
            )
            policy_support_sha256 = health_support_mask_sha256(
                policy.valid_mask,
                frame_window=bounds,
            )
            fixed_sum, policy_fixed_sum, fixed_count = _common_support_loss_sums(
                fixed,
                policy,
                frame_window=bounds,
            )
            if target_drop is None:
                target_count = None
                policy_target_sum = None
                target_sum = None
            else:
                policy_target_sum, target_sum, target_count = _common_support_loss_sums(
                    policy,
                    target_drop,
                    frame_window=bounds,
                )
            if frame_oracle is None:
                oracle_count = None
                policy_oracle_sum = None
                oracle_sum = None
                oracle_support_sha256 = None
            else:
                oracle_support_sha256 = health_support_mask_sha256(
                    frame_oracle.valid_mask,
                    frame_window=bounds,
                )
                policy_oracle_sum, oracle_sum, oracle_count = _common_support_loss_sums(
                    policy,
                    frame_oracle,
                    frame_window=bounds,
                )
                identical_support = (
                    fixed_support_sha256 == policy_support_sha256 == oracle_support_sha256
                )
                if identical_support and fixed_count > 0:
                    if oracle_count != fixed_count:
                        raise AssertionError("identical support produced unequal counts")
                    policy_oracle_sum = policy_fixed_sum
            records.append(
                HealthSequenceContrastV1(
                    schema="ffb.health-sequence-contrast/v1",
                    sequence_id=evaluation.sequence_id,
                    condition_id=evaluation.condition_id,
                    policy=policy_id,
                    window=window,
                    fixed_support_sha256=fixed_support_sha256,
                    policy_support_sha256=policy_support_sha256,
                    fixed_policy_common_count=fixed_count,
                    fixed_on_common_loss_sum_m2=fixed_sum,
                    policy_on_fixed_common_loss_sum_m2=policy_fixed_sum,
                    target_drop_applicable=target_drop is not None,
                    policy_target_drop_common_count=target_count,
                    policy_on_target_common_loss_sum_m2=policy_target_sum,
                    target_drop_on_common_loss_sum_m2=target_sum,
                    frame_oracle_applicable=frame_oracle is not None,
                    policy_frame_oracle_common_count=oracle_count,
                    policy_on_oracle_common_loss_sum_m2=policy_oracle_sum,
                    frame_oracle_on_common_loss_sum_m2=oracle_sum,
                    frame_oracle_support_sha256=oracle_support_sha256,
                )
            )
    return tuple(records)


def _healthy_to_nonhealthy_transitions(
    labels: tuple[HealthLabel, ...],
    *,
    start: int,
    end: int,
) -> tuple[tuple[int, HealthLabel], ...]:
    transitions: list[tuple[int, HealthLabel]] = []
    previous: HealthLabel = "healthy" if start == 0 else labels[start - 1]
    for frame_index in range(start, end):
        current = labels[frame_index]
        if previous == "healthy" and current != "healthy":
            transitions.append((frame_index, current))
        previous = current
    return tuple(transitions)


def _nonhealthy_to_healthy_transitions(
    labels: tuple[HealthLabel, ...],
    *,
    start: int,
    end: int,
) -> tuple[int, ...]:
    transitions: list[int] = []
    previous: HealthLabel = "healthy" if start == 0 else labels[start - 1]
    for frame_index in range(start, end):
        current = labels[frame_index]
        if previous != "healthy" and current == "healthy":
            transitions.append(frame_index)
        previous = current
    return tuple(transitions)


def _count_labels(
    values: tuple[HealthLabel, ...],
) -> tuple[int, int, int, int]:
    return (
        values.count("healthy"),
        values.count("camera-fault"),
        values.count("lidar-fault"),
        values.count("ambiguous"),
    )


def _count_actions(
    values: tuple[ExecutedAction, ...],
) -> tuple[int, int, int, int]:
    return (
        values.count("camera-only"),
        values.count("lidar-only"),
        values.count("fixed-fusion"),
        values.count("undefined"),
    )


def sequence_event_record(
    trace: HealthPolicyTrace,
    *,
    observations: HealthObservationSequence,
    condition_id: str,
    fault: HealthFaultSpec,
) -> HealthSequenceEventV1:
    """Reduce one policy trace with exact event and censoring semantics."""

    if len(trace.latched_labels) != observations.frame_count:
        raise ValueError("event trace and observations must align")
    schedule = health_event_schedule(fault.schedule)
    event_start, event_end = schedule.fault_active_frames
    recovery_start, recovery_end = schedule.recovery_frames
    if event_end - event_start != 24:
        raise AssertionError("M4 active event must contain exactly 24 frames")
    event_transitions = _healthy_to_nonhealthy_transitions(
        trace.latched_labels,
        start=event_start,
        end=event_end,
    )
    first_transition = event_transitions[0] if event_transitions else None

    realized_dropout: bool | None = None
    first_missing_frame: int | None = None
    if fault.family == "dropout":
        target_availability = (
            observations.camera_available
            if fault.target == "camera"
            else observations.lidar_available
        )
        missing = np.flatnonzero(~target_availability[event_start:event_end])
        realized_dropout = bool(missing.size)
        if realized_dropout:
            first_missing_frame = event_start + int(missing[0])

    targetless = fault.target in {"none", "both"}
    force_missed = fault.family == "identity" or (
        fault.family == "dropout" and realized_dropout is False
    )
    detected = first_transition is not None and not force_missed
    detection_frame = first_transition[0] if detected and first_transition is not None else None
    first_latch_label = first_transition[1] if detected and first_transition is not None else None
    if not detected:
        outcome = "missed"
    elif targetless or first_latch_label == "ambiguous":
        outcome = "ambiguous"
    elif first_latch_label == f"{fault.target}-fault":
        outcome = "correct"
    else:
        outcome = "wrong-sensor"

    correct_transitions = (
        tuple(
            frame_index
            for frame_index, label in event_transitions
            if label == f"{fault.target}-fault"
        )
        if not targetless and not force_missed
        else ()
    )
    correctly_attributed = bool(correct_transitions)
    recovery_transitions = _nonhealthy_to_healthy_transitions(
        trace.latched_labels,
        start=recovery_start,
        end=recovery_end,
    )
    final_active_state = trace.latched_labels[event_end - 1]
    recovery_eligible = final_active_state != "healthy"
    recovered = recovery_eligible and bool(recovery_transitions)
    active_clear_frames = _nonhealthy_to_healthy_transitions(
        trace.latched_labels,
        start=event_start,
        end=event_end,
    )
    early_clear = detected and any(
        detection_frame is not None and frame_index > detection_frame
        for frame_index in active_clear_frames
    )

    score_start, score_end = schedule.score_frames
    score_transitions = _healthy_to_nonhealthy_transitions(
        trace.latched_labels,
        start=score_start,
        end=score_end,
    )
    if fault.family in {"identity", "clean-predictor-mismatch"}:
        false_alert_count = len(score_transitions)
    elif fault.family == "dropout" and realized_dropout is False:
        false_alert_count = len(event_transitions)
    else:
        false_alert_count = 0

    active_states = trace.latched_labels[event_start:event_end]
    active_actions = trace.actions[event_start:event_end]
    healthy_frames, camera_fault_frames, lidar_fault_frames, ambiguous_frames = _count_labels(
        active_states
    )
    camera_actions, lidar_actions, fixed_actions, undefined_actions = _count_actions(active_actions)
    return HealthSequenceEventV1(
        schema="ffb.health-sequence-event/v1",
        sequence_id=observations.sequence_id,
        condition_id=condition_id,
        policy=trace.policy,
        detected=detected,
        detection_latency_frames=(
            None if detection_frame is None else detection_frame - event_start
        ),
        first_latch_label=first_latch_label,
        outcome=outcome,
        correctly_attributed=correctly_attributed,
        attribution_latency_frames=(
            correct_transitions[0] - event_start if correct_transitions else None
        ),
        realized_dropout=realized_dropout,
        first_missing_frame_minus_event_start=(
            first_missing_frame - event_start if first_missing_frame is not None else None
        ),
        detection_minus_first_missing_frames=(
            detection_frame - first_missing_frame
            if detection_frame is not None and first_missing_frame is not None
            else None
        ),
        latch_episode_count=len(event_transitions),
        false_alert_episode_count=false_alert_count,
        early_clear=early_clear,
        final_active_state=final_active_state,
        active_frame_count=24,
        active_healthy_frames=healthy_frames,
        active_camera_fault_frames=camera_fault_frames,
        active_lidar_fault_frames=lidar_fault_frames,
        active_ambiguous_frames=ambiguous_frames,
        active_camera_action_frames=camera_actions,
        active_lidar_action_frames=lidar_actions,
        active_fixed_action_frames=fixed_actions,
        active_undefined_action_frames=undefined_actions,
        recovery_eligible=recovery_eligible,
        recovered=recovered,
        recovery_latency_frames=(recovery_transitions[0] - recovery_start if recovered else None),
    )
