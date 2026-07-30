"""Variable-support M5-B health evaluation in the current ego scoring frame."""

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
    HealthWindow,
)
from fusion_fault_bench.contracts.replay_health_v1 import (
    ReplayHealthResultV1,
    ReplayHealthSequenceEventV1,
)
from fusion_fault_bench.geometry._arrays import immutable_float64_copy
from fusion_fault_bench.health import (
    ExecutedAction,
    HealthCalibration,
    HealthMethodId,
    ModalityMeasurement,
    ObjectHealthInput,
)
from fusion_fault_bench.replay_experiments import (
    ReplayEstimateFrame,
    ReplayEstimateSequence,
    ReplayObjectEstimate,
)
from fusion_fault_bench.replay_fit import (
    FrozenReplayHealthFit,
    validate_frozen_replay_health_fit,
)
from fusion_fault_bench.replay_health import (
    ReplayHealthFrameEvidence,
    ReplayHealthFrameInput,
    ReplayHealthPolicyTrace,
    ReplayHealthScorer,
    evaluate_replay_policy_trace,
    replay_health_schedule,
    replay_sequence_event_record,
)
from fusion_fault_bench.replay_inference import ReplayHealthSequenceContrast
from fusion_fault_bench.replay_plan import ReplayHealthCaseSpec

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type RowKey = tuple[int, str]

_POLICY_METHODS: tuple[HealthMethodId, ...] = (
    "self-nis-gate",
    "cross-nis-gate",
    "direct-telemetry-gate",
    "combined-health-gate",
)
_POLICY_RESULT_METHODS: tuple[HealthPolicyMethod, ...] = (
    *_POLICY_METHODS,
    "combined-health-gate-abstain",
)
_BASE_METHODS: tuple[HealthMethod, ...] = (
    "camera-only",
    "lidar-only",
    "fixed-fusion",
)
_WINDOW_ORDER: tuple[HealthWindow, ...] = ("score", "event", "recovery")
_SUPPORT_DOMAIN = b"fusion-fault-bench/m5-health-support/v1\x00"


def _immutable_bool(value: npt.ArrayLike) -> BoolArray:
    source = np.ascontiguousarray(np.asarray(value, dtype=np.bool_))
    return np.frombuffer(source.tobytes(order="C"), dtype=np.bool_).reshape(source.shape)


def _field(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) >= 2**32:
        raise ValueError("support commitment field is too long")
    return len(encoded).to_bytes(4, "big") + encoded


def _support_sha256(
    *,
    sequence_id: str,
    condition_selector: str,
    window: HealthWindow,
    row_keys: tuple[RowKey, ...],
    valid: BoolArray,
) -> str:
    if valid.shape != (len(row_keys),):
        raise ValueError("support mask does not align with replay row keys")
    payload = bytearray(_SUPPORT_DOMAIN)
    payload.extend(_field(sequence_id))
    payload.extend(_field(condition_selector))
    payload.extend(_field(window))
    selected = tuple(key for key, is_valid in zip(row_keys, valid, strict=True) if bool(is_valid))
    payload.extend(len(selected).to_bytes(8, "big"))
    for frame_index, object_id in selected:
        payload.extend(frame_index.to_bytes(8, "big", signed=False))
        payload.extend(_field(object_id))
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class ReplayLossTrace:
    """One method's flattened object-row losses and exact validity mask."""

    method: HealthMethod
    row_keys: tuple[RowKey, ...]
    loss_m2: FloatArray
    valid: BoolArray

    def __post_init__(self) -> None:
        row_count = len(self.row_keys)
        loss = np.asarray(self.loss_m2, dtype=np.float64)
        valid = np.asarray(self.valid, dtype=np.bool_)
        if loss.shape != (row_count,) or valid.shape != (row_count,):
            raise ValueError("replay loss trace arrays must align with row keys")
        if not bool(np.all(np.isfinite(loss))) or bool(np.any(loss < 0.0)):
            raise ValueError("replay losses must be finite and nonnegative")
        if bool(np.any(loss[~valid] != 0.0)):
            raise ValueError("invalid replay rows must carry zero loss")
        object.__setattr__(self, "loss_m2", immutable_float64_copy(loss))
        object.__setattr__(self, "valid", _immutable_bool(valid))


@dataclass(frozen=True, slots=True)
class ReplayHealthSceneEvaluation:
    """All local sequence-level M5-B evidence before population inference."""

    replay_experiment_identity_sha256: str
    sequence_id: str
    condition_selector: str
    evidence: tuple[ReplayHealthFrameEvidence, ...]
    policy_traces: tuple[ReplayHealthPolicyTrace, ...]
    losses: tuple[ReplayLossTrace, ...]
    results: tuple[ReplayHealthResultV1, ...]
    contrasts: tuple[ReplayHealthSequenceContrast, ...]
    events: tuple[ReplayHealthSequenceEventV1, ...]

    def __post_init__(self) -> None:
        if not self.evidence:
            raise ValueError("replay health evaluation requires frame evidence")
        if len(self.policy_traces) != len(_POLICY_RESULT_METHODS):
            raise ValueError("replay health evaluation has an incomplete policy set")
        if tuple(trace.policy for trace in self.policy_traces[:4]) != _POLICY_METHODS:
            raise ValueError("replay health policies are out of frozen order")

    def loss(self, method: HealthMethod) -> ReplayLossTrace:
        """Return one exact method loss trace."""

        for trace in self.losses:
            if trace.method == method:
                return trace
        raise KeyError(method)


def _health_evidence(
    sequence: ReplayEstimateSequence,
    *,
    calibration: HealthCalibration,
) -> tuple[ReplayHealthFrameEvidence, ...]:
    scorer = ReplayHealthScorer(calibration)
    evidence: list[ReplayHealthFrameEvidence] = []
    for frame in sequence.frames:
        objects = tuple(
            ObjectHealthInput(
                object_id=item.object_id,
                camera=(
                    ModalityMeasurement(
                        value_xy_m=item.camera_monitoring_scene.point_m,
                        reported_covariance_xy_m2=(
                            item.camera_monitoring_scene.reported_covariance_m2
                        ),
                        reported_time_s=item.camera_reported_state_time_s,
                    )
                    if frame.camera_available
                    else None
                ),
                lidar=(
                    ModalityMeasurement(
                        value_xy_m=item.lidar_monitoring_scene.point_m,
                        reported_covariance_xy_m2=(
                            item.lidar_monitoring_scene.reported_covariance_m2
                        ),
                        reported_time_s=item.lidar_reported_state_time_s,
                    )
                    if frame.lidar_available
                    else None
                ),
            )
            for item in frame.objects
        )
        evidence.append(
            scorer.process_frame(
                ReplayHealthFrameInput(
                    reference_time_s=frame.reference_time_s,
                    camera_available=frame.camera_available,
                    lidar_available=frame.lidar_available,
                    objects=objects,
                )
            )
        )
    return tuple(evidence)


def _action_defined(action: ExecutedAction, frame: ReplayEstimateFrame) -> bool:
    if action == "camera-only":
        return frame.camera_available
    if action == "lidar-only":
        return frame.lidar_available
    if action == "fixed-fusion":
        return frame.camera_available and frame.lidar_available
    return False


def _value_for_action(
    item: ReplayObjectEstimate,
    *,
    action: ExecutedAction,
) -> FloatArray:
    if action == "camera-only":
        return item.camera_current_ego.point_m
    if action == "lidar-only":
        return item.lidar_current_ego.point_m
    if action == "fixed-fusion":
        return item.fixed_current_ego_xy_m
    raise ValueError("undefined action has no estimator value")


def _row_keys(sequence: ReplayEstimateSequence) -> tuple[RowKey, ...]:
    return tuple(
        (frame.frame_index, item.object_id) for frame in sequence.frames for item in frame.objects
    )


def _loss_trace(
    sequence: ReplayEstimateSequence,
    *,
    method: HealthMethod,
    actions: tuple[ExecutedAction, ...],
) -> ReplayLossTrace:
    if len(actions) != len(sequence.frames):
        raise ValueError("replay action trace does not align with frames")
    keys = _row_keys(sequence)
    loss = np.zeros(len(keys), dtype=np.float64)
    valid = np.zeros(len(keys), dtype=np.bool_)
    row_index = 0
    for frame, action in zip(sequence.frames, actions, strict=True):
        defined = _action_defined(action, frame)
        for item in frame.objects:
            if defined:
                error = _value_for_action(item, action=action) - item.truth_current_ego_xy_m
                loss[row_index] = float(error @ error)
                valid[row_index] = True
            row_index += 1
    return ReplayLossTrace(
        method=method,
        row_keys=keys,
        loss_m2=loss,
        valid=valid,
    )


def _repeat_action(
    action: ExecutedAction,
    *,
    frame_count: int,
) -> tuple[ExecutedAction, ...]:
    return tuple(action for _ in range(frame_count))


def _target_drop_actions(
    sequence: ReplayEstimateSequence,
) -> tuple[ExecutedAction, ...] | None:
    condition = sequence.condition
    if condition.family == "common-mode-position-bias":
        return None
    actions: list[ExecutedAction] = ["fixed-fusion"] * len(sequence.frames)
    if condition.target == "camera":
        healthy: ExecutedAction = "lidar-only"
    elif condition.target == "lidar":
        healthy = "camera-only"
    else:
        return tuple(actions)
    if condition.active_frames is None:
        start, end = 0, len(actions)
    else:
        start, end = condition.active_frames
    for frame_index in range(start, end):
        actions[frame_index] = healthy
    return tuple(actions)


def _frame_oracle_actions(
    sequence: ReplayEstimateSequence,
) -> tuple[ExecutedAction, ...] | None:
    if sequence.condition.family == "common-mode-position-bias":
        return None
    actions: list[ExecutedAction] = []
    ordered_actions: tuple[ExecutedAction, ...] = (
        "camera-only",
        "lidar-only",
        "fixed-fusion",
    )
    for frame in sequence.frames:
        candidates: list[tuple[float, ExecutedAction]] = []
        if frame.objects:
            for action in ordered_actions:
                if not _action_defined(action, frame):
                    continue
                losses = tuple(
                    float(
                        (_value_for_action(item, action=action) - item.truth_current_ego_xy_m)
                        @ (_value_for_action(item, action=action) - item.truth_current_ego_xy_m)
                    )
                    for item in frame.objects
                )
                candidates.append((math.fsum(losses) / len(losses), action))
        actions.append(
            "undefined" if not candidates else min(candidates, key=lambda item: item[0])[1]
        )
    return tuple(actions)


def _window_ranges(
    sequence: ReplayEstimateSequence,
) -> tuple[tuple[HealthWindow, tuple[int, int]], ...]:
    schedule = replay_health_schedule(len(sequence.frames))
    return (
        ("score", schedule.score_frames),
        ("event", schedule.fault_active_frames),
        ("recovery", schedule.recovery_frames),
    )


def _window_mask(
    row_keys: tuple[RowKey, ...],
    bounds: tuple[int, int],
) -> BoolArray:
    start, end = bounds
    return _immutable_bool(
        np.asarray(
            [start <= frame_index < end for frame_index, _ in row_keys],
            dtype=np.bool_,
        )
    )


def _result_rows(
    sequence: ReplayEstimateSequence,
    *,
    replay_experiment_identity_sha256: str,
    losses: tuple[ReplayLossTrace, ...],
) -> tuple[ReplayHealthResultV1, ...]:
    rows: list[ReplayHealthResultV1] = []
    for window, bounds in _window_ranges(sequence):
        mask = _window_mask(losses[0].row_keys, bounds)
        eligible_count = int(np.count_nonzero(mask))
        for trace in losses:
            valid = mask & trace.valid
            rows.append(
                ReplayHealthResultV1(
                    schema="ffb.replay-health-result/v1",
                    replay_experiment_identity_sha256=(replay_experiment_identity_sha256),
                    sequence_id=sequence.sequence_id,
                    condition_id=sequence.condition.experiment_id,
                    condition_selector=sequence.condition.selector,
                    method=trace.method,
                    window=window,
                    loss_sum_m2=math.fsum(float(value) for value in trace.loss_m2[valid]),
                    valid_object_frame_count=int(np.count_nonzero(valid)),
                    eligible_object_frame_count=eligible_count,
                )
            )
    return tuple(rows)


def _paired_sums(
    left: ReplayLossTrace,
    right: ReplayLossTrace,
    *,
    window_mask: BoolArray,
) -> tuple[BoolArray, int, float, float]:
    common = _immutable_bool(window_mask & left.valid & right.valid)
    return (
        common,
        int(np.count_nonzero(common)),
        math.fsum(float(value) for value in left.loss_m2[common]),
        math.fsum(float(value) for value in right.loss_m2[common]),
    )


def _contrast_rows(
    sequence: ReplayEstimateSequence,
    *,
    replay_experiment_identity_sha256: str,
    losses: tuple[ReplayLossTrace, ...],
) -> tuple[ReplayHealthSequenceContrast, ...]:
    by_method = {trace.method: trace for trace in losses}
    fixed = by_method["fixed-fusion"]
    target = by_method.get("fault-target-drop-policy")
    oracle = by_method.get("frame-action-performance-oracle")
    rows: list[ReplayHealthSequenceContrast] = []
    for window, bounds in _window_ranges(sequence):
        window_mask = _window_mask(fixed.row_keys, bounds)
        for policy in _POLICY_RESULT_METHODS:
            policy_trace = by_method[policy]
            _, fixed_count, fixed_sum, policy_fixed_sum = _paired_sums(
                fixed,
                policy_trace,
                window_mask=window_mask,
            )
            policy_support = _immutable_bool(window_mask & policy_trace.valid)
            fixed_support = _immutable_bool(window_mask & fixed.valid)

            target_count: int | None = None
            policy_target_sum: float | None = None
            target_sum: float | None = None
            target_support_sha256: str | None = None
            if target is not None:
                _, target_count, policy_target_sum, target_sum = _paired_sums(
                    policy_trace,
                    target,
                    window_mask=window_mask,
                )
                target_support_sha256 = _support_sha256(
                    sequence_id=sequence.sequence_id,
                    condition_selector=sequence.condition.selector,
                    window=window,
                    row_keys=target.row_keys,
                    valid=_immutable_bool(window_mask & target.valid),
                )

            oracle_count: int | None = None
            policy_oracle_sum: float | None = None
            oracle_sum: float | None = None
            oracle_support_sha256: str | None = None
            if oracle is not None:
                _, oracle_count, policy_oracle_sum, oracle_sum = _paired_sums(
                    policy_trace,
                    oracle,
                    window_mask=window_mask,
                )
                oracle_support_sha256 = _support_sha256(
                    sequence_id=sequence.sequence_id,
                    condition_selector=sequence.condition.selector,
                    window=window,
                    row_keys=oracle.row_keys,
                    valid=_immutable_bool(window_mask & oracle.valid),
                )

            rows.append(
                ReplayHealthSequenceContrast(
                    replay_experiment_identity_sha256=(replay_experiment_identity_sha256),
                    sequence_id=sequence.sequence_id,
                    condition_id=sequence.condition.experiment_id,
                    condition_selector=sequence.condition.selector,
                    policy=policy,
                    window=window,
                    fixed_support_sha256=_support_sha256(
                        sequence_id=sequence.sequence_id,
                        condition_selector=sequence.condition.selector,
                        window=window,
                        row_keys=fixed.row_keys,
                        valid=fixed_support,
                    ),
                    policy_support_sha256=_support_sha256(
                        sequence_id=sequence.sequence_id,
                        condition_selector=sequence.condition.selector,
                        window=window,
                        row_keys=policy_trace.row_keys,
                        valid=policy_support,
                    ),
                    fixed_policy_common_count=fixed_count,
                    fixed_on_common_loss_sum_m2=fixed_sum,
                    policy_on_fixed_common_loss_sum_m2=policy_fixed_sum,
                    target_drop_applicable=target is not None,
                    policy_target_drop_common_count=target_count,
                    policy_on_target_common_loss_sum_m2=policy_target_sum,
                    target_drop_on_common_loss_sum_m2=target_sum,
                    target_drop_support_sha256=target_support_sha256,
                    frame_oracle_applicable=oracle is not None,
                    policy_frame_oracle_common_count=oracle_count,
                    policy_on_oracle_common_loss_sum_m2=policy_oracle_sum,
                    frame_oracle_on_common_loss_sum_m2=oracle_sum,
                    frame_oracle_support_sha256=oracle_support_sha256,
                )
            )
    return tuple(rows)


def evaluate_replay_health_sequence(
    sequence: ReplayEstimateSequence,
    *,
    case: ReplayHealthCaseSpec,
    fit: FrozenReplayHealthFit,
) -> ReplayHealthSceneEvaluation:
    """Evaluate the frozen M4 fit without refitting or changing M4 behavior."""

    validate_frozen_replay_health_fit(fit)
    schedule = replay_health_schedule(len(sequence.frames))
    if sequence.condition != case.for_frame_count(len(sequence.frames)):
        raise ValueError("M5-B sequence does not bind the exact frozen case coordinate")
    identity_sha256 = case.identity_sha256
    evidence = _health_evidence(sequence, calibration=fit.calibration)
    regular_policy_traces = tuple(
        evaluate_replay_policy_trace(
            evidence,
            method=method,
            thresholds=fit.thresholds,
        )
        for method in _POLICY_METHODS
    )
    abstain_trace = evaluate_replay_policy_trace(
        evidence,
        method="combined-health-gate",
        thresholds=fit.thresholds,
        abstain_on_ambiguous=True,
    )
    combined_trace = regular_policy_traces[-1]
    if (
        abstain_trace.raw_labels != combined_trace.raw_labels
        or abstain_trace.evidence_statuses != combined_trace.evidence_statuses
        or abstain_trace.latched_labels != combined_trace.latched_labels
    ):
        raise AssertionError("replay abstention changed the combined health-state trace")
    policy_traces = (*regular_policy_traces, abstain_trace)

    frame_count = len(sequence.frames)
    action_by_method: list[tuple[HealthMethod, tuple[ExecutedAction, ...]]] = [
        ("camera-only", _repeat_action("camera-only", frame_count=frame_count)),
        ("lidar-only", _repeat_action("lidar-only", frame_count=frame_count)),
        ("fixed-fusion", _repeat_action("fixed-fusion", frame_count=frame_count)),
        *(
            (cast(HealthMethod, method), trace.actions)
            for method, trace in zip(
                _POLICY_RESULT_METHODS,
                policy_traces,
                strict=True,
            )
        ),
    ]
    target_actions = _target_drop_actions(sequence)
    if target_actions is not None:
        action_by_method.append(("fault-target-drop-policy", target_actions))
    oracle_actions = _frame_oracle_actions(sequence)
    if oracle_actions is not None:
        action_by_method.append(("frame-action-performance-oracle", oracle_actions))
    losses = tuple(
        _loss_trace(sequence, method=method, actions=actions)
        for method, actions in action_by_method
    )
    events = tuple(
        replay_sequence_event_record(
            trace,
            schedule=schedule,
            replay_experiment_identity_sha256=identity_sha256,
            sequence_id=sequence.sequence_id,
            condition_id=sequence.condition.experiment_id,
            condition_selector=sequence.condition.selector,
            fault_family=sequence.condition.family,
            fault_target=sequence.condition.target,
        )
        for trace in regular_policy_traces
    )
    return ReplayHealthSceneEvaluation(
        replay_experiment_identity_sha256=identity_sha256,
        sequence_id=sequence.sequence_id,
        condition_selector=sequence.condition.selector,
        evidence=evidence,
        policy_traces=policy_traces,
        losses=losses,
        results=_result_rows(
            sequence,
            replay_experiment_identity_sha256=identity_sha256,
            losses=losses,
        ),
        contrasts=_contrast_rows(
            sequence,
            replay_experiment_identity_sha256=identity_sha256,
            losses=losses,
        ),
        events=events,
    )
