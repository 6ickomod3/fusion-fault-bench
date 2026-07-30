"""Sequence-level M5-A persistent-fault evaluation on frozen replay support."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from fusion_fault_bench.contracts.manifest_v1alpha1 import MethodId
from fusion_fault_bench.replay_experiments import (
    ReplayEstimateFrame,
    ReplayEstimateSequence,
    ReplayObjectEstimate,
)
from fusion_fault_bench.replay_plan import ReplayPersistentCase

type PersistentMetric = Literal[
    "matched-center-mse",
    "conditional-matched-center-mse",
    "coverage",
    "undefined-output-rate",
]

_BASE_METHODS: tuple[MethodId, ...] = (
    "camera-only",
    "lidar-only",
    "fixed-fusion",
)
_ORACLE_TIE_ORDER = _BASE_METHODS


@dataclass(frozen=True, slots=True)
class ReplayPersistentMethodResult:
    """Sufficient statistics for one M5-A scene, condition, and method."""

    method: MethodId
    loss_sum_m2: float
    valid_object_frame_count: int
    eligible_object_frame_count: int

    def __post_init__(self) -> None:
        loss_sum = float(self.loss_sum_m2)
        if not math.isfinite(loss_sum) or loss_sum < 0.0:
            raise ValueError("persistent replay loss sum must be finite and nonnegative")
        if (
            type(self.valid_object_frame_count) is not int
            or type(self.eligible_object_frame_count) is not int
            or self.valid_object_frame_count < 0
            or self.eligible_object_frame_count <= 0
            or self.valid_object_frame_count > self.eligible_object_frame_count
        ):
            raise ValueError("persistent replay support counts are invalid")
        if self.valid_object_frame_count == 0 and loss_sum != 0.0:
            raise ValueError("zero persistent replay support requires a zero loss sum")
        object.__setattr__(self, "loss_sum_m2", loss_sum)

    @property
    def coverage(self) -> float:
        """Return valid output coverage on the frozen eligible denominator."""

        return self.valid_object_frame_count / self.eligible_object_frame_count

    @property
    def conditional_loss_m2(self) -> float | None:
        """Return matched-center MSE conditional on a defined output."""

        if self.valid_object_frame_count == 0:
            return None
        return self.loss_sum_m2 / self.valid_object_frame_count

    def metric(self, name: PersistentMetric) -> float | None:
        """Return one frozen M3 metric from the sufficient statistics."""

        if name in {"matched-center-mse", "conditional-matched-center-mse"}:
            return self.conditional_loss_m2
        if name == "coverage":
            return self.coverage
        return 1.0 - self.coverage


@dataclass(frozen=True, slots=True)
class ReplayPersistentSceneEvaluation:
    """Complete local M5-A sequence evidence before population inference."""

    replay_experiment_identity_sha256: str
    sequence_id: str
    condition_id: str
    condition_selector: str
    results: tuple[ReplayPersistentMethodResult, ...]
    cross_modal_disagreement_sum_m2: float
    cross_modal_common_count: int

    def __post_init__(self) -> None:
        if len(self.replay_experiment_identity_sha256) != 64:
            raise ValueError("persistent replay identity digest is invalid")
        if not self.sequence_id or not self.condition_id or not self.condition_selector:
            raise ValueError("persistent replay evaluation identifiers must be nonempty")
        methods = tuple(result.method for result in self.results)
        if not methods or len(methods) != len(set(methods)):
            raise ValueError("persistent replay evaluation methods must be unique")
        eligible_counts = {result.eligible_object_frame_count for result in self.results}
        if len(eligible_counts) != 1:
            raise ValueError("persistent replay methods must share frozen eligible support")
        disagreement_sum = float(self.cross_modal_disagreement_sum_m2)
        if (
            not math.isfinite(disagreement_sum)
            or disagreement_sum < 0.0
            or type(self.cross_modal_common_count) is not int
            or self.cross_modal_common_count < 0
            or self.cross_modal_common_count > next(iter(eligible_counts))
            or (self.cross_modal_common_count == 0 and disagreement_sum != 0.0)
        ):
            raise ValueError("persistent replay cross-modal disagreement support is invalid")
        object.__setattr__(
            self,
            "cross_modal_disagreement_sum_m2",
            disagreement_sum,
        )

    def result(self, method: MethodId) -> ReplayPersistentMethodResult:
        """Return one method's scene sufficient statistics."""

        for row in self.results:
            if row.method == method:
                return row
        raise KeyError(method)


def _value(
    item: ReplayObjectEstimate,
    method: MethodId,
) -> np.ndarray:
    if method == "camera-only":
        return item.camera_current_ego.point_m
    if method == "lidar-only":
        return item.lidar_current_ego.point_m
    if method == "fixed-fusion":
        return item.fixed_current_ego_xy_m
    raise ValueError("persistent replay value requires a base method")


def _available(frame: ReplayEstimateFrame, method: MethodId) -> bool:
    if method == "camera-only":
        return frame.camera_available
    if method == "lidar-only":
        return frame.lidar_available
    if method == "fixed-fusion":
        return frame.camera_available and frame.lidar_available
    raise ValueError("persistent replay availability requires a base method")


def _base_result(
    sequence: ReplayEstimateSequence,
    method: MethodId,
) -> ReplayPersistentMethodResult:
    loss_sum = 0.0
    valid_count = 0
    for frame in sequence.frames:
        if not _available(frame, method):
            continue
        for item in frame.objects:
            error = _value(item, method) - item.truth_current_ego_xy_m
            loss_sum += float(error @ error)
            valid_count += 1
    return ReplayPersistentMethodResult(
        method=method,
        loss_sum_m2=loss_sum,
        valid_object_frame_count=valid_count,
        eligible_object_frame_count=sequence.eligible_object_frame_count,
    )


def _selected_result(
    source: ReplayPersistentMethodResult,
    *,
    method: MethodId,
) -> ReplayPersistentMethodResult:
    return ReplayPersistentMethodResult(
        method=method,
        loss_sum_m2=source.loss_sum_m2,
        valid_object_frame_count=source.valid_object_frame_count,
        eligible_object_frame_count=source.eligible_object_frame_count,
    )


def evaluate_replay_persistent_sequence(
    case: ReplayPersistentCase,
    sequence: ReplayEstimateSequence,
) -> ReplayPersistentSceneEvaluation:
    """Evaluate one complete scene with unchanged M3 method semantics."""

    if sequence.condition != case.fault_condition:
        raise ValueError("persistent replay sequence uses the wrong fault coordinate")
    if sequence.condition.active_frames is not None:
        raise ValueError("M5-A persistent faults must span the complete scene")
    if sequence.eligible_object_frame_count <= 0:
        raise ValueError("M5-A requires nonempty frozen base support in every scene")

    by_method = {method: _base_result(sequence, method) for method in _BASE_METHODS}
    expected_methods = tuple(case.source_manifest.methods)

    if "fault-target-drop-policy" in expected_methods:
        if case.fault_condition.identity:
            target_source: MethodId = "fixed-fusion"
        elif case.fault_condition.target == "camera":
            target_source = "lidar-only"
        elif case.fault_condition.target == "lidar":
            target_source = "camera-only"
        else:
            raise ValueError("target-drop policy requires one declared faulty modality")
        by_method["fault-target-drop-policy"] = _selected_result(
            by_method[target_source],
            method="fault-target-drop-policy",
        )

    if "performance-oracle" in expected_methods:
        if any(
            by_method[method].valid_object_frame_count != sequence.eligible_object_frame_count
            for method in _ORACLE_TIE_ORDER
        ):
            raise ValueError("complete-sequence performance oracle requires common support")
        oracle_source = min(
            _ORACLE_TIE_ORDER,
            key=lambda method: cast(float, by_method[method].conditional_loss_m2),
        )
        by_method["performance-oracle"] = _selected_result(
            by_method[oracle_source],
            method="performance-oracle",
        )

    try:
        ordered = tuple(by_method[method] for method in expected_methods)
    except KeyError as error:
        raise ValueError("M5-A source manifest contains an unsupported method") from error
    if set(by_method) != set(expected_methods):
        raise ValueError("M5-A computed method set disagrees with the frozen source manifest")

    is_dropout = case.fault_condition.family == "dropout"
    if not is_dropout and any(
        result.valid_object_frame_count != result.eligible_object_frame_count for result in ordered
    ):
        raise ValueError("non-dropout M5-A methods must remain defined on frozen support")

    disagreement_sum = 0.0
    disagreement_count = 0
    for frame in sequence.frames:
        if not (frame.camera_available and frame.lidar_available):
            continue
        for item in frame.objects:
            disagreement = item.camera_current_ego.point_m - item.lidar_current_ego.point_m
            disagreement_sum += float(disagreement @ disagreement)
            disagreement_count += 1

    return ReplayPersistentSceneEvaluation(
        replay_experiment_identity_sha256=case.identity_sha256,
        sequence_id=sequence.sequence_id,
        condition_id=case.identity.experiment_id,
        condition_selector=case.fault_condition.selector,
        results=ordered,
        cross_modal_disagreement_sum_m2=disagreement_sum,
        cross_modal_common_count=disagreement_count,
    )
