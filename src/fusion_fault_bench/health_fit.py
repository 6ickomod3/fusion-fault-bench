"""Immutable M4 feature traces, clean ECDF fitting, and threshold selection."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.contracts.health_result_v1 import (
    CROSS_THRESHOLDS,
    SELF_THRESHOLDS,
    HealthThresholdCandidateV1,
)
from fusion_fault_bench.geometry._arrays import immutable_float64_copy
from fusion_fault_bench.health import (
    HealthCalibration,
    HealthFrameEvidence,
    HealthFrameInput,
    HealthScorer,
    NumericChannelEvidence,
    ecdf_rank,
)
from fusion_fault_bench.health_inference import equal_target_family_condition_mean
from fusion_fault_bench.inference import (
    paired_bootstrap_indices,
    percentile_interval,
)

type FloatArray = npt.NDArray[np.float64]
type NumericScoreStatus = Literal["defined", "insufficient-support"]
type ValidationTarget = Literal["camera", "lidar"]

TRAIN_SEQUENCE_COUNT = 200
FRAMES_PER_SEQUENCE = 48
ECDF_START_FRAME = 2
ECDF_STOP_FRAME = 48
ECDF_VALUES_PER_CHANNEL = 9_200
CANDIDATE_COUNT = 36
VALIDATION_SEQUENCE_COUNT = 200
VALIDATION_BOOTSTRAP_SEED = 2718
VALIDATION_BOOTSTRAP_REPLICATES = 2_000
SELECTION_TIE_TOLERANCE_M2 = 1e-12

_DUMMY_CALIBRATION = HealthCalibration(
    camera_self_mean=np.asarray([0.0], dtype=np.float64),
    camera_self_maximum=np.asarray([0.0], dtype=np.float64),
    lidar_self_mean=np.asarray([0.0], dtype=np.float64),
    lidar_self_maximum=np.asarray([0.0], dtype=np.float64),
    camera_from_lidar_cross_mean=np.asarray([0.0], dtype=np.float64),
    camera_from_lidar_cross_maximum=np.asarray([0.0], dtype=np.float64),
    lidar_from_camera_cross_mean=np.asarray([0.0], dtype=np.float64),
    lidar_from_camera_cross_maximum=np.asarray([0.0], dtype=np.float64),
)


def _finite_scalar(value: float, *, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _immutable_float_matrix(
    value: npt.ArrayLike,
    *,
    shape: tuple[int, int],
    field_name: str,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{field_name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    return immutable_float64_copy(array)


def _immutable_float_vector(
    value: npt.ArrayLike,
    *,
    length: int,
    field_name: str,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"{field_name} must have shape ({length},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    return immutable_float64_copy(array)


@dataclass(frozen=True, slots=True)
class UnscoredNumericChannel:
    """Raw frame-level NIS statistics before clean-ECDF calibration."""

    status: NumericScoreStatus
    mature_object_count: int
    current_object_count: int
    mature_fraction: float
    mean_nis: float | None
    maximum_nis: float | None

    def __post_init__(self) -> None:
        if self.status not in {"defined", "insufficient-support"}:
            raise ValueError("unknown numeric score status")
        if self.current_object_count <= 0:
            raise ValueError("current_object_count must be positive")
        if not 0 <= self.mature_object_count <= self.current_object_count:
            raise ValueError("mature_object_count must lie within current support")
        expected_fraction = self.mature_object_count / self.current_object_count
        if not math.isclose(self.mature_fraction, expected_fraction, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("mature_fraction does not match support counts")
        if self.status == "defined":
            if self.mature_object_count < 2:
                raise ValueError("defined channel requires at least two mature objects")
            if self.mean_nis is None or self.maximum_nis is None:
                raise ValueError("defined channel requires mean and maximum NIS")
            for value in (self.mean_nis, self.maximum_nis):
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError("NIS statistics must be finite and nonnegative")
            if self.maximum_nis < self.mean_nis:
                raise ValueError("maximum NIS cannot be below mean NIS")
        else:
            if self.mature_object_count >= 2:
                raise ValueError("insufficient support requires fewer than two mature objects")
            if self.mean_nis is not None or self.maximum_nis is not None:
                raise ValueError("insufficient-support channel cannot carry NIS statistics")


@dataclass(frozen=True, slots=True)
class UnscoredHealthFrame:
    """Direct telemetry and four raw NIS channels for one frame."""

    reference_time_s: float
    camera_available: bool
    lidar_available: bool
    camera_timestamp_suspicious: bool
    lidar_timestamp_suspicious: bool
    camera_missing_fraction_last_four: float
    lidar_missing_fraction_last_four: float
    camera_self: UnscoredNumericChannel
    lidar_self: UnscoredNumericChannel
    camera_from_lidar_cross: UnscoredNumericChannel
    lidar_from_camera_cross: UnscoredNumericChannel

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_time_s",
            _finite_scalar(self.reference_time_s, field_name="reference_time_s"),
        )
        for field_name in (
            "camera_missing_fraction_last_four",
            "lidar_missing_fraction_last_four",
        ):
            value = _finite_scalar(getattr(self, field_name), field_name=field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must lie in [0, 1]")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class HealthFeatureTrace:
    """One sequence's immutable, calibration-independent observable trace."""

    frames: tuple[UnscoredHealthFrame, ...]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("feature trace must contain at least one frame")
        previous_time: float | None = None
        for frame in self.frames:
            if previous_time is not None and frame.reference_time_s <= previous_time:
                raise ValueError("feature trace reference times must be strictly increasing")
            previous_time = frame.reference_time_s


@dataclass(frozen=True, slots=True)
class ScoredHealthTrace:
    """One immutable feature trace rescored by a frozen clean calibration."""

    frames: tuple[HealthFrameEvidence, ...]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("scored trace must contain at least one frame")


@dataclass(frozen=True, slots=True)
class ValidationConditionRegret:
    """Per-candidate, per-sequence policy-minus-frame-oracle regret."""

    condition_id: str
    target: ValidationTarget
    family: str
    regret_m2_by_candidate_sequence: FloatArray

    def __post_init__(self) -> None:
        if not self.condition_id:
            raise ValueError("condition_id must be nonempty")
        if self.target not in {"camera", "lidar"}:
            raise ValueError("validation target must be camera or lidar")
        if not self.family:
            raise ValueError("family must be nonempty")
        regrets = _immutable_float_matrix(
            self.regret_m2_by_candidate_sequence,
            shape=(CANDIDATE_COUNT, VALIDATION_SEQUENCE_COUNT),
            field_name="regret_m2_by_candidate_sequence",
        )
        if np.any(regrets < 0.0):
            raise ValueError("policy-minus-frame-oracle regret cannot be negative")
        object.__setattr__(self, "regret_m2_by_candidate_sequence", regrets)


@dataclass(frozen=True, slots=True)
class HealthThresholdSelection:
    """All retained candidates and the uniquely selected frozen grid point."""

    candidates: tuple[HealthThresholdCandidateV1, ...]
    selected_candidate_index: int
    selected_self_threshold: float
    selected_cross_threshold: float

    def __post_init__(self) -> None:
        if len(self.candidates) != CANDIDATE_COUNT:
            raise ValueError("selection must retain all 36 threshold candidates")
        if tuple(candidate.candidate_index for candidate in self.candidates) != tuple(
            range(CANDIDATE_COUNT)
        ):
            raise ValueError("threshold candidates must remain in exact grid order")
        if not 0 <= self.selected_candidate_index < CANDIDATE_COUNT:
            raise ValueError("selected candidate index lies outside the grid")
        selected = self.candidates[self.selected_candidate_index]
        if not selected.feasible:
            raise ValueError("selected threshold candidate must be feasible")
        if (
            selected.self_threshold != self.selected_self_threshold
            or selected.cross_threshold != self.selected_cross_threshold
        ):
            raise ValueError("selected thresholds disagree with the selected candidate")


def _unscored_channel(evidence: NumericChannelEvidence) -> UnscoredNumericChannel:
    return UnscoredNumericChannel(
        status=evidence.status,
        mature_object_count=evidence.mature_object_count,
        current_object_count=evidence.current_object_count,
        mature_fraction=evidence.mature_fraction,
        mean_nis=evidence.mean_nis,
        maximum_nis=evidence.maximum_nis,
    )


def compute_health_feature_trace(
    frames: Sequence[HealthFrameInput],
) -> HealthFeatureTrace:
    """Compute causal raw features exactly once for one ordered sequence."""

    if not frames:
        raise ValueError("frames must contain at least one HealthFrameInput")
    scorer = HealthScorer(_DUMMY_CALIBRATION)
    unscored: list[UnscoredHealthFrame] = []
    for frame in frames:
        evidence = scorer.process_frame(frame)
        unscored.append(
            UnscoredHealthFrame(
                reference_time_s=evidence.reference_time_s,
                camera_available=evidence.camera_available,
                lidar_available=evidence.lidar_available,
                camera_timestamp_suspicious=evidence.camera_timestamp_suspicious,
                lidar_timestamp_suspicious=evidence.lidar_timestamp_suspicious,
                camera_missing_fraction_last_four=evidence.camera_missing_fraction_last_four,
                lidar_missing_fraction_last_four=evidence.lidar_missing_fraction_last_four,
                camera_self=_unscored_channel(evidence.camera_self),
                lidar_self=_unscored_channel(evidence.lidar_self),
                camera_from_lidar_cross=_unscored_channel(evidence.camera_from_lidar_cross),
                lidar_from_camera_cross=_unscored_channel(evidence.lidar_from_camera_cross),
            )
        )
    return HealthFeatureTrace(frames=tuple(unscored))


def _rescore_channel(
    channel: UnscoredNumericChannel,
    *,
    clean_mean: FloatArray,
    clean_maximum: FloatArray,
) -> NumericChannelEvidence:
    if channel.status == "insufficient-support":
        return NumericChannelEvidence(
            status="insufficient-support",
            mature_object_count=channel.mature_object_count,
            current_object_count=channel.current_object_count,
            mature_fraction=channel.mature_fraction,
            mean_nis=None,
            maximum_nis=None,
            score=None,
        )
    assert channel.mean_nis is not None
    assert channel.maximum_nis is not None
    return NumericChannelEvidence(
        status="defined",
        mature_object_count=channel.mature_object_count,
        current_object_count=channel.current_object_count,
        mature_fraction=channel.mature_fraction,
        mean_nis=channel.mean_nis,
        maximum_nis=channel.maximum_nis,
        score=max(
            ecdf_rank(clean_mean, channel.mean_nis),
            ecdf_rank(clean_maximum, channel.maximum_nis),
        ),
    )


def rescore_health_feature_trace(
    trace: HealthFeatureTrace,
    calibration: HealthCalibration,
) -> ScoredHealthTrace:
    """Apply a frozen ECDF fit without recomputing predictor features."""

    frames = tuple(
        HealthFrameEvidence(
            reference_time_s=frame.reference_time_s,
            camera_available=frame.camera_available,
            lidar_available=frame.lidar_available,
            camera_timestamp_suspicious=frame.camera_timestamp_suspicious,
            lidar_timestamp_suspicious=frame.lidar_timestamp_suspicious,
            camera_missing_fraction_last_four=frame.camera_missing_fraction_last_four,
            lidar_missing_fraction_last_four=frame.lidar_missing_fraction_last_four,
            camera_self=_rescore_channel(
                frame.camera_self,
                clean_mean=calibration.camera_self_mean,
                clean_maximum=calibration.camera_self_maximum,
            ),
            lidar_self=_rescore_channel(
                frame.lidar_self,
                clean_mean=calibration.lidar_self_mean,
                clean_maximum=calibration.lidar_self_maximum,
            ),
            camera_from_lidar_cross=_rescore_channel(
                frame.camera_from_lidar_cross,
                clean_mean=calibration.camera_from_lidar_cross_mean,
                clean_maximum=calibration.camera_from_lidar_cross_maximum,
            ),
            lidar_from_camera_cross=_rescore_channel(
                frame.lidar_from_camera_cross,
                clean_mean=calibration.lidar_from_camera_cross_mean,
                clean_maximum=calibration.lidar_from_camera_cross_maximum,
            ),
        )
        for frame in trace.frames
    )
    return ScoredHealthTrace(frames=frames)


def _defined_statistic(
    channel: UnscoredNumericChannel,
    *,
    statistic: Literal["mean", "maximum"],
    sequence_index: int,
    frame_index: int,
    channel_name: str,
) -> float:
    if channel.status != "defined":
        raise ValueError(
            f"{channel_name} is not defined at train sequence {sequence_index}, frame {frame_index}"
        )
    value = channel.mean_nis if statistic == "mean" else channel.maximum_nis
    assert value is not None
    return value


def fit_clean_health_calibration(
    traces: Sequence[HealthFeatureTrace],
) -> HealthCalibration:
    """Fit the exact eight 9,200-value clean arrays from train frames ``[2,48)``."""

    if len(traces) != TRAIN_SEQUENCE_COUNT:
        raise ValueError("clean fit requires exactly 200 train sequences")
    channel_values: dict[str, list[float]] = {
        "camera_self_mean": [],
        "camera_self_maximum": [],
        "lidar_self_mean": [],
        "lidar_self_maximum": [],
        "camera_from_lidar_cross_mean": [],
        "camera_from_lidar_cross_maximum": [],
        "lidar_from_camera_cross_mean": [],
        "lidar_from_camera_cross_maximum": [],
    }
    for sequence_index, trace in enumerate(traces):
        if len(trace.frames) != FRAMES_PER_SEQUENCE:
            raise ValueError("every clean train trace must contain exactly 48 frames")
        for frame_index in range(ECDF_START_FRAME, ECDF_STOP_FRAME):
            frame = trace.frames[frame_index]
            for prefix, channel in (
                ("camera_self", frame.camera_self),
                ("lidar_self", frame.lidar_self),
                ("camera_from_lidar_cross", frame.camera_from_lidar_cross),
                ("lidar_from_camera_cross", frame.lidar_from_camera_cross),
            ):
                channel_values[f"{prefix}_mean"].append(
                    _defined_statistic(
                        channel,
                        statistic="mean",
                        sequence_index=sequence_index,
                        frame_index=frame_index,
                        channel_name=prefix,
                    )
                )
                channel_values[f"{prefix}_maximum"].append(
                    _defined_statistic(
                        channel,
                        statistic="maximum",
                        sequence_index=sequence_index,
                        frame_index=frame_index,
                        channel_name=prefix,
                    )
                )

    sorted_arrays: dict[str, FloatArray] = {}
    for name, values in channel_values.items():
        if len(values) != ECDF_VALUES_PER_CHANNEL:
            raise AssertionError(f"{name} did not produce exactly 9,200 values")
        sorted_arrays[name] = np.asarray(
            np.sort(np.asarray(values, dtype=np.float64), kind="stable"),
            dtype=np.float64,
        )
    return HealthCalibration(**sorted_arrays)


def _sequence_mean(values: FloatArray) -> float:
    return math.fsum(float(value) for value in values) / values.size


def _candidate_regret(
    condition_regrets: tuple[ValidationConditionRegret, ...],
    *,
    candidate_index: int,
) -> float:
    condition_means = np.asarray(
        [
            _sequence_mean(condition.regret_m2_by_candidate_sequence[candidate_index])
            for condition in condition_regrets
        ],
        dtype=np.float64,
    )
    return equal_target_family_condition_mean(
        condition_means,
        targets=tuple(condition.target for condition in condition_regrets),
        families=tuple(condition.family for condition in condition_regrets),
    )


def _select_tied_candidate(
    feasible: Sequence[HealthThresholdCandidateV1],
) -> HealthThresholdCandidateV1:
    minimum_regret = min(candidate.validation_regret_m2 for candidate in feasible)
    tied = tuple(
        candidate
        for candidate in feasible
        if candidate.validation_regret_m2 <= minimum_regret + SELECTION_TIE_TOLERANCE_M2
    )
    return min(
        tied,
        key=lambda candidate: (
            candidate.false_alert_episode_starts_per_sequence,
            candidate.mean_clean_regression_m2,
            -candidate.self_threshold,
            -candidate.cross_threshold,
        ),
    )


def select_health_thresholds(
    *,
    clean_regression_m2_by_candidate_sequence: npt.ArrayLike,
    clean_coverage_by_candidate_sequence: npt.ArrayLike,
    fixed_clean_coverage_by_sequence: npt.ArrayLike,
    false_alert_starts_by_candidate_sequence: npt.ArrayLike,
    condition_regrets: Sequence[ValidationConditionRegret],
) -> HealthThresholdSelection:
    """Evaluate all 36 candidates and apply the frozen feasibility/tie rules."""

    shape = (CANDIDATE_COUNT, VALIDATION_SEQUENCE_COUNT)
    clean_regression = _immutable_float_matrix(
        clean_regression_m2_by_candidate_sequence,
        shape=shape,
        field_name="clean_regression_m2_by_candidate_sequence",
    )
    clean_coverage = _immutable_float_matrix(
        clean_coverage_by_candidate_sequence,
        shape=shape,
        field_name="clean_coverage_by_candidate_sequence",
    )
    fixed_coverage = _immutable_float_vector(
        fixed_clean_coverage_by_sequence,
        length=VALIDATION_SEQUENCE_COUNT,
        field_name="fixed_clean_coverage_by_sequence",
    )
    false_alert_starts = _immutable_float_matrix(
        false_alert_starts_by_candidate_sequence,
        shape=shape,
        field_name="false_alert_starts_by_candidate_sequence",
    )
    if np.any((clean_coverage < 0.0) | (clean_coverage > 1.0)):
        raise ValueError("clean coverage must lie in [0, 1]")
    if np.any((fixed_coverage < 0.0) | (fixed_coverage > 1.0)):
        raise ValueError("fixed clean coverage must lie in [0, 1]")
    if np.any(false_alert_starts < 0.0):
        raise ValueError("false-alert starts cannot be negative")

    conditions = tuple(condition_regrets)
    if not conditions:
        raise ValueError("condition_regrets must contain at least one condition")
    condition_ids = tuple(condition.condition_id for condition in conditions)
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("condition regret IDs must be unique")
    if {condition.target for condition in conditions} != {"camera", "lidar"}:
        raise ValueError("selection utility must contain both camera and lidar targets")

    bootstrap_indices = paired_bootstrap_indices(
        seed=VALIDATION_BOOTSTRAP_SEED,
        replicates=VALIDATION_BOOTSTRAP_REPLICATES,
        sequence_count=VALIDATION_SEQUENCE_COUNT,
    )
    fixed_coverage_mean = _sequence_mean(fixed_coverage)
    candidates: list[HealthThresholdCandidateV1] = []
    for candidate_index in range(CANDIDATE_COUNT):
        regression_values = clean_regression[candidate_index]
        bootstrap_means = regression_values[bootstrap_indices].mean(axis=1)
        _, upper = percentile_interval(bootstrap_means, confidence_level=0.95)
        mean_regression = _sequence_mean(regression_values)
        mean_false_alerts = _sequence_mean(false_alert_starts[candidate_index])
        mean_clean_coverage = _sequence_mean(clean_coverage[candidate_index])
        feasible = (
            mean_regression <= 0.002
            and upper <= 0.005
            and mean_false_alerts <= 0.05
            and mean_clean_coverage == fixed_coverage_mean
        )
        candidates.append(
            HealthThresholdCandidateV1(
                schema="ffb.health-threshold-candidate/v1",
                candidate_index=candidate_index,
                self_threshold=SELF_THRESHOLDS[candidate_index // len(CROSS_THRESHOLDS)],
                cross_threshold=CROSS_THRESHOLDS[candidate_index % len(CROSS_THRESHOLDS)],
                mean_clean_regression_m2=mean_regression,
                upper_95pct_clean_regression_m2=upper,
                false_alert_episode_starts_per_sequence=mean_false_alerts,
                clean_coverage=mean_clean_coverage,
                fixed_clean_coverage=fixed_coverage_mean,
                feasible=feasible,
                validation_regret_m2=_candidate_regret(
                    conditions,
                    candidate_index=candidate_index,
                ),
            )
        )

    feasible_candidates = tuple(candidate for candidate in candidates if candidate.feasible)
    if not feasible_candidates:
        raise RuntimeError("no threshold candidate passes the frozen validation gates")
    selected = _select_tied_candidate(feasible_candidates)
    return HealthThresholdSelection(
        candidates=tuple(candidates),
        selected_candidate_index=selected.candidate_index,
        selected_self_threshold=selected.self_threshold,
        selected_cross_threshold=selected.cross_threshold,
    )
