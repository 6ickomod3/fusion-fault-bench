"""Observable, causal health scoring and fallback policy for M4.

The scorer deliberately accepts only current estimator outputs, reported
covariances, availability, reference time, reported sensor time, and an opaque
object identifier.  Scenario metadata and truth are not part of this API.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.geometry._arrays import immutable_float64_copy

type FloatArray = npt.NDArray[np.float64]
type ModalityId = Literal["camera", "lidar"]
type HealthLabel = Literal["healthy", "camera-fault", "lidar-fault", "ambiguous"]
type NumericScoreStatus = Literal["defined", "insufficient-support"]
type RawEvidenceStatus = Literal["update-eligible", "insufficient-support"]
type HealthMethodId = Literal[
    "self-nis-gate",
    "cross-nis-gate",
    "direct-telemetry-gate",
    "combined-health-gate",
]
type ExecutedAction = Literal[
    "camera-only",
    "lidar-only",
    "fixed-fusion",
    "undefined",
]

_NONHEALTHY_LABELS: tuple[HealthLabel, ...] = (
    "camera-fault",
    "lidar-fault",
    "ambiguous",
)
_TIMESTAMP_TOLERANCE_S = 1e-12
_MINIMUM_MATURE_OBJECT_COUNT = 2
_MISSING_WINDOW_LENGTH = 4


def _finite_scalar(value: float, *, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _immutable_vector2(value: npt.ArrayLike, *, field_name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError(f"{field_name} must have shape (2,)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    return immutable_float64_copy(array)


def _immutable_spd2(value: npt.ArrayLike, *, field_name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2, 2):
        raise ValueError(f"{field_name} must have shape (2, 2)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    if not np.allclose(array, array.T, rtol=0.0, atol=1e-12):
        raise ValueError(f"{field_name} must be symmetric")
    symmetric = np.asarray((array + array.T) / 2.0, dtype=np.float64)
    try:
        np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{field_name} must be positive definite") from error
    return immutable_float64_copy(symmetric)


def _immutable_sorted_ecdf(value: npt.ArrayLike, *, field_name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{field_name} must be a nonempty vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    if np.any(array[1:] < array[:-1]):
        raise ValueError(f"{field_name} must be sorted in nondecreasing order")
    return immutable_float64_copy(array)


@dataclass(frozen=True, slots=True)
class ModalityMeasurement:
    """One current aligned estimate and its estimator-reported uncertainty."""

    value_xy_m: FloatArray
    reported_covariance_xy_m2: FloatArray
    reported_time_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "value_xy_m",
            _immutable_vector2(self.value_xy_m, field_name="value_xy_m"),
        )
        object.__setattr__(
            self,
            "reported_covariance_xy_m2",
            _immutable_spd2(
                self.reported_covariance_xy_m2,
                field_name="reported_covariance_xy_m2",
            ),
        )
        object.__setattr__(
            self,
            "reported_time_s",
            _finite_scalar(self.reported_time_s, field_name="reported_time_s"),
        )


@dataclass(frozen=True, slots=True)
class ObjectHealthInput:
    """Current measurements for one opaque known object identifier."""

    object_id: str
    camera: ModalityMeasurement | None
    lidar: ModalityMeasurement | None

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("object_id must be a nonempty opaque string")


@dataclass(frozen=True, slots=True)
class HealthFrameInput:
    """Leakage-bounded input for one reference-time frame."""

    reference_time_s: float
    camera_available: bool
    lidar_available: bool
    objects: tuple[ObjectHealthInput, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_time_s",
            _finite_scalar(self.reference_time_s, field_name="reference_time_s"),
        )
        if not self.objects:
            raise ValueError("objects must be a nonempty tuple")
        identifiers: set[str] = set()
        for item in self.objects:
            if item.object_id in identifiers:
                raise ValueError("object_id values must be unique within a frame")
            identifiers.add(item.object_id)
            if not self.camera_available and item.camera is not None:
                raise ValueError("camera measurement cannot exist when camera is unavailable")
            if not self.lidar_available and item.lidar is not None:
                raise ValueError("lidar measurement cannot exist when lidar is unavailable")


@dataclass(frozen=True, slots=True)
class HealthCalibration:
    """The eight exact sorted clean ECDF arrays frozen by the fit artifact."""

    camera_self_mean: FloatArray
    camera_self_maximum: FloatArray
    lidar_self_mean: FloatArray
    lidar_self_maximum: FloatArray
    camera_from_lidar_cross_mean: FloatArray
    camera_from_lidar_cross_maximum: FloatArray
    lidar_from_camera_cross_mean: FloatArray
    lidar_from_camera_cross_maximum: FloatArray

    def __post_init__(self) -> None:
        for field_name in (
            "camera_self_mean",
            "camera_self_maximum",
            "lidar_self_mean",
            "lidar_self_maximum",
            "camera_from_lidar_cross_mean",
            "camera_from_lidar_cross_maximum",
            "lidar_from_camera_cross_mean",
            "lidar_from_camera_cross_maximum",
        ):
            object.__setattr__(
                self,
                field_name,
                _immutable_sorted_ecdf(
                    getattr(self, field_name),
                    field_name=field_name,
                ),
            )


@dataclass(frozen=True, slots=True)
class HealthThresholds:
    """One globally selected self/cross threshold pair."""

    self_score: float
    cross_score: float

    def __post_init__(self) -> None:
        for field_name in ("self_score", "cross_score"):
            value = _finite_scalar(getattr(self, field_name), field_name=field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must lie in [0, 1]")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class NumericChannelEvidence:
    """One frame-level NIS channel and its strict clean-ECDF score."""

    status: NumericScoreStatus
    mature_object_count: int
    current_object_count: int
    mature_fraction: float
    mean_nis: float | None
    maximum_nis: float | None
    score: float | None

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
        defined_values = (self.mean_nis, self.maximum_nis, self.score)
        if self.status == "defined":
            if self.mature_object_count < _MINIMUM_MATURE_OBJECT_COUNT:
                raise ValueError("defined numeric evidence requires at least two mature objects")
            if any(value is None for value in defined_values):
                raise ValueError("defined numeric evidence requires finite statistics and score")
            for value in defined_values:
                assert value is not None
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError("numeric statistics and score must be finite and nonnegative")
            assert self.score is not None
            if self.score > 1.0:
                raise ValueError("numeric score cannot exceed one")
        else:
            if self.mature_object_count >= _MINIMUM_MATURE_OBJECT_COUNT:
                raise ValueError("insufficient-support requires fewer than two mature objects")
            if any(value is not None for value in defined_values):
                raise ValueError("insufficient-support numeric evidence cannot contain statistics")


@dataclass(frozen=True, slots=True)
class HealthFrameEvidence:
    """Immutable observable features produced before current history updates."""

    reference_time_s: float
    camera_available: bool
    lidar_available: bool
    camera_timestamp_suspicious: bool
    lidar_timestamp_suspicious: bool
    camera_missing_fraction_last_four: float
    lidar_missing_fraction_last_four: float
    camera_self: NumericChannelEvidence
    lidar_self: NumericChannelEvidence
    camera_from_lidar_cross: NumericChannelEvidence
    lidar_from_camera_cross: NumericChannelEvidence

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

    @property
    def camera_direct_suspicious(self) -> bool:
        return not self.camera_available or self.camera_timestamp_suspicious

    @property
    def lidar_direct_suspicious(self) -> bool:
        return not self.lidar_available or self.lidar_timestamp_suspicious


@dataclass(frozen=True, slots=True)
class RawHealthDecision:
    """One raw label, kept distinct from latch state and estimator action."""

    method: HealthMethodId
    label: HealthLabel
    evidence_status: RawEvidenceStatus
    camera_alarm: bool | None
    lidar_alarm: bool | None
    any_cross_alarm: bool | None

    def __post_init__(self) -> None:
        if self.method not in {
            "self-nis-gate",
            "cross-nis-gate",
            "direct-telemetry-gate",
            "combined-health-gate",
        }:
            raise ValueError("unknown health method")
        if self.label not in {"healthy", *_NONHEALTHY_LABELS}:
            raise ValueError("unknown raw health label")
        if self.evidence_status not in {"update-eligible", "insufficient-support"}:
            raise ValueError("unknown raw evidence status")


@dataclass(frozen=True, slots=True)
class HealthLatchState:
    """Frozen recurrence state for exact activation/recovery semantics."""

    label: HealthLabel = "healthy"
    activation_candidate: HealthLabel | None = None
    activation_count: int = 0
    recovery_count: int = 0

    def __post_init__(self) -> None:
        if self.label not in {"healthy", *_NONHEALTHY_LABELS}:
            raise ValueError("unknown latched health label")
        if self.activation_candidate not in {None, *_NONHEALTHY_LABELS}:
            raise ValueError("activation_candidate must be nonhealthy or None")
        if self.activation_count not in {0, 1}:
            raise ValueError("activation_count must be zero or one between frames")
        if self.recovery_count not in {0, 1, 2}:
            raise ValueError("recovery_count must lie in {0, 1, 2}")
        if self.label == "healthy":
            if self.recovery_count != 0:
                raise ValueError("healthy state cannot retain a recovery count")
            if (self.activation_candidate is None) != (self.activation_count == 0):
                raise ValueError("healthy activation candidate/count are inconsistent")
        elif self.activation_candidate is not None or self.activation_count != 0:
            raise ValueError("nonhealthy state cannot retain an activation candidate")


@dataclass(frozen=True, slots=True)
class HealthPolicyOutput:
    """Raw evidence, resulting latch, and current executed action."""

    evidence: HealthFrameEvidence
    raw_decision: RawHealthDecision
    latched_state: HealthLatchState
    executed_action: ExecutedAction


@dataclass(frozen=True, slots=True)
class _HistoryItem:
    value_xy_m: FloatArray
    reported_covariance_xy_m2: FloatArray
    reference_time_s: float


@dataclass(frozen=True, slots=True)
class _Prediction:
    value_xy_m: FloatArray
    covariance_xy_m2: FloatArray


def ecdf_rank(clean_sorted_values: npt.ArrayLike, value: float) -> float:
    """Return ``count(clean_value < value) / n`` with exact strict tie handling."""

    clean = _immutable_sorted_ecdf(clean_sorted_values, field_name="clean_sorted_values")
    checked_value = _finite_scalar(value, field_name="value")
    return int(np.searchsorted(clean, checked_value, side="left")) / clean.size


def _predict(
    history: deque[_HistoryItem],
    *,
    reference_time_s: float,
) -> _Prediction | None:
    if len(history) < 2:
        return None
    first, second = history
    if not first.reference_time_s < second.reference_time_s < reference_time_s:
        raise ValueError("prediction history must satisfy t_a < t_b < t_k")
    denominator = second.reference_time_s - first.reference_time_s
    h = (reference_time_s - second.reference_time_s) / denominator
    value = (1.0 + h) * second.value_xy_m - h * first.value_xy_m
    covariance = (
        1.0 + h
    ) ** 2 * second.reported_covariance_xy_m2 + h**2 * first.reported_covariance_xy_m2
    return _Prediction(
        value_xy_m=immutable_float64_copy(value),
        covariance_xy_m2=immutable_float64_copy(covariance),
    )


def _nis(
    *,
    current: ModalityMeasurement,
    prediction: _Prediction,
) -> float:
    residual = current.value_xy_m - prediction.value_xy_m
    innovation_covariance = prediction.covariance_xy_m2 + current.reported_covariance_xy_m2
    try:
        solved = np.linalg.solve(innovation_covariance, residual)
    except np.linalg.LinAlgError as error:
        raise ValueError("NIS innovation covariance must be nonsingular") from error
    result = float(residual @ solved)
    if not math.isfinite(result):
        raise ValueError("NIS must be finite")
    if result < 0.0:
        tolerance = np.finfo(np.float64).eps * max(1.0, float(residual @ residual)) * 32.0
        if result < -tolerance:
            raise ValueError("NIS cannot be negative")
        return 0.0
    return result


def _channel(
    values: list[float],
    *,
    current_object_count: int,
    clean_mean: FloatArray,
    clean_maximum: FloatArray,
) -> NumericChannelEvidence:
    mature_count = len(values)
    mature_fraction = mature_count / current_object_count
    if mature_count < _MINIMUM_MATURE_OBJECT_COUNT:
        return NumericChannelEvidence(
            status="insufficient-support",
            mature_object_count=mature_count,
            current_object_count=current_object_count,
            mature_fraction=mature_fraction,
            mean_nis=None,
            maximum_nis=None,
            score=None,
        )
    mean_nis = float(np.mean(np.asarray(values, dtype=np.float64)))
    maximum_nis = max(values)
    score = max(
        ecdf_rank(clean_mean, mean_nis),
        ecdf_rank(clean_maximum, maximum_nis),
    )
    return NumericChannelEvidence(
        status="defined",
        mature_object_count=mature_count,
        current_object_count=current_object_count,
        mature_fraction=mature_fraction,
        mean_nis=mean_nis,
        maximum_nis=maximum_nis,
        score=score,
    )


class HealthScorer:
    """Causal feature engine with independent per-modality/object histories."""

    def __init__(self, calibration: HealthCalibration) -> None:
        self._calibration = calibration
        self._camera_history: dict[str, deque[_HistoryItem]] = {}
        self._lidar_history: dict[str, deque[_HistoryItem]] = {}
        self._camera_missing: deque[bool] = deque(maxlen=_MISSING_WINDOW_LENGTH)
        self._lidar_missing: deque[bool] = deque(maxlen=_MISSING_WINDOW_LENGTH)
        self._last_reference_time_s: float | None = None

    def process_frame(self, frame: HealthFrameInput) -> HealthFrameEvidence:
        """Score one frame from strict prior history, then commit current values."""

        if (
            self._last_reference_time_s is not None
            and frame.reference_time_s <= self._last_reference_time_s
        ):
            raise ValueError("frame reference times must be strictly increasing")

        camera_self_values: list[float] = []
        lidar_self_values: list[float] = []
        camera_from_lidar_values: list[float] = []
        lidar_from_camera_values: list[float] = []
        camera_timestamp_suspicious = False
        lidar_timestamp_suspicious = False

        for item in frame.objects:
            camera_prediction = _predict(
                self._camera_history.get(item.object_id, deque()),
                reference_time_s=frame.reference_time_s,
            )
            lidar_prediction = _predict(
                self._lidar_history.get(item.object_id, deque()),
                reference_time_s=frame.reference_time_s,
            )
            if item.camera is not None:
                camera_timestamp_suspicious |= (
                    abs(item.camera.reported_time_s - frame.reference_time_s)
                    > _TIMESTAMP_TOLERANCE_S
                )
                if camera_prediction is not None:
                    camera_self_values.append(
                        _nis(current=item.camera, prediction=camera_prediction)
                    )
                if lidar_prediction is not None:
                    camera_from_lidar_values.append(
                        _nis(current=item.camera, prediction=lidar_prediction)
                    )
            if item.lidar is not None:
                lidar_timestamp_suspicious |= (
                    abs(item.lidar.reported_time_s - frame.reference_time_s)
                    > _TIMESTAMP_TOLERANCE_S
                )
                if lidar_prediction is not None:
                    lidar_self_values.append(_nis(current=item.lidar, prediction=lidar_prediction))
                if camera_prediction is not None:
                    lidar_from_camera_values.append(
                        _nis(current=item.lidar, prediction=camera_prediction)
                    )

        calibration = self._calibration
        current_count = len(frame.objects)
        evidence = HealthFrameEvidence(
            reference_time_s=frame.reference_time_s,
            camera_available=frame.camera_available,
            lidar_available=frame.lidar_available,
            camera_timestamp_suspicious=camera_timestamp_suspicious,
            lidar_timestamp_suspicious=lidar_timestamp_suspicious,
            camera_missing_fraction_last_four=self._updated_missing_fraction(
                self._camera_missing,
                missing=not frame.camera_available,
            ),
            lidar_missing_fraction_last_four=self._updated_missing_fraction(
                self._lidar_missing,
                missing=not frame.lidar_available,
            ),
            camera_self=_channel(
                camera_self_values,
                current_object_count=current_count,
                clean_mean=calibration.camera_self_mean,
                clean_maximum=calibration.camera_self_maximum,
            ),
            lidar_self=_channel(
                lidar_self_values,
                current_object_count=current_count,
                clean_mean=calibration.lidar_self_mean,
                clean_maximum=calibration.lidar_self_maximum,
            ),
            camera_from_lidar_cross=_channel(
                camera_from_lidar_values,
                current_object_count=current_count,
                clean_mean=calibration.camera_from_lidar_cross_mean,
                clean_maximum=calibration.camera_from_lidar_cross_maximum,
            ),
            lidar_from_camera_cross=_channel(
                lidar_from_camera_values,
                current_object_count=current_count,
                clean_mean=calibration.lidar_from_camera_cross_mean,
                clean_maximum=calibration.lidar_from_camera_cross_maximum,
            ),
        )

        # All predictions above were formed before either current modality
        # updates, preserving both self- and cross-channel causality.
        for item in frame.objects:
            self._commit(
                self._camera_history,
                object_id=item.object_id,
                measurement=item.camera,
                reference_time_s=frame.reference_time_s,
            )
            self._commit(
                self._lidar_history,
                object_id=item.object_id,
                measurement=item.lidar,
                reference_time_s=frame.reference_time_s,
            )
        self._last_reference_time_s = frame.reference_time_s
        return evidence

    @staticmethod
    def _updated_missing_fraction(history: deque[bool], *, missing: bool) -> float:
        history.append(missing)
        return sum(history) / len(history)

    @staticmethod
    def _commit(
        histories: dict[str, deque[_HistoryItem]],
        *,
        object_id: str,
        measurement: ModalityMeasurement | None,
        reference_time_s: float,
    ) -> None:
        if measurement is None:
            return
        history = histories.setdefault(object_id, deque(maxlen=2))
        history.append(
            _HistoryItem(
                value_xy_m=measurement.value_xy_m,
                reported_covariance_xy_m2=measurement.reported_covariance_xy_m2,
                reference_time_s=reference_time_s,
            )
        )


def _required_defined(*channels: NumericChannelEvidence) -> bool:
    return all(channel.status == "defined" for channel in channels)


def _alarm(channel: NumericChannelEvidence, threshold: float) -> bool:
    if channel.status != "defined" or channel.score is None:
        raise ValueError("cannot alarm on undefined numeric evidence")
    return channel.score > threshold


def _two_sensor_label(camera_flag: bool, lidar_flag: bool) -> HealthLabel:
    if camera_flag and lidar_flag:
        return "ambiguous"
    if camera_flag:
        return "camera-fault"
    if lidar_flag:
        return "lidar-fault"
    return "healthy"


def decide_raw(
    *,
    method: HealthMethodId,
    evidence: HealthFrameEvidence,
    thresholds: HealthThresholds,
) -> RawHealthDecision:
    """Apply one of the four frozen raw observable decision rules."""

    if method not in {
        "self-nis-gate",
        "cross-nis-gate",
        "direct-telemetry-gate",
        "combined-health-gate",
    }:
        raise ValueError("unknown health method")
    direct_label = _two_sensor_label(
        evidence.camera_direct_suspicious,
        evidence.lidar_direct_suspicious,
    )
    if method == "direct-telemetry-gate":
        return RawHealthDecision(
            method=method,
            label=direct_label,
            evidence_status="update-eligible",
            camera_alarm=None,
            lidar_alarm=None,
            any_cross_alarm=None,
        )

    self_defined = _required_defined(evidence.camera_self, evidence.lidar_self)
    cross_defined = _required_defined(
        evidence.camera_from_lidar_cross,
        evidence.lidar_from_camera_cross,
    )
    if method == "combined-health-gate" and direct_label != "healthy":
        return RawHealthDecision(
            method=method,
            label=direct_label,
            evidence_status="update-eligible",
            camera_alarm=None,
            lidar_alarm=None,
            any_cross_alarm=None,
        )

    required_defined = (
        self_defined
        if method == "self-nis-gate"
        else cross_defined
        if method == "cross-nis-gate"
        else self_defined and cross_defined
    )
    if not required_defined:
        return RawHealthDecision(
            method=method,
            label="ambiguous",
            evidence_status="insufficient-support",
            camera_alarm=None,
            lidar_alarm=None,
            any_cross_alarm=None,
        )

    camera_alarm: bool | None = None
    lidar_alarm: bool | None = None
    any_cross_alarm: bool | None = None
    if method in {"self-nis-gate", "combined-health-gate"}:
        camera_alarm = _alarm(evidence.camera_self, thresholds.self_score)
        lidar_alarm = _alarm(evidence.lidar_self, thresholds.self_score)
        self_label = _two_sensor_label(camera_alarm, lidar_alarm)
        if method == "self-nis-gate" or self_label != "healthy":
            return RawHealthDecision(
                method=method,
                label=self_label,
                evidence_status="update-eligible",
                camera_alarm=camera_alarm,
                lidar_alarm=lidar_alarm,
                any_cross_alarm=None,
            )

    any_cross_alarm = _alarm(
        evidence.camera_from_lidar_cross,
        thresholds.cross_score,
    ) or _alarm(
        evidence.lidar_from_camera_cross,
        thresholds.cross_score,
    )
    return RawHealthDecision(
        method=method,
        label="ambiguous" if any_cross_alarm else "healthy",
        evidence_status="update-eligible",
        camera_alarm=camera_alarm,
        lidar_alarm=lidar_alarm,
        any_cross_alarm=any_cross_alarm,
    )


def advance_latch(
    state: HealthLatchState,
    decision: RawHealthDecision,
) -> HealthLatchState:
    """Apply the exact two-frame activation and three-frame recovery recurrence."""

    if decision.evidence_status == "insufficient-support":
        return state

    if state.label == "healthy":
        if decision.label == "healthy":
            return HealthLatchState()
        activation_count = (
            state.activation_count + 1 if state.activation_candidate == decision.label else 1
        )
        if activation_count == 2:
            return HealthLatchState(label=decision.label)
        return HealthLatchState(
            activation_candidate=decision.label,
            activation_count=1,
        )

    if decision.label != "healthy":
        return HealthLatchState(label=state.label)
    recovery_count = state.recovery_count + 1
    if recovery_count == 3:
        return HealthLatchState()
    return HealthLatchState(
        label=state.label,
        recovery_count=recovery_count,
    )


def choose_executed_action(
    *,
    camera_available: bool,
    lidar_available: bool,
    latched_label: HealthLabel,
    abstain_on_ambiguous: bool = False,
) -> ExecutedAction:
    """Map current availability and post-transition latch to an estimator action."""

    if latched_label not in {"healthy", *_NONHEALTHY_LABELS}:
        raise ValueError("unknown latched health label")
    if not camera_available and not lidar_available:
        return "undefined"
    if camera_available and not lidar_available:
        return "camera-only"
    if lidar_available and not camera_available:
        return "lidar-only"
    if latched_label == "camera-fault":
        return "lidar-only"
    if latched_label == "lidar-fault":
        return "camera-only"
    if latched_label == "ambiguous" and abstain_on_ambiguous:
        return "undefined"
    return "fixed-fusion"


class HealthPolicy:
    """Stateful latch/action layer over immutable precomputed frame evidence."""

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
        return self._state

    def apply(self, evidence: HealthFrameEvidence) -> HealthPolicyOutput:
        """Advance the recurrence once and choose the current executed action."""

        decision = decide_raw(
            method=self._method,
            evidence=evidence,
            thresholds=self._thresholds,
        )
        state = advance_latch(self._state, decision)
        self._state = state
        return HealthPolicyOutput(
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
