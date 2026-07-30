"""M5 complete-scene inference and preregistered persistence classification."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.contracts.replay_v1 import (
    M5_HEALTH_EXPERIMENT_IDS,
    M5_SCENE_NAMES,
)
from fusion_fault_bench.inference import paired_bootstrap_indices, percentile_interval

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type ExpectedDirection = Literal["positive", "negative"]
type PersistenceLabel = Literal[
    "robustly-persistent",
    "directionally-consistent",
    "non-persistent",
    "undefined",
]
type H5AssessmentRule = Literal[
    "persistence",
    "nonpositive-control",
    "nonattributable-diagnostic",
]
type ReplayContrastKind = Literal[
    "fixed-policy",
    "policy-target-drop",
    "policy-frame-oracle",
]

_REPLAY_BOOTSTRAP_SEED = 1_618_033
_REPLAY_BOOTSTRAP_REPLICATES = 2_000
_REPLAY_SCENE_COUNT = 10
_CONFIDENCE_LEVEL = 0.95
_REPLAY_SEQUENCE_IDS = frozenset(f"nuscenes:{name}" for name in M5_SCENE_NAMES)


@dataclass(frozen=True, slots=True)
class ReplayInterval:
    """Point estimate and complete-scene bootstrap interval support."""

    estimate: float | None
    lower: float | None
    upper: float | None
    defined_replicates: int
    bootstrap_replicates: int

    def __post_init__(self) -> None:
        if type(self.bootstrap_replicates) is not int or self.bootstrap_replicates <= 0:
            raise ValueError("bootstrap_replicates must be positive")
        if (
            type(self.defined_replicates) is not int
            or not 0 <= self.defined_replicates <= self.bootstrap_replicates
        ):
            raise ValueError("defined_replicates must lie within the bootstrap count")
        endpoints = (self.estimate, self.lower, self.upper)
        if all(value is None for value in endpoints):
            return
        if any(value is None for value in endpoints):
            raise ValueError("an interval must define the point and both endpoints together")
        if not _has_strict_conditional_support(
            self.defined_replicates,
            self.bootstrap_replicates,
        ):
            raise ValueError("a defined interval requires strictly more than 97.5% support")
        assert self.estimate is not None
        assert self.lower is not None
        assert self.upper is not None
        if not all(math.isfinite(value) for value in endpoints if value is not None):
            raise ValueError("replay interval values must be finite")
        if self.lower > self.upper:
            raise ValueError("replay interval lower endpoint cannot exceed its upper endpoint")

    @property
    def defined_fraction(self) -> float:
        """Return the fraction of bootstrap replicates with a denominator."""

        return self.defined_replicates / self.bootstrap_replicates


@dataclass(frozen=True, slots=True)
class LeaveOutEstimate:
    """One public opaque leave-one-cluster-out estimate."""

    cluster_id: str
    estimate: float | None

    def __post_init__(self) -> None:
        if not self.cluster_id:
            raise ValueError("cluster_id must be nonempty")
        if self.estimate is not None and not math.isfinite(self.estimate):
            raise ValueError("leave-out estimate must be finite when defined")


@dataclass(frozen=True, slots=True)
class SceneSignCounts:
    """Exact signs of the ten scene-level primary contrasts."""

    positive: int
    zero: int
    negative: int
    undefined: int = 0

    def __post_init__(self) -> None:
        if min(self.positive, self.zero, self.negative, self.undefined) < 0:
            raise ValueError("scene sign counts must be nonnegative")

    @property
    def total(self) -> int:
        """Return the number of scene slots represented."""

        return self.positive + self.zero + self.negative + self.undefined


@dataclass(frozen=True, slots=True)
class PersistenceAssessment:
    """Frozen M5 direction and cluster-sensitivity classification."""

    label: PersistenceLabel
    expected_direction: ExpectedDirection
    scene_signs: SceneSignCounts
    leave_one_scene_out: tuple[LeaveOutEstimate, ...]
    distinct_log_group_count: int
    leave_one_log_group_out: tuple[LeaveOutEstimate, ...]

    def __post_init__(self) -> None:
        if self.distinct_log_group_count < 1:
            raise ValueError("at least one log group is required")
        if len(self.leave_one_log_group_out) != self.distinct_log_group_count:
            raise ValueError("log-group count must match leave-one-group rows")


@dataclass(frozen=True, slots=True)
class H5Selector:
    """One exact H5-B hypothesis selector frozen before replay outcomes."""

    hypothesis_id: Literal["h5-b1", "h5-b2", "h5-b3", "h5-b4"]
    selector: str
    method: Literal["combined-health-gate"]
    metric_name: Literal["policy-gain-vs-fixed"]
    window: Literal["event"]
    unit: Literal["m^2"]
    assessment_rule: H5AssessmentRule
    expected_direction: ExpectedDirection | None


def _require_digest(value: str, *, field_name: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_identifier(value: str, *, field_name: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value) is None:
        raise ValueError(f"{field_name} must be a safe replay identifier")


def _require_condition_selector(value: str, *, condition_id: str) -> None:
    pattern = (
        r"replay-[a-z0-9][a-z0-9-]*:"
        r"(?:0|[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))"
    )
    if len(value) > 128 or re.fullmatch(pattern, value) is None:
        raise ValueError("condition_selector must be a safe exact replay selector")
    if not value.startswith(f"{condition_id}:"):
        raise ValueError("condition_selector must bind the base condition_id")


def _require_support_pair(
    *,
    count: int,
    left_sum: float,
    right_sum: float,
    label: str,
) -> None:
    if type(count) is not int or count < 0:
        raise ValueError(f"{label} common support count must be a nonnegative integer")
    for value in (left_sum, right_sum):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{label} loss sums must be finite and nonnegative")
    if count == 0 and (float(left_sum) != 0.0 or float(right_sum) != 0.0):
        raise ValueError(f"{label} zero common support requires zero loss sums")


@dataclass(frozen=True, slots=True)
class ReplayHealthSequenceContrast:
    """Paired common-support sufficient statistics for one replay scene."""

    replay_experiment_identity_sha256: str
    sequence_id: str
    condition_id: str
    condition_selector: str
    policy: Literal[
        "self-nis-gate",
        "cross-nis-gate",
        "direct-telemetry-gate",
        "combined-health-gate",
        "combined-health-gate-abstain",
    ]
    window: Literal["score", "event", "recovery"]
    fixed_support_sha256: str
    policy_support_sha256: str
    fixed_policy_common_count: int
    fixed_on_common_loss_sum_m2: float
    policy_on_fixed_common_loss_sum_m2: float
    target_drop_applicable: bool
    policy_target_drop_common_count: int | None
    policy_on_target_common_loss_sum_m2: float | None
    target_drop_on_common_loss_sum_m2: float | None
    target_drop_support_sha256: str | None
    frame_oracle_applicable: bool
    policy_frame_oracle_common_count: int | None
    policy_on_oracle_common_loss_sum_m2: float | None
    frame_oracle_on_common_loss_sum_m2: float | None
    frame_oracle_support_sha256: str | None

    def __post_init__(self) -> None:
        if self.policy not in {
            "self-nis-gate",
            "cross-nis-gate",
            "direct-telemetry-gate",
            "combined-health-gate",
            "combined-health-gate-abstain",
        }:
            raise ValueError("unknown replay health policy")
        if self.window not in {"score", "event", "recovery"}:
            raise ValueError("unknown replay health window")
        if type(self.target_drop_applicable) is not bool:
            raise ValueError("target_drop_applicable must be a boolean")
        if type(self.frame_oracle_applicable) is not bool:
            raise ValueError("frame_oracle_applicable must be a boolean")
        for field_name in (
            "replay_experiment_identity_sha256",
            "fixed_support_sha256",
            "policy_support_sha256",
        ):
            _require_digest(getattr(self, field_name), field_name=field_name)
        for field_name in ("sequence_id", "condition_id"):
            _require_identifier(getattr(self, field_name), field_name=field_name)
        if self.sequence_id not in _REPLAY_SEQUENCE_IDS:
            raise ValueError("replay contrast lies outside the frozen scene population")
        if self.condition_id not in M5_HEALTH_EXPERIMENT_IDS:
            raise ValueError("replay contrast condition is not in the frozen M5-B matrix")
        _require_condition_selector(
            self.condition_selector,
            condition_id=self.condition_id,
        )
        _require_support_pair(
            count=self.fixed_policy_common_count,
            left_sum=self.fixed_on_common_loss_sum_m2,
            right_sum=self.policy_on_fixed_common_loss_sum_m2,
            label="fixed-policy",
        )

        target_values = (
            self.policy_target_drop_common_count,
            self.policy_on_target_common_loss_sum_m2,
            self.target_drop_on_common_loss_sum_m2,
            self.target_drop_support_sha256,
        )
        if self.target_drop_applicable != all(value is not None for value in target_values):
            raise ValueError(
                "target-drop applicability must match every paired sufficient statistic"
            )
        if not self.target_drop_applicable and any(value is not None for value in target_values):
            raise ValueError("inapplicable target-drop statistics must all be null")
        if self.target_drop_applicable:
            assert self.policy_target_drop_common_count is not None
            assert self.policy_on_target_common_loss_sum_m2 is not None
            assert self.target_drop_on_common_loss_sum_m2 is not None
            assert self.target_drop_support_sha256 is not None
            _require_digest(
                self.target_drop_support_sha256,
                field_name="target_drop_support_sha256",
            )
            _require_support_pair(
                count=self.policy_target_drop_common_count,
                left_sum=self.policy_on_target_common_loss_sum_m2,
                right_sum=self.target_drop_on_common_loss_sum_m2,
                label="policy-target-drop",
            )

        oracle_values = (
            self.policy_frame_oracle_common_count,
            self.policy_on_oracle_common_loss_sum_m2,
            self.frame_oracle_on_common_loss_sum_m2,
            self.frame_oracle_support_sha256,
        )
        if self.frame_oracle_applicable != all(value is not None for value in oracle_values):
            raise ValueError(
                "frame-oracle applicability must match every paired sufficient statistic"
            )
        if not self.frame_oracle_applicable and any(value is not None for value in oracle_values):
            raise ValueError("inapplicable frame-oracle statistics must all be null")
        if self.frame_oracle_applicable:
            assert self.policy_frame_oracle_common_count is not None
            assert self.policy_on_oracle_common_loss_sum_m2 is not None
            assert self.frame_oracle_on_common_loss_sum_m2 is not None
            assert self.frame_oracle_support_sha256 is not None
            _require_digest(
                self.frame_oracle_support_sha256,
                field_name="frame_oracle_support_sha256",
            )
            _require_support_pair(
                count=self.policy_frame_oracle_common_count,
                left_sum=self.policy_on_oracle_common_loss_sum_m2,
                right_sum=self.frame_oracle_on_common_loss_sum_m2,
                label="policy-frame-oracle",
            )

        if self.identical_support_recovery_applicable and (
            self.fixed_policy_common_count != self.policy_frame_oracle_common_count
            or self.policy_on_fixed_common_loss_sum_m2 != self.policy_on_oracle_common_loss_sum_m2
        ):
            raise ValueError("recovery applicability requires identical nonempty three-way support")

    @property
    def identical_support_recovery_applicable(self) -> bool:
        """Whether fixed, policy, and oracle bind one nonempty support."""

        return (
            self.frame_oracle_applicable
            and self.fixed_policy_common_count > 0
            and self.fixed_support_sha256
            == self.policy_support_sha256
            == self.frame_oracle_support_sha256
        )


H5_B_SELECTORS: tuple[H5Selector, ...] = (
    H5Selector(
        hypothesis_id="h5-b1",
        selector="replay-lidar-output-y-bias:+3",
        method="combined-health-gate",
        metric_name="policy-gain-vs-fixed",
        window="event",
        unit="m^2",
        assessment_rule="persistence",
        expected_direction="positive",
    ),
    H5Selector(
        hypothesis_id="h5-b1",
        selector="replay-lidar-timestamp-offset:+0.6",
        method="combined-health-gate",
        metric_name="policy-gain-vs-fixed",
        window="event",
        unit="m^2",
        assessment_rule="persistence",
        expected_direction="positive",
    ),
    H5Selector(
        hypothesis_id="h5-b1",
        selector="replay-camera-noise-underreported:3",
        method="combined-health-gate",
        metric_name="policy-gain-vs-fixed",
        window="event",
        unit="m^2",
        assessment_rule="persistence",
        expected_direction="positive",
    ),
    H5Selector(
        hypothesis_id="h5-b1",
        selector="replay-camera-timestamp-offset:+0.6",
        method="combined-health-gate",
        metric_name="policy-gain-vs-fixed",
        window="event",
        unit="m^2",
        assessment_rule="persistence",
        expected_direction="positive",
    ),
    H5Selector(
        hypothesis_id="h5-b1",
        selector="replay-camera-calibration-x:+3",
        method="combined-health-gate",
        metric_name="policy-gain-vs-fixed",
        window="event",
        unit="m^2",
        assessment_rule="persistence",
        expected_direction="positive",
    ),
    H5Selector(
        hypothesis_id="h5-b1",
        selector="replay-camera-output-y-bias:+3",
        method="combined-health-gate",
        metric_name="policy-gain-vs-fixed",
        window="event",
        unit="m^2",
        assessment_rule="persistence",
        expected_direction="positive",
    ),
    H5Selector(
        hypothesis_id="h5-b1",
        selector="replay-camera-calibration-yaw:+0.06",
        method="combined-health-gate",
        metric_name="policy-gain-vs-fixed",
        window="event",
        unit="m^2",
        assessment_rule="persistence",
        expected_direction="positive",
    ),
    H5Selector(
        hypothesis_id="h5-b2",
        selector="replay-lidar-noise-underreported:3",
        method="combined-health-gate",
        metric_name="policy-gain-vs-fixed",
        window="event",
        unit="m^2",
        assessment_rule="persistence",
        expected_direction="negative",
    ),
    H5Selector(
        hypothesis_id="h5-b3",
        selector="replay-camera-noise-correctly-reported:3",
        method="combined-health-gate",
        metric_name="policy-gain-vs-fixed",
        window="event",
        unit="m^2",
        assessment_rule="nonpositive-control",
        expected_direction=None,
    ),
    H5Selector(
        hypothesis_id="h5-b3",
        selector="replay-lidar-noise-correctly-reported:3",
        method="combined-health-gate",
        metric_name="policy-gain-vs-fixed",
        window="event",
        unit="m^2",
        assessment_rule="nonpositive-control",
        expected_direction=None,
    ),
    H5Selector(
        hypothesis_id="h5-b4",
        selector="replay-common-mode-x:+4",
        method="combined-health-gate",
        metric_name="policy-gain-vs-fixed",
        window="event",
        unit="m^2",
        assessment_rule="nonattributable-diagnostic",
        expected_direction=None,
    ),
)


def replay_bootstrap_indices(
    *,
    seed: int = _REPLAY_BOOTSTRAP_SEED,
    replicates: int = _REPLAY_BOOTSTRAP_REPLICATES,
    scene_count: int = _REPLAY_SCENE_COUNT,
) -> IntArray:
    """Draw complete-scene indices with the frozen PCG64DXSM defaults."""

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(replicates) is not int or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    if type(scene_count) is not int or scene_count <= 0:
        raise ValueError("scene_count must be a positive integer")
    return paired_bootstrap_indices(
        seed=seed,
        replicates=replicates,
        sequence_count=scene_count,
    )


def _finite_vector(value: npt.ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def _count_vector(
    value: npt.ArrayLike,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> IntArray:
    raw = np.asarray(value)
    if raw.shape != expected_shape or raw.dtype.kind not in {"i", "u"} or np.any(raw < 0):
        raise ValueError(f"{name} must be an aligned nonnegative integer vector")
    return np.asarray(raw, dtype=np.int64)


def _indices(value: npt.ArrayLike, *, scene_count: int) -> IntArray:
    raw = np.asarray(value)
    if (
        raw.ndim != 2
        or raw.shape[0] == 0
        or raw.shape[1] != scene_count
        or raw.dtype.kind not in {"i", "u"}
    ):
        raise ValueError("bootstrap indices must be a nonempty aligned integer matrix")
    result = np.asarray(raw, dtype=np.int64)
    if np.any(result < 0) or np.any(result >= scene_count):
        raise ValueError("bootstrap index lies outside the scene population")
    return result


def _undefined_interval(
    *,
    defined_replicates: int,
    bootstrap_replicates: int,
) -> ReplayInterval:
    return ReplayInterval(
        estimate=None,
        lower=None,
        upper=None,
        defined_replicates=defined_replicates,
        bootstrap_replicates=bootstrap_replicates,
    )


def _complete_interval(
    values: FloatArray,
    bootstrap_indices: IntArray,
) -> ReplayInterval:
    replicates = values[bootstrap_indices].mean(axis=1)
    lower, upper = percentile_interval(
        replicates,
        confidence_level=_CONFIDENCE_LEVEL,
    )
    return ReplayInterval(
        estimate=math.fsum(float(value) for value in values) / values.size,
        lower=lower,
        upper=upper,
        defined_replicates=int(replicates.size),
        bootstrap_replicates=int(replicates.size),
    )


def _has_strict_conditional_support(defined: int, total: int) -> bool:
    """Evaluate ``defined / total > 0.975`` without floating-point drift."""

    return defined * 40 > total * 39


def equal_scene_loss_interval(
    loss_sums: npt.ArrayLike,
    valid_counts: npt.ArrayLike,
    eligible_counts: npt.ArrayLike,
    bootstrap_indices: npt.ArrayLike,
) -> ReplayInterval:
    """Average full-support scene loss without treating missing output as zero."""

    losses = _finite_vector(loss_sums, name="loss_sums")
    if np.any(losses < 0.0):
        raise ValueError("loss_sums must be nonnegative")
    valid = _count_vector(
        valid_counts,
        name="valid_counts",
        expected_shape=losses.shape,
    )
    eligible = _count_vector(
        eligible_counts,
        name="eligible_counts",
        expected_shape=losses.shape,
    )
    indices = _indices(bootstrap_indices, scene_count=losses.size)
    if np.any(valid > eligible):
        raise ValueError("valid support cannot exceed eligible support")
    if np.any((valid == 0) & (losses != 0.0)):
        raise ValueError("zero valid support requires zero accumulated loss")
    if np.any(eligible == 0) or np.any(valid != eligible):
        return _undefined_interval(
            defined_replicates=0,
            bootstrap_replicates=indices.shape[0],
        )
    scene_losses = np.asarray(losses / eligible, dtype=np.float64)
    return _complete_interval(scene_losses, indices)


def equal_scene_value_interval(
    values: npt.ArrayLike,
    bootstrap_indices: npt.ArrayLike,
) -> ReplayInterval:
    """Average one finite signed value per complete replay scene."""

    checked = _finite_vector(values, name="values")
    indices = _indices(bootstrap_indices, scene_count=checked.size)
    return _complete_interval(checked, indices)


def equal_scene_ratio_interval(
    numerators: npt.ArrayLike,
    denominators: npt.ArrayLike,
    bootstrap_indices: npt.ArrayLike,
) -> ReplayInterval:
    """Average scene-level ratios only when every scene denominator is nonzero."""

    raw_numerators = np.asarray(numerators)
    if raw_numerators.ndim != 1 or raw_numerators.size == 0:
        raise ValueError("numerators must be a nonempty integer vector")
    numerator = _count_vector(
        raw_numerators,
        name="numerators",
        expected_shape=raw_numerators.shape,
    )
    denominator = _count_vector(
        denominators,
        name="denominators",
        expected_shape=numerator.shape,
    )
    indices = _indices(bootstrap_indices, scene_count=numerator.size)
    if np.any(numerator > denominator):
        raise ValueError("ratio numerator cannot exceed its denominator")
    if np.any(denominator == 0):
        return _undefined_interval(
            defined_replicates=0,
            bootstrap_replicates=indices.shape[0],
        )
    scene_ratios = np.asarray(numerator / denominator, dtype=np.float64)
    return _complete_interval(scene_ratios, indices)


def equal_scene_contrast_interval(
    rows: Sequence[ReplayHealthSequenceContrast],
    bootstrap_indices: npt.ArrayLike,
    *,
    contrast: ReplayContrastKind = "fixed-policy",
) -> ReplayInterval:
    """Infer an all-scene contrast only from paired common-support rows."""

    values = replay_sequence_contrast_values(rows, contrast=contrast)
    indices = _indices(bootstrap_indices, scene_count=len(values))
    if any(value is None for value in values):
        return _undefined_interval(
            defined_replicates=0,
            bootstrap_replicates=indices.shape[0],
        )
    return _complete_interval(np.asarray(values, dtype=np.float64), indices)


def replay_sequence_contrast_values(
    rows: Sequence[ReplayHealthSequenceContrast],
    *,
    contrast: ReplayContrastKind = "fixed-policy",
) -> tuple[float | None, ...]:
    """Return scene contrasts reconstructed on each row's paired support."""

    checked = tuple(rows)
    if not checked:
        raise ValueError("replay contrast rows must be nonempty")
    cohort_fields = (
        "replay_experiment_identity_sha256",
        "condition_id",
        "condition_selector",
        "policy",
        "window",
    )
    for field_name in cohort_fields:
        if len({getattr(row, field_name) for row in checked}) != 1:
            raise ValueError(f"replay contrast rows disagree on {field_name}")
    sequence_ids = tuple(row.sequence_id for row in checked)
    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("replay contrast rows contain a duplicate scene")

    values: list[float | None] = []
    for row in checked:
        if contrast == "fixed-policy":
            count = row.fixed_policy_common_count
            left = row.fixed_on_common_loss_sum_m2
            right = row.policy_on_fixed_common_loss_sum_m2
        elif contrast == "policy-target-drop":
            if not row.target_drop_applicable:
                raise ValueError("target-drop contrast is inapplicable")
            assert row.policy_target_drop_common_count is not None
            assert row.policy_on_target_common_loss_sum_m2 is not None
            assert row.target_drop_on_common_loss_sum_m2 is not None
            count = row.policy_target_drop_common_count
            left = row.policy_on_target_common_loss_sum_m2
            right = row.target_drop_on_common_loss_sum_m2
        elif contrast == "policy-frame-oracle":
            if not row.frame_oracle_applicable:
                raise ValueError("frame-oracle contrast is inapplicable")
            assert row.policy_frame_oracle_common_count is not None
            assert row.policy_on_oracle_common_loss_sum_m2 is not None
            assert row.frame_oracle_on_common_loss_sum_m2 is not None
            count = row.policy_frame_oracle_common_count
            left = row.policy_on_oracle_common_loss_sum_m2
            right = row.frame_oracle_on_common_loss_sum_m2
        else:
            raise ValueError("unknown replay sequence contrast")
        values.append(None if count == 0 else (left - right) / count)
    return tuple(values)


def _pooled_ratio_interval(
    numerator: FloatArray,
    denominator: IntArray,
    bootstrap_indices: IntArray,
) -> ReplayInterval:
    replicate_numerator = numerator[bootstrap_indices].sum(axis=1, dtype=np.float64)
    replicate_denominator = denominator[bootstrap_indices].sum(axis=1, dtype=np.int64)
    defined = replicate_denominator > 0
    replicates = np.asarray(
        replicate_numerator[defined] / replicate_denominator[defined],
        dtype=np.float64,
    )
    defined_count = int(replicates.size)
    replicate_count = int(bootstrap_indices.shape[0])
    point_denominator = int(denominator.sum(dtype=np.int64))
    if point_denominator == 0 or not _has_strict_conditional_support(
        defined_count,
        replicate_count,
    ):
        return _undefined_interval(
            defined_replicates=defined_count,
            bootstrap_replicates=replicate_count,
        )
    lower, upper = percentile_interval(
        replicates,
        confidence_level=_CONFIDENCE_LEVEL,
    )
    return ReplayInterval(
        estimate=math.fsum(float(value) for value in numerator) / point_denominator,
        lower=lower,
        upper=upper,
        defined_replicates=defined_count,
        bootstrap_replicates=replicate_count,
    )


def pooled_availability_interval(
    valid_counts: npt.ArrayLike,
    eligible_counts: npt.ArrayLike,
    bootstrap_indices: npt.ArrayLike,
) -> ReplayInterval:
    """Reconstruct pooled availability coverage inside each scene bootstrap."""

    raw_valid = np.asarray(valid_counts)
    if raw_valid.ndim != 1 or raw_valid.size == 0:
        raise ValueError("valid_counts must be a nonempty integer vector")
    valid = _count_vector(
        raw_valid,
        name="valid_counts",
        expected_shape=raw_valid.shape,
    )
    eligible = _count_vector(
        eligible_counts,
        name="eligible_counts",
        expected_shape=valid.shape,
    )
    if np.any(valid > eligible):
        raise ValueError("valid availability count cannot exceed eligible count")
    indices = _indices(bootstrap_indices, scene_count=valid.size)
    return _pooled_ratio_interval(
        np.asarray(valid, dtype=np.float64),
        eligible,
        indices,
    )


def pooled_conditional_loss_interval(
    loss_sums: npt.ArrayLike,
    valid_counts: npt.ArrayLike,
    bootstrap_indices: npt.ArrayLike,
) -> ReplayInterval:
    """Reconstruct pooled conditional loss and exclude zero-valid replicates."""

    losses = _finite_vector(loss_sums, name="loss_sums")
    if np.any(losses < 0.0):
        raise ValueError("loss_sums must be nonnegative")
    valid = _count_vector(
        valid_counts,
        name="valid_counts",
        expected_shape=losses.shape,
    )
    if np.any((valid == 0) & (losses != 0.0)):
        raise ValueError("zero valid support requires zero accumulated loss")
    indices = _indices(bootstrap_indices, scene_count=losses.size)
    return _pooled_ratio_interval(losses, valid, indices)


def conditional_observed_mean_interval(
    values: Sequence[float | None],
    bootstrap_indices: npt.ArrayLike,
) -> ReplayInterval:
    """Infer a conditional scene mean without imputing censored observations."""

    if len(values) == 0:
        raise ValueError("conditional values must be nonempty")
    sums = np.zeros(len(values), dtype=np.float64)
    counts = np.zeros(len(values), dtype=np.int64)
    for index, value in enumerate(values):
        if value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("conditional values must be finite when observed")
        sums[index] = numeric
        counts[index] = 1
    indices = _indices(bootstrap_indices, scene_count=len(values))
    return _pooled_ratio_interval(sums, counts, indices)


def observed_fraction_interval(
    values: Sequence[float | None],
    bootstrap_indices: npt.ArrayLike,
) -> ReplayInterval:
    """Publish observed-latency support before its conditional mean."""

    if len(values) == 0:
        raise ValueError("conditional values must be nonempty")
    checked = _optional_values(values)
    observed = np.asarray(
        [value is not None for value in checked],
        dtype=np.float64,
    )
    indices = _indices(bootstrap_indices, scene_count=len(checked))
    return _complete_interval(observed, indices)


def _optional_values(values: Sequence[float | None]) -> tuple[float | None, ...]:
    if len(values) == 0:
        raise ValueError("scene values must be nonempty")
    checked: list[float | None] = []
    for value in values:
        if value is None:
            checked.append(None)
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("scene values must be finite when defined")
        checked.append(numeric)
    return tuple(checked)


def _complete_optional_mean(values: Sequence[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return math.fsum(float(value) for value in values if value is not None) / len(values)


def leave_one_scene_out(
    values: Sequence[float | None],
) -> tuple[LeaveOutEstimate, ...]:
    """Compute every leave-one-scene-out estimate with opaque public ordinals."""

    checked = _optional_values(values)
    return tuple(
        LeaveOutEstimate(
            cluster_id=f"scene-ordinal:{omitted:02d}",
            estimate=_complete_optional_mean(
                checked[:omitted] + checked[omitted + 1 :],
            ),
        )
        for omitted in range(len(checked))
    )


def leave_one_log_group_out(
    values: Sequence[float | None],
    log_groups: Sequence[str],
) -> tuple[LeaveOutEstimate, ...]:
    """Omit complete private log groups while returning only opaque ordinals."""

    checked = _optional_values(values)
    if len(log_groups) != len(checked):
        raise ValueError("log group labels must align with scene values")
    if any(not group for group in log_groups):
        raise ValueError("log group labels must be nonempty strings")
    ordered_groups = tuple(
        sorted(
            set(log_groups),
            key=lambda value: value.encode("utf-8"),
        )
    )
    return tuple(
        LeaveOutEstimate(
            cluster_id=f"log-group:{ordinal:02d}",
            estimate=_complete_optional_mean(
                tuple(
                    value
                    for value, group in zip(checked, log_groups, strict=True)
                    if group != omitted_group
                )
            ),
        )
        for ordinal, omitted_group in enumerate(ordered_groups)
    )


def scene_sign_counts(values: Sequence[float | None]) -> SceneSignCounts:
    """Count exact positive, zero, negative, and undefined scene values."""

    checked = _optional_values(values)
    defined = tuple(value for value in checked if value is not None)
    return SceneSignCounts(
        positive=sum(value > 0.0 for value in defined),
        zero=sum(value == 0.0 for value in defined),
        negative=sum(value < 0.0 for value in defined),
        undefined=len(checked) - len(defined),
    )


def _matches_direction(value: float | None, expected: ExpectedDirection) -> bool:
    if value is None:
        return False
    return value > 0.0 if expected == "positive" else value < 0.0


def classify_persistence(
    interval: ReplayInterval,
    scene_values: Sequence[float | None],
    log_groups: Sequence[str],
    expected_direction: ExpectedDirection,
) -> PersistenceAssessment:
    """Apply the exact M5 robust/directional/nonpersistent/undefined rules."""

    checked = _optional_values(scene_values)
    if len(checked) != _REPLAY_SCENE_COUNT:
        raise ValueError("M5 persistence classification requires exactly ten scenes")
    if len(log_groups) != len(checked):
        raise ValueError("log group labels must align with scene values")
    if expected_direction not in {"positive", "negative"}:
        raise ValueError("expected_direction must be positive or negative")
    loso = leave_one_scene_out(checked)
    lolo = leave_one_log_group_out(checked, log_groups)
    signs = scene_sign_counts(checked)
    distinct_group_count = len(lolo)

    if (
        interval.estimate is None
        or interval.lower is None
        or interval.upper is None
        or signs.undefined > 0
    ):
        label: PersistenceLabel = "undefined"
    elif not _matches_direction(interval.estimate, expected_direction):
        label = "non-persistent"
    else:
        interval_matches = (
            interval.lower > 0.0 if expected_direction == "positive" else interval.upper < 0.0
        )
        matching_scene_count = (
            signs.positive if expected_direction == "positive" else signs.negative
        )
        leave_out_matches = all(
            _matches_direction(row.estimate, expected_direction) for row in (*loso, *lolo)
        )
        robust = (
            interval_matches
            and matching_scene_count >= 8
            and leave_out_matches
            and distinct_group_count >= 2
        )
        label = "robustly-persistent" if robust else "directionally-consistent"

    return PersistenceAssessment(
        label=label,
        expected_direction=expected_direction,
        scene_signs=signs,
        leave_one_scene_out=loso,
        distinct_log_group_count=distinct_group_count,
        leave_one_log_group_out=lolo,
    )


def supports_nonpositive_control(interval: ReplayInterval) -> bool | None:
    """Evaluate the exact H5-B3 point-and-upper-bound nonpositive rule."""

    if interval.estimate is None or interval.upper is None:
        return None
    return interval.estimate <= 0.0 and interval.upper <= 0.0
