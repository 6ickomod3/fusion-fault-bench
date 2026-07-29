"""M4 sequence-first population aggregation with one paired bootstrap matrix."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.contracts.health_result_v1 import (
    HealthAggregateMetricV1,
    HealthMethod,
    HealthPolicyMethod,
    HealthSequenceContrastV1,
    HealthSequenceEventV1,
    HealthSequenceLossV1,
    HealthWindow,
)
from fusion_fault_bench.health_inference import (
    HealthInterval,
    conditional_mean_interval,
    recovery_fraction_interval,
    sequence_mean_interval,
)
from fusion_fault_bench.inference import (
    bootstrap_count_ratio,
    paired_bootstrap_indices,
    percentile_interval,
)
from fusion_fault_bench.scenarios.health import HealthFaultSpec

type IntArray = npt.NDArray[np.int64]

_BOOTSTRAP_SEED = 314159
_BOOTSTRAP_REPLICATES = 2_000
_POLICY_METHODS: tuple[HealthPolicyMethod, ...] = (
    "self-nis-gate",
    "cross-nis-gate",
    "direct-telemetry-gate",
    "combined-health-gate",
    "combined-health-gate-abstain",
)
_EVENT_POLICIES = _POLICY_METHODS[:-1]


def _record(
    *,
    condition_id: str,
    method: HealthMethod | None,
    metric_name: str,
    window: HealthWindow | None,
    unit: str,
    interval: HealthInterval,
    sequence_count: int,
) -> HealthAggregateMetricV1:
    status = "ok" if interval.estimate is not None else "undefined"
    return HealthAggregateMetricV1.model_validate(
        {
            "schema": "ffb.health-aggregate/v1",
            "condition_id": condition_id,
            "method": method,
            "metric_name": metric_name,
            "window": window,
            "unit": unit,
            "status": status,
            "estimate": interval.estimate,
            "interval_lower": interval.lower,
            "interval_upper": interval.upper,
            "sequence_count": sequence_count,
            "bootstrap_replicates": _BOOTSTRAP_REPLICATES,
            "defined_bootstrap_replicates": interval.defined_replicates,
        }
    )


def _fraction_interval(
    numerators: npt.ArrayLike,
    denominators: npt.ArrayLike,
    bootstrap_indices: IntArray,
) -> HealthInterval:
    raw_numerators = np.asarray(numerators)
    raw_denominators = np.asarray(denominators)
    if (
        raw_numerators.shape != raw_denominators.shape
        or raw_numerators.ndim != 1
        or raw_numerators.size == 0
        or raw_numerators.dtype.kind not in {"i", "u"}
        or raw_denominators.dtype.kind not in {"i", "u"}
    ):
        raise ValueError("fraction counts must be aligned nonempty integer vectors")
    numerator = np.asarray(raw_numerators, dtype=np.int64)
    denominator = np.asarray(raw_denominators, dtype=np.int64)
    if np.any(numerator < 0) or np.any(denominator <= 0) or np.any(numerator > denominator):
        raise ValueError("fraction counts are outside their denominators")
    replicates = bootstrap_count_ratio(numerator, denominator, bootstrap_indices)
    lower, upper = percentile_interval(replicates, confidence_level=0.95)
    return HealthInterval(
        estimate=int(numerator.sum(dtype=np.int64)) / int(denominator.sum(dtype=np.int64)),
        lower=lower,
        upper=upper,
        defined_replicates=int(replicates.size),
    )


def _conditional_scalar_interval(
    values: Sequence[float | int | None],
    bootstrap_indices: IntArray,
) -> HealthInterval:
    sums = np.asarray([0.0 if value is None else float(value) for value in values])
    counts = np.asarray([0 if value is None else 1 for value in values], dtype=np.int64)
    return conditional_mean_interval(sums, counts, bootstrap_indices)


def _loss_record_index(
    records: Sequence[HealthSequenceLossV1],
    *,
    sequence_ids: tuple[str, ...],
    condition_id: str,
) -> dict[tuple[str, HealthMethod, HealthWindow], HealthSequenceLossV1]:
    index: dict[tuple[str, HealthMethod, HealthWindow], HealthSequenceLossV1] = {}
    for record in records:
        if record.condition_id != condition_id or record.sequence_id not in sequence_ids:
            raise ValueError("sequence loss row lies outside the condition population")
        key = (record.sequence_id, record.method, record.window)
        if key in index:
            raise ValueError("duplicate M4 sequence loss row")
        index[key] = record
    return index


def _aggregate_method_losses(
    *,
    condition_id: str,
    sequence_ids: tuple[str, ...],
    index: dict[tuple[str, HealthMethod, HealthWindow], HealthSequenceLossV1],
    bootstrap_indices: IntArray,
) -> list[HealthAggregateMetricV1]:
    method_window_list: list[tuple[HealthMethod, HealthWindow]] = []
    for _, method_id, window_id in index:
        pair = (method_id, window_id)
        if pair not in method_window_list:
            method_window_list.append(pair)
    method_windows = tuple(method_window_list)
    aggregates: list[HealthAggregateMetricV1] = []
    for method, window in method_windows:
        rows = tuple(index[(sequence_id, method, window)] for sequence_id in sequence_ids)
        sequence_losses = tuple(
            (
                row.loss_sum_m2 / row.valid_object_frame_count
                if row.valid_object_frame_count
                else None
            )
            for row in rows
        )
        aggregates.append(
            _record(
                condition_id=condition_id,
                method=method,
                metric_name="matched-center-mse",
                window=window,
                unit="m^2",
                interval=_conditional_scalar_interval(
                    sequence_losses,
                    bootstrap_indices,
                ),
                sequence_count=len(sequence_ids),
            )
        )
        valid = np.asarray(
            [row.valid_object_frame_count for row in rows],
            dtype=np.int64,
        )
        eligible = np.asarray(
            [row.eligible_object_frame_count for row in rows],
            dtype=np.int64,
        )
        coverage = _fraction_interval(valid, eligible, bootstrap_indices)
        aggregates.append(
            _record(
                condition_id=condition_id,
                method=method,
                metric_name="coverage",
                window=window,
                unit="fraction",
                interval=coverage,
                sequence_count=len(sequence_ids),
            )
        )
        aggregates.append(
            _record(
                condition_id=condition_id,
                method=method,
                metric_name="undefined-output-rate",
                window=window,
                unit="fraction",
                interval=_fraction_interval(eligible - valid, eligible, bootstrap_indices),
                sequence_count=len(sequence_ids),
            )
        )
    return aggregates


def _contrast_record_index(
    records: Sequence[HealthSequenceContrastV1],
    *,
    sequence_ids: tuple[str, ...],
    condition_id: str,
) -> dict[tuple[str, HealthPolicyMethod, HealthWindow], HealthSequenceContrastV1]:
    index: dict[
        tuple[str, HealthPolicyMethod, HealthWindow],
        HealthSequenceContrastV1,
    ] = {}
    for record in records:
        if record.condition_id != condition_id or record.sequence_id not in sequence_ids:
            raise ValueError("sequence contrast row lies outside the condition population")
        key = (record.sequence_id, record.policy, record.window)
        if key in index:
            raise ValueError("duplicate M4 sequence contrast row")
        index[key] = record
    expected = {
        (sequence_id, policy, window)
        for sequence_id in sequence_ids
        for policy in _POLICY_METHODS
        for window in ("score", "event", "recovery")
    }
    if set(index) != expected:
        raise ValueError("condition sequence contrast rows are incomplete")
    return index


def _contrast_value(
    row: HealthSequenceContrastV1,
    *,
    contrast: str,
) -> float | None:
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
        raise ValueError("unknown M4 sequence contrast")
    return None if count == 0 else (left - right) / count


def _contrast_interval(
    rows: tuple[HealthSequenceContrastV1, ...],
    *,
    contrast: str,
    bootstrap_indices: IntArray,
) -> HealthInterval:
    return _conditional_scalar_interval(
        tuple(_contrast_value(row, contrast=contrast) for row in rows),
        bootstrap_indices,
    )


def _aggregate_policy_contrasts(
    *,
    condition_id: str,
    sequence_ids: tuple[str, ...],
    index: dict[
        tuple[str, HealthPolicyMethod, HealthWindow],
        HealthSequenceContrastV1,
    ],
    bootstrap_indices: IntArray,
) -> list[HealthAggregateMetricV1]:
    aggregates: list[HealthAggregateMetricV1] = []
    for policy in _POLICY_METHODS:
        for window_name in ("score", "event", "recovery"):
            rows = tuple(index[(sequence_id, policy, window_name)] for sequence_id in sequence_ids)
            aggregates.append(
                _record(
                    condition_id=condition_id,
                    method=policy,
                    metric_name="policy-gain-vs-fixed",
                    window=window_name,
                    unit="m^2",
                    interval=_contrast_interval(
                        rows,
                        contrast="fixed-policy",
                        bootstrap_indices=bootstrap_indices,
                    ),
                    sequence_count=len(sequence_ids),
                )
            )
            target_applicability = {row.target_drop_applicable for row in rows}
            if len(target_applicability) != 1:
                raise ValueError("target-drop applicability differs across a condition")
            if target_applicability == {True}:
                aggregates.append(
                    _record(
                        condition_id=condition_id,
                        method=policy,
                        metric_name="gap-vs-fault-target-drop",
                        window=window_name,
                        unit="m^2",
                        interval=_contrast_interval(
                            rows,
                            contrast="policy-target-drop",
                            bootstrap_indices=bootstrap_indices,
                        ),
                        sequence_count=len(sequence_ids),
                    )
                )
            oracle_applicability = {row.frame_oracle_applicable for row in rows}
            if len(oracle_applicability) != 1:
                raise ValueError("frame-oracle applicability differs across a condition")
            if oracle_applicability == {True}:
                aggregates.append(
                    _record(
                        condition_id=condition_id,
                        method=policy,
                        metric_name="gap-vs-frame-oracle",
                        window=window_name,
                        unit="m^2",
                        interval=_contrast_interval(
                            rows,
                            contrast="policy-frame-oracle",
                            bootstrap_indices=bootstrap_indices,
                        ),
                        sequence_count=len(sequence_ids),
                    )
                )
                identical_support = all(row.identical_support_recovery_applicable for row in rows)
                if identical_support:
                    fixed_values = tuple(
                        row.fixed_on_common_loss_sum_m2 / row.fixed_policy_common_count
                        for row in rows
                    )
                    policy_values = tuple(
                        row.policy_on_fixed_common_loss_sum_m2 / row.fixed_policy_common_count
                        for row in rows
                    )
                    oracle_values = tuple(
                        row.frame_oracle_on_common_loss_sum_m2 / row.fixed_policy_common_count
                        for row in rows
                        if row.frame_oracle_on_common_loss_sum_m2 is not None
                    )
                    if len(oracle_values) != len(rows):
                        raise ValueError("recovery rows are missing oracle statistics")
                    aggregates.append(
                        _record(
                            condition_id=condition_id,
                            method=policy,
                            metric_name="frame-oracle-recoverable-loss-fraction",
                            window=window_name,
                            unit="fraction",
                            interval=recovery_fraction_interval(
                                fixed_values,
                                policy_values,
                                oracle_values,
                                bootstrap_indices,
                            ),
                            sequence_count=len(sequence_ids),
                        )
                    )
    return aggregates


def _event_fraction_record(
    *,
    condition_id: str,
    policy: HealthMethod | None,
    metric_name: str,
    predicate: Sequence[bool],
    row_count: int,
    bootstrap_indices: IntArray,
) -> HealthAggregateMetricV1:
    return _record(
        condition_id=condition_id,
        method=policy,
        metric_name=metric_name,
        window="event",
        unit="fraction",
        interval=_fraction_interval(
            np.asarray(predicate, dtype=np.int64),
            np.ones(row_count, dtype=np.int64),
            bootstrap_indices,
        ),
        sequence_count=row_count,
    )


def _aggregate_event_metrics(
    *,
    condition_id: str,
    fault: HealthFaultSpec,
    events: tuple[HealthSequenceEventV1, ...],
    sequence_ids: tuple[str, ...],
    bootstrap_indices: IntArray,
) -> list[HealthAggregateMetricV1]:
    by_policy: dict[HealthMethod, list[HealthSequenceEventV1]] = defaultdict(list)
    for event in events:
        if event.condition_id != condition_id or event.sequence_id not in sequence_ids:
            raise ValueError("event row lies outside the condition population")
        by_policy[event.policy].append(event)
    aggregates: list[HealthAggregateMetricV1] = []
    if fault.family == "dropout":
        rows_by_policy: dict[HealthMethod, tuple[HealthSequenceEventV1, ...]] = {}
        for policy in _EVENT_POLICIES:
            row_index = {row.sequence_id: row for row in by_policy[policy]}
            if len(row_index) != len(sequence_ids):
                raise ValueError("dropout event policy rows are incomplete or duplicated")
            rows_by_policy[policy] = tuple(row_index[sequence_id] for sequence_id in sequence_ids)
        canonical_rows = rows_by_policy[_EVENT_POLICIES[0]]
        for sequence_index in range(len(sequence_ids)):
            observation_values = {
                (
                    rows_by_policy[policy][sequence_index].realized_dropout,
                    rows_by_policy[policy][sequence_index].first_missing_frame_minus_event_start,
                )
                for policy in _EVENT_POLICIES
            }
            if len(observation_values) != 1:
                raise ValueError("dropout observation metrics differ across policies")
        aggregates.append(
            _event_fraction_record(
                condition_id=condition_id,
                policy=None,
                metric_name="realized-dropout-fraction",
                predicate=tuple(bool(row.realized_dropout) for row in canonical_rows),
                row_count=len(canonical_rows),
                bootstrap_indices=bootstrap_indices,
            )
        )
        aggregates.append(
            _record(
                condition_id=condition_id,
                method=None,
                metric_name="first-missing-frame-minus-event-start",
                window="event",
                unit="frames",
                interval=_conditional_scalar_interval(
                    tuple(row.first_missing_frame_minus_event_start for row in canonical_rows),
                    bootstrap_indices,
                ),
                sequence_count=len(canonical_rows),
            )
        )
    for policy, raw_rows in by_policy.items():
        row_index = {row.sequence_id: row for row in raw_rows}
        if len(row_index) != len(sequence_ids):
            raise ValueError("event policy rows are incomplete or duplicated")
        rows = tuple(row_index[sequence_id] for sequence_id in sequence_ids)

        aggregates.append(
            _event_fraction_record(
                condition_id=condition_id,
                policy=policy,
                metric_name="detection-fraction",
                predicate=tuple(row.detected for row in rows),
                row_count=len(rows),
                bootstrap_indices=bootstrap_indices,
            )
        )
        outcomes = (
            ("ambiguous", "missed")
            if fault.target in {"none", "both"}
            else ("correct", "ambiguous", "wrong-sensor", "missed")
        )
        for outcome in outcomes:
            aggregates.append(
                _event_fraction_record(
                    condition_id=condition_id,
                    policy=policy,
                    metric_name=f"event-outcome-{outcome}-fraction",
                    predicate=tuple(row.outcome == outcome for row in rows),
                    row_count=len(rows),
                    bootstrap_indices=bootstrap_indices,
                )
            )
        if fault.family == "common-mode-position-bias":
            for label in ("camera-fault", "lidar-fault", "ambiguous"):
                aggregates.append(
                    _event_fraction_record(
                        condition_id=condition_id,
                        policy=policy,
                        metric_name=f"first-latch-label-{label}-fraction",
                        predicate=tuple(row.first_latch_label == label for row in rows),
                        row_count=len(rows),
                        bootstrap_indices=bootstrap_indices,
                    )
                )
        if fault.target in {"camera", "lidar"}:
            aggregates.append(
                _event_fraction_record(
                    condition_id=condition_id,
                    policy=policy,
                    metric_name="attribution-fraction",
                    predicate=tuple(row.correctly_attributed for row in rows),
                    row_count=len(rows),
                    bootstrap_indices=bootstrap_indices,
                )
            )
        aggregates.append(
            _event_fraction_record(
                condition_id=condition_id,
                policy=policy,
                metric_name="early-clear-fraction",
                predicate=tuple(row.early_clear for row in rows),
                row_count=len(rows),
                bootstrap_indices=bootstrap_indices,
            )
        )
        aggregates.append(
            _event_fraction_record(
                condition_id=condition_id,
                policy=policy,
                metric_name="recovery-denominator-fraction",
                predicate=tuple(row.recovery_eligible for row in rows),
                row_count=len(rows),
                bootstrap_indices=bootstrap_indices,
            )
        )
        recovery_sums = np.asarray(
            [int(row.recovered) for row in rows],
            dtype=np.int64,
        )
        recovery_denominators = np.asarray(
            [int(row.recovery_eligible) for row in rows],
            dtype=np.int64,
        )
        aggregates.append(
            _record(
                condition_id=condition_id,
                method=policy,
                metric_name="recovery-fraction",
                window="recovery",
                unit="fraction",
                interval=conditional_mean_interval(
                    recovery_sums.astype(np.float64),
                    recovery_denominators,
                    bootstrap_indices,
                ),
                sequence_count=len(rows),
            )
        )
        for state in ("healthy", "camera-fault", "lidar-fault", "ambiguous"):
            aggregates.append(
                _event_fraction_record(
                    condition_id=condition_id,
                    policy=policy,
                    metric_name=f"final-active-state-{state}-fraction",
                    predicate=tuple(row.final_active_state == state for row in rows),
                    row_count=len(rows),
                    bootstrap_indices=bootstrap_indices,
                )
            )
        if fault.family == "dropout":
            aggregates.append(
                _record(
                    condition_id=condition_id,
                    method=policy,
                    metric_name="detection-among-realized-dropout-fraction",
                    window="event",
                    unit="fraction",
                    interval=conditional_mean_interval(
                        np.asarray(
                            [int(bool(row.realized_dropout) and row.detected) for row in rows],
                            dtype=np.float64,
                        ),
                        np.asarray(
                            [int(bool(row.realized_dropout)) for row in rows],
                            dtype=np.int64,
                        ),
                        bootstrap_indices,
                    ),
                    sequence_count=len(rows),
                )
            )
        latency_metrics: list[tuple[str, tuple[int | None, ...]]] = [
            ("detection-latency", tuple(row.detection_latency_frames for row in rows)),
        ]
        if fault.target in {"camera", "lidar"}:
            latency_metrics.append(
                (
                    "attribution-latency",
                    tuple(row.attribution_latency_frames for row in rows),
                )
            )
        latency_metrics.append(
            ("recovery-latency", tuple(row.recovery_latency_frames for row in rows)),
        )
        if fault.family == "dropout":
            latency_metrics.append(
                (
                    "detection-minus-first-missing",
                    tuple(row.detection_minus_first_missing_frames for row in rows),
                )
            )
        for metric, values in latency_metrics:
            interval = _conditional_scalar_interval(values, bootstrap_indices)
            aggregates.append(
                _record(
                    condition_id=condition_id,
                    method=policy,
                    metric_name=metric,
                    window="event" if metric != "recovery-latency" else "recovery",
                    unit="frames",
                    interval=interval,
                    sequence_count=len(rows),
                )
            )
        false_alerts = np.asarray(
            [row.false_alert_episode_count for row in rows],
            dtype=np.float64,
        )
        aggregates.append(
            _record(
                condition_id=condition_id,
                method=policy,
                metric_name="false-alert-episode-starts",
                window="score",
                unit="count-per-sequence",
                interval=sequence_mean_interval(false_alerts, bootstrap_indices),
                sequence_count=len(rows),
            )
        )
        latch_episodes = np.asarray(
            [row.latch_episode_count for row in rows],
            dtype=np.float64,
        )
        aggregates.append(
            _record(
                condition_id=condition_id,
                method=policy,
                metric_name="latch-episode-starts",
                window="event",
                unit="count-per-sequence",
                interval=sequence_mean_interval(latch_episodes, bootstrap_indices),
                sequence_count=len(rows),
            )
        )
        occupancy_fields = (
            ("state-healthy-occupancy", "active_healthy_frames"),
            ("state-camera-fault-occupancy", "active_camera_fault_frames"),
            ("state-lidar-fault-occupancy", "active_lidar_fault_frames"),
            ("state-ambiguous-occupancy", "active_ambiguous_frames"),
            ("action-camera-occupancy", "active_camera_action_frames"),
            ("action-lidar-occupancy", "active_lidar_action_frames"),
            ("action-fixed-occupancy", "active_fixed_action_frames"),
            ("action-undefined-occupancy", "active_undefined_action_frames"),
        )
        denominators = np.asarray(
            [row.active_frame_count for row in rows],
            dtype=np.int64,
        )
        for metric, field_name in occupancy_fields:
            numerators = np.asarray(
                [getattr(row, field_name) for row in rows],
                dtype=np.int64,
            )
            aggregates.append(
                _record(
                    condition_id=condition_id,
                    method=policy,
                    metric_name=metric,
                    window="event",
                    unit="fraction",
                    interval=_fraction_interval(
                        numerators,
                        denominators,
                        bootstrap_indices,
                    ),
                    sequence_count=len(rows),
                )
            )
    return aggregates


def aggregate_health_condition(
    *,
    condition_id: str,
    fault: HealthFaultSpec,
    sequence_ids: Sequence[str],
    sequence_losses: Sequence[HealthSequenceLossV1],
    sequence_contrasts: Sequence[HealthSequenceContrastV1],
    sequence_events: Sequence[HealthSequenceEventV1],
) -> tuple[HealthAggregateMetricV1, ...]:
    """Aggregate one complete condition with a shared test bootstrap matrix."""

    population_ids = tuple(sequence_ids)
    if not population_ids:
        raise ValueError("health condition aggregation requires sequence evaluations")
    if len(set(population_ids)) != len(population_ids):
        raise ValueError("condition evaluations must have unique aligned sequence IDs")
    bootstrap_indices = paired_bootstrap_indices(
        seed=_BOOTSTRAP_SEED,
        replicates=_BOOTSTRAP_REPLICATES,
        sequence_count=len(population_ids),
    )
    index = _loss_record_index(
        sequence_losses,
        sequence_ids=population_ids,
        condition_id=condition_id,
    )
    method_windows = {
        (method, window)
        for sequence_id, method, window in index
        if sequence_id == population_ids[0]
    }
    expected_loss_keys = {
        (sequence_id, method, window)
        for sequence_id in population_ids
        for method, window in method_windows
    }
    if not method_windows or set(index) != expected_loss_keys:
        raise ValueError("condition sequence loss rows are incomplete")
    contrast_index = _contrast_record_index(
        sequence_contrasts,
        sequence_ids=population_ids,
        condition_id=condition_id,
    )
    aggregates = _aggregate_method_losses(
        condition_id=condition_id,
        sequence_ids=population_ids,
        index=index,
        bootstrap_indices=bootstrap_indices,
    )
    aggregates.extend(
        _aggregate_policy_contrasts(
            condition_id=condition_id,
            sequence_ids=population_ids,
            index=contrast_index,
            bootstrap_indices=bootstrap_indices,
        )
    )
    aggregates.extend(
        _aggregate_event_metrics(
            condition_id=condition_id,
            fault=fault,
            events=tuple(sequence_events),
            sequence_ids=population_ids,
            bootstrap_indices=bootstrap_indices,
        )
    )
    return tuple(aggregates)


def recompute_row_derived_health_aggregates(
    *,
    condition_id: str,
    fault: HealthFaultSpec,
    sequence_ids: tuple[str, ...],
    sequence_losses: Sequence[HealthSequenceLossV1],
    sequence_contrasts: Sequence[HealthSequenceContrastV1],
    sequence_events: Sequence[HealthSequenceEventV1],
) -> tuple[HealthAggregateMetricV1, ...]:
    """Independently recompute every aggregate from retained sequence rows."""

    if not sequence_ids or len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("row-derived aggregation requires unique sequence IDs")
    bootstrap_indices = paired_bootstrap_indices(
        seed=_BOOTSTRAP_SEED,
        replicates=_BOOTSTRAP_REPLICATES,
        sequence_count=len(sequence_ids),
    )
    index = _loss_record_index(
        sequence_losses,
        sequence_ids=sequence_ids,
        condition_id=condition_id,
    )
    contrast_index = _contrast_record_index(
        sequence_contrasts,
        sequence_ids=sequence_ids,
        condition_id=condition_id,
    )
    aggregates = _aggregate_method_losses(
        condition_id=condition_id,
        sequence_ids=sequence_ids,
        index=index,
        bootstrap_indices=bootstrap_indices,
    )
    aggregates.extend(
        _aggregate_policy_contrasts(
            condition_id=condition_id,
            sequence_ids=sequence_ids,
            index=contrast_index,
            bootstrap_indices=bootstrap_indices,
        )
    )
    aggregates.extend(
        _aggregate_event_metrics(
            condition_id=condition_id,
            fault=fault,
            events=tuple(sequence_events),
            sequence_ids=sequence_ids,
            bootstrap_indices=bootstrap_indices,
        )
    )
    return tuple(aggregates)
