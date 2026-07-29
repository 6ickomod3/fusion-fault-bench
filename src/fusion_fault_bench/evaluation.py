"""Manifest-ordered analytic aggregation and crossover evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.bundle_v1alpha1 import (
    ConditionKey,
    expected_conditions,
    expected_sequence_ids,
)
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AnalyticCrossoverManifest,
    FaultAxis,
    FaultFamily,
    MethodId,
    SeverityDirection,
    SeverityUnit,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    AggregateMetricRecordV1Alpha1,
    CrossoverRecordV1Alpha1,
    LocalizationMetricRecord,
    MetricRecordV1Alpha1,
    SeverityCoordinate,
)
from fusion_fault_bench.inference import (
    bootstrap_crossover_roots,
    bootstrap_crossover_status,
    bootstrap_mean,
    first_zero_crossover,
    paired_bootstrap_indices,
    pava_non_decreasing,
    percentile_interval,
)

type CrossoverDirection = Literal["negative", "positive", "increase"]
type CrossoverStatus = Literal["observed", "not-observed", "undetermined"]
type CrossoverCensoring = Literal[
    "none",
    "right-above-tested-maximum",
    "mixed-bootstrap",
]
type CrossoverUpper = float | Literal["positive-infinity"] | None
type AggregateMetricName = Literal["matched-center-mse", "fused-minus-healthy"]


@dataclass(frozen=True, slots=True)
class EvaluatedRecords:
    """Sequence and population records before artifact serialization."""

    metrics: tuple[MetricRecordV1Alpha1, ...]
    aggregates: tuple[AggregateMetricRecordV1Alpha1, ...]
    crossovers: tuple[CrossoverRecordV1Alpha1, ...]


def _severity(condition: ConditionKey) -> SeverityCoordinate:
    return SeverityCoordinate(
        index=condition.severity_index,
        magnitude=condition.magnitude,
        direction=cast(SeverityDirection, condition.direction),
        unit=cast(SeverityUnit, condition.unit),
    )


def _metric_values(
    *,
    sequence_ids: tuple[str, ...],
    condition: ConditionKey,
    method: MethodId,
    index: dict[tuple[str, ConditionKey, MethodId], LocalizationMetricRecord],
) -> np.ndarray:
    values: list[float] = []
    for sequence_id in sequence_ids:
        record = index[(sequence_id, condition, method)]
        assert record.value is not None
        values.append(record.value)
    return np.asarray(values, dtype=np.float64)


def _aggregate_record(
    *,
    manifest: AnalyticCrossoverManifest,
    run_id: str,
    digest: str,
    condition: ConditionKey,
    method: MethodId,
    metric_name: AggregateMetricName,
    values: np.ndarray,
    bootstrap_indices: npt.NDArray[np.int64],
) -> AggregateMetricRecordV1Alpha1:
    bootstrap_values = bootstrap_mean(values, bootstrap_indices)
    lower, upper = percentile_interval(
        bootstrap_values,
        confidence_level=manifest.evaluation.bootstrap.confidence_level,
    )
    return AggregateMetricRecordV1Alpha1(
        schema="ffb.aggregate-metric/v1alpha1",
        record_level="aggregate",
        run_id=run_id,
        manifest_sha256=digest,
        fault_family=cast(FaultFamily, condition.fault_family),
        fault_axis=cast(FaultAxis, condition.fault_axis),
        severity=_severity(condition),
        method_id=method,
        metric_name=metric_name,
        status="ok",
        estimate=math.fsum(float(value) for value in values) / values.size,
        interval_lower=lower,
        interval_upper=upper,
        unit="m^2",
        sequence_count=values.size,
        contributing_sequence_count=values.size,
        bootstrap_replicates=manifest.evaluation.bootstrap.replicates,
        defined_bootstrap_replicates=manifest.evaluation.bootstrap.replicates,
        confidence_level=manifest.evaluation.bootstrap.confidence_level,
        interval_method="paired-sequence-percentile-pointwise",
        aggregation="object-frame-mean-then-sequence-mean",
    )


def evaluate_analytic_records(
    manifest: AnalyticCrossoverManifest,
    *,
    run_id: str,
    metrics: tuple[MetricRecordV1Alpha1, ...],
) -> EvaluatedRecords:
    """Compute aggregate and crossover records from ordered sequence evidence."""

    digest = sha256_digest(manifest)
    conditions = expected_conditions(manifest)
    sequence_ids = expected_sequence_ids(manifest)
    expected_metric_count = len(sequence_ids) * len(conditions) * len(manifest.methods)
    if len(metrics) != expected_metric_count:
        raise ValueError(
            "analytic sequence metric count is inconsistent: "
            f"expected {expected_metric_count}, got {len(metrics)}"
        )
    index: dict[tuple[str, ConditionKey, MethodId], LocalizationMetricRecord] = {}
    cursor = 0
    for sequence_id in sequence_ids:
        for condition in conditions:
            for method in manifest.methods:
                record = metrics[cursor]
                cursor += 1
                if not isinstance(record, LocalizationMetricRecord):
                    raise TypeError("analytic evaluation requires localization rows")
                actual_key = (
                    record.sequence_id,
                    ConditionKey(
                        fault_family=record.fault_family,
                        fault_axis=record.fault_axis,
                        severity_index=record.severity.index,
                        magnitude=record.severity.magnitude,
                        direction=record.severity.direction,
                        unit=record.severity.unit,
                    ),
                    record.method_id,
                )
                expected_key = (sequence_id, condition, method)
                if actual_key != expected_key:
                    raise ValueError(
                        "analytic sequence metrics are not in contractual order: "
                        f"expected {expected_key!r}, got {actual_key!r}"
                    )
                index[(sequence_id, condition, method)] = record
    assert cursor == len(metrics)

    bootstrap_indices = paired_bootstrap_indices(
        seed=manifest.rng.bootstrap_seed,
        replicates=manifest.evaluation.bootstrap.replicates,
        sequence_count=len(sequence_ids),
    )
    aggregates: list[AggregateMetricRecordV1Alpha1] = []
    healthy_method: MethodId = (
        "lidar-only" if manifest.fault_sweep.target == "camera" else "camera-only"
    )
    for condition in conditions:
        for method in manifest.methods:
            values = _metric_values(
                sequence_ids=sequence_ids,
                condition=condition,
                method=method,
                index=index,
            )
            aggregates.append(
                _aggregate_record(
                    manifest=manifest,
                    run_id=run_id,
                    digest=digest,
                    condition=condition,
                    method=method,
                    metric_name="matched-center-mse",
                    values=values,
                    bootstrap_indices=bootstrap_indices,
                )
            )
        contrasts = _metric_values(
            sequence_ids=sequence_ids,
            condition=condition,
            method="fixed-fusion",
            index=index,
        ) - _metric_values(
            sequence_ids=sequence_ids,
            condition=condition,
            method=healthy_method,
            index=index,
        )
        aggregates.append(
            _aggregate_record(
                manifest=manifest,
                run_id=run_id,
                digest=digest,
                condition=condition,
                method="fixed-fusion",
                metric_name="fused-minus-healthy",
                values=contrasts,
                bootstrap_indices=bootstrap_indices,
            )
        )

    directions = tuple(
        dict.fromkeys(
            condition.direction for condition in conditions if condition.direction != "identity"
        )
    )
    crossovers: list[CrossoverRecordV1Alpha1] = []
    for raw_direction in directions:
        curve_conditions = tuple(
            condition
            for condition in conditions
            if condition.direction in {"identity", raw_direction}
        )
        magnitudes = np.asarray(
            [condition.magnitude for condition in curve_conditions],
            dtype=np.float64,
        )
        contrasts = np.stack(
            [
                _metric_values(
                    sequence_ids=sequence_ids,
                    condition=condition,
                    method="fixed-fusion",
                    index=index,
                )
                - _metric_values(
                    sequence_ids=sequence_ids,
                    condition=condition,
                    method=healthy_method,
                    index=index,
                )
                for condition in curve_conditions
            ]
        )
        point_curve = np.asarray(
            [math.fsum(float(value) for value in row) / len(sequence_ids) for row in contrasts],
            dtype=np.float64,
        )
        point_root = first_zero_crossover(
            magnitudes,
            pava_non_decreasing(point_curve),
            zero_tolerance=manifest.evaluation.crossover.zero_tolerance_m2,
        )
        roots = bootstrap_crossover_roots(
            magnitudes=magnitudes,
            sequence_contrasts=contrasts,
            indices=bootstrap_indices,
            zero_tolerance=manifest.evaluation.crossover.zero_tolerance_m2,
        )
        finite_roots = [root for root in roots if root is not None]
        crossing_count = len(finite_roots)
        replicate_count = len(roots)
        fraction = crossing_count / replicate_count
        status: CrossoverStatus = bootstrap_crossover_status(
            point_crossed=point_root is not None,
            crossing_count=crossing_count,
            bootstrap_replicates=replicate_count,
        )
        censoring: CrossoverCensoring
        interval_upper: CrossoverUpper
        if status == "observed":
            censored = np.asarray(
                [root if root is not None else np.inf for root in roots],
                dtype=np.float64,
            )
            lower, upper = percentile_interval(
                censored,
                confidence_level=manifest.evaluation.bootstrap.confidence_level,
            )
            interval_lower: float | None = lower
            interval_upper = upper
            censoring = "none"
        elif status == "not-observed":
            interval_lower = float(magnitudes[-1])
            interval_upper = "positive-infinity"
            censoring = "right-above-tested-maximum"
        else:
            status = "undetermined"
            interval_lower = None
            interval_upper = None
            censoring = "mixed-bootstrap"
        crossovers.append(
            CrossoverRecordV1Alpha1(
                schema="ffb.crossover/v1alpha1",
                run_id=run_id,
                manifest_sha256=digest,
                fault_family=manifest.fault_sweep.kind,
                fault_axis=manifest.fault_sweep.axis,
                direction=cast(CrossoverDirection, raw_direction),
                severity_unit=manifest.fault_sweep.unit,
                status=status,
                point_curve_crossed=point_root is not None,
                point_estimate=point_root,
                interval_lower=interval_lower,
                interval_upper=interval_upper,
                tested_maximum=float(magnitudes[-1]),
                censoring=censoring,
                bootstrap_crossing_fraction=fraction,
                sequence_count=len(sequence_ids),
                bootstrap_replicates=manifest.evaluation.bootstrap.replicates,
                confidence_level=manifest.evaluation.bootstrap.confidence_level,
                interval_method="right-censored-percentile",
            )
        )
    return EvaluatedRecords(
        metrics=metrics,
        aggregates=tuple(aggregates),
        crossovers=tuple(crossovers),
    )
