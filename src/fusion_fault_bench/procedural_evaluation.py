"""Population evaluation for procedural geometry and control experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.bundle_v1alpha1 import (
    ConditionKey,
    expected_conditions,
    expected_sequence_ids,
    validate_result_bundle,
)
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AvailabilityControlManifest,
    CommonModeControlManifest,
    FaultAxis,
    FaultFamily,
    GeometryCrossoverManifest,
    MethodId,
    ProceduralSource,
    SeverityDirection,
    SeverityUnit,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    AggregateMetricRecordV1Alpha1,
    CrossoverRecordV1Alpha1,
    LocalizationMetricRecord,
    MetricRecordV1Alpha1,
    RateMetricRecord,
    RunRecordV1Alpha1,
    SeverityCoordinate,
)
from fusion_fault_bench.evaluation import EvaluatedRecords
from fusion_fault_bench.inference import (
    bootstrap_conditional_loss,
    bootstrap_count_ratio,
    bootstrap_crossover_roots,
    bootstrap_crossover_status,
    bootstrap_mean,
    first_zero_crossover,
    paired_bootstrap_indices,
    pava_non_decreasing,
    percentile_interval,
)

type ProceduralManifest = (
    GeometryCrossoverManifest | CommonModeControlManifest | AvailabilityControlManifest
)
type IntArray = npt.NDArray[np.int64]
type CrossoverDirection = Literal["negative", "positive", "increase"]
type CrossoverStatus = Literal["observed", "not-observed", "undetermined"]
type CrossoverCensoring = Literal[
    "none",
    "right-above-tested-maximum",
    "mixed-bootstrap",
]
type CrossoverUpper = float | Literal["positive-infinity"] | None


def _severity(condition: ConditionKey) -> SeverityCoordinate:
    return SeverityCoordinate(
        index=condition.severity_index,
        magnitude=condition.magnitude,
        direction=cast(SeverityDirection, condition.direction),
        unit=cast(SeverityUnit, condition.unit),
    )


def _record_condition(record: MetricRecordV1Alpha1) -> ConditionKey:
    return ConditionKey(
        fault_family=record.fault_family,
        fault_axis=record.fault_axis,
        severity_index=record.severity.index,
        magnitude=record.severity.magnitude,
        direction=record.severity.direction,
        unit=record.severity.unit,
    )


def _expected_metric_pairs(
    manifest: ProceduralManifest,
) -> tuple[tuple[MethodId, str], ...]:
    if isinstance(manifest, AvailabilityControlManifest):
        return tuple(
            (method, metric_name)
            for method in manifest.methods
            for metric_name in manifest.evaluation.metrics
        )
    return tuple((method, "matched-center-mse") for method in manifest.methods)


def _ordered_metric_index(
    manifest: ProceduralManifest,
    metrics: Sequence[MetricRecordV1Alpha1],
) -> dict[tuple[str, ConditionKey, MethodId, str], MetricRecordV1Alpha1]:
    sequence_ids = expected_sequence_ids(manifest)
    conditions = expected_conditions(manifest)
    pairs = _expected_metric_pairs(manifest)
    expected_count = len(sequence_ids) * len(conditions) * len(pairs)
    if len(metrics) != expected_count:
        raise ValueError(
            "procedural sequence metric count is inconsistent: "
            f"expected {expected_count}, got {len(metrics)}"
        )

    result: dict[tuple[str, ConditionKey, MethodId, str], MetricRecordV1Alpha1] = {}
    cursor = 0
    for sequence_id in sequence_ids:
        for condition in conditions:
            for method, metric_name in pairs:
                record = metrics[cursor]
                cursor += 1
                actual = (
                    record.sequence_id,
                    _record_condition(record),
                    record.method_id,
                    record.metric_name,
                )
                expected = (sequence_id, condition, method, metric_name)
                if actual != expected:
                    raise ValueError(
                        "procedural sequence metrics are not in contractual order: "
                        f"expected {expected!r}, got {actual!r}"
                    )
                result[expected] = record
    return result


def _localization_values(
    *,
    sequence_ids: tuple[str, ...],
    condition: ConditionKey,
    method: MethodId,
    metric_name: Literal["matched-center-mse", "conditional-matched-center-mse"],
    index: dict[tuple[str, ConditionKey, MethodId, str], MetricRecordV1Alpha1],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    values: list[float] = []
    valid_counts: list[int] = []
    for sequence_id in sequence_ids:
        record = index[(sequence_id, condition, method, metric_name)]
        if not isinstance(record, LocalizationMetricRecord):
            raise TypeError(f"{metric_name} requires localization sequence rows")
        valid_counts.append(record.valid_object_frame_count)
        values.append(0.0 if record.value is None else record.value)
    return (
        np.asarray(values, dtype=np.float64),
        np.asarray(valid_counts, dtype=np.int64),
    )


def _count_values(
    *,
    sequence_ids: tuple[str, ...],
    condition: ConditionKey,
    method: MethodId,
    metric_name: Literal["coverage", "undefined-output-rate"],
    index: dict[tuple[str, ConditionKey, MethodId, str], MetricRecordV1Alpha1],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    numerators: list[int] = []
    denominators: list[int] = []
    for sequence_id in sequence_ids:
        record = index[(sequence_id, condition, method, metric_name)]
        if not isinstance(record, RateMetricRecord):
            raise TypeError(f"{metric_name} requires rate sequence rows")
        denominators.append(record.eligible_object_frame_count)
        if metric_name == "coverage":
            numerators.append(record.valid_object_frame_count)
        else:
            numerators.append(record.eligible_object_frame_count - record.valid_object_frame_count)
    return (
        np.asarray(numerators, dtype=np.int64),
        np.asarray(denominators, dtype=np.int64),
    )


def _aggregate(
    *,
    manifest: ProceduralManifest,
    run_id: str,
    digest: str,
    sequence_ids: tuple[str, ...],
    condition: ConditionKey,
    method: MethodId,
    metric_name: Literal[
        "matched-center-mse",
        "conditional-matched-center-mse",
        "coverage",
        "undefined-output-rate",
        "fused-minus-healthy",
    ],
    index: dict[tuple[str, ConditionKey, MethodId, str], MetricRecordV1Alpha1],
    bootstrap_indices: IntArray,
) -> AggregateMetricRecordV1Alpha1:
    bootstrap = manifest.evaluation.bootstrap
    sequence_count = len(sequence_ids)
    contributing = sequence_count
    unit: Literal["m^2", "fraction"]
    aggregation: Literal[
        "object-frame-mean-then-sequence-mean",
        "valid-object-frame-ratio-with-sequence-bootstrap",
        "count-ratio-with-sequence-bootstrap",
    ]

    if metric_name == "fused-minus-healthy":
        if not isinstance(manifest, GeometryCrossoverManifest):
            raise TypeError("fused-minus-healthy requires a geometry crossover manifest")
        healthy: MethodId = (
            "lidar-only" if manifest.fault_sweep.target == "camera" else "camera-only"
        )
        fused, _ = _localization_values(
            sequence_ids=sequence_ids,
            condition=condition,
            method="fixed-fusion",
            metric_name="matched-center-mse",
            index=index,
        )
        healthy_values, _ = _localization_values(
            sequence_ids=sequence_ids,
            condition=condition,
            method=healthy,
            metric_name="matched-center-mse",
            index=index,
        )
        values = fused - healthy_values
        point = math.fsum(float(value) for value in values) / sequence_count
        bootstrap_values = bootstrap_mean(values, bootstrap_indices)
        unit = "m^2"
        aggregation = "object-frame-mean-then-sequence-mean"
    elif metric_name == "matched-center-mse":
        values, valid = _localization_values(
            sequence_ids=sequence_ids,
            condition=condition,
            method=method,
            metric_name=metric_name,
            index=index,
        )
        if np.any(valid <= 0):
            raise ValueError("matched-center-mse rows must all be defined")
        point = math.fsum(float(value) for value in values) / sequence_count
        bootstrap_values = bootstrap_mean(values, bootstrap_indices)
        unit = "m^2"
        aggregation = "object-frame-mean-then-sequence-mean"
    elif metric_name == "conditional-matched-center-mse":
        values, valid = _localization_values(
            sequence_ids=sequence_ids,
            condition=condition,
            method=method,
            metric_name=metric_name,
            index=index,
        )
        loss_sums = values * valid
        total_valid = int(valid.sum(dtype=np.int64))
        point = (
            math.fsum(float(value) for value in loss_sums) / total_valid
            if total_valid > 0
            else None
        )
        bootstrap_values = bootstrap_conditional_loss(
            loss_sums,
            valid,
            bootstrap_indices,
        )
        contributing = int(np.count_nonzero(valid))
        unit = "m^2"
        aggregation = "valid-object-frame-ratio-with-sequence-bootstrap"
    else:
        numerators, denominators = _count_values(
            sequence_ids=sequence_ids,
            condition=condition,
            method=method,
            metric_name=metric_name,
            index=index,
        )
        point = int(numerators.sum(dtype=np.int64)) / int(denominators.sum(dtype=np.int64))
        bootstrap_values = bootstrap_count_ratio(
            numerators,
            denominators,
            bootstrap_indices,
        )
        unit = "fraction"
        aggregation = "count-ratio-with-sequence-bootstrap"

    defined_replicates = int(bootstrap_values.size)
    alpha = 1.0 - bootstrap.confidence_level
    has_support = defined_replicates / bootstrap.replicates > 1.0 - alpha / 2.0
    status: Literal["ok", "undefined"] = "ok" if point is not None and has_support else "undefined"
    if status == "ok":
        interval_lower, interval_upper = percentile_interval(
            bootstrap_values,
            confidence_level=bootstrap.confidence_level,
        )
        estimate = point
    else:
        estimate = None
        interval_lower = None
        interval_upper = None

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
        status=status,
        estimate=estimate,
        interval_lower=interval_lower,
        interval_upper=interval_upper,
        unit=unit,
        sequence_count=sequence_count,
        contributing_sequence_count=contributing,
        bootstrap_replicates=bootstrap.replicates,
        defined_bootstrap_replicates=defined_replicates,
        confidence_level=bootstrap.confidence_level,
        interval_method="paired-sequence-percentile-pointwise",
        aggregation=aggregation,
    )


def _crossovers(
    *,
    manifest: GeometryCrossoverManifest,
    run_id: str,
    digest: str,
    sequence_ids: tuple[str, ...],
    conditions: tuple[ConditionKey, ...],
    index: dict[tuple[str, ConditionKey, MethodId, str], MetricRecordV1Alpha1],
    bootstrap_indices: IntArray,
) -> tuple[CrossoverRecordV1Alpha1, ...]:
    healthy: MethodId = "lidar-only" if manifest.fault_sweep.target == "camera" else "camera-only"
    directions = tuple(
        dict.fromkeys(
            condition.direction for condition in conditions if condition.direction != "identity"
        )
    )
    result: list[CrossoverRecordV1Alpha1] = []
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
                _localization_values(
                    sequence_ids=sequence_ids,
                    condition=condition,
                    method="fixed-fusion",
                    metric_name="matched-center-mse",
                    index=index,
                )[0]
                - _localization_values(
                    sequence_ids=sequence_ids,
                    condition=condition,
                    method=healthy,
                    metric_name="matched-center-mse",
                    index=index,
                )[0]
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
        crossing_count = sum(root is not None for root in roots)
        fraction = crossing_count / len(roots)
        status: CrossoverStatus = bootstrap_crossover_status(
            point_crossed=point_root is not None,
            crossing_count=crossing_count,
            bootstrap_replicates=len(roots),
        )
        censoring: CrossoverCensoring
        interval_upper: CrossoverUpper
        if status == "observed":
            censored = np.asarray(
                [root if root is not None else np.inf for root in roots],
                dtype=np.float64,
            )
            interval_lower, upper = percentile_interval(
                censored,
                confidence_level=manifest.evaluation.bootstrap.confidence_level,
            )
            interval_upper = upper
            censoring = "none"
        elif status == "not-observed":
            interval_lower = float(magnitudes[-1])
            interval_upper = "positive-infinity"
            censoring = "right-above-tested-maximum"
        else:
            interval_lower = None
            interval_upper = None
            censoring = "mixed-bootstrap"

        result.append(
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
    return tuple(result)


def evaluate_procedural_records(
    manifest: ProceduralManifest,
    *,
    run_id: str,
    metrics: tuple[MetricRecordV1Alpha1, ...],
) -> EvaluatedRecords:
    """Aggregate ordered procedural rows using the canonical bundle estimands."""

    if not isinstance(manifest.source, ProceduralSource):
        raise TypeError("procedural evaluation requires a procedural source")
    digest = sha256_digest(manifest)
    sequence_ids = expected_sequence_ids(manifest)
    conditions = expected_conditions(manifest)
    index = _ordered_metric_index(manifest, metrics)
    bootstrap_indices = paired_bootstrap_indices(
        seed=manifest.rng.bootstrap_seed,
        replicates=manifest.evaluation.bootstrap.replicates,
        sequence_count=len(sequence_ids),
    )

    aggregates: list[AggregateMetricRecordV1Alpha1] = []
    for condition in conditions:
        if isinstance(manifest, AvailabilityControlManifest):
            for method in manifest.methods:
                for metric_name in manifest.evaluation.metrics:
                    aggregates.append(
                        _aggregate(
                            manifest=manifest,
                            run_id=run_id,
                            digest=digest,
                            sequence_ids=sequence_ids,
                            condition=condition,
                            method=method,
                            metric_name=metric_name,
                            index=index,
                            bootstrap_indices=bootstrap_indices,
                        )
                    )
            continue

        for method in manifest.methods:
            aggregates.append(
                _aggregate(
                    manifest=manifest,
                    run_id=run_id,
                    digest=digest,
                    sequence_ids=sequence_ids,
                    condition=condition,
                    method=method,
                    metric_name="matched-center-mse",
                    index=index,
                    bootstrap_indices=bootstrap_indices,
                )
            )
        if isinstance(manifest, GeometryCrossoverManifest):
            aggregates.append(
                _aggregate(
                    manifest=manifest,
                    run_id=run_id,
                    digest=digest,
                    sequence_ids=sequence_ids,
                    condition=condition,
                    method="fixed-fusion",
                    metric_name="fused-minus-healthy",
                    index=index,
                    bootstrap_indices=bootstrap_indices,
                )
            )

    crossovers = (
        _crossovers(
            manifest=manifest,
            run_id=run_id,
            digest=digest,
            sequence_ids=sequence_ids,
            conditions=conditions,
            index=index,
            bootstrap_indices=bootstrap_indices,
        )
        if isinstance(manifest, GeometryCrossoverManifest)
        else ()
    )
    return EvaluatedRecords(
        metrics=metrics,
        aggregates=tuple(aggregates),
        crossovers=crossovers,
    )


def validate_evaluated_procedural_records(
    manifest: ProceduralManifest,
    *,
    run: RunRecordV1Alpha1,
    records: EvaluatedRecords,
) -> None:
    """Apply the independent generic bundle validator to evaluated records."""

    validate_result_bundle(
        manifest,
        run,
        records.metrics,
        records.aggregates,
        records.crossovers,
    )
