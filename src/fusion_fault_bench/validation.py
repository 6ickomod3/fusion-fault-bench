"""Independent analytic-oracle evidence assembled from sequence-level results."""

from __future__ import annotations

import math
from collections import defaultdict

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.artifact_v1alpha1 import (
    AnalyticCrossoverReferenceV1Alpha1,
    AnalyticPopulationPointV1Alpha1,
    AnalyticValidationV1Alpha1,
)
from fusion_fault_bench.contracts.bundle_v1alpha1 import ConditionKey
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AnalyticCrossoverManifest,
    MethodId,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    LocalizationMetricRecord,
    MetricRecordV1Alpha1,
    SeverityCoordinate,
)
from fusion_fault_bench.reference.analytic import (
    population_crossover_references,
    population_points,
)


def _record_condition(record: LocalizationMetricRecord) -> ConditionKey:
    return ConditionKey(
        fault_family=record.fault_family,
        fault_axis=record.fault_axis,
        severity_index=record.severity.index,
        magnitude=record.severity.magnitude,
        direction=record.severity.direction,
        unit=record.severity.unit,
    )


def build_analytic_validation(
    manifest: AnalyticCrossoverManifest,
    *,
    run_id: str,
    metrics: tuple[MetricRecordV1Alpha1, ...],
) -> AnalyticValidationV1Alpha1:
    """Compare empirical sequence means with the independent scalar oracle."""

    digest = sha256_digest(manifest)
    values: defaultdict[tuple[ConditionKey, MethodId], list[float]] = defaultdict(list)
    for record in metrics:
        if not isinstance(record, LocalizationMetricRecord):
            raise TypeError("analytic validation requires localization sequence metrics")
        if record.run_id != run_id or record.manifest_sha256 != digest:
            raise ValueError("analytic validation received mismatched metric provenance")
        if record.metric_name != "matched-center-mse" or record.value is None:
            raise ValueError("analytic validation requires defined matched-center MSE rows")
        values[(_record_condition(record), record.method_id)].append(record.value)

    threshold = manifest.analytic_validation.monte_carlo_standard_error_multiplier
    sequence_count = manifest.source.sequence_count
    expected_key_count = len(population_points(manifest))
    if len(values) < expected_key_count:
        raise ValueError("analytic validation is missing empirical method-condition rows")

    points: list[AnalyticPopulationPointV1Alpha1] = []
    for reference in population_points(manifest):
        condition = reference.condition
        key = (
            ConditionKey(
                fault_family=condition.fault_family,
                fault_axis=condition.fault_axis,
                severity_index=condition.index,
                magnitude=condition.magnitude,
                direction=condition.direction,
                unit=condition.unit,
            ),
            reference.population.method_id,
        )
        samples = values.get(key)
        if samples is None or len(samples) != sequence_count:
            raise ValueError(
                "analytic validation requires one value per sequence for every reference point"
            )
        empirical = math.fsum(samples) / sequence_count
        standard_error = reference.population.mse_standard_error_m2(sequence_count)
        standardized_error = abs(empirical - reference.population.mse_m2) / standard_error
        points.append(
            AnalyticPopulationPointV1Alpha1(
                severity=SeverityCoordinate(
                    index=condition.index,
                    magnitude=condition.magnitude,
                    direction=condition.direction,
                    unit=condition.unit,
                ),
                method_id=reference.population.method_id,
                mean_unit="m",
                variance_unit="m^2",
                loss_unit="m^2",
                expected_mean_xy_m=reference.population.mean_xy_m,
                expected_actual_variance_xy_m2=reference.population.actual_variance_xy_m2,
                expected_reported_variance_xy_m2=reference.population.reported_variance_xy_m2,
                expected_mse_m2=reference.population.mse_m2,
                empirical_mse_m2=empirical,
                analytic_mse_standard_error_m2=standard_error,
                absolute_standardized_error=standardized_error,
                monte_carlo_passed=standardized_error <= threshold,
            )
        )

    references = tuple(
        AnalyticCrossoverReferenceV1Alpha1(
            direction=reference.direction,
            severity_unit=reference.severity_unit,
            tested_maximum=reference.tested_maximum,
            grid_status=reference.grid_status,
            grid_point_estimate=reference.grid_point_estimate,
            grid_censoring=reference.grid_censoring,
            continuous_status=reference.continuous_status,
            continuous_point_estimate=reference.continuous_point_estimate,
        )
        for reference in population_crossover_references(manifest)
    )
    return AnalyticValidationV1Alpha1(
        schema="ffb.analytic-validation/v1alpha1",
        run_id=run_id,
        manifest_sha256=digest,
        reference_model="independent-diagonal-gaussian-closed-form-v1",
        variance_representation="diagonal-xy-m2",
        monte_carlo_standard_error_multiplier=threshold,
        population_points=tuple(points),
        crossover_references=references,
        all_monte_carlo_checks_passed=all(point.monte_carlo_passed for point in points),
    )
