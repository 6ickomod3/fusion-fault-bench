from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.bundle_v1alpha1 import (
    ResultBundleValidationError,
    expected_conditions,
    expected_sequence_ids,
    validate_result_bundle,
)
from fusion_fault_bench.contracts.io import validate_manifest_mapping
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AnalyticCrossoverManifest,
    AvailabilityControlManifest,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    AggregateMetricRecordV1Alpha1,
    CrossoverRecordV1Alpha1,
    LocalizationMetricRecord,
    MetricRecordV1Alpha1,
    RateMetricRecord,
    RunRecordV1Alpha1,
    RuntimeEnvironment,
    SeverityCoordinate,
)

GIT_REVISION = "b" * 40
LOCK_DIGEST = "c" * 64


def _small_analytic(data: dict[str, Any]) -> AnalyticCrossoverManifest:
    data = copy.deepcopy(data)
    data["source"]["sequence_count"] = 2
    data["fault_sweep"]["magnitude_values_m"] = [0.0, 1.0]
    manifest = validate_manifest_mapping(data)
    assert isinstance(manifest, AnalyticCrossoverManifest)
    return manifest


def _run(manifest: AnalyticCrossoverManifest | AvailabilityControlManifest) -> RunRecordV1Alpha1:
    now = datetime.now(UTC)
    return RunRecordV1Alpha1(
        schema="ffb.run/v1alpha1",
        run_id="run-001",
        manifest_sha256=sha256_digest(manifest),
        package_version="0.1.0",
        git_revision=GIT_REVISION,
        source_dirty=False,
        lockfile_sha256=LOCK_DIGEST,
        command=("ffb", "run", "manifest.json"),
        environment=RuntimeEnvironment(
            python_version="3.12.13",
            os_name="macOS",
            os_release="15.5",
            machine="arm64",
            cpu_model="Apple-M4",
            logical_cpu_count=10,
            memory_bytes=32_000_000_000,
        ),
        started_at=now,
        ended_at=now + timedelta(seconds=1),
        status="succeeded",
        artifact_sha256="d" * 64,
    )


def _analytic_bundle(
    manifest: AnalyticCrossoverManifest,
) -> tuple[
    list[MetricRecordV1Alpha1],
    list[AggregateMetricRecordV1Alpha1],
    list[CrossoverRecordV1Alpha1],
]:
    digest = sha256_digest(manifest)
    metrics: list[MetricRecordV1Alpha1] = []
    aggregates: list[AggregateMetricRecordV1Alpha1] = []
    for condition in expected_conditions(manifest):
        severity = SeverityCoordinate(
            index=condition.severity_index,
            magnitude=condition.magnitude,
            direction=condition.direction,
            unit=condition.unit,
        )
        for method in manifest.methods:
            for sequence_id in expected_sequence_ids(manifest):
                metrics.append(
                    LocalizationMetricRecord(
                        schema="ffb.sequence-metric/v1alpha1",
                        record_level="sequence",
                        run_id="run-001",
                        manifest_sha256=digest,
                        sequence_id=sequence_id,
                        fault_family=condition.fault_family,
                        fault_axis=condition.fault_axis,
                        severity=severity,
                        method_id=method,
                        eligible_object_frame_count=10,
                        valid_object_frame_count=10,
                        metric_name="matched-center-mse",
                        status="ok",
                        value=0.25,
                        unit="m^2",
                    )
                )
            aggregates.append(
                AggregateMetricRecordV1Alpha1(
                    schema="ffb.aggregate-metric/v1alpha1",
                    record_level="aggregate",
                    run_id="run-001",
                    manifest_sha256=digest,
                    fault_family=condition.fault_family,
                    fault_axis=condition.fault_axis,
                    severity=severity,
                    method_id=method,
                    metric_name="matched-center-mse",
                    status="ok",
                    estimate=0.25,
                    interval_lower=0.25,
                    interval_upper=0.25,
                    unit="m^2",
                    sequence_count=2,
                    contributing_sequence_count=2,
                    bootstrap_replicates=2000,
                    defined_bootstrap_replicates=2000,
                    confidence_level=0.95,
                    interval_method="paired-sequence-percentile-pointwise",
                    aggregation="object-frame-mean-then-sequence-mean",
                )
            )
        aggregates.append(
            AggregateMetricRecordV1Alpha1(
                schema="ffb.aggregate-metric/v1alpha1",
                record_level="aggregate",
                run_id="run-001",
                manifest_sha256=digest,
                fault_family=condition.fault_family,
                fault_axis=condition.fault_axis,
                severity=severity,
                method_id="fixed-fusion",
                metric_name="fused-minus-healthy",
                status="ok",
                estimate=0.0,
                interval_lower=0.0,
                interval_upper=0.0,
                unit="m^2",
                sequence_count=2,
                contributing_sequence_count=2,
                bootstrap_replicates=2000,
                defined_bootstrap_replicates=2000,
                confidence_level=0.95,
                interval_method="paired-sequence-percentile-pointwise",
                aggregation="object-frame-mean-then-sequence-mean",
            )
        )

    crossovers = [
        CrossoverRecordV1Alpha1(
            schema="ffb.crossover/v1alpha1",
            run_id="run-001",
            manifest_sha256=digest,
            fault_family="additive-position-bias",
            fault_axis="x",
            direction=direction,
            severity_unit="m",
            status="observed",
            point_curve_crossed=True,
            point_estimate=0.0,
            interval_lower=0.0,
            interval_upper=0.0,
            tested_maximum=1.0,
            censoring="none",
            bootstrap_crossing_fraction=1.0,
            sequence_count=2,
            bootstrap_replicates=2000,
            confidence_level=0.95,
            interval_method="right-censored-percentile",
        )
        for direction in ("negative", "positive")
    ]
    return metrics, aggregates, crossovers


def _availability_manifest(
    data: dict[str, Any],
    probabilities: tuple[float, ...] = (0.0, 1.0),
) -> AvailabilityControlManifest:
    data = copy.deepcopy(data)
    data.pop("analytic_validation")
    data.update(
        {
            "kind": "availability-control",
            "source": {
                "kind": "procedural",
                "split": "test",
                "profile_id": "tiny-availability-v1",
                "profile_sha256": "a" * 64,
                "sequence_count": 2,
            },
            "roi": {
                "frame": "ego-bev",
                "x_min_m": 5.0,
                "x_max_m": 60.0,
                "abs_y_max_m": 30.0,
                "camera_half_fov_rad": 0.6,
            },
            "fault_sweep": {
                "kind": "dropout",
                "target": "camera",
                "axis": "availability",
                "unit": "probability",
                "injection_site": "availability",
                "process": "shared-target-modality-frame-bernoulli",
                "probability_values": list(probabilities),
            },
            "methods": [
                "camera-only",
                "lidar-only",
                "fixed-fusion",
                "fault-target-drop-policy",
            ],
            "evaluation": {
                "mode": "availability-control",
                "metrics": [
                    "coverage",
                    "conditional-matched-center-mse",
                    "undefined-output-rate",
                ],
                "missing_output_policy": "undefined-no-localization-penalty",
                "rate_aggregation": "count-ratio-with-sequence-bootstrap",
                "conditional_loss_aggregation": (
                    "valid-object-frame-ratio-with-sequence-bootstrap"
                ),
                "undefined_bootstrap_replicate_action": ("exclude-and-require-two-sided-support"),
                "unimodal_missing_input_action": "undefined",
                "fixed_fusion_missing_input_action": "undefined",
                "target_drop_identity_action": "fixed-fusion",
                "target_drop_nonidentity_action": "use-nontarget-modality",
                "bootstrap": {
                    "method": "percentile",
                    "unit": "sequence",
                    "resampling": "paired-indices-across-severities-and-methods",
                    "interval_scope": "pointwise",
                    "replicates": 200,
                    "confidence_level": 0.95,
                },
                "crossover": "not-applicable",
            },
        }
    )
    manifest = validate_manifest_mapping(data)
    assert isinstance(manifest, AvailabilityControlManifest)
    return manifest


def _availability_bundle(
    manifest: AvailabilityControlManifest,
) -> tuple[list[MetricRecordV1Alpha1], list[AggregateMetricRecordV1Alpha1]]:
    digest = sha256_digest(manifest)
    metrics: list[MetricRecordV1Alpha1] = []
    aggregates: list[AggregateMetricRecordV1Alpha1] = []
    for condition in expected_conditions(manifest):
        severity = SeverityCoordinate(
            index=condition.severity_index,
            magnitude=condition.magnitude,
            direction=condition.direction,
            unit=condition.unit,
        )
        for method in manifest.methods:
            valid = (
                0
                if condition.magnitude == 1.0 and method in {"camera-only", "fixed-fusion"}
                else 10
            )
            for sequence_id in expected_sequence_ids(manifest):
                common = {
                    "schema": "ffb.sequence-metric/v1alpha1",
                    "record_level": "sequence",
                    "run_id": "run-001",
                    "manifest_sha256": digest,
                    "sequence_id": sequence_id,
                    "fault_family": condition.fault_family,
                    "fault_axis": condition.fault_axis,
                    "severity": severity,
                    "method_id": method,
                    "eligible_object_frame_count": 10,
                    "valid_object_frame_count": valid,
                }
                metrics.extend(
                    (
                        RateMetricRecord(
                            **common,
                            metric_name="coverage",
                            status="ok",
                            value=valid / 10,
                            unit="fraction",
                        ),
                        LocalizationMetricRecord(
                            **common,
                            metric_name="conditional-matched-center-mse",
                            status="ok" if valid else "undefined",
                            value=0.25 if valid else None,
                            unit="m^2",
                        ),
                        RateMetricRecord(
                            **common,
                            metric_name="undefined-output-rate",
                            status="ok",
                            value=(10 - valid) / 10,
                            unit="fraction",
                        ),
                    )
                )

            for metric_name in (
                "coverage",
                "conditional-matched-center-mse",
                "undefined-output-rate",
            ):
                is_conditional = metric_name == "conditional-matched-center-mse"
                is_rate = not is_conditional
                defined = valid > 0
                aggregates.append(
                    AggregateMetricRecordV1Alpha1(
                        schema="ffb.aggregate-metric/v1alpha1",
                        record_level="aggregate",
                        run_id="run-001",
                        manifest_sha256=digest,
                        fault_family=condition.fault_family,
                        fault_axis=condition.fault_axis,
                        severity=severity,
                        method_id=method,
                        metric_name=metric_name,
                        status="ok" if is_rate or defined else "undefined",
                        estimate=(
                            valid / 10
                            if metric_name == "coverage"
                            else (
                                (10 - valid) / 10
                                if metric_name == "undefined-output-rate"
                                else (0.25 if defined else None)
                            )
                        ),
                        interval_lower=(
                            valid / 10
                            if metric_name == "coverage"
                            else (
                                (10 - valid) / 10
                                if metric_name == "undefined-output-rate"
                                else (0.25 if defined else None)
                            )
                        ),
                        interval_upper=(
                            valid / 10
                            if metric_name == "coverage"
                            else (
                                (10 - valid) / 10
                                if metric_name == "undefined-output-rate"
                                else (0.25 if defined else None)
                            )
                        ),
                        unit="fraction" if is_rate else "m^2",
                        sequence_count=2,
                        contributing_sequence_count=2 if is_rate or defined else 0,
                        bootstrap_replicates=200,
                        defined_bootstrap_replicates=200 if is_rate or defined else 0,
                        confidence_level=0.95,
                        interval_method="paired-sequence-percentile-pointwise",
                        aggregation=(
                            "count-ratio-with-sequence-bootstrap"
                            if is_rate
                            else "valid-object-frame-ratio-with-sequence-bootstrap"
                        ),
                    )
                )
    return metrics, aggregates


def test_valid_analytic_bundle_is_complete_and_order_independent(
    manifest_data: dict[str, Any],
) -> None:
    manifest = _small_analytic(manifest_data)
    metrics, aggregates, crossovers = _analytic_bundle(manifest)

    validate_result_bundle(
        manifest,
        _run(manifest),
        tuple(reversed(metrics)),
        tuple(reversed(aggregates)),
        tuple(reversed(crossovers)),
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing-row", "missing sequence record key"),
        ("duplicate-row", "duplicate sequence record key"),
        ("wrong-sequence", "unexpected sequence record key"),
        ("denominator-change", "eligible denominator changes"),
        ("missing-aggregate", "missing aggregate record key"),
        ("missing-crossover", "missing crossover record key"),
        ("fraction-not-representable", "not representable"),
        ("duplicate-aggregate", "duplicate aggregate record key"),
        ("duplicate-crossover", "duplicate crossover record key"),
        ("row-run-id", "different run_id"),
        ("row-digest", "different manifest digest"),
        ("run-digest", "run record manifest digest"),
        ("run-failed", "must have succeeded"),
        ("run-dirty", "dirty source tree"),
        ("nonavailability-missing", "non-availability record has missing outputs"),
        ("aggregate-count", "sequence_count disagrees"),
        ("aggregate-replicates", "replicate count disagrees"),
        ("aggregate-confidence", "confidence disagrees"),
        ("aggregate-contributors", "contributing_sequence_count"),
        ("aggregate-defined", "defined bootstrap count"),
        ("aggregate-estimate", "point estimate disagrees"),
        ("crossover-maximum", "tested maximum disagrees"),
        ("crossover-count", "crossover sequence_count"),
        ("crossover-replicates", "crossover replicate count"),
        ("crossover-confidence", "crossover confidence"),
        ("healthy-drift", "healthy unimodal row changes"),
        ("oracle-value", "performance oracle"),
        ("target-drop-value", "target-drop policy differs"),
    ],
)
def test_bundle_rejects_cross_record_integrity_failures(
    manifest_data: dict[str, Any],
    mutation: str,
    message: str,
) -> None:
    manifest = _small_analytic(manifest_data)
    metrics, aggregates, crossovers = _analytic_bundle(manifest)
    if mutation == "missing-row":
        metrics.pop()
    elif mutation == "duplicate-row":
        metrics.append(metrics[0])
    elif mutation == "wrong-sequence":
        metrics[0] = metrics[0].model_copy(update={"sequence_id": "analytic:wrong:000000"})
    elif mutation == "denominator-change":
        metrics[0] = metrics[0].model_copy(
            update={
                "eligible_object_frame_count": 11,
                "valid_object_frame_count": 11,
            }
        )
    elif mutation == "missing-aggregate":
        aggregates.pop()
    elif mutation == "missing-crossover":
        crossovers.pop()
    elif mutation == "fraction-not-representable":
        crossovers[0] = crossovers[0].model_copy(update={"bootstrap_crossing_fraction": 0.9999})
    elif mutation == "duplicate-aggregate":
        aggregates.append(aggregates[0])
    elif mutation == "duplicate-crossover":
        crossovers.append(crossovers[0])
    elif mutation == "row-run-id":
        metrics[0] = metrics[0].model_copy(update={"run_id": "run-002"})
    elif mutation == "row-digest":
        metrics[0] = metrics[0].model_copy(update={"manifest_sha256": "e" * 64})
    elif mutation == "nonavailability-missing":
        metrics[0] = metrics[0].model_copy(update={"valid_object_frame_count": 9})
    elif mutation == "aggregate-count":
        aggregates[0] = aggregates[0].model_copy(update={"sequence_count": 3})
    elif mutation == "aggregate-replicates":
        aggregates[0] = aggregates[0].model_copy(update={"bootstrap_replicates": 2001})
    elif mutation == "aggregate-confidence":
        aggregates[0] = aggregates[0].model_copy(update={"confidence_level": 0.9})
    elif mutation == "aggregate-contributors":
        aggregates[0] = aggregates[0].model_copy(update={"contributing_sequence_count": 1})
    elif mutation == "aggregate-defined":
        aggregates[0] = aggregates[0].model_copy(update={"defined_bootstrap_replicates": 1999})
    elif mutation == "aggregate-estimate":
        aggregates[0] = aggregates[0].model_copy(
            update={
                "estimate": 100.0,
                "interval_lower": 99.0,
                "interval_upper": 101.0,
            }
        )
    elif mutation == "crossover-maximum":
        crossovers[0] = crossovers[0].model_copy(update={"tested_maximum": 1.1})
    elif mutation == "crossover-count":
        crossovers[0] = crossovers[0].model_copy(update={"sequence_count": 3})
    elif mutation == "crossover-replicates":
        crossovers[0] = crossovers[0].model_copy(update={"bootstrap_replicates": 2001})
    elif mutation == "crossover-confidence":
        crossovers[0] = crossovers[0].model_copy(update={"confidence_level": 0.9})
    elif mutation == "healthy-drift":
        index = next(
            index
            for index, metric in enumerate(metrics)
            if metric.method_id == "lidar-only" and metric.severity.index > 0
        )
        metrics[index] = metrics[index].model_copy(update={"value": 0.3})
    elif mutation == "oracle-value":
        index = next(
            index
            for index, metric in enumerate(metrics)
            if metric.method_id == "performance-oracle"
        )
        metrics[index] = metrics[index].model_copy(update={"value": 1.0})
    elif mutation == "target-drop-value":
        index = next(
            index
            for index, metric in enumerate(metrics)
            if metric.method_id == "fault-target-drop-policy"
        )
        metrics[index] = metrics[index].model_copy(update={"value": 1.0})

    run = _run(manifest)
    if mutation == "run-digest":
        run = run.model_copy(update={"manifest_sha256": "e" * 64})
    elif mutation == "run-failed":
        run = run.model_copy(update={"status": "failed", "artifact_sha256": None})
    elif mutation == "run-dirty":
        run = run.model_copy(update={"source_dirty": True})
    with pytest.raises(ResultBundleValidationError, match=message):
        validate_result_bundle(
            manifest,
            run,
            metrics,
            aggregates,
            crossovers,
        )


def test_bundle_rejects_manifest_grid_mismatch(
    manifest_data: dict[str, Any],
) -> None:
    manifest = _small_analytic(manifest_data)
    metrics, aggregates, crossovers = _analytic_bundle(manifest)
    severity = metrics[-1].severity.model_copy(update={"magnitude": 0.75})
    metrics[-1] = metrics[-1].model_copy(update={"severity": severity})

    with pytest.raises(ResultBundleValidationError, match=r"outside|unexpected"):
        validate_result_bundle(
            manifest,
            _run(manifest),
            metrics,
            aggregates,
            crossovers,
        )


def test_valid_availability_bundle_enforces_explicit_missingness(
    manifest_data: dict[str, Any],
) -> None:
    manifest = _availability_manifest(manifest_data)
    metrics, aggregates = _availability_bundle(manifest)

    validate_result_bundle(manifest, _run(manifest), metrics, aggregates, ())


def test_availability_bundle_rejects_rate_count_disagreement(
    manifest_data: dict[str, Any],
) -> None:
    manifest = _availability_manifest(manifest_data)
    metrics, aggregates = _availability_bundle(manifest)
    coverage_index = next(
        index
        for index, metric in enumerate(metrics)
        if metric.metric_name == "coverage" and metric.severity.magnitude == 1.0
    )
    metrics[coverage_index] = metrics[coverage_index].model_copy(update={"value": 0.5})

    with pytest.raises(ResultBundleValidationError, match="coverage disagrees"):
        validate_result_bundle(manifest, _run(manifest), metrics, aggregates, ())


def _replace_availability_triplet(
    metrics: list[MetricRecordV1Alpha1],
    *,
    magnitude: float,
    method: str,
    eligible: int,
    valid: int,
    sequence_id: str | None = None,
) -> None:
    for index, metric in enumerate(metrics):
        if (
            metric.severity.magnitude != magnitude
            or metric.method_id != method
            or (sequence_id is not None and metric.sequence_id != sequence_id)
        ):
            continue
        update: dict[str, object] = {
            "eligible_object_frame_count": eligible,
            "valid_object_frame_count": valid,
        }
        if metric.metric_name == "coverage":
            update["value"] = valid / eligible
        elif metric.metric_name == "undefined-output-rate":
            update["value"] = (eligible - valid) / eligible
        else:
            update["status"] = "ok" if valid else "undefined"
            update["value"] = 0.25 if valid else None
        metrics[index] = metric.model_copy(update=update)


def test_availability_bundle_suppresses_conditional_point_without_interval_support(
    manifest_data: dict[str, Any],
) -> None:
    manifest = _availability_manifest(
        manifest_data,
        probabilities=(0.0, 0.5, 1.0),
    )
    metrics, aggregates = _availability_bundle(manifest)
    unavailable_sequence = expected_sequence_ids(manifest)[1]
    for method in ("camera-only", "fixed-fusion"):
        _replace_availability_triplet(
            metrics,
            magnitude=0.5,
            method=method,
            eligible=10,
            valid=0,
            sequence_id=unavailable_sequence,
        )

    for index, aggregate in enumerate(aggregates):
        if aggregate.severity.magnitude != 0.5 or aggregate.method_id not in {
            "camera-only",
            "fixed-fusion",
        }:
            continue
        if aggregate.metric_name == "coverage":
            update: dict[str, object] = {
                "estimate": 0.5,
                "interval_lower": 0.0,
                "interval_upper": 1.0,
                "contributing_sequence_count": 2,
                "defined_bootstrap_replicates": 200,
            }
        elif aggregate.metric_name == "undefined-output-rate":
            update = {
                "estimate": 0.5,
                "interval_lower": 0.0,
                "interval_upper": 1.0,
                "contributing_sequence_count": 2,
                "defined_bootstrap_replicates": 200,
            }
        else:
            update = {
                "status": "undefined",
                "estimate": None,
                "interval_lower": None,
                "interval_upper": None,
                "contributing_sequence_count": 1,
                "defined_bootstrap_replicates": 156,
            }
        aggregates[index] = aggregate.model_copy(update=update)

    validate_result_bundle(manifest, _run(manifest), metrics, aggregates, ())


def test_unexpected_aggregate_is_reported_without_recomputation_crash(
    manifest_data: dict[str, Any],
) -> None:
    manifest = _availability_manifest(manifest_data)
    metrics, aggregates = _availability_bundle(manifest)
    analytic_manifest = _small_analytic(manifest_data)
    _, analytic_aggregates, _ = _analytic_bundle(analytic_manifest)
    unexpected = next(
        aggregate
        for aggregate in analytic_aggregates
        if aggregate.metric_name == "fused-minus-healthy"
    ).model_copy(update={"manifest_sha256": sha256_digest(manifest)})
    aggregates.append(unexpected)

    with pytest.raises(ResultBundleValidationError, match="unexpected aggregate record key"):
        validate_result_bundle(manifest, _run(manifest), metrics, aggregates, ())


def test_unexpected_crossover_is_reported_without_recomputation_crash(
    manifest_data: dict[str, Any],
) -> None:
    manifest = _small_analytic(manifest_data)
    metrics, aggregates, crossovers = _analytic_bundle(manifest)
    crossovers.append(
        CrossoverRecordV1Alpha1(
            schema="ffb.crossover/v1alpha1",
            run_id="run-001",
            manifest_sha256=sha256_digest(manifest),
            fault_family="increased-noise-underreported",
            fault_axis="xy",
            direction="increase",
            severity_unit="std-scale",
            status="not-observed",
            point_curve_crossed=False,
            point_estimate=None,
            interval_lower=2.0,
            interval_upper="positive-infinity",
            tested_maximum=2.0,
            censoring="right-above-tested-maximum",
            bootstrap_crossing_fraction=0.0,
            sequence_count=2,
            bootstrap_replicates=2000,
            confidence_level=0.95,
            interval_method="right-censored-percentile",
        )
    )

    with pytest.raises(ResultBundleValidationError, match="unexpected crossover record key"):
        validate_result_bundle(
            manifest,
            _run(manifest),
            metrics,
            aggregates,
            crossovers,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("triplet-counts", "triplet has inconsistent counts"),
        ("undefined-rate", "undefined-output-rate disagrees"),
        ("conditional-status", "conditional loss status disagrees"),
        ("eligible-method", "eligible counts differ by method"),
        ("healthy-missing", "healthy modality is not fully available"),
        ("target-drop-missing", "target-drop policy is not fully available"),
        ("fixed-mismatch", "fixed-fusion availability differs"),
        ("identity-missing", "dropout identity is not fully available"),
        ("full-dropout-output", "full dropout still has target outputs"),
        ("target-drop-loss", "target-drop policy differs from its source method"),
    ],
)
def test_availability_bundle_rejects_method_and_denominator_contradictions(
    manifest_data: dict[str, Any],
    mutation: str,
    message: str,
) -> None:
    manifest = _availability_manifest(manifest_data)
    metrics, aggregates = _availability_bundle(manifest)
    if mutation == "triplet-counts":
        index = next(
            index for index, metric in enumerate(metrics) if metric.metric_name == "coverage"
        )
        metrics[index] = metrics[index].model_copy(
            update={
                "eligible_object_frame_count": 11,
                "valid_object_frame_count": 11,
                "value": 1.0,
            }
        )
    elif mutation == "undefined-rate":
        index = next(
            index
            for index, metric in enumerate(metrics)
            if metric.metric_name == "undefined-output-rate"
        )
        metrics[index] = metrics[index].model_copy(update={"value": 0.5})
    elif mutation == "conditional-status":
        index = next(
            index
            for index, metric in enumerate(metrics)
            if metric.metric_name == "conditional-matched-center-mse"
        )
        metrics[index] = metrics[index].model_copy(update={"status": "undefined", "value": None})
    elif mutation == "eligible-method":
        _replace_availability_triplet(
            metrics,
            magnitude=0.0,
            method="lidar-only",
            eligible=11,
            valid=11,
        )
    elif mutation == "healthy-missing":
        _replace_availability_triplet(
            metrics,
            magnitude=1.0,
            method="lidar-only",
            eligible=10,
            valid=9,
        )
    elif mutation == "target-drop-missing":
        _replace_availability_triplet(
            metrics,
            magnitude=1.0,
            method="fault-target-drop-policy",
            eligible=10,
            valid=9,
        )
    elif mutation == "fixed-mismatch":
        _replace_availability_triplet(
            metrics,
            magnitude=1.0,
            method="fixed-fusion",
            eligible=10,
            valid=1,
        )
    elif mutation == "identity-missing":
        for method in ("camera-only", "fixed-fusion"):
            _replace_availability_triplet(
                metrics,
                magnitude=0.0,
                method=method,
                eligible=10,
                valid=9,
            )
    elif mutation == "full-dropout-output":
        for method in ("camera-only", "fixed-fusion"):
            _replace_availability_triplet(
                metrics,
                magnitude=1.0,
                method=method,
                eligible=10,
                valid=1,
            )
    else:
        index = next(
            index
            for index, metric in enumerate(metrics)
            if metric.severity.magnitude == 1.0
            and metric.method_id == "fault-target-drop-policy"
            and metric.metric_name == "conditional-matched-center-mse"
        )
        metrics[index] = metrics[index].model_copy(update={"value": 0.5})

    with pytest.raises(ResultBundleValidationError, match=message):
        validate_result_bundle(manifest, _run(manifest), metrics, aggregates, ())


def test_availability_bundle_rejects_non_nested_dropout_counts(
    manifest_data: dict[str, Any],
) -> None:
    manifest = _availability_manifest(
        manifest_data,
        probabilities=(0.0, 0.25, 0.5, 1.0),
    )
    metrics, aggregates = _availability_bundle(manifest)
    for method in ("camera-only", "fixed-fusion"):
        _replace_availability_triplet(
            metrics,
            magnitude=0.25,
            method=method,
            eligible=10,
            valid=5,
        )
        _replace_availability_triplet(
            metrics,
            magnitude=0.5,
            method=method,
            eligible=10,
            valid=6,
        )

    with pytest.raises(ResultBundleValidationError, match="increases with dropout"):
        validate_result_bundle(manifest, _run(manifest), metrics, aggregates, ())
