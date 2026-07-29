"""Manifest-aware integrity checks over one complete result bundle."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AdditivePositionBiasFault,
    AnalyticCrossoverManifest,
    AnalyticSource,
    AvailabilityControlManifest,
    CalibrationTranslationFault,
    CalibrationYawFault,
    CommonModeControlManifest,
    CommonModePositionBiasFault,
    CorrectlyReportedNoiseFault,
    DropoutFault,
    ExperimentManifestV1Alpha1,
    GeometryCrossoverManifest,
    ProceduralSource,
    TimestampOffsetFault,
    UnderreportedNoiseFault,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    AggregateMetricRecordV1Alpha1,
    CrossoverRecordV1Alpha1,
    LocalizationMetricRecord,
    MetricRecordV1Alpha1,
    RateMetricRecord,
    RunRecordV1Alpha1,
)
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

type AnyFault = (
    AdditivePositionBiasFault
    | CorrectlyReportedNoiseFault
    | UnderreportedNoiseFault
    | CalibrationTranslationFault
    | CalibrationYawFault
    | TimestampOffsetFault
    | CommonModePositionBiasFault
    | DropoutFault
)
type IntArray = npt.NDArray[np.int64]

_RECONCILIATION_ABS_TOLERANCE = 1e-12


@dataclass(frozen=True, order=True)
class ConditionKey:
    """Exact manifest-grid coordinate used to index result rows."""

    fault_family: str
    fault_axis: str
    severity_index: int
    magnitude: float
    direction: str
    unit: str


class ResultBundleValidationError(ValueError):
    """All deterministic integrity failures found in a result bundle."""

    def __init__(self, messages: Sequence[str]) -> None:
        self.messages = tuple(sorted(messages))
        rendered = "\n".join(f"- {message}" for message in self.messages)
        super().__init__(f"invalid result bundle:\n{rendered}")


def _matches_recomputed_float(actual: float | None, expected: float | None) -> bool:
    if actual is None or expected is None:
        return actual is expected
    return math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=_RECONCILIATION_ABS_TOLERANCE,
    )


def expected_sequence_ids(manifest: ExperimentManifestV1Alpha1) -> tuple[str, ...]:
    """Expand the source into the exact public sequence identifiers."""

    source = manifest.source
    if isinstance(source, AnalyticSource):
        return tuple(
            f"analytic:{source.case_id}:{index:06d}" for index in range(source.sequence_count)
        )
    if isinstance(source, ProceduralSource):
        return tuple(
            f"procedural:{source.profile_id}:{source.split}:{index:06d}"
            for index in range(source.sequence_count)
        )
    return tuple(f"nuscenes:{scene_name}" for scene_name in source.scene_names)


def _fault_grid(fault: AnyFault) -> tuple[float, ...]:
    if isinstance(
        fault,
        (
            AdditivePositionBiasFault,
            CalibrationTranslationFault,
            CommonModePositionBiasFault,
        ),
    ):
        return fault.magnitude_values_m
    if isinstance(fault, CalibrationYawFault):
        return fault.magnitude_values_rad
    if isinstance(fault, TimestampOffsetFault):
        return fault.magnitude_values_s
    if isinstance(
        fault,
        (CorrectlyReportedNoiseFault, UnderreportedNoiseFault),
    ):
        return fault.std_scale_values
    return fault.probability_values


def _nonidentity_directions(fault: AnyFault) -> tuple[str, ...]:
    if isinstance(
        fault,
        (
            CorrectlyReportedNoiseFault,
            UnderreportedNoiseFault,
            DropoutFault,
        ),
    ):
        return ("increase",)
    return ("negative", "positive")


def expected_conditions(
    manifest: ExperimentManifestV1Alpha1,
) -> tuple[ConditionKey, ...]:
    """Expand identity and non-identity fault coordinates in stable order."""

    fault = manifest.fault_sweep
    grid = _fault_grid(fault)
    conditions = [
        ConditionKey(
            fault_family=fault.kind,
            fault_axis=fault.axis,
            severity_index=0,
            magnitude=grid[0],
            direction="identity",
            unit=fault.unit,
        )
    ]
    for index, magnitude in enumerate(grid[1:], start=1):
        conditions.extend(
            ConditionKey(
                fault_family=fault.kind,
                fault_axis=fault.axis,
                severity_index=index,
                magnitude=magnitude,
                direction=direction,
                unit=fault.unit,
            )
            for direction in _nonidentity_directions(fault)
        )
    return tuple(conditions)


def _metric_condition(record: MetricRecordV1Alpha1) -> ConditionKey:
    return ConditionKey(
        fault_family=record.fault_family,
        fault_axis=record.fault_axis,
        severity_index=record.severity.index,
        magnitude=record.severity.magnitude,
        direction=record.severity.direction,
        unit=record.severity.unit,
    )


def _aggregate_condition(record: AggregateMetricRecordV1Alpha1) -> ConditionKey:
    return ConditionKey(
        fault_family=record.fault_family,
        fault_axis=record.fault_axis,
        severity_index=record.severity.index,
        magnitude=record.severity.magnitude,
        direction=record.severity.direction,
        unit=record.severity.unit,
    )


def _expected_pairs(
    manifest: ExperimentManifestV1Alpha1,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    methods = tuple(str(method) for method in manifest.methods)
    if isinstance(manifest, AvailabilityControlManifest):
        metrics = (
            "coverage",
            "conditional-matched-center-mse",
            "undefined-output-rate",
        )
        pairs = tuple((method, metric) for method in methods for metric in metrics)
        return pairs, pairs

    losses = tuple((method, "matched-center-mse") for method in methods)
    if isinstance(
        manifest,
        (AnalyticCrossoverManifest, GeometryCrossoverManifest),
    ):
        return losses, (*losses, ("fixed-fusion", "fused-minus-healthy"))
    return losses, losses


def _record_provenance_errors(
    *,
    digest: str,
    run_id: str,
    metrics: Sequence[MetricRecordV1Alpha1],
    aggregates: Sequence[AggregateMetricRecordV1Alpha1],
    crossovers: Sequence[CrossoverRecordV1Alpha1],
) -> list[str]:
    errors: list[str] = []
    for level, records in (
        ("sequence", metrics),
        ("aggregate", aggregates),
        ("crossover", crossovers),
    ):
        for index, record in enumerate(records):
            if record.run_id != run_id:
                errors.append(f"{level} record {index} has a different run_id")
            if record.manifest_sha256 != digest:
                errors.append(f"{level} record {index} has a different manifest digest")
    return errors


def validate_result_bundle(
    manifest: ExperimentManifestV1Alpha1,
    run: RunRecordV1Alpha1,
    metrics: Sequence[MetricRecordV1Alpha1],
    aggregates: Sequence[AggregateMetricRecordV1Alpha1],
    crossovers: Sequence[CrossoverRecordV1Alpha1],
) -> None:
    """Reject incomplete, contradictory, or cross-manifest result collections."""

    errors: list[str] = []
    digest = sha256_digest(manifest)
    if run.status != "succeeded":
        errors.append("run record must have succeeded status")
    if run.source_dirty:
        errors.append("release bundle cannot use a dirty source tree")
    if run.manifest_sha256 != digest:
        errors.append("run record manifest digest does not match the manifest")

    errors.extend(
        _record_provenance_errors(
            digest=digest,
            run_id=run.run_id,
            metrics=metrics,
            aggregates=aggregates,
            crossovers=crossovers,
        )
    )

    sequence_ids = expected_sequence_ids(manifest)
    conditions = expected_conditions(manifest)
    sequence_pairs, aggregate_pairs = _expected_pairs(manifest)
    bootstrap_indices = paired_bootstrap_indices(
        seed=manifest.rng.bootstrap_seed,
        replicates=manifest.evaluation.bootstrap.replicates,
        sequence_count=len(sequence_ids),
    )

    metric_index: dict[
        tuple[str, ConditionKey, str, str],
        MetricRecordV1Alpha1,
    ] = {}
    for record in metrics:
        key = (
            record.sequence_id,
            _metric_condition(record),
            record.method_id,
            record.metric_name,
        )
        if key in metric_index:
            errors.append(f"duplicate sequence record key: {key!r}")
        else:
            metric_index[key] = record

    aggregate_index: dict[
        tuple[ConditionKey, str, str],
        AggregateMetricRecordV1Alpha1,
    ] = {}
    for record in aggregates:
        key = (
            _aggregate_condition(record),
            record.method_id,
            record.metric_name,
        )
        if key in aggregate_index:
            errors.append(f"duplicate aggregate record key: {key!r}")
        else:
            aggregate_index[key] = record

    crossover_index: dict[
        tuple[str, str, str, str],
        CrossoverRecordV1Alpha1,
    ] = {}
    for record in crossovers:
        key = (
            record.fault_family,
            record.fault_axis,
            record.direction,
            record.severity_unit,
        )
        if key in crossover_index:
            errors.append(f"duplicate crossover record key: {key!r}")
        else:
            crossover_index[key] = record

    expected_metric_keys = {
        (sequence_id, condition, method, metric)
        for sequence_id in sequence_ids
        for condition in conditions
        for method, metric in sequence_pairs
    }
    expected_aggregate_keys = {
        (condition, method, metric)
        for condition in conditions
        for method, metric in aggregate_pairs
    }
    actual_metric_keys = set(metric_index)
    actual_aggregate_keys = set(aggregate_index)
    for key in sorted(expected_metric_keys - actual_metric_keys):
        errors.append(f"missing sequence record key: {key!r}")
    for key in sorted(actual_metric_keys - expected_metric_keys):
        errors.append(f"unexpected sequence record key: {key!r}")
    for key in sorted(expected_aggregate_keys - actual_aggregate_keys):
        errors.append(f"missing aggregate record key: {key!r}")
    for key in sorted(actual_aggregate_keys - expected_aggregate_keys):
        errors.append(f"unexpected aggregate record key: {key!r}")

    expected_crossover_keys: set[tuple[str, str, str, str]] = set()
    if isinstance(
        manifest,
        (AnalyticCrossoverManifest, GeometryCrossoverManifest),
    ):
        fault = manifest.fault_sweep
        expected_crossover_keys = {
            (fault.kind, fault.axis, direction, fault.unit)
            for direction in _nonidentity_directions(fault)
        }
    actual_crossover_keys = set(crossover_index)
    for key in sorted(expected_crossover_keys - actual_crossover_keys):
        errors.append(f"missing crossover record key: {key!r}")
    for key in sorted(actual_crossover_keys - expected_crossover_keys):
        errors.append(f"unexpected crossover record key: {key!r}")

    eligible_by_sequence: dict[str, set[int]] = defaultdict(set)
    for (sequence_id, _, _, _), record in metric_index.items():
        eligible_by_sequence[sequence_id].add(record.eligible_object_frame_count)
    for sequence_id, counts in eligible_by_sequence.items():
        if len(counts) != 1:
            errors.append(f"eligible denominator changes within sequence {sequence_id!r}")

    if isinstance(manifest, AvailabilityControlManifest):
        _validate_availability_rows(
            manifest=manifest,
            sequence_ids=sequence_ids,
            conditions=conditions,
            metric_index=metric_index,
            errors=errors,
        )
    else:
        for key, record in metric_index.items():
            if record.valid_object_frame_count != record.eligible_object_frame_count:
                errors.append(f"non-availability record has missing outputs: {key!r}")
            if (
                not isinstance(record, LocalizationMetricRecord)
                or record.status != "ok"
                or record.metric_name != "matched-center-mse"
            ):
                errors.append(f"invalid non-availability sequence metric: {key!r}")

    _validate_healthy_unimodal_invariance(
        manifest=manifest,
        sequence_ids=sequence_ids,
        conditions=conditions,
        metric_index=metric_index,
        errors=errors,
    )
    _validate_method_semantics(
        manifest=manifest,
        sequence_ids=sequence_ids,
        conditions=conditions,
        metric_index=metric_index,
        errors=errors,
    )
    _validate_aggregate_rows(
        manifest=manifest,
        sequence_ids=sequence_ids,
        conditions=conditions,
        expected_keys=expected_aggregate_keys,
        metric_index=metric_index,
        aggregate_index=aggregate_index,
        bootstrap_indices=bootstrap_indices,
        errors=errors,
    )
    _validate_crossover_rows(
        manifest=manifest,
        sequence_count=len(sequence_ids),
        sequence_ids=sequence_ids,
        conditions=conditions,
        expected_keys=expected_crossover_keys,
        metric_index=metric_index,
        bootstrap_indices=bootstrap_indices,
        crossover_index=crossover_index,
        errors=errors,
    )

    if errors:
        raise ResultBundleValidationError(errors)


def _validate_availability_rows(
    *,
    manifest: AvailabilityControlManifest,
    sequence_ids: tuple[str, ...],
    conditions: tuple[ConditionKey, ...],
    metric_index: dict[
        tuple[str, ConditionKey, str, str],
        MetricRecordV1Alpha1,
    ],
    errors: list[str],
) -> None:
    target_method = f"{manifest.fault_sweep.target}-only"
    healthy_method = "lidar-only" if target_method == "camera-only" else "camera-only"
    for sequence_id in sequence_ids:
        for condition in conditions:
            valid_by_method: dict[str, int] = {}
            eligible_by_method: dict[str, int] = {}
            for method in manifest.methods:
                rows = {
                    metric_name: metric_index.get((sequence_id, condition, method, metric_name))
                    for metric_name in (
                        "coverage",
                        "conditional-matched-center-mse",
                        "undefined-output-rate",
                    )
                }
                if any(record is None for record in rows.values()):
                    continue
                coverage = rows["coverage"]
                conditional = rows["conditional-matched-center-mse"]
                undefined_rate = rows["undefined-output-rate"]
                assert isinstance(coverage, RateMetricRecord)
                assert isinstance(conditional, LocalizationMetricRecord)
                assert isinstance(undefined_rate, RateMetricRecord)

                counts = {
                    (
                        record.eligible_object_frame_count,
                        record.valid_object_frame_count,
                    )
                    for record in (coverage, conditional, undefined_rate)
                }
                if len(counts) != 1:
                    errors.append(
                        "availability metric triplet has inconsistent counts: "
                        f"{(sequence_id, condition, method)!r}"
                    )
                    continue
                eligible = coverage.eligible_object_frame_count
                valid = coverage.valid_object_frame_count
                eligible_by_method[method] = eligible
                valid_by_method[method] = valid
                if coverage.value != valid / eligible:
                    errors.append(
                        f"coverage disagrees with counts: {(sequence_id, condition, method)!r}"
                    )
                if undefined_rate.value != (eligible - valid) / eligible:
                    errors.append(
                        f"undefined-output-rate disagrees with counts: "
                        f"{(sequence_id, condition, method)!r}"
                    )
                expected_status = "ok" if valid > 0 else "undefined"
                if conditional.status != expected_status:
                    errors.append(
                        f"conditional loss status disagrees with valid count: "
                        f"{(sequence_id, condition, method)!r}"
                    )

            if len(set(eligible_by_method.values())) > 1:
                errors.append(
                    f"availability eligible counts differ by method: {(sequence_id, condition)!r}"
                )
            if not valid_by_method:
                continue
            eligible = next(iter(eligible_by_method.values()))
            if valid_by_method.get(healthy_method) != eligible:
                errors.append(
                    f"healthy modality is not fully available: {(sequence_id, condition)!r}"
                )
            if valid_by_method.get("fault-target-drop-policy") != eligible:
                errors.append(
                    f"target-drop policy is not fully available: {(sequence_id, condition)!r}"
                )
            if valid_by_method.get("fixed-fusion") != valid_by_method.get(target_method):
                errors.append(
                    f"fixed-fusion availability differs from the target modality: "
                    f"{(sequence_id, condition)!r}"
                )
            if condition.magnitude == 0.0 and valid_by_method.get(target_method) != eligible:
                errors.append(
                    f"dropout identity is not fully available: {(sequence_id, condition)!r}"
                )
            if condition.magnitude == 1.0 and valid_by_method.get(target_method) != 0:
                errors.append(
                    f"full dropout still has target outputs: {(sequence_id, condition)!r}"
                )

        previous_valid: int | None = None
        for condition in sorted(
            conditions,
            key=lambda item: item.severity_index,
        ):
            coverage = metric_index.get((sequence_id, condition, target_method, "coverage"))
            if not isinstance(coverage, RateMetricRecord):
                continue
            if previous_valid is not None and coverage.valid_object_frame_count > previous_valid:
                errors.append(
                    f"target availability increases with dropout probability: "
                    f"{(sequence_id, condition)!r}"
                )
            previous_valid = coverage.valid_object_frame_count


def _validate_healthy_unimodal_invariance(
    *,
    manifest: ExperimentManifestV1Alpha1,
    sequence_ids: tuple[str, ...],
    conditions: tuple[ConditionKey, ...],
    metric_index: dict[
        tuple[str, ConditionKey, str, str],
        MetricRecordV1Alpha1,
    ],
    errors: list[str],
) -> None:
    if isinstance(manifest, CommonModeControlManifest):
        return
    healthy_method = "lidar-only" if manifest.fault_sweep.target == "camera" else "camera-only"
    metric_names = (
        (
            "coverage",
            "conditional-matched-center-mse",
            "undefined-output-rate",
        )
        if isinstance(manifest, AvailabilityControlManifest)
        else ("matched-center-mse",)
    )
    identity = conditions[0]
    for sequence_id in sequence_ids:
        for metric_name in metric_names:
            reference = metric_index.get((sequence_id, identity, healthy_method, metric_name))
            if reference is None:
                continue
            for condition in conditions[1:]:
                record = metric_index.get((sequence_id, condition, healthy_method, metric_name))
                if record is None:
                    continue
                if (
                    record.status != reference.status
                    or record.value != reference.value
                    or record.eligible_object_frame_count != reference.eligible_object_frame_count
                    or record.valid_object_frame_count != reference.valid_object_frame_count
                ):
                    errors.append(
                        "healthy unimodal row changes across fault conditions: "
                        f"{(sequence_id, condition, metric_name)!r}"
                    )


def _validate_method_semantics(
    *,
    manifest: ExperimentManifestV1Alpha1,
    sequence_ids: tuple[str, ...],
    conditions: tuple[ConditionKey, ...],
    metric_index: dict[
        tuple[str, ConditionKey, str, str],
        MetricRecordV1Alpha1,
    ],
    errors: list[str],
) -> None:
    if isinstance(manifest, CommonModeControlManifest):
        return

    healthy_method = "lidar-only" if manifest.fault_sweep.target == "camera" else "camera-only"
    if isinstance(manifest, AvailabilityControlManifest):
        for sequence_id in sequence_ids:
            for condition in conditions:
                source_method = "fixed-fusion" if condition.severity_index == 0 else healthy_method
                for metric_name in (
                    "coverage",
                    "conditional-matched-center-mse",
                    "undefined-output-rate",
                ):
                    target_drop = metric_index.get(
                        (
                            sequence_id,
                            condition,
                            "fault-target-drop-policy",
                            metric_name,
                        )
                    )
                    source = metric_index.get((sequence_id, condition, source_method, metric_name))
                    if target_drop is None or source is None:
                        continue
                    if (
                        target_drop.status != source.status
                        or target_drop.value != source.value
                        or target_drop.eligible_object_frame_count
                        != source.eligible_object_frame_count
                        or target_drop.valid_object_frame_count != source.valid_object_frame_count
                    ):
                        errors.append(
                            "availability target-drop policy differs from its source method: "
                            f"{(sequence_id, condition, metric_name)!r}"
                        )
        return

    assert isinstance(
        manifest,
        (AnalyticCrossoverManifest, GeometryCrossoverManifest),
    )
    for sequence_id in sequence_ids:
        for condition in conditions:
            rows = {
                method: metric_index.get((sequence_id, condition, method, "matched-center-mse"))
                for method in manifest.methods
            }
            if any(row is None for row in rows.values()):
                continue
            camera = rows["camera-only"]
            lidar = rows["lidar-only"]
            fused = rows["fixed-fusion"]
            target_drop = rows["fault-target-drop-policy"]
            oracle = rows["performance-oracle"]
            assert isinstance(camera, LocalizationMetricRecord)
            assert isinstance(lidar, LocalizationMetricRecord)
            assert isinstance(fused, LocalizationMetricRecord)
            assert isinstance(target_drop, LocalizationMetricRecord)
            assert isinstance(oracle, LocalizationMetricRecord)
            assert camera.value is not None
            assert lidar.value is not None
            assert fused.value is not None
            assert target_drop.value is not None
            assert oracle.value is not None

            expected_oracle = min(camera.value, lidar.value, fused.value)
            if oracle.value != expected_oracle:
                errors.append(
                    f"performance oracle is not the sequence candidate minimum: "
                    f"{(sequence_id, condition)!r}"
                )
            source = fused if condition.severity_index == 0 else rows[healthy_method]
            assert isinstance(source, LocalizationMetricRecord)
            if (
                target_drop.value != source.value
                or target_drop.eligible_object_frame_count != source.eligible_object_frame_count
                or target_drop.valid_object_frame_count != source.valid_object_frame_count
            ):
                errors.append(
                    f"target-drop policy differs from its source method: "
                    f"{(sequence_id, condition)!r}"
                )


def _validate_aggregate_rows(
    *,
    manifest: ExperimentManifestV1Alpha1,
    sequence_ids: tuple[str, ...],
    conditions: tuple[ConditionKey, ...],
    expected_keys: set[tuple[ConditionKey, str, str]],
    metric_index: dict[
        tuple[str, ConditionKey, str, str],
        MetricRecordV1Alpha1,
    ],
    aggregate_index: dict[
        tuple[ConditionKey, str, str],
        AggregateMetricRecordV1Alpha1,
    ],
    bootstrap_indices: IntArray,
    errors: list[str],
) -> None:
    bootstrap = manifest.evaluation.bootstrap
    sequence_count = len(sequence_ids)
    for key, record in aggregate_index.items():
        if key not in expected_keys:
            continue
        condition, method, metric_name = key
        if record.sequence_count != sequence_count:
            errors.append(f"aggregate sequence_count disagrees with source: {key!r}")
        if record.bootstrap_replicates != bootstrap.replicates:
            errors.append(f"aggregate replicate count disagrees with manifest: {key!r}")
        if record.confidence_level != bootstrap.confidence_level:
            errors.append(f"aggregate confidence disagrees with manifest: {key!r}")

        if metric_name == "conditional-matched-center-mse":
            contributing = sum(
                1
                for sequence_id in sequence_ids
                if (row := metric_index.get((sequence_id, condition, method, metric_name)))
                is not None
                and row.valid_object_frame_count > 0
            )
        else:
            contributing = sequence_count
        if record.contributing_sequence_count != contributing:
            errors.append(f"aggregate contributing_sequence_count is inconsistent: {key!r}")

        point_estimate = _aggregate_point_estimate(
            manifest=manifest,
            sequence_ids=sequence_ids,
            condition=condition,
            method=method,
            metric_name=metric_name,
            metric_index=metric_index,
        )

        bootstrap_values = _aggregate_bootstrap_values(
            manifest=manifest,
            sequence_ids=sequence_ids,
            condition=condition,
            method=method,
            metric_name=metric_name,
            metric_index=metric_index,
            bootstrap_indices=bootstrap_indices,
        )
        if bootstrap_values is None:
            continue
        defined_replicates = int(bootstrap_values.size)
        if record.defined_bootstrap_replicates != defined_replicates:
            errors.append(f"aggregate defined bootstrap count is inconsistent: {key!r}")
        alpha = 1.0 - bootstrap.confidence_level
        has_two_sided_support = defined_replicates / bootstrap.replicates > 1.0 - alpha / 2.0
        expected_status = (
            "ok" if point_estimate is not None and has_two_sided_support else "undefined"
        )
        published_estimate = point_estimate if expected_status == "ok" else None
        if not _matches_recomputed_float(record.estimate, published_estimate):
            errors.append(f"aggregate point estimate disagrees with sequence rows: {key!r}")
        if record.status != expected_status:
            errors.append(f"aggregate status disagrees with bootstrap evidence: {key!r}")
        if expected_status == "ok":
            expected_lower, expected_upper = percentile_interval(
                bootstrap_values,
                confidence_level=bootstrap.confidence_level,
            )
            if not _matches_recomputed_float(
                record.interval_lower,
                expected_lower,
            ) or not _matches_recomputed_float(record.interval_upper, expected_upper):
                errors.append(f"aggregate interval disagrees with bootstrap evidence: {key!r}")
        elif record.interval_lower is not None or record.interval_upper is not None:
            errors.append(f"undefined aggregate has a reported interval: {key!r}")

    expected_condition_set = set(conditions)
    for condition, _, _ in aggregate_index:
        if condition not in expected_condition_set:
            errors.append(f"aggregate condition is outside the manifest grid: {condition!r}")


def _aggregate_point_estimate(
    *,
    manifest: ExperimentManifestV1Alpha1,
    sequence_ids: tuple[str, ...],
    condition: ConditionKey,
    method: str,
    metric_name: str,
    metric_index: dict[
        tuple[str, ConditionKey, str, str],
        MetricRecordV1Alpha1,
    ],
) -> float | None:
    if metric_name == "fused-minus-healthy":
        assert isinstance(
            manifest,
            (AnalyticCrossoverManifest, GeometryCrossoverManifest),
        )
        healthy_method = "lidar-only" if manifest.fault_sweep.target == "camera" else "camera-only"
        differences: list[float] = []
        for sequence_id in sequence_ids:
            fused = metric_index.get((sequence_id, condition, "fixed-fusion", "matched-center-mse"))
            healthy = metric_index.get(
                (sequence_id, condition, healthy_method, "matched-center-mse")
            )
            if (
                not isinstance(fused, LocalizationMetricRecord)
                or not isinstance(healthy, LocalizationMetricRecord)
                or fused.value is None
                or healthy.value is None
            ):
                return None
            differences.append(fused.value - healthy.value)
        return math.fsum(differences) / len(sequence_ids)

    rows = [
        metric_index.get((sequence_id, condition, method, metric_name))
        for sequence_id in sequence_ids
    ]
    if any(row is None for row in rows):
        return None
    present_rows = [row for row in rows if row is not None]

    if metric_name == "matched-center-mse":
        values = [
            row.value
            for row in present_rows
            if isinstance(row, LocalizationMetricRecord) and row.value is not None
        ]
        if len(values) != len(sequence_ids):
            return None
        return math.fsum(values) / len(values)

    eligible = sum(row.eligible_object_frame_count for row in present_rows)
    valid = sum(row.valid_object_frame_count for row in present_rows)
    if metric_name == "coverage":
        return valid / eligible
    if metric_name == "undefined-output-rate":
        return (eligible - valid) / eligible
    if valid == 0:
        return None
    weighted_losses = [
        row.value * row.valid_object_frame_count
        for row in present_rows
        if isinstance(row, LocalizationMetricRecord) and row.value is not None
    ]
    return math.fsum(weighted_losses) / valid


def _aggregate_bootstrap_values(
    *,
    manifest: ExperimentManifestV1Alpha1,
    sequence_ids: tuple[str, ...],
    condition: ConditionKey,
    method: str,
    metric_name: str,
    metric_index: dict[
        tuple[str, ConditionKey, str, str],
        MetricRecordV1Alpha1,
    ],
    bootstrap_indices: IntArray,
) -> npt.NDArray[np.float64] | None:
    if metric_name == "fused-minus-healthy":
        assert isinstance(
            manifest,
            (AnalyticCrossoverManifest, GeometryCrossoverManifest),
        )
        healthy_method = "lidar-only" if manifest.fault_sweep.target == "camera" else "camera-only"
        contrasts: list[float] = []
        for sequence_id in sequence_ids:
            fused = metric_index.get((sequence_id, condition, "fixed-fusion", "matched-center-mse"))
            healthy = metric_index.get(
                (sequence_id, condition, healthy_method, "matched-center-mse")
            )
            if (
                not isinstance(fused, LocalizationMetricRecord)
                or not isinstance(healthy, LocalizationMetricRecord)
                or fused.value is None
                or healthy.value is None
            ):
                return None
            contrasts.append(fused.value - healthy.value)
        return bootstrap_mean(
            np.asarray(contrasts, dtype=np.float64),
            bootstrap_indices,
        )

    rows = [
        metric_index.get((sequence_id, condition, method, metric_name))
        for sequence_id in sequence_ids
    ]
    if any(row is None for row in rows):
        return None
    present_rows = [row for row in rows if row is not None]
    if metric_name == "matched-center-mse":
        values = [
            row.value
            for row in present_rows
            if isinstance(row, LocalizationMetricRecord) and row.value is not None
        ]
        if len(values) != len(sequence_ids):
            return None
        return bootstrap_mean(
            np.asarray(values, dtype=np.float64),
            bootstrap_indices,
        )

    eligible = np.asarray(
        [row.eligible_object_frame_count for row in present_rows],
        dtype=np.int64,
    )
    valid = np.asarray(
        [row.valid_object_frame_count for row in present_rows],
        dtype=np.int64,
    )
    if metric_name == "coverage":
        return bootstrap_count_ratio(valid, eligible, bootstrap_indices)
    if metric_name == "undefined-output-rate":
        return bootstrap_count_ratio(eligible - valid, eligible, bootstrap_indices)
    loss_sums = np.asarray(
        [
            (
                row.value * row.valid_object_frame_count
                if isinstance(row, LocalizationMetricRecord) and row.value is not None
                else 0.0
            )
            for row in present_rows
        ],
        dtype=np.float64,
    )
    return bootstrap_conditional_loss(loss_sums, valid, bootstrap_indices)


def _validate_crossover_rows(
    *,
    manifest: ExperimentManifestV1Alpha1,
    sequence_count: int,
    sequence_ids: tuple[str, ...],
    conditions: tuple[ConditionKey, ...],
    expected_keys: set[tuple[str, str, str, str]],
    metric_index: dict[
        tuple[str, ConditionKey, str, str],
        MetricRecordV1Alpha1,
    ],
    bootstrap_indices: IntArray,
    crossover_index: dict[
        tuple[str, str, str, str],
        CrossoverRecordV1Alpha1,
    ],
    errors: list[str],
) -> None:
    if not isinstance(
        manifest,
        (AnalyticCrossoverManifest, GeometryCrossoverManifest),
    ):
        return
    bootstrap = manifest.evaluation.bootstrap
    tested_maximum = _fault_grid(manifest.fault_sweep)[-1]
    for key, record in crossover_index.items():
        if key not in expected_keys:
            continue
        if record.tested_maximum != tested_maximum:
            errors.append(f"crossover tested maximum disagrees with manifest: {key!r}")
        if record.sequence_count != sequence_count:
            errors.append(f"crossover sequence_count disagrees with source: {key!r}")
        if record.bootstrap_replicates != bootstrap.replicates:
            errors.append(f"crossover replicate count disagrees with manifest: {key!r}")
        if record.confidence_level != bootstrap.confidence_level:
            errors.append(f"crossover confidence disagrees with manifest: {key!r}")
        crossings = round(record.bootstrap_crossing_fraction * record.bootstrap_replicates)
        representable = crossings / record.bootstrap_replicates
        if record.bootstrap_crossing_fraction != representable:
            errors.append(
                f"crossover fraction is not representable by its replicate count: {key!r}"
            )

        curve_conditions = [
            condition
            for condition in conditions
            if condition.direction in {"identity", record.direction}
        ]
        curve_conditions.sort(key=lambda condition: condition.severity_index)
        contrasts: list[list[float]] = []
        complete = True
        healthy_method = "lidar-only" if manifest.fault_sweep.target == "camera" else "camera-only"
        for condition in curve_conditions:
            severity_values: list[float] = []
            for sequence_id in sequence_ids:
                fused = metric_index.get(
                    (sequence_id, condition, "fixed-fusion", "matched-center-mse")
                )
                healthy = metric_index.get(
                    (sequence_id, condition, healthy_method, "matched-center-mse")
                )
                if (
                    not isinstance(fused, LocalizationMetricRecord)
                    or not isinstance(healthy, LocalizationMetricRecord)
                    or fused.value is None
                    or healthy.value is None
                ):
                    complete = False
                    break
                severity_values.append(fused.value - healthy.value)
            if not complete:
                break
            contrasts.append(severity_values)
        if not complete:
            continue

        magnitudes = np.asarray(
            [condition.magnitude for condition in curve_conditions],
            dtype=np.float64,
        )
        contrast_array = np.asarray(contrasts, dtype=np.float64)
        point_values = np.asarray(
            [math.fsum(severity_values) / sequence_count for severity_values in contrasts],
            dtype=np.float64,
        )
        point_root = first_zero_crossover(
            magnitudes,
            pava_non_decreasing(point_values),
            zero_tolerance=manifest.evaluation.crossover.zero_tolerance_m2,
        )
        roots = bootstrap_crossover_roots(
            magnitudes=magnitudes,
            sequence_contrasts=contrast_array,
            indices=bootstrap_indices,
            zero_tolerance=manifest.evaluation.crossover.zero_tolerance_m2,
        )
        finite_roots = [root for root in roots if root is not None]
        crossing_count = len(finite_roots)
        crossing_fraction = crossing_count / bootstrap.replicates
        point_crossed = point_root is not None
        expected_status = bootstrap_crossover_status(
            point_crossed=point_crossed,
            crossing_count=crossing_count,
            bootstrap_replicates=bootstrap.replicates,
        )

        if record.point_curve_crossed != point_crossed:
            errors.append(f"crossover point-curve status disagrees with sequence rows: {key!r}")
        if not _matches_recomputed_float(record.point_estimate, point_root):
            errors.append(f"crossover point estimate disagrees with sequence rows: {key!r}")
        if record.bootstrap_crossing_fraction != crossing_fraction:
            errors.append(f"crossover fraction disagrees with bootstrap evidence: {key!r}")
        if record.status != expected_status:
            errors.append(f"crossover status disagrees with bootstrap evidence: {key!r}")

        if expected_status == "observed":
            roots_with_censoring = np.asarray(
                [root if root is not None else np.inf for root in roots],
                dtype=np.float64,
            )
            expected_lower, expected_upper = percentile_interval(
                roots_with_censoring,
                confidence_level=bootstrap.confidence_level,
            )
            if (
                not _matches_recomputed_float(record.interval_lower, expected_lower)
                or not isinstance(record.interval_upper, float)
                or not _matches_recomputed_float(record.interval_upper, expected_upper)
                or record.censoring != "none"
            ):
                errors.append(f"crossover interval disagrees with bootstrap evidence: {key!r}")
        elif expected_status == "not-observed":
            if (
                record.interval_lower != tested_maximum
                or record.interval_upper != "positive-infinity"
                or record.censoring != "right-above-tested-maximum"
            ):
                errors.append(f"crossover censoring disagrees with bootstrap evidence: {key!r}")
        elif (
            record.interval_lower is not None
            or record.interval_upper is not None
            or record.censoring != "mixed-bootstrap"
        ):
            errors.append(
                f"undetermined crossover fields disagree with bootstrap evidence: {key!r}"
            )
