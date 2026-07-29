from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.bundle_v1alpha1 import (
    ConditionKey,
    expected_conditions,
    expected_sequence_ids,
)
from fusion_fault_bench.contracts.io import load_manifest
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AdditivePositionBiasFault,
    AnalyticCrossoverManifest,
    MethodId,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    LocalizationMetricRecord,
    MetricRecordV1Alpha1,
    RateMetricRecord,
)
from fusion_fault_bench.evaluation import EvaluatedRecords, evaluate_analytic_records
from fusion_fault_bench.experiments.analytic import generate_analytic_sequence_metrics
from fusion_fault_bench.rng import draw_standard_normal_xy

MANIFESTS = Path("examples/manifests")
RUN_ID = "run:analytic-test"

BIAS_MANIFEST = "analytic-bias-v1alpha1.json"
CORRECT_NOISE_MANIFEST = "analytic-noise-correct-v1alpha1.json"
UNDERREPORTED_NOISE_MANIFEST = "analytic-noise-underreported-v1alpha1.json"


@lru_cache
def _manifest(name: str) -> AnalyticCrossoverManifest:
    manifest = load_manifest(MANIFESTS / name)
    assert isinstance(manifest, AnalyticCrossoverManifest)
    return manifest


@lru_cache
def _metrics(name: str) -> tuple[MetricRecordV1Alpha1, ...]:
    return generate_analytic_sequence_metrics(_manifest(name), run_id=RUN_ID)


@lru_cache
def _evaluated(name: str) -> EvaluatedRecords:
    return evaluate_analytic_records(
        _manifest(name),
        run_id=RUN_ID,
        metrics=_metrics(name),
    )


def _localization_index(
    metrics: tuple[MetricRecordV1Alpha1, ...],
) -> dict[tuple[str, int, str, MethodId], LocalizationMetricRecord]:
    index: dict[tuple[str, int, str, MethodId], LocalizationMetricRecord] = {}
    for record in metrics:
        assert isinstance(record, LocalizationMetricRecord)
        key = (
            record.sequence_id,
            record.severity.index,
            record.severity.direction,
            record.method_id,
        )
        assert key not in index
        index[key] = record
    return index


def _loss(value_xy: tuple[float, float]) -> float:
    return math.fsum((value_xy[0] * value_xy[0], value_xy[1] * value_xy[1]))


def test_bias_sequence_rows_have_exact_contractual_order_count_and_provenance() -> None:
    manifest = _manifest(BIAS_MANIFEST)
    metrics = _metrics(BIAS_MANIFEST)
    sequence_ids = expected_sequence_ids(manifest)
    conditions = expected_conditions(manifest)

    expected_keys = [
        (sequence_id, condition, method)
        for sequence_id in sequence_ids
        for condition in conditions
        for method in manifest.methods
    ]
    actual_keys: list[tuple[str, ConditionKey, MethodId]] = []
    for record in metrics:
        assert isinstance(record, LocalizationMetricRecord)
        actual_keys.append(
            (
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
        )
        assert record.run_id == RUN_ID
        assert record.manifest_sha256 == sha256_digest(manifest)
        assert record.eligible_object_frame_count == 1
        assert record.valid_object_frame_count == 1
        assert record.metric_name == "matched-center-mse"
        assert record.status == "ok"

    assert len(metrics) == 200 * 13 * 5 == 13_000
    assert actual_keys == expected_keys
    assert sum(key[1].direction == "identity" for key in actual_keys) == 200 * 5


def test_first_sequence_losses_follow_renamed_rng_bit_golden() -> None:
    metrics = _metrics(BIAS_MANIFEST)
    first_five = metrics[:5]
    assert all(isinstance(record, LocalizationMetricRecord) for record in first_five)

    # These are the float64 values represented by the frozen first-draw bit goldens.
    camera_normal = (1.731555707700361, 0.2860795487534173)
    lidar_normal = (0.47971738760077415, -1.8763282159480077)
    camera_error = (1.5 * camera_normal[0], 0.6 * camera_normal[1])
    lidar_error = (0.25 * lidar_normal[0], 0.25 * lidar_normal[1])
    camera_weight = (1.0 / 37.0, 25.0 / 169.0)
    fused_error = (
        camera_weight[0] * camera_error[0] + (1.0 - camera_weight[0]) * lidar_error[0],
        camera_weight[1] * camera_error[1] + (1.0 - camera_weight[1]) * lidar_error[1],
    )
    expected = (
        _loss(camera_error),
        _loss(lidar_error),
        _loss(fused_error),
        _loss(fused_error),
        min(_loss(camera_error), _loss(lidar_error), _loss(fused_error)),
    )

    assert [record.method_id for record in first_five] == list(_manifest(BIAS_MANIFEST).methods)
    assert [cast(LocalizationMetricRecord, record).value for record in first_five] == (
        pytest.approx(expected, rel=0.0, abs=2e-15)
    )


def test_bias_branches_reuse_each_sequence_draw_without_severity_accumulation() -> None:
    manifest = _manifest(BIAS_MANIFEST)
    index = _localization_index(_metrics(BIAS_MANIFEST))

    for sequence_id in expected_sequence_ids(manifest):
        camera_normal = draw_standard_normal_xy(
            data_master_seed=manifest.rng.data_master_seed,
            stream_name="camera",
            sequence_id=sequence_id,
            object_frame_count=1,
        )[0]
        camera_error = (
            manifest.observations.camera.actual_std_xy_m[0] * float(camera_normal[0]),
            manifest.observations.camera.actual_std_xy_m[1] * float(camera_normal[1]),
        )
        identity_camera = index[(sequence_id, 0, "identity", "camera-only")]
        identity_lidar = index[(sequence_id, 0, "identity", "lidar-only")]
        assert identity_camera.value == pytest.approx(_loss(camera_error), abs=2e-15)

        for severity_index, magnitude in enumerate(
            cast(AdditivePositionBiasFault, manifest.fault_sweep).magnitude_values_m[1:],
            start=1,
        ):
            negative_camera = index[(sequence_id, severity_index, "negative", "camera-only")]
            positive_camera = index[(sequence_id, severity_index, "positive", "camera-only")]
            assert negative_camera.value == pytest.approx(
                _loss((camera_error[0] - magnitude, camera_error[1])),
                abs=2e-14,
            )
            assert positive_camera.value == pytest.approx(
                _loss((camera_error[0] + magnitude, camera_error[1])),
                abs=2e-14,
            )
            assert (
                0.5 * (cast(float, negative_camera.value) + cast(float, positive_camera.value))
                - magnitude * magnitude
            ) == pytest.approx(identity_camera.value, rel=2e-15, abs=1e-13)
            assert (
                index[(sequence_id, severity_index, "negative", "lidar-only")].value
                == identity_lidar.value
            )
            assert (
                index[(sequence_id, severity_index, "positive", "lidar-only")].value
                == identity_lidar.value
            )


def test_noise_controls_share_actual_errors_but_use_different_reported_uncertainty() -> None:
    correct_manifest = _manifest(CORRECT_NOISE_MANIFEST)
    correct = _localization_index(_metrics(CORRECT_NOISE_MANIFEST))
    underreported = _localization_index(_metrics(UNDERREPORTED_NOISE_MANIFEST))

    for sequence_id in expected_sequence_ids(correct_manifest):
        camera_normal = draw_standard_normal_xy(
            data_master_seed=correct_manifest.rng.data_master_seed,
            stream_name="camera",
            sequence_id=sequence_id,
            object_frame_count=1,
        )[0]
        lidar_normal = draw_standard_normal_xy(
            data_master_seed=correct_manifest.rng.data_master_seed,
            stream_name="lidar",
            sequence_id=sequence_id,
            object_frame_count=1,
        )[0]
        nominal_camera = (
            np.asarray(
                correct_manifest.observations.camera.actual_std_xy_m,
                dtype=np.float64,
            )
            * camera_normal
        )
        lidar = (
            np.asarray(
                correct_manifest.observations.lidar.actual_std_xy_m,
                dtype=np.float64,
            )
            * lidar_normal
        )
        identity_camera_loss = _loss((float(nominal_camera[0]), float(nominal_camera[1])))
        identity_lidar_loss = _loss((float(lidar[0]), float(lidar[1])))

        for condition in expected_conditions(correct_manifest):
            key_prefix = (sequence_id, condition.severity_index, condition.direction)
            correct_camera = correct[(*key_prefix, "camera-only")]
            underreported_camera = underreported[(*key_prefix, "camera-only")]
            correct_lidar = correct[(*key_prefix, "lidar-only")]
            underreported_lidar = underreported[(*key_prefix, "lidar-only")]

            assert correct_camera.value == underreported_camera.value
            assert correct_lidar.value == underreported_lidar.value == identity_lidar_loss
            assert correct_camera.value == pytest.approx(
                identity_camera_loss * condition.magnitude**2,
                rel=1e-15,
                abs=1e-14,
            )

            camera = nominal_camera * condition.magnitude
            nominal_camera_variance = np.square(
                np.asarray(
                    correct_manifest.observations.camera.reported_std_xy_m,
                    dtype=np.float64,
                )
            )
            lidar_variance = np.square(
                np.asarray(
                    correct_manifest.observations.lidar.reported_std_xy_m,
                    dtype=np.float64,
                )
            )
            correct_camera_variance = nominal_camera_variance * condition.magnitude**2
            correct_weight = lidar_variance / (correct_camera_variance + lidar_variance)
            underreported_weight = lidar_variance / (nominal_camera_variance + lidar_variance)
            expected_correct_fused = correct_weight * camera + (1.0 - correct_weight) * lidar
            expected_underreported_fused = (
                underreported_weight * camera + (1.0 - underreported_weight) * lidar
            )
            assert correct[(*key_prefix, "fixed-fusion")].value == pytest.approx(
                _loss(
                    (
                        float(expected_correct_fused[0]),
                        float(expected_correct_fused[1]),
                    )
                ),
                abs=2e-15,
            )
            assert underreported[(*key_prefix, "fixed-fusion")].value == pytest.approx(
                _loss(
                    (
                        float(expected_underreported_fused[0]),
                        float(expected_underreported_fused[1]),
                    )
                ),
                abs=2e-15,
            )


@pytest.mark.parametrize(
    "manifest_name",
    [BIAS_MANIFEST, CORRECT_NOISE_MANIFEST, UNDERREPORTED_NOISE_MANIFEST],
)
def test_target_drop_and_per_sequence_oracle_method_semantics(manifest_name: str) -> None:
    manifest = _manifest(manifest_name)
    index = _localization_index(_metrics(manifest_name))
    healthy_method: MethodId = (
        "lidar-only" if manifest.fault_sweep.target == "camera" else "camera-only"
    )

    for sequence_id in expected_sequence_ids(manifest):
        for condition in expected_conditions(manifest):
            key_prefix = (sequence_id, condition.severity_index, condition.direction)
            camera = cast(float, index[(*key_prefix, "camera-only")].value)
            lidar = cast(float, index[(*key_prefix, "lidar-only")].value)
            fused = cast(float, index[(*key_prefix, "fixed-fusion")].value)
            target_drop = index[(*key_prefix, "fault-target-drop-policy")].value
            oracle = index[(*key_prefix, "performance-oracle")].value

            expected_target_drop = (
                fused
                if condition.direction == "identity"
                else cast(float, index[(*key_prefix, healthy_method)].value)
            )
            assert target_drop == expected_target_drop
            assert oracle == min(camera, lidar, fused)


def test_bias_aggregates_are_ordered_and_reconcile_to_paired_sequence_rows() -> None:
    manifest = _manifest(BIAS_MANIFEST)
    evaluated = _evaluated(BIAS_MANIFEST)
    index = _localization_index(evaluated.metrics)
    conditions = expected_conditions(manifest)
    sequence_ids = expected_sequence_ids(manifest)

    expected_keys = [
        (condition, method, metric_name)
        for condition in conditions
        for method, metric_name in (
            *((method, "matched-center-mse") for method in manifest.methods),
            ("fixed-fusion", "fused-minus-healthy"),
        )
    ]
    actual_keys = [
        (
            ConditionKey(
                fault_family=record.fault_family,
                fault_axis=record.fault_axis,
                severity_index=record.severity.index,
                magnitude=record.severity.magnitude,
                direction=record.severity.direction,
                unit=record.severity.unit,
            ),
            record.method_id,
            record.metric_name,
        )
        for record in evaluated.aggregates
    ]
    assert len(evaluated.aggregates) == 13 * 6 == 78
    assert actual_keys == expected_keys

    aggregate_index = {
        (record.severity.index, record.severity.direction, record.method_id, record.metric_name): (
            record
        )
        for record in evaluated.aggregates
    }
    for condition in conditions:
        for method in manifest.methods:
            values = [
                cast(
                    float,
                    index[
                        (
                            sequence_id,
                            condition.severity_index,
                            condition.direction,
                            method,
                        )
                    ].value,
                )
                for sequence_id in sequence_ids
            ]
            aggregate = aggregate_index[
                (
                    condition.severity_index,
                    condition.direction,
                    method,
                    "matched-center-mse",
                )
            ]
            assert aggregate.estimate == math.fsum(values) / len(values)

        fused_values = [
            cast(
                float,
                index[
                    (
                        sequence_id,
                        condition.severity_index,
                        condition.direction,
                        "fixed-fusion",
                    )
                ].value,
            )
            for sequence_id in sequence_ids
        ]
        healthy_values = [
            cast(
                float,
                index[
                    (
                        sequence_id,
                        condition.severity_index,
                        condition.direction,
                        "lidar-only",
                    )
                ].value,
            )
            for sequence_id in sequence_ids
        ]
        contrast = aggregate_index[
            (
                condition.severity_index,
                condition.direction,
                "fixed-fusion",
                "fused-minus-healthy",
            )
        ]
        assert contrast.estimate == (
            math.fsum(
                fused - healthy for fused, healthy in zip(fused_values, healthy_values, strict=True)
            )
            / len(sequence_ids)
        )

    first_camera = aggregate_index[(0, "identity", "camera-only", "matched-center-mse")]
    first_camera_values = np.asarray(
        [
            cast(float, index[(sequence_id, 0, "identity", "camera-only")].value)
            for sequence_id in sequence_ids
        ],
        dtype=np.float64,
    )
    bootstrap_indices = np.random.Generator(
        np.random.PCG64DXSM(manifest.rng.bootstrap_seed)
    ).integers(
        0,
        len(sequence_ids),
        size=(manifest.evaluation.bootstrap.replicates, len(sequence_ids)),
        dtype=np.int64,
    )
    bootstrap_means = first_camera_values[bootstrap_indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975), method="linear")
    assert first_camera.interval_lower == float(lower)
    assert first_camera.interval_upper == float(upper)

    healthy_summaries = {
        (
            aggregate_index[
                (
                    condition.severity_index,
                    condition.direction,
                    "lidar-only",
                    "matched-center-mse",
                )
            ].estimate,
            aggregate_index[
                (
                    condition.severity_index,
                    condition.direction,
                    "lidar-only",
                    "matched-center-mse",
                )
            ].interval_lower,
            aggregate_index[
                (
                    condition.severity_index,
                    condition.direction,
                    "lidar-only",
                    "matched-center-mse",
                )
            ].interval_upper,
        )
        for condition in conditions
    }
    assert len(healthy_summaries) == 1


def test_bias_finite_sample_crossover_matches_preregistered_goldens() -> None:
    crossovers = _evaluated(BIAS_MANIFEST).crossovers
    assert [record.direction for record in crossovers] == ["negative", "positive"]

    expected = (
        (
            3.2126457655204974,
            1.310352730727073,
            4.971038275465614,
        ),
        (
            2.9641840935526558,
            1.1287754735374393,
            4.580193640694969,
        ),
    )
    for record, (point, lower, upper) in zip(crossovers, expected, strict=True):
        assert record.status == "observed"
        assert record.point_curve_crossed
        assert record.point_estimate == pytest.approx(point, rel=0.0, abs=1e-12)
        assert record.interval_lower == pytest.approx(lower, rel=0.0, abs=1e-12)
        assert record.interval_upper == pytest.approx(upper, rel=0.0, abs=1e-12)
        assert record.bootstrap_crossing_fraction == 1.0
        assert record.tested_maximum == 8.0
        assert record.censoring == "none"
        assert record.sequence_count == 200
        assert record.bootstrap_replicates == 2000


def test_generation_supports_lidar_target_and_y_bias_without_swapping_method_semantics() -> None:
    manifest = _manifest(BIAS_MANIFEST)
    fault = cast(AdditivePositionBiasFault, manifest.fault_sweep)
    generic = manifest.model_copy(
        update={
            "source": manifest.source.model_copy(update={"sequence_count": 2}),
            "fault_sweep": fault.model_copy(update={"target": "lidar", "axis": "y"}),
        }
    )
    metrics = generate_analytic_sequence_metrics(generic, run_id=RUN_ID)
    index = _localization_index(metrics)
    sequence_id = expected_sequence_ids(generic)[0]
    magnitude = fault.magnitude_values_m[1]
    camera_normal = draw_standard_normal_xy(
        data_master_seed=generic.rng.data_master_seed,
        stream_name="camera",
        sequence_id=sequence_id,
        object_frame_count=1,
    )[0]
    lidar_normal = draw_standard_normal_xy(
        data_master_seed=generic.rng.data_master_seed,
        stream_name="lidar",
        sequence_id=sequence_id,
        object_frame_count=1,
    )[0]
    camera = np.asarray(generic.observations.camera.actual_std_xy_m) * camera_normal
    lidar = np.asarray(generic.observations.lidar.actual_std_xy_m) * lidar_normal

    for direction, sign in (("negative", -1.0), ("positive", 1.0)):
        expected_lidar = np.array(lidar, copy=True)
        expected_lidar[1] += sign * magnitude
        camera_weight = np.asarray((1.0 / 37.0, 25.0 / 169.0))
        expected_fused = camera_weight * camera + (1.0 - camera_weight) * expected_lidar
        prefix = (sequence_id, 1, direction)
        assert index[(*prefix, "camera-only")].value == _loss((float(camera[0]), float(camera[1])))
        assert index[(*prefix, "lidar-only")].value == pytest.approx(
            _loss((float(expected_lidar[0]), float(expected_lidar[1]))),
            abs=2e-15,
        )
        assert index[(*prefix, "fixed-fusion")].value == pytest.approx(
            _loss((float(expected_fused[0]), float(expected_fused[1]))),
            abs=2e-15,
        )
        assert (
            index[(*prefix, "fault-target-drop-policy")].value
            == index[(*prefix, "camera-only")].value
        )


def test_evaluation_rejects_missing_and_extra_sequence_rows() -> None:
    manifest = _manifest(BIAS_MANIFEST)
    metrics = _metrics(BIAS_MANIFEST)

    with pytest.raises(ValueError, match="count"):
        evaluate_analytic_records(manifest, run_id=RUN_ID, metrics=metrics[:-1])
    with pytest.raises(ValueError, match="count"):
        evaluate_analytic_records(
            manifest,
            run_id=RUN_ID,
            metrics=(*metrics, metrics[-1]),
        )


def test_evaluation_rejects_out_of_order_and_nonlocalization_rows() -> None:
    manifest = _manifest(BIAS_MANIFEST)
    metrics = _metrics(BIAS_MANIFEST)
    reordered = (metrics[1], metrics[0], *metrics[2:])
    with pytest.raises(ValueError, match="contractual order"):
        evaluate_analytic_records(manifest, run_id=RUN_ID, metrics=reordered)

    first = cast(LocalizationMetricRecord, metrics[0])
    rate_row = RateMetricRecord(
        schema="ffb.sequence-metric/v1alpha1",
        record_level="sequence",
        run_id=first.run_id,
        manifest_sha256=first.manifest_sha256,
        sequence_id=first.sequence_id,
        fault_family=first.fault_family,
        fault_axis=first.fault_axis,
        severity=first.severity,
        method_id=first.method_id,
        eligible_object_frame_count=1,
        valid_object_frame_count=1,
        metric_name="coverage",
        status="ok",
        value=1.0,
        unit="fraction",
    )
    malformed: tuple[MetricRecordV1Alpha1, ...] = (rate_row, *metrics[1:])
    with pytest.raises(TypeError, match="localization"):
        evaluate_analytic_records(manifest, run_id=RUN_ID, metrics=malformed)


def test_generation_is_reproducible_independent_of_prior_calls() -> None:
    manifest = _manifest(CORRECT_NOISE_MANIFEST)
    first = generate_analytic_sequence_metrics(manifest, run_id=RUN_ID)
    _ = generate_analytic_sequence_metrics(_manifest(BIAS_MANIFEST), run_id="run:other")
    second = generate_analytic_sequence_metrics(manifest, run_id=RUN_ID)

    assert first == second
