from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
from typing import Any

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.bundle_v1alpha1 import (
    expected_conditions,
    expected_sequence_ids,
)
from fusion_fault_bench.contracts.io import validate_manifest_mapping
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AvailabilityControlManifest,
    CommonModeControlManifest,
    GeometryCrossoverManifest,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    LocalizationMetricRecord,
    MetricRecordV1Alpha1,
    RateMetricRecord,
    RunRecordV1Alpha1,
    RuntimeEnvironment,
    SeverityCoordinate,
)
from fusion_fault_bench.procedural_evaluation import (
    evaluate_procedural_records,
    validate_evaluated_procedural_records,
)


def _base_procedural_data(manifest_data: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(manifest_data)
    data.pop("analytic_validation")
    data["source"] = {
        "kind": "procedural",
        "split": "test",
        "profile_id": "constant-velocity-ci-smoke-v1",
        "profile_sha256": "a" * 64,
        "sequence_count": 2,
    }
    data["roi"] = {
        "frame": "ego-bev",
        "x_min_m": 5.0,
        "x_max_m": 60.0,
        "abs_y_max_m": 40.0,
        "camera_half_fov_rad": 0.7,
    }
    data["evaluation"]["bootstrap"]["replicates"] = 200
    return data


def _geometry_manifest(manifest_data: dict[str, Any]) -> GeometryCrossoverManifest:
    data = _base_procedural_data(manifest_data)
    data["kind"] = "geometry-crossover"
    data["fault_sweep"] = {
        "kind": "additive-position-bias",
        "target": "lidar",
        "axis": "y",
        "unit": "m",
        "injection_site": "estimator-output",
        "direction_policy": "symmetric-paired",
        "persistence": "sequence",
        "magnitude_values_m": [0.0, 2.0],
    }
    manifest = validate_manifest_mapping(data)
    assert isinstance(manifest, GeometryCrossoverManifest)
    return manifest


def _availability_manifest(
    manifest_data: dict[str, Any],
) -> AvailabilityControlManifest:
    data = _base_procedural_data(manifest_data)
    data["kind"] = "availability-control"
    data["fault_sweep"] = {
        "kind": "dropout",
        "target": "camera",
        "axis": "availability",
        "unit": "probability",
        "injection_site": "availability",
        "process": "shared-target-modality-frame-bernoulli",
        "probability_values": [0.0, 0.5, 1.0],
    }
    data["methods"] = [
        "camera-only",
        "lidar-only",
        "fixed-fusion",
        "fault-target-drop-policy",
    ]
    data["evaluation"] = {
        "mode": "availability-control",
        "metrics": [
            "coverage",
            "conditional-matched-center-mse",
            "undefined-output-rate",
        ],
        "missing_output_policy": "undefined-no-localization-penalty",
        "rate_aggregation": "count-ratio-with-sequence-bootstrap",
        "conditional_loss_aggregation": ("valid-object-frame-ratio-with-sequence-bootstrap"),
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
    }
    manifest = validate_manifest_mapping(data)
    assert isinstance(manifest, AvailabilityControlManifest)
    return manifest


def _common_mode_manifest(
    manifest_data: dict[str, Any],
) -> CommonModeControlManifest:
    data = _base_procedural_data(manifest_data)
    data["kind"] = "common-mode-control"
    data["fault_sweep"] = {
        "kind": "common-mode-position-bias",
        "target": "both",
        "axis": "x",
        "unit": "m",
        "injection_site": "estimator-output",
        "direction_policy": "symmetric-paired",
        "persistence": "sequence",
        "magnitude_values_m": [0.0, 1.0],
    }
    data["methods"] = ["camera-only", "lidar-only", "fixed-fusion"]
    data["evaluation"] = {
        "mode": "common-mode-blind-spot-control",
        "primary_loss": "matched-center-mse",
        "loss_unit": "m^2",
        "aggregation": "object-frame-mean-then-sequence-mean",
        "primary_contrast": "none",
        "bootstrap": {
            "method": "percentile",
            "unit": "sequence",
            "resampling": "paired-indices-across-severities-and-methods",
            "interval_scope": "pointwise",
            "replicates": 200,
            "confidence_level": 0.95,
        },
        "crossover": "not-applicable",
    }
    manifest = validate_manifest_mapping(data)
    assert isinstance(manifest, CommonModeControlManifest)
    return manifest


def _run(
    manifest: GeometryCrossoverManifest | AvailabilityControlManifest | CommonModeControlManifest,
) -> RunRecordV1Alpha1:
    now = datetime.now(UTC)
    return RunRecordV1Alpha1(
        schema="ffb.run/v1alpha1",
        run_id="procedural-test-run",
        manifest_sha256=sha256_digest(manifest),
        package_version="0.1.0",
        git_revision="b" * 40,
        source_dirty=False,
        lockfile_sha256="c" * 64,
        command=("ffb", "run", "manifest.json"),
        environment=RuntimeEnvironment(
            python_version="3.12.13",
            os_name="macOS",
            os_release="15.5",
            machine="arm64",
            cpu_model="test-cpu",
            logical_cpu_count=1,
            memory_bytes=1,
        ),
        started_at=now,
        ended_at=now + timedelta(seconds=1),
        status="succeeded",
        artifact_sha256="d" * 64,
    )


def _severity(condition: Any) -> SeverityCoordinate:
    return SeverityCoordinate(
        index=condition.severity_index,
        magnitude=condition.magnitude,
        direction=condition.direction,
        unit=condition.unit,
    )


def test_geometry_evaluation_recomputes_crossovers_and_validates_bundle(
    manifest_data: dict[str, Any],
) -> None:
    manifest = _geometry_manifest(manifest_data)
    digest = sha256_digest(manifest)
    metrics: list[MetricRecordV1Alpha1] = []
    for sequence_id in expected_sequence_ids(manifest):
        for condition in expected_conditions(manifest):
            values = (
                {
                    "camera-only": 2.0,
                    "lidar-only": 0.18,
                    "fixed-fusion": 0.16,
                    "fault-target-drop-policy": 0.16,
                    "performance-oracle": 0.16,
                }
                if condition.severity_index == 0
                else {
                    "camera-only": 2.0,
                    "lidar-only": 4.0,
                    "fixed-fusion": 2.5,
                    "fault-target-drop-policy": 2.0,
                    "performance-oracle": 2.0,
                }
            )
            for method in manifest.methods:
                metrics.append(
                    LocalizationMetricRecord(
                        schema="ffb.sequence-metric/v1alpha1",
                        record_level="sequence",
                        run_id="procedural-test-run",
                        manifest_sha256=digest,
                        sequence_id=sequence_id,
                        fault_family=condition.fault_family,
                        fault_axis=condition.fault_axis,
                        severity=_severity(condition),
                        method_id=method,
                        eligible_object_frame_count=10,
                        valid_object_frame_count=10,
                        metric_name="matched-center-mse",
                        status="ok",
                        value=values[method],
                        unit="m^2",
                    )
                )

    evaluated = evaluate_procedural_records(
        manifest,
        run_id="procedural-test-run",
        metrics=tuple(metrics),
    )

    assert len(evaluated.crossovers) == 2
    assert {record.status for record in evaluated.crossovers} == {"observed"}
    validate_evaluated_procedural_records(
        manifest,
        run=_run(manifest),
        records=evaluated,
    )


def test_availability_evaluation_preserves_undefined_conditional_support(
    manifest_data: dict[str, Any],
) -> None:
    manifest = _availability_manifest(manifest_data)
    digest = sha256_digest(manifest)
    metrics: list[MetricRecordV1Alpha1] = []
    for sequence_index, sequence_id in enumerate(expected_sequence_ids(manifest)):
        for condition in expected_conditions(manifest):
            for method in manifest.methods:
                if condition.magnitude == 0.0 or method in {
                    "lidar-only",
                    "fault-target-drop-policy",
                }:
                    valid = 10
                elif condition.magnitude == 1.0 or sequence_index == 1:
                    valid = 0
                else:
                    valid = 10
                eligible = 10
                value = (
                    0.5
                    if method in {"camera-only", "fixed-fusion"}
                    or (method == "fault-target-drop-policy" and condition.severity_index == 0)
                    else 0.25
                )
                common = {
                    "schema": "ffb.sequence-metric/v1alpha1",
                    "record_level": "sequence",
                    "run_id": "procedural-test-run",
                    "manifest_sha256": digest,
                    "sequence_id": sequence_id,
                    "fault_family": condition.fault_family,
                    "fault_axis": condition.fault_axis,
                    "severity": _severity(condition),
                    "method_id": method,
                    "eligible_object_frame_count": eligible,
                    "valid_object_frame_count": valid,
                }
                metrics.extend(
                    (
                        RateMetricRecord(
                            **common,
                            metric_name="coverage",
                            status="ok",
                            value=valid / eligible,
                            unit="fraction",
                        ),
                        LocalizationMetricRecord(
                            **common,
                            metric_name="conditional-matched-center-mse",
                            status="ok" if valid else "undefined",
                            value=value if valid else None,
                            unit="m^2",
                        ),
                        RateMetricRecord(
                            **common,
                            metric_name="undefined-output-rate",
                            status="ok",
                            value=(eligible - valid) / eligible,
                            unit="fraction",
                        ),
                    )
                )

    evaluated = evaluate_procedural_records(
        manifest,
        run_id="procedural-test-run",
        metrics=tuple(metrics),
    )

    unsupported = next(
        record
        for record in evaluated.aggregates
        if record.severity.magnitude == 0.5
        and record.method_id == "camera-only"
        and record.metric_name == "conditional-matched-center-mse"
    )
    assert unsupported.status == "undefined"
    assert unsupported.estimate is None
    assert evaluated.crossovers == ()
    validate_evaluated_procedural_records(
        manifest,
        run=_run(manifest),
        records=evaluated,
    )


def test_common_mode_evaluation_has_absolute_losses_without_crossover(
    manifest_data: dict[str, Any],
) -> None:
    manifest = _common_mode_manifest(manifest_data)
    digest = sha256_digest(manifest)
    metrics: list[MetricRecordV1Alpha1] = []
    for sequence_id in expected_sequence_ids(manifest):
        for condition in expected_conditions(manifest):
            for method in manifest.methods:
                metrics.append(
                    LocalizationMetricRecord(
                        schema="ffb.sequence-metric/v1alpha1",
                        record_level="sequence",
                        run_id="procedural-test-run",
                        manifest_sha256=digest,
                        sequence_id=sequence_id,
                        fault_family=condition.fault_family,
                        fault_axis=condition.fault_axis,
                        severity=_severity(condition),
                        method_id=method,
                        eligible_object_frame_count=10,
                        valid_object_frame_count=10,
                        metric_name="matched-center-mse",
                        status="ok",
                        value=0.2 + condition.magnitude**2,
                        unit="m^2",
                    )
                )

    evaluated = evaluate_procedural_records(
        manifest,
        run_id="procedural-test-run",
        metrics=tuple(metrics),
    )

    assert evaluated.crossovers == ()
    assert all(record.metric_name == "matched-center-mse" for record in evaluated.aggregates)
    validate_evaluated_procedural_records(
        manifest,
        run=_run(manifest),
        records=evaluated,
    )
