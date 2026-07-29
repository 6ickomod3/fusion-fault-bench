from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import TypeAdapter, ValidationError

from fusion_fault_bench.contracts.result_v1alpha1 import (
    METRIC_RECORD_ADAPTER,
    AggregateMetricRecordV1Alpha1,
    CrossoverRecordV1Alpha1,
    LocalizationMetricRecord,
    RateMetricRecord,
    RunRecordV1Alpha1,
    RuntimeEnvironment,
)

DIGEST = "a" * 64
GIT_REVISION = "b" * 40


def _environment() -> RuntimeEnvironment:
    return RuntimeEnvironment(
        python_version="3.12.13",
        os_name="macOS",
        os_release="15.5",
        machine="arm64",
        cpu_model="Apple-M4",
        logical_cpu_count=10,
        memory_bytes=32_000_000_000,
    )


def _run(**updates) -> dict:
    now = datetime.now(UTC)
    value = {
        "schema": "ffb.run/v1alpha1",
        "run_id": "run-001",
        "manifest_sha256": DIGEST,
        "package_version": "0.1.0",
        "git_revision": GIT_REVISION,
        "source_dirty": False,
        "lockfile_sha256": DIGEST,
        "command": ("ffb", "run", "analytic", "manifest.json"),
        "environment": _environment(),
        "started_at": now,
        "ended_at": now + timedelta(seconds=1),
        "status": "succeeded",
        "artifact_sha256": DIGEST,
    }
    value.update(updates)
    return value


def _metric_common() -> dict:
    return {
        "schema": "ffb.sequence-metric/v1alpha1",
        "record_level": "sequence",
        "run_id": "run-001",
        "manifest_sha256": DIGEST,
        "sequence_id": "sequence-001",
        "fault_family": "additive-position-bias",
        "fault_axis": "x",
        "severity": {
            "index": 1,
            "magnitude": 0.5,
            "direction": "positive",
            "unit": "m",
        },
        "method_id": "fixed-fusion",
        "eligible_object_frame_count": 20,
        "valid_object_frame_count": 20,
    }


def _aggregate() -> dict:
    return {
        "schema": "ffb.aggregate-metric/v1alpha1",
        "record_level": "aggregate",
        "run_id": "run-001",
        "manifest_sha256": DIGEST,
        "fault_family": "additive-position-bias",
        "fault_axis": "x",
        "severity": {
            "index": 1,
            "magnitude": 0.5,
            "direction": "positive",
            "unit": "m",
        },
        "method_id": "fixed-fusion",
        "metric_name": "matched-center-mse",
        "status": "ok",
        "estimate": 0.25,
        "interval_lower": 0.2,
        "interval_upper": 0.3,
        "unit": "m^2",
        "sequence_count": 200,
        "contributing_sequence_count": 200,
        "bootstrap_replicates": 1000,
        "defined_bootstrap_replicates": 1000,
        "confidence_level": 0.95,
        "interval_method": "paired-sequence-percentile-pointwise",
        "aggregation": "object-frame-mean-then-sequence-mean",
    }


def _crossover() -> dict:
    return {
        "schema": "ffb.crossover/v1alpha1",
        "run_id": "run-001",
        "manifest_sha256": DIGEST,
        "fault_family": "additive-position-bias",
        "fault_axis": "x",
        "direction": "positive",
        "severity_unit": "m",
        "status": "observed",
        "point_curve_crossed": True,
        "point_estimate": 1.1,
        "interval_lower": 0.9,
        "interval_upper": 1.4,
        "tested_maximum": 4.0,
        "censoring": "none",
        "bootstrap_crossing_fraction": 0.99,
        "sequence_count": 200,
        "bootstrap_replicates": 1000,
        "confidence_level": 0.95,
        "interval_method": "right-censored-percentile",
    }


def test_succeeded_run_has_structured_complete_provenance() -> None:
    run = RunRecordV1Alpha1(**_run())

    assert run.status == "succeeded"
    assert run.environment.cpu_model == "Apple-M4"


@pytest.mark.parametrize(
    "updates",
    [
        {"ended_at": None},
        {"artifact_sha256": None},
        {
            "ended_at": datetime(2026, 1, 1, tzinfo=UTC),
            "started_at": datetime(2026, 1, 2, tzinfo=UTC),
        },
        {"started_at": datetime(2026, 1, 1)},
        {
            "status": "running",
            "artifact_sha256": DIGEST,
        },
        {"command": ("ffb", "/Users/example/private.json")},
        {"command": ("ffb", "--manifest=/Users/example/private.json")},
        {"command": ("ffb", "--manifest=C:/Users/example/private.json")},
        {"command": ("ffb", "--manifest=file:///Users/example/private.json")},
    ],
)
def test_run_rejects_invalid_lifecycle_or_local_paths(updates: dict) -> None:
    with pytest.raises(ValidationError):
        RunRecordV1Alpha1(**_run(**updates))


def test_running_record_forbids_completion_fields() -> None:
    record = RunRecordV1Alpha1(**_run(status="running", ended_at=None, artifact_sha256=None))
    assert record.ended_at is None


def test_localization_metric_is_structured_by_severity() -> None:
    metric = LocalizationMetricRecord(
        **_metric_common(),
        metric_name="matched-center-mse",
        status="ok",
        value=0.25,
        unit="m^2",
    )

    assert metric.severity.index == 1
    assert metric.severity.magnitude == 0.5


def test_undefined_localization_retains_prefault_denominator() -> None:
    value = _metric_common()
    value["valid_object_frame_count"] = 0
    metric = LocalizationMetricRecord(
        **value,
        metric_name="conditional-matched-center-mse",
        status="undefined",
        value=None,
        unit="m^2",
    )

    assert metric.eligible_object_frame_count == 20
    assert metric.value is None


def test_metric_adapter_distinguishes_rate_and_localization() -> None:
    value = _metric_common()
    value.update(
        {
            "metric_name": "coverage",
            "status": "ok",
            "value": 0.5,
            "unit": "fraction",
            "valid_object_frame_count": 10,
        }
    )
    metric = METRIC_RECORD_ADAPTER.validate_python(value)

    assert isinstance(metric, RateMetricRecord)


def test_metric_rejects_invalid_counts_and_status_value() -> None:
    value = _metric_common()
    value["valid_object_frame_count"] = 21
    with pytest.raises(ValidationError, match="cannot exceed"):
        LocalizationMetricRecord(
            **value,
            metric_name="matched-center-mse",
            status="ok",
            value=0.1,
            unit="m^2",
        )

    value["valid_object_frame_count"] = 0
    with pytest.raises(ValidationError, match="require a value and valid"):
        LocalizationMetricRecord(
            **value,
            metric_name="matched-center-mse",
            status="ok",
            value=None,
            unit="m^2",
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"fault_axis": "time"}, "axis time is invalid"),
        (
            {"severity": {"index": 1, "magnitude": 0.5, "direction": "positive", "unit": "s"}},
            "severity unit",
        ),
        (
            {"severity": {"index": 0, "magnitude": 0.5, "direction": "identity", "unit": "m"}},
            "identity severity",
        ),
        (
            {"severity": {"index": 1, "magnitude": 0.0, "direction": "positive", "unit": "m"}},
            "must exceed",
        ),
    ],
)
def test_metric_rejects_fault_coordinate_contradictions(updates: dict, message: str) -> None:
    value = _metric_common()
    value.update(updates)

    with pytest.raises(ValidationError, match=message):
        LocalizationMetricRecord(
            **value,
            metric_name="matched-center-mse",
            status="ok",
            value=0.1,
            unit="m^2",
        )


def test_aggregate_interval_must_be_complete_and_contain_estimate() -> None:
    base = _aggregate()
    assert AggregateMetricRecordV1Alpha1(**base).estimate == 0.25

    base["interval_lower"] = 0.31
    with pytest.raises(ValidationError, match="bounds must be ordered"):
        AggregateMetricRecordV1Alpha1(**base)

    base.update({"interval_lower": 0.26, "interval_upper": 0.3})
    assert AggregateMetricRecordV1Alpha1(**base).estimate == 0.25


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {
                "fault_family": "dropout",
                "fault_axis": "availability",
                "severity": {
                    "index": 1,
                    "magnitude": 0.5,
                    "direction": "increase",
                    "unit": "probability",
                },
                "metric_name": "coverage",
                "unit": "m^2",
                "aggregation": "count-ratio-with-sequence-bootstrap",
            },
            "requires unit fraction",
        ),
        (
            {
                "fault_family": "dropout",
                "fault_axis": "availability",
                "severity": {
                    "index": 1,
                    "magnitude": 0.5,
                    "direction": "increase",
                    "unit": "probability",
                },
                "metric_name": "coverage",
                "unit": "fraction",
                "aggregation": "count-ratio-with-sequence-bootstrap",
                "estimate": 2.0,
                "interval_lower": 1.5,
                "interval_upper": 2.5,
            },
            r"\[0, 1\]",
        ),
        (
            {
                "estimate": -1.0,
            },
            "must be non-negative",
        ),
    ],
)
def test_aggregate_rejects_metric_unit_and_domain_contradictions(
    updates: dict, message: str
) -> None:
    base = _aggregate()
    base.update(updates)

    with pytest.raises(ValidationError, match=message):
        AggregateMetricRecordV1Alpha1(**base)


def test_crossover_status_controls_estimate_and_censoring() -> None:
    base = _crossover()
    assert CrossoverRecordV1Alpha1(**base).status == "observed"

    base.update(
        {
            "status": "not-observed",
            "point_curve_crossed": False,
            "point_estimate": None,
            "interval_lower": 4.0,
            "interval_upper": "positive-infinity",
            "censoring": "right-above-tested-maximum",
            "bootstrap_crossing_fraction": 0.0,
        }
    )
    assert CrossoverRecordV1Alpha1(**base).status == "not-observed"

    base["censoring"] = "none"
    with pytest.raises(ValidationError, match="right-censored"):
        CrossoverRecordV1Alpha1(**base)


def test_undetermined_crossover_encodes_point_curve_without_an_interval() -> None:
    base = _crossover()
    base.update(
        {
            "status": "undetermined",
            "point_estimate": 1.1,
            "interval_lower": None,
            "interval_upper": None,
            "censoring": "mixed-bootstrap",
            "bootstrap_crossing_fraction": 0.5,
        }
    )
    assert CrossoverRecordV1Alpha1(**base).point_curve_crossed

    base["point_curve_crossed"] = False
    with pytest.raises(ValidationError, match="must match point-curve"):
        CrossoverRecordV1Alpha1(**base)


@pytest.mark.parametrize(
    ("point_crossed", "point_estimate", "crossing_fraction"),
    [
        (False, None, 50 / 2000),
        (True, 1.1, 1950 / 2000),
    ],
)
def test_exact_bootstrap_support_boundaries_are_undetermined(
    point_crossed: bool,
    point_estimate: float | None,
    crossing_fraction: float,
) -> None:
    base = _crossover()
    base.update(
        {
            "status": "undetermined",
            "point_curve_crossed": point_crossed,
            "point_estimate": point_estimate,
            "interval_lower": None,
            "interval_upper": None,
            "censoring": "mixed-bootstrap",
            "bootstrap_crossing_fraction": crossing_fraction,
            "bootstrap_replicates": 2000,
        }
    )

    assert CrossoverRecordV1Alpha1(**base).status == "undetermined"


def test_crossover_fraction_must_be_representable_by_replicate_count() -> None:
    base = _crossover()
    base["bootstrap_crossing_fraction"] = 0.9901

    with pytest.raises(ValidationError, match="representable"):
        CrossoverRecordV1Alpha1(**base)


@pytest.mark.parametrize(
    ("point_crossed", "point_estimate", "crossing_fraction"),
    [
        (True, 1.1, 0.0),
        (False, None, 1.0),
    ],
)
def test_undetermined_crossover_allows_conflicting_point_and_bootstrap_evidence(
    point_crossed: bool,
    point_estimate: float | None,
    crossing_fraction: float,
) -> None:
    base = _crossover()
    base.update(
        {
            "status": "undetermined",
            "point_curve_crossed": point_crossed,
            "point_estimate": point_estimate,
            "interval_lower": None,
            "interval_upper": None,
            "censoring": "mixed-bootstrap",
            "bootstrap_crossing_fraction": crossing_fraction,
        }
    )

    assert CrossoverRecordV1Alpha1(**base).status == "undetermined"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"bootstrap_crossing_fraction": 0.0}, "sufficient bootstrap support"),
        ({"tested_maximum": -1.0}, "must exceed the identity"),
        (
            {
                "point_estimate": 4.5,
                "interval_lower": 4.2,
                "interval_upper": 4.8,
            },
            "within the tested severity grid",
        ),
        (
            {
                "status": "not-observed",
                "point_curve_crossed": False,
                "point_estimate": None,
                "interval_lower": 4.0,
                "interval_upper": "positive-infinity",
                "censoring": "right-above-tested-maximum",
                "bootstrap_crossing_fraction": 1.0,
            },
            "negligible bootstrap support",
        ),
    ],
)
def test_crossover_rejects_status_evidence_and_severity_contradictions(
    updates: dict, message: str
) -> None:
    base = _crossover()
    base.update(updates)

    with pytest.raises(ValidationError, match=message):
        CrossoverRecordV1Alpha1(**base)


def test_metric_union_schema_is_buildable() -> None:
    schema = TypeAdapter(LocalizationMetricRecord | RateMetricRecord).json_schema()
    assert "$defs" in schema
