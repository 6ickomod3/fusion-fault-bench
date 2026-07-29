"""Structured runtime, sequence, aggregate, and crossover records."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    FiniteFloat,
    TypeAdapter,
    field_validator,
    model_validator,
)

from fusion_fault_bench.contracts.common import ContractModel
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    FaultAxis,
    FaultFamily,
    MethodId,
    SeverityDirection,
    SeverityUnit,
)

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GitRevision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Identifier = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"),
]

_FAULT_AXES: dict[str, frozenset[str]] = {
    "additive-position-bias": frozenset({"x", "y"}),
    "increased-noise-correctly-reported": frozenset({"xy"}),
    "increased-noise-underreported": frozenset({"xy"}),
    "calibration-translation": frozenset({"x", "y"}),
    "calibration-yaw": frozenset({"yaw"}),
    "timestamp-offset": frozenset({"time"}),
    "dropout": frozenset({"availability"}),
    "common-mode-position-bias": frozenset({"x", "y"}),
}
_FAULT_UNITS: dict[str, str] = {
    "additive-position-bias": "m",
    "increased-noise-correctly-reported": "std-scale",
    "increased-noise-underreported": "std-scale",
    "calibration-translation": "m",
    "calibration-yaw": "rad",
    "timestamp-offset": "s",
    "dropout": "probability",
    "common-mode-position-bias": "m",
}
_NOISE_FAULTS = {
    "increased-noise-correctly-reported",
    "increased-noise-underreported",
}
_RATE_METRICS = {"coverage", "undefined-output-rate"}
_NONNEGATIVE_LOSS_METRICS = {
    "matched-center-mse",
    "conditional-matched-center-mse",
}
_METRIC_AGGREGATIONS = {
    "matched-center-mse": "object-frame-mean-then-sequence-mean",
    "fused-minus-healthy": "object-frame-mean-then-sequence-mean",
    "conditional-matched-center-mse": ("valid-object-frame-ratio-with-sequence-bootstrap"),
    "coverage": "count-ratio-with-sequence-bootstrap",
    "undefined-output-rate": "count-ratio-with-sequence-bootstrap",
}


def _validate_fault_coordinate(
    family: FaultFamily,
    axis: FaultAxis,
    severity: SeverityCoordinate,
) -> None:
    if axis not in _FAULT_AXES[family]:
        raise ValueError(f"axis {axis} is invalid for fault family {family}")
    if severity.unit != _FAULT_UNITS[family]:
        raise ValueError(f"severity unit is invalid for fault family {family}")
    identity_magnitude = 1.0 if family in _NOISE_FAULTS else 0.0
    if severity.index == 0 and severity.magnitude != identity_magnitude:
        raise ValueError("identity severity magnitude does not match the fault operator")
    if severity.index > 0:
        if severity.magnitude <= identity_magnitude:
            raise ValueError("non-identity severity must exceed the identity magnitude")
        expected_direction = "increase" if family in _NOISE_FAULTS | {"dropout"} else None
        if expected_direction is not None and severity.direction != expected_direction:
            raise ValueError(f"{family} requires increase direction")
        if expected_direction is None and severity.direction not in {"negative", "positive"}:
            raise ValueError(f"{family} requires a signed direction")


class RuntimeEnvironment(ContractModel):
    """Named machine facts required for reproducibility and throughput claims."""

    python_version: Identifier
    os_name: Identifier
    os_release: Identifier
    machine: Identifier
    cpu_model: Annotated[str, Field(min_length=1, max_length=256)]
    logical_cpu_count: Annotated[int, Field(ge=1)]
    memory_bytes: Annotated[int, Field(ge=1)]


class RunRecordV1Alpha1(ContractModel):
    """Runtime provenance kept separate from immutable experiment intent."""

    schema_id: Literal["ffb.run/v1alpha1"] = Field(alias="schema")
    run_id: Identifier
    manifest_sha256: Digest
    package_version: Identifier
    git_revision: GitRevision
    source_dirty: bool
    lockfile_sha256: Digest
    command: Annotated[tuple[str, ...], Field(min_length=1)]
    environment: RuntimeEnvironment
    started_at: datetime
    ended_at: datetime | None
    status: Literal["running", "succeeded", "failed"]
    artifact_sha256: Digest | None

    @field_validator("command")
    @classmethod
    def reject_absolute_command_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for argument in value:
            if not argument:
                raise ValueError("command arguments must not be empty")
            candidates = (argument, argument.split("=", maxsplit=1)[-1])
            for candidate in candidates:
                if (
                    PurePosixPath(candidate).is_absolute()
                    or PureWindowsPath(candidate).is_absolute()
                    or candidate.lower().startswith("file:")
                ):
                    raise ValueError("command must not contain absolute local paths")
        return value

    @field_validator("started_at", "ended_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.utcoffset() is None:
            raise ValueError("run timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_lifecycle(self) -> Self:
        if self.status == "running":
            if self.ended_at is not None or self.artifact_sha256 is not None:
                raise ValueError("running records cannot have end time or artifact digest")
            return self
        if self.ended_at is None:
            raise ValueError("completed run records require ended_at")
        if self.ended_at < self.started_at:
            raise ValueError("ended_at must not precede started_at")
        if self.status == "succeeded" and self.artifact_sha256 is None:
            raise ValueError("succeeded run records require artifact_sha256")
        return self


class SeverityCoordinate(ContractModel):
    """Machine-readable condition coordinate from a manifest fault grid."""

    index: Annotated[int, Field(ge=0)]
    magnitude: FiniteFloat
    direction: SeverityDirection
    unit: SeverityUnit

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.index == 0 and self.direction != "identity":
            raise ValueError("severity index 0 must use identity direction")
        if self.index > 0 and self.direction == "identity":
            raise ValueError("non-identity severity cannot use identity direction")
        if self.magnitude < 0.0:
            raise ValueError("severity magnitude must be non-negative")
        if self.magnitude == 0.0 and math.copysign(1.0, self.magnitude) < 0.0:
            raise ValueError("severity magnitude must use canonical positive zero")
        return self


class _SequenceMetricBase(ContractModel):
    schema_id: Literal["ffb.sequence-metric/v1alpha1"] = Field(alias="schema")
    record_level: Literal["sequence"]
    run_id: Identifier
    manifest_sha256: Digest
    sequence_id: Identifier
    fault_family: FaultFamily
    fault_axis: FaultAxis
    severity: SeverityCoordinate
    method_id: MethodId
    eligible_object_frame_count: Annotated[int, Field(ge=1)]
    valid_object_frame_count: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.valid_object_frame_count > self.eligible_object_frame_count:
            raise ValueError("valid count cannot exceed pre-fault eligible count")
        _validate_fault_coordinate(self.fault_family, self.fault_axis, self.severity)
        return self


class LocalizationMetricRecord(_SequenceMetricBase):
    """Sequence localization loss, including explicitly undefined dropout rows."""

    metric_name: Literal["matched-center-mse", "conditional-matched-center-mse"]
    status: Literal["ok", "undefined"]
    value: FiniteFloat | None
    unit: Literal["m^2"]

    @model_validator(mode="after")
    def validate_value_status(self) -> Self:
        if self.status == "ok":
            if self.value is None or self.valid_object_frame_count == 0:
                raise ValueError("ok localization rows require a value and valid samples")
            if self.value < 0.0:
                raise ValueError("localization loss must be non-negative")
        elif self.value is not None:
            raise ValueError("undefined rows cannot contain a value")
        return self


class RateMetricRecord(_SequenceMetricBase):
    """Coverage or undefined-output fraction over the pre-fault denominator."""

    metric_name: Literal["coverage", "undefined-output-rate"]
    status: Literal["ok"]
    value: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    unit: Literal["fraction"]


type MetricRecordV1Alpha1 = Annotated[
    LocalizationMetricRecord | RateMetricRecord,
    Field(discriminator="metric_name"),
]
METRIC_RECORD_ADAPTER = TypeAdapter(MetricRecordV1Alpha1)


class AggregateMetricRecordV1Alpha1(ContractModel):
    """Pointwise sequence-clustered aggregate with a percentile interval."""

    schema_id: Literal["ffb.aggregate-metric/v1alpha1"] = Field(alias="schema")
    record_level: Literal["aggregate"]
    run_id: Identifier
    manifest_sha256: Digest
    fault_family: FaultFamily
    fault_axis: FaultAxis
    severity: SeverityCoordinate
    method_id: MethodId
    metric_name: Literal[
        "matched-center-mse",
        "conditional-matched-center-mse",
        "coverage",
        "undefined-output-rate",
        "fused-minus-healthy",
    ]
    status: Literal["ok", "undefined"]
    estimate: FiniteFloat | None
    interval_lower: FiniteFloat | None
    interval_upper: FiniteFloat | None
    unit: Literal["m^2", "fraction"]
    sequence_count: Annotated[int, Field(ge=2)]
    contributing_sequence_count: Annotated[int, Field(ge=0)]
    bootstrap_replicates: Annotated[int, Field(ge=200, multiple_of=40)]
    defined_bootstrap_replicates: Annotated[int, Field(ge=0)]
    confidence_level: Annotated[FiniteFloat, Field(ge=0.95, le=0.95)]
    interval_method: Literal["paired-sequence-percentile-pointwise"]
    aggregation: Literal[
        "object-frame-mean-then-sequence-mean",
        "valid-object-frame-ratio-with-sequence-bootstrap",
        "count-ratio-with-sequence-bootstrap",
    ]

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        _validate_fault_coordinate(self.fault_family, self.fault_axis, self.severity)
        if self.contributing_sequence_count > self.sequence_count:
            raise ValueError("contributing_sequence_count cannot exceed sequence_count")
        if self.defined_bootstrap_replicates > self.bootstrap_replicates:
            raise ValueError("defined bootstrap replicates cannot exceed requested replicates")
        if self.aggregation != _METRIC_AGGREGATIONS[self.metric_name]:
            raise ValueError(f"{self.metric_name} has an invalid aggregation")
        expected_unit = "fraction" if self.metric_name in _RATE_METRICS else "m^2"
        if self.unit != expected_unit:
            raise ValueError(f"{self.metric_name} requires unit {expected_unit}")
        if self.metric_name in _RATE_METRICS and self.status != "ok":
            raise ValueError(f"{self.metric_name} must have status ok")
        if (
            self.metric_name
            in {
                "coverage",
                "undefined-output-rate",
                "conditional-matched-center-mse",
            }
            and self.fault_family != "dropout"
        ):
            raise ValueError(f"{self.metric_name} is only defined for dropout")
        if self.metric_name == "fused-minus-healthy":
            if self.fault_family in {"dropout", "common-mode-position-bias"}:
                raise ValueError("fused-minus-healthy requires a single-sensor crossover fault")
            if self.method_id != "fixed-fusion":
                raise ValueError("fused-minus-healthy must use fixed-fusion method_id")
        values = (self.estimate, self.interval_lower, self.interval_upper)
        alpha = 1.0 - self.confidence_level
        defined_fraction = self.defined_bootstrap_replicates / self.bootstrap_replicates
        if self.status == "ok":
            if self.contributing_sequence_count == 0:
                raise ValueError("ok aggregate records require contributing sequences")
            if defined_fraction <= 1.0 - alpha / 2.0:
                raise ValueError("ok aggregate records require two-sided bootstrap support")
            if any(value is None for value in values):
                raise ValueError("ok aggregate records require estimate and interval")
            assert self.estimate is not None
            assert self.interval_lower is not None
            assert self.interval_upper is not None
            if self.interval_lower > self.interval_upper:
                raise ValueError("aggregate interval bounds must be ordered")
            if self.metric_name in _RATE_METRICS and any(
                value < 0.0 or value > 1.0
                for value in (
                    self.interval_lower,
                    self.estimate,
                    self.interval_upper,
                )
            ):
                raise ValueError("rate aggregate values must be in [0, 1]")
            if self.metric_name in _NONNEGATIVE_LOSS_METRICS and any(
                value < 0.0
                for value in (
                    self.interval_lower,
                    self.estimate,
                    self.interval_upper,
                )
            ):
                raise ValueError("localization aggregate values must be non-negative")
        elif any(value is not None for value in values):
            raise ValueError("non-ok aggregate records cannot contain estimates")
        elif (
            self.metric_name == "conditional-matched-center-mse"
            and self.contributing_sequence_count > 0
            and defined_fraction > 1.0 - alpha / 2.0
        ):
            raise ValueError("conditional aggregate with two-sided support must have status ok")
        return self


class CrossoverRecordV1Alpha1(ContractModel):
    """Crossover estimate with explicit censoring and bootstrap support."""

    schema_id: Literal["ffb.crossover/v1alpha1"] = Field(alias="schema")
    run_id: Identifier
    manifest_sha256: Digest
    fault_family: FaultFamily
    fault_axis: FaultAxis
    direction: Literal["negative", "positive", "increase"]
    severity_unit: SeverityUnit
    status: Literal["observed", "not-observed", "undetermined"]
    point_curve_crossed: bool
    point_estimate: FiniteFloat | None
    interval_lower: FiniteFloat | None
    interval_upper: FiniteFloat | Literal["positive-infinity"] | None
    tested_maximum: FiniteFloat
    censoring: Literal["none", "right-above-tested-maximum", "mixed-bootstrap"]
    bootstrap_crossing_fraction: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
    sequence_count: Annotated[int, Field(ge=2)]
    bootstrap_replicates: Annotated[int, Field(ge=200, multiple_of=40)]
    confidence_level: Annotated[FiniteFloat, Field(ge=0.95, le=0.95)]
    interval_method: Literal["right-censored-percentile"]

    @model_validator(mode="after")
    def validate_status_fields(self) -> Self:
        if self.fault_family in {"dropout", "common-mode-position-bias"}:
            raise ValueError(f"{self.fault_family} has no crossover estimand")
        if self.fault_axis not in _FAULT_AXES[self.fault_family]:
            raise ValueError("crossover axis is invalid for the fault family")
        if self.severity_unit != _FAULT_UNITS[self.fault_family]:
            raise ValueError("crossover unit is invalid for the fault family")
        if self.fault_family in _NOISE_FAULTS and self.direction != "increase":
            raise ValueError("noise crossover requires increase direction")
        if self.fault_family not in _NOISE_FAULTS and self.direction == "increase":
            raise ValueError("signed crossover requires positive or negative direction")
        identity_magnitude = 1.0 if self.fault_family in _NOISE_FAULTS else 0.0
        if self.tested_maximum <= identity_magnitude:
            raise ValueError("tested_maximum must exceed the identity magnitude")

        alpha = 1.0 - self.confidence_level
        lower_support = alpha / 2.0
        upper_support = 1.0 - alpha / 2.0
        if self.status == "observed":
            if not self.point_curve_crossed:
                raise ValueError("observed crossover requires a point-curve crossing")
            if self.bootstrap_crossing_fraction <= upper_support:
                raise ValueError("observed crossover requires sufficient bootstrap support")
            if (
                self.point_estimate is None
                or self.interval_lower is None
                or self.interval_upper is None
                or self.interval_upper == "positive-infinity"
                or self.censoring != "none"
            ):
                raise ValueError("observed crossover requires uncensored estimate and interval")
            assert isinstance(self.interval_upper, float)
            if self.interval_lower > self.interval_upper:
                raise ValueError("crossover interval bounds must be ordered")
            if (
                self.interval_lower < identity_magnitude
                or self.interval_upper > self.tested_maximum
            ):
                raise ValueError("crossover interval must lie within the tested severity grid")
        elif self.status == "not-observed":
            if self.point_curve_crossed:
                raise ValueError("not-observed crossover requires a censored point curve")
            if self.bootstrap_crossing_fraction >= lower_support:
                raise ValueError("not-observed crossover requires negligible bootstrap support")
            if self.point_estimate is not None:
                raise ValueError("not-observed crossover cannot contain an estimate")
            if (
                self.interval_lower != self.tested_maximum
                or self.interval_upper != "positive-infinity"
            ):
                raise ValueError(
                    "not-observed crossover must encode [tested_maximum, positive-infinity]"
                )
            if self.censoring != "right-above-tested-maximum":
                raise ValueError("not-observed crossover must be right-censored")
        else:
            if self.interval_lower is not None or self.interval_upper is not None:
                raise ValueError("undetermined crossover cannot report a two-sided interval")
            if self.censoring != "mixed-bootstrap":
                raise ValueError("undetermined crossover must record mixed-bootstrap censoring")
            if self.point_curve_crossed != (self.point_estimate is not None):
                raise ValueError(
                    "undetermined point estimate must match point-curve crossing status"
                )

        if self.point_estimate is not None and not (
            identity_magnitude <= self.point_estimate <= self.tested_maximum
        ):
            raise ValueError("point_estimate must lie within the tested severity grid")
        return self


def metric_record_json_schema() -> dict[str, object]:
    return METRIC_RECORD_ADAPTER.json_schema(by_alias=True)
