"""Strict contracts for the M1 deterministic scientific artifact."""

from __future__ import annotations

import math
from typing import Annotated, Literal, Self

from pydantic import Field, FiniteFloat, PositiveFloat, model_validator

from fusion_fault_bench.contracts.common import ContractModel
from fusion_fault_bench.contracts.result_v1alpha1 import (
    Digest,
    Identifier,
    SeverityCoordinate,
)

MANIFEST_FILE = "manifest.json"
SEQUENCE_METRICS_FILE = "sequence-metrics.ndjson"
AGGREGATE_METRICS_FILE = "aggregate-metrics.ndjson"
CROSSOVERS_FILE = "crossovers.ndjson"
ANALYTIC_VALIDATION_FILE = "analytic-validation.json"
PAYLOAD_INDEX_FILE = "payload-index.json"
RUN_FILE = "run.json"
SUCCESS_FILE = "_SUCCESS"

INDEXED_PAYLOAD_PATHS = (
    MANIFEST_FILE,
    SEQUENCE_METRICS_FILE,
    AGGREGATE_METRICS_FILE,
    CROSSOVERS_FILE,
    ANALYTIC_VALIDATION_FILE,
)
ARTIFACT_PATHS = (*INDEXED_PAYLOAD_PATHS, PAYLOAD_INDEX_FILE, RUN_FILE, SUCCESS_FILE)

type IndexedPayloadPath = Literal[
    "manifest.json",
    "sequence-metrics.ndjson",
    "aggregate-metrics.ndjson",
    "crossovers.ndjson",
    "analytic-validation.json",
]
type AnalyticMethodId = Literal["camera-only", "lidar-only", "fixed-fusion"]
type AnalyticSeverityUnit = Literal["m", "std-scale"]
type CrossoverDirection = Literal["negative", "positive", "increase"]
type FloatPair = tuple[FiniteFloat, FiniteFloat]


class PayloadFileEntryV1Alpha1(ContractModel):
    """One exact byte-level member committed by the payload index."""

    path: IndexedPayloadPath
    byte_length: Annotated[int, Field(ge=1, le=512 * 1024 * 1024)]
    sha256: Digest


class PayloadIndexV1Alpha1(ContractModel):
    """Deterministic envelope over the five scientific payload members."""

    schema_id: Literal["ffb.payload-index/v1alpha1"] = Field(alias="schema")
    artifact_contract: Literal["ffb.scientific-payload/v1"]
    run_id: Identifier
    manifest_sha256: Digest
    files: Annotated[
        tuple[PayloadFileEntryV1Alpha1, ...],
        Field(min_length=5, max_length=5),
    ]

    @model_validator(mode="after")
    def require_exact_file_order(self) -> Self:
        paths = tuple(entry.path for entry in self.files)
        if paths != INDEXED_PAYLOAD_PATHS:
            raise ValueError("payload index files must use the fixed five-member order")
        return self


class SuccessMarkerV1Alpha1(ContractModel):
    """Final completion record committing to scientific and run provenance bytes."""

    schema_id: Literal["ffb.success/v1alpha1"] = Field(alias="schema")
    artifact_sha256: Digest
    run_sha256: Digest


class AnalyticPopulationPointV1Alpha1(ContractModel):
    """One closed-form and empirical check for a method at one condition."""

    severity: SeverityCoordinate
    method_id: AnalyticMethodId
    mean_unit: Literal["m"]
    variance_unit: Literal["m^2"]
    loss_unit: Literal["m^2"]
    expected_mean_xy_m: FloatPair
    expected_actual_variance_xy_m2: FloatPair
    expected_reported_variance_xy_m2: FloatPair
    expected_mse_m2: Annotated[FiniteFloat, Field(ge=0.0)]
    empirical_mse_m2: Annotated[FiniteFloat, Field(ge=0.0)]
    analytic_mse_standard_error_m2: PositiveFloat
    absolute_standardized_error: Annotated[FiniteFloat, Field(ge=0.0)]
    monte_carlo_passed: bool

    @model_validator(mode="after")
    def validate_analytic_point(self) -> Self:
        if self.severity.unit not in {"m", "std-scale"}:
            raise ValueError("analytic point severity unit must be m or std-scale")
        if any(value < 0.0 for value in self.expected_actual_variance_xy_m2):
            raise ValueError("expected actual variances must be non-negative")
        if any(value <= 0.0 for value in self.expected_reported_variance_xy_m2):
            raise ValueError("expected reported variances must be positive")
        expected_standardized_error = (
            abs(self.empirical_mse_m2 - self.expected_mse_m2) / self.analytic_mse_standard_error_m2
        )
        if not math.isclose(
            self.absolute_standardized_error,
            expected_standardized_error,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("absolute standardized error disagrees with MSE and analytic SE")
        return self


class AnalyticCrossoverReferenceV1Alpha1(ContractModel):
    """Population crossover reference, separate from finite-sample inference."""

    direction: CrossoverDirection
    severity_unit: AnalyticSeverityUnit
    tested_maximum: FiniteFloat
    grid_status: Literal["crossed", "not-crossed"]
    grid_point_estimate: FiniteFloat | None
    grid_censoring: Literal["none", "right-above-tested-maximum"]
    continuous_status: Literal["finite", "no-finite-root"]
    continuous_point_estimate: FiniteFloat | None

    @model_validator(mode="after")
    def validate_reference_status(self) -> Self:
        identity = 1.0 if self.severity_unit == "std-scale" else 0.0
        if self.tested_maximum <= identity:
            raise ValueError("tested maximum must exceed the identity condition")
        if self.grid_status == "crossed":
            if self.grid_point_estimate is None or self.grid_censoring != "none":
                raise ValueError("crossed grid reference requires an uncensored root")
            if not identity <= self.grid_point_estimate <= self.tested_maximum:
                raise ValueError("grid root must lie within the tested interval")
        elif (
            self.grid_point_estimate is not None
            or self.grid_censoring != "right-above-tested-maximum"
        ):
            raise ValueError("not-crossed grid reference must be right-censored")
        if self.continuous_status == "finite":
            if self.continuous_point_estimate is None:
                raise ValueError("finite continuous status requires a root")
            if self.continuous_point_estimate < identity:
                raise ValueError("continuous root must not precede identity")
        elif self.continuous_point_estimate is not None:
            raise ValueError("no-finite-root status requires a null root")
        return self


class AnalyticValidationV1Alpha1(ContractModel):
    """Independent diagonal-Gaussian validation evidence for an M1 run."""

    schema_id: Literal["ffb.analytic-validation/v1alpha1"] = Field(alias="schema")
    run_id: Identifier
    manifest_sha256: Digest
    reference_model: Literal["independent-diagonal-gaussian-closed-form-v1"]
    variance_representation: Literal["diagonal-xy-m2"]
    monte_carlo_standard_error_multiplier: Annotated[
        FiniteFloat,
        Field(ge=6.0, le=6.0),
    ]
    population_points: Annotated[
        tuple[AnalyticPopulationPointV1Alpha1, ...],
        Field(min_length=3),
    ]
    crossover_references: Annotated[
        tuple[AnalyticCrossoverReferenceV1Alpha1, ...],
        Field(min_length=1, max_length=2),
    ]
    all_monte_carlo_checks_passed: bool

    @model_validator(mode="after")
    def validate_order_and_checks(self) -> Self:
        if len(self.population_points) % 3 != 0:
            raise ValueError("analytic population points must contain complete method triples")
        expected_methods = ("camera-only", "lidar-only", "fixed-fusion")
        for offset in range(0, len(self.population_points), 3):
            group = self.population_points[offset : offset + 3]
            if tuple(point.method_id for point in group) != expected_methods:
                raise ValueError("analytic population methods must use the fixed order")
            if any(point.severity != group[0].severity for point in group[1:]):
                raise ValueError("analytic method triples must share one severity")
        directions = tuple(reference.direction for reference in self.crossover_references)
        if directions not in {("negative", "positive"), ("increase",)}:
            raise ValueError("analytic crossover references have an invalid direction order")
        for point in self.population_points:
            expected_pass = (
                point.absolute_standardized_error <= self.monte_carlo_standard_error_multiplier
            )
            if point.monte_carlo_passed != expected_pass:
                raise ValueError("Monte Carlo point pass flag disagrees with the threshold")
        if self.all_monte_carlo_checks_passed != all(
            point.monte_carlo_passed for point in self.population_points
        ):
            raise ValueError("top-level Monte Carlo pass flag must be the conjunction")
        return self
