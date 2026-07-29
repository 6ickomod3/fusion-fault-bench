from __future__ import annotations

import ast
import inspect
import math
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import cast

import pytest

import fusion_fault_bench.reference.analytic as analytic_module
from fusion_fault_bench.contracts.io import load_manifest
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AdditivePositionBiasFault,
    AnalyticCrossoverManifest,
    UnderreportedNoiseFault,
)
from fusion_fault_bench.reference.analytic import (
    ANALYTIC_GAUSSIAN_METHODS,
    AnalyticCondition,
    continuous_crossover_root,
    expand_conditions,
    first_zero_grid_root,
    gaussian_population,
    grid_crossover_root,
    pava_non_decreasing,
    population_contrast,
    population_crossover_references,
    population_points,
)

MANIFESTS = Path("examples/manifests")


def _manifest(name: str) -> AnalyticCrossoverManifest:
    manifest = load_manifest(MANIFESTS / name)
    assert isinstance(manifest, AnalyticCrossoverManifest)
    return manifest


def _condition(
    manifest: AnalyticCrossoverManifest,
    *,
    index: int,
    direction: str,
) -> AnalyticCondition:
    return next(
        condition
        for condition in expand_conditions(manifest)
        if condition.index == index and condition.direction == direction
    )


def test_reference_module_has_an_independent_import_boundary() -> None:
    tree = ast.parse(inspect.getsource(analytic_module))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
    forbidden_prefixes = {
        "fusion_fault_bench.rng",
        "fusion_fault_bench.fusion",
        "fusion_fault_bench.fault",
        "fusion_fault_bench.experiments",
        "fusion_fault_bench.evaluation",
        "fusion_fault_bench.inference",
    }

    assert not {
        module
        for module in imported_modules
        for prefix in forbidden_prefixes
        if module == prefix or module.startswith(f"{prefix}.")
    }


def test_condition_expansion_is_stable_and_identity_is_emitted_once() -> None:
    bias = _manifest("analytic-bias-v1alpha1.json")
    noise = _manifest("analytic-noise-correct-v1alpha1.json")

    bias_conditions = expand_conditions(bias)
    assert len(bias_conditions) == 13
    assert [(item.index, item.magnitude, item.direction) for item in bias_conditions[:5]] == [
        (0, 0.0, "identity"),
        (1, 0.25, "negative"),
        (1, 0.25, "positive"),
        (2, 0.5, "negative"),
        (2, 0.5, "positive"),
    ]
    assert len([item for item in bias_conditions if item.direction == "identity"]) == 1

    assert [(item.index, item.magnitude, item.direction) for item in expand_conditions(noise)] == [
        (0, 1.0, "identity"),
        (1, 1.25, "increase"),
        (2, 1.5, "increase"),
        (3, 2.0, "increase"),
        (4, 4.0, "increase"),
    ]


def test_bias_population_matches_documented_rational_goldens() -> None:
    manifest = _manifest("analytic-bias-v1alpha1.json")
    expected_base_contrast = Fraction(-547, 50024)
    expected_actual_variance = (Fraction(9, 148), Fraction(9, 169))

    for condition in expand_conditions(manifest):
        fused = gaussian_population(manifest, condition, "fixed-fusion")
        signed_magnitude = (
            -condition.magnitude if condition.direction == "negative" else condition.magnitude
        )
        if condition.direction == "identity":
            signed_magnitude = 0.0

        assert fused.mean_xy_m == pytest.approx((signed_magnitude / 37.0, 0.0), abs=1e-15)
        assert fused.actual_variance_xy_m2 == pytest.approx(
            tuple(float(value) for value in expected_actual_variance),
            abs=1e-15,
        )
        assert fused.reported_variance_xy_m2 == pytest.approx(
            tuple(float(value) for value in expected_actual_variance),
            abs=1e-15,
        )
        expected_contrast = (
            expected_base_contrast + Fraction.from_float(condition.magnitude) ** 2 / 1369
        )
        assert population_contrast(manifest, condition) == pytest.approx(
            float(expected_contrast),
            abs=1e-15,
        )

    identity_fused = gaussian_population(
        manifest,
        _condition(manifest, index=0, direction="identity"),
        "fixed-fusion",
    )
    expected_mse_variance = 2.0 * (float(Fraction(9, 148)) ** 2 + float(Fraction(9, 169)) ** 2)
    assert identity_fused.mse_variance_m4 == pytest.approx(expected_mse_variance, abs=1e-15)
    assert identity_fused.mse_standard_error_m2(200) == pytest.approx(
        (expected_mse_variance / 200.0) ** 0.5,
        abs=1e-15,
    )
    with pytest.raises(ValueError, match="positive"):
        identity_fused.mse_standard_error_m2(0)


def test_bias_grid_and_continuous_roots_match_documented_rationals() -> None:
    manifest = _manifest("analytic-bias-v1alpha1.json")
    expected_grid_root = Fraction(31055, 8112)
    expected_continuous_root = (Fraction(20239, 1352)) ** Fraction(1, 2)

    references = population_crossover_references(manifest)
    assert [reference.direction for reference in references] == ["negative", "positive"]
    for reference in references:
        assert reference.grid_status == "crossed"
        assert reference.grid_censoring == "none"
        assert reference.grid_point_estimate == pytest.approx(
            float(expected_grid_root),
            abs=manifest.analytic_validation.grid_crossover_abs_tolerance,
        )
        assert reference.continuous_status == "finite"
        assert reference.continuous_point_estimate == pytest.approx(
            float(expected_continuous_root),
            abs=manifest.analytic_validation.grid_crossover_abs_tolerance,
        )


def test_correctly_reported_noise_matches_exact_contrasts_and_has_no_finite_root() -> None:
    manifest = _manifest("analytic-noise-correct-v1alpha1.json")
    exact_contrasts = (
        Fraction(-547, 50024),
        Fraction(-269, 36640),
        Fraction(-2399, 457888),
        Fraction(-2113, 697160),
        Fraction(-8377, 10750664),
    )

    for condition, expected in zip(expand_conditions(manifest), exact_contrasts, strict=True):
        camera = gaussian_population(manifest, condition, "camera-only")
        fused = gaussian_population(manifest, condition, "fixed-fusion")
        assert camera.actual_variance_xy_m2 == camera.reported_variance_xy_m2
        assert fused.actual_variance_xy_m2 == pytest.approx(
            fused.reported_variance_xy_m2,
            abs=1e-15,
        )
        assert population_contrast(manifest, condition) == pytest.approx(
            float(expected),
            abs=1e-15,
        )

    assert continuous_crossover_root(manifest) is None
    assert population_crossover_references(manifest) == (
        analytic_module.CrossoverReference(
            direction="increase",
            severity_unit="std-scale",
            tested_maximum=4.0,
            grid_status="not-crossed",
            grid_point_estimate=None,
            grid_censoring="right-above-tested-maximum",
            continuous_status="no-finite-root",
            continuous_point_estimate=None,
        ),
    )


def test_underreported_noise_matches_documented_population_formula_and_roots() -> None:
    manifest = _manifest("analytic-noise-underreported-v1alpha1.json")
    base = Fraction(-547, 50024)
    coefficient = Fraction(1489149, 156400036)

    for condition in expand_conditions(manifest):
        camera = gaussian_population(manifest, condition, "camera-only")
        fused = gaussian_population(manifest, condition, "fixed-fusion")
        assert camera.actual_variance_xy_m2 == pytest.approx(
            (2.25 * condition.magnitude**2, 0.36 * condition.magnitude**2),
            abs=1e-15,
        )
        assert camera.reported_variance_xy_m2 == (2.25, 0.36)
        assert fused.reported_variance_xy_m2 == pytest.approx(
            (float(Fraction(9, 148)), float(Fraction(9, 169))),
            abs=1e-15,
        )
        expected = float(base) + (condition.magnitude**2 - 1.0) * float(coefficient)
        assert population_contrast(manifest, condition) == pytest.approx(expected, abs=1e-15)

    reference = population_crossover_references(manifest)[0]
    assert reference.grid_point_estimate == pytest.approx(
        float(Fraction(47931991, 32761278)),
        abs=manifest.analytic_validation.grid_crossover_abs_tolerance,
    )
    assert reference.continuous_point_estimate == pytest.approx(
        float(Fraction(6398689, 2978298) ** Fraction(1, 2)),
        abs=manifest.analytic_validation.grid_crossover_abs_tolerance,
    )


def test_reference_is_generic_across_fault_target_and_bias_axis() -> None:
    manifest = _manifest("analytic-bias-v1alpha1.json")
    fault = cast(AdditivePositionBiasFault, manifest.fault_sweep)
    generic = manifest.model_copy(
        update={
            "fault_sweep": fault.model_copy(update={"target": "lidar", "axis": "y"}),
        }
    )
    positive = _condition(generic, index=1, direction="positive")

    healthy = gaussian_population(generic, positive, "camera-only")
    target = gaussian_population(generic, positive, "lidar-only")
    fused = gaussian_population(generic, positive, "fixed-fusion")
    assert healthy.mean_xy_m == (0.0, 0.0)
    assert target.mean_xy_m == (0.0, 0.25)
    assert fused.mean_xy_m == pytest.approx((0.0, 0.25 * 144.0 / 169.0), abs=1e-15)
    assert continuous_crossover_root(generic) is not None


def test_reference_is_generic_for_lidar_underreported_noise() -> None:
    manifest = _manifest("analytic-noise-underreported-v1alpha1.json")
    fault = cast(UnderreportedNoiseFault, manifest.fault_sweep)
    generic = manifest.model_copy(
        update={"fault_sweep": fault.model_copy(update={"target": "lidar"})}
    )
    condition = _condition(generic, index=2, direction="increase")

    lidar = gaussian_population(generic, condition, "lidar-only")
    fused = gaussian_population(generic, condition, "fixed-fusion")
    assert lidar.actual_variance_xy_m2 == pytest.approx(
        (0.0625 * 1.5**2, 0.0625 * 1.5**2),
        abs=1e-15,
    )
    assert lidar.reported_variance_xy_m2 == (0.0625, 0.0625)
    assert fused.actual_variance_xy_m2 != fused.reported_variance_xy_m2
    assert continuous_crossover_root(generic) is not None


def test_population_points_are_condition_then_method_ordered() -> None:
    manifest = _manifest("analytic-noise-correct-v1alpha1.json")
    points = population_points(manifest)

    assert len(points) == len(expand_conditions(manifest)) * 3
    assert tuple(point.population.method_id for point in points[:3]) == (ANALYTIC_GAUSSIAN_METHODS)
    assert all(point.condition == points[0].condition for point in points[:3])


def test_independent_pava_and_grid_root_cover_pooling_and_boundaries() -> None:
    assert pava_non_decreasing((0.0, -2.0, 1.0, 0.0)) == (-1.0, -1.0, 0.5, 0.5)
    assert (
        grid_crossover_root(
            (0.0, 1.0, 2.0, 3.0),
            (-2.0, -1.0, 1.0, 2.0),
            zero_tolerance=1e-12,
        )
        == 1.5
    )
    assert (
        first_zero_grid_root(
            (0.0, 1.0, 2.0),
            (-1.0, 0.0, 1.0),
            zero_tolerance=1e-12,
        )
        == 1.0
    )
    assert (
        first_zero_grid_root(
            (0.0, 1.0),
            (-2.0, -1.0),
            zero_tolerance=1e-12,
        )
        is None
    )
    assert (
        first_zero_grid_root(
            (0.0, 1.0),
            (0.0, 1.0),
            zero_tolerance=1e-12,
        )
        == 0.0
    )


@pytest.mark.parametrize("values", [(), (0.0, math.nan)])
def test_pava_rejects_empty_or_nonfinite_values(values: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        pava_non_decreasing(values)


@pytest.mark.parametrize(
    ("magnitudes", "values", "tolerance", "message"),
    [
        ((0.0,), (-1.0,), 1e-12, "aligned vectors"),
        ((0.0, 0.0), (-1.0, 1.0), 1e-12, "strictly increasing"),
        ((0.0, 1.0), (1.0, -1.0), 1e-12, "nondecreasing"),
        ((0.0, 1.0), (-1.0, 1.0), 0.0, "finite and positive"),
    ],
)
def test_first_zero_grid_root_rejects_invalid_inputs(
    magnitudes: tuple[float, ...],
    values: tuple[float, ...],
    tolerance: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        first_zero_grid_root(magnitudes, values, zero_tolerance=tolerance)


def test_gaussian_population_rejects_a_condition_from_another_manifest() -> None:
    bias = _manifest("analytic-bias-v1alpha1.json")
    noise = _manifest("analytic-noise-correct-v1alpha1.json")

    with pytest.raises(ValueError, match="does not match"):
        gaussian_population(bias, expand_conditions(noise)[0], "fixed-fusion")


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"index": 99}, "outside"),
        ({"magnitude": 0.1}, "magnitude"),
        ({"direction": "positive"}, "direction"),
    ],
)
def test_gaussian_population_rejects_malformed_conditions(
    update: dict[str, object],
    message: str,
) -> None:
    manifest = _manifest("analytic-noise-correct-v1alpha1.json")
    identity = expand_conditions(manifest)[0]
    malformed = replace(identity, **update)

    with pytest.raises(ValueError, match=message):
        gaussian_population(manifest, malformed, "fixed-fusion")


def test_gaussian_population_rejects_non_gaussian_method() -> None:
    manifest = _manifest("analytic-bias-v1alpha1.json")
    condition = expand_conditions(manifest)[0]
    invalid_method = cast(analytic_module.GaussianMethodId, "performance-oracle")

    with pytest.raises(ValueError, match="not a Gaussian analytic method"):
        gaussian_population(manifest, condition, invalid_method)
