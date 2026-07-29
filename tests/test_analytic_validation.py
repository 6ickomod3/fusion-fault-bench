from __future__ import annotations

from pathlib import Path

import pytest

from fusion_fault_bench.contracts.io import load_manifest
from fusion_fault_bench.contracts.manifest_v1alpha1 import AnalyticCrossoverManifest
from fusion_fault_bench.experiments.analytic import generate_analytic_sequence_metrics
from fusion_fault_bench.validation import build_analytic_validation

_MANIFESTS = (
    "analytic-bias-v1alpha1.json",
    "analytic-noise-correct-v1alpha1.json",
    "analytic-noise-underreported-v1alpha1.json",
)


def _manifest(name: str) -> AnalyticCrossoverManifest:
    manifest = load_manifest(Path("examples/manifests") / name)
    assert isinstance(manifest, AnalyticCrossoverManifest)
    return manifest


@pytest.mark.parametrize("name", _MANIFESTS)
def test_analytic_validation_is_complete_and_passes(name: str) -> None:
    manifest = _manifest(name)
    run_id = "run:analytic-validation-test"
    metrics = generate_analytic_sequence_metrics(manifest, run_id=run_id)

    validation = build_analytic_validation(manifest, run_id=run_id, metrics=metrics)

    condition_count = len(validation.population_points) // 3
    assert condition_count in {5, 13}
    assert validation.all_monte_carlo_checks_passed
    assert tuple(point.method_id for point in validation.population_points[:3]) == (
        "camera-only",
        "lidar-only",
        "fixed-fusion",
    )
    assert all(point.monte_carlo_passed for point in validation.population_points)
    assert max(point.absolute_standardized_error for point in validation.population_points) < 1.1


def test_population_crossover_references_match_preregistration() -> None:
    expected = {
        "analytic-bias-v1alpha1.json": (
            ("negative", "crossed", 3.8282790927021697, "finite", 3.8690663675120667),
            ("positive", "crossed", 3.8282790927021697, "finite", 3.8690663675120667),
        ),
        "analytic-noise-correct-v1alpha1.json": (
            ("increase", "not-crossed", None, "no-finite-root", None),
        ),
        "analytic-noise-underreported-v1alpha1.json": (
            ("increase", "crossed", 1.46306841265472, "finite", 1.4657551414886731),
        ),
    }
    for name, expected_references in expected.items():
        manifest = _manifest(name)
        run_id = "run:analytic-validation-test"
        validation = build_analytic_validation(
            manifest,
            run_id=run_id,
            metrics=generate_analytic_sequence_metrics(manifest, run_id=run_id),
        )
        assert len(validation.crossover_references) == len(expected_references)
        for actual, reference in zip(
            validation.crossover_references,
            expected_references,
            strict=True,
        ):
            direction, grid_status, grid_root, continuous_status, continuous_root = reference
            assert actual.direction == direction
            assert actual.grid_status == grid_status
            assert actual.grid_point_estimate == pytest.approx(grid_root)
            assert actual.continuous_status == continuous_status
            assert actual.continuous_point_estimate == pytest.approx(continuous_root)


def test_validation_rejects_mismatched_provenance() -> None:
    manifest = _manifest("analytic-noise-correct-v1alpha1.json")
    metrics = generate_analytic_sequence_metrics(manifest, run_id="run:original")

    with pytest.raises(ValueError, match="mismatched metric provenance"):
        build_analytic_validation(manifest, run_id="run:different", metrics=metrics)


def test_validation_rejects_missing_sequence_value() -> None:
    manifest = _manifest("analytic-noise-underreported-v1alpha1.json")
    run_id = "run:missing-row"
    metrics = generate_analytic_sequence_metrics(manifest, run_id=run_id)

    with pytest.raises(ValueError, match="one value per sequence"):
        build_analytic_validation(manifest, run_id=run_id, metrics=metrics[1:])


def test_validation_rejects_undefined_metric() -> None:
    manifest = _manifest("analytic-noise-correct-v1alpha1.json")
    run_id = "run:undefined-row"
    metrics = generate_analytic_sequence_metrics(manifest, run_id=run_id)
    undefined = metrics[0].model_copy(
        update={
            "status": "undefined",
            "value": None,
            "valid_object_frame_count": 0,
        }
    )

    with pytest.raises(ValueError, match="defined matched-center MSE"):
        build_analytic_validation(
            manifest,
            run_id=run_id,
            metrics=(undefined, *metrics[1:]),
        )
