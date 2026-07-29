from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import fusion_fault_bench.procedural_validation as validation_module
from fusion_fault_bench.contracts.io import (
    load_manifest,
    validate_manifest_mapping,
)
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AvailabilityControlManifest,
    CommonModeControlManifest,
    GeometryCrossoverManifest,
    ProceduralSource,
)
from fusion_fault_bench.contracts.procedural_profile_v1 import (
    ProceduralProfileV1,
    load_procedural_profile,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    LocalizationMetricRecord,
    MetricRecordV1Alpha1,
)
from fusion_fault_bench.experiments.procedural import (
    generate_procedural_sequence_metrics,
)
from fusion_fault_bench.procedural_validation import (
    ProceduralManifest,
    build_procedural_validation,
)
from fusion_fault_bench.scenarios.procedural import (
    ProceduralSequence,
    generate_procedural_sequences,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
SMOKE_MANIFEST_PATH = REPOSITORY_ROOT / "examples/manifests/procedural-ci-smoke-v1alpha1.json"
SMOKE_PROFILE_PATH = REPOSITORY_ROOT / "examples/profiles/constant-velocity-ci-smoke-v1.json"
MAIN_PROFILE_PATH = REPOSITORY_ROOT / "examples/profiles/constant-velocity-front-roi-v1.json"
RUN_ID = "run:procedural-validation-test"


def _smoke_manifest() -> GeometryCrossoverManifest:
    manifest = load_manifest(SMOKE_MANIFEST_PATH)
    assert isinstance(manifest, GeometryCrossoverManifest)
    return manifest


def _smoke_profile() -> ProceduralProfileV1:
    return load_procedural_profile(SMOKE_PROFILE_PATH)


def _availability_manifest() -> AvailabilityControlManifest:
    data = _smoke_manifest().model_dump(mode="json", by_alias=True)
    data["kind"] = "availability-control"
    data["experiment"] = "procedural-validation-dropout-test"
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


def _common_mode_manifest() -> CommonModeControlManifest:
    data = _smoke_manifest().model_dump(mode="json", by_alias=True)
    data["kind"] = "common-mode-control"
    data["experiment"] = "procedural-validation-common-mode-test"
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


def _noise_manifest(*, correctly_reported: bool) -> GeometryCrossoverManifest:
    data = _smoke_manifest().model_dump(mode="json", by_alias=True)
    kind = (
        "increased-noise-correctly-reported"
        if correctly_reported
        else "increased-noise-underreported"
    )
    data["experiment"] = f"procedural-validation-{kind}-test"
    data["fault_sweep"] = {
        "kind": kind,
        "target": "camera",
        "axis": "xy",
        "unit": "std-scale",
        "injection_site": "true-error-model",
        "reported_uncertainty": "tracks-actual" if correctly_reported else "nominal",
        "persistence": "sequence-configuration",
        "std_scale_values": [1.0, 2.0, 4.0],
    }
    manifest = validate_manifest_mapping(data)
    assert isinstance(manifest, GeometryCrossoverManifest)
    return manifest


def _smoke_sized_release_fault(manifest_name: str) -> GeometryCrossoverManifest:
    source_manifest = load_manifest(REPOSITORY_ROOT / "examples/manifests" / manifest_name)
    assert isinstance(source_manifest, GeometryCrossoverManifest)
    data = source_manifest.model_dump(mode="json", by_alias=True)
    smoke = _smoke_manifest().model_dump(mode="json", by_alias=True)
    data["experiment"] = f"validation-{source_manifest.experiment}"
    data["source"] = smoke["source"]
    data["roi"] = smoke["roi"]
    data["evaluation"]["bootstrap"]["replicates"] = 200
    manifest = validate_manifest_mapping(data)
    assert isinstance(manifest, GeometryCrossoverManifest)
    return manifest


def _inputs(
    manifest: ProceduralManifest,
) -> tuple[
    ProceduralProfileV1,
    tuple[ProceduralSequence, ...],
    tuple[MetricRecordV1Alpha1, ...],
]:
    profile = _smoke_profile()
    source = manifest.source
    assert isinstance(source, ProceduralSource)
    sequences = generate_procedural_sequences(
        profile,
        split=source.split,
        sequence_count=source.sequence_count,
        data_master_seed=manifest.rng.data_master_seed,
    )
    metrics = generate_procedural_sequence_metrics(
        manifest,
        profile=profile,
        run_id=RUN_ID,
    )
    return profile, sequences, metrics


def _build(manifest: ProceduralManifest):
    profile, sequences, metrics = _inputs(manifest)
    return build_procedural_validation(
        manifest,
        profile=profile,
        run_id=RUN_ID,
        sequences=sequences,
        metrics=metrics,
    )


def test_geometry_smoke_validation_has_exact_semantic_check_sets() -> None:
    manifest = _smoke_manifest()
    first = _build(manifest)
    second = _build(manifest)

    assert first.all_checks_passed
    assert first.model_dump(mode="json", by_alias=True) == second.model_dump(
        mode="json",
        by_alias=True,
    )
    assert len(first.moment_checks) == 14
    assert {check.statistic for check in first.moment_checks} == {
        "mean",
        "variance",
        "within-sensor-covariance",
        "camera-lidar-cross-covariance",
    }
    assert sum(check.statistic == "mean" for check in first.moment_checks) == 4
    assert sum(check.statistic == "variance" for check in first.moment_checks) == 4
    assert sum(check.statistic == "within-sensor-covariance" for check in first.moment_checks) == 2
    assert (
        sum(check.statistic == "camera-lidar-cross-covariance" for check in first.moment_checks)
        == 4
    )
    # Three conditions times four affine methods plus one signed contrast.
    assert len(first.expected_loss_checks) == 3 * 5
    assert all(check.method_id != "performance-oracle" for check in first.expected_loss_checks)
    assert first.identity_comparison.status == "deferred-to-matrix"
    assert first.deterministic_model_checks.identity_row_reconstruction == "passed"
    assert first.resources.implied_sequence_row_count == 4 * 3 * 5


def test_independent_main_split_support_and_double_corruption_mutation() -> None:
    main_profile = load_procedural_profile(MAIN_PROFILE_PATH)
    assert validation_module._independent_split_family_support(
        main_profile,
        data_master_seed=1729,
    )

    truth = np.asarray(((12.0, -2.0), (30.0, 4.0)), dtype=np.float64)
    mutated = validation_module._double_corrupted_camera_reconstruction(
        truth,
        camera_true_translation_m=(1.5, 0.0, 1.5),
        camera_true_quaternion_wxyz=(0.5, -0.5, 0.5, -0.5),
        ego_translation_fault_m=(0.25, 0.0, 0.0),
    )
    assert np.allclose(mutated, truth, rtol=0.0, atol=1e-12)
    assert not np.allclose(
        mutated,
        truth + np.asarray((0.25, 0.0)),
        rtol=0.0,
        atol=1e-12,
    )


def test_double_corruption_gate_rejects_a_non_canceling_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def non_canceling_mutation(
        truth_xy_m: Any,
        **kwargs: Any,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        del kwargs
        return np.asarray(truth_xy_m, dtype=np.float64) + np.asarray((0.25, 0.0))

    monkeypatch.setattr(
        validation_module,
        "_double_corrupted_camera_reconstruction",
        non_canceling_mutation,
    )
    validation = _build(_smoke_manifest())

    assert not validation.oracle_checks.fault_cancellation_mutation_rejected
    assert not validation.oracle_checks.all_checks_passed
    assert not validation.all_checks_passed


def test_dropout_validation_rebuilds_exact_masks_and_endpoints() -> None:
    manifest = _availability_manifest()
    validation = _build(manifest)

    assert validation.all_checks_passed
    assert validation.expected_loss_checks == ()
    assert validation.dropout_validation.status == "applicable"
    assert validation.dropout_validation.exact_mask_comparison_count == 4 * 3
    assert validation.dropout_validation.maximum_mask_discrepancy == 0
    assert validation.dropout_validation.frame_sharing_passed
    assert validation.dropout_validation.nesting_passed
    assert validation.dropout_validation.endpoint_behavior_passed
    assert validation.identity_comparison.status == "deferred-to-matrix"
    assert validation.deterministic_model_checks.identity_row_reconstruction == "passed"
    assert validation.resources.implied_sequence_row_count == 4 * 3 * 4 * 3


def test_common_mode_validation_checks_disagreement_and_no_peer_identity() -> None:
    validation = _build(_common_mode_manifest())

    assert validation.all_checks_passed
    assert validation.common_mode_validation.status == "applicable"
    assert validation.common_mode_validation.maximum_disagreement_discrepancy_m <= 1e-12
    assert validation.identity_comparison.status == "not-applicable"
    assert len(validation.expected_loss_checks) == 3 * 3


@pytest.mark.parametrize("correctly_reported", [True, False])
def test_noise_validation_enforces_reported_covariance_and_expected_curve(
    correctly_reported: bool,
) -> None:
    validation = _build(_noise_manifest(correctly_reported=correctly_reported))

    assert validation.all_checks_passed
    assert len(validation.expected_loss_checks) == 3 * 5
    contrasts = [
        check.expected_value_m2
        for check in validation.expected_loss_checks
        if check.metric_name == "fused-minus-healthy"
    ]
    assert contrasts == sorted(contrasts)
    if correctly_reported:
        assert all(value < 0.0 for value in contrasts)


@pytest.mark.parametrize(
    "manifest_name",
    (
        "procedural-lidar-y-bias-v1alpha1.json",
        "procedural-camera-calibration-yaw-v1alpha1.json",
        "procedural-camera-timestamp-offset-v1alpha1.json",
    ),
)
def test_all_signed_fault_equations_pass_independent_smoke_sized_validation(
    manifest_name: str,
) -> None:
    validation = _build(_smoke_sized_release_fault(manifest_name))

    assert validation.all_checks_passed
    assert len(validation.expected_loss_checks) == 55
    assert validation.deterministic_model_checks.expected_curve_response == "passed"


def test_validation_rejects_a_mutated_nonidentity_sequence_loss() -> None:
    manifest = _smoke_manifest()
    profile, sequences, metrics = _inputs(manifest)
    mutable = list(metrics)
    target_index = next(
        index
        for index, record in enumerate(mutable)
        if isinstance(record, LocalizationMetricRecord)
        and record.severity.index == 1
        and record.method_id == "fixed-fusion"
    )
    target = cast(LocalizationMetricRecord, mutable[target_index])
    assert target.value is not None
    mutable[target_index] = target.model_copy(update={"value": target.value + 1.0})

    with pytest.raises(ValueError, match="independent reconstruction"):
        build_procedural_validation(
            manifest,
            profile=profile,
            run_id=RUN_ID,
            sequences=sequences,
            metrics=mutable,
        )


def test_dropout_reference_rejects_mutated_production_uniforms() -> None:
    manifest = _availability_manifest()
    profile, sequences, metrics = _inputs(manifest)
    altered = np.array(sequences[0].fault_uniform_by_frame, copy=True)
    altered[0] = 0.0 if altered[0] >= 0.5 else 0.999
    mutated_sequences = (
        replace(sequences[0], fault_uniform_by_frame=altered),
        *sequences[1:],
    )

    with pytest.raises(ValueError, match="dropout"):
        build_procedural_validation(
            manifest,
            profile=profile,
            run_id=RUN_ID,
            sequences=mutated_sequences,
            metrics=metrics,
        )


@pytest.mark.parametrize(
    "field_name",
    ("eligible_truth_xy_m", "eligible_velocity_xy_mps"),
)
def test_independent_profile_gate_rejects_mutated_eligible_extraction(
    field_name: str,
) -> None:
    manifest = _smoke_manifest()
    profile, sequences, metrics = _inputs(manifest)
    changed = np.roll(np.asarray(getattr(sequences[0], field_name)), shift=1, axis=0)
    mutated_sequences = (
        replace(sequences[0], **{field_name: changed}),
        *sequences[1:],
    )

    validation = build_procedural_validation(
        manifest,
        profile=profile,
        run_id=RUN_ID,
        sequences=mutated_sequences,
        metrics=metrics,
    )

    assert not validation.eligibility.eligibility_invariant
    assert not validation.all_checks_passed


def test_profile_or_metric_provenance_mismatch_fails_closed() -> None:
    manifest = _smoke_manifest()
    profile, sequences, metrics = _inputs(manifest)
    copied: dict[str, Any] = copy.deepcopy(manifest.model_dump(mode="json", by_alias=True))
    copied["experiment"] = "different-experiment"
    different = validate_manifest_mapping(copied)
    assert isinstance(different, GeometryCrossoverManifest)

    with pytest.raises(ValueError, match="provenance"):
        build_procedural_validation(
            different,
            profile=profile,
            run_id=RUN_ID,
            sequences=sequences,
            metrics=metrics,
        )
