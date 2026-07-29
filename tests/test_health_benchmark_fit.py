# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fusion_fault_bench.health_benchmark as benchmark
from fusion_fault_bench.contracts.health_result_v1 import (
    CROSS_THRESHOLDS,
    SELF_THRESHOLDS,
    HealthFitSummaryV1,
)
from fusion_fault_bench.contracts.health_v1 import (
    M4_HEALTH_INTENT_SHA256,
    load_health_benchmark_intent,
)
from fusion_fault_bench.health import HealthCalibration
from fusion_fault_bench.health_fit import (
    CANDIDATE_COUNT,
    HealthFeatureTrace,
    UnscoredHealthFrame,
    UnscoredNumericChannel,
    ValidationConditionRegret,
    select_health_thresholds,
)
from fusion_fault_bench.scenarios.health import (
    HealthFaultSpec,
    generate_health_base_sequences,
)

ROOT = Path(__file__).resolve().parents[1]


def _intent():
    return load_health_benchmark_intent(source_root=ROOT).intent


def _dummy_calibration(*, count: int = 1) -> HealthCalibration:
    values = np.zeros(count, dtype=np.float64)
    return HealthCalibration(
        camera_self_mean=values,
        camera_self_maximum=values,
        lidar_self_mean=values,
        lidar_self_maximum=values,
        camera_from_lidar_cross_mean=values,
        camera_from_lidar_cross_maximum=values,
        lidar_from_camera_cross_mean=values,
        lidar_from_camera_cross_maximum=values,
    )


def _defined_channel(value: float = 1.0) -> UnscoredNumericChannel:
    return UnscoredNumericChannel(
        status="defined",
        mature_object_count=6,
        current_object_count=6,
        mature_fraction=1.0,
        mean_nis=value,
        maximum_nis=value,
    )


def _synthetic_feature_trace() -> HealthFeatureTrace:
    channel = _defined_channel()
    return HealthFeatureTrace(
        frames=tuple(
            UnscoredHealthFrame(
                reference_time_s=frame_index * 0.1,
                camera_available=True,
                lidar_available=True,
                camera_timestamp_suspicious=False,
                lidar_timestamp_suspicious=False,
                camera_missing_fraction_last_four=0.0,
                lidar_missing_fraction_last_four=0.0,
                camera_self=channel,
                lidar_self=channel,
                camera_from_lidar_cross=channel,
                lidar_from_camera_cross=channel,
            )
            for frame_index in range(48)
        )
    )


def _zero_selection(
    cases: tuple[benchmark.HealthCaseDescriptor, ...],
):
    regrets = tuple(
        ValidationConditionRegret(
            condition_id=case.condition_id,
            target="camera" if case.fault.target == "camera" else "lidar",
            family=case.fault.family,
            regret_m2_by_candidate_sequence=np.zeros((36, 200)),
        )
        for case in cases
    )
    return (
        select_health_thresholds(
            clean_regression_m2_by_candidate_sequence=np.zeros((36, 200)),
            clean_coverage_by_candidate_sequence=np.ones((36, 200)),
            fixed_clean_coverage_by_sequence=np.ones(200),
            false_alert_starts_by_candidate_sequence=np.zeros((36, 200)),
            condition_regrets=regrets,
        ),
        regrets,
    )


def test_expands_globally_unique_value_level_case_catalog() -> None:
    intent = _intent()
    validation = benchmark.expand_validation_cases(intent)
    test = benchmark.expand_test_cases(intent)

    assert len(validation) == 33
    assert len(test) == 47
    assert len({case.condition_id for case in (*validation, *test)}) == 80
    assert all(case.condition_id == case.value_id for case in (*validation, *test))
    assert validation[0].condition_group_id == "validation-main-clean"
    assert validation[0].population == "main-validation"
    assert validation[0].fault.family == "identity"
    assert test[-1].condition_group_id == "test-cold-start-lidar-y-bias"
    assert test[-1].population == "main-test"
    assert test[-1].fault.schedule == "cold_start"

    selected = benchmark._selection_cases(intent, validation)
    assert len(selected) == 20
    assert tuple(dict.fromkeys(case.condition_group_id for case in selected)) == (
        intent.threshold_selection.selection_conditions
    )
    assert {case.fault.family for case in selected}.isdisjoint({"timestamp-offset", "dropout"})


def test_case_descriptor_rejects_noncanonical_or_wrong_population() -> None:
    scenario = benchmark.HealthScenarioMetadata(
        population="main-validation",
        fault=benchmark._IDENTITY_FAULT,
    )
    with pytest.raises(ValueError, match="canonical"):
        benchmark.HealthCaseDescriptor(
            value_id="wrong",
            condition_id="wrong",
            condition_group_id="validation-main-clean",
            value_index=0,
            stage="validation",
            role="clean",
            scenario=scenario,
        )
    with pytest.raises(ValueError, match="main validation"):
        benchmark.HealthCaseDescriptor(
            value_id="validation-main-clean.value-00",
            condition_id="validation-main-clean.value-00",
            condition_group_id="validation-main-clean",
            value_index=0,
            stage="validation",
            role="clean",
            scenario=benchmark.HealthScenarioMetadata(
                population="main-test",
                fault=benchmark._IDENTITY_FAULT,
            ),
        )


def test_loads_exact_profiles_and_rejects_swapped_profile(tmp_path: Path) -> None:
    intent = _intent()
    profiles = benchmark.load_health_population_profiles(intent, source_root=ROOT)
    assert profiles.main_profile.profile_id == "constant-velocity-front-roi-v1"
    assert profiles.edge_profile.profile_id == "constant-velocity-fov-edge-v1"
    assert profiles.main_profile_sha256 == intent.source_population.profile_sha256
    assert profiles.edge_profile_sha256 == intent.source_population.edge_profile_sha256

    with pytest.raises(FileNotFoundError):
        benchmark.load_health_population_profiles(intent, source_root=tmp_path)


def test_clean_candidate_cache_and_causality_checks_use_real_observations() -> None:
    profiles = benchmark.load_health_population_profiles(_intent(), source_root=ROOT)
    base = generate_health_base_sequences(
        profiles.main_profile,
        split="validation",
        sequence_count=2,
        data_master_seed=1729,
    )[0]
    calibration = _dummy_calibration()
    cached = benchmark._cache_clean_validation((base,), calibration)

    regression, coverage, fixed_coverage, false_alerts = benchmark._evaluate_clean_candidates(
        cached
    )
    assert regression.shape == (36, 1)
    assert coverage.shape == (36, 1)
    assert fixed_coverage.shape == (1,)
    assert false_alerts.shape == (36, 1)
    assert np.array_equal(coverage, np.ones((36, 1)))
    assert np.array_equal(fixed_coverage, np.ones(1))
    assert np.all(false_alerts >= 0.0)

    checks = benchmark._causality_checks(cached[0])
    assert checks == benchmark._CausalityChecks(
        metadata_boundary=True,
        future_prefix=True,
        current_preupdate=True,
        independent_histories=True,
    )


def test_utility_regret_is_nonnegative_and_observations_are_identity_outside_event() -> None:
    profiles = benchmark.load_health_population_profiles(_intent(), source_root=ROOT)
    base = generate_health_base_sequences(
        profiles.main_profile,
        split="validation",
        sequence_count=2,
        data_master_seed=1729,
    )[0]
    clean = benchmark.generate_health_observations(
        base,
        fault=benchmark._IDENTITY_FAULT,
    )
    fault = HealthFaultSpec(
        family="additive-position-bias",
        target="camera",
        axis="y",
        unit="m",
        value=2.0,
    )
    faulted = benchmark.generate_health_observations(base, fault=fault)
    assert benchmark._observation_identity_outside_event(
        clean,
        faulted,
        fault=fault,
    )
    features = benchmark.compute_health_feature_trace(faulted.health_frame_inputs())
    scored = benchmark.rescore_health_feature_trace(
        features,
        _dummy_calibration(),
    )
    regrets, dominance = benchmark._candidate_regrets_for_sequence(
        faulted,
        scored,
    )
    assert regrets.shape == (36,)
    assert np.all(regrets >= 0.0)
    assert dominance


def test_one_complete_value_level_utility_matrix_reuses_200_paired_sequences() -> None:
    intent = _intent()
    profiles = benchmark.load_health_population_profiles(intent, source_root=ROOT)
    base = generate_health_base_sequences(
        profiles.main_profile,
        split="validation",
        sequence_count=2,
        data_master_seed=1729,
    )[0]
    calibration = _dummy_calibration()
    clean = benchmark._cache_clean_validation((base,), calibration)[0]
    case = benchmark._selection_cases(
        intent,
        benchmark.expand_validation_cases(intent),
    )[0]

    conditions, identity_comparisons, identity_violations, oracle_violations = (
        benchmark._evaluate_utility_conditions(
            cases=(case,),
            bases=(base,) * 200,
            clean=(clean,) * 200,
            calibration=calibration,
        )
    )

    assert len(conditions) == 1
    assert conditions[0].condition_id == case.condition_id
    assert conditions[0].regret_m2_by_candidate_sequence.shape == (36, 200)
    assert identity_comparisons == 200
    assert identity_violations == 0
    assert oracle_violations == 0


def test_fit_validation_is_an_exact_passing_conjunction() -> None:
    intent = _intent()
    profiles = benchmark.load_health_population_profiles(intent, source_root=ROOT)
    validation_cases = benchmark.expand_validation_cases(intent)
    test_cases = benchmark.expand_test_cases(intent)
    selected_cases = benchmark._selection_cases(intent, validation_cases)
    selection, _ = _zero_selection(selected_cases)
    calibration = _dummy_calibration(count=9_200)
    evidence = benchmark._build_fit_validation(
        intent=intent,
        intent_sha256=M4_HEALTH_INTENT_SHA256,
        profiles=profiles,
        train_ids=tuple(f"train-{index}" for index in range(200)),
        validation_ids=tuple(f"validation-{index}" for index in range(200)),
        validation_cases=validation_cases,
        test_cases=test_cases,
        selection_cases=selected_cases,
        calibration=calibration,
        selection=selection,
        identity_comparisons=4_000,
        identity_violations=0,
        oracle_violations=0,
        causality=benchmark._CausalityChecks(True, True, True, True),
    )
    check_ids = tuple(check.check_id for check in evidence.checks)
    assert evidence.all_checks_passed
    assert check_ids == (
        "intent-digest",
        "main-profile-digest",
        "edge-profile-digest",
        "train-sequence-count",
        "validation-sequence-count",
        "train-validation-sequence-id-overlap",
        "validation-value-case-count",
        "test-value-case-count",
        "value-case-id-uniqueness",
        "selection-value-case-count",
        "identity-outside-active-event",
        "ecdf-channel-count",
        "ecdf-values-per-channel",
        "threshold-candidate-count",
        "threshold-candidate-order",
        "metadata-leakage-boundary",
        "future-prefix-causality",
        "current-preupdate-causality",
        "independent-modality-histories",
        "frame-oracle-comparison-count",
        "frame-oracle-dominance",
        "candidate-frame-evaluation-cap",
        "bootstrap-cell-cap",
        "scientific-feature-trace-count",
        "selected-candidate-feasible",
    )
    assert (
        next(
            check for check in evidence.checks if check.check_id == "ecdf-values-per-channel"
        ).observed
        == 9_200
    )
    assert (
        next(
            check for check in evidence.checks if check.check_id == "frame-oracle-comparison-count"
        ).observed
        == 144_000
    )


def test_exact_public_fit_orchestrates_without_test_population_or_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _synthetic_feature_trace()
    generated_splits: list[str] = []

    def fake_bases(
        _profile: object,
        *,
        split: str,
        sequence_count: int,
        data_master_seed: int,
    ) -> tuple[SimpleNamespace, ...]:
        generated_splits.append(split)
        assert sequence_count == 200
        assert data_master_seed == 1729
        return tuple(
            SimpleNamespace(sequence_id=f"{split}-{index:03d}") for index in range(sequence_count)
        )

    sentinel_cache = tuple(object() for _ in range(200))

    def fake_observations(
        _base: object,
        *,
        fault: HealthFaultSpec,
    ) -> SimpleNamespace:
        return SimpleNamespace(health_frame_inputs=lambda: (fault.family,))

    def fake_features(_frames: object) -> HealthFeatureTrace:
        return trace

    def fake_clean_cache(
        _bases: object,
        _calibration: HealthCalibration,
    ) -> tuple[object, ...]:
        return sentinel_cache

    def fake_clean_candidates(
        _cache: object,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.zeros((36, 200)),
            np.ones((36, 200)),
            np.ones(200),
            np.zeros((36, 200)),
        )

    monkeypatch.setattr(benchmark, "generate_health_base_sequences", fake_bases)
    monkeypatch.setattr(
        benchmark,
        "generate_health_observations",
        fake_observations,
    )
    monkeypatch.setattr(
        benchmark,
        "compute_health_feature_trace",
        fake_features,
    )
    monkeypatch.setattr(
        benchmark,
        "_cache_clean_validation",
        fake_clean_cache,
    )
    monkeypatch.setattr(
        benchmark,
        "_evaluate_clean_candidates",
        fake_clean_candidates,
    )

    def fake_utility(
        *,
        cases: tuple[benchmark.HealthCaseDescriptor, ...],
        bases: tuple[object, ...],
        clean: tuple[object, ...],
        calibration: HealthCalibration,
    ) -> tuple[tuple[ValidationConditionRegret, ...], int, int, int]:
        assert len(cases) == 20
        assert len(bases) == len(clean) == 200
        assert calibration.camera_self_mean.size == 9_200
        conditions = tuple(
            ValidationConditionRegret(
                condition_id=case.condition_id,
                target="camera" if case.fault.target == "camera" else "lidar",
                family=case.fault.family,
                regret_m2_by_candidate_sequence=np.zeros((36, 200)),
            )
            for case in cases
        )
        return conditions, 4_000, 0, 0

    monkeypatch.setattr(benchmark, "_evaluate_utility_conditions", fake_utility)

    def fake_causality(_clean: object) -> benchmark._CausalityChecks:
        return benchmark._CausalityChecks(True, True, True, True)

    monkeypatch.setattr(
        benchmark,
        "_causality_checks",
        fake_causality,
    )

    fit = benchmark.fit_health_benchmark(source_root=ROOT)

    assert generated_splits == ["train", "validation"]
    assert fit.intent_sha256 == M4_HEALTH_INTENT_SHA256
    assert len(fit.calibration.camera_self_mean) == 9_200
    assert len(fit.candidates) == CANDIDATE_COUNT
    assert tuple(
        (candidate.self_threshold, candidate.cross_threshold) for candidate in fit.candidates
    ) == tuple(
        (self_threshold, cross_threshold)
        for self_threshold in SELF_THRESHOLDS
        for cross_threshold in CROSS_THRESHOLDS
    )
    assert fit.selection.selected_candidate_index == 35
    assert fit.summary == HealthFitSummaryV1(
        schema="ffb.health-fit-summary/v1",
        intent_sha256=M4_HEALTH_INTENT_SHA256,
        main_profile_sha256=fit.profiles.main_profile_sha256,
        edge_profile_sha256=fit.profiles.edge_profile_sha256,
        train_sequence_count=200,
        validation_sequence_count=200,
        ecdf_channel_count=8,
        ecdf_values_per_channel=9_200,
        candidate_count=36,
        selected_candidate_index=35,
        selected_self_threshold=1.0,
        selected_cross_threshold=1.0,
        selection_status="selected",
    )
    assert fit.validation.all_checks_passed
    assert len(fit.condition_regrets) == 20


def test_fit_bundle_rejects_candidate_or_validation_disagreement() -> None:
    intent = _intent()
    profiles = benchmark.load_health_population_profiles(intent, source_root=ROOT)
    validation_cases = benchmark.expand_validation_cases(intent)
    test_cases = benchmark.expand_test_cases(intent)
    selected_cases = benchmark._selection_cases(intent, validation_cases)
    selection, regrets = _zero_selection(selected_cases)
    calibration = _dummy_calibration(count=9_200)
    validation = benchmark._build_fit_validation(
        intent=intent,
        intent_sha256=M4_HEALTH_INTENT_SHA256,
        profiles=profiles,
        train_ids=tuple(f"train-{index}" for index in range(200)),
        validation_ids=tuple(f"validation-{index}" for index in range(200)),
        validation_cases=validation_cases,
        test_cases=test_cases,
        selection_cases=selected_cases,
        calibration=calibration,
        selection=selection,
        identity_comparisons=4_000,
        identity_violations=0,
        oracle_violations=0,
        causality=benchmark._CausalityChecks(True, True, True, True),
    )
    summary = HealthFitSummaryV1(
        schema="ffb.health-fit-summary/v1",
        intent_sha256=M4_HEALTH_INTENT_SHA256,
        main_profile_sha256=profiles.main_profile_sha256,
        edge_profile_sha256=profiles.edge_profile_sha256,
        train_sequence_count=200,
        validation_sequence_count=200,
        ecdf_channel_count=8,
        ecdf_values_per_channel=9_200,
        candidate_count=36,
        selected_candidate_index=35,
        selected_self_threshold=1.0,
        selected_cross_threshold=1.0,
        selection_status="selected",
    )
    with pytest.raises(ValueError, match="candidate records"):
        benchmark.HealthBenchmarkFit(
            intent=intent,
            intent_sha256=M4_HEALTH_INTENT_SHA256,
            profiles=profiles,
            validation_cases=validation_cases,
            test_cases=test_cases,
            calibration=calibration,
            condition_regrets=regrets,
            candidates=tuple(reversed(selection.candidates)),
            selection=selection,
            summary=summary,
            validation=validation,
        )


def test_internal_window_and_false_alert_helpers_reject_empty_support() -> None:
    profiles = benchmark.load_health_population_profiles(_intent(), source_root=ROOT)
    base = generate_health_base_sequences(
        profiles.main_profile,
        split="validation",
        sequence_count=2,
        data_master_seed=1729,
    )[0]
    clean = benchmark.generate_health_observations(
        base,
        fault=benchmark._IDENTITY_FAULT,
    )
    fused, _ = benchmark.fixed_fusion_values(clean)
    fixed = benchmark._fixed_loss(clean, fused_xy_m=fused)
    with pytest.raises(ValueError, match="nonempty"):
        benchmark._window_statistic(clean, fixed, start=2, end=2)

    trace = benchmark.HealthPolicyTrace(
        policy="combined-health-gate",
        raw_labels=("healthy", "camera-fault", "camera-fault", "healthy", "lidar-fault"),
        evidence_statuses=("update-eligible",) * 5,
        latched_labels=("healthy", "camera-fault", "camera-fault", "healthy", "lidar-fault"),
        actions=("fixed-fusion",) * 5,
    )
    assert benchmark._false_alert_episode_starts(trace, start=0, end=5) == 2
