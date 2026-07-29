from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from fusion_fault_bench.health import (
    HealthCalibration,
    HealthFrameInput,
    ModalityMeasurement,
    ObjectHealthInput,
)
from fusion_fault_bench.health_fit import (
    CANDIDATE_COUNT,
    ECDF_VALUES_PER_CHANNEL,
    FRAMES_PER_SEQUENCE,
    TRAIN_SEQUENCE_COUNT,
    VALIDATION_BOOTSTRAP_REPLICATES,
    VALIDATION_BOOTSTRAP_SEED,
    VALIDATION_SEQUENCE_COUNT,
    HealthFeatureTrace,
    ScoredHealthTrace,
    UnscoredHealthFrame,
    UnscoredNumericChannel,
    ValidationConditionRegret,
    compute_health_feature_trace,
    fit_clean_health_calibration,
    rescore_health_feature_trace,
    select_health_thresholds,
)
from fusion_fault_bench.inference import (
    paired_bootstrap_indices,
    percentile_interval,
)


def _measurement(
    x: float,
    time_s: float,
    *,
    reported_time_s: float | None = None,
) -> ModalityMeasurement:
    return ModalityMeasurement(
        value_xy_m=np.asarray([x, 0.0], dtype=np.float64),
        reported_covariance_xy_m2=np.eye(2, dtype=np.float64),
        reported_time_s=time_s if reported_time_s is None else reported_time_s,
    )


def _input_frame(
    time_s: float,
    *,
    camera_x: float | None,
    lidar_x: float | None,
    camera_available: bool = True,
    lidar_available: bool = True,
    camera_reported_time_s: float | None = None,
) -> HealthFrameInput:
    return HealthFrameInput(
        reference_time_s=time_s,
        camera_available=camera_available,
        lidar_available=lidar_available,
        objects=tuple(
            ObjectHealthInput(
                object_id=f"object-{index}",
                camera=(
                    None
                    if camera_x is None
                    else _measurement(
                        camera_x,
                        time_s,
                        reported_time_s=camera_reported_time_s,
                    )
                ),
                lidar=None if lidar_x is None else _measurement(lidar_x, time_s),
            )
            for index in range(2)
        ),
    )


def _calibration(values: tuple[float, ...]) -> HealthCalibration:
    arrays = tuple(np.asarray(values, dtype=np.float64) for _ in range(8))
    return HealthCalibration(*arrays)


def _unscored(
    mean: float | None,
    *,
    maximum: float | None = None,
) -> UnscoredNumericChannel:
    if mean is None:
        return UnscoredNumericChannel(
            status="insufficient-support",
            mature_object_count=0,
            current_object_count=2,
            mature_fraction=0.0,
            mean_nis=None,
            maximum_nis=None,
        )
    return UnscoredNumericChannel(
        status="defined",
        mature_object_count=2,
        current_object_count=2,
        mature_fraction=1.0,
        mean_nis=mean,
        maximum_nis=mean + 0.5 if maximum is None else maximum,
    )


def _raw_frame(frame_index: int, sequence_index: int = 0) -> UnscoredHealthFrame:
    if frame_index < 2:
        camera = lidar = camera_cross = lidar_cross = _unscored(None)
    else:
        base = float(sequence_index * FRAMES_PER_SEQUENCE + frame_index)
        camera = _unscored(base)
        lidar = _unscored(base + 10_000.0)
        camera_cross = _unscored(base + 20_000.0)
        lidar_cross = _unscored(base + 30_000.0)
    return UnscoredHealthFrame(
        reference_time_s=frame_index * 0.1,
        camera_available=True,
        lidar_available=True,
        camera_timestamp_suspicious=False,
        lidar_timestamp_suspicious=False,
        camera_missing_fraction_last_four=0.0,
        lidar_missing_fraction_last_four=0.0,
        camera_self=camera,
        lidar_self=lidar,
        camera_from_lidar_cross=camera_cross,
        lidar_from_camera_cross=lidar_cross,
    )


def _train_trace(sequence_index: int) -> HealthFeatureTrace:
    return HealthFeatureTrace(
        frames=tuple(
            _raw_frame(frame_index, sequence_index) for frame_index in range(FRAMES_PER_SEQUENCE)
        )
    )


def _condition(
    condition_id: str,
    target: str,
    family: str,
    values_by_candidate: np.ndarray,
) -> ValidationConditionRegret:
    matrix = np.repeat(
        np.asarray(values_by_candidate, dtype=np.float64)[:, None],
        VALIDATION_SEQUENCE_COUNT,
        axis=1,
    )
    return ValidationConditionRegret(
        condition_id=condition_id,
        target=target,  # type: ignore[arg-type]
        family=family,
        regret_m2_by_candidate_sequence=matrix,
    )


def _selection_inputs() -> dict[str, object]:
    return {
        "clean_regression_m2_by_candidate_sequence": np.zeros(
            (CANDIDATE_COUNT, VALIDATION_SEQUENCE_COUNT),
            dtype=np.float64,
        ),
        "clean_coverage_by_candidate_sequence": np.ones(
            (CANDIDATE_COUNT, VALIDATION_SEQUENCE_COUNT),
            dtype=np.float64,
        ),
        "fixed_clean_coverage_by_sequence": np.ones(
            VALIDATION_SEQUENCE_COUNT,
            dtype=np.float64,
        ),
        "false_alert_starts_by_candidate_sequence": np.zeros(
            (CANDIDATE_COUNT, VALIDATION_SEQUENCE_COUNT),
            dtype=np.float64,
        ),
        "condition_regrets": (
            _condition(
                "camera-bias",
                "camera",
                "bias",
                np.ones(CANDIDATE_COUNT, dtype=np.float64),
            ),
            _condition(
                "lidar-bias",
                "lidar",
                "bias",
                np.ones(CANDIDATE_COUNT, dtype=np.float64),
            ),
        ),
    }


def test_feature_trace_computes_causal_raw_statistics_once_without_calibration() -> None:
    trace = compute_health_feature_trace(
        tuple(_input_frame(time, camera_x=time, lidar_x=time) for time in (0.0, 1.0, 2.0))
    )

    assert len(trace.frames) == 3
    assert trace.frames[0].camera_self.status == "insufficient-support"
    assert trace.frames[1].camera_self.status == "insufficient-support"
    for channel in (
        trace.frames[2].camera_self,
        trace.frames[2].lidar_self,
        trace.frames[2].camera_from_lidar_cross,
        trace.frames[2].lidar_from_camera_cross,
    ):
        assert channel.status == "defined"
        assert channel.mean_nis == pytest.approx(0.0)
        assert channel.maximum_nis == pytest.approx(0.0)
    with pytest.raises(FrozenInstanceError):
        trace.frames[2].camera_available = False  # type: ignore[misc]


def test_feature_trace_retains_only_observable_direct_telemetry() -> None:
    trace = compute_health_feature_trace(
        (
            _input_frame(0.0, camera_x=0.0, lidar_x=0.0),
            _input_frame(
                1.0,
                camera_x=None,
                lidar_x=1.0,
                camera_available=False,
            ),
            _input_frame(
                2.0,
                camera_x=2.0,
                lidar_x=2.0,
                camera_reported_time_s=2.1,
            ),
        )
    )
    assert not trace.frames[0].camera_timestamp_suspicious
    assert not trace.frames[1].camera_available
    assert trace.frames[1].camera_missing_fraction_last_four == pytest.approx(0.5)
    assert trace.frames[2].camera_timestamp_suspicious


def test_frozen_rescoring_changes_only_ecdf_score_not_raw_trace() -> None:
    trace = compute_health_feature_trace(
        (
            _input_frame(0.0, camera_x=0.0, lidar_x=0.0),
            _input_frame(1.0, camera_x=1.0, lidar_x=1.0),
            _input_frame(2.0, camera_x=10.0, lidar_x=2.0),
        )
    )
    raw_mean = trace.frames[2].camera_self.mean_nis
    low_fit = rescore_health_feature_trace(trace, _calibration((0.0,)))
    high_fit = rescore_health_feature_trace(trace, _calibration((0.0, 100.0)))

    assert raw_mean == pytest.approx(64.0 / 6.0)
    assert low_fit.frames[2].camera_self.score == 1.0
    assert high_fit.frames[2].camera_self.score == 0.5
    assert low_fit.frames[2].camera_self.mean_nis == raw_mean
    assert high_fit.frames[2].camera_self.mean_nis == raw_mean
    assert low_fit.frames[0].camera_self.status == "insufficient-support"
    assert low_fit.frames[0].camera_self.score is None


def test_rescoring_preserves_strict_less_than_ecdf_ties() -> None:
    frame = _raw_frame(2)
    tied_channel = _unscored(2.0, maximum=2.0)
    trace = HealthFeatureTrace(frames=(replace(frame, camera_self=tied_channel),))
    calibration = HealthCalibration(
        camera_self_mean=np.asarray([0.0, 1.0, 2.0, 2.0]),
        camera_self_maximum=np.asarray([0.0, 1.0, 2.0, 2.0]),
        lidar_self_mean=np.asarray([0.0]),
        lidar_self_maximum=np.asarray([0.0]),
        camera_from_lidar_cross_mean=np.asarray([0.0]),
        camera_from_lidar_cross_maximum=np.asarray([0.0]),
        lidar_from_camera_cross_mean=np.asarray([0.0]),
        lidar_from_camera_cross_maximum=np.asarray([0.0]),
    )
    scored = rescore_health_feature_trace(trace, calibration)
    assert scored.frames[0].camera_self.score == 0.5


def test_clean_fit_produces_exact_sorted_immutable_9200_value_arrays() -> None:
    traces = tuple(_train_trace(index) for index in range(TRAIN_SEQUENCE_COUNT))
    calibration = fit_clean_health_calibration(traces)

    for array in (
        calibration.camera_self_mean,
        calibration.camera_self_maximum,
        calibration.lidar_self_mean,
        calibration.lidar_self_maximum,
        calibration.camera_from_lidar_cross_mean,
        calibration.camera_from_lidar_cross_maximum,
        calibration.lidar_from_camera_cross_mean,
        calibration.lidar_from_camera_cross_maximum,
    ):
        assert array.shape == (ECDF_VALUES_PER_CHANNEL,)
        assert np.all(array[1:] >= array[:-1])
        assert not array.flags.writeable
    assert calibration.camera_self_mean[0] == 2.0
    assert calibration.camera_self_mean[-1] == 9_599.0
    assert calibration.camera_self_maximum[0] == 2.5
    assert calibration.lidar_from_camera_cross_mean[0] == 30_002.0


def test_clean_fit_rejects_wrong_population_frame_count_or_undefined_fit_frame() -> None:
    traces = tuple(_train_trace(index) for index in range(TRAIN_SEQUENCE_COUNT))
    with pytest.raises(ValueError, match="exactly 200"):
        fit_clean_health_calibration(traces[:-1])

    shortened = (HealthFeatureTrace(frames=traces[0].frames[:-1]), *traces[1:])
    with pytest.raises(ValueError, match="exactly 48"):
        fit_clean_health_calibration(shortened)

    bad_frame = replace(traces[0].frames[2], camera_self=_unscored(None))
    bad_trace = HealthFeatureTrace(
        frames=(traces[0].frames[0], traces[0].frames[1], bad_frame, *traces[0].frames[3:])
    )
    with pytest.raises(ValueError, match="camera_self is not defined"):
        fit_clean_health_calibration((bad_trace, *traces[1:]))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {
                "status": "bogus",
                "mature_object_count": 0,
                "current_object_count": 2,
                "mature_fraction": 0.0,
                "mean_nis": None,
                "maximum_nis": None,
            },
            "unknown",
        ),
        (
            {
                "status": "insufficient-support",
                "mature_object_count": 0,
                "current_object_count": 0,
                "mature_fraction": 0.0,
                "mean_nis": None,
                "maximum_nis": None,
            },
            "positive",
        ),
        (
            {
                "status": "insufficient-support",
                "mature_object_count": 3,
                "current_object_count": 2,
                "mature_fraction": 1.5,
                "mean_nis": None,
                "maximum_nis": None,
            },
            "within",
        ),
        (
            {
                "status": "insufficient-support",
                "mature_object_count": 1,
                "current_object_count": 2,
                "mature_fraction": 0.0,
                "mean_nis": None,
                "maximum_nis": None,
            },
            "fraction",
        ),
        (
            {
                "status": "defined",
                "mature_object_count": 1,
                "current_object_count": 2,
                "mature_fraction": 0.5,
                "mean_nis": 0.0,
                "maximum_nis": 0.0,
            },
            "at least two",
        ),
        (
            {
                "status": "defined",
                "mature_object_count": 2,
                "current_object_count": 2,
                "mature_fraction": 1.0,
                "mean_nis": None,
                "maximum_nis": 0.0,
            },
            "mean and maximum",
        ),
        (
            {
                "status": "defined",
                "mature_object_count": 2,
                "current_object_count": 2,
                "mature_fraction": 1.0,
                "mean_nis": -1.0,
                "maximum_nis": 0.0,
            },
            "nonnegative",
        ),
        (
            {
                "status": "defined",
                "mature_object_count": 2,
                "current_object_count": 2,
                "mature_fraction": 1.0,
                "mean_nis": 2.0,
                "maximum_nis": 1.0,
            },
            "below mean",
        ),
        (
            {
                "status": "insufficient-support",
                "mature_object_count": 2,
                "current_object_count": 2,
                "mature_fraction": 1.0,
                "mean_nis": None,
                "maximum_nis": None,
            },
            "fewer than two",
        ),
        (
            {
                "status": "insufficient-support",
                "mature_object_count": 1,
                "current_object_count": 2,
                "mature_fraction": 0.5,
                "mean_nis": 0.0,
                "maximum_nis": None,
            },
            "cannot carry",
        ),
    ],
)
def test_unscored_channel_rejects_inconsistent_raw_statistics(
    arguments: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        UnscoredNumericChannel(**arguments)  # type: ignore[arg-type]


def test_unscored_frame_rejects_nonfinite_time_and_invalid_missing_fraction() -> None:
    frame = _raw_frame(0)
    with pytest.raises(ValueError, match="finite"):
        replace(frame, reference_time_s=np.nan)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        replace(frame, lidar_missing_fraction_last_four=-0.1)


def test_selection_retains_exact_grid_order_and_minimizes_equal_weighted_regret() -> None:
    arguments = _selection_inputs()
    camera_values = np.ones(CANDIDATE_COUNT, dtype=np.float64)
    lidar_values = np.ones(CANDIDATE_COUNT, dtype=np.float64)
    camera_values[5] = 0.1
    lidar_values[5] = 0.1
    arguments["condition_regrets"] = (
        _condition("camera-bias", "camera", "bias", camera_values),
        _condition("lidar-bias", "lidar", "bias", lidar_values),
    )
    selection = select_health_thresholds(**arguments)  # type: ignore[arg-type]

    assert len(selection.candidates) == 36
    assert selection.selected_candidate_index == 5
    assert selection.selected_self_threshold == 0.95
    assert selection.selected_cross_threshold == 1.0
    assert [
        (
            candidate.self_threshold,
            candidate.cross_threshold,
        )
        for candidate in selection.candidates[:7]
    ] == [
        (0.95, 0.95),
        (0.95, 0.975),
        (0.95, 0.99),
        (0.95, 0.995),
        (0.95, 0.999),
        (0.95, 1.0),
        (0.975, 0.95),
    ]


def test_selection_objective_weights_target_then_family_then_condition() -> None:
    arguments = _selection_inputs()
    constants = (
        ("camera-bias-1", "camera", "bias", 0.0),
        ("camera-bias-2", "camera", "bias", 2.0),
        ("camera-noise", "camera", "noise", 10.0),
        ("lidar-bias", "lidar", "bias", 20.0),
    )
    arguments["condition_regrets"] = tuple(
        _condition(
            condition_id,
            target,
            family,
            np.full(CANDIDATE_COUNT, value, dtype=np.float64),
        )
        for condition_id, target, family, value in constants
    )
    selection = select_health_thresholds(**arguments)  # type: ignore[arg-type]

    assert selection.candidates[0].validation_regret_m2 == 12.75
    # Every earlier tie-break is equal, so larger self then larger cross wins.
    assert selection.selected_candidate_index == 35


def test_selection_recomputes_exact_paired_bootstrap_upper_bound() -> None:
    arguments = _selection_inputs()
    regression = arguments["clean_regression_m2_by_candidate_sequence"]
    assert isinstance(regression, np.ndarray)
    regression[1, :100] = -0.05
    regression[1, 100:] = 0.05
    selection = select_health_thresholds(**arguments)  # type: ignore[arg-type]

    indices = paired_bootstrap_indices(
        seed=VALIDATION_BOOTSTRAP_SEED,
        replicates=VALIDATION_BOOTSTRAP_REPLICATES,
        sequence_count=VALIDATION_SEQUENCE_COUNT,
    )
    bootstrap_means = regression[1][indices].mean(axis=1)
    _, expected_upper = percentile_interval(bootstrap_means, confidence_level=0.95)
    assert selection.candidates[1].upper_95pct_clean_regression_m2 == expected_upper
    assert expected_upper > 0.005
    assert not selection.candidates[1].feasible


def test_all_four_clean_feasibility_gates_are_applied_without_relaxation() -> None:
    arguments = _selection_inputs()
    regression = arguments["clean_regression_m2_by_candidate_sequence"]
    coverage = arguments["clean_coverage_by_candidate_sequence"]
    false_alerts = arguments["false_alert_starts_by_candidate_sequence"]
    assert isinstance(regression, np.ndarray)
    assert isinstance(coverage, np.ndarray)
    assert isinstance(false_alerts, np.ndarray)
    regression[0] = 0.003
    regression[1, :100] = -0.05
    regression[1, 100:] = 0.05
    false_alerts[2] = 0.051
    coverage[3] = 0.99

    selection = select_health_thresholds(**arguments)  # type: ignore[arg-type]

    assert [candidate.feasible for candidate in selection.candidates[:5]] == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_selection_tolerance_then_false_alert_and_threshold_tie_breaks() -> None:
    arguments = _selection_inputs()
    first = np.ones(CANDIDATE_COUNT, dtype=np.float64)
    second = np.ones(CANDIDATE_COUNT, dtype=np.float64)
    first[0] = second[0] = 0.1
    first[35] = second[35] = 0.1 + 0.5e-12
    arguments["condition_regrets"] = (
        _condition("camera", "camera", "bias", first),
        _condition("lidar", "lidar", "bias", second),
    )
    selected_by_threshold = select_health_thresholds(
        **arguments  # type: ignore[arg-type]
    )
    assert selected_by_threshold.selected_candidate_index == 35

    false_alerts = arguments["false_alert_starts_by_candidate_sequence"]
    assert isinstance(false_alerts, np.ndarray)
    false_alerts[35] = 0.01
    selected_by_false_alert = select_health_thresholds(
        **arguments  # type: ignore[arg-type]
    )
    assert selected_by_false_alert.selected_candidate_index == 0


def test_selection_hard_fails_when_no_candidate_is_feasible() -> None:
    arguments = _selection_inputs()
    arguments["clean_regression_m2_by_candidate_sequence"] = np.full(
        (CANDIDATE_COUNT, VALIDATION_SEQUENCE_COUNT),
        0.01,
        dtype=np.float64,
    )
    with pytest.raises(RuntimeError, match="no threshold candidate"):
        select_health_thresholds(**arguments)  # type: ignore[arg-type]


def test_selection_rejects_invalid_shapes_support_and_condition_metadata() -> None:
    arguments = _selection_inputs()
    with pytest.raises(ValueError, match="shape"):
        select_health_thresholds(
            **(
                arguments
                | {
                    "clean_coverage_by_candidate_sequence": np.ones(
                        (35, VALIDATION_SEQUENCE_COUNT),
                        dtype=np.float64,
                    )
                }
            )  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="both camera and lidar"):
        select_health_thresholds(
            **(
                arguments
                | {
                    "condition_regrets": (
                        _condition(
                            "camera-only",
                            "camera",
                            "bias",
                            np.ones(CANDIDATE_COUNT),
                        ),
                    )
                }
            )  # type: ignore[arg-type]
        )
    duplicated = _condition(
        "duplicate",
        "camera",
        "bias",
        np.ones(CANDIDATE_COUNT),
    )
    with pytest.raises(ValueError, match="unique"):
        select_health_thresholds(
            **(
                arguments
                | {
                    "condition_regrets": (
                        duplicated,
                        duplicated,
                        _condition(
                            "lidar",
                            "lidar",
                            "bias",
                            np.ones(CANDIDATE_COUNT),
                        ),
                    )
                }
            )  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        (
            "clean_regression_m2_by_candidate_sequence",
            np.full((CANDIDATE_COUNT, VALIDATION_SEQUENCE_COUNT), np.nan),
            "finite",
        ),
        (
            "fixed_clean_coverage_by_sequence",
            np.ones(VALIDATION_SEQUENCE_COUNT - 1),
            "shape",
        ),
        (
            "fixed_clean_coverage_by_sequence",
            np.full(VALIDATION_SEQUENCE_COUNT, np.nan),
            "finite",
        ),
        (
            "clean_coverage_by_candidate_sequence",
            np.full((CANDIDATE_COUNT, VALIDATION_SEQUENCE_COUNT), 1.1),
            r"\[0, 1\]",
        ),
        (
            "fixed_clean_coverage_by_sequence",
            np.full(VALIDATION_SEQUENCE_COUNT, -0.1),
            r"\[0, 1\]",
        ),
        (
            "false_alert_starts_by_candidate_sequence",
            np.full((CANDIDATE_COUNT, VALIDATION_SEQUENCE_COUNT), -1.0),
            "cannot be negative",
        ),
        (
            "condition_regrets",
            (),
            "at least one",
        ),
    ],
)
def test_selection_rejects_nonfinite_or_out_of_range_supplied_statistics(
    field_name: str,
    bad_value: object,
    message: str,
) -> None:
    arguments = _selection_inputs()
    arguments[field_name] = bad_value
    with pytest.raises(ValueError, match=message):
        select_health_thresholds(**arguments)  # type: ignore[arg-type]


def test_validation_condition_regret_is_immutable_and_nonnegative() -> None:
    source = np.ones(
        (CANDIDATE_COUNT, VALIDATION_SEQUENCE_COUNT),
        dtype=np.float64,
    )
    condition = ValidationConditionRegret(
        condition_id="condition",
        target="camera",
        family="bias",
        regret_m2_by_candidate_sequence=source,
    )
    source[0, 0] = 99.0
    assert condition.regret_m2_by_candidate_sequence[0, 0] == 1.0
    assert not condition.regret_m2_by_candidate_sequence.flags.writeable

    negative = np.ones_like(source)
    negative[0, 0] = -1.0
    with pytest.raises(ValueError, match="cannot be negative"):
        ValidationConditionRegret(
            condition_id="negative",
            target="camera",
            family="bias",
            regret_m2_by_candidate_sequence=negative,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"condition_id": ""}, "nonempty"),
        ({"target": "both"}, "camera or lidar"),
        ({"family": ""}, "nonempty"),
        (
            {"regret_m2_by_candidate_sequence": np.ones((35, 200))},
            "shape",
        ),
        (
            {
                "regret_m2_by_candidate_sequence": np.full(
                    (CANDIDATE_COUNT, VALIDATION_SEQUENCE_COUNT),
                    np.nan,
                )
            },
            "finite",
        ),
    ],
)
def test_validation_condition_regret_rejects_invalid_metadata_and_matrix(
    updates: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "condition_id": "condition",
        "target": "camera",
        "family": "bias",
        "regret_m2_by_candidate_sequence": np.ones((CANDIDATE_COUNT, VALIDATION_SEQUENCE_COUNT)),
    }
    arguments.update(updates)
    with pytest.raises(ValueError, match=message):
        ValidationConditionRegret(**arguments)  # type: ignore[arg-type]


def test_threshold_selection_output_rejects_inconsistent_selected_candidate() -> None:
    selection = select_health_thresholds(
        **_selection_inputs()  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="all 36"):
        replace(selection, candidates=selection.candidates[:-1])
    swapped = (
        selection.candidates[1],
        selection.candidates[0],
        *selection.candidates[2:],
    )
    with pytest.raises(ValueError, match="exact grid order"):
        replace(selection, candidates=swapped)
    with pytest.raises(ValueError, match="outside"):
        replace(selection, selected_candidate_index=36)
    with pytest.raises(ValueError, match="disagree"):
        replace(selection, selected_self_threshold=0.95)

    arguments = _selection_inputs()
    regression = arguments["clean_regression_m2_by_candidate_sequence"]
    assert isinstance(regression, np.ndarray)
    regression[0] = 0.003
    with_infeasible = select_health_thresholds(
        **arguments  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="must be feasible"):
        replace(
            with_infeasible,
            selected_candidate_index=0,
            selected_self_threshold=0.95,
            selected_cross_threshold=0.95,
        )


def test_trace_contracts_reject_empty_or_noncausal_frame_sequences() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        HealthFeatureTrace(frames=())
    frame = _raw_frame(0)
    with pytest.raises(ValueError, match="strictly increasing"):
        HealthFeatureTrace(frames=(frame, frame))
    with pytest.raises(ValueError, match="at least one frame"):
        ScoredHealthTrace(frames=())
    with pytest.raises(ValueError, match="at least one"):
        compute_health_feature_trace(())
