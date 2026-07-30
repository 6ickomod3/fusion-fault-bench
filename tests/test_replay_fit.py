from __future__ import annotations

import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fusion_fault_bench.health import HealthThresholds
from fusion_fault_bench.replay_fit import (
    M4_FROZEN_CALIBRATION_SHA256,
    M4_SELECTED_CANDIDATE_INDEX,
    M4_SELECTED_CROSS_THRESHOLD,
    M4_SELECTED_SELF_THRESHOLD,
    M4_SOURCE_FIT_ARTIFACT_SHA256,
    M4_SOURCE_FIT_RUN_SHA256,
    M4_SOURCE_INTENT_SHA256,
    ReplayFitError,
    load_frozen_replay_health_fit,
    replay_health_calibration_sha256,
    validate_frozen_replay_health_fit,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_replay_health_fit_loads_only_released_m4_state() -> None:
    fit = load_frozen_replay_health_fit(ROOT)

    assert fit.selected_candidate_index == M4_SELECTED_CANDIDATE_INDEX
    assert fit.thresholds.self_score == M4_SELECTED_SELF_THRESHOLD
    assert fit.thresholds.cross_score == M4_SELECTED_CROSS_THRESHOLD
    assert fit.source_intent_sha256 == M4_SOURCE_INTENT_SHA256
    assert fit.source_fit_artifact_sha256 == M4_SOURCE_FIT_ARTIFACT_SHA256
    assert fit.source_fit_run_sha256 == M4_SOURCE_FIT_RUN_SHA256
    assert fit.calibration_sha256 == M4_FROZEN_CALIBRATION_SHA256
    assert replay_health_calibration_sha256(fit.calibration) == fit.calibration_sha256
    for channel in (
        fit.calibration.camera_self_mean,
        fit.calibration.camera_self_maximum,
        fit.calibration.lidar_self_mean,
        fit.calibration.lidar_self_maximum,
        fit.calibration.camera_from_lidar_cross_mean,
        fit.calibration.camera_from_lidar_cross_maximum,
        fit.calibration.lidar_from_camera_cross_mean,
        fit.calibration.lidar_from_camera_cross_maximum,
    ):
        assert channel.shape == (9200,)
        assert np.all(channel[1:] >= channel[:-1])
        assert not channel.flags.writeable


def test_frozen_replay_health_fit_failure_is_path_sanitized(tmp_path: Path) -> None:
    secret_named_root = tmp_path / "private-dataset-location"
    secret_named_root.mkdir()

    with pytest.raises(
        ReplayFitError,
        match=r"^frozen M4 health fit validation failed$",
    ) as captured:
        load_frozen_replay_health_fit(secret_named_root)

    assert "private-dataset-location" not in str(captured.value)
    rendered = "".join(
        traceback.format_exception(
            captured.type,
            captured.value,
            captured.tb,
        )
    )
    assert "private-dataset-location" not in rendered
    assert captured.value.__cause__ is None


def test_frozen_replay_health_fit_rejects_forged_thresholds_and_calibration() -> None:
    fit = load_frozen_replay_health_fit(ROOT)
    with pytest.raises(ReplayFitError, match="does not match"):
        validate_frozen_replay_health_fit(
            replace(
                fit,
                thresholds=HealthThresholds(self_score=0.999, cross_score=0.999),
            )
        )

    modified = fit.calibration.camera_self_mean.copy()
    modified[-1] += 1.0
    forged_calibration = replace(fit.calibration, camera_self_mean=modified)
    with pytest.raises(ReplayFitError, match="does not match"):
        validate_frozen_replay_health_fit(
            replace(
                fit,
                calibration=forged_calibration,
            )
        )
