"""Authenticated apply-only loading of the released M4 health calibration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from fusion_fault_bench.health import HealthCalibration, HealthThresholds
from fusion_fault_bench.health_artifacts import (
    HEALTH_ECDF_CHANNEL_ORDER,
    HealthEcdfArraysV1,
)
from fusion_fault_bench.health_release import (
    HealthReleaseValidationError,
    load_health_release,
)

M4_RELEASE_RELATIVE_PATH = Path("reports/releases/m4-health-v0.1.0")
M4_SOURCE_INTENT_SHA256 = "c19573b72a6ef58ad5efdd7039110da79beacd3b398007a508eb7ecfc4c81357"
M4_SOURCE_FIT_ARTIFACT_SHA256 = "abd1540f292fe51a7a23a47b679fe8e1522d8c5e20a03125a880eb9242a608ee"
M4_SOURCE_FIT_RUN_SHA256 = "0311aec90df031bd0d1720d5fa15aae91e2e5c3dfea923dacc2eefe518134fcd"
M4_SOURCE_SCIENTIFIC_REVISION = "a829a9f3af541c1b92b89d051b7c8b7003dc5a15"
M4_SELECTED_CANDIDATE_INDEX = 27
M4_SELECTED_SELF_THRESHOLD = 0.999
M4_SELECTED_CROSS_THRESHOLD = 0.995
M4_FROZEN_CALIBRATION_SHA256 = "cb07c30d646983f7231c66020ae930ac48f596467d0539c056eac3d546801ee2"

_ECDF_RELEASE_MEMBER = "primary-fit-ecdf-arrays.json"
_CALIBRATION_DOMAIN = b"fusion-fault-bench/m5-frozen-health-calibration/v1\x00"
_ECDF_VALUES_PER_CHANNEL = 9_200


class ReplayFitError(ValueError):
    """The frozen M4 fit could not be authenticated for replay use."""


@dataclass(frozen=True, slots=True)
class FrozenReplayHealthFit:
    """Only the released M4 state permitted to enter the M5 apply-only panel."""

    calibration: HealthCalibration
    thresholds: HealthThresholds
    selected_candidate_index: int
    source_intent_sha256: str
    source_fit_artifact_sha256: str
    source_fit_run_sha256: str
    source_scientific_revision: str
    calibration_sha256: str


def replay_health_calibration_sha256(calibration: HealthCalibration) -> str:
    """Commit the exact ordered float64 arrays used by the apply-only scorer."""

    digest = hashlib.sha256(_CALIBRATION_DOMAIN)
    for channel in HEALTH_ECDF_CHANNEL_ORDER:
        encoded = channel.encode("utf-8")
        values = np.asarray(getattr(calibration, channel), dtype="<f8")
        if (
            values.shape != (_ECDF_VALUES_PER_CHANNEL,)
            or not bool(np.all(np.isfinite(values)))
            or bool(np.any(values[1:] < values[:-1]))
        ):
            raise ReplayFitError("frozen M4 health calibration is invalid")
        contiguous = np.ascontiguousarray(values)
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(contiguous.size.to_bytes(8, "big"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def validate_frozen_replay_health_fit(fit: FrozenReplayHealthFit) -> None:
    """Reject any fit state that differs from the released M4 evidence."""

    if (
        fit.selected_candidate_index != M4_SELECTED_CANDIDATE_INDEX
        or fit.source_intent_sha256 != M4_SOURCE_INTENT_SHA256
        or fit.source_fit_artifact_sha256 != M4_SOURCE_FIT_ARTIFACT_SHA256
        or fit.source_fit_run_sha256 != M4_SOURCE_FIT_RUN_SHA256
        or fit.source_scientific_revision != M4_SOURCE_SCIENTIFIC_REVISION
        or fit.thresholds.self_score != M4_SELECTED_SELF_THRESHOLD
        or fit.thresholds.cross_score != M4_SELECTED_CROSS_THRESHOLD
        or fit.calibration_sha256 != M4_FROZEN_CALIBRATION_SHA256
        or replay_health_calibration_sha256(fit.calibration) != M4_FROZEN_CALIBRATION_SHA256
    ):
        raise ReplayFitError("replay health fit does not match the frozen M4 release")


def _load_frozen_replay_health_fit(source_root: Path) -> FrozenReplayHealthFit:
    release = load_health_release(source_root / M4_RELEASE_RELATIVE_PATH)
    summary = release.summary
    repeat = release.repeat
    expected_summary = (
        summary.intent_sha256 == M4_SOURCE_INTENT_SHA256
        and summary.official_fit_artifact_sha256 == M4_SOURCE_FIT_ARTIFACT_SHA256
        and summary.official_fit_run_sha256 == M4_SOURCE_FIT_RUN_SHA256
        and summary.git_revision == M4_SOURCE_SCIENTIFIC_REVISION
        and summary.selected_candidate_index == M4_SELECTED_CANDIDATE_INDEX
        and summary.selected_self_threshold == M4_SELECTED_SELF_THRESHOLD
        and summary.selected_cross_threshold == M4_SELECTED_CROSS_THRESHOLD
        and summary.all_checks_passed
    )
    expected_repeat = (
        repeat.intent_sha256 == M4_SOURCE_INTENT_SHA256
        and repeat.official_fit_artifact_sha256 == M4_SOURCE_FIT_ARTIFACT_SHA256
        and repeat.official_fit_run_sha256 == M4_SOURCE_FIT_RUN_SHA256
        and repeat.repeat_fit_artifact_sha256 == M4_SOURCE_FIT_ARTIFACT_SHA256
        and repeat.all_checks_passed
    )
    if not expected_summary or not expected_repeat:
        raise ReplayFitError("frozen M4 health fit identity is invalid")

    ecdf_path = release.path / _ECDF_RELEASE_MEMBER
    ecdf = HealthEcdfArraysV1.model_validate_json(ecdf_path.read_bytes())
    if tuple(channel.channel for channel in ecdf.channels) != HEALTH_ECDF_CHANNEL_ORDER:
        raise ReplayFitError("frozen M4 health calibration channel order is invalid")
    values = {channel.channel: channel.values for channel in ecdf.channels}
    calibration = HealthCalibration(
        camera_self_mean=np.asarray(values["camera_self_mean"], dtype=np.float64),
        camera_self_maximum=np.asarray(values["camera_self_maximum"], dtype=np.float64),
        lidar_self_mean=np.asarray(values["lidar_self_mean"], dtype=np.float64),
        lidar_self_maximum=np.asarray(values["lidar_self_maximum"], dtype=np.float64),
        camera_from_lidar_cross_mean=np.asarray(
            values["camera_from_lidar_cross_mean"], dtype=np.float64
        ),
        camera_from_lidar_cross_maximum=np.asarray(
            values["camera_from_lidar_cross_maximum"], dtype=np.float64
        ),
        lidar_from_camera_cross_mean=np.asarray(
            values["lidar_from_camera_cross_mean"], dtype=np.float64
        ),
        lidar_from_camera_cross_maximum=np.asarray(
            values["lidar_from_camera_cross_maximum"], dtype=np.float64
        ),
    )
    fit = FrozenReplayHealthFit(
        calibration=calibration,
        thresholds=HealthThresholds(
            self_score=M4_SELECTED_SELF_THRESHOLD,
            cross_score=M4_SELECTED_CROSS_THRESHOLD,
        ),
        selected_candidate_index=M4_SELECTED_CANDIDATE_INDEX,
        source_intent_sha256=M4_SOURCE_INTENT_SHA256,
        source_fit_artifact_sha256=M4_SOURCE_FIT_ARTIFACT_SHA256,
        source_fit_run_sha256=M4_SOURCE_FIT_RUN_SHA256,
        source_scientific_revision=M4_SOURCE_SCIENTIFIC_REVISION,
        calibration_sha256=replay_health_calibration_sha256(calibration),
    )
    validate_frozen_replay_health_fit(fit)
    return fit


def load_frozen_replay_health_fit(source_root: Path) -> FrozenReplayHealthFit:
    """Strictly authenticate and load the tracked M4 fit without refitting."""

    try:
        return _load_frozen_replay_health_fit(source_root)
    except ReplayFitError:
        raise
    except (
        HealthReleaseValidationError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        ValidationError,
    ):
        raise ReplayFitError("frozen M4 health fit validation failed") from None
