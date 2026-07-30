"""Authenticated expansion of the frozen M5 persistent and health matrices."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.bundle_v1alpha1 import ConditionKey, expected_conditions
from fusion_fault_bench.contracts.io import load_json_object
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AvailabilityControlManifest,
    CommonModeControlManifest,
    GeometryCrossoverManifest,
)
from fusion_fault_bench.contracts.matrix_v1 import (
    M3_PROCEDURAL_MATRIX_SHA256,
    LoadedExperimentMatrix,
    load_experiment_matrix,
)
from fusion_fault_bench.contracts.replay_v1 import (
    M5_HEALTH_EXPERIMENT_IDS,
    M5_HEALTH_PANEL_ID,
    M5_PERSISTENT_EXPERIMENT_IDS,
    M5_PERSISTENT_PANEL_ID,
    LoadedReplayIntent,
    ReplayExperimentIdentityV1,
    expected_replay_identities,
    load_replay_intent,
    replay_experiment_identity_sha256,
)
from fusion_fault_bench.replay_experiments import (
    FaultFamily,
    FaultTarget,
    ReplayFaultCondition,
)
from fusion_fault_bench.replay_health import replay_health_schedule

type PersistentManifest = (
    GeometryCrossoverManifest | AvailabilityControlManifest | CommonModeControlManifest
)

M5_PERSISTENT_MATRIX_PATH = Path("examples/matrices/m3-procedural-v1.json")

_M5_HEALTH_COORDINATES: dict[
    str,
    tuple[FaultFamily, FaultTarget, str, str, tuple[float, ...]],
] = {
    "replay-camera-output-y-bias": (
        "additive-position-bias",
        "camera",
        "y",
        "m",
        (-3.0, -0.75, 0.75, 3.0),
    ),
    "replay-lidar-output-y-bias": (
        "additive-position-bias",
        "lidar",
        "y",
        "m",
        (-3.0, -0.75, 0.75, 3.0),
    ),
    "replay-camera-noise-underreported": (
        "increased-noise-underreported",
        "camera",
        "xy",
        "std-scale",
        (1.25, 3.0),
    ),
    "replay-lidar-noise-underreported": (
        "increased-noise-underreported",
        "lidar",
        "xy",
        "std-scale",
        (1.25, 3.0),
    ),
    "replay-camera-noise-correctly-reported": (
        "increased-noise-correctly-reported",
        "camera",
        "xy",
        "std-scale",
        (1.25, 3.0),
    ),
    "replay-lidar-noise-correctly-reported": (
        "increased-noise-correctly-reported",
        "lidar",
        "xy",
        "std-scale",
        (1.25, 3.0),
    ),
    "replay-camera-timestamp-offset": (
        "timestamp-offset",
        "camera",
        "time",
        "s",
        (-0.6, -0.15, 0.15, 0.6),
    ),
    "replay-lidar-timestamp-offset": (
        "timestamp-offset",
        "lidar",
        "time",
        "s",
        (-0.6, -0.15, 0.15, 0.6),
    ),
    "replay-camera-dropout": (
        "dropout",
        "camera",
        "availability",
        "probability",
        (0.1, 0.5, 1.0),
    ),
    "replay-lidar-dropout": (
        "dropout",
        "lidar",
        "availability",
        "probability",
        (0.1, 0.5, 1.0),
    ),
    "replay-camera-calibration-x": (
        "calibration-translation",
        "camera",
        "x",
        "m",
        (-3.0, -0.75, 0.75, 3.0),
    ),
    "replay-camera-calibration-yaw": (
        "calibration-yaw",
        "camera",
        "yaw",
        "rad",
        (-0.06, -0.015, 0.015, 0.06),
    ),
    "replay-common-mode-x": (
        "common-mode-position-bias",
        "both",
        "x",
        "m",
        (-4.0, -1.0, 1.0, 4.0),
    ),
    "replay-clean": ("identity", "none", "none", "identity", (0.0,)),
}


@dataclass(frozen=True, slots=True)
class ReplayPersistentCase:
    """One exact M5-A manifest/severity coordinate."""

    identity: ReplayExperimentIdentityV1
    identity_sha256: str
    source_manifest: PersistentManifest
    source_condition: ConditionKey
    fault_condition: ReplayFaultCondition

    def __post_init__(self) -> None:
        if self.identity.panel_id != M5_PERSISTENT_PANEL_ID:
            raise ValueError("persistent case uses the wrong replay panel")
        if self.identity_sha256 != replay_experiment_identity_sha256(self.identity):
            raise ValueError("persistent case identity digest is inconsistent")
        if sha256_digest(self.source_manifest) != self.identity.source_sha256:
            raise ValueError("persistent case does not bind its source manifest")
        if self.fault_condition.experiment_id != self.identity.experiment_id:
            raise ValueError("persistent fault coordinate uses the wrong experiment")


@dataclass(frozen=True, slots=True)
class ReplayHealthCaseSpec:
    """One exact M5-B value before its scene-length schedule is applied."""

    identity: ReplayExperimentIdentityV1
    identity_sha256: str
    family: FaultFamily
    target: FaultTarget
    axis: str
    unit: str
    value_index: int
    value: float

    def __post_init__(self) -> None:
        if self.identity.panel_id != M5_HEALTH_PANEL_ID:
            raise ValueError("health case uses the wrong replay panel")
        if self.identity_sha256 != replay_experiment_identity_sha256(self.identity):
            raise ValueError("health case identity digest is inconsistent")
        if type(self.value_index) is not int or self.value_index < 0:
            raise ValueError("health value_index must be a non-negative integer")
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError("health severity value must be finite")
        expected = _M5_HEALTH_COORDINATES.get(self.identity.experiment_id)
        if expected is None:
            raise ValueError("health case is not in the frozen M5-B matrix")
        family, target, axis, unit, values = expected
        if (
            (self.family, self.target, self.axis, self.unit) != (family, target, axis, unit)
            or self.value_index >= len(values)
            or value != values[self.value_index]
        ):
            raise ValueError("health case coordinate differs from the frozen M5-B matrix")
        object.__setattr__(self, "value", value)

    @property
    def selector(self) -> str:
        """Return the exact frozen M5-B public selector."""

        return self.for_frame_count(16).selector

    def for_frame_count(self, frame_count: int) -> ReplayFaultCondition:
        """Bind this value to the schedule derived only from scene length."""

        schedule = replay_health_schedule(frame_count)
        return ReplayFaultCondition(
            experiment_id=self.identity.experiment_id,
            family=self.family,
            target=self.target,
            axis=self.axis,
            unit=self.unit,
            value=self.value,
            identity=self.family == "identity",
            active_frames=schedule.fault_active_frames,
        )


@dataclass(frozen=True, slots=True)
class LoadedReplayPlan:
    """Every authenticated coordinate needed before local data execution."""

    intent: LoadedReplayIntent
    persistent_matrix: LoadedExperimentMatrix
    persistent_cases: tuple[ReplayPersistentCase, ...]
    health_cases: tuple[ReplayHealthCaseSpec, ...]

    def __post_init__(self) -> None:
        if self.persistent_matrix.matrix_sha256 != M3_PROCEDURAL_MATRIX_SHA256:
            raise ValueError("replay plan uses the wrong persistent matrix")
        if (
            tuple(dict.fromkeys(case.identity.experiment_id for case in self.persistent_cases))
            != M5_PERSISTENT_EXPERIMENT_IDS
        ):
            raise ValueError("replay plan has an incomplete persistent experiment order")
        if (
            tuple(dict.fromkeys(case.identity.experiment_id for case in self.health_cases))
            != M5_HEALTH_EXPERIMENT_IDS
        ):
            raise ValueError("replay plan has an incomplete health condition order")


def _persistent_manifest(value: object) -> PersistentManifest:
    if isinstance(
        value,
        (GeometryCrossoverManifest, AvailabilityControlManifest, CommonModeControlManifest),
    ):
        return value
    raise ValueError("M5-A source matrix contains an unsupported manifest")


def _signed_value(condition: ConditionKey) -> float:
    if condition.direction == "identity":
        return float(condition.magnitude)
    if condition.direction == "negative":
        return -float(condition.magnitude)
    if condition.direction in {"positive", "increase"}:
        return float(condition.magnitude)
    raise ValueError("M5-A source condition has an unsupported direction")


def _persistent_fault_condition(
    experiment_id: str,
    condition: ConditionKey,
) -> ReplayFaultCondition:
    family = cast(FaultFamily, condition.fault_family)
    target_by_experiment: dict[str, FaultTarget] = {
        "replay-lidar-y-bias": "lidar",
        "replay-camera-noise-correctly-reported": "camera",
        "replay-camera-noise-underreported": "camera",
        "replay-camera-calibration-x": "camera",
        "replay-camera-calibration-yaw": "camera",
        "replay-camera-timestamp-offset": "camera",
        "replay-camera-dropout": "camera",
        "replay-common-mode-x": "both",
    }
    try:
        target = target_by_experiment[experiment_id]
    except KeyError as error:
        raise ValueError("M5-A source experiment is not preregistered") from error
    return ReplayFaultCondition(
        experiment_id=experiment_id,
        family=family,
        target=target,
        axis=condition.fault_axis,
        unit=condition.unit,
        value=_signed_value(condition),
        identity=condition.direction == "identity",
        active_frames=None,
    )


def _health_rows(intent_path: Path) -> list[dict[str, Any]]:
    intent = load_json_object(intent_path)
    health = intent.get("health_transfer_panel")
    if not isinstance(health, dict):
        raise ValueError("frozen M5 health panel is invalid")
    conditions = health.get("conditions")
    if not isinstance(conditions, list):
        raise ValueError("frozen M5 health conditions are invalid")
    rows: list[dict[str, Any]] = []
    for value in conditions:
        if not isinstance(value, dict):
            raise ValueError("frozen M5 health condition row is invalid")
        rows.append(cast("dict[str, Any]", value))
    return rows


def _numeric_values(value: object) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError("frozen M5 health values are invalid")
    result: list[float] = []
    for item in cast("list[object]", value):
        if type(item) not in {int, float}:
            raise ValueError("frozen M5 health values are invalid")
        result.append(float(cast("int | float", item)))
    return tuple(result)


def _health_case_specs(
    intent: LoadedReplayIntent,
) -> tuple[ReplayHealthCaseSpec, ...]:
    identities = {
        identity.experiment_id: identity
        for identity in expected_replay_identities()
        if identity.panel_id == M5_HEALTH_PANEL_ID
    }
    rows = _health_rows(intent.path)
    result: list[ReplayHealthCaseSpec] = []
    for row in rows:
        condition_id = row.get("condition_id")
        family = row.get("family")
        target = row.get("target")
        axis = row.get("axis")
        unit = row.get("unit")
        values = _numeric_values(row.get("values"))
        if (
            not isinstance(condition_id, str)
            or family
            not in {
                "identity",
                "additive-position-bias",
                "increased-noise-underreported",
                "increased-noise-correctly-reported",
                "timestamp-offset",
                "dropout",
                "calibration-translation",
                "calibration-yaw",
                "common-mode-position-bias",
            }
            or target not in {"camera", "lidar", "both", "none"}
            or not isinstance(axis, str)
            or not isinstance(unit, str)
        ):
            raise ValueError("frozen M5 health condition coordinate is invalid")
        identity = identities.get(condition_id)
        if identity is None:
            raise ValueError("frozen M5 health condition identity is missing")
        for value_index, value in enumerate(values):
            result.append(
                ReplayHealthCaseSpec(
                    identity=identity,
                    identity_sha256=replay_experiment_identity_sha256(identity),
                    family=cast(FaultFamily, family),
                    target=cast(FaultTarget, target),
                    axis=axis,
                    unit=unit,
                    value_index=value_index,
                    value=value,
                )
            )
    return tuple(result)


def load_replay_plan(*, source_root: Path) -> LoadedReplayPlan:
    """Authenticate and expand the exact frozen M5 intent and M3 source matrix."""

    intent = load_replay_intent(source_root=source_root)
    matrix = load_experiment_matrix(
        M5_PERSISTENT_MATRIX_PATH,
        source_root=source_root,
    )
    if matrix.matrix_sha256 != M3_PROCEDURAL_MATRIX_SHA256:
        raise ValueError("M5-A requires the frozen M3 release matrix")
    persistent_identities = tuple(
        identity
        for identity in expected_replay_identities()
        if identity.panel_id == M5_PERSISTENT_PANEL_ID
    )
    if len(persistent_identities) != len(matrix.manifests):
        raise ValueError("M5-A identity and source-manifest counts disagree")
    persistent_cases: list[ReplayPersistentCase] = []
    for identity, raw_manifest in zip(
        persistent_identities,
        matrix.manifests,
        strict=True,
    ):
        manifest = _persistent_manifest(raw_manifest)
        if sha256_digest(manifest) != identity.source_sha256:
            raise ValueError("M5-A identity order disagrees with the source matrix")
        for source_condition in expected_conditions(manifest):
            persistent_cases.append(
                ReplayPersistentCase(
                    identity=identity,
                    identity_sha256=replay_experiment_identity_sha256(identity),
                    source_manifest=manifest,
                    source_condition=source_condition,
                    fault_condition=_persistent_fault_condition(
                        identity.experiment_id,
                        source_condition,
                    ),
                )
            )
    return LoadedReplayPlan(
        intent=intent,
        persistent_matrix=matrix,
        persistent_cases=tuple(persistent_cases),
        health_cases=_health_case_specs(intent),
    )
