"""Strict contracts for the frozen M5 nuScenes-mini replay intent."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self, cast

from pydantic import Field, TypeAdapter, model_validator

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.common import ContractModel
from fusion_fault_bench.contracts.io import load_json_object

M5_REPLAY_INTENT_PATH = Path("examples/replay/m5-nuscenes-mini-replay-v1.json")
M5_REPLAY_INTENT_SHA256 = "d429e36e2ce17ec8628c9bad4b5051fd54e0d88bcdeb966d112972e4c3dc2836"
M5_REPLAY_INTENT_BYTE_SHA256 = "d465a4b57de8af0c390395026e150c36922a9e44f7f09dafe9b85534808ccc0c"
M5_PERSISTENT_PANEL_ID = "m5-a-m3-persistent-replay"
M5_HEALTH_PANEL_ID = "m5-b-m4-apply-only-replay"
M5_PERSISTENT_MATRIX_SHA256 = "7162a01bb766a38e8f7196cba9eb2c6c546f5866bc640f220252bb6e7aab6f9b"
M5_HEALTH_FIT_SHA256 = "abd1540f292fe51a7a23a47b679fe8e1522d8c5e20a03125a880eb9242a608ee"

M5_SCENE_NAMES = (
    "scene-0061",
    "scene-0103",
    "scene-0553",
    "scene-0655",
    "scene-0757",
    "scene-0796",
    "scene-0916",
    "scene-1077",
    "scene-1094",
    "scene-1100",
)

_INTENT_FILE_CAP_BYTES = 512 * 1024

type Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
type Identifier = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")]
type ReplayPanelId = Literal[
    "m5-a-m3-persistent-replay",
    "m5-b-m4-apply-only-replay",
]


@dataclass(frozen=True, slots=True)
class ReplayIdentityBinding:
    """One preregistered experiment-to-source binding."""

    experiment_id: str
    source_sha256: str


M5_PERSISTENT_IDENTITY_BINDINGS = (
    ReplayIdentityBinding(
        "replay-lidar-y-bias",
        "e5a4aa3ddf9832cd8cc88eb0be87151bfb44efbf99363f826f3b360c81960056",
    ),
    ReplayIdentityBinding(
        "replay-camera-noise-correctly-reported",
        "4359f4f5cc172017b4cbf2eb5d7470692a25846501066d999ae416b56caa1add",
    ),
    ReplayIdentityBinding(
        "replay-camera-noise-underreported",
        "900b4893ca33eb8ce84d10cc14a3e350e6b07de4783641c4212b4ee0337c549d",
    ),
    ReplayIdentityBinding(
        "replay-camera-calibration-x",
        "463637c5dc2b8a8135e40dab4b23e1a3fd61b9475364a523b999e61d1787cdce",
    ),
    ReplayIdentityBinding(
        "replay-camera-calibration-yaw",
        "b6ec17bd745483af6863857db93376e5d560ccc9d4a6cfffc13e027e2e4289fa",
    ),
    ReplayIdentityBinding(
        "replay-camera-timestamp-offset",
        "292d47f2711223382cca48e229a2cb7a1bd6ebfe392bb36d0730505b5e9f3d57",
    ),
    ReplayIdentityBinding(
        "replay-camera-dropout",
        "79ae8c67ff9994b7d6b764e8ef8b7c2185c3cb4871b6489d64bb4385f786022a",
    ),
    ReplayIdentityBinding(
        "replay-common-mode-x",
        "1b78059d62b016ca8a25cc23d22a73576ef1e61742c08f35394e9ad273c06d3c",
    ),
)
M5_PERSISTENT_EXPERIMENT_IDS = tuple(
    binding.experiment_id for binding in M5_PERSISTENT_IDENTITY_BINDINGS
)
M5_HEALTH_EXPERIMENT_IDS = (
    "replay-camera-output-y-bias",
    "replay-lidar-output-y-bias",
    "replay-camera-noise-underreported",
    "replay-lidar-noise-underreported",
    "replay-camera-noise-correctly-reported",
    "replay-lidar-noise-correctly-reported",
    "replay-camera-timestamp-offset",
    "replay-lidar-timestamp-offset",
    "replay-camera-dropout",
    "replay-lidar-dropout",
    "replay-camera-calibration-x",
    "replay-camera-calibration-yaw",
    "replay-common-mode-x",
    "replay-clean",
)
M5_HEALTH_IDENTITY_BINDINGS = tuple(
    ReplayIdentityBinding(experiment_id, M5_HEALTH_FIT_SHA256)
    for experiment_id in M5_HEALTH_EXPERIMENT_IDS
)

_PERSISTENT_SOURCE_BY_EXPERIMENT = MappingProxyType(
    {item.experiment_id: item.source_sha256 for item in M5_PERSISTENT_IDENTITY_BINDINGS}
)
_HEALTH_SOURCE_BY_EXPERIMENT = MappingProxyType(
    {item.experiment_id: item.source_sha256 for item in M5_HEALTH_IDENTITY_BINDINGS}
)


class ReplayExperimentIdentityV1(ContractModel):
    """Content-addressed identity bound to every M5 experiment result."""

    schema_id: Literal["ffb.replay-experiment-identity/v1"] = Field(alias="schema")
    replay_intent_sha256: Digest
    panel_id: ReplayPanelId
    source_sha256: Digest
    experiment_id: Identifier

    @model_validator(mode="after")
    def require_frozen_binding(self) -> Self:
        if self.replay_intent_sha256 != M5_REPLAY_INTENT_SHA256:
            raise ValueError("replay identity does not bind the frozen M5 intent")
        if self.panel_id == M5_PERSISTENT_PANEL_ID:
            expected_source = _PERSISTENT_SOURCE_BY_EXPERIMENT.get(self.experiment_id)
            if expected_source is None or self.source_sha256 != expected_source:
                raise ValueError("replay identity does not bind the exact M5-A manifest")
        else:
            expected_source = _HEALTH_SOURCE_BY_EXPERIMENT.get(self.experiment_id)
            if expected_source is None or self.source_sha256 != expected_source:
                raise ValueError("replay identity does not bind the frozen M5-B fit")
        return self


REPLAY_EXPERIMENT_IDENTITY_ADAPTER = TypeAdapter(ReplayExperimentIdentityV1)


@dataclass(frozen=True, slots=True)
class LoadedReplayIntent:
    """Validated public coordinates extracted from the exact frozen intent."""

    path: Path
    intent_sha256: str
    byte_sha256: str
    scene_names: tuple[str, ...]
    persistent_identity_bindings: tuple[ReplayIdentityBinding, ...]
    health_identity_bindings: tuple[ReplayIdentityBinding, ...]


def replay_experiment_identity_sha256(identity: ReplayExperimentIdentityV1) -> str:
    """Return the canonical digest of one exact replay identity envelope."""

    return sha256_digest(identity)


def expected_replay_identities() -> tuple[ReplayExperimentIdentityV1, ...]:
    """Return identities in the preregistered panel and execution order."""

    persistent = tuple(
        ReplayExperimentIdentityV1(
            schema="ffb.replay-experiment-identity/v1",
            replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
            panel_id=M5_PERSISTENT_PANEL_ID,
            source_sha256=binding.source_sha256,
            experiment_id=binding.experiment_id,
        )
        for binding in M5_PERSISTENT_IDENTITY_BINDINGS
    )
    health = tuple(
        ReplayExperimentIdentityV1(
            schema="ffb.replay-experiment-identity/v1",
            replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
            panel_id=M5_HEALTH_PANEL_ID,
            source_sha256=binding.source_sha256,
            experiment_id=binding.experiment_id,
        )
        for binding in M5_HEALTH_IDENTITY_BINDINGS
    )
    return (*persistent, *health)


def _safe_intent_path(path: Path, *, source_root: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(source_root)))
    if not root.is_dir() or root.is_symlink() or root.resolve(strict=True) != root:
        raise ValueError("source_root must be one existing real directory")
    candidate = root / path if not path.is_absolute() else path
    absolute = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise ValueError("replay intent path must remain inside source_root") from error
    if relative != M5_REPLAY_INTENT_PATH:
        raise ValueError(f"replay intent must be {M5_REPLAY_INTENT_PATH.as_posix()}")
    current = root
    try:
        for part in relative.parts:
            current /= part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("replay intent path must not use symlinks")
    except ValueError:
        raise
    except OSError as error:
        raise ValueError("replay intent path is unavailable") from error
    metadata = absolute.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _INTENT_FILE_CAP_BYTES:
        raise ValueError("replay intent must be one bounded regular file")
    return absolute


def _mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return cast("dict[str, Any]", value)


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a JSON string array")
    return tuple(cast("list[str]", value))


def _persistent_bindings(
    value: object,
) -> tuple[ReplayIdentityBinding, ...]:
    if not isinstance(value, list):
        raise ValueError("persistent execution_order must be a JSON object array")
    result: list[ReplayIdentityBinding] = []
    for item in cast("list[object]", value):
        row = _mapping(item, field_name="persistent execution_order")
        experiment_id = row.get("experiment_id")
        manifest_sha256 = row.get("manifest_sha256")
        if not isinstance(experiment_id, str) or not isinstance(manifest_sha256, str):
            raise ValueError("persistent execution_order contains an invalid binding")
        result.append(ReplayIdentityBinding(experiment_id, manifest_sha256))
    return tuple(result)


def _health_bindings(value: object) -> tuple[ReplayIdentityBinding, ...]:
    if not isinstance(value, list):
        raise ValueError("health conditions must be a JSON object array")
    result: list[ReplayIdentityBinding] = []
    for item in cast("list[object]", value):
        row = _mapping(item, field_name="health conditions")
        condition_id = row.get("condition_id")
        if not isinstance(condition_id, str):
            raise ValueError("health conditions contain an invalid identifier")
        result.append(ReplayIdentityBinding(condition_id, M5_HEALTH_FIT_SHA256))
    return tuple(result)


def load_replay_intent(
    path: Path = M5_REPLAY_INTENT_PATH,
    *,
    source_root: Path,
) -> LoadedReplayIntent:
    """Load only the exact byte-frozen and canonically frozen M5 intent."""

    intent_path = _safe_intent_path(path, source_root=source_root)
    value = load_json_object(intent_path)
    raw = intent_path.read_bytes()
    byte_digest = hashlib.sha256(raw).hexdigest()
    if byte_digest != M5_REPLAY_INTENT_BYTE_SHA256:
        raise ValueError("M5 replay intent bytes are not preregistered")
    digest = sha256_digest(value)
    if digest != M5_REPLAY_INTENT_SHA256:
        raise ValueError("M5 replay intent canonical digest is not preregistered")

    if (
        value.get("schema") != "ffb.replay-benchmark-intent/v1"
        or value.get("benchmark_id") != "m5-nuscenes-mini-replay-v1"
        or value.get("preregistration_status") != "frozen-before-replay-descriptors-or-outcomes"
    ):
        raise ValueError("M5 replay intent freeze markers are invalid")
    source = _mapping(value.get("source_population"), field_name="source_population")
    persistent = _mapping(
        value.get("persistent_fault_panel"),
        field_name="persistent_fault_panel",
    )
    health = _mapping(
        value.get("health_transfer_panel"),
        field_name="health_transfer_panel",
    )
    scene_names = _string_tuple(source.get("scene_names"), field_name="scene_names")
    persistent_bindings = _persistent_bindings(persistent.get("execution_order"))
    health_bindings = _health_bindings(health.get("conditions"))
    if (
        scene_names != M5_SCENE_NAMES
        or persistent.get("panel_id") != M5_PERSISTENT_PANEL_ID
        or persistent.get("source_matrix_sha256") != M5_PERSISTENT_MATRIX_SHA256
        or persistent_bindings != M5_PERSISTENT_IDENTITY_BINDINGS
        or health.get("panel_id") != M5_HEALTH_PANEL_ID
        or health.get("source_fit_artifact_sha256") != M5_HEALTH_FIT_SHA256
        or health_bindings != M5_HEALTH_IDENTITY_BINDINGS
    ):
        raise ValueError("M5 replay intent coordinates are inconsistent")
    return LoadedReplayIntent(
        path=intent_path,
        intent_sha256=digest,
        byte_sha256=byte_digest,
        scene_names=scene_names,
        persistent_identity_bindings=persistent_bindings,
        health_identity_bindings=health_bindings,
    )
