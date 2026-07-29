"""Fail-closed loading for the two preregistered M3 experiment matrices."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.common import ContractModel
from fusion_fault_bench.contracts.io import load_manifest
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AnalyticCrossoverManifest,
    ExperimentManifestV1Alpha1,
    ProceduralSource,
)
from fusion_fault_bench.contracts.procedural_profile_v1 import (
    ProceduralProfileV1,
    load_procedural_profile,
    profile_sequence_count,
)

type MatrixId = Literal["m3-procedural-v1", "m3-ci-smoke-v1"]
type ResultSelection = Literal[
    "publish-complete-matrix-without-favorable-result-selection",
    "ci-only-not-release-evidence",
]

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

M3_PROCEDURAL_MATRIX_SHA256 = "7162a01bb766a38e8f7196cba9eb2c6c546f5866bc640f220252bb6e7aab6f9b"
M3_CI_SMOKE_MATRIX_SHA256 = "fd52418c58867d7ddd6a09a0907f797c030b3c424f6b49eca7fd09334d48d186"

_MATRIX_FILE_CAP_BYTES = 1024 * 1024
_REFERENCE_FILE_CAP_BYTES = 4 * 1024 * 1024

_RELEASE_EXECUTION = (
    (
        "examples/manifests/procedural-lidar-y-bias-v1alpha1.json",
        "e5a4aa3ddf9832cd8cc88eb0be87151bfb44efbf99363f826f3b360c81960056",
    ),
    (
        "examples/manifests/procedural-camera-noise-correct-v1alpha1.json",
        "4359f4f5cc172017b4cbf2eb5d7470692a25846501066d999ae416b56caa1add",
    ),
    (
        "examples/manifests/procedural-camera-noise-underreported-v1alpha1.json",
        "900b4893ca33eb8ce84d10cc14a3e350e6b07de4783641c4212b4ee0337c549d",
    ),
    (
        "examples/manifests/procedural-camera-calibration-x-v1alpha1.json",
        "463637c5dc2b8a8135e40dab4b23e1a3fd61b9475364a523b999e61d1787cdce",
    ),
    (
        "examples/manifests/procedural-camera-calibration-yaw-v1alpha1.json",
        "b6ec17bd745483af6863857db93376e5d560ccc9d4a6cfffc13e027e2e4289fa",
    ),
    (
        "examples/manifests/procedural-camera-timestamp-offset-v1alpha1.json",
        "292d47f2711223382cca48e229a2cb7a1bd6ebfe392bb36d0730505b5e9f3d57",
    ),
    (
        "examples/manifests/procedural-camera-dropout-v1alpha1.json",
        "79ae8c67ff9994b7d6b764e8ef8b7c2185c3cb4871b6489d64bb4385f786022a",
    ),
    (
        "examples/manifests/procedural-common-mode-x-edge-v1alpha1.json",
        "1b78059d62b016ca8a25cc23d22a73576ef1e61742c08f35394e9ad273c06d3c",
    ),
)
_RELEASE_PROFILES = (
    (
        "examples/profiles/constant-velocity-front-roi-v1.json",
        "4771a6e69d75b9af41f99ab794c0af1b51e6103e43474c8e0f07df3e6f3ca68c",
    ),
    (
        "examples/profiles/constant-velocity-fov-edge-v1.json",
        "ca1544f69023847af7bdad9f1306ae3885f2e5d067d6afc026038f87ae36448d",
    ),
    (
        "examples/profiles/constant-velocity-ci-smoke-v1.json",
        "7f2479c064e0f8104789dfc3ce704a78aabdd46c1be7a31fdd2e75dbe3b407ed",
    ),
)
_SMOKE_EXECUTION = (
    (
        "examples/manifests/procedural-ci-smoke-v1alpha1.json",
        "cc1c26f8ebce3bf17143cef89238719dde14568091838fcabf6ecaeef0e702fd",
    ),
)
_SMOKE_PROFILES = (
    (
        "examples/profiles/constant-velocity-ci-smoke-v1.json",
        "7f2479c064e0f8104789dfc3ce704a78aabdd46c1be7a31fdd2e75dbe3b407ed",
    ),
)

M3_RELEASE_MANIFEST_SHA256S: tuple[str, ...] = tuple(digest for _, digest in _RELEASE_EXECUTION)
M3_SMOKE_MANIFEST_SHA256S: tuple[str, ...] = tuple(digest for _, digest in _SMOKE_EXECUTION)


def _validate_repository_relative_path(value: str, *, directory: str) -> None:
    if not value or "\\" in value:
        raise ValueError("matrix paths must be normalized repository-relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError("matrix paths must be normalized repository-relative POSIX paths")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("matrix paths must not contain empty, dot, or parent segments")
    if len(path.parts) != 3 or path.parts[:2] != ("examples", directory):
        raise ValueError(f"matrix path must name one file under examples/{directory}")
    if not path.name.endswith(".json"):
        raise ValueError("matrix paths must name JSON files")


class MatrixManifestEntryV1(ContractModel):
    """One content-addressed manifest in exact execution order."""

    manifest: str
    manifest_sha256: Digest

    @model_validator(mode="after")
    def require_safe_path(self) -> Self:
        _validate_repository_relative_path(self.manifest, directory="manifests")
        return self


class MatrixProfileEntryV1(ContractModel):
    """One content-addressed procedural profile in exact declared order."""

    profile: str
    profile_sha256: Digest

    @model_validator(mode="after")
    def require_safe_path(self) -> Self:
        _validate_repository_relative_path(self.profile, directory="profiles")
        return self


class ExperimentMatrixV1(ContractModel):
    """The exact release or CI-only M3 matrix; no scientific overrides exist."""

    schema_id: Literal["ffb.experiment-matrix/v1"] = Field(alias="schema")
    matrix_id: MatrixId
    execution_order: tuple[MatrixManifestEntryV1, ...]
    profiles: tuple[MatrixProfileEntryV1, ...]
    release_split: Literal["test"]
    result_selection: ResultSelection
    scientific_overrides: Literal["forbidden"]

    @model_validator(mode="after")
    def require_frozen_matrix(self) -> Self:
        actual_execution = tuple(
            (entry.manifest, entry.manifest_sha256) for entry in self.execution_order
        )
        actual_profiles = tuple((entry.profile, entry.profile_sha256) for entry in self.profiles)
        if len({path for path, _ in actual_execution}) != len(actual_execution):
            raise ValueError("matrix manifest paths must be unique")
        if len({path for path, _ in actual_profiles}) != len(actual_profiles):
            raise ValueError("matrix profile paths must be unique")

        if self.matrix_id == "m3-procedural-v1":
            if (
                actual_execution != _RELEASE_EXECUTION
                or actual_profiles != _RELEASE_PROFILES
                or self.result_selection
                != "publish-complete-matrix-without-favorable-result-selection"
            ):
                raise ValueError("release matrix differs from the preregistered order or content")
        elif (
            actual_execution != _SMOKE_EXECUTION
            or actual_profiles != _SMOKE_PROFILES
            or self.result_selection != "ci-only-not-release-evidence"
        ):
            raise ValueError("CI matrix differs from the preregistered order or content")
        return self


EXPERIMENT_MATRIX_ADAPTER = TypeAdapter(ExperimentMatrixV1)


@dataclass(frozen=True, slots=True)
class LoadedExperimentMatrix:
    """One matrix plus every strictly loaded referenced input."""

    path: Path
    matrix: ExperimentMatrixV1
    matrix_sha256: str
    manifests: tuple[ExperimentManifestV1Alpha1, ...]
    profiles: tuple[ProceduralProfileV1, ...]


def _absolute_normalized(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_real_source_root(source_root: Path) -> Path:
    root = _absolute_normalized(source_root)
    if not root.exists() or not root.is_dir():
        raise ValueError("source_root must be an existing directory")
    if root.is_symlink() or root.resolve(strict=True) != root:
        raise ValueError("source_root must not use symlink components")
    return root


def _relative_matrix_path(path: Path, *, source_root: Path) -> str:
    if not path.is_absolute():
        if any(part in {".", ".."} for part in path.parts):
            raise ValueError("matrix path must be normalized")
        candidate = source_root / path
    else:
        candidate = path
    candidate = _absolute_normalized(candidate)
    try:
        relative = candidate.relative_to(source_root)
    except ValueError as error:
        raise ValueError("matrix path must remain inside source_root") from error
    value = relative.as_posix()
    _validate_repository_relative_path(value, directory="matrices")
    return value


def _require_regular_no_symlink(path: Path, *, source_root: Path, byte_cap: int) -> None:
    try:
        relative = path.relative_to(source_root)
    except ValueError as error:
        raise ValueError("referenced path escapes source_root") from error
    current = source_root
    for part in relative.parts:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError as error:
            raise ValueError("matrix references a missing repository file") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("matrix paths and references must not use symlinks")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("matrix references must be regular files")
    if metadata.st_size > byte_cap:
        raise ValueError("matrix reference exceeds its byte cap")


def _strict_json_object(path: Path) -> tuple[dict[str, Any], str]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON number is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key is forbidden")
            result[key] = value
        return result

    raw = path.read_text(encoding="utf-8")
    value = json.loads(
        raw,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return value, raw


def _matrix_file_name(matrix_id: MatrixId) -> str:
    return f"examples/matrices/{matrix_id}.json"


def _expected_matrix_digest(matrix_id: MatrixId) -> str:
    if matrix_id == "m3-procedural-v1":
        return M3_PROCEDURAL_MATRIX_SHA256
    return M3_CI_SMOKE_MATRIX_SHA256


def _load_references(
    matrix: ExperimentMatrixV1,
    *,
    source_root: Path,
) -> tuple[tuple[ExperimentManifestV1Alpha1, ...], tuple[ProceduralProfileV1, ...]]:
    loaded_profiles: list[ProceduralProfileV1] = []
    profiles_by_id: dict[str, ProceduralProfileV1] = {}
    for entry in matrix.profiles:
        profile_path = source_root / entry.profile
        _require_regular_no_symlink(
            profile_path,
            source_root=source_root,
            byte_cap=_REFERENCE_FILE_CAP_BYTES,
        )
        profile = load_procedural_profile(profile_path)
        if sha256_digest(profile) != entry.profile_sha256:
            raise ValueError("matrix profile digest disagrees with the referenced profile")
        if profile.profile_id in profiles_by_id:
            raise ValueError("matrix resolves duplicate procedural profile IDs")
        profiles_by_id[profile.profile_id] = profile
        loaded_profiles.append(profile)

    loaded_manifests: list[ExperimentManifestV1Alpha1] = []
    for entry in matrix.execution_order:
        manifest_path = source_root / entry.manifest
        _require_regular_no_symlink(
            manifest_path,
            source_root=source_root,
            byte_cap=_REFERENCE_FILE_CAP_BYTES,
        )
        manifest = load_manifest(manifest_path)
        if sha256_digest(manifest) != entry.manifest_sha256:
            raise ValueError("matrix manifest digest disagrees with the referenced manifest")
        if isinstance(manifest, AnalyticCrossoverManifest):
            raise ValueError("M3 matrix manifests cannot use the analytic experiment kind")
        source = manifest.source
        if not isinstance(source, ProceduralSource):
            raise ValueError("M3 matrix manifests must use a procedural source")
        profile = profiles_by_id.get(source.profile_id)
        if profile is None:
            raise ValueError("matrix manifest references an undeclared profile")
        if sha256_digest(profile) != source.profile_sha256:
            raise ValueError("manifest profile digest disagrees with the matrix profile")
        if source.split != matrix.release_split:
            raise ValueError("manifest source split disagrees with the matrix release split")
        if profile_sequence_count(profile, source.split) != source.sequence_count:
            raise ValueError("manifest sequence count disagrees with its procedural profile")
        if (
            manifest.roi.x_min_m,
            manifest.roi.x_max_m,
            manifest.roi.abs_y_max_m,
            manifest.roi.camera_half_fov_rad,
        ) != (
            profile.eligibility.x_min_m,
            profile.eligibility.x_max_m,
            profile.eligibility.abs_y_max_m,
            profile.eligibility.camera_half_fov_rad,
        ):
            raise ValueError("manifest ROI disagrees with its procedural profile")
        loaded_manifests.append(manifest)
    return tuple(loaded_manifests), tuple(loaded_profiles)


def load_experiment_matrix(
    path: Path,
    *,
    source_root: Path,
) -> LoadedExperimentMatrix:
    """Load one exact frozen matrix and all content-addressed references."""

    root = _require_real_source_root(source_root)
    relative_path = _relative_matrix_path(path, source_root=root)
    matrix_path = root / relative_path
    _require_regular_no_symlink(
        matrix_path,
        source_root=root,
        byte_cap=_MATRIX_FILE_CAP_BYTES,
    )
    _, raw = _strict_json_object(matrix_path)
    matrix = EXPERIMENT_MATRIX_ADAPTER.validate_json(raw)
    if relative_path != _matrix_file_name(matrix.matrix_id):
        raise ValueError("matrix ID and repository-relative filename disagree")
    digest = sha256_digest(matrix)
    if digest != _expected_matrix_digest(matrix.matrix_id):
        raise ValueError("experiment matrix canonical digest is not preregistered")
    manifests, profiles = _load_references(matrix, source_root=root)
    return LoadedExperimentMatrix(
        path=matrix_path,
        matrix=matrix,
        matrix_sha256=digest,
        manifests=manifests,
        profiles=profiles,
    )


def experiment_matrix_json_schema() -> dict[str, Any]:
    """Return the strict schema for both accepted M3 matrices."""

    return EXPERIMENT_MATRIX_ADAPTER.json_schema(by_alias=True)
