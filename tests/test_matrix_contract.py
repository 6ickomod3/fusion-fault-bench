from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fusion_fault_bench.contracts.matrix_v1 import (
    EXPERIMENT_MATRIX_ADAPTER,
    M3_CI_SMOKE_MATRIX_SHA256,
    M3_PROCEDURAL_MATRIX_SHA256,
    experiment_matrix_json_schema,
    load_experiment_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
RELEASE_MATRIX = Path("examples/matrices/m3-procedural-v1.json")
SMOKE_MATRIX = Path("examples/matrices/m3-ci-smoke-v1.json")


def _copy_inputs(tmp_path: Path) -> Path:
    shutil.copytree(ROOT / "examples", tmp_path / "examples")
    return tmp_path


def _mutate_matrix(
    root: Path,
    relative: Path,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    path = root / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    mutation(value)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_release_matrix_loads_exact_frozen_inputs() -> None:
    loaded = load_experiment_matrix(RELEASE_MATRIX, source_root=ROOT)

    assert loaded.path == ROOT / RELEASE_MATRIX
    assert loaded.matrix.matrix_id == "m3-procedural-v1"
    assert loaded.matrix_sha256 == M3_PROCEDURAL_MATRIX_SHA256
    assert len(loaded.manifests) == 8
    assert tuple(profile.profile_id for profile in loaded.profiles) == (
        "constant-velocity-front-roi-v1",
        "constant-velocity-fov-edge-v1",
        "constant-velocity-ci-smoke-v1",
    )
    assert tuple(manifest.experiment for manifest in loaded.manifests) == (
        "procedural-lidar-y-bias",
        "procedural-camera-noise-correctly-reported",
        "procedural-camera-noise-underreported",
        "procedural-camera-calibration-x",
        "procedural-camera-calibration-yaw",
        "procedural-camera-timestamp-offset",
        "procedural-camera-dropout",
        "procedural-common-mode-x-fov-edge",
    )


def test_smoke_matrix_is_distinct_and_never_release_evidence() -> None:
    loaded = load_experiment_matrix(ROOT / SMOKE_MATRIX, source_root=ROOT)

    assert loaded.matrix.matrix_id == "m3-ci-smoke-v1"
    assert loaded.matrix.result_selection == "ci-only-not-release-evidence"
    assert loaded.matrix_sha256 == M3_CI_SMOKE_MATRIX_SHA256
    assert len(loaded.manifests) == 1
    assert loaded.manifests[0].experiment == "procedural-ci-smoke"
    assert len(loaded.profiles) == 1
    assert loaded.profiles[0].profile_id == "constant-velocity-ci-smoke-v1"


def test_schema_is_strict_and_uses_public_alias() -> None:
    schema = experiment_matrix_json_schema()
    assert "schema" in schema["properties"]
    assert schema["additionalProperties"] is False

    value = json.loads((ROOT / SMOKE_MATRIX).read_text(encoding="utf-8"))
    value["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EXPERIMENT_MATRIX_ADAPTER.validate_json(json.dumps(value))


@pytest.mark.parametrize(
    "bad_path",
    [
        "/absolute/manifest.json",
        "../manifest.json",
        "examples/manifests/../manifest.json",
        "examples\\manifests\\manifest.json",
        "examples//manifests/manifest.json",
        "examples/profiles/not-a-manifest.json",
        "examples/manifests/not-json.txt",
    ],
)
def test_matrix_contract_rejects_unsafe_manifest_paths(
    tmp_path: Path,
    bad_path: str,
) -> None:
    root = _copy_inputs(tmp_path)
    _mutate_matrix(
        root,
        SMOKE_MATRIX,
        lambda value: value["execution_order"][0].update({"manifest": bad_path}),
    )

    with pytest.raises((ValidationError, ValueError)):
        load_experiment_matrix(SMOKE_MATRIX, source_root=root)


def test_matrix_contract_rejects_duplicate_reordered_or_changed_entries(tmp_path: Path) -> None:
    root = _copy_inputs(tmp_path)
    _mutate_matrix(
        root,
        RELEASE_MATRIX,
        lambda value: value["execution_order"].reverse(),
    )
    with pytest.raises(ValidationError, match="release matrix differs"):
        load_experiment_matrix(RELEASE_MATRIX, source_root=root)

    root = _copy_inputs(tmp_path / "duplicate")
    _mutate_matrix(
        root,
        RELEASE_MATRIX,
        lambda value: value["execution_order"].__setitem__(
            1,
            dict(value["execution_order"][0]),
        ),
    )
    with pytest.raises(ValidationError, match="paths must be unique"):
        load_experiment_matrix(RELEASE_MATRIX, source_root=root)

    root = _copy_inputs(tmp_path / "selection")
    _mutate_matrix(
        root,
        RELEASE_MATRIX,
        lambda value: value.update({"result_selection": "ci-only-not-release-evidence"}),
    )
    with pytest.raises(ValidationError, match="release matrix differs"):
        load_experiment_matrix(RELEASE_MATRIX, source_root=root)


def test_matrix_id_filename_and_digest_are_all_bound(tmp_path: Path) -> None:
    root = _copy_inputs(tmp_path)
    release_bytes = (root / RELEASE_MATRIX).read_bytes()
    (root / SMOKE_MATRIX).write_bytes(release_bytes)
    with pytest.raises(ValueError, match="filename disagree"):
        load_experiment_matrix(SMOKE_MATRIX, source_root=root)

    root = _copy_inputs(tmp_path / "unknown")
    _mutate_matrix(
        root,
        SMOKE_MATRIX,
        lambda value: value.update({"matrix_id": "m3-other-v1"}),
    )
    with pytest.raises(ValidationError):
        load_experiment_matrix(SMOKE_MATRIX, source_root=root)

    root = _copy_inputs(tmp_path / "digest")
    path = root / SMOKE_MATRIX
    value = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    loaded = load_experiment_matrix(SMOKE_MATRIX, source_root=root)
    assert loaded.matrix_sha256 == M3_CI_SMOKE_MATRIX_SHA256


def test_referenced_manifest_and_profile_digests_are_recomputed(tmp_path: Path) -> None:
    root = _copy_inputs(tmp_path)
    manifest_path = root / "examples/manifests/procedural-ci-smoke-v1alpha1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["experiment"] = "procedural-ci-smoke-mutated"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest digest"):
        load_experiment_matrix(SMOKE_MATRIX, source_root=root)

    root = _copy_inputs(tmp_path / "profile")
    profile_path = root / "examples/profiles/constant-velocity-ci-smoke-v1.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["source"]["frame_period_s"] = 0.2
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match=r"frozen profile|profile digest"):
        load_experiment_matrix(SMOKE_MATRIX, source_root=root)


def test_duplicate_keys_and_non_object_json_are_rejected(tmp_path: Path) -> None:
    root = _copy_inputs(tmp_path)
    path = root / SMOKE_MATRIX
    raw = path.read_text(encoding="utf-8")
    path.write_text(raw.replace("{", '{"schema":"ffb.experiment-matrix/v1",', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_experiment_matrix(SMOKE_MATRIX, source_root=root)

    root = _copy_inputs(tmp_path / "array")
    (root / SMOKE_MATRIX).write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level JSON"):
        load_experiment_matrix(SMOKE_MATRIX, source_root=root)


def test_paths_must_stay_in_a_real_normalized_source_tree(tmp_path: Path) -> None:
    root = _copy_inputs(tmp_path / "root")
    with pytest.raises(ValueError, match="inside source_root"):
        load_experiment_matrix(ROOT / SMOKE_MATRIX, source_root=root)
    with pytest.raises(ValueError, match="normalized"):
        load_experiment_matrix(
            Path("examples/matrices/../matrices/m3-ci-smoke-v1.json"),
            source_root=root,
        )
    with pytest.raises(ValueError, match="existing directory"):
        load_experiment_matrix(SMOKE_MATRIX, source_root=tmp_path / "missing")

    file_root = tmp_path / "not-directory"
    file_root.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="existing directory"):
        load_experiment_matrix(SMOKE_MATRIX, source_root=file_root)

    symlink_root = tmp_path / "source-link"
    symlink_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        load_experiment_matrix(SMOKE_MATRIX, source_root=symlink_root)


def test_matrix_and_references_reject_symlinks_missing_files_and_oversize(
    tmp_path: Path,
) -> None:
    root = _copy_inputs(tmp_path / "leaf-link")
    manifest_path = root / "examples/manifests/procedural-ci-smoke-v1alpha1.json"
    backup = root / "manifest-backup.json"
    manifest_path.rename(backup)
    manifest_path.symlink_to(backup)
    with pytest.raises(ValueError, match="symlinks"):
        load_experiment_matrix(SMOKE_MATRIX, source_root=root)

    root = _copy_inputs(tmp_path / "component-link")
    profiles = root / "examples/profiles"
    profiles_real = root / "examples/profiles-real"
    profiles.rename(profiles_real)
    profiles.symlink_to(profiles_real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        load_experiment_matrix(SMOKE_MATRIX, source_root=root)

    root = _copy_inputs(tmp_path / "missing")
    (root / "examples/profiles/constant-velocity-ci-smoke-v1.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        load_experiment_matrix(SMOKE_MATRIX, source_root=root)

    root = _copy_inputs(tmp_path / "oversize")
    (root / SMOKE_MATRIX).write_text(" " * (1024 * 1024 + 1), encoding="utf-8")
    with pytest.raises(ValueError, match="byte cap"):
        load_experiment_matrix(SMOKE_MATRIX, source_root=root)


def test_matrix_path_itself_must_be_a_regular_file(tmp_path: Path) -> None:
    root = _copy_inputs(tmp_path)
    path = root / SMOKE_MATRIX
    path.unlink()
    path.mkdir()
    with pytest.raises(ValueError, match="regular files"):
        load_experiment_matrix(SMOKE_MATRIX, source_root=root)
