from __future__ import annotations

import io
import json
import traceback
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from fusion_fault_bench.adapters.nuscenes import (
    DATASET_AUTHENTICATION,
    PROFILE,
    VERSION_DIRECTORY,
    NuScenesAdapterError,
    NuScenesAdapterErrorCode,
    load_nuscenes_mini,
    validate_nuscenes_mini,
)

type Tables = dict[str, list[dict[str, Any]]]

_TABLE_NAMES = (
    "attribute",
    "calibrated_sensor",
    "category",
    "ego_pose",
    "instance",
    "log",
    "sample",
    "sample_annotation",
    "sample_data",
    "scene",
    "sensor",
    "visibility",
)


def _fixture_tables() -> Tables:
    identity = [1.0, 0.0, 0.0, 0.0]
    translation = [0.0, 0.0, 0.0]
    intrinsic = [[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]]
    return {
        "attribute": [
            {
                "token": "attribute-secret",
                "name": "vehicle.moving",
                "description": "",
            }
        ],
        "calibrated_sensor": [
            {
                "token": "calibrated-camera-secret",
                "sensor_token": "sensor-camera-secret",
                "translation": translation,
                "rotation": identity,
                "camera_intrinsic": intrinsic,
            },
            {
                "token": "calibrated-lidar-secret",
                "sensor_token": "sensor-lidar-secret",
                "translation": translation,
                "rotation": identity,
                "camera_intrinsic": [],
            },
        ],
        "category": [
            {
                "token": "category-secret",
                "name": "vehicle.car",
                "description": "",
                "index": 1,
            }
        ],
        "ego_pose": [
            {
                "token": "pose-camera-one-secret",
                "translation": translation,
                "rotation": identity,
                "timestamp": 1_000_000,
            },
            {
                "token": "pose-lidar-one-secret",
                "translation": translation,
                "rotation": identity,
                "timestamp": 1_000_100,
            },
            {
                "token": "pose-camera-two-secret",
                "translation": translation,
                "rotation": identity,
                "timestamp": 2_000_000,
            },
            {
                "token": "pose-lidar-two-secret",
                "translation": translation,
                "rotation": identity,
                "timestamp": 2_000_100,
            },
        ],
        "instance": [
            {
                "token": "instance-secret",
                "category_token": "category-secret",
                "nbr_annotations": 2,
                "first_annotation_token": "annotation-one-secret",
                "last_annotation_token": "annotation-two-secret",
            }
        ],
        "log": [
            {
                "token": "log-secret",
                "logfile": "logfile",
                "vehicle": "vehicle",
                "date_captured": "2020-01-01",
                "location": "test-location",
            }
        ],
        "sample": [
            {
                "token": "sample-one-secret",
                "timestamp": 1_000_000,
                "scene_token": "scene-secret",
                "prev": "",
                "next": "sample-two-secret",
            },
            {
                "token": "sample-two-secret",
                "timestamp": 2_000_000,
                "scene_token": "scene-secret",
                "prev": "sample-one-secret",
                "next": "",
            },
        ],
        "sample_annotation": [
            {
                "token": "annotation-one-secret",
                "sample_token": "sample-one-secret",
                "instance_token": "instance-secret",
                "attribute_tokens": ["attribute-secret"],
                "visibility_token": "",
                "translation": [10.0, 0.0, 0.5],
                "size": [2.0, 4.0, 1.5],
                "rotation": identity,
                "num_lidar_pts": 5,
                "num_radar_pts": 0,
                "prev": "",
                "next": "annotation-two-secret",
            },
            {
                "token": "annotation-two-secret",
                "sample_token": "sample-two-secret",
                "instance_token": "instance-secret",
                "attribute_tokens": [],
                "visibility_token": "visibility-secret",
                "translation": [11.0, 0.0, 0.5],
                "size": [2.0, 4.0, 1.5],
                "rotation": identity,
                "num_lidar_pts": 4,
                "num_radar_pts": 1,
                "prev": "annotation-one-secret",
                "next": "",
            },
        ],
        "sample_data": [
            {
                "token": "camera-data-one-secret",
                "sample_token": "sample-one-secret",
                "ego_pose_token": "pose-camera-one-secret",
                "calibrated_sensor_token": "calibrated-camera-secret",
                "filename": "samples/CAM_FRONT/camera-one-secret.jpg",
                "fileformat": "jpg",
                "width": 1600,
                "height": 900,
                "timestamp": 1_000_000,
                "is_key_frame": True,
                "prev": "",
                "next": "camera-data-two-secret",
            },
            {
                "token": "camera-data-two-secret",
                "sample_token": "sample-two-secret",
                "ego_pose_token": "pose-camera-two-secret",
                "calibrated_sensor_token": "calibrated-camera-secret",
                "filename": "samples/CAM_FRONT/camera-two-secret.jpg",
                "fileformat": "jpg",
                "width": 1600,
                "height": 900,
                "timestamp": 2_000_000,
                "is_key_frame": True,
                "prev": "camera-data-one-secret",
                "next": "",
            },
            {
                "token": "lidar-data-one-secret",
                "sample_token": "sample-one-secret",
                "ego_pose_token": "pose-lidar-one-secret",
                "calibrated_sensor_token": "calibrated-lidar-secret",
                "filename": "samples/LIDAR_TOP/lidar-one-secret.bin",
                "fileformat": "bin",
                "width": 0,
                "height": 0,
                "timestamp": 1_000_100,
                "is_key_frame": True,
                "prev": "",
                "next": "lidar-data-two-secret",
            },
            {
                "token": "lidar-data-two-secret",
                "sample_token": "sample-two-secret",
                "ego_pose_token": "pose-lidar-two-secret",
                "calibrated_sensor_token": "calibrated-lidar-secret",
                "filename": "samples/LIDAR_TOP/lidar-two-secret.bin",
                "fileformat": "bin",
                "width": 0,
                "height": 0,
                "timestamp": 2_000_100,
                "is_key_frame": True,
                "prev": "lidar-data-one-secret",
                "next": "",
            },
        ],
        "scene": [
            {
                "token": "scene-secret",
                "name": "scene-test",
                "description": "",
                "log_token": "log-secret",
                "nbr_samples": 2,
                "first_sample_token": "sample-one-secret",
                "last_sample_token": "sample-two-secret",
            }
        ],
        "sensor": [
            {
                "token": "sensor-camera-secret",
                "channel": "CAM_FRONT",
                "modality": "camera",
            },
            {
                "token": "sensor-lidar-secret",
                "channel": "LIDAR_TOP",
                "modality": "lidar",
            },
        ],
        "visibility": [
            {
                "token": "visibility-secret",
                "level": "v80-100",
                "description": "",
            }
        ],
    }


def _write_table(root: Path, table: str, rows: object) -> None:
    version_root = root / VERSION_DIRECTORY
    version_root.mkdir(parents=True, exist_ok=True)
    (version_root / f"{table}.json").write_text(
        json.dumps(rows, allow_nan=False),
        encoding="utf-8",
    )


def _write_dataset(tmp_path: Path) -> tuple[Path, Tables]:
    root = tmp_path / "private-dataset-root"
    tables = _fixture_tables()
    for table, rows in tables.items():
        _write_table(root, table, rows)
    for row in tables["sample_data"]:
        blob = root / row["filename"]
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b"not-read")
    return root, tables


def _rewrite(root: Path, tables: Tables, table: str) -> None:
    _write_table(root, table, tables[table])


def _assert_sanitized_error(
    root: Path,
    expected_code: NuScenesAdapterErrorCode,
) -> None:
    with pytest.raises(NuScenesAdapterError) as caught:
        validate_nuscenes_mini(root)
    error = caught.value
    assert error.code is expected_code
    rendered = f"{error!s} {error!r} {''.join(traceback.format_exception(error))}"
    assert str(root) not in rendered
    for private_value in (
        "scene-secret",
        "sample-one-secret",
        "annotation-one-secret",
        "camera-one-secret.jpg",
    ):
        assert private_value not in rendered


def test_valid_fixture_loads_typed_indexes_and_returns_only_safe_aggregate_fields(
    tmp_path: Path,
) -> None:
    root, _ = _write_dataset(tmp_path)

    metadata = load_nuscenes_mini(root)
    validation = validate_nuscenes_mini(root)

    assert metadata.samples["sample-one-secret"].timestamp == 1_000_000
    assert metadata.sample_annotations["annotation-one-secret"].visibility_token == ""
    assert metadata.calibrated_sensors["calibrated-camera-secret"].camera_intrinsic[0][0] == 1000
    assert repr(metadata) == "NuScenesMiniMetadata()"
    assert all(item.name != "root" for item in fields(metadata))
    with pytest.raises(TypeError):
        metadata.samples["new-secret"] = metadata.samples["sample-one-secret"]  # type: ignore[index]

    assert validation.profile == PROFILE
    assert dict(validation.expected_headline_counts) == {
        "scene": 10,
        "sample": 404,
        "sample_annotation": 18_538,
    }
    assert not validation.headline_profile_passed_attested
    assert validation.structural_integrity_passed_attested
    assert validation.keyframe_blob_check_count == 4
    assert validation.keyframe_blob_validation_passed_attested
    assert validation.dataset_authentication == DATASET_AUTHENTICATION

    public = validation.to_public_mapping()
    assert set(public) == {
        "profile",
        "expected_headline_counts",
        "headline_profile_passed_attested",
        "structural_integrity_passed_attested",
        "keyframe_blob_check_count",
        "keyframe_blob_validation_passed_attested",
        "dataset_authentication",
    }
    encoded = json.dumps(public, sort_keys=True)
    assert str(root) not in encoded
    assert "-secret" not in encoded


def test_blob_payloads_are_never_opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _ = _write_dataset(tmp_path)
    original_open = io.open

    def metadata_only_open(file: str | bytes | int | Path, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, Path)):
            assert str(file).endswith(".json")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", metadata_only_open)

    assert validate_nuscenes_mini(root).keyframe_blob_check_count == 4


@pytest.mark.parametrize("table", _TABLE_NAMES)
def test_every_required_table_must_exist_and_be_a_list(tmp_path: Path, table: str) -> None:
    root, _ = _write_dataset(tmp_path)
    (root / VERSION_DIRECTORY / f"{table}.json").unlink()

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.TABLE_INVALID)


@pytest.mark.parametrize(
    ("contents", "expected_code"),
    [
        ("{}", NuScenesAdapterErrorCode.TABLE_INVALID),
        ("not-json", NuScenesAdapterErrorCode.TABLE_INVALID),
        ('[{"token":"first","token":"second"}]', NuScenesAdapterErrorCode.TABLE_INVALID),
        ("[7]", NuScenesAdapterErrorCode.ROW_INVALID),
    ],
)
def test_malformed_table_boundaries_are_rejected(
    tmp_path: Path,
    contents: str,
    expected_code: NuScenesAdapterErrorCode,
) -> None:
    root, _ = _write_dataset(tmp_path)
    (root / VERSION_DIRECTORY / "attribute.json").write_text(contents, encoding="utf-8")

    _assert_sanitized_error(root, expected_code)


def test_nonstandard_json_numbers_are_rejected(tmp_path: Path) -> None:
    root, _ = _write_dataset(tmp_path)
    (root / VERSION_DIRECTORY / "attribute.json").write_text(
        '[{"token":"private","name":"private","description":NaN}]',
        encoding="utf-8",
    )

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.TABLE_INVALID)


@pytest.mark.parametrize("root_variant", ["missing", "file", "version-symlink"])
def test_invalid_dataset_roots_are_sanitized(tmp_path: Path, root_variant: str) -> None:
    root = tmp_path / "private-root-secret"
    if root_variant == "file":
        root.write_bytes(b"private")
    elif root_variant == "version-symlink":
        root.mkdir()
        outside = tmp_path / "private-version-secret"
        outside.mkdir()
        (root / VERSION_DIRECTORY).symlink_to(outside, target_is_directory=True)

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.ROOT_INVALID)


def test_non_path_root_is_rejected_without_echoing_it() -> None:
    private_root = "private-root-secret"
    with pytest.raises(NuScenesAdapterError) as caught:
        validate_nuscenes_mini(private_root)  # type: ignore[arg-type]

    assert caught.value.code is NuScenesAdapterErrorCode.ROOT_INVALID
    assert private_root not in f"{caught.value!s} {caught.value!r}"


def test_symlinked_metadata_tables_are_rejected(tmp_path: Path) -> None:
    root, _ = _write_dataset(tmp_path)
    table = root / VERSION_DIRECTORY / "attribute.json"
    outside = tmp_path / "private-table-secret.json"
    outside.write_text("[]", encoding="utf-8")
    table.unlink()
    table.symlink_to(outside)

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.TABLE_INVALID)


@pytest.mark.parametrize("token", ["", "attribute-secret"])
def test_empty_or_duplicate_primary_tokens_are_rejected(tmp_path: Path, token: str) -> None:
    root, tables = _write_dataset(tmp_path)
    duplicate = dict(tables["attribute"][0])
    duplicate["token"] = token
    tables["attribute"].append(duplicate)
    _rewrite(root, tables, "attribute")

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.ROW_INVALID)


@pytest.mark.parametrize(
    ("table", "row_index", "field_name", "bad_value"),
    [
        ("calibrated_sensor", 0, "sensor_token", "missing-secret"),
        ("instance", 0, "category_token", "missing-secret"),
        ("sample", 0, "scene_token", "missing-secret"),
        ("sample_annotation", 0, "sample_token", "missing-secret"),
        ("sample_annotation", 0, "instance_token", "missing-secret"),
        ("sample_annotation", 0, "attribute_tokens", ["missing-secret"]),
        ("sample_annotation", 1, "visibility_token", "missing-secret"),
        ("sample_data", 0, "sample_token", "missing-secret"),
        ("sample_data", 0, "ego_pose_token", "missing-secret"),
        ("sample_data", 0, "calibrated_sensor_token", "missing-secret"),
        ("scene", 0, "log_token", "missing-secret"),
    ],
)
def test_missing_foreign_keys_are_rejected(
    tmp_path: Path,
    table: str,
    row_index: int,
    field_name: str,
    bad_value: object,
) -> None:
    root, tables = _write_dataset(tmp_path)
    tables[table][row_index][field_name] = bad_value
    _rewrite(root, tables, table)

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.RELATION_INVALID)


@pytest.mark.parametrize(
    ("table", "row_index", "field_name", "bad_value"),
    [
        ("sample", 0, "timestamp", 0),
        ("sample", 0, "timestamp", True),
        ("ego_pose", 0, "translation", "not-a-vector"),
        ("ego_pose", 0, "translation", [0.0, 0.0]),
        ("ego_pose", 0, "translation", [0.0, 0.0, "not-a-number"]),
        ("ego_pose", 0, "rotation", "not-a-quaternion"),
        ("ego_pose", 0, "rotation", [1.0, 0.0, 0.0]),
        ("ego_pose", 0, "rotation", [2.0, 0.0, 0.0, 0.0]),
        ("sample_annotation", 0, "size", [0.0, 4.0, 1.5]),
        ("sample_annotation", 0, "num_lidar_pts", -1),
        ("sample_annotation", 0, "num_radar_pts", 1.5),
        ("sample_annotation", 0, "attribute_tokens", "not-a-list"),
        ("sample_data", 0, "width", 0),
        ("sample_data", 0, "is_key_frame", 1),
        ("calibrated_sensor", 0, "camera_intrinsic", "not-a-matrix"),
        ("calibrated_sensor", 0, "camera_intrinsic", [[1.0, 0.0, 0.0]]),
        ("calibrated_sensor", 0, "camera_intrinsic", []),
        (
            "calibrated_sensor",
            1,
            "camera_intrinsic",
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        (
            "calibrated_sensor",
            0,
            "camera_intrinsic",
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        ("sensor", 0, "modality", "thermal"),
    ],
)
def test_invalid_numeric_and_modality_rows_are_rejected(
    tmp_path: Path,
    table: str,
    row_index: int,
    field_name: str,
    bad_value: object,
) -> None:
    root, tables = _write_dataset(tmp_path)
    tables[table][row_index][field_name] = bad_value
    _rewrite(root, tables, table)

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.ROW_INVALID)


def test_missing_required_row_field_is_rejected(tmp_path: Path) -> None:
    root, tables = _write_dataset(tmp_path)
    del tables["attribute"][0]["description"]
    _rewrite(root, tables, "attribute")

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.ROW_INVALID)


def test_channel_modality_mismatch_is_rejected(tmp_path: Path) -> None:
    root, tables = _write_dataset(tmp_path)
    tables["sensor"][0]["modality"] = "lidar"
    _rewrite(root, tables, "sensor")

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.RELATION_INVALID)


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (
            lambda tables: tables["sample"][1].__setitem__("prev", ""),
            NuScenesAdapterErrorCode.CHAIN_INVALID,
        ),
        (
            lambda tables: tables["sample"][0].__setitem__("next", ""),
            NuScenesAdapterErrorCode.CHAIN_INVALID,
        ),
        (
            lambda tables: tables["scene"][0].__setitem__("nbr_samples", 1),
            NuScenesAdapterErrorCode.CHAIN_INVALID,
        ),
        (
            lambda tables: tables["scene"][0].__setitem__(
                "first_sample_token", "sample-two-secret"
            ),
            NuScenesAdapterErrorCode.CHAIN_INVALID,
        ),
        (
            lambda tables: tables["instance"][0].__setitem__("nbr_annotations", 1),
            NuScenesAdapterErrorCode.CHAIN_INVALID,
        ),
        (
            lambda tables: tables["sample_annotation"][1].__setitem__("prev", ""),
            NuScenesAdapterErrorCode.CHAIN_INVALID,
        ),
        (
            lambda tables: tables["sample_annotation"][0].__setitem__("next", ""),
            NuScenesAdapterErrorCode.CHAIN_INVALID,
        ),
        (
            lambda tables: tables["instance"][0].__setitem__(
                "first_annotation_token", "annotation-two-secret"
            ),
            NuScenesAdapterErrorCode.CHAIN_INVALID,
        ),
        (
            lambda tables: tables["sample"][1].__setitem__("timestamp", 1_000_000),
            NuScenesAdapterErrorCode.CHAIN_INVALID,
        ),
        (
            lambda tables: tables["sample_annotation"][1].__setitem__(
                "sample_token", "sample-one-secret"
            ),
            NuScenesAdapterErrorCode.CHAIN_INVALID,
        ),
    ],
)
def test_sample_and_annotation_chain_contracts_are_enforced(
    tmp_path: Path,
    mutate: Callable[[Tables], None],
    expected_code: NuScenesAdapterErrorCode,
) -> None:
    root, tables = _write_dataset(tmp_path)
    mutate(tables)
    for table in ("sample", "scene", "instance", "sample_annotation"):
        _rewrite(root, tables, table)

    _assert_sanitized_error(root, expected_code)


def test_sample_data_pose_timestamp_must_match(tmp_path: Path) -> None:
    root, tables = _write_dataset(tmp_path)
    tables["ego_pose"][0]["timestamp"] = 1_000_001
    _rewrite(root, tables, "ego_pose")

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.RELATION_INVALID)


def test_sample_data_chain_timestamp_must_increase(tmp_path: Path) -> None:
    root, tables = _write_dataset(tmp_path)
    tables["sample_data"][1]["timestamp"] = 1_000_000
    tables["ego_pose"][2]["timestamp"] = 1_000_000
    _rewrite(root, tables, "sample_data")
    _rewrite(root, tables, "ego_pose")

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.CHAIN_INVALID)


def test_sample_data_chain_cannot_change_calibration_identity(tmp_path: Path) -> None:
    root, tables = _write_dataset(tmp_path)
    alternate = dict(tables["calibrated_sensor"][0])
    alternate["token"] = "calibrated-camera-alternate-secret"
    tables["calibrated_sensor"].append(alternate)
    tables["sample_data"][1]["calibrated_sensor_token"] = alternate["token"]
    _rewrite(root, tables, "calibrated_sensor")
    _rewrite(root, tables, "sample_data")

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.CHAIN_INVALID)


def test_calibration_change_cannot_hide_behind_split_channel_chains(
    tmp_path: Path,
) -> None:
    root, tables = _write_dataset(tmp_path)
    alternate = dict(tables["calibrated_sensor"][0])
    alternate["token"] = "calibrated-camera-alternate-secret"
    tables["calibrated_sensor"].append(alternate)
    tables["sample_data"][0]["next"] = ""
    tables["sample_data"][1]["prev"] = ""
    tables["sample_data"][1]["calibrated_sensor_token"] = alternate["token"]
    _rewrite(root, tables, "calibrated_sensor")
    _rewrite(root, tables, "sample_data")

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.CHAIN_INVALID)


def test_sample_data_prev_next_reciprocity_is_required(tmp_path: Path) -> None:
    root, tables = _write_dataset(tmp_path)
    tables["sample_data"][0]["next"] = ""
    _rewrite(root, tables, "sample_data")

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.CHAIN_INVALID)


def test_sample_data_group_must_be_one_complete_chain(tmp_path: Path) -> None:
    root, tables = _write_dataset(tmp_path)
    tables["sample_data"][0]["next"] = ""
    tables["sample_data"][1]["prev"] = ""
    _rewrite(root, tables, "sample_data")

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.CHAIN_INVALID)


def test_sample_data_chain_cannot_change_scene(tmp_path: Path) -> None:
    root, tables = _write_dataset(tmp_path)
    tables["sample"][0]["next"] = ""
    tables["sample"][1]["prev"] = ""
    tables["sample"][1]["scene_token"] = "scene-two-secret"
    tables["scene"][0]["nbr_samples"] = 1
    tables["scene"][0]["last_sample_token"] = "sample-one-secret"
    tables["scene"].append(
        {
            "token": "scene-two-secret",
            "name": "scene-test-two",
            "description": "",
            "log_token": "log-secret",
            "nbr_samples": 1,
            "first_sample_token": "sample-two-secret",
            "last_sample_token": "sample-two-secret",
        }
    )
    tables["sample_annotation"][0]["next"] = ""
    tables["sample_annotation"][1]["prev"] = ""
    tables["sample_annotation"][1]["instance_token"] = "instance-two-secret"
    tables["instance"][0]["nbr_annotations"] = 1
    tables["instance"][0]["last_annotation_token"] = "annotation-one-secret"
    tables["instance"].append(
        {
            "token": "instance-two-secret",
            "category_token": "category-secret",
            "nbr_annotations": 1,
            "first_annotation_token": "annotation-two-secret",
            "last_annotation_token": "annotation-two-secret",
        }
    )
    for table in ("sample", "scene", "sample_annotation", "instance"):
        _rewrite(root, tables, table)

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.CHAIN_INVALID)


def test_each_sample_requires_exact_front_camera_and_top_lidar_keyframes(
    tmp_path: Path,
) -> None:
    root, tables = _write_dataset(tmp_path)
    tables["sample_data"][1]["is_key_frame"] = False
    _rewrite(root, tables, "sample_data")

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.KEYFRAME_INVALID)


@pytest.mark.parametrize("failure", ["missing", "empty", "directory", "escape"])
def test_keyframe_blob_path_and_file_contract_is_enforced(
    tmp_path: Path,
    failure: str,
) -> None:
    root, tables = _write_dataset(tmp_path)
    original = root / tables["sample_data"][0]["filename"]
    if failure == "missing":
        original.unlink()
    elif failure == "empty":
        original.write_bytes(b"")
    elif failure == "directory":
        original.unlink()
        original.mkdir()
    else:
        tables["sample_data"][0]["filename"] = "../outside-secret.bin"
        (tmp_path / "outside-secret.bin").write_bytes(b"outside")
        _rewrite(root, tables, "sample_data")

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.BLOB_INVALID)


def test_keyframe_blob_symlinks_are_rejected(tmp_path: Path) -> None:
    root, tables = _write_dataset(tmp_path)
    original = root / tables["sample_data"][0]["filename"]
    target = tmp_path / "outside-secret.jpg"
    target.write_bytes(b"outside")
    original.unlink()
    original.symlink_to(target)

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.BLOB_INVALID)


def test_symlinked_blob_path_components_are_rejected(tmp_path: Path) -> None:
    root, tables = _write_dataset(tmp_path)
    outside = tmp_path / "outside-directory"
    outside.mkdir()
    (outside / "camera-secret.jpg").write_bytes(b"outside")
    link = root / "linked-secret"
    link.symlink_to(outside, target_is_directory=True)
    tables["sample_data"][0]["filename"] = "linked-secret/camera-secret.jpg"
    _rewrite(root, tables, "sample_data")

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.BLOB_INVALID)


def test_non_directory_blob_path_components_are_rejected(tmp_path: Path) -> None:
    root, tables = _write_dataset(tmp_path)
    tables["sample_data"][0]["filename"] = "samples/CAM_FRONT/camera-one-secret.jpg/child-secret"
    _rewrite(root, tables, "sample_data")

    _assert_sanitized_error(root, NuScenesAdapterErrorCode.BLOB_INVALID)
