from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fusion_fault_bench.contracts.procedural_profile_v1 import (
    EDGE_PROFILE_SHA256,
    MAIN_PROFILE_SHA256,
    PROCEDURAL_PROFILE_ADAPTER,
    SMOKE_PROFILE_SHA256,
    EdgeProceduralProfile,
    MainProceduralProfile,
    SmokeProceduralProfile,
    load_procedural_profile,
    procedural_profile_json_schema,
    profile_sequence_count,
)

PROFILE_ROOT = Path("examples/profiles")


@pytest.mark.parametrize(
    ("filename", "model_type", "digest", "frame_count", "object_count", "test_count"),
    [
        (
            "constant-velocity-front-roi-v1.json",
            MainProceduralProfile,
            MAIN_PROFILE_SHA256,
            48,
            6,
            200,
        ),
        (
            "constant-velocity-fov-edge-v1.json",
            EdgeProceduralProfile,
            EDGE_PROFILE_SHA256,
            48,
            4,
            100,
        ),
        (
            "constant-velocity-ci-smoke-v1.json",
            SmokeProceduralProfile,
            SMOKE_PROFILE_SHA256,
            8,
            3,
            4,
        ),
    ],
)
def test_load_frozen_profiles(
    filename: str,
    model_type: type[object],
    digest: str,
    frame_count: int,
    object_count: int,
    test_count: int,
) -> None:
    profile = load_procedural_profile(PROFILE_ROOT / filename)
    assert isinstance(profile, model_type)
    assert len(digest) == 64
    assert profile.source.frame_count == frame_count
    assert profile.source.object_count == object_count
    assert profile_sequence_count(profile, "test") == test_count
    assert profile.eligibility.camera_half_fov_rad == 0.7
    assert profile.rig.camera_true_extrinsic.translation_m == (1.5, 0.0, 1.5)


def test_main_profile_exposes_all_split_counts() -> None:
    profile = load_procedural_profile(PROFILE_ROOT / "constant-velocity-front-roi-v1.json")
    assert profile_sequence_count(profile, "train") == 200
    assert profile_sequence_count(profile, "validation") == 200
    assert profile_sequence_count(profile, "test") == 200


def test_single_split_profiles_reject_other_splits() -> None:
    edge = load_procedural_profile(PROFILE_ROOT / "constant-velocity-fov-edge-v1.json")
    with pytest.raises(ValueError, match="only the test split"):
        profile_sequence_count(edge, "train")


def test_profile_contract_rejects_nested_extra_and_semantic_mutation() -> None:
    path = PROFILE_ROOT / "constant-velocity-ci-smoke-v1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["source"]["unexpected"] = True
    with pytest.raises(ValidationError, match="unexpected"):
        PROCEDURAL_PROFILE_ADAPTER.validate_json(json.dumps(value))

    value = json.loads(path.read_text(encoding="utf-8"))
    value["splits"]["test"]["initial_x_m"]["equation"] = "11+10*u0"
    with pytest.raises(ValidationError, match="smoke equations"):
        PROCEDURAL_PROFILE_ADAPTER.validate_json(json.dumps(value))


def test_profile_schema_is_discriminated_and_extra_forbid() -> None:
    schema = procedural_profile_json_schema()
    assert schema["discriminator"]["propertyName"] == "profile_id"
    assert len(schema["oneOf"]) == 3
    definitions = schema["$defs"]
    assert definitions["MainProceduralProfile"]["additionalProperties"] is False
    assert definitions["ProceduralSourceSpec"]["additionalProperties"] is False


def _mutate_path(value: object, path: tuple[str | int, ...], replacement: object) -> None:
    cursor = value
    for component in path[:-1]:
        cursor = cursor[component]  # type: ignore[index]
    cursor[path[-1]] = replacement  # type: ignore[index]


@pytest.mark.parametrize(
    ("filename", "path", "replacement", "message"),
    (
        (
            "constant-velocity-ci-smoke-v1.json",
            ("rng_contract", "camera_error_draw", "stream_name"),
            "lidar",
            "camera stream",
        ),
        (
            "constant-velocity-ci-smoke-v1.json",
            ("rng_contract", "lidar_error_draw", "stream_name"),
            "camera",
            "lidar stream",
        ),
        (
            "constant-velocity-ci-smoke-v1.json",
            ("eligibility", "x_min_m"),
            6.0,
            "common ROI",
        ),
        (
            "constant-velocity-ci-smoke-v1.json",
            ("rig", "camera_true_extrinsic", "translation_m"),
            [1.4, 0.0, 1.5],
            "camera translation",
        ),
        (
            "constant-velocity-ci-smoke-v1.json",
            ("rig", "camera_true_extrinsic", "quaternion_wxyz"),
            [1.0, 0.0, 0.0, 0.0],
            "camera quaternion",
        ),
        (
            "constant-velocity-ci-smoke-v1.json",
            ("timing", "physical_camera_time_offset_s"),
            0.1,
            "camera time offset",
        ),
        (
            "constant-velocity-ci-smoke-v1.json",
            ("timing", "physical_lidar_time_offset_s"),
            0.1,
            "lidar time offset",
        ),
        (
            "constant-velocity-front-roi-v1.json",
            ("splits", "train", "initial_x_m", "support"),
            [28.0, 10.0],
            "support bounds",
        ),
        (
            "constant-velocity-front-roi-v1.json",
            ("splits", "train", "initial_y_m", "lane_centers_m"),
            [-3.0, 3.0],
            "train lateral",
        ),
        (
            "constant-velocity-front-roi-v1.json",
            ("splits", "train", "velocity_x_mps", "support"),
            [-2.0, 2.0],
            "train split equations",
        ),
        (
            "constant-velocity-front-roi-v1.json",
            ("splits", "validation", "initial_y_m", "absolute_support"),
            [8.0, 5.0],
            "absolute support",
        ),
        (
            "constant-velocity-front-roi-v1.json",
            ("splits", "validation", "velocity_y_mps", "absolute_support"),
            [1.0, 3.0],
            "validation lateral velocity",
        ),
        (
            "constant-velocity-front-roi-v1.json",
            ("splits", "validation", "object_side_by_parity", "even"),
            1,
            "validation side parity",
        ),
        (
            "constant-velocity-front-roi-v1.json",
            ("splits", "validation", "initial_x_m", "support"),
            [29.0, 40.0],
            "validation equations",
        ),
        (
            "constant-velocity-front-roi-v1.json",
            ("splits", "test", "initial_y_m", "lateral_centers_m"),
            [-6.0, -4.0, -1.0, 1.0, 4.0, 7.0],
            "test lateral centers",
        ),
        (
            "constant-velocity-front-roi-v1.json",
            ("splits", "test", "initial_y_m", "jitter_half_width_m"),
            0.2,
            "test lateral jitter",
        ),
        (
            "constant-velocity-front-roi-v1.json",
            ("splits", "test", "velocity_x_mps", "support"),
            [-6.0, -3.0],
            "test equations",
        ),
        (
            "constant-velocity-fov-edge-v1.json",
            ("splits", "test", "bearing_rad", "absolute_support"),
            [0.67, 0.695],
            "edge bearing support",
        ),
        (
            "constant-velocity-fov-edge-v1.json",
            ("splits", "test", "bearing_rad", "inside_half_fov_margin_rad"),
            [0.004, 0.02],
            "edge FOV margin",
        ),
        (
            "constant-velocity-fov-edge-v1.json",
            ("splits", "test", "object_side_by_parity", "even"),
            -1,
            "edge side parity",
        ),
        (
            "constant-velocity-fov-edge-v1.json",
            ("splits", "test", "initial_range_m", "support"),
            [19.0, 40.0],
            "edge equations",
        ),
        (
            "constant-velocity-ci-smoke-v1.json",
            ("splits", "test", "initial_y_m", "lateral_centers_m"),
            [-1.0, 0.0, 2.0],
            "smoke lateral centers",
        ),
        (
            "constant-velocity-ci-smoke-v1.json",
            ("splits", "test", "initial_y_m", "jitter_half_width_m"),
            0.2,
            "smoke jitter",
        ),
        (
            "constant-velocity-ci-smoke-v1.json",
            ("splits", "test", "velocity_x_mps", "support"),
            [-1.0, 0.5],
            "smoke equations",
        ),
        (
            "constant-velocity-ci-smoke-v1.json",
            ("source", "frame_count"),
            9,
            "source dimensions",
        ),
        (
            "constant-velocity-ci-smoke-v1.json",
            ("rng_contract", "latent_draw", "shape"),
            [2, 4],
            "latent draw shape",
        ),
        (
            "constant-velocity-ci-smoke-v1.json",
            ("rng_contract", "latent_draw", "component_order"),
            ["u0", "u1", "u3", "u2"],
            "latent component order",
        ),
        (
            "constant-velocity-ci-smoke-v1.json",
            ("rng_contract", "dropout_draw", "shape"),
            [9],
            "dropout draw shape",
        ),
    ),
)
def test_profile_semantic_mutations_fail_closed(
    filename: str,
    path: tuple[str | int, ...],
    replacement: object,
    message: str,
) -> None:
    value = json.loads((PROFILE_ROOT / filename).read_text(encoding="utf-8"))
    _mutate_path(value, path, replacement)

    with pytest.raises(ValidationError, match=message):
        PROCEDURAL_PROFILE_ADAPTER.validate_json(json.dumps(value))
