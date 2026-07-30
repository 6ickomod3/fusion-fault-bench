# pyright: reportPrivateUsage=false

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import fusion_fault_bench.replay_plan as replay_plan_module
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.replay_v1 import (
    M5_HEALTH_EXPERIMENT_IDS,
    M5_HEALTH_PANEL_ID,
    M5_PERSISTENT_EXPERIMENT_IDS,
    M5_PERSISTENT_PANEL_ID,
    replay_experiment_identity_sha256,
)
from fusion_fault_bench.replay_plan import (
    _health_case_specs,
    _health_rows,
    _numeric_values,
    _persistent_fault_condition,
    _persistent_manifest,
    _signed_value,
    load_replay_plan,
)

ROOT = Path(__file__).resolve().parents[1]


def test_replay_plan_expands_complete_frozen_matrices_in_order() -> None:
    plan = load_replay_plan(source_root=ROOT)

    assert len(plan.persistent_cases) == 71
    assert len(plan.health_cases) == 43
    assert (
        tuple(dict.fromkeys(case.identity.experiment_id for case in plan.persistent_cases))
        == M5_PERSISTENT_EXPERIMENT_IDS
    )
    assert tuple(dict.fromkeys(case.identity.experiment_id for case in plan.health_cases)) == (
        M5_HEALTH_EXPERIMENT_IDS
    )
    assert sum(case.fault_condition.identity for case in plan.persistent_cases) == 8
    assert sum(case.family == "identity" for case in plan.health_cases) == 1


def test_persistent_cases_bind_each_source_manifest_not_the_matrix() -> None:
    plan = load_replay_plan(source_root=ROOT)

    for case in plan.persistent_cases:
        assert case.identity.source_sha256 == sha256_digest(case.source_manifest)
        assert case.identity_sha256 == replay_experiment_identity_sha256(case.identity)
        assert case.fault_condition.experiment_id == case.identity.experiment_id


def test_persistent_signed_coordinates_preserve_the_m3_grids() -> None:
    plan = load_replay_plan(source_root=ROOT)
    lidar_bias = tuple(
        case.fault_condition
        for case in plan.persistent_cases
        if case.identity.experiment_id == "replay-lidar-y-bias"
    )
    correct_noise = tuple(
        case.fault_condition
        for case in plan.persistent_cases
        if case.identity.experiment_id == "replay-camera-noise-correctly-reported"
    )

    assert tuple(item.value for item in lidar_bias) == (
        0.0,
        -0.25,
        0.25,
        -0.5,
        0.5,
        -1.0,
        1.0,
        -2.0,
        2.0,
        -4.0,
        4.0,
    )
    assert tuple(item.value for item in correct_noise) == (1.0, 1.25, 1.5, 2.0, 4.0)
    assert lidar_bias[0].identity
    assert correct_noise[0].identity
    assert all(item.active_frames is None for item in (*lidar_bias, *correct_noise))


def test_health_selectors_and_dynamic_windows_match_preregistration() -> None:
    plan = load_replay_plan(source_root=ROOT)
    selectors = {case.selector for case in plan.health_cases}

    assert {
        "replay-lidar-output-y-bias:+3",
        "replay-lidar-timestamp-offset:+0.6",
        "replay-camera-noise-underreported:3",
        "replay-camera-timestamp-offset:+0.6",
        "replay-camera-calibration-x:+3",
        "replay-camera-output-y-bias:+3",
        "replay-camera-calibration-yaw:+0.06",
        "replay-lidar-noise-underreported:3",
        "replay-camera-noise-correctly-reported:3",
        "replay-lidar-noise-correctly-reported:3",
        "replay-common-mode-x:+4",
    }.issubset(selectors)

    for case in plan.health_cases:
        condition = case.for_frame_count(40)
        assert condition.active_frames == (10, 30)
        assert condition.selector == case.selector


def test_health_case_contract_rejects_out_of_grid_and_semantic_rebinding() -> None:
    case = load_replay_plan(source_root=ROOT).health_cases[0]

    with pytest.raises(ValueError, match="frozen M5-B matrix"):
        replace(case, value=100.0)
    with pytest.raises(ValueError, match="frozen M5-B matrix"):
        replace(case, target="lidar")
    with pytest.raises(ValueError, match="frozen M5-B matrix"):
        replace(case, value_index=1, value=case.value)


def test_persistent_case_contract_authenticates_panel_digest_manifest_and_coordinate() -> None:
    plan = load_replay_plan(source_root=ROOT)
    case = plan.persistent_cases[0]
    with pytest.raises(ValueError, match="wrong replay panel"):
        replace(
            case,
            identity=case.identity.model_copy(update={"panel_id": M5_HEALTH_PANEL_ID}),
        )
    with pytest.raises(ValueError, match="identity digest"):
        replace(case, identity_sha256="b" * 64)
    with pytest.raises(ValueError, match="bind its source manifest"):
        replace(case, source_manifest=plan.persistent_cases[-1].source_manifest)
    with pytest.raises(ValueError, match="wrong experiment"):
        replace(
            case,
            fault_condition=replace(case.fault_condition, experiment_id="replay-other"),
        )


def test_health_case_contract_authenticates_panel_digest_index_and_identity() -> None:
    case = load_replay_plan(source_root=ROOT).health_cases[0]
    with pytest.raises(ValueError, match="wrong replay panel"):
        replace(
            case,
            identity=case.identity.model_copy(update={"panel_id": M5_PERSISTENT_PANEL_ID}),
        )
    with pytest.raises(ValueError, match="identity digest"):
        replace(case, identity_sha256="b" * 64)
    with pytest.raises(ValueError, match="non-negative integer"):
        replace(case, value_index=True)
    with pytest.raises(ValueError, match="non-negative integer"):
        replace(case, value_index=-1)
    with pytest.raises(ValueError, match="finite"):
        replace(case, value=float("inf"))
    with pytest.raises(ValueError, match="not in the frozen M5-B matrix"):
        replace(
            case,
            identity=case.identity.model_copy(update={"experiment_id": "replay-other"}),
            identity_sha256=replay_experiment_identity_sha256(
                case.identity.model_copy(update={"experiment_id": "replay-other"})
            ),
        )


def test_loaded_plan_requires_exact_matrix_and_complete_panel_orders() -> None:
    plan = load_replay_plan(source_root=ROOT)
    with pytest.raises(ValueError, match="wrong persistent matrix"):
        replace(
            plan,
            persistent_matrix=replace(plan.persistent_matrix, matrix_sha256="b" * 64),
        )
    with pytest.raises(ValueError, match="persistent experiment order"):
        replace(
            plan,
            persistent_cases=tuple(
                case
                for case in plan.persistent_cases
                if case.identity.experiment_id != M5_PERSISTENT_EXPERIMENT_IDS[0]
            ),
        )
    with pytest.raises(ValueError, match="health condition order"):
        replace(
            plan,
            health_cases=tuple(
                case
                for case in plan.health_cases
                if case.identity.experiment_id != M5_HEALTH_EXPERIMENT_IDS[0]
            ),
        )


def test_plan_helpers_reject_unsupported_manifest_direction_and_experiment() -> None:
    case = load_replay_plan(source_root=ROOT).persistent_cases[0]
    with pytest.raises(ValueError, match="unsupported manifest"):
        _persistent_manifest(object())
    with pytest.raises(ValueError, match="unsupported direction"):
        _signed_value(replace(case.source_condition, direction="other"))
    with pytest.raises(ValueError, match="not preregistered"):
        _persistent_fault_condition("replay-other", case.source_condition)


def test_health_intent_helpers_reject_malformed_rows_and_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def missing_panel(_path: Path) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(replay_plan_module, "load_json_object", missing_panel)
    with pytest.raises(ValueError, match="health panel is invalid"):
        _health_rows(tmp_path / "intent.json")

    def missing_conditions(_path: Path) -> dict[str, Any]:
        return {"health_transfer_panel": {}}

    monkeypatch.setattr(replay_plan_module, "load_json_object", missing_conditions)
    with pytest.raises(ValueError, match="health conditions are invalid"):
        _health_rows(tmp_path / "intent.json")

    def invalid_row(_path: Path) -> dict[str, Any]:
        return {"health_transfer_panel": {"conditions": [None]}}

    monkeypatch.setattr(replay_plan_module, "load_json_object", invalid_row)
    with pytest.raises(ValueError, match="condition row is invalid"):
        _health_rows(tmp_path / "intent.json")

    with pytest.raises(ValueError, match="health values are invalid"):
        _numeric_values((1.0,))
    with pytest.raises(ValueError, match="health values are invalid"):
        _numeric_values([True])
    assert _numeric_values([1, 2.5]) == (1.0, 2.5)


def test_health_case_expansion_rejects_invalid_coordinate_and_missing_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent = load_replay_plan(source_root=ROOT).intent
    invalid_coordinate = {
        "condition_id": "replay-clean",
        "family": "unsupported",
        "target": "none",
        "axis": "none",
        "unit": "identity",
        "values": [0.0],
    }

    def invalid_coordinate_rows(_path: Path) -> list[dict[str, Any]]:
        return [invalid_coordinate]

    monkeypatch.setattr(replay_plan_module, "_health_rows", invalid_coordinate_rows)
    with pytest.raises(ValueError, match="condition coordinate is invalid"):
        _health_case_specs(intent)

    missing_identity = {
        "condition_id": "replay-other",
        "family": "identity",
        "target": "none",
        "axis": "none",
        "unit": "identity",
        "values": [0.0],
    }

    def missing_identity_rows(_path: Path) -> list[dict[str, Any]]:
        return [missing_identity]

    monkeypatch.setattr(replay_plan_module, "_health_rows", missing_identity_rows)
    with pytest.raises(ValueError, match="condition identity is missing"):
        _health_case_specs(intent)


def test_plan_loader_rejects_wrong_source_matrix_and_manifest_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = load_replay_plan(source_root=ROOT)

    def wrong_matrix(_path: Path, *, source_root: Path):
        del source_root
        return replace(plan.persistent_matrix, matrix_sha256="b" * 64)

    monkeypatch.setattr(
        replay_plan_module,
        "load_experiment_matrix",
        wrong_matrix,
    )
    with pytest.raises(ValueError, match="requires the frozen M3 release matrix"):
        load_replay_plan(source_root=ROOT)

    reordered = replace(
        plan.persistent_matrix,
        manifests=(
            plan.persistent_matrix.manifests[1],
            plan.persistent_matrix.manifests[0],
            *plan.persistent_matrix.manifests[2:],
        ),
    )

    def reordered_matrix(_path: Path, *, source_root: Path):
        del source_root
        return reordered

    monkeypatch.setattr(
        replay_plan_module,
        "load_experiment_matrix",
        reordered_matrix,
    )
    with pytest.raises(ValueError, match="identity order disagrees"):
        load_replay_plan(source_root=ROOT)
