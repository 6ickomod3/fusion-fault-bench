from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.health_v1 import (
    EXPECTED_M4_METHODS,
    EXPECTED_M4_SELECTION_CONDITIONS,
    EXPECTED_M4_THRESHOLDS,
    HEALTH_BENCHMARK_INTENT_ADAPTER,
    M4_HEALTH_INTENT_PATH,
    M4_HEALTH_INTENT_SHA256,
    health_benchmark_intent_json_schema,
    load_health_benchmark_intent,
)

ROOT = Path(__file__).resolve().parents[1]


def _copy_intent(tmp_path: Path) -> Path:
    destination = tmp_path / M4_HEALTH_INTENT_PATH
    destination.parent.mkdir(parents=True)
    shutil.copy2(ROOT / M4_HEALTH_INTENT_PATH, destination)
    return destination


def _mutate(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
) -> Path:
    path = _copy_intent(tmp_path)
    value = cast(
        dict[str, Any],
        json.loads(path.read_text(encoding="utf-8")),
    )
    mutation(value)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def test_loads_exact_frozen_m4_health_intent() -> None:
    loaded = load_health_benchmark_intent(source_root=ROOT)
    intent = loaded.intent

    assert loaded.path == ROOT / M4_HEALTH_INTENT_PATH
    assert loaded.intent_sha256 == M4_HEALTH_INTENT_SHA256
    assert sha256_digest(intent) == M4_HEALTH_INTENT_SHA256
    assert len(intent.validation_matrix) == 11
    assert len(intent.validation_controls) == 1
    assert len(intent.test_matrix) == 12
    assert len(intent.test_controls) == 6
    assert intent.methods == EXPECTED_M4_METHODS
    assert intent.threshold_selection.self_thresholds == EXPECTED_M4_THRESHOLDS
    assert intent.threshold_selection.cross_thresholds == EXPECTED_M4_THRESHOLDS
    assert intent.threshold_selection.candidate_count == 36
    assert intent.threshold_selection.selection_conditions == EXPECTED_M4_SELECTION_CONDITIONS


def test_schema_is_strict_and_exposes_public_schema_alias() -> None:
    schema = health_benchmark_intent_json_schema()

    assert "schema" in schema["properties"]
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["EventSchedulesV1"]["additionalProperties"] is False

    value = json.loads((ROOT / M4_HEALTH_INTENT_PATH).read_text(encoding="utf-8"))
    value["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        HEALTH_BENCHMARK_INTENT_ADAPTER.validate_json(json.dumps(value))


def test_loader_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    path = _copy_intent(tmp_path)
    raw = path.read_text(encoding="utf-8")
    path.write_text(
        raw.replace(
            '  "benchmark_id": "m4-health-v1",',
            ('  "benchmark_id": "m4-health-v1",\n  "benchmark_id": "m4-health-v1",'),
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_health_benchmark_intent(source_root=tmp_path)


def test_rejects_duplicate_condition_ids(tmp_path: Path) -> None:
    def duplicate(value: dict[str, Any]) -> None:
        rows = cast(list[dict[str, Any]], value["validation_matrix"])
        rows[1]["condition_id"] = rows[0]["condition_id"]

    _mutate(tmp_path, duplicate)
    with pytest.raises(ValidationError, match="globally unique"):
        load_health_benchmark_intent(source_root=tmp_path)


def test_rejects_changed_literal_validation_clean_condition(tmp_path: Path) -> None:
    def change_clean(value: dict[str, Any]) -> None:
        controls = cast(list[dict[str, Any]], value["validation_controls"])
        controls[0]["values"] = [1.0]

    _mutate(tmp_path, change_clean)
    with pytest.raises(ValidationError, match="literal sole validation clean"):
        load_health_benchmark_intent(source_root=tmp_path)


@pytest.mark.parametrize(
    "field",
    ("self_thresholds", "cross_thresholds"),
)
def test_rejects_changed_36_candidate_threshold_grid(
    tmp_path: Path,
    field: str,
) -> None:
    def shorten_grid(value: dict[str, Any]) -> None:
        selection = cast(dict[str, Any], value["threshold_selection"])
        thresholds = cast(list[float], selection[field])
        thresholds.pop()

    _mutate(tmp_path, shorten_grid)
    with pytest.raises(ValidationError, match="threshold grid"):
        load_health_benchmark_intent(source_root=tmp_path)


def test_rejects_selection_condition_outside_numeric_utility_set(
    tmp_path: Path,
) -> None:
    def select_direct_condition(value: dict[str, Any]) -> None:
        selection = cast(dict[str, Any], value["threshold_selection"])
        conditions = cast(list[str], selection["selection_conditions"])
        conditions[0] = "validation-camera-timestamp-offset"

    _mutate(tmp_path, select_direct_condition)
    with pytest.raises(ValidationError, match="ineligible condition"):
        load_health_benchmark_intent(source_root=tmp_path)


def test_rejects_reordered_selection_conditions(tmp_path: Path) -> None:
    def reorder(value: dict[str, Any]) -> None:
        selection = cast(dict[str, Any], value["threshold_selection"])
        conditions = cast(list[str], selection["selection_conditions"])
        conditions[0], conditions[1] = conditions[1], conditions[0]

    _mutate(tmp_path, reorder)
    with pytest.raises(ValidationError, match="condition set or order"):
        load_health_benchmark_intent(source_root=tmp_path)


def test_rejects_camera_yaw_leaking_out_of_held_out_test_family(
    tmp_path: Path,
) -> None:
    def change_yaw_role(value: dict[str, Any]) -> None:
        rows = cast(list[dict[str, Any]], value["test_matrix"])
        yaw = next(row for row in rows if row["condition_id"] == "test-camera-calibration-yaw")
        yaw["test_role"] = "unseen-severity"

    _mutate(tmp_path, change_yaw_role)
    with pytest.raises(ValidationError, match="held-out test family"):
        load_health_benchmark_intent(source_root=tmp_path)


def test_rejects_changed_cold_start_control_membership(tmp_path: Path) -> None:
    def change_schedule(value: dict[str, Any]) -> None:
        controls = cast(list[dict[str, Any]], value["test_controls"])
        cold = next(
            row for row in controls if row["condition_id"] == "test-cold-start-lidar-y-bias"
        )
        cold["schedule"] = "standard"

    _mutate(tmp_path, change_schedule)
    with pytest.raises(ValidationError, match="cold-start diagnostic set"):
        load_health_benchmark_intent(source_root=tmp_path)


def test_rejects_changed_cold_start_evaluation_windows(tmp_path: Path) -> None:
    def change_window(value: dict[str, Any]) -> None:
        evaluation = cast(dict[str, Any], value["evaluation"])
        schedules = cast(dict[str, Any], evaluation["loss_windows_by_schedule"])
        cold = cast(dict[str, Any], schedules["cold_start"])
        cold["score"] = [2, 48]

    _mutate(tmp_path, change_window)
    with pytest.raises(ValidationError, match="evaluation windows"):
        load_health_benchmark_intent(source_root=tmp_path)


def test_rejects_changed_fault_operator_values(tmp_path: Path) -> None:
    def change_operator(value: dict[str, Any]) -> None:
        rows = cast(list[dict[str, Any]], value["validation_matrix"])
        rows[0]["values"] = [-2.0, -0.25, 0.25, 2.0]

    _mutate(tmp_path, change_operator)
    with pytest.raises(ValidationError, match="validation fault-operator matrix"):
        load_health_benchmark_intent(source_root=tmp_path)


def test_rejects_changed_test_control_fault_operator(tmp_path: Path) -> None:
    def change_control_target(value: dict[str, Any]) -> None:
        controls = cast(list[dict[str, Any]], value["test_controls"])
        common_mode = next(
            row for row in controls if row["condition_id"] == "test-common-mode-x-edge"
        )
        common_mode["target"] = "camera"

    _mutate(tmp_path, change_control_target)
    with pytest.raises(ValidationError, match="test control fault-operator matrix"):
        load_health_benchmark_intent(source_root=tmp_path)


def test_rejects_changed_method_set_or_order(tmp_path: Path) -> None:
    def duplicate_method(value: dict[str, Any]) -> None:
        methods = cast(list[str], value["methods"])
        methods[-1] = methods[0]

    _mutate(tmp_path, duplicate_method)
    with pytest.raises(ValidationError, match="method set or order"):
        load_health_benchmark_intent(source_root=tmp_path)


def test_rejects_changed_oracle_exact_loss_tie_order(tmp_path: Path) -> None:
    def reorder_oracle(value: dict[str, Any]) -> None:
        oracle = cast(dict[str, Any], value["oracle"])
        tie_order = cast(list[str], oracle["tie_break_order"])
        tie_order[0], tie_order[1] = tie_order[1], tie_order[0]

    _mutate(tmp_path, reorder_oracle)
    with pytest.raises(ValidationError, match="oracle candidate or tie order"):
        load_health_benchmark_intent(source_root=tmp_path)


def test_rejects_changed_resource_cap(tmp_path: Path) -> None:
    def change_cap(value: dict[str, Any]) -> None:
        caps = cast(dict[str, Any], value["resource_caps"])
        caps["peak_rss_bytes_max"] = 2_147_483_648

    _mutate(tmp_path, change_cap)
    with pytest.raises(ValidationError):
        load_health_benchmark_intent(source_root=tmp_path)


def test_canonical_digest_rejects_otherwise_well_typed_text_change(
    tmp_path: Path,
) -> None:
    def change_text(value: dict[str, Any]) -> None:
        acceptance = cast(dict[str, Any], value["acceptance"])
        controls = cast(list[str], acceptance["required_controls"])
        controls[0] = "well-typed-but-not-preregistered-control"

    _mutate(tmp_path, change_text)
    with pytest.raises(ValidationError, match="canonical digest"):
        load_health_benchmark_intent(source_root=tmp_path)


@pytest.mark.parametrize(
    "path",
    (
        Path("examples/health/other.json"),
        Path("../m4-health-v1.json"),
    ),
)
def test_loader_rejects_noncanonical_intent_paths(
    tmp_path: Path,
    path: Path,
) -> None:
    _copy_intent(tmp_path)

    with pytest.raises(ValueError, match="health intent"):
        load_health_benchmark_intent(path, source_root=tmp_path)
