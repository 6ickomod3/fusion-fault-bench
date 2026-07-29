from __future__ import annotations

from pathlib import Path

import pytest

from fusion_fault_bench.contracts.io import load_json_object, load_manifest


def test_load_json_object_requires_object(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level JSON"):
        load_json_object(path)


def test_load_manifest_rejects_nonstandard_nan(tmp_path: Path) -> None:
    path = tmp_path / "nan.json"
    path.write_text('{"value": NaN}', encoding="utf-8")

    with pytest.raises(ValueError, match="non-standard JSON number"):
        load_manifest(path)


def test_load_manifest_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"seed": 1, "seed": 2}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_manifest(path)


@pytest.mark.parametrize("spelling", ["-0", "-0.0", "-0e0", "-0.000E+10"])
def test_load_json_rejects_every_negative_zero_spelling(tmp_path: Path, spelling: str) -> None:
    path = tmp_path / "negative-zero.json"
    path.write_text(f'{{"value": {spelling}}}', encoding="utf-8")

    with pytest.raises(ValueError, match="negative zero"):
        load_json_object(path)


def test_load_manifest_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "missing.json")
