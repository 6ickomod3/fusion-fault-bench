from __future__ import annotations

import json

from fusion_fault_bench.cli import main


def test_manifest_validate_json_output(example_path, capsys) -> None:
    result = main(["manifest", "validate", str(example_path), "--json"])

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "manifest_sha256": "a603d090f77ad97f20f92b4ec685fe19624d0974dc7d1be4328e2cd3c963bd3e",
        "schema": "ffb.manifest/v1alpha1",
        "valid": True,
    }


def test_manifest_validate_human_output(example_path, capsys) -> None:
    result = main(["manifest", "validate", str(example_path)])

    assert result == 0
    assert capsys.readouterr().out.startswith("valid ffb.manifest/v1alpha1 ")


def test_manifest_digest_output(example_path, capsys) -> None:
    result = main(["manifest", "digest", str(example_path)])

    assert result == 0
    assert capsys.readouterr().out.strip() == (
        "a603d090f77ad97f20f92b4ec685fe19624d0974dc7d1be4328e2cd3c963bd3e"
    )


def test_schema_show_outputs_json(capsys) -> None:
    result = main(["schema", "show", "metric"])

    assert result == 0
    schema = json.loads(capsys.readouterr().out)
    assert "oneOf" in schema
    assert {"LocalizationMetricRecord", "RateMetricRecord"}.issubset(schema["$defs"])


def test_invalid_manifest_returns_exit_code_two(tmp_path, capsys) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{}", encoding="utf-8")

    assert main(["manifest", "validate", str(path)]) == 2
    assert capsys.readouterr().err.startswith("error:")
