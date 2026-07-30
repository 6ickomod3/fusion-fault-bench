from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fusion_fault_bench.cli import main

M5_RELEASE_SCHEMA_NAMES = (
    "replay-review-candidate-index",
    "replay-implementation-review-attestation",
    "replay-software-verification",
    "replay-privacy-license-attestation",
    "replay-validation-inputs",
    "replay-public-claim-projections",
    "replay-figure-spec",
    "replay-figure-source-binding",
    "replay-results-review-attestation",
    "replay-release-sidecar-index",
)


def test_m5_release_schemas_are_exposed_under_stable_names(capsys) -> None:
    for name in M5_RELEASE_SCHEMA_NAMES:
        assert main(["schema", "show", name]) == 0
        schema = json.loads(capsys.readouterr().out)
        assert schema["type"] == "object"
        assert "schema" in schema["properties"]


def test_schema_commands_do_not_import_the_release_workflow(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    def forbidden_import(_name: str) -> object:
        raise AssertionError("schema rendering must not load the release workflow")

    monkeypatch.setattr("fusion_fault_bench.cli.importlib.import_module", forbidden_import)

    assert main(["schema", "show", "replay-release-sidecar-index"]) == 0
    assert json.loads(capsys.readouterr().out)["type"] == "object"


def test_replay_release_cli_lazily_validates_without_echoing_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    private_release = tmp_path / "private-release"
    observed: list[Path] = []

    def validate_release_package(*, path: Path) -> str:
        observed.append(path)
        return "d" * 64

    def import_workflow(name: str) -> object:
        assert name == "fusion_fault_bench.replay_release_workflow"
        return SimpleNamespace(validate_release_package=validate_release_package)

    monkeypatch.setattr("fusion_fault_bench.cli.importlib.import_module", import_workflow)

    assert main(["replay", "release", "validate", str(private_release)]) == 0
    output = capsys.readouterr().out
    assert output == (f"valid ffb.m5-release-package/v1 release_package_sha256={'d' * 64}\n")
    assert str(tmp_path) not in output
    assert observed == [private_release]
