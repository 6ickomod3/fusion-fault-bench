from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github/workflows/ci.yml"
_M5_RELEASE_GLOB = "reports/releases/m5-nuscenes-replay-v0.1.0/**"
_M5_RELEASE = "reports/releases/m5-nuscenes-replay-v0.1.0"


def test_installed_wheel_validates_tracked_m5_release_offline_after_install() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    install = "uv pip install --python /tmp/ffb-wheel-smoke/bin/python dist/*.whl"
    step = "- name: Validate tracked M5 release with installed wheel offline"
    command = "/tmp/ffb-wheel-smoke/bin/ffb replay release validate"

    assert workflow.count(step) == 1
    assert workflow.index(install) < workflow.index(step) < workflow.index(command)
    installed_step = workflow[workflow.index(step) :]
    assert f"hashFiles('{_M5_RELEASE_GLOB}') != ''" in installed_step
    assert "env -u NUSCENES_ROOT UV_OFFLINE=1" in installed_step
    assert command in installed_step
    assert _M5_RELEASE in installed_step


def test_any_partial_m5_release_tree_activates_all_validation_gates() -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    tree_condition = f"hashFiles('{_M5_RELEASE_GLOB}') != ''"
    index_only_condition = (
        "hashFiles('reports/releases/m5-nuscenes-replay-v0.1.0/release-sidecar-index.json') != ''"
    )

    assert workflow.count(tree_condition) == 3
    assert index_only_condition not in workflow
    assert "fetch-depth: 0" in workflow
    assert workflow.count("- name: Reject deletion of an established M5 release") == 1
    assert "git cat-file -e" in workflow
    assert "M5_BASE_SHA:" in workflow
