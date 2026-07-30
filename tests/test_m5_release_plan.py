from __future__ import annotations

from pathlib import Path


def test_frozen_workflow_creates_private_uv_cache_before_first_use() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = (root / "docs/m5-release-pipeline-plan.md").read_text(encoding="utf-8")
    export = "export UV_CACHE_DIR=<private-temporary-directory>/ffb-m5-uv-cache-<revision>"
    create = 'mkdir -m 700 "$UV_CACHE_DIR"'
    verify = "uv run --frozen --no-sync python tools/m5_release.py verify-software"

    export_at = plan.index(export)
    create_at = plan.index(create, export_at)
    first_exported_cache_run_at = plan.index("uv run --frozen --no-sync", export_at)

    assert plan.count(create) == 1
    assert export_at < create_at < first_exported_cache_run_at
    assert first_exported_cache_run_at == plan.index(verify, export_at)
