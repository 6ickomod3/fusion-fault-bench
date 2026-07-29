from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from fusion_fault_bench.contracts.health_result_v1 import (
    HealthAggregateMetricV1,
    HealthFitSummaryV1,
)
from fusion_fault_bench.health_release import (
    AggregateQuantitativeClaimV1,
    FitQuantitativeClaimV1,
    HealthRunResourceEvidenceV1,
    ResourceQuantitativeClaimV1,
)
from fusion_fault_bench.provenance import CleanSourceSnapshot

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
m4_release = importlib.import_module("tools.m4_release")


def _aggregate(key: m4_release.AggregateKey) -> HealthAggregateMetricV1:
    condition_id, method, metric_name, window = key
    return HealthAggregateMetricV1(
        schema="ffb.health-aggregate/v1",
        condition_id=condition_id,
        method=method,
        metric_name=metric_name,
        window=window,
        unit="m^2",
        status="ok",
        estimate=1.0,
        interval_lower=0.5,
        interval_upper=1.5,
        sequence_count=200,
        bootstrap_replicates=2000,
        defined_bootstrap_replicates=2000,
    )


def _summary() -> HealthFitSummaryV1:
    return cast(
        HealthFitSummaryV1,
        SimpleNamespace(
            selected_candidate_index=27,
            selected_self_threshold=0.999,
            selected_cross_threshold=0.995,
        ),
    )


def _resources() -> tuple[HealthRunResourceEvidenceV1, ...]:
    return tuple(
        cast(
            HealthRunResourceEvidenceV1,
            SimpleNamespace(
                run_label=label,
                wall_time_seconds=10.0 + index,
                peak_rss_bytes=100_000_000 + index,
                cpu_model="Test CPU",
            ),
        )
        for index, label in enumerate(
            (
                "primary-fit",
                "repeat-fit",
                "primary-evaluation",
                "repeat-evaluation",
            )
        )
    )


def test_frozen_claim_projection_includes_all_predeclared_rows_fit_and_resources() -> None:
    aggregates = tuple(_aggregate(key) for key in m4_release._OUTCOME_CLAIM_KEYS)

    claims = m4_release.derive_quantitative_claims(
        fit_summary=_summary(),
        aggregates=aggregates,
        resources=_resources(),
    )

    aggregate_claims = tuple(
        claim for claim in claims if isinstance(claim, AggregateQuantitativeClaimV1)
    )
    fit_claims = tuple(claim for claim in claims if isinstance(claim, FitQuantitativeClaimV1))
    resource_claims = tuple(
        claim for claim in claims if isinstance(claim, ResourceQuantitativeClaimV1)
    )
    assert tuple(claim.aggregate for claim in aggregate_claims) == aggregates
    assert len(fit_claims) == 3
    assert len(resource_claims) == 8
    assert len({claim.claim_id for claim in claims}) == len(claims) == 29


def test_frozen_claim_projection_rejects_missing_or_duplicate_aggregate_keys() -> None:
    aggregates = tuple(_aggregate(key) for key in m4_release._OUTCOME_CLAIM_KEYS)
    with pytest.raises(m4_release.HealthReleaseDriverError, match="missing 1"):
        m4_release.derive_quantitative_claims(
            fit_summary=_summary(),
            aggregates=aggregates[:-1],
            resources=_resources(),
        )
    with pytest.raises(m4_release.HealthReleaseDriverError, match="duplicate keys"):
        m4_release.derive_quantitative_claims(
            fit_summary=_summary(),
            aggregates=(*aggregates, aggregates[0]),
            resources=_resources(),
        )


def _snapshot(tmp_path: Path) -> CleanSourceSnapshot:
    return CleanSourceSnapshot(
        source_root=tmp_path,
        git_revision="a" * 40,
        git_dir=tmp_path / ".git",
        git_common_dir=tmp_path / ".git",
        lockfile_sha256="b" * 64,
        package_version="0.1.0",
        manifest_relative_path="examples/health/m4-health-v1.json",
    )


def _run_artifact(
    snapshot: CleanSourceSnapshot,
    *,
    git_revision: str | None = None,
) -> object:
    return SimpleNamespace(
        run=SimpleNamespace(
            git_revision=snapshot.git_revision if git_revision is None else git_revision,
            lockfile_sha256=snapshot.lockfile_sha256,
            package_version=snapshot.package_version,
            source_dirty=False,
        )
    )


def test_release_requires_all_artifacts_from_current_common_source(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    current = tuple(_run_artifact(snapshot) for _ in range(4))
    m4_release._require_current_common_provenance(snapshot, current)

    stale = (*current[:3], _run_artifact(snapshot, git_revision="c" * 40))
    with pytest.raises(m4_release.HealthReleaseDriverError, match="current clean source"):
        m4_release._require_current_common_provenance(snapshot, stale)


def test_resource_logs_require_four_distinct_paths_and_files(
    tmp_path: Path,
) -> None:
    paths = tuple(tmp_path / f"run-{index}.txt" for index in range(4))
    for index, path in enumerate(paths):
        path.write_text(f"{index}.00 real\n{100 + index}  maximum resident set size\n")
    labels = tuple(f"run-{index}" for index in range(4))

    logs = m4_release._read_independent_resource_logs(dict(zip(labels, paths, strict=True)))

    assert tuple(log.value for log in logs.values()) == tuple(path.read_bytes() for path in paths)
    with pytest.raises(m4_release.HealthReleaseDriverError, match="distinct paths"):
        m4_release._read_independent_resource_logs(
            {label: paths[0] for label in labels},
        )


def test_generated_input_rejects_symlinked_parent(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "time.txt").write_text("1.00 real\n100 maximum resident set size\n")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "generated").symlink_to(external, target_is_directory=True)

    with pytest.raises(m4_release.HealthReleaseDriverError, match="symlink components"):
        m4_release._repository_path(
            Path("reports/generated/time.txt"),
            source_root=tmp_path,
            label="resource log",
            generated_only=True,
        )


@pytest.mark.parametrize("path", (Path("/tmp/release"), Path("../release")))
def test_release_validation_rejects_paths_outside_repository(
    path: Path,
) -> None:
    with pytest.raises(m4_release.HealthReleaseDriverError, match="repository-relative"):
        m4_release.validate_release(path)
