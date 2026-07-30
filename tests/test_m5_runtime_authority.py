from __future__ import annotations

import os
from pathlib import Path

import pytest

from fusion_fault_bench.replay_release_workflow import (
    ReplayReleaseWorkflowError,
    _authenticated_input_directory,
    _paths_overlap,
)


def test_runtime_input_directories_are_real_and_identity_bound(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    cache = tmp_path / "cache"
    dataset.mkdir()
    cache.mkdir(mode=0o700)
    cache.chmod(0o700)

    observed_dataset, dataset_identity = _authenticated_input_directory(
        os.fspath(dataset),
        label="dataset root",
        require_private=False,
    )
    observed_cache, cache_identity = _authenticated_input_directory(
        os.fspath(cache),
        label="cache root",
        require_private=True,
    )

    assert observed_dataset == dataset
    assert observed_cache == cache
    assert dataset_identity == (dataset.stat().st_dev, dataset.stat().st_ino)
    assert cache_identity == (cache.stat().st_dev, cache.stat().st_ino)


def test_runtime_input_rejects_nonprivate_redirected_and_unnormalized_paths(
    tmp_path: Path,
) -> None:
    public_cache = tmp_path / "public-cache"
    public_cache.mkdir(mode=0o755)
    public_cache.chmod(0o755)
    with pytest.raises(ReplayReleaseWorkflowError, match="safe real directory"):
        _authenticated_input_directory(
            os.fspath(public_cache),
            label="cache root",
            require_private=True,
        )

    target = tmp_path / "target"
    target.mkdir()
    redirected = tmp_path / "redirected"
    redirected.symlink_to(target, target_is_directory=True)
    with pytest.raises(ReplayReleaseWorkflowError, match="safe real directory"):
        _authenticated_input_directory(
            os.fspath(redirected),
            label="dataset root",
            require_private=False,
        )

    with pytest.raises(ReplayReleaseWorkflowError, match="normalized absolute"):
        _authenticated_input_directory(
            f"{target}/",
            label="dataset root",
            require_private=False,
        )


def test_runtime_dataset_cache_and_source_overlap_is_detectable(tmp_path: Path) -> None:
    child = tmp_path / "child"
    sibling = tmp_path.parent / "sibling"
    assert _paths_overlap(tmp_path, child)
    assert _paths_overlap(child, tmp_path)
    assert not _paths_overlap(child, sibling)
