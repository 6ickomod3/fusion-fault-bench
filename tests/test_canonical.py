from __future__ import annotations

from fusion_fault_bench.canonical import canonical_json, sha256_digest
from fusion_fault_bench.contracts.io import load_manifest


def test_mapping_order_does_not_change_canonical_digest() -> None:
    first = {"b": 2, "a": {"d": 4, "c": 3}}
    second = {"a": {"c": 3, "d": 4}, "b": 2}

    assert canonical_json(first) == '{"a":{"c":3,"d":4},"b":2}'
    assert sha256_digest(first) == sha256_digest(second)


def test_example_manifest_has_stable_golden_digest(example_path) -> None:
    manifest = load_manifest(example_path)

    assert sha256_digest(manifest) == (
        "a603d090f77ad97f20f92b4ec685fe19624d0974dc7d1be4328e2cd3c963bd3e"
    )


def test_manifest_digest_changes_with_experimental_intent(manifest_data, validate_manifest) -> None:
    original = validate_manifest(manifest_data)
    manifest_data["rng"]["data_master_seed"] += 1
    changed = validate_manifest(manifest_data)

    assert sha256_digest(original) != sha256_digest(changed)
