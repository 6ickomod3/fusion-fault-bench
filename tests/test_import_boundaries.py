from __future__ import annotations

import subprocess
import sys


def test_base_import_does_not_load_scientific_or_dataset_packages() -> None:
    script = """
import sys
import fusion_fault_bench
forbidden = {"matplotlib", "numpy", "nuscenes", "scipy", "sklearn", "torch"}
assert forbidden.isdisjoint(sys.modules)
print(fusion_fault_bench.__version__)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "0.1.0"
