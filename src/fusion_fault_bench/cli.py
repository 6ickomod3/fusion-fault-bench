"""Small dependency-light command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from fusion_fault_bench import __version__
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.io import load_manifest
from fusion_fault_bench.contracts.manifest_v1alpha1 import manifest_json_schema
from fusion_fault_bench.contracts.result_v1alpha1 import (
    AggregateMetricRecordV1Alpha1,
    CrossoverRecordV1Alpha1,
    RunRecordV1Alpha1,
    metric_record_json_schema,
)

SchemaBuilder = Callable[[], dict[str, Any]]
SCHEMA_BUILDERS: dict[str, SchemaBuilder] = {
    "aggregate": lambda: AggregateMetricRecordV1Alpha1.model_json_schema(by_alias=True),
    "crossover": lambda: CrossoverRecordV1Alpha1.model_json_schema(by_alias=True),
    "manifest": manifest_json_schema,
    "metric": metric_record_json_schema,
    "run": lambda: RunRecordV1Alpha1.model_json_schema(by_alias=True),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffb",
        description="Fusion Fault Bench reproducibility and experiment tools.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    schema = commands.add_parser("schema", help="Inspect a versioned JSON Schema.")
    schema_commands = schema.add_subparsers(dest="schema_command", required=True)
    schema_show = schema_commands.add_parser("show", help="Print a JSON Schema.")
    schema_show.add_argument("record_type", choices=sorted(SCHEMA_BUILDERS))

    manifest = commands.add_parser("manifest", help="Validate or fingerprint a manifest.")
    manifest_commands = manifest.add_subparsers(dest="manifest_command", required=True)
    manifest_validate = manifest_commands.add_parser("validate", help="Validate a manifest.")
    manifest_validate.add_argument("path", type=Path)
    manifest_validate.add_argument("--json", action="store_true", dest="as_json")
    manifest_digest = manifest_commands.add_parser("digest", help="Print a manifest digest.")
    manifest_digest.add_argument("path", type=Path)
    return parser


def _schema_payload(record_type: str) -> dict[str, Any]:
    return SCHEMA_BUILDERS[record_type]()


def _run(args: argparse.Namespace) -> int:
    if args.command == "schema":
        print(json.dumps(_schema_payload(args.record_type), indent=2, sort_keys=True))
        return 0

    manifest = load_manifest(args.path)
    digest = sha256_digest(manifest)
    if args.manifest_command == "digest":
        print(digest)
        return 0
    if args.as_json:
        print(
            json.dumps(
                {
                    "manifest_sha256": digest,
                    "schema": manifest.schema_id,
                    "valid": True,
                },
                sort_keys=True,
            )
        )
    else:
        print(f"valid {manifest.schema_id} {digest}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI, returning a process-style exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (OSError, ValueError, ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
