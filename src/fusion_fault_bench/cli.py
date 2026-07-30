"""Small dependency-light command line interface."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Never, cast

from pydantic import ValidationError

from fusion_fault_bench import __version__
from fusion_fault_bench.artifacts import load_artifact
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.artifact_v1alpha1 import (
    AnalyticValidationV1Alpha1,
    PayloadIndexV1Alpha1,
    SuccessMarkerV1Alpha1,
)
from fusion_fault_bench.contracts.geometry_validation_v1 import (
    GeometryPayloadIndexV1,
    GeometryValidationManifestV1,
    GeometryValidationV1,
)
from fusion_fault_bench.contracts.health_artifact_v1 import (
    HealthFitReferenceV1,
    health_payload_index_json_schema,
)
from fusion_fault_bench.contracts.health_result_v1 import (
    HealthAggregateMetricV1,
    HealthFitSummaryV1,
    HealthSequenceContrastV1,
    HealthSequenceEventV1,
    HealthSequenceLossV1,
    HealthThresholdCandidateV1,
    HealthValidationV1,
)
from fusion_fault_bench.contracts.health_v1 import (
    health_benchmark_intent_json_schema,
)
from fusion_fault_bench.contracts.io import load_manifest
from fusion_fault_bench.contracts.manifest_v1alpha1 import manifest_json_schema
from fusion_fault_bench.contracts.matrix_v1 import experiment_matrix_json_schema
from fusion_fault_bench.contracts.procedural_artifact_v1 import (
    ProceduralPayloadIndexV1Alpha2,
)
from fusion_fault_bench.contracts.procedural_profile_v1 import (
    procedural_profile_json_schema,
)
from fusion_fault_bench.contracts.procedural_release_v1 import (
    m3_matrix_validation_json_schema,
    repeat_verification_json_schema,
)
from fusion_fault_bench.contracts.procedural_validation_v1 import (
    procedural_validation_json_schema,
)
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    ReplayClusterSensitivityV1,
    ReplayDescriptorAggregateV1,
    ReplayExecutionResourceEvidenceV1,
    ReplayFigureRecordV1,
    ReplayHealthAggregateV1,
    ReplayPersistentAggregateV1,
    ReplayPersistentCrossoverV1,
    ReplayProfileSummaryV1,
    ReplayReleaseIndexV1,
    ReplayRepeatVerificationV1,
    ReplaySourceMemberCommitmentV1,
    ReplayValidationV1,
)
from fusion_fault_bench.contracts.replay_release_v1 import (
    replay_release_contract_json_schemas,
)
from fusion_fault_bench.contracts.replay_v1 import ReplayExperimentIdentityV1
from fusion_fault_bench.contracts.result_v1alpha1 import (
    AggregateMetricRecordV1Alpha1,
    CrossoverRecordV1Alpha1,
    RunRecordV1Alpha1,
    metric_record_json_schema,
)
from fusion_fault_bench.geometry_artifacts import load_geometry_validation_artifact
from fusion_fault_bench.geometry_runner import run_geometry_validation
from fusion_fault_bench.health_artifacts import (
    load_health_evaluation_artifact,
    load_health_fit_artifact,
)
from fusion_fault_bench.health_runner import (
    evaluate_health_benchmark_artifact,
    fit_health_benchmark_artifact,
)
from fusion_fault_bench.procedural_artifacts import load_procedural_artifact
from fusion_fault_bench.procedural_runner import run_procedural_matrix
from fusion_fault_bench.replay_artifacts import load_replay_curated_artifact
from fusion_fault_bench.replay_runner import (
    curate_replay_verified_repeat,
    load_replay_local_artifact,
    run_replay_local,
    run_replay_repeat,
    verify_replay_repeat_artifacts,
)
from fusion_fault_bench.runner import run_analytic_experiment

SchemaBuilder = Callable[[], dict[str, Any]]


def _constant_schema_builder(schema: dict[str, Any]) -> SchemaBuilder:
    def build() -> dict[str, Any]:
        return schema

    return build


SCHEMA_BUILDERS: dict[str, SchemaBuilder] = {
    "aggregate": lambda: AggregateMetricRecordV1Alpha1.model_json_schema(by_alias=True),
    "analytic-validation": lambda: AnalyticValidationV1Alpha1.model_json_schema(by_alias=True),
    "crossover": lambda: CrossoverRecordV1Alpha1.model_json_schema(by_alias=True),
    "geometry-manifest": lambda: GeometryValidationManifestV1.model_json_schema(by_alias=True),
    "geometry-payload-index": lambda: GeometryPayloadIndexV1.model_json_schema(by_alias=True),
    "geometry-validation": lambda: GeometryValidationV1.model_json_schema(by_alias=True),
    "health-aggregate": lambda: HealthAggregateMetricV1.model_json_schema(by_alias=True),
    "health-fit-reference": lambda: HealthFitReferenceV1.model_json_schema(by_alias=True),
    "health-fit-summary": lambda: HealthFitSummaryV1.model_json_schema(by_alias=True),
    "health-intent": health_benchmark_intent_json_schema,
    "health-payload-index": health_payload_index_json_schema,
    "health-sequence-event": lambda: HealthSequenceEventV1.model_json_schema(by_alias=True),
    "health-sequence-contrast": lambda: HealthSequenceContrastV1.model_json_schema(by_alias=True),
    "health-sequence-loss": lambda: HealthSequenceLossV1.model_json_schema(by_alias=True),
    "health-threshold-candidate": lambda: HealthThresholdCandidateV1.model_json_schema(
        by_alias=True
    ),
    "health-validation": lambda: HealthValidationV1.model_json_schema(by_alias=True),
    "manifest": manifest_json_schema,
    "m3-matrix-validation": m3_matrix_validation_json_schema,
    "matrix": experiment_matrix_json_schema,
    "metric": metric_record_json_schema,
    "payload-index": lambda: PayloadIndexV1Alpha1.model_json_schema(by_alias=True),
    "procedural-payload-index": lambda: ProceduralPayloadIndexV1Alpha2.model_json_schema(
        by_alias=True
    ),
    "procedural-profile": procedural_profile_json_schema,
    "procedural-validation": procedural_validation_json_schema,
    "replay-cluster-sensitivity": lambda: ReplayClusterSensitivityV1.model_json_schema(
        by_alias=True
    ),
    "replay-descriptor-aggregate": lambda: ReplayDescriptorAggregateV1.model_json_schema(
        by_alias=True
    ),
    "replay-experiment-identity": lambda: ReplayExperimentIdentityV1.model_json_schema(
        by_alias=True
    ),
    "replay-execution-resource-evidence": lambda: (
        ReplayExecutionResourceEvidenceV1.model_json_schema(by_alias=True)
    ),
    "replay-figure-record": lambda: ReplayFigureRecordV1.model_json_schema(by_alias=True),
    "replay-health-aggregate": lambda: ReplayHealthAggregateV1.model_json_schema(by_alias=True),
    "replay-persistent-aggregate": lambda: ReplayPersistentAggregateV1.model_json_schema(
        by_alias=True
    ),
    "replay-persistent-crossover": lambda: ReplayPersistentCrossoverV1.model_json_schema(
        by_alias=True
    ),
    "replay-profile-summary": lambda: ReplayProfileSummaryV1.model_json_schema(by_alias=True),
    **{
        name: _constant_schema_builder(schema)
        for name, schema in replay_release_contract_json_schemas().items()
    },
    "replay-release-index": lambda: ReplayReleaseIndexV1.model_json_schema(by_alias=True),
    "replay-repeat-verification": lambda: ReplayRepeatVerificationV1.model_json_schema(
        by_alias=True
    ),
    "replay-source-commitment": lambda: ReplaySourceMemberCommitmentV1.model_json_schema(
        by_alias=True
    ),
    "replay-validation": lambda: ReplayValidationV1.model_json_schema(by_alias=True),
    "repeat-verification": repeat_verification_json_schema,
    "run": lambda: RunRecordV1Alpha1.model_json_schema(by_alias=True),
    "success": lambda: SuccessMarkerV1Alpha1.model_json_schema(by_alias=True),
}


class _CliArgumentError(ValueError):
    """An argument failure whose message never repeats user-provided values."""


class _SanitizedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise _CliArgumentError("invalid command arguments")


def _build_parser() -> argparse.ArgumentParser:
    parser = _SanitizedArgumentParser(
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

    run = commands.add_parser("run", help="Run a clean-source analytic experiment.")
    run.add_argument("path", type=Path, metavar="MANIFEST")
    run.add_argument("--output-dir", type=Path, required=True, metavar="DEST")

    bundle = commands.add_parser("bundle", help="Inspect a scientific result bundle.")
    bundle_commands = bundle.add_subparsers(dest="bundle_command", required=True)
    bundle_validate = bundle_commands.add_parser(
        "validate",
        help="Strictly validate a complete result bundle.",
    )
    bundle_validate.add_argument("path", type=Path, metavar="DEST")

    geometry = commands.add_parser(
        "geometry",
        help="Run or inspect the frozen M2 geometry validation.",
    )
    geometry_commands = geometry.add_subparsers(
        dest="geometry_command",
        required=True,
    )
    geometry_validate = geometry_commands.add_parser(
        "validate",
        help="Run the clean-source M2 geometry validation.",
    )
    geometry_validate.add_argument("path", type=Path, metavar="MANIFEST")
    geometry_validate.add_argument(
        "--dataset-root-env",
        required=True,
    )
    geometry_validate.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        metavar="DEST",
    )
    geometry_bundle = geometry_commands.add_parser(
        "bundle",
        help="Inspect an M2 geometry-validation artifact.",
    )
    geometry_bundle_commands = geometry_bundle.add_subparsers(
        dest="geometry_bundle_command",
        required=True,
    )
    geometry_bundle_validate = geometry_bundle_commands.add_parser(
        "validate",
        help="Strictly validate a complete M2 artifact.",
    )
    geometry_bundle_validate.add_argument("path", type=Path, metavar="DEST")

    procedural = commands.add_parser(
        "procedural",
        help="Run or inspect a frozen M3 procedural matrix.",
    )
    procedural_commands = procedural.add_subparsers(
        dest="procedural_command",
        required=True,
    )
    procedural_matrix = procedural_commands.add_parser(
        "matrix",
        help="Run a content-addressed procedural matrix.",
    )
    procedural_matrix_commands = procedural_matrix.add_subparsers(
        dest="procedural_matrix_command",
        required=True,
    )
    procedural_matrix_run = procedural_matrix_commands.add_parser(
        "run",
        help="Run every manifest in the frozen matrix.",
    )
    procedural_matrix_run.add_argument("path", type=Path, metavar="MATRIX")
    procedural_matrix_run.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        metavar="DEST",
    )
    procedural_bundle = procedural_commands.add_parser(
        "bundle",
        help="Inspect an M3 procedural artifact.",
    )
    procedural_bundle_commands = procedural_bundle.add_subparsers(
        dest="procedural_bundle_command",
        required=True,
    )
    procedural_bundle_validate = procedural_bundle_commands.add_parser(
        "validate",
        help="Strictly validate a complete M3 artifact.",
    )
    procedural_bundle_validate.add_argument("path", type=Path, metavar="DEST")

    health = commands.add_parser(
        "health",
        help="Fit, evaluate, or inspect the frozen M4 health benchmark.",
    )
    health_commands = health.add_subparsers(
        dest="health_command",
        required=True,
    )
    health_fit = health_commands.add_parser(
        "fit",
        help="Fit the frozen M4 calibration and threshold policy.",
    )
    health_fit.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        metavar="DEST",
    )
    health_evaluate = health_commands.add_parser(
        "evaluate",
        help="Apply one authenticated M4 fit to the frozen test matrix.",
    )
    health_evaluate.add_argument("path", type=Path, metavar="FIT")
    health_evaluate.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        metavar="DEST",
    )
    health_bundle = health_commands.add_parser(
        "bundle",
        help="Inspect an M4 health artifact.",
    )
    health_bundle_commands = health_bundle.add_subparsers(
        dest="health_bundle_kind",
        required=True,
    )
    health_fit_bundle = health_bundle_commands.add_parser(
        "fit",
        help="Inspect a fit artifact.",
    )
    health_fit_bundle_commands = health_fit_bundle.add_subparsers(
        dest="health_fit_bundle_command",
        required=True,
    )
    health_fit_bundle_validate = health_fit_bundle_commands.add_parser(
        "validate",
        help="Strictly validate a complete M4 fit artifact.",
    )
    health_fit_bundle_validate.add_argument("path", type=Path, metavar="DEST")
    health_evaluation_bundle = health_bundle_commands.add_parser(
        "evaluation",
        help="Inspect an evaluation artifact.",
    )
    health_evaluation_bundle_commands = health_evaluation_bundle.add_subparsers(
        dest="health_evaluation_bundle_command",
        required=True,
    )
    health_evaluation_bundle_validate = health_evaluation_bundle_commands.add_parser(
        "validate",
        help="Strictly validate a fit-bound M4 evaluation artifact.",
    )
    health_evaluation_bundle_validate.add_argument("path", type=Path, metavar="DEST")
    health_evaluation_bundle_validate.add_argument(
        "--fit-artifact",
        type=Path,
        required=True,
        metavar="FIT",
    )

    replay = commands.add_parser(
        "replay",
        help="Run or inspect the frozen M5 nuScenes-mini latent replay.",
    )
    replay_commands = replay.add_subparsers(
        dest="replay_command",
        required=True,
    )
    replay_run = replay_commands.add_parser(
        "run",
        help="Run one clean metadata-only M5 replay from NUSCENES_ROOT.",
    )
    replay_run.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        metavar="DEST",
    )
    replay_repeat = replay_commands.add_parser(
        "repeat",
        help=(
            "Run an in-process diagnostic repeat; release evidence uses two separately "
            "timed runs and verify-repeat."
        ),
    )
    replay_repeat.add_argument(
        "--primary-output-dir",
        type=Path,
        required=True,
        metavar="PRIMARY",
    )
    replay_repeat.add_argument(
        "--repeat-output-dir",
        type=Path,
        required=True,
        metavar="REPEAT",
    )
    replay_verify_repeat = replay_commands.add_parser(
        "verify-repeat",
        help="Verify two separately timed local artifacts and reconstruct curation input.",
    )
    replay_verify_repeat.add_argument(
        "--primary-artifact",
        type=Path,
        required=True,
        metavar="PRIMARY",
    )
    replay_verify_repeat.add_argument(
        "--repeat-artifact",
        type=Path,
        required=True,
        metavar="REPEAT",
    )
    replay_verify_repeat.add_argument(
        "--primary-time-log",
        type=Path,
        required=True,
        metavar="PRIMARY_TIME",
    )
    replay_verify_repeat.add_argument(
        "--repeat-time-log",
        type=Path,
        required=True,
        metavar="REPEAT_TIME",
    )
    replay_local = replay_commands.add_parser(
        "local",
        help="Inspect an ignored local M5 scientific-source artifact.",
    )
    replay_local_commands = replay_local.add_subparsers(
        dest="replay_local_command",
        required=True,
    )
    replay_local_validate = replay_local_commands.add_parser(
        "validate",
        help="Strictly validate an ignored local M5 artifact.",
    )
    replay_local_validate.add_argument("path", type=Path, metavar="DEST")
    replay_bundle = replay_commands.add_parser(
        "bundle",
        help="Inspect an aggregate-only curated M5 artifact.",
    )
    replay_bundle_commands = replay_bundle.add_subparsers(
        dest="replay_bundle_command",
        required=True,
    )
    replay_bundle_validate = replay_bundle_commands.add_parser(
        "validate",
        help="Strictly validate a complete curated M5 artifact.",
    )
    replay_bundle_validate.add_argument("path", type=Path, metavar="DEST")
    replay_release = replay_commands.add_parser(
        "release",
        help="Inspect a complete aggregate-only M5 release package.",
    )
    replay_release_commands = replay_release.add_subparsers(
        dest="replay_release_command",
        required=True,
    )
    replay_release_validate = replay_release_commands.add_parser(
        "validate",
        help="Strictly validate the complete 41-file M5 release package.",
    )
    replay_release_validate.add_argument("path", type=Path, metavar="RELEASE")
    return parser


def _schema_payload(record_type: str) -> dict[str, Any]:
    return SCHEMA_BUILDERS[record_type]()


def _run(args: argparse.Namespace) -> int:
    if args.command == "schema":
        print(json.dumps(_schema_payload(args.record_type), indent=2, sort_keys=True))
        return 0

    if args.command == "run":
        artifact = run_analytic_experiment(args.path, output_dir=args.output_dir)
        print(
            f"wrote {artifact.path} "
            f"artifact_sha256={artifact.artifact_sha256} "
            f"run_sha256={artifact.run_sha256}"
        )
        return 0

    if args.command == "bundle":
        artifact = load_artifact(args.path)
        print(
            f"valid {artifact.payload_index.artifact_contract} "
            f"artifact_sha256={artifact.artifact_sha256} "
            f"run_sha256={artifact.run_sha256}"
        )
        return 0

    if args.command == "geometry":
        if args.geometry_command == "validate":
            artifact = run_geometry_validation(
                args.path,
                dataset_root_env=args.dataset_root_env,
                output_dir=args.output_dir,
            )
            print(
                f"wrote {args.output_dir.as_posix()} "
                f"artifact_sha256={artifact.artifact_sha256} "
                f"run_sha256={artifact.run_sha256}"
            )
            return 0
        artifact = load_geometry_validation_artifact(args.path)
        print(
            f"valid {artifact.payload_index.artifact_contract} "
            f"artifact_sha256={artifact.artifact_sha256} "
            f"run_sha256={artifact.run_sha256}"
        )
        return 0

    if args.command == "procedural":
        if args.procedural_command == "matrix":
            artifacts = run_procedural_matrix(
                args.path,
                output_dir=args.output_dir,
            )
            for artifact in artifacts:
                logical_path = args.output_dir / artifact.path.name
                print(
                    f"wrote {logical_path.as_posix()} "
                    f"artifact_sha256={artifact.artifact_sha256} "
                    f"run_sha256={artifact.run_sha256}"
                )
            print(f"completed procedural matrix artifact_count={len(artifacts)}")
            return 0
        artifact = load_procedural_artifact(args.path)
        print(
            f"valid {artifact.payload_index.artifact_contract} "
            f"artifact_sha256={artifact.artifact_sha256} "
            f"run_sha256={artifact.run_sha256}"
        )
        return 0

    if args.command == "health":
        if args.health_command == "fit":
            artifact = fit_health_benchmark_artifact(output_dir=args.output_dir)
            print(
                f"wrote {args.output_dir.as_posix()} "
                f"artifact_sha256={artifact.artifact_sha256} "
                f"run_sha256={artifact.run_sha256}"
            )
            return 0
        if args.health_command == "evaluate":
            artifact = evaluate_health_benchmark_artifact(
                args.path,
                output_dir=args.output_dir,
            )
            print(
                f"wrote {args.output_dir.as_posix()} "
                f"artifact_sha256={artifact.artifact_sha256} "
                f"run_sha256={artifact.run_sha256}"
            )
            return 0
        if args.health_bundle_kind == "fit":
            artifact = load_health_fit_artifact(args.path)
        else:
            fit_artifact = load_health_fit_artifact(args.fit_artifact)
            artifact = load_health_evaluation_artifact(
                args.path,
                fit_artifact=fit_artifact,
            )
        print(
            f"valid {artifact.payload_index.artifact_contract} "
            f"artifact_sha256={artifact.artifact_sha256} "
            f"run_sha256={artifact.run_sha256}"
        )
        return 0

    if args.command == "replay":
        if args.replay_command == "run":
            execution = run_replay_local(args.output_dir)
            print(
                "wrote M5 replay local artifact "
                f"artifact_sha256={execution.artifact.artifact_sha256} "
                f"run_sha256={execution.artifact.run_sha256}"
            )
            return 0
        if args.replay_command == "repeat":
            execution = run_replay_repeat(
                primary_output_dir=args.primary_output_dir,
                repeat_output_dir=args.repeat_output_dir,
            )
            print(
                "verified M5 replay repeat "
                f"primary_artifact_sha256={execution.primary.artifact.artifact_sha256} "
                f"repeat_artifact_sha256={execution.repeat.artifact.artifact_sha256} "
                f"verification_sha256={sha256_digest(execution.repeat_verification)}"
            )
            return 0
        if args.replay_command == "verify-repeat":
            repeat = verify_replay_repeat_artifacts(
                primary_path=args.primary_artifact,
                repeat_path=args.repeat_artifact,
            )
            curated = curate_replay_verified_repeat(
                repeat,
                primary_log_path=args.primary_time_log,
                repeat_log_path=args.repeat_time_log,
            )
            print(
                "verified separately timed M5 replay artifacts "
                f"primary_artifact_sha256={repeat.primary.artifact_sha256} "
                f"repeat_artifact_sha256={repeat.repeat.artifact_sha256} "
                f"verification_sha256={sha256_digest(repeat.repeat_verification)} "
                f"profile_sha256={sha256_digest(curated.profile_summary)}"
            )
            return 0
        if args.replay_command == "local":
            artifact = load_replay_local_artifact(args.path)
            print(
                "valid ffb.replay-local-source/v1 "
                f"artifact_sha256={artifact.artifact_sha256} "
                f"run_sha256={artifact.run_sha256}"
            )
            return 0
        if args.replay_command == "release":
            workflow = importlib.import_module("fusion_fault_bench.replay_release_workflow")
            validator = cast(
                Callable[..., str],
                workflow.validate_release_package,
            )
            release_package_sha256 = validator(path=args.path)
            print(
                f"valid ffb.m5-release-package/v1 release_package_sha256={release_package_sha256}"
            )
            return 0
        artifact = load_replay_curated_artifact(args.path)
        print(
            f"valid {artifact.release_index.artifact_contract} "
            f"artifact_sha256={artifact.artifact_sha256} "
            f"run_sha256={artifact.run_sha256}"
        )
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
    try:
        args = parser.parse_args(argv)
        return _run(args)
    except (OSError, ValueError, ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
