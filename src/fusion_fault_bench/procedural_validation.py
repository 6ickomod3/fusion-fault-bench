"""Independent runtime validation for one M3 procedural experiment.

The validator deliberately reconstructs the frozen profile, random streams,
fault equations, affine loss moments, and dropout masks without importing the
production RNG helpers.  Production condition outputs are used only as the
implementation under test for the lower-level geometry and availability
oracles.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.bundle_v1alpha1 import (
    ConditionKey,
    expected_conditions,
    expected_sequence_ids,
)
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AdditivePositionBiasFault,
    AvailabilityControlManifest,
    CalibrationTranslationFault,
    CalibrationYawFault,
    CommonModeControlManifest,
    CommonModePositionBiasFault,
    CorrectlyReportedNoiseFault,
    FaultAxis,
    FaultFamily,
    GeometryCrossoverManifest,
    MethodId,
    ProceduralSource,
    SeverityDirection,
    SeverityUnit,
    TimestampOffsetFault,
    UnderreportedNoiseFault,
)
from fusion_fault_bench.contracts.procedural_artifact_v1 import (
    PROCEDURAL_MAX_BOOTSTRAP_CELLS,
    PROCEDURAL_MAX_BOOTSTRAP_REPLICATES,
    PROCEDURAL_MAX_SEQUENCE_COUNT,
    PROCEDURAL_MAX_SEQUENCE_ROWS,
)
from fusion_fault_bench.contracts.procedural_profile_v1 import (
    EdgeProceduralProfile,
    ProceduralProfileV1,
    SmokeProceduralProfile,
    profile_sequence_count,
)
from fusion_fault_bench.contracts.procedural_validation_v1 import (
    CommonModeValidationApplicableV1,
    CommonModeValidationNotApplicableV1,
    DeterministicModelChecksV1,
    DropoutValidationApplicableV1,
    DropoutValidationNotApplicableV1,
    EligibilityValidationV1,
    ExpectedLossCheckV1,
    IdentityComparisonDeferredV1,
    IdentityComparisonNotApplicableV1,
    MomentCheckV1,
    OracleDiscrepancyV1,
    ProceduralOracleChecksV1,
    ProceduralValidationV1,
    ProfileValidationChecksV1,
    ResourceValidationV1,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    LocalizationMetricRecord,
    MetricRecordV1Alpha1,
    SeverityCoordinate,
)
from fusion_fault_bench.experiments.procedural import (
    ProceduralConditionOutputs,
    generate_procedural_condition_outputs,
)
from fusion_fault_bench.reference.procedural import (
    AffineLossMoments,
    affine_signed_contrast_moments,
    affine_squared_loss_moments,
    covariance_six_se_bound,
    equal_sequence_population_moments,
    independent_dropout_mask,
    independent_fault_uniforms,
    mean_six_se_bound,
    reference_eligibility_mask,
    reference_latent_state,
    reference_truth,
    timestamp_displacement_xy,
    variance_six_se_bound,
    yaw_displacement_xy,
)
from fusion_fault_bench.scenarios.procedural import ProceduralSequence

type ProceduralManifest = (
    GeometryCrossoverManifest | CommonModeControlManifest | AvailabilityControlManifest
)
type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type MetricKey = tuple[str, ConditionKey, MethodId, str]
type StreamName = Literal["latent", "camera", "lidar"]

_ABS_TOLERANCE = 1e-12
_RNG_DOMAIN = b"fusion-fault-bench/rng/v1"
_ELIGIBILITY_DOMAIN = b"fusion-fault-bench/eligibility/v1\x00"
_ORDERED_ELIGIBILITY_DOMAIN = b"fusion-fault-bench/ordered-eligibility/v1\x00"
_UNIFORM_COMMITMENT_DOMAIN = b"fusion-fault-bench/dropout-uniforms/v1\x00"
_PRIMARY_METHODS: tuple[MethodId, ...] = (
    "camera-only",
    "lidar-only",
    "fixed-fusion",
    "fault-target-drop-policy",
    "performance-oracle",
)


@dataclass(frozen=True, slots=True)
class _AffineError:
    matrix: FloatArray
    bias: FloatArray


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=_ABS_TOLERANCE)


def _maximum_absolute(left: npt.ArrayLike, right: npt.ArrayLike) -> float:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != second.shape:
        return math.inf
    return float(np.max(np.abs(first - second), initial=0.0))


def _framed(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(4, "big") + encoded


def _independent_stream_seed(
    *,
    data_master_seed: int,
    stream_name: StreamName,
    sequence_id: str,
) -> int:
    """Duplicate the frozen framing without calling the production seed helper."""

    if type(data_master_seed) is not int or not 0 <= data_master_seed < 2**128:
        raise ValueError("data_master_seed must be an unsigned 128-bit integer")
    stream = stream_name.encode("utf-8")
    sequence = sequence_id.encode("utf-8")
    payload = b"".join(
        (
            _RNG_DOMAIN,
            b"\x00",
            data_master_seed.to_bytes(16, "big"),
            len(stream).to_bytes(4, "big"),
            stream,
            len(sequence).to_bytes(4, "big"),
            sequence,
        )
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _independent_latent_uniforms(
    *,
    data_master_seed: int,
    sequence_id: str,
    object_count: int,
) -> FloatArray:
    generator = np.random.Generator(
        np.random.PCG64DXSM(
            _independent_stream_seed(
                data_master_seed=data_master_seed,
                stream_name="latent",
                sequence_id=sequence_id,
            )
        )
    )
    return generator.random((object_count, 4), dtype=np.float64)


def _independent_standard_normals(
    *,
    data_master_seed: int,
    stream_name: Literal["camera", "lidar"],
    sequence_id: str,
    object_frame_count: int,
) -> FloatArray:
    generator = np.random.Generator(
        np.random.PCG64DXSM(
            _independent_stream_seed(
                data_master_seed=data_master_seed,
                stream_name=stream_name,
                sequence_id=sequence_id,
            )
        )
    )
    return generator.standard_normal((object_frame_count, 2), dtype=np.float64)


def _eligibility_sha256(sequence_id: str, identifiers: tuple[str, ...]) -> str:
    payload = bytearray(_ELIGIBILITY_DOMAIN)
    payload.extend(_framed(sequence_id))
    payload.extend(len(identifiers).to_bytes(8, "big"))
    for identifier in identifiers:
        payload.extend(_framed(identifier))
    return hashlib.sha256(payload).hexdigest()


def _ordered_eligibility_sha256(sequences: Sequence[ProceduralSequence]) -> str:
    payload = bytearray(_ORDERED_ELIGIBILITY_DOMAIN)
    payload.extend(len(sequences).to_bytes(8, "big"))
    for sequence in sequences:
        payload.extend(_framed(sequence.sequence_id))
        payload.extend(bytes.fromhex(sequence.eligibility_sha256))
    return hashlib.sha256(payload).hexdigest()


def _uniform_vectors_sha256(
    reference_uniforms: Sequence[tuple[str, FloatArray]],
) -> str:
    payload = bytearray(_UNIFORM_COMMITMENT_DOMAIN)
    payload.extend(len(reference_uniforms).to_bytes(8, "big"))
    for sequence_id, uniforms in reference_uniforms:
        payload.extend(_framed(sequence_id))
        payload.extend(uniforms.size.to_bytes(8, "big"))
        payload.extend(np.asarray(uniforms, dtype=">f8").tobytes(order="C"))
    return hashlib.sha256(payload).hexdigest()


def _severity(condition: ConditionKey) -> SeverityCoordinate:
    return SeverityCoordinate(
        index=condition.severity_index,
        magnitude=condition.magnitude,
        direction=cast(SeverityDirection, condition.direction),
        unit=cast(SeverityUnit, condition.unit),
    )


def _condition_for(record: MetricRecordV1Alpha1) -> ConditionKey:
    return ConditionKey(
        fault_family=record.fault_family,
        fault_axis=record.fault_axis,
        severity_index=record.severity.index,
        magnitude=record.severity.magnitude,
        direction=record.severity.direction,
        unit=record.severity.unit,
    )


def _metric_index(
    manifest: ProceduralManifest,
    *,
    run_id: str,
    metrics: Sequence[MetricRecordV1Alpha1],
) -> dict[MetricKey, MetricRecordV1Alpha1]:
    digest = sha256_digest(manifest)
    result: dict[MetricKey, MetricRecordV1Alpha1] = {}
    for record in metrics:
        if record.run_id != run_id or record.manifest_sha256 != digest:
            raise ValueError("sequence metric provenance does not match the validation run")
        key = (
            record.sequence_id,
            _condition_for(record),
            record.method_id,
            record.metric_name,
        )
        if key in result:
            raise ValueError("duplicate procedural sequence metric")
        result[key] = record

    if isinstance(manifest, AvailabilityControlManifest):
        metric_names = manifest.evaluation.metrics
    else:
        metric_names = ("matched-center-mse",)
    expected_keys = {
        (sequence_id, condition, method, metric_name)
        for sequence_id in expected_sequence_ids(manifest)
        for condition in expected_conditions(manifest)
        for method in manifest.methods
        for metric_name in metric_names
    }
    if set(result) != expected_keys:
        raise ValueError("procedural sequence metrics are incomplete or contain extra rows")
    return result


def _independent_split_family_support(
    profile: ProceduralProfileV1,
    *,
    data_master_seed: int,
) -> bool:
    """Check held-out split identities and supports without production generation."""

    if isinstance(profile, EdgeProceduralProfile):
        return profile.splits.test.layout_family == "fov-edge-radial-motion"
    if isinstance(profile, SmokeProceduralProfile):
        return profile.splits.test.layout_family == "small-front-roi-smoke"

    split_states: dict[str, tuple[FloatArray, FloatArray]] = {}
    split_ids: dict[str, frozenset[str]] = {}
    for split in ("train", "validation", "test"):
        initial_rows: list[FloatArray] = []
        velocity_rows: list[FloatArray] = []
        sequence_ids = tuple(
            f"procedural:{profile.profile_id}:{split}:{index:06d}" for index in range(200)
        )
        split_ids[split] = frozenset(sequence_ids)
        for sequence_id in sequence_ids:
            uniforms = _independent_latent_uniforms(
                data_master_seed=data_master_seed,
                sequence_id=sequence_id,
                object_count=profile.source.object_count,
            )
            state = reference_latent_state(
                profile.profile_id,
                split,
                uniforms,
            )
            initial_rows.append(state.initial_xy_m)
            velocity_rows.append(state.velocity_xy_mps)
        split_states[split] = (
            np.concatenate(initial_rows, axis=0),
            np.concatenate(velocity_rows, axis=0),
        )

    train_initial, train_velocity = split_states["train"]
    validation_initial, validation_velocity = split_states["validation"]
    test_initial, test_velocity = split_states["test"]
    identifiers_disjoint = (
        split_ids["train"].isdisjoint(split_ids["validation"])
        and split_ids["train"].isdisjoint(split_ids["test"])
        and split_ids["validation"].isdisjoint(split_ids["test"])
    )
    layout_families_distinct = (
        profile.splits.train.layout_family,
        profile.splits.validation.layout_family,
        profile.splits.test.layout_family,
    ) == (
        "near-two-lane-flow",
        "mid-lateral-crossing",
        "far-fast-approach",
    )
    initial_range_slices_disjoint = float(np.max(train_initial[:, 0])) < float(
        np.min(validation_initial[:, 0])
    ) and float(np.max(validation_initial[:, 0])) < float(np.min(test_initial[:, 0]))
    test_approach_speed_held_out = float(np.max(test_velocity[:, 0])) < min(
        float(np.min(train_velocity[:, 0])),
        float(np.min(validation_velocity[:, 0])),
    )
    validation_crossing_speed_held_out = float(np.min(np.abs(validation_velocity[:, 1]))) > max(
        float(np.max(np.abs(train_velocity[:, 1]))),
        float(np.max(np.abs(test_velocity[:, 1]))),
    )
    return all(
        (
            identifiers_disjoint,
            layout_families_distinct,
            initial_range_slices_disjoint,
            test_approach_speed_held_out,
            validation_crossing_speed_held_out,
        )
    )


def _independent_profile_checks(
    manifest: ProceduralManifest,
    profile: ProceduralProfileV1,
    sequences: Sequence[ProceduralSequence],
) -> tuple[ProfileValidationChecksV1, EligibilityValidationV1]:
    source = manifest.source
    if not isinstance(source, ProceduralSource):
        raise TypeError("M3 runtime validation requires a procedural source")

    profile_digest_valid = sha256_digest(profile) == source.profile_sha256
    profile_id_valid = profile.profile_id == source.profile_id
    split_count_valid = profile_sequence_count(profile, source.split) == source.sequence_count
    roi_valid = (
        profile.eligibility.frame == manifest.roi.frame
        and profile.eligibility.x_min_m == manifest.roi.x_min_m
        and profile.eligibility.x_max_m == manifest.roi.x_max_m
        and profile.eligibility.abs_y_max_m == manifest.roi.abs_y_max_m
        and profile.eligibility.camera_half_fov_rad == manifest.roi.camera_half_fov_rad
    )
    isotropic_yaw_compatible = (
        manifest.observations.camera.actual_std_xy_m[0]
        == manifest.observations.camera.actual_std_xy_m[1]
        and manifest.observations.camera.reported_std_xy_m[0]
        == manifest.observations.camera.reported_std_xy_m[1]
    )

    expected_ids = expected_sequence_ids(manifest)
    canonical_ordering_valid = (
        len(sequences) == source.sequence_count
        and tuple(sequence.sequence_id for sequence in sequences) == expected_ids
        and tuple(sequence.sequence_index for sequence in sequences)
        == tuple(range(source.sequence_count))
    )
    split_family_support_valid = canonical_ordering_valid and _independent_split_family_support(
        profile,
        data_master_seed=manifest.rng.data_master_seed,
    )
    independent_eligibility_valid = canonical_ordering_valid

    for sequence in sequences:
        expected_object_ids = tuple(
            f"object:{index:02d}" for index in range(profile.source.object_count)
        )
        expected_frames = np.arange(profile.source.frame_count, dtype=np.int64)
        expected_times = expected_frames.astype(np.float64) * profile.source.frame_period_s
        canonical_ordering_valid = canonical_ordering_valid and (
            sequence.profile_id == profile.profile_id
            and sequence.split == source.split
            and sequence.object_ids == expected_object_ids
            and np.array_equal(sequence.frame_indices, expected_frames)
            and np.array_equal(sequence.frame_times_s, expected_times)
        )

        uniforms = _independent_latent_uniforms(
            data_master_seed=manifest.rng.data_master_seed,
            sequence_id=sequence.sequence_id,
            object_count=profile.source.object_count,
        )
        state = reference_latent_state(profile.profile_id, source.split, uniforms)
        truth = reference_truth(
            state,
            frame_count=profile.source.frame_count,
            frame_period_s=profile.source.frame_period_s,
        )
        eligibility = reference_eligibility_mask(
            truth,
            x_min_m=manifest.roi.x_min_m,
            x_max_m=manifest.roi.x_max_m,
            abs_y_max_m=manifest.roi.abs_y_max_m,
            camera_half_fov_rad=manifest.roi.camera_half_fov_rad,
        )
        eligible_frames, eligible_objects = np.nonzero(eligibility)
        expected_eligible_truth = truth[eligible_frames, eligible_objects]
        expected_eligible_velocity = state.velocity_xy_mps[eligible_objects]
        identifiers = tuple(
            f"{int(frame):06d}:{expected_object_ids[int(obj)]}"
            for frame, obj in zip(eligible_frames, eligible_objects, strict=True)
        )
        split_family_support_valid = split_family_support_valid and (
            _maximum_absolute(sequence.initial_xy_m, state.initial_xy_m) <= _ABS_TOLERANCE
            and _maximum_absolute(sequence.velocity_xy_mps, state.velocity_xy_mps) <= _ABS_TOLERANCE
            and _maximum_absolute(sequence.truth_xy_m, truth) <= _ABS_TOLERANCE
        )
        independent_eligibility_valid = independent_eligibility_valid and (
            np.array_equal(sequence.eligibility_mask, eligibility)
            and np.array_equal(sequence.eligible_frame_indices, eligible_frames)
            and np.array_equal(sequence.eligible_object_indices, eligible_objects)
            and sequence.eligible_object_frame_ids == identifiers
            and sequence.eligibility_sha256
            == _eligibility_sha256(sequence.sequence_id, identifiers)
            and _maximum_absolute(
                sequence.eligible_truth_xy_m,
                expected_eligible_truth,
            )
            <= _ABS_TOLERANCE
            and _maximum_absolute(
                sequence.eligible_velocity_xy_mps,
                expected_eligible_velocity,
            )
            <= _ABS_TOLERANCE
        )
        canonical_ordering_valid = canonical_ordering_valid and (
            tuple(
                zip(
                    sequence.eligible_frame_indices.tolist(),
                    sequence.eligible_object_indices.tolist(),
                    strict=True,
                )
            )
            == tuple(sorted(zip(eligible_frames.tolist(), eligible_objects.tolist(), strict=True)))
        )

        count = sequence.eligible_object_frame_count
        expected_camera = _independent_standard_normals(
            data_master_seed=manifest.rng.data_master_seed,
            stream_name="camera",
            sequence_id=sequence.sequence_id,
            object_frame_count=count,
        )
        expected_lidar = _independent_standard_normals(
            data_master_seed=manifest.rng.data_master_seed,
            stream_name="lidar",
            sequence_id=sequence.sequence_id,
            object_frame_count=count,
        )
        canonical_ordering_valid = canonical_ordering_valid and (
            sequence.camera_standard_normal_xy.tobytes(order="C")
            == expected_camera.tobytes(order="C")
            and sequence.lidar_standard_normal_xy.tobytes(order="C")
            == expected_lidar.tobytes(order="C")
        )

    counts = tuple(sequence.eligible_object_frame_count for sequence in sequences)
    if not counts:
        raise ValueError("procedural validation requires at least one sequence")
    profile_checks = ProfileValidationChecksV1(
        schema_valid=profile.schema_id == "ffb.procedural-profile/v1",
        profile_id_valid=profile_id_valid,
        profile_digest_valid=profile_digest_valid,
        split_count_valid=split_count_valid,
        roi_valid=roi_valid,
        isotropic_yaw_compatible=isotropic_yaw_compatible,
        split_family_support_valid=split_family_support_valid,
        canonical_ordering_valid=canonical_ordering_valid,
        all_checks_passed=all(
            (
                profile.schema_id == "ffb.procedural-profile/v1",
                profile_id_valid,
                profile_digest_valid,
                split_count_valid,
                roi_valid,
                isotropic_yaw_compatible,
                split_family_support_valid,
                canonical_ordering_valid,
            )
        ),
    )
    eligibility_validation = EligibilityValidationV1(
        ordered_sequence_commitments_sha256=_ordered_eligibility_sha256(sequences),
        minimum_eligible_object_frame_count=min(counts),
        maximum_eligible_object_frame_count=max(counts),
        total_eligible_object_frame_count=sum(counts),
        eligibility_invariant=independent_eligibility_valid,
    )
    return profile_checks, eligibility_validation


def _oracle_geometry_manifest(
    manifest: ProceduralManifest,
    *,
    fault_sweep: dict[str, object],
) -> GeometryCrossoverManifest:
    """Construct a fixed lower-level fixture while retaining run observations."""

    return GeometryCrossoverManifest.model_validate_json(
        json.dumps(
            {
                "schema": "ffb.manifest/v1alpha1",
                "kind": "geometry-crossover",
                "experiment": "procedural-runtime-oracle",
                "rng": manifest.rng.model_dump(mode="json"),
                "source": manifest.source.model_dump(mode="json"),
                "roi": manifest.roi.model_dump(mode="json"),
                "observations": manifest.observations.model_dump(mode="json"),
                "fault_sweep": fault_sweep,
                "methods": _PRIMARY_METHODS,
                "evaluation": {
                    "mode": "primary-single-sensor-crossover",
                    "primary_loss": "matched-center-mse",
                    "loss_unit": "m^2",
                    "aggregation": "object-frame-mean-then-sequence-mean",
                    "primary_contrast": "fused-minus-healthy",
                    "bootstrap": manifest.evaluation.bootstrap.model_dump(mode="json"),
                    "crossover": {
                        "fit": "nondecreasing-isotonic-pava",
                        "fit_weights": "equal-severity",
                        "interpolation": "linear-first-zero",
                        "zero_tolerance_m2": 1e-12,
                        "no_crossing_handling": "right-censored-above-tested-maximum",
                        "status_rule": "two-sided-bootstrap-crossing-fraction",
                    },
                    "target_drop_identity_action": "fixed-fusion",
                    "performance_oracle_selection_unit": "sequence",
                    "performance_oracle_candidates": (
                        "camera-only",
                        "lidar-only",
                        "fixed-fusion",
                    ),
                    "oracle_recovery_denominator_tolerance_m2": 1e-12,
                },
            }
        )
    )


def _positive_condition(manifest: GeometryCrossoverManifest) -> ConditionKey:
    return next(
        condition
        for condition in expected_conditions(manifest)
        if condition.direction == "positive"
    )


def _condition_outputs(
    manifest: ProceduralManifest,
    condition: ConditionKey,
    *,
    profile: ProceduralProfileV1,
    truth_xy_m: npt.ArrayLike,
    velocity_xy_mps: npt.ArrayLike,
    camera_normal_xy: npt.ArrayLike,
    lidar_normal_xy: npt.ArrayLike,
    frame_indices: npt.ArrayLike,
    fault_uniforms: npt.ArrayLike,
) -> ProceduralConditionOutputs:
    extrinsic = profile.rig.camera_true_extrinsic
    return generate_procedural_condition_outputs(
        manifest,
        condition=condition,
        truth_xy_m=truth_xy_m,
        velocity_xy_mps=velocity_xy_mps,
        eligible_frame_indices=frame_indices,
        camera_standard_normal_xy=camera_normal_xy,
        lidar_standard_normal_xy=lidar_normal_xy,
        fault_uniform_by_frame=fault_uniforms,
        camera_true_translation_m=extrinsic.translation_m,
        camera_true_quaternion_wxyz=extrinsic.quaternion_wxyz,
    )


def _oracle_record(
    check_id: str, unit: Literal["m", "m^2"], discrepancy: float
) -> OracleDiscrepancyV1:
    return OracleDiscrepancyV1(
        check_id=check_id,
        unit=unit,
        maximum_absolute_discrepancy=discrepancy,
        tolerance=_ABS_TOLERANCE,
        passed=discrepancy <= _ABS_TOLERANCE,
    )


def _double_corrupted_camera_reconstruction(
    truth_xy_m: npt.ArrayLike,
    *,
    camera_true_translation_m: npt.ArrayLike,
    camera_true_quaternion_wxyz: npt.ArrayLike,
    ego_translation_fault_m: npt.ArrayLike,
) -> FloatArray:
    """Apply one corrupted extrinsic to both generation and reconstruction.

    This intentionally incorrect mutation should self-cancel to the true
    center, demonstrating why generation must retain the physical transform.
    The scalar quaternion implementation is local to this independent runtime
    validator and does not call production SE(3) utilities.
    """

    truth = np.asarray(truth_xy_m, dtype=np.float64)
    translation = np.asarray(camera_true_translation_m, dtype=np.float64)
    quaternion = np.asarray(camera_true_quaternion_wxyz, dtype=np.float64)
    fault = np.asarray(ego_translation_fault_m, dtype=np.float64)
    if (
        truth.ndim != 2
        or truth.shape[1] != 2
        or translation.shape != (3,)
        or quaternion.shape != (4,)
        or fault.shape != (3,)
        or not all(np.all(np.isfinite(value)) for value in (truth, translation, quaternion, fault))
    ):
        raise ValueError("double-corruption mutation inputs have invalid shape or values")
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("double-corruption mutation quaternion must be unit length")
    w, x, y, z = (float(value) for value in quaternion)
    rotation = np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    truth_xyz = np.column_stack((truth, np.zeros(truth.shape[0], dtype=np.float64)))
    reported_translation = translation + fault
    corrupted_camera = (truth_xyz - reported_translation) @ rotation
    reconstructed = corrupted_camera @ rotation.T + reported_translation
    return np.asarray(reconstructed[:, :2], dtype=np.float64)


def _build_oracle_checks(
    manifest: ProceduralManifest,
    profile: ProceduralProfileV1,
    sequences: Sequence[ProceduralSequence],
) -> ProceduralOracleChecksV1:
    sequence = sequences[0]
    truth = np.asarray(
        sequence.eligible_truth_xy_m[: min(16, sequence.eligible_object_frame_count)]
    )
    velocity = np.asarray(
        sequence.eligible_velocity_xy_mps[: min(16, sequence.eligible_object_frame_count)]
    )
    count = truth.shape[0]
    zeros = np.zeros((count, 2), dtype=np.float64)
    frames = np.zeros(count, dtype=np.int64)
    uniforms = np.ones(profile.source.frame_count, dtype=np.float64) * 0.5

    translation_manifest = _oracle_geometry_manifest(
        manifest,
        fault_sweep={
            "kind": "calibration-translation",
            "target": "camera",
            "axis": "x",
            "unit": "m",
            "injection_site": "calibration-metadata",
            "perturbation_frame": "ego",
            "direction_policy": "symmetric-paired",
            "persistence": "sequence",
            "magnitude_values_m": (0.0, 0.25),
        },
    )
    bias_manifest = _oracle_geometry_manifest(
        manifest,
        fault_sweep={
            "kind": "additive-position-bias",
            "target": "camera",
            "axis": "x",
            "unit": "m",
            "injection_site": "estimator-output",
            "direction_policy": "symmetric-paired",
            "persistence": "sequence",
            "magnitude_values_m": (0.0, 0.25),
        },
    )
    yaw_manifest = _oracle_geometry_manifest(
        manifest,
        fault_sweep={
            "kind": "calibration-yaw",
            "target": "camera",
            "axis": "yaw",
            "unit": "rad",
            "injection_site": "calibration-metadata",
            "perturbation_frame": "ego",
            "direction_policy": "symmetric-paired",
            "persistence": "sequence",
            "magnitude_values_rad": (0.0, 0.02),
        },
    )
    timestamp_manifest = _oracle_geometry_manifest(
        manifest,
        fault_sweep={
            "kind": "timestamp-offset",
            "target": "camera",
            "axis": "time",
            "unit": "s",
            "injection_site": "timestamp-metadata",
            "timestamp_convention": "reported-minus-true",
            "direction_policy": "symmetric-paired",
            "persistence": "sequence",
            "magnitude_values_s": (0.0, 0.2),
        },
    )

    identity = expected_conditions(translation_manifest)[0]
    identity_outputs = _condition_outputs(
        translation_manifest,
        identity,
        profile=profile,
        truth_xy_m=truth,
        velocity_xy_mps=velocity,
        camera_normal_xy=zeros,
        lidar_normal_xy=zeros,
        frame_indices=frames,
        fault_uniforms=uniforms,
    )
    identity_discrepancy = max(
        _maximum_absolute(identity_outputs.camera_value_xy_m, truth),
        _maximum_absolute(identity_outputs.lidar_value_xy_m, truth),
        _maximum_absolute(identity_outputs.fixed_fusion_value_xy_m, truth),
    )

    translation_condition = _positive_condition(translation_manifest)
    translation_outputs = _condition_outputs(
        translation_manifest,
        translation_condition,
        profile=profile,
        truth_xy_m=truth,
        velocity_xy_mps=velocity,
        camera_normal_xy=zeros,
        lidar_normal_xy=zeros,
        frame_indices=frames,
        fault_uniforms=uniforms,
    )
    translation_expected = truth + np.asarray((0.25, 0.0))
    translation_discrepancy = _maximum_absolute(
        translation_outputs.camera_value_xy_m,
        translation_expected,
    )

    bias_outputs = _condition_outputs(
        bias_manifest,
        _positive_condition(bias_manifest),
        profile=profile,
        truth_xy_m=truth,
        velocity_xy_mps=velocity,
        camera_normal_xy=zeros,
        lidar_normal_xy=zeros,
        frame_indices=frames,
        fault_uniforms=uniforms,
    )
    equivalence_discrepancy = _maximum_absolute(
        translation_outputs.camera_value_xy_m,
        bias_outputs.camera_value_xy_m,
    )
    translation_loss = np.mean(
        np.sum(np.square(translation_outputs.camera_value_xy_m - truth), axis=1)
    )
    bias_loss = np.mean(np.sum(np.square(bias_outputs.camera_value_xy_m - truth), axis=1))
    equivalence_loss_discrepancy = abs(float(translation_loss - bias_loss))

    yaw_condition = _positive_condition(yaw_manifest)
    yaw_outputs = _condition_outputs(
        yaw_manifest,
        yaw_condition,
        profile=profile,
        truth_xy_m=truth,
        velocity_xy_mps=velocity,
        camera_normal_xy=zeros,
        lidar_normal_xy=zeros,
        frame_indices=frames,
        fault_uniforms=uniforms,
    )
    yaw_expected = truth + np.asarray(
        [yaw_displacement_xy(point, yaw_condition.magnitude) for point in truth],
        dtype=np.float64,
    )
    yaw_discrepancy = _maximum_absolute(yaw_outputs.camera_value_xy_m, yaw_expected)

    timestamp_condition = _positive_condition(timestamp_manifest)
    timestamp_outputs = _condition_outputs(
        timestamp_manifest,
        timestamp_condition,
        profile=profile,
        truth_xy_m=truth,
        velocity_xy_mps=velocity,
        camera_normal_xy=zeros,
        lidar_normal_xy=zeros,
        frame_indices=frames,
        fault_uniforms=uniforms,
    )
    timing_expected = truth + np.asarray(
        [timestamp_displacement_xy(row, timestamp_condition.magnitude) for row in velocity],
        dtype=np.float64,
    )
    timestamp_discrepancy = _maximum_absolute(
        timestamp_outputs.camera_value_xy_m,
        timing_expected,
    )
    static_outputs = _condition_outputs(
        timestamp_manifest,
        timestamp_condition,
        profile=profile,
        truth_xy_m=truth,
        velocity_xy_mps=zeros,
        camera_normal_xy=zeros,
        lidar_normal_xy=zeros,
        frame_indices=frames,
        fault_uniforms=uniforms,
    )
    static_discrepancy = _maximum_absolute(static_outputs.camera_value_xy_m, truth)

    extrinsic = profile.rig.camera_true_extrinsic
    canceled_generation_and_reconstruction = _double_corrupted_camera_reconstruction(
        truth,
        camera_true_translation_m=extrinsic.translation_m,
        camera_true_quaternion_wxyz=extrinsic.quaternion_wxyz,
        ego_translation_fault_m=(0.25, 0.0, 0.0),
    )
    mutation_cancels_to_truth = (
        _maximum_absolute(canceled_generation_and_reconstruction, truth) <= _ABS_TOLERANCE
    )
    mutation_disagrees_with_physical_oracle = (
        _maximum_absolute(
            canceled_generation_and_reconstruction,
            translation_expected,
        )
        > _ABS_TOLERANCE
    )
    mutation_rejected = mutation_cancels_to_truth and mutation_disagrees_with_physical_oracle

    records = (
        _oracle_record("identity-center", "m", identity_discrepancy),
        _oracle_record("calibration-translation-center", "m", translation_discrepancy),
        _oracle_record(
            "translation-bias-equivalence-center",
            "m",
            equivalence_discrepancy,
        ),
        _oracle_record(
            "translation-bias-equivalence-sequence-loss",
            "m^2",
            equivalence_loss_discrepancy,
        ),
        _oracle_record("calibration-yaw-center", "m", yaw_discrepancy),
        _oracle_record("timestamp-alignment-center", "m", timestamp_discrepancy),
        _oracle_record("static-timestamp-center", "m", static_discrepancy),
    )
    return ProceduralOracleChecksV1(
        identity_center=records[0],
        calibration_translation_center=records[1],
        translation_bias_equivalence_center=records[2],
        translation_bias_equivalence_sequence_loss=records[3],
        calibration_yaw_center=records[4],
        timestamp_alignment_center=records[5],
        static_timestamp_center=records[6],
        fault_cancellation_mutation_rejected=mutation_rejected,
        all_checks_passed=mutation_rejected and all(record.passed for record in records),
    )


def _moment_record(
    *,
    check_id: str,
    statistic: Literal[
        "mean",
        "variance",
        "within-sensor-covariance",
        "camera-lidar-cross-covariance",
    ],
    sensor_a: Literal["camera", "lidar"],
    coordinate_a: Literal["x", "y"],
    sensor_b: Literal["camera", "lidar"] | None,
    coordinate_b: Literal["x", "y"] | None,
    sample_count: int,
    expectation: float,
    observed: float,
    bound: float,
) -> MomentCheckV1:
    discrepancy = abs(observed - expectation)
    return MomentCheckV1(
        check_id=check_id,
        statistic=statistic,
        sensor_a=sensor_a,
        coordinate_a=coordinate_a,
        sensor_b=sensor_b,
        coordinate_b=coordinate_b,
        sample_count=sample_count,
        ddof=1,
        expectation=expectation,
        observed_value=observed,
        six_standard_error_bound=bound,
        absolute_discrepancy=discrepancy,
        unit="m" if statistic == "mean" else "m^2",
        passed=discrepancy <= bound,
    )


def _build_moment_checks(
    manifest: ProceduralManifest,
    sequences: Sequence[ProceduralSequence],
) -> tuple[MomentCheckV1, ...]:
    camera_std = np.asarray(manifest.observations.camera.actual_std_xy_m, dtype=np.float64)
    lidar_std = np.asarray(manifest.observations.lidar.actual_std_xy_m, dtype=np.float64)
    camera = np.concatenate(
        [sequence.camera_standard_normal_xy * camera_std for sequence in sequences],
        axis=0,
    )
    lidar = np.concatenate(
        [sequence.lidar_standard_normal_xy * lidar_std for sequence in sequences],
        axis=0,
    )
    sample_count = camera.shape[0]
    if lidar.shape != camera.shape or sample_count < 2:
        raise ValueError("moment validation requires aligned camera and LiDAR draws")

    checks: list[MomentCheckV1] = []
    sensors = (("camera", camera, camera_std), ("lidar", lidar, lidar_std))
    for sensor_name, values, standard_deviations in sensors:
        for coordinate_index, coordinate_name in enumerate(("x", "y")):
            observed_mean = float(np.mean(values[:, coordinate_index]))
            checks.append(
                _moment_record(
                    check_id=f"{sensor_name}-{coordinate_name}-mean",
                    statistic="mean",
                    sensor_a=cast(Literal["camera", "lidar"], sensor_name),
                    coordinate_a=coordinate_name,
                    sensor_b=None,
                    coordinate_b=None,
                    sample_count=sample_count,
                    expectation=0.0,
                    observed=observed_mean,
                    bound=mean_six_se_bound(
                        standard_deviation=float(standard_deviations[coordinate_index]),
                        sample_count=sample_count,
                    ),
                )
            )
            expected_variance = float(standard_deviations[coordinate_index] ** 2)
            observed_variance = float(np.var(values[:, coordinate_index], ddof=1))
            checks.append(
                _moment_record(
                    check_id=f"{sensor_name}-{coordinate_name}-variance",
                    statistic="variance",
                    sensor_a=cast(Literal["camera", "lidar"], sensor_name),
                    coordinate_a=coordinate_name,
                    sensor_b=None,
                    coordinate_b=None,
                    sample_count=sample_count,
                    expectation=expected_variance,
                    observed=observed_variance,
                    bound=variance_six_se_bound(
                        variance=expected_variance,
                        sample_count=sample_count,
                    ),
                )
            )
        observed_covariance = float(np.cov(values[:, 0], values[:, 1], ddof=1)[0, 1])
        checks.append(
            _moment_record(
                check_id=f"{sensor_name}-xy-covariance",
                statistic="within-sensor-covariance",
                sensor_a=cast(Literal["camera", "lidar"], sensor_name),
                coordinate_a="x",
                sensor_b=cast(Literal["camera", "lidar"], sensor_name),
                coordinate_b="y",
                sample_count=sample_count,
                expectation=0.0,
                observed=observed_covariance,
                bound=covariance_six_se_bound(
                    first_standard_deviation=float(standard_deviations[0]),
                    second_standard_deviation=float(standard_deviations[1]),
                    sample_count=sample_count,
                ),
            )
        )

    for camera_index, camera_coordinate in enumerate(("x", "y")):
        for lidar_index, lidar_coordinate in enumerate(("x", "y")):
            observed_covariance = float(
                np.cov(camera[:, camera_index], lidar[:, lidar_index], ddof=1)[0, 1]
            )
            checks.append(
                _moment_record(
                    check_id=f"camera-{camera_coordinate}-lidar-{lidar_coordinate}-covariance",
                    statistic="camera-lidar-cross-covariance",
                    sensor_a="camera",
                    coordinate_a=camera_coordinate,
                    sensor_b="lidar",
                    coordinate_b=lidar_coordinate,
                    sample_count=sample_count,
                    expectation=0.0,
                    observed=observed_covariance,
                    bound=covariance_six_se_bound(
                        first_standard_deviation=float(camera_std[camera_index]),
                        second_standard_deviation=float(lidar_std[lidar_index]),
                        sample_count=sample_count,
                    ),
                )
            )
    return tuple(checks)


def _fault_sign(condition: ConditionKey) -> float:
    if condition.direction == "identity":
        return 0.0
    if condition.direction == "negative":
        return -1.0
    return 1.0


def _affine_errors(
    manifest: ProceduralManifest,
    condition: ConditionKey,
    *,
    truth_xy_m: FloatArray,
    velocity_xy_mps: FloatArray,
) -> dict[MethodId, tuple[_AffineError, ...]]:
    camera_std = np.asarray(manifest.observations.camera.actual_std_xy_m, dtype=np.float64)
    lidar_std = np.asarray(manifest.observations.lidar.actual_std_xy_m, dtype=np.float64)
    fault = manifest.fault_sweep
    if isinstance(fault, (CorrectlyReportedNoiseFault, UnderreportedNoiseFault)):
        if fault.target == "camera":
            camera_std = camera_std * condition.magnitude
        else:
            lidar_std = lidar_std * condition.magnitude

    camera_reported = np.square(
        np.asarray(manifest.observations.camera.reported_std_xy_m, dtype=np.float64)
    )
    lidar_reported = np.square(
        np.asarray(manifest.observations.lidar.reported_std_xy_m, dtype=np.float64)
    )
    if isinstance(fault, CorrectlyReportedNoiseFault):
        if fault.target == "camera":
            camera_reported = camera_reported * condition.magnitude**2
        else:
            lidar_reported = lidar_reported * condition.magnitude**2
    camera_weight = (1.0 / camera_reported) / ((1.0 / camera_reported) + (1.0 / lidar_reported))
    lidar_weight = 1.0 - camera_weight

    sign_magnitude = _fault_sign(condition) * condition.magnitude
    rows: dict[MethodId, list[_AffineError]] = {
        "camera-only": [],
        "lidar-only": [],
        "fixed-fusion": [],
    }
    for truth, velocity in zip(truth_xy_m, velocity_xy_mps, strict=True):
        camera_matrix = np.zeros((2, 4), dtype=np.float64)
        lidar_matrix = np.zeros((2, 4), dtype=np.float64)
        camera_matrix[0, 0] = camera_std[0]
        camera_matrix[1, 1] = camera_std[1]
        lidar_matrix[0, 2] = lidar_std[0]
        lidar_matrix[1, 3] = lidar_std[1]
        camera_bias = np.zeros(2, dtype=np.float64)
        lidar_bias = np.zeros(2, dtype=np.float64)

        if isinstance(fault, CalibrationYawFault):
            cosine = math.cos(sign_magnitude)
            sine = math.sin(sign_magnitude)
            rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
            camera_matrix = rotation @ camera_matrix
            camera_bias = rotation @ truth - truth
        elif isinstance(
            fault,
            (
                AdditivePositionBiasFault,
                CalibrationTranslationFault,
                CommonModePositionBiasFault,
            ),
        ):
            axis_index = 0 if fault.axis == "x" else 1
            if isinstance(fault, CommonModePositionBiasFault):
                camera_bias[axis_index] += sign_magnitude
                lidar_bias[axis_index] += sign_magnitude
            else:
                target_bias = camera_bias if fault.target == "camera" else lidar_bias
                target_bias[axis_index] += sign_magnitude
        elif isinstance(fault, TimestampOffsetFault):
            target_bias = camera_bias if fault.target == "camera" else lidar_bias
            target_bias -= sign_magnitude * velocity

        fusion_matrix = (
            np.diag(camera_weight) @ camera_matrix + np.diag(lidar_weight) @ lidar_matrix
        )
        fusion_bias = camera_weight * camera_bias + lidar_weight * lidar_bias
        rows["camera-only"].append(_AffineError(camera_matrix, camera_bias))
        rows["lidar-only"].append(_AffineError(lidar_matrix, lidar_bias))
        rows["fixed-fusion"].append(_AffineError(fusion_matrix, fusion_bias))

    if isinstance(manifest, GeometryCrossoverManifest):
        healthy: MethodId = (
            "lidar-only" if manifest.fault_sweep.target == "camera" else "camera-only"
        )
        source: MethodId = "fixed-fusion" if condition.severity_index == 0 else healthy
        rows["fault-target-drop-policy"] = list(rows[source])
    return {method: tuple(values) for method, values in rows.items()}


def _sample_loss(errors: Sequence[_AffineError], normals: FloatArray) -> float:
    values = np.asarray(
        [error.matrix @ normal + error.bias for error, normal in zip(errors, normals, strict=True)]
    )
    return float(np.mean(np.sum(np.square(values), axis=1)))


def _metric_localization_value(
    index: dict[MetricKey, MetricRecordV1Alpha1],
    *,
    sequence_id: str,
    condition: ConditionKey,
    method: MethodId,
    metric_name: str,
) -> float | None:
    record = index[(sequence_id, condition, method, metric_name)]
    if not isinstance(record, LocalizationMetricRecord):
        raise TypeError("localization validation requires a localization metric row")
    return record.value


def _expected_loss_checks(
    manifest: ProceduralManifest,
    *,
    sequences: Sequence[ProceduralSequence],
    index: dict[MetricKey, MetricRecordV1Alpha1],
) -> tuple[ExpectedLossCheckV1, ...]:
    if isinstance(manifest, AvailabilityControlManifest):
        return ()

    checks: list[ExpectedLossCheckV1] = []
    expected_by_key: dict[tuple[ConditionKey, MethodId, str], float] = {}
    conditions = expected_conditions(manifest)
    affine_methods: tuple[MethodId, ...] = tuple(
        cast(MethodId, method) for method in manifest.methods if method != "performance-oracle"
    )
    for condition in conditions:
        rows_by_method: dict[MethodId, list[tuple[AffineLossMoments, ...]]] = {
            method: [] for method in affine_methods
        }
        contrast_by_sequence: list[tuple[AffineLossMoments, ...]] = []
        for sequence in sequences:
            representations = _affine_errors(
                manifest,
                condition,
                truth_xy_m=sequence.eligible_truth_xy_m,
                velocity_xy_mps=sequence.eligible_velocity_xy_mps,
            )
            normals = np.column_stack(
                (
                    sequence.camera_standard_normal_xy,
                    sequence.lidar_standard_normal_xy,
                )
            )
            candidate_sample_losses: dict[MethodId, float] = {}
            for method in affine_methods:
                method_rows = representations[method]
                moments = tuple(
                    affine_squared_loss_moments(row.matrix, row.bias) for row in method_rows
                )
                rows_by_method[method].append(moments)
                sample_loss = _sample_loss(method_rows, normals)
                candidate_sample_losses[method] = sample_loss
                observed = _metric_localization_value(
                    index,
                    sequence_id=sequence.sequence_id,
                    condition=condition,
                    method=method,
                    metric_name="matched-center-mse",
                )
                if observed is None or not _close(observed, sample_loss):
                    raise ValueError(
                        "sequence localization row disagrees with independent reconstruction"
                    )

            if isinstance(manifest, GeometryCrossoverManifest):
                healthy: MethodId = (
                    "lidar-only" if manifest.fault_sweep.target == "camera" else "camera-only"
                )
                contrast_by_sequence.append(
                    tuple(
                        affine_signed_contrast_moments(
                            first.matrix,
                            first.bias,
                            second.matrix,
                            second.bias,
                        )
                        for first, second in zip(
                            representations["fixed-fusion"],
                            representations[healthy],
                            strict=True,
                        )
                    )
                )
                oracle_observed = _metric_localization_value(
                    index,
                    sequence_id=sequence.sequence_id,
                    condition=condition,
                    method="performance-oracle",
                    metric_name="matched-center-mse",
                )
                exact_oracle = min(
                    candidate_sample_losses["camera-only"],
                    candidate_sample_losses["lidar-only"],
                    candidate_sample_losses["fixed-fusion"],
                )
                if oracle_observed is None or not _close(oracle_observed, exact_oracle):
                    raise ValueError(
                        "performance-oracle row is not an exact complete-sequence candidate minimum"
                    )

        for method in affine_methods:
            population = equal_sequence_population_moments(rows_by_method[method])
            empirical_values = [
                _metric_localization_value(
                    index,
                    sequence_id=sequence.sequence_id,
                    condition=condition,
                    method=method,
                    metric_name="matched-center-mse",
                )
                for sequence in sequences
            ]
            if any(value is None for value in empirical_values):
                raise ValueError("non-availability loss cannot be undefined")
            empirical = math.fsum(cast(float, value) for value in empirical_values) / len(sequences)
            standardized = abs(empirical - population.expected_m2) / (population.standard_error_m2)
            check = ExpectedLossCheckV1(
                check_id=(f"loss-{condition.severity_index:02d}-{condition.direction}-{method}"),
                fault_family=cast(FaultFamily, condition.fault_family),
                fault_axis=cast(FaultAxis, condition.fault_axis),
                severity=_severity(condition),
                method_id=method,
                metric_name="matched-center-mse",
                expected_value_m2=population.expected_m2,
                empirical_value_m2=empirical,
                analytic_standard_error_m2=population.standard_error_m2,
                absolute_standardized_error=standardized,
                standard_error_multiplier=6.0,
                passed=standardized <= 6.0,
            )
            checks.append(check)
            expected_by_key[(condition, method, "matched-center-mse")] = population.expected_m2

        if isinstance(manifest, GeometryCrossoverManifest):
            contrast = equal_sequence_population_moments(contrast_by_sequence)
            healthy: MethodId = (
                "lidar-only" if manifest.fault_sweep.target == "camera" else "camera-only"
            )
            empirical_contrasts: list[float] = []
            for sequence in sequences:
                fixed = _metric_localization_value(
                    index,
                    sequence_id=sequence.sequence_id,
                    condition=condition,
                    method="fixed-fusion",
                    metric_name="matched-center-mse",
                )
                healthy_value = _metric_localization_value(
                    index,
                    sequence_id=sequence.sequence_id,
                    condition=condition,
                    method=healthy,
                    metric_name="matched-center-mse",
                )
                if fixed is None or healthy_value is None:
                    raise ValueError("signed contrast inputs cannot be undefined")
                empirical_contrasts.append(fixed - healthy_value)
            empirical = math.fsum(empirical_contrasts) / len(sequences)
            standardized = abs(empirical - contrast.expected_m2) / contrast.standard_error_m2
            checks.append(
                ExpectedLossCheckV1(
                    check_id=(f"contrast-{condition.severity_index:02d}-{condition.direction}"),
                    fault_family=cast(FaultFamily, condition.fault_family),
                    fault_axis=cast(FaultAxis, condition.fault_axis),
                    severity=_severity(condition),
                    method_id="fixed-fusion",
                    metric_name="fused-minus-healthy",
                    expected_value_m2=contrast.expected_m2,
                    empirical_value_m2=empirical,
                    analytic_standard_error_m2=contrast.standard_error_m2,
                    absolute_standardized_error=standardized,
                    standard_error_multiplier=6.0,
                    passed=standardized <= 6.0,
                )
            )
            expected_by_key[(condition, "fixed-fusion", "fused-minus-healthy")] = (
                contrast.expected_m2
            )

    _validate_expected_response(manifest, conditions, expected_by_key, sequences)
    return tuple(checks)


def _validate_reported_covariances(
    manifest: ProceduralManifest,
    *,
    profile: ProceduralProfileV1,
    sequence: ProceduralSequence,
) -> None:
    """Check the declared actual/reported noise distinction at every condition."""

    nominal_camera = np.square(
        np.asarray(manifest.observations.camera.reported_std_xy_m, dtype=np.float64)
    )
    nominal_lidar = np.square(
        np.asarray(manifest.observations.lidar.reported_std_xy_m, dtype=np.float64)
    )
    fault = manifest.fault_sweep
    for condition in expected_conditions(manifest):
        outputs = _condition_outputs(
            manifest,
            condition,
            profile=profile,
            truth_xy_m=sequence.eligible_truth_xy_m,
            velocity_xy_mps=sequence.eligible_velocity_xy_mps,
            camera_normal_xy=sequence.camera_standard_normal_xy,
            lidar_normal_xy=sequence.lidar_standard_normal_xy,
            frame_indices=sequence.eligible_frame_indices,
            fault_uniforms=sequence.fault_uniform_by_frame,
        )
        expected_camera = np.array(nominal_camera, copy=True)
        expected_lidar = np.array(nominal_lidar, copy=True)
        if isinstance(fault, CorrectlyReportedNoiseFault):
            target = expected_camera if fault.target == "camera" else expected_lidar
            target *= condition.magnitude**2
        # Underreported-noise deliberately leaves both reported arrays nominal.
        if (
            _maximum_absolute(
                outputs.camera_reported_variance_xy_m2,
                expected_camera,
            )
            > _ABS_TOLERANCE
            or _maximum_absolute(
                outputs.lidar_reported_variance_xy_m2,
                expected_lidar,
            )
            > _ABS_TOLERANCE
        ):
            raise ValueError(
                "reported covariance disagrees with the declared noise-reporting behavior"
            )


def _validate_expected_response(
    manifest: ProceduralManifest,
    conditions: Sequence[ConditionKey],
    expected: dict[tuple[ConditionKey, MethodId, str], float],
    sequences: Sequence[ProceduralSequence],
) -> None:
    fault = manifest.fault_sweep
    if isinstance(
        fault,
        (AdditivePositionBiasFault, CalibrationTranslationFault, TimestampOffsetFault),
    ):
        for magnitude in sorted({condition.magnitude for condition in conditions[1:]}):
            paired = [
                condition
                for condition in conditions
                if condition.magnitude == magnitude
                and condition.direction in {"negative", "positive"}
            ]
            if len(paired) != 2:
                raise ValueError("signed response grid is not paired")
            affine_methods: tuple[MethodId, ...] = tuple(
                cast(MethodId, method)
                for method in manifest.methods
                if method != "performance-oracle"
            )
            for method in affine_methods:
                values = [
                    expected[(condition, method, "matched-center-mse")] for condition in paired
                ]
                if not _close(values[0], values[1]):
                    raise ValueError("expected signed affine response is not symmetric")
            contrast_values = [
                expected[(condition, "fixed-fusion", "fused-minus-healthy")] for condition in paired
            ]
            if not _close(contrast_values[0], contrast_values[1]):
                raise ValueError("expected signed contrast response is not symmetric")

    if isinstance(fault, (CorrectlyReportedNoiseFault, UnderreportedNoiseFault)):
        curve = [
            expected[(condition, "fixed-fusion", "fused-minus-healthy")] for condition in conditions
        ]
        if any(later < earlier for earlier, later in pairwise(curve)):
            raise ValueError("expected noise contrast is not nondecreasing")
        if isinstance(fault, CorrectlyReportedNoiseFault) and any(value >= 0.0 for value in curve):
            raise ValueError("correctly reported noise contrast must approach zero from below")
        if isinstance(fault, UnderreportedNoiseFault) and any(
            later <= earlier for earlier, later in pairwise(curve)
        ):
            raise ValueError("underreported-noise contrast must be strictly increasing")

    if isinstance(fault, CalibrationYawFault):
        positive_conditions = [
            condition for condition in conditions if condition.direction in {"identity", "positive"}
        ]
        positive_curve = [
            expected[(condition, "fixed-fusion", "fused-minus-healthy")]
            for condition in positive_conditions
        ]
        if any(later + _ABS_TOLERANCE < earlier for earlier, later in pairwise(positive_curve)):
            raise ValueError("expected yaw contrast is not nondecreasing")
        for condition in conditions:
            if condition.direction == "identity":
                continue
            theta = _fault_sign(condition) * condition.magnitude
            for sequence in sequences:
                for point in sequence.eligible_truth_xy_m:
                    displacement = yaw_displacement_xy(point, theta)
                    radius = math.hypot(float(point[0]), float(point[1]))
                    expected_norm = 2.0 * radius * math.sin(abs(theta) / 2.0)
                    expected_squared = 2.0 * radius * radius * (1.0 - math.cos(theta))
                    if not _close(math.hypot(*displacement), expected_norm):
                        raise ValueError("yaw displacement norm oracle failed")
                    if not _close(
                        displacement[0] ** 2 + displacement[1] ** 2,
                        expected_squared,
                    ):
                        raise ValueError("yaw squared-displacement oracle failed")


def _identity_comparison(
    manifest: ProceduralManifest,
    *,
    sequences: Sequence[ProceduralSequence],
    index: dict[MetricKey, MetricRecordV1Alpha1],
) -> IdentityComparisonDeferredV1 | IdentityComparisonNotApplicableV1:
    if isinstance(manifest, CommonModeControlManifest):
        return IdentityComparisonNotApplicableV1(
            status="not-applicable",
            reason="edge-common-mode-has-no-comparable-profile-peer",
        )
    identity = expected_conditions(manifest)[0]
    discrepancies: list[float] = []
    metric_name = (
        "conditional-matched-center-mse"
        if isinstance(manifest, AvailabilityControlManifest)
        else "matched-center-mse"
    )
    for sequence in sequences:
        representations = _affine_errors(
            manifest,
            identity,
            truth_xy_m=sequence.eligible_truth_xy_m,
            velocity_xy_mps=sequence.eligible_velocity_xy_mps,
        )
        normals = np.column_stack(
            (
                sequence.camera_standard_normal_xy,
                sequence.lidar_standard_normal_xy,
            )
        )
        candidate_losses = {
            method: _sample_loss(rows, normals) for method, rows in representations.items()
        }
        expected_values: dict[MethodId, float] = {
            "camera-only": candidate_losses["camera-only"],
            "lidar-only": candidate_losses["lidar-only"],
            "fixed-fusion": candidate_losses["fixed-fusion"],
            "fault-target-drop-policy": candidate_losses["fixed-fusion"],
        }
        if isinstance(manifest, GeometryCrossoverManifest):
            expected_values["performance-oracle"] = min(
                candidate_losses["camera-only"],
                candidate_losses["lidar-only"],
                candidate_losses["fixed-fusion"],
            )
        for method in manifest.methods:
            observed = _metric_localization_value(
                index,
                sequence_id=sequence.sequence_id,
                condition=identity,
                method=method,
                metric_name=metric_name,
            )
            if observed is None:
                discrepancies.append(math.inf)
            else:
                discrepancies.append(abs(observed - expected_values[method]))
    maximum = max(discrepancies, default=math.inf)
    if maximum > _ABS_TOLERANCE:
        raise ValueError("identity rows disagree with independent reconstruction")
    return IdentityComparisonDeferredV1(
        status="deferred-to-matrix",
        reason="cross-manifest-identity-requires-complete-matrix",
    )


def _validate_metric_counts_and_healthy_invariance(
    manifest: ProceduralManifest,
    *,
    sequences: Sequence[ProceduralSequence],
    index: dict[MetricKey, MetricRecordV1Alpha1],
) -> bool:
    eligibility_invariant = True
    conditions = expected_conditions(manifest)
    for sequence in sequences:
        expected_count = sequence.eligible_object_frame_count
        for key, record in index.items():
            if key[0] != sequence.sequence_id:
                continue
            eligibility_invariant = eligibility_invariant and (
                record.eligible_object_frame_count == expected_count
            )

    if isinstance(manifest, CommonModeControlManifest):
        return eligibility_invariant
    healthy: MethodId = "lidar-only" if manifest.fault_sweep.target == "camera" else "camera-only"
    metric_names = (
        manifest.evaluation.metrics
        if isinstance(manifest, AvailabilityControlManifest)
        else ("matched-center-mse",)
    )
    identity = conditions[0]
    for sequence in sequences:
        for metric_name in metric_names:
            baseline = index[(sequence.sequence_id, identity, healthy, metric_name)]
            for condition in conditions[1:]:
                current = index[(sequence.sequence_id, condition, healthy, metric_name)]
                if type(current) is not type(baseline):
                    raise ValueError("healthy row type changed across severity")
                if (
                    current.status != baseline.status
                    or current.value != baseline.value
                    or current.eligible_object_frame_count != baseline.eligible_object_frame_count
                    or current.valid_object_frame_count != baseline.valid_object_frame_count
                ):
                    raise ValueError("healthy modality changed across fault severity")
    return eligibility_invariant


def _availability_validation(
    manifest: AvailabilityControlManifest,
    *,
    profile: ProceduralProfileV1,
    sequences: Sequence[ProceduralSequence],
    index: dict[MetricKey, MetricRecordV1Alpha1],
) -> DropoutValidationApplicableV1:
    reference_vectors: list[tuple[str, FloatArray]] = []
    comparison_count = 0
    maximum_discrepancy = 0
    frame_sharing = True
    nesting = True
    endpoint = True
    conditions = expected_conditions(manifest)
    target_method: MethodId = (
        "camera-only" if manifest.fault_sweep.target == "camera" else "lidar-only"
    )
    healthy_method: MethodId = (
        "lidar-only" if manifest.fault_sweep.target == "camera" else "camera-only"
    )

    for sequence in sequences:
        reference_uniforms = independent_fault_uniforms(
            data_master_seed=manifest.rng.data_master_seed,
            sequence_id=sequence.sequence_id,
            frame_count=profile.source.frame_count,
        )
        reference_vectors.append((sequence.sequence_id, reference_uniforms))
        maximum_discrepancy = max(
            maximum_discrepancy,
            int(
                reference_uniforms.tobytes(order="C")
                != sequence.fault_uniform_by_frame.tobytes(order="C")
            ),
        )
        previous = np.zeros(profile.source.frame_count, dtype=np.bool_)
        for condition in conditions:
            reference_dropped = independent_dropout_mask(
                reference_uniforms,
                condition.magnitude,
            )
            outputs = _condition_outputs(
                manifest,
                condition,
                profile=profile,
                truth_xy_m=sequence.eligible_truth_xy_m,
                velocity_xy_mps=sequence.eligible_velocity_xy_mps,
                camera_normal_xy=sequence.camera_standard_normal_xy,
                lidar_normal_xy=sequence.lidar_standard_normal_xy,
                frame_indices=sequence.eligible_frame_indices,
                fault_uniforms=sequence.fault_uniform_by_frame,
            )
            target_available = (
                outputs.camera_available
                if manifest.fault_sweep.target == "camera"
                else outputs.lidar_available
            )
            observed_dropped = ~target_available
            expected_row_mask = reference_dropped[sequence.eligible_frame_indices]
            comparison_count += 1
            maximum_discrepancy = max(
                maximum_discrepancy,
                int(not np.array_equal(observed_dropped, expected_row_mask)),
            )
            for frame_index in np.unique(sequence.eligible_frame_indices):
                values = observed_dropped[sequence.eligible_frame_indices == frame_index]
                frame_sharing = frame_sharing and bool(np.all(values == values[0]))
            nesting = nesting and bool(np.all(previous <= reference_dropped))
            previous = reference_dropped
            representations = _affine_errors(
                manifest,
                condition,
                truth_xy_m=sequence.eligible_truth_xy_m,
                velocity_xy_mps=sequence.eligible_velocity_xy_mps,
            )
            normals = np.column_stack(
                (
                    sequence.camera_standard_normal_xy,
                    sequence.lidar_standard_normal_xy,
                )
            )
            sample_errors = {
                method: np.asarray(
                    [
                        error.matrix @ normal + error.bias
                        for error, normal in zip(rows, normals, strict=True)
                    ]
                )
                for method, rows in representations.items()
            }
            masks: dict[MethodId, BoolArray] = {
                "camera-only": outputs.camera_available,
                "lidar-only": outputs.lidar_available,
                "fixed-fusion": outputs.fixed_fusion_available,
            }
            source_method: MethodId = (
                "fixed-fusion" if condition.severity_index == 0 else healthy_method
            )
            sample_errors["fault-target-drop-policy"] = sample_errors[source_method]
            masks["fault-target-drop-policy"] = masks[source_method]

            for method in manifest.methods:
                valid_mask = masks[method]
                valid_count = int(np.count_nonzero(valid_mask))
                coverage = valid_count / sequence.eligible_object_frame_count
                undefined = 1.0 - coverage
                conditional = (
                    float(np.mean(np.sum(np.square(sample_errors[method][valid_mask]), axis=1)))
                    if valid_count
                    else None
                )
                expected_rows: tuple[tuple[str, float | None], ...] = (
                    ("coverage", coverage),
                    ("conditional-matched-center-mse", conditional),
                    ("undefined-output-rate", undefined),
                )
                for metric_name, expected_value in expected_rows:
                    record = index[(sequence.sequence_id, condition, method, metric_name)]
                    if record.valid_object_frame_count != valid_count:
                        raise ValueError("dropout valid count disagrees with independent mask")
                    if expected_value is None:
                        if (
                            not isinstance(record, LocalizationMetricRecord)
                            or record.status != "undefined"
                            or record.value is not None
                        ):
                            raise ValueError("dropout undefined endpoint semantics are invalid")
                    elif not _close(cast(float, record.value), expected_value):
                        raise ValueError("dropout metric disagrees with independent mask")

            if condition.magnitude == 0.0:
                endpoint = endpoint and (
                    not np.any(reference_dropped)
                    and np.all(outputs.camera_available)
                    and np.all(outputs.lidar_available)
                    and np.all(outputs.fixed_fusion_available)
                )
                fusion = _metric_localization_value(
                    index,
                    sequence_id=sequence.sequence_id,
                    condition=condition,
                    method="fixed-fusion",
                    metric_name="conditional-matched-center-mse",
                )
                drop_policy = _metric_localization_value(
                    index,
                    sequence_id=sequence.sequence_id,
                    condition=condition,
                    method="fault-target-drop-policy",
                    metric_name="conditional-matched-center-mse",
                )
                endpoint = endpoint and fusion == drop_policy
            if condition.magnitude == 1.0:
                endpoint = endpoint and (
                    np.all(reference_dropped)
                    and not np.any(target_available)
                    and not np.any(outputs.fixed_fusion_available)
                )
                for method in (target_method, "fixed-fusion"):
                    record = index[
                        (
                            sequence.sequence_id,
                            condition,
                            method,
                            "conditional-matched-center-mse",
                        )
                    ]
                    endpoint = endpoint and record.status == "undefined"
                for method in (healthy_method, "fault-target-drop-policy"):
                    coverage = index[(sequence.sequence_id, condition, method, "coverage")]
                    endpoint = endpoint and coverage.value == 1.0

    return DropoutValidationApplicableV1(
        status="applicable",
        uniform_vectors_sha256=_uniform_vectors_sha256(reference_vectors),
        exact_mask_comparison_count=comparison_count,
        frame_sharing_passed=frame_sharing,
        nesting_passed=nesting,
        endpoint_behavior_passed=bool(endpoint),
        maximum_mask_discrepancy=maximum_discrepancy,
        all_checks_passed=bool(maximum_discrepancy == 0 and frame_sharing and nesting and endpoint),
    )


def _common_mode_validation(
    manifest: CommonModeControlManifest,
    *,
    profile: ProceduralProfileV1,
    sequences: Sequence[ProceduralSequence],
) -> CommonModeValidationApplicableV1:
    conditions = expected_conditions(manifest)
    identity = conditions[0]
    maximum = 0.0
    for sequence in sequences:
        base = _condition_outputs(
            manifest,
            identity,
            profile=profile,
            truth_xy_m=sequence.eligible_truth_xy_m,
            velocity_xy_mps=sequence.eligible_velocity_xy_mps,
            camera_normal_xy=sequence.camera_standard_normal_xy,
            lidar_normal_xy=sequence.lidar_standard_normal_xy,
            frame_indices=sequence.eligible_frame_indices,
            fault_uniforms=sequence.fault_uniform_by_frame,
        )
        base_disagreement = base.camera_value_xy_m - base.lidar_value_xy_m
        zeros = np.zeros_like(sequence.camera_standard_normal_xy)
        for condition in conditions:
            outputs = _condition_outputs(
                manifest,
                condition,
                profile=profile,
                truth_xy_m=sequence.eligible_truth_xy_m,
                velocity_xy_mps=sequence.eligible_velocity_xy_mps,
                camera_normal_xy=sequence.camera_standard_normal_xy,
                lidar_normal_xy=sequence.lidar_standard_normal_xy,
                frame_indices=sequence.eligible_frame_indices,
                fault_uniforms=sequence.fault_uniform_by_frame,
            )
            maximum = max(
                maximum,
                _maximum_absolute(
                    outputs.camera_value_xy_m - outputs.lidar_value_xy_m,
                    base_disagreement,
                ),
            )
            noiseless = _condition_outputs(
                manifest,
                condition,
                profile=profile,
                truth_xy_m=sequence.eligible_truth_xy_m,
                velocity_xy_mps=sequence.eligible_velocity_xy_mps,
                camera_normal_xy=zeros,
                lidar_normal_xy=zeros,
                frame_indices=sequence.eligible_frame_indices,
                fault_uniforms=sequence.fault_uniform_by_frame,
            )
            expected = np.array(sequence.eligible_truth_xy_m, copy=True)
            axis_index = 0 if manifest.fault_sweep.axis == "x" else 1
            expected[:, axis_index] += _fault_sign(condition) * condition.magnitude
            maximum = max(
                maximum,
                _maximum_absolute(noiseless.camera_value_xy_m, expected),
                _maximum_absolute(noiseless.lidar_value_xy_m, expected),
                _maximum_absolute(noiseless.fixed_fusion_value_xy_m, expected),
            )
    return CommonModeValidationApplicableV1(
        status="applicable",
        maximum_disagreement_discrepancy_m=maximum,
        tolerance_m=_ABS_TOLERANCE,
        passed=maximum <= _ABS_TOLERANCE,
    )


def _resource_validation(manifest: ProceduralManifest) -> ResourceValidationV1:
    source = manifest.source
    if not isinstance(source, ProceduralSource):
        raise TypeError("M3 runtime validation requires a procedural source")
    condition_count = len(expected_conditions(manifest))
    if isinstance(manifest, AvailabilityControlManifest):
        pair_count = len(manifest.methods) * len(manifest.evaluation.metrics)
    else:
        pair_count = len(manifest.methods)
    sequence_rows = source.sequence_count * condition_count * pair_count
    replicates = manifest.evaluation.bootstrap.replicates
    bootstrap_cells = source.sequence_count * replicates
    return ResourceValidationV1(
        implied_sequence_row_count=sequence_rows,
        sequence_row_cap=PROCEDURAL_MAX_SEQUENCE_ROWS,
        implied_bootstrap_cell_count=bootstrap_cells,
        bootstrap_cell_cap=PROCEDURAL_MAX_BOOTSTRAP_CELLS,
        sequence_count=source.sequence_count,
        sequence_count_cap=PROCEDURAL_MAX_SEQUENCE_COUNT,
        bootstrap_replicates=replicates,
        bootstrap_replicate_cap=PROCEDURAL_MAX_BOOTSTRAP_REPLICATES,
        sequence_rows_within_cap=sequence_rows <= PROCEDURAL_MAX_SEQUENCE_ROWS,
        bootstrap_cells_within_cap=bootstrap_cells <= PROCEDURAL_MAX_BOOTSTRAP_CELLS,
        sequence_count_within_cap=source.sequence_count <= PROCEDURAL_MAX_SEQUENCE_COUNT,
        bootstrap_replicates_within_cap=replicates <= PROCEDURAL_MAX_BOOTSTRAP_REPLICATES,
        all_checks_passed=all(
            (
                sequence_rows <= PROCEDURAL_MAX_SEQUENCE_ROWS,
                bootstrap_cells <= PROCEDURAL_MAX_BOOTSTRAP_CELLS,
                source.sequence_count <= PROCEDURAL_MAX_SEQUENCE_COUNT,
                replicates <= PROCEDURAL_MAX_BOOTSTRAP_REPLICATES,
            )
        ),
    )


def build_procedural_validation(
    manifest: ProceduralManifest,
    *,
    profile: ProceduralProfileV1,
    run_id: str,
    sequences: Sequence[ProceduralSequence],
    metrics: Sequence[MetricRecordV1Alpha1],
) -> ProceduralValidationV1:
    """Build typed, independently recomputed validation evidence for one run."""

    source = manifest.source
    if not isinstance(source, ProceduralSource):
        raise TypeError("M3 runtime validation requires a procedural source")
    sequence_tuple = tuple(sequences)
    metric_index = _metric_index(manifest, run_id=run_id, metrics=metrics)
    profile_checks, eligibility = _independent_profile_checks(
        manifest,
        profile,
        sequence_tuple,
    )
    eligibility_invariant = _validate_metric_counts_and_healthy_invariance(
        manifest,
        sequences=sequence_tuple,
        index=metric_index,
    )
    eligibility = eligibility.model_copy(
        update={
            "eligibility_invariant": (eligibility.eligibility_invariant and eligibility_invariant)
        }
    )
    oracle_checks = _build_oracle_checks(manifest, profile, sequence_tuple)
    moment_checks = _build_moment_checks(manifest, sequence_tuple)
    expected_loss_checks = _expected_loss_checks(
        manifest,
        sequences=sequence_tuple,
        index=metric_index,
    )
    _validate_reported_covariances(
        manifest,
        profile=profile,
        sequence=sequence_tuple[0],
    )
    identity = _identity_comparison(
        manifest,
        sequences=sequence_tuple,
        index=metric_index,
    )
    if isinstance(manifest, AvailabilityControlManifest):
        dropout = _availability_validation(
            manifest,
            profile=profile,
            sequences=sequence_tuple,
            index=metric_index,
        )
    else:
        dropout = DropoutValidationNotApplicableV1(status="not-applicable")
    if isinstance(manifest, CommonModeControlManifest):
        common_mode = _common_mode_validation(
            manifest,
            profile=profile,
            sequences=sequence_tuple,
        )
    else:
        common_mode = CommonModeValidationNotApplicableV1(status="not-applicable")
    resources = _resource_validation(manifest)
    deterministic_model_checks = DeterministicModelChecksV1(
        reported_covariance_behavior="passed",
        expected_curve_response=(
            "not-applicable" if isinstance(manifest, AvailabilityControlManifest) else "passed"
        ),
        complete_sequence_performance_oracle=(
            "passed" if isinstance(manifest, GeometryCrossoverManifest) else "not-applicable"
        ),
        identity_row_reconstruction=(
            "not-applicable" if isinstance(manifest, CommonModeControlManifest) else "passed"
        ),
        all_checks_passed=True,
    )

    component_passes = (
        profile_checks.all_checks_passed,
        eligibility.eligibility_invariant,
        oracle_checks.all_checks_passed,
        all(check.passed for check in moment_checks),
        all(check.passed for check in expected_loss_checks),
        dropout.status == "not-applicable" or dropout.all_checks_passed,
        identity.status in {"not-applicable", "deferred-to-matrix"},
        common_mode.status == "not-applicable" or common_mode.passed,
        resources.all_checks_passed,
    )
    return ProceduralValidationV1(
        schema="ffb.procedural-validation/v1",
        run_id=run_id,
        manifest_sha256=sha256_digest(manifest),
        profile_id=profile.profile_id,
        profile_sha256=sha256_digest(profile),
        split=source.split,
        sequence_count=source.sequence_count,
        frame_count=profile.source.frame_count,
        object_count=profile.source.object_count,
        total_eligible_object_frame_count=eligibility.total_eligible_object_frame_count,
        profile_checks=profile_checks,
        eligibility=eligibility,
        oracle_checks=oracle_checks,
        moment_checks=moment_checks,
        expected_loss_checks=expected_loss_checks,
        dropout_validation=dropout,
        identity_comparison=identity,
        common_mode_validation=common_mode,
        deterministic_model_checks=deterministic_model_checks,
        resources=resources,
        all_checks_passed=all(component_passes),
    )
