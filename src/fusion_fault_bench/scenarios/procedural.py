"""Seeded constant-velocity M3 scenes with frozen pre-fault eligibility."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.contracts.procedural_profile_v1 import (
    EdgeProceduralProfile,
    MainProceduralProfile,
    ProceduralProfileV1,
    SplitId,
    profile_sequence_count,
)
from fusion_fault_bench.geometry._arrays import immutable_float64_copy
from fusion_fault_bench.geometry.roi import CommonRoi, procedural_center_eligible
from fusion_fault_bench.rng import (
    draw_fault_uniforms,
    draw_latent_uniforms,
    draw_standard_normal_xy,
)

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type BoolArray = npt.NDArray[np.bool_]

_ELIGIBILITY_DOMAIN = b"fusion-fault-bench/eligibility/v1\x00"


def _immutable_int64(value: npt.ArrayLike) -> IntArray:
    array = np.asarray(value, dtype=np.int64)
    return np.frombuffer(array.tobytes(order="C"), dtype=np.int64).reshape(array.shape)


def _immutable_bool(value: npt.ArrayLike) -> BoolArray:
    array = np.asarray(value, dtype=np.bool_)
    return np.frombuffer(array.tobytes(order="C"), dtype=np.bool_).reshape(array.shape)


def _field(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(4, "big") + encoded


def _eligibility_commitment(sequence_id: str, identifiers: tuple[str, ...]) -> str:
    payload = bytearray(_ELIGIBILITY_DOMAIN)
    payload.extend(_field(sequence_id))
    payload.extend(len(identifiers).to_bytes(8, "big"))
    for identifier in identifiers:
        payload.extend(_field(identifier))
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class ProceduralSequence:
    """One immutable complete sequence and its paired base random draws."""

    sequence_id: str
    profile_id: str
    split: SplitId
    sequence_index: int
    object_ids: tuple[str, ...]
    frame_indices: IntArray
    frame_times_s: FloatArray
    initial_xy_m: FloatArray
    velocity_xy_mps: FloatArray
    truth_xy_m: FloatArray
    eligibility_mask: BoolArray
    eligible_frame_indices: IntArray
    eligible_object_indices: IntArray
    eligible_object_frame_ids: tuple[str, ...]
    eligible_truth_xy_m: FloatArray
    eligible_velocity_xy_mps: FloatArray
    camera_standard_normal_xy: FloatArray
    lidar_standard_normal_xy: FloatArray
    fault_uniform_by_frame: FloatArray
    eligibility_sha256: str

    @property
    def frame_count(self) -> int:
        return int(self.frame_indices.size)

    @property
    def object_count(self) -> int:
        return len(self.object_ids)

    @property
    def eligible_object_frame_count(self) -> int:
        return len(self.eligible_object_frame_ids)


def _main_state(
    profile: MainProceduralProfile,
    *,
    split: SplitId,
    uniforms: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    del profile
    object_indices = np.arange(6, dtype=np.int64)
    if split == "train":
        initial_x = 10.0 + 18.0 * uniforms[:, 0]
        lane = np.where(object_indices % 2 == 0, -3.5, 3.5)
        initial_y = lane + 0.25 * (2.0 * uniforms[:, 1] - 1.0)
        velocity_x = -1.0 + 2.0 * uniforms[:, 2]
        velocity_y = 0.1 * (2.0 * uniforms[:, 3] - 1.0)
    elif split == "validation":
        side = np.where(object_indices % 2 == 0, -1.0, 1.0)
        initial_x = 30.0 + 10.0 * uniforms[:, 0]
        initial_y = side * (5.0 + 3.0 * uniforms[:, 1])
        velocity_x = -1.0 + 2.0 * uniforms[:, 2]
        velocity_y = -side * (1.5 + 1.5 * uniforms[:, 3])
    elif split == "test":
        lateral_centers = np.asarray([-7.0, -4.0, -1.0, 1.0, 4.0, 7.0])
        initial_x = 44.0 + 12.0 * uniforms[:, 0]
        initial_y = lateral_centers + 0.25 * (2.0 * uniforms[:, 1] - 1.0)
        velocity_x = -(3.0 + 2.0 * uniforms[:, 2])
        velocity_y = 0.2 * (2.0 * uniforms[:, 3] - 1.0)
    else:
        raise ValueError("unknown main-profile split")
    return (
        immutable_float64_copy(np.column_stack((initial_x, initial_y))),
        immutable_float64_copy(np.column_stack((velocity_x, velocity_y))),
    )


def _edge_state(uniforms: FloatArray) -> tuple[FloatArray, FloatArray]:
    object_indices = np.arange(4, dtype=np.int64)
    side = np.where(object_indices % 2 == 0, 1.0, -1.0)
    radius = 20.0 + 20.0 * uniforms[:, 0]
    bearing = side * (0.7 - (0.005 + 0.015 * uniforms[:, 1]))
    direction = np.column_stack((np.cos(bearing), np.sin(bearing)))
    speed = -0.5 + uniforms[:, 2]
    return (
        immutable_float64_copy(radius[:, np.newaxis] * direction),
        immutable_float64_copy(speed[:, np.newaxis] * direction),
    )


def _smoke_state(uniforms: FloatArray) -> tuple[FloatArray, FloatArray]:
    lateral_centers = np.asarray([-2.0, 0.0, 2.0])
    initial = np.column_stack(
        (
            10.0 + 10.0 * uniforms[:, 0],
            lateral_centers + 0.1 * (2.0 * uniforms[:, 1] - 1.0),
        )
    )
    velocity = np.column_stack(
        (
            -0.5 + uniforms[:, 2],
            0.05 * (2.0 * uniforms[:, 3] - 1.0),
        )
    )
    return immutable_float64_copy(initial), immutable_float64_copy(velocity)


def _latent_state(
    profile: ProceduralProfileV1,
    *,
    split: SplitId,
    uniforms: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    if isinstance(profile, MainProceduralProfile):
        return _main_state(profile, split=split, uniforms=uniforms)
    if split != "test":
        raise ValueError("this profile defines only the test split")
    if isinstance(profile, EdgeProceduralProfile):
        return _edge_state(uniforms)
    return _smoke_state(uniforms)


def generate_procedural_sequence(
    profile: ProceduralProfileV1,
    *,
    split: SplitId,
    sequence_index: int,
    data_master_seed: int,
) -> ProceduralSequence:
    """Generate exactly one paired sequence from immutable manifest intent."""

    split_count = profile_sequence_count(profile, split)
    if type(sequence_index) is not int or not 0 <= sequence_index < split_count:
        raise ValueError("sequence_index is outside the declared split")
    sequence_id = f"procedural:{profile.profile_id}:{split}:{sequence_index:06d}"
    source = profile.source
    uniforms = draw_latent_uniforms(
        data_master_seed=data_master_seed,
        sequence_id=sequence_id,
        object_count=source.object_count,
    )
    initial, velocity = _latent_state(profile, split=split, uniforms=uniforms)
    frame_indices = np.arange(source.frame_count, dtype=np.int64)
    frame_times = frame_indices.astype(np.float64) * source.frame_period_s
    truth = (
        initial[np.newaxis, :, :]
        + frame_times[:, np.newaxis, np.newaxis] * velocity[np.newaxis, :, :]
    )
    roi = CommonRoi(
        x_min_m=profile.eligibility.x_min_m,
        x_max_m=profile.eligibility.x_max_m,
        abs_y_max_m=profile.eligibility.abs_y_max_m,
    )
    eligibility = np.zeros((source.frame_count, source.object_count), dtype=np.bool_)
    for frame_index in range(source.frame_count):
        for object_index in range(source.object_count):
            eligibility[frame_index, object_index] = procedural_center_eligible(
                center_ego_m=truth[frame_index, object_index],
                roi=roi,
                camera_half_fov_rad=profile.eligibility.camera_half_fov_rad,
                lidar_support_available=profile.eligibility.lidar_support_available,
            )
    eligible_frame_indices, eligible_object_indices = np.nonzero(eligibility)
    if eligible_frame_indices.size == 0:
        raise ValueError("procedural sequence has no pre-fault eligible object-frame")
    object_ids = tuple(f"object:{index:02d}" for index in range(source.object_count))
    if object_ids != tuple(sorted(object_ids, key=lambda value: value.encode("utf-8"))):
        raise AssertionError("profile object IDs are not in UTF-8 byte order")
    identifiers = tuple(
        f"{int(frame_index):06d}:{object_ids[int(object_index)]}"
        for frame_index, object_index in zip(
            eligible_frame_indices,
            eligible_object_indices,
            strict=True,
        )
    )
    eligible_truth = truth[eligible_frame_indices, eligible_object_indices]
    eligible_velocity = velocity[eligible_object_indices]
    eligible_count = len(identifiers)
    camera_normal = draw_standard_normal_xy(
        data_master_seed=data_master_seed,
        stream_name="camera",
        sequence_id=sequence_id,
        object_frame_count=eligible_count,
    )
    lidar_normal = draw_standard_normal_xy(
        data_master_seed=data_master_seed,
        stream_name="lidar",
        sequence_id=sequence_id,
        object_frame_count=eligible_count,
    )
    fault_uniforms = draw_fault_uniforms(
        data_master_seed=data_master_seed,
        sequence_id=sequence_id,
        frame_count=source.frame_count,
    )
    return ProceduralSequence(
        sequence_id=sequence_id,
        profile_id=profile.profile_id,
        split=split,
        sequence_index=sequence_index,
        object_ids=object_ids,
        frame_indices=_immutable_int64(frame_indices),
        frame_times_s=immutable_float64_copy(frame_times),
        initial_xy_m=initial,
        velocity_xy_mps=velocity,
        truth_xy_m=immutable_float64_copy(truth),
        eligibility_mask=_immutable_bool(eligibility),
        eligible_frame_indices=_immutable_int64(eligible_frame_indices),
        eligible_object_indices=_immutable_int64(eligible_object_indices),
        eligible_object_frame_ids=identifiers,
        eligible_truth_xy_m=immutable_float64_copy(eligible_truth),
        eligible_velocity_xy_mps=immutable_float64_copy(eligible_velocity),
        camera_standard_normal_xy=immutable_float64_copy(camera_normal),
        lidar_standard_normal_xy=immutable_float64_copy(lidar_normal),
        fault_uniform_by_frame=immutable_float64_copy(fault_uniforms),
        eligibility_sha256=_eligibility_commitment(sequence_id, identifiers),
    )


def generate_procedural_sequences(
    profile: ProceduralProfileV1,
    *,
    split: SplitId,
    sequence_count: int,
    data_master_seed: int,
) -> tuple[ProceduralSequence, ...]:
    """Generate the declared prefix of one split without retries or exclusions."""

    declared_count = profile_sequence_count(profile, split)
    if type(sequence_count) is not int or not 2 <= sequence_count <= declared_count:
        raise ValueError("sequence_count must be in [2, declared split count]")
    if not math.isfinite(float(data_master_seed)):
        raise ValueError("data_master_seed must be finite")
    return tuple(
        generate_procedural_sequence(
            profile,
            split=split,
            sequence_index=index,
            data_master_seed=data_master_seed,
        )
        for index in range(sequence_count)
    )
