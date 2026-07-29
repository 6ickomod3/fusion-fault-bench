# Dataset Preparation

## Dataset status

The dataset is not required for M0, M1, the procedural M3/M4 paths, or public
release validation. The released M2 local geometry check and planned M5 replay
use only the official **nuScenes v1.0-mini** split. Do not download the full
train/validation or test datasets for the CPU-first milestones.

The project initially needs:

- Sample and sample-data timestamps.
- Calibrated-sensor extrinsics.
- Camera intrinsics.
- Ego poses.
- 3D sample annotations and tracked instance links.
- Visibility and LiDAR-point-count metadata.

The mini split is sufficient for developing and validating the replay adapter.
Procedural experiments will remain the primary controlled benchmark.

## 1. Accept the dataset terms

1. Create or sign in to an account on the
   [nuScenes download page](https://www.nuscenes.org/nuscenes#download).
2. Review and accept the current nuScenes terms.
3. Under **Full dataset (v1.0)**, download only the **Mini** archive,
   conventionally named `v1.0-mini.tgz`.

Dataset archives must not be committed to this repository or redistributed.
Use the official download rather than a third-party mirror.

## 2. Choose storage outside the repository

Choose an absolute dataset directory outside this Git checkout and expose it as
`NUSCENES_ROOT` in your local shell or experiment configuration.

Example:

```bash
export NUSCENES_ROOT=/absolute/path/to/datasets/nuscenes
mkdir -p "$NUSCENES_ROOT"
tar -xzf /absolute/path/to/v1.0-mini.tgz -C "$NUSCENES_ROOT"
```

Do not extract the archive under this repository. The project ignores local
`data/` and `datasets/` directories as a secondary safeguard, but external
storage is preferred.

## 3. Expected layout

After extraction, the root should resemble:

```text
$NUSCENES_ROOT/
|-- maps/
|-- samples/
|-- sweeps/
`-- v1.0-mini/
    |-- calibrated_sensor.json
    |-- ego_pose.json
    |-- instance.json
    |-- sample.json
    |-- sample_annotation.json
    |-- sample_data.json
    `-- ...
```

The exact set of JSON files is defined by the official
[nuScenes schema](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md).

## 4. Manual validation

Before project code is available, verify the main directories and tables:

```bash
test -d "$NUSCENES_ROOT/v1.0-mini"
test -d "$NUSCENES_ROOT/samples"
test -d "$NUSCENES_ROOT/sweeps"
test -f "$NUSCENES_ROOT/v1.0-mini/calibrated_sensor.json"
test -f "$NUSCENES_ROOT/v1.0-mini/ego_pose.json"
test -f "$NUSCENES_ROOT/v1.0-mini/sample_annotation.json"
```

The official tutorial reports that v1.0-mini contains 10 scenes, 404 annotated
samples, and 18,538 sample annotations. Run the implemented M2 check from a
clean checkout:

```bash
uv run ffb geometry validate \
  examples/validation/m2-geometry-v1.json \
  --dataset-root-env NUSCENES_ROOT \
  --output-dir reports/generated/m2-geometry
uv run ffb geometry bundle validate reports/generated/m2-geometry
```

The command checks the fixed 12-table profile, declared links and chains,
CAM_FRONT/LIDAR_TOP calibration and pose consistency, annotation records, and
the existence of 808 referenced key-frame files. It does not read image or
point-cloud contents. It writes one sanitized five-file aggregate artifact and
one ignored local projection diagnostic; it uploads or copies no dataset file.
The no-overwrite output path must be absent before each execution.

## 5. Do not download yet

The following are outside the current milestone:

- nuScenes full train/validation or test archives.
- lidarseg or panoptic extensions.
- CAN bus expansion.
- MultiCorrupt generated datasets.
- Waymo Open Dataset.
- BEVFusion or other neural-model checkpoints.

These additions would increase storage and dependency cost without helping the
first CPU vertical slice.

## 6. What may be committed

Allowed public artifacts:

- Dataset configuration templates containing no credentials.
- Code that reads user-provided data.
- Tiny synthetic fixtures created by this project.
- Aggregate metrics and selected generated figures.
- Dataset citations and setup instructions.

Never commit:

- nuScenes images, point clouds, JSON tables, maps, or archives.
- Authentication cookies, access tokens, or download URLs tied to an account.
- Derived artifacts that reproduce or redistribute protected dataset content.

## 7. Released geometry milestone and planned replay

The released M2 adapter canonicalizes:

1. Global-frame object annotations.
2. Ego pose at each sensor timestamp.
3. Sensor-to-ego extrinsics.
4. Camera intrinsics and image bounds.
5. Global-to-ego-to-sensor transform chains.

It renders one deterministic local diagnostic and compares the production
projection against a repository-owned independent scalar implementation based
on conventions reviewed at a pinned official devkit revision. This is not an
execution-parity claim for the devkit. The diagnostic remains untracked and no
detector inference is involved.

M5 will use eligible annotations, poses, and timing as latent matched-center
sequence structure. Estimator errors will still be simulated; replay will not
turn the benchmark into a raw-sensor evaluation.
