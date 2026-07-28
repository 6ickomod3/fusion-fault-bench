# Fusion Fault Bench

Fusion Fault Bench is a CPU-first framework for studying when camera-LiDAR
fusion stops helping under controlled sensor faults, and whether an observable
sensor-health signal can select a safer fallback.

The project operates on object-level and bird's-eye-view measurements generated
from known latent scenes. It does **not** train a 3D detector or claim to
simulate photorealistic camera images or LiDAR point clouds.

## Core questions

1. When does fixed fusion become worse than the best available unimodal
   estimator?
2. Which observable temporal and cross-modal signals identify the faulty
   modality?
3. How much fault-induced loss can a health-aware fallback recover without
   regressing clean performance?

## Planned scope

- Seeded procedural temporal scenes.
- nuScenes-mini replay using annotations, sensor calibration, ego poses, and
  timestamps as latent geometry.
- Camera and LiDAR measurement models with explicit uncertainty.
- Deterministic dropout, degradation, calibration, and timing faults.
- Camera-only, LiDAR-only, fixed-fusion, robust-fusion, and fallback baselines.
- Fusion-benefit, harmful-fusion-gap, crossover-severity, health-calibration,
  and uncertainty analyses.
- Reproducible manifests, CPU tests, machine-readable results, and a concise
  technical report.

## Explicit non-goals

- GPU training or inference.
- Reproducing a BEV neural detector.
- Raw-sensor or photorealistic simulation.
- Estimating real fleet fault rates.
- Safety-certification or production-readiness claims.

## Documentation

- [Project plan](docs/project-plan.md)
- [Dataset preparation](docs/dataset-preparation.md)

## Status

Planning and repository bootstrap. Results and performance claims will be added
only after the corresponding experiments are implemented and reproduced.
