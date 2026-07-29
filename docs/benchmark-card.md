# Benchmark Card

## Summary

Fusion Fault Bench is a deterministic benchmark for camera-LiDAR
estimator-output fusion under controlled proxy faults. The v0.1 contract
studies known-object BEV center estimates in a common front-camera/LiDAR region
of interest. The released M1 slice is a narrower one-object Gaussian analytic
case; geometry, temporal, and dataset-grounded modes remain planned.

The benchmark is designed to answer a narrow evaluation question: when does
fixed information fusion increase matched-center loss relative to a modality
declared healthy by a controlled single-sensor target? A later milestone asks
how much of that loss an observable health gate can recover.

## Intended uses

- Verify fusion and fault-injection implementations against analytic cases.
- Compare fusion policies under paired, manifest-defined fault sweeps.
- Study calibration, timing, uncertainty-reporting, and dropout sensitivities.
- Test evaluator blind spots with difficult-clean and common-mode controls.
- Reproduce small benchmarks on CPU-only hardware.

## Out-of-scope uses

- Estimating real sensor-fault rates or severity distributions.
- Evaluating raw camera images or point clouds.
- Claiming detector, vehicle, or production-safety robustness.
- Planning, collision, closed-loop behavior, or simulator-policy evaluation.
- Certification, operational fallback selection, or fleet deployment.

## v0.1 data modes

**Analytic — released in M1:** one-object Gaussian camera bias and
uncertainty-reporting cases with closed-form checks.

**Procedural — planned:** seeded constant-velocity scenes with controlled
layouts and paired observation draws.

**nuScenes-mini grounding — planned:** recorded annotations, poses,
calibration, and timing provide latent scene structure. Estimator outputs
remain simulated. Ten mini scenes are treated as exploratory grounding rather
than external validation.

## Status vocabulary

- **Planned:** specified but not implemented.
- **Implemented:** code and targeted tests exist.
- **Validated:** acceptance checks and adversarial review passed.
- **Released:** evidence is committed under `reports/releases/` and traces to a
  tagged software revision.

The current quantitative evidence is limited to
[M1 analytic estimator-output stress tests](../reports/releases/m1-analytic-v0.1.0/README.md).
The correctly reported-noise control is published with an `undetermined`
finite-sample status; it is not converted into a favorable conclusion.

## Primary validity boundaries

- v0.1 assumes known object identity and independent modality errors.
- Camera and LiDAR are compared only on shared support.
- Crossover depends on the configured true error and reported uncertainty.
- Dropout is evaluated through coverage and conditional loss, not crossover.
- A common-mode error can evade cross-modal consistency checks.
- Object-level estimator outputs do not contain raw-sensor failure modes.

See [Benchmark Contract v0.1](benchmark-contract-v0.1.md) for normative
definitions and [Limitations](limitations.md) for the public claim boundary.
