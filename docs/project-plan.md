# Fusion Fault Bench: Project Plan

## 1. Project statement

Fusion Fault Bench is a deterministic, CPU-only framework for evaluating
camera-LiDAR fusion under controlled sensor faults. It generates paired
multimodal measurements from known latent scenes, quantifies when fusion becomes
harmful, and evaluates sensor-health-aware fallback policies.

The central question is:

> Can an evaluator detect when camera and LiDAR observations no longer support
> reliable fusion, attribute the unhealthy modality, and choose a fallback with
> a measured clean-versus-fault tradeoff?

This project focuses on the sensor-observation and fusion layer. Agent behavior,
planning quality, collision metrics, and closed-loop policy evaluation are
outside its scope.

## 2. Goals

- Build deterministic procedural temporal scenes and a nuScenes-mini replay
  adapter.
- Simulate object-level camera and LiDAR measurements using explicit geometry,
  timing, and uncertainty models.
- Inject reproducible dropout, degradation, extrinsic, and temporal faults.
- Compare unimodal estimation, fixed fusion, robust fusion, and health-aware
  fallback.
- Define fusion-benefit, harmful-fusion-gap, and crossover-severity analyses
  with uncertainty intervals.
- Validate sensor-health scores on scene-disjoint data and unseen fault
  severities.
- Publish reproducible configurations, manifests, tests, results, limitations,
  and a concise technical report.

## 3. Non-goals

- Rendering photorealistic images or full LiDAR point clouds.
- Training, reproducing, or benchmarking a neural BEV detector.
- Reproducing the published CalibRobustBEV method.
- Estimating naturally occurring or fleet-level sensor-fault distributions.
- Radar, ROS, CARLA, or full 3D ray tracing.
- Making production-safety, certification, or broad real-world generalization
  claims.
- Requiring CUDA, GPU inference, or a full nuScenes download.

## 4. Research questions

1. At what fault severity does fixed fusion stop outperforming the best
   available unimodal estimator?
2. Which temporal and cross-modal consistency signals best identify camera
   faults, LiDAR faults, and ambiguous common-mode failures?
3. Can a health-aware policy recover fault-induced loss without causing an
   unacceptable clean-condition regression?
4. How stable are these conclusions across scene geometry, object range,
   motion, fault family, severity, and random seed?

The project does not require a positive result. A well-supported negative result
or a documented evaluator blind spot is a valid outcome.

## 5. System architecture

```text
Latent temporal scene
  +-- Seeded procedural generator
  `-- nuScenes-mini annotation replay
             |
             v
Nominal measurement models
  +-- Camera projection, bearing, and noisy depth
  `-- LiDAR position, extent, and range
             |
             v
Manifest-driven fault injection
             |
             v
Camera-only / LiDAR-only / fixed fusion / robust fusion / fallback
             |
             v
Task loss / fusion risk / health quality / uncertainty / reports
```

Every run will be defined by an immutable manifest containing the scenario,
sequence seed, timestamps, sensor configuration, fault parameters, experiment
configuration version, and software revision.

### 5.1 Latent scenes

Two scene sources are planned:

- **Procedural mode:** Seeded sequences of moving 3D cuboids with object class,
  size, pose, velocity, ego motion, misses, and false observations.
- **nuScenes replay mode:** nuScenes-mini annotations, ego poses, timestamps,
  camera intrinsics, and sensor extrinsics provide real scene geometry. Sensor
  observations remain simulated.

Procedural scenes provide controlled coverage. nuScenes replay tests whether
conclusions transfer to a less synthetic scene distribution.

### 5.2 Camera measurement model

- Project cuboid centers and corners through a calibrated pinhole camera.
- Apply image-bound and field-of-view filtering.
- Produce bearing, projected-box, estimated-depth, and uncertainty fields.
- Support range-dependent covariance, missed observations, false positives,
  and confidence degradation.

### 5.3 LiDAR measurement model

- Produce noisy bird's-eye-view position and object-extent measurements.
- Support range-dependent variance, sparsity, bias, missed observations, and
  false positives.
- Preserve sensor pose and field-of-regard assumptions explicitly in config.

These are measurement-level models, not raw-sensor simulators.

## 6. Fault model

Primary experiments use scene-persistent single-sensor faults. Burst and
compound faults follow only after the one-factor experiments are valid.

| Family | Initial cases |
|---|---|
| Dropout | Missing observations, burst dropout, full modality loss |
| Degradation | Increased variance, misses, outliers, depth/range bias |
| Extrinsic | Translation bias and roll/pitch/yaw bias |
| Temporal | Timestamp skew, stale observations, repeated frames |

Extrinsic corruption will use an explicit convention such as

\[
T_{\mathrm{used}} = \exp(\hat{\xi}) T_{\mathrm{true}},
\]

with the coordinate frame and left/right multiplication convention documented
and tested. Severity grids will be declared in versioned configurations before
the corresponding benchmark is run. They are controlled stress-test settings,
not estimates of real fault priors.

## 7. Baselines

The initial comparison set is intentionally interpretable:

1. Camera-only estimation.
2. LiDAR-only estimation.
3. Fixed covariance-weighted information fusion.
4. Robust fusion using clipped or Huber-weighted innovations.
5. Rule-based health gating using normalized innovation squared.
6. Optional calibrated CPU classifier using observable health features.
7. Oracle-health fallback using injected fault labels as an upper bound.

Oracle object correspondence will be used first to isolate fusion behavior.
Hungarian association will be added as a separate experiment so association
failure is measured rather than silently conflated with geometric error.

## 8. Sensor-health estimator

A constant-velocity temporal predictor will produce sensor-specific innovation
residuals. Candidate observable features include:

- Per-sensor normalized innovation squared mean, maximum, variance, and trend.
- Cross-modal position and extent disagreement.
- Missing-observation rate.
- Association failure rate.
- Timestamp inconsistency.
- Temporal jerk and track inconsistency.

The learned option is a small CPU model that predicts healthy, camera-fault, or
LiDAR-fault probabilities. Fault type, severity, seed, and manifest fields are
prohibited inputs. Probability calibration and decision thresholds are selected
on validation data only.

## 9. Metrics

Let \(L_C\), \(L_L\), and \(L_F\) be lower-is-better task losses for camera,
LiDAR, and fixed fusion. The primary task loss is planned to be GOSPA or an
equivalently explicit Hungarian-matched loss, with localization, missed-object,
and false-object components reported separately.

### 9.1 Fusion benefit

\[
\mathrm{FB}(s) = \min(L_C(s), L_L(s)) - L_F(s).
\]

A positive value means fusion improves over the best unimodal estimator.

### 9.2 Harmful-fusion gap

For a single-sensor fault with known healthy modality \(H\):

\[
\mathrm{HFG}(s) = \max(0, L_F(s) - L_H(s)).
\]

### 9.3 Crossover severity

\[
s^* = \inf\{s : \mathrm{FB}(s) \le 0\}.
\]

Crossover will be estimated with paired sequence bootstrap intervals and a
predeclared monotonic fitting rule when appropriate. If no supported crossing
occurs, the result is "not observed" rather than a forced estimate.

### 9.4 Fallback evaluation

- Gain over always using fixed fusion.
- Clean-condition regression.
- Fraction of oracle-recoverable loss recovered.
- Regret relative to the oracle fallback.
- Risk-coverage behavior when abstention is allowed.

### 9.5 Health evaluation

- AUROC and AUPRC.
- Brier score and expected calibration error.
- Fault-attribution confusion matrix.
- Performance on unseen severities and held-out fault families.
- Runtime, peak memory, and throughput on named CPU hardware.

## 10. Experiment protocol

- Split complete temporal sequences, never individual frames.
- Use paired clean and faulted versions of the same latent scene.
- Reuse the same stochastic-noise realization across competing fusion methods.
- Generate procedural train, validation, and test sets from disjoint seeds.
- Train health estimators on procedural scenarios before nuScenes replay.
- Treat nuScenes replay as a separate scene-distribution test.
- Tune thresholds and probability calibration on validation data only.
- Predeclare fault families, severities, seeds, primary metrics, and exclusions.
- Use paired bootstrap confidence intervals over sequences.
- Run broad sweeps with oracle association, then repeat selected cases with
  Hungarian association.
- Report non-monotonic responses and failed hypotheses.

Required negative controls include:

- A common-mode coordinate bias applied to both sensors.
- Increased unbiased noise without a systematic pose fault.
- Clean but geometrically difficult scenes.

The common-mode case is important: cross-modal agreement can remain high even
when both modalities are wrong in the same way.

## 11. Work plan and acceptance criteria

### M0: Reproducible project foundation

Deliver:

- Public project plan and data-preparation guide.
- Privacy-safe Git configuration.
- Pinned CPU environment design.
- Initial experiment and result schemas.

Accept when:

- Private interview material and datasets are verified as ignored.
- A clean checkout contains no local data or secrets.
- All planned claims are labeled as hypotheses rather than results.

### M1: Geometry and reproducibility

Deliver:

- Named coordinate frames, SE(3), and camera projection utilities.
- Procedural fixtures and manifest schema.
- CPU property and unit tests.

Accept when:

- Transform composition and inverse tests pass.
- Projection agrees with independent fixtures or the nuScenes devkit within a
  declared tolerance.
- Analytic yaw-error and timing-error checks pass.
- Re-running a manifest produces identical measurements and metrics.

### M2: Measurement simulation

Deliver:

- Procedural temporal scenes.
- Camera and LiDAR measurement models.
- Single-factor fault implementations.

Accept when:

- Zero-noise measurements recover latent geometry within numerical tolerance.
- Empirical noise moments match configured moments within declared tolerance.
- Every fault has an identity case and a severity-response test.
- Smoke experiments run in CPU-only continuous integration.

### M3: Fusion baselines and decision metrics

Deliver:

- Unimodal, fixed-fusion, robust-fusion, and oracle baselines.
- Matched-object loss decomposition.
- Fusion metrics, bootstrap intervals, and crossover estimation.

Accept when:

- Metric sign and boundary tests pass.
- Independent Gaussian fusion behaves as analytically expected.
- Crossover handling covers single crossing, multiple crossings, and no
  crossing.
- Results are stored in machine-readable, versioned records.

### M4: Health-aware fallback

Deliver:

- Temporal innovation gate.
- Optional calibrated CPU classifier.
- Fixed, estimated, and oracle fallback comparison.

Accept when:

- No prohibited manifest metadata enters health features.
- Scene-disjoint and held-out-fault evaluations are present.
- Clean regression, fault recovery, and oracle gap are reported together.
- Negative controls and known attribution failures are documented.

No minimum improvement is required for completion; a valid negative result is
acceptable.

### M5: nuScenes grounding

Deliver:

- nuScenes-mini metadata and replay adapter.
- Coordinate-frame validation visualization.
- Comparison of procedural and replay distributions.

Accept when:

- The adapter requires only documented user-provided mini data.
- No dataset file is committed or redistributed.
- Scene counts, ranges, dimensions, motion, visibility, and timing differences
  are reported rather than hidden.

### M6: Public report

Deliver:

- Fixed experiment matrix and final CPU benchmark.
- Generated figures and concise technical report.
- One-command procedural reproduction path.
- Resume-ready evidence table containing only measured values.

Accept when:

- A clean CPU environment reproduces the procedural report.
- Public figures trace back to versioned manifests.
- The report distinguishes measurement simulation from raw-sensor simulation.
- Limitations and related work are explicit.

## 12. Planned repository layout

```text
.
|-- configs/
|-- docs/
|-- reports/
|   `-- generated/          # ignored
|-- src/fusion_fault_bench/
|   |-- experiments/
|   |-- faults/
|   |-- fusion/
|   |-- geometry/
|   |-- health/
|   |-- metrics/
|   |-- scenarios/
|   |-- sensors/
|   `-- visualization/
`-- tests/
```

Only documentation exists at repository bootstrap. Source and test directories
will be introduced with the first executable vertical slice.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Simulator appears too synthetic | Ground scene geometry in nuScenes replay and compare procedural/replay distributions |
| Noise priors are not realistic | Expose all priors in configs and run sensitivity analyses |
| Cross-modal residual cannot identify the faulty sensor | Add sensor-specific temporal innovations and an oracle bound |
| Common-mode failures remain invisible | Include a required common-mode negative control |
| Health model learns fault-generator shortcuts | Hold out fault families and forbid manifest fields as features |
| Crossover is unstable or absent | Use paired intervals and permit "not observed" |
| Association obscures pose effects | Report oracle and Hungarian association separately |
| Coordinate-frame error invalidates results | Make geometry tests and overlays release blockers |
| Results overstate realism or safety | Use controlled-stress-test language and publish limitations |

## 14. Optional future extension

After the CPU milestone is complete, the measurement and policy interfaces may
be connected to externally generated neural-detector predictions. That extension
is not required for the current project and will not influence the CPU-first
architecture.

## 15. Related work and data

- [nuScenes dataset and devkit](https://github.com/nutonomy/nuscenes-devkit)
- [MultiCorrupt](https://github.com/ika-rwth-aachen/MultiCorrupt)
- [CalibRobustBEV](https://just.ustc.edu.cn/article/cstr/32290.14.JUSTC-2024-0028)

The project will cite and distinguish these works rather than presenting their
fault taxonomies or model contributions as new.
