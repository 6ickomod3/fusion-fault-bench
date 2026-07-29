# Fusion Fault Bench: Project Plan

The normative definitions for the first release are specified in
[Benchmark Contract v0.1](benchmark-contract-v0.1.md). This roadmap contains
later extensions; where it is broader, the benchmark contract controls v0.1.

## 1. Project statement

Fusion Fault Bench is a deterministic framework for evaluating object-level
camera-LiDAR estimator fusion under controlled proxy faults. It generates paired
multimodal estimator outputs from known latent scenes, quantifies when fixed
fusion has higher matched-center loss than a designated healthy modality, and
evaluates observable health-aware fallback policies.

The central question is:

> Under a declared estimator-output model, when does fusion increase benchmark
> loss, can observable evidence attribute the inconsistency, and can a fallback
> recover loss with a measured clean-versus-fault tradeoff?

This project focuses on the sensor-observation and fusion layer. Agent behavior,
planning quality, collision metrics, and closed-loop policy evaluation are
outside its scope. CPU-only reproduction is a design property, not the research
claim.

## 2. Goals

- Build analytic cases, deterministic procedural temporal scenes, and a
  nuScenes-mini latent-scene replay adapter.
- Simulate object-level camera and LiDAR estimator outputs using explicit
  geometry, timing, actual error, and reported-uncertainty models.
- Inject reproducible dropout, degradation, extrinsic, and temporal faults.
- Compare unimodal estimation, fixed fusion, health-aware fallback,
  fault-target-drop policy, and a performance oracle.
- Define signed healthy-modality delta, fusion-benefit, harmful-fusion-gap, and
  predeclared crossover analyses with sequence-clustered uncertainty intervals.
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

1. At what configured fault severity does fixed fusion have higher
   matched-center loss than the modality designated healthy?
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
  +-- Analytic Gaussian oracle
  +-- Seeded procedural generator
  `-- nuScenes-mini annotation replay
             |
             v
Estimator-output models
  +-- Actual sampling error
  `-- Reported covariance, pose, and timestamp
             |
             v
Manifest-driven fault injection
             |
             v
Camera-only / LiDAR-only / fixed fusion / health fallback / oracles
             |
             v
Matched-center loss / signed contrasts / health quality / uncertainty / reports
```

Every run will be defined by an immutable manifest containing the scenario,
sequence seed, sensor configuration, fault parameters, estimand, and experiment
schema version. Runtime timestamps, software revision, platform, and local paths
belong in a separate run-provenance record and do not affect the intent digest.

### 5.1 Latent scenes

Three scene sources are planned:

- **Analytic mode:** One-object Gaussian cases with known expected fixed-fusion
  loss and bias crossover.
- **Procedural mode:** Seeded sequences of matched moving BEV centers with
  constant velocity and controlled scene-layout families. Detection misses,
  false positives, extents, and unknown association are deferred to v0.2.
- **nuScenes replay mode:** nuScenes-mini annotations, ego poses, timestamps,
  camera intrinsics, and sensor extrinsics provide recorded latent geometry and
  motion. Sensor estimator outputs remain simulated.

Procedural scenes provide controlled coverage. nuScenes replay tests whether
findings persist when latent geometry, motion, calibration, and timing are drawn
from ten recorded mini scenes; it is exploratory grounding, not real-sensor
transfer validation.

### 5.2 Camera measurement model

- v0.1 analytic mode produces a noisy ego-BEV center with separate actual and
  reported covariance.
- Geometry mode projects centers through a calibrated pinhole camera, applies
  image bounds, and produces bearing/depth with covariance propagated into
  ego-BEV coordinates.
- Camera support defines the common front-camera/LiDAR scoring region.

### 5.3 LiDAR measurement model

- Produce noisy ego-BEV center estimates with separate actual and reported
  covariance.
- Restrict nominal LiDAR eligibility to the same object-frame support as the
  front camera.
- Preserve sensor pose and field-of-regard assumptions explicitly in config.

These are estimator-output models, not raw-sensor simulators.

## 6. Fault model

Primary experiments use sequence-persistent, single-axis proxy faults. Burst,
compound, and frame-varying faults follow only after the one-factor experiments
are valid.

| Family | Initial cases |
|---|---|
| Bias | Additive estimator-output position bias with nominal covariance |
| Uncertainty | Increased noise with either correct or underreported covariance |
| Dropout | Partial or complete modality unavailability |
| Extrinsic metadata | Camera translation and yaw error |
| Temporal metadata | Timestamp offset or omitted latency compensation |

Physical observations are generated with the true pose and timestamp. An
extrinsic fault modifies only estimator-consumed metadata:

\[
\widetilde T_{e\leftarrow c}=\Delta T_e T^{\mathrm{true}}_{e\leftarrow c}.
\]

All transforms use right-handed frames, column vectors, and
\(T_{a\leftarrow b}p_b=p_a\). Fault generation and reconstruction may not reuse
the same corrupted transform because that would cancel the injected error.
Severity grids are declared before the benchmark and are controlled stress-test
settings, not estimates of real fault priors.

## 7. Baselines

The initial comparison set is intentionally interpretable:

1. Camera-only estimation.
2. LiDAR-only estimation.
3. Fixed covariance-weighted information fusion.
4. Rule-based health gating using pre-update normalized innovation squared.
5. Fault-label oracle that drops the injected-fault modality.
6. Performance oracle that selects the lowest-loss method with hindsight.

Known object correspondence is used in v0.1 to isolate fusion behavior.
Robust fusion, learned health classification, Hungarian association, set loss,
and false detections are v0.2 candidates added only after the matched-center
task is valid.

## 8. Sensor-health estimator

A constant-velocity temporal predictor will form a prediction before consuming
the current measurements. Sensor-specific or leave-one-sensor-out state tracks
produce innovation residuals while recording which measurements update them.
Candidate observable features include:

- Per-sensor normalized innovation squared mean, maximum, variance, and trend.
- Cross-modal position disagreement.
- Missing-observation rate.
- Timestamp inconsistency.
- Temporal jerk and track inconsistency.

The primary v0.1 policy is rule-based and includes an unknown/ambiguous outcome.
An optional later CPU model may predict healthy, camera-fault, LiDAR-fault, or
ambiguous probabilities. Fault type, severity, seed, split, and manifest fields
are prohibited inputs. Probability calibration and thresholds are selected on
validation data only.

## 9. Metrics

Let \(L_{m,j}(s)\) be the mean squared BEV center error for method \(m\),
complete sequence \(j\), and severity \(s\). Population method loss
\(\bar L_m(s)\) is the mean of sequence losses. The v0.1 primary task is
matched-center MSE in \(\mathrm{m}^2\).

### 9.1 Primary signed contrast

For a single-sensor fault with a predeclared healthy modality \(H\):

\[
D_H(s)=\frac{1}{N}\sum_j\left(L_{F,j}(s)-L_{H,j}(s)\right).
\]

Negative values mean fixed fusion helps; positive values mean it is harmful.
Inference and intervals operate on this signed quantity.

### 9.2 Fusion benefit

\[
\mathrm{FB}(s) =
\min\left(\bar L_C(s),\bar L_L(s)\right)-\bar L_F(s).
\]

The minimum is taken over population method means, never per object, frame, or
sequence.

### 9.3 Harmful-fusion gap

\(\mathrm{HFG}(s)=\max(0,D_H(s))\) is derived only after inference so
beneficial evidence is not truncated.

### 9.4 Crossover severity

Crossover is the first zero of a predeclared nondecreasing isotonic fit to
\(D_H(s)\), computed separately for each fault family and physical axis. Raw
points remain visible. Complete sequences are paired-bootstrap resampled and
the curve is refit in each replicate. The result may be **not observed** or
**undetermined** rather than a forced threshold.

### 9.5 Fallback evaluation

- Gain over always using fixed fusion.
- Clean-condition regression.
- Fraction of oracle-recoverable loss recovered.
- Gap to the fault-target-drop policy and performance oracle.
- Loss-coverage behavior when abstention is allowed.

The oracle-recovery fraction is reported only when its aggregate denominator is
positive and above a declared numerical tolerance.

### 9.6 Health evaluation

- Event-level attribution including unknown/ambiguous.
- Time to detect, time to recover, and false alerts per clean sequence.
- Frame AUROC and AUPRC as secondary diagnostics.
- Brier score and expected calibration error.
- Performance on unseen severities and held-out fault families.
- Runtime, peak memory, and throughput on named CPU hardware.

## 10. Experiment protocol

- Split complete temporal sequences, never individual frames.
- Use paired clean and faulted versions of the same latent scene.
- Reuse the same stochastic-noise realization across competing fusion methods.
- Keep every variant of one latent sequence in the same split.
- Hold out layout families, range/velocity slices, fault families, and severity
  intervals in addition to using disjoint seeds.
- Train health estimators on procedural scenarios before nuScenes replay.
- Treat nuScenes-mini replay as exploratory latent-geometry grounding.
- Tune thresholds and probability calibration on validation data only.
- Predeclare fault families, severities, seeds, primary metrics, and exclusions.
- Use paired bootstrap confidence intervals over sequences.
- Report non-monotonic responses and failed hypotheses.

Required negative controls include:

- A common-mode coordinate bias applied to both sensors.
- Increased unbiased noise without a systematic pose fault.
- Clean but geometrically difficult scenes.

The common-mode case is important: cross-modal agreement can remain high even
when both modalities are wrong in the same way.

## 11. Work plan and acceptance criteria

### M0: Freeze estimands and reproducible foundation

Status: **complete**. The contract foundation passed its adversarial release
gate; no quantitative benchmark result is implied.

Deliver:

- Normative v0.1 benchmark contract, public benchmark card, and claim policy.
- Privacy-safe Git configuration.
- Pinned executable CPU environment and CI.
- Strict alpha experiment, run, and metric schemas.
- Canonical manifest serialization and digest CLI.
- Public methodology plus ignored private learning system.

Accept when:

- Private interview material and datasets are verified as ignored.
- A clean checkout contains no local data or secrets.
- Transform, fault-insertion, uncertainty, ROI, task-loss, aggregation,
  bootstrap, and crossover semantics are frozen.
- Lint, strict source type checking, tests, schema drift checks, and package
  builds pass from the lockfile.
- All planned claims are labeled as hypotheses rather than results.

### M1: Analytic end-to-end vertical slice

Status: **released**. See the
[M1 analytic evidence](../reports/releases/m1-analytic-v0.1.0/README.md).

Deliver:

- One-object 2D Gaussian camera/LiDAR generator.
- Camera bias and uncertainty-reporting faults.
- Camera-only, LiDAR-only, fixed-information fusion, target-drop policy, and
  performance-oracle semantics.
- Matched-center MSE, signed healthy-modality contrast, paired intervals, and
  predeclared crossover.
- Machine-readable run and metric records.

Accept when:

- Independent Gaussian fusion mean, covariance, and expected MSE match
  closed-form results.
- The population contract-grid/PAVA crossover matches an independent analytic
  grid calculation at the fixed tolerance; the continuous model root is
  reported separately as a discretization reference.
- Correctly reported increased noise is distinguished from overconfidence.
- Metric signs, oracle semantics, no-crossing, and undetermined cases pass.
- Re-running a manifest produces byte-identical raw records.

### M2: Geometry and nuScenes grounding

Status: **pre-registered**. See the
[M2 geometry validation plan](m2-geometry-plan.md). Implementation and local
geometry execution have not started.

Deliver:

- Named frames, SE(3), quaternion, and camera projection utilities.
- Minimal nuScenes-mini metadata adapter and referential-integrity validator.
- One-sample global-to-ego-to-camera projection diagnostic.
- Front-camera/common-LiDAR ROI implementation.

Accept when:

- Transform composition, inverse, and round-trip property tests pass.
- nuScenes scalar-first quaternion and sensor-to-ego conventions are tested.
- Projection agrees with an independent fixture or official devkit path within
  a declared tolerance.
- Camera bearing/depth covariance propagation agrees with Monte Carlo.
- No dataset content or absolute local path enters tracked artifacts.

### M3: Temporal procedural benchmark

Deliver:

- Constant-velocity matched-center sequences and declared layout families.
- Camera and LiDAR estimator-output models in the common ROI.
- Calibration translation/yaw, timestamp-offset, bias, and
  uncertainty-reporting crossover families plus a separate dropout
  availability control.
- Fixed procedural experiment matrix with paired sequence records.

Accept when:

- Zero-fault and zero-noise identity cases recover the latent state.
- Empirical noise and propagated covariance match configured moments.
- Every fault has an identity and analytic or monotonic severity-response check.
- Procedural splits hold out declared layout and motion/range slices.
- Smoke experiments run in CPU-only CI and fixed sweeps are deterministic.

### M4: Health-aware fallback

Deliver:

- Pre-update sensor-specific or leave-one-sensor-out temporal predictors.
- Cross-modal, camera-NIS, LiDAR-NIS, and combined health gates.
- Healthy, camera, LiDAR, and unknown/ambiguous decisions.
- Fixed, estimated, fault-target-drop, and performance-oracle comparisons.

Accept when:

- No prohibited manifest metadata enters health features.
- No current measurement leaks into its own innovation prediction.
- Scene-disjoint, held-out-layout, unseen-severity, and held-out-fault
  evaluations are present.
- Clean regression, fault recovery, and oracle gap are reported together.
- Event attribution, time to detect/recover, and false alerts are reported.
- Common-mode and difficult-clean attribution failures are documented.

No minimum improvement is required for completion; a valid negative result is
acceptable.

### M5: nuScenes latent replay and selected v0.2 complexity

Deliver:

- nuScenes-mini matched-center temporal replay using local metadata.
- Comparison of procedural and replay ranges, motions, visibility, and timing.
- Repetition of the predeclared v0.1 fault matrix on eligible replay slices.
- Only if v0.1 remains valid: selected association, miss/false-positive, set
  loss, robust-fusion, or learned-health extensions.

Accept when:

- The adapter requires only documented user-provided mini data.
- No dataset file is committed or redistributed.
- Scene counts, ranges, dimensions, motion, visibility, and timing differences
  are reported rather than hidden.
- Replay claims are limited to persistence under recorded latent geometry and
  motion, with scene-level uncertainty.
- Any v0.2 extension is reported separately from the v0.1 estimand.

### M6: Public report

Deliver:

- Fixed experiment matrix and final CPU benchmark.
- Curated aggregate figures and concise technical report.
- One-command procedural reproduction path.
- Resume-ready evidence table containing only measured values.
- Complete private code tour, derivations, question bank, project narratives,
  and failure postmortems.

Accept when:

- A clean CPU environment reproduces the procedural report.
- Public figures trace back to versioned manifests.
- The report distinguishes estimator-output simulation from raw-sensor
  simulation.
- Limitations and related work are explicit.
- An adversarial claims audit passes and every headline statement has released
  evidence plus a validity boundary.

## 12. Planned repository layout

```text
.
|-- examples/manifests/
|-- docs/
|-- reports/
|   |-- generated/          # ignored raw outputs
|   `-- releases/           # curated aggregate evidence
|-- schemas/
|-- src/fusion_fault_bench/
|   |-- contracts/
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

Foundation contracts and tests are introduced in M0. Scientific dependencies
and modules are added only when their milestone uses them.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Estimator-output abstraction appears too synthetic | Ground latent geometry in nuScenes replay and publish the abstraction boundary |
| Noise priors are not realistic | Expose all priors in configs and run sensitivity analyses |
| Cross-modal residual cannot identify the faulty sensor | Add sensor-specific temporal innovations and an oracle bound |
| Common-mode failures remain invisible | Include a required common-mode negative control |
| Health model learns fault-generator shortcuts | Hold out fault families and forbid manifest fields as features |
| Crossover is unstable or absent | Use paired intervals and permit "not observed" |
| Association obscures pose effects | Validate matched IDs first and report later Hungarian results separately |
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
