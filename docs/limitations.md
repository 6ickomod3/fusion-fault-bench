# Limitations

## Measurement abstraction

Fusion Fault Bench v0.1 evaluates simulated object-level estimator outputs. It
does not render images, generate point clouds, run a learned detector, or model
every raw-sensor artifact. The abstraction isolates fusion behavior but cannot
establish detector robustness.

The released M1 evidence is narrower still: a one-object, two-dimensional
Gaussian analytic model with additive output bias and
uncertainty-reporting stress. M2 validates SE(3), projection, ROI, box, and
covariance implementations, but it does not add a fault-performance result.
M3 adds timestamps, dropout, and procedural constant-velocity motion. M4 adds
one observable health-aware rule, but neither milestone adds latent-scene
replay or a learned detector.

## Conditional crossover

A crossover depends on the chosen actual error model, reported covariance,
field of view, region of interest, task loss, and fault persistence. It is not a
universal threshold for a physical sensor.

The M1 meter and standard-deviation-scale roots belong to different declared
axes and cannot be combined or ranked as one severity. The correctly reported
noise control has an `undetermined` finite-sample status; neither its point
curve root nor its population no-root reference licenses a stronger
finite-sample conclusion.

## Association

Known object IDs remove data-association failure from v0.1. This is useful for
causal isolation but optimistic relative to a complete perception stack.
Hungarian association and set losses are deferred until the matched-center
benchmark is valid.

## Independence and common-mode error

Initial information fusion assumes independent sensor errors. Shared ego-pose
or map-frame errors violate that assumption. A required common-mode control
demonstrates that cross-modal agreement can remain high while both modalities
are wrong. In M3, a shared `±4 m` proxy bias raised fixed-fusion loss above
`16 m²` while disagreement remained invariant within `7.11e-15 m`; this is a
constructed blind spot, not a claim about common-mode fault prevalence.

## nuScenes-mini

nuScenes-mini contains ten scenes. Replaying its annotations, poses,
calibration, and timestamps tests whether findings persist under those latent
geometries and motions; it does not validate real sensor noise or support
fleet-scale inference.

The released M2 record is narrower than replay. One user-provided tree matching
the declared official-mini profile passed fixed metadata, link,
referenced-key-frame existence, and projection attestations. The public
aggregate does not authenticate archive or table bytes. The 808 file checks do
not read image or point-cloud contents, the independent scalar reference is
not an official-devkit execution, and the omitted local diagnostic cannot
support a localization-loss or physical calibration-accuracy claim.

## Health attribution

Residual-based health scoring can identify inconsistency but may not uniquely
identify its cause. Difficult geometry, covariance misspecification, and
common-mode faults can be ambiguous. The benchmark retains an
unknown/ambiguous decision and publishes attribution failures.

M4 evaluates an observable health policy separately from its diagnostic
target-drop and frame-action performance oracles. The latter two use fault
metadata or hindsight and are not deployable.

M4 also demonstrates that attribution is not action utility. Under `3×`
underreported LiDAR noise, the combined gate detected and attributed every
event but worsened matched-center loss relative to fixed fusion. Under shared
common-mode bias there is no uniquely healthy target, and under held-out edge
clean support false-alert and loss regression increased. The selected
thresholds are frozen benchmark choices, not operational thresholds.
The frozen three-frame recovery latch also created post-event loss after a
strongly beneficial LiDAR-bias event; hysteresis is therefore part of the
policy tradeoff, not a cost-free implementation detail.

The clean ECDF anomaly ranks are not calibrated probabilities. Their training
population is procedural constant-velocity output, and threshold uncertainty
is not propagated into test intervals. Test intervals are pointwise and
condition on the selected fit.

The M4 public release retains every aggregate row but omits 433,700
sequence-level loss, contrast, and event rows. Exact commitments authenticate
what was omitted, but a third party cannot independently recompute bootstrap
inference from the curated release without rerunning the benchmark.

## Safety boundary

Matched-center loss and fallback regret are benchmark metrics, not collision or
safety metrics. The project makes no production-readiness, certification, or
operational fallback claim.
