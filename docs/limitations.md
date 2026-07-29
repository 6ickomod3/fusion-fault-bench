# Limitations

## Measurement abstraction

Fusion Fault Bench v0.1 evaluates simulated object-level estimator outputs. It
does not render images, generate point clouds, run a learned detector, or model
every raw-sensor artifact. The abstraction isolates fusion behavior but cannot
establish detector robustness.

The released M1 evidence is narrower still: a one-object, two-dimensional
Gaussian analytic model with additive output bias and
uncertainty-reporting stress. It does not yet exercise SE(3), projection,
timestamps, dropout, procedural motion, nuScenes, or a health-aware gate.

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
are wrong.

## nuScenes-mini

nuScenes-mini contains ten scenes. Replaying its annotations, poses,
calibration, and timestamps tests whether findings persist under those latent
geometries and motions; it does not validate real sensor noise or support
fleet-scale inference.

## Health attribution

Residual-based health scoring can identify inconsistency but may not uniquely
identify its cause. Difficult geometry, covariance misspecification, and
common-mode faults can be ambiguous. The benchmark retains an
unknown/ambiguous decision and publishes attribution failures.

## Safety boundary

Matched-center loss and fallback regret are benchmark metrics, not collision or
safety metrics. The project makes no production-readiness, certification, or
operational fallback claim.
