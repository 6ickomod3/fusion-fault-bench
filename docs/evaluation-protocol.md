# Evaluation Protocol

## Pre-registration

Each released experiment is defined by an immutable JSON manifest. The
canonical, key-sorted UTF-8 representation is fingerprinted with SHA-256. A
manifest records experimental intent but excludes local paths, timestamps,
hostnames, Git revisions, and runtime state.

Every experiment manifest predeclares:

- one fault family, target, physical axis, unit, and injection site;
- an identity condition, direction policy, operator equation, and strictly
  increasing magnitudes;
- actual and reported estimator uncertainty;
- named RNG streams plus separate data and bootstrap seeds;
- methods and evaluation mode;
- loss, aggregation unit, interval method, and—only for single-sensor
  localization experiments—the crossover rule.

The healthy modality is derived from a single-sensor fault target. Common-mode
and availability controls have no healthy-modality crossover.

## Paired design

Clean and faulted conditions reuse the same latent sequences and base random
draws. Competing methods consume the same observations. This turns method and
severity comparisons into paired sequence contrasts rather than independent
Monte Carlo estimates.

## Splits

- Complete sequences are indivisible.
- All corruptions and augmentations of one latent sequence remain in one split.
- Procedural train, validation, and test sets use disjoint sequence seeds and
  declared layout families.
- Thresholds and calibration use validation only.
- Final metrics are computed once on test data.
- nuScenes-mini scenes are reported separately from procedural test sequences.

Seed separation alone is not treated as generalization. Later health models
must also face held-out layout families, fault families, range/velocity slices,
and unseen severity intervals.

## Primary statistic

The primary statistic is the sequence-clustered mean of

\[
d_j(s)=L_{F,j}(s)-L_{H,j}(s).
\]

Intervals use paired bootstrap resampling of complete sequences. The bootstrap
seed and replicate count are manifest fields. Raw severity points remain
visible even when an isotonic curve is used to summarize crossover.

## Required controls

- Every fault's identity condition.
- Analytic independent-Gaussian fusion.
- Correctly reported increased noise.
- Difficult but clean geometry.
- Common-mode position bias.
- Target-drop policy and sequence-level performance oracle kept semantically
  separate.

## Health evaluation

Deployable health features are produced before current-measurement updates and
cannot include injected labels or manifest metadata. Report:

- event-level fault attribution including unknown/ambiguous;
- time to detect and time to recover;
- false alerts per clean sequence;
- clean-condition regression;
- gain over fixed fusion and gaps to the target-drop policy and performance
  oracle;
- probability calibration when probabilities are emitted.

Frame-level AUROC/AUPRC are secondary because persistent faults create
correlated frames.

## Claim release rule

A quantitative statement may enter the README or resume evidence only when it
traces to:

1. a released manifest and its digest;
2. a named Git revision;
3. machine-readable aggregate records;
4. a reproducible command;
5. named CPU hardware and runtime;
6. uncertainty and a stated validity boundary.

Negative or inconclusive results are valid releases. A result is never selected
for publication solely because it supports a preferred narrative.
