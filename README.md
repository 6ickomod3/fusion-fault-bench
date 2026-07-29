# Fusion Fault Bench

[![CI](https://github.com/6ickomod3/fusion-fault-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/6ickomod3/fusion-fault-bench/actions/workflows/ci.yml)

**When does camera-LiDAR fusion become worse than trusting the modality
declared healthy in a controlled single-sensor fault?**

Fusion Fault Bench is a deterministic estimator-output benchmark for answering
that question under controlled calibration, timing, uncertainty, bias, and
dropout faults. It uses paired counterfactual sequences, explicit covariance,
sequence-level inference, and predeclared crossover rules.

The benchmark runs on CPU. That is a reproducibility property, not the research
contribution.

## What it evaluates

```text
Latent matched-object states
          |
          v
Camera and LiDAR estimator outputs
  - actual error model
  - reported covariance
  - true/reported pose and time
          |
          v
Manifest-defined proxy fault
          |
          v
Camera-only | LiDAR-only | fixed fusion | health gate | oracles
          |
          v
Matched-center loss | signed healthy-modality delta | crossover
```

The v0.1 contract restricts evaluation to known object IDs and BEV centers in a
common front-camera/LiDAR region of interest. This is designed to isolate fusion
behavior from detection and data association. A later version may add set
prediction and Hungarian association after the matched-center benchmark is
validated.

## Why the contract is narrow

Fusion can appear robust or harmful simply because an evaluator chose a
convenient covariance, field of view, loss aggregation, or fault insertion
point. This project fixes those choices before running the benchmark:

- physical observations use the true pose and timestamp;
- calibration and timing faults modify estimator-consumed metadata;
- actual error and reported uncertainty remain separate;
- clean and faulted conditions reuse latent scenes and random draws;
- losses are aggregated by complete sequence;
- inference uses a signed delta from the designated healthy modality;
- crossover fitting and uncertainty rules are predeclared.

See the frozen
[v0.1 benchmark contract](docs/benchmark-contract-v0.1.md).

## Quickstart

```bash
uv sync --locked --group dev
uv run ffb --version
uv run ffb manifest validate \
  examples/manifests/analytic-bias-v1alpha1.json
uv run pytest
```

The M1 analytic path can also produce and strictly reload a complete scientific
artifact from a clean checkout:

```bash
uv run ffb run \
  examples/manifests/analytic-bias-v1alpha1.json \
  --output-dir reports/generated/analytic-camera-x-bias-a603d090f77a
uv run ffb bundle validate \
  reports/generated/analytic-camera-x-bias-a603d090f77a
```

The current tools require no GPU, model checkpoint, or dataset. nuScenes-mini
is user-provided and optional for the planned geometry and latent-scene
grounding.

## Evidence status

**Current release scope: M0 foundation plus the preregistered M1 analytic
implementation. There are no released scientific findings yet.**

| Component | Status |
|---|---|
| Versioned manifest and result contracts | Validated |
| Canonical manifest fingerprinting | Validated |
| Analytic fusion/fault vertical slice | Implemented; release evidence pending |
| SE(3) and nuScenes projection grounding | Planned |
| Temporal procedural benchmark | Planned |
| Health-aware fallback | Planned |
| Released quantitative results | None |

Results enter this README only after they trace to a released manifest,
software revision, aggregate record, named CPU run, uncertainty interval, and
reproduction command.

## Documentation

- [Benchmark card](docs/benchmark-card.md)
- [Benchmark contract v0.1](docs/benchmark-contract-v0.1.md)
- [Evaluation protocol](docs/evaluation-protocol.md)
- [Reproducibility](docs/reproducibility.md)
- [Dataset preparation](docs/dataset-preparation.md)
- [Limitations](docs/limitations.md)
- [Results](docs/results.md)
- [Project plan](docs/project-plan.md)
- [M1 analytic pre-registration](docs/m1-analytic-plan.md)

## Explicit non-goals

- Raw camera or point-cloud simulation.
- Training or reproducing a neural BEV detector.
- Estimating naturally occurring or fleet-level fault rates.
- Planning, collisions, or closed-loop policy evaluation.
- Production-safety, certification, or real-world robustness claims.

The Apache-2.0 license covers project code only. nuScenes and other external
assets remain governed by their own terms; see
[Data and model terms](DATA_AND_MODEL_TERMS.md).
