# M4 Verification Record

## Frozen source

- Scientific revision:
  `a829a9f3af541c1b92b89d051b7c8b7003dc5a15`
- Lock digest:
  `ac20e73938328ee6ca0929b7ac8b39b76f81a26251025aa373aaf7ac181bb06f`
- Intent digest:
  `c19573b72a6ef58ad5efdd7039110da79beacd3b398007a508eb7ecfc4c81357`
- Runtime: Python `3.12.13`, Apple M3 Pro, Darwin `24.5.0`
- Accelerator requested: false

## Software verification

Before the scientific runs:

```text
ruff format --check: pass
ruff check: pass
pyright: 0 errors
pytest: 1,134 passed, 1 skipped
branch-aware coverage: 90.67% (gate: 90%)
source distribution build: pass
wheel build: pass
isolated wheel import/version/schema CLI: pass
```

The skipped test requires an optional user-provided local nuScenes tree and is
not part of the dataset-independent M4 result path.

Release closeout repeated the complete check after documentation and CI
integration: 1,134 tests passed, one optional local-data test skipped,
branch-aware coverage was 90.69%, Ruff and Pyright passed, both distributions
built, and the isolated wheel exposed the version, health CLI, intent schema,
and validation schema.

## Exact run commands

All runs set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`NUMEXPR_NUM_THREADS=1`. Each command was one direct child of
`/usr/bin/time -l`; the raw logs are retained in the strict release.

```bash
uv run ffb health fit \
  --output-dir reports/generated/m4-health-fit-primary-a829a9f-r1

uv run ffb health fit \
  --output-dir reports/generated/m4-health-fit-repeat-a829a9f-r1

uv run ffb health evaluate \
  reports/generated/m4-health-fit-primary-a829a9f-r1 \
  --output-dir reports/generated/m4-health-eval-primary-a829a9f-r1

uv run ffb health evaluate \
  reports/generated/m4-health-fit-primary-a829a9f-r1 \
  --output-dir reports/generated/m4-health-eval-repeat-a829a9f-r1
```

Each artifact passed its strict CLI reload:

```bash
uv run ffb health bundle fit validate <fit>
uv run ffb health bundle evaluation validate <evaluation> \
  --fit-artifact reports/generated/m4-health-fit-primary-a829a9f-r1
```

## Repeat result

| Phase | Primary artifact | Repeat artifact | Indexed comparisons | Mismatches |
|---|---|---|---:|---:|
| Fit | `abd1540f…608ee` | `abd1540f…608ee` | 7 | 0 |
| Evaluation | `2d9ee626…c913f` | `2d9ee626…c913f` | 9 | 0 |

Only volatile `run.json` and `_SUCCESS` envelopes differed. The strict
[`repeat-verification.json`](m4-health-v0.1.0/repeat-verification.json)
recomputes member equality, normalized run identity, environment equality,
path/inode independence, and resource gates.

## Resource gates

| Run | Wall time | Limit | Peak RSS | Limit | Status |
|---|---:|---:|---:|---:|---|
| Primary fit | 279.62 s | 1,800 s | 121,585,664 B | 1,073,741,824 B | Pass |
| Repeat fit | 286.17 s | 1,800 s | 123,486,208 B | 1,073,741,824 B | Pass |
| Primary evaluation | 592.01 s | 1,800 s | 169,869,312 B | 1,073,741,824 B | Pass |
| Repeat evaluation | 559.82 s | 1,800 s | 168,116,224 B | 1,073,741,824 B | Pass |

The resource records are raw-log-backed operator measurements, not
cryptographic proof of process execution.

## Curation and validation

```bash
uv run python tools/m4_release.py build-release \
  --official-fit reports/generated/m4-health-fit-primary-a829a9f-r1 \
  --repeat-fit reports/generated/m4-health-fit-repeat-a829a9f-r1 \
  --primary-evaluation reports/generated/m4-health-eval-primary-a829a9f-r1 \
  --repeat-evaluation reports/generated/m4-health-eval-repeat-a829a9f-r1 \
  --primary-fit-time-l \
    reports/generated/m4-health-fit-primary-a829a9f-r1.time-l.txt \
  --repeat-fit-time-l \
    reports/generated/m4-health-fit-repeat-a829a9f-r1.time-l.txt \
  --primary-evaluation-time-l \
    reports/generated/m4-health-eval-primary-a829a9f-r1.time-l.txt \
  --repeat-evaluation-time-l \
    reports/generated/m4-health-eval-repeat-a829a9f-r1.time-l.txt \
  --output-dir reports/releases/m4-health-v0.1.0

uv run python tools/m4_release.py validate-release \
  reports/releases/m4-health-v0.1.0
```

Validation returned release artifact
`f41dc1a10c0e5ac940e2760baee159a911f95446345dc70c3fb1a5afa8d325be`.
The curated tree is `7,108,875` bytes (`6.8 MiB`) against the 50 MiB cap.

## Evidence limitations

- The public release retains every aggregate but omits sequence rows.
  Commitment validation cannot independently recompute bootstrap inference.
- Repeat evidence establishes byte equality and internal provenance
  consistency, not cryptographic proof of two executions.
- The exact measurements apply to the named machine and revision; no
  cross-architecture byte-identity or throughput claim is made.
- Public CI verifies the tracked package, tests, and strict curated-release
  reload without dataset access; it does not rerun the complete named-CPU
  release benchmark.
