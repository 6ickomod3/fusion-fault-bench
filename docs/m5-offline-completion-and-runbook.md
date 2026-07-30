# M5 offline completion status and authoritative-run runbook

This document records exactly what was completed for M5 in a **network-isolated**
environment and gives the precise commands to finish the **authoritative,
tagged** release on a machine with GitHub access. It changes no scientific
intent, estimand, or claim; it is a process/handoff record.

## 1. What was completed offline (and verified)

1. **The in-flight M5 revision is green.** `ruff format --check`, `ruff check`,
   `pyright` (0 errors), and `pytest` all pass with coverage ≥ 90%. Two defects
   in the in-flight work were fixed:
   - the "dashboard = 9th closeout document" feature was completed by re-pinning
     the frozen methodology digest (`_FROZEN_METHODOLOGY_SHA256`, both copies) to
     the amended pre-outcome plan bytes;
   - a pre-existing runner defect was fixed: local NDJSON members were reloaded
     with strict **Python-mode** validation, which rejects JSON arrays for
     `tuple[int, int]` schedule fields; the reload now uses JSON mode
     (`model_validate_json` / `validate_json`), matching the release loaders.
     This defect only surfaces on real data and would otherwise have burned the
     no-retry authoritative revision.
   A P2 hardening from the independent review was applied: a direct byte-binding
   test for the security-critical `replay_release._FROZEN_METHODOLOGY_SHA256`.

2. **The real replay runs and is deterministic.** `ffb replay run` was executed
   twice on the user-provided all-ten-scene nuScenes-mini tree; `ffb replay
   verify-repeat` PASSED (two separately-timed runs are scientifically
   byte-identical). Named CPU: **Apple M4 Max**; peak RSS 270 MB (< 1 GiB cap);
   elapsed ~163 s (< 1800 s cap); `raw_sensor_payload_reads = 0`; no GPU/torch.
   These artifacts are **exploratory / non-authoritative** (they did not pass the
   network-gated authoritative lifecycle) and live under gitignored
   `reports/generated/`.

3. **Exploratory scientific result (both panels persist on recorded geometry):**
   - **M5-A**: 9 of 10 predeclared crossovers observed (all q = 2000/2000); the
     correctly-reported camera-noise control correctly did **not** cross
     (q = 9/2000). Mirrors the M3 procedural finding.
   - **M5-B** (apply-only M4 rule): large positive event-window gains for LiDAR
     output y-bias (~+3.6 m²) and LiDAR timestamp-offset (~+5.7 m²); the M4
     counterexample transfers — underreported LiDAR noise gives **−0.55 m²** gain
     despite 100 % detection; common-mode remains a blind spot; small clean-
     condition regression.

4. **Independent adversarial reviews: both permit-release, zero P0/P1.** See
   [`docs/reviews/m5-exploratory-implementation-review.md`](reviews/m5-exploratory-implementation-review.md)
   and [`docs/reviews/m5-exploratory-results-review.md`](reviews/m5-exploratory-results-review.md).

## 2. Why the authoritative release cannot complete in this environment

The frozen pipeline (`m5-release-pipeline-plan.md`) requires live network to
GitHub, which is blocked here (SSH :22 and HTTPS :443 both denied):

- every `run-replay` preflight proves **live upstream equality** via
  `git ls-remote` (`replay_release_workflow.py` `_require_upstream_sync`, called
  from `_execution_authority`);
- closeout requires the release commit to **pass GitHub CI** and an annotated
  **tag pushed** to the exact commit (acceptance gates 2, 15, 16).

None of these can be satisfied offline. The authoritative steps below must be
run on a machine that can reach the `origin` remote.

## 3. Authoritative runbook (run on a networked machine)

Preconditions: the final revision (this branch, including the fixes above) is
**committed and pushed**, `HEAD == @{upstream}`, and the working tree is clean.
Let `REV` be that commit SHA and `NUSCENES_ROOT` the user's mini tree
(here: `/Users/jdai/Interview/FaultyBench/datasets/nuscenes`).

```bash
# 0. Environment
git status --short --untracked-files=all      # must be empty
uv lock --check && uv sync --locked --group dev
export NUSCENES_ROOT=<user-provided-nuscenes-root>
export UV_CACHE_DIR=<private-tmp>/ffb-m5-uv-cache-$REV && mkdir -m 700 "$UV_CACHE_DIR"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
       MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# 1. Independent whole-revision implementation review -> attest -> commit + push
#    (author the report; the tool only canonicalizes the reviewer disposition)
uv run --frozen --no-sync python tools/m5_release.py attest-implementation-review \
  --review-report docs/reviews/m5-release-implementation-review.md \
  --decision reports/generated/m5-release-implementation-review-decision.json \
  --output docs/reviews/m5-release-implementation-review-attestation.json
#    commit + push the report + attestation (no implementation-snapshot change)

# 2. Software verification (offline-capable; requires step 1's attestation)
uv run --frozen --no-sync python tools/m5_release.py verify-software \
  --output reports/generated/m5-software-verification-$REV.json

# 3. Authoritative replay: exactly one primary r1 + one repeat r1 (NO retry)
uv run --frozen --no-sync python tools/m5_release.py run-replay --run-label primary \
  --output-dir reports/generated/m5-replay-primary-$REV-r1 \
  --time-l-output reports/generated/m5-replay-primary-$REV-r1.time-l.txt
uv run --frozen --no-sync python tools/m5_release.py run-replay --run-label repeat \
  --output-dir reports/generated/m5-replay-repeat-$REV-r1 \
  --time-l-output reports/generated/m5-replay-repeat-$REV-r1.time-l.txt

# 4. Review candidate (34 files) + validate
uv run --frozen --no-sync python tools/m5_release.py prepare-review \
  --primary-artifact reports/generated/m5-replay-primary-$REV-r1 \
  --repeat-artifact  reports/generated/m5-replay-repeat-$REV-r1 \
  --primary-time-l   reports/generated/m5-replay-primary-$REV-r1.time-l.txt \
  --repeat-time-l    reports/generated/m5-replay-repeat-$REV-r1.time-l.txt \
  --software-verification reports/generated/m5-software-verification-$REV.json \
  --output-dir reports/generated/m5-review-candidate-$REV
uv run --frozen --no-sync python tools/m5_release.py validate-review-candidate \
  reports/generated/m5-review-candidate-$REV

# 5. Independent results-and-claims review of that exact candidate -> attest
uv run --frozen --no-sync python tools/m5_release.py attest-results-review \
  --candidate reports/generated/m5-review-candidate-$REV \
  --review-report reports/generated/m5-results-review-$REV.md \
  --decision reports/generated/m5-results-review-decision-$REV.json \
  --output reports/generated/m5-results-review-attestation-$REV.json

# 6. Final release package (41 files) -> sync review copies -> validate
uv run --frozen --no-sync python tools/m5_release.py build-release \
  --candidate reports/generated/m5-review-candidate-$REV \
  --results-review reports/generated/m5-results-review-$REV.md \
  --results-review-attestation reports/generated/m5-results-review-attestation-$REV.json \
  --primary-artifact reports/generated/m5-replay-primary-$REV-r1 \
  --repeat-artifact  reports/generated/m5-replay-repeat-$REV-r1 \
  --primary-time-l   reports/generated/m5-replay-primary-$REV-r1.time-l.txt \
  --repeat-time-l    reports/generated/m5-replay-repeat-$REV-r1.time-l.txt \
  --software-verification reports/generated/m5-software-verification-$REV.json \
  --output-dir reports/releases/m5-nuscenes-replay-v0.1.0
uv run --frozen --no-sync python tools/m5_release.py sync-reviewed-evidence \
  --release reports/releases/m5-nuscenes-replay-v0.1.0 \
  --review-report-output docs/reviews/m5-results-review.md \
  --review-attestation-output docs/reviews/m5-results-review-attestation.json
uv run --frozen --no-sync python tools/m5_release.py validate-release \
  reports/releases/m5-nuscenes-replay-v0.1.0
uv run --frozen --no-sync ffb replay bundle validate \
  reports/releases/m5-nuscenes-replay-v0.1.0/artifact

# 7. Closeout: 9 projection docs, offline validators, commit, CI, tag
#    (fill the frozen projection markers, then:)
env -u NUSCENES_ROOT UV_OFFLINE=1 uv run --frozen --no-sync \
  python tools/m5_release.py validate-release reports/releases/m5-nuscenes-replay-v0.1.0
env -u NUSCENES_ROOT UV_OFFLINE=1 uv run --frozen --no-sync \
  python tools/m5_release.py validate-publication \
  --release reports/releases/m5-nuscenes-replay-v0.1.0 --source-root .
#    push the release commit; wait for GitHub CI to pass on that exact commit;
#    then create and push the annotated tag m5-nuscenes-replay-v0.1.0.
```

The expected authoritative outcome matches the exploratory result in §1.3 (up to
run-to-run-stable values); a negative/undetermined result would not by itself
block release, but privacy leakage, favorable selection, review blockers, or an
unauthenticated package would.
