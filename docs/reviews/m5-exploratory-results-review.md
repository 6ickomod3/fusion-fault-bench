# M5 exploratory results-and-claims review

Status: **independent adversarial review of the M5 scientific findings from the
offline exploratory replay**. Reviewer identity scope:
`operator-recorded-not-cryptographically-authenticated`.

This is a checkpoint review of an **exploratory, non-authoritative** replay
(produced with `ffb replay run`, not the authoritative `run-replay` lifecycle,
which is network-gated). It is not the independent results-and-claims review
required by `docs/m5-release-pipeline-plan.md` §9, which binds the exact review
candidate on the final pushed revision. It records that the findings are sound
so the authoritative reviewer can confirm them against the released candidate.

## Provenance of the reviewed artifact

- Source: all ten nuScenes-mini scenes (metadata only), local user-provided tree.
- Scientific revision: `e99c097db967c82399b703272e6a77dc747c8a78`
  (`source_dirty=false`, lockfile `ac20e739…`).
- Run id: `run:3adb501e…`; primary artifact `42ba30fb…`, repeat `a8a6ec1b…`;
  `verify-repeat` PASSED (`verification_sha256=6beec1c8…`,
  `profile_sha256=184f9e09…`).
- Named CPU: **Apple M4 Max**. `cuda_used=false`, `gpu_used=false`,
  `torch_imported=false`, `raw_sensor_payload_reads=0`, peak RSS
  `269,991,936` B (< 1 GiB cap), elapsed `162.9 s` (< 1800 s cap),
  `scientific_replay_worker_count=1`.

## Disposition: permit-release (0 P0, 0 P1, 3 P2)

Every headline number was independently re-derived from the raw NDJSON.

### M5-A — persistent crossover panel (CONFIRMED)

All 10 predeclared crossovers present: **9 observed, 1 not-observed**. The
correctly-reported camera-noise case is a genuine **non-crossing control**
(`bootstrap_crossing_count 9/2000`, `status=not-observed`, point `null`,
right-censored `[4.0, +inf]`; its fixed-fusion fused-minus-healthy stays
negative through 4×). Its sibling — underreported camera noise — **does** cross
(~1.45 std-scale), the physically correct contrast; labels are not swapped. Sign
convention holds throughout (fused − healthy < 0 near severity 0 = fusion helps;
> 0 at high severity = fusion harmful). Observed crossovers: LiDAR y-bias
(~1.38 / 1.40 m), camera calibration-x (~1.39 / 1.43 m), camera calibration-yaw
(~0.040 / 0.046 rad), camera timestamp-offset (~0.23 / 0.28 s), underreported
camera noise (~1.45). Every observed point estimate lies inside its interval;
observed cases are 2000/2000. **The M3 fusion-harm pattern persists on recorded
latent geometry.**

### M5-B — health-transfer panel (CONFIRMED on all four sub-claims)

Combined-health-gate event-window policy gain vs fixed fusion (m², equal-scene-mean):

- **(a) large positive gains** for LiDAR output y-bias (+3.63 / +3.69 m²) and
  LiDAR timestamp-offset (+5.63 / +5.81 m²), CIs excluding zero;
- **(b) the key counterexample transfers and is retained**: underreported LiDAR
  noise yields **−0.5459 m²** (CI [−0.754, −0.392], excludes zero) with
  `detection-fraction = 1.0` — detection without benefit, matching the M4
  finding (M4 was −0.5786 m²; the replay value differs as expected for recorded
  geometry, not from refitting);
- **(c) common-mode is a blind spot** (all four gains negative, −0.076 to −0.363);
- **(d) a small clean-condition regression** (−0.083, CI touching zero).

The M4 rule is provably **apply-only**: `replay_fit.py` hard-pins
`M4_FROZEN_CALIBRATION_SHA256` (byte-matches the artifact's
`fit_calibration_sha256`) with frozen thresholds (self 0.999 / cross 0.995) and
rejects any fit differing from the released M4 evidence.

### Integrity and retention

All 392 persistent and 7,654 health equal-scene-mean `ok` rows re-derive
exactly from their scene numerator/denominator arrays (0 mismatches); all 14,144
numeric intervals bracket their estimates (0 violations). Negative, undefined,
and not-applicable results are retained (987 undefined + 319 not-applicable in
the health panel; full-dropout severity-1 rows carried as `undefined` /
`null`, never zero-imputed). The claim boundary in the docs is explicit and
conservative and disavows raw-sensor robustness, detector performance, physical
tolerance, fleet generalization, planning/collision, production readiness, and
safety.

## Findings (all P2, all already acknowledged by the design)

- **P2-1** — the 10 scenes derive from only 8 log-groups (log-group 06 backs 3
  scenes); the equal-scene-mean bootstrap treats scenes as independent, mildly
  narrowing nominal CIs. Already disclosed: results are scoped to "finite
  scene/log-group sensitivity", CIs are stated as not fleet-population intervals,
  and leave-one-cluster rows ship. Keep public narrative anchored to the
  log-group-sensitivity caveat.
- **P2-2** — LiDAR timestamp-offset gain is sign-robust but magnitude-uncertain
  (CI [1.4, 10.6] m²). Public phrasing should emphasize sign/persistence, not
  the point magnitude.
- **P2-3** — byte-level determinism is asserted by design and internally
  consistent (seed 1729, uniform 2000 replicates, content-addressed `_SUCCESS`),
  and `verify-repeat` PASSED across two separately timed runs, but the reviewer
  did not re-execute end-to-end; rely on the pipeline's repeat-verification gate.

## Note for the authoritative reviewer

These findings are from the exploratory run at `e99c097`. The authoritative
results-and-claims review must bind the exact review candidate built on the
final pushed revision per `m5-release-pipeline-plan.md` §9.
