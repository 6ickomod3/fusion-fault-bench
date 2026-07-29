# M3 adversarial results review — m3-procedural-v0.1.0

Verdict: **PASS**

Artifact set: `a870f05a372f727a2dbca432079299c58b3bea04a0053d1fc923ebf232f95cef`

Scientific revision: `e8595fe428bcb9dfb269069e4b02972aff10f4ee`

Matrix SHA-256: `7162a01bb766a38e8f7196cba9eb2c6c546f5866bc640f220252bb6e7aab6f9b`

Public CI: [workflow `ci`, run 30456056647](https://github.com/6ickomod3/fusion-fault-bench/actions/runs/30456056647)

## Review scope

An independent adversarial agent reviewed semantics, geometry, leakage,
statistics, result selection, claims, privacy, resources, and provenance. The
review used both complete local run roots and the persisted repeat evidence.
It did not edit the source, derive the official identity, or build the public
release.

The reviewer independently:

- strict-reloaded and rebuilt both complete artifact roots;
- recomputed all 429 aggregates and pointwise intervals from all 71,700
  sequence rows;
- recomputed all 10 PAVA/bootstrap crossover records;
- checked every validation bundle, repeat pair, run record, and completion
  marker;
- inspected all outcomes, including undefined and right-censored results; and
- audited generated roots for absolute user paths, nuScenes references,
  credentials, and private material.

No P0 or P1 blocker remained.

## Evidence gates

- All eight per-experiment validation bundles passed.
- All 303 analytic expected-loss checks passed; the maximum absolute
  standardized error was `2.079942534683295`, below the preregistered limit of
  six.
- All 112 empirical moment checks passed.
- Cross-manifest identity passed all 5,800 comparisons with zero discrepancy.
- All 48 indexed scientific members were byte-identical between the two roots.
- All eight corresponding volatile run-record digests differed, as required.
- Independent aggregate reconciliation differed by at most
  `4.34e-19` from floating-point alpha rounding; crossover discrepancy was
  zero.
- The retained statuses were 427 `ok` aggregates, two intentionally
  `undefined` dropout aggregates, nine observed crossovers, and one
  not-observed crossover.

## Crossover findings

All intervals below are pointwise 95% paired complete-sequence bootstrap
intervals. These are conditional procedural stress-test results, not physical
sensor tolerances.

| Fault and direction | Crossover result |
|---|---|
| LiDAR y-bias, negative | `1.3968849130460446 m` `[1.389999281805564, 1.4037114600964853]` |
| LiDAR y-bias, positive | `1.3942566175140465 m` `[1.3874881459452584, 1.4008480010938364]` |
| Camera noise, correctly reported | Not observed through `4×`; `[4, +∞)`, crossing fraction `0` |
| Camera noise, underreported | `1.4475323333484358×` `[1.4258107359522219, 1.468957584519013]` |
| Camera calibration x, negative | `1.383889433246956 m` `[1.345955729458017, 1.4209496094435923]` |
| Camera calibration x, positive | `1.4237279263474245 m` `[1.3806381804922199, 1.4627830366308743]` |
| Camera yaw, negative | `0.03521127235349965 rad` `[0.03409377684383041, 0.036400627447693006]` |
| Camera yaw, positive | `0.03408296916867806 rad` `[0.032988505139442084, 0.03507853034101522]` |
| Timestamp offset, negative | `0.3535791578648241 s` `[0.3417235221585074, 0.3662034173717133]` |
| Timestamp offset, positive | `0.36708164425734324 s` `[0.3527324507856683, 0.3800755226303754]` |

Every observed crossover had bootstrap crossing fraction `1.0`. All ten raw
point curves were already nondecreasing, so point-level PAVA required no
corrective pooling.

## Required counterexamples and controls

At `4×` actual camera noise, correctly reported covariance kept the signed
fusion delta below zero at `-0.0010478879078088413 m²`; underreported
covariance produced `+0.1897431948470576 m²`. Covariance-aware downweighting,
not noise magnitude alone, determines whether fixed fusion remains beneficial
in this declared model.

The fault-target-drop policy is not universally beneficial. It discards useful
fusion below crossover and throughout the correctly reported-noise sweep. It
uses the known injected target and is diagnostic, not deployable. The
performance oracle uses complete-sequence hindsight and is also not deployable.

For camera dropout probabilities `[0, .1, .25, .5, .75, 1]`, camera-only and
fixed-fusion coverage was
`[1, .896875, .7467708333333334, .5001041666666667, .241875, 0]`.
LiDAR-only and target-drop coverage remained `1`. At complete dropout,
camera-only and fixed-fusion conditional loss was undefined and was never
zero-imputed. Dropout has no crossover estimand.

Under the common-mode edge control, fixed-fusion MSE rose from
`0.1650957575247963 m²` at identity to `16.15591270574043 m²` at `-4 m`
and `16.174278809309165 m²` at `+4 m`. Camera-LiDAR disagreement nevertheless
changed by at most `7.105427357601002e-15 m`, below the `1e-12 m` gate. This
is the intended agreement-based health blind spot and has no healthy-reference
crossover.

## Runtime, provenance, and privacy

The two complete-matrix measurements on the named Apple M3 Pro environment
were:

- `1218.376157041639 s` wall and `369000448` bytes peak RSS; and
- `1258.861101000104 s` wall and `389431296` bytes peak RSS.

These resource observations are self-reported by the tracked `wait4` driver
and are not independently recomputable from the curated package. Distinct
paths, inodes, commands, volatile records, and completion markers are
consistency evidence, not cryptographic proof of execution.

GitHub Actions run `30456056647` was observed successful for workflow `ci` on
the scientific revision, including the full test suite, two-run M3 smoke,
distribution build, and wheel smoke. The tracked CI attestation is an
operator-recorded external reference; the offline release validator does not
query GitHub.

Generated roots are ignored. The audit found no absolute user path, dataset
payload reference, credential, or private interview material in the reviewed
artifacts. The public release omits all 71,700 sequence rows and retains their
content commitments.

## Claim boundary

The supported statement is that a deterministic CPU estimator-output benchmark
measured matched-center fusion behavior under declared procedural geometry and
proxy metadata faults.

The evidence does not establish physical fault tolerances, detector behavior,
real sensor-noise transfer, nuScenes persistence, health-aware deployable
fallback, planning or safety benefit, production readiness, or fleet
generalization. Intervals are pointwise, not simultaneous.

The reviewer identity and review process are not cryptographically
authenticated by the offline validator; the exact tracked report and
artifact-set binding are the review boundary.
