# M5 adversarial plan review

Status: **PASS after revision**.

Scope: replay semantics, full-frame geometry, calibration and timing causality,
health-feature leakage, support, sequence-first statistics, persistence labels,
selection, identity binding, M4 non-mutation, resources, privacy, licensing,
and claim boundaries.

Two independent adversarial agents reviewed the M5 plan before implementation
or inspection of replay descriptors or fault outcomes. Reviewer identity is
operator-recorded and is not cryptographically authenticated.

Frozen files:

- plan byte SHA-256:
  `7779783ccd1bb6e71400cc6d39301e1c96c5ee81fa92e3978917cb3879a97388`;
- machine-intent byte SHA-256:
  `d465a4b57de8af0c390395026e150c36922a9e44f7f09dafe9b85534808ccc0c`;
  and
- machine-intent canonical SHA-256:
  `d429e36e2ce17ec8628c9bad4b5051fd54e0d88bcdeb966d112972e4c3dc2836`.

## Initial blockers and resolutions

1. **The health-frame transform was incomplete.** The final contract retains
   each full 3D modality reconstruction, applies the complete recorded ego
   \(SE(3)\), then projects global XY into a yaw-anchored monitoring frame.
   Exact modality- and severity-specific Jacobians push reported covariance
   forward. A stationary-global-object roll/pitch/yaw/translation oracle
   blocks early XY truncation.
2. **Variable scene length and zero-object frames could not reuse the fixed
   M4 schemas literally.** M5 now declares replay-only schedule, frame,
   evidence, event, and result versions. Empty numeric support, direct-evidence
   priority, history updates, latch/action timing, occupancy, dynamic censor
   bounds, and exact step/elapsed-time endpoints are frozen. Released M4
   contracts remain unchanged.
3. **Replay identity was ambiguous.** Every local row and curated member binds
   an exact canonical replay-identity envelope containing the replay-intent
   digest, panel, frozen source digest, and experiment/condition ID.
   Completeness validation rejects missing, extra, duplicate, or mismatched
   identities without rewriting an M3 v1alpha1 manifest.
4. **RNG prose omitted a machine-declared stream.** The latent stream is now
   explicitly declared but unconsumed because recorded metadata supplies the
   replay latent state.
5. **Availability and conditional support were underspecified.** Zero-support
   scenes remain in the bootstrap. Pooled coverage and conditional loss have
   exact point denominators, undefined-replicate behavior, and a strict
   greater-than-97.5% defined-replicate gate. Equal-scene metrics never become
   reduced-scene averages.

## Exact-file follow-up corrections

- Direct-only empty-frame evidence remains update-eligible from availability;
  only numeric-dependent decisions hold on healthy direct evidence.
- Detection, attribution, recovery, first-missing, and signed dropout-response
  latency have exact frame and timestamp equations.
- The scene-frame copy is monitoring-only. Camera/LiDAR/fusion/oracle outputs
  and matched-center loss remain in current LiDAR-time ego BEV.
- M5-A uses the exact full-chain \(A C A^\top\) covariance projection, reducing
  to released M3 behavior on its planar source.
- Identity keys follow the repository canonical serializer rather than textual
  key order.
- Availability's pooled metrics are explicit exceptions to equal-scene
  zero-window-support rules.
- H5-B1 through H5-B4 bind the exact method, metric, window, unit, and condition
  selectors from the released M4 comparisons.
- Machine and prose rules agree on strict signs, scene counts, LOSO/LOLO,
  undefined support, and the nonpositive H5-B3 control.

## Final verdict

Both reviewers reread the frozen plan and machine intent, independently
verified the byte hashes, canonical digest, parseability, and frozen status,
and returned **PASS with no remaining P0/P1 blocker**.

This verdict approves only the preregistered implementation target. It does not
approve an implementation, result, persistence claim, dataset-authentication
claim, or release.
