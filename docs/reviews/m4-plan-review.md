# M4 adversarial plan review

Status: **PASS after revision**.

Scope: semantics, temporal causality, leakage, covariance, statistics,
selection, support, oracle boundaries, controls, resources, privacy, and
claims.

The review was performed before M4 implementation. The reviewer first returned
`REVISE` with five release blockers and ten major issues. The final
machine-readable intent has canonical SHA-256
`391d41a8c4b1ec191649e8a86fc43ee32b03849b5e17b658f2b3e9c5b1cc9ae7`.

## Initial blockers and resolutions

1. **M3's sequence oracle was not a ceiling for a frame-switching policy.**
   M4 now versions a frame-action oracle at the same action granularity and
   tests dominance on identical support. M3 semantics remain unchanged.
2. **M3 persistent manifests could not be reinterpreted as transient events.**
   M4 has a new event intent and will use new fit/evaluation contracts.
3. **Train/validation/test and held-out claims were underspecified.** The
   intent now enumerates clean fit, every validation/test target and value,
   held-out camera yaw, edge, motion-mismatch, common-mode, and cold-start
   controls. Test is apply-only through a content-addressed fit.
4. **Ambiguous/missing/abstaining actions could enable coverage gaming.** A
   complete action table now separates raw label, latch, and execution, and
   requires coverage/undefined rate before conditional loss.
5. **Repeated variants could be treated as independent.** One base-sequence
   bootstrap index carries every condition and method variant.

## Major corrections

- Predictor history, reference-time clock, historical covariance propagation,
  directional cross-NIS equations, maturity, and post-score updates are exact.
- Strict ECDF rank `count(v<x)/n`, tie handling, and strict alarms make
  threshold 1.0 a real no-numeric-alert candidate.
- A typed observable input structurally excludes truth, velocity, labels,
  severities, split, seed, event phase, and manifest metadata.
- The literal latch transition table fixes when the second alert and third
  healthy frame affect the current action.
- Attribution and downstream action utility are separate estimands.
- Generic camera/LiDAR faults are mirrored, and validation utility weights
  targets before families.
- Direct timestamp/dropout evidence is published separately from self/cross
  NIS.
- Detection, early clear, final-event state, recovery denominators, occupancy,
  and false-alert episode starts prevent latency censoring from looking
  favorable.
- A bounded-acceleration clean control supplements the edge support shift.
- Feature caching and explicit CPU/memory/time/artifact caps bound the matrix.

## Final follow-up

The follow-up review found no remaining P0 blocker. Its five P1 requests were
all made literal in the frozen plan:

1. frame-reference predictor clocks and both directional cross equations;
2. separate raw label, latched state, and executed action;
3. complete event censoring, early-clear, recovery, and false-alert rules;
4. mechanically unique threshold iteration, feasibility, tie-break, and
   no-feasible behavior; and
5. every matrix value plus the frame-oracle support equation and dominance
   gate.

An exact-file audit then found seven remaining machine/prose gaps. The revised
intent now adds the literal `validation-main-clean` feasibility condition;
explicit insufficient-support and activation-counter semantics;
schedule-specific cold-start loss windows; paired-common-support dropout
contrasts; a first-latch event-outcome reduction with targetless common-mode
reporting; realized-dropout denominators; and literal acceleration and
fault-target-drop operators.

The final exact-file reread returned **PASS**: all seven findings are resolved,
machine intent and prose agree, and no P0/P1 preregistration blocker remains.

Verdict: **PASS for preregistration freeze.**

This pass approves only the plan for implementation, not an implementation,
result, or hypothesis.
