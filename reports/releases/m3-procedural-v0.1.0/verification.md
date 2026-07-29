# Verification — m3-procedural-v0.1.0

Run from the repository root in the locked `.venv`:

```bash
uv run python tools/m3_release.py validate-release reports/releases/m3-procedural-v0.1.0 --official-identity examples/release-identities/m3-procedural-v0.1.0.json
```

The validator fails closed unless all of the following hold:

- the exact `m3-procedural-v1` matrix and canonical digest are present;
- the external Git-bound official identity exactly matches the included copy;
- the exact Git-bound results-review report matches its included copy and
  content-addressed review attestation;
- all eight matrix entries occur once in frozen order;
- exactly 429 aggregate and 10 crossover rows are present in source order;
- the omitted 71,700 sequence rows retain exact source hashes, byte lengths,
  and independently derived counts;
- every curated member matches its source payload index and repeat pair;
- payload indexes, artifact digests, matrix evidence, both run records,
  run IDs, and both completion markers form one identity graph;
- all sixteen runs share the pinned source, lock, package, full environment,
  named CPU, clean-success state, and exact logical command;
- generated summaries, documents, and all three SVG figures reproduce byte
  for byte; and
- the canonical release index exactly hashes and allowlists every member.

Frozen official identity:

- source revision: `e8595fe428bcb9dfb269069e4b02972aff10f4ee`
- artifact set: `a870f05a372f727a2dbca432079299c58b3bea04a0053d1fc923ebf232f95cef`
- matrix evidence SHA-256: `936552ed518b49db787f09796c74bf3ecd59d23c8b84afedf2f9e68822b946d1`
- repeat evidence SHA-256: `3677bd2df40635c915315cebbe87b73d170a5c901692fc9972b69f32ed3fec4a`

Completeness facts:

- experiments: 8
- omitted sequence rows: 71700
- curated aggregate rows: 429
- curated crossover rows: 10
- repeat member comparisons: 48
- repeat member mismatches: 0

The release intentionally does not contain `sequence-metrics.ndjson`.
Therefore standalone public validation cannot recompute aggregate values
from sequence rows. Regenerate both source roots from the frozen matrix to
repeat that computation.

For a full fresh-clone regeneration, check out the scientific source
revision above, create the locked environment, and run:

```bash
uv run python tools/m3_release.py execute examples/matrices/m3-procedural-v1.json \
  --first-output-dir reports/generated/m3-reproduction-first \
  --second-output-dir reports/generated/m3-reproduction-second \
  --evidence-dir reports/generated/m3-reproduction-evidence
uv run python tools/m3_release.py validate examples/matrices/m3-procedural-v1.json \
  --first-output-dir reports/generated/m3-reproduction-first \
  --second-output-dir reports/generated/m3-reproduction-second \
  --evidence-dir reports/generated/m3-reproduction-evidence
```

Compare the regenerated artifact-set digest and indexed scientific-member
hashes with `evidence/official-identity.json`. Volatile run-record hashes,
wall time, and RSS are expected to differ. Byte-identical scientific
members were demonstrated for the two named runs on the named locked CPU
environment; cross-architecture byte identity is not claimed.

Resource and execution authenticity boundary:

- `self-reported-by-tracked-wait4-driver-not-independently-recomputable`
- `distinct-path-inode-and-run-record-consistency-not-cryptographic-proof`

Those literal scopes mean elapsed time and RSS are preserved observations,
not independently reproducible facts, and the consistency controls do not
cryptographically prove two executions. Git history binds the tracked
driver and official identity. CI smoke is not M3 release evidence.

Machine-bound review and CI attestations:

- public CI run `30456056647` concluded `success` for `e8595fe428bcb9dfb269069e4b02972aff10f4ee`;
- results review `pass` covers artifact set `a870f05a372f727a2dbca432079299c58b3bea04a0053d1fc923ebf232f95cef` at `docs/reviews/m3-results-review.md`.

These are content-addressed, Git-bound declarations. The offline validator
checks their exact contents and links, but does not query GitHub or
cryptographically authenticate the human or agent reviewer.
