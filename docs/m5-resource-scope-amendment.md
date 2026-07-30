# M5 outcome-blind resource-scope clarification

Status: **preregistered clarification; M5 outcomes remain uninspected**.

Date: 2026-07-29.

This note clarifies, without changing, the frozen M5 scientific intent bound by:

- machine-intent byte SHA-256
  `d465a4b57de8af0c390395026e150c36922a9e44f7f09dafe9b85534808ccc0c`;
- machine-intent canonical SHA-256
  `d429e36e2ce17ec8628c9bad4b5051fd54e0d88bcdeb966d112972e4c3dc2836`;
  and
- plan byte SHA-256
  `7779783ccd1bb6e71400cc6d39301e1c96c5ee81fa92e3978917cb3879a97388`.

The machine field `resource_caps.cpu_processes: 1` and the plan phrase “one CPU
process” mean exactly one **scientific replay worker** with no benchmark
multiprocessing. They do not prohibit sequential helper processes needed for
environment discovery, clean-source provenance checks, Git inspection, or
external resource measurement. Helpers do not evaluate scenes or contribute
scientific rows.

The primary and repeat complete replay CLI lifetimes are each wrapped by Darwin
`/usr/bin/time -l`. The imported external measurements are operator-recorded
self-reports, not independent resource attestations. The public evidence binds
the raw-log digest and byte length, parsed values, command, environment, and
corresponding local artifact/run commitments without serializing the raw-log
path.

This scope clarification was written before M5 dataset execution or inspection
of replay descriptors, fault outcomes, acceptance gates, or resource results.
It changes no hypothesis, estimand, threshold, experiment identity, dataset
selection, fault schedule, statistical method, or release gate.
