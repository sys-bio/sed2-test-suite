# Notes for Claude

## Never hide a problem; expose it

One of the main goals of this project is to improve the field of simulators: to find bugs and fix them wherever
they are, in the simulators, in libsed2, in the SED2 specification, and in the translator that runs these tests.
A suite that is green because problems were covered up defeats that goal.

* A disagreement between backends, or between a backend and the expected results, is a finding.  Do not loosen a
  tolerance, drop a backend from a case's `backends` list, change expected results, or edit a case to make it pass
  unless that is the correct resolution and the reason is written down.
* Record each finding in `disagreements.json` (format: `docs/FORMATS.md`, check with
  `python -m sed2suite.disagreements --check`): the test, the backends, the size of the difference, the diagnosis
  (solver setting, translator bug, simulator bug, spec ambiguity) and the resolution.  Say plainly when it is a bug
  in someone else's software, and how it was checked.
* When a case has to be changed to avoid a known problem, keep what the case is about intact, say in the log entry
  and the case's description what was changed and why, and keep the problematic variant documented.
* Expected results come from hand calculation or from simulators that agree; `provenance` in `settings.json` says
  which.  Never record results from a simulator as canonical just because it runs; check them.
* Specification and libsed2 questions belong in the translator repository's `GAPS.md`; the SED2 author makes the
  changes.

## Standing rules

* **Do not commit or push** anything, in any repository, unless asked to.  Leave changes in the working tree.
* **libsed2 and the SED2 specification belong to someone else.**  Report gaps in the translator repository's
  `GAPS.md`; never work around them here.  Specification changes are made by the user; propose them, do not
  make them.
* **ASCII only** in files (cases, documents, data, source, documentation).
* **Case numbers** are assigned in order of addition, from `00001`, and are never reused or renumbered.
* **Ask only when blocked.**  Make the call and say what was decided, unless a gap or a real question about
  intent stops the work.
* **Do not retry a rate-limited site** (for example EBI OLS4 answers 429): note it and carry on without it.
* **Checks before calling work done:** `python -m pytest`, `python -m sed2suite.validate_suite`,
  `python generate_tags.py --all --check`, `python -m sed2suite.make_inputs --all --check` and
  `python -m sed2suite.disagreements --check`.
