# Notes for Claude

## Who is who

The user is the author of Antimony and a maintainer of libSBML, roadrunner and Tellurium.  The user and Claude are the authors of everything SED2 related: SED2 (the specification and libsed2), pySED2Translate, sed2-test-suite and pbg2sed2.  These projects are developed in tandem, and more will be added in the coming months, so a change in one often has a consequence in the others.  A defect found in any of them is ours to fix, in the right repository; one in roadrunner, Tellurium, libSBML or Antimony is the user's own project (say which project it is in and record it as usual).  COPASI, OpenCOR, process-bigraph, the viva wrapper packages and every other project are other people's software.

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
* Specification and libsed2 questions belong in `SED2/TODO.md` in the SED2 repository (a new section at the end; change
  nothing else there); the changes are made in the SED2 repository.
* **A case is never made to pass by changing how a result is handled or compared.**  A `compare.mode` other than `full` (`lastRow`, `keyedRows`) is for what the case is about (the solver chooses the output points), never for a backend's inaccuracy; if a backend is outside the tolerance, that is a disagreement, and the backend is not admitted.  The translator must return a simulator's plain result (pySED2Translate's `CLAUDE.md` and `docs/compensations.md`); if a backend passes a case only because of a correction there, that is the finding.  Do not admit a backend (`--admit`) without checking the results it produced are the simulator's own.

## Standing rules

* **Do not commit or push** anything, in any repository, unless asked to.  Leave changes in the working tree.
* **libsed2 and the SED2 specification are developed in the SED2 repository, not from this one.**  Report gaps in
  `SED2/TODO.md` (a new section at the end); never work around them here.  Specification changes are decided by the user; propose them, do not
  make them.
* **ASCII only** in files (cases, documents, data, source, documentation).
* **Markdown files: no hard line breaks inside a paragraph or list item.**  Write each paragraph (and each list item) on one line and let the editor wrap it; the files are read in a text editor with word wrap.  Line breaks are for blank lines between paragraphs, list items, headings, table rows and code blocks.  Do not re-flow existing Markdown files unless asked.
* **Case numbers** are assigned in order of addition, from `00001`, and are never reused or renumbered.
* **Ask only when blocked.**  Make the call and say what was decided, unless a gap or a real question about
  intent stops the work.
* **Do not retry a rate-limited site** (for example EBI OLS4 answers 429): note it and carry on without it.
* **Checks before calling work done:** `python -m pytest`, `python -m sed2suite.validate_suite`,
  `python generate_tags.py --all --check`, `python -m sed2suite.make_inputs --all --check` and
  `python -m sed2suite.disagreements --check` and `python -m sed2suite.coverage --check`.
