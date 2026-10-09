# From a draft to a canonical test case

A case is canonical when its expected results are right, whatever a simulator produces, and at least one backend of
pySED2Translate is known to agree with them.  Each case records this in its description (the "Sign-off" section) and in
its `settings.json`.  `python -m sed2suite.validate_suite` fails a case whose sign-off is incomplete.

## Checklist

1. **Derive the results by hand.**  The description has an "Expected results" section that says how each number was
   worked out: a formula, a closed-form solution of the ODE, or a counting argument.  The authoring script computes the
   expected tables from that formula, never from a simulator.  Models are chosen so that a closed form exists; when
   none does, say so and record the simulators in `provenance.simulators` (`source: "simulation"`), and ask for the
   results of at least two independent simulators to agree before they are accepted.
2. **Make every value explicit.**  Set every initial value (an unset SBML species is simulator-dependent), use
   compartments of volume 1 where concentration and amount would otherwise be confused, and avoid anything the
   specification leaves undefined (SED2/TODO.md lists the open points).  A point that is undefined is not tested.
3. **Check the document.**  libsed2 must accept it (the validator runs it), and there is at least one report.
4. **Run it.**  `python -m pysed2translate.suite_runner NNNNN --crosscheck` runs the case through each backend and
   compares the results with the expected files and with each other.
5. **Triage every difference; never hide one.**  A difference is one of:
   * a mistake in the expected results or the case (fix the case, then start again from step 1),
   * a bug in the translator (fix it, add a test to pySED2Translate),
   * a bug in a simulator (record it in `disagreements.json` with the model and the numbers; the translator may refuse
     the case for that backend with a clear reason, as it does for OpenCOR and models with events),
   * a point the specification leaves open (add a section to the end of SED2/TODO.md and remove the case until it is
     decided).
6. **Admit the backends.**  `python -m pysed2translate.suite_runner NNNNN --admit` writes the backends that pass into
   `settings.json` and the "Backends" line of the sign-off.  A backend that cannot run a case (a documented limit)
   is skipped, not listed.  Once a backend is listed, the translator must keep supporting it for that case: losing it
   is a failure (a mistake in the capability table must not turn real tests into skips).
7. **Sign-off.**  The description's "Sign-off" section then has three lines: how the results were obtained, the
   tolerances, and the admitted backends with the date.  Rebuilding a case with `authoring/build.py` keeps the admission
   only if nothing about the case changed; a changed case must be run and admitted again.
8. **Full runs.**  Before the work is called done: the suite's own tests, the validator
   (`python -m sed2suite.validate_suite`), `python -m sed2suite.disagreements --check`, the translator's tests, and a run
   of every case through every backend.

Tolerances are the defaults (absolute 1e-7, relative 1e-4) unless the description says why a case needs others.  A
tolerance is never loosened to make a backend pass.
