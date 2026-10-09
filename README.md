# sed2-test-suite

A test suite for interpreters of SED2 documents.  It is modeled after the
[SBML Test Suite](https://github.com/sbmlteam/sbml-test-suite): each test case is a SED2 document plus the
input files it needs and the results a correct interpreter must produce.

Status: under construction.  The semantic cases 00001-00282 exist.  Every expected result was derived by hand, and
every case has been run through the pySED2Translate backends (roadrunner, COPASI, OpenCOR; COBRApy for flux balance
analysis) and agrees on each backend that can run it (`settings.json` lists them).  They cover constants, calculations and data manipulation, ODE time
courses, steady states and Jacobians, flux balance analysis, ranges and repeats (Scatter, Loop, ParameterScan), ModelChange (`setValues`,
`removeElements`), ModelElementList, CsvImport and the data behind Plot2D and Plot3D.  What is not yet covered, and why,
is in [docs/COVERAGE.md](docs/COVERAGE.md) and [docs/deferred.md](docs/deferred.md): chiefly stochastic simulation,
DataImport, aggregations and task parameters.

## Layout

```
cases/
  semantic/    valid SED2 documents with known, deterministic results (the current focus)
  syntactic/   valid and invalid documents for validation rules (filled in later, from the SED2 project)
  stochastic/  stochastic simulations and DrawFromDistribution (filled in later)
docs/FORMATS.md        exact file formats for results and settings (normative)
docs/PROMOTION.md      the checklist that makes a case canonical, and the sign-off in each description
docs/COVERAGE.md       which SED2 elements have cases (generated; deferred elements and why)
docs/deferred.md       what is not tested yet because the specification has not decided it
authoring/             the scripts that write the cases (see below)
schemas/               JSON Schemas (settings.json, disagreements.json)
disagreements.json     the log of cases where backends disagree (docs/FORMATS.md)
tools/sed2suite/       Python tools: results reader/writer, comparison, settings validation, suite validation
tests/                 pytest tests for the tools
tags.json              the tag vocabulary ('component' and 'semantic')
generate_tags.py       reads a .sed2.json and creates or updates its description.md tag block
```

## A semantic test case

Each folder `cases/semantic/NNNNN/` contains:

| File | Contents |
|---|---|
| `NNNNN.sed2.json` | The SED2 document being tested. |
| `NNNNN.description.md` | What the test checks, plus a generated list of component and semantic tags. |
| input files | Anything the document needs: models (`NNNNN.sbml`), data (`NNNNN.csv`), ... SBML models are made from Antimony; the Antimony source `NNNNN.ant` is kept next to the SBML it produced. |
| `NNNNN.[report_id].csv` or `.h5` | Expected output of one `report` in the document: CSV for 0, 1 or 2 dimensions, HDF5 for more. |
| `NNNNN.[plot_id]_as_data.csv` or `.h5` | The data behind a `plot` (CSV for Plot2D, HDF5 for Plot3D). |
| `NNNNN.[plot_id].png` | Optional illustration of a plot.  Not compared. |
| `NNNNN.settings.json` | Tolerances, which file belongs to which report or plot, comparison rules, and where the expected results came from. |

The exact formats are in [docs/FORMATS.md](docs/FORMATS.md).

A test is added to the suite once its document validates, it runs on every simulator backend that can run it,
and all of those backends agree with the expected results within the tolerances in `settings.json`.
Expected results are either 'Analytical' (calculable by hand) or produced by simulators; `settings.json`
records which.

## Numbering

Cases are numbered in the order they are added, starting at `00001`.  A number is never reused or renumbered, so
the log, commit messages and bug reports can refer to it for good.  The topic of a case is in its tags, not in its
number.

## Using the tools

Python 3.10 or later (3.13 recommended).  From this directory:

```
pip install -r requirements.txt
python -m pytest                              # tests for the tools
python generate_tags.py --all --check         # tag blocks are up to date
python -m sed2suite.validate_suite            # every case is complete and valid, with its sign-off
python -m sed2suite.disagreements --check     # the disagreement log is valid
python -m sed2suite.coverage --check          # every element has a case or a stated reason, COVERAGE.md is current
python -m sed2suite.compare CASE_DIR ACTUAL_DIR   # compare a run's output against a case
```

`validate_suite` needs the `libsed2` Python wheel (build artifact of https://github.com/sys-bio/SED2/ , see its
releases page); it is not on PyPI.

## Writing cases

The cases are written by the scripts in `authoring/` (one module per series, listed in `authoring/build.py`).  Each case
is a call that gives the SED2 document, the expected results computed from a formula, and a description of the
derivation:

```
PYTHONPATH=tools python authoring/build.py            # every series; one series: build.py repeats
```

A rebuild leaves a case exactly as it was (including which backends are admitted) when nothing about it changed.  After
a change, run the case through pySED2Translate and record the backends that agree:

```
python -m pysed2translate.suite_runner 00200 --crosscheck     # compare the backends with the expected results and with each other
python -m pysed2translate.suite_runner 00200 --admit          # record the backends that pass
```

See [docs/PROMOTION.md](docs/PROMOTION.md) for the full checklist.

## License

MIT, see `LICENSE`.
