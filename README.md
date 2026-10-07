# sed2-test-suite

A test suite for interpreters of SED2 documents.  It is modeled after the
[SBML Test Suite](https://github.com/sbmlteam/sbml-test-suite): each test case is a SED2 document plus the
input files it needs and the results a correct interpreter must produce.

Status: under construction.  Only the infrastructure exists so far; semantic test cases are being added.

## Layout

```
cases/
  semantic/    valid SED2 documents with known, deterministic results (the current focus)
  syntactic/   valid and invalid documents for validation rules (filled in later, from the SED2 project)
  stochastic/  stochastic simulations and DrawFromDistribution (filled in later)
docs/FORMATS.md        exact file formats for results and settings (normative)
schemas/               JSON Schemas (settings.json)
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

Test numbers are stable: once a number is used it is never reused or renumbered.  Blocks are reserved by topic
so related tests sit together:

| Range | Topic |
|---|---|
| 00001-00099 | Constants and reports only (no tasks) |
| 00100-00199 | Calculations and data manipulation |
| 00200-00299 | ODE time-course simulations |
| 00300-00399 | Steady state and Jacobian |
| 00400-00499 | Ranges, loops and scans |
| 00500-00599 | Model changes, data imports, plots |
| 00600 and up | Reserved |

## Using the tools

Python 3.10 or later (3.13 recommended).  From this directory:

```
pip install -r requirements.txt
python -m pytest                              # tests for the tools
python generate_tags.py --all --check         # tag blocks are up to date
python -m sed2suite.validate_suite            # every case is complete and valid
python -m sed2suite.compare CASE_DIR ACTUAL_DIR   # compare a run's output against a case
```

`validate_suite` needs the `libsed2` Python wheel (build artifact of https://github.com/sys-bio/SED2/ , see its
releases page); it is not on PyPI.

## License

MIT, see `LICENSE`.
