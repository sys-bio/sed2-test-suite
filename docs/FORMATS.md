# File formats

This document is normative for the files in a semantic test case.  The reference implementation is
`tools/sed2suite/results_io.py` (reading and writing), `compare.py` (comparison) and `settings.py` (settings
validation).  SED2 itself does not define any of these formats.

All text files are UTF-8 with LF line endings and contain no byte-order mark.

## Reports

A report's `data` is always AnnotatedData: an n-dimensional array plus optional labels (and a name) for each
dimension.  Expected results for one report are stored in one file, `NNNNN.[report_id].csv` or `.h5`.

Data with 0, 1 or 2 dimensions is written as CSV; data with 3 or more dimensions is written as HDF5.  The
settings file records the format and which dimensions carry labels, so no reader has to guess.

### CSV

* Comma separator, RFC 4180 quoting, LF line endings.
* 2-D: rows by columns.  If the column dimension has labels they form a header row.  If the row dimension has
  labels they form the first column.  If both exist, the top-left cell is empty.
* 1-D: a column vector (n rows, one value column).  Labels, if any, are the first column.  No header row.
* 0-D: one line holding one value.
* Numbers are written with Python `repr` (shortest text that round-trips).  Special values are `nan`, `inf` and
  `-inf`, lower case.  Booleans are written as `1` and `0`.  Readers also accept `NaN`, `INF` and `+inf`.
* Strings are written as-is, quoted where RFC 4180 requires.  One array is either all numbers or all strings;
  `nan`, `inf` and `-inf` may appear in numeric arrays only.
* Dimension names are not stored.

### HDF5

* Dataset `data`: the array in row-major order; float64 for numbers, UTF-8 variable-length strings for strings.
* Dataset `labels/<i>` for each dimension `i` that has labels.  Dimensions without labels have no such dataset.
* Root attribute `dims`: the dimension names (empty strings when unnamed).

## Plots as data

Only the data a plot was drawn from is stored.  Styles, axis scales and limits, legends, sizes and curve types
are not recorded or compared.  Number and special-value rules are those of report CSV files.

### Plot2D: `NNNNN.[plot_id]_as_data.csv`

* One column per data field per curve, in this order: `x`, `y`, `xErrorLower`, `xErrorUpper`, `yErrorLower`,
  `yErrorUpper`, `yFrom`, `yTo`.  Fields the curve does not define are omitted.
* Header row `[curve_id].[field]` (for example `c1.x`), no index column.
* Curves appear in ascending `order`, ties broken by position in the `curves` dictionary.
* Every value resolves to 0-D or 1-D data (0-D counts as length 1).
* Curves may differ in length.  Shorter columns are padded with empty cells.  An empty cell means "no value";
  a real NaN is written `nan`.

### Plot3D: `NNNNN.[plot_id]_as_data.h5`

* One group per surface, named for the surface id.
* Datasets `x`, `y` and `z`, each with the shape its reference resolves to.
* Group attributes: `surfaceType` (string) and `index` (position in the `surfaces` dictionary).
* Surfaces appear in ascending `order`, ties broken by position.  HDF5 groups have no order of their own, so the file is
  created with creation order tracked (h5py: `track_order=True`) and the groups are created in that order; a reader
  lists them in creation order.  The comparison checks the list of surface ids in that order, so an interpreter that
  ignores `order` fails.

### PNG

`NNNNN.[plot_id].png` files are optional illustrations and are never compared.

### Plotly figure

`NNNNN.[plot_id].plotly.json` is an optional illustration written by pySED2Translate's generated scripts.  It is a plotly figure (`data`, `layout`, `meta`) built from the same arrays as the plot-as-data file, so a browser can draw it, but it also carries styles (axis scale and limits, legend, size, curve types), nan and the infinities are null, and it is never compared and not listed in `settings.json`.  The `_as_data` file stays the only compared form of a plot.

## Comparison

Numbers match when `|actual - expected| <= absolute + relative * |expected|`.  `nan` matches `nan` at the same
position; `inf` matches `inf` of the same sign.  Strings must match exactly.  Labels are compared the same way
as values when both look numeric (for example time points), and exactly otherwise.  Row and column order matter.

Plot2D: the header rows must be equal and in the same order; values use the tolerance; empty cells must be in
the same places.  Plot3D: the same surfaces in the same order, the same `surfaceType`, and `x`, `y` and `z` of
the same shape within tolerance.

The comparison tool lists at most 10 mismatches per item and then counts the rest.

### Tolerances

`relative` is a fraction of the absolute value of the expected number.  The absolute part is given in one of two ways, never both in one `tolerances` object:

* `absolute`: a fixed number (default 1e-7).
* `absoluteScale`: a factor.  The absolute tolerance of a column is `absoluteScale * |v0|`, where `v0` is the value in the first row of that column of the expected result, or `absoluteScale` itself when `v0` is 0, nan or infinite.  It suits tables whose columns differ by orders of magnitude.  It applies to two-dimensional numeric reports and to the columns of Plot2D files.  Everything else (0-D and 1-D data, HDF5 arrays, Plot3D, numeric labels, and the key column of `keyedRows`) uses the factor as it is.

A `tolerances` object in a report or plot replaces the case's `absolute` or `absoluteScale` with its own (giving `absolute` there turns the case's `absoluteScale` off for that item, and the other way round), and replaces `relative` separately.

### Score

Every comparison also has a score: the largest, over all the numbers compared, of `|actual - expected|` divided by `absolute + relative * |expected|` (with the absolute part as above).  Numeric labels count in the same way.

* 0 means identical, 1 means exactly at the limit, 2 means twice the allowed difference.
* An item passes when its score is at most 1.  The verdict is the tolerance rule itself; the score of a value that fails is never below the smallest number above 1, and that of a value that passes never above 1.
* A mismatch that has no size scores infinity: a different shape or number of labels, strings or non-numeric labels that differ, nan or an infinity against a different value, a missing output file, a column of different length.
* If the allowed difference is 0, an equal pair scores 0 and a different one infinity.
* The score of a case is the largest score of its reports and plots.

The score is computed when results are compared and is not stored in `settings.json`.

### Comparison modes

Chosen per report in `settings.json` (`compare.mode`); the default is `full`.  A mode other than `full` should
carry a `compare.note` saying why.

| Mode | Meaning |
|---|---|
| `full` | Same shape, same labels, all values within tolerance. |
| `lastRow` | Only the last row is compared (and the labels of the other dimensions).  For output whose number of rows depends on the solver, such as variable-step output where only the final state is meaningful. |
| `keyedRows` | Each expected row is matched to the actual row with the same first-column value (within tolerance), and the remaining columns of those rows are compared.  Extra actual rows are ignored.  For 2-D numeric reports whose first column is a key such as time. |

## settings.json

`NNNNN.settings.json` is validated by `schemas/settings.schema.json` (JSON Schema draft 2020-12), and by
`sed2suite.settings.validate_settings` for the checks a schema cannot express.

```json
{
  "schemaVersion": 1,
  "tolerances": { "absolute": 1e-7, "relative": 1e-4 },
  "provenance": {
    "source": "simulation",
    "simulators": [ { "name": "roadrunner", "version": "2.10.0" } ],
    "notes": "free text"
  },
  "backends": [ "roadrunner", "copasi", "opencor" ],
  "reports": {
    "rep1": {
      "file": "00200.rep1.csv", "format": "csv", "ndim": 2, "dtype": "number",
      "labels": { "rows": false, "columns": true },
      "tolerances": { "relative": 1e-3 },
      "compare": { "mode": "full", "note": "" }
    }
  },
  "plots": {
    "plt1": { "file": "00200.plt1_as_data.csv", "format": "csv", "type": "plot2D" }
  },
  "notes": "free text"
}
```

| Key | Meaning |
|---|---|
| `schemaVersion` | Always `1`. |
| `tolerances` | Default tolerances for the case: `relative`, and either `absolute` or `absoluteScale` (see Tolerances).  Required. |
| `provenance.source` | `analytical` (worked out by hand or from a closed form) or `simulation`. |
| `provenance.simulators` | Required for `simulation`: name and version of each simulator that produced the expected results. |
| `backends` | Translator backends that were run and agreed with the expected results.  At least one.  The translator must keep supporting these for this case; losing one is a regression. |
| `reports` | Per report id: `file`, `format` (`csv` or `h5`); for CSV also `ndim` (0-2); optional `dtype` (`number` by default, or `string`), `labels` (which dimensions are labelled), `tolerances` (override), `compare`. |
| `plots` | Per plot id: `file`, `format`, `type` (`plot2D` is CSV, `plot3D` is HDF5), optional `tolerances`. |
| `notes` | Free text. |

Cross-checks in `validate_settings`: the file extension agrees with `format`; every listed file exists in the
case folder; and the set of report ids and the set of plot ids equal those in the SED2 document.  Ids are
identifiers (letters, digits, underscore; not starting with a digit), and file names contain no path separators.

## Disagreement log

`disagreements.json` at the root of the suite records every case where two backends, or a backend and the recorded
results, differ by more than the tolerance.  It is validated by `schemas/disagreements.schema.json` and by
`python -m sed2suite.disagreements --check` (which also checks that each test exists, that ids are unique, and that
an entry that is no longer open says what was done).  `--list [--open]` prints one line per entry.

```json
{
  "schemaVersion": 1,
  "entries": [
    {
      "id": "D-001",
      "test": "00003",
      "date": "2026-10-08",
      "backends": ["roadrunner", "copasi"],
      "report": "rep1",
      "size": { "maxAbsolute": 3.2e-5, "maxRelative": 1.1e-3, "where": "row 12 (4.0), column 1 (S1)" },
      "symptom": "COPASI's LSODA drifts from CVODE over the last third of the run",
      "status": "resolved",
      "diagnosis": "solver setting",
      "resolution": "set relativeTolerance 1e-10 in the document's algorithm parameters",
      "reference": ""
    }
  ]
}
```

| Key | Meaning |
|---|---|
| `id` | `D-001`, `D-002`, ...; never reused. |
| `test` | The five-digit case number. |
| `backends` | The two or more things that disagree: backend names, and `expected` for the results recorded in the case. |
| `report` | The report id the difference was found in; leave out for a failure that is not about one report (the script crashed, no output). |
| `size` | `maxAbsolute` and `maxRelative` (the largest differences found, whatever the tolerance; the number or `"inf"`) and `where` (the position of the largest one, with labels).  Left out when the results could not be compared at all (different shapes, strings against numbers); `symptom` then says why. |
| `symptom` | What was seen, in words. |
| `status` | `open`, `resolved` or `wontfix`. |
| `diagnosis` | `undiagnosed` (still being looked at), `solver setting`, `translator bug`, `simulator bug` or `spec ambiguity`. |
| `resolution` | What was done, or what is being waited for.  Required once the status is not `open`. |
| `reference` | Optional pointer: a commit, issue, SED2/TODO.md item or specification section. |

A mismatch only becomes an entry when the cause is not obvious from the output.  The suite runner
(`pysed2translate.suite_runner --log-draft`) prints draft entries for the disagreements it finds; they are added to
the file by hand once someone has looked at them.
