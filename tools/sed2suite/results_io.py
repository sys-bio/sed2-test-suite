"""Reading and writing SED2 test-suite result files (see docs/FORMATS.md).

Data model
----------
AnnotatedData: an n-dimensional array plus, per dimension, optional labels and a dimension name.
  values : numpy array, float64 for numbers or object dtype holding str for strings
  labels : list with one entry per dimension; each entry is None or a list of str/float labels
  dims   : list of dimension names ('' when unnamed); only HDF5 stores them

Plot2DData : ordered mapping column name -> list of float / str / None (None = empty cell)
Plot3DData : ordered mapping surface id -> Surface (x, y, z arrays, surface_type, index)

This module is deliberately independent of pySED2Translate's own writer.
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:  # h5py is only needed for the HDF5 formats
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_SPECIAL = {"nan": math.nan, "inf": math.inf, "+inf": math.inf, "-inf": -math.inf}


class FormatError(ValueError):
    """A results file does not follow docs/FORMATS.md."""


# --------------------------------------------------------------------------- numbers

def parse_number(text: str) -> float:
    """Parse one numeric cell.  Accepts decimal and exponent forms plus nan, inf, -inf (any case)."""
    s = text.strip()
    low = s.lower()
    if low in _SPECIAL:
        return _SPECIAL[low]
    if _NUMBER_RE.match(s):
        return float(s)
    raise FormatError(f"not a number: {text!r}")


def looks_numeric(text) -> bool:
    """True for ints/floats and for strings that parse as a finite or special number."""
    if isinstance(text, (int, float, np.integer, np.floating)) and not isinstance(text, bool):
        return True
    if isinstance(text, str):
        s = text.strip().lower()
        return s in _SPECIAL or bool(_NUMBER_RE.match(s))
    return False


def format_number(x) -> str:
    """Format a number for a results file: nan/inf/-inf lowercase, floats with Python's round-trip repr."""
    if isinstance(x, (bool, np.bool_)):
        return "1" if x else "0"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    f = float(x)
    if math.isnan(f):
        return "nan"
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    return repr(f)


# --------------------------------------------------------------------------- data model

@dataclass
class AnnotatedData:
    values: np.ndarray
    labels: list = field(default_factory=list)
    dims: list = field(default_factory=list)

    def __post_init__(self):
        self.values = np.asarray(self.values)
        n = self.values.ndim
        if not self.labels:
            self.labels = [None] * n
        if not self.dims:
            self.dims = [""] * n
        if len(self.labels) != n or len(self.dims) != n:
            raise FormatError("labels and dims must have one entry per dimension")
        for i, lab in enumerate(self.labels):
            if lab is not None and len(lab) != self.values.shape[i]:
                raise FormatError(f"dimension {i} has {self.values.shape[i]} entries but {len(lab)} labels")

    @property
    def ndim(self) -> int:
        return self.values.ndim

    @property
    def shape(self):
        return self.values.shape

    @property
    def is_string(self) -> bool:
        return self.values.dtype == object or self.values.dtype.kind in "US"


@dataclass
class Plot2DData:
    columns: dict  # name -> list of float | str | None


@dataclass
class Surface:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    surface_type: str
    index: int = 0


@dataclass
class Plot3DData:
    surfaces: dict  # surface id -> Surface, ordered by index


# --------------------------------------------------------------------------- CSV: reports

def _cell(value, is_string: bool) -> str:
    return str(value) if is_string else format_number(value)


def _label_cell(lab) -> str:
    return lab if isinstance(lab, str) else format_number(lab)


def csv_text(data: AnnotatedData) -> str:
    """Serialize 0-, 1- or 2-D AnnotatedData to CSV text (LF line endings, RFC 4180 quoting)."""
    import io

    if data.ndim > 2:
        raise FormatError("CSV holds at most two dimensions; use HDF5")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    s = data.is_string
    if data.ndim == 0:
        w.writerow([_cell(data.values.item(), s)])
    elif data.ndim == 1:
        rows = data.labels[0]
        for i in range(data.shape[0]):
            row = [_cell(data.values[i], s)]
            if rows is not None:
                row.insert(0, _label_cell(rows[i]))
            w.writerow(row)
    else:
        rows, cols = data.labels
        if cols is not None:
            head = [_label_cell(c) for c in cols]
            if rows is not None:
                head.insert(0, "")
            w.writerow(head)
        for i in range(data.shape[0]):
            row = [_cell(v, s) for v in data.values[i]]
            if rows is not None:
                row.insert(0, _label_cell(rows[i]))
            w.writerow(row)
    return buf.getvalue()


def write_csv(path, data: AnnotatedData) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(csv_text(data))


def _read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [row for row in csv.reader(f)]


def read_csv(path, *, ndim: int, row_labels: bool = False, column_labels: bool = False,
             dtype: str = "number") -> AnnotatedData:
    """Read a report CSV.  The caller (normally from settings.json) says how to interpret it."""
    if dtype not in ("number", "string"):
        raise FormatError(f"dtype must be 'number' or 'string', not {dtype!r}")
    if ndim not in (0, 1, 2):
        raise FormatError("CSV ndim must be 0, 1 or 2")
    if ndim == 0 and (row_labels or column_labels):
        raise FormatError("a 0-D file has no labels")
    if ndim == 1 and column_labels:
        raise FormatError("a 1-D file has no column labels")
    rows = _read_rows(path)
    conv = (lambda t: t) if dtype == "string" else parse_number
    try:
        if ndim == 0:
            if len(rows) != 1 or len(rows[0]) != 1:
                raise FormatError("a 0-D file must contain exactly one cell")
            return AnnotatedData(np.array(conv(rows[0][0]), dtype=object if dtype == "string" else float))
        col_lab = None
        if ndim == 2 and column_labels:
            if not rows:
                raise FormatError("missing header row")
            header = rows.pop(0)
            col_lab = header[1:] if row_labels else header
        width = None
        row_lab, vals = ([] if row_labels else None), []
        for r, row in enumerate(rows):
            if row_labels:
                if not row:
                    raise FormatError(f"row {r} is empty")
                row_lab.append(row[0])
                row = row[1:]
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise FormatError(f"row {r} has {len(row)} cells, expected {width}")
            vals.append([conv(c) for c in row])
        if ndim == 1:
            if width not in (None, 1):
                raise FormatError("a 1-D file must have a single value column")
            arr = np.array([v[0] for v in vals], dtype=object if dtype == "string" else float)
            return AnnotatedData(arr, [row_lab])
        if width is None:
            width = len(col_lab) if col_lab is not None else 0
        arr = np.empty((len(vals), width), dtype=object if dtype == "string" else float)
        for i, v in enumerate(vals):
            arr[i, :] = v
        if col_lab is not None and len(col_lab) != width:
            raise FormatError(f"header has {len(col_lab)} labels but rows have {width} values")
        return AnnotatedData(arr, [row_lab, col_lab])
    except FormatError as e:
        raise FormatError(f"{path}: {e}") from None


# --------------------------------------------------------------------------- HDF5: reports

def _need_h5():
    if h5py is None:  # pragma: no cover
        raise ImportError("h5py is required for HDF5 results")


def _store(group, name, array):
    arr = np.asarray(array)
    if arr.dtype == object or arr.dtype.kind in "US":
        ds = group.create_dataset(name, data=np.array([str(x) for x in arr.ravel()], dtype=object).reshape(arr.shape),
                                  dtype=h5py.string_dtype("utf-8"))
    else:
        group.create_dataset(name, data=arr.astype(np.float64))


def _load(ds):
    arr = ds[()]
    if ds.dtype.kind == "O" or ds.dtype.kind == "S":
        flat = np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in np.asarray(arr).ravel()],
                        dtype=object)
        return flat.reshape(np.asarray(arr).shape)
    return np.asarray(arr, dtype=np.float64)


def write_h5(path, data: AnnotatedData) -> None:
    """HDF5 layout: dataset 'data'; datasets 'labels/<i>' for labelled dimensions; root attribute 'dims'."""
    _need_h5()
    with h5py.File(path, "w") as f:
        _store(f, "data", data.values)
        for i, lab in enumerate(data.labels):
            if lab is not None:
                if all(looks_numeric(x) and not isinstance(x, str) for x in lab):
                    f.create_dataset(f"labels/{i}", data=np.array(lab, dtype=np.float64))
                else:
                    _store(f, f"labels/{i}", np.array([_label_cell(x) for x in lab], dtype=object))
        f.attrs.create("dims", np.array(list(data.dims), dtype=object), dtype=h5py.string_dtype("utf-8"))


def read_h5(path) -> AnnotatedData:
    _need_h5()
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise FormatError(f"{path}: no dataset named 'data'")
        values = _load(f["data"])
        labels = []
        for i in range(values.ndim):
            key = f"labels/{i}"
            if key in f:
                lab = _load(f[key])
                labels.append([x if isinstance(x, str) else float(x) for x in lab.tolist()])
            else:
                labels.append(None)
        dims = [d.decode("utf-8") if isinstance(d, bytes) else str(d) for d in f.attrs.get("dims", [""] * values.ndim)]
        if len(dims) != values.ndim:
            raise FormatError(f"{path}: 'dims' has {len(dims)} entries for {values.ndim} dimensions")
    return AnnotatedData(values, labels, dims)


# --------------------------------------------------------------------------- Plot2D CSV

def _plot_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return format_number(v)


def plot2d_csv_text(plot: Plot2DData) -> str:
    import io

    names = list(plot.columns)
    n = max((len(c) for c in plot.columns.values()), default=0)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(names)
    for i in range(n):
        w.writerow([_plot_cell(plot.columns[c][i]) if i < len(plot.columns[c]) else "" for c in names])
    return buf.getvalue()


def write_plot2d_csv(path, plot: Plot2DData) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(plot2d_csv_text(plot))


def read_plot2d_csv(path) -> Plot2DData:
    rows = _read_rows(path)
    if not rows:
        raise FormatError(f"{path}: missing header row")
    names = rows[0]
    cols = {n: [] for n in names}
    if len(cols) != len(names):
        raise FormatError(f"{path}: duplicate column names")
    for r, row in enumerate(rows[1:], start=1):
        if len(row) != len(names):
            raise FormatError(f"{path}: row {r} has {len(row)} cells, expected {len(names)}")
        for n, cell in zip(names, row):
            if cell == "":
                cols[n].append(None)
            elif looks_numeric(cell):
                cols[n].append(parse_number(cell))
            else:
                cols[n].append(cell)
    return Plot2DData(cols)


# --------------------------------------------------------------------------- Plot3D HDF5

def write_plot3d_h5(path, plot: Plot3DData) -> None:
    """One group per surface (named by surface id) with datasets x, y, z and attributes surfaceType, index."""
    _need_h5()
    with h5py.File(path, "w") as f:
        for sid, s in sorted(plot.surfaces.items(), key=lambda kv: kv[1].index):
            g = f.create_group(sid)
            for name in ("x", "y", "z"):
                _store(g, name, getattr(s, name))
            g.attrs["surfaceType"] = s.surface_type
            g.attrs["index"] = int(s.index)


def read_plot3d_h5(path) -> Plot3DData:
    _need_h5()
    out = {}
    with h5py.File(path, "r") as f:
        for sid in f:
            g = f[sid]
            for name in ("x", "y", "z"):
                if name not in g:
                    raise FormatError(f"{path}: surface {sid!r} lacks dataset {name!r}")
            st = g.attrs.get("surfaceType", "")
            st = st.decode("utf-8") if isinstance(st, bytes) else str(st)
            out[sid] = Surface(_load(g["x"]), _load(g["y"]), _load(g["z"]), st, int(g.attrs.get("index", 0)))
    return Plot3DData(dict(sorted(out.items(), key=lambda kv: kv[1].index)))
