"""Comparison of results against a test case's expected results (see docs/FORMATS.md).

Number rule:  |actual - expected| <= absolute + relative * |expected|
nan matches nan; inf matches inf of the same sign; strings and string labels must match exactly.
Numeric-looking labels (for example time values) are compared with the tolerance as well.

Command line:
    python -m sed2suite.compare CASE_DIR ACTUAL_DIR [-v]
CASE_DIR holds NNNNN.settings.json and the expected files; ACTUAL_DIR holds output files with the same names.
Exit status 0 if every report and plot matches, 1 otherwise, 2 for usage or format errors.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import results_io as rio
from .settings import load_settings, find_settings_file

MAX_LISTED = 10


@dataclass(frozen=True)
class Tolerance:
    absolute: float = 1e-7
    relative: float = 1e-4

    def merged(self, override: Optional[dict]) -> "Tolerance":
        if not override:
            return self
        return Tolerance(override.get("absolute", self.absolute), override.get("relative", self.relative))


@dataclass
class Mismatch:
    where: str
    message: str

    def __str__(self):
        return f"{self.where}: {self.message}" if self.where else self.message


@dataclass
class ItemResult:
    name: str
    kind: str  # 'report' or 'plot'
    file: str
    mismatches: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches


# --------------------------------------------------------------------------- scalars and arrays

def numbers_close(actual: float, expected: float, tol: Tolerance) -> bool:
    if math.isnan(expected) or math.isnan(actual):
        return math.isnan(expected) and math.isnan(actual)
    if math.isinf(expected) or math.isinf(actual):
        return actual == expected
    return abs(actual - expected) <= tol.absolute + tol.relative * abs(expected)


def _close_mask(actual: np.ndarray, expected: np.ndarray, tol: Tolerance) -> np.ndarray:
    a = actual.astype(np.float64)
    e = expected.astype(np.float64)
    both_nan = np.isnan(a) & np.isnan(e)
    either_nan = np.isnan(a) | np.isnan(e)
    inf_any = np.isinf(a) | np.isinf(e)
    with np.errstate(invalid="ignore"):
        finite_ok = np.abs(a - e) <= tol.absolute + tol.relative * np.abs(e)
        inf_ok = a == e
    return np.where(either_nan, both_nan, np.where(inf_any, inf_ok, finite_ok))


def _py(x):
    """Plain Python value for messages (numpy scalars print as np.float64(...))."""
    return x.item() if isinstance(x, np.generic) else x


def _describe(a, e, tol: Tolerance) -> str:
    a, e = _py(a), _py(e)
    try:
        diff = abs(float(a) - float(e))
        limit = tol.absolute + tol.relative * abs(float(e))
        return f"actual {a!r}, expected {e!r} (difference {diff:.6g} > allowed {limit:.6g})"
    except (TypeError, ValueError):
        return f"actual {a!r}, expected {e!r}"


def _label_equal(a, e, tol: Tolerance) -> bool:
    if rio.looks_numeric(a) and rio.looks_numeric(e):
        fa = a if not isinstance(a, str) else rio.parse_number(a)
        fe = e if not isinstance(e, str) else rio.parse_number(e)
        return numbers_close(float(fa), float(fe), tol)
    return str(a) == str(e)


def _limit(msgs: list, where: str, found: list) -> None:
    for m in found[:MAX_LISTED]:
        msgs.append(Mismatch(where, m))
    if len(found) > MAX_LISTED:
        msgs.append(Mismatch(where, f"... and {len(found) - MAX_LISTED} more"))


def _compare_values(actual: np.ndarray, expected: np.ndarray, tol: Tolerance, where: str) -> list:
    out: list = []
    if actual.shape != expected.shape:
        return [Mismatch(where, f"shape {actual.shape}, expected {expected.shape}")]
    a_str = actual.dtype == object or actual.dtype.kind in "US"
    e_str = expected.dtype == object or expected.dtype.kind in "US"
    if a_str != e_str:
        return [Mismatch(where, "one result holds strings and the other numbers")]
    found = []
    if e_str:
        bad = np.asarray(actual != expected)
        for idx in zip(*np.nonzero(np.atleast_1d(bad))):
            at = () if actual.ndim == 0 else idx
            found.append(f"{list(map(int, at))}: actual {_py(actual[at])!r}, expected {_py(expected[at])!r}")
    else:
        bad = ~_close_mask(actual, expected, tol)
        for idx in zip(*np.nonzero(np.atleast_1d(bad))):
            at = () if actual.ndim == 0 else idx
            found.append(f"{list(map(int, at))}: " + _describe(actual[at], expected[at], tol))
    _limit(out, where, found)
    return out


def _compare_labels(actual: list, expected: list, tol: Tolerance, where: str) -> list:
    out = []
    for dim, (a, e) in enumerate(zip(actual, expected)):
        w = f"{where} labels of dimension {dim}".strip()
        if (a is None) != (e is None):
            out.append(Mismatch(w, "labels present in one result and absent in the other"))
        elif a is not None:
            if len(a) != len(e):
                out.append(Mismatch(w, f"{len(a)} labels, expected {len(e)}"))
                continue
            found = [f"[{i}]: actual {x!r}, expected {y!r}" for i, (x, y) in enumerate(zip(a, e))
                     if not _label_equal(x, y, tol)]
            _limit(out, w, found)
    return out


# --------------------------------------------------------------------------- AnnotatedData

def compare_annotated(actual: rio.AnnotatedData, expected: rio.AnnotatedData, tol: Tolerance,
                      mode: str = "full", where: str = "") -> list:
    """Return a list of Mismatch; empty means the results agree."""
    if actual.ndim != expected.ndim:
        return [Mismatch(where, f"{actual.ndim} dimensions, expected {expected.ndim}")]
    if mode == "full":
        out = _compare_labels(actual.labels, expected.labels, tol, where)
        out += _compare_values(actual.values, expected.values, tol, where)
        return out
    if mode == "lastRow":
        if expected.ndim < 1 or actual.shape[0] == 0 or expected.shape[0] == 0:
            return [Mismatch(where, "lastRow needs at least one row in both results")]
        a_vals, e_vals = actual.values[-1:], expected.values[-1:]
        a_lab = [None] + list(actual.labels[1:])
        e_lab = [None] + list(expected.labels[1:])
        out = _compare_labels(a_lab, e_lab, tol, where)
        out += _compare_values(a_vals, e_vals, tol, where)
        return out
    if mode == "keyedRows":
        return _compare_keyed_rows(actual, expected, tol, where)
    return [Mismatch(where, f"unknown comparison mode {mode!r}")]


def _compare_keyed_rows(actual, expected, tol, where) -> list:
    """Match each expected row to the actual row with the same first-column value; compare the rest."""
    if expected.ndim != 2 or actual.ndim != 2 or actual.is_string or expected.is_string:
        return [Mismatch(where, "keyedRows needs two-dimensional numeric results")]
    out = _compare_labels([None, actual.labels[1]], [None, expected.labels[1]], tol, where)
    if actual.shape[1] != expected.shape[1]:
        return out + [Mismatch(where, f"{actual.shape[1]} columns, expected {expected.shape[1]}")]
    keys = actual.values[:, 0]
    found = []
    for r in range(expected.shape[0]):
        key = expected.values[r, 0]
        hits = [i for i in range(actual.shape[0]) if numbers_close(float(keys[i]), float(key), tol)]
        if not hits:
            found.append(f"row {r}: no actual row has first-column value {_py(key)!r}")
            continue
        sub = _compare_values(actual.values[hits[0]:hits[0] + 1], expected.values[r:r + 1], tol, "")
        for m in sub:
            found.append(f"row {r} (key {_py(key)!r}): {m.message}")
    _limit(out, where, found)
    return out


# --------------------------------------------------------------------------- plots

def compare_plot2d(actual: rio.Plot2DData, expected: rio.Plot2DData, tol: Tolerance, where: str = "") -> list:
    out = []
    a_names, e_names = list(actual.columns), list(expected.columns)
    if a_names != e_names:
        return [Mismatch(where, f"columns {a_names}, expected {e_names}")]
    for name in e_names:
        a, e = actual.columns[name], expected.columns[name]
        w = f"{where} column {name!r}".strip()
        if len(a) != len(e):
            out.append(Mismatch(w, f"{len(a)} rows, expected {len(e)}"))
            continue
        found = []
        for i, (x, y) in enumerate(zip(a, e)):
            if x is None or y is None:
                if x is not y:
                    found.append(f"row {i}: actual {x!r}, expected {y!r}")
            elif isinstance(x, str) or isinstance(y, str):
                if x != y:
                    found.append(f"row {i}: actual {x!r}, expected {y!r}")
            elif not numbers_close(float(x), float(y), tol):
                found.append(f"row {i}: " + _describe(x, y, tol))
        _limit(out, w, found)
    return out


def compare_plot3d(actual: rio.Plot3DData, expected: rio.Plot3DData, tol: Tolerance, where: str = "") -> list:
    out = []
    if list(actual.surfaces) != list(expected.surfaces):
        return [Mismatch(where, f"surfaces {list(actual.surfaces)}, expected {list(expected.surfaces)}")]
    for sid, e in expected.surfaces.items():
        a = actual.surfaces[sid]
        w = f"{where} surface {sid!r}".strip()
        if a.surface_type != e.surface_type:
            out.append(Mismatch(w, f"surfaceType {a.surface_type!r}, expected {e.surface_type!r}"))
        for name in ("x", "y", "z"):
            out += _compare_values(getattr(a, name), getattr(e, name), tol, f"{w} {name}")
    return out


# --------------------------------------------------------------------------- whole cases

def _read_report(path: str, spec: dict) -> rio.AnnotatedData:
    if spec["format"] == "h5":
        return rio.read_h5(path)
    labels = spec.get("labels", {})
    return rio.read_csv(path, ndim=spec["ndim"], row_labels=labels.get("rows", False),
                        column_labels=labels.get("columns", False), dtype=spec.get("dtype", "number"))


def _describe_position(index: tuple, data: rio.AnnotatedData) -> str:
    """'row 3 (0.5), column 1 (S1)' for a position in the data, using labels where there are some."""
    names = ["row", "column"] if len(index) <= 2 else [f"dimension {i}" for i in range(len(index))]
    parts = []
    for d, i in enumerate(index):
        label = data.labels[d][i] if d < len(data.labels) and data.labels[d] else None
        parts.append(f"{names[d]} {i}" + (f" ({label})" if label is not None else ""))
    return ", ".join(parts) if parts else "the value"


def measure_difference(actual: rio.AnnotatedData, expected: rio.AnnotatedData) -> dict:
    """How far apart two results are, whatever the tolerance.  Returns a dict with:
      comparable   False if the shapes differ or one holds strings and the other numbers (then `reason` says so)
      maxAbsolute  the largest |actual - expected| (nan against a number counts as infinity)
      maxRelative  the largest |actual - expected| / |expected| over the entries where expected is finite and not 0
      where        the position of the largest absolute difference, with labels
    Strings compare exactly: maxAbsolute is the number of entries that differ."""
    a, e = np.asarray(actual.values), np.asarray(expected.values)
    if a.shape != e.shape:
        return {"comparable": False, "reason": f"shape {a.shape}, expected {e.shape}"}
    a_str, e_str = a.dtype == object or a.dtype.kind in "US", e.dtype == object or e.dtype.kind in "US"
    if a_str != e_str:
        return {"comparable": False, "reason": "one result holds strings and the other numbers"}
    if a_str:
        differ = np.array([x != y for x, y in zip(a.ravel(), e.ravel())], dtype=bool).reshape(a.shape)
        count = int(differ.sum())
        where = _describe_position(tuple(np.argwhere(differ)[0]), expected) if count else ""
        return {"comparable": True, "maxAbsolute": float(count), "maxRelative": float(count > 0), "where": where}
    a, e = a.astype(np.float64), e.astype(np.float64)
    if a.size == 0:
        return {"comparable": True, "maxAbsolute": 0.0, "maxRelative": 0.0, "where": ""}
    with np.errstate(invalid="ignore"):
        diff = np.abs(a - e)
        same = (np.isnan(a) & np.isnan(e)) | ((np.isinf(a) | np.isinf(e)) & (a == e))
        diff = np.where(same, 0.0, np.where(np.isnan(diff), np.inf, diff))
        usable = (e != 0) & np.isfinite(e)
        rel = np.where(usable, diff / np.where(usable, np.abs(e), 1.0), np.where(np.isinf(diff), np.inf, 0.0))
    worst = np.unravel_index(int(np.argmax(diff)), diff.shape)
    return {"comparable": True, "maxAbsolute": float(diff.max()), "maxRelative": float(rel.max()),
            "where": _describe_position(tuple(int(i) for i in worst), expected) if diff.max() > 0 else ""}


def measure_case(expected_dir: str, actual_dir: str, settings: dict) -> list:
    """For every report in the settings, how far the result in `actual_dir` is from the one in `expected_dir`
    (a case folder, or another backend's output folder with the same file names).  A list of dicts with the keys of
    `measure_difference` plus `name`, `file` and `ok` (within the report's tolerance, as `compare_case` judges)."""
    base = Tolerance().merged(settings.get("tolerances"))
    out = []
    for name, spec in settings.get("reports", {}).items():
        entry = {"name": name, "file": spec["file"]}
        exp_path, act_path = os.path.join(expected_dir, spec["file"]), os.path.join(actual_dir, spec["file"])
        if not os.path.exists(act_path) or not os.path.exists(exp_path):
            entry.update(comparable=False, reason=f"{spec['file']} was not produced", ok=False)
        else:
            try:
                actual, expected = _read_report(act_path, spec), _read_report(exp_path, spec)
                entry.update(measure_difference(actual, expected))
                tol = base.merged(spec.get("tolerances"))
                entry["ok"] = not compare_annotated(actual, expected, tol, spec.get("compare", {}).get("mode", "full"))
            except (rio.FormatError, OSError, KeyError) as e:
                entry.update(comparable=False, reason=f"cannot read: {e}", ok=False)
        out.append(entry)
    return out


def compare_case(case_dir: str, actual_dir: str, settings: Optional[dict] = None) -> list:
    """Compare every report and plot listed in the case's settings.json.  Returns a list of ItemResult."""
    if settings is None:
        settings = load_settings(find_settings_file(case_dir))
    base = Tolerance().merged(settings.get("tolerances"))
    results = []
    for kind, section in (("report", "reports"), ("plot", "plots")):
        for name, spec in settings.get(section, {}).items():
            item = ItemResult(name, kind, spec["file"])
            tol = base.merged(spec.get("tolerances"))
            exp_path = os.path.join(case_dir, spec["file"])
            act_path = os.path.join(actual_dir, spec["file"])
            try:
                if not os.path.exists(act_path):
                    item.mismatches.append(Mismatch("", f"output file {spec['file']} was not produced"))
                elif kind == "report":
                    mode = spec.get("compare", {}).get("mode", "full")
                    item.mismatches += compare_annotated(_read_report(act_path, spec), _read_report(exp_path, spec),
                                                         tol, mode)
                elif spec["format"] == "csv":
                    item.mismatches += compare_plot2d(rio.read_plot2d_csv(act_path), rio.read_plot2d_csv(exp_path), tol)
                else:
                    item.mismatches += compare_plot3d(rio.read_plot3d_h5(act_path), rio.read_plot3d_h5(exp_path), tol)
            except (rio.FormatError, OSError, KeyError) as e:
                item.mismatches.append(Mismatch("", f"cannot compare: {e}"))
            results.append(item)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare results against a test case's expected results.")
    ap.add_argument("case_dir")
    ap.add_argument("actual_dir")
    ap.add_argument("-v", "--verbose", action="store_true", help="also list passing items")
    args = ap.parse_args(argv)
    try:
        results = compare_case(args.case_dir, args.actual_dir)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    failed = 0
    for r in results:
        if r.ok:
            if args.verbose:
                print(f"PASS  {r.kind} {r.name} ({r.file})")
        else:
            failed += 1
            print(f"FAIL  {r.kind} {r.name} ({r.file})")
            for m in r.mismatches:
                print(f"        {m}")
    print(f"{len(results) - failed} of {len(results)} items match")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
