"""Comparison of results against a test case's expected results (see docs/FORMATS.md).

Number rule:  |actual - expected| <= absolute + relative * |expected|
              (absolute is a fixed number, or absoluteScale times the column's first expected value; see Tolerance)
Score:        the largest |actual - expected| / (absolute + relative * |expected|); an item passes when it is at most 1
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
    """How close a number must be: |actual - expected| <= absolute + relative * |expected|.

    With `absolute_scale` set, it replaces `absolute`: the absolute part for a column of a two-dimensional table is
    absolute_scale * |the column's first value in the expected result|, or absolute_scale itself when that value is 0,
    nan or infinite.  Data that is not such a table (and numeric labels) use absolute_scale as it is."""
    absolute: float = 1e-7
    relative: float = 1e-4
    absolute_scale: Optional[float] = None

    def merged(self, override: Optional[dict]) -> "Tolerance":
        """These tolerances with `override` (a settings.json tolerances object) applied.  Giving `absolute` replaces a
        scale and giving `absoluteScale` replaces an absolute; giving both is an error."""
        if not override:
            return self
        if "absolute" in override and "absoluteScale" in override:
            raise ValueError("a tolerances object gives both 'absolute' and 'absoluteScale'; use one of them")
        absolute, scale = self.absolute, self.absolute_scale
        if "absolute" in override:
            absolute, scale = override["absolute"], None
        if "absoluteScale" in override:
            scale = override["absoluteScale"]
        return Tolerance(absolute, override.get("relative", self.relative), scale)

    def absolute_for(self, initial=None):
        """The absolute tolerance for data whose first value is `initial` (a number, an array of them, or None when
        the data has no first value to scale by).  A float for None or a number, an array for an array."""
        if self.absolute_scale is None:
            return float(self.absolute)
        if initial is None:
            return float(self.absolute_scale)
        first = np.asarray(initial, dtype=np.float64)
        usable = np.isfinite(first) & (first != 0)
        scaled = np.where(usable, self.absolute_scale * np.abs(first), self.absolute_scale)
        return float(scaled) if scaled.ndim == 0 else scaled


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
    score: float = 0.0     # see Scored; infinity when there is nothing to measure
    worst: str = ""        # where the score comes from

    @property
    def ok(self) -> bool:
        return not self.mismatches


# --------------------------------------------------------------------------- scalars and arrays

ABOVE_ONE = float(np.nextafter(1.0, 2.0))     # the smallest score that fails


@dataclass
class Scored:
    """The mismatches found, and the score of what was compared: the largest ratio of a difference to the difference
    allowed.  0 means identical, at most 1 within tolerance, above 1 beyond it, infinity for a mismatch that has no
    size (a shape, a string, nan against a number).  `worst` says where the score comes from."""
    mismatches: list = field(default_factory=list)
    score: float = 0.0
    worst: str = ""


def combine(*parts: Scored) -> Scored:
    best = max(parts, key=lambda p: p.score, default=Scored())
    return Scored([m for p in parts for m in p.mismatches], best.score, best.worst)


def numbers_close(actual: float, expected: float, tol: Tolerance, initial: Optional[float] = None) -> bool:
    if math.isnan(expected) or math.isnan(actual):
        return math.isnan(expected) and math.isnan(actual)
    if math.isinf(expected) or math.isinf(actual):
        return actual == expected
    return abs(actual - expected) <= tol.absolute_for(initial) + tol.relative * abs(expected)


def _outcome_scores(actual: np.ndarray, expected: np.ndarray, tol: Tolerance, initial=None) -> tuple:
    """(ok, score) for every entry of two numeric arrays of one shape.  `initial` is an array that broadcasts against
    them (the first value of each column), or None.  An entry is ok exactly when the tolerance rule holds; its score
    is |difference| / allowed, kept at most 1 for an ok entry and above 1 for a failing one."""
    a = actual.astype(np.float64)
    e = expected.astype(np.float64)
    either_nan = np.isnan(a) | np.isnan(e)
    both_nan = np.isnan(a) & np.isnan(e)
    inf_any = np.isinf(a) | np.isinf(e)
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = np.abs(a - e)
        allowed = tol.absolute_for(initial) + tol.relative * np.abs(e)
        finite_ok = diff <= allowed
        ratio = np.where(allowed > 0, diff / allowed, np.where(diff == 0, 0.0, np.inf))
        ok = np.where(either_nan, both_nan, np.where(inf_any, a == e, finite_ok))
        ratio = np.where(ok, np.minimum(ratio, 1.0), np.maximum(ratio, ABOVE_ONE))
        score = np.where(either_nan | inf_any, np.where(ok, 0.0, np.inf), ratio)
    return ok, score


def number_score(actual: float, expected: float, tol: Tolerance, initial: Optional[float] = None) -> float:
    first = None if initial is None else np.array([initial])
    return float(_outcome_scores(np.array([actual]), np.array([expected]), tol, first)[1][0])


def _py(x):
    """Plain Python value for messages (numpy scalars print as np.float64(...))."""
    return x.item() if isinstance(x, np.generic) else x


def _describe(a, e, tol: Tolerance, initial=None) -> str:
    a, e = _py(a), _py(e)
    try:
        diff = abs(float(a) - float(e))
        limit = tol.absolute_for(initial) + tol.relative * abs(float(e))
        return f"actual {a!r}, expected {e!r} (difference {diff:.6g} > allowed {limit:.6g})"
    except (TypeError, ValueError):
        return f"actual {a!r}, expected {e!r}"


def _label_equal(a, e, tol: Tolerance) -> bool:
    if rio.looks_numeric(a) and rio.looks_numeric(e):
        fa = a if not isinstance(a, str) else rio.parse_number(a)
        fe = e if not isinstance(e, str) else rio.parse_number(e)
        return numbers_close(float(fa), float(fe), tol)
    return str(a) == str(e)


def _label_score(a, e, tol: Tolerance) -> float:
    if rio.looks_numeric(a) and rio.looks_numeric(e):
        fa = a if not isinstance(a, str) else rio.parse_number(a)
        fe = e if not isinstance(e, str) else rio.parse_number(e)
        return number_score(float(fa), float(fe), tol)
    return 0.0 if str(a) == str(e) else math.inf


def _limit(msgs: list, where: str, found: list) -> None:
    for m in found[:MAX_LISTED]:
        msgs.append(Mismatch(where, m))
    if len(found) > MAX_LISTED:
        msgs.append(Mismatch(where, f"... and {len(found) - MAX_LISTED} more"))


def _position_text(position: tuple, like: Optional[rio.AnnotatedData]) -> str:
    position = tuple(int(i) for i in position)
    return _describe_position(position, like) if like is not None else str(list(position))


def _compare_values_scored(actual: np.ndarray, expected: np.ndarray, tol: Tolerance, where: str,
                           initial=None, like: Optional[rio.AnnotatedData] = None) -> Scored:
    """`initial` holds the first value of each column (for a two-dimensional numeric result), `like` supplies labels
    for the position of the worst entry."""
    if actual.shape != expected.shape:
        return Scored([Mismatch(where, f"shape {actual.shape}, expected {expected.shape}")], math.inf)
    a_str = actual.dtype == object or actual.dtype.kind in "US"
    e_str = expected.dtype == object or expected.dtype.kind in "US"
    if a_str != e_str:
        return Scored([Mismatch(where, "one result holds strings and the other numbers")], math.inf)
    found = []
    if e_str:
        bad = np.asarray(actual != expected)
        positions = list(zip(*np.nonzero(np.atleast_1d(bad))))
        for idx in positions:
            at = () if actual.ndim == 0 else idx
            found.append(f"{list(map(int, at))}: actual {_py(actual[at])!r}, expected {_py(expected[at])!r}")
        score = math.inf if positions else 0.0
        worst = () if (positions and actual.ndim == 0) else (positions[0] if positions else None)
    else:
        ok, scores = _outcome_scores(actual, expected, tol, initial)
        for idx in zip(*np.nonzero(np.atleast_1d(~ok))):
            at = () if actual.ndim == 0 else idx
            column_first = None if initial is None else initial[at[-1]]
            found.append(f"{list(map(int, at))}: " + _describe(actual[at], expected[at], tol, column_first))
        flat = np.atleast_1d(scores)
        score = float(flat.max()) if flat.size else 0.0
        worst = None
        if score > 0:
            worst = () if actual.ndim == 0 else np.unravel_index(int(np.argmax(flat)), flat.shape)
    out: list = []
    _limit(out, where, found)
    return Scored(out, score, _position_text(worst, like) if worst is not None else "")


def _labels_scored(actual: list, expected: list, tol: Tolerance, where: str) -> Scored:
    out = Scored()
    for dim, (a, e) in enumerate(zip(actual, expected)):
        w = f"{where} labels of dimension {dim}".strip()
        if (a is None) != (e is None):
            out = combine(out, Scored([Mismatch(w, "labels present in one result and absent in the other")], math.inf, w))
        elif a is not None:
            if len(a) != len(e):
                out = combine(out, Scored([Mismatch(w, f"{len(a)} labels, expected {len(e)}")], math.inf, w))
                continue
            scores = [_label_score(x, y, tol) for x, y in zip(a, e)]
            found = [f"[{i}]: actual {x!r}, expected {y!r}" for i, (x, y) in enumerate(zip(a, e))
                     if not _label_equal(x, y, tol)]
            msgs: list = []
            _limit(msgs, w, found)
            top = max(scores, default=0.0)
            out = combine(out, Scored(msgs, top, f"{w} [{scores.index(top)}]" if top > 0 else ""))
    return out


# --------------------------------------------------------------------------- AnnotatedData

def _initial_values(expected: rio.AnnotatedData) -> Optional[np.ndarray]:
    """The first-row value of every column of a two-dimensional numeric result; None for any other shape."""
    if expected.ndim != 2 or expected.is_string or expected.values.shape[0] == 0:
        return None
    return expected.values[0].astype(np.float64)


def compare_annotated_scored(actual: rio.AnnotatedData, expected: rio.AnnotatedData, tol: Tolerance,
                             mode: str = "full", where: str = "") -> Scored:
    """The mismatches (none means the results agree) and the score of the comparison."""
    if actual.ndim != expected.ndim:
        return Scored([Mismatch(where, f"{actual.ndim} dimensions, expected {expected.ndim}")], math.inf)
    initial = _initial_values(expected)
    if mode == "full":
        return combine(_labels_scored(actual.labels, expected.labels, tol, where),
                       _compare_values_scored(actual.values, expected.values, tol, where, initial, expected))
    if mode == "lastRow":
        if expected.ndim < 1 or actual.shape[0] == 0 or expected.shape[0] == 0:
            return Scored([Mismatch(where, "lastRow needs at least one row in both results")], math.inf)
        a_vals, e_vals = actual.values[-1:], expected.values[-1:]
        a_lab = [None] + list(actual.labels[1:])
        e_lab = [None] + list(expected.labels[1:])
        return combine(_labels_scored(a_lab, e_lab, tol, where),
                       _compare_values_scored(a_vals, e_vals, tol, where, initial))
    if mode == "keyedRows":
        return _compare_keyed_rows(actual, expected, tol, where, initial)
    return Scored([Mismatch(where, f"unknown comparison mode {mode!r}")], math.inf)


def compare_annotated(actual: rio.AnnotatedData, expected: rio.AnnotatedData, tol: Tolerance,
                      mode: str = "full", where: str = "") -> list:
    """Return a list of Mismatch; empty means the results agree."""
    return compare_annotated_scored(actual, expected, tol, mode, where).mismatches


def _compare_keyed_rows(actual, expected, tol, where, initial=None) -> Scored:
    """Match each expected row to the actual row with the same first-column value; compare the rest."""
    if expected.ndim != 2 or actual.ndim != 2 or actual.is_string or expected.is_string:
        return Scored([Mismatch(where, "keyedRows needs two-dimensional numeric results")], math.inf)
    labels = _labels_scored([None, actual.labels[1]], [None, expected.labels[1]], tol, where)
    if actual.shape[1] != expected.shape[1]:
        return combine(labels, Scored([Mismatch(where, f"{actual.shape[1]} columns, expected {expected.shape[1]}")],
                                      math.inf))
    row_initial = None if initial is None else np.where(np.arange(len(initial)) == 0, np.nan, initial)   # the key column is not scaled
    keys = actual.values[:, 0]
    found = []
    score, worst = 0.0, ""
    for r in range(expected.shape[0]):
        key = expected.values[r, 0]
        hits = [i for i in range(actual.shape[0]) if numbers_close(float(keys[i]), float(key), tol)]
        if not hits:
            found.append(f"row {r}: no actual row has first-column value {_py(key)!r}")
            score, worst = math.inf, f"row {r}"
            continue
        sub = _compare_values_scored(actual.values[hits[0]:hits[0] + 1], expected.values[r:r + 1], tol, "", row_initial)
        for m in sub.mismatches:
            found.append(f"row {r} (key {_py(key)!r}): {m.message}")
        if sub.score > score:
            score, worst = sub.score, f"row {r} (key {_py(key)!r})"
    msgs: list = []
    _limit(msgs, where, found)
    return combine(labels, Scored(msgs, score, worst))


# --------------------------------------------------------------------------- plots

def compare_plot2d_scored(actual: rio.Plot2DData, expected: rio.Plot2DData, tol: Tolerance, where: str = "") -> Scored:
    a_names, e_names = list(actual.columns), list(expected.columns)
    if a_names != e_names:
        return Scored([Mismatch(where, f"columns {a_names}, expected {e_names}")], math.inf)
    out = Scored()
    for name in e_names:
        a, e = actual.columns[name], expected.columns[name]
        w = f"{where} column {name!r}".strip()
        if len(a) != len(e):
            out = combine(out, Scored([Mismatch(w, f"{len(a)} rows, expected {len(e)}")], math.inf, w))
            continue
        initial = None if not e or e[0] is None or isinstance(e[0], str) else float(e[0])
        found = []
        score, worst = 0.0, ""
        for i, (x, y) in enumerate(zip(a, e)):
            s = 0.0
            if x is None or y is None:
                if x is not y:
                    found.append(f"row {i}: actual {x!r}, expected {y!r}")
                    s = math.inf
            elif isinstance(x, str) or isinstance(y, str):
                if x != y:
                    found.append(f"row {i}: actual {x!r}, expected {y!r}")
                    s = math.inf
            else:
                s = number_score(float(x), float(y), tol, initial)
                if not numbers_close(float(x), float(y), tol, initial):
                    found.append(f"row {i}: " + _describe(x, y, tol, initial))
            if s > score:
                score, worst = s, f"{w} row {i}"
        msgs: list = []
        _limit(msgs, w, found)
        out = combine(out, Scored(msgs, score, worst))
    return out


def compare_plot2d(actual: rio.Plot2DData, expected: rio.Plot2DData, tol: Tolerance, where: str = "") -> list:
    return compare_plot2d_scored(actual, expected, tol, where).mismatches


def compare_plot3d_scored(actual: rio.Plot3DData, expected: rio.Plot3DData, tol: Tolerance, where: str = "") -> Scored:
    if list(actual.surfaces) != list(expected.surfaces):
        return Scored([Mismatch(where, f"surfaces {list(actual.surfaces)}, expected {list(expected.surfaces)}")],
                      math.inf)
    out = Scored()
    for sid, e in expected.surfaces.items():
        a = actual.surfaces[sid]
        w = f"{where} surface {sid!r}".strip()
        if a.surface_type != e.surface_type:
            out = combine(out, Scored([Mismatch(w, f"surfaceType {a.surface_type!r}, expected {e.surface_type!r}")],
                                      math.inf, w))
        for name in ("x", "y", "z"):
            part = _compare_values_scored(getattr(a, name), getattr(e, name), tol, f"{w} {name}")
            part.worst = f"{w} {name} {part.worst}".strip() if part.score > 0 else ""
            out = combine(out, part)
    return out


def compare_plot3d(actual: rio.Plot3DData, expected: rio.Plot3DData, tol: Tolerance, where: str = "") -> list:
    return compare_plot3d_scored(actual, expected, tol, where).mismatches


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


def _last_row(data: rio.AnnotatedData) -> rio.AnnotatedData:
    """Only the last row, with the column labels; the row labels are dropped, as `lastRow` comparison drops them."""
    return rio.AnnotatedData(data.values[-1:], [None] + list(data.labels[1:]), data.dims)


def measure_difference(actual: rio.AnnotatedData, expected: rio.AnnotatedData, mode: str = "full") -> dict:
    """How far apart two results are, whatever the tolerance.  `mode` is the report's comparison mode: with "lastRow"
    only the last rows are measured, as `compare_annotated` compares them.  Returns a dict with:
      comparable   False if the shapes differ or one holds strings and the other numbers (then `reason` says so)
      maxAbsolute  the largest |actual - expected| (nan against a number counts as infinity)
      maxRelative  the largest |actual - expected| / |expected| over the entries where expected is finite and not 0
      where        the position of the largest absolute difference, with labels
    Strings compare exactly: maxAbsolute is the number of entries that differ."""
    if mode == "lastRow" and actual.ndim >= 1 and expected.ndim >= 1 and actual.shape[0] > 0 and expected.shape[0] > 0:
        actual, expected = _last_row(actual), _last_row(expected)
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
                mode = spec.get("compare", {}).get("mode", "full")
                entry.update(measure_difference(actual, expected, mode))
                tol = base.merged(spec.get("tolerances"))
                scored = compare_annotated_scored(actual, expected, tol, mode)
                entry["ok"], entry["score"] = not scored.mismatches, scored.score
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
                    item.score = math.inf
                else:
                    if kind == "report":
                        mode = spec.get("compare", {}).get("mode", "full")
                        scored = compare_annotated_scored(_read_report(act_path, spec), _read_report(exp_path, spec),
                                                          tol, mode)
                    elif spec["format"] == "csv":
                        scored = compare_plot2d_scored(rio.read_plot2d_csv(act_path), rio.read_plot2d_csv(exp_path), tol)
                    else:
                        scored = compare_plot3d_scored(rio.read_plot3d_h5(act_path), rio.read_plot3d_h5(exp_path), tol)
                    item.mismatches += scored.mismatches
                    item.score, item.worst = scored.score, scored.worst
            except (rio.FormatError, OSError, KeyError) as e:
                item.mismatches.append(Mismatch("", f"cannot compare: {e}"))
                item.score = math.inf
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
        score = "inf" if math.isinf(r.score) else f"{r.score:.3g}"
        if r.ok:
            if args.verbose:
                print(f"PASS  {r.kind} {r.name} ({r.file})  score {score}")
        else:
            failed += 1
            print(f"FAIL  {r.kind} {r.name} ({r.file})  score {score}")
            for m in r.mismatches:
                print(f"        {m}")
    print(f"{len(results) - failed} of {len(results)} items match")
    if results:
        w = max(results, key=lambda r: r.score)
        score = "inf" if math.isinf(w.score) else f"{w.score:.3g}"
        print(f"worst score {score} ({w.kind} {w.name}{' at ' + w.worst if w.worst else ''}); "
              f"an item passes when its score is at most 1")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
