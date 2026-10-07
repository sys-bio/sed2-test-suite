import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from sed2suite import compare as cmp
from sed2suite import results_io as rio
from sed2suite.compare import Tolerance, compare_annotated, numbers_close
from sed2suite.results_io import AnnotatedData

TOL = Tolerance(absolute=1e-7, relative=1e-4)


def ad(values, labels=None, dims=None):
    return AnnotatedData(np.array(values), labels or [], dims or [])


# ---------------------------------------------------------------- numbers_close

@pytest.mark.parametrize("a, e, ok", [
    (1.0, 1.0, True),
    (1.0 + 0.9e-4, 1.0, True),
    (1.0 + 1.2e-4, 1.0, False),
    (0.0, 0.0, True),
    (5e-8, 0.0, True),
    (2e-7, 0.0, False),
    (-1.00005, -1.0, True),
    (math.nan, math.nan, True),
    (math.nan, 1.0, False),
    (1.0, math.nan, False),
    (math.inf, math.inf, True),
    (math.inf, -math.inf, False),
    (math.inf, 1e300, False),
    (1e300, math.inf, False),
    (-math.inf, -math.inf, True),
])
def test_numbers_close(a, e, ok):
    assert numbers_close(a, e, TOL) is ok


def test_tolerance_is_relative_to_expected_not_actual():
    # expected 100, actual 100.0105: allowed 1e-7 + 1e-4*100 = 0.0100001
    assert not numbers_close(100.0105, 100.0, TOL)
    assert numbers_close(100.0099, 100.0, TOL)


def test_tolerance_merge():
    t = Tolerance(1e-7, 1e-4)
    assert t.merged(None) == t
    assert t.merged({"relative": 1e-2}) == Tolerance(1e-7, 1e-2)
    assert t.merged({"absolute": 1.0, "relative": 0.0}) == Tolerance(1.0, 0.0)


# ---------------------------------------------------------------- compare_annotated, full mode

def test_identical():
    d = ad([[1, 2], [3, 4]], [["a", "b"], ["x", "y"]])
    assert compare_annotated(d, d, TOL) == []


def test_within_tolerance_passes():
    assert compare_annotated(ad([1.0, 2.0]), ad([1.00001, 2.0]), TOL) == []


def test_value_mismatch_reports_index():
    out = compare_annotated(ad([1.0, 2.0, 3.0]), ad([1.0, 2.5, 3.0]), TOL, where="rep")
    assert len(out) == 1
    assert "[1]" in out[0].message and "2.0" in out[0].message and "2.5" in out[0].message
    assert out[0].where == "rep"


def test_shape_mismatch():
    out = compare_annotated(ad([[1, 2], [3, 4]]), ad([[1, 2, 3], [4, 5, 6]]), TOL)
    assert len(out) == 1 and "shape" in out[0].message


def test_ndim_mismatch():
    out = compare_annotated(ad([1, 2]), ad([[1], [2]]), TOL)
    assert len(out) == 1 and "dimensions" in out[0].message


def test_zero_dim():
    assert compare_annotated(ad(1.0), ad(1.0), TOL) == []
    assert compare_annotated(ad(1.0), ad(2.0), TOL) != []


def test_nan_positions():
    assert compare_annotated(ad([math.nan, 1.0]), ad([math.nan, 1.0]), TOL) == []
    assert compare_annotated(ad([math.nan, 1.0]), ad([1.0, math.nan]), TOL) != []


def test_inf_sign():
    assert compare_annotated(ad([math.inf]), ad([math.inf]), TOL) == []
    assert compare_annotated(ad([math.inf]), ad([-math.inf]), TOL) != []


def test_strings_exact():
    a = AnnotatedData(np.array(["a", "b"], dtype=object))
    b = AnnotatedData(np.array(["a", "c"], dtype=object))
    assert compare_annotated(a, a, TOL) == []
    out = compare_annotated(a, b, TOL)
    assert len(out) == 1 and "'b'" in out[0].message


def test_string_vs_number():
    a = AnnotatedData(np.array(["1"], dtype=object))
    out = compare_annotated(a, ad([1.0]), TOL)
    assert len(out) == 1 and "strings" in out[0].message


def test_labels_text_exact():
    a = ad([1.0, 2.0], [["S1", "S2"]])
    b = ad([1.0, 2.0], [["S1", "S3"]])
    out = compare_annotated(a, b, TOL)
    assert len(out) == 1 and "labels" in out[0].where


def test_labels_numeric_use_tolerance():
    a = ad([1.0, 2.0], [["0", "0.1000001"]])
    b = ad([1.0, 2.0], [["0", "0.1"]])
    assert compare_annotated(a, b, TOL) == []
    c = ad([1.0, 2.0], [["0", "0.2"]])
    assert compare_annotated(c, b, TOL) != []


def test_labels_float_versus_text():
    a = ad([1.0, 2.0], [[0.0, 0.1]])
    b = ad([1.0, 2.0], [["0", "0.1"]])
    assert compare_annotated(a, b, TOL) == []


def test_labels_present_in_one_only():
    out = compare_annotated(ad([1.0, 2.0], [["a", "b"]]), ad([1.0, 2.0]), TOL)
    assert len(out) == 1 and "absent" in out[0].message


def test_label_count_mismatch_is_reported_with_shape_check():
    # shapes differ too; both problems may be listed but at least one must be
    out = compare_annotated(ad([1.0, 2.0], [["a", "b"]]), ad([1.0], [["a"]]), TOL)
    assert out


def test_listing_cap():
    n = cmp.MAX_LISTED + 7
    out = compare_annotated(ad(np.zeros(n)), ad(np.ones(n)), TOL)
    assert len(out) == cmp.MAX_LISTED + 1
    assert "7 more" in out[-1].message


def test_unknown_mode():
    out = compare_annotated(ad([1.0]), ad([1.0]), TOL, mode="bogus")
    assert len(out) == 1 and "unknown" in out[0].message


# ---------------------------------------------------------------- lastRow

def test_last_row_ignores_earlier_rows_and_row_count():
    a = ad([[0, 9], [1, 5], [2, 7.0]])
    e = ad([[0, 1], [2, 7.0]])
    assert compare_annotated(a, e, TOL, mode="lastRow") == []


def test_last_row_detects_difference():
    a = ad([[0, 9], [2, 7.5]])
    e = ad([[0, 1], [2, 7.0]])
    assert compare_annotated(a, e, TOL, mode="lastRow") != []


def test_last_row_column_labels_still_checked():
    a = ad([[1.0, 2.0]], [None, ["A", "B"]])
    e = ad([[1.0, 2.0]], [None, ["A", "C"]])
    assert compare_annotated(a, e, TOL, mode="lastRow") != []


def test_last_row_one_dimensional_column_vector():
    assert compare_annotated(ad([1.0, 2.0, 3.0]), ad([9.0, 3.0]), TOL, mode="lastRow") == []


def test_last_row_empty():
    out = compare_annotated(ad(np.zeros((0, 2))), ad([[1.0, 2.0]]), TOL, mode="lastRow")
    assert len(out) == 1 and "at least one row" in out[0].message


# ---------------------------------------------------------------- keyedRows

def test_keyed_rows_match_in_any_order_and_extra_rows():
    a = ad([[2.0, 20.0], [0.0, 0.5], [1.0, 10.0], [3.0, 99.0]])
    e = ad([[0.0, 0.5], [1.0, 10.0], [2.0, 20.0]])
    assert compare_annotated(a, e, TOL, mode="keyedRows") == []


def test_keyed_rows_missing_key():
    a = ad([[0.0, 0.5], [1.0, 10.0]])
    e = ad([[0.0, 0.5], [2.0, 20.0]])
    out = compare_annotated(a, e, TOL, mode="keyedRows")
    assert len(out) == 1 and "row 1" in out[0].message


def test_keyed_rows_value_mismatch():
    a = ad([[0.0, 0.5], [1.0, 11.0]])
    e = ad([[0.0, 0.5], [1.0, 10.0]])
    out = compare_annotated(a, e, TOL, mode="keyedRows")
    assert len(out) == 1 and "key 1.0" in out[0].message


def test_keyed_rows_key_tolerance():
    a = ad([[0.1 + 1e-9, 5.0]])
    e = ad([[0.1, 5.0]])
    assert compare_annotated(a, e, TOL, mode="keyedRows") == []


def test_keyed_rows_requires_2d_numeric():
    out = compare_annotated(ad([1.0, 2.0]), ad([1.0, 2.0]), TOL, mode="keyedRows")
    assert len(out) == 1 and "two-dimensional" in out[0].message


def test_keyed_rows_column_count():
    out = compare_annotated(ad([[0.0, 1.0]]), ad([[0.0, 1.0, 2.0]]), TOL, mode="keyedRows")
    assert out and "columns" in out[-1].message


# ---------------------------------------------------------------- plots

def p2(**cols):
    return rio.Plot2DData(columns=dict(cols))


def test_plot2d_equal_and_padding():
    a = p2(**{"c1.x": [0.0, 1.0, None], "c1.y": [5.0, 6.0, None]})
    assert cmp.compare_plot2d(a, a, TOL) == []


def test_plot2d_column_names_must_match():
    out = cmp.compare_plot2d(p2(a=[1.0]), p2(b=[1.0]), TOL)
    assert len(out) == 1 and "columns" in out[0].message


def test_plot2d_row_count():
    out = cmp.compare_plot2d(p2(a=[1.0, 2.0]), p2(a=[1.0]), TOL)
    assert len(out) == 1 and "rows" in out[0].message


def test_plot2d_value_and_padding_mismatch():
    out = cmp.compare_plot2d(p2(a=[1.0, 2.0]), p2(a=[1.0, 3.0]), TOL)
    assert len(out) == 1
    out = cmp.compare_plot2d(p2(a=[1.0, None]), p2(a=[1.0, 3.0]), TOL)
    assert len(out) == 1


def test_plot2d_strings():
    assert cmp.compare_plot2d(p2(a=["x"]), p2(a=["x"]), TOL) == []
    assert cmp.compare_plot2d(p2(a=["x"]), p2(a=["y"]), TOL) != []


def surf(z, kind="surface", index=0):
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    return rio.Surface(x=x, y=y, z=np.array(z, dtype=float), surface_type=kind, index=index)


def test_plot3d_equal():
    a = rio.Plot3DData(surfaces={"s": surf([[1, 2], [3, 4]])})
    assert cmp.compare_plot3d(a, a, TOL) == []


def test_plot3d_differences():
    a = rio.Plot3DData(surfaces={"s": surf([[1, 2], [3, 4]])})
    b = rio.Plot3DData(surfaces={"s": surf([[1, 2], [3, 5]])})
    out = cmp.compare_plot3d(a, b, TOL)
    assert len(out) == 1 and "surface 's' z" in out[0].where
    c = rio.Plot3DData(surfaces={"t": surf([[1, 2], [3, 4]])})
    assert cmp.compare_plot3d(a, c, TOL)
    d = rio.Plot3DData(surfaces={"s": surf([[1, 2], [3, 4]], kind="contour")})
    assert cmp.compare_plot3d(a, d, TOL)


# ---------------------------------------------------------------- whole cases

def make_case(tmp_path, actual_values=None, with_plot=False):
    """Create case/ with expected files and out/ with actual files; return (case, out, settings)."""
    case = tmp_path / "case"
    out = tmp_path / "out"
    case.mkdir()
    out.mkdir()
    expected = AnnotatedData(np.array([[0.0, 1.0], [1.0, 2.0]]), [None, ["time", "S1"]])
    rio.write_csv(str(case / "r.csv"), expected)
    if not (isinstance(actual_values, str) and actual_values == "skip"):
        actual = expected if actual_values is None else AnnotatedData(np.array(actual_values), [None, ["time", "S1"]])
        rio.write_csv(str(out / "r.csv"), actual)
    settings = {
        "schemaVersion": 1,
        "tolerances": {"absolute": 1e-7, "relative": 1e-4},
        "provenance": {"source": "analytical"},
        "backends": ["roadrunner"],
        "reports": {"r": {"file": "r.csv", "format": "csv", "ndim": 2, "labels": {"columns": True}}},
    }
    if with_plot:
        pl = p2(**{"c.x": [0.0, 1.0], "c.y": [1.0, 2.0]})
        rio.write_plot2d_csv(str(case / "p.csv"), pl)
        rio.write_plot2d_csv(str(out / "p.csv"), pl)
        settings["plots"] = {"p": {"file": "p.csv", "format": "csv", "type": "plot2D"}}
    (case / "00001.settings.json").write_text(json.dumps(settings))
    return str(case), str(out), settings


def test_compare_case_passes(tmp_path):
    case, out, _ = make_case(tmp_path, with_plot=True)
    res = cmp.compare_case(case, out)
    assert [r.name for r in res] == ["r", "p"]
    assert all(r.ok for r in res)


def test_compare_case_detects_mismatch(tmp_path):
    case, out, _ = make_case(tmp_path, actual_values=[[0.0, 1.0], [1.0, 2.5]])
    res = cmp.compare_case(case, out)
    assert not res[0].ok


def test_compare_case_missing_output(tmp_path):
    case, out, _ = make_case(tmp_path, actual_values="skip")
    res = cmp.compare_case(case, out)
    assert not res[0].ok and "not produced" in str(res[0].mismatches[0])


def test_compare_case_unreadable_output(tmp_path):
    case, out, _ = make_case(tmp_path)
    (tmp_path / "out" / "r.csv").write_text("a,b\n1,2,3\n")
    res = cmp.compare_case(case, out)
    assert not res[0].ok and "cannot compare" in str(res[0].mismatches[0])


def test_compare_case_tolerance_override(tmp_path):
    case, out, settings = make_case(tmp_path, actual_values=[[0.0, 1.0], [1.0, 2.1]])
    assert not cmp.compare_case(case, out)[0].ok
    settings["reports"]["r"]["tolerances"] = {"relative": 0.1}
    assert cmp.compare_case(case, out, settings)[0].ok


def test_compare_case_mode_from_settings(tmp_path):
    case, out, settings = make_case(tmp_path, actual_values=[[0.0, 7.0], [0.5, 7.0], [1.0, 2.0]])
    assert not cmp.compare_case(case, out)[0].ok
    settings["reports"]["r"]["compare"] = {"mode": "lastRow"}
    assert cmp.compare_case(case, out, settings)[0].ok


def test_compare_case_needs_one_settings_file(tmp_path):
    (tmp_path / "x").mkdir()
    with pytest.raises(FileNotFoundError):
        cmp.compare_case(str(tmp_path / "x"), str(tmp_path))


# ---------------------------------------------------------------- command line

TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")


def run_cli(*args):
    env = dict(os.environ, PYTHONPATH=TOOLS)
    return subprocess.run([sys.executable, "-m", "sed2suite.compare", *args], capture_output=True, text=True, env=env)


def test_cli_exit_codes(tmp_path):
    case, out, _ = make_case(tmp_path)
    p = run_cli(case, out)
    assert p.returncode == 0 and "1 of 1 items match" in p.stdout
    p = run_cli(case, out, "-v")
    assert "PASS" in p.stdout
    rio.write_csv(os.path.join(out, "r.csv"), AnnotatedData(np.array([[0.0, 1.0], [1.0, 9.0]]), [None, ["time", "S1"]]))
    p = run_cli(case, out)
    assert p.returncode == 1 and "FAIL" in p.stdout
    p = run_cli(str(tmp_path / "nonexistent"), out)
    assert p.returncode == 2
