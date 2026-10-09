import math
import random

import numpy as np
import pytest

from sed2suite import results_io as rio
from sed2suite.results_io import AnnotatedData, FormatError


# ---------------------------------------------------------------- numbers

@pytest.mark.parametrize("value, text", [
    (0.1, "0.1"), (1.5e-4, "0.00015"), (1e-5, "1e-05"), (3.0, "3.0"), (-2.5, "-2.5"), (1e22, "1e+22"),
    (math.nan, "nan"), (math.inf, "inf"), (-math.inf, "-inf"),
    (True, "1"), (False, "0"), (7, "7"), (np.int64(-3), "-3"), (np.float64(0.5), "0.5"),
])
def test_format_number(value, text):
    assert rio.format_number(value) == text


@pytest.mark.parametrize("text, value", [
    ("1", 1.0), ("-1.5", -1.5), (".5", 0.5), ("5.", 5.0), ("1e3", 1000.0), ("1.5E-4", 1.5e-4), ("+2", 2.0),
    ("inf", math.inf), ("-inf", -math.inf), ("INF", math.inf), ("-Inf", -math.inf), (" 3 ", 3.0),
])
def test_parse_number(text, value):
    assert rio.parse_number(text) == value


def test_parse_nan_any_case():
    assert math.isnan(rio.parse_number("nan"))
    assert math.isnan(rio.parse_number("NaN"))


@pytest.mark.parametrize("text", ["", "abc", "1_000", "0x10", "1,5", "--1", "1e", "infinity "[:-1] + "x"])
def test_parse_number_rejects(text):
    with pytest.raises(FormatError):
        rio.parse_number(text)


def test_float_round_trip_is_exact():
    rng = random.Random(1)
    for _ in range(500):
        x = rng.uniform(-1, 1) * 10 ** rng.randint(-300, 300)
        assert rio.parse_number(rio.format_number(x)) == x


# ---------------------------------------------------------------- CSV text, exact

def test_csv_0d():
    assert rio.csv_text(AnnotatedData(np.array(2.5))) == "2.5\n"


def test_csv_1d_unlabeled():
    assert rio.csv_text(AnnotatedData(np.array([1.0, 2.0, math.nan]))) == "1.0\n2.0\nnan\n"


def test_csv_1d_labeled():
    d = AnnotatedData(np.array([1.0, 2.0]), [["a", "b"]])
    assert rio.csv_text(d) == "a,1.0\nb,2.0\n"


def test_csv_2d_no_labels():
    assert rio.csv_text(AnnotatedData(np.array([[1.0, 2.0], [3.0, 4.0]]))) == "1.0,2.0\n3.0,4.0\n"


def test_csv_2d_column_labels_only():
    d = AnnotatedData(np.array([[0.0, 1.5e-4]]), [None, ["time", "S1"]])
    assert rio.csv_text(d) == "time,S1\n0.0,0.00015\n"


def test_csv_2d_both_labels_has_empty_corner():
    d = AnnotatedData(np.array([[1.0, 2.0]]), [["r"], ["c1", "c2"]])
    assert rio.csv_text(d) == ",c1,c2\nr,1.0,2.0\n"


def test_csv_numeric_labels_formatted_as_numbers():
    d = AnnotatedData(np.array([1.0, 2.0]), [[0.0, 0.5]])
    assert rio.csv_text(d) == "0.0,1.0\n0.5,2.0\n"


def test_csv_strings_quoted_when_needed_and_lf_endings():
    v = np.array(["plain", 'has,comma', 'has"quote', "two\nlines"], dtype=object)
    text = rio.csv_text(AnnotatedData(v))
    assert text == 'plain\n"has,comma"\n"has""quote"\n"two\nlines"\n'
    assert "\r" not in text


def test_csv_booleans_as_one_zero():
    assert rio.csv_text(AnnotatedData(np.array([True, False]))) == "1\n0\n"


def test_csv_refuses_three_dimensions():
    with pytest.raises(FormatError):
        rio.csv_text(AnnotatedData(np.zeros((1, 1, 1))))


# ---------------------------------------------------------------- CSV round trips

def _roundtrip(tmp_path, data, **kw):
    p = tmp_path / "x.csv"
    rio.write_csv(p, data)
    return rio.read_csv(p, **kw)


def test_roundtrip_0d(tmp_path):
    r = _roundtrip(tmp_path, AnnotatedData(np.array(math.inf)), ndim=0)
    assert r.ndim == 0 and r.values.item() == math.inf


def test_roundtrip_1d_labeled(tmp_path):
    d = AnnotatedData(np.array([1.0, -2.0]), [["a", "b"]])
    r = _roundtrip(tmp_path, d, ndim=1, row_labels=True)
    assert r.labels == [["a", "b"]] and r.values.tolist() == [1.0, -2.0]


def test_roundtrip_2d_both_labels(tmp_path):
    d = AnnotatedData(np.array([[1.0, math.nan], [math.inf, -math.inf]]), [["r1", "r2"], ["c1", "c2"]])
    r = _roundtrip(tmp_path, d, ndim=2, row_labels=True, column_labels=True)
    assert r.labels == [["r1", "r2"], ["c1", "c2"]]
    assert math.isnan(r.values[0, 1]) and r.values[1, 0] == math.inf and r.values[1, 1] == -math.inf


def test_roundtrip_2d_no_labels(tmp_path):
    d = AnnotatedData(np.arange(6.0).reshape(2, 3))
    r = _roundtrip(tmp_path, d, ndim=2)
    assert r.labels == [None, None] and r.values.tolist() == d.values.tolist()


def test_roundtrip_strings_with_awkward_text(tmp_path):
    v = np.array([["a,b", 'q"t'], ["x\ny", ""]], dtype=object)
    r = _roundtrip(tmp_path, AnnotatedData(v), ndim=2, dtype="string")
    assert r.values.tolist() == v.tolist()


def test_read_rejects_ragged_rows(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("1,2\n3\n")
    with pytest.raises(FormatError, match="row 1"):
        rio.read_csv(p, ndim=2)


def test_read_rejects_bad_number(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("1,two\n")
    with pytest.raises(FormatError):
        rio.read_csv(p, ndim=2)


def test_read_0d_requires_single_cell(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("1,2\n")
    with pytest.raises(FormatError):
        rio.read_csv(p, ndim=0)


def test_read_1d_rejects_two_value_columns(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("1,2\n3,4\n")
    with pytest.raises(FormatError):
        rio.read_csv(p, ndim=1)


def test_read_header_width_must_match(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b,c\n1,2\n")
    with pytest.raises(FormatError):
        rio.read_csv(p, ndim=2, column_labels=True)


def test_annotated_data_checks_label_count():
    with pytest.raises(FormatError):
        AnnotatedData(np.zeros(3), [["a", "b"]])


# ---------------------------------------------------------------- HDF5 reports

def test_h5_roundtrip_3d_with_labels_and_dims(tmp_path):
    v = np.arange(24.0).reshape(2, 3, 4)
    d = AnnotatedData(v, [["p", "q"], None, [0.0, 0.5, 1.0, 1.5]], ["scan", "", "time"])
    p = tmp_path / "x.h5"
    rio.write_h5(p, d)
    r = rio.read_h5(p)
    assert r.values.tolist() == v.tolist()
    assert r.labels == [["p", "q"], None, [0.0, 0.5, 1.0, 1.5]]
    assert r.dims == ["scan", "", "time"]


def test_h5_roundtrip_strings(tmp_path):
    v = np.array([["a", "b"], ["c", "\u00e9"]], dtype=object)
    p = tmp_path / "x.h5"
    rio.write_h5(p, AnnotatedData(v))
    r = rio.read_h5(p)
    assert r.is_string and r.values.tolist() == v.tolist()


def test_h5_special_values(tmp_path):
    v = np.array([math.nan, math.inf, -math.inf, 1.0])
    p = tmp_path / "x.h5"
    rio.write_h5(p, AnnotatedData(v))
    r = rio.read_h5(p).values
    assert math.isnan(r[0]) and r[1] == math.inf and r[2] == -math.inf and r[3] == 1.0


def test_h5_missing_data_dataset(tmp_path):
    import h5py
    p = tmp_path / "x.h5"
    with h5py.File(p, "w"):
        pass
    with pytest.raises(FormatError):
        rio.read_h5(p)


# ---------------------------------------------------------------- Plot2D

def test_plot2d_text_pads_with_empty_cells_and_keeps_nan():
    p = rio.Plot2DData({"c1.x": [0.0, 1.0, 2.0], "c1.y": [5.0, math.nan], "c2.x": []})
    assert rio.plot2d_csv_text(p) == "c1.x,c1.y,c2.x\n0.0,5.0,\n1.0,nan,\n2.0,,\n"


def test_plot2d_roundtrip(tmp_path):
    p = rio.Plot2DData({"c1.x": [0.0, 1.0, 2.0], "c1.y": [5.0, math.nan], "lab": ["a", "b", "c"]})
    path = tmp_path / "p.csv"
    rio.write_plot2d_csv(path, p)
    r = rio.read_plot2d_csv(path)
    assert list(r.columns) == ["c1.x", "c1.y", "lab"]
    assert r.columns["c1.x"] == [0.0, 1.0, 2.0]
    assert r.columns["c1.y"][0] == 5.0 and math.isnan(r.columns["c1.y"][1]) and r.columns["c1.y"][2] is None
    assert r.columns["lab"] == ["a", "b", "c"]


def test_plot2d_rejects_duplicate_columns(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("a,a\n1,2\n")
    with pytest.raises(FormatError):
        rio.read_plot2d_csv(path)


# ---------------------------------------------------------------- Plot3D

def test_plot3d_roundtrip_keeps_the_stored_order(tmp_path):
    """The groups come back in the order they were written (ascending `order`), not by name or by `index`."""
    z = np.arange(6.0).reshape(2, 3)
    s_b = rio.Surface(np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0]), z, "heatMap", 0)
    s_a = rio.Surface(np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0]), z * 2, "surfaceMesh", 1)
    s_c = rio.Surface(np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0]), z * 3, "contour", 2)
    path = tmp_path / "p.h5"
    rio.write_plot3d_h5(path, rio.Plot3DData({"a": s_a, "c": s_c, "b": s_b}))
    r = rio.read_plot3d_h5(path)
    assert list(r.surfaces) == ["a", "c", "b"]
    assert r.surfaces["a"].surface_type == "surfaceMesh" and r.surfaces["a"].index == 1
    assert r.surfaces["a"].z.tolist() == (z * 2).tolist()
    assert r.surfaces["b"].x.tolist() == [0.0, 1.0]
