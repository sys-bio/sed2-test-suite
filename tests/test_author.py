import json
import os

import numpy as np
import pytest

from sed2suite import author, compare, results_io as rio, settings as st, validate_suite

DOC = {"constants": {"v": [1, 2, 3]}, "outputs": {"r": {"_type": "report", "data": "#constants:v"}}}


def make(tmp_path, number=7, **kw):
    expected = kw.pop("expected", {"r": rio.AnnotatedData(np.array([1.0, 2.0, 3.0]), [None])})
    return author.write_case(number, "Reports a vector.", DOC, expected, root=str(tmp_path), **kw)


def test_files_and_settings(tmp_path):
    folder = make(tmp_path, derivation="The vector is reported unchanged.")
    names = sorted(os.listdir(folder))
    assert names == ["00007.description.md", "00007.r.csv", "00007.sed2.json", "00007.settings.json"]
    s = st.load_settings(os.path.join(folder, "00007.settings.json"))
    assert s["reports"]["r"] == {"file": "00007.r.csv", "format": "csv", "ndim": 1, "dtype": "number"}
    assert s["backends"] == [] and s["provenance"] == {"source": "analytical"} and s["tolerances"] == author.DEFAULT_TOLERANCES
    assert json.load(open(os.path.join(folder, "00007.sed2.json")))["version"] == "v1.0.0"
    assert rio.read_csv(os.path.join(folder, "00007.r.csv"), ndim=1).values.tolist() == [1.0, 2.0, 3.0]


def test_description_has_sections_sign_off_and_tags(tmp_path):
    text = open(os.path.join(make(tmp_path, derivation="Reported unchanged.", notes="A note.",
                                  tolerance_note="Tight: exact copy."), "00007.description.md")).read()
    for part in ("# Test 00007", "Reports a vector.", "## Expected results", "Reported unchanged.", "## Notes", "A note.",
                 "## Sign-off", "derived by hand", "Tight: exact copy.", author.BACKENDS_LINE, "<!-- tags:begin -->"):
        assert part in text
    assert "semanticTags: analytical, constants-only" in text


def test_the_case_folder_is_emptied_first(tmp_path):
    folder = make(tmp_path)
    open(os.path.join(folder, "stray.txt"), "w").close()
    author.reset_written()   # a later build run
    make(tmp_path)
    assert not os.path.exists(os.path.join(folder, "stray.txt"))


def test_labelled_two_dimensional_and_hdf5(tmp_path):
    pytest.importorskip("h5py")
    two = rio.AnnotatedData(np.arange(6.0).reshape(2, 3), [["a", "b"], ["x", "y", "z"]])
    three = rio.AnnotatedData(np.arange(8.0).reshape(2, 2, 2), [["a", "b"], None, None], ["d0", "", ""])
    folder = make(tmp_path, expected={"two": two, "three": three})
    s = st.load_settings(os.path.join(folder, "00007.settings.json"))
    assert s["reports"]["two"]["labels"] == {"rows": True, "columns": True}
    assert s["reports"]["three"]["format"] == "h5" and s["reports"]["three"]["file"] == "00007.three.h5"
    assert "ndim" not in s["reports"]["three"]
    text = open(os.path.join(folder, "00007.description.md")).read()
    assert "labeled-data" in text and "multi-dimensional" in text


def test_tolerances_compare_and_provenance(tmp_path):
    folder = make(tmp_path, tolerances={"absolute": 1e-9, "relative": 1e-6}, report_tolerances={"r": {"relative": 0.01}},
                  compare={"r": {"mode": "lastRow"}}, source="simulation",
                  simulators=[{"name": "roadrunner", "version": "2.10.0"}], provenance_notes="checked by hand")
    s = st.load_settings(os.path.join(folder, "00007.settings.json"))
    assert s["tolerances"]["absolute"] == 1e-9 and s["reports"]["r"]["tolerances"] == {"relative": 0.01}
    assert s["reports"]["r"]["compare"] == {"mode": "lastRow"}
    assert s["provenance"]["source"] == "simulation" and s["provenance"]["notes"] == "checked by hand"
    assert "from simulators that agree" in open(os.path.join(folder, "00007.description.md")).read()


def test_inputs_and_antimony(tmp_path):
    pytest.importorskip("antimony")
    folder = make(tmp_path, antimony={"": "model m\n  species S1 = 1\nend\n", "second": "model n\n  species X = 2\nend"},
                  inputs={"00007.data.csv": "a,b\n1,2\n", "00007.bin": b"\x00\x01"})
    names = sorted(os.listdir(folder))
    for n in ("00007.ant", "00007.sbml", "00007.second.ant", "00007.second.sbml", "00007.data.csv", "00007.bin"):
        assert n in names
    assert "<species" in open(os.path.join(folder, "00007.sbml")).read()


def test_plots(tmp_path):
    pytest.importorskip("h5py")
    p2 = rio.Plot2DData({"c.x": [0.0, 1.0], "c.y": [1.0, 2.0]})
    p3 = rio.Plot3DData({"s": rio.Surface(np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2)), "heatMap")})
    folder = make(tmp_path, plots={"p": p2, "q": p3})
    s = st.load_settings(os.path.join(folder, "00007.settings.json"))
    assert s["plots"]["p"] == {"file": "00007.p_as_data.csv", "format": "csv", "type": "plot2D"}
    assert s["plots"]["q"] == {"file": "00007.q_as_data.h5", "format": "h5", "type": "plot3D"}
    assert rio.read_plot3d_h5(os.path.join(folder, "00007.q_as_data.h5")).surfaces["s"].surface_type == "heatMap"


def test_a_written_case_is_consistent_enough_for_the_comparer(tmp_path):
    folder = make(tmp_path)
    s = st.load_settings(os.path.join(folder, "00007.settings.json"))
    assert all(r.ok for r in compare.compare_case(folder, folder, s))


def test_case_numbers():
    assert author.case_id(1) == "00001" and author.case_id(123) == "00123"
    for bad in (0, 100000):
        with pytest.raises(ValueError):
            author.case_id(bad)


def test_registry_records_cases_and_rejects_duplicates(tmp_path):
    author.reset_written()
    make(tmp_path, number=7)
    make(tmp_path, number=9)
    assert author.written_numbers() == [7, 9]
    with pytest.raises(ValueError, match="already written"):
        make(tmp_path, number=7)
    author.reset_written()
    assert author.written_numbers() == []
    make(tmp_path, number=7)   # a fresh run may write it again
    author.reset_written()


def _admit(folder, number=7):
    """What `suite_runner --admit` does to a case: list the backends in settings.json and the sign-off line."""
    spath = os.path.join(folder, f"{number:05d}.settings.json")
    s = json.load(open(spath))
    s["backends"] = ["roadrunner"]
    open(spath, "w").write(json.dumps(s, indent=2) + "\n")
    dpath = os.path.join(folder, f"{number:05d}.description.md")
    text = open(dpath).read().replace(author.BACKENDS_LINE, "- Backends: roadrunner (admitted).")
    open(dpath, "w").write(text)


def test_rebuilding_an_unchanged_case_keeps_its_admission(tmp_path):
    folder = make(tmp_path, derivation="Same.")
    _admit(folder)
    author.reset_written()
    make(tmp_path, derivation="Same.")
    s = st.load_settings(os.path.join(folder, "00007.settings.json"))
    assert s["backends"] == ["roadrunner"]
    assert "- Backends: roadrunner (admitted)." in open(os.path.join(folder, "00007.description.md")).read()
    assert not os.path.exists(folder + ".new")


def test_rebuilding_a_changed_case_drops_its_admission(tmp_path):
    folder = make(tmp_path, derivation="Same.")
    _admit(folder)
    author.reset_written()
    make(tmp_path, derivation="Different words.")
    s = st.load_settings(os.path.join(folder, "00007.settings.json"))
    assert s["backends"] == []
    assert author.BACKENDS_LINE in open(os.path.join(folder, "00007.description.md")).read()
    author.reset_written()
    make(tmp_path, derivation="Different words.", expected={"r": rio.AnnotatedData(np.array([1.0, 2.0, 4.0]), [None])})
    assert rio.read_csv(os.path.join(folder, "00007.r.csv"), ndim=1).values.tolist() == [1.0, 2.0, 4.0]


def test_a_failed_write_leaves_nothing_behind(tmp_path):
    pytest.importorskip("antimony")
    with pytest.raises(Exception):
        make(tmp_path, antimony={"": "model m\n  A <-> B\nend"})
    assert os.listdir(str(tmp_path)) == []
