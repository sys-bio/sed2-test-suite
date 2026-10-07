import copy
import json

import pytest

from sed2suite import settings as st

GOOD = {
    "schemaVersion": 1,
    "tolerances": {"absolute": 1e-7, "relative": 1e-4},
    "provenance": {"source": "analytical", "notes": "closed form"},
    "backends": ["roadrunner", "copasi"],
    "reports": {"rep1": {"file": "00001.rep1.csv", "format": "csv", "ndim": 2, "labels": {"columns": True}}},
    "plots": {
        "plt1": {"file": "00001.plt1.csv", "format": "csv", "type": "plot2D"},
        "plt2": {"file": "00001.plt2.h5", "format": "h5", "type": "plot3D"},
    },
}


def good():
    return copy.deepcopy(GOOD)


def test_valid():
    assert st.validate_settings(good()) == []


def test_valid_minimal():
    s = good()
    del s["reports"], s["plots"]
    assert st.validate_settings(s) == []


def test_simulation_provenance_needs_simulators():
    s = good()
    s["provenance"] = {"source": "simulation"}
    out = st.validate_settings(s)
    assert out and "simulators" in out[0]
    s["provenance"]["simulators"] = [{"name": "roadrunner", "version": "2.10.0"}]
    assert st.validate_settings(s) == []


def test_simulators_not_empty_when_simulation():
    s = good()
    s["provenance"] = {"source": "simulation", "simulators": []}
    assert st.validate_settings(s)


@pytest.mark.parametrize("key", ["schemaVersion", "tolerances", "provenance", "backends"])
def test_required_top_level(key):
    s = good()
    del s[key]
    out = st.validate_settings(s)
    assert out and key in out[0]


def test_unknown_top_level_key():
    s = good()
    s["extra"] = 1
    assert st.validate_settings(s)


def test_wrong_schema_version():
    s = good()
    s["schemaVersion"] = 2
    assert st.validate_settings(s)


def test_backends_rules():
    s = good()
    s["backends"] = []
    assert st.validate_settings(s)
    s["backends"] = ["roadrunner", "roadrunner"]
    assert st.validate_settings(s)
    s["backends"] = ["RoadRunner"]
    assert st.validate_settings(s)


def test_negative_tolerance():
    s = good()
    s["tolerances"]["absolute"] = -1
    assert st.validate_settings(s)


def test_csv_report_requires_ndim():
    s = good()
    del s["reports"]["rep1"]["ndim"]
    out = st.validate_settings(s)
    assert out and "ndim" in out[0]


def test_h5_report_without_ndim_ok():
    s = good()
    s["reports"]["rep1"] = {"file": "00001.rep1.h5", "format": "h5"}
    assert st.validate_settings(s) == []


def test_report_ndim_range():
    s = good()
    s["reports"]["rep1"]["ndim"] = 3
    assert st.validate_settings(s)


def test_report_compare_mode():
    s = good()
    s["reports"]["rep1"]["compare"] = {"mode": "lastRow", "note": "adaptive steps"}
    assert st.validate_settings(s) == []
    s["reports"]["rep1"]["compare"] = {"mode": "sometimes"}
    assert st.validate_settings(s)


def test_file_must_be_plain_name():
    s = good()
    s["reports"]["rep1"]["file"] = "sub/rep1.csv"
    assert st.validate_settings(s)
    s["reports"]["rep1"]["file"] = "..\\rep1.csv"
    assert st.validate_settings(s)


def test_bad_ids():
    s = good()
    s["reports"]["1bad"] = s["reports"].pop("rep1")
    assert st.validate_settings(s)


def test_plot2d_requires_csv_plot3d_requires_h5():
    s = good()
    s["plots"]["plt1"]["format"] = "h5"
    assert st.validate_settings(s)
    s = good()
    s["plots"]["plt2"]["format"] = "csv"
    assert st.validate_settings(s)


def test_extension_must_match_format():
    s = good()
    s["reports"]["rep1"]["file"] = "00001.rep1.txt"
    out = st.validate_settings(s)
    assert any("extension" in p for p in out)


def test_id_sets_checked():
    s = good()
    out = st.validate_settings(s, report_ids=["rep1", "rep2"], plot_ids=["plt1"])
    assert any("rep2" in p and "no entry" in p for p in out)
    assert any("plt2" in p and "not in the SED2 document" in p for p in out)
    assert st.validate_settings(s, report_ids=["rep1"], plot_ids=["plt1", "plt2"]) == []


def test_id_sets_skipped_when_none():
    assert st.validate_settings(good(), report_ids=None, plot_ids=None) == []


def test_files_must_exist(tmp_path):
    s = good()
    out = st.validate_settings(s, case_dir=str(tmp_path))
    assert len([p for p in out if "does not exist" in p]) == 3
    for name in ("00001.rep1.csv", "00001.plt1.csv", "00001.plt2.h5"):
        (tmp_path / name).write_text("x")
    assert st.validate_settings(s, case_dir=str(tmp_path)) == []


def test_find_and_load(tmp_path):
    with pytest.raises(FileNotFoundError):
        st.find_settings_file(str(tmp_path))
    p = tmp_path / "00001.settings.json"
    p.write_text(json.dumps(good()))
    assert st.find_settings_file(str(tmp_path)) == str(p)
    assert st.load_settings(str(p)) == good()
    (tmp_path / "00002.settings.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        st.find_settings_file(str(tmp_path))
