import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from sed2suite import author, results_io as rio, tags, validate_suite as vs

VOCAB = tags.load_vocabulary()
TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")

DOC = {
    "version": "v1.0.0",
    "constants": {"k_array": [10, 20, 30]},
    "outputs": {"rep1": {"_type": "report", "data": "#constants:k_array"}},
}
SETTINGS = {
    "schemaVersion": 1,
    "tolerances": {"absolute": 1e-7, "relative": 1e-4},
    "provenance": {"source": "analytical"},
    "backends": ["roadrunner", "copasi", "opencor"],
    "reports": {"rep1": {"file": "00001.rep1.csv", "format": "csv", "ndim": 1}},
}


def make_case(root, doc=None, settings=None, semantic=("analytical", "constants-only"), number="00001",
              default_csv=True, signoff=True):
    d = root / number
    d.mkdir()
    doc = copy.deepcopy(doc or DOC)
    settings = copy.deepcopy(settings or SETTINGS)
    (d / f"{number}.sed2.json").write_text(json.dumps(doc, indent=2))
    (d / f"{number}.settings.json").write_text(json.dumps(settings, indent=2))
    if default_csv:
        rio.write_csv(str(d / f"{number}.rep1.csv"), rio.AnnotatedData(np.array([10.0, 20.0, 30.0])))
    if signoff:
        text = author.description_text(number, "A test.", "Worked out by hand: the array is 10, 20, 30.", "", "analytical", "")
        text = text.replace(author.BACKENDS_LINE, "- Backends: " + ", ".join(settings["backends"]) + " (agree; admitted).")
        (d / f"{number}.description.md").write_text(text, encoding="utf-8", newline="\n")
    tags.process_case(str(d / f"{number}.sed2.json"), VOCAB, list(semantic))
    return d


def errors(problems):
    return [p for p in problems if p.level == "error"]


def messages(problems):
    return " | ".join(str(p) for p in errors(problems))


def test_valid_case_has_no_errors(tmp_path):
    d = make_case(tmp_path)
    assert errors(vs.validate_case(str(d), VOCAB)) == []


def test_document_helpers():
    doc = {"outputs": {"r": {"_type": "report"}, "p": {"_type": "plot2D"}, "q": {"_type": "plot3D"}},
           "tasks": {"m": {"_type": "dataImport", "location": "d.csv", "sub": [{"location": "e.csv"}]}}}
    assert vs.output_ids(doc) == (["r"], {"p": "plot2D", "q": "plot3D"})
    assert vs.input_locations(doc) == {"d.csv", "e.csv"}


def test_missing_required_files(tmp_path):
    d = make_case(tmp_path)
    os.remove(d / "00001.description.md")
    os.remove(d / "00001.settings.json")
    m = messages(vs.validate_case(str(d), VOCAB))
    assert "00001.description.md" in m and "00001.settings.json" in m


def test_bad_folder_name(tmp_path):
    d = make_case(tmp_path, number="abc")
    assert "five digits" in messages(vs.validate_case(str(d), VOCAB))


def test_invalid_document_reported_by_libsed2(tmp_path):
    pytest.importorskip("libsed2")
    doc = copy.deepcopy(DOC)
    doc["outputs"]["rep1"]["data"] = "#constants:nonexistent"
    d = make_case(tmp_path, doc=doc)
    assert errors(vs.validate_case(str(d), VOCAB))
    assert errors(vs.validate_case(str(d), VOCAB, use_libsed2=False)) == []


def test_no_report(tmp_path):
    doc = {"version": "v1.0.0", "constants": {"a": 1}}
    s = copy.deepcopy(SETTINGS)
    del s["reports"]
    d = make_case(tmp_path, doc=doc, settings=s)
    assert "no report" in messages(vs.validate_case(str(d), VOCAB, use_libsed2=False))


def test_settings_and_document_disagree(tmp_path):
    s = copy.deepcopy(SETTINGS)
    s["reports"]["other"] = {"file": "00001.rep1.csv", "format": "csv", "ndim": 1}
    d = make_case(tmp_path, settings=s)
    assert "other" in messages(vs.validate_case(str(d), VOCAB, use_libsed2=False))


def test_missing_result_file(tmp_path):
    d = make_case(tmp_path)
    os.remove(d / "00001.rep1.csv")
    assert "does not exist" in messages(vs.validate_case(str(d), VOCAB, use_libsed2=False))


def test_unreadable_result_and_dtype(tmp_path):
    d = make_case(tmp_path)
    (d / "00001.rep1.csv").write_text("1\nabc\n")
    assert "cannot be read" in messages(vs.validate_case(str(d), VOCAB, use_libsed2=False))


def test_h5_dtype_must_match_settings(tmp_path):
    s = copy.deepcopy(SETTINGS)
    s["reports"]["rep1"] = {"file": "00001.rep1.h5", "format": "h5", "dtype": "number"}
    d = make_case(tmp_path, settings=s, semantic=("analytical", "constants-only", "multi-dimensional"),
                  default_csv=False)
    strings = np.full((2, 2, 2), "a", dtype=object)
    rio.write_h5(str(d / "00001.rep1.h5"), rio.AnnotatedData(strings))
    assert "dtype is number" in messages(vs.validate_case(str(d), VOCAB, use_libsed2=False))


def test_stray_files(tmp_path):
    d = make_case(tmp_path)
    (d / "00001.stray.csv").write_text("1\n")
    (d / "notes.txt").write_text("x")
    (d / "00001.nonplot.png").write_bytes(b"png")
    problems = vs.validate_case(str(d), VOCAB, use_libsed2=False)
    assert "00001.stray.csv" in messages(problems)
    warned = " ".join(p.message for p in problems if p.level == "warning")
    assert "notes.txt" in warned and "nonplot" in warned


def test_input_files_are_not_stray(tmp_path):
    doc = copy.deepcopy(DOC)
    doc["tasks"] = {"d": {"_type": "csvImport", "location": "00001.csv"}}
    d = make_case(tmp_path, doc=doc, semantic=("analytical",))
    (d / "00001.csv").write_text("1,2\n")
    assert errors(vs.validate_case(str(d), VOCAB, use_libsed2=False)) == []
    os.remove(d / "00001.csv")
    assert "does not exist" in messages(vs.validate_case(str(d), VOCAB, use_libsed2=False))


def test_tag_block_out_of_date(tmp_path):
    d = make_case(tmp_path)
    p = d / "00001.description.md"
    p.write_text(p.read_text().replace("componentTags: ", "componentTags: report, "))
    assert "out-of-date" in messages(vs.validate_case(str(d), VOCAB, use_libsed2=False))


def test_derived_semantic_tags_checked(tmp_path):
    d = make_case(tmp_path, semantic=("analytical",))  # constants-only missing
    assert "constants-only" in messages(vs.validate_case(str(d), VOCAB, use_libsed2=False))
    d2 = make_case(tmp_path, semantic=("analytical", "constants-only", "special-values"), number="00002")
    rio.write_csv(str(d2 / "00001.rep1.csv"), rio.AnnotatedData(np.array([1.0, 2.0])))
    assert "special-values" in messages(vs.validate_case(str(d2), VOCAB, use_libsed2=False))


def test_derive_special_values(tmp_path):
    d = make_case(tmp_path, semantic=("analytical", "constants-only", "special-values"))
    rio.write_csv(str(d / "00001.rep1.csv"), rio.AnnotatedData(np.array([1.0, float("nan")])))
    assert errors(vs.validate_case(str(d), VOCAB, use_libsed2=False)) == []


def test_h5_report_must_be_3d(tmp_path):
    s = copy.deepcopy(SETTINGS)
    s["reports"]["rep1"] = {"file": "00001.rep1.h5", "format": "h5"}
    d = make_case(tmp_path, settings=s, semantic=("analytical", "constants-only", "multi-dimensional"),
                  default_csv=False)
    rio.write_h5(str(d / "00001.rep1.h5"), rio.AnnotatedData(np.zeros((2, 2))))
    assert "3 or more" in messages(vs.validate_case(str(d), VOCAB, use_libsed2=False))
    rio.write_h5(str(d / "00001.rep1.h5"), rio.AnnotatedData(np.zeros((2, 2, 2))))
    assert errors(vs.validate_case(str(d), VOCAB, use_libsed2=False)) == []


def test_plot_type_must_match_document(tmp_path):
    doc = copy.deepcopy(DOC)
    doc["outputs"]["plt"] = {"_type": "plot2D", "xAxis": {}, "curves": {}}
    s = copy.deepcopy(SETTINGS)
    s["plots"] = {"plt": {"file": "00001.plt_as_data.h5", "format": "h5", "type": "plot3D"}}
    d = make_case(tmp_path, doc=doc, settings=s, semantic=("analytical", "constants-only", "multi-dimensional"))
    (d / "00001.plt_as_data.h5").write_bytes(b"")
    assert "plot 'plt' is plot3D but the document says plot2D" in messages(vs.validate_case(str(d), VOCAB, False))


def test_sbml_must_match_ant(tmp_path):
    d = make_case(tmp_path)
    (d / "00001.ant").write_text("model m\n species S1; S1 = 1; J: S1 -> ; 0.1*S1\nend\n")
    assert "00001.sbml" in messages(vs.validate_case(str(d), VOCAB, use_libsed2=False))
    from sed2suite import make_inputs
    make_inputs.process(str(d / "00001.ant"))
    assert errors(vs.validate_case(str(d), VOCAB, use_libsed2=False)) == []


def test_duplicate_numbers_and_case_dirs(tmp_path):
    a = make_case(tmp_path)
    problems = vs.validate_suite([str(a), str(a)], VOCAB, use_libsed2=False)
    assert any("used twice" in p.message for p in problems)
    assert vs.case_dirs([str(a)]) == [str(a)]
    assert vs.case_dirs([str(a / "00001.sed2.json")]) == [str(a)]
    with pytest.raises(FileNotFoundError):
        vs.case_dirs(["nothing-here"])


def test_cli(tmp_path):
    env = dict(os.environ, PYTHONPATH=TOOLS)

    def run(*args):
        return subprocess.run([sys.executable, "-m", "sed2suite.validate_suite", *args], capture_output=True,
                              text=True, env=env)

    d = make_case(tmp_path)
    r = run(str(d), "--no-libsed2")
    assert r.returncode == 0 and "0 error(s)" in r.stdout
    os.remove(d / "00001.rep1.csv")
    assert run(str(d), "--no-libsed2").returncode == 1
    assert run("nothing-here").returncode == 2
    assert run("--no-libsed2").returncode == 0  # the real suite as it stands


# --------------------------------------------------------------------------- the sign-off in the description

def test_signoff_is_required_and_checked(tmp_path):
    d = make_case(tmp_path, signoff=False)
    m = messages(vs.validate_case(str(d), VOCAB, use_libsed2=False))
    assert "Expected results" in m and "Sign-off" in m


def test_signoff_pieces(tmp_path):
    good = author.description_text("00001", "What.", "By hand.", "", "analytical", "").replace(
        author.BACKENDS_LINE, "- Backends: roadrunner, copasi, opencor (agree)")
    settings = {"provenance": {"source": "analytical"}, "backends": ["roadrunner", "copasi", "opencor"]}
    assert vs.signoff_problems(good, settings) == []
    # no derivation
    assert any("Expected results" in p for p in vs.signoff_problems(good.replace("## Expected results\n\nBy hand.\n", ""), settings))
    # not admitted
    notyet = good.replace("- Backends: roadrunner, copasi, opencor (agree)", author.BACKENDS_LINE)
    assert any("not been admitted" in p for p in vs.signoff_problems(notyet, settings))
    # a backend in settings.json that the sign-off does not mention
    assert any("'x'" in p for p in vs.signoff_problems(good, dict(settings, backends=["x"])))
    # provenance and the sign-off must agree
    sim = {"provenance": {"source": "simulation"}, "backends": ["roadrunner"]}
    assert any("simulators" in p for p in vs.signoff_problems(good, sim))
    # missing lines
    assert any("Tolerances" in p for p in vs.signoff_problems(good.replace("- Tolerances:", "- Tol:"), settings))
