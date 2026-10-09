import json
import math

import pytest

from sed2suite import disagreements as dg


def entry(**kw):
    e = {"id": "D-001", "test": "00200", "backends": ["copasi", "roadrunner"], "status": "open",
         "diagnosis": "undiagnosed", "resolution": ""}
    e.update(kw)
    return e


def log_of(*entries):
    return {"schemaVersion": 1, "entries": list(entries)}


def test_empty_log_is_valid():
    assert dg.validate_log(dg.empty_log()) == []


def test_missing_file_is_an_empty_log(tmp_path):
    assert dg.load_log(str(tmp_path / "none.json")) == dg.empty_log()


def test_save_and_load_round_trip(tmp_path):
    log = log_of(entry())
    path = str(tmp_path / "d.json")
    dg.save_log(log, path)
    assert dg.load_log(path) == log
    assert open(path, encoding="utf-8").read().endswith("}\n")


def test_valid_entry_with_everything():
    e = entry(status="resolved", diagnosis="solver setting", resolution="tightened the relative tolerance to 1e-8",
              report="rep1", date="2026-10-08", size={"maxAbsolute": 1e-5, "maxRelative": "inf", "where": "row 3"},
              symptom="drift", reference="GAPS.md S-008")
    assert dg.validate_log(log_of(e)) == []


@pytest.mark.parametrize("change", [
    {"id": "X-1"}, {"test": "200"}, {"backends": ["roadrunner"]}, {"backends": ["a", "a"]},
    {"status": "done"}, {"diagnosis": "bug"}, {"date": "8/10/2026"}, {"extra": 1},
])
def test_schema_rejects(change):
    assert dg.validate_log(log_of(entry(**change)))


def test_missing_required_key():
    e = entry()
    del e["diagnosis"]
    assert dg.validate_log(log_of(e))


def test_resolved_needs_a_resolution_and_a_diagnosis():
    problems = dg.validate_log(log_of(entry(status="resolved", diagnosis="translator bug")))
    assert any("resolution is empty" in p for p in problems)
    problems = dg.validate_log(log_of(entry(status="resolved", resolution="fixed")))
    assert any("undiagnosed" in p for p in problems)
    assert dg.validate_log(log_of(entry(status="wontfix", diagnosis="simulator bug", resolution="upstream"))) == []


def test_duplicate_ids_and_unknown_cases():
    problems = dg.validate_log(log_of(entry(), entry()))
    assert any("duplicate" in p for p in problems)
    problems = dg.validate_log(log_of(entry()), case_ids={"00100"})
    assert any("no test case 00200" in p for p in problems)
    assert dg.validate_log(log_of(entry()), case_ids={"00200"}) == []


def test_next_id():
    assert dg.next_id(dg.empty_log()) == "D-001"
    assert dg.next_id(log_of(entry(id="D-007"), entry(id="D-003"))) == "D-008"


def test_draft_entry_from_a_measurement():
    m = {"comparable": True, "maxAbsolute": 0.5, "maxRelative": math.inf, "where": "row 1 (t), column 1 (S1)"}
    e = dg.draft_entry(log_of(entry()), "00201", ["roadrunner", "copasi"], m, report="r", date="2026-10-08")
    assert e["id"] == "D-002" and e["status"] == "open" and e["diagnosis"] == "undiagnosed"
    assert e["size"] == {"maxAbsolute": 0.5, "maxRelative": "inf", "where": "row 1 (t), column 1 (S1)"}
    assert dg.validate_log(log_of(e)) == []
    assert json.dumps(e)  # no infinity or nan left in the entry


def test_draft_entry_for_results_that_cannot_be_compared():
    m = {"comparable": False, "reason": "shape (2,), expected (3,)"}
    e = dg.draft_entry(dg.empty_log(), "00200", ["a", "b"], m)
    assert "size" not in e and e["symptom"] == "shape (2,), expected (3,)"
    assert dg.validate_log(log_of(e)) == []


def test_find_matches_test_backends_and_report():
    log = log_of(entry(report="r"))
    assert dg.find(log, "00200", ["roadrunner", "copasi"], "r")["id"] == "D-001"
    assert dg.find(log, "00200", ["roadrunner", "copasi"]) is None
    assert dg.find(log, "00201", ["roadrunner", "copasi"], "r") is None


def test_cli_check_and_list(tmp_path, capsys):
    path = tmp_path / "d.json"
    path.write_text(json.dumps(log_of(entry(test="00999"), entry(test="00999", id="D-002", status="wontfix", diagnosis="simulator bug",
                                                       resolution="upstream"))))
    assert dg.main(["--log", str(path), "--list", "--open"]) == 0
    out = capsys.readouterr().out
    assert "D-001" in out and "D-002" not in out
    assert dg.main(["--log", str(path), "--list"]) == 0
    assert "D-002" in capsys.readouterr().out
    # the suite has no case 00999, so --check complains about it
    assert dg.main(["--log", str(path), "--check"]) == 1
    assert "no test case 00999" in capsys.readouterr().out


def test_cli_bad_file(tmp_path, capsys):
    path = tmp_path / "d.json"
    path.write_text("{not json")
    assert dg.main(["--log", str(path), "--check"]) == 2
