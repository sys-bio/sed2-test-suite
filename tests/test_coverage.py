import json
import os

from sed2suite import coverage

VOCAB = {"schemaVersion": 1, "description": "", "component": {
    "tags": {"calculation": "c", "report": "r", "constants": "k", "scatter": "s", "style": "y"},
    "attributeTags": {"constants": "constants"}, "note": ""}, "semantic": {"tags": {}, "prefixes": {}, "note": ""}}


def make_suite(tmp_path, cases):
    (tmp_path / "tags.json").write_text(json.dumps(VOCAB))
    for cid, (doc, backends) in cases.items():
        d = tmp_path / "cases" / "semantic" / cid
        d.mkdir(parents=True)
        (d / f"{cid}.sed2.json").write_text(json.dumps(doc))
        (d / f"{cid}.settings.json").write_text(json.dumps({"backends": backends}))


DOC = {"version": "v1.0.0", "tasks": {"a": {"_type": "calculation", "math": "1"}},
       "outputs": {"r": {"_type": "report", "data": "#tasks:a"}}}


def test_counts_cases_and_backends(tmp_path):
    make_suite(tmp_path, {"00001": (DOC, ["roadrunner", "copasi"]), "00002": (DOC, ["roadrunner"])})
    data = coverage.collect(str(tmp_path))
    assert data["calculation"]["cases"] == ["00001", "00002"]
    assert data["calculation"]["backends"]["roadrunner"] == ["00001", "00002"]
    assert data["calculation"]["backends"]["copasi"] == ["00001"]
    assert data["calculation"]["backends"]["opencor"] == []
    assert data["scatter"]["cases"] == []


def test_uncovered_ignores_deferred(tmp_path, monkeypatch):
    make_suite(tmp_path, {"00001": (DOC, ["roadrunner"])})
    data = coverage.collect(str(tmp_path))
    monkeypatch.setattr(coverage, "DEFERRED", {"style": "placeholder"})
    assert coverage.uncovered(data) == ["constants", "scatter"]
    monkeypatch.setattr(coverage, "DEFERRED", {"style": "placeholder", "calculation": "no longer true"})
    assert coverage.stale_deferrals(data) == ["calculation"]
    text = coverage.render(data)
    assert "| scatter | 0 | 0 | 0 | 0 | 0 | **NO CASE** |" in text
    assert "| style | 0 | 0 | 0 | 0 | 0 | deferred: placeholder |" in text
    assert "| calculation | 1 | 1 | 0 | 0 | 0 | covered |" in text


def test_every_tag_of_the_real_suite_is_covered_or_deferred():
    data = coverage.collect()
    assert coverage.uncovered(data) == []
    assert coverage.stale_deferrals(data) == []
    assert set(coverage.DEFERRED) <= set(data)
