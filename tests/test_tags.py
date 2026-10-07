import copy
import json
import os
import subprocess
import sys

import pytest

from sed2suite import tags

VOCAB = tags.load_vocabulary()
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DOC = {
    "version": "v1.0.0",
    "constants": {"a": 1, "annotations": {"x": 1}, "styles": 3},
    "tasks": {
        "annotations": {"_type": "calculation", "math": "#constants:a"},
        "m": {"_type": "modelImport", "location": "m.sbml", "language": "urn:sedml:language:sbml"},
        "loop1": {
            "_type": "loop",
            "loopVariables": {"v": {"initialValue": 1, "subsequentValues": "#tasks:loop1.sub.x"}},
            "subTasks": {"sub": {"_type": "calculation", "math": "1"}},
        },
    },
    "outputs": {
        "r": {"_type": "report", "data": "#tasks:annotations"},
        "p": {
            "_type": "plot2D",
            "xAxis": {"title": "t"},
            "curves": {"c1": {"_type": "curve", "x": "#a", "y": "#b"}},
        },
    },
}


def test_component_tags():
    got = tags.component_tags(DOC, VOCAB)
    assert got == sorted(["constants", "calculation", "modelImport", "loop", "loopVariable", "report",
                          "plot2D", "axis", "curve"])


def test_task_id_named_like_an_attribute_is_not_a_tag():
    # the task id 'annotations' and constants keys 'annotations'/'styles' must not produce tags
    got = tags.component_tags(DOC, VOCAB)
    assert "annotation" not in got and "style" not in got


def test_empty_attributes_give_no_tag():
    doc = {"constants": {}, "tasks": {"c": {"_type": "calculation", "math": "1", "annotations": []}}}
    assert tags.component_tags(doc, VOCAB) == ["calculation"]


def test_annotations_attribute_tag():
    doc = {"tasks": {"c": {"_type": "calculation", "math": "1", "annotations": [{"qualifier": "q", "value": "v"}]}}}
    assert tags.component_tags(doc, VOCAB) == ["annotation", "calculation"]


def test_unknown_type():
    with pytest.raises(tags.TagError, match="unknown"):
        tags.component_tags({"tasks": {"t": {"_type": "bogus"}}}, VOCAB)


def test_list_type_rejected():
    with pytest.raises(tags.TagError):
        tags.component_tags({"tasks": {"t": {"_type": ["calculation"]}}}, VOCAB)


def test_not_an_object():
    with pytest.raises(tags.TagError):
        tags.component_tags([], VOCAB)


def test_every_vocabulary_tag_is_reachable_or_untyped():
    comp = VOCAB["component"]
    assert set(comp["attributeTags"].values()) <= set(comp["tags"])


# ---------------------------------------------------------------- semantic tags

@pytest.mark.parametrize("tag, ok", [
    ("analytical", True), ("constants-only", True), ("sbml:events", True), ("csv:columnHeaders", True),
    ("cellml:foo", True), ("bogus", False), ("foo:bar", False), ("sbml:", False), ("sbml:a b", False),
])
def test_check_semantic(tag, ok):
    assert (tags.check_semantic_tags([tag], VOCAB) == []) is ok


def test_duplicate_semantic():
    assert tags.check_semantic_tags(["analytical", "analytical"], VOCAB)


# ---------------------------------------------------------------- description files

def test_new_description():
    text = tags.update_description(None, "00001", ["constants", "report"], ["constants-only"])
    assert text.startswith("# Test 00001\n")
    assert "componentTags: constants, report\nsemanticTags: constants-only\n" in text
    assert text.count(tags.BEGIN) == 1 and text.endswith(tags.END + "\n")


def test_update_preserves_surrounding_text():
    old = "# My test\n\nHand written.\n\n" + tags.render_block(["report"], ["analytical"]) + "\n\nTrailing notes.\n"
    new = tags.update_description(old, "00001", ["report", "constants"], ["analytical"])
    assert new.startswith("# My test\n\nHand written.\n\n")
    assert new.endswith("\n\nTrailing notes.\n")
    assert "componentTags: report, constants" in new


def test_update_appends_when_no_block():
    new = tags.update_description("# T\n\ntext\n", "1", ["report"], [])
    assert new.startswith("# T\n\ntext\n\n" + tags.BEGIN)
    assert tags.update_description(new, "1", ["report"], []) == new  # idempotent


def test_read_semantic():
    d = tags.update_description(None, "1", ["report"], ["analytical", "sbml:events"])
    assert tags.read_semantic_tags(d) == ["analytical", "sbml:events"]
    assert tags.read_semantic_tags("no block") is None
    assert tags.read_semantic_tags(tags.update_description(None, "1", ["report"], [])) == []


def make_case(tmp_path, doc=None, name="00001"):
    d = tmp_path / name
    d.mkdir()
    p = d / f"{name}.sed2.json"
    p.write_text(json.dumps(doc or DOC))
    return str(p)


def test_process_case_create_update_check(tmp_path):
    p = make_case(tmp_path)
    changed, problems = tags.process_case(p, VOCAB, ["analytical"], check=True)
    assert changed and not problems
    assert not os.path.exists(tags.case_paths(p)[1])  # check does not write
    changed, problems = tags.process_case(p, VOCAB, ["analytical"])
    assert changed and not problems
    desc = open(tags.case_paths(p)[1], encoding="utf-8").read()
    assert "semanticTags: analytical" in desc and "\r" not in desc
    # second run: nothing to do, semantic tags kept
    assert tags.process_case(p, VOCAB, None, check=True) == (False, [])
    # a changed document makes it out of date
    doc = copy.deepcopy(DOC)
    del doc["outputs"]["p"]
    open(p, "w").write(json.dumps(doc))
    assert tags.process_case(p, VOCAB, None, check=True)[0] is True


def test_process_case_problems(tmp_path):
    p = make_case(tmp_path)
    changed, problems = tags.process_case(p, VOCAB, ["bogus"])
    assert not changed and problems
    q = make_case(tmp_path, {"tasks": {"t": {"_type": "bogus"}}}, name="00002")
    changed, problems = tags.process_case(q, VOCAB)
    assert not changed and "unknown" in problems[0]
    (tmp_path / "00003").mkdir()
    bad = tmp_path / "00003" / "00003.sed2.json"
    bad.write_text("{not json")
    assert tags.process_case(str(bad), VOCAB)[1]


def test_case_paths_requires_name():
    with pytest.raises(tags.TagError):
        tags.case_paths("foo.json")


# ---------------------------------------------------------------- command line

def run(*args, cwd=ROOT):
    return subprocess.run([sys.executable, os.path.join(ROOT, "generate_tags.py"), *args],
                          capture_output=True, text=True, cwd=cwd)


def test_cli_file_and_check(tmp_path):
    p = make_case(tmp_path)
    r = run(p, "--semantic", "analytical")
    assert r.returncode == 0 and "updated" in r.stdout
    assert run(p, "--check").returncode == 0
    desc = tags.case_paths(p)[1]
    text = open(desc, encoding="utf-8").read().replace("componentTags: ", "componentTags: bogus, ")
    open(desc, "w", encoding="utf-8").write(text)
    r = run(p, "--check")
    assert r.returncode == 1 and "out of date" in r.stdout


def test_cli_errors(tmp_path):
    assert run("--semantic", "x").returncode == 2  # argparse error: neither cases nor --all
    assert run("99999").returncode == 2
    p = make_case(tmp_path)
    assert run(p, "--semantic", "bogus").returncode == 2


def test_cli_all_on_current_suite():
    # The suite may be empty or populated; either way its tag blocks must be current.
    r = run("--all", "--check")
    assert r.returncode == 0, r.stdout + r.stderr
