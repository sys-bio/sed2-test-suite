"""Loading and validating NNNNN.settings.json (schema: schemas/settings.schema.json, docs: docs/FORMATS.md)."""
from __future__ import annotations

import glob
import json
import os
from typing import Optional

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "schemas", "settings.schema.json")


def find_settings_file(case_dir: str) -> str:
    hits = sorted(glob.glob(os.path.join(case_dir, "*.settings.json")))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly one *.settings.json in {case_dir}, found {len(hits)}")
    return hits[0]


def load_settings(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _schema() -> dict:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_settings(settings: dict, case_dir: Optional[str] = None, report_ids=None, plot_ids=None) -> list:
    """Return a list of problem strings (empty if valid).

    With case_dir, also checks that every listed file exists and that formats and extensions agree.
    With report_ids / plot_ids (ids of the report and plot outputs in the SED2 document), checks that the
    two sets match exactly.
    """
    import jsonschema

    problems = []
    validator = jsonschema.Draft202012Validator(_schema())
    for err in sorted(validator.iter_errors(settings), key=lambda e: list(map(str, e.absolute_path))):
        loc = "/".join(str(p) for p in err.absolute_path) or "(top level)"
        problems.append(f"{loc}: {err.message}")
    if problems:
        return problems
    for section, ids in (("reports", report_ids), ("plots", plot_ids)):
        if ids is None:
            continue
        have = set(settings.get(section, {}))
        want = set(ids)
        for missing in sorted(want - have):
            problems.append(f"{section}: no entry for {missing!r}, which is in the SED2 document")
        for extra in sorted(have - want):
            problems.append(f"{section}: {extra!r} is not in the SED2 document")
    for section in ("reports", "plots"):
        for name, spec in settings.get(section, {}).items():
            ext = os.path.splitext(spec["file"])[1].lower()
            if ext != "." + spec["format"]:
                problems.append(f"{section}/{name}: file {spec['file']!r} does not have a .{spec['format']} extension")
            if case_dir is not None and not os.path.exists(os.path.join(case_dir, spec["file"])):
                problems.append(f"{section}/{name}: file {spec['file']!r} does not exist")
    return problems
