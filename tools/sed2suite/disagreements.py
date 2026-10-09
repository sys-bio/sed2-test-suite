"""The disagreement log: cases where two backends (or a backend and the recorded results) differ.

    python -m sed2suite.disagreements --check            # validate disagreements.json (exit 1 on problems)
    python -m sed2suite.disagreements --list [--open]    # one line per entry

Format: schemas/disagreements.schema.json, docs/FORMATS.md ("Disagreement log").  Each entry says which test,
which backends, how big the difference is, the diagnosis (solver setting, translator bug, simulator bug, spec
ambiguity, or undiagnosed while still being looked at) and the resolution.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import sys
from typing import Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LOG_PATH = os.path.join(ROOT, "disagreements.json")
SCHEMA_PATH = os.path.join(ROOT, "schemas", "disagreements.schema.json")

DIAGNOSES = ("undiagnosed", "solver setting", "translator bug", "simulator bug", "spec ambiguity")


def empty_log() -> dict:
    return {"schemaVersion": 1, "entries": []}


def load_log(path: str = LOG_PATH) -> dict:
    """The log at `path`; an empty one if the file does not exist."""
    if not os.path.exists(path):
        return empty_log()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_log(log: dict, path: str = LOG_PATH) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(log, f, indent=2)
        f.write("\n")


def next_id(log: dict) -> str:
    numbers = [int(m.group(1)) for e in log.get("entries", []) if (m := re.match(r"^D-(\d+)$", str(e.get("id", ""))))]
    return f"D-{max(numbers, default=0) + 1:03d}"


def _number(x: float):
    """JSON cannot hold infinity or nan; they are written as the strings 'inf' and 'nan'."""
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return x


def draft_entry(log: dict, test: str, backends: list, measure: Optional[dict] = None, report: Optional[str] = None,
                symptom: str = "", date: Optional[str] = None) -> dict:
    """A new, undiagnosed, open entry (not added to the log) for `test`, from a `compare.measure_case` item."""
    entry = {"id": next_id(log), "test": test, "date": date or datetime.date.today().isoformat(),
             "backends": list(backends), "status": "open", "diagnosis": "undiagnosed", "resolution": ""}
    if report:
        entry["report"] = report
    if measure and measure.get("comparable"):
        size = {"maxAbsolute": _number(measure["maxAbsolute"]), "maxRelative": _number(measure["maxRelative"])}
        if measure.get("where"):
            size["where"] = measure["where"]
        entry["size"] = size
    if symptom or (measure and not measure.get("comparable", True)):
        entry["symptom"] = symptom or (measure or {}).get("reason", "")
    return entry


def find(log: dict, test: str, backends: list, report: Optional[str] = None) -> Optional[dict]:
    """The entry for the same test, the same set of backends and the same report, if there is one."""
    for e in log.get("entries", []):
        if e["test"] == test and set(e["backends"]) == set(backends) and e.get("report") == report:
            return e
    return None


def validate_log(log: dict, case_ids: Optional[set] = None) -> list:
    """Problem strings (empty if the log is fine).  With `case_ids`, every entry's test must be one of them."""
    import jsonschema

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    problems = []
    for err in sorted(jsonschema.Draft202012Validator(schema).iter_errors(log), key=lambda e: list(map(str, e.absolute_path))):
        problems.append(f"{'/'.join(str(p) for p in err.absolute_path) or '(top level)'}: {err.message}")
    if problems:
        return problems
    seen = set()
    for i, e in enumerate(log["entries"]):
        where = f"entries/{i} ({e['id']})"
        if e["id"] in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(e["id"])
        if e["status"] != "open" and not e["resolution"].strip():
            problems.append(f"{where}: status is {e['status']!r} but the resolution is empty")
        if e["status"] == "resolved" and e["diagnosis"] == "undiagnosed":
            problems.append(f"{where}: resolved but still undiagnosed")
        if case_ids is not None and e["test"] not in case_ids:
            problems.append(f"{where}: there is no test case {e['test']}")
    return problems


def format_entry(e: dict) -> str:
    size = e.get("size")
    size_text = f"  abs {size['maxAbsolute']}, rel {size['maxRelative']}" if size else ""
    report = f" report {e['report']}" if e.get("report") else ""
    return (f"{e['id']}  {e['test']}{report}  {' vs '.join(e['backends'])}{size_text}  "
            f"[{e['status']}; {e['diagnosis']}]")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check or list the disagreement log.")
    ap.add_argument("--log", default=LOG_PATH, help="the log file (default: disagreements.json at the suite root)")
    ap.add_argument("--check", action="store_true", help="validate the log")
    ap.add_argument("--list", action="store_true", help="list the entries")
    ap.add_argument("--open", action="store_true", help="with --list, only open entries")
    args = ap.parse_args(argv)
    try:
        log = load_log(args.log)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    status = 0
    if args.check or not args.list:
        from . import validate_suite

        case_ids = {os.path.basename(p) for p in validate_suite.case_dirs([])}
        problems = validate_log(log, case_ids)
        for p in problems:
            print(f"error: {p}")
        print(f"{len(log.get('entries', []))} entries, {len(problems)} problems")
        status = 1 if problems else 0
    if args.list:
        for e in log.get("entries", []):
            if not args.open or e["status"] == "open":
                print(format_entry(e))
    return status


if __name__ == "__main__":
    sys.exit(main())
