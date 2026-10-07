"""Check that test cases are complete and consistent.

    python -m sed2suite.validate_suite                 # every case in cases/semantic
    python -m sed2suite.validate_suite 00200 00201     # chosen cases (number, folder or .sed2.json file)
    python -m sed2suite.validate_suite --no-libsed2    # skip validating documents with libsed2

Per case this checks: required files; the SED2 document validates (libsed2, errors only fail); settings.json
is valid and agrees with the document's reports and plots; expected result files are readable and match what
settings.json says about them; no stray result files; the description's tag block is current and its
semantic tags are valid and agree with what can be derived from the case; .sbml files are current with their
.ant sources.  Exit status 0 if there are no errors, 1 if there are, 2 for usage errors.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import compare, make_inputs, results_io as rio, settings as st, tags

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SEMANTIC_DIR = os.path.join(ROOT, "cases", "semantic")

ALL_BACKENDS = ("roadrunner", "copasi", "opencor")
CASE_ID_RE = re.compile(r"^\d{5}$")
REPORT_TYPE = "report"
PLOT_TYPES = ("plot2D", "plot3D")


@dataclass
class Problem:
    level: str  # 'error' or 'warning'
    case: str
    message: str

    def __str__(self):
        return f"{self.level}: {self.case}: {self.message}"


# --------------------------------------------------------------------------- document helpers

def output_ids(doc: dict) -> tuple:
    """(report ids, {plot id: plot type}) from the document's top-level outputs."""
    reports, plots = [], {}
    outputs = doc.get("outputs", {})
    if isinstance(outputs, dict):
        for oid, o in outputs.items():
            t = o.get("_type") if isinstance(o, dict) else None
            if t == REPORT_TYPE:
                reports.append(oid)
            elif t in PLOT_TYPES:
                plots[oid] = t
    return reports, plots


def input_locations(node) -> set:
    """Every string found under a 'location' key anywhere in the document (input file names)."""
    found = set()
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "location" and isinstance(v, str):
                found.add(v)
            else:
                found |= input_locations(v)
    elif isinstance(node, list):
        for v in node:
            found |= input_locations(v)
    return found


def libsed2_problems(path: str) -> list:
    """Messages (level, text) from libsed2 for a document."""
    import libsed2

    try:
        doc = libsed2.read_from_file(path)
    except Exception as e:  # malformed JSON or unreadable file
        return [("error", f"libsed2 cannot read the document: {e}")]
    out = []
    for p in doc.validate():
        level = "warning" if str(p.severity).lower().startswith("warn") else "error"
        out.append((level, f"{p.rule_id} at {p.location}: {p.message}"))
    return out


# --------------------------------------------------------------------------- derived semantic tags

def _has_special(values: np.ndarray) -> bool:
    if values.dtype == object or values.dtype.kind in "US":
        return False
    v = values.astype(np.float64)
    return bool(np.isnan(v).any() or np.isinf(v).any())


def derive_semantic(doc: dict, settings: dict, case_dir: str) -> set:
    """The semantic tags that follow mechanically from the case.  Unreadable results contribute nothing."""
    derived = set()
    if not doc.get("tasks"):
        derived.add("constants-only")
    if settings["provenance"]["source"] == "analytical":
        derived.add("analytical")
    if not set(ALL_BACKENDS) <= set(settings["backends"]):
        derived.add("backend-subset")
    for spec in settings.get("reports", {}).values():
        if spec.get("dtype") == "string":
            derived.add("string-data")
        if spec["format"] == "h5":
            derived.add("multi-dimensional")
        lab = spec.get("labels", {})
        if lab.get("rows") or lab.get("columns"):
            derived.add("labeled-data")
        try:
            data = compare._read_report(os.path.join(case_dir, spec["file"]), spec)
        except Exception:
            continue
        if _has_special(data.values):
            derived.add("special-values")
        if spec["format"] == "h5" and any(l is not None for l in data.labels):
            derived.add("labeled-data")
    for spec in settings.get("plots", {}).values():
        try:
            if spec["format"] == "csv":
                cols = rio.read_plot2d_csv(os.path.join(case_dir, spec["file"])).columns
                if any(isinstance(x, float) and (np.isnan(x) or np.isinf(x)) for c in cols.values() for x in c):
                    derived.add("special-values")
        except Exception:
            continue
    return derived


# Semantic tags that derive_semantic can decide.  A case must carry exactly the derived ones among these.
CHECKED_SEMANTIC = {"constants-only", "analytical", "backend-subset", "string-data", "multi-dimensional",
                    "labeled-data", "special-values"}


# --------------------------------------------------------------------------- one case

def validate_case(case_dir: str, vocab: dict, use_libsed2: bool = True) -> list:
    case_dir = os.path.abspath(case_dir)
    cid = os.path.basename(case_dir)
    problems: list = []

    def err(msg):
        problems.append(Problem("error", cid, msg))

    def warn(msg):
        problems.append(Problem("warning", cid, msg))

    if not CASE_ID_RE.match(cid):
        err("folder name must be five digits")
    sed2 = os.path.join(case_dir, f"{cid}.sed2.json")
    desc = os.path.join(case_dir, f"{cid}.description.md")
    setf = os.path.join(case_dir, f"{cid}.settings.json")
    missing = [os.path.basename(p) for p in (sed2, desc, setf) if not os.path.isfile(p)]
    for m in missing:
        err(f"missing required file {m}")
    if not os.path.isfile(sed2):
        return problems

    # the SED2 document
    try:
        with open(sed2, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        err(f"{os.path.basename(sed2)} is not readable JSON: {e}")
        return problems
    if not isinstance(doc, dict):
        err("the SED2 document is not a JSON object")
        return problems
    if use_libsed2:
        for level, msg in libsed2_problems(sed2):
            (err if level == "error" else warn)(msg)
    reports, plots = output_ids(doc)
    if not reports:
        err("the document has no report; every test needs at least one report")

    # description / tags
    if os.path.isfile(desc):
        changed, tag_problems = tags.process_case(sed2, vocab, None, check=True)
        for m in tag_problems:
            err(m)
        if changed and not tag_problems:
            err(f"{os.path.basename(desc)} has a missing or out-of-date tag block (run generate_tags.py)")

    # settings
    if not os.path.isfile(setf):
        return problems
    try:
        settings = st.load_settings(setf)
    except (OSError, json.JSONDecodeError) as e:
        err(f"{os.path.basename(setf)} is not readable JSON: {e}")
        return problems
    sp = st.validate_settings(settings, case_dir=case_dir, report_ids=reports, plot_ids=list(plots))
    for m in sp:
        err(f"settings: {m}")
    if sp:
        return problems
    for pid, spec in settings.get("plots", {}).items():
        if spec["type"] != plots[pid]:
            err(f"settings: plot {pid!r} is {spec['type']} but the document says {plots[pid]}")

    # result files: readable and consistent with settings
    for rid, spec in settings.get("reports", {}).items():
        try:
            data = compare._read_report(os.path.join(case_dir, spec["file"]), spec)
        except (rio.FormatError, OSError, KeyError) as e:
            err(f"report {rid!r}: {spec['file']} cannot be read: {e}")
            continue
        if spec["format"] == "h5" and data.ndim < 3:
            err(f"report {rid!r}: HDF5 is for 3 or more dimensions but {spec['file']} has {data.ndim}")
        if spec.get("dtype", "number") == "string" and not data.is_string:
            err(f"report {rid!r}: dtype is string but {spec['file']} holds numbers")
        if spec.get("dtype", "number") == "number" and data.is_string:
            err(f"report {rid!r}: dtype is number but {spec['file']} holds strings")
    for pid, spec in settings.get("plots", {}).items():
        try:
            if spec["format"] == "csv":
                rio.read_plot2d_csv(os.path.join(case_dir, spec["file"]))
            else:
                rio.read_plot3d_h5(os.path.join(case_dir, spec["file"]))
        except (rio.FormatError, OSError, KeyError) as e:
            err(f"plot {pid!r}: {spec['file']} cannot be read: {e}")

    # stray files
    listed = {s["file"] for sec in ("reports", "plots") for s in settings.get(sec, {}).values()}
    inputs = {os.path.basename(x) for x in input_locations(doc)}
    for name in sorted(os.listdir(case_dir)):
        full = os.path.join(case_dir, name)
        if os.path.isdir(full) or name in listed or name in inputs:
            continue
        ext = os.path.splitext(name)[1].lower()
        if name in (os.path.basename(p) for p in (sed2, desc, setf)):
            continue
        if ext in (".csv", ".h5"):
            err(f"{name} is not listed in settings.json and is not an input of the document")
        elif ext == ".png":
            m = re.fullmatch(re.escape(cid) + r"\.(.+)\.png", name)
            if not m or m.group(1) not in plots:
                warn(f"{name} does not match any plot id in the document")
        elif ext in (".ant", ".sbml"):
            continue
        else:
            warn(f"{name} is not a recognised test file")

    # SBML from Antimony
    for ant in sorted(glob.glob(os.path.join(case_dir, "*.ant"))):
        try:
            if make_inputs.process(ant, check=True):
                err(f"{os.path.basename(make_inputs.sbml_path(ant))} is missing or differs from "
                    f"{os.path.basename(ant)} (run make_inputs)")
        except make_inputs.InputError as e:
            err(f"{os.path.basename(ant)}: {e}")
        except ImportError as e:
            warn(f"cannot check {os.path.basename(ant)}: {e}")
    for name in sorted(inputs):
        if not os.path.exists(os.path.join(case_dir, name)):
            err(f"input file {name} named in the document does not exist in the case folder")

    # semantic tags that can be derived
    if os.path.isfile(desc):
        with open(desc, "r", encoding="utf-8") as f:
            have = set(tags.read_semantic_tags(f.read()) or [])
        want = derive_semantic(doc, settings, case_dir)
        for t in sorted((want - have) & CHECKED_SEMANTIC):
            err(f"semantic tag {t!r} applies to this case but is missing from the description")
        for t in sorted((have - want) & CHECKED_SEMANTIC):
            err(f"semantic tag {t!r} is in the description but does not apply to this case")
    return problems


# --------------------------------------------------------------------------- whole suite

def case_dirs(args: list) -> list:
    if not args:
        return sorted(d for d in glob.glob(os.path.join(SEMANTIC_DIR, "*")) if os.path.isdir(d))
    out = []
    for a in args:
        for c in (a, os.path.join(SEMANTIC_DIR, a)):
            if os.path.isdir(c):
                out.append(c)
                break
            if os.path.isfile(c) and c.endswith(".sed2.json"):
                out.append(os.path.dirname(os.path.abspath(c)))
                break
        else:
            raise FileNotFoundError(f"cannot find case {a!r}")
    return out


def validate_suite(dirs: list, vocab: dict, use_libsed2: bool = True) -> list:
    problems = []
    seen = {}
    for d in dirs:
        cid = os.path.basename(os.path.abspath(d))
        if cid in seen:
            problems.append(Problem("error", cid, "case number used twice"))
        seen[cid] = d
        problems += validate_case(d, vocab, use_libsed2)
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", help="case numbers, folders or .sed2.json files (default: all)")
    ap.add_argument("--no-libsed2", action="store_true", help="do not validate documents with libsed2")
    ap.add_argument("--tags-file", default=tags.DEFAULT_TAGS_PATH, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    use_libsed2 = not args.no_libsed2
    if use_libsed2:
        try:
            import libsed2  # noqa: F401
        except ImportError:
            print("error: the libsed2 wheel is not installed (see README); use --no-libsed2 to skip document validation",
                  file=sys.stderr)
            return 2
    try:
        dirs = case_dirs(args.cases)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    problems = validate_suite(dirs, tags.load_vocabulary(args.tags_file), use_libsed2)
    for p in problems:
        print(p)
    errors = sum(p.level == "error" for p in problems)
    print(f"{len(dirs)} case(s) checked: {errors} error(s), {len(problems) - errors} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
