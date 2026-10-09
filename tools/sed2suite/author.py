"""Write a complete test case from Python: the document, inputs, expected results, settings and description.

    from sed2suite.author import write_case, AnnotatedData
    write_case(7, "Exponential decay ...", document, {"r": AnnotatedData(values, [None, ["time", "S1"]])},
               antimony={"": "model m ... end"}, derivation="S1(t) = 3 exp(-0.5 t)", semantic=["labeled-data"])

The authoring scripts in `authoring/` use this, so a whole series of cases can be regenerated after a change
(`python authoring/build.py`).  What `write_case` makes in `cases/semantic/NNNNN/`:

    NNNNN.sed2.json            the document (a "version" is added if it has none)
    NNNNN.ant, NNNNN.sbml      Antimony sources (key '' or a name -> NNNNN.ant, NNNNN.name.ant) and the SBML made from them
    <inputs>                   other input files, by file name (the name should start with NNNNN.)
    NNNNN.<report>.csv|h5      expected reports (CSV for 0-2 dimensions, HDF5 beyond)
    NNNNN.<plot>_as_data.*     expected plot data
    NNNNN.settings.json        tolerances, provenance, reports and plots; `backends` is [] until the runner admits
                               the case (`pysed2translate.suite_runner --admit`)
    NNNNN.description.md       what is tested, how the expected results were derived, a sign-off block, the tag block

The case folder is emptied first, so a case is exactly what its script says.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from typing import Optional

import numpy as np

from . import make_inputs, results_io as rio, tags, validate_suite
from .results_io import AnnotatedData, Plot2DData, Plot3DData  # noqa: F401  (for the authoring scripts)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SEMANTIC_DIR = os.path.join(ROOT, "cases", "semantic")
DEFAULT_TOLERANCES = {"absolute": 1e-7, "relative": 1e-4}
BACKENDS_LINE = "- Backends: not yet admitted (run `pysed2translate.suite_runner --admit`)."


_WRITTEN: dict = {}  # (base folder, number) -> case folder, in the order written


def reset_written() -> None:
    _WRITTEN.clear()


def written_numbers() -> list:
    """Case numbers written since the last reset, in the order written."""
    return [number for (_base, number) in _WRITTEN]


def case_id(number: int) -> str:
    if not (1 <= number <= 99999):
        raise ValueError(f"case number {number} is out of range")
    return f"{number:05d}"


def report_entry(data: AnnotatedData, file_name: str) -> dict:
    """The settings.json entry that describes an expected report."""
    entry = {"file": file_name, "format": "csv" if data.ndim <= 2 else "h5",
             "dtype": "string" if data.is_string else "number"}
    if entry["format"] == "csv":   # an HDF5 file carries its own shape; settings.json gives `ndim` for CSV only
        entry["ndim"] = data.ndim
    labels = {}
    if data.ndim >= 1 and data.labels[0] is not None:
        labels["rows"] = True
    if data.ndim == 2 and data.labels[1] is not None:
        labels["columns"] = True
    if labels and entry["format"] == "csv":
        entry["labels"] = {"rows": labels.get("rows", False), "columns": labels.get("columns", False)}
    return entry


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def description_text(cid: str, what: str, derivation: str, tolerance_note: str, source: str, notes: str) -> str:
    lines = [f"# Test {cid}", "", what.strip(), ""]
    if derivation.strip():
        lines += ["## Expected results", "", derivation.strip(), ""]
    if notes.strip():
        lines += ["## Notes", "", notes.strip(), ""]
    lines += ["## Sign-off", "",
              "- Expected results: " + ("derived by hand (see above), not taken from a simulator." if source == "analytical"
                                       else "from simulators that agree (see settings.json provenance)."),
              f"- Tolerances: {tolerance_note.strip() or 'the defaults (absolute 1e-7, relative 1e-4).'}",
              BACKENDS_LINE, ""]
    return "\n".join(lines)


def _write_case(number: int, what: str, document: dict, expected: Optional[dict] = None, *,
               derivation: str = "", antimony: Optional[dict] = None, inputs: Optional[dict] = None,
               plots: Optional[dict] = None, tolerances: Optional[dict] = None,
               report_tolerances: Optional[dict] = None, compare: Optional[dict] = None,
               semantic=(), notes: str = "", tolerance_note: str = "", source: str = "analytical",
               simulators: Optional[list] = None, provenance_notes: str = "", backends: Optional[list] = None,
               root: Optional[str] = None) -> str:
    """Write case `number` and return its folder.  `expected` maps report ids to AnnotatedData, `plots` maps plot ids
    to Plot2DData or Plot3DData, `report_tolerances` maps report ids to tolerance overrides and `compare` maps report
    ids to {"mode": ..., "note": ...}."""
    cid = case_id(number)
    base = root or SEMANTIC_DIR
    if (base, number) in _WRITTEN:
        raise ValueError(f"case {cid} was already written in this run")
    final = os.path.join(base, cid)
    folder = final + ".new"          # written here first, so that an unchanged case can be left exactly as it was
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

    doc = dict(document)
    if "version" not in doc:
        doc = {"version": "v1.0.0", **doc}
    sed2 = os.path.join(folder, f"{cid}.sed2.json")
    _write_text(sed2, json.dumps(doc, indent=2) + "\n")

    for key, text in (antimony or {}).items():
        ant = os.path.join(folder, f"{cid}.ant" if not key else f"{cid}.{key}.ant")
        _write_text(ant, text if text.endswith("\n") else text + "\n")
        make_inputs.process(ant)
    for name, content in (inputs or {}).items():
        path = os.path.join(folder, name)
        if isinstance(content, bytes):
            with open(path, "wb") as f:
                f.write(content)
        else:
            _write_text(path, content)

    settings = {"schemaVersion": 1, "tolerances": dict(tolerances or DEFAULT_TOLERANCES),
                "provenance": {"source": source}, "backends": list(backends or [])}
    if source == "simulation":
        settings["provenance"]["simulators"] = list(simulators or [])
    if provenance_notes:
        settings["provenance"]["notes"] = provenance_notes
    reports = {}
    for rid, data in (expected or {}).items():
        ext = "csv" if data.ndim <= 2 else "h5"
        file_name = f"{cid}.{rid}.{ext}"
        (rio.write_csv if ext == "csv" else rio.write_h5)(os.path.join(folder, file_name), data)
        entry = report_entry(data, file_name)
        if report_tolerances and rid in report_tolerances:
            entry["tolerances"] = report_tolerances[rid]
        if compare and rid in compare:
            entry["compare"] = compare[rid]
        reports[rid] = entry
    if reports:
        settings["reports"] = reports
    plot_entries = {}
    for pid, plot in (plots or {}).items():
        if isinstance(plot, Plot2DData):
            file_name = f"{cid}.{pid}_as_data.csv"
            rio.write_plot2d_csv(os.path.join(folder, file_name), plot)
            plot_entries[pid] = {"file": file_name, "format": "csv", "type": "plot2D"}
        else:
            file_name = f"{cid}.{pid}_as_data.h5"
            rio.write_plot3d_h5(os.path.join(folder, file_name), plot)
            plot_entries[pid] = {"file": file_name, "format": "h5", "type": "plot3D"}
    if plot_entries:
        settings["plots"] = plot_entries
    _write_text(os.path.join(folder, f"{cid}.settings.json"), json.dumps(settings, indent=2) + "\n")

    _write_text(os.path.join(folder, f"{cid}.description.md"),
                description_text(cid, what, derivation, tolerance_note, source, notes))
    sem = sorted(set(semantic) | validate_suite.derive_semantic(doc, settings, folder))
    vocab = tags.load_vocabulary()
    changed, problems = tags.process_case(sed2, vocab, sem)
    if problems:
        raise ValueError("; ".join(problems))
    if os.path.isdir(final) and _same_case(final, folder):
        shutil.rmtree(folder)        # nothing changed: keep the admission record in the existing folder
    else:
        if os.path.isdir(final):
            shutil.rmtree(final)
        os.rename(folder, final)
    _WRITTEN[(base, number)] = final
    return final


def write_case(number: int, *args, **kwargs) -> str:
    """Write case `number` (see `_write_case` for the arguments) and return its folder.  A failure leaves no
    half-written folder behind."""
    base = kwargs.get("root") or SEMANTIC_DIR
    try:
        return _write_case(number, *args, **kwargs)
    except BaseException:
        shutil.rmtree(os.path.join(base, case_id(number)) + ".new", ignore_errors=True)
        raise


_BACKENDS_RE = re.compile(r"^- Backends:.*$", re.M)


def _normalised(path: str, name: str):
    """File content with the admission record removed (the `backends` list and the Backends sign-off line)."""
    with open(path, "rb") as f:
        data = f.read()
    if name.endswith(".settings.json"):
        d = json.loads(data.decode("utf-8"))
        d.pop("backends", None)
        return d
    if name.endswith(".description.md"):
        return _BACKENDS_RE.sub("- Backends:", data.decode("utf-8"))
    return data


def _same_case(old: str, new: str) -> bool:
    """True when the two case folders hold the same files, ignoring which backends have been admitted."""
    if sorted(os.listdir(old)) != sorted(os.listdir(new)):
        return False
    return all(_normalised(os.path.join(old, n), n) == _normalised(os.path.join(new, n), n) for n in os.listdir(new))
