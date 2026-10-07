"""Create SBML input files from the Antimony sources kept in test cases.

    python -m sed2suite.make_inputs 00200            # case number, folder, or NNNNN.ant file
    python -m sed2suite.make_inputs --all --check    # exit 1 if any .sbml differs from its .ant

For every NNNNN.ant (or NNNNN.something.ant) the SBML is written next to it with the same name and a .sbml
extension.  The main module of the Antimony file is exported, as SBML Level 3 Version 2, with no timestamp,
so the output is reproducible and `--check` can compare it with the committed file.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SEMANTIC_DIR = os.path.join(ROOT, "cases", "semantic")

LEVEL, VERSION = 3, 2


class InputError(Exception):
    pass


def antimony_to_sbml(text: str) -> str:
    """Convert Antimony source text to an SBML L3V2 string."""
    import antimony
    import libsbml

    antimony.clearPreviousLoads()
    if antimony.loadAntimonyString(text) < 0:
        raise InputError("Antimony error: " + antimony.getLastError().strip())
    antimony.setWriteSBMLTimestamp(False)
    try:
        sbml = antimony.getSBMLString(antimony.getMainModuleName())
    finally:
        antimony.clearPreviousLoads()
    doc = libsbml.readSBMLFromString(sbml)
    if doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR) > 0 or doc.getModel() is None:
        raise InputError("antimony produced SBML that libsbml cannot read")
    if (doc.getLevel(), doc.getVersion()) != (LEVEL, VERSION):
        if not doc.setLevelAndVersion(LEVEL, VERSION, False):
            raise InputError(f"cannot convert the model to SBML L{LEVEL}V{VERSION}")
    return libsbml.writeSBMLToString(doc)


def sbml_path(ant_path: str) -> str:
    return os.path.splitext(ant_path)[0] + ".sbml"


def resolve(arg: str) -> list:
    """The .ant files named by a file path, a case folder, or a bare case number."""
    for c in (arg, os.path.join(SEMANTIC_DIR, arg)):
        if os.path.isfile(c):
            return [c]
        if os.path.isdir(c):
            return sorted(glob.glob(os.path.join(c, "*.ant")))
    raise FileNotFoundError(f"cannot find {arg!r}")


def process(ant_path: str, check: bool = False) -> bool:
    """Write (or with check, only compare) the SBML for one .ant file.  Returns True if it changed/differs."""
    with open(ant_path, "r", encoding="utf-8") as f:
        text = f.read()
    sbml = antimony_to_sbml(text)
    out = sbml_path(ant_path)
    old = None
    if os.path.exists(out):
        with open(out, "r", encoding="utf-8", newline="") as f:
            old = f.read()
    changed = sbml != old
    if changed and not check:
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write(sbml)
    return changed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", help=".ant files, case folders or case numbers")
    ap.add_argument("--all", action="store_true", help="every case in cases/semantic")
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if any SBML is missing or differs")
    args = ap.parse_args(argv)
    if args.all == bool(args.cases):
        ap.error("give case(s) or --all, not both and not neither")
    try:
        if args.all:
            files = sorted(glob.glob(os.path.join(SEMANTIC_DIR, "*", "*.ant")))
        else:
            files = [f for a in args.cases for f in resolve(a)]
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    status = 0
    for f in files:
        try:
            changed = process(f, args.check)
        except InputError as e:
            print(f"error: {f}: {e}", file=sys.stderr)
            status = 2
            continue
        if changed:
            print(("out of date: " if args.check else "wrote: ") + os.path.relpath(sbml_path(f), ROOT))
            if args.check and status == 0:
                status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
