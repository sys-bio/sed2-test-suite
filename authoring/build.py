#!/usr/bin/env python3
"""Regenerate the test cases from the authoring scripts.

    python authoring/build.py                    # every series
    python authoring/build.py constants          # one series (authoring/constants.py)
    python authoring/build.py --prune            # after a full build, delete case folders no script produced

Each series module has a `build()` that calls `write_case` with explicit case numbers.  Numbers are assigned in the
order cases are added and are never reused.  A build writes expected results and settings with `backends` empty;
`python -m pysed2translate.suite_runner NNNNN --admit` then records which backends agree.
"""
import argparse
import importlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

from sed2suite import author  # noqa: E402

SERIES = ["constants", "calculations", "timecourses", "steadystate", "repeats", "modelchange", "dataplots", "misc", "fba"]   # in the order the numbers were handed out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("series", nargs="*", help=f"series to build (default: all of {', '.join(SERIES)})")
    ap.add_argument("--prune", action="store_true", help="delete case folders that no script wrote (full builds only)")
    args = ap.parse_args(argv)
    chosen = args.series or SERIES
    unknown = [s for s in chosen if s not in SERIES]
    if unknown:
        print(f"error: unknown series {unknown}", file=sys.stderr)
        return 2
    if args.prune and args.series:
        print("error: --prune needs a full build", file=sys.stderr)
        return 2
    author.reset_written()
    for name in chosen:
        before = len(author.written_numbers())
        importlib.import_module(name).build()
        print(f"{name}: {len(author.written_numbers()) - before} cases")
    if args.prune:
        keep = {author.case_id(n) for n in author.written_numbers()}
        for d in sorted(os.listdir(author.SEMANTIC_DIR)):
            path = os.path.join(author.SEMANTIC_DIR, d)
            if os.path.isdir(path) and d not in keep:
                shutil.rmtree(path)
                print(f"pruned {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
