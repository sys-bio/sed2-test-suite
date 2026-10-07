#!/usr/bin/env python3
"""Create or update the tag block of a test case's description file.

    python generate_tags.py 00001/00001.sed2.json          (relative to cases/semantic if not found as given)
    python generate_tags.py 00001                          (a case folder name works too)
    python generate_tags.py 00001 --semantic constants-only,analytical
    python generate_tags.py --all --check                  (exit 1 if any description is out of date)

Component tags come from the SED2 document.  Semantic tags are kept from the existing description unless
--semantic replaces them; they must be listed in tags.json (or be prefix:name for a listed prefix).
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tools"))

from sed2suite import tags  # noqa: E402

SEMANTIC_DIR = os.path.join(HERE, "cases", "semantic")


def resolve(arg: str) -> str:
    """Find NNNNN.sed2.json for a file path, a case folder, or a bare case number."""
    candidates = [arg, os.path.join(SEMANTIC_DIR, arg)]
    for c in candidates:
        if os.path.isfile(c):
            return c
        if os.path.isdir(c):
            hits = glob.glob(os.path.join(c, "*.sed2.json"))
            if len(hits) == 1:
                return hits[0]
    raise FileNotFoundError(f"cannot find a SED2 file for {arg!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", help="NNNNN.sed2.json files, case folders or case numbers")
    ap.add_argument("--all", action="store_true", help="process every case in cases/semantic")
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if anything would change")
    ap.add_argument("--semantic", help="comma-separated semantic tags that replace the current ones")
    ap.add_argument("--tags-file", default=tags.DEFAULT_TAGS_PATH, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.all == bool(args.cases):
        ap.error("give case(s) or --all, not both and not neither")
    if args.all and args.semantic is not None:
        ap.error("--semantic applies to explicitly named cases, not --all")
    vocab = tags.load_vocabulary(args.tags_file)
    semantic = None if args.semantic is None else tags.parse_tag_line(args.semantic)
    try:
        paths = sorted(glob.glob(os.path.join(SEMANTIC_DIR, "*", "*.sed2.json"))) if args.all \
            else [resolve(a) for a in args.cases]
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    status = 0
    for p in paths:
        changed, problems = tags.process_case(p, vocab, semantic, check=args.check)
        if problems:
            status = 2
            for m in problems:
                print(f"error: {m}", file=sys.stderr)
        elif changed:
            print(("out of date: " if args.check else "updated: ") + os.path.relpath(p, HERE))
            if args.check and status == 0:
                status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
