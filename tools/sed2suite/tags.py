"""Component and semantic tags for test cases (see tags.json and README.md).

Component tags are derived from a SED2 document: one tag per `_type` value found, plus tags for untyped
elements found through the attribute that holds them (tags.json 'attributeTags').  Semantic tags are chosen
by the test author and kept in the description file.

The tag block in NNNNN.description.md looks like

    <!-- tags:begin -->
    componentTags: calculation, constants, report
    semanticTags: constants-only
    <!-- tags:end -->
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

BEGIN = "<!-- tags:begin -->"
END = "<!-- tags:end -->"
_BLOCK_RE = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_TAGS_PATH = os.path.join(ROOT, "tags.json")

# Attributes whose value is an element (not a dictionary of elements) and whose contents hold no further tags.
_LEAF_ATTRIBUTES = {"constants", "styles", "annotations"}


class TagError(ValueError):
    pass


def load_vocabulary(path: str = DEFAULT_TAGS_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- component tags

def component_tags(doc: dict, vocab: dict) -> list:
    """Return the sorted list of component tags for a parsed SED2 document.  Raises TagError for unknown types."""
    known = vocab["component"]["tags"]
    attr_tags = vocab["component"]["attributeTags"]
    found: set = set()
    problems: list = []

    def has_content(v) -> bool:
        return v not in (None, {}, [], "")

    def walk_value(v, where: str) -> None:
        """v is the value of an attribute that is not itself named in attributeTags."""
        if isinstance(v, dict):
            if "_type" in v:
                walk_element(v, where)
            else:  # a dictionary of elements, keyed by id: the keys are ids, not attributes
                for k, child in v.items():
                    if isinstance(child, (dict, list)):
                        walk_value(child, f"{where}/{k}")
        elif isinstance(v, list):
            for i, child in enumerate(v):
                if isinstance(child, (dict, list)):
                    walk_value(child, f"{where}[{i}]")

    def walk_element(el: dict, where: str) -> None:
        t = el.get("_type")
        if t is not None:
            if isinstance(t, str) and t in known:
                found.add(t)
            else:
                problems.append(f"{where or '/'}: unknown or invalid _type {t!r}")
        for key, val in el.items():
            if key == "_type":
                continue
            here = f"{where}/{key}"
            if key in attr_tags:
                if has_content(val):
                    found.add(attr_tags[key])
                if key in _LEAF_ATTRIBUTES:
                    continue
                if isinstance(val, dict) and key.endswith("Axis"):
                    walk_element(val, here)  # a single element
                else:
                    walk_value(val, here)
            else:
                walk_value(val, here)

    if not isinstance(doc, dict):
        raise TagError("the document is not a JSON object")
    walk_element(doc, "")
    if problems:
        raise TagError("; ".join(problems))
    for t in found:
        if t not in known:
            raise TagError(f"tag {t!r} is not in tags.json")
    return sorted(found)


# --------------------------------------------------------------------------- semantic tags

def check_semantic_tags(tags, vocab: dict) -> list:
    """Return a list of problems (empty if all tags are valid)."""
    listed = vocab["semantic"]["tags"]
    prefixes = vocab["semantic"].get("prefixes", {})
    problems = []
    seen = set()
    for t in tags:
        if t in seen:
            problems.append(f"semantic tag {t!r} is listed twice")
        seen.add(t)
        if t in listed:
            continue
        if ":" in t:
            prefix, _, rest = t.partition(":")
            if prefix in prefixes and rest and re.fullmatch(r"[A-Za-z0-9_.-]+", rest):
                continue
        problems.append(f"semantic tag {t!r} is not in tags.json (and is not an allowed prefix:name)")
    return problems


def parse_tag_line(text: str) -> list:
    return [t.strip() for t in text.split(",") if t.strip()]


def read_semantic_tags(description: str) -> Optional[list]:
    """Semantic tags in an existing description's tag block, or None if there is no block."""
    m = _BLOCK_RE.search(description)
    if not m:
        return None
    for line in m.group(0).splitlines():
        if line.startswith("semanticTags:"):
            return parse_tag_line(line[len("semanticTags:"):])
    return []


# --------------------------------------------------------------------------- description files

def render_block(component: list, semantic: list) -> str:
    return "\n".join([BEGIN,
                      "componentTags: " + ", ".join(component),
                      "semanticTags: " + ", ".join(semantic),
                      END])


def new_description(case_id: str, block: str) -> str:
    return (f"# Test {case_id}\n\n"
            "Describe what this test checks and how the expected results were obtained.\n\n"
            f"{block}\n")


def update_description(existing: Optional[str], case_id: str, component: list, semantic: list) -> str:
    """Return the description text with its tag block created or replaced; everything else is preserved."""
    block = render_block(component, semantic)
    if existing is None:
        return new_description(case_id, block)
    if _BLOCK_RE.search(existing):
        return _BLOCK_RE.sub(lambda _m: block, existing, count=1)
    sep = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
    return existing + sep + block + "\n"


def case_paths(sed2_path: str) -> tuple:
    """(case id, description path) for NNNNN/NNNNN.sed2.json."""
    name = os.path.basename(sed2_path)
    if not name.endswith(".sed2.json"):
        raise TagError(f"{sed2_path}: expected a file named NNNNN.sed2.json")
    case_id = name[: -len(".sed2.json")]
    return case_id, os.path.join(os.path.dirname(os.path.abspath(sed2_path)), case_id + ".description.md")


def process_case(sed2_path: str, vocab: dict, semantic: Optional[list] = None, check: bool = False) -> tuple:
    """Create or update the description's tag block.

    semantic: replacement semantic tags; None keeps the ones already in the description.
    Returns (changed, problems).  With check=True nothing is written.
    """
    case_id, desc_path = case_paths(sed2_path)
    try:
        with open(sed2_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        comp = component_tags(doc, vocab)
    except (OSError, json.JSONDecodeError, TagError) as e:
        return False, [f"{sed2_path}: {e}"]
    existing = None
    if os.path.exists(desc_path):
        with open(desc_path, "r", encoding="utf-8", newline="") as f:
            existing = f.read()
    sem = semantic
    if sem is None:
        sem = (read_semantic_tags(existing) if existing is not None else None) or []
    problems = [f"{desc_path}: {p}" for p in check_semantic_tags(sem, vocab)]
    if problems:
        return False, problems
    updated = update_description(existing, case_id, comp, sem)
    changed = updated != existing
    if changed and not check:
        with open(desc_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(updated)
    return changed, []
