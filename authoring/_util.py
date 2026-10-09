"""Shared helpers for the authoring scripts (see tools/sed2suite/author.py)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

import numpy as np  # noqa: E402

from sed2suite.author import AnnotatedData, write_case  # noqa: E402,F401

IMPORT_SBML = "urn:sedml:language:sbml"


def A(values, rows=None, cols=None, dims=None):
    """AnnotatedData from nested lists; `rows`/`cols` are labels of dimension 0/1, `dims` dimension names."""
    arr = np.array(values, dtype=object if _has_str(values) else float)
    labels = [None] * arr.ndim
    if rows is not None:
        labels[0] = list(rows)
    if cols is not None:
        labels[1] = list(cols)
    return AnnotatedData(arr, labels, list(dims) if dims else None)


def _has_str(v):
    if isinstance(v, str):
        return True
    if isinstance(v, (list, tuple)):
        return any(_has_str(x) for x in v)
    return False


def outputs(refs):
    """{report id: reference} -> the document's outputs."""
    return {rid: {"_type": "report", "data": ref} for rid, ref in refs.items()}


def model_import(cid, name="model", language=IMPORT_SBML):
    return {"_type": "modelImport", "location": f"{cid}.sbml", "language": language}


def constants_case(number, what, constants, refs, expected, derivation, **kw):
    """A case with constants and reports only."""
    doc = {"constants": constants, "outputs": outputs(refs)}
    return write_case(number, what, doc, expected, derivation=derivation, **kw)
