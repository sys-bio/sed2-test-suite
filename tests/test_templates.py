"""Every ported template in templates/ is a valid SED2 document (see templates/PORTING.md)."""
import glob
import os

import pytest

libsed2 = pytest.importorskip("libsed2")

TEMPLATES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "templates")
FILES = sorted(glob.glob(os.path.join(TEMPLATES, "*.sed2.json")))


def test_there_are_templates():
    assert len(FILES) == 39


@pytest.mark.parametrize("path", FILES, ids=os.path.basename)
def test_template_validates(path):
    doc = libsed2.read_from_file(path)
    errors = [f"{p.rule_id} {p.location}: {p.message}" for p in doc.validate() if not str(p.severity).lower().startswith("warn")]
    assert errors == []
