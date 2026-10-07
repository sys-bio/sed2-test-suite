import os
import subprocess
import sys

import pytest

from sed2suite import make_inputs as mi

TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
ANT = "model m\n  species S1; S1 = 1; k = 0.1\n  J: S1 -> ; k*S1\nend\n"


def test_converts_to_l3v2():
    import libsbml
    s = mi.antimony_to_sbml(ANT)
    d = libsbml.readSBMLFromString(s)
    assert (d.getLevel(), d.getVersion()) == (3, 2)
    assert d.getModel().getNumSpecies() == 1
    assert "\r" not in s


def test_deterministic():
    assert mi.antimony_to_sbml(ANT) == mi.antimony_to_sbml(ANT)


def test_state_not_carried_between_calls():
    other = "model other\n  species A; A = 2; J: A -> ; 1*A\nend\n"
    a = mi.antimony_to_sbml(ANT)
    mi.antimony_to_sbml(other)
    assert mi.antimony_to_sbml(ANT) == a


def test_bad_antimony():
    with pytest.raises(mi.InputError):
        mi.antimony_to_sbml("model m\n  this is not antimony\nend\n")


def test_process_write_and_check(tmp_path):
    p = tmp_path / "00200.ant"
    p.write_text(ANT)
    assert mi.process(str(p), check=True) is True  # missing sbml
    assert not (tmp_path / "00200.sbml").exists()
    assert mi.process(str(p)) is True
    assert (tmp_path / "00200.sbml").exists()
    assert mi.process(str(p), check=True) is False
    (tmp_path / "00200.sbml").write_text("stale")
    assert mi.process(str(p), check=True) is True


def test_resolve(tmp_path):
    (tmp_path / "a.ant").write_text(ANT)
    assert mi.resolve(str(tmp_path)) == [str(tmp_path / "a.ant")]
    assert mi.resolve(str(tmp_path / "a.ant")) == [str(tmp_path / "a.ant")]
    with pytest.raises(FileNotFoundError):
        mi.resolve("no-such-case")


def run(*args):
    env = dict(os.environ, PYTHONPATH=TOOLS)
    return subprocess.run([sys.executable, "-m", "sed2suite.make_inputs", *args], capture_output=True, text=True, env=env)


def test_cli(tmp_path):
    p = tmp_path / "00200.ant"
    p.write_text(ANT)
    r = run(str(p), "--check")
    assert r.returncode == 1 and "out of date" in r.stdout
    r = run(str(p))
    assert r.returncode == 0 and "wrote" in r.stdout
    assert run(str(p), "--check").returncode == 0
    bad = tmp_path / "00201.ant"
    bad.write_text("model m\n nonsense here\nend\n")
    assert run(str(bad)).returncode == 2
    assert run("--all", "--check").returncode == 0
