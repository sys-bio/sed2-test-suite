"""Series "misc": ModelElementList, and the SEDBase fields (name, description, notes, annotations) that never change a result.

ModelElementList: the specification gives one example of the SBML vocabulary ("species" in includeTypes), says that an
element in both an include and an exclude list is excluded, and does not say in what order the ids come back.  The
cases therefore select exactly one element (or use "species" on a model with exactly one species), so that the order
cannot matter; SED2/TODO.md has the open points.
"""
import numpy as np

from _util import A, write_case, outputs, IMPORT_SBML

FIRST = 264
TIME = "urn:sedml:symbol:time"

ONE = "model one\n  compartment c = 1\n  species S1 in c\n  S1 = 1\n  J0: S1 -> ; k*S1\n  k = 1\nend"
CHAIN = ("model chain\n  compartment c = 1\n  species S1 in c, S2 in c\n  S1 = 1\n  S2 = 0.5\n  $Src -> S1; k0\n"
         "  J1: S1 -> S2; k1*S1\n  J2: S2 -> $Snk; k2*S2\n  k0 = 2\n  k1 = 0.5\n  k2 = 1\nend")
DECAY = "model dec\n  compartment c = 1\n  species S in c\n  S = 3\n  S -> ; k1*S\n  k1 = 0.5\nend"


def build():
    counter = iter(range(FIRST, FIRST + 1000))

    def case(what, ant, tasks, refs, expected, derivation, constants=None, plots=None, **kw):
        n = next(counter)
        cid = f"{n:05d}"
        all_tasks = {"m": {"_type": "modelImport", "location": f"{cid}.sbml", "language": IMPORT_SBML}}
        all_tasks.update(tasks)
        doc = {}
        if constants:
            doc["constants"] = constants
        doc["tasks"] = all_tasks
        doc["outputs"] = outputs(refs)
        return write_case(n, what, doc, expected, derivation=derivation, antimony={"": ant}, **kw)

    def mel(**kw):
        return {"_type": "modelElementList", "model": "#tasks:m.model", **kw}

    R = {"r": "#tasks:l.strings"}

    case("ModelElementList: includeElements names one element.", ONE, {"l": mel(includeElements=["J0"])}, R,
         {"r": A(["J0"])}, "Only J0 is included, so the list is (J0).")
    case("ModelElementList: includeTypes with 'species' selects the species (the model has one).", ONE,
         {"l": mel(includeTypes=["species"])}, R, {"r": A(["S1"])}, "The only species is S1.")
    case("ModelElementList: an element in both an include and an exclude list is excluded; the specification's example "
         "in a model with one species.", ONE,
         {"l": mel(includeElements=["J0"], includeTypes=["species"], excludeElements=["S1"])}, R, {"r": A(["J0"])},
         "'Every species plus the reaction J0, except S1': S1 is the only species and is excluded, leaving J0.")
    case("ModelElementList: an element named in includeElements and in excludeElements is excluded.", ONE,
         {"l": mel(includeElements=["J0", "S1"], excludeElements=["J0"])}, R, {"r": A(["S1"])},
         "J0 and S1 are included and J0 is also excluded: the list is (S1).")
    case("ModelElementList: excludeTypes removes the species.", ONE,
         {"l": mel(includeElements=["S1", "J0"], excludeTypes=["species"])}, R, {"r": A(["J0"])},
         "S1 and J0 are included; every species is excluded, so S1 goes: (J0).")
    case("ModelElementList: the lists are given by references to list constants.", ONE,
         {"l": mel(includeElements="#constants:inc", excludeElements="#constants:exc")}, R, {"r": A(["S1"])},
         "J0 and S1 are included and J0 excluded: (S1).", constants={"inc": ["J0", "S1"], "exc": ["J0"]})
    case("ModelElementList used by ModelChange.removeElements: the selected reaction J1 is removed from a chain.",
         CHAIN, {"l": mel(includeElements=["J1"]),
                 "c": {"_type": "modelChange", "inputModel": "#tasks:m.model", "removeElements": "#tasks:l.strings"},
                 "s": {"_type": "explicitODESimulation", "model": "#tasks:c.model", "independentVariable": TIME,
                       "outputVariables": ["S1", "S2"],
                       "independentVariableRange": {"_type": "numericRange", "start": 0, "end": 2, "numberOfSteps": 2}}},
         {"r": "#tasks:s"},
         {"r": A(np.column_stack([[0.0, 1.0, 2.0], 1 + 2 * np.array([0.0, 1.0, 2.0]),
                                  0.5 * np.exp(-np.array([0.0, 1.0, 2.0]))]), cols=[TIME, "S1", "S2"])},
         "Without J1: S1' = 2 (S1 = 1 + 2 t) and S2' = -S2 (S2 = 0.5 exp(-t)).")
    case("ModelElementList used by StringFormation.", ONE, {"l": mel(includeElements=["J0"]),
                                                            "sf": {"_type": "stringFormation",
                                                                   "concatenate": ["id=", "#tasks:l.strings"]}},
         {"r": "#tasks:sf"}, {"r": A(["id=J0"])}, "'id=' joined to each string of the list (J0): id=J0.")

    # ------------------------------------------------------------------ name, description, notes, annotations
    meta = {"name": "A name", "description": "A description.", "notes": "Some *markdown* notes.",
            "annotations": [{"qualifier": "dc:title", "value": "A title"},
                            {"qualifier": "dc:license", "value": "http://creativecommons.org/publicdomain/zero/1.0/"}]}
    n = next(counter)
    write_case(n, "name, description, notes and annotations on a calculation and a report do not change the result.",
               {"constants": {"x": 4},
                "tasks": {"c": {"_type": "calculation", "math": "#constants:x * 2 + 1", **meta}},
                "outputs": {"r": {"_type": "report", "data": "#tasks:c", **meta}}},
               {"r": A(9.0)}, derivation="4 * 2 + 1 = 9.  The extra fields are ignored by an interpreter.")
    n = next(counter)
    cid = f"{n:05d}"
    meta2 = dict(meta, annotations=[{"qualifier": "dc:title", "value": "#constants:title"}])
    write_case(n, "Metadata (with an annotation value that is a reference) on a model import, a simulation "
                  "and a report.",
               {"constants": {"title": "A referenced title"},
                "tasks": {"m": {"_type": "modelImport", "location": f"{cid}.sbml", "language": IMPORT_SBML, **meta2},
                          "s": {"_type": "oneStepODESimulation", "model": "#tasks:m.model", "independentVariable": TIME,
                                "outputVariables": ["S"], "independentStep": 2, **meta2}},
                "outputs": {"r": {"_type": "report", "data": "#tasks:s['S']", **meta2}}},
               {"r": A(3 * np.exp(-1.0))}, derivation="S(2) = 3 exp(-0.5 * 2) = 3 exp(-1).  Every element carries a name, a "
               "description, notes and an annotation whose value refers to a constant.", antimony={"": DECAY})
