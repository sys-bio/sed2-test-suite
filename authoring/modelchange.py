"""Series "modelchange": ModelChange (setValues and removeElements), ModelImport and the labels of `.model`.

addElements and replaceElements are not tested (SED2/TODO.md), nor is the order in which several attributes of one
ModelChange apply, so no case combines setValues and removeElements in one task.  The SBML-specific keys and removal
rules are still "to be determined" in model_formats/SBML, so the cases keep to the clearest meanings: a parameter or
compartment id sets its value, and every species lives in a compartment of volume 1 where a concentration and an amount
are the same number; removing a reaction removes its contribution to the dynamics (SED2/TODO.md has the open points).
"""
import numpy as np

from _util import A, write_case, outputs, IMPORT_SBML

FIRST = 222
TIME = "urn:sedml:symbol:time"

DECAY = "model dec\n  compartment c = 1\n  species S in c\n  S = 3\n  J: S -> ; k1*S\n  k1 = 0.5\nend"
PARS = "model pars\n  species X\n  X = 1\n  k = 2\n  j = 3\nend"
CELL = "model cellm\n  compartment cell = 2\n  species X in cell\n  X = 1\n  k = 2\nend"
CHAIN = ("model chain\n  compartment c = 1\n  species S1 in c, S2 in c\n  S1 = 1\n  S2 = 0.5\n  $Src -> S1; k0\n"
         "  J1: S1 -> S2; k1*S1\n  J2: S2 -> $Snk; k2*S2\n  k0 = 2\n  k1 = 0.5\n  k2 = 1\nend")
SOURCE = "model src\n  -> A; v\n  A -> ; k1*A\n  v = 2\n  k1 = 1\n  A = 1\nend"
LAB = ("model lab\n  compartment cell = 2\n  species Sa in cell, Sb in cell\n  substanceOnly species Sc in cell\n"
       "  Sa = 3\n  Sb = 0\n  Sc = 4\n  p = 5\n  q := 2*p\n  J: Sa -> Sb; k*Sa*cell\n  k = 0.5\nend")


def table(times, **cols):
    data = np.column_stack([np.asarray(times, float)] + [np.asarray(v, float) for v in cols.values()])
    return A(data, cols=[TIME] + list(cols))


def change(model, values=None, remove=None):
    c = {"_type": "modelChange", "inputModel": model}
    if values is not None:
        c["setValues"] = values
    if remove is not None:
        c["removeElements"] = remove
    return c


def one_step(model, step, outvars=("S",)):
    return {"_type": "oneStepODESimulation", "model": model, "independentVariable": TIME,
            "outputVariables": list(outvars), "independentStep": step}


def course(model, outvars, end=2, steps=2):
    return {"_type": "explicitODESimulation", "model": model, "independentVariable": TIME,
            "outputVariables": list(outvars),
            "independentVariableRange": {"_type": "numericRange", "start": 0, "end": end, "numberOfSteps": steps}}


def build():
    counter = iter(range(FIRST, FIRST + 1000))

    def case(what, ant, tasks, refs, expected, derivation, imports=("m",), constants=None, **kw):
        """The model task(s) named in `imports` all import the case's model; "MODEL" stands for `#tasks:m.model`."""
        n = next(counter)
        cid = f"{n:05d}"
        all_tasks = {name: {"_type": "modelImport", "location": f"{cid}.sbml", "language": IMPORT_SBML}
                     for name in imports}
        all_tasks.update(tasks)
        doc = {}
        if constants:
            doc["constants"] = constants
        doc["tasks"] = all_tasks
        doc["outputs"] = outputs(refs)
        return write_case(n, what, doc, expected, derivation=derivation, antimony={"": ant}, **kw)

    M = "#tasks:m.model"

    # ------------------------------------------------------------------ setValues
    case("setValues on a parameter: the changed model has the new value and the model it came from keeps the old one.",
         PARS, {"c": change(M, {"k": 7})},
         {"new": "#tasks:c.model['k']", "old": "#tasks:m.model['k']", "other": "#tasks:c.model['j']"},
         {"new": A(7.0), "old": A(2.0), "other": A(3.0)},
         "k is set from 2 to 7.  The imported model still has k = 2, and j, which was not mentioned, is still 3.")
    case("setValues with several keys in one ModelChange.",
         PARS, {"c": change(M, {"k": 7, "j": -1.5})},
         {"k": "#tasks:c.model['k']", "j": "#tasks:c.model['j']"}, {"k": A(7.0), "j": A(-1.5)},
         "k = 7 and j = -1.5.")
    case("setValues on a compartment sets its size.",
         CELL, {"c": change(M, {"cell": 5})},
         {"new": "#tasks:c.model['cell']", "old": "#tasks:m.model['cell']"}, {"new": A(5.0), "old": A(2.0)},
         "The compartment 'cell' of size 2 becomes 5; the imported model is unchanged.")
    case("setValues on a species in a compartment of volume 1.",
         DECAY, {"c": change(M, {"S": 9})},
         {"new": "#tasks:c.model['S']", "old": "#tasks:m.model['S']"}, {"new": A(9.0), "old": A(3.0)},
         "The compartment has volume 1, so amount and concentration are the same number: S becomes 9.")
    case("Chained ModelChanges: each keeps the changes before it, and none alters the model it started from.",
         PARS, {"c1": change(M, {"k": 7}), "c2": change("#tasks:c1.model", {"j": 8})},
         {"k2": "#tasks:c2.model['k']", "j2": "#tasks:c2.model['j']", "k1": "#tasks:c1.model['k']",
          "j1": "#tasks:c1.model['j']", "k0": "#tasks:m.model['k']", "j0": "#tasks:m.model['j']"},
         {"k2": A(7.0), "j2": A(8.0), "k1": A(7.0), "j1": A(3.0), "k0": A(2.0), "j0": A(3.0)},
         "c1 sets k = 7; c2 starts from c1 and sets j = 8, so c2 has (7, 8), c1 has (7, 3) and the import has (2, 3).")
    case("setValues values given as references: a constant, an element of a dictionary constant and a calculation.",
         PARS, {"calc": {"_type": "calculation", "math": "#constants:base * 4"},
                "c": change(M, {"k": "#constants:newk", "j": "#tasks:calc"})},
         {"k": "#tasks:c.model['k']", "j": "#tasks:c.model['j']"}, {"k": A(0.25), "j": A(12.0)},
         "k takes the constant newk = 0.25 and j the calculation base * 4 = 12.",
         constants={"newk": 0.25, "base": 3})
    case("setValues given as a whole dictionary: a reference to a dictionary constant.",
         PARS, {"c": change(M, "#constants:chg")},
         {"k": "#tasks:c.model['k']", "j": "#tasks:c.model['j']"}, {"k": A(5.0), "j": A(6.0)},
         "The dictionary {k: 5, j: 6} sets both.", constants={"chg": {"k": 5, "j": 6}})
    case("setValues given as a whole dictionary: a reference to a data block.",
         PARS, {"blk": {"_type": "createDataBlock", "data": {"k": 0.1, "j": 7}}, "c": change(M, "#tasks:blk")},
         {"k": "#tasks:c.model['k']", "j": "#tasks:c.model['j']"}, {"k": A(0.1), "j": A(7.0)},
         "The data block has the labels k and j with the values 0.1 and 7; each label names an element to set.")
    case("The same model imported twice gives two independent models.",
         PARS, {"c": change("#tasks:m1.model", {"k": 9})},
         {"changed": "#tasks:c.model['k']", "m1": "#tasks:m1.model['k']", "m2": "#tasks:m2.model['k']"},
         {"changed": A(9.0), "m1": A(2.0), "m2": A(2.0)},
         "Changing a model made from the first import leaves both imports alone.", imports=("m1", "m2"))

    # ------------------------------------------------------------------ changed models used by other tasks
    case("A changed model is simulated and the original is simulated too: S' = -k1 S, S(0) = 3, k1 = 0.5 or 2.",
         DECAY, {"c": change(M, {"k1": 2}), "s0": one_step(M, 2), "s1": one_step("#tasks:c.model", 2)},
         {"orig": "#tasks:s0['S']", "new": "#tasks:s1['S']"}, {"orig": A(3 * np.exp(-1.0)), "new": A(3 * np.exp(-4.0))},
         "S(2) = 3 exp(-2 k1): 3 exp(-1) for k1 = 0.5 and 3 exp(-4) for k1 = 2.")
    case("A ModelChange applied to the end state of a simulation: the change takes effect from there.",
         DECAY, {"s1": one_step(M, 1), "c": change("#tasks:s1.model", {"k1": 1}), "s2": one_step("#tasks:c.model", 1)},
         {"first": "#tasks:s1['S']", "second": "#tasks:s2['S']"},
         {"first": A(3 * np.exp(-0.5)), "second": A(3 * np.exp(-0.5) * np.exp(-1.0))},
         "S(1) = 3 exp(-0.5) with k1 = 0.5.  Then k1 is set to 1 without touching the state, and one more time unit "
         "gives S(2) = S(1) exp(-1).")
    case("A changed model feeds a steady state: -> A at rate v and A -> at rate k1 A, with v = 6 and k1 = 2.",
         SOURCE, {"c": change(M, {"v": 6, "k1": 2}),
                  "s": {"_type": "steadyState", "model": "#tasks:c.model", "outputVariables": ["A"]}},
         {"r": "#tasks:s"}, {"r": A([3.0], rows=["A"])},
         "A = v/k1 = 6/2 = 3 for the changed model.")

    # ------------------------------------------------------------------ removeElements
    t = np.array([0.0, 1.0, 2.0])
    case("removeElements: removing the reaction J1 from a chain -> S1 -> S2 -> .  The original is simulated for comparison.",
         CHAIN, {"c": change(M, remove=["J1"]), "s0": course(M, ["S1", "S2"]),
                 "s1": course("#tasks:c.model", ["S1", "S2"])},
         {"orig": "#tasks:s0", "cut": "#tasks:s1"},
         {"orig": table(t, S1=4 - 3 * np.exp(-0.5 * t), S2=2 - 3 * np.exp(-0.5 * t) + 1.5 * np.exp(-t)),
          "cut": table(t, S1=1 + 2 * t, S2=0.5 * np.exp(-t))},
         "Original: S1' = 2 - 0.5 S1 gives S1 = 4 - 3 exp(-t/2); S2' = 0.5 S1 - S2 gives S2 = 2 - 3 exp(-t/2) + 1.5 exp(-t).  "
         "Without J1 nothing converts S1: S1' = 2, S1 = 1 + 2 t, and S2' = -S2 gives S2 = 0.5 exp(-t).")
    case("removeElements with the names given by a reference to a list constant, removing two reactions.",
         CHAIN, {"c": change(M, remove="#constants:gone"), "s": course("#tasks:c.model", ["S1", "S2"])},
         {"r": "#tasks:s"}, {"r": table(t, S1=1 + 2 * t, S2=[0.5, 0.5, 0.5])},
         "Without J1 and J2, S1' = 2 (S1 = 1 + 2 t) and S2 does not change (0.5).", constants={"gone": ["J1", "J2"]})
    case("removeElements followed by setValues in a separate ModelChange.",
         CHAIN, {"c1": change(M, remove=["J1"]), "c2": change("#tasks:c1.model", {"k2": 2}),
                 "s": course("#tasks:c2.model", ["S1", "S2"])},
         {"r": "#tasks:s"}, {"r": table(t, S1=1 + 2 * t, S2=0.5 * np.exp(-2 * t))},
         "With J1 removed and k2 = 2: S1 = 1 + 2 t and S2' = -2 S2, S2 = 0.5 exp(-2 t).")

    # ------------------------------------------------------------------ labels of .model
    case("Labels on .model: a species concentration, a compartment size, a parameter, an assigned value, a reaction flux and "
         "an amount-valued species.",
         LAB, {}, {"Sa": "#tasks:m.model['Sa']", "cell": "#tasks:m.model['cell']", "p": "#tasks:m.model['p']",
                   "q": "#tasks:m.model['q']", "J": "#tasks:m.model['J']", "Sc": "#tasks:m.model['Sc']"},
         {"Sa": A(3.0), "cell": A(2.0), "p": A(5.0), "q": A(10.0), "J": A(3.0), "Sc": A(4.0)},
         "Sa is a concentration, 3, in a compartment of size 2; p = 5; q := 2 p = 10; the flux of J is k Sa cell = "
         "0.5 * 3 * 2 = 3 (amount per time); Sc has only substance units, so its value is its amount, 4.")
    case("Labels on .model after a simulation: the values at the end state.",
         DECAY, {"s": one_step(M, 2)},
         {"S": "#tasks:s.model['S']", "J": "#tasks:s.model['J']", "k1": "#tasks:s.model['k1']"},
         {"S": A(3 * np.exp(-1.0)), "J": A(0.5 * 3 * np.exp(-1.0)), "k1": A(0.5)},
         "At t = 2, S = 3 exp(-1); the flux of J is k1 S; k1 is still 0.5.")
