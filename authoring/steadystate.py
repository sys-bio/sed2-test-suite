"""Series "steadystate": SteadyState, JacobianFull and JacobianReduced against hand-derived answers.

OpenCOR (libopencor) has neither a steady-state solver nor a Jacobian that can be called, so these cases run on
roadrunner and COPASI.  Every species starts at a non-zero value: COPASI's Jacobian is wrong for a species whose value
is exactly zero (see disagreements.json D-001), and the translator's COPASI backend refuses that case.
"""
import math

import numpy as np

from _util import A, write_case, outputs, IMPORT_SBML

FIRST = 152
TIME = "urn:sedml:symbol:time"


def build():
    counter = iter(range(FIRST, FIRST + 1000))

    def case(what, ant, tasks, refs, expected, derivation, constants=None, **kw):
        n = next(counter)
        cid = f"{n:05d}"
        all_tasks = {"m": {"_type": "modelImport", "location": f"{cid}.sbml", "language": IMPORT_SBML}}
        for tid, t in tasks.items():
            t = dict(t)
            if t.get("model") == "MODEL":
                t["model"] = "#tasks:m.model"
            all_tasks[tid] = t
        doc = {}
        if constants:
            doc["constants"] = constants
        doc["tasks"] = all_tasks
        doc["outputs"] = outputs(refs)
        return write_case(n, what, doc, expected, derivation=derivation, antimony={"": ant}, **kw)

    def steady(outvars, algorithm=None, variable=None, model="MODEL"):
        t = {"_type": "steadyState", "model": model, "outputVariables": outvars}
        if algorithm:
            t["workingAlgorithms"] = [{"algorithm": algorithm}]
        if variable:
            t["independentVariable"] = variable
        return t

    def jac(kind="jacobianFull", model="MODEL"):
        return {"_type": kind, "model": model}

    def mat(values, species):
        return A(np.array(values, dtype=float), rows=species, cols=species)

    def vec(values, names):
        return A(np.array(values, dtype=float), rows=names)

    # ------------------------------------------------------------------ steady states
    SRC = "model src\n  -> A; v\n  A -> B; k1*A\n  B -> ; k2*B\n  v = 2\n  k1 = 1\n  k2 = 0.5\n  A = 1\n  B = 1\nend"
    case("Steady state of a source feeding a two-step chain: -> A at rate 2, A -> B at rate A, B -> at rate B/2.",
         SRC, {"s": steady(["A", "B"])}, {"r": "#tasks:s"}, {"r": vec([2.0, 4.0], ["A", "B"])},
         "dA/dt = v - k1 A = 0 gives A = v/k1 = 2; dB/dt = k1 A - k2 B = 0 gives B = k1 A/k2 = 4.  The result is a vector "
         "labelled by the output variables.  The starting values (1, 1) do not matter for this linear model.")
    case("SteadyState with the independent variable given, in a different order of output variables, with an element picked out.",
         SRC, {"s": steady(["B", "A"], variable=TIME)}, {"r": "#tasks:s", "b": "#tasks:s['B']"},
         {"r": vec([4.0, 2.0], ["B", "A"]), "b": A(4.0)},
         "As before, with the output variables in the order B, A: B = 4, A = 2.")
    case("SteadyState with the generic root-finding algorithm named (KISAO:0000407).",
         SRC, {"s": steady(["A", "B"], algorithm="KISAO:0000407")}, {"r": "#tasks:s"}, {"r": vec([2.0, 4.0], ["A", "B"])},
         "As the first case: A = 2, B = 4.")
    case("A non-linear steady state: production at rate 2 and removal at rate 0.5 S^2, from S = 1.5.",
         "model sq\n  -> S; v\n  S -> ; k*S^2\n  v = 2\n  k = 0.5\n  S = 1.5\nend",
         {"s": steady(["S"])}, {"r": "#tasks:s"}, {"r": vec([2.0], ["S"])},
         "dS/dt = v - k S^2 = 0 gives S = sqrt(v/k) = sqrt(4) = 2 (the positive root, which the starting value 1.5 is near).")
    case("A closed system with a conservation law: A -> B reversible with kf = 1, kr = 0.5 and A + B = 3.",
         "model closed\n  A -> B; kf*A - kr*B\n  kf = 1\n  kr = 0.5\n  A = 2\n  B = 1\nend",
         {"s": steady(["A", "B"])}, {"r": "#tasks:s"}, {"r": vec([1.0, 2.0], ["A", "B"])},
         "At steady state kf A = kr B, so B = 2 A, and A + B = 3 gives A = 1, B = 2.  The Jacobian of this model is singular "
         "because of the conservation law A + B = 3.")
    case("Parameters and assigned values as output variables of a steady state: x is produced at rate a and removed at "
         "rate b x, and z := 2 x + 1.",
         "model par\n  -> x; a\n  x -> ; b*x\n  a = 6\n  b = 3\n  x = 1\n  z := 2*x + 1\nend",
         {"s": steady(["x", "z", "a"])}, {"r": "#tasks:s"}, {"r": vec([2.0, 5.0, 6.0], ["x", "z", "a"])},
         "x = a/b = 2; z = 2 x + 1 = 5; the constant a is 6.")
    case("The steady state does not depend on the volume of the compartment: -> S at rate 2 and S -> at rate 0.5 [S] in a compartment of volume 4.",
         "model vol\n  compartment c = 4\n  species S in c\n  S = 1\n  -> S; v\n  S -> ; k*S\n  v = 2\n  k = 0.5\nend",
         {"s": steady(["S", "c"])}, {"r": "#tasks:s"}, {"r": vec([4.0, 4.0], ["S", "c"])},
         "Amounts: dn/dt = v - k [S] = 0 gives [S] = v/k = 4 (the volume only scales the amount n = 4 [S] = 16).  The "
         "compartment size is 4.")
    case("The model at the steady state: elements of the resulting model can be read with a label.",
         SRC, {"s": steady(["A"])}, {"a": "#tasks:s.model['A']", "b": "#tasks:s.model['B']", "v": "#tasks:s.model['v']"},
         {"a": A(2.0), "b": A(4.0), "v": A(2.0)},
         "The whole state is at the steady state, not only the output variables: A = 2, B = 4; the parameter v is 2.")
    case("A steady state reached after a time course: the second task starts from where the first ended.",
         SRC, {"t": {"_type": "oneStepODESimulation", "model": "MODEL", "independentVariable": TIME,
                     "outputVariables": ["A"], "independentStep": 0.5},
               "s": steady(["A", "B"], model="#tasks:t.model")},
         {"r": "#tasks:s"}, {"r": vec([2.0, 4.0], ["A", "B"])},
         "Whatever the model has reached after t = 0.5, the steady state of this stable linear model is the same: A = 2, B = 4.")

    # ------------------------------------------------------------------ Jacobians
    CHAIN = "model jc\n  A -> B; k1*A\n  B -> ; k2*B\n  k1 = 1\n  k2 = 2\n  A = 1\n  B = 0.5\nend"
    case("JacobianFull of a linear chain A -> B -> (k1 = 1, k2 = 2) at the initial state A = 1, B = 0.5.",
         CHAIN, {"j": jac()}, {"r": "#tasks:j", "ba": "#tasks:j['B', 'A']", "row": "#tasks:j['B']"},
         {"r": mat([[-1, 0], [1, -2]], ["A", "B"]), "ba": A(1.0), "row": A([1.0, -2.0], rows=["A", "B"])},
         "dA/dt = -k1 A and dB/dt = k1 A - k2 B, so J = [[-k1, 0], [k1, -k2]] = [[-1, 0], [1, -2]] with rows and columns "
         "labelled A, B.  J[B, A] = dB'/dA = 1; the row B is (1, -2) and keeps the column labels.")
    case("JacobianFull of a non-linear model at the initial state: 2 A -> B at rate 0.5 A^2, A = 2, B = 1.",
         "model dim\n  A + A -> B; k*A^2\n  k = 0.5\n  A = 2\n  B = 1\nend",
         {"j": jac()}, {"r": "#tasks:j"}, {"r": mat([[-4, 0], [2, 0]], ["A", "B"])},
         "dA/dt = -2 k A^2 and dB/dt = k A^2, so dA'/dA = -4 k A = -4, dB'/dA = 2 k A = 2 and neither depends on B.")
    case("JacobianFull of a one-species model with a saturable removal: S' = -Vmax S/(Km + S) at S = 3 (Vmax = 2, Km = 1).",
         "model mm\n  S -> ; Vmax*S/(Km + S)\n  Vmax = 2\n  Km = 1\n  S = 3\nend",
         {"j": jac()}, {"r": "#tasks:j"}, {"r": mat([[-2 * 1 / 16]], ["S"])},
         "d/dS of -Vmax S/(Km + S) is -Vmax Km/(Km + S)^2 = -2/16 = -0.125.")
    case("JacobianFull of a bimolecular reaction: A + B -> C at rate 0.5 A B, A = 2, B = 3, C = 1.",
         "model bim\n  A + B -> C; k*A*B\n  k = 0.5\n  A = 2\n  B = 3\n  C = 1\nend",
         {"j": jac()}, {"r": "#tasks:j"},
         {"r": mat([[-1.5, -1.0, 0], [-1.5, -1.0, 0], [1.5, 1.0, 0]], ["A", "B", "C"])},
         "v = k A B: dv/dA = k B = 1.5, dv/dB = k A = 1.  A and B lose v, C gains it: rows A and B are (-1.5, -1, 0), row C is "
         "(1.5, 1, 0).")
    case("The Jacobian at the end state of a time course: 2 A -> B at rate 0.5 A^2 run until t = 2, from A = 2.",
         "model dim\n  A + A -> B; k*A^2\n  k = 0.5\n  A = 2\n  B = 1\nend",
         {"s": {"_type": "oneStepODESimulation", "model": "MODEL", "independentVariable": TIME, "outputVariables": ["A"],
                "independentStep": 2},
          "j": jac(model="#tasks:s.model")},
         {"r": "#tasks:j"},
         {"r": mat([[-4 * 0.5 * (2 / (1 + 2 * 0.5 * 2 * 2)), 0], [2 * 0.5 * (2 / (1 + 2 * 0.5 * 2 * 2)), 0]], ["A", "B"])},
         "A(t) = A0/(1 + 2 k A0 t) = 2/(1 + 4) = 0.4 at t = 2.  Then dA'/dA = -4 k A = -0.8 and dB'/dA = 2 k A = 0.4.")
    case("The Jacobian at a steady state: -> S at rate 2 and S -> at rate 0.5 S^2.",
         "model sq\n  -> S; v\n  S -> ; k*S^2\n  v = 2\n  k = 0.5\n  S = 1.5\nend",
         {"s": steady(["S"]), "j": jac(model="#tasks:s.model")}, {"r": "#tasks:j"}, {"r": mat([[-2.0]], ["S"])},
         "S = 2 at the steady state and d(v - k S^2)/dS = -2 k S = -2.")
    case("JacobianReduced of a model without a conservation law is the full Jacobian.",
         CHAIN, {"j": jac("jacobianReduced")}, {"r": "#tasks:j"}, {"r": mat([[-1, 0], [1, -2]], ["A", "B"])},
         "Nothing is conserved by the chain A -> B -> , so no species is removed and the matrix is the full one: "
         "[[-1, 0], [1, -2]].")
    case("JacobianFull of a model with a conservation law: A -> B reversible, kf = 1, kr = 0.5, A = 2, B = 1.",
         "model closed\n  A -> B; kf*A - kr*B\n  kf = 1\n  kr = 0.5\n  A = 2\n  B = 1\nend",
         {"j": jac()}, {"r": "#tasks:j"}, {"r": mat([[-1.0, 0.5], [1.0, -0.5]], ["A", "B"])},
         "dA/dt = -kf A + kr B and dB/dt = kf A - kr B give J = [[-kf, kr], [kf, -kr]] = [[-1, 0.5], [1, -0.5]], which is "
         "singular (A + B is conserved).")

    # ------------------------------------------------------------------ more Jacobians, including reduced ones
    ENZ = ("model enz\n  E + S -> ES; k1*E*S - k2*ES\n  ES -> E + P; k3*ES\n  k1 = 1\n  k2 = 0.5\n  k3 = 2\n"
           "  E = 1\n  S = 5\n  ES = 0.5\n  P = 0.5\nend")
    case("JacobianFull of an enzyme mechanism, E + S <-> ES -> E + P, with E = 1, S = 5, ES = 0.5, P = 0.5 (k1 = 1, k2 = 0.5, k3 = 2).",
         ENZ, {"j": jac()}, {"r": "#tasks:j"},
         {"r": mat([[-5, -1, 2.5, 0], [-5, -1, 0.5, 0], [5, 1, -2.5, 0], [0, 0, 2, 0]], ["E", "S", "ES", "P"])},
         "With v1 = k1 E S - k2 ES and v2 = k3 ES: dE/dt = -v1 + v2, dS/dt = -v1, dES/dt = v1 - v2, dP/dt = v2.  "
         "dv1/dE = k1 S = 5, dv1/dS = k1 E = 1, dv1/dES = -k2 = -0.5, dv2/dES = k3 = 2.  Rows: E = (-5, -1, 2.5, 0), "
         "S = (-5, -1, 0.5, 0), ES = (5, 1, -2.5, 0), P = (0, 0, 2, 0).")
    case("JacobianReduced with one conservation law: A -> B reversible (kf = 1, kr = 0.5), A + B = 3.  The independent species is A.",
         "model closed\n  A -> B; kf*A - kr*B\n  kf = 1\n  kr = 0.5\n  A = 2\n  B = 1\nend",
         {"j": jac("jacobianReduced")}, {"r": "#tasks:j"}, {"r": mat([[-1.5]], ["A"])},
         "B = 3 - A, so dA/dt = -kf A + kr (3 - A) and the derivative with respect to A is -(kf + kr) = -1.5.",
         notes="Which species is dropped is not defined by the specification (SED2/TODO.md: JacobianReduced); roadrunner and "
               "COPASI both keep the first species of the conservation law (A).")
    case("JacobianReduced of a chain with one conservation law: A <-> B <-> C, A + B + C = 6.  The independent species are A and B.",
         "model c3\n  A -> B; k1*A - k2*B\n  B -> C; k3*B - k4*C\n  k1 = 1\n  k2 = 0.5\n  k3 = 2\n  k4 = 0.25\n"
         "  A = 1\n  B = 2\n  C = 3\nend",
         {"j": jac("jacobianReduced")}, {"r": "#tasks:j"}, {"r": mat([[-1.0, 0.5], [0.75, -2.75]], ["A", "B"])},
         "C = 6 - A - B.  dA/dt = -k1 A + k2 B gives (-1, 0.5).  dB/dt = k1 A - k2 B - k3 B + k4 C = k1 A - (k2 + k3) B + "
         "k4 (6 - A - B) gives d/dA = k1 - k4 = 0.75 and d/dB = -(k2 + k3) - k4 = -2.75.",
         notes="Which species is dropped is not defined by the specification (SED2/TODO.md: JacobianReduced).")
    case("JacobianReduced of the enzyme mechanism, which has two conservation laws (E + ES and S + ES + P).  The independent species are E and S.",
         ENZ, {"j": jac("jacobianReduced")}, {"r": "#tasks:j"}, {"r": mat([[-7.5, -1.0], [-5.5, -1.0]], ["E", "S"])},
         "E + ES = 1.5 and S + ES + P = 6, so ES = 1.5 - E.  dE/dt = -k1 E S + (k2 + k3)(1.5 - E) and dS/dt = -k1 E S + "
         "k2 (1.5 - E): d(dE)/dE = -k1 S - (k2 + k3) = -7.5, d(dE)/dS = -k1 E = -1, d(dS)/dE = -k1 S - k2 = -5.5, "
         "d(dS)/dS = -k1 E = -1.",
         notes="Which species are dropped is not defined by the specification (SED2/TODO.md: JacobianReduced).")
