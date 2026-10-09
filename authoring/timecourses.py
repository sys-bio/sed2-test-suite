"""Series "timecourses": ODE time courses against closed-form solutions (explicit, one-step and bounded simulations).

Every model has an analytic solution, written out in the case description; the expected tables are computed from
that formula, never from a simulator.  The generated scripts set the solver tolerances to 1e-12 (absolute) and 1e-10
(relative) unless the document says otherwise, far tighter than the suite's comparison tolerances.
"""
import math

import numpy as np

from _util import A, write_case, outputs, IMPORT_SBML

FIRST = 105
TIME = "urn:sedml:symbol:time"


def table(times, _label=TIME, **cols):
    """The time course table; the first column is labelled with the independentVariable attribute as written."""
    times = np.asarray(times, dtype=float)
    data = np.column_stack([times] + [np.asarray(v, dtype=float) for v in cols.values()])
    return A(data, cols=[_label] + list(cols))


# Note: the first column's label is the independentVariable attribute as written, because outputs.json gives the labels
# of a time course as `[independentVariable] + outputVariables`.


def lin(start, end, steps):
    return np.linspace(start, end, steps + 1)


def build():
    counter = iter(range(FIRST, FIRST + 1000))

    def case(what, ant, sims, refs, expected, derivation, extra_tasks=None, constants=None, models=1, **kw):
        """`sims` maps task ids to simulation dicts that may use "MODEL" as the model reference.  The model task is
        `m` (`m2` for a second model written to NNNNN.second.sbml)."""
        n = next(counter)
        cid = f"{n:05d}"
        tasks = {"m": {"_type": "modelImport", "location": f"{cid}.sbml", "language": IMPORT_SBML}}
        if models == 2:
            tasks["m2"] = {"_type": "modelImport", "location": f"{cid}.second.sbml", "language": IMPORT_SBML}
        for tid, sim in (extra_tasks or {}).items():
            tasks[tid] = sim
        for tid, sim in sims.items():
            sim = dict(sim)
            if sim.get("model") == "MODEL":
                sim["model"] = "#tasks:m.model"
            tasks[tid] = sim
        doc = {}
        if constants:
            doc["constants"] = constants
        doc["tasks"] = tasks
        doc["outputs"] = outputs(refs)
        antimony = ant if isinstance(ant, dict) else {"": ant}
        return write_case(n, what, doc, expected, derivation=derivation, antimony=antimony, **kw)

    def explicit(outvars, start, end, steps, init=None, algorithm=None, scale=None, **settings):
        rng = {"_type": "numericRange", "start": start, "end": end, "numberOfSteps": steps}
        if scale:
            rng["scale"] = scale
        return _sim("explicitODESimulation", outvars, algorithm, init, independentVariableRange=rng, **settings)

    def _sim(kind, outvars, algorithm=None, init=None, **rest):
        sim = {"_type": kind, "model": "MODEL", "independentVariable": TIME, "outputVariables": outvars}
        if init is not None:
            sim["independentVariableInit"] = init
        if algorithm:
            sim["workingAlgorithms"] = [{"algorithm": algorithm}]
        sim.update(rest)
        return sim

    def onestep(outvars, step, init=None, algorithm=None, **settings):
        return _sim("oneStepODESimulation", outvars, algorithm, init, independentStep=step, **settings)

    def bounded(outvars, start, end, init=None, algorithm=None, **settings):
        return _sim("boundedODESimulation", outvars, algorithm, init,
                    independentVariableSpan={"_type": "span", "start": start, "end": end}, **settings)

    def last_row(note="the solver chooses the output points, so only the end of the run is compared"):
        return {"r": {"mode": "lastRow", "note": note}}

    DECAY = "model decay\n  S -> ; k*S\n  k = 0.5\n  S = 10\nend"
    T10 = lin(0, 10, 20)
    S_decay = 10 * np.exp(-0.5 * T10)

    # ------------------------------------------------------------------ the basic shapes
    case("A time course with explicit output points: first order decay S -> with k = 0.5 and S(0) = 10.",
         DECAY, {"s": explicit(["S"], 0, 10, 20)}, {"r": "#tasks:s"}, {"r": table(T10, S=S_decay)},
         "dS/dt = -k S gives S(t) = 10 exp(-t/2).  The 21 rows are t = 0, 0.5, ..., 10; the columns are labelled time and S.")
    case("Parts of a time course picked by index: a column by label, one value, a row, a range of rows.",
         DECAY, {"s": explicit(["S"], 0, 10, 20)},
         {"col": "#tasks:s[:, 'S']", "t": "#tasks:s[:, 0]", "last": "#tasks:s[-1, 'S']", "row": "#tasks:s[4]",
          "mid": "#tasks:s[2:5, 'S']"},
         {"col": A(S_decay), "t": A(T10), "last": A(S_decay[-1]), "row": A([T10[4], S_decay[4]], rows=[TIME, "S"]),
          "mid": A(S_decay[2:5])},
         "The table of the previous case (S(t) = 10 exp(-t/2), t = 0, 0.5, ..., 10) indexed: col is the S column, t the "
         "time column, last is S(10) = 10 exp(-5), row is row 4 (t = 2: 2, 10 exp(-1)) and keeps the column labels as its "
         "labels, mid is rows 2, 3, 4 of S.")
    case("The independent variable may be written time instead of the URN.",
         DECAY, {"s": _sim("explicitODESimulation", ["S"], independentVariableRange={
             "_type": "numericRange", "start": 0, "end": 4, "numberOfSteps": 4}) | {"independentVariable": "time"}},
         {"r": "#tasks:s"}, {"r": table(lin(0, 4, 4), _label="time", S=10 * np.exp(-0.5 * lin(0, 4, 4)))},
         "S(t) = 10 exp(-t/2) at t = 0, 1, 2, 3, 4.  The first column is labelled with the independentVariable as written, "
         "time.")
    k1, k2 = 0.3, 0.7
    T = lin(0, 12, 24)
    A_, B_ = np.exp(-k1 * T), k1 / (k2 - k1) * (np.exp(-k1 * T) - np.exp(-k2 * T))
    case("A chain A -> B -> C with k1 = 0.3, k2 = 0.7, starting with A = 1.",
         "model chain\n  A -> B; k1*A\n  B -> C; k2*B\n  k1 = 0.3\n  k2 = 0.7\n  A = 1\n  B = 0\n  C = 0\nend",
         {"s": explicit(["A", "B", "C"], 0, 12, 24)}, {"r": "#tasks:s"},
         {"r": table(T, A=A_, B=B_, C=1 - A_ - B_)},
         "A = exp(-k1 t); B = k1/(k2 - k1) (exp(-k1 t) - exp(-k2 t)); C = 1 - A - B (mass is conserved).")
    kf, kr = 1.0, 0.5
    T = lin(0, 6, 12)
    aeq = kr / (kf + kr)
    Aa = aeq + (1 - aeq) * np.exp(-(kf + kr) * T)
    case("A reversible reaction A <-> B (kf = 1, kr = 0.5) from A = 1, B = 0, which has a conservation law A + B = 1.",
         "model iso\n  A -> B; kf*A - kr*B\n  kf = 1\n  kr = 0.5\n  A = 1\n  B = 0\nend",
         {"s": explicit(["B", "A"], 0, 6, 12)}, {"r": "#tasks:s"},
         {"r": table(T, B=1 - Aa, A=Aa)},
         "A(t) = A_eq + (1 - A_eq) exp(-(kf + kr) t) with A_eq = kr/(kf + kr) = 1/3; B = 1 - A.  The output variables are "
         "listed in the order B, A and the columns follow that order.")

    # ------------------------------------------------------------------ what can be an output variable
    T = lin(0, 4, 8)
    x = np.exp(0.25 * T)
    case("Parameters are output variables too: a rate rule, an assignment rule and a linear rate rule.",
         "model params\n  x = 1\n  x' = r*x\n  r = 0.25\n  z := 2*x + 1\n  y = 5\n  y' = 3\nend",
         {"s": explicit(["x", "z", "y", "r"], 0, 4, 8)}, {"r": "#tasks:s"},
         {"r": table(T, x=x, z=2 * x + 1, y=5 + 3 * T, r=np.full_like(T, 0.25))},
         "x' = r x gives x = exp(t/4); z = 2 x + 1 follows; y' = 3 gives y = 5 + 3 t; the constant r stays 0.25.")
    T = lin(0, 6, 12)
    case("Concentration and volume: S has concentration 4 in a compartment of volume 2, and the rate k*S is in amount per time.",
         "model vol\n  compartment c = 2\n  species S in c\n  S = 4\n  S -> ; k*S\n  k = 0.5\nend",
         {"s": explicit(["S", "c"], 0, 6, 12)}, {"r": "#tasks:s"},
         {"r": table(T, S=4 * np.exp(-0.25 * T), c=np.full_like(T, 2.0))},
         "Amount n = 2 S, dn/dt = -k S = -(k/2) n, so n and [S] both decay with rate k/2 = 0.25: [S] = 4 exp(-t/4).  The "
         "compartment size stays 2.")
    case("A species with hasOnlySubstanceUnits is its amount: 4 in a compartment of volume 2.",
         "model amt\n  compartment c = 2\n  substanceOnly species S in c\n  S = 4\n  S -> ; k*S\n  k = 0.5\nend",
         {"s": explicit(["S"], 0, 6, 12)}, {"r": "#tasks:s"}, {"r": table(T, S=4 * np.exp(-0.5 * T))},
         "S is the amount n, dn/dt = -k n, so n = 4 exp(-t/2).")
    T = lin(0, 6, 12)
    nA = 3 * np.exp(-0.5 * T)
    case("Two compartments of different volume: A (volume 1) turns into B (volume 2) at rate k*A in amount per time.",
         "model two\n  compartment c1 = 1\n  compartment c2 = 2\n  species A in c1\n  species B in c2\n  A = 3\n  B = 0\n"
         "  A -> B; k*A\n  k = 0.5\nend",
         {"s": explicit(["A", "B"], 0, 6, 12)}, {"r": "#tasks:s"},
         {"r": table(T, A=nA, B=(3 - nA) / 2)},
         "Amounts: n_A = 3 exp(-t/2) (volume 1, so [A] = n_A) and n_B = 3 - n_A, so [B] = n_B / 2.")

    # ------------------------------------------------------------------ non-linear and special model features
    T = lin(0, 5, 10)
    a0, kk = 2.0, 0.5
    Aq = a0 / (1 + 2 * kk * a0 * T)
    case("Second order: 2 A -> B with rate k A^2 and A(0) = 2, k = 0.5 (each reaction event uses two A).",
         "model dimer\n  A + A -> B; k*A^2\n  k = 0.5\n  A = 2\n  B = 0\nend",
         {"s": explicit(["A", "B"], 0, 5, 10)}, {"r": "#tasks:s"}, {"r": table(T, A=Aq, B=(a0 - Aq) / 2)},
         "The rate of the reaction is v = k A^2 and A is consumed twice per event, so dA/dt = -2 k A^2: A = A0/(1 + 2 k "
         "A0 t) and B = (A0 - A)/2.")
    c0 = 1.0
    Ab = 1.0 / (1 + 0.5 * 1.0 * T)
    case("Bimolecular: A + B -> C with equal starting amounts A = B = 1, k = 0.5.",
         "model bimol\n  A + B -> C; k*A*B\n  k = 0.5\n  A = 1\n  B = 1\n  C = 0\nend",
         {"s": explicit(["A", "B", "C"], 0, 5, 10)}, {"r": "#tasks:s"}, {"r": table(T, A=Ab, B=Ab, C=1 - Ab)},
         "With A = B, dA/dt = -k A^2 so A = A0/(1 + k A0 t) = 1/(1 + t/2); C = 1 - A.")
    T = lin(0, 20, 20)
    r_, K_, x0_ = 0.5, 10.0, 1.0
    logi = K_ / (1 + (K_ - x0_) / x0_ * np.exp(-r_ * T))
    case("Logistic growth by a rate rule: x' = r x (1 - x/K) with r = 0.5, K = 10, x(0) = 1.",
         "model logistic\n  x = 1\n  x' = r*x*(1 - x/K)\n  r = 0.5\n  K = 10\nend",
         {"s": explicit(["x"], 0, 20, 20)}, {"r": "#tasks:s"}, {"r": table(T, x=logi)},
         "x(t) = K / (1 + ((K - x0)/x0) exp(-r t)).")
    T = lin(0, 10, 20)
    v_, k_, s0 = 2.0, 0.5, 1.0
    Sz = v_ / k_ + (s0 - v_ / k_) * np.exp(-k_ * T)
    case("Zero order production and first order decay: -> S at rate 2, S -> at rate 0.5 S, S(0) = 1.",
         "model prodecay\n  -> S; v\n  S -> ; k*S\n  v = 2\n  k = 0.5\n  S = 1\nend",
         {"s": explicit(["S"], 0, 10, 20)}, {"r": "#tasks:s"}, {"r": table(T, S=Sz)},
         "dS/dt = v - k S gives S = v/k + (S0 - v/k) exp(-k t) = 4 - 3 exp(-t/2).")
    T = lin(0, 5, 10)
    case("A fixed (boundary) species feeds a product at a constant rate: $A -> B with k = 1 and A = 2.",
         "model bound\n  $A -> B; k*A\n  A = 2\n  B = 0.5\n  k = 1\nend",
         {"s": explicit(["A", "B"], 0, 5, 10)}, {"r": "#tasks:s"}, {"r": table(T, A=np.full_like(T, 2.0), B=0.5 + 2 * T)},
         "A is a boundary species and stays at 2; dB/dt = k A = 2, so B = 0.5 + 2 t.")
    T = lin(0, 20, 80)
    w = 2.0
    case("The harmonic oscillator x' = v, v' = -w^2 x with w = 2, x(0) = 1, v(0) = 0.",
         "model osc\n  x = 1\n  v = 0\n  x' = v\n  v' = -w^2*x\n  w = 2\nend",
         {"s": explicit(["x", "v"], 0, 20, 80)}, {"r": "#tasks:s"},
         {"r": table(T, x=np.cos(w * T), v=-w * np.sin(w * T))},
         "x = cos(2 t), v = -2 sin(2 t).  The model has a pair of imaginary eigenvalues, so the solution does not decay.")
    T = np.array([0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 5.0])
    x0, y0 = 1.0, 0.0
    xs = x0 * np.exp(-1000 * T)
    ys = (y0 + x0 / 999) * np.exp(-T) - x0 / 999 * np.exp(-1000 * T)
    case("A stiff pair: x' = -1000 x and y' = -y + x, from x = 1, y = 0, sampled at points that span four decades.",
         "model stiff\n  x = 1\n  y = 0\n  x' = -1000*x\n  y' = -y + x\nend",
         {"s": _sim("explicitODESimulation", ["x", "y"], independentVariableRange={
             "_type": "numericRange", "values": [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 5.0]})},
         {"r": "#tasks:s"}, {"r": table(T, x=xs, y=ys)},
         "x = exp(-1000 t).  y' + y = exp(-1000 t) has the particular solution -exp(-1000 t)/999, so with y(0) = 0: "
         "y = (exp(-t) - exp(-1000 t))/999.")
    T = lin(0, 4, 8)
    case("A function definition used in a rate rule: f(x) = 2 x and S' = -f(S) with S(0) = 3.",
         "function f(x)\n  2*x\nend\nmodel fd\n  S = 3\n  S' = -f(S)\nend",
         {"s": explicit(["S"], 0, 4, 8)}, {"r": "#tasks:s"}, {"r": table(T, S=3 * np.exp(-2 * T))},
         "S' = -2 S gives S = 3 exp(-2 t).")
    case("An initial assignment: k = 2*kb with kb = 3 and S(0) = 4.",
         "model ia\n  S = 4\n  S -> ; k*S\n  k = 2*kb\n  kb = 3\nend",
         {"s": explicit(["S", "k"], 0, 1, 10)}, {"r": "#tasks:s"},
         {"r": table(lin(0, 1, 10), S=4 * np.exp(-6 * lin(0, 1, 10)), k=np.full(11, 6.0))},
         "k = 6 from the initial assignment, so S = 4 exp(-6 t).")
    ts = np.array([0.0, 1.0, 1.5, 2.5, 3.0, 4.0, 6.0])
    ev = np.where(ts < 2, 10 * np.exp(-0.5 * ts), 5 * np.exp(-0.5 * (ts - 2)))
    case("An event: at time 2 the amount of S is reset to 5 (decay k = 0.5 from S = 10).  No output point falls on the event time.",
         "model reset\n  S = 10\n  S -> ; k*S\n  k = 0.5\n  at (time > 2): S = 5\nend",
         {"s": _sim("explicitODESimulation", ["S"], independentVariableRange={
             "_type": "numericRange", "values": [0.0, 1.0, 1.5, 2.5, 3.0, 4.0, 6.0]})},
         {"r": "#tasks:s"}, {"r": table(ts, S=ev)},
         "S = 10 exp(-t/2) before t = 2 and 5 exp(-(t - 2)/2) after it.")
    T = lin(0, 12, 48)
    case("The time symbol inside the model: x' = cos(time) with x(0) = 0 and v := sin(time).",
         "model timedep\n  x = 0\n  x' = cos(time)\n  v := sin(time)\nend",
         {"s": explicit(["x", "v"], 0, 12, 48)}, {"r": "#tasks:s"}, {"r": table(T, x=np.sin(T), v=np.sin(T))},
         "x = sin(t) and v = sin(t).")

    # ------------------------------------------------------------------ the output range
    sd = lambda t: 10 * np.exp(-0.5 * t)
    case("The output range may start after zero: with independentVariableInit 0 the first output is at 3.",
         DECAY, {"s": explicit(["S"], 3, 7, 4, init=0)}, {"r": "#tasks:s"}, {"r": table(lin(3, 7, 4), S=sd(lin(3, 7, 4)))},
         "The model runs from 0 and the output starts at t = 3: 3, 4, 5, 6, 7, with S = 10 exp(-t/2).")
    tt = np.logspace(-1, 1, 5)
    case("A logarithmic output range: log10 scale from 0.1 to 10 in four steps.",
         DECAY, {"s": explicit(["S"], 0.1, 10, 4, scale="log10")}, {"r": "#tasks:s"}, {"r": table(tt, S=sd(tt))},
         "The points are 10^-1, 10^-0.5, 10^0, 10^0.5, 10^1.")
    tv = np.array([0.0, 0.3, 1.0, 4.0, 9.0])
    case("An output range given by a list of values, not evenly spaced.",
         DECAY, {"s": _sim("explicitODESimulation", ["S"], independentVariableRange={"_type": "numericRange",
                                                                                    "values": [0.0, 0.3, 1.0, 4.0, 9.0]})},
         {"r": "#tasks:s"}, {"r": table(tv, S=sd(tv))}, "S = 10 exp(-t/2) at t = 0, 0.3, 1, 4, 9.")
    ti = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
    case("An output range given by start, end and interval.",
         DECAY, {"s": _sim("explicitODESimulation", ["S"], independentVariableRange={
             "_type": "numericRange", "start": 0, "end": 10, "interval": 2.5})},
         {"r": "#tasks:s"}, {"r": table(ti, S=sd(ti))}, "The points 0, 2.5, 5, 7.5, 10.")
    case("The range, the output variables and the tolerances may be references to constants.",
         DECAY, {"s": _sim("explicitODESimulation", "#constants:vars", independentVariableRange={
             "_type": "numericRange", "start": "#constants:t0", "end": "#constants:t1", "numberOfSteps": "#constants:n"},
                           relativeTolerance="#constants:rtol", absoluteTolerance="#constants:atol")},
         {"r": "#tasks:s"}, {"r": table(lin(1, 5, 4), S=sd(lin(1, 5, 4)))},
         "t0 = 1, t1 = 5, n = 4: the times 1, 2, 3, 4, 5.  The tolerances are 1e-10 and 1e-12.",
         constants={"vars": ["S"], "t0": 1, "t1": 5, "n": 4, "rtol": 1e-10, "atol": 1e-12})
    case("A single output point: the range has one step, so two points (start and end).",
         DECAY, {"s": explicit(["S"], 0, 10, 1)}, {"r": "#tasks:s"}, {"r": table([0.0, 10.0], S=sd(np.array([0.0, 10.0])))},
         "S(0) = 10 and S(10) = 10 exp(-5).")
    case("Two simulations of the same model do not disturb one another or the imported model.",
         DECAY, {"s1": explicit(["S"], 0, 4, 4), "s2": explicit(["S"], 0, 2, 2)}, {"a": "#tasks:s1", "b": "#tasks:s2"},
         {"a": table(lin(0, 4, 4), S=sd(lin(0, 4, 4))), "b": table(lin(0, 2, 2), S=sd(lin(0, 2, 2)))},
         "Each starts from S = 10 at t = 0: a has t = 0..4, b has t = 0..2.")

    # ------------------------------------------------------------------ one-step and bounded simulations
    case("OneStepODESimulation: the values after advancing time by 5 (a vector of the output variables).",
         "model chain\n  A -> B; k1*A\n  B -> C; k2*B\n  k1 = 0.3\n  k2 = 0.7\n  A = 1\n  B = 0\n  C = 0\nend",
         {"s": onestep(["A", "B", "C"], 5)}, {"r": "#tasks:s"},
         {"r": A([math.exp(-1.5), 0.3 / 0.4 * (math.exp(-1.5) - math.exp(-3.5)), 1 - math.exp(-1.5) - 0.3 / 0.4 * (math.exp(-1.5) - math.exp(-3.5))],
                 rows=["A", "B", "C"])},
         "At t = 5: A = exp(-1.5), B = 0.75 (exp(-1.5) - exp(-3.5)), C = 1 - A - B.  The result is a vector labelled by the "
         "output variables.")
    case("OneStepODESimulation of a model that depends on time, started at independentVariableInit 1 and advanced by 2.",
         "model timedep\n  x = 0.5\n  x' = cos(time)\nend",
         {"s": onestep(["x"], 2, init=1)}, {"r": "#tasks:s"}, {"r": A([0.5 + math.sin(3) - math.sin(1)], rows=["x"])},
         "x(3) = x(1) + sin(3) - sin(1), and x at time 1 is the initial value 0.5, because the model starts at time 1.")
    case("OneStepODESimulation: a labelled value picked out of the result.",
         DECAY, {"s": onestep(["S"], 3)}, {"r": "#tasks:s", "s": "#tasks:s['S']"},
         {"r": A([10 * math.exp(-1.5)], rows=["S"]), "s": A(10 * math.exp(-1.5))},
         "S(3) = 10 exp(-1.5).")
    case("BoundedODESimulation: the solver chooses the output points, so only the last row (time 10) is compared.",
         DECAY, {"s": bounded(["S"], 0, 10)}, {"r": "#tasks:s"}, {"r": table([10.0], S=[sd(10.0)])},
         "The run ends at t = 10 with S = 10 exp(-5).", compare=last_row())
    case("BoundedODESimulation of a chain; only the end of the run is compared.",
         "model chain\n  A -> B; k1*A\n  B -> C; k2*B\n  k1 = 0.3\n  k2 = 0.7\n  A = 1\n  B = 0\n  C = 0\nend",
         {"s": bounded(["A", "B", "C"], 0, 8)}, {"r": "#tasks:s"},
         {"r": table([8.0], A=[math.exp(-2.4)], B=[0.75 * (math.exp(-2.4) - math.exp(-5.6))],
                     C=[1 - math.exp(-2.4) - 0.75 * (math.exp(-2.4) - math.exp(-5.6))])},
         "At t = 8: A = exp(-2.4), B = 0.75 (exp(-2.4) - exp(-5.6)), C = 1 - A - B.", compare=last_row())

    # ------------------------------------------------------------------ the model after a run
    case("The model at the end of a run: its elements can be read with a label.",
         "model chain\n  A -> B; k1*A\n  B -> C; k2*B\n  k1 = 0.3\n  k2 = 0.7\n  A = 1\n  B = 0\n  C = 0\nend",
         {"s": explicit(["A"], 0, 10, 5)}, {"a": "#tasks:s.model['A']", "b": "#tasks:s.model['B']", "k1": "#tasks:s.model['k1']"},
         {"a": A(math.exp(-3.0)), "b": A(0.75 * (math.exp(-3.0) - math.exp(-7.0))), "k1": A(0.3)},
         "After t = 10: A = exp(-3), B = 0.75 (exp(-3) - exp(-7)) (B was not an output variable, but it is in the model), "
         "and the parameter k1 is 0.3.")
    case("A second simulation can start from the end state of the first.",
         DECAY, {"s1": explicit(["S"], 0, 2, 2),
                 "s2": {"_type": "explicitODESimulation", "model": "#tasks:s1.model", "independentVariable": TIME,
                        "independentVariableInit": 0, "outputVariables": ["S"],
                        "independentVariableRange": {"_type": "numericRange", "start": 0, "end": 4, "numberOfSteps": 4}}},
         {"a": "#tasks:s1", "b": "#tasks:s2"},
         {"a": table(lin(0, 2, 2), S=sd(lin(0, 2, 2))), "b": table(lin(0, 4, 4), S=sd(2) * np.exp(-0.5 * lin(0, 4, 4)))},
         "s1 ends at t = 2 with S = 10 exp(-1).  s2 starts from that state and its own clock from 0: S = 10 exp(-1) exp(-t/2).")
    case("A one-step simulation from the end of an explicit one.",
         DECAY, {"s1": explicit(["S"], 0, 2, 2),
                 "s2": {"_type": "oneStepODESimulation", "model": "#tasks:s1.model", "independentVariable": TIME,
                        "outputVariables": ["S"], "independentStep": 3}},
         {"r": "#tasks:s2", "m": "#tasks:s2.model['S']"},
         {"r": A([10 * math.exp(-2.5)], rows=["S"]), "m": A(10 * math.exp(-2.5))},
         "After 2 and then 3 more time units S = 10 exp(-5/2).")

    # ------------------------------------------------------------------ algorithms (a case per KiSAO term)
    T = lin(0, 10, 20)
    tol_loose = {"absolute": 1e-4, "relative": 1e-3}
    tol_rk = {"absolute": 1e-3, "relative": 1e-3}
    algos = [
        ("KISAO:0000694", "the generic ODE solver: each simulator uses its own default", None, None),
        ("KISAO:0000019", "CVODE", None, None),
        ("KISAO:0000288", "BDF (CVODE with the stiff method)", None, None),
        ("KISAO:0000280", "Adams-Moulton (CVODE with the non-stiff method)", None, None),
        ("KISAO:0000088", "LSODA", None, None),
        ("KISAO:0000304", "Radau5", None, None),
        ("KISAO:0000032", "a fourth order Runge-Kutta method with a fixed step", None, tol_rk),
        ("KISAO:0000030", "the forward Euler method with a small fixed step (first order, so a loose tolerance)",
         {"initialStepSize": 0.0001}, tol_loose),
    ]
    for term, name, extra, tol in algos:
        why = ("a first order method with step 1e-4 is accurate only to about 1e-4" if "Euler" in name else
               "a fixed step method whose step the simulator chooses (here about the output spacing) is accurate only to "
               "about 2e-4 relative")
        kw = {"tolerances": tol, "tolerance_note": f"loose (absolute {tol['absolute']:g}, relative {tol['relative']:g}): {why}."} if tol else {}
        case(f"{name} ({term}), first order decay S -> with k = 0.5 and S(0) = 10.",
             DECAY, {"s": explicit(["S"], 0, 10, 20, algorithm=term, **(extra or {}))}, {"r": "#tasks:s"},
             {"r": table(T, S=S_decay)}, "S(t) = 10 exp(-t/2) as in the first case.  A simulator that does not provide "
             "this algorithm skips the case.", **kw)

    # ------------------------------------------------------------------ solver settings
    case("Solver settings: tight tolerances and a step limit (relative 1e-12, absolute 1e-14, at most 100000 steps).",
         DECAY, {"s": explicit(["S"], 0, 10, 20, relativeTolerance=1e-12, absoluteTolerance=1e-14, maxNumberOfSteps=100000)},
         {"r": "#tasks:s"}, {"r": table(T, S=S_decay)}, "S(t) = 10 exp(-t/2) as before.")
    case("Solver settings: the stiff solver requested on a stiff problem (x' = -1000 x, y' = -y + x).",
         "model stiff\n  x = 1\n  y = 0\n  x' = -1000*x\n  y' = -y + x\nend",
         {"s": _sim("explicitODESimulation", ["x", "y"], useStiffSolver=True, independentVariableRange={
             "_type": "numericRange", "values": [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 5.0]})},
         {"r": "#tasks:s"}, {"r": table(np.array([0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 5.0]), x=xs,
                                        y=ys)},
         "As the stiff pair above: x = exp(-1000 t), y = (exp(-t) - exp(-1000 t))/999.")
    case("Solver settings: a maximum internal step size.",
         DECAY, {"s": explicit(["S"], 0, 10, 20, maxInternalStepSize=0.1)},
         {"r": "#tasks:s"}, {"r": table(T, S=S_decay)}, "S(t) = 10 exp(-t/2) as before.")
    case("Solver settings: an initial step size.",
         DECAY, {"s": explicit(["S"], 0, 10, 20, initialStepSize=0.01)},
         {"r": "#tasks:s"}, {"r": table(T, S=S_decay)}, "S(t) = 10 exp(-t/2) as before.")
