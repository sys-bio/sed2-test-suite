"""Series "fba": FluxBalanceAnalysis against hand-solved linear programs, and a simplified ODE + FBA cosimulation.

The FBA models are small SBML models with the FBC package, written here by `fbc_model` (not exported from a tool, so
that every id is known): every reaction has its own two bound parameters, called `<reaction id>_lower_bound` and
`<reaction id>_upper_bound`, so a ModelChange can change one bound without changing another.  Reaction ids start with
`R_` and species ids with `M_`, as in the SBML files that COBRApy writes.

The toy network ("overflow"):

    R_EX_glc   M_glc_e ->                 uptake of glucose when negative; bounds -10 .. 1000
    R_GLCt     M_glc_e -> M_glc_c         0 .. 1000
    R_RESP     M_glc_c -> 0.024 M_bm      respiration, capacity 6 (upper bound); 0 .. 6
    R_FERM     M_glc_c -> 2 M_ac_c + 0.006 M_bm    fermentation (acetate overflow); 0 .. 1000
    R_EX_ac    M_ac_c ->                  acetate secretion; 0 .. 1000
    R_ATPM     M_glc_c ->                 maintenance drain that must carry at least 0.5; 0.5 .. 1000
    R_BIOMASS  M_bm ->                    biomass drain, the objective; 0 .. 1000

With g units of glucose taken up and a respiration capacity c, g - 0.5 reaches the cytosol after the maintenance drain;
respiration, which yields four times more biomass per glucose, takes min(c, g - 0.5) and fermentation the rest, so the
optimum is unique: biomass = 0.024 min(c, g - 0.5) + 0.006 max(0, g - 0.5 - c).  The values below were also checked
against scipy.optimize.linprog when the series was written.

The cosimulation case uses the shape of templates/cosimulation.sed2.json (a Loop that runs an ODE model for an interval,
hands its state to an FBA model as a bound, runs the FBA, and hands the fluxes back to the ODE model as parameters) with
a one-substrate network small enough to be solved on paper.
"""
import numpy as np

from _util import A, AnnotatedData, write_case, outputs, IMPORT_SBML

FIRST = 274
TIME = "urn:sedml:symbol:time"
FBA_KISAO = "KISAO:0000437"


# ---------------------------------------------------------------------------- SBML with the FBC package

def fbc_model(model_id, species, reactions, objective, coefficient=1.0):
    """The text of an SBML Level 3 Version 1 model with FBC version 2.

    species: list of (id, compartment); reactions: list of (id, reactants, products, lower, upper) where reactants and
    products map a species id to its stoichiometry.  Compartments e and c are always declared."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" '
             'xmlns:fbc="http://www.sbml.org/sbml/level3/version1/fbc/version2" level="3" version="1" fbc:required="false">',
             f'  <model id="{model_id}" fbc:strict="true">',
             '    <listOfCompartments>',
             '      <compartment id="e" constant="true"/>',
             '      <compartment id="c" constant="true"/>',
             '    </listOfCompartments>',
             '    <listOfSpecies>']
    for sid, comp in species:
        lines.append(f'      <species id="{sid}" compartment="{comp}" hasOnlySubstanceUnits="false" '
                     'boundaryCondition="false" constant="false"/>')
    lines += ['    </listOfSpecies>', '    <listOfParameters>']
    for rid, _r, _p, lo, hi in reactions:
        lines.append(f'      <parameter id="{rid}_lower_bound" value="{lo!r}" constant="true"/>')
        lines.append(f'      <parameter id="{rid}_upper_bound" value="{hi!r}" constant="true"/>')
    lines += ['    </listOfParameters>', '    <listOfReactions>']
    for rid, reactants, products, _lo, _hi in reactions:
        lines.append(f'      <reaction id="{rid}" reversible="{"true" if _lo < 0 else "false"}" fast="false" '
                     f'fbc:lowerFluxBound="{rid}_lower_bound" fbc:upperFluxBound="{rid}_upper_bound">')
        for tag, side in (("listOfReactants", reactants), ("listOfProducts", products)):
            if side:
                lines.append(f'        <{tag}>')
                for sid, st in side.items():
                    lines.append(f'          <speciesReference species="{sid}" stoichiometry="{st!r}" constant="true"/>')
                lines.append(f'        </{tag}>')
        lines.append('      </reaction>')
    lines += ['    </listOfReactions>',
              '    <fbc:listOfObjectives fbc:activeObjective="obj">',
              '      <fbc:objective fbc:id="obj" fbc:type="maximize">',
              '        <fbc:listOfFluxObjectives>',
              f'          <fbc:fluxObjective fbc:reaction="{objective}" fbc:coefficient="{coefficient!r}"/>',
              '        </fbc:listOfFluxObjectives>',
              '      </fbc:objective>',
              '    </fbc:listOfObjectives>',
              '  </model>', '</sbml>', '']
    text = "\n".join(lines)
    _check_sbml(text)
    return text


def _check_sbml(text):
    import libsbml

    doc = libsbml.readSBMLFromString(text)
    doc.checkConsistency()
    errors = [doc.getError(i).getMessage() for i in range(doc.getNumErrors()) if doc.getError(i).isError()]
    if errors:
        raise ValueError("invalid SBML: " + "; ".join(errors[:3]))


GLC, RESP_CAPACITY, MAINTENANCE = 10.0, 6.0, 0.5
Y_RESP, Y_FERM = 0.024, 0.006
OVERFLOW_REACTIONS = ["R_EX_glc", "R_GLCt", "R_RESP", "R_FERM", "R_EX_ac", "R_ATPM", "R_BIOMASS"]


def overflow_model():
    return fbc_model(
        "overflow",
        [("M_glc_e", "e"), ("M_glc_c", "c"), ("M_ac_c", "c"), ("M_bm", "c")],
        [("R_EX_glc", {"M_glc_e": 1.0}, {}, -GLC, 1000.0),
         ("R_GLCt", {"M_glc_e": 1.0}, {"M_glc_c": 1.0}, 0.0, 1000.0),
         ("R_RESP", {"M_glc_c": 1.0}, {"M_bm": Y_RESP}, 0.0, RESP_CAPACITY),
         ("R_FERM", {"M_glc_c": 1.0}, {"M_ac_c": 2.0, "M_bm": Y_FERM}, 0.0, 1000.0),
         ("R_EX_ac", {"M_ac_c": 1.0}, {}, 0.0, 1000.0),
         ("R_ATPM", {"M_glc_c": 1.0}, {}, MAINTENANCE, 1000.0),
         ("R_BIOMASS", {"M_bm": 1.0}, {}, 0.0, 1000.0)],
        "R_BIOMASS")


def optimum(glucose, capacity=RESP_CAPACITY):
    """The unique optimal fluxes of the overflow network for `glucose` taken up, as a dict by reaction id."""
    avail = glucose - MAINTENANCE
    resp = min(capacity, avail)
    ferm = avail - resp
    return {"R_EX_glc": -glucose, "R_GLCt": glucose, "R_RESP": resp, "R_FERM": ferm, "R_EX_ac": 2 * ferm,
            "R_ATPM": MAINTENANCE, "R_BIOMASS": Y_RESP * resp + Y_FERM * ferm}


def vec(names, values=None, fluxes=None):
    values = values if values is not None else [fluxes[n] for n in names]
    return A(np.array(values, dtype=float), rows=names)


def text(x):
    x = float(x)
    return str(int(x)) if x == int(x) else repr(x)


def AL(values, labels, dims=None):
    return AnnotatedData(np.array(values, dtype=float), list(labels), list(dims) if dims else None)


# ---------------------------------------------------------------------------- the series

def build():
    counter = iter(range(FIRST, FIRST + 1000))
    model_text = overflow_model()

    def case(what, tasks, refs, expected, derivation, **kw):
        n = next(counter)
        cid = f"{n:05d}"
        all_tasks = {"m": {"_type": "modelImport", "location": f"{cid}.sbml", "language": IMPORT_SBML}}
        all_tasks.update(tasks)
        doc = {"tasks": all_tasks, "outputs": outputs(refs)}
        return write_case(n, what, doc, expected, derivation=derivation, inputs={f"{cid}.sbml": model_text}, **kw)

    def fba(outvars, model="#tasks:m.model", algorithm=None):
        t = {"_type": "fluxBalanceAnalysis", "model": model, "outputVariables": outvars}
        if algorithm:
            t["workingAlgorithms"] = [{"algorithm": algorithm}]
        return t

    ALL = OVERFLOW_REACTIONS
    opt = optimum(GLC)
    DERIV = ("Glucose uptake is at most 10 and the maintenance drain R_ATPM must carry at least 0.5, so 9.5 reach "
             "respiration and fermentation.  Respiration yields 0.024 biomass per glucose and is limited to 6; "
             "fermentation yields 0.006 and takes the remaining 3.5 (and makes 7 acetate).  So R_RESP = 6, R_FERM = 3.5, "
             "R_EX_ac = 7, R_ATPM = 0.5, R_GLCt = 10, R_EX_glc = -10 and the objective R_BIOMASS = 0.024 * 6 + 0.006 * 3.5 = 0.165.  "
             "The optimum is unique: a glucose spent on fermentation instead of respiration gives less biomass.")

    # ---- a single analysis
    case("FluxBalanceAnalysis of a small network with a maintenance drain and an overflow pathway: the fluxes named in "
         "outputVariables, labelled by reaction id.",
         {"f": fba(["R_BIOMASS", "R_RESP", "R_FERM", "R_EX_glc", "R_EX_ac", "R_ATPM"])}, {"r": "#tasks:f"},
         {"r": vec(["R_BIOMASS", "R_RESP", "R_FERM", "R_EX_glc", "R_EX_ac", "R_ATPM"], fluxes=opt)},
         DERIV + "  The result is a vector labelled by the output variables, in the order given.",
         notes="The model is an SBML file with the FBC package; the bounds are the parameters named "
               "`<reaction>_lower_bound` and `<reaction>_upper_bound`.")
    case("FluxBalanceAnalysis with the generic FBA algorithm named (KISAO:0000437), and one flux picked out by its label.",
         {"f": fba(["R_RESP", "R_BIOMASS"], algorithm=FBA_KISAO)}, {"r": "#tasks:f", "b": "#tasks:f['R_BIOMASS']"},
         {"r": vec(["R_RESP", "R_BIOMASS"], fluxes=opt), "b": A(opt["R_BIOMASS"])},
         DERIV)
    case("FluxBalanceAnalysis whose output variables are all the reactions of the model, taken from a ModelElementList.",
         {"l": {"_type": "modelElementList", "model": "#tasks:m.model", "includeTypes": ["reaction"]},
          "f": fba("#tasks:l.strings")}, {"r": "#tasks:f"},
         {"r": vec(ALL, fluxes=opt)},
         DERIV + "  The reactions are listed in the order of the model.")
    case("FluxBalanceAnalysis can report a model parameter: the lower bound of the glucose exchange, together with the biomass flux.",
         {"f": fba(["R_BIOMASS", "R_EX_glc_lower_bound"])}, {"r": "#tasks:f"},
         {"r": vec(["R_BIOMASS", "R_EX_glc_lower_bound"], [opt["R_BIOMASS"], -GLC])},
         DERIV + "  The parameter R_EX_glc_lower_bound is -10; a variable that is not a reaction has the model's value.")

    # ---- changing bounds
    g5 = optimum(5.0)
    case("A ModelChange sets the glucose uptake bound to -5 before the FluxBalanceAnalysis.",
         {"c": {"_type": "modelChange", "inputModel": "#tasks:m.model", "setValues": {"R_EX_glc_lower_bound": -5}},
          "f": fba(["R_BIOMASS", "R_RESP", "R_FERM"], model="#tasks:c.model")}, {"r": "#tasks:f"},
         {"r": vec(["R_BIOMASS", "R_RESP", "R_FERM"], fluxes=g5)},
         "With 5 glucose, 4.5 reach the network after maintenance; that is below the respiration capacity of 6, so all of it "
         "is respired: R_RESP = 4.5, R_FERM = 0, R_BIOMASS = 0.024 * 4.5 = 0.108.")
    scan = [10.0, 5.0, 2.0, 1.0]
    case("A Scatter over the glucose uptake bound: -10, -5, -2, -1, each followed by a FluxBalanceAnalysis.",
         {"sc": {"_type": "scatter", "range": {"_type": "range", "values": [-g for g in scan]},
                 "subTasks": {"c": {"_type": "modelChange", "inputModel": "#tasks:m.model",
                                    "setValues": {"R_EX_glc_lower_bound": "#tasks:sc.range"}},
                              "f": fba(["R_BIOMASS", "R_EX_ac"], model="#tasks:sc:subTasks:c.model")},
                 "outputVariableMap": {"growth": "#tasks:sc:subTasks:f['R_BIOMASS']", "acetate": "#tasks:sc:subTasks:f['R_EX_ac']"}}},
         {"r": "#tasks:sc"},
         {"r": AL([[optimum(g)["R_BIOMASS"], optimum(g)["R_EX_ac"]] for g in scan], [[text(-g) for g in scan], ["growth", "acetate"]])},
         "biomass = 0.024 min(6, g - 0.5) + 0.006 max(0, g - 0.5 - 6) for g glucose: g = 10: 0.165 (7 acetate); g = 5: 0.108; "
         "g = 2: 0.024 * 1.5 = 0.036; g = 1: 0.024 * 0.5 = 0.012; below 6.5 glucose there is no fermentation and no acetate.")
    grid_g, grid_c = [10.0, 5.0], [2.0, 6.0]
    values = [[[optimum(g, c)["R_BIOMASS"], optimum(g, c)["R_FERM"]] for c in grid_c] for g in grid_g]
    case("A ParameterScan of two bounds of the network, the glucose uptake and the respiration capacity, with a FluxBalanceAnalysis for each pair.",
         {"ps": {"_type": "parameterScan", "model": "#tasks:m.model",
                 "parameterRanges": [{"_type": "parameterRange", "modelElement": "R_EX_glc_lower_bound", "values": [-g for g in grid_g]},
                                     {"_type": "parameterRange", "modelElement": "R_RESP_upper_bound", "values": grid_c}],
                 "subTasks": {"c": {"_type": "modelChange", "inputModel": "#tasks:m.model",
                                    "setValues": {"R_EX_glc_lower_bound": "#tasks:ps.ranges['R_EX_glc_lower_bound']",
                                                  "R_RESP_upper_bound": "#tasks:ps.ranges['R_RESP_upper_bound']"}},
                              "f": fba(["R_BIOMASS", "R_FERM"], model="#tasks:ps:subTasks:c.model")},
                 "outputVariableMap": {"growth": "#tasks:ps:subTasks:f['R_BIOMASS']", "fermentation": "#tasks:ps:subTasks:f['R_FERM']"}}},
         {"r": "#tasks:ps"},
         {"r": AL(values, [[text(-g) for g in grid_g], [text(c) for c in grid_c], ["growth", "fermentation"]],
                  ["R_EX_glc_lower_bound", "R_RESP_upper_bound", ""])},
         "With avail = g - 0.5: respiration takes min(c, avail) and fermentation the rest.  (g, c) = (10, 2): resp 2, ferm 7.5, "
         "biomass 0.048 + 0.045 = 0.093; (10, 6): 0.165 with ferm 3.5; (5, 2): resp 2, ferm 2.5, biomass 0.048 + 0.015 = 0.063; "
         "(5, 6): resp 4.5, ferm 0, biomass 0.108.  The result has one dimension per parameter range, then the two entries.")
    halves = [10.0, 5.0, 2.5, 1.25]
    case("A Loop whose loop variable is the glucose uptake bound, halved on every iteration, with a FluxBalanceAnalysis in each.",
         {"lp": {"_type": "loop", "range": {"_type": "numericRange", "start": 1, "numberOfSteps": len(halves) - 1, "interval": 1},
                 "loopVariables": {"bound": {"initialValue": -10.0, "subsequentValues": "#tasks:lp:subTasks:half"}},
                 "subTasks": {"c": {"_type": "modelChange", "inputModel": "#tasks:m.model",
                                    "setValues": {"R_EX_glc_lower_bound": "#tasks:lp:loopVariables:bound"}},
                              "f": fba(["R_BIOMASS"], model="#tasks:lp:subTasks:c.model"),
                              "half": {"_type": "calculation", "math": "#tasks:lp:loopVariables:bound * 0.5"}},
                 "outputVariableMap": {"growth": "#tasks:lp:subTasks:f['R_BIOMASS']"}}},
         {"r": "#tasks:lp"},
         {"r": AL([[optimum(g)["R_BIOMASS"]] for g in halves], [[str(i + 1) for i in range(len(halves))], ["growth"]])},
         "The bound is -10, -5, -2.5, -1.25 on the four iterations.  biomass = 0.024 min(6, g - 0.5) + 0.006 max(0, g - 0.5 - 6): "
         "0.165, 0.108, 0.024 * 2 = 0.048 and 0.024 * 0.75 = 0.018.")
    cosimulation(next(counter))


# ---------------------------------------------------------------------------- ODE + FBA

COSIM_ODE = """model monod
  species S, B
  S = 10
  B = 0.02
  Vmax = 5
  Km = 2
  # the uptake and growth rates, set from the FBA solution between intervals; they start at the values for S = 10
  R_EX_S = -4.166666666666667
  R_BIOMASS = 0.4166666666666667
  # the uptake limit the FBA model is given: Monod kinetics at the current substrate concentration
  R_EX_S_lower_bound := -Vmax*S/(Km + S)
  uptake: S => ; -R_EX_S*B
  growth: => B; R_BIOMASS*B
end"""


def cosim_model():
    return fbc_model(
        "growth_fba",
        [("M_S_e", "e"), ("M_bm", "c")],
        [("R_EX_S", {"M_S_e": 1.0}, {}, -4.0, 1000.0),
         ("R_GROW", {"M_S_e": 1.0}, {"M_bm": 0.1}, 0.0, 1000.0),
         ("R_BIOMASS", {"M_bm": 1.0}, {}, 0.0, 1000.0)],
        "R_BIOMASS")


def cosim_trace(steps, points, interval, s0, b0, vmax, km, yield_):
    """Closed-form trace of the loop: for each iteration the rows (time, S, B, bound) at the output points."""
    s, b = s0, b0
    u = vmax * s / (km + s)            # the initial uptake and growth rates are those of S = s0
    out = []
    for _ in range(steps):
        mu = yield_ * u
        rows = []
        for k in range(points + 1):
            t = interval * k / points
            growth = np.exp(mu * t)
            bt = b * growth
            st = s - u * b * (growth - 1.0) / mu
            rows.append([t, st, bt, -vmax * st / (km + st)])
        out.append(rows)
        s, b = rows[-1][1], rows[-1][2]
        u = vmax * s / (km + s)        # the FBA: uptake at its bound, growth is the yield times the uptake
    return out


def cosimulation(n):
    cid = f"{n:05d}"
    steps, points, interval = 8, 4, 1.0
    trace = cosim_trace(steps, points, interval, 10.0, 0.02, 5.0, 2.0, 0.1)
    assert min(r[1] for it in trace for r in it) > 0.5, "the substrate must not run out"
    doc = {
        "constants": {"ode_interval": interval},
        "tasks": {
            "ODEmodel": {"_type": "modelImport", "location": f"{cid}.ode.sbml", "language": IMPORT_SBML},
            "FBAmodel": {"_type": "modelImport", "location": f"{cid}.fba.sbml", "language": IMPORT_SBML},
            "repeat": {
                "_type": "loop",
                "range": {"_type": "numericRange", "start": 1, "numberOfSteps": steps - 1, "interval": 1},
                "loopVariables": {
                    "ODE_model": {"initialValue": "#tasks:ODEmodel.model", "subsequentValues": "#tasks:repeat:subTasks:new_ode.model"},
                    "FBA_model": {"initialValue": "#tasks:FBAmodel.model", "subsequentValues": "#tasks:repeat:subTasks:fbasim.model"},
                },
                "subTasks": {
                    "odesim": {
                        "_type": "explicitODESimulation",
                        "workingAlgorithms": [{"algorithm": "KISAO:0000694"}],
                        "model": "#tasks:repeat:loopVariables:ODE_model",
                        "independentVariable": TIME,
                        "independentVariableInit": 0,
                        "independentVariableRange": {"_type": "numericRange", "start": 0, "end": "#constants:ode_interval",
                                                     "numberOfSteps": points, "scale": "linear"},
                        "outputVariables": ["S", "B", "R_EX_S_lower_bound"],
                    },
                    "new_fba": {
                        "_type": "modelChange",
                        "inputModel": "#tasks:repeat:loopVariables:FBA_model",
                        "setValues": {"R_EX_S_lower_bound": "#tasks:repeat:subTasks:odesim[-1, 'R_EX_S_lower_bound']"},
                    },
                    "fbasim": {
                        "_type": "fluxBalanceAnalysis",
                        "workingAlgorithms": [{"algorithm": FBA_KISAO}],
                        "model": "#tasks:repeat:subTasks:new_fba.model",
                        "outputVariables": ["R_EX_S", "R_BIOMASS"],
                    },
                    "new_ode": {
                        "_type": "modelChange",
                        "inputModel": "#tasks:repeat:subTasks:odesim.model",
                        "setValues": {"R_EX_S": "#tasks:repeat:subTasks:fbasim['R_EX_S']",
                                      "R_BIOMASS": "#tasks:repeat:subTasks:fbasim['R_BIOMASS']"},
                    },
                },
                "outputVariableMap": {"species_trace": "#tasks:repeat:subTasks:odesim"},
            },
        },
        "outputs": {"cosim_output": {"_type": "report", "data": "#tasks:repeat[:, 'species_trace']"}},
    }
    expected = AL(trace, [[str(i + 1) for i in range(steps)], None, [TIME, "S", "B", "R_EX_S_lower_bound"]])
    write_case(
        n,
        "Cosimulation of an ODE model and an FBA model in a Loop (the shape of templates/cosimulation.sed2.json): each iteration "
        "runs the ODE model for one time unit, gives its uptake limit to the FBA model as a bound, runs the FBA, and gives the "
        "fluxes back to the ODE model as its uptake and growth rates.  The models are a one-substrate Monod culture and a "
        "three-reaction growth network, much simplified from the CRM-FBA example.",
        doc, {"cosim_output": expected},
        derivation=("ODE model: dS/dt = R_EX_S B and dB/dt = R_BIOMASS B, with R_EX_S < 0 the uptake flux and R_BIOMASS the growth "
                    "rate; R_EX_S_lower_bound = -Vmax S/(Km + S) (Vmax 5, Km 2) is the Monod uptake limit at the current S.  "
                    "FBA model: R_EX_S >= bound, R_GROW makes 0.1 biomass per substrate and R_BIOMASS drains it, so the optimum "
                    "is R_EX_S = bound (the uptake limit) and R_BIOMASS = 0.1 * |bound|.  Over one interval u = -R_EX_S and "
                    "mu = R_BIOMASS are constant, so B(t) = B0 exp(mu t) and S(t) = S0 - u B0 (exp(mu t) - 1)/mu.  The first "
                    "interval uses the rates for S = 10 (u = 4.1667, mu = 0.41667, the initial values of the ODE model); at the "
                    "end of every interval the FBA sets u = Vmax S/(Km + S) for the S reached and mu = 0.1 u, which the next "
                    "interval uses.  The expected data is that recurrence, 8 intervals with 4 output steps each (5 rows, the "
                    "first being the start of the interval).  The result has the iteration, the output row and the four "
                    "columns time, S, B, R_EX_S_lower_bound."),
        antimony={"ode": COSIM_ODE},
        inputs={f"{cid}.fba.sbml": cosim_model()},
        notes=("The FluxBalanceAnalysis runs on cobra whichever ODE backend is chosen; the Backends line below names the ODE backend. "
               "The bound of R_EX_S in the FBA model starts at -4 and is replaced by the ODE model's value in every iteration. "
               "The report takes the entry by its label in dimension 1 (`[:, 'species_trace']`); the template writes "
               "`repeat['species_trace']`, which would index the iterations (SED2/TODO.md). "
               "In templates/cosimulation.sed2.json the ModelChange takes the whole last row of the ODE result; that row also has "
               "a time column, which is not an element of the FBA model, so this case names the one element it hands over."),
    )
