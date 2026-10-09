"""Series "dataplots": CsvImport and the data behind Plot2D and Plot3D (stored as `*_as_data.csv` / `*_as_data.h5`).

Only the data a plot is drawn from is compared (docs/FORMATS.md): the columns of a Plot2D (`curveId.field`, in the
order of the curves' `order`, ties by position in the dictionary) and the x, y and z arrays of each Plot3D surface.
CsvImport is tested for what is clear in its description: a 2-D numeric table, `separator`, `headers`, `columnNames`,
`nrows` and `ncols`; the open points (the default for `headers`, a header row together with `columnNames`, `organization`)
are in SED2/TODO.md and not tested.
"""
import numpy as np

from _util import A, AnnotatedData, write_case, outputs, IMPORT_SBML
from sed2suite.results_io import Plot2DData, Plot3DData, Surface

FIRST = 239
TIME = "urn:sedml:symbol:time"
NAN, INF = float("nan"), float("inf")

DECAY = "model dec\n  compartment c = 1\n  species S in c\n  S = 3\n  S -> ; k1*S\n  k1 = 0.5\nend"


def curve(x, y, **kw):
    c = {"_type": "curve", "curveType": "points", "x": x, "y": y}
    c.update(kw)
    return c


def surface(kind, x, y, z, **kw):
    s = {"surfaceType": kind, "x": x, "y": y, "z": z}
    s.update(kw)
    return s


def build():
    counter = iter(range(FIRST, FIRST + 1000))

    def case(what, doc_outputs, expected_plots, derivation, constants=None, tasks=None, ant=None, refs=None,
             expected=None, inputs=None, **kw):
        n = next(counter)
        cid = f"{n:05d}"
        all_tasks = {}
        if ant:
            all_tasks["m"] = {"_type": "modelImport", "location": f"{cid}.sbml", "language": IMPORT_SBML}
        for name, task in (tasks or {}).items():
            all_tasks[name] = task
        doc = {}
        if constants:
            doc["constants"] = constants
        if all_tasks:
            doc["tasks"] = all_tasks
        outs = dict(outputs(refs or {}))
        outs.update(doc_outputs)
        doc["outputs"] = outs
        if inputs:
            inputs = {f"{cid}.{name}": text for name, text in inputs.items()}
        return write_case(n, what, doc, expected, plots=expected_plots, derivation=derivation, inputs=inputs,
                          antimony={"": ant} if ant else None, **kw)

    def p2(curves, **kw):
        return {"_type": "plot2D", "curves": curves, **kw}

    def p3(surfaces, **kw):
        return {"_type": "plot3D", "surfaces": surfaces, **kw}

    X4, Y4 = [1, 2, 3, 4], [10, 20, 30, 40]

    # ------------------------------------------------------------------ Plot2D
    case("Plot2D with one curve: the data are the x and y columns, named for the curve.",
         {"p": p2({"c": curve("#constants:x", "#constants:y")})},
         {"p": Plot2DData({"c.x": [1.0, 2.0, 3.0, 4.0], "c.y": [10.0, 20.0, 30.0, 40.0]})},
         "The curve c gives the columns c.x and c.y.  The report r is y, so that the case has a report.", constants={"x": X4, "y": Y4},
         refs={"r": "#constants:y"}, expected={"r": A(Y4)})
    case("Plot2D: the curve type, axes, legend and size are not part of the data.",
         {"p": p2({"c": curve("#constants:x", "#constants:y", curveType="bar", yAxis="right"), },
                  legend=True, width=300, height=200, xAxis={"scale": "log10", "min": 0.5, "max": 5},
                  yAxis={"min": 0}, rightYAxis={"max": 100})},
         {"p": Plot2DData({"c.x": [1.0, 2.0, 3.0, 4.0], "c.y": [10.0, 20.0, 30.0, 40.0]})},
         "A bar curve on the right axis, with styled axes, still stores just c.x and c.y.", constants={"x": X4, "y": Y4},
         refs={"r": "#constants:x"}, expected={"r": A(X4)})
    case("Plot2D: curves are ordered by their `order`; equal orders keep the order of the dictionary.",
         {"p": p2({"late": curve("#constants:x", "#constants:y", order=2),
                   "first": curve("#constants:x", "#constants:x", order=1),
                   "tied": curve("#constants:y", "#constants:y", order=1)})},
         {"p": Plot2DData({"first.x": [1.0, 2.0, 3.0, 4.0], "first.y": [1.0, 2.0, 3.0, 4.0],
                           "tied.x": [10.0, 20.0, 30.0, 40.0], "tied.y": [10.0, 20.0, 30.0, 40.0],
                           "late.x": [1.0, 2.0, 3.0, 4.0], "late.y": [10.0, 20.0, 30.0, 40.0]})},
         "first (order 1), tied (order 1, after first in the dictionary), then late (order 2), although late is written "
         "first.", constants={"x": X4, "y": Y4},
         refs={"r": "#constants:x"}, expected={"r": A(X4)})
    case("Plot2D: every data field of a curve, in the fixed order x, y, xErrorLower, xErrorUpper, yErrorLower, yErrorUpper, yFrom, yTo.",
         {"p": p2({"c": curve("#constants:x", "#constants:y", yTo="#constants:hi", yFrom="#constants:lo",
                              yErrorUpper="#constants:e2", yErrorLower="#constants:e1",
                              xErrorUpper="#constants:e2", xErrorLower="#constants:e1")})},
         {"p": Plot2DData({"c.x": [1.0, 2.0], "c.y": [3.0, 4.0], "c.xErrorLower": [0.1, 0.1], "c.xErrorUpper": [0.2, 0.2],
                           "c.yErrorLower": [0.1, 0.1], "c.yErrorUpper": [0.2, 0.2], "c.yFrom": [0.0, 0.0],
                           "c.yTo": [9.0, 9.0]})},
         "The columns follow the fixed field order whatever the order the attributes are written in.",
         constants={"x": [1, 2], "y": [3, 4], "e1": [0.1, 0.1], "e2": [0.2, 0.2], "lo": [0, 0], "hi": [9, 9]},
         refs={"r": "#constants:y"}, expected={"r": A([3.0, 4.0])})
    case("Plot2D: curves of different lengths; the shorter columns are padded with empty cells.",
         {"p": p2({"a": curve("#constants:x", "#constants:y"), "b": curve("#constants:x2", "#constants:y2")})},
         {"p": Plot2DData({"a.x": [1.0, 2.0, 3.0, 4.0], "a.y": [10.0, 20.0, 30.0, 40.0],
                           "b.x": [5.0, 6.0, None, None], "b.y": [50.0, 60.0, None, None]})},
         "Curve b has two points; its columns have two empty cells.",
         constants={"x": X4, "y": Y4, "x2": [5, 6], "y2": [50, 60]},
         refs={"r": "#constants:y2"}, expected={"r": A([50.0, 60.0])})
    case("Plot2D: a single number counts as a column of length 1.",
         {"p": p2({"c": curve("#constants:x", "#constants:one")})},
         {"p": Plot2DData({"c.x": [1.0, 2.0, 3.0, 4.0], "c.y": [7.0, None, None, None]})},
         "y is the number 7, one value, followed by empty cells.", constants={"x": X4, "one": 7}, refs={"r": "#constants:one"}, expected={"r": A(7.0)})
    case("Plot2D: text x values (categories) are kept as text.",
         {"p": p2({"c": curve("#constants:cats", "#constants:y3", curveType="bar")})},
         {"p": Plot2DData({"c.x": ["a", "b", "c"], "c.y": [1.0, 2.0, 3.0]})},
         "The categories a, b, c with the values 1, 2, 3.", constants={"cats": ["a", "b", "c"], "y3": [1, 2, 3]},
         refs={"r": "#constants:cats"}, expected={"r": A(["a", "b", "c"])})
    case("Plot2D: NaN and infinity in the data are kept.",
         {"p": p2({"c": curve("#constants:x3", "#constants:y")})},
         {"p": Plot2DData({"c.x": [1.0, 2.0, 3.0], "c.y": [1.0, NAN, INF]})},
         "nan is a real NaN, not an empty cell; inf is +infinity.", constants={"x3": [1, 2, 3], "y": [1, "nan", "inf"]},
         refs={"r": "#constants:y"}, expected={"r": A([1.0, NAN, INF])})
    case("Plot2D of a time course: the curve's x and y are columns of the simulation result.",
         {"p": p2({"c": curve("#tasks:s[:, 0]", "#tasks:s[:, 'S']")})},
         {"p": Plot2DData({"c.x": [0.0, 1.0, 2.0, 3.0, 4.0],
                           "c.y": [float(3 * np.exp(-0.5 * t)) for t in (0, 1, 2, 3, 4)]})},
         "S(t) = 3 exp(-t/2) at t = 0..4; column 0 of the table is time and the column S is picked by label.",
         ant=DECAY, refs={"r": "#tasks:s"},
         expected={"r": A([[float(t), float(3 * np.exp(-0.5 * t))] for t in (0, 1, 2, 3, 4)], cols=[TIME, "S"])},
         tasks={"s": {"_type": "explicitODESimulation", "model": "#tasks:m.model", "independentVariable": TIME,
                                 "outputVariables": ["S"],
                                 "independentVariableRange": {"_type": "numericRange", "start": 0, "end": 4,
                                                              "numberOfSteps": 4}}})
    case("Plot2D of a parameter scan together with a report of the same data.",
         {"p": p2({"c": curve("#constants:ks", "#tasks:ps[:, 'S']")})},
         {"p": Plot2DData({"c.x": [0.25, 0.5, 1.0], "c.y": [float(3 * np.exp(-k)) for k in (0.25, 0.5, 1.0)]})},
         "S(1) = 3 exp(-k1) for k1 = 0.25, 0.5, 1.  The report r is the scan result itself.",
         constants={"ks": [0.25, 0.5, 1]}, ant=DECAY,
         tasks={"ps": {"_type": "parameterScan", "model": "#tasks:m.model",
                       "parameterRanges": [{"_type": "parameterRange", "modelElement": "k1", "values": [0.25, 0.5, 1]}],
                       "subTasks": {"s": {"_type": "oneStepODESimulation", "model": "#tasks:ps.model",
                                          "independentVariable": TIME, "outputVariables": ["S"], "independentStep": 1}},
                       "outputVariableMap": {"S": "#tasks:ps:subTasks:s['S']"}}},
         refs={"r": "#tasks:ps"},
         expected={"r": A([[float(3 * np.exp(-k))] for k in (0.25, 0.5, 1.0)], rows=["0.25", "0.5", "1"], cols=["S"])})

    # ------------------------------------------------------------------ Plot3D
    gx, gy, gz = [0, 1, 2], [0, 1], [[1, 2, 3], [4, 5, 6]]
    case("Plot3D with one surface: x, y and z are stored as the references resolve.",
         {"q": p3({"s": surface("heatMap", "#constants:gx", "#constants:gy", "#constants:gz")})},
         {"q": Plot3DData({"s": Surface(np.array(gx, float), np.array(gy, float), np.array(gz, float), "heatMap", 0)})},
         "A heat map of a 2 x 3 grid: x has 3 values, y has 2, z has 2 rows of 3.",
         constants={"gx": gx, "gy": gy, "gz": gz}, refs={"r": "#constants:gz"}, expected={"r": A(gz)})
    case("Plot3D: surfaces are ordered by `order` and each records its position in the dictionary and its type.",
         {"q": p3({"b": surface("heatMap", "#constants:gx", "#constants:gy", "#constants:gz", order=2),
                   "a": surface("surfaceMesh", "#constants:gx", "#constants:gy", "#constants:gz", order=1),
                   "c": surface("parametricCurve", "#constants:x", "#constants:y", "#constants:e", order=1)},
                  legend=True)},
         {"q": Plot3DData({"a": Surface(np.array(gx, float), np.array(gy, float), np.array(gz, float), "surfaceMesh", 1),
                           "c": Surface(np.array([1, 2, 3, 4], float), np.array(Y4, float), np.array([0.5] * 4),
                                        "parametricCurve", 2),
                           "b": Surface(np.array(gx, float), np.array(gy, float), np.array(gz, float), "heatMap", 0)})},
         "a (order 1, second in the dictionary), c (order 1, third) and b (order 2, first): a and c tie on the order, "
         "so the dictionary position decides.  The stored index is the position in the dictionary: a 1, c 2, b 0.  "
         "The parametric curve c has three vectors of the same length.",
         constants={"gx": gx, "gy": gy, "gz": gz, "x": X4, "y": Y4, "e": [0.5] * 4},
         refs={"r": "#constants:gz"}, expected={"r": A(gz)})
    case("Plot3D of a two-parameter scan: S(2) over S(0) (rows) and k1 (columns).",
         {"q": p3({"s": surface("surfaceMesh", "#constants:ks", "#constants:s0s", "#tasks:ps[:, :, 'S']")})},
         {"q": Plot3DData({"s": Surface(np.array([0.25, 0.5, 0.75]), np.array([1.0, 2.0]),
                                        np.array([[s * np.exp(-2 * k) for k in (0.25, 0.5, 0.75)] for s in (1.0, 2.0)]),
                                        "surfaceMesh", 0)})},
         "z[i][j] = S0_i exp(-2 k1_j): the scan has S0 first (2 values) and k1 second (3 values), so the slice "
         "[:, :, 'S'] is 2 x 3 like the grid, and keeps the labels of the two ranges (the report r).",
         constants={"ks": [0.25, 0.5, 0.75], "s0s": [1, 2]}, ant=DECAY, refs={"r": "#tasks:ps[:, :, 'S']"},
         expected={"r": A([[float(s * np.exp(-2 * k)) for k in (0.25, 0.5, 0.75)] for s in (1.0, 2.0)],
                          rows=["1", "2"], cols=["0.25", "0.5", "0.75"])},
         tasks={"ps": {"_type": "parameterScan", "model": "#tasks:m.model",
                       "parameterRanges": [{"_type": "parameterRange", "modelElement": "S", "values": [1, 2]},
                                           {"_type": "parameterRange", "modelElement": "k1", "values": [0.25, 0.5, 0.75]}],
                       "subTasks": {"s": {"_type": "oneStepODESimulation", "model": "#tasks:ps.model",
                                          "independentVariable": TIME, "outputVariables": ["S"], "independentStep": 2}},
                       "outputVariableMap": {"S": "#tasks:ps:subTasks:s['S']"}}})

    # ------------------------------------------------------------------ CsvImport
    # numbers are assigned in order; the csv cases follow the plots
    def csv(what, task, refs, expected, derivation, text, constants=None, extra_tasks=None, **kw):
        n = next(counter)
        cid = f"{n:05d}"
        task = dict(task, location=f"{cid}.d.csv")
        tasks = {"d": task}
        tasks.update(extra_tasks or {})
        doc = {}
        if constants:
            doc["constants"] = constants
        doc["tasks"] = tasks
        doc["outputs"] = outputs(refs)
        return write_case(n, what, doc, expected, derivation=derivation, inputs={f"{cid}.d.csv": text}, **kw)

    def csvtype(**kw):
        return {"_type": "csvImport", **kw}

    csv("CsvImport of plain numbers: a 2 x 3 table without labels.", csvtype(), {"r": "#tasks:d"},
        {"r": A([[1, 2, 3], [4, 5, 6]])}, "Two lines of three numbers: a 2 x 3 table.", "1,2,3\n4,5,6\n")
    csv("CsvImport: numbers written in several forms (decimal, exponent, negative, no leading digit).", csvtype(),
        {"r": "#tasks:d"}, {"r": A([[0.001, -250.0], [0.5, 12.0]])},
        "1e-3 = 0.001, -2.5e2 = -250, .5 = 0.5, 12.", "1e-3,-2.5e2\n.5,12\n")
    csv("CsvImport with headers: the first line names the columns.", csvtype(headers=True),
        {"r": "#tasks:d", "s1": "#tasks:d[:, 'S1']", "cell": "#tasks:d[1, 'S2']"},
        {"r": A([[0, 1.5, 9], [1, 2.5, 8], [2, 3.5, 7]], cols=["time", "S1", "S2"]),
         "s1": A([1.5, 2.5, 3.5]), "cell": A(8.0)},
        "The header gives the column labels time, S1, S2 and the three data lines are the rows.  Column S1 is "
        "1.5, 2.5, 3.5; row 1 of column S2 is 8.", "time,S1,S2\n0,1.5,9\n1,2.5,8\n2,3.5,7\n")
    csv("CsvImport with columnNames and no header line.", csvtype(columnNames=["u", "v"]),
        {"r": "#tasks:d", "v": "#tasks:d[:, 'v']"},
        {"r": A([[1, 2], [3, 4], [5, 6]], cols=["u", "v"]), "v": A([2.0, 4.0, 6.0])},
        "The columns are named u and v; there are three rows.", "1,2\n3,4\n5,6\n")
    csv("CsvImport with a semicolon as the separator.", csvtype(separator=";"), {"r": "#tasks:d"},
        {"r": A([[1.5, 2], [3, 4.5]])}, "The separator is ';'.", "1.5;2\n3;4.5\n")
    csv("CsvImport with a tab as the separator.", csvtype(separator="\t", headers=True), {"r": "#tasks:d"},
        {"r": A([[1, 2], [3, 4]], cols=["a", "b"])}, "The separator is a tab; the header names the columns a, b.",
        "a\tb\n1\t2\n3\t4\n")
    csv("CsvImport reading only the first two data rows (nrows).", csvtype(headers=True, nrows=2), {"r": "#tasks:d"},
        {"r": A([[1, 2, 3], [4, 5, 6]], cols=["a", "b", "c"])}, "nrows = 2 reads the first two lines after the header.",
        "a,b,c\n1,2,3\n4,5,6\n7,8,9\n")
    csv("CsvImport reading only the first two columns (ncols).", csvtype(headers=True, ncols=2), {"r": "#tasks:d"},
        {"r": A([[1, 2], [4, 5], [7, 8]], cols=["a", "b"])}, "ncols = 2 keeps the columns a and b.",
        "a,b,c\n1,2,3\n4,5,6\n7,8,9\n")
    csv("CsvImport: nrows and ncols together.", csvtype(headers=True, ncols=2, nrows=1), {"r": "#tasks:d"},
        {"r": A([[1, 2]], cols=["a", "b"])}, "The first row of the first two columns.", "a,b,c\n1,2,3\n4,5,6\n")
    csv("CsvImport with every option given by a reference to a constant.",
        csvtype(separator="#constants:sep", headers="#constants:hdr", nrows="#constants:n", ncols="#constants:nc"),
        {"r": "#tasks:d"}, {"r": A([[1, 2], [4, 5]], cols=["a", "b"])},
        "Separator ';' (so the header has the columns a, b, c), 2 rows and 2 columns.",
        "a;b;c\n1;2;3\n4;5;6\n7;8;9\n", constants={"sep": ";", "hdr": True, "n": 2, "nc": 2})
    csv("CsvImport followed by calculations on its columns.", csvtype(headers=True),
        {"sum": "#tasks:total", "scaled": "#tasks:scaled"}, {"sum": A([5.0, 7.0, 9.0]), "scaled": A([2.0, 4.0, 6.0])},
        "The columns x = (1, 2, 3) and y = (4, 5, 6): x + y = (5, 7, 9) and 2 x = (2, 4, 6).",
        "x,y\n1,4\n2,5\n3,6\n",
        extra_tasks={"total": {"_type": "calculation", "math": "#tasks:d[:, 'x'] + #tasks:d[:, 'y']"},
                     "scaled": {"_type": "calculation", "math": "#tasks:d[:, 'x'] * 2"}})
    n = next(counter)
    cid = f"{n:05d}"
    write_case(n, "CsvImport used as the data of a Plot2D curve.",
               {"tasks": {"d": {"_type": "csvImport", "location": f"{cid}.d.csv", "headers": True}},
                "outputs": {"p": p2({"c": curve("#tasks:d[:, 'time']", "#tasks:d[:, 'v']")}),
                            "r": {"_type": "report", "data": "#tasks:d"}}},
               {"r": A([[0, 5], [1, 6], [2, 8]], cols=["time", "v"])}, plots={"p": Plot2DData({"c.x": [0.0, 1.0, 2.0], "c.y": [5.0, 6.0, 8.0]})},
               derivation="The curve plots column v against column time: (0, 5), (1, 6), (2, 8).",
               inputs={f"{cid}.d.csv": "time,v\n0,5\n1,6\n2,8\n"})
