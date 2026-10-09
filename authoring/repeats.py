"""Series "repeats": ranges, Scatter, Loop and ParameterScan (and nesting of them) against hand-computed answers.

Result layout of a repeat, as the specification describes it: dimension 0 has one entry per iteration, labelled by the
value of the range written as text ("1", "0.5"); the next dimension has one entry per outputVariableMap key, labelled
by the key; an entry that is not a single number adds its own dimensions after those.  A ParameterScan has one
dimension per parameterRange (named by its modelElement and labelled by the range values), then the entries.
Results with more than two dimensions are stored as HDF5.

Models are small and have closed-form solutions, so the expected values are computed here from the formula.
"""
import json

import numpy as np

from _util import A, AnnotatedData, write_case, outputs, IMPORT_SBML

FIRST = 173
TIME = "urn:sedml:symbol:time"

DECAY = "model dec\n  compartment c = 1\n  species S in c\n  S = 3\n  S -> ; k1*S\n  k1 = 0.5\nend"
TWO = ("model two\n  compartment c = 1\n  species A in c, B in c\n  A = 3\n  B = 0\n  A -> B; k1*A\n  k1 = 0.5\nend")
PARS = "model pars\n  species X\n  X = 1\n  a = 1\n  b = 1\nend"
SOURCE = "model src\n  -> A; v\n  A -> ; k1*A\n  v = 2\n  k1 = 1\n  A = 1\nend"


def AL(values, labels, dims=None):
    """AnnotatedData of any dimension: `labels` has one list (or None) per dimension."""
    return AnnotatedData(np.array(values, dtype=float), list(labels), list(dims) if dims else None)


def text(x):
    """A number as the spec writes a range value into a label."""
    x = float(x)
    return str(int(x)) if x == int(x) else repr(x)


def numeric(start, steps, interval):
    return {"_type": "numericRange", "start": start, "numberOfSteps": steps, "interval": interval}


def calc(math):
    return {"_type": "calculation", "math": math}


def one_step(model, step, outvars=("S",)):
    return {"_type": "oneStepODESimulation", "model": model, "independentVariable": TIME,
            "outputVariables": list(outvars), "independentStep": step}


def change(model, values):
    return {"_type": "modelChange", "inputModel": model, "setValues": values}


def build():
    counter = iter(range(FIRST, FIRST + 1000))

    def case(what, tasks, refs, expected, derivation, ant=None, constants=None, **kw):
        n = next(counter)
        cid = f"{n:05d}"
        all_tasks = {}
        if ant:
            all_tasks["m"] = {"_type": "modelImport", "location": f"{cid}.sbml", "language": IMPORT_SBML}
        all_tasks.update(json.loads(json.dumps(tasks).replace('"MODEL"', '"#tasks:m.model"')))
        doc = {}
        if constants:
            doc["constants"] = constants
        doc["tasks"] = all_tasks
        doc["outputs"] = outputs(refs)
        return write_case(n, what, doc, expected, derivation=derivation,
                          antimony={"": ant} if ant else None, **kw)

    def scatter(rng, subtasks, out_map):
        return {"_type": "scatter", "range": rng, "subTasks": subtasks, "outputVariableMap": out_map}

    def loop(rng, loop_vars, subtasks, out_map):
        return {"_type": "loop", "range": rng, "loopVariables": loop_vars, "subTasks": subtasks,
                "outputVariableMap": out_map}

    def table(values, points, keys):
        return A(np.array(values, dtype=float), rows=[text(p) for p in points], cols=keys)

    # ------------------------------------------------------------------ standalone ranges
    def range_case(what, rng, expected, derivation, **kw):
        return case(what, {"r": rng}, {"r": "#tasks:r"}, {"r": A(expected)}, derivation, **kw)

    range_case("NumericRange: a list of values is used as it is.", {"_type": "numericRange", "values": [5, 6, 7]},
               [5.0, 6.0, 7.0], "The values 5, 6, 7.")
    range_case("NumericRange: only numberOfSteps counts from 0 in steps of 1.",
               {"_type": "numericRange", "numberOfSteps": 3}, [0.0, 1.0, 2.0, 3.0],
               "3 steps of size 1 from 0 give the 4 points 0, 1, 2, 3.")
    range_case("NumericRange: start, end and numberOfSteps, linear scale.",
               {"_type": "numericRange", "start": 0, "end": 1, "numberOfSteps": 4}, [0.0, 0.25, 0.5, 0.75, 1.0],
               "4 equal steps from 0 to 1: 0, 0.25, 0.5, 0.75, 1.")
    range_case("NumericRange: start, end and numberOfSteps with the scale written out as linear.",
               {"_type": "numericRange", "start": 2, "end": 4, "numberOfSteps": 2, "scale": "linear"}, [2.0, 3.0, 4.0],
               "2 equal steps from 2 to 4: 2, 3, 4.")
    range_case("NumericRange: start, end and numberOfSteps, log10 scale.",
               {"_type": "numericRange", "start": 1, "end": 1000, "numberOfSteps": 3, "scale": "log10"},
               [1.0, 10.0, 100.0, 1000.0], "The exponents run 0, 1, 2, 3 in equal steps: 10^0 ... 10^3 = 1, 10, 100, 1000.")
    range_case("NumericRange: start, end and interval; end is the last point when the interval does not divide the span.",
               {"_type": "numericRange", "start": 0, "end": 10, "interval": 3}, [0.0, 3.0, 6.0, 9.0, 10.0],
               "Steps of 3 from 0 reach 9; the span ends at 10, which is included as the last point.")
    range_case("NumericRange: start, end and interval that divides the span exactly.",
               {"_type": "numericRange", "start": 1, "end": 2, "interval": 0.25}, [1.0, 1.25, 1.5, 1.75, 2.0],
               "0.25 divides the span of 1 four times: 1, 1.25, 1.5, 1.75, 2.")
    range_case("NumericRange: start, numberOfSteps and interval.",
               {"_type": "numericRange", "start": 2, "numberOfSteps": 3, "interval": 2}, [2.0, 4.0, 6.0, 8.0],
               "3 steps of 2 from 2: 2, 4, 6, 8.")
    range_case("NumericRange: end, numberOfSteps and interval.",
               {"_type": "numericRange", "end": 10, "numberOfSteps": 4, "interval": 2.5}, [0.0, 2.5, 5.0, 7.5, 10.0],
               "The start is end - numberOfSteps * interval = 10 - 4 * 2.5 = 0, so the points are 0, 2.5, 5, 7.5, 10.")
    range_case("NumericRange: a range that descends, from start 5 to end 1 in 4 steps.",
               {"_type": "numericRange", "start": 5, "end": 1, "numberOfSteps": 4}, [5.0, 4.0, 3.0, 2.0, 1.0],
               "Equal steps of (1 - 5)/4 = -1: 5, 4, 3, 2, 1.")
    range_case("NumericRange: a list with a single value.", {"_type": "numericRange", "values": [7]}, [7.0],
               "One value, 7.  (numberOfSteps must be positive, so a one-point range is written with values.)")
    range_case("NumericRange: the attributes may be references to constants.",
               {"_type": "numericRange", "start": "#constants:lo", "interval": "#constants:h", "numberOfSteps": "#constants:n"},
               [1.0, 1.5, 2.0, 2.5], "Start 1, interval 0.5, 3 steps: 1, 1.5, 2, 2.5.",
               constants={"lo": 1.0, "h": 0.5, "n": 3})
    range_case("Range: the values may be a reference to a list constant.",
               {"_type": "range", "values": "#constants:vals"}, [4.0, 9.0, 16.0], "The constant list 4, 9, 16.",
               constants={"vals": [4, 9, 16]})
    case("Range: a range of strings.", {"r": {"_type": "range", "values": ["a", "b", "c"]}}, {"r": "#tasks:r"},
         {"r": A(["a", "b", "c"])}, "The three strings a, b, c.")
    case("A range task can be indexed like any vector of numbers.",
         {"r": {"_type": "numericRange", "start": 10, "end": 50, "numberOfSteps": 4}},
         {"second": "#tasks:r[1]", "last": "#tasks:r[-1]", "mid": "#tasks:r[1:3]"},
         {"second": A(20.0), "last": A(50.0), "mid": A([20.0, 30.0])},
         "The points are 10, 20, 30, 40, 50.  [1] is 20, [-1] is the last point 50 and [1:3] is the points 1 and 2: 20, 30.")
    case("A calculation can use a range task.",
         {"r": {"_type": "numericRange", "values": [1, 2, 3]}, "c": calc("#tasks:r * 10 + 1")},
         {"c": "#tasks:c"}, {"c": A([11.0, 21.0, 31.0])}, "Element by element: 1*10+1, 2*10+1, 3*10+1.")
    case("ParameterRange as a task is a list of values like any other range.",
         {"r": {"_type": "parameterRange", "modelElement": "k1", "values": [1, 4]}},
         {"r": "#tasks:r"}, {"r": A([1.0, 4.0])}, "The values 1, 4.")

    # ------------------------------------------------------------------ Scatter
    def xy():
        return {"x": calc("#tasks:sc.range * 2 + #tasks:sc.index * 10"), "y": calc("#tasks:sc:subTasks:x + 1")}
    xy_map = {"X": "#tasks:sc:subTasks:x", "Y": "#tasks:sc:subTasks:y"}
    xy_rows = [[2 * r + 10 * i, 2 * r + 10 * i + 1] for i, r in enumerate([0, 1, 2, 3])]

    case("Scatter: two calculations per iteration, using the range value and the iteration index.",
         {"sc": scatter(numeric(0, 3, 1), xy(), xy_map)}, {"r": "#tasks:sc"},
         {"r": table(xy_rows, [0, 1, 2, 3], ["X", "Y"])},
         "Iteration i has range value r = i (0..3): X = 2 r + 10 i and Y = X + 1.  Rows are labelled by the range values "
         "0, 1, 2, 3; the columns by the outputVariableMap keys X, Y.")
    case("Scatter: the result can be indexed by iteration index, by label, by range of iterations and from the end.",
         {"sc": scatter(numeric(0, 3, 1), xy(), xy_map)},
         {"one": "#tasks:sc[2, 'Y']", "col": "#tasks:sc[1:3, 'X']", "row": "#tasks:sc['3']", "last": "#tasks:sc[-1][0]"},
         {"one": A(25.0), "col": A([12.0, 24.0], rows=["1", "2"]), "row": A([36.0, 37.0], rows=["X", "Y"]),
          "last": A(36.0)},
         "As in the first Scatter case X = 0, 12, 24, 36 and Y = 1, 13, 25, 37.  [2, 'Y'] is iteration 2, column Y: 25.  "
         "[1:3, 'X'] is iterations 1 and 2 of X: 12, 24, labelled 1, 2.  ['3'] is the iteration labelled 3 (the range value 3): "
         "36, 37, labelled X, Y.  [-1][0] is the last iteration, entry 0: 36.")
    case("Scatter over a plain Range: the labels are the values, whatever they are.",
         {"sc": scatter({"_type": "range", "values": [10, 20, 30]}, {"x": calc("#tasks:sc.range + 1")},
                        {"X": "#tasks:sc:subTasks:x"})}, {"r": "#tasks:sc"},
         {"r": table([[11], [21], [31]], [10, 20, 30], ["X"])}, "X = value + 1: 11, 21, 31, labelled 10, 20, 30.")
    case("Scatter over fractional range values: labels are the shortest text that reads back the same number.",
         {"sc": scatter(numeric(1, 2, 0.5), {"x": calc("#tasks:sc.range ^ 2")}, {"X": "#tasks:sc:subTasks:x"})},
         {"r": "#tasks:sc"}, {"r": table([[1.0], [2.25], [4.0]], [1, 1.5, 2], ["X"])},
         "The range is 1, 1.5, 2 (labels '1', '1.5', '2'); X = r^2 = 1, 2.25, 4.")
    case("Scatter over a range given only by numberOfSteps: 0, 1, 2.",
         {"sc": scatter({"_type": "numericRange", "numberOfSteps": 2}, {"x": calc("#tasks:sc.range * 3")},
                        {"X": "#tasks:sc:subTasks:x"})}, {"r": "#tasks:sc"},
         {"r": table([[0.0], [3.0], [6.0]], [0, 1, 2], ["X"])}, "X = 3 r = 0, 3, 6.")
    case("Scatter over a log10 range.",
         {"sc": scatter({"_type": "numericRange", "start": 1, "end": 1000, "numberOfSteps": 3, "scale": "log10"},
                        {"x": calc("#tasks:sc.range + 0")}, {"X": "#tasks:sc:subTasks:x"})}, {"r": "#tasks:sc"},
         {"r": table([[1.0], [10.0], [100.0], [1000.0]], [1, 10, 100, 1000], ["X"])},
         "The points are 1, 10, 100, 1000 and X is the range value itself.")
    case("Scatter with a range given by a reference to a list constant.",
         {"sc": scatter({"_type": "range", "values": "#constants:vals"}, {"x": calc("#tasks:sc.range * #tasks:sc.range")},
                        {"X": "#tasks:sc:subTasks:x"})}, {"r": "#tasks:sc"},
         {"r": table([[16.0], [25.0]], [4, 5], ["X"])}, "X = r^2 = 16, 25, labelled 4, 5.", constants={"vals": [4, 5]})
    case("Scatter with a single iteration (a range of one value).",
         {"sc": scatter({"_type": "range", "values": [5]}, {"x": calc("#tasks:sc.range + 1")},
                        {"X": "#tasks:sc:subTasks:x"})}, {"r": "#tasks:sc"},
         {"r": table([[6.0]], [5], ["X"])}, "One value, 5: X = 6.")
    case("Scatter: only one of two sub-task results is put in the output map.",
         {"sc": scatter(numeric(1, 2, 1), {"x": calc("#tasks:sc.range * 2"), "y": calc("#tasks:sc:subTasks:x + 100")},
                        {"Y": "#tasks:sc:subTasks:y"})}, {"r": "#tasks:sc"},
         {"r": table([[102.0], [104.0], [106.0]], [1, 2, 3], ["Y"])},
         "x = 2 r and y = x + 100 = 102, 104, 106; only Y is reported.")
    case("Scatter with no output variables: the result has an entry for each iteration but no columns.",
         {"sc": scatter(numeric(1, 2, 0.5), {"x": calc("1")}, {})}, {"r": "#tasks:sc"},
         {"r": AL(np.zeros((3, 0)), [["1", "1.5", "2"], []])},
         "Three iterations (range 1, 1.5, 2) and an empty outputVariableMap: a 3 x 0 table with the iterations as row labels.")
    case("Scatter: a string result of a sub-task.",
         {"sc": scatter(numeric(0, 1, 1), {"x": {"_type": "stringFormation", "concatenate": ["n=", "#tasks:sc.index"]}},
                        {"X": "#tasks:sc:subTasks:x"})}, {"r": "#tasks:sc"},
         {"r": A([["n=0"], ["n=1"]], rows=["0", "1"], cols=["X"])},
         "The index is 0 then 1, and a whole number is written without a decimal point: 'n=0', 'n=1'.")
    case("Scatter of a parameter sweep: the range value is set into the model, which is simulated for 2 time units.",
         {"sc": scatter(numeric(0.25, 3, 0.25),
                        {"c": change("MODEL", {"k1": "#tasks:sc.range"}), "s": one_step("#tasks:sc:subTasks:c.model", 2)},
                        {"S": "#tasks:sc:subTasks:s['S']"})}, {"r": "#tasks:sc"},
         {"r": table([[3 * np.exp(-2 * k)] for k in (0.25, 0.5, 0.75, 1.0)], [0.25, 0.5, 0.75, 1.0], ["S"])},
         "S' = -k1 S from S = 3 gives S(2) = 3 exp(-2 k1) for k1 = 0.25, 0.5, 0.75, 1.  Each iteration starts from the "
         "original model, since the ModelChange is applied to the imported model, not to the previous result.", ant=DECAY)
    case("Scatter in which each iteration returns a whole time course: the result has the iterations, the entry, and then the "
         "table's own two dimensions.",
         {"sc": scatter(numeric(1, 2, 1),
                        {"c": change("MODEL", {"k1": "#tasks:sc.range"}),
                         "t": {"_type": "explicitODESimulation", "model": "#tasks:sc:subTasks:c.model",
                               "independentVariable": TIME, "outputVariables": ["A", "B"],
                               "independentVariableRange": {"_type": "numericRange", "start": 0, "end": 1,
                                                            "numberOfSteps": 4}}},
                        {"tc": "#tasks:sc:subTasks:t"})}, {"r": "#tasks:sc"},
         {"r": AL([[[[t, 3 * np.exp(-k * t), 3 - 3 * np.exp(-k * t)] for t in np.linspace(0, 1, 5)]] for k in (1, 2, 3)],
                  [["1", "2", "3"], ["tc"], None, [TIME, "A", "B"]])},
         "For k1 = 1, 2, 3: A(t) = 3 exp(-k1 t) and B = 3 - A at t = 0, 0.25, 0.5, 0.75, 1.  The result has dimensions "
         "(iteration, entry, time point, column): 3 x 1 x 5 x 3; the table's columns keep their labels (time, A, B).",
         ant=TWO)

    # ------------------------------------------------------------------ Loop
    case("Loop: a number is carried from one iteration to the next and doubled each time.",
         {"lp": loop(numeric(1, 3, 1), {"acc": {"initialValue": 1.0, "subsequentValues": "#tasks:lp:subTasks:d"}},
                     {"d": calc("#tasks:lp:loopVariables:acc * 2")}, {"D": "#tasks:lp:subTasks:d"})},
         {"r": "#tasks:lp"}, {"r": table([[2.0], [4.0], [8.0], [16.0]], [1, 2, 3, 4], ["D"])},
         "acc starts at 1.  Iteration 1: d = 2, and acc becomes 2.  Then d = 4, 8, 16.")
    case("Loop: a running sum, adding the range value to the loop variable.",
         {"lp": loop(numeric(1, 3, 1), {"a": {"initialValue": 0.0, "subsequentValues": "#tasks:lp:subTasks:d"}},
                     {"d": calc("#tasks:lp:loopVariables:a + #tasks:lp.range")}, {"D": "#tasks:lp:subTasks:d"})},
         {"r": "#tasks:lp"}, {"r": table([[1.0], [3.0], [6.0], [10.0]], [1, 2, 3, 4], ["D"])},
         "a starts at 0 and the range is 1, 2, 3, 4: the sums are 1, 3, 6, 10.")
    case("Loop: the initial value is a constant.",
         {"lp": loop(numeric(1, 3, 1), {"a": {"initialValue": "#constants:x0", "subsequentValues": "#tasks:lp:subTasks:d"}},
                     {"d": calc("#tasks:lp:loopVariables:a - 1")}, {"D": "#tasks:lp:subTasks:d"})},
         {"r": "#tasks:lp"}, {"r": table([[4.0], [3.0], [2.0], [1.0]], [1, 2, 3, 4], ["D"])},
         "a starts at 5; each iteration subtracts 1: 4, 3, 2, 1.", constants={"x0": 5})
    case("Loop with two loop variables updated together: (a, b) -> (b, a + b) from (0, 1).",
         {"lp": loop(numeric(0, 5, 1),
                     {"a": {"initialValue": 0, "subsequentValues": "#tasks:lp:subTasks:nb"},
                      "b": {"initialValue": 1, "subsequentValues": "#tasks:lp:subTasks:nsum"}},
                     {"nb": calc("#tasks:lp:loopVariables:b"), "nsum": calc("#tasks:lp:loopVariables:a + #tasks:lp:loopVariables:b")},
                     {"B": "#tasks:lp:subTasks:nb"})},
         {"r": "#tasks:lp"}, {"r": table([[1.0], [1.0], [2.0], [3.0], [5.0], [8.0]], [0, 1, 2, 3, 4, 5], ["B"])},
         "Both new values are computed from the old pair before either replaces it.  The pairs are (0,1), (1,1), (1,2), "
         "(2,3), (3,5), (5,8); b is reported: 1, 1, 2, 3, 5, 8.")
    case("Loop with a vector as the loop variable: each iteration doubles every element.",
         {"lp": loop(numeric(1, 2, 1), {"v": {"initialValue": "#constants:v0", "subsequentValues": "#tasks:lp:subTasks:d"}},
                     {"d": calc("#tasks:lp:loopVariables:v * 2")}, {"D": "#tasks:lp:subTasks:d"})},
         {"r": "#tasks:lp"}, {"r": AL([[[2, 4]], [[4, 8]], [[8, 16]]], [["1", "2", "3"], ["D"], None])},
         "v starts at (1, 2): the iterations report (2, 4), (4, 8), (8, 16).  The vector adds a third dimension.",
         constants={"v0": [1, 2]})
    case("Loop carrying the model state: each iteration simulates one more time unit from where the last one stopped.",
         {"lp": loop(numeric(1, 3, 1),
                     {"m": {"initialValue": "#tasks:m.model", "subsequentValues": "#tasks:lp:subTasks:s.model"}},
                     {"s": one_step("#tasks:lp:loopVariables:m", 1)}, {"S": "#tasks:lp:subTasks:s['S']"})},
         {"r": "#tasks:lp"}, {"r": table([[3 * np.exp(-0.5 * t)] for t in (1, 2, 3, 4)], [1, 2, 3, 4], ["S"])},
         "S' = -0.5 S from S = 3: after each further time unit S = 3 exp(-0.5 t) for t = 1, 2, 3, 4.", ant=DECAY)
    case("Loop with a number loop variable used as a parameter: k1 doubles each iteration, the model is restarted each time.",
         {"lp": loop(numeric(1, 3, 1), {"k": {"initialValue": 0.25, "subsequentValues": "#tasks:lp:subTasks:nk"}},
                     {"c": change("MODEL", {"k1": "#tasks:lp:loopVariables:k"}),
                      "s": one_step("#tasks:lp:subTasks:c.model", 1), "nk": calc("#tasks:lp:loopVariables:k * 2")},
                     {"S": "#tasks:lp:subTasks:s['S']"})},
         {"r": "#tasks:lp"}, {"r": table([[3 * np.exp(-k)] for k in (0.25, 0.5, 1.0, 2.0)], [1, 2, 3, 4], ["S"])},
         "k1 = 0.25, 0.5, 1, 2 in the four iterations; each simulation starts from the original model and runs 1 time "
         "unit: S = 3 exp(-k1).", ant=DECAY)

    # ------------------------------------------------------------------ ParameterScan
    def scan(ranges, subtasks, out_map, model="MODEL"):
        return {"_type": "parameterScan", "model": model, "parameterRanges": ranges, "subTasks": subtasks,
                "outputVariableMap": out_map}

    def prange(element, **kw):
        return {"_type": "parameterRange", "modelElement": element, **kw}

    ks, s0s = [0.25, 0.5, 0.75], [1.0, 2.0]
    grid = [[[s * np.exp(-2 * k), 10 * i + j] for j, s in enumerate(s0s)] for i, k in enumerate(ks)]
    both = [prange("k1", start=0.25, numberOfSteps=2, interval=0.25), prange("S", values=[1, 2])]

    case("ParameterScan of two parameters: k1 = 0.25, 0.5, 0.75 and S(0) = 1, 2, simulated for 2 time units.  The ModelChange "
         "applies the current values with .ranges.",
         {"ps": scan(both,
                     {"c": change("MODEL", {"k1": "#tasks:ps.ranges['k1']", "S": "#tasks:ps.ranges['S']"}),
                      "s": one_step("#tasks:ps:subTasks:c.model", 2),
                      "idx": calc("#tasks:ps.indexes['k1'] * 10 + #tasks:ps.indexes['S']")},
                     {"S": "#tasks:ps:subTasks:s['S']", "idx": "#tasks:ps:subTasks:idx"})},
         {"r": "#tasks:ps"}, {"r": AL(grid, [["0.25", "0.5", "0.75"], ["1", "2"], ["S", "idx"]], ["k1", "S", ""])},
         "S(2) = S0 exp(-2 k1) for each pair.  idx = 10 * (index of k1) + (index of S).  The result has one dimension per "
         "parameter range, named by its model element and labelled by the range values (the first range varies slowest), "
         "then the two entries: 3 x 2 x 2.  The compartment has volume 1, so setting S sets amount and concentration alike.",
         ant=DECAY)
    case("ParameterScan in which the sub-task takes the scan's own .model, which already has the current values.",
         {"ps": scan(both, {"s": one_step("#tasks:ps.model", 2),
                            "idx": calc("#tasks:ps.indexes['k1'] * 10 + #tasks:ps.indexes['S']")},
                     {"S": "#tasks:ps:subTasks:s['S']", "idx": "#tasks:ps:subTasks:idx"})},
         {"r": "#tasks:ps"}, {"r": AL(grid, [["0.25", "0.5", "0.75"], ["1", "2"], ["S", "idx"]], ["k1", "S", ""])},
         "The same scan as the previous case, without the ModelChange: #tasks:ps.model is the scanned model with the "
         "current value of each range already set.", ant=DECAY)
    case("ParameterScan of one parameter: the result is a table with one row per value.",
         {"ps": scan([prange("k1", values=[0.5, 1, 2])], {"s": one_step("#tasks:ps.model", 1)},
                     {"S": "#tasks:ps:subTasks:s['S']"})},
         {"r": "#tasks:ps"}, {"r": A([[3 * np.exp(-k)] for k in (0.5, 1.0, 2.0)], rows=["0.5", "1", "2"], cols=["S"])},
         "S(1) = 3 exp(-k1) for k1 = 0.5, 1, 2.  One parameter range gives dimension 0 (labelled by the values) and the "
         "entries are dimension 1.", ant=DECAY)
    case("ParameterScan whose values come from a constant list, using a single-parameter slice of the result.",
         {"ps": scan([prange("k1", values="#constants:ks")], {"s": one_step("#tasks:ps.model", 1)},
                     {"S": "#tasks:ps:subTasks:s['S']"})},
         {"all": "#tasks:ps", "second": "#tasks:ps[1, 'S']", "byvalue": "#tasks:ps['0.5']"},
         {"all": A([[3 * np.exp(-k)] for k in (0.25, 0.5)], rows=["0.25", "0.5"], cols=["S"]),
          "second": A(3 * np.exp(-0.5)), "byvalue": A([3 * np.exp(-0.5)], rows=["S"])},
         "k1 = 0.25 and 0.5; S(1) = 3 exp(-k1).  [1, 'S'] is the second value of k1 and entry S; ['0.5'] is the row "
         "labelled 0.5, a vector with the entry S.", ant=DECAY, constants={"ks": [0.25, 0.5]})
    ab = [prange("a", values=[1, 2]), prange("b", values=[10, 20, 30])]
    case("ParameterScan with calculations only: the .ranges and .indexes of the current point.",
         {"ps": scan(ab, {"v": calc("#tasks:ps.ranges['a'] * 100 + #tasks:ps.ranges['b']"),
                          "idx": calc("#tasks:ps.indexes['a'] * 10 + #tasks:ps.indexes['b']")},
                     {"v": "#tasks:ps:subTasks:v", "idx": "#tasks:ps:subTasks:idx"})},
         {"r": "#tasks:ps"},
         {"r": AL([[[100 * a + b, 10 * i + j] for j, b in enumerate([10, 20, 30])] for i, a in enumerate([1, 2])],
                  [["1", "2"], ["10", "20", "30"], ["v", "idx"]], ["a", "b", ""])},
         "v = 100 a + b and idx = 10 i + j (i, j are the indexes of a and b): v = 110, 120, 130 for a = 1 and 210, 220, "
         "230 for a = 2.  The result is 2 x 3 x 2 with dimensions named a, b and unnamed.", ant=PARS)
    case("ParameterScan over a log10 range.",
         {"ps": scan([prange("a", start=1, end=1000, numberOfSteps=3, scale="log10")],
                     {"v": calc("#tasks:ps.ranges['a'] * 2")}, {"v": "#tasks:ps:subTasks:v"})},
         {"r": "#tasks:ps"}, {"r": A([[2.0], [20.0], [200.0], [2000.0]], rows=["1", "10", "100", "1000"], cols=["v"])},
         "a = 1, 10, 100, 1000 and v = 2 a.", ant=PARS)
    case("ParameterScan of a steady state: the production rate v and the removal constant k1 of -> A -> .",
         {"ps": scan([prange("v", values=[2, 4, 6]), prange("k1", values=[1, 2])],
                     {"s": {"_type": "steadyState", "model": "#tasks:ps.model", "outputVariables": ["A"]}},
                     {"A": "#tasks:ps:subTasks:s['A']"})},
         {"r": "#tasks:ps"},
         {"r": AL([[[v / k] for k in (1, 2)] for v in (2, 4, 6)], [["2", "4", "6"], ["1", "2"], ["A"]], ["v", "k1", ""])},
         "dA/dt = v - k1 A = 0 gives A = v/k1: 2, 1 for v = 2; 4, 2 for v = 4; 6, 3 for v = 6.", ant=SOURCE)
    case("Indexing a ParameterScan result: a slice along the first parameter.",
         {"ps": scan(both, {"s": one_step("#tasks:ps.model", 2),
                            "idx": calc("#tasks:ps.indexes['k1'] * 10 + #tasks:ps.indexes['S']")},
                     {"S": "#tasks:ps:subTasks:s['S']", "idx": "#tasks:ps:subTasks:idx"})},
         {"slice": "#tasks:ps[1]", "one": "#tasks:ps[2, 1, 'idx']"},
         {"slice": AL(grid[1], [["1", "2"], ["S", "idx"]], ["S", ""]), "one": A(21.0)},
         "[1] is the second value of k1 (0.5): its 2 x 2 table over S0 = 1, 2 and the entries S, idx: "
         "(exp(-1), 10), (2 exp(-1), 11).  [2, 1, 'idx'] is k1 = 0.75, S0 = 2: idx = 2 * 10 + 1 = 21.", ant=DECAY)

    # ------------------------------------------------------------------ nesting
    inner = {"_type": "scatter", "range": numeric(10, 1, 10),
             "subTasks": {"p": calc("#tasks:outer.range * #tasks:outer:subTasks:inner.range")},
             "outputVariableMap": {"P": "#tasks:outer:subTasks:inner:subTasks:p"}}
    case("A Scatter inside a Scatter: the outer sub-task reads the inner result by index and label.",
         {"outer": scatter(numeric(1, 2, 1),
                           {"inner": inner, "last": calc("#tasks:outer:subTasks:inner[1]['P']")},
                           {"L": "#tasks:outer:subTasks:last"})},
         {"r": "#tasks:outer"}, {"r": table([[20.0], [40.0], [60.0]], [1, 2, 3], ["L"])},
         "The inner range is 10, 20 and p = outer * inner; the inner result's second row is outer * 20: 20, 40, 60.")
    case("A Scatter inside a Scatter, reporting the whole inner result: the result has four dimensions.",
         {"outer": scatter(numeric(1, 2, 1), {"inner": inner}, {"I": "#tasks:outer:subTasks:inner"})},
         {"r": "#tasks:outer"},
         {"r": AL([[[[o * i] for i in (10, 20)]] for o in (1, 2, 3)],
                  [["1", "2", "3"], ["I"], ["10", "20"], ["P"]])},
         "p = outer * inner.  The outer iteration (1, 2, 3) is dimension 0, the entry I is dimension 1, and the inner "
         "result's own dimensions follow: its iterations (labelled 10, 20) and its entry P: 3 x 1 x 2 x 1.")
    case("A Loop inside a Scatter, the loop's initial value taken from the outer range.",
         {"outer": scatter(numeric(1, 1, 1),
                           {"lp": loop(numeric(1, 2, 1),
                                       {"a": {"initialValue": "#tasks:outer.range",
                                              "subsequentValues": "#tasks:outer:subTasks:lp:subTasks:d"}},
                                       {"d": calc("#tasks:outer:subTasks:lp:loopVariables:a * 2")},
                                       {"D": "#tasks:outer:subTasks:lp:subTasks:d"}),
                            "last": calc("#tasks:outer:subTasks:lp[-1]['D']")},
                           {"L": "#tasks:outer:subTasks:last"})},
         {"r": "#tasks:outer"}, {"r": table([[8.0], [16.0]], [1, 2], ["L"])},
         "The inner loop doubles its variable three times starting from the outer range value r: 2r, 4r, 8r.  The last "
         "is 8r: 8 for r = 1 and 16 for r = 2.")
    case("A ParameterScan inside a Scatter: the scan sets the rate and the outer range value scales the result.",
         {"outer": scatter(numeric(1, 1, 1),
                           {"ps": scan([prange("k1", values=[0.5, 1])], {"s": one_step("#tasks:outer:subTasks:ps.model", 2)},
                                       {"S": "#tasks:outer:subTasks:ps:subTasks:s['S']"}),
                            "last": calc("#tasks:outer:subTasks:ps[1, 'S'] * #tasks:outer.range")},
                           {"L": "#tasks:outer:subTasks:last"})},
         {"r": "#tasks:outer"}, {"r": table([[3 * np.exp(-2) * 1], [3 * np.exp(-2) * 2]], [1, 2], ["L"])},
         "The inner scan simulates 2 time units for k1 = 0.5 and 1; its second value (k1 = 1) gives S = 3 exp(-2).  "
         "The outer range value (1, 2) scales it.", ant=DECAY)
