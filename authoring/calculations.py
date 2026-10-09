"""Series "calculations": Calculation, CreateDataBlock, RelabelData and StringFormation (cases 00051-).

Calculations are evaluated by the generated script itself (not by the simulator), so a cross-check between backends can
only show that they all run it alike; the expected values, worked out by hand here, are what really test the math.
Where a closed form exists it is used (for example sinh(ln 2) = (2 - 1/2)/2 = 0.75).
"""
import math

import numpy as np

from _util import A, write_case, outputs

FIRST = 51
NAN, INF = float("nan"), float("inf")
S3, S2 = math.sqrt(3), math.sqrt(2)
LN2 = math.log(2)
PI = math.pi


def show(v):
    arr = np.asarray(v.values if hasattr(v, "values") else v)
    return str(arr.tolist()) if arr.ndim else repr(arr.item())


def build():
    counter = iter(range(FIRST, FIRST + 1000))

    def doc_case(what, constants, tasks, refs, expected, derivation, **kw):
        doc = {}
        if constants:
            doc["constants"] = constants
        doc["tasks"] = tasks
        doc["outputs"] = outputs(refs)
        return write_case(next(counter), what, doc, expected, derivation=derivation, **kw)

    def calcs_case(what, calcs, constants=None, derivation_intro="", **kw):
        """`calcs` maps an id to (math, expected value or AnnotatedData, how).  Every calculation is reported."""
        tasks, refs, expected, lines = {}, {}, {}, []
        for cid, (m, value, how) in calcs.items():
            tasks[cid] = {"_type": "calculation", "math": m}
            refs[cid] = f"#tasks:{cid}"
            data = value if hasattr(value, "values") and hasattr(value, "labels") else A(value)
            expected[cid] = data
            lines.append(f"- `{cid}`: `{m}` = {show(data)}" + (f" ({how})" if how else ""))
        derivation = (derivation_intro + "\n\n" if derivation_intro else "") + "\n".join(lines)
        return doc_case(what, constants, tasks, refs, expected, derivation, **kw)

    # ------------------------------------------------------------------ arithmetic
    calcs_case("The four arithmetic operators on numbers.  Division of integers is not integer division.", {
        "add": ("1.5 + 2.25", 3.75, ""), "sub": ("10 - 4.5", 5.5, ""), "mul": ("3 * 4", 12.0, ""),
        "div": ("7 / 2", 3.5, "not 3"), "mixed": ("1 + 2 * 3 - 4 / 2", 5.0, "1 + 6 - 2")})
    calcs_case("Precedence, associativity and parentheses.", {
        "p1": ("2 + 3 * 4", 14.0, "multiplication first"), "p2": ("(2 + 3) * 4", 20.0, ""),
        "p3": ("10 - 4 - 3", 3.0, "left to right: (10 - 4) - 3"), "p4": ("100 / 10 / 5", 2.0, "(100 / 10) / 5"),
        "p5": ("2 * 3 / 4", 1.5, "(2 * 3) / 4"), "p6": ("8 / (2 * 2)", 2.0, ""),
        "p7": ("((1 + 2) * (3 + 4))", 21.0, "nested parentheses")})
    calcs_case("The power operator: right associative, tighter than unary minus.", {
        "pow1": ("2 ^ 10", 1024.0, ""), "pow2": ("2 ^ 3 ^ 2", 512.0, "right to left: 2 ^ 9, not 8 ^ 2 = 64"),
        "pow3": ("-2 ^ 2", -4.0, "-(2 ^ 2)"), "pow4": ("(-2) ^ 2", 4.0, ""), "pow5": ("4 ^ 0.5", 2.0, ""),
        "pow6": ("2 ^ -1", 0.5, ""), "pow7": ("10 ^ -2", 0.01, ""), "pow8": ("5 ^ 0", 1.0, "")})
    calcs_case("Unary minus and plus.", {
        "neg1": ("-5", -5.0, ""), "neg2": ("-(-5)", 5.0, ""), "neg3": ("3 - -2", 5.0, "subtract minus two"),
        "neg4": ("-3 * -4", 12.0, ""), "plus": ("+4", 4.0, ""), "neg5": ("-(2 + 3)", -5.0, "")})
    calcs_case("The remainder, as an operator and as functions: the result has the sign of the dividend.", {
        "r1": ("7 % 3", 1.0, ""), "r2": ("-7 % 3", -1.0, "x - y * trunc(x / y) = -7 - 3 * (-2)"),
        "r3": ("7 % -3", 1.0, "7 - (-3) * (-2)"), "r4": ("5.5 % 2", 1.5, ""), "r5": ("rem(-7, 3)", -1.0, ""),
        "r6": ("rem(7.5, 2)", 1.5, ""), "q1": ("quotient(7, 2)", 3.0, "integer part of 3.5"),
        "q2": ("quotient(10, 5)", 2.0, ""), "q3": ("2 + 7 % 4 * 2", 8.0, "% binds like *: 2 + (7 % 4) * 2 = 2 + 6")})
    calcs_case("abs, floor and ceiling.", {
        "abs1": ("abs(-3.5)", 3.5, ""), "abs2": ("abs(2)", 2.0, ""), "abs3": ("abs(0)", 0.0, ""),
        "fl1": ("floor(2.7)", 2.0, ""), "fl2": ("floor(-2.7)", -3.0, "toward minus infinity"),
        "fl3": ("floor(5)", 5.0, ""), "ce1": ("ceiling(2.1)", 3.0, ""),
        "ce2": ("ceiling(-2.7)", -2.0, "toward plus infinity"), "ce3": ("ceiling(5)", 5.0, "")})
    calcs_case("exp, ln and log (base 10 with one argument; log(base, x) with two).", {
        "e1": ("exp(0)", 1.0, ""), "e2": ("exp(1)", math.e, "Euler's number"), "l1": ("ln(1)", 0.0, ""),
        "l2": ("ln(exponentiale)", 1.0, ""), "l3": ("ln(exp(2.5))", 2.5, ""),
        "g1": ("log(1000)", 3.0, "base 10"), "g2": ("log(0.01)", -2.0, ""), "g3": ("log(2, 8)", 3.0, "base 2 of 8"),
        "g4": ("log(10, 100000)", 5.0, ""), "g5": ("log(3, 81)", 4.0, "")})
    calcs_case("root with one argument (square root) and with two (root(degree, x)); factorial.", {
        "sq1": ("root(16)", 4.0, ""), "sq2": ("root(2, 9)", 3.0, ""), "cu": ("root(3, 27)", 3.0, ""),
        "r4": ("root(4, 16)", 2.0, ""), "f0": ("factorial(0)", 1.0, ""), "f1": ("factorial(1)", 1.0, ""),
        "f5": ("factorial(5)", 120.0, ""), "f10": ("factorial(10)", 3628800.0, "")})
    calcs_case("min and max of several numbers.", {
        "mn1": ("min(3, 1, 2)", 1.0, ""), "mx1": ("max(3, 1, 2)", 3.0, ""), "mn2": ("min(5)", 5.0, "one argument"),
        "mx2": ("max(-1, -2)", -1.0, ""), "mn3": ("min(2.5, 2.25, 9)", 2.25, ""), "mx3": ("max(1, 1, 1)", 1.0, "")})
    calcs_case("Whitespace, exponent notation and decimal forms inside math.", {
        "w1": ("  1e3  +  2.5E-1 ", 1000.25, ""), "w2": ("0.1 + 0.2", 0.3, "within the tolerance"),
        "w3": ("1E2*3", 300.0, ""), "w4": ("2.5\t*\t4", 10.0, "tabs separate tokens too"), "w5": ("1.5e+2 / 3", 50.0, "")})

    # ------------------------------------------------------------------ circular and hyperbolic functions
    # x = pi * f with f a fraction of pi: closed forms for sin, cos, tan and their reciprocals
    fs = [0.0, 1 / 6, 0.25, 1 / 3, 0.5]
    calcs_case("sin, cos and tan at multiples of pi/12 with closed forms.  f = 0, 1/6, 1/4, 1/3, 1/2 and the angle is pi * f.", {
        "sin": ("sin(pi * #constants:f)", A([0.0, 0.5, S2 / 2, S3 / 2, 1.0]), "sin: 0, 1/2, sqrt2/2, sqrt3/2, 1"),
        "cos": ("cos(pi * #constants:f)", A([1.0, S3 / 2, S2 / 2, 0.5, 0.0]), "cos: 1, sqrt3/2, sqrt2/2, 1/2, 0"),
        "tan": ("tan(pi * #constants:g)", A([0.0, 1 / S3, 1.0, S3]), "tan at the first four angles of g: 0, 1/sqrt3, 1, sqrt3")},
        constants={"f": fs, "g": fs[:4]})
    calcs_case("sec, csc and cot at angles where they are finite.", {
        "sec": ("sec(pi * #constants:f)", A([1.0, 2 / S3, S2, 2.0]), "1/cos at f = 0, 1/6, 1/4, 1/3"),
        "csc": ("csc(pi * #constants:g)", A([2.0, S2, 2 / S3, 1.0]), "1/sin at g = 1/6, 1/4, 1/3, 1/2"),
        "cot": ("cot(pi * #constants:g)", A([S3, 1.0, 1 / S3, 0.0]), "cos/sin at g = 1/6, 1/4, 1/3, 1/2")},
        constants={"f": fs[:4], "g": fs[1:]})
    calcs_case("The six hyperbolic functions at x = ln 2, where e^x = 2 and e^-x = 1/2.", {
        "sinh": ("sinh(ln(2))", 0.75, "(2 - 1/2) / 2"), "cosh": ("cosh(ln(2))", 1.25, "(2 + 1/2) / 2"),
        "tanh": ("tanh(ln(2))", 0.6, "0.75 / 1.25"), "sech": ("sech(ln(2))", 0.8, "1 / 1.25"),
        "csch": ("csch(ln(2))", 4 / 3, "1 / 0.75"), "coth": ("coth(ln(2))", 5 / 3, "1.25 / 0.75"),
        "s0": ("sinh(0)", 0.0, ""), "c0": ("cosh(0)", 1.0, "")})
    calcs_case("arcsin, arccos and arctan: principal values.", {
        "as1": ("arcsin(#constants:u)", A([-PI / 2, -PI / 6, 0.0, PI / 6, PI / 2]), "u = -1, -1/2, 0, 1/2, 1"),
        "ac1": ("arccos(#constants:u)", A([PI, 2 * PI / 3, PI / 2, PI / 3, 0.0]), "arccos(-1/2) = 2 pi/3"),
        "at1": ("arctan(#constants:t)", A([-PI / 4, 0.0, PI / 4, PI / 3]), "t = -1, 0, 1, sqrt3"),
        "at2": ("arctan(1) * 4", PI, "")},
        constants={"u": [-1, -0.5, 0, 0.5, 1], "t": [-1, 0, 1, S3]})
    calcs_case("arcsec, arccsc and arccot for positive arguments (where the principal value is not in doubt).", {
        "asec": ("arcsec(#constants:x)", A([0.0, PI / 3, PI / 4]), "x = 1, 2, sqrt2: arccos(1/x)"),
        "acsc": ("arccsc(#constants:x)", A([PI / 2, PI / 6, PI / 4]), "arcsin(1/x)"),
        "acot": ("arccot(#constants:y)", A([PI / 4, PI / 3, PI / 6]), "y = 1, 1/sqrt3, sqrt3: arctan(1/y)")},
        constants={"x": [1, 2, S2], "y": [1, 1 / S3, S3]})
    calcs_case("The inverse hyperbolic functions; each of these arguments is the value of the matching hyperbolic function at ln 2, so the answer is ln 2.", {
        "asinh": ("arcsinh(0.75)", LN2, "sinh(ln 2) = 0.75"), "acosh": ("arccosh(1.25)", LN2, "cosh(ln 2) = 1.25"),
        "atanh": ("arctanh(0.6)", LN2, "tanh(ln 2) = 0.6"), "asech": ("arcsech(0.8)", LN2, "sech(ln 2) = 0.8"),
        "acsch": ("arccsch(4 / 3)", LN2, "csch(ln 2) = 4/3"), "acoth": ("arccoth(5 / 3)", LN2, "coth(ln 2) = 5/3"),
        "z1": ("arcsinh(0)", 0.0, ""), "z2": ("arccosh(1)", 0.0, "")})
    calcs_case("The predefined constants.", {
        "pi": ("pi", PI, ""), "e": ("exponentiale", math.e, ""), "t": ("true", 1.0, "booleans are parsed as 1"),
        "f": ("false", 0.0, "and 0"), "nan": ("notanumber", NAN, ""), "inf": ("infinity", INF, ""),
        "ninf": ("-infinity", -INF, ""), "circ": ("pi * 2 ^ 2", 4 * PI, "area of a circle of radius 2")})
    calcs_case("Arithmetic with NaN and infinity follows IEEE rules.", {
        "n1": ("notanumber + 1", NAN, ""), "n2": ("notanumber * 0", NAN, ""), "i1": ("infinity + 1", INF, ""),
        "i2": ("infinity - infinity", NAN, "undefined"), "i3": ("1 / infinity", 0.0, ""),
        "i4": ("infinity * -1", -INF, ""), "i5": ("0 * infinity", NAN, "undefined"), "i6": ("infinity * infinity", INF, ""),
        "i7": ("-infinity - 1", -INF, "")})

    # ------------------------------------------------------------------ relational and logical, through piecewise
    # (a bare comparison is a boolean, which can only be 'parsed as 1 or 0'; piecewise makes the test unambiguous)
    def cond(expr):
        return f"piecewise(1, {expr}, 0)"

    rel_true = {"==": "2 == 2", "!=": "2 != 3", "<>": "2 <> 3", "><": "2 >< 3", "<": "2 < 3", ">": "3 > 2", "<=": "2 <= 2",
                ">=": "3 >= 3", "eq": "eq(2, 2)", "neq": "neq(2, 3)", "lt": "lt(2, 3)", "gt": "gt(3, 2)", "leq": "leq(2, 2)",
                "geq": "geq(3, 3)", "chain_lt": "1 < 2 < 3", "chain_mixed": "1 < 2 <= 2", "chain_eq": "2 == 2 == 2",
                "multi_lt": "lt(1, 2, 3)", "multi_eq": "eq(4, 4, 4)", "multi_geq": "geq(3, 3, 1)"}
    names = {k: "r" + str(i) for i, k in enumerate(rel_true)}
    calcs_case("Relational operators and their call forms, each true: the test is wrapped in piecewise(1, test, 0), so every result is 1.", {
        names[k]: (cond(v), 1.0, "true") for k, v in rel_true.items()})
    rel_false = {"==": "2 == 3", "!=": "2 != 2", "<>": "2 <> 2", "><": "2 >< 2", "<": "3 < 2", ">": "2 > 3", "<=": "3 <= 2",
                 ">=": "2 >= 3", "eq": "eq(2, 3)", "neq": "neq(2, 2)", "lt": "lt(3, 2)", "gt": "gt(2, 3)", "leq": "leq(3, 2)",
                 "geq": "geq(2, 3)", "chain_lt": "1 < 3 < 2", "chain_mixed": "1 < 2 <= 1", "chain_eq": "2 == 2 == 3",
                 "multi_lt": "lt(1, 3, 2)", "multi_eq": "eq(4, 4, 5)", "multi_geq": "geq(3, 1, 3)"}
    calcs_case("The same relational operators, each false: every result is 0.", {
        names[k]: (cond(v), 0.0, "false") for k, v in rel_false.items()})
    calcs_case("Logical operators and their call forms, each true: every result is 1.", {
        "and1": (cond("1 < 2 && 2 < 3"), 1.0, ""), "or1": (cond("1 > 2 || 2 > 1"), 1.0, ""),
        "not1": (cond("!(1 > 2)"), 1.0, ""), "and2": (cond("and(1 < 2, 2 < 3, 3 < 4)"), 1.0, ""),
        "or2": (cond("or(1 > 2, 2 > 3, 3 < 4)"), 1.0, ""), "xor1": (cond("xor(1 < 2, 1 > 2)"), 1.0, "exactly one true"),
        "xor3": (cond("xor(1 < 2, 1 < 3, 1 < 4)"), 1.0, "an odd number are true"),
        "not2": (cond("not(2 < 1)"), 1.0, ""), "imp1": (cond("implies(1 > 2, 1 > 3)"), 1.0, "false implies anything"),
        "imp2": (cond("implies(1 < 2, 2 < 3)"), 1.0, "true implies true"), "and0": (cond("and()"), 1.0, "empty and is true"),
        "t": (cond("true"), 1.0, ""), "prec1": (cond("1 + 1 == 2"), 1.0, "+ binds tighter than =="),
        "prec2": (cond("1 == 2 && 1 == 1 || 1 == 1"), 1.0, "&& and || share a level, left to right: (false && true) || true"),
        "prec3": (cond("!(1 == 2) && 2 == 2"), 1.0, "")})
    calcs_case("Logical operators and their call forms, each false: every result is 0.", {
        "and1": (cond("1 < 2 && 2 > 3"), 0.0, ""), "or1": (cond("1 > 2 || 2 > 3"), 0.0, ""),
        "not1": (cond("!(1 < 2)"), 0.0, ""), "and2": (cond("and(1 < 2, 2 < 3, 3 > 4)"), 0.0, ""),
        "or2": (cond("or(1 > 2, 2 > 3, 3 > 4)"), 0.0, ""), "xor1": (cond("xor(1 < 2, 1 < 3)"), 0.0, "two are true"),
        "xor0": (cond("xor()"), 0.0, "empty xor is false"), "or0": (cond("or()"), 0.0, "empty or is false"),
        "imp1": (cond("implies(1 < 2, 2 > 3)"), 0.0, "true implies false"), "f": (cond("false"), 0.0, ""),
        "prec1": (cond("1 == 1 || 1 == 2 && 1 == 2"), 0.0, "(true || false) && false = false")})
    calcs_case("Comparisons with NaN are false except !=.", {
        "eq": (cond("notanumber == notanumber"), 0.0, ""), "lt": (cond("notanumber < 1"), 0.0, ""),
        "gt": (cond("notanumber > 1"), 0.0, ""), "neq": (cond("notanumber != notanumber"), 1.0, ""),
        "inf": (cond("infinity > 1e300"), 1.0, ""), "ninf": (cond("-infinity < -1e300"), 1.0, "")})
    calcs_case("piecewise: the first condition that is true wins; the last lone argument is the 'otherwise' value.", {
        "first": ("piecewise(10, 1 > 2, 20, 2 > 1, 30, 3 > 2, 99)", 20.0, "the second pair is the first true one"),
        "other": ("piecewise(10, 1 > 2, 99)", 99.0, "no condition is true"),
        "noother": ("piecewise(5, 1 < 2)", 5.0, ""), "only": ("piecewise(99)", 99.0, "a lone argument is the otherwise value"),
        "none": ("piecewise(5, 1 > 2)", NAN, "no condition is true and there is no otherwise value: undefined (NaN)"),
        "calc": ("piecewise(2 * 3, 1 < 2, 0)", 6.0, "values are expressions too"),
        "nested": ("piecewise(piecewise(1, 1 > 2, 2), 1 < 2, 3)", 2.0, "inner piecewise gives 2")})

    # ------------------------------------------------------------------ broadcasting
    V, W = [1, 2, 3], [10, 20, 30]
    M = [[1, 2, 3], [4, 5, 6]]
    calcs_case("A number combined with a vector acts on every element.  v = 1, 2, 3.", {
        "add": ("5 + #constants:v", A([6.0, 7.0, 8.0]), ""), "sub": ("#constants:v - 1", A([0.0, 1.0, 2.0]), ""),
        "rsub": ("10 - #constants:v", A([9.0, 8.0, 7.0]), ""), "mul": ("2 * #constants:v", A([2.0, 4.0, 6.0]), ""),
        "div": ("12 / #constants:v", A([12.0, 6.0, 4.0]), ""), "pow": ("#constants:v ^ 2", A([1.0, 4.0, 9.0]), ""),
        "rpow": ("2 ^ #constants:v", A([2.0, 4.0, 8.0]), ""), "neg": ("-#constants:v", A([-1.0, -2.0, -3.0]), "")},
        constants={"v": V})
    calcs_case("Two vectors of the same length combine element by element.  v = 1, 2, 3 and w = 10, 20, 30.", {
        "add": ("#constants:v + #constants:w", A([11.0, 22.0, 33.0]), ""),
        "sub": ("#constants:w - #constants:v", A([9.0, 18.0, 27.0]), ""),
        "mul": ("#constants:v * #constants:w", A([10.0, 40.0, 90.0]), ""),
        "div": ("#constants:w / #constants:v", A([10.0, 10.0, 10.0]), ""),
        "pow": ("#constants:v ^ #constants:v", A([1.0, 4.0, 27.0]), ""),
        "expr": ("(#constants:v + 1) * (#constants:w - 5)", A([10.0, 45.0, 100.0]), "2*5, 3*15, 4*25")},
        constants={"v": V, "w": W})
    calcs_case("A number combined with a matrix.  m has rows 1 2 3 / 4 5 6.", {
        "mul": ("#constants:m * 2", A([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]), ""),
        "rsub": ("1 - #constants:m", A([[0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]]), ""),
        "div": ("12 / #constants:m", A([[12.0, 6.0, 4.0], [3.0, 2.4, 2.0]]), "")}, constants={"m": M})
    calcs_case("Two matrices of the same shape combine element by element.", {
        "add": ("#constants:m + #constants:n", A([[11.0, 22.0, 33.0], [44.0, 55.0, 66.0]]), ""),
        "mul": ("#constants:m * #constants:n", A([[10.0, 40.0, 90.0], [160.0, 250.0, 360.0]]), "")},
        constants={"m": M, "n": [[10, 20, 30], [40, 50, 60]]})
    calcs_case("A vector combined with a matrix applies to each row: [1D] * [2D] multiplies each row.  v = 1, 2, 3; m rows 1 2 3 / 4 5 6.", {
        "mul": ("#constants:v * #constants:m", A([[1.0, 4.0, 9.0], [4.0, 10.0, 18.0]]), "each row times v"),
        "add": ("#constants:m + #constants:v", A([[2.0, 4.0, 6.0], [5.0, 7.0, 9.0]]), "v added to each row"),
        "sub": ("#constants:m - #constants:v", A([[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]]), "")},
        constants={"v": V, "m": M})
    t3 = np.arange(1.0, 13.0).reshape(2, 3, 2)
    calcs_case("A matrix combined with a three-dimensional array applies to each slab.  t is 2 x 3 x 2 holding 1..12; p is 3 x 2.", {
        "add": ("#constants:t + #constants:p", A((t3 + np.array([[10, 20], [30, 40], [50, 60]])).tolist()),
                "p added to each of the two 3 x 2 slabs"),
        "mul": ("#constants:t * 2", A((t3 * 2).tolist()), "")},
        constants={"t": t3.tolist(), "p": [[10, 20], [30, 40], [50, 60]]})
    calcs_case("Functions act on every element of a vector or matrix.", {
        "abs": ("abs(#constants:v)", A([1.5, 0.0, 2.0]), ""), "floor": ("floor(#constants:v)", A([-2.0, 0.0, 2.0]), ""),
        "ceil": ("ceiling(#constants:v)", A([-1.0, 0.0, 2.0]), ""),
        "root": ("root(#constants:sq)", A([[1.0, 2.0], [3.0, 4.0]]), "square roots of 1 4 / 9 16"),
        "exp": ("exp(#constants:z)", A([1.0, 1.0]), "exp(0) twice")},
        constants={"v": [-1.5, 0.0, 2.0], "sq": [[1, 4], [9, 16]], "z": [0, 0]})
    calcs_case("A dictionary constant used in math keeps its keys as labels.  d = {a: 1, b: 2, c: 3}, e = {a: 10, b: 20, c: 30}.", {
        "scaled": ("#constants:d * 10", A([10.0, 20.0, 30.0], rows=["a", "b", "c"]), "labels a, b, c are kept"),
        "summed": ("#constants:d + #constants:e", A([11.0, 22.0, 33.0], rows=["a", "b", "c"]), "same keys: entries with the same key combine")},
        constants={"d": {"a": 1, "b": 2, "c": 3}, "e": {"a": 10, "b": 20, "c": 30}})

    # ------------------------------------------------------------------ sum and references inside math
    calcs_case("sum reduces the outermost dimension: a vector gives a number and a matrix gives a vector (column sums).", {
        "vec": ("sum(#constants:v)", 150.0, "10 + 20 + 30 + 40 + 50"),
        "mat": ("sum(#constants:m)", A([5.0, 7.0, 9.0]), "rows 1 2 3 and 4 5 6 added together"),
        "part": ("sum(#constants:v[1:4])", 90.0, "20 + 30 + 40"),
        "row": ("sum(#constants:m[1])", 15.0, "4 + 5 + 6"),
        "col": ("sum(#constants:m[0:2, 1])", 7.0, "2 + 5"),
        "expr": ("sum(#constants:v) / 5", 30.0, "the mean"),
        "sq": ("sum(#constants:v ^ 2)", 5500.0, "100 + 400 + 900 + 1600 + 2500")},
        constants={"v": [10, 20, 30, 40, 50], "m": M})
    calcs_case("sum of a three-dimensional array reduces the first dimension.", {
        "s": ("sum(#constants:t)", A((t3[0] + t3[1]).tolist()), "slab 0 + slab 1: 8 10 / 12 14 / 16 18")},
        constants={"t": t3.tolist()})
    calcs_case("References with indices inside math.  v = 10, 20, 30, 40, 50; m rows 1 2 3 / 4 5 6.", {
        "ends": ("#constants:v[0] + #constants:v[-1]", 60.0, "10 + 50"),
        "slice": ("#constants:v[1:3] * 2", A([40.0, 60.0]), ""),
        "rows": ("#constants:m[0] * #constants:m[1]", A([4.0, 10.0, 18.0]), "row 0 times row 1"),
        "col": ("#constants:m[0:2, 1] + 1", A([3.0, 6.0]), "column 1 plus 1"),
        "elem": ("#constants:m[1, 2] ^ 2", 36.0, "")},
        constants={"v": [10, 20, 30, 40, 50], "m": M})
    doc_case("A calculation can use the result of another calculation.  v = 1, 2, 3.", {"v": V},
             {"c1": {"_type": "calculation", "math": "#constants:v * 2"},
              "c2": {"_type": "calculation", "math": "#tasks:c1 + 1"},
              "c3": {"_type": "calculation", "math": "sum(#tasks:c2) + #tasks:c1[0]"}},
             {"c1": "#tasks:c1", "c2": "#tasks:c2", "c3": "#tasks:c3"},
             {"c1": A([2.0, 4.0, 6.0]), "c2": A([3.0, 5.0, 7.0]), "c3": A(17.0)},
             "c1 = 2, 4, 6; c2 = c1 + 1 = 3, 5, 7; c3 = sum(c2) + c1[0] = 15 + 2 = 17.")
    doc_case("A math string that is a single reference is that value, not the text of the reference.", {"v": V},
             {"same": {"_type": "calculation", "math": "#constants:v"},
              "idx": {"_type": "calculation", "math": "#constants:v[2]"}},
             {"same": "#tasks:same", "idx": "#tasks:idx"}, {"same": A([1.0, 2.0, 3.0]), "idx": A(3.0)},
             "same is the vector 1, 2, 3; idx is its entry 2, which is 3.")

    # ------------------------------------------------------------------ CreateDataBlock
    doc_case("CreateDataBlock: a dictionary of numbers becomes a labelled vector, in the order of the keys.",
             {"k": 7},
             {"blk": {"_type": "createDataBlock", "data": {"b": "#constants:k", "a": 2.5, "c": -1}}},
             {"r": "#tasks:blk"}, {"r": A([7.0, 2.5, -1.0], rows=["b", "a", "c"])},
             "One entry per key, labelled by the key, in the order written: b = 7 (the constant k), a = 2.5, c = -1.")
    doc_case("CreateDataBlock: a block of vectors is a matrix with one row per key.  Labels then select a row, and a row can be indexed again.",
             {"v": [1, 2, 3]},
             {"blk": {"_type": "createDataBlock", "data": {"r1": "#constants:v", "r2": [4, 5, 6]}}},
             {"all": "#tasks:blk", "row": "#tasks:blk['r2']", "chained": "#tasks:blk['r2'][1]",
              "comma": "#tasks:blk['r2', 1]", "col": "#tasks:blk[:, 2]", "pos": "#tasks:blk[0]"},
             {"all": A([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], rows=["r1", "r2"]), "row": A([4.0, 5.0, 6.0]),
              "chained": A(5.0), "comma": A(5.0), "col": A([3.0, 6.0], rows=["r1", "r2"]), "pos": A([1.0, 2.0, 3.0])},
             "all is the 2 x 3 matrix with rows r1 = 1 2 3 and r2 = 4 5 6, row-labelled r1, r2.  row is r2 (4 5 6, label gone); "
             "chained and comma are entry 1 of r2 = 5; col is column 2 of both rows = 3, 6 with the row labels kept; "
             "pos is row 0 = 1 2 3.")
    doc_case("CreateDataBlock: the data may be given by a reference to a dictionary constant.",
             {"spec": {"p": 1, "q": 2, "r": 3}},
             {"blk": {"_type": "createDataBlock", "data": "#constants:spec"}},
             {"r": "#tasks:blk"}, {"r": A([1.0, 2.0, 3.0], rows=["p", "q", "r"])},
             "The labelled vector with the keys p, q, r and the values 1, 2, 3.")
    doc_case("CreateDataBlock: the entries may be results of other tasks.",
             {"v": [1, 2, 3]},
             {"total": {"_type": "calculation", "math": "sum(#constants:v)"},
              "scaled": {"_type": "calculation", "math": "#constants:v * 10"},
              "blk": {"_type": "createDataBlock", "data": {"total": "#tasks:total", "one": 1, "x": "#constants:v[2]"}},
              "mat": {"_type": "createDataBlock", "data": {"base": "#constants:v", "scaled": "#tasks:scaled"}}},
             {"blk": "#tasks:blk", "mat": "#tasks:mat"},
             {"blk": A([6.0, 1.0, 3.0], rows=["total", "one", "x"]),
              "mat": A([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], rows=["base", "scaled"])},
             "total = 1 + 2 + 3 = 6, one = 1, x = v[2] = 3.  mat has the row base = 1 2 3 and the row scaled = 10 20 30.")
    doc_case("CreateDataBlock: a block of matrices is a three-dimensional array whose first dimension is labelled.",
             {"m": M, "n": [[10, 20, 30], [40, 50, 60]]},
             {"blk": {"_type": "createDataBlock", "data": {"m": "#constants:m", "n": "#constants:n"}}},
             {"all": "#tasks:blk", "slab": "#tasks:blk['n']", "entry": "#tasks:blk['n', 1, 2]"},
             {"all": A([M, [[10, 20, 30], [40, 50, 60]]], rows=["m", "n"]), "slab": A([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
              "entry": A(60.0)},
             "all has shape 2 x 2 x 3 (key, row, column), slab is the matrix n, and entry is row 1 column 2 of n, 60.")
    doc_case("CreateDataBlock: a single entry still gives a one-element vector.",
             {}, {"blk": {"_type": "createDataBlock", "data": {"only": 4}}},
             {"r": "#tasks:blk", "one": "#tasks:blk['only']"},
             {"r": A([4.0], rows=["only"]), "one": A(4.0)},
             "r is a vector of one entry labelled only; indexing by that label drops the dimension, leaving the number 4.")

    # ------------------------------------------------------------------ RelabelData
    doc_case("RelabelData: the labels of the topmost dimension are replaced; the values and the original are unchanged.",
             {},
             {"blk": {"_type": "createDataBlock", "data": {"a": 1, "b": 2, "c": 3}},
              "rel": {"_type": "relabelData", "input": "#tasks:blk", "labels": ["P", "Q", "R"]}},
             {"rel": "#tasks:rel", "orig": "#tasks:blk"},
             {"rel": A([1.0, 2.0, 3.0], rows=["P", "Q", "R"]), "orig": A([1.0, 2.0, 3.0], rows=["a", "b", "c"])},
             "The values 1, 2, 3 now carry the labels P, Q, R; the block the labels came from still has a, b, c.")
    doc_case("RelabelData: the new labels may come from a constant.",
             {"names": ["x", "y", "z"]},
             {"blk": {"_type": "createDataBlock", "data": {"a": 10, "b": 20, "c": 30}},
              "rel": {"_type": "relabelData", "input": "#tasks:blk", "labels": "#constants:names"}},
             {"rel": "#tasks:rel"},
             {"rel": A([10.0, 20.0, 30.0], rows=["x", "y", "z"])},
             "Labels x, y, z replace a, b, c; the values stay 10, 20, 30.",
             notes="Indexing by a new label is not covered yet: the validator rejects it (SED2/TODO.md: RelabelData).")
    doc_case("RelabelData on a matrix replaces the row labels only.",
             {},
             {"blk": {"_type": "createDataBlock", "data": {"r1": [1, 2, 3], "r2": [4, 5, 6]}},
              "rel": {"_type": "relabelData", "input": "#tasks:blk", "labels": ["top", "bottom"]}},
             {"rel": "#tasks:rel"},
             {"rel": A([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], rows=["top", "bottom"])},
             "The rows are now called top and bottom; the values are the same.",
             notes="Indexing by a new label is not covered yet: the validator rejects it (SED2/TODO.md: RelabelData).")
    doc_case("RelabelData with labels built by StringFormation (the .strings accessor).",
             {"n": [1, 2, 3]},
             {"blk": {"_type": "createDataBlock", "data": {"a": 5, "b": 6, "c": 7}},
              "lab": {"_type": "stringFormation", "concatenate": ["S", "#constants:n"]},
              "rel": {"_type": "relabelData", "input": "#tasks:blk", "labels": "#tasks:lab.strings"}},
             {"rel": "#tasks:rel"},
             {"rel": A([5.0, 6.0, 7.0], rows=["S1", "S2", "S3"])},
             "The labels are S1, S2, S3 and the values 5, 6, 7.",
             notes="Indexing by a new label is not covered yet: the validator rejects it (SED2/TODO.md: RelabelData).")

    # ------------------------------------------------------------------ StringFormation
    doc_case("StringFormation: strings are concatenated as they are.", {},
             {"sf": {"_type": "stringFormation", "concatenate": ["a", "b", "c"]},
              "sp": {"_type": "stringFormation", "concatenate": ["x = ", "1", " unit"]},
              "one": {"_type": "stringFormation", "concatenate": ["alone"]}},
             {"sf": "#tasks:sf", "sp": "#tasks:sp", "one": "#tasks:one"},
             {"sf": A("abc"), "sp": A("x = 1 unit"), "one": A("alone")},
             "abc; the spaces inside the strings are kept; a single string is returned unchanged.")
    doc_case("StringFormation: numbers.  An integral number has no decimal point; any other is the shortest decimal that reads back the same.",
             {"k": 2.5, "third": 1 / 3, "n": 3.0, "neg": -2, "big": 1234567, "small": 0.125},
             {"lit": {"_type": "stringFormation", "concatenate": ["a", 3, "b", 2.5, "c", -2, "d", 0.1, "e", 100]},
              "ref": {"_type": "stringFormation", "concatenate": ["k=", "#constants:k", " third=", "#constants:third",
                                                                   " n=", "#constants:n", " neg=", "#constants:neg",
                                                                   " big=", "#constants:big", " small=", "#constants:small"]},
              "float": {"_type": "stringFormation", "concatenate": ["v", 4.0]}},
             {"lit": "#tasks:lit", "ref": "#tasks:ref", "float": "#tasks:float"},
             {"lit": A("a3b2.5c-2d0.1e100"),
              "ref": A("k=2.5 third=0.3333333333333333 n=3 neg=-2 big=1234567 small=0.125"), "float": A("v4")},
             "3 -> 3, 2.5 -> 2.5, -2 -> -2, 0.1 -> 0.1, 100 -> 100.  The constant 3.0 is written 3.  1/3 needs sixteen "
             "digits to read back the same: 0.3333333333333333.  4.0 is written 4.")
    doc_case("StringFormation: booleans.  A literal is true or false; a reference to a boolean constant is a number, so 1 or 0.",
             {"yes": True, "no": False},
             {"lit": {"_type": "stringFormation", "concatenate": ["lit:", True, ",", False]},
              "ref": {"_type": "stringFormation", "concatenate": ["ref:", "#constants:yes", ",", "#constants:no"]}},
             {"lit": "#tasks:lit", "ref": "#tasks:ref"},
             {"lit": A("lit:true,false"), "ref": A("ref:1,0")},
             "As stated in the StringFormation description.")
    doc_case("StringFormation: a list element gives a list of strings.", {"v": [1, 2, 3], "s": ["a", "b", "c"]},
             {"lit": {"_type": "stringFormation", "concatenate": ["n = ", [1, 2, 3]]},
              "ref": {"_type": "stringFormation", "concatenate": ["x", "#constants:v"]},
              "strs": {"_type": "stringFormation", "concatenate": ["<", "#constants:s", ">"]}},
             {"lit": "#tasks:lit", "ref": "#tasks:ref", "strs": "#tasks:strs"},
             {"lit": A(["n = 1", "n = 2", "n = 3"]), "ref": A(["x1", "x2", "x3"]), "strs": A(["<a>", "<b>", "<c>"])},
             "Each member of the list is concatenated into a separate string, as in the description's own example.")
    doc_case("StringFormation: several lists are combined pairwise.", {"v": [1, 2, 3], "s": ["a", "b", "c"], "name": "S"},
             {"pair": {"_type": "stringFormation", "concatenate": ["#constants:name", "#constants:v", "_", "#constants:s"]},
              "lits": {"_type": "stringFormation", "concatenate": [[1, 2], "-", ["x", "y"]]}},
             {"pair": "#tasks:pair", "lits": "#tasks:lits"},
             {"pair": A(["S1_a", "S2_b", "S3_c"]), "lits": A(["1-x", "2-y"])},
             "The first element of each list goes together, then the second, and so on.")
    doc_case("StringFormation: two-dimensional lists give a two-dimensional result.",
             {"m": [[1, 2], [3, 4]], "t": [["a", "b"], ["c", "d"]]},
             {"grid": {"_type": "stringFormation", "concatenate": ["r", "#constants:m", "c", "#constants:t"]}},
             {"grid": "#tasks:grid"}, {"grid": A([["r1ca", "r2cb"], ["r3cc", "r4cd"]])},
             "The result has the shape of the lists: entry [i][j] is r, m[i][j], c, t[i][j].")
    doc_case("StringFormation: .strings gives the same strings as the task itself.", {"v": [1, 2]},
             {"s": {"_type": "stringFormation", "concatenate": ["id", "#constants:v"]}},
             {"plain": "#tasks:s", "strings": "#tasks:s.strings", "second": "#tasks:s.strings[1]"},
             {"plain": A(["id1", "id2"]), "strings": A(["id1", "id2"]), "second": A("id2")},
             "Both are id1, id2; indexing [1] gives id2.")
