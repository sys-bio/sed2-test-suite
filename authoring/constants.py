"""Series "constants": documents with constants and reports only (cases 00001-).

Every expected value is worked out by hand from the SED2 specification (Types: Indexing, Special number rules).
Constants carry no labels, so label indexing is not covered here (it needs task output; see the later series).
"""
import numpy as np

from _util import A, constants_case

V = [10, 20, 30, 40, 50]
M = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
T = [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]   # shape (2, 3, 2)
NAN, INF = float("nan"), float("inf")


def build():
    n = iter(range(1, 1000))

    def case(what, constants, refs, expected, derivation, **kw):
        constants_case(next(n), what, constants, refs, expected, derivation, **kw)

    # ---- shapes and types of literal constants
    case("A single number is reported unchanged.  A scalar is zero-dimensional data.",
         {"x": 3.5}, {"r": "#constants:x"}, {"r": A(3.5)}, "The report is the scalar 3.5.")
    case("An integer constant is reported as a number.",
         {"x": 7}, {"r": "#constants:x"}, {"r": A(7.0)}, "The report is the scalar 7.")
    case("A list of numbers is a one-dimensional result.",
         {"v": V}, {"r": "#constants:v"}, {"r": A(V)}, "The report is the vector 10, 20, 30, 40, 50.")
    case("A list of lists is a two-dimensional result (rows first).",
         {"m": M}, {"r": "#constants:m"}, {"r": A(M)},
         "The report is the 4 x 3 matrix whose rows are the inner lists, in order.")
    case("A list of lists of lists is a three-dimensional result.",
         {"t": T}, {"r": "#constants:t"}, {"r": A(T)},
         "The report is the 2 x 3 x 2 array given, outermost list first.")
    case("Negative numbers, fractions and exponent notation survive unchanged.",
         {"v": [-2.5, 0.001, 1.5e10, -3e-8, 0]}, {"r": "#constants:v"}, {"r": A([-2.5, 0.001, 1.5e10, -3e-8, 0])},
         "The report is the five numbers given.",
         tolerances={"absolute": 1e-9, "relative": 1e-9},
         tolerance_note="tight (1e-9 absolute and relative): the values are copied, not computed.")
    case("Several reports in one document, each referencing a different constant.",
         {"a": 1, "b": [2, 3], "c": [[4, 5], [6, 7]]}, {"ra": "#constants:a", "rb": "#constants:b", "rc": "#constants:c"},
         {"ra": A(1.0), "rb": A([2.0, 3.0]), "rc": A([[4.0, 5.0], [6.0, 7.0]])},
         "Each report is its constant unchanged: 1; the vector 2, 3; and the 2 x 2 matrix 4, 5 / 6, 7.")

    # ---- special numbers and booleans
    case("The strings nan, inf and -inf stand for NaN and the infinities when they appear among numbers.",
         {"v": [1, "nan", "inf", "-inf", 2]}, {"r": "#constants:v"}, {"r": A([1.0, NAN, INF, -INF, 2.0])},
         "Entries 0 and 4 are 1 and 2; entry 1 is NaN, entry 2 is +infinity and entry 3 is -infinity.")
    case("The special strings are recognised in any capitalisation.",
         {"v": ["NaN", "NAN", "Inf", "INF", "-Inf", "-INF"]}, {"r": "#constants:v"},
         {"r": A([NAN, NAN, INF, INF, -INF, -INF])},
         "Every entry is one of nan, inf, -inf in some capitalisation, so the list is numeric: NaN, NaN, +inf, +inf, "
         "-inf, -inf.")
    case("Special strings inside a two-dimensional constant.",
         {"m": [[1, "inf"], ["nan", 4]]}, {"r": "#constants:m"}, {"r": A([[1.0, INF], [NAN, 4.0]])},
         "The 2 x 2 matrix with +infinity at row 0 column 1 and NaN at row 1 column 0.")
    case("Booleans are parsed numerically: true is 1 and false is 0.",
         {"flag": True, "flags": [True, False, True]}, {"rs": "#constants:flag", "rv": "#constants:flags"},
         {"rs": A(1.0), "rv": A([1.0, 0.0, 1.0])},
         "The scalar is 1 and the vector is 1, 0, 1.",
         notes="The specification says booleans 'may be parsed numerically as 1 or 0', so this case tests the "
               "behaviour that is allowed, not one that is required.")

    # ---- strings
    case("A string constant is reported as string data.",
         {"s": "hello"}, {"r": "#constants:s"}, {"r": A("hello")}, "The report is the scalar string hello.")
    case("A list of strings is a one-dimensional string result.",
         {"s": ["alpha", "beta", "gamma"]}, {"r": "#constants:s"}, {"r": A(["alpha", "beta", "gamma"])},
         "The report is the three strings in order.")
    case("A list of lists of strings is a two-dimensional string result.",
         {"s": [["a", "b", "c"], ["d", "e", "f"]]}, {"r": "#constants:s"}, {"r": A([["a", "b", "c"], ["d", "e", "f"]])},
         "The report is the 2 x 3 table of strings.")

    # ---- indexing, one form at a time (vector v = 10, 20, 30, 40, 50)
    idx = [
        ("[n]: a positive index selects one entry and removes the dimension.", "#constants:v[1]", A(20.0),
         "Entry 1 (0-based) of v is 20, a scalar."),
        ("[-n]: a negative index counts from the end.", "#constants:v[-1]", A(50.0), "Entry -1 is the last, 50."),
        ("[-n] other than -1.", "#constants:v[-4]", A(20.0), "Entry -4 of five is entry 1, 20."),
        ("[a:b]: entries a up to but not including b.", "#constants:v[1:3]", A([20.0, 30.0]),
         "Entries 1 and 2: 20, 30.  A range keeps the dimension."),
        ("[a:]: from a to the end.", "#constants:v[2:]", A([30.0, 40.0, 50.0]), "Entries 2, 3, 4."),
        ("[:b]: from the beginning up to b.", "#constants:v[:2]", A([10.0, 20.0]), "Entries 0 and 1."),
        ("[:]: everything.", "#constants:v[:]", A(V), "All five entries."),
        ("[a:b] with negative bounds.", "#constants:v[-3:-1]", A([30.0, 40.0]),
         "-3 is entry 2 and -1 is entry 4, not included: entries 2 and 3."),
        ("[a:b] with one negative bound.", "#constants:v[1:-1]", A([20.0, 30.0, 40.0]),
         "Entries 1, 2, 3 (-1 is entry 4, not included)."),
        ("A range of a single entry keeps the dimension.", "#constants:v[2:3]", A([30.0]),
         "Entry 2 only, but still a one-element vector, not a scalar."),
        ("Whitespace is allowed inside the brackets.", "#constants:v[ 1 : 3 ]", A([20.0, 30.0]),
         "The same as v[1:3]."),
    ]
    for what, ref, exp, how in idx:
        case(what + "  Vector v = 10, 20, 30, 40, 50.", {"v": V}, {"r": ref}, {"r": exp}, how)

    # ---- indexing a matrix (rows 1 2 3 / 4 5 6 / 7 8 9 / 10 11 12)
    midx = [
        ("One index on a matrix selects a row.", "#constants:m[1]", A([4.0, 5.0, 6.0]), "Row 1."),
        ("A negative index on a matrix selects a row from the end.", "#constants:m[-1]", A([10.0, 11.0, 12.0]),
         "The last row."),
        ("A range on a matrix selects rows and keeps both dimensions.", "#constants:m[1:3]",
         A([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), "Rows 1 and 2, a 2 x 3 matrix."),
        ("Comma form with two integers selects one entry.", "#constants:m[1, 2]", A(6.0), "Row 1, column 2: 6."),
        ("Comma form with a range and an integer, like numpy m[a:b, n].", "#constants:m[1:3, 0]", A([4.0, 7.0]),
         "Column 0 of rows 1 and 2: 4, 7."),
        ("Comma form with an integer and a range.", "#constants:m[2, 1:]", A([8.0, 9.0]), "Row 2, columns 1 and 2."),
        ("Comma form with two ranges.", "#constants:m[0:2, 1:3]", A([[2.0, 3.0], [5.0, 6.0]]),
         "Rows 0 and 1, columns 1 and 2."),
        ("Comma form with negative indices.", "#constants:m[-1, -1]", A(12.0), "Last row, last column."),
        ("An open range in each position of the comma form.", "#constants:m[:, 1]", A([2.0, 5.0, 8.0, 11.0]),
         "Column 1 of every row."),
        ("Chained brackets with two integers equal the comma form.", "#constants:m[3][1]", A(11.0),
         "m[3] is row 3 (10, 11, 12) and [1] of that is 11."),
        ("A range followed by an integer: the second bracket applies to the result of the first.",
         "#constants:m[1:3][0]", A([4.0, 5.0, 6.0]),
         "m[1:3] is rows 1 and 2 and keeps both dimensions, so [0] is its first row, which is row 1."),
        ("An integer followed by a range.", "#constants:m[2][0:2]", A([7.0, 8.0]), "m[2] is row 2; [0:2] its first two."),
        ("Whitespace is allowed around the comma and the colon.", "#constants:m[ 0 : 2 , 1 ]", A([2.0, 5.0]),
         "The same as m[0:2, 1]: column 1 of rows 0 and 1."),
    ]
    for what, ref, exp, how in midx:
        case(what + "  Matrix m has rows 1 2 3 / 4 5 6 / 7 8 9 / 10 11 12.", {"m": M}, {"r": ref}, {"r": exp}, how)

    # ---- indexing a three-dimensional array, t[i][j][k] = 6 i + 2 j + k + 1
    case("Comma form with three indices on a three-dimensional constant.  t is 2 x 3 x 2, t[i][j][k] = 6i + 2j + k + 1.",
         {"t": T}, {"r": "#constants:t[1, 2, 0]"}, {"r": A(11.0)}, "6 + 4 + 0 + 1 = 11.")
    case("Mixed range, open range and integer on a three-dimensional constant (t as in the previous case).",
         {"t": T}, {"r": "#constants:t[:, 1:, 0]"}, {"r": A([[3.0, 5.0], [9.0, 11.0]])},
         "First dimension all (2 entries), second dimension 1 and 2 (2 entries), third dimension 0 (removed): "
         "[[t[0][1][0], t[0][2][0]], [t[1][1][0], t[1][2][0]]] = [[3, 5], [9, 11]].")
    case("Fewer indices than dimensions in the comma form leave the remaining dimensions whole (t as above).",
         {"t": T}, {"r": "#constants:t[1, 0]"}, {"r": A([7.0, 8.0])}, "t[1][0] is the pair 7, 8.")

    # ---- constants referencing constants
    case("A constant may be defined by indexing an earlier constant.",
         {"v": V, "w": "#constants:v[1:4]"}, {"r": "#constants:w"}, {"r": A([20.0, 30.0, 40.0])},
         "w is v[1:4] = 20, 30, 40.")
    case("A constant defined as another constant, then indexed in a report.",
         {"m": M, "n": "#constants:m"}, {"r": "#constants:n[2, 0]"}, {"r": A(7.0)},
         "n is a copy of m and n[2, 0] is row 2 column 0: 7.")
    # (A list holding a reference, such as ["#constants:a", 6], is not covered: the specification does not say what it
    # means; see SED2/TODO.md "References inside list and dictionary constants".)

    # ---- dictionary constants: their keys are labels of the first dimension.  d = {a: 1, b: 2, c: 3}
    D = {"a": 1, "b": 2, "c": 3}
    DL = {"a": [1, 2, 3], "b": [4, 5, 6]}
    # Only labels index a dictionary constant: SEDBase-0012 requires an array for an integer index and an object for a
    # label, so d[0] and d[1:3] are invalid documents (they belong with the validation tests).
    labeled = [
        ("A label in single quotes selects one entry of a dictionary constant.", D, "#constants:d['b']", A(2.0),
         "The entry labelled b is 2."),
        ("A label in double quotes works the same.", D, '#constants:d["c"]', A(3.0), "The entry labelled c is 3."),
        ("A label selects a whole list entry of a dictionary constant.", DL, "#constants:d['b']", A([4.0, 5.0, 6.0]),
         "The entry labelled b is the list 4, 5, 6."),
        ("A label followed by a position (chained brackets).", DL, "#constants:d['b'][1]", A(5.0),
         "d['b'] is 4, 5, 6 and [1] of that is 5."),
        ("A label and a position in the comma form.", DL, "#constants:d['a', 2]", A(3.0),
         "Entry a is 1, 2, 3 and index 2 of it is 3."),
        ("A label and a range in the comma form.", DL, "#constants:d['b', 0:2]", A([4.0, 5.0]),
         "Entry b is 4, 5, 6 and 0:2 selects 4, 5."),
    ]
    for what, const, ref, exp, how in labeled:
        case(what + "  d = " + str(const).replace("'", '"') + ".", {"d": const}, {"r": ref}, {"r": exp}, how)

    # ---- unusual but valid containers
    case("A dictionary constant that no report uses does not disturb the others.",
         {"d": {"k1": 1, "k2": 2}, "x": 4}, {"r": "#constants:x"}, {"r": A(4.0)},
         "The report is x, 4; the dictionary is carried but not reported.")
