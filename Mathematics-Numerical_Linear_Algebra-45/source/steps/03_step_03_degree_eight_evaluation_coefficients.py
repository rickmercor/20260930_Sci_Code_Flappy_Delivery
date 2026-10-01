"""
Solve for the ten evaluation coefficients that let an arbitrary degree-eight matrix polynomial be assembled from three non-scalar multiplications.

Writing the polynomial as S5 = f0*I + f1*X + f2*X**2 + f4*S4 with S2 = X**2, S3 = S2*(c1*S1 + S2) and S4 = (d0*I + d1*X + d2*X**2 + S3)*(e1*X + e2*X**2 + S3), and matching monomials against b0..b8 gives nine equations in the ten unknowns c1, d0, d1, d2, e1, e2, f0, f1, f2, f4. The system is left deliberately underdetermined so that a tenth condition can force the discriminant of the quadratic for e2 to equal one, which makes every coefficient real and the construction well defined whenever b8 is non-zero.

Returns
-------
np.ndarray, float, shape (10,): the evaluation coefficients [c1, d0, d1, d2, e1, e2, f0, f1, f2, f4] of the three-multiplication scheme.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def degree_eight_evaluation_coefficients(coeffs: np.ndarray) -> np.ndarray:
    """Solve the coefficient-matching system of the three-multiplication scheme.

    Parameters
    ----------
    coeffs : np.ndarray
        Shape (9,) array [b0, b1, ..., b8] of monomial coefficients of a
        polynomial of degree exactly eight, so that b8 is non-zero.

    Returns
    -------
    evaluation : np.ndarray
        Shape (10,) float array [c1, d0, d1, d2, e1, e2, f0, f1, f2, f4] of
        evaluation coefficients, in that order.

    Raises
    ------
    ValueError
        If ``coeffs`` is not a one-dimensional array of length 9, if any of
        its entries is not finite, or if its leading entry b8 is zero, in
        which case the polynomial is not of degree exactly eight and the
        scheme is undefined. The function must raise rather than return a
        placeholder, a NaN or a lower-degree fallback.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(10, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_degree_eight_evaluation_coefficients(coeffs: np.ndarray) -> np.ndarray:
    import numpy as np

    b = np.asarray(coeffs, dtype=float)
    if b.ndim != 1 or b.shape[0] != 9:
        raise ValueError("coeffs must be a one-dimensional array of length 9")
    if not np.all(np.isfinite(b)):
        raise ValueError("coeffs must be finite")
    if b[8] == 0.0:
        raise ValueError("the polynomial must have degree exactly eight (b8 != 0)")

    # The leading three matched monomials fix f4, c1 and the two sums
    # t2 = d2 + e2 and t1 = d1 + e1 outright.
    f4 = b[8]
    c1 = b[7] / (2.0 * f4)
    t2 = b[6] / f4 - c1 ** 2
    t1 = b[5] / f4 - c1 * t2

    # Tenth condition: the discriminant of the quadratic for e2 is set to one,
    # which both removes the square root and pins d0.
    d0 = 0.25 * (1.0 - t2 ** 2 + 4.0 * b[4] / f4 - 4.0 * c1 * t1)
    e2 = 0.5 * (t2 + 1.0)
    d2 = 0.5 * (t2 - 1.0)

    # With e2 - d2 = 1 the remaining split of t1 is explicit.
    e1 = c1 * d0 + t1 * e2 - b[3] / f4
    d1 = t1 - e1

    f2 = b[2] - f4 * (d0 * e2 + d1 * e1)
    f1 = b[1] - f4 * d0 * e1
    f0 = b[0]

    return np.array([c1, d0, d1, d2, e1, e2, f0, f1, f2, f4], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values obtained independently of the oracle. For the pure
        # power x**8 the matching system collapses to c1 = 0, t2 = t1 = 0,
        # hence d0 = 1/4, e2 = 1/2, d2 = -1/2, e1 = d1 = 0, f2 = -1/8, f1 = 0,
        # f0 = 0 and f4 = 1; the alternating probe separates all ten entries.
        {
            "setup": """import numpy as np
b = np.zeros(9); b[8] = 1.0
exact = np.array([0.0, 0.25, 0.0, -0.5, 0.0, 0.5, 0.0, 0.0, -0.125, 1.0])
w = np.array([(-3.0) ** j for j in range(10)])
EXPECTED = float(np.dot(w, exact))
""",
            "call": ("float(np.dot(w, degree_eight_evaluation_coefficients(b))"
                     " + 1e6 * degree_eight_evaluation_coefficients(b)[0])"),
            "gold_call": "float(EXPECTED + 1e6 * exact[0])",
        },
        # --- Pinned: the scheme must reproduce the polynomial itself. Assembling
        # the three products from the returned coefficients at a scalar argument
        # has to return the plain Horner value, which no oracle is needed to
        # state. A scheme that solved a different tenth condition still passes
        # only if it is genuinely an evaluation of the same polynomial.
        {
            "setup": """import numpy as np
b = np.array([0.0, 0.0, 0.0, 56.0, -210.0, 336.0, -280.0, 120.0, -21.0])
x = 0.37
def assemble(k, x):
    c1, d0, d1, d2, e1, e2, f0, f1, f2, f4 = [float(v) for v in k]
    s2 = x * x
    s3 = s2 * (c1 * x + s2)
    s4 = (d0 + d1 * x + d2 * s2 + s3) * (e1 * x + e2 * s2 + s3)
    return f0 + f1 * x + f2 * s2 + f4 * s4
EXPECTED = float(sum(b[i] * x ** i for i in range(9)))
""",
            "call": ("float(assemble(degree_eight_evaluation_coefficients(b), x)"
                     " + 1000.0 * degree_eight_evaluation_coefficients(b)[0])"),
            "gold_call": "float(EXPECTED + 1000.0 * b[7] / (2.0 * b[8]))",
        },
        # --- Valid: a tabulated member of the component-polynomial family ---
        {
            "setup": """import numpy as np
b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 56.0, -140.0, 120.0, -35.0])
w = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
""",
            "call": ("float(degree_eight_evaluation_coefficients(b)[0]"
                     " + 1e-12 * np.dot(w, degree_eight_evaluation_coefficients(b)))"),
            "gold_call": ("float(_oracle_degree_eight_evaluation_coefficients(b)[0]"
                          " + 1e-12 * np.dot(w, _oracle_degree_eight_evaluation_coefficients(b)))"),
        },
        # --- Boundary: a polynomial with a vanishing seventh-degree term, which
        # is one of the cases that break the original nine-unknown ansatz ---
        {
            "setup": """import numpy as np
b = np.array([0.5, -1.25, 2.0, 0.0, -3.5, 0.0, 4.25, 0.0, 1.75])
w = np.array([(-1.5) ** j for j in range(10)])
""",
            "call": ("float(np.dot(w, degree_eight_evaluation_coefficients(b))"
                     " + 1e6 * degree_eight_evaluation_coefficients(b)[0])"),
            "gold_call": ("float(np.dot(w, _oracle_degree_eight_evaluation_coefficients(b))"
                          " + 1e6 * _oracle_degree_eight_evaluation_coefficients(b)[0])"),
        },
        # --- Edge: a polynomial whose leading coefficient is small and negative ---
        {
            "setup": """import numpy as np
b = np.array([-2.0, 3.0, -0.5, 1.5, -4.0, 6.0, -2.5, 0.75, -0.03125])
w = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
""",
            "call": ("float(np.dot(w, degree_eight_evaluation_coefficients(b))"
                     " + 1e6 * degree_eight_evaluation_coefficients(b)[0])"),
            "gold_call": ("float(np.dot(w, _oracle_degree_eight_evaluation_coefficients(b))"
                          " + 1e6 * _oracle_degree_eight_evaluation_coefficients(b)[0])"),
        },
        # --- Invalid: the leading coefficient vanishes ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        degree_eight_evaluation_coefficients(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_degree_eight_evaluation_coefficients(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: wrong number of monomial coefficients ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        degree_eight_evaluation_coefficients(np.ones(7))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_degree_eight_evaluation_coefficients(np.ones(7))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
