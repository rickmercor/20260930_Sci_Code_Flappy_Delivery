"""
Apply a degree-eight matrix polynomial to a square matrix using exactly three non-scalar multiplications and three matrix-sized workspace slots.

Starting from the input in the first slot, the three products are the square of the input, the square of the running second slot and one final product of the second and third slots, with every other operation a scaled matrix addition. The rearrangement scalars restore the two factors that the second product would otherwise have spoiled, so the polynomial is completed as f4 times that final product added to f0*I + f1*X + f2*X**2.

Returns
-------
np.ndarray, float, shape (n, n): the degree-eight polynomial evaluated at the input matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def apply_degree_eight_polynomial(matrix: np.ndarray, evaluation: np.ndarray,
                                  scalars: np.ndarray,
                                  drop_tolerance: float = 0.0) -> np.ndarray:
    """Evaluate a degree-eight matrix polynomial with three matrix products.

    Parameters
    ----------
    matrix : np.ndarray
        Shape (n, n) square float array, the argument of the polynomial.
    evaluation : np.ndarray
        Shape (10,) array [c1, d0, d1, d2, e1, e2, f0, f1, f2, f4] of
        evaluation coefficients, in that order.
    scalars : np.ndarray
        Shape (4,) array [r1, r2, r3, r4] of workspace rearrangement scalars.
    drop_tolerance : float
        Finite nonnegative threshold. Zero disables filtering. When positive,
        every raw product P is symmetrized as 0.5*(P + P.T), then entries
        whose absolute value is strictly below the threshold are set to zero,
        before any subsequent scaled-matrix addition.

    Returns
    -------
    result : np.ndarray
        Shape (n, n) float array holding the polynomial evaluated at
        ``matrix``.

    Raises
    ------
    ValueError
        If ``matrix`` is not a non-empty square two-dimensional array or has
        a non-finite entry; if ``evaluation`` is not a finite
        one-dimensional array of length 10; if ``scalars`` is not a finite
        one-dimensional array of length 4; if ``drop_tolerance`` is not a
        finite nonnegative real scalar; or if filtering is requested for a
        matrix that is not symmetric to absolute tolerance 1e-10. The
        function must raise rather than return a placeholder or broadcast a
        rectangular input.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros_like(np.asarray(matrix, dtype=float))  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_apply_degree_eight_polynomial(matrix: np.ndarray, evaluation: np.ndarray,
                                          scalars: np.ndarray,
                                          drop_tolerance: float = 0.0) -> np.ndarray:
    import numpy as np

    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError("matrix must be a non-empty square two-dimensional array")
    if not np.all(np.isfinite(a)):
        raise ValueError("matrix must be finite")

    k = np.asarray(evaluation, dtype=float)
    if k.ndim != 1 or k.shape[0] != 10 or not np.all(np.isfinite(k)):
        raise ValueError("evaluation must be a finite array of length 10")
    r = np.asarray(scalars, dtype=float)
    if r.ndim != 1 or r.shape[0] != 4 or not np.all(np.isfinite(r)):
        raise ValueError("scalars must be a finite array of length 4")
    if isinstance(drop_tolerance, bool) or not isinstance(
        drop_tolerance, (int, float, np.integer, np.floating)
    ):
        raise ValueError("drop_tolerance must be a real number")
    tolerance = float(drop_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("drop_tolerance must be finite and nonnegative")
    if tolerance > 0.0:
        if not np.allclose(a, a.T, rtol=0.0, atol=1e-10):
            raise ValueError("matrix must be symmetric when filtering is requested")
        a = 0.5 * (a + a.T)

    c1, d0 = k[0], k[1]
    f0, f2, f4 = k[6], k[8], k[9]
    r1, r2, r3, r4 = r[0], r[1], r[2], r[3]

    eye = np.eye(a.shape[0], dtype=float)
    m1 = a.copy()

    def multiply(left, right):
        product = left @ right
        if tolerance > 0.0:
            product = 0.5 * (product + product.T)
            product[np.abs(product) < tolerance] = 0.0
        return product

    m2 = multiply(m1, m1)              # first non-scalar multiplication
    m2 = m2 + 0.5 * c1 * m1
    m3 = multiply(m2, m2)              # second non-scalar multiplication

    # Rebuild the two factors of the final product inside the same three slots.
    m3 = m3 + r1 * m1
    m3 = m3 + r2 * m2
    m2 = m2 + r3 * m1
    m1 = r4 * m1 + f2 * m2 + f0 * eye
    m2 = m2 + m3
    m3 = m3 + d0 * eye

    m1 = f4 * multiply(m2, m3) + m1    # third non-scalar multiplication
    return m1

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values obtained independently of the oracle: for the pure
        # power x**8 the scheme must return the eighth matrix power, which is
        # reproduced here by repeated squaring. A scheme that misplaces one of
        # the rearrangement scalars leaves a stray low-order term and fails.
        {
            "setup": """import numpy as np
a = np.array([[0.6, 0.2, -0.1], [0.2, 0.45, 0.15], [-0.1, 0.15, 0.8]])
k = np.array([0.0, 0.25, 0.0, -0.5, 0.0, 0.5, 0.0, 0.0, -0.125, 1.0])
r = np.array([0.0, -0.5, 0.0, 0.0])
a2 = a @ a
a4 = a2 @ a2
EXPECTED = float(np.trace(a4 @ a4))
""",
            "call": "float(np.trace(apply_degree_eight_polynomial(a, k, r)))",
            "gold_call": "EXPECTED",
        },
        # --- Pinned: the scheme must agree with a direct monomial expansion of
        # the same polynomial, built here from explicit matrix powers.
        {
            "setup": """import numpy as np
b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 56.0, -140.0, 120.0, -35.0])
k = np.array([-1.7142857142857142, 0.34429404405034876, 0.5835068720998319,
              0.030612244897959162, -0.36426488962280604, 1.0306122448979591,
              0.0, -4.389498119837221, 4.979890698105264, -35.0])
r = np.array([-0.019991670137442735, -0.7040816326530612,
              -0.09062890462387913, 0.33030166042484756])
a = np.array([[0.31, 0.08, 0.02], [0.08, 0.52, -0.06], [0.02, -0.06, 0.74]])
w = np.array([1.0, 2.0, 3.0])
direct = sum(b[i] * np.linalg.matrix_power(a, i) for i in range(9))
EXPECTED = float(w @ direct @ w)
""",
            "call": "float(w @ apply_degree_eight_polynomial(a, k, r) @ w)",
            "gold_call": "EXPECTED",
        },
        # --- Valid: a non-symmetric argument, where the ordering of the final
        # product matters (normal scenario) ---
        {
            "setup": """import numpy as np
def rearrange(k):
    c1, d1, d2, e1, f1, f2 = k[0], k[2], k[3], k[4], k[7], k[8]
    r2 = d2 - 0.25 * c1 ** 2
    return np.array([d1 - 0.5 * c1 * r2, r2, e1 - d1 - 0.5 * c1,
                     f1 - f2 * (e1 - d1)])
a = np.array([[0.2, 0.7, 0.1], [-0.3, 0.4, 0.25], [0.05, -0.15, 0.6]])
k = np.array([1.3, -0.4, 0.9, 2.1, -1.7, 3.1, 0.5, 2.2, -0.8, 1.4])
r = rearrange(k)
w = np.array([1.0, -2.0, 3.0])
""",
            "call": "float(w @ apply_degree_eight_polynomial(a, k, r) @ w)",
            "gold_call": "float(w @ _oracle_apply_degree_eight_polynomial(a, k, r) @ w)",
        },
        # --- Boundary: a one-by-one matrix, where the whole scheme reduces to
        # scalar arithmetic ---
        {
            "setup": """import numpy as np
a = np.array([[0.43]])
k = np.array([-2.2857142857142856, 0.19618908789657, 0.4981257809215,
              0.8877551020408163, -0.5541024573206, 1.8877551020408163,
              0.0, 3.8048099493, -3.3020481318, 35.0])
r = np.array([0.019991670137442735, -0.41836734693877553,
              0.09062890462387913, 0.33030166042484756])
""",
            "call": "float(apply_degree_eight_polynomial(a, k, r)[0, 0])",
            "gold_call": "float(_oracle_apply_degree_eight_polynomial(a, k, r)[0, 0])",
        },
        # --- Valid: positive filtering must occur at each raw product, before
        # the cancellation-sensitive additions rebuild the final factors. ---
        {
            "setup": """import numpy as np
a = np.array([[0.31, 0.008, 0.002], [0.008, 0.52, -0.006],
              [0.002, -0.006, 0.74]])
k = np.array([-1.7142857142857142, 0.34429404405034876,
              0.5835068720998319, 0.030612244897959162,
              -0.36426488962280604, 1.0306122448979591,
              0.0, -4.389498119837221, 4.979890698105264, -35.0])
r = np.array([-0.019991670137442735, -0.7040816326530612,
              -0.09062890462387913, 0.33030166042484756])
w = np.array([1.0, -2.0, 3.0])
tau = 1.0e-3
""",
            "call": "float(w @ apply_degree_eight_polynomial(a, k, r, tau) @ w)",
            "gold_call": "float(w @ _oracle_apply_degree_eight_polynomial(a, k, r, tau) @ w)",
        },
        # --- Edge: a larger deterministic matrix with entries spanning both
        # signs, reduced to a single weighted trace ---
        {
            "setup": """import numpy as np
def rearrange(k):
    c1, d1, d2, e1, f1, f2 = k[0], k[2], k[3], k[4], k[7], k[8]
    r2 = d2 - 0.25 * c1 ** 2
    return np.array([d1 - 0.5 * c1 * r2, r2, e1 - d1 - 0.5 * c1,
                     f1 - f2 * (e1 - d1)])
n = 7
idx = np.arange(n)
a = 0.4 * np.cos(0.7 * np.add.outer(idx, 2 * idx)) - 0.1 * np.eye(n)
k = np.array([0.9, 0.15, -0.6, 1.1, 0.35, 2.1, -0.2, 0.8, 1.3, -2.5])
r = rearrange(k)
""",
            "call": "float(np.trace(apply_degree_eight_polynomial(a, k, r) @ a))",
            "gold_call": "float(np.trace(_oracle_apply_degree_eight_polynomial(a, k, r) @ a))",
        },
        # --- Invalid: a rectangular argument ---
        {
            "setup": """import numpy as np
k = np.ones(10); r = np.ones(4)
def run_model():
    try:
        apply_degree_eight_polynomial(np.ones((3, 4)), k, r)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_apply_degree_eight_polynomial(np.ones((3, 4)), k, r)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: wrong number of rearrangement scalars ---
        {
            "setup": """import numpy as np
a = np.eye(3); k = np.ones(10)
def run_model():
    try:
        apply_degree_eight_polynomial(a, k, np.ones(3))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_apply_degree_eight_polynomial(a, k, np.ones(3))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative filtering tolerance is not meaningful. ---
        {
            "setup": """import numpy as np
a = np.eye(3); k = np.ones(10); r = np.ones(4)
def run_model():
    try:
        apply_degree_eight_polynomial(a, k, r, -1e-6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_apply_degree_eight_polynomial(a, k, r, -1e-6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: symmetrizing a genuinely non-symmetric input would
        # silently change the matrix-polynomial contract. ---
        {
            "setup": """import numpy as np
a = np.array([[0.2, 0.3], [-0.1, 0.4]])
k = np.ones(10); r = np.ones(4)
def run_model():
    try:
        apply_degree_eight_polynomial(a, k, r, 1e-6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_apply_degree_eight_polynomial(a, k, r, 1e-6)
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
