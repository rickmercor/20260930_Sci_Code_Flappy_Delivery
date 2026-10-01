"""
Compute the exact value of the error-bound constant gamma that enters the paper's adaptive step-size rule, for the sparsity-promoting objective, from the coefficient matrix, the exact solution and the sparsity parameter.

The paper's step-size rule depends on a constant gamma defined through its error-bound assumption (Assumption 3.2): the squared residual ||Ax - b||_2^2 dominates the Bregman distance to the exact solution up to a constant theta(x_hat), and gamma is theta(x_hat) rescaled by the squared Frobenius norm of A. The paper does not derive theta itself; it states that for f(x) = lam*||x||_1 + (1/2)||x||_2^2 an explicit constant is available in the reference it cites (Schöpfer and Lorenz, Linear convergence of the randomized sparse Kaczmarz method, Math. Program. 2019, Lemma 3.1), where the bound is written in the reverse direction (Bregman distance bounded by a constant times the squared residual) and the constant depends on the smallest positive singular value over column submatrices of A, the smallest nonzero absolute entry of the exact solution, and lam. Consult both sources for the exact constant and for how it maps onto the paper's gamma; neither is restated here.

Returns
-------
float — gamma, the paper's error-bound constant for the given A, x_hat and lam, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def error_bound_gamma(A: "np.ndarray", x_hat: "np.ndarray", lam: float) -> float:
    '''Compute the paper's error-bound constant gamma (Assumption 3.2, with
    gamma := theta(x_hat) / ||A||_F^2) for f(x) = lam*||x||_1 + (1/2)||x||_2^2.

    Parameters
    ----------
    A : np.ndarray
        (m, n) coefficient matrix, not identically zero.
    x_hat : np.ndarray
        (n,) exact (minimum-f) solution of A x = b, with at least one
        nonzero entry.
    lam : float
        Sparsity parameter lam >= 0 of the objective f.

    Returns
    -------
    gamma : float
        The paper's constant gamma, as a native Python float, using the
        explicit error-bound constant of the cited reference for this
        objective (not restated here). The singular-value quantity in that
        constant is the minimum, over all nonempty column subsets J of A
        with A_J nonzero, of the smallest positive singular value of the
        column submatrix A_J.

    Raises
    ------
    ValueError
        If A is not a 2D array or is identically zero, if x_hat does not
        have shape (n,) matching A's columns or has no nonzero entry, or if
        lam is negative.
    '''
    return gamma  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools

import numpy as np

def _sigma_tilde_min(A: "np.ndarray") -> float:
    # min over nonempty column subsets J (A_J != 0) of the smallest positive
    # singular value of A_J (definition used by the cited error bound).
    m, n = A.shape
    best = np.inf
    for r in range(1, n + 1):
        for J in itertools.combinations(range(n), r):
            sub = A[:, J]
            if not np.any(sub):
                continue
            s = np.linalg.svd(sub, compute_uv=False)
            s = s[s > 1e-12 * s.max()]
            best = min(best, float(s.min()))
    return best


def _oracle_error_bound_gamma(A: "np.ndarray", x_hat: "np.ndarray", lam: float) -> float:
    A = np.asarray(A, dtype=float)
    x_hat = np.asarray(x_hat, dtype=float)
    if A.ndim != 2 or not np.any(A):
        raise ValueError("A must be a nonzero 2D array")
    m, n = A.shape
    if x_hat.shape != (n,):
        raise ValueError("x_hat must have shape (n,) matching A's columns")
    if not np.any(x_hat):
        raise ValueError("x_hat must have at least one nonzero entry")
    if lam < 0:
        raise ValueError("lam must be nonnegative")
    # Schöpfer–Lorenz (Lemma 3.1): D_f^{x*}(x, x_hat) <= gamma_SL ||Ax - b||^2 with
    #   gamma_SL = (|x_hat|_min + 2 lam) / (sigma~_min(A)^2 |x_hat|_min).
    # The paper's Assumption 3.2 is the reverse inequality, so theta = 1/gamma_SL,
    # and the paper defines gamma := theta / ||A||_F^2.
    x_min = float(np.min(np.abs(x_hat[x_hat != 0])))
    theta = _sigma_tilde_min(A) ** 2 * x_min / (x_min + 2.0 * lam)
    return float(theta / np.sum(A ** 2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the exact problem instance (full column rank, lam = 0.1) ---
        {
            "setup": """import numpy as np
A = np.array([[2.0, 1.0, 0.0, 1.0], [1.0, 3.0, 1.0, 0.0], [0.0, 1.0, 4.0, 1.0],
              [3.0, 0.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, -1.0, 2.0, 0.0]])
x_hat = np.array([2.0, 0.0, -1.0, 0.0])
lam = 0.1
""",
            "call": "error_bound_gamma(A, x_hat, lam)",
            "gold_call": "_oracle_error_bound_gamma(A, x_hat, lam)",
        },
        # --- Boundary: lam = 0 (Euclidean objective), where the constant
        #     reduces to a pure singular-value ratio. ---
        {
            "setup": """import numpy as np
A = np.array([[2.0, 1.0, 0.0, 1.0], [1.0, 3.0, 1.0, 0.0], [0.0, 1.0, 4.0, 1.0],
              [3.0, 0.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, -1.0, 2.0, 0.0]])
x_hat = np.array([2.0, 0.0, -1.0, 0.0])
lam = 0.0
""",
            "call": "error_bound_gamma(A, x_hat, lam)",
            "gold_call": "_oracle_error_bound_gamma(A, x_hat, lam)",
        },
        # --- Edge: a small nonzero entry in x_hat (|x_hat|_min = 0.05) with
        #     a sparsity parameter of the same order, so the lam-dependent
        #     factor dominates. ---
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
x_hat = np.array([1.0, 0.05])
lam = 0.1
""",
            "call": "error_bound_gamma(A, x_hat, lam)",
            "gold_call": "_oracle_error_bound_gamma(A, x_hat, lam)",
        },
        # --- Edge: rank-deficient A (two proportional columns), where the
        #     subset-minimum singular value differs from the smallest
        #     positive singular value of A itself. ---
        {
            "setup": """import numpy as np
A = np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 1.0], [0.0, 0.0, 3.0], [1.0, 2.0, -1.0]])
x_hat = np.array([0.5, 1.0, -2.0])
lam = 0.2
""",
            "call": "error_bound_gamma(A, x_hat, lam)",
            "gold_call": "_oracle_error_bound_gamma(A, x_hat, lam)",
        },
        # --- Edge: a single-column system, where the only submatrix is A
        #     itself and its singular value is the column norm. ---
        {
            "setup": """import numpy as np
A = np.array([[3.0], [4.0]])
x_hat = np.array([-2.0])
lam = 0.5
""",
            "call": "error_bound_gamma(A, x_hat, lam)",
            "gold_call": "_oracle_error_bound_gamma(A, x_hat, lam)",
        },
        # --- Invalid: x_hat identically zero -> ValueError ---
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
x_hat = np.array([0.0, 0.0])
lam = 0.1
def run_model():
    try:
        error_bound_gamma(A, x_hat, lam)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_error_bound_gamma(A, x_hat, lam)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative lam -> ValueError ---
        {
            "setup": """import numpy as np
A = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
x_hat = np.array([1.0, -1.0])
lam = -0.1
def run_model():
    try:
        error_bound_gamma(A, x_hat, lam)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_error_bound_gamma(A, x_hat, lam)
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
