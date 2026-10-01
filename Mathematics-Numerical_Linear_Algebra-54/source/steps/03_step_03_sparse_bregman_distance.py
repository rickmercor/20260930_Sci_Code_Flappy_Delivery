"""
Evaluate the Bregman distance, with respect to the sparsity-promoting strongly convex objective f(x) = lam*||x||_1 + (1/2)||x||_2^2, from the primal point encoded by a dual vector x_star to a target point y.

The paper measures progress in the Bregman distance D_f^{x*}(x, y) = f(y) - f(x) - <x*, y - x> with respect to the objective f(x) = lam*||x||_1 + (1/2)||x||_2^2 and a subgradient x* of f at x (Definition 3.1). The method keeps a dual iterate x* and recovers the primal iterate as x = grad f*(x*), where f* is the Fenchel conjugate of f; for this f the paper identifies grad f* with a componentwise soft-shrinkage map and gives f* in closed form, and it rewrites the Bregman distance through Fenchel's equality as D_f^{x*}(x, y) = f*(x*) - <x*, y> + f(y) (Section 3). Consult Section 3 of the paper for the exact conjugate and shrinkage map.

Returns
-------
float — the Bregman distance D_f^{x*}(x, y), as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sparse_bregman_distance(x_star: "np.ndarray", y: "np.ndarray",
                            lam: float) -> float:
    '''Compute D_f^{x*}(x, y) for f(x) = lam*||x||_1 + (1/2)||x||_2^2, where
    the primal point x is the one encoded by the dual vector x_star, i.e.
    x = grad f*(x_star).

    Parameters
    ----------
    x_star : np.ndarray
        (n,) dual vector (a subgradient of f at the primal point x it
        encodes).
    y : np.ndarray
        (n,) target point.
    lam : float
        Sparsity parameter lam >= 0 of the objective f.

    Returns
    -------
    dist : float
        The Bregman distance D_f^{x*}(x, y) as a native Python float
        (nonnegative).

    Raises
    ------
    ValueError
        If x_star and y do not have the same 1D shape, or if lam is
        negative.
    '''
    return dist  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _soft_shrink(v: "np.ndarray", lam: float) -> "np.ndarray":
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)


def _oracle_sparse_bregman_distance(x_star: "np.ndarray", y: "np.ndarray",
                                    lam: float) -> float:
    x_star = np.asarray(x_star, dtype=float)
    y = np.asarray(y, dtype=float)
    if x_star.ndim != 1 or x_star.shape != y.shape:
        raise ValueError("x_star and y must be 1D arrays of the same shape")
    if lam < 0:
        raise ValueError("lam must be nonnegative")
    # Section 3: f*(x*) = (1/2)||S_lam(x*)||^2 and
    # D_f^{x*}(x, y) = f*(x*) - <x*, y> + f(y).
    f_conj = 0.5 * float(np.sum(_soft_shrink(x_star, lam) ** 2))
    f_y = lam * float(np.sum(np.abs(y))) + 0.5 * float(np.sum(y ** 2))
    return float(f_conj - float(x_star @ y) + f_y)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the exact problem instance at initialization
        #     (x_star = 0, target = x_hat, lam = 0.1) ---
        {
            "setup": """import numpy as np
x_star = np.zeros(4)
y = np.array([2.0, 0.0, -1.0, 0.0])
lam = 0.1
""",
            "call": "sparse_bregman_distance(x_star, y, lam)",
            "gold_call": "_oracle_sparse_bregman_distance(x_star, y, lam)",
        },
        # --- Normal: a dual vector with entries on both sides of the
        #     shrinkage threshold, at a nonzero target. ---
        {
            "setup": """import numpy as np
x_star = np.array([1.1376524076, 0.2736953563, -0.0745495891, 0.5725379394])
y = np.array([2.0, 0.0, -1.0, 0.0])
lam = 0.1
""",
            "call": "sparse_bregman_distance(x_star, y, lam)",
            "gold_call": "_oracle_sparse_bregman_distance(x_star, y, lam)",
        },
        # --- Boundary: lam = 0 (Euclidean case), where the Bregman distance
        #     is exactly half the squared Euclidean distance. ---
        {
            "setup": """import numpy as np
x_star = np.array([1.0, -2.0, 0.5])
y = np.array([0.0, 1.0, 0.5])
lam = 0.0
""",
            "call": "sparse_bregman_distance(x_star, y, lam)",
            "gold_call": "_oracle_sparse_bregman_distance(x_star, y, lam)",
        },
        # --- Boundary: target equals the encoded primal point, which must
        #     give a zero distance even though x_star itself is nonzero. ---
        {
            "setup": """import numpy as np
lam = 0.3
x_star = np.array([1.3, -0.8, 0.2])
y = np.sign(x_star) * np.maximum(np.abs(x_star) - lam, 0.0)
""",
            "call": "sparse_bregman_distance(x_star, y, lam)",
            "gold_call": "_oracle_sparse_bregman_distance(x_star, y, lam)",
        },
        # --- Edge: every dual entry lies strictly inside the shrinkage
        #     band (encoded primal point is 0) while the target is far away. ---
        {
            "setup": """import numpy as np
x_star = np.array([0.4, -0.3, 0.1])
y = np.array([5.0, -5.0, 5.0])
lam = 0.5
""",
            "call": "sparse_bregman_distance(x_star, y, lam)",
            "gold_call": "_oracle_sparse_bregman_distance(x_star, y, lam)",
        },
        # --- Invalid: shape mismatch -> ValueError ---
        {
            "setup": """import numpy as np
x_star = np.array([1.0, 2.0])
y = np.array([1.0, 2.0, 3.0])
lam = 0.1
def run_model():
    try:
        sparse_bregman_distance(x_star, y, lam)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sparse_bregman_distance(x_star, y, lam)
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
x_star = np.array([1.0, 2.0])
y = np.array([1.0, 2.0])
lam = -0.1
def run_model():
    try:
        sparse_bregman_distance(x_star, y, lam)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sparse_bregman_distance(x_star, y, lam)
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
