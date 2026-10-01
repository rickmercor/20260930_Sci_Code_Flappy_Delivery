"""
Take one dual step along the averaged direction with the given step size and map the new dual iterate back to the primal iterate through the conjugate of the sparsity-promoting objective.

The paper's iteration is a mirror-descent-type scheme: it keeps a dual variable x*_k, moves it against the direction d_k scaled by the step eta_k, and recovers the primal iterate as x_{k+1} = grad f*(x*_{k+1}) (eq. (5)). For the objective f(x) = lam*||x||_1 + (1/2)||x||_2^2 the paper identifies grad f* with a componentwise soft-shrinkage map with threshold lam (Section 3), which is what produces sparse primal iterates. Consult eq. (5) and Section 3 of the paper for the exact update and the exact shrinkage map.

Returns
-------
tuple (np.ndarray of shape (n,), np.ndarray of shape (n,)) — (x*_{k+1}, x_{k+1}), both dtype float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bregman_dual_update(x_star: "np.ndarray", d: "np.ndarray", eta: float,
                        lam: float) -> tuple:
    '''Perform the dual step and primal recovery of eq. (5) for
    f(x) = lam*||x||_1 + (1/2)||x||_2^2.

    Parameters
    ----------
    x_star : np.ndarray
        (n,) current dual iterate x*_k.
    d : np.ndarray
        (n,) direction d_k from aabk_averaged_direction.
    eta : float
        Step size eta_k from aabk_adaptive_step, >= 0.
    lam : float
        Sparsity parameter lam >= 0 of the objective f.

    Returns
    -------
    result : tuple of (np.ndarray, np.ndarray)
        (x_star_next, x_next): the new dual iterate x*_{k+1} and the new
        primal iterate x_{k+1} = grad f*(x*_{k+1}), both of shape (n,), as
        defined in eq. (5) of the paper (not restated here).

    Raises
    ------
    ValueError
        If x_star and d do not have the same 1D shape, if eta is negative,
        or if lam is negative.
    '''
    return x_star_next, x_next  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_bregman_dual_update(x_star: "np.ndarray", d: "np.ndarray", eta: float,
                                lam: float) -> tuple:
    x_star = np.asarray(x_star, dtype=float)
    d = np.asarray(d, dtype=float)
    if x_star.ndim != 1 or x_star.shape != d.shape:
        raise ValueError("x_star and d must be 1D arrays of the same shape")
    if eta < 0:
        raise ValueError("eta must be nonnegative")
    if lam < 0:
        raise ValueError("lam must be nonnegative")
    # eq. (5): x*_{k+1} = x*_k - eta_k d_k, x_{k+1} = grad f*(x*_{k+1}) = S_lam(x*_{k+1}).
    x_star_next = x_star - eta * d
    x_next = np.sign(x_star_next) * np.maximum(np.abs(x_star_next) - lam, 0.0)
    return x_star_next, x_next

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: iteration 0 of the exact problem instance (x*_0 = 0) ---
        {
            "setup": """import numpy as np
x_star = np.zeros(4)
d = np.array([-0.0659065489, -0.1253669162, 0.1602514178, 0.0486328723])
eta = 0.5048250013072225
lam = 0.1
""",
            "call": "np.concatenate(bregman_dual_update(x_star, d, eta, lam))",
            "gold_call": "np.concatenate(_oracle_bregman_dual_update(x_star, d, eta, lam))",
        },
        # --- Normal: iteration 3 of the exact problem instance, where
        #     several dual entries cross the threshold at once. ---
        {
            "setup": """import numpy as np
x_star = np.array([0.0358159853, 0.0389386975, -0.1859316817, -0.0489007460])
d = np.array([-2.0837653277, -0.3896283929, -0.3353146135, -1.2308660207])
eta = 0.5010831161701362
lam = 0.1
""",
            "call": "np.concatenate(bregman_dual_update(x_star, d, eta, lam))",
            "gold_call": "np.concatenate(_oracle_bregman_dual_update(x_star, d, eta, lam))",
        },
        # --- Boundary: lam = 0 (plain Kaczmarz-type update, primal equals dual) ---
        {
            "setup": """import numpy as np
x_star = np.array([1.0, -2.0, 0.5])
d = np.array([0.5, 0.5, -1.0])
eta = 1.2
lam = 0.0
""",
            "call": "np.concatenate(bregman_dual_update(x_star, d, eta, lam))",
            "gold_call": "np.concatenate(_oracle_bregman_dual_update(x_star, d, eta, lam))",
        },
        # --- Boundary: a dual entry lands exactly on the threshold, and one
        #     lands exactly on its negative. ---
        {
            "setup": """import numpy as np
x_star = np.array([0.5, -0.5, 2.0])
d = np.array([0.2, -0.2, 0.5])
eta = 1.0
lam = 0.3
""",
            "call": "np.concatenate(bregman_dual_update(x_star, d, eta, lam))",
            "gold_call": "np.concatenate(_oracle_bregman_dual_update(x_star, d, eta, lam))",
        },
        # --- Edge: zero step (nothing moves, primal must still be the
        #     shrinkage of the unchanged dual). ---
        {
            "setup": """import numpy as np
x_star = np.array([0.05, -1.5, 0.3])
d = np.array([10.0, 10.0, 10.0])
eta = 0.0
lam = 0.1
""",
            "call": "np.concatenate(bregman_dual_update(x_star, d, eta, lam))",
            "gold_call": "np.concatenate(_oracle_bregman_dual_update(x_star, d, eta, lam))",
        },
        # --- Invalid: negative eta -> ValueError ---
        {
            "setup": """import numpy as np
x_star = np.array([1.0, 2.0])
d = np.array([1.0, 1.0])
eta = -0.5
lam = 0.1
def run_model():
    try:
        bregman_dual_update(x_star, d, eta, lam)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_bregman_dual_update(x_star, d, eta, lam)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: shape mismatch -> ValueError ---
        {
            "setup": """import numpy as np
x_star = np.array([1.0, 2.0])
d = np.array([1.0, 1.0, 1.0])
eta = 0.5
lam = 0.1
def run_model():
    try:
        bregman_dual_update(x_star, d, eta, lam)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_bregman_dual_update(x_star, d, eta, lam)
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
