"""
Reduce the prior mixing probabilities and the per-component shrinkage constants to the pair of numbers each non-null component contributes to the posterior inclusion probability of a single variant.

The posterior odds that a variant belongs to non-null component k rather than to the null component is the prior odds times the Gaussian marginal-likelihood ratio, which splits into a data-free shrinkage factor sqrt(lambda_k / C_k) and an exponential in the single-variant chi-square statistic. Collecting the two pieces gives the amplitude A_k = (pi_k / pi_1) * sqrt(lambda_k / C_k) and the rate B_k = n / (2 * C_k), a pure number multiplying the statistic.

Returns
-------
np.ndarray, float, shape (n_components, 2): the amplitude A_k and the rate B_k of every non-null mixture component.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def pip_curve_constants(mixture_probs: np.ndarray, scale_factors: np.ndarray,
                        n: float) -> np.ndarray:
    """Build the amplitude and rate constants of each non-null component.

    Parameters
    ----------
    mixture_probs : np.ndarray
        Shape ``(1 + n_components,)`` prior mixing probabilities, with the
        null probability first.
    scale_factors : np.ndarray
        Shape ``(n_components, 2)`` array holding the shrinkage ratio in
        column 0 and the posterior precision in column 1.
    n : float
        Sample size of the study, on the scale at which the phenotype has
        unit variance (n > 0).

    Returns
    -------
    constants : np.ndarray
        Shape ``(n_components, 2)`` float array with the amplitude in
        column 0 and the rate in column 1, one row per non-null component.

    Raises
    ------
    ValueError
        If ``n`` is not a finite real number greater than zero; if
        ``scale_factors`` does not have shape ``(n_components, 2)`` or holds
        a value that is not finite and positive; if ``mixture_probs`` does
        not hold exactly one more entry than ``scale_factors`` has rows; if
        any probability is negative or not finite; if the null probability
        is not strictly positive; or if the probabilities do not sum to 1
        within a tolerance of 1e-9.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros((np.asarray(scale_factors).shape[0], 2), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_pip_curve_constants(mixture_probs: np.ndarray, scale_factors: np.ndarray,
                                n: float) -> np.ndarray:
    import numpy as np

    probs = np.asarray(mixture_probs, dtype=float).ravel()
    factors = np.asarray(scale_factors, dtype=float)
    if isinstance(n, bool) or not isinstance(n, (int, float, np.integer, np.floating)):
        raise ValueError("n must be a real number")
    n = float(n)
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("n must be a finite number > 0")
    if factors.ndim != 2 or factors.shape[1] != 2 or factors.shape[0] < 1:
        raise ValueError("scale_factors must have shape (n_components, 2)")
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        raise ValueError("scale_factors must be finite and > 0")
    if probs.size != factors.shape[0] + 1:
        raise ValueError("mixture_probs must hold one null and one per non-null component")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("mixture_probs must be finite and non-negative")
    if probs[0] <= 0.0:
        raise ValueError("the null mixing probability must be > 0")
    if abs(float(np.sum(probs)) - 1.0) > 1e-9:
        raise ValueError("mixture_probs must sum to 1")

    lam = factors[:, 0]
    c_precision = factors[:, 1]

    # Prior odds against the null, times the shrinkage factor the Gaussian
    # marginal likelihood contributes; the rate is the coefficient of the
    # chi-square statistic in the log posterior odds and carries no variance.
    amplitude = (probs[1:] / probs[0]) * np.sqrt(lam / c_precision)
    rate = n / (2.0 * c_precision)

    constants = np.empty((lam.size, 2), dtype=float)
    constants[:, 0] = amplitude
    constants[:, 1] = rate
    return constants

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values: both columns are written out from the definitions
        # without touching the oracle. The two probes weight the amplitudes and
        # the rates differently, so swapping the columns, dropping the square
        # root, or using the prior probability instead of the prior odds all
        # fail.
        {
            "setup": """import numpy as np
probs = np.array([0.99, 0.008, 0.002])
lam = np.array([1.0e4, 1.0e2])
n = 250000.0
c = n + lam
factors = np.column_stack([lam, c])
amp_exact = (probs[1:] / probs[0]) * np.sqrt(lam / c)
rate_exact = n / (2.0 * c)
EXPECTED = float(1e3 * np.dot(amp_exact, [1.0, 5.0]) + np.dot(rate_exact, [1.0, 9.0]))
""",
            "call": ("float(1e3 * np.dot(pip_curve_constants(probs, factors, n)[:, 0], [1.0, 5.0])"
                     " + np.dot(pip_curve_constants(probs, factors, n)[:, 1], [1.0, 9.0]))"),
            "gold_call": "EXPECTED",
        },
        # --- Valid: a four-component prior with a very sparse architecture ---
        {
            "setup": """import numpy as np
probs = np.array([0.9963, 0.0030, 0.0006, 0.00008, 0.00002])
lam = np.array([2.5e5, 2.5e4, 2.5e3, 2.5e2])
n = 900000.0
factors = np.column_stack([lam, n + lam])
w = np.array([1.0, 3.0, 9.0, 27.0])
""",
            "call": ("float(1e4 * np.dot(pip_curve_constants(probs, factors, n)[:, 0], w)"
                     " + np.dot(pip_curve_constants(probs, factors, n)[:, 1], w))"),
            "gold_call": ("float(1e4 * np.dot(_oracle_pip_curve_constants(probs, factors, n)[:, 0], w)"
                          " + np.dot(_oracle_pip_curve_constants(probs, factors, n)[:, 1], w))"),
        },
        # --- Boundary: a sample so small that the rate is far below one half,
        # which is the limit it approaches as the sample grows ---
        {
            "setup": """import numpy as np
probs = np.array([0.95, 0.05])
lam = np.array([4.0e5])
n = 100.0
factors = np.column_stack([lam, n + lam])
""",
            "call": "float(pip_curve_constants(probs, factors, n)[0, 1])",
            "gold_call": "float(_oracle_pip_curve_constants(probs, factors, n)[0, 1])",
        },
        # --- Edge: a single non-null component whose prior odds exceed one ---
        {
            "setup": """import numpy as np
probs = np.array([0.4, 0.6])
lam = np.array([1.7e3])
n = 55000.0
factors = np.column_stack([lam, n + lam])
""",
            "call": "float(pip_curve_constants(probs, factors, n)[0, 0])",
            "gold_call": "float(_oracle_pip_curve_constants(probs, factors, n)[0, 0])",
        },
        # --- Invalid: mixing probabilities that do not sum to one ---
        {
            "setup": """import numpy as np
probs = np.array([0.90, 0.05])
factors = np.column_stack([np.array([1.0e4]), np.array([1.1e4])])
def run_model():
    try:
        pip_curve_constants(probs, factors, 1000.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_pip_curve_constants(probs, factors, 1000.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: one fewer mixing probability than the prior needs ---
        {
            "setup": """import numpy as np
probs = np.array([0.98, 0.02])
factors = np.column_stack([np.array([1.0e4, 1.0e3]), np.array([1.1e4, 1.1e3])])
def run_model():
    try:
        pip_curve_constants(probs, factors, 1000.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_pip_curve_constants(probs, factors, 1000.0)
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
