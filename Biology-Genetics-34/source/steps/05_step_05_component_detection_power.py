"""
Compute, for each non-null mixture component, the probability that a causal variant drawn from that component produces a chi-square statistic above the threshold set by the PIP cut-off.

A causal variant from component k contributes both prior effect variance and sampling error to the same normal deviate, so its chi-square statistic is marginally a central chi-square on one degree of freedom scaled by C_k / lambda_k rather than a noncentral one. The probability of clearing a threshold z is therefore the two-sided standard normal tail beyond sqrt(z * lambda_k / C_k).

Returns
-------
np.ndarray, float, shape (n_components,): the probability that a causal variant of each non-null component is detected at the given threshold.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def component_detection_power(scale_factors: np.ndarray, z_threshold: float) -> np.ndarray:
    """Probability that a causal variant of each component clears the threshold.

    Parameters
    ----------
    scale_factors : np.ndarray
        Shape ``(n_components, 2)`` array holding the shrinkage ratio in
        column 0 and the posterior precision in column 1.
    z_threshold : float
        Chi-square statistic at which the PIP threshold is reached
        (z_threshold >= 0).

    Returns
    -------
    power : np.ndarray
        Shape ``(n_components,)`` float array of detection probabilities, one
        per non-null mixture component.

    Raises
    ------
    ValueError
        If ``scale_factors`` does not have shape ``(n_components, 2)`` or
        holds a value that is not finite and positive; if any posterior
        precision does not strictly exceed its shrinkage ratio, which no
        positive sample size can produce; or if ``z_threshold`` is not a
        finite real number greater than or equal to zero.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(np.asarray(scale_factors).shape[0], dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_component_detection_power(scale_factors: np.ndarray,
                                      z_threshold: float) -> np.ndarray:
    import numpy as np
    from math import erfc, sqrt

    factors = np.asarray(scale_factors, dtype=float)
    if factors.ndim != 2 or factors.shape[1] != 2 or factors.shape[0] < 1:
        raise ValueError("scale_factors must have shape (n_components, 2)")
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        raise ValueError("scale_factors must be finite and > 0")
    lam = factors[:, 0]
    c_precision = factors[:, 1]
    if np.any(c_precision <= lam):
        raise ValueError("the posterior precision must exceed the shrinkage ratio")
    if isinstance(z_threshold, bool) or not isinstance(
            z_threshold, (int, float, np.integer, np.floating)):
        raise ValueError("z_threshold must be a real number")
    z_threshold = float(z_threshold)
    if not np.isfinite(z_threshold) or z_threshold < 0.0:
        raise ValueError("z_threshold must be a finite number >= 0")

    # Marginally over the prior, the statistic of a causal variant is the
    # central chi-square scaled by C_k / lambda_k, so the threshold moves to
    # z * lambda_k / C_k on the standard scale.
    scaled = z_threshold * lam / c_precision

    # Two-sided standard normal tail, written with the complementary error
    # function so that the far tail keeps its relative accuracy.
    power = np.array([erfc(sqrt(0.5 * value)) for value in scaled], dtype=float)
    return power

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values: the ratios below put the standard-scale threshold
        # at exactly 1, 0.5 and 1.5 standard deviations, whose two-sided normal
        # tails are textbook constants. A one-sided tail, or a tail taken
        # beyond sqrt(z * C / lambda) instead, misses all three.
        {
            "setup": """import numpy as np
factors = np.array([[1000.0, 4000.0], [625.0, 10000.0], [5625.0, 10000.0]])
exact = np.array([0.31731050786291415, 0.6170750774519738, 0.13361440253771614])
w = np.array([1.0, 4.0, 16.0])
EXPECTED = float(np.dot(w, exact))
""",
            "call": "float(np.dot(w, component_detection_power(factors, 4.0)))",
            "gold_call": "EXPECTED",
        },
        # --- Valid: a four-component prior at a realistic PIP threshold, where
        # the powers span two orders of magnitude ---
        {
            "setup": """import numpy as np
factors = np.array([[2.4e5, 1.94e6], [2.4e4, 1.724e6],
                    [2.4e3, 1.7024e6], [2.4e2, 1.70024e6]])
w = np.array([1.0, 2.0, 3.0, 4.0])
""",
            "call": "float(np.dot(w, component_detection_power(factors, 17.3)))",
            "gold_call": "float(np.dot(w, _oracle_component_detection_power(factors, 17.3)))",
        },
        # --- Boundary: a threshold of zero, where every causal variant is
        # detected with certainty ---
        {
            "setup": """import numpy as np
factors = np.array([[1.0e4, 5.0e5], [1.0e2, 5.0e5]])
""",
            "call": "float(np.sum(component_detection_power(factors, 0.0)))",
            "gold_call": "float(np.sum(_oracle_component_detection_power(factors, 0.0)))",
        },
        # --- Edge: a small study against a small-effect component, where the
        # tail is deep enough that a squared or exponentiated form would show ---
        {
            "setup": """import numpy as np
factors = np.array([[8.0e5, 8.3e5]])
""",
            "call": "float(component_detection_power(factors, 34.0)[0])",
            "gold_call": "float(_oracle_component_detection_power(factors, 34.0)[0])",
        },
        # --- Invalid: a posterior precision below the shrinkage ratio, which no
        # positive sample size can produce ---
        {
            "setup": """import numpy as np
factors = np.array([[1.0e4, 9.0e3]])
def run_model():
    try:
        component_detection_power(factors, 10.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_component_detection_power(factors, 10.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative threshold ---
        {
            "setup": """import numpy as np
factors = np.array([[1.0e4, 5.0e5]])
def run_model():
    try:
        component_detection_power(factors, -3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_component_detection_power(factors, -3.0)
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
