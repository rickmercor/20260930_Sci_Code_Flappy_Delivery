"""
Compute, for each non-null mixture component, the second moment of the standardised statistic restricted to the region in which the variant is detected.

The variance a detected variant carries is weighted by the size of its statistic, so the detection region contributes its second moment and not only its probability. For a standard normal deviate cut at c, integration by parts gives E[W**2 * 1(|W| > c)] = 2 * c * phi(c) + P, where phi is the standard normal density and P is the two-sided tail already available as the detection probability.

Returns
-------
np.ndarray, float, shape (n_components,): the second moment of the standardised statistic restricted to the detection region, per component.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def component_tail_second_moment(scale_factors: np.ndarray, z_threshold: float,
                                 detection_power: np.ndarray) -> np.ndarray:
    """Second moment of the standardised statistic over the detection region.

    Parameters
    ----------
    scale_factors : np.ndarray
        Shape ``(n_components, 2)`` array holding the shrinkage ratio in
        column 0 and the posterior precision in column 1.
    z_threshold : float
        Chi-square statistic at which the PIP threshold is reached
        (z_threshold >= 0).
    detection_power : np.ndarray
        Shape ``(n_components,)`` array of detection probabilities, one per
        non-null mixture component.

    Returns
    -------
    second_moment : np.ndarray
        Shape ``(n_components,)`` float array of restricted second moments,
        one per non-null mixture component.

    Raises
    ------
    ValueError
        If ``scale_factors`` does not have shape ``(n_components, 2)`` or
        holds a value that is not finite and positive; if any posterior
        precision does not strictly exceed its shrinkage ratio; if
        ``detection_power`` does not hold exactly one entry per component;
        if any detection probability is not finite or falls outside the
        interval [0, 1]; or if ``z_threshold`` is not a finite real number
        greater than or equal to zero.

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

def _oracle_component_tail_second_moment(scale_factors: np.ndarray, z_threshold: float,
                                         detection_power: np.ndarray) -> np.ndarray:
    import numpy as np

    factors = np.asarray(scale_factors, dtype=float)
    power = np.asarray(detection_power, dtype=float).ravel()
    if factors.ndim != 2 or factors.shape[1] != 2 or factors.shape[0] < 1:
        raise ValueError("scale_factors must have shape (n_components, 2)")
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        raise ValueError("scale_factors must be finite and > 0")
    lam = factors[:, 0]
    c_precision = factors[:, 1]
    if np.any(c_precision <= lam):
        raise ValueError("the posterior precision must exceed the shrinkage ratio")
    if power.size != factors.shape[0]:
        raise ValueError("detection_power must hold one entry per component")
    if not np.all(np.isfinite(power)) or np.any(power < 0.0) or np.any(power > 1.0):
        raise ValueError("detection_power must lie in [0, 1]")
    if isinstance(z_threshold, bool) or not isinstance(
            z_threshold, (int, float, np.integer, np.floating)):
        raise ValueError("z_threshold must be a real number")
    z_threshold = float(z_threshold)
    if not np.isfinite(z_threshold) or z_threshold < 0.0:
        raise ValueError("z_threshold must be a finite number >= 0")

    # The detection region is |W| > c on the standard scale, with the same cut
    # point the detection probability uses.
    cut = np.sqrt(z_threshold * lam / c_precision)
    density = np.exp(-0.5 * cut ** 2) / np.sqrt(2.0 * np.pi)

    # Integrating w**2 over both tails leaves the boundary term plus the tail
    # probability itself, so the moment reduces to a two-term expression.
    return np.asarray(2.0 * cut * density + power, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values: the ratios put the cut at exactly 1, 0.5 and 1.5
        # standard deviations, whose two-sided normal tails are textbook
        # constants; the moment is then rebuilt from the density without
        # touching the oracle. Dropping the boundary term, or halving it,
        # misses all three.
        {
            "setup": """import numpy as np
factors = np.array([[1000.0, 4000.0], [625.0, 10000.0], [5625.0, 10000.0]])
cut = np.array([1.0, 0.5, 1.5])
tails = np.array([0.31731050786291415, 0.6170750774519738, 0.13361440253771614])
exact = 2.0 * cut * np.exp(-0.5 * cut ** 2) / np.sqrt(2.0 * np.pi) + tails
w = np.array([1.0, 4.0, 16.0])
EXPECTED = float(np.dot(w, exact))
""",
            "call": "float(np.dot(w, component_tail_second_moment(factors, 4.0, tails)))",
            "gold_call": "EXPECTED",
        },
        # --- Valid: a four-component prior at a realistic PIP threshold ---
        {
            "setup": """import numpy as np
factors = np.array([[2.4e5, 1.94e6], [2.4e4, 1.724e6],
                    [2.4e3, 1.7024e6], [2.4e2, 1.70024e6]])
power = np.array([0.12, 0.55, 0.84, 0.94])
w = np.array([1.0, 2.0, 3.0, 4.0])
""",
            "call": "float(np.dot(w, component_tail_second_moment(factors, 17.3, power)))",
            "gold_call": ("float(np.dot(w, _oracle_component_tail_second_moment("
                          "factors, 17.3, power)))"),
        },
        # --- Boundary: a threshold of zero, where the region is the whole line
        # and the moment must collapse to the unit variance ---
        {
            "setup": """import numpy as np
factors = np.array([[1.0e4, 5.0e5], [1.0e2, 5.0e5]])
power = np.array([1.0, 1.0])
""",
            "call": "float(np.sum(component_tail_second_moment(factors, 0.0, power)))",
            "gold_call": "float(np.sum(_oracle_component_tail_second_moment(factors, 0.0, power)))",
        },
        # --- Edge: a deep tail, where the boundary term dominates the moment ---
        {
            "setup": """import numpy as np
factors = np.array([[8.0e5, 8.3e5]])
power = np.array([1.5e-8])
""",
            "call": "float(component_tail_second_moment(factors, 34.0, power)[0])",
            "gold_call": "float(_oracle_component_tail_second_moment(factors, 34.0, power)[0])",
        },
        # --- Invalid: a detection probability above one ---
        {
            "setup": """import numpy as np
factors = np.array([[1.0e4, 5.0e5]])
def run_model():
    try:
        component_tail_second_moment(factors, 10.0, np.array([1.2]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_component_tail_second_moment(factors, 10.0, np.array([1.2]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: one detection probability short of the component count ---
        {
            "setup": """import numpy as np
factors = np.array([[1.0e4, 5.0e5], [1.0e2, 5.0e5]])
def run_model():
    try:
        component_tail_second_moment(factors, 10.0, np.array([0.4]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_component_tail_second_moment(factors, 10.0, np.array([0.4]))
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
