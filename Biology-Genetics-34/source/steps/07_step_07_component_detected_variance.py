"""
Compute, for each non-null mixture component, the expected squared effect of a causal variant restricted to the outcomes in which that variant is detected.

Conditioning the prior effect on the observed deviate makes its conditional mean proportional to that deviate and its conditional variance sigma_e**2 / C_k, so the expected squared effect over the detection region splits into a flat piece weighted by the detection probability and a piece weighted by the restricted second moment. The two carry coefficients sigma_e**2 / C_k and n * sigma_e**2 / (lambda_k * C_k), with sigma_e**2 the residual variance.

Returns
-------
np.ndarray, float, shape (n_components,): the expected squared causal effect restricted to the detection region, per non-null mixture component.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def component_detected_variance(scale_factors: np.ndarray, detection_power: np.ndarray,
                                tail_second_moment: np.ndarray, h2_snp: float,
                                n: float) -> np.ndarray:
    """Expected squared effect of a causal variant over the detection region.

    Parameters
    ----------
    scale_factors : np.ndarray
        Shape ``(n_components, 2)`` array holding the shrinkage ratio in
        column 0 and the posterior precision in column 1.
    detection_power : np.ndarray
        Shape ``(n_components,)`` array of detection probabilities.
    tail_second_moment : np.ndarray
        Shape ``(n_components,)`` array of restricted second moments of the
        standardised statistic.
    h2_snp : float
        SNP-based heritability of the trait (0 < h2_snp < 1).
    n : float
        Sample size of the study, on the scale at which the phenotype has
        unit variance (n > 0).

    Returns
    -------
    detected_variance : np.ndarray
        Shape ``(n_components,)`` float array of expected squared effects
        restricted to the detection region, one per non-null component.

    Raises
    ------
    ValueError
        If ``scale_factors`` does not have shape ``(n_components, 2)`` or
        holds a value that is not finite and positive; if
        ``detection_power`` or ``tail_second_moment`` does not hold exactly
        one entry per component; if any detection probability is not finite
        or falls outside the interval [0, 1]; if any restricted second
        moment is not finite or is negative; if ``h2_snp`` is not a real
        number strictly between 0 and 1; if ``n`` is not a finite real
        number greater than zero; or if ``scale_factors`` and ``n`` describe
        different sample sizes, that is if any posterior precision minus its
        shrinkage ratio differs from ``n``.

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

def _oracle_component_detected_variance(scale_factors: np.ndarray, detection_power: np.ndarray,
                                        tail_second_moment: np.ndarray, h2_snp: float,
                                        n: float) -> np.ndarray:
    import numpy as np

    factors = np.asarray(scale_factors, dtype=float)
    power = np.asarray(detection_power, dtype=float).ravel()
    moment = np.asarray(tail_second_moment, dtype=float).ravel()
    if factors.ndim != 2 or factors.shape[1] != 2 or factors.shape[0] < 1:
        raise ValueError("scale_factors must have shape (n_components, 2)")
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        raise ValueError("scale_factors must be finite and > 0")
    if power.size != factors.shape[0] or moment.size != factors.shape[0]:
        raise ValueError("detection_power and tail_second_moment need one entry per component")
    if not np.all(np.isfinite(power)) or np.any(power < 0.0) or np.any(power > 1.0):
        raise ValueError("detection_power must lie in [0, 1]")
    if not np.all(np.isfinite(moment)) or np.any(moment < 0.0):
        raise ValueError("tail_second_moment must be finite and non-negative")
    if isinstance(h2_snp, bool) or not isinstance(h2_snp, (int, float, np.integer, np.floating)):
        raise ValueError("h2_snp must be a real number")
    h2_snp = float(h2_snp)
    if not np.isfinite(h2_snp) or h2_snp <= 0.0 or h2_snp >= 1.0:
        raise ValueError("h2_snp must lie strictly between 0 and 1")
    if isinstance(n, bool) or not isinstance(n, (int, float, np.integer, np.floating)):
        raise ValueError("n must be a real number")
    n = float(n)
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("n must be a finite number > 0")

    lam = factors[:, 0]
    c_precision = factors[:, 1]
    # The precision is the sample size plus the shrinkage ratio, so the three
    # arguments have to describe the same study.
    if np.any(np.abs(c_precision - lam - n) > 1e-6 * np.maximum(1.0, c_precision)):
        raise ValueError("scale_factors and n describe different sample sizes")

    sigma_e2 = 1.0 - h2_snp

    # Flat piece: the conditional variance of the effect, which survives even
    # where the deviate itself carries no information.
    flat = (sigma_e2 / c_precision) * power
    # Regression piece: the conditional mean squared, whose coefficient grows
    # with the sample size and shrinks with the component's shrinkage ratio.
    regression = (n * sigma_e2 / (lam * c_precision)) * moment
    return np.asarray(flat + regression, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values: with the arguments below the two coefficients are
        # exact decimals, so the expected squared effects can be written down
        # directly. Dropping the flat piece, or pairing the regression
        # coefficient with the detection probability instead of the moment,
        # fails both components.
        {
            "setup": """import numpy as np
n = 9000.0
lam = np.array([1000.0, 100.0])
factors = np.column_stack([lam, n + lam])
power = np.array([0.25, 0.80])
moment = np.array([0.60, 0.95])
sigma_e2 = 1.0 - 0.40
exact = (sigma_e2 / (n + lam)) * power + (n * sigma_e2 / (lam * (n + lam))) * moment
EXPECTED = float(1e3 * np.dot(exact, [1.0, 6.0]))
""",
            "call": ("float(1e3 * np.dot(component_detected_variance("
                     "factors, power, moment, 0.40, n), [1.0, 6.0]))"),
            "gold_call": "EXPECTED",
        },
        # --- Valid: a four-component prior at a realistic study size ---
        {
            "setup": """import numpy as np
n = 1.7e6
lam = np.array([2.4e5, 2.4e4, 2.4e3, 2.4e2])
factors = np.column_stack([lam, n + lam])
power = np.array([0.12, 0.55, 0.84, 0.94])
moment = np.array([0.170, 0.860, 0.990, 0.9995])
w = np.array([1.0, 10.0, 100.0, 1000.0])
""",
            "call": ("float(np.dot(w, component_detected_variance("
                     "factors, power, moment, 0.31, n)))"),
            "gold_call": ("float(np.dot(w, _oracle_component_detected_variance("
                          "factors, power, moment, 0.31, n)))"),
        },
        # --- Boundary: full detection, where the result must return the prior
        # effect variance of each component ---
        {
            "setup": """import numpy as np
n = 5.0e5
lam = np.array([1.0e5, 1.0e3])
factors = np.column_stack([lam, n + lam])
power = np.array([1.0, 1.0])
moment = np.array([1.0, 1.0])
""",
            "call": ("float(1e4 * np.sum(component_detected_variance("
                     "factors, power, moment, 0.5, n)))"),
            "gold_call": ("float(1e4 * np.sum(_oracle_component_detected_variance("
                          "factors, power, moment, 0.5, n)))"),
        },
        # --- Edge: no detection at all, where the expected squared effect must
        # vanish rather than fall back on the prior ---
        {
            "setup": """import numpy as np
n = 1.0e4
lam = np.array([2.0e5])
factors = np.column_stack([lam, n + lam])
""",
            "call": ("float(component_detected_variance("
                     "factors, np.array([0.0]), np.array([0.0]), 0.15, n)[0])"),
            "gold_call": ("float(_oracle_component_detected_variance("
                          "factors, np.array([0.0]), np.array([0.0]), 0.15, n)[0])"),
        },
        # --- Invalid: a sample size that contradicts the posterior precision ---
        {
            "setup": """import numpy as np
factors = np.array([[1.0e4, 5.1e5]])
def run_model():
    try:
        component_detected_variance(factors, np.array([0.5]), np.array([0.9]), 0.3, 1.0e4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_component_detected_variance(factors, np.array([0.5]), np.array([0.9]), 0.3, 1.0e4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative restricted second moment ---
        {
            "setup": """import numpy as np
factors = np.array([[1.0e4, 5.1e5]])
def run_model():
    try:
        component_detected_variance(factors, np.array([0.5]), np.array([-0.1]), 0.3, 5.0e5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_component_detected_variance(factors, np.array([0.5]), np.array([-0.1]), 0.3, 5.0e5)
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
