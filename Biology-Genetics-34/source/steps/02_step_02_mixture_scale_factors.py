"""
Convert the prior variance scale factors of the mixture model into the per-component shrinkage and precision constants that a single-variant posterior at a given sample size depends on.

For standardised genotypes and a phenotype of unit variance the genetic variance is the SNP-based heritability and the residual variance is its complement, so component k of the prior has effect variance gamma_k times the genetic variance and a shrinkage ratio lambda_k = sigma_e**2 / (gamma_k * sigma_g**2). At sample size n the posterior precision of a single-variant effect is C_k = n + lambda_k, and the two together fix everything the posterior odds of that component need.

Returns
-------
np.ndarray, float, shape (n_components, 2): the shrinkage ratio lambda_k and the posterior precision C_k of every non-null mixture component.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def mixture_scale_factors(n: float, h2_snp: float, gamma: tuple) -> np.ndarray:
    """Build the shrinkage and precision constants of the non-null components.

    Parameters
    ----------
    n : float
        Sample size of the study, on the scale at which the phenotype has
        unit variance (n > 0).
    h2_snp : float
        SNP-based heritability of the trait (0 < h2_snp < 1).
    gamma : tuple
        Prior variance scale factors of the mixture, one per component and in
        the same order as the mixing probabilities. The first entry is the
        null component and must be exactly zero; the rest must be strictly
        positive.

    Returns
    -------
    factors : np.ndarray
        Shape ``(len(gamma) - 1, 2)`` float array with one row per non-null
        component, holding the shrinkage ratio in column 0 and the posterior
        precision in column 1.

    Raises
    ------
    ValueError
        If ``n`` is not a finite real number greater than zero; if
        ``h2_snp`` is not a real number strictly between 0 and 1; if
        ``gamma`` is not a finite sequence holding a null component and at
        least one non-null one; if the first ``gamma`` entry is not exactly
        zero; or if any later ``gamma`` entry is not strictly positive.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros((len(gamma) - 1, 2), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_mixture_scale_factors(n: float, h2_snp: float, gamma: tuple) -> np.ndarray:
    import numpy as np

    if isinstance(n, bool) or not isinstance(n, (int, float, np.integer, np.floating)):
        raise ValueError("n must be a real number")
    n = float(n)
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("n must be a finite number > 0")
    if isinstance(h2_snp, bool) or not isinstance(h2_snp, (int, float, np.integer, np.floating)):
        raise ValueError("h2_snp must be a real number")
    h2_snp = float(h2_snp)
    if not np.isfinite(h2_snp) or h2_snp <= 0.0 or h2_snp >= 1.0:
        raise ValueError("h2_snp must lie strictly between 0 and 1")
    if not isinstance(gamma, (tuple, list, np.ndarray)):
        raise ValueError("gamma must be a sequence")
    scales = np.asarray(gamma, dtype=float).ravel()
    if scales.size < 2:
        raise ValueError("gamma must hold a null component and at least one non-null one")
    if not np.all(np.isfinite(scales)):
        raise ValueError("gamma must be finite")
    if scales[0] != 0.0:
        raise ValueError("the first gamma entry is the null component and must be zero")
    if np.any(scales[1:] <= 0.0):
        raise ValueError("every non-null gamma entry must be > 0")

    # Standardised genotypes with a unit-variance phenotype put the genetic
    # variance at the heritability and the residual variance at its complement.
    sigma_g2 = h2_snp
    sigma_e2 = 1.0 - h2_snp

    # The shrinkage ratio is the residual variance over the prior effect
    # variance of the component, so a large-effect component has a small ratio.
    lam = sigma_e2 / (scales[1:] * sigma_g2)
    c_precision = n + lam

    factors = np.empty((lam.size, 2), dtype=float)
    factors[:, 0] = lam
    factors[:, 1] = c_precision
    return factors

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values: with h2 = 0.5 the residual and genetic variances
        # are both 0.5, so lambda_k is exactly 1 / gamma_k and C_k is
        # n + 1 / gamma_k. Probing lambda and C with different weights
        # separates the two columns, so a transposed return or a C_k built
        # from the genetic instead of the residual variance cannot pass.
        {
            "setup": """import numpy as np
n = 250000.0
gamma = (0.0, 1e-4, 1e-2)
lam_exact = np.array([1.0e4, 1.0e2])
c_exact = n + lam_exact
EXPECTED = float(np.dot(lam_exact, [1.0, 3.0]) + 1e-3 * np.dot(c_exact, [1.0, 7.0]))
""",
            "call": ("float(np.dot(mixture_scale_factors(n, 0.5, gamma)[:, 0], [1.0, 3.0])"
                     " + 1e-3 * np.dot(mixture_scale_factors(n, 0.5, gamma)[:, 1], [1.0, 7.0]))"),
            "gold_call": "EXPECTED",
        },
        # --- Valid: a four-component prior at a realistic study size ---
        {
            "setup": """import numpy as np
n = 480000.0
gamma = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)
w = np.array([1.0, 2.0, 4.0, 8.0])
""",
            "call": ("float(np.dot(mixture_scale_factors(n, 0.31, gamma)[:, 0], w)"
                     " + 1e-2 * np.dot(mixture_scale_factors(n, 0.31, gamma)[:, 1], w))"),
            "gold_call": ("float(np.dot(_oracle_mixture_scale_factors(n, 0.31, gamma)[:, 0], w)"
                          " + 1e-2 * np.dot(_oracle_mixture_scale_factors(n, 0.31, gamma)[:, 1], w))"),
        },
        # --- Boundary: a very small sample, where the shrinkage ratio rather
        # than the sample size dominates the posterior precision ---
        {
            "setup": """import numpy as np
gamma = (0.0, 1e-6, 1e-3)
""",
            "call": "float(np.sum(mixture_scale_factors(12.0, 0.08, gamma)[:, 1]))",
            "gold_call": "float(np.sum(_oracle_mixture_scale_factors(12.0, 0.08, gamma)[:, 1]))",
        },
        # --- Edge: a nearly fully heritable trait, where the residual variance
        # is small and every shrinkage ratio collapses toward zero ---
        {
            "setup": """import numpy as np
gamma = (0.0, 1e-3)
""",
            "call": "float(mixture_scale_factors(1e6, 0.995, gamma)[0, 0])",
            "gold_call": "float(_oracle_mixture_scale_factors(1e6, 0.995, gamma)[0, 0])",
        },
        # --- Invalid: a non-zero leading gamma entry ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        mixture_scale_factors(1e5, 0.3, (1e-6, 1e-4, 1e-2))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_mixture_scale_factors(1e5, 0.3, (1e-6, 1e-4, 1e-2))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a heritability outside the open unit interval ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        mixture_scale_factors(1e5, 1.0, (0.0, 1e-4))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_mixture_scale_factors(1e5, 1.0, (0.0, 1e-4))
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
