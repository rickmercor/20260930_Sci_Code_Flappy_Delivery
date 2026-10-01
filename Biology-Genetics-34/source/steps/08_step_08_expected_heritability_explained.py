"""
Combine the per-component detected effect variances with the prior mixing probabilities into the expected proportion of SNP-based heritability carried by the variants that pass the PIP threshold.

Each of the n_snps variants contributes its component's detected squared effect with the prior probability of belonging to that component, so the expected genetic variance carried by the detected variants is n_snps times the probability-weighted sum of those contributions. Dividing by the SNP-based heritability turns it into a proportion, which is an effect-size-weighted quantity and not the fraction of causal variants found.

Returns
-------
float: the expected proportion of SNP-based heritability carried by the detected variants, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def expected_heritability_explained(mixture_probs: np.ndarray, detected_variance: np.ndarray,
                                    h2_snp: float, n_snps: int) -> float:
    """Expected proportion of SNP-based heritability carried by detected variants.

    Parameters
    ----------
    mixture_probs : np.ndarray
        Shape ``(1 + n_components,)`` prior mixing probabilities, with the
        null probability first.
    detected_variance : np.ndarray
        Shape ``(n_components,)`` array of expected squared causal effects
        restricted to the detection region.
    h2_snp : float
        SNP-based heritability of the trait (0 < h2_snp < 1).
    n_snps : int
        Total number of SNPs fitted by the model (n_snps > 0).

    Returns
    -------
    proportion : float
        Expected proportion of the SNP-based heritability carried by the
        variants that pass the threshold, as a native Python float.

    Raises
    ------
    ValueError
        If ``detected_variance`` is empty, holds a value that is not finite,
        or holds a negative value; if ``mixture_probs`` does not hold
        exactly one more entry than ``detected_variance``; if any
        probability is negative or not finite; if the probabilities do not
        sum to 1 within a tolerance of 1e-9; if ``h2_snp`` is not a real
        number strictly between 0 and 1; or if ``n_snps`` is not an integer
        greater than zero.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_expected_heritability_explained(mixture_probs: np.ndarray,
                                            detected_variance: np.ndarray,
                                            h2_snp: float, n_snps: int) -> float:
    import numpy as np

    probs = np.asarray(mixture_probs, dtype=float).ravel()
    detected = np.asarray(detected_variance, dtype=float).ravel()
    if detected.size < 1:
        raise ValueError("detected_variance must hold at least one component")
    if probs.size != detected.size + 1:
        raise ValueError("mixture_probs must hold one null and one per non-null component")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("mixture_probs must be finite and non-negative")
    if abs(float(np.sum(probs)) - 1.0) > 1e-9:
        raise ValueError("mixture_probs must sum to 1")
    if not np.all(np.isfinite(detected)) or np.any(detected < 0.0):
        raise ValueError("detected_variance must be finite and non-negative")
    if isinstance(h2_snp, bool) or not isinstance(h2_snp, (int, float, np.integer, np.floating)):
        raise ValueError("h2_snp must be a real number")
    h2_snp = float(h2_snp)
    if not np.isfinite(h2_snp) or h2_snp <= 0.0 or h2_snp >= 1.0:
        raise ValueError("h2_snp must lie strictly between 0 and 1")
    if isinstance(n_snps, bool) or not isinstance(n_snps, (int, np.integer)):
        raise ValueError("n_snps must be an integer")
    n_snps = int(n_snps)
    if n_snps <= 0:
        raise ValueError("n_snps must be > 0")

    # Every SNP in the panel contributes, weighted by the prior probability of
    # the component it would come from; the null component contributes nothing.
    detected_genetic_variance = float(n_snps) * float(np.sum(probs[1:] * detected))
    return float(detected_genetic_variance / h2_snp)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned value: the weighted sum below is an exact decimal, so the
        # proportion can be written down directly. Weighting by the null
        # probability, or omitting the panel size, changes it by orders of
        # magnitude.
        {
            "setup": """import numpy as np
probs = np.array([0.9960, 0.0032, 0.0006, 0.0002])
detected = np.array([1.0e-6, 2.0e-5, 2.4e-4])
EXPECTED = float(1000000.0 * (0.0032 * 1.0e-6 + 0.0006 * 2.0e-5
                              + 0.0002 * 2.4e-4) / 0.32)
""",
            "call": "expected_heritability_explained(probs, detected, 0.32, 1000000)",
            "gold_call": "EXPECTED",
        },
        # --- Valid: a four-component architecture on a dense panel ---
        {
            "setup": """import numpy as np
probs = np.array([0.9972, 0.0024, 0.00035, 0.00005])
detected = np.array([1.4e-6, 3.1e-5, 3.0e-4])
""",
            "call": "expected_heritability_explained(probs, detected, 0.31, 9000000)",
            "gold_call": ("_oracle_expected_heritability_explained("
                          "probs, detected, 0.31, 9000000)"),
        },
        # --- Boundary: no variant detected anywhere, where the proportion must
        # be exactly zero ---
        {
            "setup": """import numpy as np
probs = np.array([0.99, 0.008, 0.002])
detected = np.zeros(2)
""",
            "call": "expected_heritability_explained(probs, detected, 0.45, 800000)",
            "gold_call": "_oracle_expected_heritability_explained(probs, detected, 0.45, 800000)",
        },
        # --- Edge: a single component whose detected variance equals its prior
        # variance, where the proportion must return the architecture's own
        # share of the heritability ---
        {
            "setup": """import numpy as np
probs = np.array([0.999, 0.001])
detected = np.array([1e-3 * 0.2])
""",
            "call": "expected_heritability_explained(probs, detected, 0.2, 1000000)",
            "gold_call": "_oracle_expected_heritability_explained(probs, detected, 0.2, 1000000)",
        },
        # --- Invalid: mixing probabilities that do not sum to one ---
        {
            "setup": """import numpy as np
probs = np.array([0.90, 0.05])
detected = np.array([1.0e-5])
def run_model():
    try:
        expected_heritability_explained(probs, detected, 0.3, 100000)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_expected_heritability_explained(probs, detected, 0.3, 100000)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-integer panel size ---
        {
            "setup": """import numpy as np
probs = np.array([0.99, 0.01])
detected = np.array([1.0e-5])
def run_model():
    try:
        expected_heritability_explained(probs, detected, 0.3, 1.5e6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_expected_heritability_explained(probs, detected, 0.3, 1.5e6)
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
