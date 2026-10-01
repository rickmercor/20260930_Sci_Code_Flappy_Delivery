"""
Determine the number and prevalence of significant double-mutant interactions from a vector of macroscopic interaction factors using the fixed 1.5-fold significance criterion. Separate significant interactions into positive and negative classes and report their aggregate counts and prevalence.

The benchmark follows the paper's use of a 1.5-fold threshold for distinguishing substantial deviations from the null prediction. An interaction factor greater than 1.5 represents a positive deviation, while an interaction factor below the reciprocal threshold 1/1.5 represents a negative deviation. Values between these thresholds are treated as non-significant. The resulting prevalence measures the fraction of tested mutant pairs displaying significant non-specific epistasis and allows the simple and extended catalytic mechanisms to be compared on the same scale.

Returns
-------
np.ndarray, a one-dimensional floating-point array of length 4 containing [significant_count, positive_count, negative_count, significant_prevalence].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def classify_epistasis(
    interactions: "np.ndarray",
    threshold: float,
) -> "np.ndarray":
    """
    Summarize significant pairwise interactions using the supplied
    significance threshold.

    Parameters
    ----------
    interactions : np.ndarray
        One-dimensional vector of interaction factors.

    threshold : float
        Multiplicative threshold used for significance classification.

    Returns
    -------
    np.ndarray
        A length-4 floating-point array containing, in order:
        [significant_count, positive_count, negative_count,
         significant_prevalence].

    Raises
    ------
    ValueError
        If the interaction vector is invalid or the threshold is invalid.
    """
    return summary

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_classify_epistasis(
    interactions: "np.ndarray",
    threshold: float,
) -> "np.ndarray":
    e = np.asarray(interactions, dtype=float)
    if e.ndim != 1 or e.size < 1 or not np.all(np.isfinite(e)) or np.any(e <= 0):
        raise ValueError("invalid interactions")
    if not np.isfinite(threshold) or threshold <= 1:
        raise ValueError("threshold must exceed one")
    pos = e > threshold
    neg = e < 1.0 / threshold
    sig = pos | neg
    return np.array([sig.sum(), pos.sum(), neg.sum(), sig.mean()], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\ne=np.array([1.,1.6,.5,2.,.8],float)\n", "call": "classify_epistasis(e,1.5)", "gold_call": "_oracle_classify_epistasis(e,1.5)"},
        {"setup": "import numpy as np\ne=np.ones(5)\n", "call": "classify_epistasis(e,1.5)", "gold_call": "_oracle_classify_epistasis(e,1.5)"},
        {"setup": "import numpy as np\ne=np.array([1e-9,1e9],float)\n", "call": "classify_epistasis(e,1.5)", "gold_call": "_oracle_classify_epistasis(e,1.5)"},
        {"setup": "import numpy as np\ne=np.ones(2)\ndef run_model():\n    try: classify_epistasis(e,1.0); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try: _oracle_classify_epistasis(e,1.0); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call": "run_model()", "gold_call": "run_gold()"},
    ]
