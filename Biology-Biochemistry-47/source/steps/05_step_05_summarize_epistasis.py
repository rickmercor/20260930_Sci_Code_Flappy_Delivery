"""
Reduce a matrix of double-mutant epistasis values to the four summary statistics that characterise how much interaction a mechanism manufactures on its own.

Epistasis, defined as a ratio of observed to expected, is a multiplicative quantity: the null value is one, a doubling and a halving are equally large departures, and the natural coordinate in which to summarise it is the base-ten logarithm of its magnitude. Working with the raw ratio instead would give a distribution with a hard floor at zero and no ceiling, so its summary statistics would be dominated by the enhancing tail and would misrepresent the diminishing one.

Experimental practice does not call every departure epistasis. Measurement error on steady-state parameters is such that a small fold departure is not distinguishable from additivity, so a variant pair counts as epistatic only once its ratio is at least the stated threshold or at most the reciprocal of that threshold. Applying the threshold symmetrically on the logarithmic scale is what makes enhancing and diminishing interactions equally hard to detect, and the fraction of pairs that clear it is the prevalence usually quoted in the literature.

Returns
-------
np.ndarray of shape (4,), float: the significant fraction, the median absolute log10 epistasis over all entries, the enhancing fraction among significant entries (0.0 when no entry clears the threshold), and the maximum absolute log10 epistasis.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def summarize_epistasis(epistasis, threshold: float = 1.5) -> np.ndarray:
    """Reduce a matrix of epistasis ratios to four summary statistics.

    Parameters
    ----------
    epistasis : array_like
        Array of strictly positive epistasis ratios, observed over expected.
    threshold : float
        Fold-change at or beyond which a departure from the null model counts
        as significant (threshold > 1).

    Returns
    -------
    summary : np.ndarray
        Array of shape (4,) holding the fraction of entries that clear the
        threshold, the median absolute base-ten logarithm over all entries,
        the fraction of the significant entries that are enhancing, and the
        largest absolute base-ten logarithm anywhere in the input. When no
        entry clears the threshold the enhancing fraction is 0.0, the empty
        set of significant entries carrying no enhancing ones.

    Raises
    ------
    ValueError
        If epistasis is empty, holds a non-finite entry or an entry that is
        not strictly positive, or if threshold is not a finite number
        greater than one.
    """
    return summary  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_summarize_epistasis(epistasis, threshold: float = 1.5) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    values = np.asarray(epistasis, dtype=float)
    if values.size < 1:
        raise ValueError("epistasis must contain at least one entry")
    if not np.all(np.isfinite(values)):
        raise ValueError("epistasis must be finite")
    if np.any(values <= 0.0):
        raise ValueError("epistasis ratios must be strictly positive")
    if not (isinstance(threshold, (int, float, np.floating, np.integer))
            and not isinstance(threshold, bool)
            and np.isfinite(threshold) and float(threshold) > 1.0):
        raise ValueError("threshold must be a finite number > 1")

    magnitude = np.abs(np.log10(values))
    significant = magnitude >= np.log10(float(threshold))

    fraction_significant = float(np.mean(significant))
    median_magnitude = float(np.median(magnitude))
    if np.any(significant):
        fraction_enhancing = float(np.mean(values[significant] > 1.0))
    else:
        fraction_enhancing = 0.0
    max_magnitude = float(np.max(magnitude))

    return np.array([fraction_significant, median_magnitude,
                     fraction_enhancing, max_magnitude], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    import numpy as np  # noqa: F401  (keeps this field self-contained)

    return [
        # --- Valid: a broad log-symmetric sweep of ratios (normal scenario) ---
        {
            "setup": """import numpy as np
generator = np.random.default_rng(31)
epistasis = 10.0 ** generator.normal(0.05, 0.4, size=(60, 60))
""",
            "call": "summarize_epistasis(epistasis)",
            "gold_call": "_oracle_summarize_epistasis(epistasis)",
        },
        # --- Valid: a stricter threshold on a skewed sweep ---
        {
            "setup": """import numpy as np
generator = np.random.default_rng(4)
epistasis = 10.0 ** generator.gumbel(0.0, 0.3, size=2000)
threshold = 5.0
""",
            "call": "summarize_epistasis(epistasis, threshold)",
            "gold_call": "_oracle_summarize_epistasis(epistasis, threshold)",
        },
        # --- Boundary: every entry exactly at the null value, so nothing is significant ---
        {
            "setup": """import numpy as np
epistasis = np.ones((8, 8))
""",
            "call": "summarize_epistasis(epistasis)",
            "gold_call": "_oracle_summarize_epistasis(epistasis)",
        },
        # --- Boundary: entries sitting exactly on the threshold and its reciprocal ---
        {
            "setup": """import numpy as np
epistasis = np.array([1.5, 1.0 / 1.5, 1.0, 2.0, 0.5])
""",
            "call": "summarize_epistasis(epistasis, 1.5)",
            "gold_call": "_oracle_summarize_epistasis(epistasis, 1.5)",
        },
        # --- Edge: a single entry many decades away from the null value ---
        {
            "setup": """import numpy as np
epistasis = np.array([1.0e-7])
""",
            "call": "summarize_epistasis(epistasis, 1.2)",
            "gold_call": "_oracle_summarize_epistasis(epistasis, 1.2)",
        },
        # --- Invalid: a non-positive epistasis ratio ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        summarize_epistasis(np.array([1.0, 0.0, 2.0]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_summarize_epistasis(np.array([1.0, 0.0, 2.0]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: threshold not above one ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        summarize_epistasis(np.array([1.0, 2.0]), 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_summarize_epistasis(np.array([1.0, 2.0]), 1.0)
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
