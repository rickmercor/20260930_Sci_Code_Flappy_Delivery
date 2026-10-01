"""
Turn the per-component counts of causal variants reported by a genome-wide Bayesian mixture model into the mixing probabilities of the prior, including the probability that a variant is null.

A genome-wide Bayesian mixture model gives every SNP the same prior, a point mass at zero followed by several normal components, so an estimated architecture quoted as a number of causal variants per component is a statement about the mixing probabilities of that prior. Dividing each count by the number of SNPs analysed gives the non-null probabilities, and the null probability is whatever is left over.

Returns
-------
np.ndarray, float, shape (1 + n_components,): the prior mixing probabilities, with the null probability first.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def mixture_probabilities(n_snps: int, causal_counts: tuple) -> np.ndarray:
    """Convert per-component causal-variant counts into prior mixing probabilities.

    Parameters
    ----------
    n_snps : int
        Total number of SNPs fitted by the model (n_snps > 0).
    causal_counts : tuple
        One non-negative integer per non-null mixture component, giving the
        number of causal variants assigned to that component. The total must
        be smaller than ``n_snps``.

    Returns
    -------
    probabilities : np.ndarray
        Shape ``(1 + len(causal_counts),)`` float array. Element 0 is the
        prior probability that a SNP is null; the remaining elements are the
        prior probabilities of the non-null components, in the given order.

    Raises
    ------
    ValueError
        If ``n_snps`` is not an integer greater than zero; if
        ``causal_counts`` is not a non-empty sequence of integers; if any
        count is negative; or if the counts sum to ``n_snps`` or more, which
        would leave no null probability.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(1 + len(causal_counts), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_mixture_probabilities(n_snps: int, causal_counts: tuple) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if isinstance(n_snps, bool) or not isinstance(n_snps, (int, np.integer)):
        raise ValueError("n_snps must be an integer")
    n_snps = int(n_snps)
    if n_snps <= 0:
        raise ValueError("n_snps must be > 0")
    if not isinstance(causal_counts, (tuple, list, np.ndarray)):
        raise ValueError("causal_counts must be a sequence")
    counts = list(causal_counts)
    if len(counts) < 1:
        raise ValueError("causal_counts must hold at least one component")
    for val in counts:
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise ValueError("causal_counts must contain integers only")
        if int(val) < 0:
            raise ValueError("causal_counts must be non-negative")
    total = sum(int(val) for val in counts)
    if total >= n_snps:
        raise ValueError("the causal variants must be fewer than the SNPs fitted")

    # Non-null mixing probabilities are the component shares of the panel; the
    # null probability absorbs the rest, so the vector sums to one by
    # construction.
    non_null = np.array([int(val) / n_snps for val in counts], dtype=float)
    probabilities = np.empty(non_null.size + 1, dtype=float)
    probabilities[0] = 1.0 - float(total) / n_snps
    probabilities[1:] = non_null
    return probabilities

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values: the probabilities are exact rationals, so the
        # null probability and the ordered non-null probabilities can be
        # written down independently of the oracle. Probing with the weights
        # [1, 10, 100, 1000, 10000] separates every element, so a reversed
        # order or a null probability computed as 1 minus one component
        # cannot pass.
        {
            "setup": """import numpy as np
n_snps = 2000000
counts = (8000, 600, 40, 2)
w = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
exact = np.array([1.0 - 8642.0 / 2000000.0, 8000.0 / 2000000.0,
                  600.0 / 2000000.0, 40.0 / 2000000.0, 2.0 / 2000000.0])
EXPECTED = float(np.dot(w, exact))
""",
            "call": "float(np.dot(w, mixture_probabilities(n_snps, counts)))",
            "gold_call": "EXPECTED",
        },
        # --- Valid: a four-component architecture on a dense panel ---
        {
            "setup": """import numpy as np
n_snps = 9000000
counts = (55000, 4100, 310, 17)
""",
            "call": "float(np.sum(mixture_probabilities(n_snps, counts) * np.arange(1.0, 6.0)))",
            "gold_call": ("float(np.sum(_oracle_mixture_probabilities(n_snps, counts)"
                          " * np.arange(1.0, 6.0)))"),
        },
        # --- Boundary: an almost entirely causal panel, where the null
        # probability is tiny but still strictly positive ---
        {
            "setup": """import numpy as np
n_snps = 1000
counts = (900, 99)
""",
            "call": "float(mixture_probabilities(n_snps, counts)[0])",
            "gold_call": "float(_oracle_mixture_probabilities(n_snps, counts)[0])",
        },
        # --- Edge: a single non-null component with one causal variant ---
        {
            "setup": """import numpy as np
n_snps = 500000
counts = (1,)
""",
            "call": "float(np.sum(mixture_probabilities(n_snps, counts)))",
            "gold_call": "float(np.sum(_oracle_mixture_probabilities(n_snps, counts)))",
        },
        # --- Invalid: more causal variants than SNPs fitted ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        mixture_probabilities(1000, (900, 200))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_mixture_probabilities(1000, (900, 200))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative component count ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        mixture_probabilities(1000000, (5000, -3))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_mixture_probabilities(1000000, (5000, -3))
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
