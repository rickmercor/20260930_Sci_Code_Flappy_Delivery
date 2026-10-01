"""
Summarize the highest-ranked conditional haplotypes as an allele burden.

Candidates are ranked by decreasing conditional probability, with binary

lexicographic order breaking exact ties. For the first L configurations, the

reported burden is the number of partner ALT alleles averaged with probabilities

renormalized within that top-L set. The retained mass is also returned so the

truncation can be inspected.

Inputs

------

configurations : expanded binary partner configurations

conditional_probabilities : probability assigned to each configuration

top_l : number of leading configurations retained

Returns

-------

mean_burden : top-L probability-weighted partner ALT count

top_configurations : ranked leading configurations

top_probabilities : their unnormalized conditional probabilities

top_mass : total conditional mass of the retained configurations

Returns
-------
tuple: native float, uint8 array (L, k), float64 array (L,), and native float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def summarize_top_burden(
    configurations: np.ndarray, conditional_probabilities: np.ndarray, top_l: int
) -> tuple[float, np.ndarray, np.ndarray, float]:
    '''Rank candidates and compute the top-L probability-weighted ALT burden.

    Parameters
    ----------
    configurations : np.ndarray
        Binary candidate configurations.
    conditional_probabilities : np.ndarray
        Conditional probability of each candidate.
    top_l : int
        Positive number of leading candidates to retain.

    Returns
    -------
    mean_burden : float
        Probability-weighted partner ALT count within the retained set.
    top_configurations : np.ndarray
        Leading configurations in rank order.
    top_probabilities : np.ndarray
        Corresponding unnormalized conditional probabilities.
    top_mass : float
        Sum of the retained probabilities.

    Raises
    ------
    ValueError
        If `configurations` is not a nonempty two-dimensional binary array; if
        `conditional_probabilities` is not a finite nonnegative vector with one
        entry per candidate row; if `top_l` is not an integer between one and
        the number of candidates, inclusive; or if the retained candidates
        have zero total probability.
    '''
    return result  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_summarize_top_burden(
    configurations: np.ndarray, conditional_probabilities: np.ndarray, top_l: int
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Reference implementation of top-L ranking and burden averaging."""
    candidates = np.asarray(configurations)
    probabilities = np.asarray(conditional_probabilities, dtype=float)
    if candidates.ndim != 2 or candidates.shape[0] < 1 or candidates.shape[1] < 1:
        raise ValueError("configurations must be a nonempty two-dimensional array")
    if not np.isin(candidates, (0, 1)).all():
        raise ValueError("configurations must be binary")
    if probabilities.shape != (candidates.shape[0],) or not np.isfinite(probabilities).all():
        raise ValueError("conditional_probabilities must be finite and match candidate rows")
    if np.any(probabilities < 0.0):
        raise ValueError("conditional_probabilities must be nonnegative")
    if not isinstance(top_l, (int, np.integer)) or not 1 <= int(top_l) <= candidates.shape[0]:
        raise ValueError("top_l must be between one and the number of candidates")

    candidates = candidates.astype(np.uint8)
    order = sorted(
        range(candidates.shape[0]),
        key=lambda index: (-float(probabilities[index]), tuple(int(v) for v in candidates[index])),
    )[: int(top_l)]
    top_configurations = candidates[order]
    top_probabilities = probabilities[order]
    top_mass = float(np.sum(top_probabilities))
    if top_mass <= 0.0:
        raise ValueError("the retained candidates have zero probability")
    burdens = np.sum(top_configurations, axis=1, dtype=float)
    mean_burden = float(np.sum(top_probabilities * burdens) / top_mass)
    return mean_burden, top_configurations, top_probabilities, top_mass

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """configurations = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
conditional_probabilities = np.array([0.1, 0.4, 0.3, 0.2])
top_l = 3
def pack(result):
    mean_burden, top_configurations, top_probabilities, top_mass = result
    return np.concatenate((
        np.array([mean_burden], dtype=float),
        np.asarray(top_configurations, dtype=float).ravel(),
        np.asarray(top_probabilities, dtype=float).ravel(),
        np.array([top_mass], dtype=float),
    ))
""",
            "call": "pack(summarize_top_burden(configurations, conditional_probabilities, top_l))",
            "gold_call": "pack(_oracle_summarize_top_burden(configurations, conditional_probabilities, top_l))",
        },
        {
            "setup": """configurations = np.array([[0], [1]], dtype=np.uint8)
conditional_probabilities = np.array([0.5, 0.5])
top_l = 1
def pack(result):
    mean_burden, top_configurations, top_probabilities, top_mass = result
    return np.concatenate((
        np.array([mean_burden], dtype=float),
        np.asarray(top_configurations, dtype=float).ravel(),
        np.asarray(top_probabilities, dtype=float).ravel(),
        np.array([top_mass], dtype=float),
    ))
""",
            "call": "pack(summarize_top_burden(configurations, conditional_probabilities, top_l))",
            "gold_call": "pack(_oracle_summarize_top_burden(configurations, conditional_probabilities, top_l))",
        },
        {
            "setup": """configurations = np.array([[0, 1], [1, 0]], dtype=np.uint8)
conditional_probabilities = np.array([0.5, 0.5])
top_l = 3
def run_model():
    try:
        summarize_top_burden(configurations, conditional_probabilities, top_l)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_summarize_top_burden(configurations, conditional_probabilities, top_l)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        {
            "setup": """configurations = np.array([[1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)
conditional_probabilities = np.array([0.2, 0.2, 0.5, 0.1])
top_l = 2
def pack(result):
    mean_burden, top_configurations, top_probabilities, top_mass = result
    return np.concatenate((
        np.array([mean_burden], dtype=float),
        np.asarray(top_configurations, dtype=float).ravel(),
        np.asarray(top_probabilities, dtype=float).ravel(),
        np.array([top_mass], dtype=float),
    ))
""",
            "call": "pack(summarize_top_burden(configurations, conditional_probabilities, top_l))",
            "gold_call": "pack(_oracle_summarize_top_burden(configurations, conditional_probabilities, top_l))",
        },
    ]
