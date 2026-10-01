"""
Turn the per-tree posterior probability of the archaic state into called ancestry

tracts by taking maximal runs of consecutive trees above a posterior threshold and

keeping only those runs that are long enough in both physical and genetic units.

Posterior decoding gives a probability per marginal tree, but the object of

interest is a tract of contiguous sequence, so adjacent trees whose posterior

exceeds the threshold are merged into one candidate and the candidate is measured

along the chromosome. Two independent length filters are applied to it. The

physical filter, in base pairs, removes short calls that reflect uncertainty in

the reconstructed genealogy rather than ancestry. The genetic filter, in

centimorgans, is the one that separates introgression from incomplete lineage

sorting: for an instantaneous pulse t_admix generations in the past the expected

genetic length of a surviving introgressed tract is of order 100 / t_admix

centimorgans, while tracts that predate the population split have had far longer

to be broken down by recombination and are shorter. A candidate has to clear both

thresholds, because recombination rate varies along the genome and a tract that is

long in base pairs may still be short in recombination distance, and it is the

recombination distance that carries the information about the age of the event.

Returns
-------
list of (int, int) tuples, the inclusive first and last tree index of every retained tract, ordered by position along the chromosome
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def archaic_segments(
    posterior_archaic: np.ndarray,
    span_bp: np.ndarray,
    span_cm: np.ndarray,
    post_threshold: float = 0.9,
    min_bp: float = 5e4,
    min_cm: float = 0.05,
) -> list:
    """Return the retained archaic tracts as inclusive tree index ranges.

    Parameters
    ----------
    posterior_archaic : np.ndarray
        Posterior probability of the archaic state, of shape (m,).
    span_bp : np.ndarray
        Physical span of each marginal tree in base pairs.
    span_cm : np.ndarray
        Genetic span of each marginal tree in centimorgans.
    post_threshold : float
        Posterior probability a tree must exceed to join a candidate tract.
    min_bp : float
        Smallest admissible physical length of a retained tract.
    min_cm : float
        Smallest admissible genetic length of a retained tract.

    Returns
    -------
    segments : list
        List of (start, end) inclusive tree index pairs of the retained tracts.

    Raises
    ------
    ValueError
        If the three arrays are not one dimensional of equal length, if any span
        is negative, if post_threshold does not lie strictly between 0 and 1, or
        if min_bp or min_cm is negative.
    """
    return segments

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_archaic_segments(
    posterior_archaic: np.ndarray,
    span_bp: np.ndarray,
    span_cm: np.ndarray,
    post_threshold: float = 0.9,
    min_bp: float = 5e4,
    min_cm: float = 0.05,
) -> list:
    """Reference implementation."""
    posterior_archaic = np.asarray(posterior_archaic, dtype=float)
    span_bp = np.asarray(span_bp, dtype=float)
    span_cm = np.asarray(span_cm, dtype=float)
    if posterior_archaic.ndim != 1 or span_bp.ndim != 1 or span_cm.ndim != 1:
        raise ValueError("posterior_archaic, span_bp and span_cm must be one dimensional")
    if not (posterior_archaic.size == span_bp.size == span_cm.size):
        raise ValueError("posterior_archaic, span_bp and span_cm must have equal length")
    if np.any(span_bp < 0.0) or np.any(span_cm < 0.0):
        raise ValueError("spans must be non-negative")
    if not np.isfinite(post_threshold) or not (0.0 < post_threshold < 1.0):
        raise ValueError("post_threshold must lie strictly between 0 and 1")
    if min_bp < 0.0 or min_cm < 0.0:
        raise ValueError("min_bp and min_cm must be non-negative")

    called = posterior_archaic > post_threshold
    segments = []
    i = 0
    m = called.size
    while i < m:
        if not called[i]:
            i += 1
            continue
        j = i
        while j + 1 < m and called[j + 1]:
            j += 1
        if span_bp[i:j + 1].sum() >= min_bp and span_cm[i:j + 1].sum() >= min_cm:
            segments.append((int(i), int(j)))
        i = j + 1
    return segments

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    fixture = '''import numpy as np

_ARCHAIC_POSTERIOR = np.array([
    0.0007, 0.0004, 0.0009, 0.0001, 1.0, 1.0, 1.0, 0.9923,
    0.0048, 0.0, 0.8767, 0.0007, 1.0, 0.0016, 0.0, 0.0001,
])
_PHYSICAL_SPAN = np.array([
    15800.0, 12500.0, 14300.0, 13000.0, 13000.0, 12000.0, 13700.0, 12000.0,
    13300.0, 11400.0, 58000.0, 11400.0, 62000.0, 13900.0, 11600.0, 13700.0,
])
_GENETIC_SPAN = np.array([
    0.016, 0.0141, 0.0157, 0.0144, 0.013, 0.012, 0.0147, 0.012,
    0.0145, 0.0138, 0.061, 0.0119, 0.028, 0.0169, 0.0133, 0.0161,
])
'''
    return [
        # --- Normal scenario: one tract passes, one fails the genetic filter ---
        {
            "setup": fixture + """posterior_archaic = _ARCHAIC_POSTERIOR.copy()
span_bp = _PHYSICAL_SPAN.copy()
span_cm = _GENETIC_SPAN.copy()
""",
            "call": "[list(s) for s in archaic_segments(posterior_archaic, span_bp, span_cm)]",
            "gold_call": "[list(s) for s in _oracle_archaic_segments(posterior_archaic, span_bp, span_cm)]",
        },
        # --- Boundary case: both filters switched off keeps every candidate run ---
        {
            "setup": fixture + """posterior_archaic = _ARCHAIC_POSTERIOR.copy()
span_bp = _PHYSICAL_SPAN.copy()
span_cm = _GENETIC_SPAN.copy()
""",
            "call": "[list(s) for s in archaic_segments(posterior_archaic, span_bp, span_cm, min_bp=0.0, min_cm=0.0)]",
            "gold_call": "[list(s) for s in _oracle_archaic_segments(posterior_archaic, span_bp, span_cm, min_bp=0.0, min_cm=0.0)]",
        },
        # --- Edge case: a posterior threshold of exactly one ---
        {
            "setup": fixture + """posterior_archaic = _ARCHAIC_POSTERIOR.copy()
span_bp = _PHYSICAL_SPAN.copy()
span_cm = _GENETIC_SPAN.copy()
def run_model():
    try:
        archaic_segments(posterior_archaic, span_bp, span_cm, post_threshold=1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_archaic_segments(posterior_archaic, span_bp, span_cm, post_threshold=1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
