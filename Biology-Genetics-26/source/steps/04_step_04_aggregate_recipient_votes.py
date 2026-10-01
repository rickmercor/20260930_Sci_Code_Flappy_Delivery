"""
Convert reconciliation-triple appearances into one direction vote per gene tree.

Within a reconciled gene tree, the candidate branch appearing more often among NNI triples is favored as the recipient. Compressed topology multiplicities must be expanded at the voting stage so that every original gene tree contributes one vote. Exact within-tree ties are random but reproducible under a fixed seed.

Returns
-------
numpy.ndarray of three integers [test_votes, focal_votes, recipient_index]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def aggregate_recipient_votes(
    reconciliation_summary: np.ndarray,
    multiplicities: np.ndarray,
    seed: int,
) -> np.ndarray:
    '''Aggregate one recipient vote per represented gene tree.

    Parameters
    ----------
    reconciliation_summary : np.ndarray
        Columns [minimum_nni_distance, number_of_shortest_paths, test_clade_appearances, focal_clade_appearances].
    multiplicities : np.ndarray
        Positive integer multiplicity for each row.
    seed : int
        Seed for np.random.default_rng; untied rows consume no draws, tied rows consume multiplicity draws in input order, and an aggregate tie consumes one final draw.

    Returns
    -------
    vote_summary : np.ndarray
        Test votes, focal votes, and the selected recipient index.

    Raises
    ------
    ValueError
        If reconciliation_summary is not a nonempty (p, 4) array of nonnegative integers, multiplicities are not one positive integer per row, or seed is not an integer.
    '''
    return vote_summary

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_aggregate_recipient_votes(
    reconciliation_summary: np.ndarray,
    multiplicities: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Reference implementation."""
    summary = np.asarray(reconciliation_summary)
    counts = np.asarray(multiplicities)
    if summary.ndim != 2 or summary.shape[1] != 4 or summary.shape[0] == 0:
        raise ValueError("reconciliation_summary must have shape (p, 4)")
    if not np.issubdtype(summary.dtype, np.integer) or np.any(summary < 0):
        raise ValueError("reconciliation_summary must contain nonnegative integers")
    if counts.shape != (summary.shape[0],) or not np.issubdtype(
        counts.dtype,
        np.integer,
    ):
        raise ValueError("multiplicities must be one integer per summary row")
    if np.any(counts <= 0):
        raise ValueError("multiplicities must be positive")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(int(seed))
    test_votes = 0
    focal_votes = 0
    for row, multiplicity in zip(summary, counts):
        test_appearances = int(row[2])
        focal_appearances = int(row[3])
        number = int(multiplicity)
        if test_appearances > focal_appearances:
            test_votes += number
        elif focal_appearances > test_appearances:
            focal_votes += number
        else:
            assignments = rng.integers(0, 2, size=number)
            test_votes += int(np.count_nonzero(assignments == 0))
            focal_votes += int(np.count_nonzero(assignments == 1))
    if test_votes > focal_votes:
        recipient_index = 0
    elif focal_votes > test_votes:
        recipient_index = 1
    else:
        recipient_index = int(rng.integers(0, 2))
    return np.asarray([test_votes, focal_votes, recipient_index], dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": '''summary = np.array([
    [2, 1, 2, 1],
    [3, 4, 2, 2],
    [2, 1, 1, 2],
    [1, 1, 0, 1],
    [2, 1, 1, 1],
], dtype=int)
multiplicities = np.array([11, 7, 13, 12, 7], dtype=int)
expected = [20, 30, 1]
def check(value):
    result = value.tolist()
    if result != expected:
        raise AssertionError("unexpected prompt-instance vote summary")
    return result
''',
            "call": "check(aggregate_recipient_votes(summary, multiplicities, 17))",
            "gold_call": "check(_oracle_aggregate_recipient_votes(summary, multiplicities, 17))",
        },
        {
            "setup": '''summary = np.array([
    [1, 1, 2, 1],
    [2, 3, 1, 1],
    [1, 1, 0, 1],
], dtype=int)
multiplicities = np.array([1, 2, 1], dtype=int)
expected = [2, 2, 1]
def check(value):
    result = value.tolist()
    if result != expected:
        raise AssertionError("unexpected aggregate-tie vote summary")
    return result
''',
            "call": "check(aggregate_recipient_votes(summary, multiplicities, 1))",
            "gold_call": "check(_oracle_aggregate_recipient_votes(summary, multiplicities, 1))",
        },
        {
            "setup": '''summary = np.array([[1, 1, 2, 1]], dtype=int)
multiplicities = np.array([0], dtype=int)
def error_code(function):
    try:
        function(summary, multiplicities, 3)
    except ValueError:
        return 2
    except Exception:
        return 3
    return 0
''',
            "call": "error_code(aggregate_recipient_votes)",
            "gold_call": "error_code(_oracle_aggregate_recipient_votes)",
        },
    ]
