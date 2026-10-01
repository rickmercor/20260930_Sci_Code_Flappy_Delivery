"""
Reduce the raw partial-path type counts of every ensemble to the local crossing probabilities that drive the chain.

A partial-path ensemble centred on an interface collects only trajectory segments that touch that interface and terminate on one of its two neighbours. Each segment therefore carries two labels beyond the interface itself: the neighbour it arrived from and the neighbour it departed to. Sorting the sampled segments by that pair gives four populations for a generic ensemble, and the quantity that governs how a long trajectory threads through the interface set is the conditional probability of departing towards one neighbour given the neighbour of arrival. Because departure is exhaustive - a segment that crosses the interface must eventually terminate on one side or the other - the two probabilities conditioned on the same arrival side are complementary, so each ensemble contributes exactly two independent numbers.

Conditioning on the arrival side rather than pooling the segments is what separates this construction from a memoryless description of the same trajectory. A segment that arrived from below has, in general, a different chance of progressing than one that arrived from above, and discarding that distinction discards precisely the memory that the sampling scheme was designed to retain. The ensemble that straddles the boundary of the reactant state is degenerate: a segment cannot both arrive from the interface above and depart towards it, because touching the boundary interface commits the trajectory to the reactant state, so one of the four populations is empty by construction and the corresponding conditional probability collapses to certainty.


Formulas:

    p[j, k, l] = n[j, k, l] / (n[j, k, -1] + n[j, k, +1])

    p[j, k, -1] + p[j, k, +1] = 1

Returns
-------
numpy.ndarray of shape (n_ensembles, 2, 2): local crossing probabilities, each row-pair summing to one.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_local_crossing_probabilities(path_counts) -> np.ndarray:
    """Convert sampled path-type populations into local crossing probabilities.

    Parameters
    ----------
    path_counts : array_like
        Numeric array of shape (n_ensembles, 4) whose entries are whole numbers
        of sampled paths; any numeric dtype is accepted, including float. Row j
        holds the sampled populations of ensemble j in the fixed order (arrive
        below / depart below, arrive below / depart above, arrive above / depart
        below, arrive above / depart above). At least two ensembles are
        required. Entries must be non-negative, and the two populations sharing
        an arrival side must not both vanish.

    Returns
    -------
    probabilities : numpy.ndarray
        Float array of shape (n_ensembles, 2, 2). Element [j, a, b] is the
        probability that a segment of ensemble j which arrived from the side
        indexed by a departs towards the side indexed by b, where index 0 means
        below and index 1 means above.

    Raises
    ------
    ValueError
        If any sampled population is negative or if the two populations
        sharing either arrival side both vanish.
    """
    return np.zeros((0, 2, 2))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_local_crossing_probabilities(path_counts) -> np.ndarray:
    # Local import: the harness compiles this oracle on its own, so the
    # module-level import is not in scope here.
    import numpy as np

    counts = np.asarray(path_counts)
    if counts.dtype == bool:
        raise ValueError("path_counts must be numeric, not boolean")
    if not np.issubdtype(counts.dtype, np.number):
        raise ValueError("path_counts must be numeric")
    counts = counts.astype(float)
    if counts.ndim != 2 or counts.shape[1] != 4:
        raise ValueError("path_counts must have shape (n_ensembles, 4)")
    if counts.shape[0] < 2:
        raise ValueError("at least two ensembles are required")
    if not np.all(np.isfinite(counts)):
        raise ValueError("path_counts must be finite")
    if np.any(counts < 0.0):
        raise ValueError("path_counts must be non-negative")
    if np.any(np.abs(counts - np.rint(counts)) > 1e-9):
        raise ValueError("path_counts must be whole numbers of sampled paths")

    # The four populations of a row split into two arrival groups, each holding
    # the segments that leave downwards and upwards respectively.
    grouped = counts.reshape(counts.shape[0], 2, 2)
    totals = grouped.sum(axis=2)
    if np.any(totals <= 0.0):
        raise ValueError("every arrival side must carry at least one sampled path")

    # Departure is exhaustive, so normalising within an arrival group is the
    # whole content of the estimator. An empty departure population is a
    # genuine zero, not a missing measurement.
    probabilities = grouped / totals[:, :, None]

    return probabilities

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: probability of progressing after arriving from below ---
        {
            "setup": """import numpy as np
counts = np.array([[6142, 1358, 2087, 0],
                   [2894, 2106, 1449, 2371],
                   [2704, 2096, 1583, 2297],
                   [2296, 2204, 1704, 2196]], dtype=float)
""",
            "call": "float(compute_local_crossing_probabilities(counts)[2, 0, 1])",
            "gold_call": "float(_oracle_compute_local_crossing_probabilities(counts)[2, 0, 1])",
        },
        # --- Valid: probability of progressing after arriving from above ---
        {
            "setup": """import numpy as np
counts = np.array([[6142, 1358, 2087, 0],
                   [2894, 2106, 1449, 2371],
                   [2704, 2096, 1583, 2297],
                   [2296, 2204, 1704, 2196]], dtype=float)
""",
            "call": "float(compute_local_crossing_probabilities(counts)[1, 1, 1])",
            "gold_call": "float(_oracle_compute_local_crossing_probabilities(counts)[1, 1, 1])",
        },
        # --- Boundary: the degenerate straddling ensemble, whose upper arrival
        #     group has an empty progression population ---
        {
            "setup": """import numpy as np
counts = np.array([[6142, 1358, 2087, 0],
                   [2894, 2106, 1449, 2371],
                   [2704, 2096, 1583, 2297]], dtype=float)
""",
            "call": "float(compute_local_crossing_probabilities(counts)[0, 1, 0])",
            "gold_call": "float(_oracle_compute_local_crossing_probabilities(counts)[0, 1, 0])",
        },
        # --- Edge: complementarity of every conditional pair, reduced to one
        #     scalar over the whole table ---
        {
            "setup": """import numpy as np
counts = np.array([[512, 41, 97, 0],
                   [77, 1023, 66, 8],
                   [1, 1, 1, 1],
                   [900, 100, 250, 750]], dtype=float)
""",
            "call": "float(np.sum(compute_local_crossing_probabilities(counts)))",
            "gold_call": "float(np.sum(_oracle_compute_local_crossing_probabilities(counts)))",
        },
        # --- Invalid: an arrival side with no sampled paths at all ---
        {
            "setup": """import numpy as np
counts = np.array([[6142, 1358, 0, 0],
                   [2894, 2106, 1449, 2371]], dtype=float)
def run_model():
    try:
        compute_local_crossing_probabilities(counts)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_local_crossing_probabilities(counts)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative population ---
        {
            "setup": """import numpy as np
counts = np.array([[6142, 1358, 2087, 0],
                   [2894, -2106, 1449, 2371]], dtype=float)
def run_model():
    try:
        compute_local_crossing_probabilities(counts)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_local_crossing_probabilities(counts)
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
