"""
Reduce the pairwise separation and cost matrices to the source-defined assignment distance and within-range fraction.

Active spectra may have unequal peak counts. Preserve separation and cost alignment; the pinned source specifies the unequal-cardinality representation, reduction, and boundary classification.

Returns
-------
A tuple of two floats: source-defined assignment distance and within-range fraction.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def modified_hungarian_distance(
    distances: "np.ndarray",
    costs: "np.ndarray",
    tolerance: float,
) -> "tuple[float, float]":
    """Return the source-defined assignment distance and match fraction.

    Preserve the alignment of the supplied separation and cost matrices.
    Follow the pinned unequal-cardinality reduction and boundary convention.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_modified_hungarian_distance(
    distances: "np.ndarray",
    costs: "np.ndarray",
    tolerance: float,
) -> "tuple[float, float]":
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    d = np.asarray(distances, dtype=float)
    c = np.asarray(costs, dtype=float)

    if d.ndim != 2 or d.shape != c.shape or min(d.shape) == 0:
        raise ValueError("equal nonempty matrices required")

    n_a, n_b = d.shape
    size = max(n_a, n_b)

    padded_cost = np.full(
        (size, size),
        float(tolerance),
        dtype=float,
    )
    padded_distance = np.full(
        (size, size),
        float(tolerance),
        dtype=float,
    )

    padded_cost[:n_a, :n_b] = c
    padded_distance[:n_a, :n_b] = d

    rows, cols = linear_sum_assignment(padded_cost)

    return (
        float(np.mean(padded_cost[rows, cols])),
        float(
            np.mean(
                padded_distance[rows, cols] <= tolerance
            )
        ),
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import copy\n'
               'import numpy as np\n'
               'from scipy.optimize import linear_sum_assignment\n'
               'd=np.array([[1.,9.],[8.,2.]]); c=d.copy(); t=10.',
      'call': 'np.asarray(modified_hungarian_distance(copy.deepcopy(d), copy.deepcopy(c), '
              'copy.deepcopy(t)), dtype=float)',
      'gold_call': 'np.asarray(_oracle_modified_hungarian_distance(copy.deepcopy(d), '
                   'copy.deepcopy(c), copy.deepcopy(t)), dtype=float)'},
     {'setup': 'import copy\n'
               'import numpy as np\n'
               'from scipy.optimize import linear_sum_assignment\n'
               'd=np.array([[1.,30.],[2.,25.],[3.,20.],[4.,15.]]); c=np.where(d<=10.,d,d+1.); '
               't=10.',
      'call': 'np.asarray(modified_hungarian_distance(copy.deepcopy(d), copy.deepcopy(c), '
              'copy.deepcopy(t)), dtype=float)',
      'gold_call': 'np.asarray(_oracle_modified_hungarian_distance(copy.deepcopy(d), '
                   'copy.deepcopy(c), copy.deepcopy(t)), dtype=float)'},
     {'setup': 'import copy\n'
               'import numpy as np\n'
               'from scipy.optimize import linear_sum_assignment\n'
               'd=np.array([[10.,1.],[1.,10.]]); c=np.array([[.1,5.],[5.,.1]]); t=10.',
      'call': 'np.asarray(modified_hungarian_distance(copy.deepcopy(d), copy.deepcopy(c), '
              'copy.deepcopy(t)), dtype=float)',
      'gold_call': 'np.asarray(_oracle_modified_hungarian_distance(copy.deepcopy(d), '
                   'copy.deepcopy(c), copy.deepcopy(t)), dtype=float)'}]
