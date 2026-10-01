"""
Given a nonnegative irreducible projection matrix of size N, a number of groups m that divides N, and a whole number of intervals s, construct the reduced matrix that is consistent with the given matrix in growth rate, grouped stable distribution, reproductive values and elasticities.

Return the reduced matrix, the reduced reproductive values scaled so that their inner product with the grouped stable distribution is one when the stable distribution sums to one, and the effectiveness defined above.

The flow-matching collapse reproduces the growth rate and the grouped stable distribution, but not the reproductive values: its left eigenvector is not the average of the original reproductive values over each group, so the elasticities of the reduced model are not the sums of the original elasticities either. The reason is that it adds flows between classes whose individuals carry different reproductive values, as though an individual of every age were worth the same.

The quality of fit is measured by an effectiveness defined from the elasticities. Let $w$ and $v$ be the stable distribution and reproductive values of $M$, let $E$ be the elasticity matrix of $M^s$ defined in step 7, and let $G$ be the partitioning matrix of step 4. For class index $j$ and group indices $I$ and $J$, define the weights $q_j = v_j w_j / (v^T w)$, their group sums $Q = G q$, the row-grouped elasticities $H = G E$, and the group-pair elasticities $K = G E G^T$. The effectiveness is

$$(\\sum_{I,J} K_{IJ}^2 / Q_J) / (\\sum_{I,j} H_{Ij}^2 / q_j).$$

With a single group the fit is perfect and that effectiveness is one.

Returns
-------
dict holding the np.ndarray reduced_matrix of shape (m, m); the np.ndarray reduced_reproductive_values of shape (m,); and the float effectiveness defined above.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def elasticity_consistent_aggregate(
    matrix: np.ndarray,
    groups: int,
    steps: int,
) -> dict:
    """Collapse a projection matrix consistently with its reproductive values and elasticities.

    Parameters
    ----------
    matrix : np.ndarray
        Nonnegative irreducible projection matrix, shape (N, N).
    groups : int
        Number of reduced classes, dividing N.
    steps : int
        Number of original intervals spanned by one reduced interval, at least one.

    Returns
    -------
    dict
        Under the keys reduced_matrix, reduced_reproductive_values and effectiveness.

    Raises
    ------
    ValueError
        When the matrix fails the checks of the Perron triplet, when groups is not an integer
        dividing N, or when steps is not an integer of at least one.
        A count given as a bool, or as a float even when its value is integral, is
        not accepted as an integer.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numbers

import numpy as np


def _check_count(value, name, lower):
    """Validate an integer count not below a lower bound and return it as an int."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(name + " must be an integer")
    if int(value) < lower:
        raise ValueError(name + " must be at least " + str(lower))
    return int(value)


def _partition_matrix(size, groups):
    """The groups-by-size matrix that sums consecutive blocks of size // groups classes."""
    return np.kron(np.eye(groups), np.ones((1, size // groups)))


def _oracle_elasticity_consistent_aggregate(
    matrix: np.ndarray,
    groups: int,
    steps: int,
) -> dict:
    """Reference implementation."""
    triplet = _oracle_perron_triplet(matrix)  # noqa: F821
    a = np.asarray(matrix, dtype=float)
    size = a.shape[0]
    m = _check_count(groups, "groups", 1)
    s = _check_count(steps, "steps", 1)
    if m > size or size % m != 0:
        raise ValueError("groups must divide the size of the matrix")
    w = triplet["stable_distribution"]
    v = triplet["reproductive_values"]

    balanced = (v[:, None] * a) / v[None, :]
    collapsed = _oracle_interstage_flow_aggregate(balanced, m, s)  # noqa: F821

    g = _partition_matrix(size, m)
    reduced_v = (g @ (v * w)) / (g @ w)
    reduced = collapsed["reduced_matrix"] * reduced_v[None, :] / reduced_v[:, None]
    return {"reduced_matrix": reduced,
            "reduced_reproductive_values": reduced_v / float(reduced_v @ (g @ w)),
            "effectiveness": collapsed["effectiveness"]}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return self-contained, single-invocation numeric test cases."""
    setup = "import numpy as np\n\ndef _flatten(value):\n    if isinstance(value, dict):\n        parts = [_flatten(value[key]) for key in sorted(value)]\n        return np.concatenate(parts) if parts else np.empty(0, dtype=float)\n    if isinstance(value, (list, tuple)):\n        parts = [_flatten(item) for item in value]\n        return np.concatenate(parts) if parts else np.empty(0, dtype=float)\n    return np.asarray(value, dtype=float).ravel()\n\ndef project(value):\n    raw = _flatten(value)\n    missing = np.isnan(raw)\n    return np.concatenate((np.where(missing, 0.0, raw), missing.astype(float)))\n\ndef expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n\ndef expect_runtime_error(fn, *args):\n    try:\n        fn(*args)\n    except RuntimeError:\n        return np.array([1.0])\n    return np.array([0.0])\n\ndef leslie(fertility, survival):\n    matrix = np.zeros((len(fertility), len(fertility)))\n    matrix[0, :] = fertility\n    for index, probability in enumerate(survival):\n        matrix[index + 1, index] = probability\n    return matrix\n\nF = np.array([0.0, 0.4, 1.0, 1.4, 1.6, 1.8, 1.6])\nP = np.array([0.60, 0.72, 0.78, 0.76, 0.70, 0.50])\n\ndef resolved(fertility, survival, subdivisions):\n    classes = len(fertility)\n    matrix = np.zeros((classes * subdivisions, classes * subdivisions))\n    for age_class in range(1, classes + 1):\n        matrix[0, age_class * subdivisions - 1] = fertility[age_class - 1]\n    for index in range(1, classes * subdivisions):\n        matrix[index, index - 1] = survival[index // subdivisions - 1] if index % subdivisions == 0 else 1.0\n    return matrix\n"
    return [
        {
            "setup": setup + "",
            "call": "project(elasticity_consistent_aggregate(resolved(F, P, 3), 3, 7))",
            "gold_call": "project(_oracle_elasticity_consistent_aggregate(resolved(F, P, 3), 3, 7))",
        },
        {
            "setup": setup + "",
            "call": "project(elasticity_consistent_aggregate(resolved([0.0, 1.0, 5.0], [0.3, 0.5], 2), 2, 3))",
            "gold_call": "project(_oracle_elasticity_consistent_aggregate(resolved([0.0, 1.0, 5.0], [0.3, 0.5], 2), 2, 3))",
        },
        {
            "setup": setup + "",
            "call": "project(elasticity_consistent_aggregate(leslie(F, P), 7, 2))",
            "gold_call": "project(_oracle_elasticity_consistent_aggregate(leslie(F, P), 7, 2))",
        },
        {
            "setup": setup + "",
            "call": "expect_value_error(elasticity_consistent_aggregate, leslie(F, P), 3, 1)",
            "gold_call": "expect_value_error(_oracle_elasticity_consistent_aggregate, leslie(F, P), 3, 1)",
        },
    ]
