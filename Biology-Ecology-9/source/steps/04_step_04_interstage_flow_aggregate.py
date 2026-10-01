"""
Given a nonnegative irreducible projection matrix of size N, a number of groups m that divides N, and a whole number of intervals s, return the reduced matrix B = G M^s Q built from the stable distribution of the given matrix, together with its effectiveness and the grouped stable distribution G w. Group I consists of the classes (I - 1) N/m + 1 to I N/m. Obtain the stable distribution from the Perron triplet of the given matrix.

With m equal to N the reduced matrix is M^s itself and the effectiveness is one.

A projection matrix M of size N advances the population by one interval. Merging its classes into m groups of N/m adjacent classes and advancing by s intervals at once gives an m-by-m model B that should satisfy the perfect-aggregation relation G M^s = B G, where G is the m-by-N partitioning matrix whose row I holds ones over the classes of group I and zeros elsewhere. That relation is overdetermined, with more equations than unknowns, so it has no exact solution in general and B has to be chosen by a weighted least-squares fit.

The interstage flow of a projection matrix is the matrix multiplied on the right by the diagonal matrix of its stable distribution, so that its entry in row i and column j is the number of individuals moving from class j to class i per interval in the stable population. Flows, unlike rates, can be added across classes. Requiring the flow matrix of the reduced model to equal the summed flows of M^s, group by group, gives B diag(G w) = G M^s diag(w) G^T, which is solved by B = G M^s Q with Q = diag(w) G^T (G diag(w) G^T)^(-1). This is the same matrix that minimises the squared residual of G M^s = B G with each column of the residual weighted by the square root of the stable distribution, and it is the long-established way to collapse an age-structured model.

The fit is diagnosed by an effectiveness, the analogue of a coefficient of determination: the squared Frobenius norm of B G diag(w)^(1/2) divided by that of G M^s diag(w)^(1/2). It equals one when the aggregation is perfect and is below one otherwise. Because the stable distribution satisfies M w = lambda w, the reduced model has growth rate lambda^s and stable distribution G w exactly, whatever the quality of the fit.

Returns
-------
dict holding the np.ndarray reduced_matrix of shape (m, m); the float effectiveness; and the np.ndarray grouped_distribution of shape (m,), the stable distribution summed over each group.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def interstage_flow_aggregate(
    matrix: np.ndarray,
    groups: int,
    steps: int,
) -> dict:
    """Collapse a projection matrix by matching summed interstage flows.

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
        Under the keys reduced_matrix, effectiveness and grouped_distribution.

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


def _oracle_interstage_flow_aggregate(
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
    g = _partition_matrix(size, m)
    power = np.linalg.matrix_power(a, s)
    weight = np.diag(w)
    q = weight @ g.T @ np.linalg.inv(g @ weight @ g.T)
    reduced = g @ power @ q

    root = np.diag(np.sqrt(w))
    fitted = float(np.sum((reduced @ g @ root) ** 2))
    actual = float(np.sum((g @ power @ root) ** 2))
    return {"reduced_matrix": reduced,
            "effectiveness": fitted / actual,
            "grouped_distribution": g @ w}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return self-contained, single-invocation numeric test cases."""
    setup = "import numpy as np\n\ndef _flatten(value):\n    if isinstance(value, dict):\n        parts = [_flatten(value[key]) for key in sorted(value)]\n        return np.concatenate(parts) if parts else np.empty(0, dtype=float)\n    if isinstance(value, (list, tuple)):\n        parts = [_flatten(item) for item in value]\n        return np.concatenate(parts) if parts else np.empty(0, dtype=float)\n    return np.asarray(value, dtype=float).ravel()\n\ndef project(value):\n    raw = _flatten(value)\n    missing = np.isnan(raw)\n    return np.concatenate((np.where(missing, 0.0, raw), missing.astype(float)))\n\ndef expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n\ndef expect_runtime_error(fn, *args):\n    try:\n        fn(*args)\n    except RuntimeError:\n        return np.array([1.0])\n    return np.array([0.0])\n\ndef leslie(fertility, survival):\n    matrix = np.zeros((len(fertility), len(fertility)))\n    matrix[0, :] = fertility\n    for index, probability in enumerate(survival):\n        matrix[index + 1, index] = probability\n    return matrix\n\nF = np.array([0.0, 0.4, 1.0, 1.4, 1.6, 1.8, 1.6])\nP = np.array([0.60, 0.72, 0.78, 0.76, 0.70, 0.50])\n\ndef resolved(fertility, survival, subdivisions):\n    classes = len(fertility)\n    matrix = np.zeros((classes * subdivisions, classes * subdivisions))\n    for age_class in range(1, classes + 1):\n        matrix[0, age_class * subdivisions - 1] = fertility[age_class - 1]\n    for index in range(1, classes * subdivisions):\n        matrix[index, index - 1] = survival[index // subdivisions - 1] if index % subdivisions == 0 else 1.0\n    return matrix\n"
    return [
        {
            "setup": setup + "",
            "call": "project(interstage_flow_aggregate(resolved(F, P, 3), 3, 7))",
            "gold_call": "project(_oracle_interstage_flow_aggregate(resolved(F, P, 3), 3, 7))",
        },
        {
            "setup": setup + "",
            "call": "project(interstage_flow_aggregate(resolved([0.0, 1.0, 5.0], [0.3, 0.5], 2), 2, 3))",
            "gold_call": "project(_oracle_interstage_flow_aggregate(resolved([0.0, 1.0, 5.0], [0.3, 0.5], 2), 2, 3))",
        },
        {
            "setup": setup + "",
            "call": "project(interstage_flow_aggregate(leslie(F, P), 1, 7))",
            "gold_call": "project(_oracle_interstage_flow_aggregate(leslie(F, P), 1, 7))",
        },
        {
            "setup": setup + "",
            "call": "expect_value_error(interstage_flow_aggregate, leslie(F, P), 2, 1)",
            "gold_call": "expect_value_error(_oracle_interstage_flow_aggregate, leslie(F, P), 2, 1)",
        },
    ]
