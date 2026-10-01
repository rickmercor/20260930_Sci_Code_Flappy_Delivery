"""
Given a nonnegative irreducible projection matrix of size N, a number of groups m that divides N, a whole number of intervals s, and a candidate reduced matrix of size m, return four nonnegative residuals. The growth-rate residual is the absolute difference between the Perron root of the candidate and lambda^s, divided by lambda^s. The stable-structure residual is the largest absolute difference between the candidate's stable distribution and the grouped stable distribution of M, both summing to one. The reproductive-value residual is the largest absolute difference between the candidate's reproductive values and the stable-distribution-weighted group averages of the reproductive values of M, divided by the largest of those averages, with both vectors scaled to unit inner product with the grouped stable distribution. The elasticity residual is the largest absolute difference between the elasticity matrix of the candidate and the elasticity matrix of M^s summed over pairs of groups. Obtain every eigen-quantity from the Perron triplet and the elasticities of the candidate from the elasticity matrix. Form the elasticities of M^s from the Perron triplet of M itself, as v_i (M^s)_ij w_j divided by lambda^s and by the inner product of v and w: M^s shares its Perron vectors with M, but it is reducible whenever the index of imprimitivity of M shares a factor with s, as it does for a semelparous model resolved into sub-classes, and its own Perron triplet is then not defined.

A reduced model of a projection matrix M, with m groups of adjacent classes projecting over s intervals, is consistent with M when four statements hold at once. Its growth rate is lambda^s, the growth rate of M over the same span of time. Its stable distribution, normalised to sum to one, is the grouped stable distribution of M. Its reproductive values, scaled to unit inner product with that distribution, are the averages of the reproductive values of M over each group weighted by the stable distribution. And its elasticity matrix is the elasticity matrix of M^s summed over pairs of groups, entry (I, J) of the reduced elasticities being the sum of the elasticities of M^s over all rows in group I and all columns in group J.

The last statement is the one that comparative demography relies on and the one that the flow-matching collapse fails. The elasticities must be those of the s-step projection M^s rather than of M itself, because the reduced model replaces s steps of the original at once: the summed elasticities of M do not even have the Leslie pattern, since they place survival within a group on the diagonal. Summing the elasticities of M^s over pairs of groups gives a matrix with the Leslie pattern whenever M is a Leslie matrix and the groups are blocks of whole classes, and it sums to one.

Measuring the four discrepancies separately is how the two constructions are told apart numerically. The flow-matching collapse leaves the first two at rounding level and the last two at the size of the aggregation error; the consistent reduction leaves all four at rounding level.

Returns
-------
dict holding the floats growth_rate_residual, stable_structure_residual, reproductive_value_residual and elasticity_residual.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def consistency_residuals(
    matrix: np.ndarray,
    groups: int,
    steps: int,
    reduced_matrix: np.ndarray,
) -> dict:
    """Discrepancies of a reduced model from the four consistency requirements.

    Parameters
    ----------
    matrix : np.ndarray
        Nonnegative irreducible projection matrix, shape (N, N).
    groups : int
        Number of reduced classes, dividing N.
    steps : int
        Number of original intervals spanned by one reduced interval, at least one.
    reduced_matrix : np.ndarray
        Candidate reduced model, nonnegative irreducible, shape (m, m).

    Returns
    -------
    dict
        Under the keys growth_rate_residual, stable_structure_residual,
        reproductive_value_residual and elasticity_residual.

    Raises
    ------
    ValueError
        When either matrix fails the checks of the Perron triplet, when groups is not an integer
        dividing N, when steps is not an integer of at least one, or when the candidate does not
        have shape (m, m).
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


def _oracle_consistency_residuals(
    matrix: np.ndarray,
    groups: int,
    steps: int,
    reduced_matrix: np.ndarray,
) -> dict:
    """Reference implementation."""
    fine = _oracle_perron_triplet(matrix)  # noqa: F821
    a = np.asarray(matrix, dtype=float)
    size = a.shape[0]
    m = _check_count(groups, "groups", 1)
    s = _check_count(steps, "steps", 1)
    if m > size or size % m != 0:
        raise ValueError("groups must divide the size of the matrix")
    b = np.asarray(reduced_matrix)
    if b.ndim != 2 or b.shape != (m, m):
        raise ValueError("reduced_matrix must have shape (groups, groups)")
    coarse = _oracle_perron_triplet(b)  # noqa: F821

    g = _partition_matrix(size, m)
    w = fine["stable_distribution"]
    v = fine["reproductive_values"]
    target_rate = fine["growth_rate"] ** s
    grouped_w = g @ w
    averaged_v = (g @ (v * w)) / grouped_w
    averaged_v = averaged_v / float(averaged_v @ grouped_w)
    reduced_v = coarse["reproductive_values"] / float(coarse["reproductive_values"] @ grouped_w)

    fine_elasticity = np.outer(v, w) * np.linalg.matrix_power(a, s) / (target_rate * float(v @ w))
    coarse_elasticity = _oracle_elasticity_matrix(b)  # noqa: F821
    return {
        "growth_rate_residual": abs(coarse["growth_rate"] - target_rate) / target_rate,
        "stable_structure_residual": float(np.max(np.abs(coarse["stable_distribution"] - grouped_w))),
        "reproductive_value_residual": float(np.max(np.abs(reduced_v - averaged_v)) / np.max(averaged_v)),
        "elasticity_residual": float(np.max(np.abs(coarse_elasticity - g @ fine_elasticity @ g.T))),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return self-contained, single-invocation numeric test cases."""
    setup = "import numpy as np\n\ndef _flatten(value):\n    if isinstance(value, dict):\n        parts = [_flatten(value[key]) for key in sorted(value)]\n        return np.concatenate(parts) if parts else np.empty(0, dtype=float)\n    if isinstance(value, (list, tuple)):\n        parts = [_flatten(item) for item in value]\n        return np.concatenate(parts) if parts else np.empty(0, dtype=float)\n    return np.asarray(value, dtype=float).ravel()\n\ndef project(value):\n    raw = _flatten(value)\n    missing = np.isnan(raw)\n    return np.concatenate((np.where(missing, 0.0, raw), missing.astype(float)))\n\ndef expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n\ndef expect_runtime_error(fn, *args):\n    try:\n        fn(*args)\n    except RuntimeError:\n        return np.array([1.0])\n    return np.array([0.0])\n\ndef leslie(fertility, survival):\n    matrix = np.zeros((len(fertility), len(fertility)))\n    matrix[0, :] = fertility\n    for index, probability in enumerate(survival):\n        matrix[index + 1, index] = probability\n    return matrix\n\nF = np.array([0.0, 0.4, 1.0, 1.4, 1.6, 1.8, 1.6])\nP = np.array([0.60, 0.72, 0.78, 0.76, 0.70, 0.50])\n\ndef resolved(fertility, survival, subdivisions):\n    classes = len(fertility)\n    matrix = np.zeros((classes * subdivisions, classes * subdivisions))\n    for age_class in range(1, classes + 1):\n        matrix[0, age_class * subdivisions - 1] = fertility[age_class - 1]\n    for index in range(1, classes * subdivisions):\n        matrix[index, index - 1] = survival[index // subdivisions - 1] if index % subdivisions == 0 else 1.0\n    return matrix\n\nC = resolved(F, P, 3)\nSTANDARD = _oracle_interstage_flow_aggregate(C, 3, 7)[\"reduced_matrix\"]\nBALANCED = _oracle_elasticity_consistent_aggregate(C, 3, 7)[\"reduced_matrix\"]\nBENT = BALANCED.copy()\nBENT[0, 1] *= 1.1\nBENT[0, 2] *= 0.8\n"
    return [
        {
            "setup": setup + "",
            "call": "project(consistency_residuals(C, 3, 7, STANDARD))",
            "gold_call": "project(_oracle_consistency_residuals(C, 3, 7, STANDARD))",
        },
        {
            "setup": setup + "",
            "call": "project(consistency_residuals(C, 3, 7, BALANCED))",
            "gold_call": "project(_oracle_consistency_residuals(C, 3, 7, BALANCED))",
        },
        {
            "setup": setup + "",
            "call": "project(consistency_residuals(C, 3, 7, BENT))",
            "gold_call": "project(_oracle_consistency_residuals(C, 3, 7, BENT))",
        },
        {
            "setup": setup + "",
            "call": "expect_value_error(consistency_residuals, leslie(F, P), 7, 1, np.eye(3))",
            "gold_call": "expect_value_error(_oracle_consistency_residuals, leslie(F, P), 7, 1, np.eye(3))",
        },
    ]
