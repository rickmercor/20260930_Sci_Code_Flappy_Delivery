"""
Validate the representation-independent depth-search certificate table and

return the minimum feasible synthesis depth together with the exact global

optima of all rejected shallower depths.



Depths must be consecutive starting from 1. The threshold_met field is a

Boolean-valued numeric flag and must be exactly 0 or 1. The entire input table

is validated before any first-feasible row is accepted.

The input table has rows



[depth, threshold_met, certificate].



depth must contain consecutive integer depths beginning at 1.



threshold_met is discrete and must be exactly 0 or exactly 1. Values such as

2, -1, or 0.5 are malformed.



epsilon must be finite and satisfy



0 <= epsilon < 1,



so the requested fidelity threshold is 1-epsilon.



For a rejected row, threshold_met=0 and certificate must be below the

threshold.



For a feasible row, threshold_met=1 and the public certificate must equal the

requested threshold, matching the fixed-depth public contract.



The returned array begins with the first feasible depth and is followed by the

exact globally optimized fidelities of all preceding rejected depths.

Returns
-------
np.ndarray: [minimum feasible depth, followed by the exact global optima of every rejected shallower depth]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def select_first_feasible_depth(
    depth_results: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    Validate depth-search results and select the first feasible depth.

    Parameters
    ----------
    depth_results : np.ndarray
        Finite numeric array of shape (m,3), with 1 <= m <= 4.
        Each row is [depth, threshold_met, certificate].
        Depths must be consecutive integers starting at 1.
        threshold_met must be exactly 0 or 1.
        certificate must lie in [0,1].
    epsilon : float
        Approximation tolerance. Must be finite and satisfy
        0 <= epsilon < 1.

    Returns
    -------
    result : np.ndarray
        One-dimensional array
        [minimum_feasible_depth,
         rejected_optimum_depth_1,
         rejected_optimum_depth_2,
         ...].

    Raises
    ------
    ValueError
        If depth_results is malformed or non-finite; if it contains
        fewer than one or more than four rows; if depth values are not
        consecutive integers starting at 1; if any threshold_met value
        is not exactly 0 or 1; if any certificate lies outside [0,1];
        if a rejected or feasible certificate is inconsistent with
        1-epsilon; if epsilon is non-finite or does not satisfy
        0 <= epsilon < 1; or if no feasible depth is present.
    """
    return np.empty(0, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_select_first_feasible_depth(
    depth_results: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    values = np.asarray(
        depth_results,
        dtype=float,
    )

    if (
        values.ndim != 2
        or values.shape[1] != 3
        or values.shape[0] < 1
        or values.shape[0] > 4
    ):
        raise ValueError(
            "depth_results must have shape (m,3) with 1 <= m <= 4"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "depth_results must be finite"
        )

    try:
        epsilon = float(epsilon)
    except Exception as exc:
        raise ValueError(
            "epsilon must be finite and satisfy 0 <= epsilon < 1"
        ) from exc

    if (
        not np.isfinite(epsilon)
        or epsilon < 0.0
        or epsilon >= 1.0
    ):
        raise ValueError(
            "epsilon must be finite and satisfy 0 <= epsilon < 1"
        )

    threshold = 1.0 - epsilon

    # Validate the complete table before selecting a feasible row.
    for index, row in enumerate(values):
        expected_depth = float(
            index + 1
        )

        depth = float(row[0])
        flag = float(row[1])
        certificate = float(row[2])

        if depth != expected_depth:
            raise ValueError(
                "depth rows must be consecutive integers starting at 1"
            )

        # Review-3 repair:
        # do this BEFORE the feasible branch.
        if flag not in (0.0, 1.0):
            raise ValueError(
                "threshold_met must be exactly 0 or 1"
            )

        if (
            certificate < 0.0
            or certificate > 1.0
        ):
            raise ValueError(
                "certificate must lie in [0,1]"
            )

        if (
            flag == 0.0
            and certificate
            >= threshold - 1e-12
        ):
            raise ValueError(
                "a rejected depth must have global optimum below threshold"
            )

        if (
            flag == 1.0
            and not np.isclose(
                certificate,
                threshold,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "a feasible public certificate must equal the requested threshold"
            )

    rejected = []

    for row in values:
        depth = int(row[0])
        flag = int(row[1])
        certificate = float(row[2])

        if flag == 1:
            return np.asarray(
                [
                    float(depth)
                ]
                + rejected,
                dtype=float,
            )

        rejected.append(
            certificate
        )

    raise ValueError(
        "no feasible depth is present"
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np

depth_results = np.array([
    [1.0, 0.0, 0.19329500596619154],
    [2.0, 0.0, 0.7063610711381948],
    [3.0, 0.0, 0.7063610711381948],
    [4.0, 1.0, 0.90],
])

epsilon = 0.10
""",
            "call": "select_first_feasible_depth(depth_results, epsilon)",
            "gold_call": "_oracle_select_first_feasible_depth(depth_results, epsilon)",
        },
        {
            "setup": """import numpy as np

depth_results = np.array([
    [1.0, 0.0, 0.19329500596619154],
    [2.0, 1.0, 0.70],
])

epsilon = 0.30
""",
            "call": "select_first_feasible_depth(depth_results, epsilon)",
            "gold_call": "_oracle_select_first_feasible_depth(depth_results, epsilon)",
        },
        {
            "setup": """import numpy as np

# Malformed depth sequence: depth 2 is missing.
depth_results = np.array([
    [1.0, 0.0, 0.1],
    [3.0, 1.0, 0.9],
])

epsilon = 0.10

def run_model():
    try:
        select_first_feasible_depth(
            depth_results,
            epsilon,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_select_first_feasible_depth(
            depth_results,
            epsilon,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

# threshold_met=2 is malformed; only 0 or 1 is allowed.
depth_results = np.array([
    [1.0, 2.0, 0.90],
])

epsilon = 0.10

def run_model():
    try:
        select_first_feasible_depth(
            depth_results,
            epsilon,
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_select_first_feasible_depth(
            depth_results,
            epsilon,
        )
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
