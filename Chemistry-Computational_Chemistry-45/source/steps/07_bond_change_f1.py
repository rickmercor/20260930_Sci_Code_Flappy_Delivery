"""
Score reported bond changes against the expected reactive-bond set using F1.

Set-based F1 balances missed reactive bonds against unrelated reported bond changes without counting dependent angular coordinates.

Returns
-------
Float F1 score in [0, 1], with two empty sets scoring 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def bond_change_f1(
    statuses: np.ndarray,
    coordinates: np.ndarray,
    expected_bonds: np.ndarray,
) -> float:
    """Compute F1 for primary and coupled-proton bond reports.

    Parameters
    ----------
    statuses : np.ndarray
        Integer status code for every coordinate.
    coordinates : np.ndarray
        Encoded coordinate rows aligned with statuses.
    expected_bonds : np.ndarray
        Expected undirected reactive bonds.

    Returns
    -------
    float
        Set-based bond F1 score.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_bond_change_f1(
    statuses: np.ndarray,
    coordinates: np.ndarray,
    expected_bonds: np.ndarray,
) -> float:
    """Reference set-based reactive-bond F1 score."""
    import numpy as np

    state = np.asarray(statuses)
    coords = np.asarray(coordinates)
    expected = np.asarray(expected_bonds)

    if (
        state.ndim != 1
        or state.size < 1
        or coords.shape != (state.size, 5)
    ):
        raise ValueError("statuses and coordinates must align")
    if (
        not np.all(np.isfinite(state))
        or not np.all(state == np.floor(state))
    ):
        raise ValueError("statuses must be finite integers")

    state = state.astype(int)
    if not np.all(np.isin(state, [0, 1, 2, 3, 4])):
        raise ValueError(
            "status codes must be between 0 and 4"
        )

    if (
        not np.all(np.isfinite(coords))
        or not np.all(coords == np.floor(coords))
    ):
        raise ValueError(
            "coordinate encoding must contain finite integers"
        )
    coords = coords.astype(int)

    if expected.ndim != 2 or expected.shape[1] != 2:
        raise ValueError(
            "expected_bonds must have shape (n_expected, 2)"
        )
    if (
        not np.all(np.isfinite(expected))
        or not np.all(expected == np.floor(expected))
    ):
        raise ValueError(
            "expected bond indices must be finite integers"
        )

    expected = expected.astype(int)
    if (
        np.any(expected < 0)
        or np.any(expected[:, 0] == expected[:, 1])
    ):
        raise ValueError(
            "expected bonds must contain distinct nonnegative indices"
        )

    detected = {
        tuple(sorted(map(int, coords[index, 1:3])))
        for index in np.flatnonzero(
            (state == 1) | (state == 2)
        )
        if coords[index, 0] == 2
    }

    target = {
        tuple(sorted(map(int, row)))
        for row in expected
    }

    if not detected and not target:
        return 1.0

    true_positive = len(detected.intersection(target))
    false_positive = len(detected - target)
    false_negative = len(target - detected)

    denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    return (
        0.0
        if denominator == 0
        else float(2 * true_positive / denominator)
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
statuses = np.array([1,2,3])
coordinates = np.array([[2,0,1,-1,-1],[2,1,2,-1,-1],[3,0,1,2,-1]])
expected_bonds = np.array([[1,0],[2,1]])""",
            "call": "bond_change_f1(statuses, coordinates, expected_bonds)",
            "gold_call": "_oracle_bond_change_f1(statuses, coordinates, expected_bonds)",
        },
        {
            "setup": """import numpy as np
statuses = np.array([1,2,0])
coordinates = np.array([[2,0,1,-1,-1],[2,1,2,-1,-1],[2,2,3,-1,-1]])
expected_bonds = np.array([[0,1],[2,3]])""",
            "call": "bond_change_f1(statuses, coordinates, expected_bonds)",
            "gold_call": "_oracle_bond_change_f1(statuses, coordinates, expected_bonds)",
        },
        {
            "setup": """import numpy as np
statuses = np.array([0])
coordinates = np.array([[3,0,1,2,-1]])
expected_bonds = np.empty((0,2), dtype=int)""",
            "call": "bond_change_f1(statuses, coordinates, expected_bonds)",
            "gold_call": "_oracle_bond_change_f1(statuses, coordinates, expected_bonds)",
        },
    ]
