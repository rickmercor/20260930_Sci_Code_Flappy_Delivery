"""
Fit one symmetric harmonic precision matrix to the projected force targets using shared upper-triangle parameters.

For a harmonic coarse-grained potential, the conservative model force is -H R with symmetric H. Each off-diagonal element contributes to two coupled force equations, so fitting separate output rows would violate the conservative constraint. The unique parameters are ordered rowwise over the upper triangle and fitted with nonnegative ridge regularization.

Returns
-------
Return the symmetric fitted precision matrix with shape (beads, beads).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def fit_conservative_precision(
    noised_positions: np.ndarray,
    target_forces: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Fit a symmetric harmonic CG precision from force targets.

    The model force is -H R with H symmetric. Unique parameters are ordered
    rowwise over the upper triangle, including the diagonal.

    Parameters
    ----------
    noised_positions
        Array with shape (replicates, frames, beads, components).
    target_forces
        Force targets with the same shape.
    ridge
        Nonnegative regularization on the unique entries of H.

    Returns
    -------
    np.ndarray
        Symmetric fitted precision with shape (beads, beads).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_fit_conservative_precision(noised_positions, target_forces, ridge):
    import numpy as np

    positions = np.asarray(noised_positions, dtype=float)
    targets = np.asarray(target_forces, dtype=float)
    if positions.ndim != 4 or min(positions.shape) == 0:
        raise ValueError("noised_positions must be a nonempty 4D array")
    if targets.shape != positions.shape:
        raise ValueError("target_forces must match noised_positions")
    if np.any(~np.isfinite(positions)) or np.any(~np.isfinite(targets)):
        raise ValueError("positions and targets must be finite")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")

    n_beads = positions.shape[2]
    coordinates = np.transpose(positions, (0, 1, 3, 2)).reshape(
        -1, n_beads
    )
    responses = np.transpose(targets, (0, 1, 3, 2)).reshape(
        -1, n_beads
    )
    pairs = [
        (row, column)
        for row in range(n_beads)
        for column in range(row, n_beads)
    ]
    design = np.zeros((len(coordinates) * n_beads, len(pairs)))
    for index, (row, column) in enumerate(pairs):
        if row == column:
            design[row::n_beads, index] = -coordinates[:, row]
        else:
            design[row::n_beads, index] = -coordinates[:, column]
            design[column::n_beads, index] = -coordinates[:, row]

    count = len(coordinates)
    normal = design.T @ design / count + ridge * np.eye(len(pairs))
    right_hand_side = design.T @ responses.reshape(-1) / count
    try:
        parameters = np.linalg.solve(normal, right_hand_side)
    except np.linalg.LinAlgError as exc:
        raise ValueError("precision fit is singular") from exc

    precision = np.zeros((n_beads, n_beads))
    for value, (row, column) in zip(parameters, pairs):
        precision[row, column] = value
        precision[column, row] = value
    return precision

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nrng=np.random.default_rng(2); R=rng.normal(size=(3,6,2,3)); "
            "H=np.array([[2.,-.4],[-.4,1.5]]); "
            "Y=-np.einsum('ij,sfjc->sfic',H,R)",
            "call": "fit_conservative_precision(R,Y,1e-8)",
            "gold_call": "_oracle_fit_conservative_precision(R,Y,1e-8)",
        },
        {
            "setup": "import numpy as np\nR=np.linspace(-1,1,12).reshape(2,3,1,2); Y=-2.5*R",
            "call": "fit_conservative_precision(R,Y,.01)",
            "gold_call": "_oracle_fit_conservative_precision(R,Y,.01)",
        },
        {
            "setup": "import numpy as np\nrng=np.random.default_rng(12); R=rng.normal(size=(2,5,3,1)); "
            "H=np.array([[3.,.2,-.1],[.2,2.,.4],[-.1,.4,1.7]]); "
            "Y=-np.einsum('ij,sfjc->sfic',H,R)+.05*rng.normal(size=R.shape)",
            "call": "fit_conservative_precision(R,Y,.03)",
            "gold_call": "_oracle_fit_conservative_precision(R,Y,.03)",
        },
        {
            "setup": "import numpy as np\nrng=np.random.default_rng(44); "
            "R=rng.normal(size=(2,7,4,2)); "
            "H=np.array([[3.,-.4,.15,0.],[-.4,2.6,-.3,.1],"
            "[.15,-.3,2.2,-.25],[0.,.1,-.25,1.8]]); "
            "Y=-np.einsum('ij,sfjc->sfic',H,R)",
            "call": "fit_conservative_precision(R,Y,0.0)",
            "gold_call": "_oracle_fit_conservative_precision(R,Y,0.0)",
        },
    ]
