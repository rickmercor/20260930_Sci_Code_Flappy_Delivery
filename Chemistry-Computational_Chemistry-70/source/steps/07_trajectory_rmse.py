"""
Implement a function that computes the whole-trajectory position root-mean-

square error between a predicted trajectory and the exact reference

trajectory, using the minimum-image displacement convention and per-atom,

per-frame normalization. This single metric function is applied to both

approximate solvers in this problem: the RBL trajectory of Step 5 and the

DINaMo trajectory of Step 6, each compared against the exact reference

trajectory of Step 3.

Because the simulation box is periodic, a naive coordinate-wise difference

between two trajectories can be dominated by an atom that has wrapped across

a box boundary even though its physical displacement is small; the

minimum-image convention removes this artifact by measuring each atom's

displacement along the shortest path through the periodic images. The

resulting whole-trajectory RMSE,




    RMSE = sqrt( (1 / (3*N*B)) * sum_b sum_i || Delta r_i(t_b) ||_2^2 ),




aggregates this minimum-image displacement over every atom and every stored

frame into the single scalar this problem ultimately compares between the

two independent solvers.

Returns
-------
float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def trajectory_rmse(predicted_trajectory: np.ndarray, reference_trajectory: np.ndarray,
                     box_length: float) -> float:
    """
    Parameters
    ----------
    predicted_trajectory : np.ndarray
        Array of shape (B, N, 3) with B >= 1 and N >= 2, the predicted
        particle positions at each stored frame.
    reference_trajectory : np.ndarray
        Array of the same shape as `predicted_trajectory`, the reference
        particle positions at each stored frame.
    box_length : float
        Side length of the cubic periodic box.

    Returns
    -------
    rmse : float
        The whole-trajectory position RMSE, as a native Python float.

    Raises
    ------
    ValueError
        If `predicted_trajectory` does not have shape (B, N, 3) with
        B >= 1 and N >= 2, or contains non-finite values; if
        `reference_trajectory` does not match `predicted_trajectory`'s
        shape or contains non-finite values; or if `box_length` is not
        finite and > 0.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_trajectory_rmse(predicted_trajectory: np.ndarray, reference_trajectory: np.ndarray,
                             box_length: float) -> float:
    """Reference implementation."""
    import numpy as np
    predicted_trajectory = np.asarray(predicted_trajectory, dtype=float)
    reference_trajectory = np.asarray(reference_trajectory, dtype=float)

    if (predicted_trajectory.ndim != 3 or predicted_trajectory.shape[2] != 3
            or predicted_trajectory.shape[0] < 1 or predicted_trajectory.shape[1] < 2):
        raise ValueError("predicted_trajectory must have shape (B, N, 3) with B >= 1 and N >= 2.")
    if not np.all(np.isfinite(predicted_trajectory)):
        raise ValueError("predicted_trajectory must contain only finite values.")
    if reference_trajectory.shape != predicted_trajectory.shape:
        raise ValueError("reference_trajectory must have the same shape as predicted_trajectory.")
    if not np.all(np.isfinite(reference_trajectory)):
        raise ValueError("reference_trajectory must contain only finite values.")
    if not np.isfinite(box_length) or box_length <= 0:
        raise ValueError("box_length must be finite and > 0.")

    b, n, _ = predicted_trajectory.shape
    diff = predicted_trajectory - reference_trajectory
    diff = diff - box_length * np.round(diff / box_length)
    sq_sum = np.sum(diff ** 2)
    rmse = np.sqrt(sq_sum / (3 * n * b))
    return float(rmse)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # Normal scenario: 4 frames, 3 atoms, small differences between
            # predicted and reference.
            "setup": (
                "import numpy as np\n"
                "reference_trajectory = np.array([\n"
                "    [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],\n"
                "    [[1.1, 1.0, 1.0], [2.0, 2.1, 2.0], [3.0, 3.0, 2.9]],\n"
                "    [[1.2, 1.0, 1.0], [2.0, 2.2, 2.0], [3.0, 3.0, 2.8]],\n"
                "    [[1.3, 1.0, 1.0], [2.0, 2.3, 2.0], [3.0, 3.0, 2.7]],\n"
                "])\n"
                "predicted_trajectory = reference_trajectory + 0.01\n"
                "box_length = 10.0\n"
            ),
            "call": "trajectory_rmse(predicted_trajectory, reference_trajectory, box_length)",
            "gold_call": "_oracle_trajectory_rmse(predicted_trajectory, reference_trajectory, box_length)",
        },
        {
            # Boundary case: predicted equals reference exactly, so the
            # RMSE must be exactly 0.0.
            "setup": (
                "import numpy as np\n"
                "reference_trajectory = np.array([\n"
                "    [[0.5, 0.5, 0.5], [4.0, 4.0, 4.0]],\n"
                "    [[0.6, 0.5, 0.5], [4.1, 4.0, 4.0]],\n"
                "])\n"
                "predicted_trajectory = reference_trajectory.copy()\n"
                "box_length = 8.0\n"
            ),
            "call": "trajectory_rmse(predicted_trajectory, reference_trajectory, box_length)",
            "gold_call": "_oracle_trajectory_rmse(predicted_trajectory, reference_trajectory, box_length)",
        },
        {
            # Edge case: a single frame in which one atom's raw coordinate
            # difference spans nearly the whole box (a wrap-around), so the
            # minimum-image convention must reduce it to a small effective
            # displacement rather than counting it as almost `box_length`.
            "setup": (
                "import numpy as np\n"
                "box_length = 10.0\n"
                "reference_trajectory = np.array([\n"
                "    [[0.2, 5.0, 5.0], [5.0, 5.0, 5.0]],\n"
                "])\n"
                "predicted_trajectory = np.array([\n"
                "    [[9.9, 5.0, 5.0], [5.0, 5.0, 5.0]],\n"
                "])\n"
            ),
            "call": "trajectory_rmse(predicted_trajectory, reference_trajectory, box_length)",
            "gold_call": "_oracle_trajectory_rmse(predicted_trajectory, reference_trajectory, box_length)",
        },
    ]
