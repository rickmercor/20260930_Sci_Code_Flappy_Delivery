"""
Return the indices of the reference states that the source's forecasting algorithm keeps as the neighbourhood of one extreme state of the target: the neighbourhood is built from the k nearest states of other trajectories in the dissimilarity matrix, the radius (the source's dissimilarity threshold) and, when weights are given, the trajectory weights, following the source's order of operations and its treatment of states that fail the radius or carry no weight. Indices are returned in ascending dissimilarity to the extreme state, ties in index order.

The forecast of a target's trajectory is driven by the reference states that resemble its extremes: the algorithm looks for the nearest states of each extreme in the trajectories of the regime and limits the neighbourhood to a dissimilarity radius so that distant states do not bias the averaging.

Returns
-------
numpy.ndarray of int with shape (m,): indices of the kept neighbouring states.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def neighbour_states(dissimilarity: "numpy.ndarray", trajectory_ids: "numpy.ndarray",
                     target_index: int, k: int, radius: float,
                     weights: "numpy.ndarray" = None) -> "numpy.ndarray":
    """Return the indices of the reference states that the source's forecasting algorithm keeps as the neighbourhood of one extreme state of the target: the neighbourhood is built from the k nearest states of other trajectories in the dissimilarity matrix, the radius (the source's dissimilarity threshold) and, when weights are given, the trajectory weights, following the source's order of operations and its treatment of states that fail the radius or carry no weight. Indices are returned in ascending dissimilarity to the extreme state, ties in index order.

    Parameters
    ----------
    dissimilarity : numpy.ndarray
        Array of shape (n, n) of state dissimilarities (step 03).
    trajectory_ids : numpy.ndarray
        One-dimensional array of length n: the trajectory of every state.
    target_index : int
        Row index of the extreme state of the target whose neighbourhood is built.
    k : int
        Number of nearest states considered, at least 1.
    radius : float
        Positive dissimilarity threshold of the neighbourhood.
    weights : numpy.ndarray
        Optional one-dimensional array of length n of nonnegative per-state trajectory weights (step 05); None when no weights are used.

    Returns
    -------
    neighbours : numpy.ndarray
        One-dimensional integer array of the kept state indices, ascending dissimilarity to the extreme state.

    Raises
    ------
    ValueError
        If dissimilarity is not a finite square matrix matching trajectory_ids, target_index is out of range, k is not a positive integer, radius is not positive and finite, or weights is not a finite nonnegative vector of length n.
    """
    return neighbours

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_neighbour_states(dissimilarity: "numpy.ndarray", trajectory_ids: "numpy.ndarray",
                             target_index: int, k: int, radius: float,
                             weights: "numpy.ndarray" = None) -> "numpy.ndarray":
    """Indices of the reference states that PETRA-EDR keeps around one target state: the k nearest states that do not
    belong to the target's trajectory, restricted afterwards (no replacement) to dissimilarities below radius and, when
    weights are given, to states of positively weighted trajectories; ascending dissimilarity, ties by index."""
    D = np.asarray(dissimilarity, dtype=np.float64); tr = np.asarray(trajectory_ids)
    n = tr.size
    if D.shape != (n, n) or n < 1:
        raise ValueError("dissimilarity must be a square matrix matching trajectory_ids")
    if not np.all(np.isfinite(D)):
        raise ValueError("dissimilarity must be finite")
    if int(target_index) != target_index or not (0 <= target_index < n):
        raise ValueError("target_index must index a row of the matrix")
    if int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("weights must be a finite nonnegative vector matching trajectory_ids")
    t = int(target_index)
    candidates = np.array([i for i in np.argsort(D[:, t], kind="stable") if tr[i] != tr[t]], dtype=int)
    kept = candidates[: int(k)]                                              # k nearest ...
    kept = kept[D[kept, t] < float(radius)]                                  # ... then the radius, strict, no replacement
    if weights is not None:
        kept = kept[w[kept] > 0.0]                                           # ... then positive trajectory weight
    return kept.astype(int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius = 25, 0.065\nweights = None\ntarget_index = first_index\n",
            "call": "neighbour_states(dissimilarity, trajectory_ids, target_index, k, radius, weights)",
            "gold_call": "_oracle_neighbour_states(dissimilarity, trajectory_ids, target_index, k, radius, weights)",
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius, min_pts, alpha = 25, 0.065, 3, 3.0\nweights = _oracle_trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)\ntarget_index = last_index\n",
            "call": "neighbour_states(dissimilarity, trajectory_ids, target_index, k, radius, weights)",
            "gold_call": "_oracle_neighbour_states(dissimilarity, trajectory_ids, target_index, k, radius, weights)",
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius, min_pts, alpha = 25, 0.065, 3, 3.0\nweights = _oracle_trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha).copy()\nweights[trajectory_ids == 29] = 0.0\ntarget_index = first_index\n",
            "call": "neighbour_states(dissimilarity, trajectory_ids, target_index, k, radius, weights)",
            "gold_call": "_oracle_neighbour_states(dissimilarity, trajectory_ids, target_index, k, radius, weights)",
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius, min_pts, alpha = 25, 0.065, 3, 3.0\nweights = _oracle_trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)\nradius = 0.055\ntarget_index = first_index\n",
            "call": "neighbour_states(dissimilarity, trajectory_ids, target_index, k, radius, weights)",
            "gold_call": "_oracle_neighbour_states(dissimilarity, trajectory_ids, target_index, k, radius, weights)",
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 12, 6, 0.2\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (3, 9)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(2, 5 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius = 8, 0.12\nweights = None\ntarget_index = last_index\n",
            "call": "neighbour_states(dissimilarity, trajectory_ids, target_index, k, radius, weights)",
            "gold_call": "_oracle_neighbour_states(dissimilarity, trajectory_ids, target_index, k, radius, weights)",
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius = 25, -0.1\nweights = None\ntarget_index = first_index\ndef run_model():\n    try:\n        neighbour_states(dissimilarity, trajectory_ids, target_index, k, radius, weights)\n        return 0\n    except ValueError:\n        return 1\ndef _fx_safe(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "_fx_safe(lambda: _oracle_neighbour_states(dissimilarity, trajectory_ids, target_index, k, radius, weights))",
        },
    ]
