"""
Return the sequence of states that the source's forecasting algorithm predicts in one direction from a neighbourhood: the sweep moves along the neighbouring trajectories, one state per step, backward (direction -1) or forward (direction +1), and forms each predicted state as the weighted mean of the state variables of the states reached, following the source's rule for which states enter each step and its stopping rule governed by min_pts. Rows are ordered from the first predicted state outward.

The algorithm predicts the states that precede and follow a target by averaging what happened before and after the target's neighbours in their own trajectories, step by step, until too few neighbouring trajectories extend far enough to support a prediction.

Returns
-------
numpy.ndarray of float64 with shape (m, S): the predicted states of one sweep.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sweep_states(states: "numpy.ndarray", trajectory_ids: "numpy.ndarray", state_ids: "numpy.ndarray",
                 neighbours: "numpy.ndarray", weights: "numpy.ndarray", min_pts: int,
                 direction: int) -> "numpy.ndarray":
    """Return the sequence of states that the source's forecasting algorithm predicts in one direction from a neighbourhood: the sweep moves along the neighbouring trajectories, one state per step, backward (direction -1) or forward (direction +1), and forms each predicted state as the weighted mean of the state variables of the states reached, following the source's rule for which states enter each step and its stopping rule governed by min_pts. Rows are ordered from the first predicted state outward.

    Parameters
    ----------
    states : numpy.ndarray
        Array of shape (n, S) of the state variables of all states.
    trajectory_ids : numpy.ndarray
        One-dimensional array of length n: the trajectory of every state.
    state_ids : numpy.ndarray
        One-dimensional array of length n of integers giving the temporal order of the states within their trajectory.
    neighbours : numpy.ndarray
        One-dimensional integer array of the kept neighbouring state indices (step 06).
    weights : numpy.ndarray
        One-dimensional array of length n of nonnegative per-state trajectory weights (step 05).
    min_pts : int
        Minimum number of states required to form a predicted state, at least 1.
    direction : int
        -1 for a backward sweep, +1 for a forward sweep.

    Returns
    -------
    predicted : numpy.ndarray
        Array of shape (m, S) of the m predicted states in sweep order (m may be 0; float64).

    Raises
    ------
    ValueError
        If the arrays do not describe the same n states, states or weights are not finite, a weight is negative, the states averaged at some step carry no weight, neighbours holds an invalid index, min_pts is not a positive integer, or direction is not -1 or +1.
    """
    return predicted

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sweep_states(states: "numpy.ndarray", trajectory_ids: "numpy.ndarray", state_ids: "numpy.ndarray",
                         neighbours: "numpy.ndarray", weights: "numpy.ndarray", min_pts: int,
                         direction: int) -> "numpy.ndarray":
    """PETRA-EDR sweep in one direction: the i-th predicted state (i = 1, 2, ...) is the weighted mean of the states lying
    i positions before (direction -1) or after (direction +1) each kept neighbour within its own trajectory, formed while at
    least min_pts such states exist; rows ordered by increasing i."""
    X = np.asarray(states, dtype=np.float64); tr = np.asarray(trajectory_ids); st = np.asarray(state_ids)
    nb = np.asarray(neighbours, dtype=int); w = np.asarray(weights, dtype=np.float64)
    n = tr.size
    if X.ndim != 2 or X.shape[0] != n or st.shape != (n,) or w.shape != (n,):
        raise ValueError("states, trajectory_ids, state_ids and weights must describe the same n states")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(w))) or np.any(w < 0.0):
        raise ValueError("states must be finite and weights finite and nonnegative")
    if nb.ndim != 1 or (nb.size and (nb.min() < 0 or nb.max() >= n)):
        raise ValueError("neighbours must be a 1-D array of valid state indices")
    if int(min_pts) != min_pts or min_pts < 1:
        raise ValueError("min_pts must be a positive integer")
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 (backward) or +1 (forward)")
    # position of every state within its own trajectory (states sorted by state_ids)
    pos_of = {}
    for lab in dict.fromkeys(tr.tolist()):
        idx = np.where(tr == lab)[0]
        idx = idx[np.argsort(st[idx], kind="stable")]
        pos_of[lab] = idx
    out = []
    i = 1
    while True:
        shifted = []
        for j in nb:
            idx = pos_of[tr[j]]
            p = int(np.where(idx == j)[0][0]) + int(direction) * i
            if 0 <= p < idx.size:
                shifted.append(int(idx[p]))
        if len(shifted) < int(min_pts):
            break
        ww = w[shifted]
        if ww.sum() <= 0.0:
            raise ValueError("the contributing states carry no weight")
        out.append((X[shifted] * ww[:, None]).sum(axis=0) / ww.sum())
        i += 1
    return np.array(out).reshape(-1, X.shape[1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius, min_pts, alpha = 25, 0.065, 3, 3.0\nweights = _oracle_trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)\nneighbours = _oracle_neighbour_states(dissimilarity, trajectory_ids, first_index, k, radius, weights)\ndirection = -1\n",
            "call": "sweep_states(states, trajectory_ids, state_ids, neighbours, weights, min_pts, direction)",
            "gold_call": "_oracle_sweep_states(states, trajectory_ids, state_ids, neighbours, weights, min_pts, direction)",
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius, min_pts, alpha = 25, 0.065, 3, 3.0\nweights = _oracle_trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)\nneighbours = _oracle_neighbour_states(dissimilarity, trajectory_ids, last_index, k, radius, weights)\ndirection = 1\n",
            "call": "sweep_states(states, trajectory_ids, state_ids, neighbours, weights, min_pts, direction)",
            "gold_call": "_oracle_sweep_states(states, trajectory_ids, state_ids, neighbours, weights, min_pts, direction)",
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius, min_pts, alpha = 25, 0.065, 3, 3.0\nweights = _oracle_trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)\nneighbours = _oracle_neighbour_states(dissimilarity, trajectory_ids, first_index, k, radius, weights)\nmin_pts = 24\ndirection = -1\n",
            "call": "sweep_states(states, trajectory_ids, state_ids, neighbours, weights, min_pts, direction)",
            "gold_call": "_oracle_sweep_states(states, trajectory_ids, state_ids, neighbours, weights, min_pts, direction)",
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 12, 6, 0.2\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (3, 9)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(2, 5 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius, min_pts, alpha = 8, 0.12, 2, 1.0\nweights = _oracle_trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)\nneighbours = _oracle_neighbour_states(dissimilarity, trajectory_ids, last_index, k, radius, weights)\ndirection = 1\n",
            "call": "sweep_states(states, trajectory_ids, state_ids, neighbours, weights, min_pts, direction)",
            "gold_call": "_oracle_sweep_states(states, trajectory_ids, state_ids, neighbours, weights, min_pts, direction)",
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius, min_pts, alpha = 25, 0.065, 3, 3.0\nweights = _oracle_trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)\nneighbours = _oracle_neighbour_states(dissimilarity, trajectory_ids, first_index, k, radius, weights)\ndirection = 0\ndef run_model():\n    try:\n        sweep_states(states, trajectory_ids, state_ids, neighbours, weights, min_pts, direction)\n        return 0\n    except ValueError:\n        return 1\ndef _fx_safe(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "_fx_safe(lambda: _oracle_sweep_states(states, trajectory_ids, state_ids, neighbours, weights, min_pts, direction))",
        },
    ]
