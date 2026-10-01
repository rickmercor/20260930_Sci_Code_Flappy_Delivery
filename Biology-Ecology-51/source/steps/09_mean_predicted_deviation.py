"""
Return the source's mean predicted deviation (its Eq 3) of the predicted trajectory of a multi-state target: every predicted state is in turn forecast as a new single-state target with the source's algorithm (the same k, radius and min_pts, through steps 03, 06, 07 and 08), with the regime trajectories (the original target removed) as the reference, the predicted state forming a one-state trajectory of its own with weight 0, and every regime state keeping the original target's trajectory weight; the deviation is the average, over every pair of an observed target state and a predicted state, of the smallest Bray-Curtis dissimilarity between the observed state and the states of the trajectory forecast from that predicted state (its step 08 assembly, the predicted state itself included).

If a target's trajectory is well determined by the regime, trajectories forecast from the states predicted for it should pass through the target's observed states; the mean predicted deviation turns that consistency requirement into an accuracy metric that needs no held-out data.

Returns
-------
float, the mean predicted deviation of the predicted trajectory.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mean_predicted_deviation(states: "numpy.ndarray", trajectory_ids: "numpy.ndarray",
                             state_ids: "numpy.ndarray", target_label: int, k: int, radius: float,
                             min_pts: int, weights: "numpy.ndarray",
                             predicted: "numpy.ndarray") -> float:
    """Return the source's mean predicted deviation (its Eq 3) of the predicted trajectory of a multi-state target: every predicted state is in turn forecast as a new single-state target with the source's algorithm (the same k, radius and min_pts, through steps 03, 06, 07 and 08), with the regime trajectories (the original target removed) as the reference, the predicted state forming a one-state trajectory of its own with weight 0, and every regime state keeping the original target's trajectory weight; the deviation is the average, over every pair of an observed target state and a predicted state, of the smallest Bray-Curtis dissimilarity between the observed state and the states of the trajectory forecast from that predicted state (its step 08 assembly, the predicted state itself included).

    Parameters
    ----------
    states : numpy.ndarray
        Array of shape (n, S) of the state variables of all states, the target's included.
    trajectory_ids : numpy.ndarray
        One-dimensional array of length n: the trajectory of every state.
    state_ids : numpy.ndarray
        One-dimensional array of length n of integers giving the temporal order within each trajectory.
    target_label : int
        Trajectory label of the target.
    k : int
        Number of nearest states of the neighbourhood, at least 1.
    radius : float
        Positive dissimilarity threshold of the neighbourhood.
    min_pts : int
        Minimum number of states required to form a predicted state.
    weights : numpy.ndarray
        One-dimensional array of length n of the per-state trajectory weights of the original target (step 05).
    predicted : numpy.ndarray
        Array of shape (m + n_X, S + 1) from step 08: labels in column 0, state variables after.

    Returns
    -------
    deviation : float
        The mean predicted deviation, in the units of the dissimilarity, as a native Python float.

    Raises
    ------
    ValueError
        If the arrays do not describe the same n states, target_label names no state, predicted does not have S + 1 columns or contains no predicted state besides the target's own, or the inputs are invalid for steps 03, 06, 07 or 08.
    """
    return deviation

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_mean_predicted_deviation(states: "numpy.ndarray", trajectory_ids: "numpy.ndarray",
                                     state_ids: "numpy.ndarray", target_label: int, k: int, radius: float,
                                     min_pts: int, weights: "numpy.ndarray",
                                     predicted: "numpy.ndarray") -> float:
    """Mean predicted deviation (source Eq 3): every predicted state becomes a single-state target in the reference set
    formed by the regime trajectories without the original target, is forecast in both directions with the same k, radius,
    min_pts and the original target's trajectory weights (steps 03, 06, 07, 08), and MPD is the mean over the observed
    target states and the predicted states of the minimum Bray-Curtis dissimilarity between the observed state and the
    states of that forecast."""
    X = np.asarray(states, dtype=np.float64); tr = np.asarray(trajectory_ids); st = np.asarray(state_ids)
    w = np.asarray(weights, dtype=np.float64); P = np.asarray(predicted, dtype=np.float64)
    n = tr.size
    if X.ndim != 2 or X.shape[0] != n or st.shape != (n,) or w.shape != (n,):
        raise ValueError("states, trajectory_ids, state_ids and weights must describe the same n states")
    tid = np.where(tr == target_label)[0]
    if tid.size < 1:
        raise ValueError("target_label must name at least one state")
    if P.ndim != 2 or P.shape[1] != X.shape[1] + 1 or P.shape[0] < 1:
        raise ValueError("predicted must have one label column plus the state variables")
    own = set(st[tid].astype(float).tolist())
    pred_rows = P[[float(l) not in own for l in P[:, 0]]]
    if pred_rows.shape[0] == 0:
        raise ValueError("the predicted trajectory holds no predicted state")
    keep = np.where(tr != target_label)[0]                                # regime without the original target
    devs = []
    for row in pred_rows:
        label, y = float(row[0]), row[1:]
        Xr = np.vstack([X[keep], y[None, :]])
        trr = np.concatenate([tr[keep].astype(object), np.array(["__predicted__"], dtype=object)])
        str_ = np.concatenate([st[keep], [label]])
        wr = np.concatenate([w[keep], [0.0]])
        ti = Xr.shape[0] - 1
        Dr = _oracle_bray_curtis_matrix(Xr)
        nb = _oracle_neighbour_states(Dr, trr, ti, k, radius, wr)
        back = _oracle_sweep_states(Xr, trr, str_, nb, wr, min_pts, -1)
        forw = _oracle_sweep_states(Xr, trr, str_, nb, wr, min_pts, 1)
        Z = _oracle_predicted_trajectory(y, np.array([label]), back, forw)[:, 1:]
        for i in tid:
            num = np.abs(Z - X[i][None, :]).sum(axis=1); den = (Z + X[i][None, :]).sum(axis=1)
            devs.append(float(np.min(num / den)))
    return float(np.mean(devs))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius, min_pts, alpha = 25, 0.065, 3, 3.0\nweights = _oracle_trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)\nneighbours_back = _oracle_neighbour_states(dissimilarity, trajectory_ids, first_index, k, radius, weights)\nneighbours_forw = _oracle_neighbour_states(dissimilarity, trajectory_ids, last_index, k, radius, weights)\nbackward = _oracle_sweep_states(states, trajectory_ids, state_ids, neighbours_back, weights, min_pts, -1)\nforward = _oracle_sweep_states(states, trajectory_ids, state_ids, neighbours_forw, weights, min_pts, 1)\npredicted = _oracle_predicted_trajectory(target_states, target_labels, backward, forward)\n",
            "call": "mean_predicted_deviation(states, trajectory_ids, state_ids, target_label, k, radius, min_pts, weights, predicted)",
            "gold_call": "_oracle_mean_predicted_deviation(states, trajectory_ids, state_ids, target_label, k, radius, min_pts, weights, predicted)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius, min_pts, alpha = 15, 0.08, 4, 2.0\nweights = _oracle_trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)\nneighbours_back = _oracle_neighbour_states(dissimilarity, trajectory_ids, first_index, k, radius, weights)\nneighbours_forw = _oracle_neighbour_states(dissimilarity, trajectory_ids, last_index, k, radius, weights)\nbackward = _oracle_sweep_states(states, trajectory_ids, state_ids, neighbours_back, weights, min_pts, -1)\nforward = _oracle_sweep_states(states, trajectory_ids, state_ids, neighbours_forw, weights, min_pts, 1)\npredicted = _oracle_predicted_trajectory(target_states, target_labels, backward, forward)\n",
            "call": "mean_predicted_deviation(states, trajectory_ids, state_ids, target_label, k, radius, min_pts, weights, predicted)",
            "gold_call": "_oracle_mean_predicted_deviation(states, trajectory_ids, state_ids, target_label, k, radius, min_pts, weights, predicted)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 12, 6, 0.2\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (3, 9)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(2, 5 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius, min_pts, alpha = 8, 0.12, 2, 1.0\nweights = _oracle_trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)\nneighbours_back = _oracle_neighbour_states(dissimilarity, trajectory_ids, first_index, k, radius, weights)\nneighbours_forw = _oracle_neighbour_states(dissimilarity, trajectory_ids, last_index, k, radius, weights)\nbackward = _oracle_sweep_states(states, trajectory_ids, state_ids, neighbours_back, weights, min_pts, -1)\nforward = _oracle_sweep_states(states, trajectory_ids, state_ids, neighbours_forw, weights, min_pts, 1)\npredicted = _oracle_predicted_trajectory(target_states, target_labels, backward, forward)\n",
            "call": "mean_predicted_deviation(states, trajectory_ids, state_ids, target_label, k, radius, min_pts, weights, predicted)",
            "gold_call": "_oracle_mean_predicted_deviation(states, trajectory_ids, state_ids, target_label, k, radius, min_pts, weights, predicted)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius, min_pts, alpha = 25, 0.065, 3, 3.0\nweights = _oracle_trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)\npredicted = np.column_stack([target_labels, target_states])\ndef run_model():\n    try:\n        mean_predicted_deviation(states, trajectory_ids, state_ids, target_label, k, radius, min_pts, weights, predicted)\n        return 0\n    except ValueError:\n        return 1\ndef _fx_safe(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "_fx_safe(lambda: _oracle_mean_predicted_deviation(states, trajectory_ids, state_ids, target_label, k, radius, min_pts, weights, predicted))",
        },
    ]
