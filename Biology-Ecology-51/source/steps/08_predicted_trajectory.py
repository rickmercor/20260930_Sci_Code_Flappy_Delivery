"""
Assemble the predicted trajectory of a multi-state target from its observed states, the backward sweep of its first state and the forward sweep of its last state (step 07): an array whose column 0 holds the state label (the observed labels for the target's own states, decreasing by one per backward predicted state from the first label and increasing by one per forward predicted state from the last label) and whose remaining columns hold the state variables; the target's own states are included and rows are sorted by label. Either sweep may be empty.

The predicted trajectory of a target is its observed states extended in both temporal directions by the states that its neighbourhoods in the dynamic regime imply; labelling the states by their temporal position keeps the order explicit.

Returns
-------
numpy.ndarray of float64 with shape (m_b + m_f + n_X, S + 1): labels in column 0, then the state variables.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def predicted_trajectory(target_states: "numpy.ndarray", target_labels: "numpy.ndarray", backward: "numpy.ndarray",
                         forward: "numpy.ndarray") -> "numpy.ndarray":
    """Assemble the predicted trajectory of a multi-state target from its observed states, the backward sweep of its first state and the forward sweep of its last state (step 07): an array whose column 0 holds the state label (the observed labels for the target's own states, decreasing by one per backward predicted state from the first label and increasing by one per forward predicted state from the last label) and whose remaining columns hold the state variables; the target's own states are included and rows are sorted by label. Either sweep may be empty.

    Parameters
    ----------
    target_states : numpy.ndarray
        Array of shape (n_X, S) of the state variables of the target's observed states in temporal order.
    target_labels : numpy.ndarray
        One-dimensional array of n_X distinct integer temporal labels of the target's states.
    backward : numpy.ndarray
        Array of shape (m_b, S) of the backward sweep (step 07, direction -1), first predicted state first; may have zero rows.
    forward : numpy.ndarray
        Array of shape (m_f, S) of the forward sweep (step 07, direction +1), first predicted state first; may have zero rows.

    Returns
    -------
    predicted : numpy.ndarray
        Array of shape (m_b + m_f + n_X, S + 1): column 0 the labels, columns 1..S the state variables (float64).

    Raises
    ------
    ValueError
        If target_states is not a finite two-dimensional array with one finite label per row, the labels are not distinct integers, or a sweep contains a non-finite value.
    """
    return predicted

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_predicted_trajectory(target_states: "numpy.ndarray", target_labels: "numpy.ndarray", backward: "numpy.ndarray",
                                 forward: "numpy.ndarray") -> "numpy.ndarray":
    """Assemble the PETRA-EDR predicted trajectory of a target from its observed states and the backward sweep of its
    first state and forward sweep of its last state (step 07): column 0 holds the label (the observed labels, then
    decreasing by one per backward step from the first label and increasing by one per forward step from the last) and
    the remaining columns the state variables; rows sorted by label."""
    X = np.asarray(target_states, dtype=np.float64); lab = np.asarray(target_labels, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[0] < 1 or lab.shape != (X.shape[0],) or not np.all(np.isfinite(X)) or not np.all(np.isfinite(lab)):
        raise ValueError("target_states must be a finite (n_X, S) array with one finite label per state")
    if np.any(lab != np.round(lab)) or np.unique(lab).size != lab.size:
        raise ValueError("target_labels must be distinct integers")
    S = X.shape[1]
    B = np.asarray(backward, dtype=np.float64).reshape(-1, S) if np.asarray(backward).size else np.empty((0, S))
    F = np.asarray(forward, dtype=np.float64).reshape(-1, S) if np.asarray(forward).size else np.empty((0, S))
    if not (np.all(np.isfinite(B)) and np.all(np.isfinite(F))):
        raise ValueError("the sweeps must be finite")
    rows = [np.concatenate([[float(l)], x]) for l, x in zip(lab, X)]
    lo, hi = float(lab.min()), float(lab.max())
    for i, row in enumerate(B, 1):
        rows.append(np.concatenate([[lo - i], row]))
    for i, row in enumerate(F, 1):
        rows.append(np.concatenate([[hi + i], row]))
    out = np.array(rows)
    return out[np.argsort(out[:, 0], kind="stable")]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius, min_pts, alpha = 25, 0.065, 3, 3.0\nweights = _oracle_trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)\nneighbours_back = _oracle_neighbour_states(dissimilarity, trajectory_ids, first_index, k, radius, weights)\nneighbours_forw = _oracle_neighbour_states(dissimilarity, trajectory_ids, last_index, k, radius, weights)\nbackward = _oracle_sweep_states(states, trajectory_ids, state_ids, neighbours_back, weights, min_pts, -1)\nforward = _oracle_sweep_states(states, trajectory_ids, state_ids, neighbours_forw, weights, min_pts, 1)\n",
            "call": "predicted_trajectory(target_states, target_labels, backward, forward)",
            "gold_call": "_oracle_predicted_trajectory(target_states, target_labels, backward, forward)",
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius, min_pts, alpha = 15, 0.08, 4, 2.0\nweights = _oracle_trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)\nneighbours_back = _oracle_neighbour_states(dissimilarity, trajectory_ids, first_index, k, radius, weights)\nneighbours_forw = _oracle_neighbour_states(dissimilarity, trajectory_ids, last_index, k, radius, weights)\nbackward = _oracle_sweep_states(states, trajectory_ids, state_ids, neighbours_back, weights, min_pts, -1)\nforward = _oracle_sweep_states(states, trajectory_ids, state_ids, neighbours_forw, weights, min_pts, 1)\n",
            "call": "predicted_trajectory(target_states, target_labels, backward, forward)",
            "gold_call": "_oracle_predicted_trajectory(target_states, target_labels, backward, forward)",
        },
        {
            "setup": "import numpy as np\ntarget_states = np.array([[0.5, 0.3, 0.2], [0.45, 0.35, 0.2]])\ntarget_labels = np.array([1, 2])\nbackward = np.empty((0, 3))\nforward = np.array([[0.4, 0.4, 0.2], [0.3, 0.5, 0.2]])\n",
            "call": "predicted_trajectory(target_states, target_labels, backward, forward)",
            "gold_call": "_oracle_predicted_trajectory(target_states, target_labels, backward, forward)",
        },
        {
            "setup": "import numpy as np\ntarget_states = np.array([[0.5, 0.3, 0.2], [0.45, 0.35, 0.2]])\ntarget_labels = np.array([1, 1])\nbackward = np.empty((0, 3))\nforward = np.array([[0.4, 0.4, 0.2]])\ndef run_model():\n    try:\n        predicted_trajectory(target_states, target_labels, backward, forward)\n        return 0\n    except ValueError:\n        return 1\ndef _fx_safe(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "_fx_safe(lambda: _oracle_predicted_trajectory(target_states, target_labels, backward, forward))",
        },
    ]
