"""
Return, as a vector with one entry per state, the trajectory weights that the source's distance-weighting scheme (its Eq 2 with the exponential function of its Table S1 and shape parameter alpha) assigns around a multi-state target: a reference trajectory T receives the weight exp(-alpha d_T / d_max) when it contains one of the k nearest states, among all states outside the target's trajectory (ties in index order), of the target's first or last state and has a state at dissimilarity at most radius from some state of the target, where d_T is the trajectory dissimilarity of step 04 from the target to T and the scale factor d_max is the larger of radius and the largest d_T over those trajectories (the trajectory that sets d_max keeps the weight exp(-alpha)); every state of a trajectory carries that trajectory's weight, states of the target's own trajectory carry 0, and trajectories outside the source's weighted set carry 0.

Rather than a hard neighbourhood, the algorithm can weight the reference trajectories by a decreasing function of their dissimilarity to the target scaled by a threshold, so that the trajectories most similar to the target dominate the averaging.

Returns
-------
numpy.ndarray of float64 with shape (n,): the per-state trajectory weights.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def trajectory_weights(dissimilarity: "numpy.ndarray", trajectory_ids: "numpy.ndarray",
                       state_ids: "numpy.ndarray", target_label: int, k: int, radius: float,
                       alpha: float) -> "numpy.ndarray":
    """Return, as a vector with one entry per state, the trajectory weights that the source's distance-weighting scheme (its Eq 2 with the exponential function of its Table S1 and shape parameter alpha) assigns around a multi-state target: a reference trajectory T receives the weight exp(-alpha d_T / d_max) when it contains one of the k nearest states, among all states outside the target's trajectory (ties in index order), of the target's first or last state and has a state at dissimilarity at most radius from some state of the target, where d_T is the trajectory dissimilarity of step 04 from the target to T and the scale factor d_max is the larger of radius and the largest d_T over those trajectories (the trajectory that sets d_max keeps the weight exp(-alpha)); every state of a trajectory carries that trajectory's weight, states of the target's own trajectory carry 0, and trajectories outside the source's weighted set carry 0.

    Parameters
    ----------
    dissimilarity : numpy.ndarray
        Array of shape (n, n) of state dissimilarities (step 03).
    trajectory_ids : numpy.ndarray
        One-dimensional array of length n: the trajectory of every state.
    state_ids : numpy.ndarray
        One-dimensional array of length n of integers giving the temporal order of the states within their trajectory.
    target_label : int
        Trajectory label of the target, which holds at least two states.
    k : int
        Number of nearest states of the neighbourhood, at least 1.
    radius : float
        Positive dissimilarity threshold of the neighbourhood.
    alpha : float
        Positive shape parameter of the weighting function.

    Returns
    -------
    weights : numpy.ndarray
        One-dimensional array of length n of nonnegative per-state weights (float64).

    Raises
    ------
    ValueError
        If dissimilarity is not a finite square matrix matching trajectory_ids and state_ids, the target has fewer than two states, k is not a positive integer, radius or alpha is not positive and finite, or the target's trajectory is the only one.
    """
    return weights

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_trajectory_weights(dissimilarity: "numpy.ndarray", trajectory_ids: "numpy.ndarray",
                               state_ids: "numpy.ndarray", target_label: int, k: int, radius: float,
                               alpha: float) -> "numpy.ndarray":
    """Per-state vector of the source's exponential distance weights around a multi-state target: w_T = exp(-alpha d_T /
    d_max) for every reference trajectory T that contains one of the k nearest states of either extreme of the target and
    has a state within the radius of some target state, with d_T the directed segment path dissimilarity from the target
    to T (step 04) and d_max = max(radius, max d_T over those trajectories); all other states, including the target's
    own, get 0."""
    D = np.asarray(dissimilarity, dtype=np.float64); tr = np.asarray(trajectory_ids); st = np.asarray(state_ids)
    n = tr.size
    if D.shape != (n, n) or n < 1 or not np.all(np.isfinite(D)) or st.shape != (n,):
        raise ValueError("dissimilarity must be a finite square matrix matching trajectory_ids and state_ids")
    tid = np.where(tr == target_label)[0]
    if tid.size < 2:
        raise ValueError("the target must hold at least two states")
    if int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    if not np.isfinite(radius) or radius <= 0.0 or not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("radius and alpha must be positive and finite")
    tid = tid[np.argsort(st[tid], kind="stable")]
    t1, t2 = int(tid[0]), int(tid[-1])
    others = np.array([i for i in range(n) if tr[i] != target_label], dtype=int)
    if others.size == 0:
        raise ValueError("the target must not be the only trajectory")
    k_traj = set()
    for ext in (t1, t2):
        order = others[np.argsort(D[others, ext], kind="stable")][: int(k)]
        k_traj |= set(tr[order].tolist())
    within = others[(D[np.ix_(others, tid)] <= float(radius)).any(axis=1)]
    eps_traj = set(tr[within].tolist())
    d_traj = {}
    for lab in dict.fromkeys(tr[others].tolist()):
        idx = others[tr[others] == lab]
        if lab in k_traj and lab in eps_traj and idx.size >= 2:
            idx = idx[np.argsort(st[idx], kind="stable")]
            d_traj[lab] = _oracle_trajectory_dissimilarity(D, tid, idx)
    d_max = max(float(radius), max(d_traj.values())) if d_traj else float(radius)
    w = np.zeros(n)
    for lab, d in d_traj.items():
        w[tr == lab] = np.exp(-float(alpha) * d / d_max)
    return w

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius, alpha = 25, 0.065, 3.0\n",
            "call": "trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)",
            "gold_call": "_oracle_trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius, alpha = 15, 0.08, 2.0\n",
            "call": "trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)",
            "gold_call": "_oracle_trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 12, 6, 0.2\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (3, 9)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(2, 5 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius, alpha = 8, 0.12, 1.0\n",
            "call": "trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)",
            "gold_call": "_oracle_trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nk, radius, alpha = 15, 0.0, 2.0\ndef run_model():\n    try:\n        trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha)\n        return 0\n    except ValueError:\n        return 1\ndef _fx_safe(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "_fx_safe(lambda: _oracle_trajectory_weights(dissimilarity, trajectory_ids, state_ids, target_label, k, radius, alpha))",
        },
    ]
