"""
Return the dissimilarity between the target trajectory and one reference trajectory that the source uses for multi-state targets: the directed segment path dissimilarity of its companion trajectory-analysis source, evaluated from the target to the reference on the given state dissimilarity matrix and constructed as follows. To project a state P onto the directed segment from state A to state B, take L = d(A, B), a = d(P, A) and b = d(P, B); if these three values violate the triangle inequality, first add to each of them the smallest constant that restores it, max(L - a - b, a - L - b, b - L - a), and use the corrected values in everything that follows. The position of the projection along the segment is p = (a^2 + L^2 - b^2) / (2 L) and the distance of P to the segment is h = sqrt(a^2 - p^2); if a^2 - p^2 < 0, p < 0 or p > L, the projection is the nearer endpoint instead, with position 0 and distance a when a < b, and position L and distance b otherwise. For two directed segments S1 = (s1 -> e1) and S2 = (s2 -> e2), project s1 and e1 onto S2 and s2 and e2 onto S1. If the projection of s1 lies strictly further along S2 than that of e1, S1 runs against S2 and the distance of e1 is replaced by d(s1, e1) + min(h(s1), h(e1)); S2 is treated the same way with respect to S1. The distance between the two segments is the largest of the four endpoint distances, and the dissimilarity from the target to the reference is the mean, over the target's segments, of the distance to the closest segment of the reference. Both trajectories are given as time-ordered state indices; consecutive indices form the segments.

When the target is observed at several times, its resemblance to a reference trajectory is not the dissimilarity between two states but between two paths through the state space; trajectory analysis compares the segments that join consecutive states, so that the direction and the shape of the paths, and not only their positions, enter the comparison.

Returns
-------
float, the directed trajectory dissimilarity from the target trajectory to the reference trajectory.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def trajectory_dissimilarity(dissimilarity: "numpy.ndarray", target_states: "numpy.ndarray",
                             reference_states: "numpy.ndarray") -> float:
    """Return the dissimilarity between the target trajectory and one reference trajectory that the source uses for multi-state targets: the directed segment path dissimilarity of its companion trajectory-analysis source, evaluated from the target to the reference on the given state dissimilarity matrix and constructed as follows. To project a state P onto the directed segment from state A to state B, take L = d(A, B), a = d(P, A) and b = d(P, B); if these three values violate the triangle inequality, first add to each of them the smallest constant that restores it, max(L - a - b, a - L - b, b - L - a), and use the corrected values in everything that follows. The position of the projection along the segment is p = (a^2 + L^2 - b^2) / (2 L) and the distance of P to the segment is h = sqrt(a^2 - p^2); if a^2 - p^2 < 0, p < 0 or p > L, the projection is the nearer endpoint instead, with position 0 and distance a when a < b, and position L and distance b otherwise. For two directed segments S1 = (s1 -> e1) and S2 = (s2 -> e2), project s1 and e1 onto S2 and s2 and e2 onto S1. If the projection of s1 lies strictly further along S2 than that of e1, S1 runs against S2 and the distance of e1 is replaced by d(s1, e1) + min(h(s1), h(e1)); S2 is treated the same way with respect to S1. The distance between the two segments is the largest of the four endpoint distances, and the dissimilarity from the target to the reference is the mean, over the target's segments, of the distance to the closest segment of the reference. Both trajectories are given as time-ordered state indices; consecutive indices form the segments.

    Parameters
    ----------
    dissimilarity : numpy.ndarray
        Array of shape (n, n) of state dissimilarities (step 03).
    target_states : numpy.ndarray
        One-dimensional integer array of at least two row indices of the target's states, in temporal order.
    reference_states : numpy.ndarray
        One-dimensional integer array of at least two row indices of the reference trajectory's states, in temporal order.

    Returns
    -------
    distance : float
        The trajectory dissimilarity from the target to the reference, as a native Python float.

    Raises
    ------
    ValueError
        If dissimilarity is not a finite nonnegative square matrix, either index array has fewer than two entries or an index outside the matrix, or a segment has zero length.
    """
    return distance

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _segment_projection(d_ref: float, d1: float, d2: float) -> "numpy.ndarray":
    """Companion-source projection of a point onto a directed segment from the three dissimilarities (segment length
    d_ref, point to start d1, point to end d2): [start-to-projection, projection-to-end, point-to-segment], with the
    triangle-inequality correction (the smallest constant added to the three values that closes the triangle) and the
    nearest-endpoint rule when the projection falls outside the segment."""
    dref, a, b = float(d_ref), float(d1), float(d2)
    k = max(0.0, dref - (a + b), b - (dref + a), a - (b + dref))       # ecotraj: correct the triangle inequality
    a, b, dref = a + k, b + k, dref + k
    a1 = (a * a + dref * dref - b * b) / (2.0 * dref)
    a2 = dref - a1
    sq = a * a - a1 * a1
    h = np.sqrt(sq) if sq >= 0.0 else np.nan
    if np.isnan(h) or a1 < 0.0 or a2 < 0.0:                          # outside the segment: nearest endpoint
        if a < b:
            return np.array([0.0, dref, a])
        return np.array([dref, 0.0, b])
    return np.array([a1, a2, h])


def _segment_pair_distance(D: "numpy.ndarray", s1: int, e1: int, s2: int, e2: int) -> float:
    """Companion-source directed distance between two segments (s1->e1, s2->e2) from the dissimilarity matrix."""
    ps1 = _segment_projection(D[s2, e2], D[s1, s2], D[s1, e2])
    pe1 = _segment_projection(D[s2, e2], D[e1, s2], D[e1, e2])
    ps2 = _segment_projection(D[s1, e1], D[s1, s2], D[e1, s2])
    pe2 = _segment_projection(D[s1, e1], D[s1, e2], D[e1, e2])
    ds1, de1, ds2, de2 = ps1[2], pe1[2], ps2[2], pe2[2]
    if ps1[0] > pe1[0]:                                               # segment 1 runs against segment 2
        de1 = min(D[s1, e1] + ps1[2], D[s1, e1] + pe1[2])
    if ps2[0] > pe2[0]:
        de2 = min(D[s2, e2] + ps2[2], D[s2, e2] + pe2[2])
    return float(max(ds1, de1, ds2, de2))


def _oracle_trajectory_dissimilarity(dissimilarity: "numpy.ndarray", target_states: "numpy.ndarray",
                                     reference_states: "numpy.ndarray") -> float:
    """Directed segment path dissimilarity from the target trajectory to a reference trajectory (companion source):
    the mean, over the target's segments, of the smallest directed segment distance to any reference segment."""
    D = np.asarray(dissimilarity, dtype=np.float64)
    ta = np.asarray(target_states).astype(int); tb = np.asarray(reference_states).astype(int)
    n = D.shape[0] if D.ndim == 2 else 0
    if D.ndim != 2 or D.shape[1] != n or n < 2 or not np.all(np.isfinite(D)) or np.any(D < 0.0):
        raise ValueError("dissimilarity must be a finite nonnegative square matrix")
    if ta.ndim != 1 or tb.ndim != 1 or ta.size < 2 or tb.size < 2:
        raise ValueError("each trajectory needs at least two states")
    if np.any(ta < 0) or np.any(ta >= n) or np.any(tb < 0) or np.any(tb >= n):
        raise ValueError("trajectory indices must be valid state indices")
    seg_a = [(int(ta[i]), int(ta[i + 1])) for i in range(ta.size - 1)]
    seg_b = [(int(tb[i]), int(tb[i + 1])) for i in range(tb.size - 1)]
    for s_, e_ in seg_a + seg_b:
        if D[s_, e_] <= 0.0:
            raise ValueError("segments must have positive length")
    M = np.array([[_segment_pair_distance(D, s1, e1, s2, e2) for (s2, e2) in seg_b] for (s1, e1) in seg_a])
    return float(np.mean(M.min(axis=1)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nreference_states = np.where(trajectory_ids == 36)[0]\ntarget_states = np.where(trajectory_ids == target_label)[0]\n",
            "call": "trajectory_dissimilarity(dissimilarity, target_states, reference_states)",
            "gold_call": "_oracle_trajectory_dissimilarity(dissimilarity, target_states, reference_states)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nreference_states = np.where(trajectory_ids == 7)[0]\ntarget_states = np.where(trajectory_ids == target_label)[0]\n",
            "call": "trajectory_dissimilarity(dissimilarity, target_states, reference_states)",
            "gold_call": "_oracle_trajectory_dissimilarity(dissimilarity, target_states, reference_states)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 12, 6, 0.2\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (3, 9)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(2, 5 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nreference_states = np.where(trajectory_ids == 9)[0]\ntarget_states = np.where(trajectory_ids == target_label)[0]\n",
            "call": "trajectory_dissimilarity(dissimilarity, target_states, reference_states)",
            "gold_call": "_oracle_trajectory_dissimilarity(dissimilarity, target_states, reference_states)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nreference_states = np.where(trajectory_ids == 36)[0][::-1]\ntarget_states = np.where(trajectory_ids == target_label)[0]\n",
            "call": "trajectory_dissimilarity(dissimilarity, target_states, reference_states)",
            "gold_call": "_oracle_trajectory_dissimilarity(dissimilarity, target_states, reference_states)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nP = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.5], [0.2, 1.0], [1.4, 1.2], [2.6, 0.9]])\ndissimilarity = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2))\ntarget_states = np.array([0, 1, 2])\nreference_states = np.array([3, 4, 5])\n",
            "call": "trajectory_dissimilarity(dissimilarity, target_states, reference_states)",
            "gold_call": "_oracle_trajectory_dissimilarity(dissimilarity, target_states, reference_states)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\nX = _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)\ntr = np.repeat(np.arange(1, n_traj + 1), n_states)\nst = np.tile(np.arange(1, n_states + 1), n_traj)\ns_idx = np.arange(1, 9)\ninits = [base * (1.0 + amplitude * (2.0 * ((1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001) / 1000.0 - 1.0)) for j in (13, 31)]\nN = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), r, K, A, n_states)\ntarget_labels = np.arange(4, 6 + 1)\ntarget_states = N[target_labels - 1] / N[target_labels - 1].sum(axis=1, keepdims=True)\nstates = np.vstack([X, target_states])\ntrajectory_ids = np.concatenate([tr, np.zeros(target_labels.size, dtype=int)])\nstate_ids = np.concatenate([st, target_labels])\ntarget_label = 0\nfirst_index = X.shape[0]\nlast_index = states.shape[0] - 1\n_bc_diff = np.abs(states[:, None, :] - states[None, :, :]).sum(axis=2)\ndissimilarity = _bc_diff / (states[:, None, :] + states[None, :, :]).sum(axis=2)\nnp.fill_diagonal(dissimilarity, 0.0)\nreference_states = np.where(trajectory_ids == 36)[0][:1]\ntarget_states = np.where(trajectory_ids == target_label)[0]\ndef run_model():\n    try:\n        trajectory_dissimilarity(dissimilarity, target_states, reference_states)\n        return 0\n    except ValueError:\n        return 1\ndef _fx_safe(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "_fx_safe(lambda: _oracle_trajectory_dissimilarity(dissimilarity, target_states, reference_states))",
        },
    ]
