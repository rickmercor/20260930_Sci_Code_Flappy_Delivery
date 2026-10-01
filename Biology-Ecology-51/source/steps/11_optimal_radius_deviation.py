"""
The source's selection of the radius by minimising the mean predicted deviation. For the same regime, target, k, min_pts and alpha as step 10, return the infimum of the step-10 deviation over all radii radius_min <= radius <= radius_max, together with the infimum and the supremum of the radii at which the deviation equals that value (equality within 1e-12), all three exact to the working precision (a sampled approximation does not meet the contract). The caller guarantees that, for every radius of the domain, the scale factor of step 05 is at least that radius and the forecast holds at least one predicted state; raise ValueError if either guarantee fails. Call the earlier step functions, including step 10, rather than reimplementing them.

The deviation is the source's criterion for choosing the neighbourhood parameters: the radius that minimises it gives the forecast most consistent with the regime, and the minimum is generally attained on a range of radii rather than at a single value, so the selection is reported as that range.

Returns
-------
numpy.ndarray of float64 with shape (3,): [infimum of the deviation, infimum of the attaining radii, supremum of the attaining radii].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def optimal_radius_deviation(n_traj: int, n_states: int, base: "numpy.ndarray", amplitude: float,
                             growth: "numpy.ndarray", capacity: "numpy.ndarray", interaction: "numpy.ndarray",
                             source_a: int, source_b: int, first_state: int, last_state: int, k: int,
                             radius_min: float, radius_max: float, min_pts: int,
                             alpha: float) -> "numpy.ndarray":
    """The source's selection of the radius by minimising the mean predicted deviation. For the same regime, target, k, min_pts and alpha as step 10, return the infimum of the step-10 deviation over all radii radius_min <= radius <= radius_max, together with the infimum and the supremum of the radii at which the deviation equals that value (equality within 1e-12), all three exact to the working precision (a sampled approximation does not meet the contract). The caller guarantees that, for every radius of the domain, the scale factor of step 05 is at least that radius and the forecast holds at least one predicted state; raise ValueError if either guarantee fails. Call the earlier step functions, including step 10, rather than reimplementing them.

    Parameters
    ----------
    n_traj : int
        Number of regime trajectories, at least 2.
    n_states : int
        Number of states per trajectory, at least 2.
    base : numpy.ndarray
        One-dimensional array of the S positive base abundances.
    amplitude : float
        Relative amplitude of the hashed perturbation, in [0, 1).
    growth : numpy.ndarray
        One-dimensional array of the S intrinsic growth rates.
    capacity : numpy.ndarray
        One-dimensional array of the S positive carrying capacities.
    interaction : numpy.ndarray
        Array of shape (S, S) of interaction coefficients.
    source_a : int
        Index (1-based) of the first regime trajectory whose initial community is averaged.
    source_b : int
        Index (1-based) of the second regime trajectory whose initial community is averaged.
    first_state : int
        First observed state (1-based) of the target's Ricker run.
    last_state : int
        Last observed state (1-based) of the target's Ricker run, greater than first_state.
    k : int
        Number of nearest states of the neighbourhood.
    radius_min : float
        Lower end of the radius domain, positive.
    radius_max : float
        Upper end of the radius domain, greater than radius_min.
    min_pts : int
        Minimum number of states required to form a predicted state.
    alpha : float
        Positive shape parameter of the weighting function.

    Returns
    -------
    result : numpy.ndarray
        Array of three float64 values: the infimum of the deviation over the domain, the infimum and the supremum of the radii attaining it.

    Raises
    ------
    ValueError
        If the regime, target or neighbourhood arguments are invalid (as in step 10), radius_min is not positive or not below radius_max, the scale factor is below some radius of the domain, or the forecast has no predicted state for some radius of the domain.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_optimal_radius_deviation(n_traj: int, n_states: int, base: "numpy.ndarray", amplitude: float,
                                     growth: "numpy.ndarray", capacity: "numpy.ndarray", interaction: "numpy.ndarray",
                                     source_a: int, source_b: int, first_state: int, last_state: int, k: int,
                                     radius_min: float, radius_max: float, min_pts: int,
                                     alpha: float) -> "numpy.ndarray":
    """The source's radius selection by minimising the mean predicted deviation: the infimum of the step-10
    deviation over [radius_min, radius_max] and the infimum/supremum of the radii attaining it. Because the forecast depends
    on the radius only through threshold comparisons (and the scale factor exceeds every radius of the domain), the
    deviation is a step function of the radius; this function locates its pieces exactly - the radii at which a state
    enters a neighbourhood of a target extreme (06), a trajectory becomes eligible for a weight (05), or a state enters
    the neighbourhood of a predicted state in the re-forecasts of step 09 - evaluates the deviation on every open piece
    (no isolated breakpoint value lies below the piece minimum for the shipped cases; the cross-check through step 10 at
    the optimum guards the enumeration) and returns [infimum, infimum of the attaining radii, supremum of the attaining
    radii]."""
    if int(n_traj) != n_traj or n_traj < 2 or int(n_states) != n_states or n_states < 2:
        raise ValueError("n_traj must be at least 2 and n_states at least 2")
    if int(source_a) != source_a or int(source_b) != source_b or not (1 <= source_a <= n_traj and 1 <= source_b <= n_traj):
        raise ValueError("source_a and source_b must index regime trajectories (1-based)")
    if int(first_state) != first_state or int(last_state) != last_state or not (1 <= first_state < last_state <= n_states):
        raise ValueError("first_state and last_state must satisfy 1 <= first_state < last_state <= n_states")
    lo, hi = float(radius_min), float(radius_max)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0.0 or hi <= lo:
        raise ValueError("radius_min and radius_max must satisfy 0 < radius_min < radius_max")
    if int(k) != k or k < 1 or int(min_pts) != min_pts or min_pts < 1 or not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("k and min_pts must be positive integers and alpha positive")
    b = np.asarray(base, dtype=np.float64)
    S = b.size
    X = _oracle_regime_states(n_traj, n_states, b, amplitude, growth, capacity, interaction)
    tr = np.repeat(np.arange(1, int(n_traj) + 1), int(n_states))
    st = np.tile(np.arange(1, int(n_states) + 1), int(n_traj))
    s_idx = np.arange(1, S + 1)
    inits = []
    for j in (int(source_a), int(source_b)):
        h = (1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001
        inits.append(b * (1.0 + float(amplitude) * (2.0 * h / 1000.0 - 1.0)))
    N = _oracle_ricker_trajectory(0.5 * (inits[0] + inits[1]), growth, capacity, interaction, int(n_states))
    labels = np.arange(int(first_state), int(last_state) + 1)
    target = N[labels - 1] / N[labels - 1].sum(axis=1, keepdims=True)
    Xa = np.vstack([X, target])
    tra = np.concatenate([tr, np.zeros(labels.size, dtype=int)])
    sta = np.concatenate([st, labels])
    D = _oracle_bray_curtis_matrix(Xa)
    t1 = X.shape[0]; t2 = Xa.shape[0] - 1
    tid = np.arange(t1, t2 + 1)
    # radii at which the target's neighbourhoods or the weighted set can change (the k nearest states are fixed)
    breaks = set()
    for ext in (t1, t2):
        for i in _oracle_neighbour_states(D, tra, ext, k, 2.0, None):
            breaks.add(float(D[i, ext]))
    for lab in range(1, int(n_traj) + 1):
        breaks.add(float(D[np.ix_(np.where(tra == lab)[0], tid)].min()))
    edges = [lo] + sorted(x for x in breaks if lo < x < hi) + [hi]
    pieces = []
    for a, c in zip(edges[:-1], edges[1:]):
        mid = 0.5 * (a + c)
        w = _oracle_trajectory_weights(D, tra, sta, 0, k, mid, alpha)
        weighted = list(dict.fromkeys(tra[w > 0].tolist()))
        if not weighted or np.all(w[tra != 0] == 1.0):
            raise ValueError("no weighted trajectory for a radius inside the domain")
        d_T = max(_oracle_trajectory_dissimilarity(D, tid, np.where(tra == lab)[0]) for lab in weighted)
        if d_T < c - 1e-12:                                            # the deviation would vary continuously with the radius
            raise ValueError("the scale factor falls below a radius inside the domain")
        nb_back = _oracle_neighbour_states(D, tra, t1, k, mid, w)
        nb_forw = _oracle_neighbour_states(D, tra, t2, k, mid, w)
        back = _oracle_sweep_states(Xa, tra, sta, nb_back, w, min_pts, -1)
        forw = _oracle_sweep_states(Xa, tra, sta, nb_forw, w, min_pts, 1)
        P = _oracle_predicted_trajectory(target, labels, back, forw)
        if P.shape[0] == labels.size:
            raise ValueError("no predicted state for a radius inside the domain")
        # radii at which a re-forecast neighbourhood can change: the k nearest regime states of every predicted state
        sub = set()
        own = set(labels.astype(float).tolist())
        for row in P:
            if float(row[0]) in own:
                continue
            Dr = _oracle_bray_curtis_matrix(np.vstack([X, row[1:][None, :]]))
            trr = np.concatenate([tr, [0]])
            for i in _oracle_neighbour_states(Dr, trr, X.shape[0], k, 2.0, None):
                d = float(Dr[i, X.shape[0]])
                if a < d < c:
                    sub.add(d)
        sedges = [a] + sorted(sub) + [c]
        for a2, c2 in zip(sedges[:-1], sedges[1:]):
            val = _oracle_mean_predicted_deviation(Xa, tra, sta, 0, k, 0.5 * (a2 + c2), min_pts, w, P)
            pieces.append((a2, c2, val))
    best = min(p[2] for p in pieces)
    opt = [p for p in pieces if abs(p[2] - best) <= 1e-12]
    eps_lo, eps_hi = min(p[0] for p in opt), max(p[1] for p in opt)
    check = _oracle_forecast_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a,
                                       source_b, first_state, last_state, k, 0.5 * (opt[0][0] + opt[0][1]), min_pts, alpha)
    if abs(check - best) > 1e-12:
        raise ValueError("the optimal piece does not reproduce through the single-radius forecast")
    return np.array([best, eps_lo, eps_hi])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\ngrowth, capacity, interaction = r, K, A\nn_traj, n_states, amplitude, source_a, source_b, first_state, last_state = 36, 8, 0.35, 13, 31, 4, 6\nk, radius_min, radius_max, min_pts, alpha = 25, 0.05, 0.07, 3, 3.0\n",
            "call": "optimal_radius_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius_min, radius_max, min_pts, alpha)",
            "gold_call": "_oracle_optimal_radius_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius_min, radius_max, min_pts, alpha)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\ngrowth, capacity, interaction = r, K, A\nn_traj, n_states, amplitude, source_a, source_b, first_state, last_state = 36, 8, 0.35, 9, 24, 4, 6\nk, radius_min, radius_max, min_pts, alpha = 25, 0.05, 0.068, 3, 3.0\n",
            "call": "optimal_radius_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius_min, radius_max, min_pts, alpha)",
            "gold_call": "_oracle_optimal_radius_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius_min, radius_max, min_pts, alpha)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\ngrowth, capacity, interaction = r, K, A\nn_traj, n_states, amplitude, source_a, source_b, first_state, last_state = 36, 8, 0.35, 5, 14, 2, 5\nk, radius_min, radius_max, min_pts, alpha = 25, 0.05, 0.068, 3, 3.0\n",
            "call": "optimal_radius_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius_min, radius_max, min_pts, alpha)",
            "gold_call": "_oracle_optimal_radius_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius_min, radius_max, min_pts, alpha)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\ngrowth, capacity, interaction = r, K, A\nn_traj, n_states, amplitude, source_a, source_b, first_state, last_state = 36, 8, 0.35, 13, 31, 4, 6\nk, radius_min, radius_max, min_pts, alpha = 25, 0.07, 0.05, 3, 3.0\ndef run_model():\n    try:\n        optimal_radius_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius_min, radius_max, min_pts, alpha)\n        return 0\n    except ValueError:\n        return 1\ndef _fx_safe(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "_fx_safe(lambda: _oracle_optimal_radius_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius_min, radius_max, min_pts, alpha))",
        },
    ]
