"""
Whole forecast at one radius. Build the synthetic regime of step 02 (n_traj trajectories of n_states states from base, amplitude, growth, capacity and interaction), define the target as states first_state to last_state of the Ricker run (step 01) started from the mean of the initial communities of trajectories source_a and source_b (the same hashed initial abundances as step 02), append it to the regime as its own trajectory, compute the Bray-Curtis matrix (step 03), the trajectory weights (step 05, exponential with alpha), the trajectory dissimilarity of step 04 from the target to every weighted trajectory (called directly, as a consistency check of the weights that raises ValueError on disagreement), the neighbourhoods of the first and last target states (step 06 with k and radius), the backward and forward sweeps (step 07 with min_pts), the predicted trajectory (step 08) and return its mean predicted deviation (step 09). Call the earlier step functions rather than reimplementing them.

The whole forecasting pipeline at a fixed radius, from a reproducible synthetic dynamic regime to the accuracy metric of the forecast of one partially observed target trajectory, yields a single number that every convention of the forecasting algorithm feeds into.

Returns
-------
float, the mean predicted deviation of the forecast at the given radius.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def forecast_deviation(n_traj: int, n_states: int, base: "numpy.ndarray", amplitude: float,
                       growth: "numpy.ndarray", capacity: "numpy.ndarray", interaction: "numpy.ndarray",
                       source_a: int, source_b: int, first_state: int, last_state: int, k: int,
                       radius: float, min_pts: int, alpha: float) -> float:
    """Whole forecast at one radius. Build the synthetic regime of step 02 (n_traj trajectories of n_states states from base, amplitude, growth, capacity and interaction), define the target as states first_state to last_state of the Ricker run (step 01) started from the mean of the initial communities of trajectories source_a and source_b (the same hashed initial abundances as step 02), append it to the regime as its own trajectory, compute the Bray-Curtis matrix (step 03), the trajectory weights (step 05, exponential with alpha), the trajectory dissimilarity of step 04 from the target to every weighted trajectory (called directly, as a consistency check of the weights that raises ValueError on disagreement), the neighbourhoods of the first and last target states (step 06 with k and radius), the backward and forward sweeps (step 07 with min_pts), the predicted trajectory (step 08) and return its mean predicted deviation (step 09). Call the earlier step functions rather than reimplementing them.

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
    radius : float
        Positive dissimilarity threshold of the neighbourhood.
    min_pts : int
        Minimum number of states required to form a predicted state.
    alpha : float
        Positive shape parameter of the weighting function.

    Returns
    -------
    deviation : float
        The mean predicted deviation of the target's predicted trajectory at this radius, as a native Python float.

    Raises
    ------
    ValueError
        If n_traj or n_states is below 2, source_a, source_b, first_state or last_state are out of range or not ordered, the weights disagree with the trajectory dissimilarities, or the inputs are invalid for the earlier steps.
    """
    return deviation

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_forecast_deviation(n_traj: int, n_states: int, base: "numpy.ndarray", amplitude: float,
                               growth: "numpy.ndarray", capacity: "numpy.ndarray", interaction: "numpy.ndarray",
                               source_a: int, source_b: int, first_state: int, last_state: int, k: int,
                               radius: float, min_pts: int, alpha: float) -> float:
    """Whole forecast at one radius: build the regime (step 02), the target (states first_state..last_state of the run
    started from the mean of the initial communities of trajectories source_a and source_b), the dissimilarity matrix
    (03), the weights (05), the trajectory dissimilarities of the weighted trajectories (04, as a consistency gate on the
    weights), the two neighbourhoods (06), the sweeps (07), the predicted trajectory (08) and its mean predicted deviation
    (09)."""
    if int(n_traj) != n_traj or n_traj < 2 or int(n_states) != n_states or n_states < 2:
        raise ValueError("n_traj must be at least 2 and n_states at least 2")
    if int(source_a) != source_a or int(source_b) != source_b or not (1 <= source_a <= n_traj and 1 <= source_b <= n_traj):
        raise ValueError("source_a and source_b must index regime trajectories (1-based)")
    if int(first_state) != first_state or int(last_state) != last_state or not (1 <= first_state < last_state <= n_states):
        raise ValueError("first_state and last_state must satisfy 1 <= first_state < last_state <= n_states")
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
    tra = np.concatenate([tr, np.zeros(labels.size, dtype=int)])          # trajectory 0 = the target
    sta = np.concatenate([st, labels])
    D = _oracle_bray_curtis_matrix(Xa)
    w = _oracle_trajectory_weights(D, tra, sta, 0, k, radius, alpha)
    t1 = X.shape[0]; t2 = Xa.shape[0] - 1
    tid = np.arange(t1, t2 + 1)
    weighted = list(dict.fromkeys(tra[w > 0].tolist()))
    if weighted and not np.all(w[tra != 0] == 1.0):                     # consistency gate on the weights (step 04)
        d_T = {lab: _oracle_trajectory_dissimilarity(D, tid, np.where(tra == lab)[0]) for lab in weighted}
        d_max = max(float(radius), max(d_T.values()))
        for lab, d in d_T.items():
            if abs(float(w[tra == lab][0]) - np.exp(-float(alpha) * d / d_max)) > 1e-9:
                raise ValueError("trajectory weights are inconsistent with the trajectory dissimilarities")
    nb_back = _oracle_neighbour_states(D, tra, t1, k, radius, w)
    nb_forw = _oracle_neighbour_states(D, tra, t2, k, radius, w)
    back = _oracle_sweep_states(Xa, tra, sta, nb_back, w, min_pts, -1)
    forw = _oracle_sweep_states(Xa, tra, sta, nb_forw, w, min_pts, 1)
    P = _oracle_predicted_trajectory(target, labels, back, forw)
    return _oracle_mean_predicted_deviation(Xa, tra, sta, 0, k, radius, min_pts, w, P)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\ngrowth, capacity, interaction = r, K, A\nn_traj, n_states, amplitude, source_a, source_b, first_state, last_state = 36, 8, 0.35, 13, 31, 4, 6\nk, radius, min_pts, alpha = 25, 0.065, 3, 3.0\n",
            "call": "forecast_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius, min_pts, alpha)",
            "gold_call": "_oracle_forecast_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius, min_pts, alpha)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\ngrowth, capacity, interaction = r, K, A\nn_traj, n_states, amplitude, source_a, source_b, first_state, last_state = 36, 8, 0.35, 3, 22, 3, 5\nk, radius, min_pts, alpha = 20, 0.08, 4, 3.0\n",
            "call": "forecast_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius, min_pts, alpha)",
            "gold_call": "_oracle_forecast_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius, min_pts, alpha)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\ngrowth, capacity, interaction = r, K, A\nn_traj, n_states, amplitude, source_a, source_b, first_state, last_state = 12, 6, 0.2, 3, 9, 2, 5\nk, radius, min_pts, alpha = 8, 0.12, 2, 1.0\n",
            "call": "forecast_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius, min_pts, alpha)",
            "gold_call": "_oracle_forecast_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius, min_pts, alpha)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\ngrowth, capacity, interaction = r, K, A\nn_traj, n_states, amplitude, source_a, source_b, first_state, last_state = 36, 8, 0.35, 7, 40, 3, 5\nk, radius, min_pts, alpha = 25, 0.065, 3, 3.0\ndef run_model():\n    try:\n        forecast_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius, min_pts, alpha)\n        return 0\n    except ValueError:\n        return 1\ndef _fx_safe(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "_fx_safe(lambda: _oracle_forecast_deviation(n_traj, n_states, base, amplitude, growth, capacity, interaction, source_a, source_b, first_state, last_state, k, radius, min_pts, alpha))",
        },
    ]
