"""
Return the relative-abundance states of a synthetic dynamic regime of n_traj trajectories with n_states states each: trajectory j (j = 1, ..., n_traj) starts from the absolute abundances base_s (1 + amplitude (2 h_{j,s} / 1000 - 1)) with the integer hash h_{j,s} = (1103 j + 617 s + 251 j s) mod 1001 (species s = 1, ..., S), follows the Ricker map of step 01 with growth, capacity and interaction for n_states states, and every state is divided by its total abundance. Rows are ordered by trajectory and, within a trajectory, by state.

An ecological dynamic regime is a set of trajectories of ecological units that share the same external conditions and dynamics but start from different states; a deterministic hash of the trajectory and species indices gives a reproducible spread of initial communities around a common composition, and relative abundances are the state variables on which community dissimilarities are computed.

Returns
-------
numpy.ndarray of float64 with shape (n_traj * n_states, S): relative abundances, trajectory-major order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def regime_states(n_traj: int, n_states: int, base: "numpy.ndarray", amplitude: float,
                  growth: "numpy.ndarray", capacity: "numpy.ndarray",
                  interaction: "numpy.ndarray") -> "numpy.ndarray":
    """Return the relative-abundance states of a synthetic dynamic regime of n_traj trajectories with n_states states each: trajectory j (j = 1, ..., n_traj) starts from the absolute abundances base_s (1 + amplitude (2 h_{j,s} / 1000 - 1)) with the integer hash h_{j,s} = (1103 j + 617 s + 251 j s) mod 1001 (species s = 1, ..., S), follows the Ricker map of step 01 with growth, capacity and interaction for n_states states, and every state is divided by its total abundance. Rows are ordered by trajectory and, within a trajectory, by state.

    Parameters
    ----------
    n_traj : int
        Number of trajectories, at least 1.
    n_states : int
        Number of states per trajectory, at least 1.
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

    Returns
    -------
    states : numpy.ndarray
        Array of shape (n_traj * n_states, S) of relative abundances (each row sums to 1; float64).

    Raises
    ------
    ValueError
        If base is not a non-empty one-dimensional array of positive finite values, n_traj or n_states is not a positive integer, amplitude is outside [0, 1), or the Ricker inputs are invalid for step 01.
    """
    return states

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_regime_states(n_traj: int, n_states: int, base: "numpy.ndarray", amplitude: float,
                          growth: "numpy.ndarray", capacity: "numpy.ndarray",
                          interaction: "numpy.ndarray") -> "numpy.ndarray":
    """Relative-abundance states of the synthetic regime: trajectory j starts from base_s (1 + amplitude (2 h/1000 - 1)),
    h = (1103 j + 617 s + 251 j s) mod 1001, runs the Ricker map for n_states states; rows ordered by trajectory then state."""
    b = np.asarray(base, dtype=np.float64)
    if b.ndim != 1 or b.size < 1 or not np.all(np.isfinite(b)) or np.any(b <= 0.0):
        raise ValueError("base must be a non-empty 1-D array of positive finite values")
    if int(n_traj) != n_traj or n_traj < 1 or int(n_states) != n_states or n_states < 1:
        raise ValueError("n_traj and n_states must be positive integers")
    if not np.isfinite(amplitude) or not (0.0 <= amplitude < 1.0):
        raise ValueError("amplitude must lie in [0, 1)")
    S = b.size
    s_idx = np.arange(1, S + 1)
    rows = []
    for j in range(1, int(n_traj) + 1):
        h = (1103 * j + 617 * s_idx + 251 * j * s_idx) % 1001                 # integer hash, exact
        N0 = b * (1.0 + float(amplitude) * (2.0 * h / 1000.0 - 1.0))
        N = _oracle_ricker_trajectory(N0, growth, capacity, interaction, int(n_states))
        rows.append(N / N.sum(axis=1, keepdims=True))                        # relative abundances
    return np.vstack(rows)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 36, 8, 0.35\n",
            "call": "regime_states(n_traj, n_states, base, amplitude, r, K, A)",
            "gold_call": "_oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)",
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 12, 6, 0.2\n",
            "call": "regime_states(n_traj, n_states, base, amplitude, r, K, A)",
            "gold_call": "_oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)",
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 5, 10, 0.5\n",
            "call": "regime_states(n_traj, n_states, base, amplitude, r, K, A)",
            "gold_call": "_oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A)",
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\nn_traj, n_states, amplitude = 12, 6, 1.5\ndef run_model():\n    try:\n        regime_states(n_traj, n_states, base, amplitude, r, K, A)\n        return 0\n    except ValueError:\n        return 1\ndef _fx_safe(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "_fx_safe(lambda: _oracle_regime_states(n_traj, n_states, base, amplitude, r, K, A))",
        },
    ]
