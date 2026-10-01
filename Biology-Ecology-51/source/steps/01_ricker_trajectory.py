"""
Return the n_states successive community states of the discrete-time Ricker-Lotka-Volterra map N_{t+1,s} = N_{t,s} exp(r_s (1 - (A N_t)_s / K_s)) started from the absolute abundances initial (row 0 of the result is initial itself, row t is the state after t applications of the map).

A discrete-time Ricker-type Lotka-Volterra community model is a standard way to generate successional trajectories of species abundances from an initial community; with growth rates decreasing and carrying capacities increasing along a pioneer-to-late-successional ordering, and later species suppressing earlier ones more than the reverse, every trajectory moves from a pioneer-dominated to a late-successional composition.

Returns
-------
numpy.ndarray of float64 with shape (n_states, S): the successive absolute abundances.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ricker_trajectory(initial: "numpy.ndarray", growth: "numpy.ndarray", capacity: "numpy.ndarray",
                      interaction: "numpy.ndarray", n_states: int) -> "numpy.ndarray":
    """Return the n_states successive community states of the discrete-time Ricker-Lotka-Volterra map N_{t+1,s} = N_{t,s} exp(r_s (1 - (A N_t)_s / K_s)) started from the absolute abundances initial (row 0 of the result is initial itself, row t is the state after t applications of the map).

    Parameters
    ----------
    initial : numpy.ndarray
        One-dimensional array of the S nonnegative initial abundances.
    growth : numpy.ndarray
        One-dimensional array of the S intrinsic growth rates r_s.
    capacity : numpy.ndarray
        One-dimensional array of the S positive carrying capacities K_s.
    interaction : numpy.ndarray
        Array of shape (S, S) of interaction coefficients A[s, u], the effect of species u on species s.
    n_states : int
        Number of states returned, at least 1.

    Returns
    -------
    trajectory : numpy.ndarray
        Array of shape (n_states, S) of absolute abundances (float64).

    Raises
    ------
    ValueError
        If initial is not a non-empty one-dimensional array, growth, capacity or interaction do not match its length, any input is not finite, an abundance is negative, a capacity is not positive, or n_states is not a positive integer.
    """
    return trajectory

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ricker_trajectory(initial: "numpy.ndarray", growth: "numpy.ndarray", capacity: "numpy.ndarray",
                              interaction: "numpy.ndarray", n_states: int) -> "numpy.ndarray":
    """Discrete-time Ricker-Lotka-Volterra map N_{t+1,s} = N_{t,s} exp(r_s (1 - (A N_t)_s / K_s)); row t = state t+1."""
    N0 = np.asarray(initial, dtype=np.float64); r = np.asarray(growth, dtype=np.float64)
    K = np.asarray(capacity, dtype=np.float64); A = np.asarray(interaction, dtype=np.float64)
    if N0.ndim != 1 or N0.size < 1:
        raise ValueError("initial must be a non-empty 1-D array")
    S = N0.size
    if r.shape != (S,) or K.shape != (S,) or A.shape != (S, S):
        raise ValueError("growth and capacity must have the length of initial and interaction must be square of that size")
    if not (np.all(np.isfinite(N0)) and np.all(np.isfinite(r)) and np.all(np.isfinite(K)) and np.all(np.isfinite(A))):
        raise ValueError("all inputs must be finite")
    if np.any(N0 < 0.0) or np.any(K <= 0.0):
        raise ValueError("initial abundances must be nonnegative and capacities positive")
    if int(n_states) != n_states or n_states < 1:
        raise ValueError("n_states must be a positive integer")
    out = np.empty((int(n_states), S))
    N = N0.copy()
    out[0] = N
    for t in range(1, int(n_states)):
        N = N * np.exp(r * (1.0 - (A @ N) / K))
        out[t] = N
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\ninitial = base.copy()\nn_states = 8\n",
            "call": "ricker_trajectory(initial, r, K, A, n_states)",
            "gold_call": "_oracle_ricker_trajectory(initial, r, K, A, n_states)",
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\ninitial = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])\nn_states = 12\n",
            "call": "ricker_trajectory(initial, r, K, A, n_states)",
            "gold_call": "_oracle_ricker_trajectory(initial, r, K, A, n_states)",
        },
        {
            "setup": "import numpy as np\ninitial = np.array([2.0, 1.0])\ngrowth = np.array([0.4, 0.3])\ncapacity = np.array([10.0, 20.0])\ninteraction = np.array([[1.0, 0.5], [0.1, 1.0]])\nn_states = 5\n",
            "call": "ricker_trajectory(initial, growth, capacity, interaction, n_states)",
            "gold_call": "_oracle_ricker_trajectory(initial, growth, capacity, interaction, n_states)",
        },
        {
            "setup": "import numpy as np\nr = np.array([0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20])\nK = np.array([20.0, 25.0, 32.0, 40.0, 50.0, 63.0, 80.0, 100.0])\nbase = np.array([12.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0])\nA = np.where(np.arange(8)[:, None] < np.arange(8)[None, :], 0.5, np.where(np.arange(8)[:, None] > np.arange(8)[None, :], 0.1, 1.0))\ninitial = base.copy()\nn_states = 0\ndef run_model():\n    try:\n        ricker_trajectory(initial, r, K, A, n_states)\n        return 0\n    except ValueError:\n        return 1\ndef _fx_safe(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "_fx_safe(lambda: _oracle_ricker_trajectory(initial, r, K, A, n_states))",
        },
    ]
