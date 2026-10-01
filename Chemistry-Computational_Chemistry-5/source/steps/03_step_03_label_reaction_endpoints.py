"""
Step name: 03_label_reaction_endpoints
Step description: Assign every frame to the atomic reactant state, the molecular product state or the intermediate region using time-smoothed hydrogen-hydrogen distances.

Step scientific background: Transition-path quantities require the two endpoint states of the reaction to be defined on the configurations themselves. In a hot, crowded adsorbate layer the instantaneous bond length rattles across any threshold, so the endpoint test is applied to a short time average, and configurations whose central atom is flanked by more than one close neighbour are excluded from both endpoints.

Returns
-------
np.ndarray: integer label array of shape (T,) with 0 atomic, 1 molecular, 2 intermediate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def label_reaction_endpoints(trajectory: np.ndarray, window_frames: int = 5,
                             r_atomic: float = 1.5, r_molecular: float = 1.0) -> np.ndarray:
    """Return the endpoint label of every frame of a trajectory.

    Each of the two hydrogen-hydrogen distances is first replaced by a centred
    moving average over ``window_frames`` frames; near the two ends of the
    trajectory the window is truncated symmetrically, so frame ``t`` averages
    the frames from ``max(t - h, 0)`` to ``min(t + h, T - 1)`` inclusive with
    ``h = window_frames // 2``. Labels are then assigned from the smoothed
    distances alone.

    Labels are ``0`` for the atomic reactant state ``A``, ``1`` for the
    molecular product state ``B``, and ``2`` for the intermediate region that
    belongs to neither. A frame whose smoothed distances to both neighbours
    are below ``r_atomic`` is intermediate whatever else holds; otherwise the
    frame is molecular when its smallest smoothed distance is below
    ``r_molecular``, atomic when that distance exceeds ``r_atomic``, and
    intermediate in between.

    Parameters
    ----------
    trajectory : np.ndarray
        Float array of shape ``(T, 3)`` with rows ``(r1, r2, s)`` and ``T``
        at least 1.
    window_frames : int
        Odd positive number of frames in the centred smoothing window.
    r_atomic : float
        Smoothed separation in angstrom above which a frame is atomic.
    r_molecular : float
        Smoothed separation in angstrom below which a frame is molecular.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(T,)`` with values in ``{0, 1, 2}``.

    Raises
    ------
    ValueError
        If ``trajectory`` is not a finite non-empty array of shape ``(T, 3)``,
        if ``window_frames`` is not an odd integer of at least 1, if either
        radius is not a positive finite float, or if ``r_molecular`` is not
        strictly smaller than ``r_atomic``.
    """
    return labels

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _centred_moving_average(values, window_frames: int):
    """Return the centred moving average of each column, truncated at the ends."""
    import numpy as np
    n_rows = values.shape[0]
    half = int(window_frames) // 2
    cumulative = np.concatenate([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    index = np.arange(n_rows)
    low = np.maximum(index - half, 0)
    high = np.minimum(index + half + 1, n_rows)
    return (cumulative[high] - cumulative[low]) / (high - low)[:, None]


import numpy as np
def _oracle_label_reaction_endpoints(trajectory: np.ndarray, window_frames: int = 5,
                                     r_atomic: float = 1.5, r_molecular: float = 1.0) -> np.ndarray:
    """Reference implementation (smoothed thresholds with the crowding exclusion)."""
    import numpy as np

    if not (_is_integer(window_frames) and int(window_frames) >= 1 and int(window_frames) % 2 == 1):
        raise ValueError("window_frames must be an odd integer of at least 1")
    if not (_is_positive_float(r_atomic) and _is_positive_float(r_molecular)):
        raise ValueError("r_atomic and r_molecular must be positive finite floats")
    if not float(r_molecular) < float(r_atomic):
        raise ValueError("r_molecular must be strictly smaller than r_atomic")
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim != 2 or traj.shape[1] != 3 or traj.shape[0] < 1:
        raise ValueError("trajectory must be a non-empty array of shape (T, 3)")
    if not np.all(np.isfinite(traj)):
        raise ValueError("trajectory must be finite")

    smoothed = _centred_moving_average(traj[:, :2], int(window_frames))
    nearest = smoothed.min(axis=1)
    crowded = np.all(smoothed < float(r_atomic), axis=1)

    labels = np.full(traj.shape[0], 2, dtype=np.int64)
    labels[(~crowded) & (nearest < float(r_molecular))] = 1
    labels[(~crowded) & (nearest > float(r_atomic))] = 0
    return labels

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    digest = (
        "import numpy as np\n"
        "def _sig(a):\n"
        "    f = np.asarray(a, dtype=float).ravel()\n"
        "    if f.size == 0:\n"
        "        return -1.0\n"
        "    w = np.cos(np.arange(f.size, dtype=float) + 1.0)\n"
        "    return float(np.dot(f, w) + np.abs(f).mean() + 1000.0 * f[0])\n"
        "traj = simulate_surface_trajectory(6000, 5)\n"
        "traj_gold = _oracle_simulate_surface_trajectory(6000, 5)\n"
        "ramp = np.column_stack([np.linspace(0.7, 2.4, 60),\n"
        "                        np.linspace(2.4, 0.7, 60),\n"
        "                        np.zeros(60)])\n"
        "flat = np.array([[1.20, 1.30, 0.0], [0.95, 2.80, 1.0], [2.90, 3.10, 2.0]])\n"
    )
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "traj = np.array([[2.0, 2.0, 0.0], [2.0, 2.0, 0.1]])\n"
    )
    return [
        {
            "setup": digest,
            "call": "_sig(label_reaction_endpoints(traj))",
            "gold_call": "_sig(_oracle_label_reaction_endpoints(traj_gold))",
        },
        {
            "setup": digest,
            "call": "float(np.mean(label_reaction_endpoints(traj) == 2) + 3.0 * np.mean(label_reaction_endpoints(traj) == 1))",
            "gold_call": "float(np.mean(_oracle_label_reaction_endpoints(traj_gold) == 2) + 3.0 * np.mean(_oracle_label_reaction_endpoints(traj_gold) == 1))",
        },
        {
            "setup": digest,
            "call": "_sig(label_reaction_endpoints(ramp, window_frames=1))",
            "gold_call": "_sig(_oracle_label_reaction_endpoints(ramp, window_frames=1))",
        },
        {
            "setup": digest,
            "call": "_sig(label_reaction_endpoints(ramp, window_frames=11))",
            "gold_call": "_sig(_oracle_label_reaction_endpoints(ramp, window_frames=11))",
        },
        {
            "setup": digest,
            "call": "float(np.dot(label_reaction_endpoints(flat), np.array([1.0, 10.0, 100.0])))",
            "gold_call": "float(np.dot(_oracle_label_reaction_endpoints(flat), np.array([1.0, 10.0, 100.0])))",
        },
        {
            "setup": digest,
            "call": "_sig(label_reaction_endpoints(traj, window_frames=5, r_atomic=1.8, r_molecular=0.9))",
            "gold_call": "_sig(_oracle_label_reaction_endpoints(traj_gold, window_frames=5, r_atomic=1.8, r_molecular=0.9))",
        },
        {
            "setup": status,
            "call": "_status(lambda: label_reaction_endpoints(traj, window_frames=4))",
            "gold_call": "_status(lambda: _oracle_label_reaction_endpoints(traj, window_frames=4))",
        },
        {
            "setup": status,
            "call": "_status(lambda: label_reaction_endpoints(traj, r_atomic=1.0, r_molecular=1.5))",
            "gold_call": "_status(lambda: _oracle_label_reaction_endpoints(traj, r_atomic=1.0, r_molecular=1.5))",
        },
        {
            "setup": status,
            "call": "_status(lambda: label_reaction_endpoints(np.zeros((0, 3))))",
            "gold_call": "_status(lambda: _oracle_label_reaction_endpoints(np.zeros((0, 3))))",
        },
    ]
