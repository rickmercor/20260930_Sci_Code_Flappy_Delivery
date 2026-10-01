"""
Step name: 09_estimate_reactive_flux
Step description: Accumulate the net reactive flux per frame from the committor and the endpoint entry times over lagged trajectory windows.

Step scientific background: The reactive current between two states is the mean rate at which trajectories carry probability across the transition region weighted by the forward and backward commitment probabilities. Evaluating the weights at the instants the dynamics are halted, rather than at a fixed lag, removes the dependence of the estimate on the lag that is chosen.

Returns
-------
float, the net reactive flux per frame as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def estimate_reactive_flux(labels: np.ndarray, committor: np.ndarray,
                           stopping: np.ndarray, lag_frames: int = 20,
                           block_frames: int = 20000) -> float:
    """Return the net reactive flux from the atomic to the molecular state.

    The estimate is the average, over all lagged windows of the trajectory, of
    the flux carried by the single-frame increments inside a window, with the
    committor of each increment's endpoints evaluated where the halted
    dynamics stop. The backward committor is obtained from the forward one
    under detailed balance rather than estimated separately.

    Conventions fixed by this step. Windows are indexed by their first frame
    ``t`` running from ``0`` to ``T - lag_frames - 1`` inclusive, and every
    window contributes the increments from its own frames ``t`` to
    ``t + lag_frames - 1``, each increment running to the next frame. Halting
    instants come from the two columns of ``stopping`` and are clamped to the
    window that contains them, so no increment ever refers to a frame outside
    its own window. The two time directions are averaged with equal weight,
    the sum over increments inside a window is divided by ``lag_frames``, and
    the sum over windows is divided by the number of windows, so the result is
    a flux per frame and not per window. ``block_frames`` only sets how many
    windows are accumulated at a time and must not change the result beyond
    floating-point summation order.

    Parameters
    ----------
    labels : np.ndarray
        Integer array of shape ``(T,)`` with values in ``{0, 1, 2}``.
    committor : np.ndarray
        Float array of shape ``(T,)`` of forward committor values.
    stopping : np.ndarray
        Integer array of shape ``(T, 2)`` of forward and backward endpoint
        entry times.
    lag_frames : int
        Lag of the halted dynamics in frames, at least 1 and smaller than
        ``T``.
    block_frames : int
        Number of windows accumulated per block, at least 1.

    Returns
    -------
    float
        Net reactive flux per frame.

    Raises
    ------
    ValueError
        If the three arrays do not have matching length and the stated shapes
        and dtypes, if a label lies outside ``{0, 1, 2}``, if ``committor`` is
        not finite, if ``lag_frames`` is not an integer in ``1 .. T - 1``, or
        if ``block_frames`` is not an integer of at least 1.
    """
    return flux

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _window_flux(forward, backward, stops, starts, lag):
    """Return the summed flux of one block of windows in both time directions."""
    import numpy as np
    offsets = np.arange(lag, dtype=np.int64)[None, :]
    origin = starts[:, None]
    steps = origin + offsets
    left = np.maximum(origin, stops[steps, 1])
    right = np.minimum(origin + lag, stops[steps + 1, 0])
    total = float(np.sum(backward[left] * (forward[steps + 1] - forward[steps]) * forward[right]))
    total += float(np.sum(forward[left] * (backward[steps + 1] - backward[steps]) * backward[right]))
    return total


import numpy as np
def _oracle_estimate_reactive_flux(labels: np.ndarray, committor: np.ndarray,
                                   stopping: np.ndarray, lag_frames: int = 20,
                                   block_frames: int = 20000) -> float:
    """Reference implementation (halted rolling-window reactive current)."""
    import numpy as np

    tags = _validated_labels(labels)
    if not np.all(np.isin(tags, (0, 1, 2))):
        raise ValueError("labels must contain only 0, 1 and 2")
    n_frames = tags.shape[0]
    forward = np.asarray(committor, dtype=float)
    if forward.ndim != 1 or forward.shape[0] != n_frames or not np.all(np.isfinite(forward)):
        raise ValueError("committor must be a finite one-dimensional array matching labels")
    stops = np.asarray(stopping)
    if stops.ndim != 2 or stops.shape != (n_frames, 2) or not np.issubdtype(stops.dtype, np.integer):
        raise ValueError("stopping must be an integer array of shape (T, 2)")
    if not (_is_integer(lag_frames) and 1 <= int(lag_frames) < n_frames):
        raise ValueError("lag_frames must be an integer in 1 .. T - 1")
    if not (_is_integer(block_frames) and int(block_frames) >= 1):
        raise ValueError("block_frames must be an integer of at least 1")

    lag = int(lag_frames)
    block = int(block_frames)
    stops = stops.astype(np.int64)
    backward = 1.0 - forward
    n_windows = n_frames - lag

    total = 0.0
    for low in range(0, n_windows, block):
        starts = np.arange(low, min(low + block, n_windows), dtype=np.int64)
        total += _window_flux(forward, backward, stops, starts, lag)
    return float(total / (2.0 * n_windows * lag))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    setup = (
        "import numpy as np\n"
        "traj = simulate_surface_trajectory(12000, 101, kt_ev=0.07)\n"
        "lab = label_reaction_endpoints(traj)\n"
        "comp = project_dynamical_components(build_local_descriptors(traj))\n"
        "mic = assign_intermediate_microstates(comp, lab, n_microstates=24)\n"
        "stp = compute_stopping_times(lab)\n"
        "sys20 = build_galerkin_system(lab, mic, stp, lag_frames=20, n_microstates=24)\n"
        "qp = solve_committor(lab, mic, sys20)\n"
        "traj_gold = _oracle_simulate_surface_trajectory(12000, 101, kt_ev=0.07)\n"
        "lab_gold = _oracle_label_reaction_endpoints(traj_gold)\n"
        "comp_gold = _oracle_project_dynamical_components(_oracle_build_local_descriptors(traj_gold))\n"
        "mic_gold = _oracle_assign_intermediate_microstates(comp_gold, lab_gold, n_microstates=24)\n"
        "stp_gold = _oracle_compute_stopping_times(lab_gold)\n"
        "sys20_gold = _oracle_build_galerkin_system(lab_gold, mic_gold, stp_gold, lag_frames=20, n_microstates=24)\n"
        "qp_gold = _oracle_solve_committor(lab_gold, mic_gold, sys20_gold)\n"
        "tiny = np.array([0, 2, 2, 1, 2, 0, 2, 2, 1, 2, 2, 0], dtype=np.int64)\n"
        "tstp = compute_stopping_times(tiny)\n"
        "tstp_gold = _oracle_compute_stopping_times(tiny)\n"
        "tq = np.array([0.0, 0.3, 0.7, 1.0, 0.6, 0.0, 0.2, 0.8, 1.0, 0.9, 0.4, 0.0])\n"
        "flat = np.full(12, 0.5)\n"
    )
    status = (
        "import numpy as np\n"
        "tiny = np.array([0, 2, 2, 1, 2, 0, 2, 2, 1, 2, 2, 0], dtype=np.int64)\n"
        "tstp = compute_stopping_times(tiny)\n"
        "tstp_gold = _oracle_compute_stopping_times(tiny)\n"
        "tq = np.array([0.0, 0.3, 0.7, 1.0, 0.6, 0.0, 0.2, 0.8, 1.0, 0.9, 0.4, 0.0])\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        {
            "setup": setup,
            "call": "estimate_reactive_flux(lab, qp, stp, lag_frames=20)",
            "gold_call": "_oracle_estimate_reactive_flux(lab_gold, qp_gold, stp_gold, lag_frames=20)",
        },
        {
            "setup": setup,
            "call": "estimate_reactive_flux(lab, qp, stp, lag_frames=3, block_frames=997)",
            "gold_call": "_oracle_estimate_reactive_flux(lab_gold, qp_gold, stp_gold, lag_frames=3, block_frames=997)",
        },
        {
            "setup": setup,
            "call": "estimate_reactive_flux(tiny, tq, tstp, lag_frames=2)",
            "gold_call": "_oracle_estimate_reactive_flux(tiny, tq, tstp_gold, lag_frames=2)",
        },
        {
            "setup": setup,
            "call": "estimate_reactive_flux(tiny, tq, tstp, lag_frames=1)",
            "gold_call": "_oracle_estimate_reactive_flux(tiny, tq, tstp_gold, lag_frames=1)",
        },
        {
            "setup": setup,
            "call": "estimate_reactive_flux(tiny, flat, tstp, lag_frames=4)",
            "gold_call": "_oracle_estimate_reactive_flux(tiny, flat, tstp_gold, lag_frames=4)",
        },
        {
            "setup": setup,
            "call": "float(estimate_reactive_flux(lab, qp, stp, lag_frames=50) * 1000.0)",
            "gold_call": "float(_oracle_estimate_reactive_flux(lab_gold, qp_gold, stp_gold, lag_frames=50) * 1000.0)",
        },
        {
            "setup": status,
            "call": "_status(lambda: estimate_reactive_flux(tiny, tq, tstp, lag_frames=12))",
            "gold_call": "_status(lambda: _oracle_estimate_reactive_flux(tiny, tq, tstp_gold, lag_frames=12))",
        },
        {
            "setup": status,
            "call": "_status(lambda: estimate_reactive_flux(tiny, tq[:5], tstp, lag_frames=2))",
            "gold_call": "_status(lambda: _oracle_estimate_reactive_flux(tiny, tq[:5], tstp_gold, lag_frames=2))",
        },
        {
            "setup": status,
            "call": "_status(lambda: estimate_reactive_flux(tiny, tq, tstp, lag_frames=2, block_frames=0))",
            "gold_call": "_status(lambda: _oracle_estimate_reactive_flux(tiny, tq, tstp_gold, lag_frames=2, block_frames=0))",
        },
    ]
