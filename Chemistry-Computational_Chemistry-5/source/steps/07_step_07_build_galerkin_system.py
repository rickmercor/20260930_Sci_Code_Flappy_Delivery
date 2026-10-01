"""
Step name: 07_build_galerkin_system
Step description: Accumulate the linear system whose solution gives the expansion coefficients of the committor in the basis of intermediate-microstate indicator functions.

Step scientific background: Projecting the committor equation onto a finite basis turns a boundary value problem for the halted dynamics into a small linear system whose matrix elements are correlation functions of basis indicators over lagged trajectory windows. Averaging each window over the trajectory and its time reversal imposes detailed balance on the estimate.

Returns
-------
np.ndarray: float array of shape (n_microstates, n_microstates + 1), the Galerkin matrix beside its reference vector.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def build_galerkin_system(labels: np.ndarray, microstates: np.ndarray,
                          stopping: np.ndarray, lag_frames: int = 20,
                          n_microstates: int = 64) -> np.ndarray:
    """Return the augmented Galerkin system of the halted committor problem.

    The trial basis is the set of indicator functions of the intermediate
    microstates, and the reference function is the indicator of the molecular
    product state. Entry ``(i, j)`` of the returned matrix, for ``j`` below
    ``n_microstates``, is the trajectory estimate of the correlation of
    indicator ``i`` with the change of indicator ``j`` produced by the halted
    dynamics over ``lag_frames``; the final column is the same estimate taken
    against the product-state indicator instead of a basis indicator.

    Conventions fixed by this step. Windows are indexed by their first frame
    ``t`` running from ``0`` to ``T - lag_frames - 1`` inclusive, so there are
    ``T - lag_frames`` of them. Every window contributes twice, once read
    forwards from ``t`` and once read backwards from ``t + lag_frames``, and
    the accumulated total is divided by twice the number of windows. A frame
    that is not intermediate carries no basis indicator and so contributes no
    row. The halting instants must be taken from ``stopping``, whose two
    columns are the forward and backward endpoint entry times, and must be
    clamped to the window being read.

    Parameters
    ----------
    labels : np.ndarray
        Integer array of shape ``(T,)`` with values in ``{0, 1, 2}``.
    microstates : np.ndarray
        Integer array of shape ``(T,)`` holding a microstate index in
        ``0 .. n_microstates - 1`` on intermediate frames and ``-1``
        elsewhere.
    stopping : np.ndarray
        Integer array of shape ``(T, 2)`` of forward and backward endpoint
        entry times.
    lag_frames : int
        Lag of the halted dynamics in frames, at least 1 and smaller than
        ``T``.
    n_microstates : int
        Size of the basis, at least 1.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_microstates, n_microstates + 1)`` holding
        the Galerkin matrix in its leading columns and the reference vector
        in its final column.

    Raises
    ------
    ValueError
        If ``labels``, ``microstates`` and ``stopping`` are not integer arrays
        of matching length and the stated shapes, if a label lies outside
        ``{0, 1, 2}``, if an intermediate frame carries a microstate index
        outside ``0 .. n_microstates - 1`` or a non-intermediate frame carries
        one that is not ``-1``, if ``lag_frames`` is not an integer in
        ``1 .. T - 1``, or if ``n_microstates`` is not an integer of at least
        1.
    """
    return system

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _check_state_arrays(labels, microstates, stopping, n_microstates):
    """Return the validated label, microstate and stopping arrays."""
    import numpy as np
    tags = _validated_labels(labels)
    if not np.all(np.isin(tags, (0, 1, 2))):
        raise ValueError("labels must contain only 0, 1 and 2")
    cells = np.asarray(microstates)
    stops = np.asarray(stopping)
    if cells.ndim != 1 or cells.shape[0] != tags.shape[0] or not np.issubdtype(cells.dtype, np.integer):
        raise ValueError("microstates must be an integer array matching labels")
    if stops.ndim != 2 or stops.shape != (tags.shape[0], 2) or not np.issubdtype(stops.dtype, np.integer):
        raise ValueError("stopping must be an integer array of shape (T, 2)")
    if not (_is_integer(n_microstates) and int(n_microstates) >= 1):
        raise ValueError("n_microstates must be an integer of at least 1")
    interior = tags == 2
    if np.any(cells[~interior] != -1):
        raise ValueError("only intermediate frames may carry a microstate index")
    if interior.any():
        inside = cells[interior]
        if inside.min() < 0 or inside.max() >= int(n_microstates):
            raise ValueError("microstate indices must lie in 0 .. n_microstates - 1")
    return tags, cells.astype(np.int64), stops.astype(np.int64)


def _accumulate_half(system, rows, cols, tags_at_cols, n_microstates):
    """Add one directional half of the Galerkin estimate in place."""
    import numpy as np
    stays = tags_at_cols == 2
    np.add.at(system, (rows[stays], cols[stays]), 1.0)
    reaches = tags_at_cols == 1
    np.add.at(system[:, n_microstates], rows[reaches], 1.0)
    np.add.at(system, (rows, rows), -1.0)


import numpy as np
def _oracle_build_galerkin_system(labels: np.ndarray, microstates: np.ndarray,
                                  stopping: np.ndarray, lag_frames: int = 20,
                                  n_microstates: int = 64) -> np.ndarray:
    """Reference implementation (forward and time-reversed halted windows)."""
    import numpy as np

    tags, cells, stops = _check_state_arrays(labels, microstates, stopping, n_microstates)
    n_frames = tags.shape[0]
    if not (_is_integer(lag_frames) and 1 <= int(lag_frames) < n_frames):
        raise ValueError("lag_frames must be an integer in 1 .. T - 1")

    lag = int(lag_frames)
    size = int(n_microstates)
    start = np.arange(n_frames - lag, dtype=np.int64)
    system = np.zeros((size, size + 1), dtype=float)

    forward = start[tags[start] == 2]
    reached = np.minimum(forward + lag, stops[forward, 0])
    _accumulate_half(system, cells[forward], cells[reached], tags[reached], size)

    backward = (start + lag)[tags[start + lag] == 2]
    left = np.maximum(backward - lag, stops[backward, 1])
    _accumulate_half(system, cells[backward], cells[left], tags[left], size)

    return system / (2.0 * (n_frames - lag))

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
        "traj = simulate_surface_trajectory(12000, 101, kt_ev=0.07)\n"
        "lab = label_reaction_endpoints(traj)\n"
        "comp = project_dynamical_components(build_local_descriptors(traj))\n"
        "mic = assign_intermediate_microstates(comp, lab, n_microstates=24)\n"
        "stp = compute_stopping_times(lab)\n"
        "traj_gold = _oracle_simulate_surface_trajectory(12000, 101, kt_ev=0.07)\n"
        "lab_gold = _oracle_label_reaction_endpoints(traj_gold)\n"
        "comp_gold = _oracle_project_dynamical_components(_oracle_build_local_descriptors(traj_gold))\n"
        "mic_gold = _oracle_assign_intermediate_microstates(comp_gold, lab_gold, n_microstates=24)\n"
        "stp_gold = _oracle_compute_stopping_times(lab_gold)\n"
        "tiny = np.array([0, 2, 2, 1, 2, 0, 2, 2, 1, 2, 2, 0], dtype=np.int64)\n"
        "tmic = np.where(tiny == 2, np.arange(12) % 3, -1).astype(np.int64)\n"
        "tstp = compute_stopping_times(tiny)\n"
        "tstp_gold = _oracle_compute_stopping_times(tiny)\n"
    )
    status = (
        "import numpy as np\n"
        "tiny = np.array([0, 2, 2, 1, 2, 0, 2, 2, 1, 2, 2, 0], dtype=np.int64)\n"
        "tmic = np.where(tiny == 2, np.arange(12) % 3, -1).astype(np.int64)\n"
        "tstp = compute_stopping_times(tiny)\n"
        "tstp_gold = _oracle_compute_stopping_times(tiny)\n"
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
            "setup": digest,
            "call": "_sig(build_galerkin_system(lab, mic, stp, lag_frames=20, n_microstates=24))",
            "gold_call": "_sig(_oracle_build_galerkin_system(lab_gold, mic_gold, stp_gold, lag_frames=20, n_microstates=24))",
        },
        {
            "setup": digest,
            "call": "float(np.trace(build_galerkin_system(lab, mic, stp, lag_frames=5, n_microstates=24)[:, :24]))",
            "gold_call": "float(np.trace(_oracle_build_galerkin_system(lab_gold, mic_gold, stp_gold, lag_frames=5, n_microstates=24)[:, :24]))",
        },
        {
            "setup": digest,
            "call": "float(np.sum(build_galerkin_system(lab, mic, stp, lag_frames=60, n_microstates=24)[:, 24]))",
            "gold_call": "float(np.sum(_oracle_build_galerkin_system(lab_gold, mic_gold, stp_gold, lag_frames=60, n_microstates=24)[:, 24]))",
        },
        {
            "setup": digest,
            "call": "_sig(build_galerkin_system(tiny, tmic, tstp, lag_frames=2, n_microstates=3))",
            "gold_call": "_sig(_oracle_build_galerkin_system(tiny, tmic, tstp_gold, lag_frames=2, n_microstates=3))",
        },
        {
            "setup": digest,
            "call": "_sig(build_galerkin_system(tiny, tmic, tstp, lag_frames=1, n_microstates=3))",
            "gold_call": "_sig(_oracle_build_galerkin_system(tiny, tmic, tstp_gold, lag_frames=1, n_microstates=3))",
        },
        {
            "setup": digest,
            "call": "float(np.linalg.cond(build_galerkin_system(lab, mic, stp, lag_frames=20, n_microstates=24)[:, :24]))",
            "gold_call": "float(np.linalg.cond(_oracle_build_galerkin_system(lab_gold, mic_gold, stp_gold, lag_frames=20, n_microstates=24)[:, :24]))",
        },
        {
            "setup": status,
            "call": "_status(lambda: build_galerkin_system(tiny, tmic, tstp, lag_frames=0, n_microstates=3))",
            "gold_call": "_status(lambda: _oracle_build_galerkin_system(tiny, tmic, tstp_gold, lag_frames=0, n_microstates=3))",
        },
        {
            "setup": status,
            "call": "_status(lambda: build_galerkin_system(tiny, tmic, tstp, lag_frames=2, n_microstates=2))",
            "gold_call": "_status(lambda: _oracle_build_galerkin_system(tiny, tmic, tstp_gold, lag_frames=2, n_microstates=2))",
        },
        {
            "setup": status,
            "call": "_status(lambda: build_galerkin_system(tiny, tmic, tstp[:5], lag_frames=2, n_microstates=3))",
            "gold_call": "_status(lambda: _oracle_build_galerkin_system(tiny, tmic, tstp_gold[:5], lag_frames=2, n_microstates=3))",
        },
        {
            "setup": status,
            "call": "_status(lambda: build_galerkin_system(tiny, np.zeros(12, dtype=np.int64), tstp, lag_frames=2, n_microstates=3))",
            "gold_call": "_status(lambda: _oracle_build_galerkin_system(tiny, np.zeros(12, dtype=np.int64), tstp_gold, lag_frames=2, n_microstates=3))",
        },
    ]
