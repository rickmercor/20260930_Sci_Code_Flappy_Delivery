"""
Step name: 06_compute_stopping_times
Step description: Tabulate, for every frame, the first later frame and the last earlier frame at which the trajectory occupies one of the two endpoint states.

Step scientific background: Committor and reactive-flux estimators are built from dynamics that are halted the moment the system reaches either endpoint state, because a trajectory that has arrived is committed and must not be propagated further. Those halting instants are the entry times into the union of the two endpoint states, tabulated forwards and backwards along the trajectory.

Returns
-------
np.ndarray: integer array of shape (T, 2) with the forward and backward endpoint entry times.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def compute_stopping_times(labels: np.ndarray) -> np.ndarray:
    """Return the forward and backward endpoint entry times of every frame.

    A frame is an endpoint frame when its label is ``0`` or ``1``. For frame
    ``t`` of a trajectory of ``T`` frames, column 0 of the output holds the
    smallest index ``t' >= t`` that is an endpoint frame and column 1 holds
    the largest index ``t' <= t`` that is an endpoint frame. An endpoint frame
    is therefore its own entry time in both columns.

    Conventions fixed by this step. When no endpoint frame exists at or after
    ``t`` the forward entry is ``T``; when none exists at or before ``t`` the
    backward entry is ``-1``. Both sentinels are chosen so that clamping a
    forward entry from above, or a backward entry from below, by any index
    inside the trajectory leaves that index unchanged.

    Parameters
    ----------
    labels : np.ndarray
        Integer array of shape ``(T,)`` with values in ``{0, 1, 2}`` and ``T``
        at least 1.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(T, 2)`` whose columns are the forward and
        backward endpoint entry times.

    Raises
    ------
    ValueError
        If ``labels`` is not a non-empty one-dimensional integer array whose
        values all lie in ``{0, 1, 2}``.
    """
    return stopping

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _validated_labels(labels):
    """Return the labels as a one-dimensional integer array of the right shape."""
    import numpy as np
    tags = np.asarray(labels)
    if tags.ndim != 1 or tags.shape[0] < 1 or not np.issubdtype(tags.dtype, np.integer):
        raise ValueError("labels must be a non-empty one-dimensional integer array")
    return tags.astype(np.int64)


import numpy as np
def _oracle_compute_stopping_times(labels: np.ndarray) -> np.ndarray:
    """Reference implementation (running minima and maxima of endpoint indices)."""
    import numpy as np

    tags = _validated_labels(labels)
    if not np.all(np.isin(tags, (0, 1, 2))):
        raise ValueError("labels must contain only 0, 1 and 2")
    n_frames = tags.shape[0]
    index = np.arange(n_frames, dtype=np.int64)
    endpoint = tags < 2

    forward = np.minimum.accumulate(np.where(endpoint, index, n_frames)[::-1])[::-1]
    backward = np.maximum.accumulate(np.where(endpoint, index, -1))
    return np.column_stack([forward, backward]).astype(np.int64)

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
        "lab = label_reaction_endpoints(simulate_surface_trajectory(6000, 5))\n"
        "lab_gold = _oracle_label_reaction_endpoints(_oracle_simulate_surface_trajectory(6000, 5))\n"
        "mixed = np.array([2, 2, 0, 2, 2, 2, 1, 2, 0, 2], dtype=np.int64)\n"
        "allmid = np.full(9, 2, dtype=np.int64)\n"
        "allend = np.array([0, 1, 0, 1], dtype=np.int64)\n"
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
    )
    return [
        {
            "setup": digest,
            "call": "_sig(compute_stopping_times(lab))",
            "gold_call": "_sig(_oracle_compute_stopping_times(lab_gold))",
        },
        {
            "setup": digest,
            "call": "_sig(compute_stopping_times(mixed))",
            "gold_call": "_sig(_oracle_compute_stopping_times(mixed))",
        },
        {
            "setup": digest,
            "call": "float(np.sum(compute_stopping_times(allmid)))",
            "gold_call": "float(np.sum(_oracle_compute_stopping_times(allmid)))",
        },
        {
            "setup": digest,
            "call": "float(np.sum(compute_stopping_times(allend)[:, 0] - compute_stopping_times(allend)[:, 1]))",
            "gold_call": "float(np.sum(_oracle_compute_stopping_times(allend)[:, 0] - _oracle_compute_stopping_times(allend)[:, 1]))",
        },
        {
            "setup": digest,
            "call": "float(compute_stopping_times(mixed)[1, 0] * 100.0 + compute_stopping_times(mixed)[7, 1])",
            "gold_call": "float(_oracle_compute_stopping_times(mixed)[1, 0] * 100.0 + _oracle_compute_stopping_times(mixed)[7, 1])",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_stopping_times(np.zeros(0, dtype=np.int64)))",
            "gold_call": "_status(lambda: _oracle_compute_stopping_times(np.zeros(0, dtype=np.int64)))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_stopping_times(np.array([0.0, 1.0, 2.0])))",
            "gold_call": "_status(lambda: _oracle_compute_stopping_times(np.array([0.0, 1.0, 2.0])))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_stopping_times(np.array([0, 5, 2], dtype=np.int64)))",
            "gold_call": "_status(lambda: _oracle_compute_stopping_times(np.array([0, 5, 2], dtype=np.int64)))",
        },
    ]
