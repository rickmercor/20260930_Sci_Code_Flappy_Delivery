"""
Summarise per-cell values at the level of spatial neighbourhoods by averaging each cell's value with those of its neighbours.

Single-cell signalling estimates are noisy, so they are reported for the small neighbourhood around each cell rather than for the cell alone.

Returns
-------
np.ndarray: neighbourhood means with the shape of values.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def average_over_neighbourhoods(values: "np.ndarray", neighbourhoods: "np.ndarray") -> "np.ndarray":
    """Return the mean of per-cell values over every cell's neighbourhood.

    ``neighbourhoods[i, k]`` is 1.0 when cell ``k`` belongs to the
    neighbourhood of cell ``i`` and 0.0 otherwise; every cell belongs to its
    own neighbourhood and membership is symmetric. Entry ``i`` of the result
    (row ``i`` when ``values`` is two-dimensional) is the arithmetic mean of
    ``values`` over the members of cell ``i``'s neighbourhood.

    Parameters
    ----------
    values : np.ndarray
        Finite float array of shape ``(n,)`` or ``(n, d)``.
    neighbourhoods : np.ndarray
        Float array of shape ``(n, n)`` holding only 0.0 and 1.0, symmetric,
        with ones on the diagonal.

    Returns
    -------
    np.ndarray
        Float array with the shape of ``values``.

    Raises
    ------
    ValueError
        If ``neighbourhoods`` is not a numeric ``(n, n)`` array of zeros and
        ones that is symmetric with ones on the diagonal, or if ``values`` is
        not a finite numeric array of shape ``(n,)`` or ``(n, d)``.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_average_over_neighbourhoods(values: "np.ndarray", neighbourhoods: "np.ndarray") -> "np.ndarray":
    """Reference implementation (row-normalised membership matrix)."""
    import numpy as np

    try:
        members = np.array(neighbourhoods, dtype=float)
        data = np.array(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("values and neighbourhoods must be numeric arrays") from None
    if members.ndim != 2 or members.shape[0] != members.shape[1] or members.shape[0] < 1:
        raise ValueError("neighbourhoods must have shape (n, n) with n >= 1")
    if not np.all((members == 0.0) | (members == 1.0)):
        raise ValueError("neighbourhoods must hold only zeros and ones")
    if not (np.all(np.diag(members) == 1.0) and np.array_equal(members, members.T)):
        raise ValueError("neighbourhoods must be symmetric with ones on the diagonal")
    n = members.shape[0]
    if data.ndim not in (1, 2) or data.shape[0] != n:
        raise ValueError("values must have shape (n,) or (n, d)")
    if not np.all(np.isfinite(data)):
        raise ValueError("values must be finite")
    means = members / members.sum(axis=1, keepdims=True)
    return means @ data

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    head = (
        "import numpy as np\n"
        "def _sig(a, shape):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != shape:\n"
        "        return -1.0\n"
        "    k = np.arange(a.size, dtype=float)\n"
        "    return float(np.sum(np.abs(a)) + np.sum(a.ravel() * np.cos(0.7 * k)))\n"
    )
    status = (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    graph = (
        "A = np.array([[1, 1, 1, 0, 0],\n"
        "              [1, 1, 1, 1, 0],\n"
        "              [1, 1, 1, 1, 1],\n"
        "              [0, 1, 1, 1, 1],\n"
        "              [0, 0, 1, 1, 1]], dtype=float)\n"
        "v = np.array([2.4, -1.1, 0.35, 3.8, -0.6])\n"
    )
    return [
        {
            "setup": head + graph,
            "call": "_sig(average_over_neighbourhoods(v, A), (5,))",
            "gold_call": "_sig(_oracle_average_over_neighbourhoods(v, A), (5,))",
        },
        {
            "setup": head + graph,
            "call": "_sig(average_over_neighbourhoods(np.column_stack([v, v ** 2]), A), (5, 2))",
            "gold_call": "_sig(_oracle_average_over_neighbourhoods(np.column_stack([v, v ** 2]), A), (5, 2))",
        },
        {
            "setup": head + graph,
            "call": "_sig(average_over_neighbourhoods(v, np.eye(5)), (5,))",
            "gold_call": "_sig(_oracle_average_over_neighbourhoods(v, np.eye(5)), (5,))",
        },
        {
            "setup": head,
            "call": "_sig(average_over_neighbourhoods(np.array([1.5, -4.0, 7.25]), np.ones((3, 3))), (3,))",
            "gold_call": "_sig(_oracle_average_over_neighbourhoods(np.array([1.5, -4.0, 7.25]), np.ones((3, 3))), (3,))",
        },
        {
            "setup": head + graph + status,
            "call": "_status(lambda: average_over_neighbourhoods(v, A - np.eye(5)))",
            "gold_call": "_status(lambda: _oracle_average_over_neighbourhoods(v, A - np.eye(5)))",
        },
        {
            "setup": head + graph + status + "B = A.copy()\nB[0, 3] = 1.0\n",
            "call": "_status(lambda: average_over_neighbourhoods(v, B))",
            "gold_call": "_status(lambda: _oracle_average_over_neighbourhoods(v, B))",
        },
        {
            "setup": head + graph + status,
            "call": "_status(lambda: average_over_neighbourhoods(v[:4], A))",
            "gold_call": "_status(lambda: _oracle_average_over_neighbourhoods(v[:4], A))",
        },
    ]
