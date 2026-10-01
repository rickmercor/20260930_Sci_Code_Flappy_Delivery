"""
Restrict a grouped mutation polynomial to a common mutation rate and normalize it by the independently computed reference partition function.

On the diagonal where every group rate equals μ, a multivariate coefficient contributes to the univariate order given by the sum of its group degrees. This contraction preserves the exact aggregate finite-substitution orders.

Returns
-------
np.ndarray: normalized diagonal coefficients in increasing total order, shape (L + 1,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def normalize_order_coefficients(
    fixed_inside_weights: "np.ndarray",
    library_coefficients: "np.ndarray",
    relative_tolerance: float = 1e-10,
) -> "np.ndarray":
    """Diagonalize grouped coefficients and normalize by the reference value.

    ``fixed_inside_weights`` is the class-resolved result of
    ``compute_stacked_inside_weights`` for a length-``L`` reference; its
    partition function is entry ``[0, 1, L]``. ``library_coefficients`` is
    the one- to four-dimensional grid returned by
    ``compute_profile_partition_polynomial``. If its shape is
    ``(n_0 + 1, ..., n_{G-1} + 1)``, require ``sum_g n_g = L``.

    Restrict the grouped polynomial ``F(mu_0,...,mu_{G-1})`` to the diagonal
    ``mu_0 = ... = mu_{G-1} = mu``. Thus every grid entry at multi-index
    ``(r_0,...,r_{G-1})`` is accumulated into univariate coefficient
    ``r_0 + ... + r_{G-1}``. Before normalization, require the all-zero grid
    entry to agree with the fixed partition function within
    ``relative_tolerance``.

    Parameters
    ----------
    fixed_inside_weights : np.ndarray
        Finite array of shape ``(4, L + 2, L + 1)`` with ``L >= 1``.
    library_coefficients : np.ndarray
        Finite coefficient grid with one through four nonempty axes and total
        degree capacity ``sum(shape[g] - 1) = L``.
    relative_tolerance : float
        Finite nonnegative relative tolerance for constant-term agreement.

    Returns
    -------
    np.ndarray
        Normalized diagonal coefficients of shape ``(L + 1,)`` in increasing
        total substitution order.

    Raises
    ------
    ValueError
        If shapes or values are invalid, the reference partition function is
        not positive, or the two constant terms do not agree.
    """
    return normalized

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_normalize_order_coefficients(
    fixed_inside_weights: "np.ndarray",
    library_coefficients: "np.ndarray",
    relative_tolerance: float = 1e-10,
) -> "np.ndarray":
    """Reference diagonal contraction, validation and normalization."""
    import numpy as np

    try:
        inside = np.array(fixed_inside_weights, dtype=float)
        coefficients = np.array(library_coefficients, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("weights and coefficients must be numeric") from None
    if inside.ndim != 3 or inside.shape[0] != 4 or inside.shape[2] < 2:
        raise ValueError("fixed_inside_weights has an invalid shape")
    length = inside.shape[2] - 1
    if inside.shape != (4, length + 2, length + 1):
        raise ValueError("fixed_inside_weights has an invalid shape")
    if coefficients.ndim < 1 or coefficients.ndim > 4 or any(size < 1 for size in coefficients.shape):
        raise ValueError("library_coefficients must have one through four nonempty axes")
    if sum(size - 1 for size in coefficients.shape) != length:
        raise ValueError("library coefficient degree capacities must sum to L")
    if not np.all(np.isfinite(inside)) or not np.all(np.isfinite(coefficients)):
        raise ValueError("weights and coefficients must be finite")
    if isinstance(relative_tolerance, bool):
        raise ValueError("relative_tolerance must be a finite nonnegative scalar")
    try:
        tolerance = float(relative_tolerance)
    except (TypeError, ValueError):
        raise ValueError("relative_tolerance must be a finite nonnegative scalar") from None
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("relative_tolerance must be a finite nonnegative scalar")

    reference = float(inside[0, 1, length])
    constant = float(coefficients[(0,) * coefficients.ndim])
    if reference <= 0.0:
        raise ValueError("the reference partition function must be positive")
    scale = max(abs(reference), abs(constant))
    if abs(constant - reference) > tolerance * scale:
        raise ValueError("the polynomial constant does not match the reference partition function")

    diagonal = np.zeros(length + 1, dtype=float)
    for multi_index in np.ndindex(coefficients.shape):
        diagonal[sum(multi_index)] += coefficients[multi_index]
    return diagonal / reference

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return diagonalization, boundary, edge and validation cases."""
    base = (
        "import numpy as np\n"
        "def _inside(length, z):\n"
        "    w = np.zeros((4, length + 2, length + 1))\n"
        "    w[0, 1, length] = z\n"
        "    return w\n"
        "def _sig(c):\n"
        "    c = np.asarray(c, dtype=float)\n"
        "    return float(np.dot(c, np.cos(np.arange(c.size) + .3)))\n"
    )
    status = (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    return [
        {
            "setup": base + "W = _inside(4, .025)\nC = np.array([[.025, .01, -.004], [.04, -.02, .003], [-.01, .005, .001]])\n",
            "call": "_sig(normalize_order_coefficients(W.copy(), C.copy()))",
            "gold_call": "_sig(_oracle_normalize_order_coefficients(W.copy(), C.copy()))",
        },
        {
            "setup": base + "W = _inside(1, 3.5)\nC = np.array([3.5, -1.25])\n",
            "call": "_sig(normalize_order_coefficients(W.copy(), C.copy(), 0.0))",
            "gold_call": "_sig(_oracle_normalize_order_coefficients(W.copy(), C.copy(), 0.0))",
        },
        {
            "setup": base + "W = _inside(3, 2.0)\nC = np.zeros((2, 2, 2)); C[0,0,0] = 2.0 + 1e-11; C[1,0,0] = 1.0; C[0,1,1] = -.2\n",
            "call": "_sig(normalize_order_coefficients(W.copy(), C.copy(), 1e-9))",
            "gold_call": "_sig(_oracle_normalize_order_coefficients(W.copy(), C.copy(), 1e-9))",
        },
        {
            "setup": base + status + "W = _inside(3, 2.0)\nC = np.zeros((2,3)); C[0,0] = 2.1\n",
            "call": "_status(lambda: normalize_order_coefficients(W, C, 1e-4))",
            "gold_call": "_status(lambda: _oracle_normalize_order_coefficients(W, C, 1e-4))",
        },
        {
            "setup": base + status + "W = _inside(2, 0.0)\nC = np.array([[0.0, 1.0], [2.0, 0.0]])\n",
            "call": "_status(lambda: normalize_order_coefficients(W, C))",
            "gold_call": "_status(lambda: _oracle_normalize_order_coefficients(W, C))",
        },
        {
            "setup": base + status + "W = _inside(2, 1.0)\nC = np.array([1.0, 1.0])\n",
            "call": "_status(lambda: normalize_order_coefficients(W, C))",
            "gold_call": "_status(lambda: _oracle_normalize_order_coefficients(W, C))",
        },
    ]
