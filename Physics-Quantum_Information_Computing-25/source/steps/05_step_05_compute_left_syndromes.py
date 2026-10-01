"""
For a guess of the error on the off-diagonal part of the decoupled matrix, compute the syndrome that each diagonal block must then explain.

Once the off-diagonal error is fixed, the remaining constraints separate into independent small problems, one per diagonal block.

Returns
-------
np.ndarray: (K, m/K) integer 0/1 left-part syndromes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_left_syndromes(
    syndrome: "np.ndarray",
    t_matrix: "np.ndarray",
    a_matrix: "np.ndarray",
    right_error: "np.ndarray",
    n_blocks: int,
) -> "np.ndarray":
    r"""Return the per-block syndromes left for the diagonal blocks.

    The decoupled matrix is $D' = T D P = (\mathrm{diag}(D_1, \dots, D_K) \mid A)$
    and ``syndrome`` is the measured syndrome of the original matrix $D$. For
    the off-diagonal error ``right_error`` (on the columns of $A$),
    return the $(K, m/K)$ array whose row $i$ is the syndrome that
    block $D_i$ must reproduce, over rows $i m/K, \dots, (i+1) m/K - 1$ of
    $D'$, for the complete permuted error to satisfy the decoupled
    syndrome equation modulo 2.

    Parameters
    ----------
    syndrome : np.ndarray
        Binary vector of length $m$.
    t_matrix : np.ndarray
        Binary $(m, m)$ transformation $T$.
    a_matrix : np.ndarray
        Binary off-diagonal part $A$ of shape $(m, n_A)$.
    right_error : np.ndarray
        Binary vector of length $n_A$.
    n_blocks : int
        Positive number of blocks $K$ dividing $m$.

    Returns
    -------
    np.ndarray
        Integer 0/1 array of shape $(K, m/K)$.

    Raises
    ------
    ValueError
        If any array is not binary with the stated shape, or ``n_blocks``
        is not a positive integer dividing $m$.
    """
    return left_syndromes

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_left_syndromes(
    syndrome: "np.ndarray",
    t_matrix: "np.ndarray",
    a_matrix: "np.ndarray",
    right_error: "np.ndarray",
    n_blocks: int,
) -> "np.ndarray":
    """Reference left-part syndrome: (T s + A r) mod 2, split by block rows."""
    import numpy as np

    def _binary(value, shape, name):
        array = np.asarray(value)
        if array.shape != shape or not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"{name} must be a binary array of shape {shape}")
        if not np.all((array == 0) | (array == 1)):
            raise ValueError(f"{name} entries must be 0 or 1")
        return array.astype(np.int64)

    s_raw = np.asarray(syndrome)
    if s_raw.ndim != 1 or s_raw.size == 0:
        raise ValueError("syndrome must be a nonempty vector")
    m = s_raw.size
    s = _binary(s_raw, (m,), "syndrome")
    t = _binary(t_matrix, (m, m), "t_matrix")
    a_raw = np.asarray(a_matrix)
    if a_raw.ndim != 2 or a_raw.shape[0] != m:
        raise ValueError("a_matrix must have m rows")
    a = _binary(a_raw, a_raw.shape, "a_matrix")
    r = _binary(right_error, (a.shape[1],), "right_error")
    if isinstance(n_blocks, bool) or not isinstance(n_blocks, (int, np.integer)) or n_blocks < 1 or m % n_blocks:
        raise ValueError("n_blocks must be a positive integer dividing m")
    left = (t @ s + a @ r) % 2
    return left.reshape(int(n_blocks), m // int(n_blocks)).astype(np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return numerical test specifications with elementwise array comparisons."""
    helpers = (
        "import numpy as np\n"
        "def _sig(v):\n"
        "    v = np.asarray(v, dtype=float)\n"
        "    k = np.arange(v.size, dtype=float)\n"
        "    return 100.0 * v.shape[0] + v.sum() + float(v.ravel() @ np.cos(0.71 * k + 0.25))\n"
        "T6 = np.array([[1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0],\n"
        "               [1, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1]])\n"
        "A6 = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1]])\n"
        "S6 = np.array([1, 1, 0, 1, 0, 0])\n"
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
    return [
        {
            "setup": helpers,
            "call": "compute_left_syndromes(S6.copy(), T6.copy(), A6.copy(), np.array([1, 0, 1]), 3)",
            "gold_call": "_oracle_compute_left_syndromes(S6.copy(), T6.copy(), A6.copy(), np.array([1, 0, 1]), 3)",
        },
        {
            "setup": helpers,
            "call": "compute_left_syndromes(S6.copy(), T6.copy(), A6.copy(), np.array([0, 1, 1]), 2)",
            "gold_call": "_oracle_compute_left_syndromes(S6.copy(), T6.copy(), A6.copy(), np.array([0, 1, 1]), 2)",
        },
        {
            "setup": helpers,
            "call": "compute_left_syndromes(S6.copy(), T6.copy(), A6.copy(), np.zeros(3, dtype=int), 6)",
            "gold_call": "_oracle_compute_left_syndromes(S6.copy(), T6.copy(), A6.copy(), np.zeros(3, dtype=int), 6)",
        },
        {
            "setup": helpers,
            "call": "compute_left_syndromes(S6.copy(), np.eye(6, dtype=int), A6.copy(), np.array([1, 1, 1]), 1)",
            "gold_call": "_oracle_compute_left_syndromes(S6.copy(), np.eye(6, dtype=int), A6.copy(), np.array([1, 1, 1]), 1)",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: compute_left_syndromes(S6.copy(), T6.copy(), A6.copy(), np.array([1, 0, 1]), 4))",
            "gold_call": "_status(lambda: _oracle_compute_left_syndromes(S6.copy(), T6.copy(), A6.copy(), np.array([1, 0, 1]), 4))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: compute_left_syndromes(S6.copy(), T6.copy(), A6.copy(), np.array([1, 0]), 3))",
            "gold_call": "_status(lambda: _oracle_compute_left_syndromes(S6.copy(), T6.copy(), A6.copy(), np.array([1, 0]), 3))",
        },
    ]
