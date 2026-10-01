"""
Apply one sketched correction expansion to the second-cycle search state that begins at the restart. The input is the current second-cycle basis (the restarted three-column basis from Step 06, followed by any correction blocks already appended in this cycle), its stored sketch and operator image, together with one active correction block. The output appends that block and returns the updated basis, stored sketch and operator image for the next second-cycle extraction.

After a restart, the second cycle starts from the basis and stored sketch produced by the restart stage, and each expansion works with them as supplied.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], containing the unchanged operator A and sketch S together with the post-restart expanded basis Vt_new, its expanded sketch representation Q_new = S @ Vt_new, and its expanded operator image W_new = A @ Vt_new
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def post_restart_expansion(state: tuple) -> tuple:
    """Apply the source-defined second-cycle expansion to a restarted state.

    Parameters
    ----------
    state : tuple
        (A, S, Vt, Q, W, T): operator, fixed sketch, current second-cycle
        full-space basis, stored sketch, operator image and one active
        correction block. The basis starts with the three-column output of
        the restart stage and may already contain correction blocks appended
        earlier in the second cycle.

    Returns
    -------
    tuple
        (A, S, Vt_new, Q_new, W_new) after appending the correction block.

    Raises
    ------
    ValueError
        If the state is malformed, dimensions are inconsistent, the sketch
        is too small for the expanded basis, or the correction block is
        numerically rank deficient after projection.

    Notes
    -----
    This step represents one second-cycle expansion. Carry the fixed
    sketch forward unchanged and do not restart or re-sketch-orthonormalize
    the columns already in the basis here.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_post_restart_expansion(state: tuple) -> tuple:
    """Reference implementation."""
    return _oracle_sketched_block_expansion(state)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return post-restart normal, boundary, and trajectory-like cases."""
    common = (
        "import copy\nimport numpy as np\n"
        "def _family(n):\n"
        "    i = np.arange(1, n + 1, dtype=float)\n"
        "    A = np.zeros((n, n))\n"
        "    A[np.diag_indices(n)] = np.log(99.0 + i)\n"
        "    off = 0.19*np.sin(0.73*i[:-1]) + 0.004*np.cos(0.41*i[:-1])\n"
        "    A[np.arange(n-1), np.arange(1, n)] = off\n"
        "    A[np.arange(1, n), np.arange(n-1)] = off\n"
        "    return A\n"
        "def _sketch(s, n, zeta, seed):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    S = np.zeros((s, n))\n"
        "    for col in range(n):\n"
        "        rows = rng.choice(s, zeta, replace=False)\n"
        "        signs = 2*rng.integers(0, 2, zeta) - 1\n"
        "        S[rows, col] = signs / np.sqrt(zeta)\n"
        "    return S\n"
        "def _restarted_state(n, s, zeta, seed, t_seed):\n"
        "    A = _family(n)\n"
        "    S = _sketch(s, n, zeta, seed)\n"
        "    V0 = np.random.default_rng(t_seed).standard_normal((n, 3))\n"
        "    Vt, _ = np.linalg.qr(V0)\n"
        "    Q = S @ Vt\n"
        "    W = A @ Vt\n"
        "    T = np.random.default_rng(t_seed + 100).standard_normal((n, 3))\n"
        "    return (A, S, Vt, Q, W, T)\n"
    )
    return [
        # Normal: restarted three-column basis, explicitly full-space orthonormal.
        {
            "setup": common +
                "state = _restarted_state(24, 45, 4, 2026, 21)\n",
            "call": "post_restart_expansion(copy.deepcopy(state))",
            "gold_call": "_oracle_post_restart_expansion(state)",
        },
        # Boundary: the expanded basis fills the sketch dimension.
        {
            "setup": common +
                "state = _restarted_state(12, 6, 2, 123, 31)\n",
            "call": "post_restart_expansion(copy.deepcopy(state))",
            "gold_call": "_oracle_post_restart_expansion(state)",
        },
        # Edge: second second-cycle expansion (j = 6 -> 9); the basis is a retained three-column
        # rotation followed by one obliquely appended, sketch-normalized block.
        {
            "setup": common + (
                "n = 24\n"
                "A = _family(n)\n"
                "S = _sketch(45, n, 4, 2026)\n"
                "V9, _ = np.linalg.qr(np.random.default_rng(41).standard_normal((n, 9)))\n"
                "_, Y9 = np.linalg.eigh(V9.T @ A @ V9)\n"
                "V3 = V9 @ Y9[:, :3]\n"
                "Q3 = S @ V3\n"
                "T1 = np.random.default_rng(43).standard_normal((n, 3))\n"
                "C1 = Q3.T @ (S @ T1)\n"
                "Tp = T1 - V3 @ C1\n"
                "_, r = np.linalg.qr(S @ Tp)\n"
                "r = np.sign(np.diag(r))[:, None] * r\n"
                "Vt = np.column_stack((V3, np.linalg.solve(r.T, Tp.T).T))\n"
                "Q = S @ Vt\n"
                "W = A @ Vt\n"
                "T = np.random.default_rng(42).standard_normal((n, 3))\n"
                "state = (A, S, Vt, Q, W, T)\n"
            ),
            "call": "post_restart_expansion(copy.deepcopy(state))",
            "gold_call": "_oracle_post_restart_expansion(state)",
        },
    ]
