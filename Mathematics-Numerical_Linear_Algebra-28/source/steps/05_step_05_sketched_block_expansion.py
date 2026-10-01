"""
Append one block of correction directions to the first-cycle search space using the source study's randomized sketch-orthogonalization. The input state holds the operator, the fixed sketch, the current full-space basis together with its stored sketch and its operator image, and the correction block produced by Step 04. The current basis is the sketch-orthonormal basis maintained during the first cycle. The output is the expanded basis with its expanded stored sketch and operator image.

Randomized orthogonalization measures the geometry of new search directions through a low-dimensional sketch instead of full-dimensional inner products, which reduces the full-space work per appended block. During the first cycle, the stored sketch provides the compressed geometry used to keep each newly appended block compatible with the existing search basis.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], containing the unchanged operator A and sketch S together with the expanded search basis Vt_new, its expanded sketch representation Q_new = S @ Vt_new, and its expanded operator image W_new = A @ Vt_new
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sketched_block_expansion(state: tuple) -> tuple:
    """Append a correction block to the search space by sketched orthogonalization.

    Parameters
    ----------
    state : tuple
        (A, S, Vt, Q, W, T): operator (n by n), sketch (s by n), current
        full-space basis (n by j), its stored sketch (s by j), its operator
        image A Vt (n by j) and the correction block (n by p).

    Returns
    -------
    tuple
        (A, S, Vt_new, Q_new, W_new) with j + p columns in the basis, its
        stored sketch and its operator image.

    Raises
    ------
    ValueError
        If the state is malformed, the dimensions are inconsistent, the
        sketch has fewer rows than the expanded basis has columns, or the
        correction block is numerically rank deficient after projection.

    Notes
    -----
    Use a single stage-1 pass (p_s = 1) of the source's randomized
    Gram-Schmidt. Column signs follow the positive-diagonal convention:
    the upper triangular factor relating the sketch of the projected block
    to its returned sketch has positive diagonal entries.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sketched_block_expansion(state: tuple) -> tuple:
    """Reference implementation."""
    if not isinstance(state, tuple) or len(state) != 6:
        raise ValueError("state must be (A, S, Vt, Q, W, T)")

    A, S, Vt, Q, W, T = [np.asarray(x, dtype=np.float64) for x in state]

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    n = A.shape[0]

    if S.ndim != 2 or S.shape[1] != n:
        raise ValueError("S must have shape (s,n)")

    if Vt.ndim != 2 or Vt.shape[0] != n or W.shape != Vt.shape:
        raise ValueError("Vt and W must have shape (n,j)")

    if Q.ndim != 2 or Q.shape != (S.shape[0], Vt.shape[1]):
        raise ValueError("Q must have shape (s,j)")

    if T.ndim != 2 or T.shape[0] != n or T.shape[1] < 1:
        raise ValueError("T must have shape (n,p)")

    if Vt.shape[1] + T.shape[1] > S.shape[0]:
        raise ValueError("sketch has too few rows for the expanded basis")

    if not all(np.all(np.isfinite(x)) for x in (A, S, Vt, Q, W, T)):
        raise ValueError("state contains non-finite values")

    # Stage 1, one pass: coefficients from the stored sketch, removed from both spaces.
    P = S @ T
    C = Q.T @ P
    T_hat = T - Vt @ C
    P = P - Q @ C

    # Stage 2: column loop in the sketched norm.
    for ell in range(T_hat.shape[1]):
        if ell:
            h = P[:, :ell].T @ P[:, ell]
            T_hat[:, ell] -= T_hat[:, :ell] @ h
            P[:, ell] -= P[:, :ell] @ h

        norm_p = float(np.linalg.norm(P[:, ell]))

        if not np.isfinite(norm_p) or norm_p <= 1e-14:
            raise ValueError("rank-deficient sketched correction block")

        T_hat[:, ell] /= norm_p
        P[:, ell] /= norm_p

    return (
        A,
        S,
        np.column_stack((Vt, T_hat)),
        np.column_stack((Q, P)),
        np.column_stack((W, A @ T_hat)),
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and invalid test specifications."""
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
        "def _sketch_orthonormal(S, V):\n"
        "    _, R = np.linalg.qr(S @ V)\n"
        "    d = np.sign(np.diag(R))\n"
        "    d[d == 0.0] = 1.0\n"
        "    return np.linalg.solve((d[:, None]*R).T, V.T).T\n"
    )
    return [
        # Normal: first-cycle sketch-orthonormal basis with nonzero correction.
        {
            "setup": common + (
                "n = 24\n"
                "A = _family(n)\n"
                "S = _sketch(45, n, 4, 2026)\n"
                "Vt = _sketch_orthonormal(S, np.random.default_rng(1).standard_normal((n, 3)))\n"
                "Q = S @ Vt\n"
                "W = A @ Vt\n"
                "T = np.random.default_rng(2).standard_normal((n, 3))\n"
                "state = (A, S, Vt, Q, W, T)\n"
            ),
            "call": "sketched_block_expansion(copy.deepcopy(state))",
            "gold_call": "_oracle_sketched_block_expansion(state)",
        },
        # Boundary: sketch rows equal the expanded basis dimension.
        {
            "setup": common + (
                "n = 12\n"
                "A = _family(n)\n"
                "S = _sketch(6, n, 2, 123)\n"
                "Vt = _sketch_orthonormal(S, np.random.default_rng(5).standard_normal((n, 3)))\n"
                "Q = S @ Vt\n"
                "W = A @ Vt\n"
                "T = np.random.default_rng(6).standard_normal((n, 3))\n"
                "state = (A, S, Vt, Q, W, T)\n"
            ),
            "call": "sketched_block_expansion(copy.deepcopy(state))",
            "gold_call": "_oracle_sketched_block_expansion(state)",
        },
        # Edge: second first-cycle expansion (j = 6 -> 9); the basis already holds an appended block.
        {
            "setup": common + (
                "n = 24\n"
                "A = _family(n)\n"
                "S = _sketch(45, n, 4, 2026)\n"
                "V3 = _sketch_orthonormal(S, np.random.default_rng(15).standard_normal((n, 3)))\n"
                "T1 = np.random.default_rng(16).standard_normal((n, 3))\n"
                "C1 = (S @ V3).T @ (S @ T1)\n"
                "Vt = np.column_stack((V3, _sketch_orthonormal(S, T1 - V3 @ C1)))\n"
                "Q = S @ Vt\n"
                "W = A @ Vt\n"
                "T = np.random.default_rng(17).standard_normal((n, 3))\n"
                "state = (A, S, Vt, Q, W, T)\n"
            ),
            "call": "sketched_block_expansion(copy.deepcopy(state))",
            "gold_call": "_oracle_sketched_block_expansion(state)",
        },
        # Invalid: the correction block lies in the span of the basis.
        {
            "setup": common + (
                "n = 12\n"
                "A = _family(n)\n"
                "S = _sketch(8, n, 2, 7)\n"
                "Vt = _sketch_orthonormal(S, np.random.default_rng(8).standard_normal((n, 3)))\n"
                "Q = S @ Vt\n"
                "W = A @ Vt\n"
                "T = Vt @ np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 2.0], [1.0, 0.0, 1.0]])\n"
                "state = (A, S, Vt, Q, W, T)\n"
                "def run_model():\n"
                "    try:\n"
                "        sketched_block_expansion(copy.deepcopy(state))\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_sketched_block_expansion(state)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
