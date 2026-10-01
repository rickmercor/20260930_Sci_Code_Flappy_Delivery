"""
Sketch-orthonormalize the prescribed initial search block for the randomized eigensolver representation used by subsequent spectral computations. The operation must preserve the dimensionality of the initial block while producing a basis together with its corresponding lower-dimensional sketch representation. The operator and sketch are carried forward unchanged so that later stages can consume the complete numerical state directly.

A low-dimensional search space must remain sufficiently well conditioned for iterative spectral methods to operate reliably. Randomized subspace embeddings provide a compressed representation of vector geometry that can be used when constructing such a basis, while the resulting basis need not be orthonormal in the original Euclidean inner product. This allows the expensive full-space manipulation of the search basis to be reduced while retaining a representation suitable for the subsequent eigenvalue extraction stage.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], containing the unchanged operator A, the fixed sketch S, the sketch-orthonormalized search basis T, and its corresponding sketch representation P = S @ T
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rcgs_initial(state: tuple) -> tuple:
    """Construct the transformed initial search representation.

    Parameters
    ----------
    state : tuple
        Numerical state produced by Step 01 containing the operator,
        initial search block, and fixed sketch.

    Returns
    -------
    tuple
        Numerical state containing the quantities required by the
        generalized Ritz stage.

    Raises
    ------
    ValueError
        If the chained state is malformed, dimensions are inconsistent,
        or the supplied search block is numerically rank deficient.

    Notes
    -----
    The returned quantities must remain consistent with the fixed
    sketch used by the preceding stage.

    Column signs follow the positive-diagonal convention: the upper
    triangular factor relating the sketch of the supplied block to the
    returned sketch has positive diagonal entries.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_rcgs_initial(state: tuple) -> tuple:
    """Reference implementation."""

    if not isinstance(state, tuple) or len(state) != 3:
        raise ValueError("state must be the Step 01 tuple")

    A, V0, S = [
        np.asarray(x, dtype=np.float64)
        for x in state
    ]

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    n = A.shape[0]

    if V0.ndim != 2 or V0.shape[0] != n:
        raise ValueError("V0 must have shape (n,p)")

    if S.ndim != 2 or S.shape[1] != n:
        raise ValueError("S must have shape (s,n)")

    if (
        not np.all(np.isfinite(A))
        or not np.all(np.isfinite(V0))
        or not np.all(np.isfinite(S))
    ):
        raise ValueError("inputs must be finite")

    T = V0.copy()
    P = S @ T

    for ell in range(T.shape[1]):
        if ell:
            h = P[:, :ell].T @ P[:, ell]
            T[:, ell] -= T[:, :ell] @ h
            P[:, ell] -= P[:, :ell] @ h

        norm_p = float(np.linalg.norm(P[:, ell]))

        if not np.isfinite(norm_p) or norm_p <= 1e-14:
            raise ValueError("rank-deficient initial block")

        T[:, ell] /= norm_p
        P[:, ell] /= norm_p

    if not np.allclose(
        P.T @ P,
        np.eye(P.shape[1]),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("initial block is not sketch-orthonormal")

    return A, S, T, P

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid test specifications."""
    return [
        # Normal: benchmark-style 96-dimensional block with the fixed seed-2026 sparse sketch.
        {
            "setup": (
                "import copy\nimport numpy as np\n"
                "n = 96\n"
                "i = np.arange(1, n + 1, dtype=float)\n"
                "A = np.zeros((n, n), dtype=float)\n"
                "A[np.diag_indices(n)] = np.log(99.0 + i)\n"
                "off = 0.19*np.sin(0.73*i[:-1]) + 0.004*np.cos(0.41*i[:-1])\n"
                "A[np.arange(n-1), np.arange(1,n)] = off\n"
                "A[np.arange(1,n), np.arange(n-1)] = off\n"
                "j = np.arange(1, n + 1, dtype=float)\n"
                "V0 = np.column_stack((\n"
                "    np.ones(n),\n"
                "    (-1.0)**(j-1.0),\n"
                "    np.tile([1.0, 1.0, -1.0, -1.0], n//4),\n"
                ")) / np.sqrt(n)\n"
                "rng = np.random.default_rng(2026)\n"
                "S = np.zeros((45, n), dtype=float)\n"
                "for col in range(n):\n"
                "    rows = rng.choice(45, 4, replace=False)\n"
                "    signs = 2*rng.integers(0, 2, 4) - 1\n"
                "    S[rows, col] = signs / 2.0\n"
                "state = (A, V0, S)\n"
            ),
            "call": "rcgs_initial(copy.deepcopy(state))",
            "gold_call": "_oracle_rcgs_initial(state)",
        },

        # Boundary: smallest supported dimension.
        {
            "setup": (
                "import copy\nimport numpy as np\n"
                "A = np.eye(4, dtype=float)\n"
                "V0 = np.column_stack((\n"
                "    np.ones(4),\n"
                "    [1., -1., 1., -1.],\n"
                "    [1., 1., -1., -1.],\n"
                ")) / 2.0\n"
                "S = np.eye(4, dtype=float)\n"
                "state = (A, V0, S)\n"
            ),
            "call": "rcgs_initial(copy.deepcopy(state))",
            "gold_call": "_oracle_rcgs_initial(state)",
        },

        # Edge: few sketch rows (s = 6) for a three-column block, n = 12, zeta = 2 (the orchestrator's third test configuration).
        {
            "setup": (
                "import copy\nimport numpy as np\n"
                "n = 12\n"
                "i = np.arange(1, n + 1, dtype=float)\n"
                "A = np.zeros((n, n), dtype=float)\n"
                "A[np.diag_indices(n)] = np.log(99.0 + i)\n"
                "off = 0.19*np.sin(0.73*i[:-1]) + 0.004*np.cos(0.41*i[:-1])\n"
                "A[np.arange(n-1), np.arange(1,n)] = off\n"
                "A[np.arange(1,n), np.arange(n-1)] = off\n"
                "V0 = np.column_stack((np.ones(n), (-1.0)**np.arange(n), np.tile([1.0, 1.0, -1.0, -1.0], n//4))) / np.sqrt(n)\n"
                "rng = np.random.default_rng(123)\n"
                "S = np.zeros((6, n), dtype=float)\n"
                "for col in range(n):\n"
                "    rows = rng.choice(6, 2, replace=False)\n"
                "    signs = 2*rng.integers(0, 2, 2) - 1\n"
                "    S[rows, col] = signs / np.sqrt(2.0)\n"
                "state = (A, V0, S)\n"
            ),
            "call": "rcgs_initial(copy.deepcopy(state))",
            "gold_call": "_oracle_rcgs_initial(state)",
        },

        # Invalid: rank-deficient supplied block.
        {
            "setup": (
                "import copy\nimport numpy as np\n"
                "A = np.eye(4, dtype=float)\n"
                "V0 = np.column_stack(([1.,0.,0.,0.], [1.,0.,0.,0.]))\n"
                "S = np.eye(4, dtype=float)\n"
                "state = (A, V0, S)\n"
                "def run_model():\n"
                "    try:\n"
                "        rcgs_initial(copy.deepcopy(state))\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_rcgs_initial(state)\n"
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
