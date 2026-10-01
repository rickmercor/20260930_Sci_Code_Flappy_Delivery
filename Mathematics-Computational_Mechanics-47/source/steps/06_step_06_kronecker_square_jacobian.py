"""
Differentiate the Kronecker square of the reduced state with respect to that state.

Every quadratic term of a projected polynomial system enters the reduced

Newton system through this state-dependent factor, allowing the quadratic

Jacobian contribution to be assembled from precomputed reduced operators.

Returns
-------
np.ndarray, the derivative array of shape (n**2, n) as a float array.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def kronecker_square_jacobian(reduced_state: np.ndarray) -> np.ndarray:
    """Return the exact derivative of v kron v with respect to v.

    For a vector v of length n, the Kronecker square v kron v is the vector of
    length n**2 whose entry p * n + q is v_p * v_q, using numpy's layout with
    zero-based p and q. This function returns the n**2 by n matrix J whose
    entry (p * n + q, r) is the partial derivative of v_p * v_q with respect to
    v_r, so that for any matrix M with n**2 columns the derivative of
    M (v kron v) with respect to v is M J.

    Parameters
    ----------
    reduced_state : np.ndarray
        Finite real array of shape (n,) with n >= 1.

    Returns
    -------
    derivative : np.ndarray
        Float array of shape (n**2, n).

    Raises
    ------
    ValueError
        If reduced_state is not a non-empty 1-D array of finite real entries.
    """
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_kronecker_square_jacobian(reduced_state: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    import numpy as np

    vector = np.asarray(reduced_state, dtype=float)
    if vector.ndim != 1 or vector.shape[0] < 1:
        raise ValueError("reduced_state must be a 1-D array with at least one entry")
    if not np.all(np.isfinite(vector)):
        raise ValueError("reduced_state entries must be finite")

    size = vector.shape[0]
    identity = np.eye(size, dtype=float)
    column = vector.reshape(size, 1)
    # The first term differentiates the left factor of the Kronecker square and
    # the second term differentiates the right factor; both are needed because
    # the two factors occupy different index positions.
    derivative = np.kron(column, identity) + np.kron(identity, column)
    return np.asarray(derivative, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return a list of test case specifications."""
    reducer = (
        "import numpy as np\n"
        "def _sig(value, scale):\n"
        "    a = np.asarray(value, dtype=float)\n"
        "    b = np.concatenate([np.asarray([a.ndim, *a.shape], dtype=float), a.ravel()])\n"
        "    k = np.arange(1.0, b.size + 1.0)\n"
        "    return float((np.sum(np.abs(b)) + np.sum(b * np.cos(k))) / scale)\n"
    )
    return [
        # Case 1: a generic reduced state with mixed signs.
        {
            "setup": reducer + "v = np.array([0.7, -1.3, 0.25, 2.0])\n",
            "call": "_sig(kronecker_square_jacobian(v), 1e1)",
            "gold_call": "_sig(_oracle_kronecker_square_jacobian(v), 1e1)",
        },
        # Case 2: a single reduced mode, where the derivative is 2 v.
        {
            "setup": reducer + "v = np.array([-1.75])\n",
            "call": "_sig(kronecker_square_jacobian(v), 1e0)",
            "gold_call": "_sig(_oracle_kronecker_square_jacobian(v), 1e0)",
        },
        # Case 3: a zero state, where the derivative vanishes identically.
        {
            "setup": reducer + "v = np.zeros(5)\n",
            "call": "_sig(kronecker_square_jacobian(v), 1e0)",
            "gold_call": "_sig(_oracle_kronecker_square_jacobian(v), 1e0)",
        },
        # --- Decisive: contracted against a matrix that is not symmetric under
        #     exchange of the two Kronecker index positions, the result must
        #     match a central finite difference of M (v kron v). Keeping only
        #     one of the two index positions, doubled, passes for symmetric M
        #     and fails here.
        {
            "setup": reducer + (
                "def directional():\n"
                "    n = 4\n"
                "    rng = np.random.default_rng(17)\n"
                "    m = np.zeros((3, n * n))\n"
                "    for p in range(n):\n"
                "        for q in range(n):\n"
                "            if p <= q:\n"
                "                m[:, p * n + q] = rng.standard_normal(3)\n"
                "    v = np.array([0.9, -0.4, 1.7, 0.05])\n"
                "    jac = m @ kronecker_square_jacobian(v)\n"
                "    h = 1e-6\n"
                "    fd = np.zeros((3, n))\n"
                "    for r in range(n):\n"
                "        e = np.zeros(n)\n"
                "        e[r] = h\n"
                "        plus = m @ np.kron(v + e, v + e)\n"
                "        minus = m @ np.kron(v - e, v - e)\n"
                "        fd[:, r] = (plus - minus) / (2.0 * h)\n"
                "    single = m @ (2.0 * np.kron(v.reshape(-1, 1), np.eye(n)))\n"
                "    return (int(np.max(np.abs(jac - fd)) < 1e-6)\n"
                "            + 2 * int(np.max(np.abs(single - fd)) > 1e-3))\n"
            ),
            "call": "directional()",
            "gold_call": "3",
        },
        # --- Decisive: the two Kronecker index positions contribute different
        #     rows, so the derivative rows p * n + q and q * n + p coincide while
        #     each differs from the doubled single-position result unless p == q.
        {
            "setup": reducer + (
                "def symmetry():\n"
                "    v = np.array([1.0, -2.0, 3.0])\n"
                "    n = v.size\n"
                "    j = kronecker_square_jacobian(v)\n"
                "    pairs = float(max(np.max(np.abs(j[p * n + q] - j[q * n + p]))\n"
                "                      for p in range(n) for q in range(n)))\n"
                "    diag_ok = int(np.max(np.abs(np.array([j[p * n + p, p] for p in range(n)])\n"
                "                                - 2.0 * v)) < 1e-12)\n"
                "    left = np.kron(v.reshape(-1, 1), np.eye(n))\n"
                "    return int(pairs < 1e-12) + 2 * diag_ok + 4 * int(np.max(np.abs(j - 2.0 * left)) > 1e-8)\n"
            ),
            "call": "symmetry()",
            "gold_call": "7",
        },
        # Case 6: invalid rank.
        {
            "setup": reducer + (
                "v = np.ones((2, 2))\n"
                "def run_model():\n"
                "    try:\n"
                "        kronecker_square_jacobian(v)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_kronecker_square_jacobian(v)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 7: invalid non-finite entry.
        {
            "setup": reducer + (
                "v = np.array([1.0, np.inf])\n"
                "def run_model():\n"
                "    try:\n"
                "        kronecker_square_jacobian(v)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_kronecker_square_jacobian(v)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
