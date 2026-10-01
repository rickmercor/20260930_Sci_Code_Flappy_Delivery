"""
Return the sketched search direction at the current iterate after it has been made tangent to the affine constraint. The sketch $S$ acts on the complementary residual. Require $S$ of shape $(m_r, q)$ with $q \ge 1$.

A direction that still has a component in the constraint-row range would leave the affine set after a line search. The complementary residual is seen only through S.

Returns
-------
ndarray of shape (n,): projected sketched direction
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def projected_sketched_direction(
    block_pack: np.ndarray, x: np.ndarray, S: np.ndarray
) -> np.ndarray:
    """Return the constraint-feasible sketched direction at x.

    Parameters
    ----------
    block_pack : np.ndarray
        Packed $(A_p, b_p, A_r, b_r)$.
    x : np.ndarray
        Current iterate, length $n$.
    S : np.ndarray
        Sketch of shape $(m_r, q)$, $q \ge 1$, acting on the complementary residual.

    Returns
    -------
    d : np.ndarray
        Constraint-feasible sketched direction of length $n$.

    Raises
    ------
    ValueError
        If block_pack is malformed, if $x$ does not have shape $(n,)$, or if
        $S$ does not have shape $(m_r, q)$ with $q \ge 1$.
    """
    return np.zeros(1)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _unpack_blocks(block_pack):
    p = np.asarray(block_pack, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("block pack is too short")
    mp, n, mr = [int(round(float(v))) for v in p[:3]]
    if min(mp, n, mr) < 1:
        raise ValueError("packed shapes must be positive")
    need = 3 + mp * n + mp + mr * n + mr
    if p.size != need:
        raise ValueError("block pack length does not match header")
    i = 3
    A_p = p[i : i + mp * n].reshape(mp, n)
    i += mp * n
    b_p = p[i : i + mp]
    i += mp
    A_r = p[i : i + mr * n].reshape(mr, n)
    i += mr * n
    b_r = p[i:]
    return A_p, b_p, A_r, b_r

def _oracle_projected_sketched_direction(block_pack, x, S):
    A_p, _b_p, A_r, b_r = _unpack_blocks(block_pack)
    n = A_p.shape[1]
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.shape != (n,):
        raise ValueError("x must have shape (n,)")
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != A_r.shape[0] or S.shape[1] < 1:
        raise ValueError("S must have shape (m_r, q) with q >= 1")
    r = A_r @ x - b_r
    Sr = S.T @ r
    g = -A_r.T @ (S @ Sr)
    d = g - np.linalg.pinv(A_p) @ (A_p @ g)
    return d

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
A_p = np.array([[1.0, 0.0, 0.5], [0.0, 2.0, -1.0]])
b_p = np.array([1.0, -0.5])
A_r = np.array([[0.3, 0.1, 0.2], [0.0, 1.0, 0.4], [0.2, 0.0, 0.8]])
b_r = np.array([0.1, 0.2, 0.3])
block_pack = np.concatenate(([2.0, 3.0, 3.0], A_p.ravel(), b_p, A_r.ravel(), b_r))
x = np.array([0.2, -0.1, 0.4])
S = np.array([[1.0, 0.0], [0.5, 1.0], [0.0, -1.0]])
""",
            "call": "projected_sketched_direction(block_pack, x, S)",
            "gold_call": "_oracle_projected_sketched_direction(block_pack, x, S)",
        },
        {
            "setup": """import numpy as np
A_p = np.array([[2.0, 1.0]])
b_p = np.array([1.0])
A_r = np.array([[1.0, 0.0], [0.0, 1.0]])
b_r = np.array([0.0, 0.0])
block_pack = np.concatenate(([1.0, 2.0, 2.0], A_p.ravel(), b_p, A_r.ravel(), b_r))
x = np.linalg.pinv(A_p) @ b_p
S = np.array([[0.7], [-0.2]])
""",
            "call": "projected_sketched_direction(block_pack, x, S)",
            "gold_call": "_oracle_projected_sketched_direction(block_pack, x, S)",
        },
        {
            "setup": """import numpy as np
A_p = np.eye(2)
b_p = np.array([0.0, 0.0])
A_r = np.array([[1.0, 2.0], [3.0, 4.0], [0.5, -1.0]])
b_r = np.array([1.0, 0.0, -1.0])
block_pack = np.concatenate(([2.0, 2.0, 3.0], A_p.ravel(), b_p, A_r.ravel(), b_r))
x = np.zeros(2)
S = np.ones((3, 2))
""",
            "call": "projected_sketched_direction(block_pack, x, S)",
            "gold_call": "_oracle_projected_sketched_direction(block_pack, x, S)",
        },
        {
            "setup": """import numpy as np
A_p = np.array([[1.0, 0.0]])
b_p = np.array([0.0])
A_r = np.array([[0.0, 1.0]])
b_r = np.array([0.0])
block_pack = np.concatenate(([1.0, 2.0, 1.0], A_p.ravel(), b_p, A_r.ravel(), b_r))
x = np.zeros(2)
S = np.ones((1, 2))
def run_model():
    try:
        projected_sketched_direction(block_pack, np.zeros(3), S)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_projected_sketched_direction(block_pack, np.zeros(3), S)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
A_p = np.array([[1.0, 0.0]])
b_p = np.array([0.0])
A_r = np.array([[0.0, 1.0]])
b_r = np.array([0.0])
block_pack = np.concatenate(([1.0, 2.0, 1.0], A_p.ravel(), b_p, A_r.ravel(), b_r))
x = np.zeros(2)
S = np.ones((2, 1))
def run_model():
    try:
        projected_sketched_direction(block_pack, x, S)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_projected_sketched_direction(block_pack, x, S)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
