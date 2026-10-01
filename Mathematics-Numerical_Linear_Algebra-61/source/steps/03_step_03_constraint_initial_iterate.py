"""
Return the minimum-norm solution $x_0 = A_p^+ b_p$ of the packed constraint block.

Every later iterate is required to stay in the affine set of the constraint rows. Starting at A_p^+ b_p places x0 in that set, so the projector correction on later sketched gradients keeps the three equations satisfied.

Returns
-------
ndarray of shape (n,): initial iterate x0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def constraint_initial_iterate(block_pack: np.ndarray) -> np.ndarray:
    """Return $x_0 = A_p^+ b_p$ from a packed row split.

    Parameters
    ----------
    block_pack : np.ndarray
        Packed $(A_p, b_p, A_r, b_r)$ from extract_row_blocks.

    Returns
    -------
    x0 : np.ndarray
        Vector of length $n$.

    Raises
    ------
    ValueError
        If block_pack is too short, if packed shapes are not positive, if
        the pack length does not match the header, or if the constraint
        block is not a finite 2d array with a matching right-hand side.
    """
    return np.zeros(1)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _unpack_constraint_init_pack(block_pack):
    p = np.asarray(block_pack, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("block pack is too short")
    mp, n, mr = [int(round(float(x))) for x in p[:3]]
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

def _oracle_constraint_initial_iterate(block_pack):
    A_p, b_p, _A_r, _b_r = _unpack_constraint_init_pack(block_pack)
    A_p = np.asarray(A_p, dtype=float)
    b_p = np.asarray(b_p, dtype=float).reshape(-1)
    if A_p.ndim != 2 or A_p.shape[0] < 1 or A_p.shape[1] < 1:
        raise ValueError("constraint block A_p must be a nonempty 2d array")
    if b_p.shape != (A_p.shape[0],):
        raise ValueError("b_p must have shape (mp,)")
    if not np.all(np.isfinite(A_p)) or not np.all(np.isfinite(b_p)):
        raise ValueError("constraint block must be finite")
    return np.linalg.pinv(A_p) @ b_p

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
mp, n, mr = 2, 3, 3
block_pack = np.concatenate(([float(mp), float(n), float(mr)], A_p.ravel(), b_p, A_r.ravel(), b_r))
""",
            "call": "constraint_initial_iterate(block_pack)",
            "gold_call": "_oracle_constraint_initial_iterate(block_pack)",
        },
        {
            "setup": """import numpy as np
A_p = np.array([[2.0, 1.0]])
b_p = np.array([1.0])
A_r = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
b_r = np.array([0.0, 0.0, 0.0])
block_pack = np.concatenate(([1.0, 2.0, 3.0], A_p.ravel(), b_p, A_r.ravel(), b_r))
""",
            "call": "constraint_initial_iterate(block_pack)",
            "gold_call": "_oracle_constraint_initial_iterate(block_pack)",
        },
        {
            "setup": """import numpy as np
A_p = np.eye(2)
b_p = np.array([3.0, -1.0])
A_r = np.array([[1.0, 2.0]])
b_r = np.array([0.5])
block_pack = np.concatenate(([2.0, 2.0, 1.0], A_p.ravel(), b_p, A_r.ravel(), b_r))
""",
            "call": "constraint_initial_iterate(block_pack)",
            "gold_call": "_oracle_constraint_initial_iterate(block_pack)",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        constraint_initial_iterate(np.array([1.0, 2.0]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_constraint_initial_iterate(np.array([1.0, 2.0]))
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
def run_model():
    try:
        constraint_initial_iterate(np.zeros(6))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_constraint_initial_iterate(np.zeros(6))
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
