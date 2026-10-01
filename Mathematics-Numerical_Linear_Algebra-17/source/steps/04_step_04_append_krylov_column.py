"""
Incorporate one packed sketched action into the Krylov state. A non-final column is accepted into the basis; the final action is stored as leftover and continuation scale rather than as an extra basis column. The action width must equal n_hess + 1.

The uncorrected Hessenberg is filled one column at a time. Treating the leftover as an (m+1)-st basis column, or accepting every leftover including the last, changes both the packed state and the later similarity restoration.

Returns
-------
1d ndarray: updated Krylov pack with n_hess increased by one
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def append_krylov_column(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Accept a mid-subspace column, or store the final leftover.

    Parameters
    ----------
    state : np.ndarray
        Packed (n, m, n_hess, U.ravel(), H.ravel(), leftover, h_next).
    action : np.ndarray
        Packed (k, h, leftover, scale) with k = n_hess + 1.

    Returns
    -------
    state : np.ndarray
        Updated Krylov pack with n_hess increased by one.

    Raises
    ------
    ValueError
        If either pack is short, the wrong length, or has an invalid
        header, if the Krylov pack is already complete, if the action
        width is not n_hess + 1, or if the continuation scale is not
        positive.
    """
    return np.zeros(1)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _unpack_append_state(pack):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("Krylov pack is too short")
    n, m, n_hess = [int(round(float(v))) for v in p[:3]]
    if n < 2 or m < 1 or n_hess < 0 or n_hess > m:
        raise ValueError("Krylov pack header is invalid")
    need = 3 + n * m + m * m + n + 1
    if p.size != need:
        raise ValueError("Krylov pack length does not match header")
    U = p[3 : 3 + n * m].reshape(n, m)
    H = p[3 + n * m : 3 + n * m + m * m].reshape(m, m)
    leftover = p[3 + n * m + m * m : 3 + n * m + m * m + n]
    h_next = float(p[-1])
    return n, m, n_hess, U, H, leftover, h_next

def _unpack_action(pack, n):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 2:
        raise ValueError("action pack is too short")
    k = int(round(float(p[0])))
    if k < 1:
        raise ValueError("action width must be positive")
    need = 1 + k + n + 1
    if p.size != need:
        raise ValueError("action pack length does not match header")
    h = p[1 : 1 + k]
    leftover = p[1 + k : 1 + k + n]
    scale = float(p[-1])
    return k, h, leftover, scale

def _pack_append_state(n, m, n_hess, U, H, leftover, h_next):
    return np.concatenate(
        (
            [float(n), float(m), float(n_hess)],
            np.asarray(U, dtype=float).ravel(),
            np.asarray(H, dtype=float).ravel(),
            np.asarray(leftover, dtype=float).ravel(),
            [float(h_next)],
        )
    )

def _oracle_append_krylov_column(state, action):
    n, m, n_hess, U, H, leftover, h_next = _unpack_append_state(state)
    if n_hess >= m:
        raise ValueError("Krylov pack is already complete")
    k, h, r, scale = _unpack_action(action, n)
    if k != n_hess + 1:
        raise ValueError("action width must equal n_hess + 1")
    if h.shape != (k,):
        raise ValueError("coefficient length must equal k")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("continuation scale must be positive")
    U = np.array(U, dtype=float, copy=True)
    H = np.array(H, dtype=float, copy=True)
    H[:k, n_hess] = h
    if n_hess + 1 < m:
        H[k, n_hess] = scale
        U[:, n_hess + 1] = r / scale
        leftover = np.zeros(n)
        h_next = 0.0
    else:
        leftover = r / scale
        h_next = scale
    return _pack_append_state(n, m, n_hess + 1, U, H, leftover, h_next)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
n, m, n_hess = 3, 2, 0
U = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
H = np.zeros((2, 2))
state = np.concatenate(([float(n), float(m), float(n_hess)], U.ravel(), H.ravel(), np.zeros(3), [0.0]))
h = np.array([0.5])
r = np.array([0.0, 2.0, 0.0])
action = np.concatenate(([1.0], h, r, [2.0]))
""",
            "call": "append_krylov_column(state, action)",
            "gold_call": "_oracle_append_krylov_column(state, action)",
        },
        {
            "setup": """import numpy as np
n, m, n_hess = 3, 2, 1
U = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
H = np.array([[0.5, 0.0], [2.0, 0.0]])
state = np.concatenate(([float(n), float(m), float(n_hess)], U.ravel(), H.ravel(), np.zeros(3), [0.0]))
h = np.array([0.1, -0.2])
r = np.array([0.3, 0.0, 0.6])
action = np.concatenate(([2.0], h, r, [0.5]))
""",
            "call": "append_krylov_column(state, action)",
            "gold_call": "_oracle_append_krylov_column(state, action)",
        },
        {
            "setup": """import numpy as np
n, m = 2, 1
U = np.array([[1.0], [0.0]])
H = np.zeros((1, 1))
state = np.concatenate(([float(n), float(m), 0.0], U.ravel(), H.ravel(), np.zeros(2), [0.0]))
action = np.concatenate(([1.0], np.array([1.5]), np.array([0.0, 2.0]), [2.0]))
""",
            "call": "append_krylov_column(state, action)",
            "gold_call": "_oracle_append_krylov_column(state, action)",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        append_krylov_column(np.ones(3), np.ones(4))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_append_krylov_column(np.ones(3), np.ones(4))
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
n, m, n_hess = 3, 2, 0
U = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
H = np.zeros((2, 2))
state = np.concatenate(([float(n), float(m), float(n_hess)], U.ravel(), H.ravel(), np.zeros(3), [0.0]))
action = np.concatenate(([2.0], np.array([0.1, 0.2]), np.ones(3), [1.0]))
def run_model():
    try:
        append_krylov_column(state, action)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_append_krylov_column(state, action)
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
