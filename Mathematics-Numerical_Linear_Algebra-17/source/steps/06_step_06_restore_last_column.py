"""
Using those coefficients, restore last-vector similarity on the uncorrected Hessenberg and return the corrected m-by-m matrix. Only the last column changes.

The implementable restoration changes only the last Hessenberg column. Updating every column, or leaving H uncorrected, yields a different spectrum.

Returns
-------
ndarray of shape (m, m): corrected Hessenberg
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def restore_last_column(state: np.ndarray, hhat: np.ndarray) -> np.ndarray:
    """Return the corrected Hessenberg after a last-column restoration.

    Parameters
    ----------
    state : np.ndarray
        Completed Krylov pack (n_hess = m).
    hhat : np.ndarray
        Euclidean leftover coefficients, length m.

    Returns
    -------
    Hbar : np.ndarray
        Corrected m-by-m Hessenberg. Only the last column changes.

    Raises
    ------
    ValueError
        If the Krylov pack is short, the wrong length, or has an invalid
        header, if n_hess is not m, or if hhat does not have length m
        or is nonfinite.
    """
    return np.zeros(1)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _unpack_restore_state(pack):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 3:
        raise ValueError("Krylov pack is too short")
    n, m, n_hess = [int(round(float(v))) for v in p[:3]]
    if n < 2 or m < 1 or n_hess < 0 or n_hess > m:
        raise ValueError("Krylov pack header is invalid")
    need = 3 + n * m + m * m + n + 1
    if p.size != need:
        raise ValueError("Krylov pack length does not match header")
    H = p[3 + n * m : 3 + n * m + m * m].reshape(m, m)
    h_next = float(p[-1])
    return n, m, n_hess, H, h_next

def _oracle_restore_last_column(state, hhat):
    _n, m, n_hess, H, h_next = _unpack_restore_state(state)
    if n_hess != m:
        raise ValueError("Krylov pack is incomplete")
    hhat = np.asarray(hhat, dtype=float).reshape(-1)
    if hhat.shape != (m,):
        raise ValueError("hhat must have length m")
    if not np.all(np.isfinite(hhat)):
        raise ValueError("hhat must be finite")
    Hbar = np.array(H, dtype=float, copy=True)
    Hbar[:, -1] = Hbar[:, -1] + h_next * hhat
    return Hbar

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
n, m, n_hess = 3, 2, 2
U = np.eye(3, 2)
H = np.array([[1.0, 2.0], [3.0, 4.0]])
state = np.concatenate(([float(n), float(m), float(n_hess)], U.ravel(), H.ravel(), np.ones(3), [0.5]))
hhat = np.array([0.2, -0.1])
""",
            "call": "restore_last_column(state, hhat)",
            "gold_call": "_oracle_restore_last_column(state, hhat)",
        },
        {
            "setup": """import numpy as np
n, m, n_hess = 2, 1, 1
U = np.array([[1.0], [0.0]])
H = np.array([[2.0]])
state = np.concatenate(([float(n), float(m), float(n_hess)], U.ravel(), H.ravel(), np.array([0.0, 1.0]), [1.5]))
hhat = np.array([0.4])
""",
            "call": "restore_last_column(state, hhat)",
            "gold_call": "_oracle_restore_last_column(state, hhat)",
        },
        {
            "setup": """import numpy as np
n, m, n_hess = 3, 2, 1
U = np.eye(3, 2)
H = np.array([[1.0, 2.0], [3.0, 4.0]])
state = np.concatenate(([float(n), float(m), float(n_hess)], U.ravel(), H.ravel(), np.ones(3), [0.5]))
hhat = np.array([0.2, -0.1])
def run_model():
    try:
        restore_last_column(state, hhat)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_restore_last_column(state, hhat)
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
n, m, n_hess = 3, 2, 2
U = np.eye(3, 2)
H = np.array([[1.0, 2.0], [3.0, 4.0]])
state = np.concatenate(([float(n), float(m), float(n_hess)], U.ravel(), H.ravel(), np.ones(3), [0.5]))
hhat = np.array([0.2])
def run_model():
    try:
        restore_last_column(state, hhat)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_restore_last_column(state, hhat)
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
        restore_last_column(np.ones(3), np.ones(2))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_restore_last_column(np.ones(3), np.ones(2))
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
