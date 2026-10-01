"""
From a packed sketched instance, Omega-normalize the start vector and pack an m-column Krylov state with that first column accepted and no Hessenberg columns written yet. Pack (n, m, n_hess, U.ravel(), H.ravel(), leftover, h_next) with n_hess = 0. Require 1 <= m < n and d >= m.

Randomized Arnoldi starts from a vector that is unit length in the sketched inner product, not the Euclidean one. Euclidean normalization of b yields a different first column and a different Hessenberg.

Returns
-------
1d ndarray: packed Krylov state with n_hess = 0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def start_krylov_pack(inst_pack: np.ndarray, m: int) -> np.ndarray:
    """Pack an m-column Krylov state with only the start vector accepted.

    Parameters
    ----------
    inst_pack : np.ndarray
        Packed (n, d, A.ravel(), b, Omega.ravel()).
    m : int
        Subspace length, 1 <= m < n, with d >= m.

    Returns
    -------
    state : np.ndarray
        Packed (n, m, n_hess, U.ravel(), H.ravel(), leftover, h_next)
        with n_hess = 0 and U[:, 0] the Omega-normalized start vector.

    Raises
    ------
    ValueError
        If the instance pack is short or the wrong length, if packed
        shapes are not positive, if m is not an integer, if m is not in
        1, ..., n-1, if d < m, or if Omega @ b is zero.
    """
    return np.zeros(1)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _unpack_instance(pack):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 2:
        raise ValueError("instance pack is too short")
    n, d = [int(round(float(v))) for v in p[:2]]
    if min(n, d) < 1:
        raise ValueError("packed shapes must be positive")
    need = 2 + n * n + n + d * n
    if p.size != need:
        raise ValueError("instance pack length does not match header")
    A = p[2 : 2 + n * n].reshape(n, n)
    b = p[2 + n * n : 2 + n * n + n]
    Omega = p[2 + n * n + n :].reshape(d, n)
    return A, b, Omega

def _pack_start_state(n, m, n_hess, U, H, leftover, h_next):
    return np.concatenate(
        (
            [float(n), float(m), float(n_hess)],
            np.asarray(U, dtype=float).ravel(),
            np.asarray(H, dtype=float).ravel(),
            np.asarray(leftover, dtype=float).ravel(),
            [float(h_next)],
        )
    )

def _oracle_start_krylov_pack(inst_pack, m):
    _A, b, Omega = _unpack_instance(inst_pack)
    n = b.shape[0]
    d = Omega.shape[0]
    if not isinstance(m, (int, np.integer)):
        raise ValueError("m must be an integer")
    m = int(m)
    if m < 1 or m >= n:
        raise ValueError("require 1 <= m < n")
    if d < m:
        raise ValueError("require d >= m")
    beta = float(np.linalg.norm(Omega @ b))
    if not np.isfinite(beta) or beta == 0.0:
        raise ValueError("Omega b must be nonzero")
    U = np.zeros((n, m))
    U[:, 0] = b / beta
    H = np.zeros((m, m))
    leftover = np.zeros(n)
    return _pack_start_state(n, m, 0, U, H, leftover, 0.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
n, d, m = 12, 8, 4
s = np.array([8.0, 6.0, 5.0, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15])
rng = np.random.default_rng(7)
Q, _ = np.linalg.qr(rng.standard_normal((n, n)), mode='reduced')
A = 0.5 * ((Q * s) @ Q.T + ((Q * s) @ Q.T).T)
b = rng.standard_normal(n)
Omega = np.random.default_rng(11).standard_normal((d, n))
inst = np.concatenate(([float(n), float(d)], A.ravel(), b, Omega.ravel()))
""",
            "call": "start_krylov_pack(inst, m)",
            "gold_call": "_oracle_start_krylov_pack(inst, m)",
        },
        {
            "setup": """import numpy as np
n, d, m = 4, 3, 2
s = np.array([3.0, 2.0, 0.4, 0.2])
rng = np.random.default_rng(1)
Q, _ = np.linalg.qr(rng.standard_normal((n, n)), mode='reduced')
A = 0.5 * ((Q * s) @ Q.T + ((Q * s) @ Q.T).T)
b = rng.standard_normal(n)
Omega = np.random.default_rng(2).standard_normal((d, n))
inst = np.concatenate(([float(n), float(d)], A.ravel(), b, Omega.ravel()))
""",
            "call": "start_krylov_pack(inst, m)",
            "gold_call": "_oracle_start_krylov_pack(inst, m)",
        },
        {
            "setup": """import numpy as np
n, d = 4, 3
s = np.array([3.0, 2.0, 0.4, 0.2])
rng = np.random.default_rng(1)
Q, _ = np.linalg.qr(rng.standard_normal((n, n)), mode='reduced')
A = 0.5 * ((Q * s) @ Q.T + ((Q * s) @ Q.T).T)
b = rng.standard_normal(n)
Omega = np.random.default_rng(2).standard_normal((d, n))
inst = np.concatenate(([float(n), float(d)], A.ravel(), b, Omega.ravel()))
def run_model():
    try:
        start_krylov_pack(inst, 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_start_krylov_pack(inst, 4)
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
n, d = 4, 2
s = np.array([3.0, 2.0, 0.4, 0.2])
rng = np.random.default_rng(1)
Q, _ = np.linalg.qr(rng.standard_normal((n, n)), mode='reduced')
A = 0.5 * ((Q * s) @ Q.T + ((Q * s) @ Q.T).T)
b = rng.standard_normal(n)
Omega = np.random.default_rng(2).standard_normal((d, n))
inst = np.concatenate(([float(n), float(d)], A.ravel(), b, Omega.ravel()))
def run_model():
    try:
        start_krylov_pack(inst, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_start_krylov_pack(inst, 3)
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
        start_krylov_pack(np.ones(3), 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_start_krylov_pack(np.ones(3), 1)
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
