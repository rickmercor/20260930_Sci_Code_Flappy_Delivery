"""
Expand an order-l Krylov decomposition A U_l = U_l H_l + uhat c_l^T back to order m by m - l sketched Gram-Schmidt steps. Place U_l, then uhat unchanged, as the first l + 1 columns; put H_l in the leading block and c_l in row l; for each new action w = A U[:, k] solve the sketched least-squares problem min ||Omega U[:, :k+1] h - Omega w||_2, subtract, and Omega-normalize. Return the standard completed Krylov pack (n, m, m, U.ravel(), H.ravel(), leftover, h_next). Require l + 1 <= m < n and d >= m.

After a restart the basis is no longer Omega-orthonormal, so the sketched coefficients must come from the pseudoinverse of Omega U rather than its transpose, and the coefficient matrix is no longer Hessenberg. No optional sketched reorthogonalization is applied.

Returns
-------
1d ndarray: completed Krylov pack (n, m, m, U.ravel(), H.ravel(), leftover, h_next)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def expand_krylov_cycle(
    inst_pack: np.ndarray, decomp: np.ndarray, m: int
) -> np.ndarray:
    """Return the completed order-m Krylov pack of one restart cycle.

    Parameters
    ----------
    inst_pack : np.ndarray
        Packed (n, d, A.ravel(), b, Omega.ravel()).
    decomp : np.ndarray
        Packed (n, l, U_l.ravel(), H_l.ravel(), uhat, c_l) satisfying
        A U_l = U_l H_l + uhat c_l^T.
    m : int
        Expansion order, l + 1 <= m < n, with d >= m.

    Returns
    -------
    state : np.ndarray
        Completed Krylov pack (n, m, m, U.ravel(), H.ravel(), leftover,
        h_next) with U[:, :l] = U_l, U[:, l] = uhat kept as given (not
        rescaled), H[:l, :l] = H_l, H[l, :l] = c_l, and columns l, ..., m-1
        of H filled by m - l sketched Gram-Schmidt steps: for each new
        action w = A U[:, k], the coefficients solve
        min || Omega U[:, :k+1] h - Omega w ||_2, the leftover is
        w - U[:, :k+1] h, and the next column (or the final leftover) is
        that vector divided by || Omega leftover ||_2, which is stored as
        the subdiagonal entry (or h_next).

    Raises
    ------
    ValueError
        If either pack is short, the wrong length, has an invalid header,
        or is nonfinite, if the two packs disagree on n, if m is not an
        integer with l + 1 <= m < n, if d < m, or if a sketched leftover
        has zero scale (breakdown).
    """
    return np.zeros(1)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _unpack_expand_instance(pack):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 2:
        raise ValueError("instance pack is too short")
    n, d = [int(round(float(v))) for v in p[:2]]
    if min(n, d) < 1:
        raise ValueError("packed shapes must be positive")
    need = 2 + n * n + n + d * n
    if p.size != need:
        raise ValueError("instance pack length does not match header")
    if not np.all(np.isfinite(p)):
        raise ValueError("instance pack must be finite")
    A = p[2 : 2 + n * n].reshape(n, n)
    Omega = p[2 + n * n + n :].reshape(d, n)
    return n, d, A, Omega


def _unpack_decomp(pack):
    p = np.asarray(pack, dtype=float).reshape(-1)
    if p.size < 2:
        raise ValueError("decomposition pack is too short")
    n, l = [int(round(float(v))) for v in p[:2]]
    if n < 2 or l < 1 or l >= n:
        raise ValueError("decomposition pack header is invalid")
    need = 2 + n * l + l * l + n + l
    if p.size != need:
        raise ValueError("decomposition pack length does not match header")
    if not np.all(np.isfinite(p)):
        raise ValueError("decomposition pack must be finite")
    U_l = p[2 : 2 + n * l].reshape(n, l)
    H_l = p[2 + n * l : 2 + n * l + l * l].reshape(l, l)
    uhat = p[2 + n * l + l * l : 2 + n * l + l * l + n]
    c = p[2 + n * l + l * l + n :]
    return n, l, U_l, H_l, uhat, c


def _oracle_expand_krylov_cycle(inst_pack, decomp, m):
    n, d, A, Omega = _unpack_expand_instance(inst_pack)
    n2, l, U_l, H_l, uhat, c = _unpack_decomp(decomp)
    if n2 != n:
        raise ValueError("instance and decomposition disagree on n")
    if not isinstance(m, (int, np.integer)):
        raise ValueError("m must be an integer")
    m = int(m)
    if m < l + 1 or m >= n:
        raise ValueError("require l + 1 <= m < n")
    if d < m:
        raise ValueError("require d >= m")
    U = np.zeros((n, m))
    H = np.zeros((m, m))
    U[:, :l] = U_l
    U[:, l] = uhat
    H[:l, :l] = H_l
    H[l, :l] = c
    leftover = np.zeros(n)
    h_next = 0.0
    for k in range(l, m):
        w = A @ U[:, k]
        hk, *_ = np.linalg.lstsq(Omega @ U[:, : k + 1], Omega @ w, rcond=None)
        r = w - U[:, : k + 1] @ hk
        scale = float(np.linalg.norm(Omega @ r))
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError("Arnoldi breakdown")
        H[: k + 1, k] = hk
        if k + 1 < m:
            H[k + 1, k] = scale
            U[:, k + 1] = r / scale
        else:
            leftover = r / scale
            h_next = scale
    return np.concatenate(
        (
            [float(n), float(m), float(m)],
            U.ravel(),
            H.ravel(),
            leftover.ravel(),
            [float(h_next)],
        )
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
n, d, l, m = 5, 4, 1, 3
s = np.array([4.0, 3.0, 1.0, 0.5, 0.25])
rng = np.random.default_rng(5)
Q, _ = np.linalg.qr(rng.standard_normal((n, n)), mode='reduced')
A = 0.5 * ((Q * s) @ Q.T + ((Q * s) @ Q.T).T)
b = rng.standard_normal(n)
Omega = np.random.default_rng(6).standard_normal((d, n))
inst = np.concatenate(([float(n), float(d)], A.ravel(), b, Omega.ravel()))
U_l = np.array([[1.0], [0.0], [0.5], [0.0], [0.0]])
H_l = np.array([[2.0]])
uhat = np.array([0.0, 1.0, 0.0, 0.3, 0.0])
c = np.array([0.6])
dec = np.concatenate(([float(n), float(l)], U_l.ravel(), H_l.ravel(), uhat, c))
""",
            "call": "expand_krylov_cycle(inst, dec, m)",
            "gold_call": "_oracle_expand_krylov_cycle(inst, dec, m)",
        },
        {
            "setup": """import numpy as np
n, d, l, m = 6, 5, 2, 4
s = np.array([5.0, 4.0, 3.0, 0.5, 0.4, 0.3])
rng = np.random.default_rng(8)
Q, _ = np.linalg.qr(rng.standard_normal((n, n)), mode='reduced')
A = 0.5 * ((Q * s) @ Q.T + ((Q * s) @ Q.T).T)
b = rng.standard_normal(n)
Omega = np.random.default_rng(9).standard_normal((d, n))
inst = np.concatenate(([float(n), float(d)], A.ravel(), b, Omega.ravel()))
U_l = rng.standard_normal((n, l))
H_l = np.array([[3.0, 0.2], [0.0, 1.5]])
uhat = rng.standard_normal(n)
c = np.array([0.4, -0.1])
dec = np.concatenate(([float(n), float(l)], U_l.ravel(), H_l.ravel(), uhat, c))
""",
            "call": "expand_krylov_cycle(inst, dec, m)",
            "gold_call": "_oracle_expand_krylov_cycle(inst, dec, m)",
        },
        {
            "setup": """import numpy as np
n, d, l, m = 4, 3, 1, 2
s = np.array([3.0, 2.0, 0.4, 0.2])
rng = np.random.default_rng(1)
Q, _ = np.linalg.qr(rng.standard_normal((n, n)), mode='reduced')
A = 0.5 * ((Q * s) @ Q.T + ((Q * s) @ Q.T).T)
b = rng.standard_normal(n)
Omega = np.random.default_rng(2).standard_normal((d, n))
inst = np.concatenate(([float(n), float(d)], A.ravel(), b, Omega.ravel()))
U_l = np.array([[0.8], [0.2], [0.0], [-0.1]])
H_l = np.array([[1.7]])
uhat = np.array([0.1, 0.0, 0.9, 0.2])
c = np.array([0.35])
dec = np.concatenate(([float(n), float(l)], U_l.ravel(), H_l.ravel(), uhat, c))
""",
            "call": "expand_krylov_cycle(inst, dec, m)",
            "gold_call": "_oracle_expand_krylov_cycle(inst, dec, m)",
        },
        {
            "setup": """import numpy as np
n, d, l = 4, 3, 1
s = np.array([3.0, 2.0, 0.4, 0.2])
rng = np.random.default_rng(1)
Q, _ = np.linalg.qr(rng.standard_normal((n, n)), mode='reduced')
A = 0.5 * ((Q * s) @ Q.T + ((Q * s) @ Q.T).T)
b = rng.standard_normal(n)
Omega = np.random.default_rng(2).standard_normal((d, n))
inst = np.concatenate(([float(n), float(d)], A.ravel(), b, Omega.ravel()))
dec = np.concatenate(([float(n), float(l)], np.array([0.8, 0.2, 0.0, -0.1]), [1.7], np.array([0.1, 0.0, 0.9, 0.2]), [0.35]))
def run_model():
    try:
        expand_krylov_cycle(inst, dec, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_expand_krylov_cycle(inst, dec, 1)
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
n, d, l = 4, 2, 1
s = np.array([3.0, 2.0, 0.4, 0.2])
rng = np.random.default_rng(1)
Q, _ = np.linalg.qr(rng.standard_normal((n, n)), mode='reduced')
A = 0.5 * ((Q * s) @ Q.T + ((Q * s) @ Q.T).T)
b = rng.standard_normal(n)
Omega = np.random.default_rng(2).standard_normal((d, n))
inst = np.concatenate(([float(n), float(d)], A.ravel(), b, Omega.ravel()))
dec = np.concatenate(([float(n), float(l)], np.array([0.8, 0.2, 0.0, -0.1]), [1.7], np.array([0.1, 0.0, 0.9, 0.2]), [0.35]))
def run_model():
    try:
        expand_krylov_cycle(inst, dec, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_expand_krylov_cycle(inst, dec, 3)
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
