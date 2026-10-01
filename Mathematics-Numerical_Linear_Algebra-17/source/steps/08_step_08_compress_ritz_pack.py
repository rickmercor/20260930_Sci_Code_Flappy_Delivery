"""
Compress a completed, corrected order-m sketched Krylov decomposition to order l for a restart. Take the completed Krylov pack, the Euclidean leftover coefficients hhat, and the corrected matrix Hbar; keep the orthonormal Schur vectors V_l of Hbar for its l largest eigenvalues (descending, each column signed so its largest-magnitude entry is positive); return the pack (n, l, (U V_l).ravel(), (V_l^T Hbar V_l).ravel(), leftover - U hhat, V_l^T (h_next e_m)). The carried vector is not rescaled. Require 1 <= l < m, real eigenvalues, and a gap between the l-th and (l+1)-th eigenvalues.

Krylov-Schur restarting compresses onto the Schur vectors of the wanted Ritz values. In the similarity-restoring variant the vector carried across the restart is the Euclidean-orthogonal corrected residual, kept at its own scale, and the continuation row is the transformed last row of the corrected decomposition.

Returns
-------
1d ndarray: packed (n, l, U_l.ravel(), S_l.ravel(), uhat, c_l)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compress_ritz_pack(
    state: np.ndarray, hhat: np.ndarray, Hbar: np.ndarray, l: int
) -> np.ndarray:
    """Pack the order-l Krylov decomposition that starts the next restart cycle.

    Parameters
    ----------
    state : np.ndarray
        Completed Krylov pack (n, m, n_hess, U.ravel(), H.ravel(),
        leftover, h_next) with n_hess = m.
    hhat : np.ndarray
        Euclidean leftover coefficients, length m.
    Hbar : np.ndarray
        Corrected m-by-m matrix, equal to H with h_next * hhat added to
        its last column.
    l : int
        Restart order, 1 <= l < m.

    Returns
    -------
    decomp : np.ndarray
        Packed (n, l, U_l.ravel(), S_l.ravel(), uhat, c_l) where
        U_l = U @ V_l, S_l = V_l.T @ Hbar @ V_l, uhat = leftover - U @ hhat
        (carried unnormalized), and c_l = V_l.T @ (h_next * e_m). V_l holds
        the orthonormal Schur vectors of Hbar for its l largest eigenvalues,
        ordered so the diagonal of S_l decreases; each column of V_l is
        signed so that its entry of largest magnitude (lowest index on ties)
        is positive.

    Raises
    ------
    ValueError
        If the Krylov pack is short, the wrong length, or has an invalid
        header, if n_hess is not m, if hhat or Hbar has the wrong shape or
        is nonfinite, if Hbar is not the last-column restoration of H, if l
        is not an integer in 1, ..., m-1, if Hbar has a non-real eigenvalue,
        or if the l-th and (l+1)-th largest eigenvalues coincide.
    """
    return np.zeros(1)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _unpack_compress_state(pack):
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


def _ordered_schur_vectors(M, l):
    vals, vecs = np.linalg.eig(M)
    if np.any(np.abs(vals.imag) > 1e-8):
        raise ValueError("corrected matrix must have real eigenvalues")
    vals = vals.real
    order = np.argsort(-vals, kind="stable")
    if abs(vals[order[l - 1]] - vals[order[l]]) <= 1e-12:
        raise ValueError("wanted and unwanted Ritz values must be separated")
    V, _ = np.linalg.qr(vecs[:, order[:l]].real)
    for j in range(l):
        i = int(np.argmax(np.abs(V[:, j])))
        if V[i, j] < 0.0:
            V[:, j] = -V[:, j]
    return V


def _oracle_compress_ritz_pack(state, hhat, Hbar, l):
    n, m, n_hess, U, H, leftover, h_next = _unpack_compress_state(state)
    if n_hess != m:
        raise ValueError("Krylov pack is incomplete")
    hhat = np.asarray(hhat, dtype=float).reshape(-1)
    if hhat.shape != (m,) or not np.all(np.isfinite(hhat)):
        raise ValueError("hhat must be a finite vector of length m")
    Hbar = np.asarray(Hbar, dtype=float)
    if Hbar.shape != (m, m) or not np.all(np.isfinite(Hbar)):
        raise ValueError("Hbar must be a finite m-by-m matrix")
    expected = np.array(H, dtype=float, copy=True)
    expected[:, -1] = expected[:, -1] + h_next * hhat
    if not np.allclose(Hbar, expected, rtol=1e-9, atol=1e-9):
        raise ValueError("Hbar must be the last-column restoration of H")
    if not isinstance(l, (int, np.integer)):
        raise ValueError("l must be an integer")
    l = int(l)
    if l < 1 or l >= m:
        raise ValueError("require 1 <= l < m")
    V = _ordered_schur_vectors(Hbar, l)
    S = V.T @ Hbar @ V
    U_l = U @ V
    uhat = leftover - U @ hhat
    c = V.T @ (h_next * np.eye(m)[m - 1])
    return np.concatenate(
        ([float(n), float(l)], U_l.ravel(), S.ravel(), uhat.ravel(), c.ravel())
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
n, m, n_hess = 4, 3, 3
U = np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.2], [0.3, 0.0, 1.0], [0.0, 0.2, 0.1]])
H = np.array([[3.0, 0.5, 0.2], [0.4, 2.0, 0.3], [0.0, 0.6, 1.0]])
leftover = np.array([0.2, -0.1, 0.4, 0.3])
h_next = 0.8
state = np.concatenate(([float(n), float(m), float(n_hess)], U.ravel(), H.ravel(), leftover, [h_next]))
hhat = np.array([0.1, -0.2, 0.05])
Hbar = H.copy()
Hbar[:, -1] += h_next * hhat
""",
            "call": "compress_ritz_pack(state, hhat, Hbar, 2)",
            "gold_call": "_oracle_compress_ritz_pack(state, hhat, Hbar, 2)",
        },
        {
            "setup": """import numpy as np
n, m, n_hess = 3, 2, 2
U = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.5]])
H = np.array([[2.0, 1.0], [0.5, 1.0]])
leftover = np.array([0.0, 0.0, 1.0])
h_next = 0.4
state = np.concatenate(([float(n), float(m), float(n_hess)], U.ravel(), H.ravel(), leftover, [h_next]))
hhat = np.array([0.0, 0.5])
Hbar = H.copy()
Hbar[:, -1] += h_next * hhat
""",
            "call": "compress_ritz_pack(state, hhat, Hbar, 1)",
            "gold_call": "_oracle_compress_ritz_pack(state, hhat, Hbar, 1)",
        },
        {
            "setup": """import numpy as np
n, m, n_hess = 5, 4, 4
rng = np.random.default_rng(3)
U = np.eye(5, 4) + 0.1 * rng.standard_normal((5, 4))
H = np.array([[4.0, 0.5, 0.1, 0.2], [0.5, 3.0, 0.4, 0.1], [0.0, 0.4, 2.0, 0.3], [0.0, 0.0, 0.3, 1.0]])
leftover = rng.standard_normal(5)
h_next = 0.7
state = np.concatenate(([float(n), float(m), float(n_hess)], U.ravel(), H.ravel(), leftover, [h_next]))
hhat = np.array([0.05, -0.05, 0.1, 0.0])
Hbar = H.copy()
Hbar[:, -1] += h_next * hhat
""",
            "call": "compress_ritz_pack(state, hhat, Hbar, 2)",
            "gold_call": "_oracle_compress_ritz_pack(state, hhat, Hbar, 2)",
        },
        {
            "setup": """import numpy as np
n, m, n_hess = 3, 2, 2
U = np.eye(3, 2)
H = np.array([[2.0, 1.0], [0.5, 1.0]])
state = np.concatenate(([float(n), float(m), float(n_hess)], U.ravel(), H.ravel(), np.ones(3), [0.4]))
hhat = np.array([0.0, 0.5])
Hbar = H.copy()
Hbar[:, -1] += 0.4 * hhat
def run_model():
    try:
        compress_ritz_pack(state, hhat, Hbar, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compress_ritz_pack(state, hhat, Hbar, 2)
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
H = np.array([[2.0, 1.0], [0.5, 1.0]])
state = np.concatenate(([float(n), float(m), float(n_hess)], U.ravel(), H.ravel(), np.ones(3), [0.4]))
hhat = np.array([0.0, 0.5])
def run_model():
    try:
        compress_ritz_pack(state, hhat, H, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compress_ritz_pack(state, hhat, H, 1)
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
