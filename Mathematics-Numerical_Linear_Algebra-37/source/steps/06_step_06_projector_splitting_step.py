"""
Apply one practical projector-splitting calculation to a packed low-rank state given a prescribed increment array, and return the updated packed state in the same layout. Both orthonormal factorisations must be resolved to a unique convention by requiring the diagonal of the triangular factor to be non-negative, transferring any sign change to the orthonormal factor. Raise ValueError if the state or increment is not two-dimensional, if the state's row count does not equal the increment's row count plus the rank plus the increment's column count, if any input contains non-finite entries, or if either matrix being factorised has collapsed to rounding error, meaning its largest singular value is at most 1e-13 times the sum of the Frobenius norms of the two terms forming it, or has lost full column rank, meaning its smallest singular value is at most 1e-15 times its largest.

The differential equations governing the factors of a low-rank approximation contain the inverse of the small coefficient matrix. The projector-splitting integrator avoids that inverse. It is used here in its practical form: the prescribed increment defines a straight path from the current state to the current state plus the increment, and the Lie-Trotter splitting of the tangent-space projection along that path is carried out in its standard K, S, L order, each substep solved exactly.

Returns
-------
np.ndarray of shape (Nx + r + Nk, r), the updated packed state in the same layout as the input, dtype float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def projector_splitting_step(state: "np.ndarray", increment: "np.ndarray") -> "np.ndarray":
    '''Apply one practical projector-splitting calculation.

    Parameters
    ----------
    state : np.ndarray
        (Nx + r + Nk, r) packed state: left factors, coefficient matrix,
        right factors, stacked in that order.
    increment : np.ndarray
        (Nx, Nk) prescribed increment of the ambient solution.

    Returns
    -------
    new_state : np.ndarray
        (Nx + r + Nk, r) updated packed state in the same layout, float64.
    '''
    return new_state  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_projector_splitting_step(state: "np.ndarray",
                                     increment: "np.ndarray") -> "np.ndarray":
    Y = np.asarray(state, dtype=float)
    dA = np.asarray(increment, dtype=float)
    if Y.ndim != 2 or dA.ndim != 2:
        raise ValueError("state and increment must be two-dimensional arrays")
    r = Y.shape[1]
    nx, nk = dA.shape
    if r < 1 or Y.shape[0] != nx + r + nk:
        raise ValueError("state rows must equal increment rows + rank + increment columns")
    if not (np.all(np.isfinite(Y)) and np.all(np.isfinite(dA))):
        raise ValueError("inputs must contain only finite values")
    U = Y[:nx, :]
    S = Y[nx:nx + r, :]
    V = Y[nx + r:, :]

    def _h_factor(term_a, term_b):
        M = term_a + term_b
        if M.shape[0] < M.shape[1]:
            raise ValueError("factorised matrix is not of full column rank")
        scale = np.linalg.norm(term_a) + np.linalg.norm(term_b)
        sv = np.linalg.svd(M, compute_uv=False)
        if scale <= 0.0 or sv[0] <= 1e-13 * scale:
            raise ValueError("factorised matrix has cancelled to rounding error")
        if sv[-1] <= 1e-15 * sv[0]:
            raise ValueError("factorised matrix is not of full column rank")
        Q, R = np.linalg.qr(M)
        d = np.sign(np.diag(R))
        d[d == 0.0] = 1.0
        return Q * d, (R.T * d).T

    U1, Shat = _h_factor(U @ S, dA @ V)
    Stil = Shat - U1.T @ dA @ V
    V1, Rt = _h_factor(V @ Stil.T, dA.T @ U1)
    return np.vstack([U1, Rt.T, V1]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: production state with a production-sized increment ---
        {
            "setup": """import numpy as np
n, P, r = 128, 20.0, 12
g = -P/2.0 + P/n*np.arange(n)
w = 2.0*np.pi*np.fft.fftfreq(n, d=P/n)
D = np.real(np.fft.ifft(1j*w[:, None]*np.fft.fft(np.eye(n), axis=0), axis=0))
th = np.pi/5.0
u = np.cos(th)*g[:, None] - np.sin(th)*g[None, :]
v = np.sin(th)*g[:, None] + np.cos(th)*g[None, :]
A = np.exp(-(u - 1.0)**2/2.0 - 2.0*v**2)/np.pi
Uf, s, Vt = np.linalg.svd(A, full_matrices=False)
_Uf, _s, _Vt = np.linalg.svd(A, full_matrices=False)
state = np.vstack([_Uf[:, :r], np.diag(_s[:r]), _Vt[:r, :].T])
Y = Uf[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]
increment = (2.0/600.0/4.0)*(-(D @ Y)*g[None, :] + g[:, None]*(Y @ D.T))
""",
            "call": "projector_splitting_step(state, increment)",
            "gold_call": "_oracle_projector_splitting_step(state, increment)",
        },
        # --- boundary: vanishing increment, which must return the state ---
        {
            "setup": """import numpy as np
n, P, r = 128, 20.0, 12
g = -P/2.0 + P/n*np.arange(n)
th = np.pi/5.0
u = np.cos(th)*g[:, None] - np.sin(th)*g[None, :]
v = np.sin(th)*g[:, None] + np.cos(th)*g[None, :]
A = np.exp(-(u - 1.0)**2/2.0 - 2.0*v**2)/np.pi
_Uf, _s, _Vt = np.linalg.svd(A, full_matrices=False)
state = np.vstack([_Uf[:, :r], np.diag(_s[:r]), _Vt[:r, :].T])
increment = np.zeros((n, n))
""",
            "call": "projector_splitting_step(state, increment)",
            "gold_call": "_oracle_projector_splitting_step(state, increment)",
        },
        # --- edge: increment two hundred times the production size ---
        {
            "setup": """import numpy as np
n, P, r = 128, 20.0, 12
g = -P/2.0 + P/n*np.arange(n)
w = 2.0*np.pi*np.fft.fftfreq(n, d=P/n)
D = np.real(np.fft.ifft(1j*w[:, None]*np.fft.fft(np.eye(n), axis=0), axis=0))
th = np.pi/5.0
u = np.cos(th)*g[:, None] - np.sin(th)*g[None, :]
v = np.sin(th)*g[:, None] + np.cos(th)*g[None, :]
A = np.exp(-(u - 1.0)**2/2.0 - 2.0*v**2)/np.pi
Uf, s, Vt = np.linalg.svd(A, full_matrices=False)
_Uf, _s, _Vt = np.linalg.svd(A, full_matrices=False)
state = np.vstack([_Uf[:, :r], np.diag(_s[:r]), _Vt[:r, :].T])
Y = Uf[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]
increment = 0.2*(-(D @ Y)*g[None, :] + g[:, None]*(Y @ D.T))
""",
            "call": "projector_splitting_step(state, increment)",
            "gold_call": "_oracle_projector_splitting_step(state, increment)",
        },
        # --- edge: coefficient matrix spanning fourteen orders of magnitude,
        #     the regime the construction is designed to tolerate. The returned
        #     factorisation is graded through the reconstruction, its shape and
        #     the orthonormality of both bases, because at a smallest singular
        #     value of 1e-14 the trailing basis column is fixed only to about
        #     eps times the condition number of the factorised matrix, while the
        #     projection the step specifies is determined to rounding ---
        {
            "setup": """import numpy as np
n, P, r = 32, 6.0, 4
g = -P/2.0 + P/n*np.arange(n)
Uf, _ = np.linalg.qr(np.cos(np.outer(g, np.arange(1, r+1))))
Vf, _ = np.linalg.qr(np.sin(np.outer(g, np.arange(1, r+1))) + 0.5)
state = np.vstack([Uf, np.diag([1.0, 1e-5, 1e-10, 1e-14]), Vf])
increment = 1e-3*np.outer(np.cos(3.0*g), np.sin(2.0*g))
def probe(fn):
    out = np.asarray(fn(state, increment), dtype=float)
    U1, S1, V1 = out[:n, :], out[n:n+r, :], out[n+r:, :]
    rec = U1 @ S1 @ V1.T
    flags = [int(out.shape == (n + r + n, r)),
             int(np.abs(U1.T @ U1 - np.eye(r)).max() < 1e-10),
             int(np.abs(V1.T @ V1 - np.eye(r)).max() < 1e-10)]
    return np.concatenate([rec.ravel(), np.array(flags, dtype=float)])
def run_model():
    return probe(projector_splitting_step)
def run_gold():
    return probe(_oracle_projector_splitting_step)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- edge: increment cancelling half the state, rectangular grids ---
        {
            "setup": """import numpy as np
nx, nk, r = 20, 13, 5
xa = np.linspace(-2.0, 2.0, nx)
yb = np.linspace(-3.0, 3.0, nk)
A = 1.0/(1.0 + (xa[:, None] - yb[None, :])**2)
Uf, s, Vt = np.linalg.svd(A, full_matrices=False)
_Uf, _s, _Vt = np.linalg.svd(A, full_matrices=False)
state = np.vstack([_Uf[:, :r], np.diag(_s[:r]), _Vt[:r, :].T])
increment = -0.5*(Uf[:, :r] @ np.diag(s[:r]) @ Vt[:r, :])
""",
            "call": "projector_splitting_step(state, increment)",
            "gold_call": "_oracle_projector_splitting_step(state, increment)",
        },
        # --- boundary: rank-admissibility, including an increment that
        #     annihilates the state so the factorisation is pure rounding ---
        {
            "setup": """import numpy as np
nx, nk, r = 20, 13, 5
xa = np.linspace(-2.0, 2.0, nx)
yb = np.linspace(-3.0, 3.0, nk)
A = 1.0/(1.0 + (xa[:, None] - yb[None, :])**2)
Uf, s, Vt = np.linalg.svd(A, full_matrices=False)
_Uf, _s, _Vt = np.linalg.svd(A, full_matrices=False)
state = np.vstack([_Uf[:, :r], np.diag(_s[:r]), _Vt[:r, :].T])
Y = Uf[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]
incs = [-Y, -0.5*Y, 0.0*Y, 1e-3*np.outer(np.cos(xa), np.sin(yb))]
def probe(fn):
    out = []
    for dA in incs:
        try:
            out.append(int(np.asarray(fn(state, dA)).shape[0]))
        except ValueError:
            out.append(-1)
        except Exception:
            out.append(-2)
    return out
def run_model():
    return probe(projector_splitting_step)
def run_gold():
    return probe(_oracle_projector_splitting_step)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: increment shape inconsistent with the state ---
        {
            "setup": """import numpy as np
nx, nk, r = 12, 9, 3
state = np.zeros((nx + r + nk, r))
state[:nx, :] = np.linalg.qr(np.arange(1.0, nx*r + 1.0).reshape(nx, r))[0]
state[nx:nx+r, :] = np.diag([3.0, 2.0, 1.0])
state[nx+r:, :] = np.linalg.qr(np.cos(np.arange(nk*r, dtype=float)).reshape(nk, r))[0]
def run_model():
    try:
        projector_splitting_step(state, np.zeros((nx, nk + 1)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_projector_splitting_step(state, np.zeros((nx, nk + 1)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: one-dimensional increment ---
        {
            "setup": """import numpy as np
nx, nk, r = 12, 9, 3
state = np.zeros((nx + r + nk, r))
def run_model():
    try:
        projector_splitting_step(state, np.zeros(nx))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_projector_splitting_step(state, np.zeros(nx))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-finite increment ---
        {
            "setup": """import numpy as np
nx, nk, r = 12, 9, 3
state = np.zeros((nx + r + nk, r))
def run_model():
    try:
        projector_splitting_step(state, np.full((nx, nk), np.inf))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_projector_splitting_step(state, np.full((nx, nk), np.inf))
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
