"""
Advance a packed low-rank state by one time step of the robust two-stage construction, using the tableau recovered from the supplied first-stage weight and the right-hand side of the preceding step for the potential given by the separation factors. Raise ValueError if the state is not two-dimensional, if the grids are not non-empty one-dimensional arrays, if the state's row count does not equal the spatial grid size plus the rank plus the wave-vector grid size, if the differentiation operator is not square and matched to the spatial grid, if h is not a finite positive real number, if any input contains non-finite entries, if the supplied weight admits no tableau, or if an evaluation of the right-hand side or an application of the practical splitting calculation raises.

Building the increment from several evaluations of the vector field is not a matter of substituting a better increment, because each splitting calculation changes the bases as well as the coefficient matrix. The construction used here is the one whose error constants stay independent of the retained singular values. It rests on a stability estimate of the splitting calculation with respect to the space the calculation starts from, and that estimate covers every stage that starts from the same space.

Returns
-------
np.ndarray of shape (Nx + r + Nk, r), the packed state advanced by one time step, in the same layout as the input, dtype float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def robust_psrk_step(state: "np.ndarray",
                     d_x: "np.ndarray",
                     x_grid: "np.ndarray",
                     k_grid: "np.ndarray",
                     factors: "np.ndarray",
                     h: float,
                     b1: float) -> "np.ndarray":
    '''Advance a packed low-rank state by one two-stage step.

    Parameters
    ----------
    state : np.ndarray
        (Nx + r + Nk, r) packed state: left factors, coefficient matrix,
        right factors, stacked in that order.
    d_x : np.ndarray
        (Nx, Nx) differentiation operator for the spatial coordinate.
    x_grid : np.ndarray
        (Nx,) spatial nodes.
    k_grid : np.ndarray
        (Nk,) uniformly spaced wave-vector nodes.
    factors : np.ndarray
        (Nx + Nk, R) separation factors of the central difference potential,
        in the layout the right-hand-side step takes.
    h : float
        Time step, finite and positive.
    b1 : float
        Final weight assigned to the first stage.

    Returns
    -------
    new_state : np.ndarray
        (Nx + r + Nk, r) advanced packed state in the same layout, float64.
    '''
    return new_state  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_robust_psrk_step(state: "np.ndarray",
                             d_x: "np.ndarray",
                             x_grid: "np.ndarray",
                             k_grid: "np.ndarray",
                             factors: "np.ndarray",
                             h: float,
                             b1: float) -> "np.ndarray":
    Y = np.asarray(state, dtype=float)
    Dx = np.asarray(d_x, dtype=float)
    xg = np.asarray(x_grid, dtype=float)
    kg = np.asarray(k_grid, dtype=float)
    if Y.ndim != 2:
        raise ValueError("state must be a two-dimensional array")
    if xg.ndim != 1 or kg.ndim != 1 or xg.size < 1 or kg.size < 1:
        raise ValueError("grids must be non-empty one-dimensional arrays")
    nx, nk, r = xg.size, kg.size, Y.shape[1]
    if r < 1 or Y.shape[0] != nx + r + nk:
        raise ValueError("state rows must equal len(x_grid) + rank + len(k_grid)")
    if Dx.shape != (nx, nx):
        raise ValueError("the differentiation operator must be square and match the spatial grid")
    if isinstance(h, bool) or not isinstance(h, (int, float, np.integer, np.floating)):
        raise ValueError("h must be a real number")
    hh = float(h)
    if not np.isfinite(hh) or hh <= 0.0:
        raise ValueError("h must be finite and positive")
    if not (np.all(np.isfinite(Y)) and np.all(np.isfinite(Dx))):
        raise ValueError("inputs must contain only finite values")
    tab = _oracle_two_stage_tableau(b1)
    a, w1, w2 = float(tab[0]), float(tab[1]), float(tab[2])
    F1 = _oracle_wigner_rhs(Y, Dx, xg, kg, factors)
    stage = _oracle_projector_splitting_step(Y, (a * hh) * F1)
    F2 = _oracle_wigner_rhs(stage, Dx, xg, kg, factors)
    return _oracle_projector_splitting_step(Y, hh * (w1 * F1 + w2 * F2)).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: one production step from the initial state ---
        {
            "setup": """import numpy as np
n, P, r = 128, 20.0, 12
g = -P/2.0 + P/n*np.arange(n)
factors = np.concatenate([g, (2.0*np.pi/P)*np.arange(-(n//2), n - n//2)])[:, None]
w = 2.0*np.pi*np.fft.fftfreq(n, d=P/n)
D = np.real(np.fft.ifft(1j*w[:, None]*np.fft.fft(np.eye(n), axis=0), axis=0))
th = np.pi/5.0
u = np.cos(th)*g[:, None] - np.sin(th)*g[None, :]
v = np.sin(th)*g[:, None] + np.cos(th)*g[None, :]
A = np.exp(-(u - 1.0)**2/2.0 - 2.0*v**2)/np.pi
_Uf, _s, _Vt = np.linalg.svd(A, full_matrices=False)
state = np.vstack([_Uf[:, :r], np.diag(_s[:r]), _Vt[:r, :].T])
h = 2.0/600.0
""",
            "call": "robust_psrk_step(state, D, g, g, factors, h, -1.0)",
            "gold_call": "_oracle_robust_psrk_step(state, D, g, g, factors, h, -1.0)",
        },
        # --- normal: the same step with the reference algorithm's own member ---
        {
            "setup": """import numpy as np
n, P, r = 128, 20.0, 12
g = -P/2.0 + P/n*np.arange(n)
factors = np.concatenate([g, (2.0*np.pi/P)*np.arange(-(n//2), n - n//2)])[:, None]
w = 2.0*np.pi*np.fft.fftfreq(n, d=P/n)
D = np.real(np.fft.ifft(1j*w[:, None]*np.fft.fft(np.eye(n), axis=0), axis=0))
th = np.pi/5.0
u = np.cos(th)*g[:, None] - np.sin(th)*g[None, :]
v = np.sin(th)*g[:, None] + np.cos(th)*g[None, :]
A = np.exp(-(u - 1.0)**2/2.0 - 2.0*v**2)/np.pi
_Uf, _s, _Vt = np.linalg.svd(A, full_matrices=False)
state = np.vstack([_Uf[:, :r], np.diag(_s[:r]), _Vt[:r, :].T])
h = 2.0/600.0
""",
            "call": "robust_psrk_step(state, D, g, g, factors, h, 0.0)",
            "gold_call": "_oracle_robust_psrk_step(state, D, g, g, factors, h, 0.0)",
        },
        # --- edge: step size sixty times the production value ---
        {
            "setup": """import numpy as np
n, P, r = 128, 20.0, 12
g = -P/2.0 + P/n*np.arange(n)
factors = np.concatenate([g, (2.0*np.pi/P)*np.arange(-(n//2), n - n//2)])[:, None]
w = 2.0*np.pi*np.fft.fftfreq(n, d=P/n)
D = np.real(np.fft.ifft(1j*w[:, None]*np.fft.fft(np.eye(n), axis=0), axis=0))
th = np.pi/5.0
u = np.cos(th)*g[:, None] - np.sin(th)*g[None, :]
v = np.sin(th)*g[:, None] + np.cos(th)*g[None, :]
A = np.exp(-(u - 1.0)**2/2.0 - 2.0*v**2)/np.pi
_Uf, _s, _Vt = np.linalg.svd(A, full_matrices=False)
state = np.vstack([_Uf[:, :r], np.diag(_s[:r]), _Vt[:r, :].T])
""",
            "call": "robust_psrk_step(state, D, g, g, factors, 0.2, -1.0)",
            "gold_call": "_oracle_robust_psrk_step(state, D, g, g, factors, 0.2, -1.0)",
        },
        # --- edge: unequal grids, unequal periods, extreme negative weight. The
        #     weight is -1e4 rather than -1e6: at -1e6 the assembled increment
        #     h(b1*F1 + b2*F2) differs two terms of size 1e6 whose sum is of
        #     order one, and the graded array stops being reproducible at the
        #     comparison tolerance ---
        {
            "setup": """import numpy as np
nx, nk, px, pk, r = 16, 24, 4.0, 9.0, 3
xg = -px/2.0 + px/nx*np.arange(nx)
kg = -pk/2.0 + pk/nk*np.arange(nk)
def dmat(n, p):
    w = 2.0*np.pi*np.fft.fftfreq(n, d=p/n)
    return np.real(np.fft.ifft(1j*w[:, None]*np.fft.fft(np.eye(n), axis=0), axis=0))
Dx = dmat(nx, px)
factors = np.concatenate([xg, (2.0*np.pi/pk)*np.arange(-(nk//2), nk - nk//2)])[:, None]
A = 1.0/(1.0 + (xg[:, None] - kg[None, :])**2) + 0.3*np.cos(xg)[:, None]*np.sin(2.0*kg)[None, :]
_Uf, _s, _Vt = np.linalg.svd(A, full_matrices=False)
state = np.vstack([_Uf[:, :r], np.diag(_s[:r]), _Vt[:r, :].T])
""",
            "call": "robust_psrk_step(state, Dx, xg, kg, factors, 0.01, -1e4)",
            "gold_call": "_oracle_robust_psrk_step(state, Dx, xg, kg, factors, 0.01, -1e4)",
        },
        # --- edge: coefficient matrix spanning fourteen orders of magnitude ---
        {
            "setup": """import numpy as np
n, P, r = 32, 6.0, 4
g = -P/2.0 + P/n*np.arange(n)
factors = np.concatenate([g, (2.0*np.pi/P)*np.arange(-(n//2), n - n//2)])[:, None]
w = 2.0*np.pi*np.fft.fftfreq(n, d=P/n)
D = np.real(np.fft.ifft(1j*w[:, None]*np.fft.fft(np.eye(n), axis=0), axis=0))
Uf, _ = np.linalg.qr(np.cos(np.outer(g, np.arange(1, r+1))))
Vf, _ = np.linalg.qr(np.sin(np.outer(g, np.arange(1, r+1))) + 0.5)
state = np.vstack([Uf, np.diag([1.0, 1e-5, 1e-10, 1e-14]), Vf])
""",
            "call": "robust_psrk_step(state, D, g, g, factors, 1e-3, -1.0)",
            "gold_call": "_oracle_robust_psrk_step(state, D, g, g, factors, 1e-3, -1.0)",
        },
        # --- boundary: the returned step must differ from the variant that
        #     propagates each stage into the next, as integers ---
        {
            "setup": """import numpy as np
n, P, r = 64, 12.0, 6
g = -P/2.0 + P/n*np.arange(n)
factors = np.concatenate([g, (2.0*np.pi/P)*np.arange(-(n//2), n - n//2)])[:, None]
w = 2.0*np.pi*np.fft.fftfreq(n, d=P/n)
D = np.real(np.fft.ifft(1j*w[:, None]*np.fft.fft(np.eye(n), axis=0), axis=0))
th = np.pi/5.0
u = np.cos(th)*g[:, None] - np.sin(th)*g[None, :]
v = np.sin(th)*g[:, None] + np.cos(th)*g[None, :]
A = np.exp(-(u - 1.0)**2/2.0 - 2.0*v**2)/np.pi
_Uf, _s, _Vt = np.linalg.svd(A, full_matrices=False)
state = np.vstack([_Uf[:, :r], np.diag(_s[:r]), _Vt[:r, :].T])
h = 0.01
def rhs(Y):
    U = Y[:n, :]; S = Y[n:n+r, :]; V = Y[n+r:, :]
    return -(D @ U) @ S @ (g[:, None]*V).T + (g[:, None]*U) @ S @ (D @ V).T
def split(Y, dA):
    U = Y[:n, :]; S = Y[n:n+r, :]; V = Y[n+r:, :]
    K1 = U @ S + dA @ V
    U1, Sh = np.linalg.qr(K1)
    d = np.sign(np.diag(Sh)); d[d == 0.0] = 1.0
    U1, Sh = U1*d, (Sh.T*d).T
    St = Sh - U1.T @ dA @ V
    L1 = V @ St.T + dA.T @ U1
    V1, Rt = np.linalg.qr(L1)
    e = np.sign(np.diag(Rt)); e[e == 0.0] = 1.0
    V1, Rt = V1*e, (Rt.T*e).T
    return np.vstack([U1, Rt.T, V1])
def chained(Y, a, w1, w2):
    F1 = rhs(Y); S1 = split(Y, (a*h)*F1)
    F2 = rhs(S1)
    return split(S1, h*(w1*F1 + w2*F2))
def probe(fn):
    out = []
    for b1 in [-1.0, 0.0, 0.5]:
        b2 = 1.0 - b1; a = 0.5/b2
        got = np.asarray(fn(state, D, g, g, factors, h, b1), dtype=float)
        out.append(int(np.abs(got - chained(state, a, b1, b2)).max() > 1e-8))
    return out
def run_model():
    return probe(robust_psrk_step)
def run_gold():
    return probe(_oracle_robust_psrk_step)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-positive step size ---
        {
            "setup": """import numpy as np
n, P, r = 16, 4.0, 2
g = -P/2.0 + P/n*np.arange(n)
factors = np.concatenate([g, (2.0*np.pi/P)*np.arange(-(n//2), n - n//2)])[:, None]
D = np.eye(n)
state = np.zeros((n + r + n, r))
def run_model():
    try:
        robust_psrk_step(state, D, g, g, factors, 0.0, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_robust_psrk_step(state, D, g, g, factors, 0.0, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: weight admitting no tableau ---
        {
            "setup": """import numpy as np
n, P, r = 16, 4.0, 2
g = -P/2.0 + P/n*np.arange(n)
factors = np.concatenate([g, (2.0*np.pi/P)*np.arange(-(n//2), n - n//2)])[:, None]
D = np.eye(n)
state = np.zeros((n + r + n, r))
def run_model():
    try:
        robust_psrk_step(state, D, g, g, factors, 0.01, 2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_robust_psrk_step(state, D, g, g, factors, 0.01, 2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: differentiation operator not matched to the spatial grid ---
        {
            "setup": """import numpy as np
n, P, r = 16, 4.0, 2
g = -P/2.0 + P/n*np.arange(n)
factors = np.concatenate([g, (2.0*np.pi/P)*np.arange(-(n//2), n - n//2)])[:, None]
D = np.eye(n)
state = np.zeros((n + r + n, r))
def run_model():
    try:
        robust_psrk_step(state, np.eye(7), g, g, factors, 0.01, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_robust_psrk_step(state, np.eye(7), g, g, factors, 0.01, -1.0)
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
