"""
Evaluate the right-hand side of the Wigner equation (hbar = m = 1) at a packed low-rank state, for a potential supplied only through the separation factors of its central difference potential, and return it as a dense array on the phase-space grid. The state arrives packed as left factors, coefficient matrix and right factors stacked in that order. The factors arrive packed as returned by the separation step: the spatial factors sampled at the spatial nodes, stacked above the dual factors sampled at the nodes of the grid dual to the wave-vector grid. For Nk wave-vector nodes with spacing dk, the dual nodes are y_m = 2*pi*m/(Nk*dk) for the integers m from -floor(Nk/2) to Nk - floor(Nk/2) - 1, in increasing order. For even Nk the unpaired dual node m = -Nk/2 contributes nothing, and the returned array is the real part of the result. Take the spatial derivative with the supplied differentiation operator. Raise ValueError if the state is not two-dimensional, if the grids are not non-empty one-dimensional arrays, if the state's row count does not equal the spatial grid size plus the rank plus the wave-vector grid size, if the differentiation operator is not square and matched to the spatial grid, if the factors are not a two-dimensional array whose row count equals the spatial grid size plus the wave-vector grid size, or if any input contains non-finite entries.

Two mechanisms drive the quasi-distribution: free streaming, and the potential acting through the pseudo-differential operator. The central difference potential enters that operator as a function of position and of the variable dual to the wave vector, so a separated form of it acts on the two sides of the low-rank factorisation independently. In the classical limit the potential term reduces to the derivative of the potential times the wave-vector derivative of the distribution.

Returns
-------
np.ndarray of shape (Nx, Nk), the right-hand side of the evolution equation evaluated at the given low-rank state, dtype float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def wigner_rhs(state: "np.ndarray",
               d_x: "np.ndarray",
               x_grid: "np.ndarray",
               k_grid: "np.ndarray",
               factors: "np.ndarray") -> "np.ndarray":
    '''Evaluate the Wigner right-hand side at a low-rank state.

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
        (Nx + Nk, R) separation factors of the central difference potential:
        R spatial factors at the spatial nodes stacked above R dual factors
        at the dual nodes, in increasing order. R may be zero.

    Returns
    -------
    rhs : np.ndarray
        (Nx, Nk) dense right-hand side evaluated at the state, float64.
    '''
    return rhs  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_wigner_rhs(state: "np.ndarray",
                       d_x: "np.ndarray",
                       x_grid: "np.ndarray",
                       k_grid: "np.ndarray",
                       factors: "np.ndarray") -> "np.ndarray":
    Y = np.asarray(state, dtype=float)
    Dx = np.asarray(d_x, dtype=float)
    xg = np.asarray(x_grid, dtype=float)
    kg = np.asarray(k_grid, dtype=float)
    Fc = np.asarray(factors, dtype=float)
    if Y.ndim != 2:
        raise ValueError("state must be a two-dimensional array")
    if xg.ndim != 1 or kg.ndim != 1 or xg.size < 1 or kg.size < 1:
        raise ValueError("x_grid and k_grid must be non-empty one-dimensional arrays")
    nx, nk, r = xg.size, kg.size, Y.shape[1]
    if r < 1 or Y.shape[0] != nx + r + nk:
        raise ValueError("state rows must equal len(x_grid) + rank + len(k_grid)")
    if Dx.shape != (nx, nx):
        raise ValueError("the differentiation operator must be square and match the spatial grid")
    if Fc.ndim != 2 or Fc.shape[0] != nx + nk:
        raise ValueError("factors must be two-dimensional with len(x_grid) + len(k_grid) rows")
    if not (np.all(np.isfinite(Y)) and np.all(np.isfinite(Dx)) and np.all(np.isfinite(Fc))):
        raise ValueError("inputs must contain only finite values")
    U = Y[:nx, :]
    S = Y[nx:nx + r, :]
    V = Y[nx + r:, :]
    rhs = -(Dx @ U) @ S @ (kg[:, None] * V).T
    if Fc.shape[1] > 0:
        # Y-truncation: move the right factors to the dual variable, multiply by each
        # dual factor, return to the wave vector, and apply 1/i.
        DX = Fc[:nx, :]
        DY = np.fft.ifftshift(Fc[nx:, :], axes=0).copy()
        if nk % 2 == 0:
            DY[nk // 2, :] = 0.0
        coef = np.fft.ifft(V, axis=0)
        for s in range(Fc.shape[1]):
            Ks = np.real(-1j * np.fft.fft(DY[:, s][:, None] * coef, axis=0))
            rhs = rhs + (DX[:, s][:, None] * U) @ S @ Ks.T
    return rhs.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: production grid, rank and harmonic potential ---
        {
            "setup": """import numpy as np
n, P, r = 128, 20.0, 12
g = -P/2.0 + P/n*np.arange(n)
w = 2.0*np.pi*np.fft.fftfreq(n, d=P/n)
D = np.real(np.fft.ifft(1j*w[:, None]*np.fft.fft(np.eye(n), axis=0), axis=0))
y = (2.0*np.pi/P)*np.arange(-(n//2), n - n//2)
_DV = (lambda t: 0.5*t**2)(g[:, None] + 0.5*y[None, :]) - (lambda t: 0.5*t**2)(g[:, None] - 0.5*y[None, :])
_a, _sv, _bt = np.linalg.svd(_DV, full_matrices=False)
_R = int(np.count_nonzero(_sv > 1e-12*_sv[0]))
factors = np.vstack([_a[:, :_R]*np.sqrt(_sv[:_R]), _bt[:_R, :].T*np.sqrt(_sv[:_R])])
th = np.pi/5.0
u = np.cos(th)*g[:, None] - np.sin(th)*g[None, :]
v = np.sin(th)*g[:, None] + np.cos(th)*g[None, :]
A = np.exp(-(u - 1.0)**2/2.0 - 2.0*v**2)/np.pi
Uf, s, Vt = np.linalg.svd(A, full_matrices=False)
state = np.vstack([Uf[:, :r], np.diag(s[:r]), Vt[:r, :].T])
""",
            "call": "wigner_rhs(state, D, g, g, factors)",
            "gold_call": "_oracle_wigner_rhs(state, D, g, g, factors)",
        },
        # --- normal: Gaussian barrier on [0, 60) x [-4, 4), dual spacing pi/4,
        #     separation rank nine ---
        {
            "setup": """import numpy as np
nx, nk, px, pk, r = 48, 32, 60.0, 8.0, 6
xg = px/nx*np.arange(nx)
kg = -pk/2.0 + pk/nk*np.arange(nk)
w = 2.0*np.pi*np.fft.fftfreq(nx, d=px/nx)
Dx = np.real(np.fft.ifft(1j*w[:, None]*np.fft.fft(np.eye(nx), axis=0), axis=0))
y = (2.0*np.pi/pk)*np.arange(-(nk//2), nk - nk//2)
V = lambda t: 0.3*np.exp(-(t - 30.0)**2/2.0)
_DV = (V)(xg[:, None] + 0.5*y[None, :]) - (V)(xg[:, None] - 0.5*y[None, :])
_a, _sv, _bt = np.linalg.svd(_DV, full_matrices=False)
_R = int(np.count_nonzero(_sv > 1e-10*_sv[0]))
factors = np.vstack([_a[:, :_R]*np.sqrt(_sv[:_R]), _bt[:_R, :].T*np.sqrt(_sv[:_R])])
u = np.cos(0.5)*(xg[:, None] - 27.0) - np.sin(0.5)*(kg[None, :] - 0.7)
v = np.sin(0.5)*(xg[:, None] - 27.0) + np.cos(0.5)*(kg[None, :] - 0.7)
A = np.exp(-u**2/8.0 - 8.0*v**2)/np.pi
Uf, s, Vt = np.linalg.svd(A, full_matrices=False)
state = np.vstack([Uf[:, :r], np.diag(s[:r]), Vt[:r, :].T])
""",
            "call": "wigner_rhs(state, Dx, xg, kg, factors)",
            "gold_call": "_oracle_wigner_rhs(state, Dx, xg, kg, factors)",
        },
        # --- edge: cosine potential, whose operator shifts the wave vector by
        #     one half, between grid nodes; odd node count, no unpaired node ---
        {
            "setup": """import numpy as np
nx, nk, px, pk, r = 40, 25, 8.0*np.pi, 12.0, 5
xg = -px/2.0 + px/nx*np.arange(nx)
kg = -pk/2.0 + pk/nk*np.arange(nk)
w = 2.0*np.pi*np.fft.fftfreq(nx, d=px/nx)
Dx = np.real(np.fft.ifft(1j*w[:, None]*np.fft.fft(np.eye(nx), axis=0), axis=0))
y = (2.0*np.pi/pk)*np.arange(-(nk//2), nk - nk//2)
_DV = (np.cos)(xg[:, None] + 0.5*y[None, :]) - (np.cos)(xg[:, None] - 0.5*y[None, :])
_a, _sv, _bt = np.linalg.svd(_DV, full_matrices=False)
_R = int(np.count_nonzero(_sv > 1e-12*_sv[0]))
factors = np.vstack([_a[:, :_R]*np.sqrt(_sv[:_R]), _bt[:_R, :].T*np.sqrt(_sv[:_R])])
u = np.cos(0.3)*(xg[:, None] - 1.0) - np.sin(0.3)*(kg[None, :] - 0.5)
v = np.sin(0.3)*(xg[:, None] - 1.0) + np.cos(0.3)*(kg[None, :] - 0.5)
A = np.exp(-u**2/4.5 - 4.5*v**2)/np.pi
Uf, s, Vt = np.linalg.svd(A, full_matrices=False)
state = np.vstack([Uf[:, :r], np.diag(s[:r]), Vt[:r, :].T])
""",
            "call": "wigner_rhs(state, Dx, xg, kg, factors)",
            "gold_call": "_oracle_wigner_rhs(state, Dx, xg, kg, factors)",
        },
        # --- edge: quartic potential, separation rank two, so the operator
        #     carries a third wave-vector derivative; unequal grids and periods ---
        {
            "setup": """import numpy as np
nx, nk, px, pk, r = 36, 30, 8.0, 14.0, 4
xg = -px/2.0 + px/nx*np.arange(nx)
kg = -pk/2.0 + pk/nk*np.arange(nk)
w = 2.0*np.pi*np.fft.fftfreq(nx, d=px/nx)
Dx = np.real(np.fft.ifft(1j*w[:, None]*np.fft.fft(np.eye(nx), axis=0), axis=0))
y = (2.0*np.pi/pk)*np.arange(-(nk//2), nk - nk//2)
_DV = (lambda t: t**4/4.0)(xg[:, None] + 0.5*y[None, :]) - (lambda t: t**4/4.0)(xg[:, None] - 0.5*y[None, :])
_a, _sv, _bt = np.linalg.svd(_DV, full_matrices=False)
_R = int(np.count_nonzero(_sv > 1e-12*_sv[0]))
factors = np.vstack([_a[:, :_R]*np.sqrt(_sv[:_R]), _bt[:_R, :].T*np.sqrt(_sv[:_R])])
u = np.cos(0.4)*(xg[:, None] - 0.5) - np.sin(0.4)*(kg[None, :] + 0.5)
v = np.sin(0.4)*(xg[:, None] - 0.5) + np.cos(0.4)*(kg[None, :] + 0.5)
A = np.exp(-u**2/1.28 - 1.28*v**2)/np.pi
Uf, s, Vt = np.linalg.svd(A, full_matrices=False)
state = np.vstack([Uf[:, :r], np.diag(s[:r]), Vt[:r, :].T])
""",
            "call": "wigner_rhs(state, Dx, xg, kg, factors)",
            "gold_call": "_oracle_wigner_rhs(state, Dx, xg, kg, factors)",
        },
        # --- edge: coefficient matrix spanning fourteen orders of magnitude,
        #     so any step that divides by it loses all accuracy ---
        {
            "setup": """import numpy as np
n, P, r = 32, 6.0, 4
g = -P/2.0 + P/n*np.arange(n)
w = 2.0*np.pi*np.fft.fftfreq(n, d=P/n)
D = np.real(np.fft.ifft(1j*w[:, None]*np.fft.fft(np.eye(n), axis=0), axis=0))
y = (2.0*np.pi/P)*np.arange(-(n//2), n - n//2)
factors = np.concatenate([g, y])[:, None]
Uf, _ = np.linalg.qr(np.cos(np.outer(g, np.arange(1, r+1))))
Vf, _ = np.linalg.qr(np.sin(np.outer(g, np.arange(1, r+1))) + 0.5)
state = np.vstack([Uf, np.diag([1.0, 1e-5, 1e-10, 1e-14]), Vf])
""",
            "call": "wigner_rhs(state, D, g, g, factors)",
            "gold_call": "_oracle_wigner_rhs(state, D, g, g, factors)",
        },
        # --- boundary: constant potential, empty separation, free streaming only;
        #     smallest sensible grid, odd node count ---
        {
            "setup": """import numpy as np
n, P, r = 7, 2.0, 2
g = -P/2.0 + P/n*np.arange(n)
w = 2.0*np.pi*np.fft.fftfreq(n, d=P/n)
D = np.real(np.fft.ifft(1j*w[:, None]*np.fft.fft(np.eye(n), axis=0), axis=0))
factors = np.zeros((2*n, 0))
A = np.exp(-g[:, None]**2) + 0.4*g[:, None]*g[None, :]
Uf, s, Vt = np.linalg.svd(A, full_matrices=False)
state = np.vstack([Uf[:, :r], np.diag(s[:r]), Vt[:r, :].T])
""",
            "call": "wigner_rhs(state, D, g, g, factors)",
            "gold_call": "_oracle_wigner_rhs(state, D, g, g, factors)",
        },
        # --- structural probe: the result depends only on the separated product,
        #     so rescaling a factor pair or splitting it into two halves leaves it
        #     unchanged, while dropping the potential changes it; as integers ---
        {
            "setup": """import numpy as np
nx, nk, px, pk, r = 24, 20, 10.0, 10.0, 3
xg = -px/2.0 + px/nx*np.arange(nx)
kg = -pk/2.0 + pk/nk*np.arange(nk)
w = 2.0*np.pi*np.fft.fftfreq(nx, d=px/nx)
Dx = np.real(np.fft.ifft(1j*w[:, None]*np.fft.fft(np.eye(nx), axis=0), axis=0))
y = (2.0*np.pi/pk)*np.arange(-(nk//2), nk - nk//2)
DV = np.cos(xg[:, None] + 0.5*y[None, :]) - np.cos(xg[:, None] - 0.5*y[None, :])
a, sv, bt = np.linalg.svd(DV, full_matrices=False)
F = np.vstack([a[:, :1]*np.sqrt(sv[0]), bt[:1, :].T*np.sqrt(sv[0])])
F_scaled = np.vstack([2.5*F[:nx], F[nx:]/2.5])
F_split = np.hstack([np.vstack([F[:nx], 0.5*F[nx:]]), np.vstack([F[:nx], 0.5*F[nx:]])])
A = 1.0/(1.0 + (xg[:, None] - 0.5*kg[None, :])**2)
Uf, s, Vt = np.linalg.svd(A, full_matrices=False)
state = np.vstack([Uf[:, :r], np.diag(s[:r]), Vt[:r, :].T])
def probe(fn):
    base = np.asarray(fn(state, Dx, xg, kg, F), dtype=float)
    sc = np.abs(base).max()
    return [int(np.abs(np.asarray(fn(state, Dx, xg, kg, F_scaled)) - base).max() < 1e-10*sc),
            int(np.abs(np.asarray(fn(state, Dx, xg, kg, F_split)) - base).max() < 1e-10*sc),
            int(np.abs(np.asarray(fn(state, Dx, xg, kg, F[:, :0])) - base).max() > 1e-3*sc)]
def run_model():
    return probe(wigner_rhs)
def run_gold():
    return probe(_oracle_wigner_rhs)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: state row count inconsistent with the grids ---
        {
            "setup": """import numpy as np
n, P, r = 16, 4.0, 2
g = -P/2.0 + P/n*np.arange(n)
D = np.eye(n)
factors = np.ones((2*n, 1))
state = np.ones((n + r + n - 1, r))
def run_model():
    try:
        wigner_rhs(state, D, g, g, factors)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_wigner_rhs(state, D, g, g, factors)
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
factors = np.ones((2*n, 1))
state = np.ones((n + r + n, r))
def run_model():
    try:
        wigner_rhs(state, np.eye(7), g, g, factors)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_wigner_rhs(state, np.eye(7), g, g, factors)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: factors with one row too few ---
        {
            "setup": """import numpy as np
n, P, r = 16, 4.0, 2
g = -P/2.0 + P/n*np.arange(n)
D = np.eye(n)
factors = np.ones((2*n - 1, 1))
state = np.ones((n + r + n, r))
def run_model():
    try:
        wigner_rhs(state, D, g, g, factors)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_wigner_rhs(state, D, g, g, factors)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-finite entries in the factors ---
        {
            "setup": """import numpy as np
n, P, r = 16, 4.0, 2
g = -P/2.0 + P/n*np.arange(n)
D = np.eye(n)
factors = np.full((2*n, 1), np.nan)
state = np.ones((n + r + n, r))
def run_model():
    try:
        wigner_rhs(state, D, g, g, factors)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_wigner_rhs(state, D, g, g, factors)
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
