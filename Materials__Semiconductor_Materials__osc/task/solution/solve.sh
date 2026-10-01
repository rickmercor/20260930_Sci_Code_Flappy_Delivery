#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

"""Canonical implementations for Material_Science-Semiconductor_Materials-10 (rebuild).
Source: arXiv:2609.05170 (CC BY 4.0) -- photocurrent versus effective voltage in organic
solar cells; one-dimensional drift-diffusion with Scharfetter-Gummel fluxes.

THE ONLY PLACE THE SCIENCE IS WRITTEN. build_osc.py slices this into sub_problems/NN_name.py.

Oracle chaining rule: every * that needs an earlier step calls that step's 
twin, NEVER the public name.
"""
import numpy as np
from scipy.linalg import solve_banded

_Q = 1.602176634e-19
_KB = 1.380649e-23
_EPS0 = 8.8541878128e-12


def _phys():
    """Physical constants. A function, not module-level state: the harness keeps
    imports and function definitions and drops module-level assignments."""
    return 1.602176634e-19, 1.380649e-23, 8.8541878128e-12

# declared device configuration (SI)
_SPEC = dict(T=300.0, d=100e-9, Eg=1.4, Phi=0.25, mu_n=2e-8, mu_p=2e-8,
             eps_r=3.5, Nc=1e26, Nv=1e26, Gex=1e28, gamma=1.0)


def _tridiag(lo, di, up, rhs):
    ab = np.zeros((3, len(di)))
    ab[0, 1:] = up[:-1]
    ab[1] = di
    ab[2, :-1] = lo[1:]
    return solve_banded((1, 1), ab, rhs)


def _tridiag_dirichlet(lo, di, up, r, left, right):
    """Solve the interior rows only, with the two end values pinned exactly.

    Putting the Dirichlet rows inside the banded system gives them a unit
    diagonal beside interior diagonals of order 1e8, so partial pivoting swaps
    them out and the pinned value comes back perturbed - by 10% or more once the
    densities span fifteen orders of magnitude. Eliminating them by hand keeps
    the contact values bit-exact and conditions the remaining system better.
    """
    N = len(di)
    if N <= 2:
        return np.array([left, right], float)[:N]
    rr = r[1:-1].copy()
    rr[0] -= lo[1] * left
    rr[-1] -= up[N - 2] * right
    inner = _tridiag(lo[1:-1], di[1:-1], up[1:-1], rr)
    out = np.empty(N, float)
    out[0] = left; out[-1] = right; out[1:-1] = inner
    return out




def derived_constants(spec: dict) -> dict:
    _Q, _KB, _EPS0 = _phys()
    if not isinstance(spec, dict) or "Eg" not in spec:
        raise ValueError("spec must be the declared device configuration")
    if spec["T"] <= 0.0:
        raise ValueError("the temperature must be positive")
    VT = _KB * spec["T"] / _Q
    return dict(VT=VT,
                Vbi=spec["Eg"] - 2.0 * spec["Phi"],
                beta_L=_Q * (spec["mu_n"] + spec["mu_p"]) / (spec["eps_r"] * _EPS0),
                ni2=spec["Nc"] * spec["Nv"] * np.exp(-spec["Eg"] / VT),
                eps=spec["eps_r"] * _EPS0)

"""Canonical implementations for Material_Science-Semiconductor_Materials-10 (rebuild).
Source: arXiv:2609.05170 (CC BY 4.0) -- photocurrent versus effective voltage in organic
solar cells; one-dimensional drift-diffusion with Scharfetter-Gummel fluxes.

THE ONLY PLACE THE SCIENCE IS WRITTEN. build_osc.py slices this into sub_problems/NN_name.py.

Oracle chaining rule: every * that needs an earlier step calls that step's 
twin, NEVER the public name.
"""
import numpy as np
from scipy.linalg import solve_banded

_Q = 1.602176634e-19
_KB = 1.380649e-23
_EPS0 = 8.8541878128e-12


def _phys():
    """Physical constants. A function, not module-level state: the harness keeps
    imports and function definitions and drops module-level assignments."""
    return 1.602176634e-19, 1.380649e-23, 8.8541878128e-12

# declared device configuration (SI)
_SPEC = dict(T=300.0, d=100e-9, Eg=1.4, Phi=0.25, mu_n=2e-8, mu_p=2e-8,
             eps_r=3.5, Nc=1e26, Nv=1e26, Gex=1e28, gamma=1.0)


def _tridiag(lo, di, up, rhs):
    ab = np.zeros((3, len(di)))
    ab[0, 1:] = up[:-1]
    ab[1] = di
    ab[2, :-1] = lo[1:]
    return solve_banded((1, 1), ab, rhs)


def _tridiag_dirichlet(lo, di, up, r, left, right):
    """Solve the interior rows only, with the two end values pinned exactly.

    Putting the Dirichlet rows inside the banded system gives them a unit
    diagonal beside interior diagonals of order 1e8, so partial pivoting swaps
    them out and the pinned value comes back perturbed - by 10% or more once the
    densities span fifteen orders of magnitude. Eliminating them by hand keeps
    the contact values bit-exact and conditions the remaining system better.
    """
    N = len(di)
    if N <= 2:
        return np.array([left, right], float)[:N]
    rr = r[1:-1].copy()
    rr[0] -= lo[1] * left
    rr[-1] -= up[N - 2] * right
    inner = _tridiag(lo[1:-1], di[1:-1], up[1:-1], rr)
    out = np.empty(N, float)
    out[0] = left; out[-1] = right; out[1:-1] = inner
    return out




def bernoulli(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("the Bernoulli argument must be finite")
    out = np.empty_like(x)
    small = np.abs(x) < 1.0e-10
    out[small] = 1.0 - x[small] / 2.0 + x[small] ** 2 / 12.0
    big = ~small
    xb = np.clip(x[big], -700.0, 700.0)
    out[big] = xb / np.expm1(xb)
    return out

"""Canonical implementations for Material_Science-Semiconductor_Materials-10 (rebuild).
Source: arXiv:2609.05170 (CC BY 4.0) -- photocurrent versus effective voltage in organic
solar cells; one-dimensional drift-diffusion with Scharfetter-Gummel fluxes.

THE ONLY PLACE THE SCIENCE IS WRITTEN. build_osc.py slices this into sub_problems/NN_name.py.

Oracle chaining rule: every * that needs an earlier step calls that step's 
twin, NEVER the public name.
"""
import numpy as np
from scipy.linalg import solve_banded

_Q = 1.602176634e-19
_KB = 1.380649e-23
_EPS0 = 8.8541878128e-12


def _phys():
    """Physical constants. A function, not module-level state: the harness keeps
    imports and function definitions and drops module-level assignments."""
    return 1.602176634e-19, 1.380649e-23, 8.8541878128e-12

# declared device configuration (SI)
_SPEC = dict(T=300.0, d=100e-9, Eg=1.4, Phi=0.25, mu_n=2e-8, mu_p=2e-8,
             eps_r=3.5, Nc=1e26, Nv=1e26, Gex=1e28, gamma=1.0)


def _tridiag(lo, di, up, rhs):
    ab = np.zeros((3, len(di)))
    ab[0, 1:] = up[:-1]
    ab[1] = di
    ab[2, :-1] = lo[1:]
    return solve_banded((1, 1), ab, rhs)


def _tridiag_dirichlet(lo, di, up, r, left, right):
    """Solve the interior rows only, with the two end values pinned exactly.

    Putting the Dirichlet rows inside the banded system gives them a unit
    diagonal beside interior diagonals of order 1e8, so partial pivoting swaps
    them out and the pinned value comes back perturbed - by 10% or more once the
    densities span fifteen orders of magnitude. Eliminating them by hand keeps
    the contact values bit-exact and conditions the remaining system better.
    """
    N = len(di)
    if N <= 2:
        return np.array([left, right], float)[:N]
    rr = r[1:-1].copy()
    rr[0] -= lo[1] * left
    rr[-1] -= up[N - 2] * right
    inner = _tridiag(lo[1:-1], di[1:-1], up[1:-1], rr)
    out = np.empty(N, float)
    out[0] = left; out[-1] = right; out[1:-1] = inner
    return out




def contact_densities(spec: dict) -> np.ndarray:
    if spec["Phi"] < 0.0 or spec["Phi"] > spec["Eg"]:
        raise ValueError("the injection barrier must lie inside the transport gap")
    c = derived_constants(spec)
    p_an = spec["Nv"] * np.exp(-spec["Phi"] / c["VT"])
    n_cat = spec["Nc"] * np.exp(-spec["Phi"] / c["VT"])
    return np.array([c["ni2"] / p_an, p_an, n_cat, c["ni2"] / n_cat])

"""Canonical implementations for Material_Science-Semiconductor_Materials-10 (rebuild).
Source: arXiv:2609.05170 (CC BY 4.0) -- photocurrent versus effective voltage in organic
solar cells; one-dimensional drift-diffusion with Scharfetter-Gummel fluxes.

THE ONLY PLACE THE SCIENCE IS WRITTEN. build_osc.py slices this into sub_problems/NN_name.py.

Oracle chaining rule: every * that needs an earlier step calls that step's 
twin, NEVER the public name.
"""
import numpy as np
from scipy.linalg import solve_banded

_Q = 1.602176634e-19
_KB = 1.380649e-23
_EPS0 = 8.8541878128e-12


def _phys():
    """Physical constants. A function, not module-level state: the harness keeps
    imports and function definitions and drops module-level assignments."""
    return 1.602176634e-19, 1.380649e-23, 8.8541878128e-12

# declared device configuration (SI)
_SPEC = dict(T=300.0, d=100e-9, Eg=1.4, Phi=0.25, mu_n=2e-8, mu_p=2e-8,
             eps_r=3.5, Nc=1e26, Nv=1e26, Gex=1e28, gamma=1.0)


def _tridiag(lo, di, up, rhs):
    ab = np.zeros((3, len(di)))
    ab[0, 1:] = up[:-1]
    ab[1] = di
    ab[2, :-1] = lo[1:]
    return solve_banded((1, 1), ab, rhs)


def _tridiag_dirichlet(lo, di, up, r, left, right):
    """Solve the interior rows only, with the two end values pinned exactly.

    Putting the Dirichlet rows inside the banded system gives them a unit
    diagonal beside interior diagonals of order 1e8, so partial pivoting swaps
    them out and the pinned value comes back perturbed - by 10% or more once the
    densities span fifteen orders of magnitude. Eliminating them by hand keeps
    the contact values bit-exact and conditions the remaining system better.
    """
    N = len(di)
    if N <= 2:
        return np.array([left, right], float)[:N]
    rr = r[1:-1].copy()
    rr[0] -= lo[1] * left
    rr[-1] -= up[N - 2] * right
    inner = _tridiag(lo[1:-1], di[1:-1], up[1:-1], rr)
    out = np.empty(N, float)
    out[0] = left; out[-1] = right; out[1:-1] = inner
    return out




def poisson_step(psi: np.ndarray, n: np.ndarray, p: np.ndarray, spec: dict, V: float) -> np.ndarray:
    _Q, _KB, _EPS0 = _phys()
    psi = np.asarray(psi, float); n = np.asarray(n, float); p = np.asarray(p, float)
    if not (psi.shape == n.shape == p.shape):
        raise ValueError("psi, n and p must share one grid")
    c = derived_constants(spec)
    N = psi.size
    h = spec["d"] / (N - 1)
    lo = np.full(N, c["eps"] / h ** 2)
    up = np.full(N, c["eps"] / h ** 2)
    di = np.full(N, -2.0 * c["eps"] / h ** 2) - _Q * (n + p) / c["VT"]
    rhs = -_Q * (p - n) - _Q * (n + p) / c["VT"] * psi
    new = _tridiag_dirichlet(lo, di, up, rhs, 0.0, c["Vbi"] - V)
    return psi + np.clip(new - psi, -0.5, 0.5)

"""Canonical implementations for Material_Science-Semiconductor_Materials-10 (rebuild).
Source: arXiv:2609.05170 (CC BY 4.0) -- photocurrent versus effective voltage in organic
solar cells; one-dimensional drift-diffusion with Scharfetter-Gummel fluxes.

THE ONLY PLACE THE SCIENCE IS WRITTEN. build_osc.py slices this into sub_problems/NN_name.py.

Oracle chaining rule: every * that needs an earlier step calls that step's 
twin, NEVER the public name.
"""
import numpy as np
from scipy.linalg import solve_banded

_Q = 1.602176634e-19
_KB = 1.380649e-23
_EPS0 = 8.8541878128e-12


def _phys():
    """Physical constants. A function, not module-level state: the harness keeps
    imports and function definitions and drops module-level assignments."""
    return 1.602176634e-19, 1.380649e-23, 8.8541878128e-12

# declared device configuration (SI)
_SPEC = dict(T=300.0, d=100e-9, Eg=1.4, Phi=0.25, mu_n=2e-8, mu_p=2e-8,
             eps_r=3.5, Nc=1e26, Nv=1e26, Gex=1e28, gamma=1.0)


def _tridiag(lo, di, up, rhs):
    ab = np.zeros((3, len(di)))
    ab[0, 1:] = up[:-1]
    ab[1] = di
    ab[2, :-1] = lo[1:]
    return solve_banded((1, 1), ab, rhs)


def _tridiag_dirichlet(lo, di, up, r, left, right):
    """Solve the interior rows only, with the two end values pinned exactly.

    Putting the Dirichlet rows inside the banded system gives them a unit
    diagonal beside interior diagonals of order 1e8, so partial pivoting swaps
    them out and the pinned value comes back perturbed - by 10% or more once the
    densities span fifteen orders of magnitude. Eliminating them by hand keeps
    the contact values bit-exact and conditions the remaining system better.
    """
    N = len(di)
    if N <= 2:
        return np.array([left, right], float)[:N]
    rr = r[1:-1].copy()
    rr[0] -= lo[1] * left
    rr[-1] -= up[N - 2] * right
    inner = _tridiag(lo[1:-1], di[1:-1], up[1:-1], rr)
    out = np.empty(N, float)
    out[0] = left; out[-1] = right; out[1:-1] = inner
    return out




def electron_density(psi: np.ndarray, p: np.ndarray, spec: dict, G: float) -> np.ndarray:
    psi = np.asarray(psi, float); p = np.asarray(p, float)
    if psi.shape != p.shape:
        raise ValueError("the potential and the carrier density must share one grid")
    if np.isscalar(G) and float(G) < 0.0:
        raise ValueError("the generation rate cannot be negative")
    c = derived_constants(spec)
    bc = contact_densities(spec)
    N = psi.size
    h = spec["d"] / (N - 1)
    a = spec["mu_n"] * c["VT"] / h ** 2
    Bp = bernoulli(np.diff(psi) / c["VT"])
    Bm = bernoulli(-np.diff(psi) / c["VT"])
    R = spec["gamma"] * c["beta_L"]
    lo = np.zeros(N); di = np.zeros(N); up = np.zeros(N); r = np.zeros(N)
    lo[1:-1] = a * Bm[0:N - 2]
    up[1:-1] = a * Bp[1:N - 1]
    di[1:-1] = -(a * Bp[0:N - 2] + a * Bm[1:N - 1]) - R * p[1:-1]
    Gv = np.full(N, float(G)) if np.isscalar(G) else np.asarray(G, float)
    r[1:-1] = -Gv[1:-1] - R * c["ni2"]
    return np.maximum(_tridiag_dirichlet(lo, di, up, r, bc[0], bc[2]), 1.0e-30)

"""Canonical implementations for Material_Science-Semiconductor_Materials-10 (rebuild).
Source: arXiv:2609.05170 (CC BY 4.0) -- photocurrent versus effective voltage in organic
solar cells; one-dimensional drift-diffusion with Scharfetter-Gummel fluxes.

THE ONLY PLACE THE SCIENCE IS WRITTEN. build_osc.py slices this into sub_problems/NN_name.py.

Oracle chaining rule: every * that needs an earlier step calls that step's 
twin, NEVER the public name.
"""
import numpy as np
from scipy.linalg import solve_banded

_Q = 1.602176634e-19
_KB = 1.380649e-23
_EPS0 = 8.8541878128e-12


def _phys():
    """Physical constants. A function, not module-level state: the harness keeps
    imports and function definitions and drops module-level assignments."""
    return 1.602176634e-19, 1.380649e-23, 8.8541878128e-12

# declared device configuration (SI)
_SPEC = dict(T=300.0, d=100e-9, Eg=1.4, Phi=0.25, mu_n=2e-8, mu_p=2e-8,
             eps_r=3.5, Nc=1e26, Nv=1e26, Gex=1e28, gamma=1.0)


def _tridiag(lo, di, up, rhs):
    ab = np.zeros((3, len(di)))
    ab[0, 1:] = up[:-1]
    ab[1] = di
    ab[2, :-1] = lo[1:]
    return solve_banded((1, 1), ab, rhs)


def _tridiag_dirichlet(lo, di, up, r, left, right):
    """Solve the interior rows only, with the two end values pinned exactly.

    Putting the Dirichlet rows inside the banded system gives them a unit
    diagonal beside interior diagonals of order 1e8, so partial pivoting swaps
    them out and the pinned value comes back perturbed - by 10% or more once the
    densities span fifteen orders of magnitude. Eliminating them by hand keeps
    the contact values bit-exact and conditions the remaining system better.
    """
    N = len(di)
    if N <= 2:
        return np.array([left, right], float)[:N]
    rr = r[1:-1].copy()
    rr[0] -= lo[1] * left
    rr[-1] -= up[N - 2] * right
    inner = _tridiag(lo[1:-1], di[1:-1], up[1:-1], rr)
    out = np.empty(N, float)
    out[0] = left; out[-1] = right; out[1:-1] = inner
    return out




def hole_density(psi: np.ndarray, n: np.ndarray, spec: dict, G: float) -> np.ndarray:
    psi = np.asarray(psi, float); n = np.asarray(n, float)
    if psi.shape != n.shape:
        raise ValueError("the potential and the carrier density must share one grid")
    if np.isscalar(G) and float(G) < 0.0:
        raise ValueError("the generation rate cannot be negative")
    c = derived_constants(spec)
    bc = contact_densities(spec)
    N = psi.size
    h = spec["d"] / (N - 1)
    b = spec["mu_p"] * c["VT"] / h ** 2
    Bp = bernoulli(np.diff(psi) / c["VT"])
    Bm = bernoulli(-np.diff(psi) / c["VT"])
    R = spec["gamma"] * c["beta_L"]
    lo = np.zeros(N); di = np.zeros(N); up = np.zeros(N); r = np.zeros(N)
    lo[1:-1] = b * Bp[0:N - 2]
    up[1:-1] = b * Bm[1:N - 1]
    di[1:-1] = -(b * Bm[0:N - 2] + b * Bp[1:N - 1]) - R * n[1:-1]
    Gv = np.full(N, float(G)) if np.isscalar(G) else np.asarray(G, float)
    r[1:-1] = -Gv[1:-1] - R * c["ni2"]
    return np.maximum(_tridiag_dirichlet(lo, di, up, r, bc[1], bc[3]), 1.0e-30)

"""Canonical implementations for Material_Science-Semiconductor_Materials-10 (rebuild).
Source: arXiv:2609.05170 (CC BY 4.0) -- photocurrent versus effective voltage in organic
solar cells; one-dimensional drift-diffusion with Scharfetter-Gummel fluxes.

THE ONLY PLACE THE SCIENCE IS WRITTEN. build_osc.py slices this into sub_problems/NN_name.py.

Oracle chaining rule: every * that needs an earlier step calls that step's 
twin, NEVER the public name.
"""
import numpy as np
from scipy.linalg import solve_banded

_Q = 1.602176634e-19
_KB = 1.380649e-23
_EPS0 = 8.8541878128e-12


def _phys():
    """Physical constants. A function, not module-level state: the harness keeps
    imports and function definitions and drops module-level assignments."""
    return 1.602176634e-19, 1.380649e-23, 8.8541878128e-12

# declared device configuration (SI)
_SPEC = dict(T=300.0, d=100e-9, Eg=1.4, Phi=0.25, mu_n=2e-8, mu_p=2e-8,
             eps_r=3.5, Nc=1e26, Nv=1e26, Gex=1e28, gamma=1.0)


def _tridiag(lo, di, up, rhs):
    ab = np.zeros((3, len(di)))
    ab[0, 1:] = up[:-1]
    ab[1] = di
    ab[2, :-1] = lo[1:]
    return solve_banded((1, 1), ab, rhs)


def _tridiag_dirichlet(lo, di, up, r, left, right):
    """Solve the interior rows only, with the two end values pinned exactly.

    Putting the Dirichlet rows inside the banded system gives them a unit
    diagonal beside interior diagonals of order 1e8, so partial pivoting swaps
    them out and the pinned value comes back perturbed - by 10% or more once the
    densities span fifteen orders of magnitude. Eliminating them by hand keeps
    the contact values bit-exact and conditions the remaining system better.
    """
    N = len(di)
    if N <= 2:
        return np.array([left, right], float)[:N]
    rr = r[1:-1].copy()
    rr[0] -= lo[1] * left
    rr[-1] -= up[N - 2] * right
    inner = _tridiag(lo[1:-1], di[1:-1], up[1:-1], rr)
    out = np.empty(N, float)
    out[0] = left; out[-1] = right; out[1:-1] = inner
    return out




def self_consistent_state(V: float, G: float, spec: dict, n_grid: int) -> tuple:
    n_grid = int(n_grid)
    if n_grid < 5:
        raise ValueError("the grid needs at least five nodes")
    c = derived_constants(spec)
    bc = contact_densities(spec)
    psi = np.linspace(0.0, c["Vbi"] - V, n_grid)
    n = np.linspace(bc[0], bc[2], n_grid)
    p = np.linspace(bc[1], bc[3], n_grid)
    for _ in range(2000):
        psi_new = poisson_step(psi, n, p, spec, V)
        dpsi = float(np.max(np.abs(psi_new - psi)))
        psi = psi_new
        n_new = electron_density(psi, p, spec, G)
        p_new = hole_density(psi, n_new, spec, G)
        # measured against each density's own scale: the minority carrier at its own
        # contact sits fourteen orders below the majority, so a pointwise relative test
        # there reports rounding noise and never settles
        rel = max(float(np.max(np.abs(n_new - n))) / float(np.max(n_new)),
                  float(np.max(np.abs(p_new - p))) / float(np.max(p_new)))
        n, p = n_new, p_new
        if max(dpsi, rel) < 1.0e-6:
            break
    else:
        raise ValueError("the coupled sweeps did not reach a self-consistent state")
    return psi, n, p

"""Canonical implementations for Material_Science-Semiconductor_Materials-10 (rebuild).
Source: arXiv:2609.05170 (CC BY 4.0) -- photocurrent versus effective voltage in organic
solar cells; one-dimensional drift-diffusion with Scharfetter-Gummel fluxes.

THE ONLY PLACE THE SCIENCE IS WRITTEN. build_osc.py slices this into sub_problems/NN_name.py.

Oracle chaining rule: every * that needs an earlier step calls that step's 
twin, NEVER the public name.
"""
import numpy as np
from scipy.linalg import solve_banded

_Q = 1.602176634e-19
_KB = 1.380649e-23
_EPS0 = 8.8541878128e-12


def _phys():
    """Physical constants. A function, not module-level state: the harness keeps
    imports and function definitions and drops module-level assignments."""
    return 1.602176634e-19, 1.380649e-23, 8.8541878128e-12

# declared device configuration (SI)
_SPEC = dict(T=300.0, d=100e-9, Eg=1.4, Phi=0.25, mu_n=2e-8, mu_p=2e-8,
             eps_r=3.5, Nc=1e26, Nv=1e26, Gex=1e28, gamma=1.0)


def _tridiag(lo, di, up, rhs):
    ab = np.zeros((3, len(di)))
    ab[0, 1:] = up[:-1]
    ab[1] = di
    ab[2, :-1] = lo[1:]
    return solve_banded((1, 1), ab, rhs)


def _tridiag_dirichlet(lo, di, up, r, left, right):
    """Solve the interior rows only, with the two end values pinned exactly.

    Putting the Dirichlet rows inside the banded system gives them a unit
    diagonal beside interior diagonals of order 1e8, so partial pivoting swaps
    them out and the pinned value comes back perturbed - by 10% or more once the
    densities span fifteen orders of magnitude. Eliminating them by hand keeps
    the contact values bit-exact and conditions the remaining system better.
    """
    N = len(di)
    if N <= 2:
        return np.array([left, right], float)[:N]
    rr = r[1:-1].copy()
    rr[0] -= lo[1] * left
    rr[-1] -= up[N - 2] * right
    inner = _tridiag(lo[1:-1], di[1:-1], up[1:-1], rr)
    out = np.empty(N, float)
    out[0] = left; out[-1] = right; out[1:-1] = inner
    return out




def terminal_current(psi: np.ndarray, n: np.ndarray, p: np.ndarray, spec: dict) -> float:
    _Q, _KB, _EPS0 = _phys()
    psi = np.asarray(psi, float); n = np.asarray(n, float); p = np.asarray(p, float)
    c = derived_constants(spec)
    N = psi.size
    h = spec["d"] / (N - 1)
    Bp = bernoulli(np.diff(psi) / c["VT"])
    Bm = bernoulli(-np.diff(psi) / c["VT"])
    Jn = _Q * spec["mu_n"] * c["VT"] / h * (n[1:] * Bp - n[:-1] * Bm)
    Jp = _Q * spec["mu_p"] * c["VT"] / h * (p[:-1] * Bp - p[1:] * Bm)
    Jt = Jn + Jp
    if float(np.max(Jt) - np.min(Jt)) > 1.0e-3 * max(1.0, float(np.max(np.abs(Jt)))):
        raise ValueError("the total current is not independent of position")
    return float(np.mean(Jt))

"""Canonical implementations for Material_Science-Semiconductor_Materials-10 (rebuild).
Source: arXiv:2609.05170 (CC BY 4.0) -- photocurrent versus effective voltage in organic
solar cells; one-dimensional drift-diffusion with Scharfetter-Gummel fluxes.

THE ONLY PLACE THE SCIENCE IS WRITTEN. build_osc.py slices this into sub_problems/NN_name.py.

Oracle chaining rule: every * that needs an earlier step calls that step's 
twin, NEVER the public name.
"""
import numpy as np
from scipy.linalg import solve_banded

_Q = 1.602176634e-19
_KB = 1.380649e-23
_EPS0 = 8.8541878128e-12


def _phys():
    """Physical constants. A function, not module-level state: the harness keeps
    imports and function definitions and drops module-level assignments."""
    return 1.602176634e-19, 1.380649e-23, 8.8541878128e-12

# declared device configuration (SI)
_SPEC = dict(T=300.0, d=100e-9, Eg=1.4, Phi=0.25, mu_n=2e-8, mu_p=2e-8,
             eps_r=3.5, Nc=1e26, Nv=1e26, Gex=1e28, gamma=1.0)


def _tridiag(lo, di, up, rhs):
    ab = np.zeros((3, len(di)))
    ab[0, 1:] = up[:-1]
    ab[1] = di
    ab[2, :-1] = lo[1:]
    return solve_banded((1, 1), ab, rhs)


def _tridiag_dirichlet(lo, di, up, r, left, right):
    """Solve the interior rows only, with the two end values pinned exactly.

    Putting the Dirichlet rows inside the banded system gives them a unit
    diagonal beside interior diagonals of order 1e8, so partial pivoting swaps
    them out and the pinned value comes back perturbed - by 10% or more once the
    densities span fifteen orders of magnitude. Eliminating them by hand keeps
    the contact values bit-exact and conditions the remaining system better.
    """
    N = len(di)
    if N <= 2:
        return np.array([left, right], float)[:N]
    rr = r[1:-1].copy()
    rr[0] -= lo[1] * left
    rr[-1] -= up[N - 2] * right
    inner = _tridiag(lo[1:-1], di[1:-1], up[1:-1], rr)
    out = np.empty(N, float)
    out[0] = left; out[-1] = right; out[1:-1] = inner
    return out




def open_circuit_voltage(spec: dict, n_grid: int) -> float:
    from scipy.optimize import brentq
    c = derived_constants(spec)

    def _J(V):
        psi, n, p = self_consistent_state(V, spec["Gex"], spec, n_grid)
        return terminal_current(psi, n, p, spec)

    lo, hi = 0.0, c["Vbi"] - 1.0e-3
    if _J(lo) * _J(hi) > 0.0:
        raise ValueError("the illuminated current does not change sign below flat band")
    return float(brentq(_J, lo, hi, xtol=1.0e-12, rtol=8.9e-16, maxiter=300))

"""Canonical implementations for Material_Science-Semiconductor_Materials-10 (rebuild).
Source: arXiv:2609.05170 (CC BY 4.0) -- photocurrent versus effective voltage in organic
solar cells; one-dimensional drift-diffusion with Scharfetter-Gummel fluxes.

THE ONLY PLACE THE SCIENCE IS WRITTEN. build_osc.py slices this into sub_problems/NN_name.py.

Oracle chaining rule: every * that needs an earlier step calls that step's 
twin, NEVER the public name.
"""
import numpy as np
from scipy.linalg import solve_banded

_Q = 1.602176634e-19
_KB = 1.380649e-23
_EPS0 = 8.8541878128e-12


def _phys():
    """Physical constants. A function, not module-level state: the harness keeps
    imports and function definitions and drops module-level assignments."""
    return 1.602176634e-19, 1.380649e-23, 8.8541878128e-12

# declared device configuration (SI)
_SPEC = dict(T=300.0, d=100e-9, Eg=1.4, Phi=0.25, mu_n=2e-8, mu_p=2e-8,
             eps_r=3.5, Nc=1e26, Nv=1e26, Gex=1e28, gamma=1.0)


def _tridiag(lo, di, up, rhs):
    ab = np.zeros((3, len(di)))
    ab[0, 1:] = up[:-1]
    ab[1] = di
    ab[2, :-1] = lo[1:]
    return solve_banded((1, 1), ab, rhs)


def _tridiag_dirichlet(lo, di, up, r, left, right):
    """Solve the interior rows only, with the two end values pinned exactly.

    Putting the Dirichlet rows inside the banded system gives them a unit
    diagonal beside interior diagonals of order 1e8, so partial pivoting swaps
    them out and the pinned value comes back perturbed - by 10% or more once the
    densities span fifteen orders of magnitude. Eliminating them by hand keeps
    the contact values bit-exact and conditions the remaining system better.
    """
    N = len(di)
    if N <= 2:
        return np.array([left, right], float)[:N]
    rr = r[1:-1].copy()
    rr[0] -= lo[1] * left
    rr[-1] -= up[N - 2] * right
    inner = _tridiag(lo[1:-1], di[1:-1], up[1:-1], rr)
    out = np.empty(N, float)
    out[0] = left; out[-1] = right; out[1:-1] = inner
    return out




def fill_factor(spec: dict, n_coarse: int, n_fine: int) -> float:
    from scipy.optimize import minimize_scalar
    if int(n_fine) <= int(n_coarse):
        raise ValueError("the fine grid must be finer than the coarse one")
    if spec["gamma"] < 0.0:
        raise ValueError("the recombination reduction factor cannot be negative")

    c = derived_constants(spec)
    bc = contact_densities(spec)

    # gate 1: the contact densities must obey mass action
    if abs(bc[0] * bc[1] - c["ni2"]) > 1.0e-6 * c["ni2"]:
        raise ValueError("the contact densities violate mass action")
    # gate 2: the Bernoulli function must satisfy B(x) - B(-x) = -x
    xs = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    if float(np.max(np.abs(bernoulli(xs) - bernoulli(-xs) + xs))) > 1.0e-12:
        raise ValueError("the Bernoulli function fails its defining identity")
    # gate 3: in the dark at zero bias the terminal current must vanish
    psi, n, p = self_consistent_state(0.0, 0.0, spec, int(n_coarse))
    if abs(terminal_current(psi, n, p, spec)) > 1.0e-6:
        raise ValueError("the dark device carries current at zero bias")

    # gate 4: the converged state must be stationary under one further sweep, measured on
    # the terminal current, which is the quantity the sweeps actually settle. The carrier
    # densities keep jittering in the minority tail long after the current has stopped moving.
    for ng in (int(n_coarse), int(n_fine)):
        psi, n, p = self_consistent_state(0.0, spec["Gex"], spec, ng)
        J0 = terminal_current(psi, n, p, spec)
        psi2 = poisson_step(psi, n, p, spec, 0.0)
        n2 = electron_density(psi2, p, spec, spec["Gex"])
        p2 = hole_density(psi2, n2, spec, spec["Gex"])
        J1 = terminal_current(psi2, n2, p2, spec)
        if abs(J1 - J0) > 1.0e-6 * abs(J0):
            raise ValueError("one further sweep still moves the terminal current")

    def _ff(n_grid):
        n_grid = int(n_grid)
        psi, ne, ph = self_consistent_state(0.0, spec["Gex"], spec, n_grid)
        Jsc = terminal_current(psi, ne, ph, spec)
        Voc = open_circuit_voltage(spec, n_grid)

        def _power(V):
            ps, nn, pp = self_consistent_state(V, spec["Gex"], spec, n_grid)
            return V * terminal_current(ps, nn, pp, spec)

        r = minimize_scalar(_power, bounds=(0.0, Voc), method="bounded",
                            options=dict(xatol=1.0e-10))
        return (-float(r.fun)) / (abs(Jsc) * Voc)

    f_c = _ff(n_coarse)
    f_f = _ff(n_fine)
    # the discretisation is second order, so Richardson removes the leading error
    r = (float(n_fine) - 1.0) / (float(n_coarse) - 1.0)
    return float(f_f + (f_f - f_c) / (r * r - 1.0))
SCICODE_GOLD_EOF
