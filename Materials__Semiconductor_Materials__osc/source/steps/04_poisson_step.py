"""
Advance the electrostatic potential by one linearised Poisson solve at fixed carrier densities, on a uniform grid spanning the active layer. Raise ValueError if the three profiles do not share one grid.

Solving Poisson with the densities held fixed diverges at these carrier levels, so the update is linearised by letting each density respond to the potential change through its Boltzmann factor before the tridiagonal system is formed. The potential is pinned at both contacts, the applied bias entering as the difference between them. The step is limited so that no single update can move the potential by more than half a volt.

Returns
-------
np.ndarray of the same shape as psi, in volts
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def poisson_step(psi: np.ndarray, n: np.ndarray, p: np.ndarray, spec: dict, V: float) -> np.ndarray:
    """Advance the electrostatic potential by one linearised Poisson solve at fixed carrier densities, on a uniform grid spanning the active layer. Raise ValueError if the three profiles do not share one grid.

    Parameters
    ----------
    psi : np.ndarray
        Current electrostatic potential on a uniform grid, in volts.
    n : np.ndarray
        Electron density on the same grid.
    p : np.ndarray
        Hole density on the same grid.
    spec : dict
        The declared device configuration, in SI units, with keys 'T' (temperature, K), 'd' (layer thickness, m), 'Eg' (transport gap, eV), 'Phi' (injection barrier at each contact, eV), 'mu_n' and 'mu_p' (mobilities, m^2 V^-1 s^-1), 'eps_r' (relative permittivity), 'Nc' and 'Nv' (effective densities of states, m^-3), 'Gex' (generation rate, m^-3 s^-1) and 'gamma' (recombination reduction factor).
    V : float
        Applied bias, in volts.

    Returns
    -------
    np.ndarray of the same shape as psi, in volts
    """
    return psi

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Canonical implementations for Material_Science-Semiconductor_Materials-10 (rebuild).
Source: arXiv:2609.05170 (CC BY 4.0) -- photocurrent versus effective voltage in organic
solar cells; one-dimensional drift-diffusion with Scharfetter-Gummel fluxes.

THE ONLY PLACE THE SCIENCE IS WRITTEN. build_osc.py slices this into sub_problems/NN_name.py.

Oracle chaining rule: every _oracle_* that needs an earlier step calls that step's _oracle_
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




def _oracle_poisson_step(psi: np.ndarray, n: np.ndarray, p: np.ndarray, spec: dict, V: float) -> np.ndarray:
    _Q, _KB, _EPS0 = _phys()
    psi = np.asarray(psi, float); n = np.asarray(n, float); p = np.asarray(p, float)
    if not (psi.shape == n.shape == p.shape):
        raise ValueError("psi, n and p must share one grid")
    c = _oracle_derived_constants(spec)
    N = psi.size
    h = spec["d"] / (N - 1)
    lo = np.full(N, c["eps"] / h ** 2)
    up = np.full(N, c["eps"] / h ** 2)
    di = np.full(N, -2.0 * c["eps"] / h ** 2) - _Q * (n + p) / c["VT"]
    rhs = -_Q * (p - n) - _Q * (n + p) / c["VT"] * psi
    new = _tridiag_dirichlet(lo, di, up, rhs, 0.0, c["Vbi"] - V)
    return psi + np.clip(new - psi, -0.5, 0.5)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': "import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ndef _fx_derived_constants(spec):\n    _Q=1.602176634e-19; _KB=1.380649e-23; _EPS0=8.8541878128e-12\n    if not isinstance(spec, dict) or 'Eg' not in spec:\n        raise ValueError('spec must be the declared device configuration')\n    if spec['T'] <= 0.0:\n        raise ValueError('the temperature must be positive')\n    VT = _KB * spec['T'] / _Q\n    return dict(VT=VT, Vbi=spec['Eg'] - 2.0 * spec['Phi'], beta_L=_Q * (spec['mu_n'] + spec['mu_p']) / (spec['eps_r'] * _EPS0), ni2=spec['Nc'] * spec['Nv'] * np.exp(-spec['Eg'] / VT), eps=spec['eps_r'] * _EPS0)\ndef _fx_contact_densities(spec):\n    if spec['Phi'] < 0.0 or spec['Phi'] > spec['Eg']:\n        raise ValueError('the injection barrier must lie inside the transport gap')\n    c = _fx_derived_constants(spec)\n    p_an = spec['Nv'] * np.exp(-spec['Phi'] / c['VT'])\n    n_cat = spec['Nc'] * np.exp(-spec['Phi'] / c['VT'])\n    return np.array([c['ni2'] / p_an, p_an, n_cat, c['ni2'] / n_cat])\ndef start(sp,V,N):\n    c=_fx_derived_constants(sp); bc=_fx_contact_densities(sp)\n    return (np.linspace(0.0,c['Vbi']-V,N),np.linspace(bc[0],bc[2],N),np.linspace(bc[1],bc[3],N))\npsi,n,p=start(SPEC,0.0,41)", 'call': 'poisson_step(psi,n,p,SPEC,0.0)', 'gold_call': '_oracle_poisson_step(psi,n,p,SPEC,0.0)', 'tol': 1e-09}, {'setup': "import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ndef _fx_derived_constants(spec):\n    _Q=1.602176634e-19; _KB=1.380649e-23; _EPS0=8.8541878128e-12\n    if not isinstance(spec, dict) or 'Eg' not in spec:\n        raise ValueError('spec must be the declared device configuration')\n    if spec['T'] <= 0.0:\n        raise ValueError('the temperature must be positive')\n    VT = _KB * spec['T'] / _Q\n    return dict(VT=VT, Vbi=spec['Eg'] - 2.0 * spec['Phi'], beta_L=_Q * (spec['mu_n'] + spec['mu_p']) / (spec['eps_r'] * _EPS0), ni2=spec['Nc'] * spec['Nv'] * np.exp(-spec['Eg'] / VT), eps=spec['eps_r'] * _EPS0)\ndef _fx_contact_densities(spec):\n    if spec['Phi'] < 0.0 or spec['Phi'] > spec['Eg']:\n        raise ValueError('the injection barrier must lie inside the transport gap')\n    c = _fx_derived_constants(spec)\n    p_an = spec['Nv'] * np.exp(-spec['Phi'] / c['VT'])\n    n_cat = spec['Nc'] * np.exp(-spec['Phi'] / c['VT'])\n    return np.array([c['ni2'] / p_an, p_an, n_cat, c['ni2'] / n_cat])\ndef start(sp,V,N):\n    c=_fx_derived_constants(sp); bc=_fx_contact_densities(sp)\n    return (np.linspace(0.0,c['Vbi']-V,N),np.linspace(bc[0],bc[2],N),np.linspace(bc[1],bc[3],N))\npsi,n,p=start(SPEC,0.5,61)", 'call': 'poisson_step(psi,n,p,SPEC,0.5)', 'gold_call': '_oracle_poisson_step(psi,n,p,SPEC,0.5)', 'tol': 1e-09}, {'setup': "import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ndef _fx_derived_constants(spec):\n    _Q=1.602176634e-19; _KB=1.380649e-23; _EPS0=8.8541878128e-12\n    if not isinstance(spec, dict) or 'Eg' not in spec:\n        raise ValueError('spec must be the declared device configuration')\n    if spec['T'] <= 0.0:\n        raise ValueError('the temperature must be positive')\n    VT = _KB * spec['T'] / _Q\n    return dict(VT=VT, Vbi=spec['Eg'] - 2.0 * spec['Phi'], beta_L=_Q * (spec['mu_n'] + spec['mu_p']) / (spec['eps_r'] * _EPS0), ni2=spec['Nc'] * spec['Nv'] * np.exp(-spec['Eg'] / VT), eps=spec['eps_r'] * _EPS0)\ndef _fx_contact_densities(spec):\n    if spec['Phi'] < 0.0 or spec['Phi'] > spec['Eg']:\n        raise ValueError('the injection barrier must lie inside the transport gap')\n    c = _fx_derived_constants(spec)\n    p_an = spec['Nv'] * np.exp(-spec['Phi'] / c['VT'])\n    n_cat = spec['Nc'] * np.exp(-spec['Phi'] / c['VT'])\n    return np.array([c['ni2'] / p_an, p_an, n_cat, c['ni2'] / n_cat])\ndef start(sp,V,N):\n    c=_fx_derived_constants(sp); bc=_fx_contact_densities(sp)\n    return (np.linspace(0.0,c['Vbi']-V,N),np.linspace(bc[0],bc[2],N),np.linspace(bc[1],bc[3],N))\npsi,n,p=start(SPEC,-1.0,31)", 'call': 'poisson_step(psi,n,p,SPEC,-1.0)', 'gold_call': '_oracle_poisson_step(psi,n,p,SPEC,-1.0)', 'tol': 1e-09}, {'setup': "import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ndef raises(f):\n    try:\n        f()\n    except ValueError:\n        return 1.0\n    return 0.0\ndef _fx_derived_constants(spec):\n    _Q=1.602176634e-19; _KB=1.380649e-23; _EPS0=8.8541878128e-12\n    if not isinstance(spec, dict) or 'Eg' not in spec:\n        raise ValueError('spec must be the declared device configuration')\n    if spec['T'] <= 0.0:\n        raise ValueError('the temperature must be positive')\n    VT = _KB * spec['T'] / _Q\n    return dict(VT=VT, Vbi=spec['Eg'] - 2.0 * spec['Phi'], beta_L=_Q * (spec['mu_n'] + spec['mu_p']) / (spec['eps_r'] * _EPS0), ni2=spec['Nc'] * spec['Nv'] * np.exp(-spec['Eg'] / VT), eps=spec['eps_r'] * _EPS0)\ndef _fx_contact_densities(spec):\n    if spec['Phi'] < 0.0 or spec['Phi'] > spec['Eg']:\n        raise ValueError('the injection barrier must lie inside the transport gap')\n    c = _fx_derived_constants(spec)\n    p_an = spec['Nv'] * np.exp(-spec['Phi'] / c['VT'])\n    n_cat = spec['Nc'] * np.exp(-spec['Phi'] / c['VT'])\n    return np.array([c['ni2'] / p_an, p_an, n_cat, c['ni2'] / n_cat])\ndef start(sp,V,N):\n    c=_fx_derived_constants(sp); bc=_fx_contact_densities(sp)\n    return (np.linspace(0.0,c['Vbi']-V,N),np.linspace(bc[0],bc[2],N),np.linspace(bc[1],bc[3],N))\npsi,n,p=start(SPEC,0.0,41)", 'call': 'raises(lambda: poisson_step(psi, n[:-1], p, SPEC, 0.0))', 'gold_call': 'raises(lambda: _oracle_poisson_step(psi, n[:-1], p, SPEC, 0.0))', 'tol': 1e-09}]
