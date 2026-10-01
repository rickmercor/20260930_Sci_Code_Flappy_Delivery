"""
Solve the electron continuity equation on the given potential for the electron density, using the source's flux discretisation, with the recombination term linearised in the electron density at the given hole density. The contact values are held fixed. Raise ValueError if the potential and the hole density do not share one grid, or if the generation rate is negative.

Written with plain finite differences the carrier flux is unstable once the potential drop between neighbouring nodes exceeds the thermal voltage, which it does throughout a device of this thickness. The source therefore uses the exponentially fitted flux, in which the two neighbouring densities are weighted by the Bernoulli function of the scaled potential difference. Recombination couples the two carriers, so it enters here through the hole density held from the previous sweep.

Returns
-------
np.ndarray of the same shape as psi, in inverse cubic metres
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def electron_density(psi: np.ndarray, p: np.ndarray, spec: dict, G: float) -> np.ndarray:
    """Solve the electron continuity equation on the given potential for the electron density, using the source's flux discretisation, with the recombination term linearised in the electron density at the given hole density. The contact values are held fixed. Raise ValueError if the potential and the hole density do not share one grid, or if the generation rate is negative.

    Parameters
    ----------
    psi : np.ndarray
        Electrostatic potential on a uniform grid, in volts.
    p : np.ndarray
        Hole density on the same grid.
    spec : dict
        The declared device configuration, in SI units, with keys 'T' (temperature, K), 'd' (layer thickness, m), 'Eg' (transport gap, eV), 'Phi' (injection barrier at each contact, eV), 'mu_n' and 'mu_p' (mobilities, m^2 V^-1 s^-1), 'eps_r' (relative permittivity), 'Nc' and 'Nv' (effective densities of states, m^-3), 'Gex' (generation rate, m^-3 s^-1) and 'gamma' (recombination reduction factor).
    G : float
        Uniform free-charge generation rate.

    Returns
    -------
    np.ndarray of the same shape as psi, in inverse cubic metres
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




def _oracle_electron_density(psi: np.ndarray, p: np.ndarray, spec: dict, G: float) -> np.ndarray:
    psi = np.asarray(psi, float); p = np.asarray(p, float)
    if psi.shape != p.shape:
        raise ValueError("the potential and the carrier density must share one grid")
    if np.isscalar(G) and float(G) < 0.0:
        raise ValueError("the generation rate cannot be negative")
    c = _oracle_derived_constants(spec)
    bc = _oracle_contact_densities(spec)
    N = psi.size
    h = spec["d"] / (N - 1)
    a = spec["mu_n"] * c["VT"] / h ** 2
    Bp = _oracle_bernoulli(np.diff(psi) / c["VT"])
    Bm = _oracle_bernoulli(-np.diff(psi) / c["VT"])
    R = spec["gamma"] * c["beta_L"]
    lo = np.zeros(N); di = np.zeros(N); up = np.zeros(N); r = np.zeros(N)
    lo[1:-1] = a * Bm[0:N - 2]
    up[1:-1] = a * Bp[1:N - 1]
    di[1:-1] = -(a * Bp[0:N - 2] + a * Bm[1:N - 1]) - R * p[1:-1]
    Gv = np.full(N, float(G)) if np.isscalar(G) else np.asarray(G, float)
    r[1:-1] = -Gv[1:-1] - R * c["ni2"]
    return np.maximum(_tridiag_dirichlet(lo, di, up, r, bc[0], bc[2]), 1.0e-30)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': "import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ndef _fx_derived_constants(spec):\n    _Q=1.602176634e-19; _KB=1.380649e-23; _EPS0=8.8541878128e-12\n    if not isinstance(spec, dict) or 'Eg' not in spec:\n        raise ValueError('spec must be the declared device configuration')\n    if spec['T'] <= 0.0:\n        raise ValueError('the temperature must be positive')\n    VT = _KB * spec['T'] / _Q\n    return dict(VT=VT, Vbi=spec['Eg'] - 2.0 * spec['Phi'], beta_L=_Q * (spec['mu_n'] + spec['mu_p']) / (spec['eps_r'] * _EPS0), ni2=spec['Nc'] * spec['Nv'] * np.exp(-spec['Eg'] / VT), eps=spec['eps_r'] * _EPS0)\ndef _fx_contact_densities(spec):\n    if spec['Phi'] < 0.0 or spec['Phi'] > spec['Eg']:\n        raise ValueError('the injection barrier must lie inside the transport gap')\n    c = _fx_derived_constants(spec)\n    p_an = spec['Nv'] * np.exp(-spec['Phi'] / c['VT'])\n    n_cat = spec['Nc'] * np.exp(-spec['Phi'] / c['VT'])\n    return np.array([c['ni2'] / p_an, p_an, n_cat, c['ni2'] / n_cat])\ndef start(sp,V,N):\n    c=_fx_derived_constants(sp); bc=_fx_contact_densities(sp)\n    return (np.linspace(0.0,c['Vbi']-V,N),np.linspace(bc[0],bc[2],N),np.linspace(bc[1],bc[3],N))\npsi,n,p=start(SPEC,0.0,41)", 'call': "electron_density(psi,p,SPEC,SPEC['Gex'])", 'gold_call': "_oracle_electron_density(psi,p,SPEC,SPEC['Gex'])", 'tol': 1e-09}, {'setup': "import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ndef _fx_derived_constants(spec):\n    _Q=1.602176634e-19; _KB=1.380649e-23; _EPS0=8.8541878128e-12\n    if not isinstance(spec, dict) or 'Eg' not in spec:\n        raise ValueError('spec must be the declared device configuration')\n    if spec['T'] <= 0.0:\n        raise ValueError('the temperature must be positive')\n    VT = _KB * spec['T'] / _Q\n    return dict(VT=VT, Vbi=spec['Eg'] - 2.0 * spec['Phi'], beta_L=_Q * (spec['mu_n'] + spec['mu_p']) / (spec['eps_r'] * _EPS0), ni2=spec['Nc'] * spec['Nv'] * np.exp(-spec['Eg'] / VT), eps=spec['eps_r'] * _EPS0)\ndef _fx_contact_densities(spec):\n    if spec['Phi'] < 0.0 or spec['Phi'] > spec['Eg']:\n        raise ValueError('the injection barrier must lie inside the transport gap')\n    c = _fx_derived_constants(spec)\n    p_an = spec['Nv'] * np.exp(-spec['Phi'] / c['VT'])\n    n_cat = spec['Nc'] * np.exp(-spec['Phi'] / c['VT'])\n    return np.array([c['ni2'] / p_an, p_an, n_cat, c['ni2'] / n_cat])\ndef start(sp,V,N):\n    c=_fx_derived_constants(sp); bc=_fx_contact_densities(sp)\n    return (np.linspace(0.0,c['Vbi']-V,N),np.linspace(bc[0],bc[2],N),np.linspace(bc[1],bc[3],N))\npsi,n,p=start(SPEC,0.0,41)", 'call': 'electron_density(psi,p,SPEC,0.0)', 'gold_call': '_oracle_electron_density(psi,p,SPEC,0.0)', 'tol': 1e-09}, {'setup': "import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ndef _fx_derived_constants(spec):\n    _Q=1.602176634e-19; _KB=1.380649e-23; _EPS0=8.8541878128e-12\n    if not isinstance(spec, dict) or 'Eg' not in spec:\n        raise ValueError('spec must be the declared device configuration')\n    if spec['T'] <= 0.0:\n        raise ValueError('the temperature must be positive')\n    VT = _KB * spec['T'] / _Q\n    return dict(VT=VT, Vbi=spec['Eg'] - 2.0 * spec['Phi'], beta_L=_Q * (spec['mu_n'] + spec['mu_p']) / (spec['eps_r'] * _EPS0), ni2=spec['Nc'] * spec['Nv'] * np.exp(-spec['Eg'] / VT), eps=spec['eps_r'] * _EPS0)\ndef _fx_contact_densities(spec):\n    if spec['Phi'] < 0.0 or spec['Phi'] > spec['Eg']:\n        raise ValueError('the injection barrier must lie inside the transport gap')\n    c = _fx_derived_constants(spec)\n    p_an = spec['Nv'] * np.exp(-spec['Phi'] / c['VT'])\n    n_cat = spec['Nc'] * np.exp(-spec['Phi'] / c['VT'])\n    return np.array([c['ni2'] / p_an, p_an, n_cat, c['ni2'] / n_cat])\ndef start(sp,V,N):\n    c=_fx_derived_constants(sp); bc=_fx_contact_densities(sp)\n    return (np.linspace(0.0,c['Vbi']-V,N),np.linspace(bc[0],bc[2],N),np.linspace(bc[1],bc[3],N))\npsi,n,p=start(SPEC,0.6,51)", 'call': "electron_density(psi,p,SPEC,SPEC['Gex'])", 'gold_call': "_oracle_electron_density(psi,p,SPEC,SPEC['Gex'])", 'tol': 1e-09}, {'setup': "import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ndef raises(f):\n    try:\n        f()\n    except ValueError:\n        return 1.0\n    return 0.0\ndef _fx_derived_constants(spec):\n    _Q=1.602176634e-19; _KB=1.380649e-23; _EPS0=8.8541878128e-12\n    if not isinstance(spec, dict) or 'Eg' not in spec:\n        raise ValueError('spec must be the declared device configuration')\n    if spec['T'] <= 0.0:\n        raise ValueError('the temperature must be positive')\n    VT = _KB * spec['T'] / _Q\n    return dict(VT=VT, Vbi=spec['Eg'] - 2.0 * spec['Phi'], beta_L=_Q * (spec['mu_n'] + spec['mu_p']) / (spec['eps_r'] * _EPS0), ni2=spec['Nc'] * spec['Nv'] * np.exp(-spec['Eg'] / VT), eps=spec['eps_r'] * _EPS0)\ndef _fx_contact_densities(spec):\n    if spec['Phi'] < 0.0 or spec['Phi'] > spec['Eg']:\n        raise ValueError('the injection barrier must lie inside the transport gap')\n    c = _fx_derived_constants(spec)\n    p_an = spec['Nv'] * np.exp(-spec['Phi'] / c['VT'])\n    n_cat = spec['Nc'] * np.exp(-spec['Phi'] / c['VT'])\n    return np.array([c['ni2'] / p_an, p_an, n_cat, c['ni2'] / n_cat])\ndef start(sp,V,N):\n    c=_fx_derived_constants(sp); bc=_fx_contact_densities(sp)\n    return (np.linspace(0.0,c['Vbi']-V,N),np.linspace(bc[0],bc[2],N),np.linspace(bc[1],bc[3],N))\npsi,n,p=start(SPEC,0.0,41)", 'call': "raises(lambda: electron_density(psi, p[:-1], SPEC, SPEC['Gex']))", 'gold_call': "raises(lambda: _oracle_electron_density(psi, p[:-1], SPEC, SPEC['Gex']))", 'tol': 1e-09}]
