"""
Return the derived constants the rest of the calculation needs, as a dictionary with keys 'VT', 'Vbi', 'beta_L', 'ni2' and 'eps': the thermal voltage, the built-in voltage, the bimolecular recombination coefficient, the square of the intrinsic density, and the absolute permittivity. Raise ValueError for a non-positive temperature.

The built-in voltage follows from the transport gap and the two injection barriers. The recombination coefficient is not a free parameter of the model: the source fixes it from the transport coefficients and the permittivity, and the declared reduction factor then scales it. The intrinsic density follows from the two effective densities of states and the gap by mass action.

Returns
-------
dict with keys 'VT', 'Vbi', 'beta_L', 'ni2', 'eps', all SI floats
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def derived_constants(spec: dict) -> dict:
    """Return the derived constants the rest of the calculation needs, as a dictionary with keys 'VT', 'Vbi', 'beta_L', 'ni2' and 'eps': the thermal voltage, the built-in voltage, the bimolecular recombination coefficient, the square of the intrinsic density, and the absolute permittivity. Raise ValueError for a non-positive temperature.

    Parameters
    ----------
    spec : dict
        The declared device configuration, in SI units, with keys 'T' (temperature, K), 'd' (layer thickness, m), 'Eg' (transport gap, eV), 'Phi' (injection barrier at each contact, eV), 'mu_n' and 'mu_p' (mobilities, m^2 V^-1 s^-1), 'eps_r' (relative permittivity), 'Nc' and 'Nv' (effective densities of states, m^-3), 'Gex' (generation rate, m^-3 s^-1) and 'gamma' (recombination reduction factor).

    Returns
    -------
    dict with keys 'VT', 'Vbi', 'beta_L', 'ni2', 'eps', all SI floats
    """
    return {}

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




def _oracle_derived_constants(spec: dict) -> dict:
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)',
         "call": "np.array([derived_constants(SPEC)[q] for q in ('VT','Vbi','beta_L','ni2','eps')])",
         "gold_call": "np.array([_oracle_derived_constants(SPEC)[q] for q in ('VT','Vbi','beta_L','ni2','eps')])"},
        {"setup": "import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ns=dict(SPEC); s['T']=200.0",
         "call": "np.array([derived_constants(s)[q] for q in ('VT','Vbi','beta_L','ni2','eps')])",
         "gold_call": "np.array([_oracle_derived_constants(s)[q] for q in ('VT','Vbi','beta_L','ni2','eps')])"},
        {"setup": "import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ns=dict(SPEC); s['Eg']=1.8; s['Phi']=0.1",
         "call": "np.array([derived_constants(s)[q] for q in ('VT','Vbi','beta_L','ni2','eps')])",
         "gold_call": "np.array([_oracle_derived_constants(s)[q] for q in ('VT','Vbi','beta_L','ni2','eps')])"},
        {"setup": 'import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ndef raises(f):\n    try:\n        f()\n    except ValueError:\n        return 1.0\n    return 0.0',
         "call": 'raises(lambda: derived_constants(dict(SPEC, T=-5.0)))',
         "gold_call": 'raises(lambda: _oracle_derived_constants(dict(SPEC, T=-5.0)))'},
    ]
