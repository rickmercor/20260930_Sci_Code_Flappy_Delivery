"""
Return the equilibrium carrier densities held at the two ohmic contacts, as the four numbers [electrons at the anode, holes at the anode, electrons at the cathode, holes at the cathode]. Raise ValueError if the injection barrier does not lie inside the transport gap.

Each contact is ohmic and pins both carrier densities at their equilibrium values. The majority density at a contact is set by that contact's injection barrier, and the minority density follows from mass action. The anode is the hole-collecting contact and the cathode the electron-collecting one.

Returns
-------
np.ndarray of shape (4,), in inverse cubic metres
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def contact_densities(spec: dict) -> np.ndarray:
    """Return the equilibrium carrier densities held at the two ohmic contacts, as the four numbers [electrons at the anode, holes at the anode, electrons at the cathode, holes at the cathode]. Raise ValueError if the injection barrier does not lie inside the transport gap.

    Parameters
    ----------
    spec : dict
        The declared device configuration, in SI units, with keys 'T' (temperature, K), 'd' (layer thickness, m), 'Eg' (transport gap, eV), 'Phi' (injection barrier at each contact, eV), 'mu_n' and 'mu_p' (mobilities, m^2 V^-1 s^-1), 'eps_r' (relative permittivity), 'Nc' and 'Nv' (effective densities of states, m^-3), 'Gex' (generation rate, m^-3 s^-1) and 'gamma' (recombination reduction factor).

    Returns
    -------
    np.ndarray of shape (4,), in inverse cubic metres
    """
    return None

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




def _oracle_contact_densities(spec: dict) -> np.ndarray:
    if spec["Phi"] < 0.0 or spec["Phi"] > spec["Eg"]:
        raise ValueError("the injection barrier must lie inside the transport gap")
    c = _oracle_derived_constants(spec)
    p_an = spec["Nv"] * np.exp(-spec["Phi"] / c["VT"])
    n_cat = spec["Nc"] * np.exp(-spec["Phi"] / c["VT"])
    return np.array([c["ni2"] / p_an, p_an, n_cat, c["ni2"] / n_cat])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)',
         "call": 'contact_densities(SPEC)',
         "gold_call": '_oracle_contact_densities(SPEC)'},
        {"setup": "import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ns=dict(SPEC); s['Phi']=0.35",
         "call": 'contact_densities(s)',
         "gold_call": '_oracle_contact_densities(s)'},
        {"setup": "import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ns=dict(SPEC); s['T']=250.0",
         "call": 'contact_densities(s)',
         "gold_call": '_oracle_contact_densities(s)'},
        {"setup": 'import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ndef raises(f):\n    try:\n        f()\n    except ValueError:\n        return 1.0\n    return 0.0',
         "call": 'raises(lambda: contact_densities(dict(SPEC, Phi=2.0)))',
         "gold_call": 'raises(lambda: _oracle_contact_densities(dict(SPEC, Phi=2.0)))'},
    ]
