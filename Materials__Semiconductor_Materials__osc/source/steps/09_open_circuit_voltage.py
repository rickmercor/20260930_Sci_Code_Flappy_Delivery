"""
Return the bias at which the illuminated device carries no terminal current, in volts. Raise ValueError if no such bias exists below flat band.

The illuminated current is negative at zero bias and turns positive before the applied bias reaches the built-in voltage, so the crossing is bracketed by those two values and can be found by bisection. Each evaluation of the current requires its own self-consistent solution, so the bracket should not be widened further than the physics requires.

Returns
-------
float, in volts
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def open_circuit_voltage(spec: dict, n_grid: int) -> float:
    """Return the bias at which the illuminated device carries no terminal current, in volts. Raise ValueError if no such bias exists below flat band.

    Parameters
    ----------
    spec : dict
        The declared device configuration, in SI units, with keys 'T' (temperature, K), 'd' (layer thickness, m), 'Eg' (transport gap, eV), 'Phi' (injection barrier at each contact, eV), 'mu_n' and 'mu_p' (mobilities, m^2 V^-1 s^-1), 'eps_r' (relative permittivity), 'Nc' and 'Nv' (effective densities of states, m^-3), 'Gex' (generation rate, m^-3 s^-1) and 'gamma' (recombination reduction factor).
    n_grid : int
        Number of uniformly spaced nodes, both contacts included.

    Returns
    -------
    float, in volts
    """
    return 0.0

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




def _oracle_open_circuit_voltage(spec: dict, n_grid: int) -> float:
    from scipy.optimize import brentq
    c = _oracle_derived_constants(spec)

    def _J(V):
        psi, n, p = _oracle_self_consistent_state(V, spec["Gex"], spec, n_grid)
        return _oracle_terminal_current(psi, n, p, spec)

    lo, hi = 0.0, c["Vbi"] - 1.0e-3
    if _J(lo) * _J(hi) > 0.0:
        raise ValueError("the illuminated current does not change sign below flat band")
    return float(brentq(_J, lo, hi, xtol=1.0e-12, rtol=8.9e-16, maxiter=300))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)',
         "call": 'open_circuit_voltage(SPEC,41)',
         "gold_call": '_oracle_open_circuit_voltage(SPEC,41)',
         "tol": 1e-06},
        {"setup": 'import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)',
         "call": 'open_circuit_voltage(SPEC,61)',
         "gold_call": '_oracle_open_circuit_voltage(SPEC,61)',
         "tol": 1e-06},
        {"setup": "import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ns=dict(SPEC); s['gamma']=0.2",
         "call": 'open_circuit_voltage(s,41)',
         "gold_call": '_oracle_open_circuit_voltage(s,41)',
         "tol": 1e-06},
        {"setup": 'import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ndef raises(f):\n    try:\n        f()\n    except ValueError:\n        return 1.0\n    return 0.0',
         "call": 'raises(lambda: open_circuit_voltage(dict(SPEC, Phi=0.69), 41))',
         "gold_call": 'raises(lambda: _oracle_open_circuit_voltage(dict(SPEC, Phi=0.69), 41))',
         "tol": 1e-06},
    ]
