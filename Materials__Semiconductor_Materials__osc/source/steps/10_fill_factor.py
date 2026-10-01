"""
Return the fill factor of the illuminated device, extrapolated to the continuum from solutions on the two given grids. Raise ValueError if the fine grid is not finer than the coarse one, or if the recombination reduction factor is negative.

The fill factor is the largest output power density the illuminated device can deliver, divided by the product of the short-circuit current density and the open-circuit voltage. Both of those and the maximum power point each need their own self-consistent solution, so the whole current-voltage characteristic is exercised. The discretisation error of the flux scheme falls with the square of the node spacing, which is what makes the two-grid extrapolation worthwhile and what its weights must be built from.

Returns
-------
float, dimensionless
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fill_factor(spec: dict, n_coarse: int, n_fine: int) -> float:
    """Return the fill factor of the illuminated device, extrapolated to the continuum from solutions on the two given grids. Raise ValueError if the fine grid is not finer than the coarse one, or if the recombination reduction factor is negative.

    Parameters
    ----------
    spec : dict
        The declared device configuration, in SI units, with keys 'T' (temperature, K), 'd' (layer thickness, m), 'Eg' (transport gap, eV), 'Phi' (injection barrier at each contact, eV), 'mu_n' and 'mu_p' (mobilities, m^2 V^-1 s^-1), 'eps_r' (relative permittivity), 'Nc' and 'Nv' (effective densities of states, m^-3), 'Gex' (generation rate, m^-3 s^-1) and 'gamma' (recombination reduction factor).
    n_coarse : int
        Node count of the coarser of the two grids.
    n_fine : int
        Node count of the finer of the two grids.

    Returns
    -------
    float, dimensionless
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




def _oracle_fill_factor(spec: dict, n_coarse: int, n_fine: int) -> float:
    from scipy.optimize import minimize_scalar
    if int(n_fine) <= int(n_coarse):
        raise ValueError("the fine grid must be finer than the coarse one")
    if spec["gamma"] < 0.0:
        raise ValueError("the recombination reduction factor cannot be negative")

    c = _oracle_derived_constants(spec)
    bc = _oracle_contact_densities(spec)

    # gate 1: the contact densities must obey mass action
    if abs(bc[0] * bc[1] - c["ni2"]) > 1.0e-6 * c["ni2"]:
        raise ValueError("the contact densities violate mass action")
    # gate 2: the Bernoulli function must satisfy B(x) - B(-x) = -x
    xs = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    if float(np.max(np.abs(_oracle_bernoulli(xs) - _oracle_bernoulli(-xs) + xs))) > 1.0e-12:
        raise ValueError("the Bernoulli function fails its defining identity")
    # gate 3: in the dark at zero bias the terminal current must vanish
    psi, n, p = _oracle_self_consistent_state(0.0, 0.0, spec, int(n_coarse))
    if abs(_oracle_terminal_current(psi, n, p, spec)) > 1.0e-6:
        raise ValueError("the dark device carries current at zero bias")

    # gate 4: the converged state must be stationary under one further sweep, measured on
    # the terminal current, which is the quantity the sweeps actually settle. The carrier
    # densities keep jittering in the minority tail long after the current has stopped moving.
    for ng in (int(n_coarse), int(n_fine)):
        psi, n, p = _oracle_self_consistent_state(0.0, spec["Gex"], spec, ng)
        J0 = _oracle_terminal_current(psi, n, p, spec)
        psi2 = _oracle_poisson_step(psi, n, p, spec, 0.0)
        n2 = _oracle_electron_density(psi2, p, spec, spec["Gex"])
        p2 = _oracle_hole_density(psi2, n2, spec, spec["Gex"])
        J1 = _oracle_terminal_current(psi2, n2, p2, spec)
        if abs(J1 - J0) > 1.0e-6 * abs(J0):
            raise ValueError("one further sweep still moves the terminal current")

    def _ff(n_grid):
        n_grid = int(n_grid)
        psi, ne, ph = _oracle_self_consistent_state(0.0, spec["Gex"], spec, n_grid)
        Jsc = _oracle_terminal_current(psi, ne, ph, spec)
        Voc = _oracle_open_circuit_voltage(spec, n_grid)

        def _power(V):
            ps, nn, pp = _oracle_self_consistent_state(V, spec["Gex"], spec, n_grid)
            return V * _oracle_terminal_current(ps, nn, pp, spec)

        r = minimize_scalar(_power, bounds=(0.0, Voc), method="bounded",
                            options=dict(xatol=1.0e-10))
        return (-float(r.fun)) / (abs(Jsc) * Voc)

    f_c = _ff(n_coarse)
    f_f = _ff(n_fine)
    # the discretisation is second order, so Richardson removes the leading error
    r = (float(n_fine) - 1.0) / (float(n_coarse) - 1.0)
    return float(f_f + (f_f - f_c) / (r * r - 1.0))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)',
         "call": 'fill_factor(SPEC,41,81)',
         "gold_call": '_oracle_fill_factor(SPEC,41,81)',
         "tol": 1e-05},
        {"setup": "import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ns=dict(SPEC); s['gamma']=0.3",
         "call": 'fill_factor(s,41,81)',
         "gold_call": '_oracle_fill_factor(s,41,81)',
         "tol": 1e-05},
        {"setup": "import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ns=dict(SPEC); s['Gex']=3e27",
         "call": 'fill_factor(s,41,81)',
         "gold_call": '_oracle_fill_factor(s,41,81)',
         "tol": 1e-05},
        {"setup": 'import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ndef raises(f):\n    try:\n        f()\n    except ValueError:\n        return 1.0\n    return 0.0',
         "call": 'raises(lambda: fill_factor(SPEC, 201, 101))',
         "gold_call": 'raises(lambda: _oracle_fill_factor(SPEC, 201, 101))',
         "tol": 1e-05},
    ]
