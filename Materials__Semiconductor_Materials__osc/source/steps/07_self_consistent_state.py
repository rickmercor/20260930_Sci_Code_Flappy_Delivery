"""
Iterate the potential and the two carrier densities to mutual self-consistency at the given bias and generation rate, and return them as the tuple (psi, n, p). Stop when the largest absolute change in the potential and the largest change in each density relative to that density's own maximum have all fallen below 1e-6, and raise ValueError if that has not happened within 2000 sweeps. Raise ValueError for a grid of fewer than five nodes.

The three equations are coupled: the densities set the space charge that shapes the potential, and the potential sets the fluxes that redistribute the densities. Solving them as one Newton system in the densities themselves is badly conditioned here, because the equilibrium electron and hole densities at opposite contacts differ by roughly fifteen orders of magnitude. Sweeping the three equations in turn until neither the potential nor either density moves is stable instead. The convergence test has to be scale aware: a pointwise relative test on the densities never passes, because each minority density at its own contact sits some fourteen orders below the majority there, so its relative motion stays large long after the solution has stopped moving in any physically meaningful sense.

Returns
-------
tuple (psi, n, p) of np.ndarray, each of shape (n_grid,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def self_consistent_state(V: float, G: float, spec: dict, n_grid: int) -> tuple:
    """Iterate the potential and the two carrier densities to mutual self-consistency at the given bias and generation rate, and return them as the tuple (psi, n, p). Stop when the largest absolute change in the potential and the largest change in each density relative to that density's own maximum have all fallen below 1e-6, and raise ValueError if that has not happened within 2000 sweeps. Raise ValueError for a grid of fewer than five nodes.

    Parameters
    ----------
    V : float
        Applied bias, in volts.
    G : float
        Uniform free-charge generation rate.
    spec : dict
        The declared device configuration, in SI units, with keys 'T' (temperature, K), 'd' (layer thickness, m), 'Eg' (transport gap, eV), 'Phi' (injection barrier at each contact, eV), 'mu_n' and 'mu_p' (mobilities, m^2 V^-1 s^-1), 'eps_r' (relative permittivity), 'Nc' and 'Nv' (effective densities of states, m^-3), 'Gex' (generation rate, m^-3 s^-1) and 'gamma' (recombination reduction factor).
    n_grid : int
        Number of uniformly spaced nodes, both contacts included.

    Returns
    -------
    tuple (psi, n, p) of np.ndarray, each of shape (n_grid,)
    """
    return (psi, n, p)

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




def _oracle_self_consistent_state(V: float, G: float, spec: dict, n_grid: int) -> tuple:
    n_grid = int(n_grid)
    if n_grid < 5:
        raise ValueError("the grid needs at least five nodes")
    c = _oracle_derived_constants(spec)
    bc = _oracle_contact_densities(spec)
    psi = np.linspace(0.0, c["Vbi"] - V, n_grid)
    n = np.linspace(bc[0], bc[2], n_grid)
    p = np.linspace(bc[1], bc[3], n_grid)
    for _ in range(2000):
        psi_new = _oracle_poisson_step(psi, n, p, spec, V)
        dpsi = float(np.max(np.abs(psi_new - psi)))
        psi = psi_new
        n_new = _oracle_electron_density(psi, p, spec, G)
        p_new = _oracle_hole_density(psi, n_new, spec, G)
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)',
         "call": "np.concatenate(self_consistent_state(0.0,SPEC['Gex'],SPEC,41))",
         "gold_call": "np.concatenate(_oracle_self_consistent_state(0.0,SPEC['Gex'],SPEC,41))",
         "tol": 0.0001},
        {"setup": 'import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)',
         "call": 'np.concatenate(self_consistent_state(0.0,0.0,SPEC,41))',
         "gold_call": 'np.concatenate(_oracle_self_consistent_state(0.0,0.0,SPEC,41))',
         "tol": 0.0001},
        {"setup": 'import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)',
         "call": "np.concatenate(self_consistent_state(0.7,SPEC['Gex'],SPEC,51))",
         "gold_call": "np.concatenate(_oracle_self_consistent_state(0.7,SPEC['Gex'],SPEC,51))",
         "tol": 0.0001},
        {"setup": 'import numpy as np\nSPEC=dict(T=300.0,d=100e-9,Eg=1.4,Phi=0.25,mu_n=2e-8,mu_p=2e-8,eps_r=3.5,Nc=1e26,Nv=1e26,Gex=1e28,gamma=1.0)\ndef raises(f):\n    try:\n        f()\n    except ValueError:\n        return 1.0\n    return 0.0',
         "call": "raises(lambda: self_consistent_state(0.0, SPEC['Gex'], SPEC, 3))",
         "gold_call": "raises(lambda: _oracle_self_consistent_state(0.0, SPEC['Gex'], SPEC, 3))",
         "tol": 0.0001},
    ]
