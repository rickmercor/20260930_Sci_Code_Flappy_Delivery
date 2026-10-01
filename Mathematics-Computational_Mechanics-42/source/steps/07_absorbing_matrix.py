"""
Return the coefficient matrix of the source's absorbing boundary condition, which relates the boundary traction to the boundary velocity so that outgoing waves leave the domain without reflecting (the paper's Eq. 19). The matrix is built from the density and the two Lame parameters and is NOT isotropic: it applies one wave impedance along the boundary normal and a different one in the directions tangential to it, because pressure and shear waves travel at different speeds. Recover the exact matrix, including which impedance goes with which direction, from the paper.

A finite computational domain would otherwise reflect outgoing waves back into the region of interest and pollute the solution; an absorbing condition mimics an unbounded medium by absorbing energy at the rate the outgoing wave carries it.

Returns
-------
return (d, d) float64: the absorbing boundary coefficient matrix
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def absorbing_matrix(rho, lam, mu, n):
    """rho: density; lam, mu: first and second Lame parameters; n: (d,) outward unit
    normal of the boundary. Returns a float64 array (d, d) with the source's absorbing
    boundary coefficient matrix (paper Eq. 19). Raises ValueError if n is not a unit
    vector."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 7: absorbing boundary coefficient matrix (Eq. 19)."""

import numpy as np


def _oracle_absorbing_matrix(rho, lam, mu, n):
    nv = np.asarray(n, dtype=np.float64)
    nn = float(nv @ nv)
    if not np.isfinite(nn) or abs(nn - 1.0) > 1e-10:
        raise ValueError("n must be a unit vector")
    P = np.outer(nv, nv)
    I = np.eye(nv.size)
    return np.sqrt(rho * (lam + 2.0 * mu)) * P + np.sqrt(rho * mu) * (I - P)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nrho=2600.0;lam=4.0e9;mu=4.0e9;n=_n.array([1.0,0.0])', "call": 'absorbing_matrix(rho, lam, mu, n)', "gold_call": '_oracle_absorbing_matrix(rho, lam, mu, n)', "tol": 1e-06},
        {"setup": 'import numpy as _n\nrho=2600.0;lam=4.0e9;mu=4.0e9;n=_n.array([0.6,0.8])', "call": 'absorbing_matrix(rho, lam, mu, n)', "gold_call": '_oracle_absorbing_matrix(rho, lam, mu, n)', "tol": 1e-06},
        {"setup": 'import numpy as _n\nrho=2200.0;lam=6.0e9;mu=3.0e9;_v=_n.array([1.0,2.0,2.0]);n=_v/_n.linalg.norm(_v)', "call": 'absorbing_matrix(rho, lam, mu, n)', "gold_call": '_oracle_absorbing_matrix(rho, lam, mu, n)', "tol": 1e-06},
    ]
