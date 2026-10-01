"""
Return the strain-energy density and the kinetic-energy density at a point from the stress, strain, density and velocity there (the paper's Eqs. 18-19). Stress and strain are given in engineering Voigt order (xx, yy, xy) with the shear entry the engineering shear strain. These two densities are the ones that enter the dynamic J-integral integrand.

The dynamic energy release rate balances stored elastic energy against kinetic energy carried by the moving material; both densities appear explicitly in the crack-tip contour and domain integrals.

Returns
-------
return (2,) float64: [strain-energy density, kinetic-energy density]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def energy_densities(sigma, eps, rho, vel):
    """sigma: (3,) Voigt stress (xx, yy, xy); eps: (3,) Voigt strain with engineering
    shear; rho: density; vel: (2,) velocity. Returns a float64 array [W, K] with the
    strain-energy and kinetic-energy densities (paper Eqs. 18-19)."""
    return np.zeros(2)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 4: strain-energy and kinetic-energy densities (Eqs. 18-19)."""

import numpy as np


def _oracle_energy_densities(sigma, eps, rho, vel):
    sigma = np.asarray(sigma, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    if sigma.shape != (3,) or eps.shape != (3,):
        raise ValueError("sigma and eps must be Voigt triples")
    if vel.shape != (2,):
        raise ValueError("vel must hold two components")
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be positive and finite")
    if not (np.all(np.isfinite(sigma)) and np.all(np.isfinite(eps)) and np.all(np.isfinite(vel))):
        raise ValueError("non-finite field value")
    W = 0.5 * (sigma[0] * eps[0] + sigma[1] * eps[1] + sigma[2] * eps[2])
    K = 0.5 * rho * (vel[0] ** 2 + vel[1] ** 2)
    return np.array([W, K], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nsigma=_n.array([1.2,0.8,0.3]);eps=_n.array([0.011,0.006,0.004]);rho=1.0;vel=_n.array([0.2,0.1])', "call": 'energy_densities(sigma, eps, rho, vel)', "gold_call": '_oracle_energy_densities(sigma, eps, rho, vel)', "tol": 1e-12},
        {"setup": 'import numpy as _n\nsigma=_n.array([2.0,-0.5,0.9]);eps=_n.array([0.02,-0.004,0.012]);rho=1.5;vel=_n.array([-0.3,0.25])', "call": 'energy_densities(sigma, eps, rho, vel)', "gold_call": '_oracle_energy_densities(sigma, eps, rho, vel)', "tol": 1e-12},
        {"setup": 'import numpy as _n\nsigma=_n.array([0.5,0.5,0.0]);eps=_n.array([0.005,0.005,0.0]);rho=0.8;vel=_n.array([0.0,0.4])', "call": 'energy_densities(sigma, eps, rho, vel)', "gold_call": '_oracle_energy_densities(sigma, eps, rho, vel)', "tol": 1e-12},
    ]
