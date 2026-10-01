"""
Orchestrator. On the uniform periodic grid of M points of the cell [0, a), build the external potential, compute the Gross-Pitaevskii ground-state density for coupling g, compute the proximal density for the regularisation parameter eps, form the potential that the source's inversion scheme determines at this finite eps from the two densities, and return the error of that potential against the potential the inversion is meant to recover for this model, measured in the norm of the source's potential space. Call the earlier step functions rather than reimplementing any of them. Raise ValueError if a, g or eps is out of range, if M is below 8, if an upstream solve does not converge, or if the proximal-density displacement is not finite.

The source's Figure 1 plots exactly this error against eps: it is the single number that tells whether the density space, the duality mapping, the guiding functional and the finite-eps potential formula have all been taken as the source prescribes.

Returns
-------
float, the potential error in the source's norm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def inversion_error(a: float, M: int, g: float, eps: float) -> float:
    """float, the potential error in the source's norm."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
 
def _oracle_inversion_error(a: float, M: int, g: float, eps: float) -> float:
    """ORCHESTRATOR (Sec IV B): reference GPE density on the M-point grid, proximal density at
    the given ε, inverted potential v_ε = (1/ε) J(ρ_ε − ρ_gs) (Eq 36), and its error against
    the sought potential v_g = g ρ_gs measured in X* = H^1_{per,hom}."""
    a, M, g, eps = float(a), int(M), float(g), float(eps)
    if a <= 0.0 or M < 8 or g < 0.0 or eps <= 0.0:
        raise ValueError("need a > 0, M >= 8, g >= 0, eps > 0")
    vext = _oracle_external_potential(a, M)
    rho_gs = _oracle_gpe_ground_state(vext, a, g)
    rho_eps = _oracle_proximal_density(rho_gs, vext, a, eps)
    dist = _oracle_density_norm(rho_eps - rho_gs, a)
    if not np.isfinite(dist):
        raise ValueError("the proximal-density displacement is not finite")
    v_eps = _oracle_duality_map(rho_eps - rho_gs, a) / eps
    return _oracle_potential_norm(v_eps - g * rho_gs, a)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "a, M, g, eps = 10.0, 256, 1.0, 1e-2",
            "call": "inversion_error(a, M, g, eps)",
            "gold_call": "_oracle_inversion_error(a, M, g, eps)",
        },
        {
            "setup": "a, M, g, eps = 8.0, 96, 0.5, 1e-1",
            "call": "inversion_error(a, M, g, eps)",
            "gold_call": "_oracle_inversion_error(a, M, g, eps)",
        },
        {
            "setup": "a, M, g, eps = 10.0, 128, 2.0, 1e-3",
            "call": "inversion_error(a, M, g, eps)",
            "gold_call": "_oracle_inversion_error(a, M, g, eps)",
        },
        {
            "setup": "a, M, g, eps = 10.0, 64, 0.0, 1e-2",
            "call": "inversion_error(a, M, g, eps)",
            "gold_call": "_oracle_inversion_error(a, M, g, eps)",
        },
    ]
