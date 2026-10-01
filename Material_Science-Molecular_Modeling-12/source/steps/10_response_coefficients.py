"""
Evaluate the macroscopic response implied by the structure-aware equation of state at (T, rho): the isochoric derivative (dP/dT) at fixed rho and the isothermal derivative (dP/drho) at fixed T, each by a central difference with the relative steps deltas = (dT/T, drho/rho), every displaced state being re-solved self-consistently on the grid (n_grid, dr) with the parameters params unchanged (theta0 and n00 stay at their reference values, so a displaced state is genuinely off-reference), and from them the implied macroscopic isothermal compressibility kappa = 1 / (rho dP/drho) and thermal expansion coefficient alpha = (dP/dT) / (rho dP/drho) in reduced units. Return [dP/dT, dP/drho, kappa, alpha]. Raise ValueError if either relative step is not positive.

The first-estimate parameters were fixed so that the structureless equation of state returns the macroscopic compressibility and expansion coefficient (both equal to their reduced inputs); with the structure included, the same parameters no longer do, and the discrepancy measures how much of the response is carried by correlations. The temperature derivative probes the explicit temperature dependence of the particle pressure and of the potential as well as the temperature dependence of the structure through beta u. The source estimates the macroscopic compressibility from simulations at displaced densities as kappa* = 0.98 against the input 1.00; the HNC route predicts a larger value.

Returns
-------
A (4,) float64 array [dP/dT at fixed rho, dP/drho at fixed T, implied kappa, implied alpha] in reduced units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def response_coefficients(temperature, rho, rcut, fcut, params, n_grid, dr, deltas):
    """Evaluate the macroscopic response implied by the structure-aware equation of state at
    (T, rho): the isochoric derivative (dP/dT) at fixed rho and the isothermal derivative
    (dP/drho) at fixed T, each by a central difference with the relative steps deltas =
    (dT/T, drho/rho), every displaced state being re-solved self-consistently on the grid
    (n_grid, dr) with the parameters params unchanged (theta0 and n00 stay at their
    reference values, so a displaced state is genuinely off-reference), and from them the
    implied macroscopic isothermal compressibility kappa = 1 / (rho dP/drho) and thermal
    expansion coefficient alpha = (dP/dT) / (rho dP/drho) in reduced units. A (4,) float64
    array [dP/dT at fixed rho, dP/drho at fixed T, implied kappa, implied alpha] in reduced
    units."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _pressure(temperature, rho, rcut, fcut, params, n_grid, dr):
    g = _oracle_self_consistent_structure(temperature, rho, rcut, fcut, params, n_grid, dr)
    return _oracle_eos_state(g, temperature, rho, rcut, fcut, params, dr)

def _oracle_response_coefficients(temperature, rho, rcut, fcut, params, n_grid, dr, deltas):
    dt_rel, dn_rel = [float(x) for x in deltas]
    if dt_rel <= 0 or dn_rel <= 0:
        raise ValueError("difference steps must be positive")
    ht = dt_rel*temperature
    hn = dn_rel*rho
    p_tp = float(_pressure(temperature + ht, rho, rcut, fcut, params, n_grid, dr)[2])
    p_tm = float(_pressure(temperature - ht, rho, rcut, fcut, params, n_grid, dr)[2])
    p_np = float(_pressure(temperature, rho + hn, rcut, fcut, params, n_grid, dr)[2])
    p_nm = float(_pressure(temperature, rho - hn, rcut, fcut, params, n_grid, dr)[2])
    dpdt = (p_tp - p_tm)/(2.0*ht)
    dpdn = (p_np - p_nm)/(2.0*hn)
    kappa_impl = 1.0/(rho*dpdn)
    alpha_impl = dpdt/(rho*dpdn)
    return np.array([dpdt, dpdn, kappa_impl, alpha_impl])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\n# normal: benchmark liquid response with symmetric relative differences\nparams = np.array([0.01105407273373585, 1.0, 0.11605782726626415, 1.0111776310806926, 29.34480813105209, 11.008324924443508])\ntemperature = 0.01105407273373585\nrho = 1.0\nrcut = 2.1564\nfcut = 1.33\nn_grid = 100\ndr = 0.025\ndeltas = (1.0e-2, 1.0e-2)\n',
         'call': 'response_coefficients(temperature, rho, rcut, fcut, params, n_grid, dr, deltas)',
         'gold_call': '_oracle_response_coefficients(temperature, rho, rcut, fcut, params, n_grid, dr, deltas)'},
        {'setup': 'import numpy as np\n# boundary: minimum permitted HNC grid length n_grid = 8\nparams = np.array([0.1, 1.0, 0.0, 1000.0, 0.0, 1.0])\ntemperature = 0.1\nrho = 0.1\nrcut = 0.5\nfcut = 1.0\nn_grid = 8\ndr = 0.06666666666666667\ndeltas = (1.0e-2, 1.0e-2)\n',
         'call': 'response_coefficients(temperature, rho, rcut, fcut, params, n_grid, dr, deltas)',
         'gold_call': '_oracle_response_coefficients(temperature, rho, rcut, fcut, params, n_grid, dr, deltas)'},
        {'setup': 'import numpy as np\n# edge: strongly unequal but positive temperature and density difference steps\nparams = np.array([0.1, 1.0, 0.0, 10.0, 0.0, 1.0])\ntemperature = 0.1\nrho = 0.1\nrcut = 1.0\nfcut = 2.0\nn_grid = 24\ndr = 0.05\ndeltas = (2.0e-3, 5.0e-2)\n',
         'call': 'response_coefficients(temperature, rho, rcut, fcut, params, n_grid, dr, deltas)',
         'gold_call': '_oracle_response_coefficients(temperature, rho, rcut, fcut, params, n_grid, dr, deltas)'},
    ]
