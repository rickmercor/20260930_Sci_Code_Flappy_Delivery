"""
Solve the mutually dependent thickness, surface-tension fit, and bulk density.

At each iteration, refit the entire thickness-tension dataset, compute signed

$\Pi_i$, impose $P_{l,i}=P_{g,i}-\Pi_i$, and update liquid density from its

pressure-dependent equation of state. The resulting thickness map is



$$

F_i(h;\lambda)=\frac{N_i/A-\rho_{g,i}L_z}

{\rho_l(P_{g,i}-\Pi_i(h);\lambda)-\rho_{g,i}}.

$$



Initialize with liquid density $c_0$ and return the fixed-point branch reached

by successive updates. Require $\max_i|F_i(h)-h_i|/h_i\leq10^{-12}$.

All earlier physical-domain restrictions apply at every iterate. Fits are

recomputed from every film after each update; measured tensions stay fixed.

Returns
-------
Return a numerical array of shape (m,) containing the converged coupled film thicknesses in angstroms.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_coupled_thickness(
    counts: np.ndarray,
    area: float,
    box_length: float,
    vapor_density: np.ndarray,
    gas_pressure: np.ndarray,
    tension: np.ndarray,
    coefficients: np.ndarray,
    coupling: float,
    bounds: np.ndarray,
) -> np.ndarray:
    r"""Find the branch reached from the zero-pressure liquid density.

    Parameters
    ----------
    counts, vapor_density, gas_pressure, tension : np.ndarray
        Aligned finite shape $(m,)$ vectors, $m\geq4$, in molecule counts,
        inverse cubic angstroms, MPa, and mN/m. Counts are positive and gas
        properties nonnegative; tensions are nonconstant.
    area, box_length : float
        Finite positive geometry in squared angstroms and angstroms.
    coefficients : np.ndarray
        Finite shape $(4,)$ coefficients ordered by increasing pressure power,
        with positive constant density, as defined in the bulk response.
    coupling : float
        Finite nonnegative dimensionless density-response multiplier.
    bounds : np.ndarray
        Finite shape $(2,)$, ordered negative decay-rate bounds.

    Returns
    -------
    result : np.ndarray
        Shape $(m,)$, converged positive thicknesses in angstroms.

    Raises
    ------
    ValueError
        If any input violates its domain or alignment, any iterate has repeated
        thicknesses, nonidentifiable fit, nonpositive excess inventory, density
        not exceeding vapor density, thickness outside $(0,L_z)$, liquid pressure
        outside $[0,200]$ MPa, or an invalid bulk response.
    RuntimeError
        If the relative thickness residual does not reach $10^{-12}$ within
        500 refitted density updates.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_coupled_thickness(
    counts: np.ndarray,
    area: float,
    box_length: float,
    vapor_density: np.ndarray,
    gas_pressure: np.ndarray,
    tension: np.ndarray,
    coefficients: np.ndarray,
    coupling: float,
    bounds: np.ndarray,
) -> np.ndarray:
    """Evaluate the reference numerical map."""
    gas_pressure = _finite_array(gas_pressure, "gas_pressure", 1)
    counts = _finite_array(counts, "counts", 1)
    coefficients = _finite_array(coefficients, "coefficients", 1)
    if gas_pressure.shape != counts.shape or np.any(gas_pressure < 0):
        raise ValueError("gas pressure must be aligned and nonnegative")
    if coefficients.shape != (4,) or coefficients[0] <= 0:
        raise ValueError(
            "four coefficients with positive baseline density are required"
        )
    thickness = _oracle_compute_film_thickness(
        counts, area, box_length, vapor_density, np.full(counts.size, coefficients[0])
    )
    for _ in range(500):
        parameters = _oracle_fit_interfacial_curve(thickness, tension, bounds)
        pressure = _oracle_differentiate_disjoining_pressure(thickness, parameters)[
            :, 0
        ]
        response = _oracle_evaluate_bulk_response(
            gas_pressure - pressure, coefficients, coupling
        )
        updated = _oracle_compute_film_thickness(
            counts, area, box_length, vapor_density, response[:, 0]
        )
        residual = np.max(np.abs(updated - thickness) / thickness)
        if residual <= 1e-12:
            return thickness
        thickness = updated
    raise RuntimeError("the thickness iteration did not converge in 500 updates")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return numerical test specifications."""
    return [
        {
            "setup": """import numpy as np
from scipy.optimize import brentq
z = np.array([-50., -30., -12., 0., 15., 35., 50.])
w = np.array([0., .25, 1., 1.4, .8, .15, 0.])
d = np.array([8.8, 10.1, 11.5, 12.9, 13.7, 14.1])
counts = np.array([330., 410., 520., 670., 860., 1110.])
vapor_density = np.array([.00011, .00012, .00010, .00013, .000115, .000105])
gas_pressure = np.array([.58, .63, .54, .68, .60, .56])
area, box_length, coupling = 1000., 100., 1.
coefficients = np.array([.028, 6e-5, -1e-7, 2e-10])
bounds = np.array([-.30, -.01])
pressure = np.broadcast_to(gas_pressure[:, None, None], (6, 7, 3)).copy()
pressure[:, :, 2] += d[:, None] * w[None, :]
tension = _oracle_integrate_surface_tension(z.copy(), pressure.copy())
""",
            "call": "solve_coupled_thickness(counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), tension.copy(), coefficients.copy(), coupling, bounds.copy())",
            "gold_call": "_oracle_solve_coupled_thickness(counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), tension.copy(), coefficients.copy(), coupling, bounds.copy())",
        },
        {
            "setup": """import numpy as np
from scipy.optimize import brentq
z = np.array([-50., -30., -12., 0., 15., 35., 50.])
w = np.array([0., .25, 1., 1.4, .8, .15, 0.])
d = np.array([8.8, 10.1, 11.5, 12.9, 13.7, 14.1])
counts = np.array([330., 410., 520., 670., 860., 1110.])
vapor_density = np.array([.00011, .00012, .00010, .00013, .000115, .000105])
gas_pressure = np.array([.58, .63, .54, .68, .60, .56])
area, box_length, coupling = 1000., 100., 1.
coefficients = np.array([.028, 6e-5, -1e-7, 2e-10])
bounds = np.array([-.30, -.01])
pressure = np.broadcast_to(gas_pressure[:, None, None], (6, 7, 3)).copy()
pressure[:, :, 2] += d[:, None] * w[None, :]
tension = _oracle_integrate_surface_tension(z.copy(), pressure.copy())
coupling = 0.
""",
            "call": "solve_coupled_thickness(counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), tension.copy(), coefficients.copy(), coupling, bounds.copy())",
            "gold_call": "_oracle_solve_coupled_thickness(counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), tension.copy(), coefficients.copy(), coupling, bounds.copy())",
        },
        {
            "setup": """import numpy as np
from scipy.optimize import brentq
z = np.array([-50., -30., -12., 0., 15., 35., 50.])
w = np.array([0., .25, 1., 1.4, .8, .15, 0.])
d = np.array([8.8, 10.1, 11.5, 12.9, 13.7, 14.1])
counts = np.array([330., 410., 520., 670., 860., 1110.])
vapor_density = np.array([.00011, .00012, .00010, .00013, .000115, .000105])
gas_pressure = np.array([.58, .63, .54, .68, .60, .56])
area, box_length, coupling = 1000., 100., 1.
coefficients = np.array([.028, 6e-5, -1e-7, 2e-10])
bounds = np.array([-.30, -.01])
pressure = np.broadcast_to(gas_pressure[:, None, None], (6, 7, 3)).copy()
pressure[:, :, 2] += d[:, None] * w[None, :]
tension = _oracle_integrate_surface_tension(z.copy(), pressure.copy())
""",
            "call": "solve_coupled_thickness(counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), tension.copy(), coefficients.copy(), coupling, bounds.copy())",
            "gold_call": "_oracle_solve_coupled_thickness(counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), tension.copy(), coefficients.copy(), coupling, bounds.copy())",
        },
        {
            "setup": """import numpy as np
from scipy.optimize import brentq
z = np.array([-50., -30., -12., 0., 15., 35., 50.])
w = np.array([0., .25, 1., 1.4, .8, .15, 0.])
d = np.array([8.8, 10.1, 11.5, 12.9, 13.7, 14.1])
counts = np.array([330., 410., 520., 670., 860., 1110.])
vapor_density = np.array([.00011, .00012, .00010, .00013, .000115, .000105])
gas_pressure = np.array([.58, .63, .54, .68, .60, .56])
area, box_length, coupling = 1000., 100., 1.
coefficients = np.array([.028, 6e-5, -1e-7, 2e-10])
bounds = np.array([-.30, -.01])
pressure = np.broadcast_to(gas_pressure[:, None, None], (6, 7, 3)).copy()
pressure[:, :, 2] += d[:, None] * w[None, :]
tension = _oracle_integrate_surface_tension(z.copy(), pressure.copy())
gas_pressure[0] = -1.

def _capture_value_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_capture_value_error(solve_coupled_thickness, counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), tension.copy(), coefficients.copy(), coupling, bounds.copy())",
            "gold_call": "_capture_value_error(_oracle_solve_coupled_thickness, counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), tension.copy(), coefficients.copy(), coupling, bounds.copy())",
        },
    ]
