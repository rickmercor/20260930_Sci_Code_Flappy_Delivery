"""
Compute the curvature of the peak pressure while its maximizing compressibility moves.

At fixed population perturbation $\eta$, let $\lambda_*(\eta)$ maximize the

first film's signed pressure on the given closed coupling interval. The

supported maximum is unique, interior, and regular, with a positive coupling

derivative at the left endpoint and a negative derivative at the right.

Implicit differentiation of its stationarity condition gives



$$

\Pi_{0,\lambda}(\lambda_*(\eta),\eta)=0,\qquad

\lambda_*'(0)=-\frac{\Pi_{0,\lambda\eta}}{\Pi_{0,\lambda\lambda}}.

$$



For $G(\eta)=\Pi_0(\lambda_*(\eta),\eta)$, the required envelope curvature is



$$

G''(0)=\Pi_{0,\eta\eta}

-\frac{\Pi_{0,\lambda\eta}^2}{\Pi_{0,\lambda\lambda}}.

$$



All partial derivatives are evaluated at the coupled equilibrium and maximizing

coupling for $\eta=0$. The peak must remain a smooth interior maximum near zero,

so $\Pi_{0,\lambda\lambda}<0$. The unique-maximum property is a precondition

on the input family; a numerical root bracket alone is not a general proof of

that property. Film order specifies the target; mean populations, not an integer

rounding of them, define the continuous perturbation.

Returns
-------
Return one finite Python float giving the second population-direction derivative of the peak signed pressure in MPa.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_nanofilm_response(
    z: np.ndarray,
    pressure: np.ndarray,
    counts: np.ndarray,
    area: float,
    box_length: float,
    vapor_density: np.ndarray,
    gas_pressure: np.ndarray,
    coefficients: np.ndarray,
    inventory_direction: np.ndarray,
    coupling_interval: np.ndarray,
    bounds: np.ndarray,
) -> float:
    r"""Compute the population-uncertainty curvature of peak signed pressure.

    Parameters
    ----------
    z : np.ndarray
        Shape $(n,)$, $n\geq2$, finite increasing sample positions in angstroms.
    pressure : np.ndarray
        Finite real shape $(m,n,3)$, $m\geq4$, MPa samples ordered $xx,yy,zz$.
    counts, vapor_density, gas_pressure : np.ndarray
        Finite aligned shape $(m,)$ mean molecule counts, vapor densities in
        inverse cubic angstroms, and gas pressures in MPa; counts positive,
        gas properties nonnegative. Counts refer to $\eta=0$.
    area, box_length : float
        Finite positive geometry in squared angstroms and angstroms.
    coefficients : np.ndarray
        Finite shape $(4,)$ bulk EOS coefficients in ascending pressure power.
    inventory_direction : np.ndarray
        Finite conservative shape $(m,)$ direction $dN/d\eta$; the sum
        tolerance is $10^{-12}\max(1,\sum_i|u_i|)$; zero is permitted.
    coupling_interval : np.ndarray
        Finite shape $(2,)$ with $0\leq\lambda_{\min}<\lambda_{\max}$.
        The first pressure has a unique regular interior maximum, a positive
        coupling derivative at the left endpoint, and a negative one at the
        right endpoint; the maximum remains interior for small $\eta$.
    bounds : np.ndarray
        Finite shape $(2,)$ with strictly ordered negative fit-decay bounds.

    Returns
    -------
    result : float
        The envelope curvature $G''(0)$ in MPa per squared dimensionless $\eta$.

    Raises
    ------
    ValueError
        If any preceding step's realness, finiteness, alignment, physical
        domain, conservative-direction, identifiable interior fit, or regular
        stationary-response requirement fails; the interval is invalid or
        does not bracket the maximum; or its coupling curvature is nonnegative.
    RuntimeError
        If a coupled thickness solve fails to converge in 500 iterations,
        peak search fails in 100 iterations, or stationarity exceeds
        $10^{-8}$ MPa at the computed maximizing coupling.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _oracle_compute_nanofilm_response(
    z: np.ndarray,
    pressure: np.ndarray,
    counts: np.ndarray,
    area: float,
    box_length: float,
    vapor_density: np.ndarray,
    gas_pressure: np.ndarray,
    coefficients: np.ndarray,
    inventory_direction: np.ndarray,
    coupling_interval: np.ndarray,
    bounds: np.ndarray,
) -> float:
    """Evaluate the reference numerical map."""
    counts = _finite_array(counts, "counts", 1)
    interval = _finite_array(coupling_interval, "coupling_interval", 1)
    direction, area = _validated_inventory_direction(
        inventory_direction, counts.size, area
    )
    if interval.shape != (2,) or not 0 <= interval[0] < interval[1]:
        raise ValueError(
            "the coupling interval must have ordered nonnegative endpoints"
        )
    tension = _oracle_integrate_surface_tension(z, pressure)

    def _response_at(coupling):
        thickness = _oracle_solve_coupled_thickness(
            counts,
            area,
            box_length,
            vapor_density,
            gas_pressure,
            tension,
            coefficients,
            coupling,
            bounds,
        )
        return _oracle_compute_coupled_sensitivity(
            thickness,
            tension,
            vapor_density,
            gas_pressure,
            coefficients,
            coupling,
            bounds,
            direction,
            area,
        )[0]

    left = _response_at(float(interval[0]))
    right = _response_at(float(interval[1]))
    if left[1] <= 0 or right[1] >= 0:
        raise ValueError("the interval must bracket a regular pressure maximum")
    peak = brentq(
        lambda coupling: _response_at(coupling)[1],
        float(interval[0]),
        float(interval[1]),
        xtol=1e-10,
        rtol=1e-12,
        maxiter=100,
    )
    response = _response_at(peak)
    if abs(response[1]) > 1e-8:
        raise RuntimeError("the maximizing coupling did not satisfy stationarity")
    if response[3] >= 0:
        raise ValueError("the stationary pressure must be a regular maximum")
    return float(response[5] - response[4] ** 2 / response[3])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and invalid-input comparisons."""
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
inventory_direction = np.array([80., -120., 60., -50., 40., -10.])
coupling_interval = np.array([0., 10.])
""",
            "call": "compute_nanofilm_response(z.copy(), pressure.copy(), counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), coefficients.copy(), inventory_direction.copy(), coupling_interval.copy(), bounds.copy())",
            "gold_call": "_oracle_compute_nanofilm_response(z.copy(), pressure.copy(), counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), coefficients.copy(), inventory_direction.copy(), coupling_interval.copy(), bounds.copy())",
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
inventory_direction = np.array([80., -120., 60., -50., 40., -10.])
coupling_interval = np.array([0., 10.])
inventory_direction = np.zeros(6)
""",
            "call": "compute_nanofilm_response(z.copy(), pressure.copy(), counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), coefficients.copy(), inventory_direction.copy(), coupling_interval.copy(), bounds.copy())",
            "gold_call": "_oracle_compute_nanofilm_response(z.copy(), pressure.copy(), counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), coefficients.copy(), inventory_direction.copy(), coupling_interval.copy(), bounds.copy())",
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
inventory_direction = np.array([80., -120., 60., -50., 40., -10.])
coupling_interval = np.array([0., 10.])
inventory_direction = np.array([0., -90., 60., -20., 80., -30.])
""",
            "call": "compute_nanofilm_response(z.copy(), pressure.copy(), counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), coefficients.copy(), inventory_direction.copy(), coupling_interval.copy(), bounds.copy())",
            "gold_call": "_oracle_compute_nanofilm_response(z.copy(), pressure.copy(), counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), coefficients.copy(), inventory_direction.copy(), coupling_interval.copy(), bounds.copy())",
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
inventory_direction = np.array([80., -120., 60., -50., 40., -10.])
coupling_interval = np.array([0., 10.])
coupling_interval = np.array([0., 1.])

def _capture_value_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_capture_value_error(compute_nanofilm_response, z.copy(), pressure.copy(), counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), coefficients.copy(), inventory_direction.copy(), coupling_interval.copy(), bounds.copy())",
            "gold_call": "_capture_value_error(_oracle_compute_nanofilm_response, z.copy(), pressure.copy(), counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), coefficients.copy(), inventory_direction.copy(), coupling_interval.copy(), bounds.copy())",
        },
    ]
