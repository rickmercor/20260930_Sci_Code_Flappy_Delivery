"""
Compute the full second-order pressure response to bulk compressibility and population redistribution.

A perturbation $N(\eta)=N+\eta u$ preserves total population when $\sum_i u_i=0$.

Let $x=(h_0,\ldots,h_{m-1},a,b,c)$, $f_i=a\exp(bh_i)+c$, $r_i=f_i-\sigma_i$,

and $J_{ik}=\partial f_i/\partial\theta_k$ for $\theta=(a,b,c)$. At an interior

joint fit and reservoir equilibrium, the augmented stationary residual is



$$

E(x;\lambda,\eta)=\begin{pmatrix}

h\,[\rho_l(P_g-\Pi;\lambda)-\rho_g]-N(\eta)/A+\rho_gL_z\\

J^Tr

\end{pmatrix}=0.

$$



The first block uses componentwise products. With $\mathcal A=E_x$, differentiate

for $p,q\in\{\lambda,\eta\}$:



$$

x_p=-\mathcal A^{-1}E_p,\qquad

x_{pq}=-\mathcal A^{-1}D^2E[(x_p,e_p),(x_q,e_q)].

$$



Here $e_p$ selects a parameter direction; $D^2E$ differentiates the augmented

residual along straight directions, excluding the unknown $x_{pq}$ term. The

fit block includes the exact residual curvature, so its second differential

contains the third derivatives of the exponential fit, not just $J^TJ$.

For $Q_i(x)=-20ab\exp(bh_i)$, the pressure jet follows from



$$

Q_{i,p}=Q_{i,x}x_p,\qquad

Q_{i,pq}=Q_{i,x}x_{pq}+x_p^TQ_{i,xx}x_q.

$$



The supplied state must already satisfy the inventory balance. This step has no

nominal count or box-length input with which to check that balance. It returns

columns $(\Pi,\Pi_\lambda,\Pi_\eta,\Pi_{\lambda\lambda},

\Pi_{\lambda\eta},\Pi_{\eta\eta})$, all in MPa because the two parameters

are dimensionless. An active fit bound or singular stationary system does not

have the supported smooth second-order response.

Returns
-------
Return a numerical array of shape (m, 6) containing pressure, two first derivatives, and three second derivatives in the stated order, all in MPa.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_coupled_sensitivity(
    thickness: np.ndarray,
    tension: np.ndarray,
    vapor_density: np.ndarray,
    gas_pressure: np.ndarray,
    coefficients: np.ndarray,
    coupling: float,
    bounds: np.ndarray,
    inventory_direction: np.ndarray,
    area: float,
) -> np.ndarray:
    r"""Compute the second-order coupled pressure response at equilibrium.

    Parameters
    ----------
    thickness, tension, vapor_density, gas_pressure : np.ndarray
        Aligned finite real shape $(m,)$ data, $m\geq4$: distinct positive
        equilibrium thicknesses in angstroms, nonconstant tensions in mN/m,
        nonnegative vapor densities in inverse cubic angstroms and gas
        pressures in MPa.
    coefficients : np.ndarray
        Finite real shape $(4,)$ bulk-density polynomial coefficients in
        ascending pressure-power order, with positive baseline density.
    coupling : float
        Finite dimensionless $\lambda\geq0$; at zero take derivatives of
        the smooth algebraic continuation of the equation of state.
    bounds : np.ndarray
        Finite shape $(2,)$, strictly ordered negative decay-rate bounds.
    inventory_direction : np.ndarray
        Finite real shape $(m,)$ direction $u=dN/d\eta$ in mean molecule
        counts per dimensionless $\eta$; its sum is zero within $10^{-12}$
        times the larger of one and its absolute-entry sum; zero is allowed.
    area : float
        Positive finite film area in squared angstroms.

    Returns
    -------
    result : np.ndarray
        Shape $(m,6)$ in the order $(\Pi,\Pi_\lambda,\Pi_\eta,
        \Pi_{\lambda\lambda},\Pi_{\lambda\eta},\Pi_{\eta\eta})$, in MPa.

    Raises
    ------
    ValueError
        If realness, finiteness, alignment, area, population conservation,
        fit or bulk-response domains fail; liquid density does not exceed
        vapor density; the fit is within $10^{-9}$ inverse angstroms of a
        bound; or the row/column-scaled augmented stationary matrix has
        condition number above $10^{12}$ or a zero row/column scale.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validated_inventory_direction(inventory_direction, size, area):
    direction = _finite_array(inventory_direction, "inventory_direction", 1)
    area = _finite_scalar(area, "area")
    if direction.shape != (size,) or area <= 0:
        raise ValueError(
            "the population direction must align with films and area must be positive"
        )
    if abs(direction.sum()) > 1e-12 * max(1.0, np.abs(direction).sum()):
        raise ValueError(
            "the population direction must conserve total mean molecule count"
        )
    return direction, area


def _scaled_stationary_solve(matrix, rhs):
    row_scale = np.max(np.abs(matrix), axis=1)
    if np.any(row_scale == 0):
        raise ValueError("the coupled stationary system is singular")
    row_scaled = matrix / row_scale[:, None]
    column_scale = np.max(np.abs(row_scaled), axis=0)
    if np.any(column_scale == 0):
        raise ValueError("the coupled stationary system is singular")
    scaled = row_scaled / column_scale[None, :]
    if np.linalg.cond(scaled) > 1e12:
        raise ValueError("the scaled coupled stationary system is numerically singular")
    if rhs.ndim == 1:
        return np.linalg.solve(scaled, rhs / row_scale) / column_scale
    return np.linalg.solve(scaled, rhs / row_scale[:, None]) / column_scale[:, None]


def _directional_interfacial_terms(
    direction, thickness, a, b, exponential, pressure_gradient
):
    size = thickness.size
    dh = direction[:size]
    da, db, dc = direction[size:]
    exponent_derivative = b * dh + thickness * db
    de = exponential * exponent_derivative
    df = da * exponential + a * de + dc
    dp = pressure_gradient @ direction
    dj = np.column_stack(
        (
            de,
            (da * thickness + a * dh) * exponential + a * thickness * de,
            np.zeros(size),
        )
    )
    return dh, da, db, exponent_derivative, de, df, dp, dj


def _oracle_compute_coupled_sensitivity(
    thickness: np.ndarray,
    tension: np.ndarray,
    vapor_density: np.ndarray,
    gas_pressure: np.ndarray,
    coefficients: np.ndarray,
    coupling: float,
    bounds: np.ndarray,
    inventory_direction: np.ndarray,
    area: float,
) -> np.ndarray:
    """Evaluate the reference numerical map."""
    thickness, tension, bounds = _fit_inputs(thickness, tension, bounds)
    size = thickness.size
    vapor_density = _finite_array(vapor_density, "vapor_density", 1)
    gas_pressure = _finite_array(gas_pressure, "gas_pressure", 1)
    coefficients = _finite_array(coefficients, "coefficients", 1)
    direction, area = _validated_inventory_direction(inventory_direction, size, area)
    if vapor_density.shape != (size,) or gas_pressure.shape != (size,):
        raise ValueError("phase data must align with the films")
    if np.any(vapor_density < 0) or np.any(gas_pressure < 0):
        raise ValueError("gas properties must be nonnegative")
    parameters = _oracle_fit_interfacial_curve(thickness, tension, bounds)
    a, b, c = parameters
    if min(b - bounds[0], bounds[1] - b) <= 1e-9:
        raise ValueError("the fit must be a regular interior optimum")
    partials = _oracle_differentiate_disjoining_pressure(thickness, parameters)
    liquid_pressure = gas_pressure - partials[:, 0]
    response = _oracle_evaluate_bulk_response(liquid_pressure, coefficients, coupling)
    if np.any(response[:, 0] <= vapor_density):
        raise ValueError("liquid density must exceed vapor density")
    exponential = np.exp(b * thickness)
    residual = a * exponential + c - tension
    jacobian = np.column_stack(
        (exponential, a * thickness * exponential, np.ones(size))
    )
    hessian = jacobian.T @ jacobian
    mixed = residual @ (thickness * exponential)
    hessian[0, 1] += mixed
    hessian[1, 0] += mixed
    hessian[1, 1] += residual @ (a * thickness**2 * exponential)
    cross = jacobian.T * (a * b * exponential)
    cross[0] += residual * b * exponential
    cross[1] += residual * a * exponential * (1.0 + b * thickness)
    pressure_gradient = np.column_stack((np.diag(partials[:, 1]), partials[:, 2:]))
    stationary_matrix = np.zeros((size + 3, size + 3))
    stationary_matrix[:size, :size] = np.diag(response[:, 0] - vapor_density)
    stationary_matrix[:size] -= (thickness * response[:, 1])[
        :, None
    ] * pressure_gradient
    stationary_matrix[size:, :size] = cross
    stationary_matrix[size:, size:] = hessian
    forcing = np.zeros((size + 3, 2))
    forcing[:size, 0] = -thickness * response[:, 2]
    forcing[:size, 1] = direction / area
    first = _scaled_stationary_solve(stationary_matrix, forcing)
    first_pressure = pressure_gradient @ first
    c1, c2, c3 = coefficients[1:]
    pressure = liquid_pressure
    correction_slope = c1 + 2 * c2 * pressure + 3 * c3 * pressure**2
    correction_curvature = 2 * c2 + 6 * c3 * pressure

    def _second_pressure(left, right, left_coupling, right_coupling):
        hp, ap, bp, zp, ep, fp, qp, jp = _directional_interfacial_terms(
            left, thickness, a, b, exponential, pressure_gradient
        )
        hq, aq, bq, zq, eq, fq, qq, jq = _directional_interfacial_terms(
            right, thickness, a, b, exponential, pressure_gradient
        )
        epq = exponential * (zp * zq + bp * hq + bq * hp)
        fpq = ap * eq + aq * ep + a * epq
        qpq = -20.0 * (
            (ap * bq + aq * bp) * exponential
            + (ap * b + a * bp) * eq
            + (aq * b + a * bq) * ep
            + a * b * epq
        )
        jpq = np.column_stack(
            (
                epq,
                (ap * hq + aq * hp) * exponential
                + (ap * thickness + a * hp) * eq
                + (aq * thickness + a * hq) * ep
                + a * thickness * epq,
                np.zeros(size),
            )
        )
        density_p = left_coupling * response[:, 2] - response[:, 1] * qp
        density_q = right_coupling * response[:, 2] - response[:, 1] * qq
        density_pq = (
            -left_coupling * correction_slope * qq
            - right_coupling * correction_slope * qp
            + coupling * (correction_curvature * qp * qq - correction_slope * qpq)
        )
        balance_second = hp * density_q + hq * density_p + thickness * density_pq
        fit_second = fpq @ jacobian + fp @ jq + fq @ jp + residual @ jpq
        second_state = _scaled_stationary_solve(
            stationary_matrix, -np.concatenate((balance_second, fit_second))
        )
        return qpq + pressure_gradient @ second_state

    lambda_lambda = _second_pressure(first[:, 0], first[:, 0], 1.0, 1.0)
    lambda_eta = _second_pressure(first[:, 0], first[:, 1], 1.0, 0.0)
    eta_eta = _second_pressure(first[:, 1], first[:, 1], 0.0, 0.0)
    return np.column_stack(
        (partials[:, 0], first_pressure, lambda_lambda, lambda_eta, eta_eta)
    )

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
tension = _oracle_integrate_surface_tension(z.copy(), pressure.copy())
thickness = _oracle_solve_coupled_thickness(counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), tension.copy(), coefficients.copy(), coupling, bounds.copy())
""",
            "call": "compute_coupled_sensitivity(thickness.copy(), tension.copy(), vapor_density.copy(), gas_pressure.copy(), coefficients.copy(), coupling, bounds.copy(), inventory_direction.copy(), area)",
            "gold_call": "_oracle_compute_coupled_sensitivity(thickness.copy(), tension.copy(), vapor_density.copy(), gas_pressure.copy(), coefficients.copy(), coupling, bounds.copy(), inventory_direction.copy(), area)",
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
tension = _oracle_integrate_surface_tension(z.copy(), pressure.copy())
thickness = _oracle_solve_coupled_thickness(counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), tension.copy(), coefficients.copy(), coupling, bounds.copy())
inventory_direction = np.zeros(6)
""",
            "call": "compute_coupled_sensitivity(thickness.copy(), tension.copy(), vapor_density.copy(), gas_pressure.copy(), coefficients.copy(), coupling, bounds.copy(), inventory_direction.copy(), area)",
            "gold_call": "_oracle_compute_coupled_sensitivity(thickness.copy(), tension.copy(), vapor_density.copy(), gas_pressure.copy(), coefficients.copy(), coupling, bounds.copy(), inventory_direction.copy(), area)",
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
area, box_length, coupling = 1000., 100., 0.
coefficients = np.array([.028, 6e-5, -1e-7, 2e-10])
bounds = np.array([-.30, -.01])
pressure = np.broadcast_to(gas_pressure[:, None, None], (6, 7, 3)).copy()
pressure[:, :, 2] += d[:, None] * w[None, :]
inventory_direction = np.array([80., -120., 60., -50., 40., -10.])
tension = _oracle_integrate_surface_tension(z.copy(), pressure.copy())
thickness = _oracle_solve_coupled_thickness(counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), tension.copy(), coefficients.copy(), coupling, bounds.copy())
""",
            "call": "compute_coupled_sensitivity(thickness.copy(), tension.copy(), vapor_density.copy(), gas_pressure.copy(), coefficients.copy(), coupling, bounds.copy(), inventory_direction.copy(), area)",
            "gold_call": "_oracle_compute_coupled_sensitivity(thickness.copy(), tension.copy(), vapor_density.copy(), gas_pressure.copy(), coefficients.copy(), coupling, bounds.copy(), inventory_direction.copy(), area)",
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
tension = _oracle_integrate_surface_tension(z.copy(), pressure.copy())
thickness = _oracle_solve_coupled_thickness(counts.copy(), area, box_length, vapor_density.copy(), gas_pressure.copy(), tension.copy(), coefficients.copy(), coupling, bounds.copy())
inventory_direction[0] += 1.

def _capture_value_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_capture_value_error(compute_coupled_sensitivity, thickness.copy(), tension.copy(), vapor_density.copy(), gas_pressure.copy(), coefficients.copy(), coupling, bounds.copy(), inventory_direction.copy(), area)",
            "gold_call": "_capture_value_error(_oracle_compute_coupled_sensitivity, thickness.copy(), tension.copy(), vapor_density.copy(), gas_pressure.copy(), coefficients.copy(), coupling, bounds.copy(), inventory_direction.copy(), area)",
        },
    ]
