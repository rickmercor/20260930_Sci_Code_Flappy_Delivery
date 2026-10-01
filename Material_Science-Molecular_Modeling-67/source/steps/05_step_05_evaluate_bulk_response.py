"""
Evaluate liquid density and the differential response of the implicit reservoir.

Let $C(P)=c_1P+c_2P^2+c_3P^3$ and



$$

\rho_l(P;\lambda)=c_0+\lambda C(P),\qquad

\rho_{l,P}=\lambda(c_1+2c_2P+3c_3P^2),\qquad

\rho_{l,\lambda}=C(P).

$$



Input coefficients are ordered by increasing power of pressure. Pressure is in

MPa, density in inverse cubic angstroms, and $\lambda$ is dimensionless. The

supported pressure interval is the closed interval $[0,200]$ MPa, with positive

density and nonnegative isothermal density response at every supplied pressure.

Returns
-------
Return a numerical array of shape (m, 3) containing density and its pressure and coupling derivatives.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_bulk_response(
    liquid_pressure: np.ndarray, coefficients: np.ndarray, coupling: float
) -> np.ndarray:
    r"""Compute the bulk density and its partial derivatives.

    Parameters
    ----------
    liquid_pressure : np.ndarray
        Shape $(m,)$, $m\geq1$, finite pressures in $[0,200]$ MPa.
    coefficients : np.ndarray
        Shape $(4,)$, real finite $(c_0,c_1,c_2,c_3)$, with $c_0>0$;
        $c_k$ has units $\mathrm{\AA}^{-3}\mathrm{MPa}^{-k}$.
    coupling : float
        Finite dimensionless $\lambda\geq0$.

    Returns
    -------
    result : np.ndarray
        Shape $(m,3)$, columns density, its pressure derivative, and its
        coupling derivative, in inverse cubic angstroms, inverse cubic
        angstroms per MPa, and inverse cubic angstroms respectively.

    Raises
    ------
    ValueError
        If input shapes, realness, finiteness, or parameter domains are invalid,
        or evaluated density is nonpositive or its pressure derivative negative.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_evaluate_bulk_response(
    liquid_pressure: np.ndarray, coefficients: np.ndarray, coupling: float
) -> np.ndarray:
    """Evaluate the reference numerical map."""
    liquid_pressure = _finite_array(liquid_pressure, "liquid_pressure", 1)
    coefficients = _finite_array(coefficients, "coefficients", 1)
    coupling = _finite_scalar(coupling, "coupling")
    if liquid_pressure.size < 1 or np.any(
        (liquid_pressure < 0) | (liquid_pressure > 200)
    ):
        raise ValueError("liquid pressure must lie in [0, 200] MPa")
    if coefficients.shape != (4,) or coefficients[0] <= 0 or coupling < 0:
        raise ValueError("invalid equation of state or coupling")
    c0, c1, c2, c3 = coefficients
    correction = liquid_pressure * (c1 + liquid_pressure * (c2 + liquid_pressure * c3))
    density = c0 + coupling * correction
    pressure_derivative = coupling * (
        c1 + liquid_pressure * (2 * c2 + 3 * liquid_pressure * c3)
    )
    if np.any(density <= 0) or np.any(pressure_derivative < 0):
        raise ValueError(
            "the liquid response must have positive density and nonnegative slope"
        )
    return np.column_stack((density, pressure_derivative, correction))

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
liquid_pressure = np.array([1., 30., 100.])
""",
            "call": "evaluate_bulk_response(liquid_pressure.copy(), coefficients.copy(), coupling)",
            "gold_call": "_oracle_evaluate_bulk_response(liquid_pressure.copy(), coefficients.copy(), coupling)",
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
liquid_pressure = np.array([0., 200.])
coupling = 0.
""",
            "call": "evaluate_bulk_response(liquid_pressure.copy(), coefficients.copy(), coupling)",
            "gold_call": "_oracle_evaluate_bulk_response(liquid_pressure.copy(), coefficients.copy(), coupling)",
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
liquid_pressure = np.array([200.])
coupling = 1.5
""",
            "call": "evaluate_bulk_response(liquid_pressure.copy(), coefficients.copy(), coupling)",
            "gold_call": "_oracle_evaluate_bulk_response(liquid_pressure.copy(), coefficients.copy(), coupling)",
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
liquid_pressure = np.array([-1.])

def _capture_value_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_capture_value_error(evaluate_bulk_response, liquid_pressure.copy(), coefficients.copy(), coupling)",
            "gold_call": "_capture_value_error(_oracle_evaluate_bulk_response, liquid_pressure.copy(), coefficients.copy(), coupling)",
        },
    ]
