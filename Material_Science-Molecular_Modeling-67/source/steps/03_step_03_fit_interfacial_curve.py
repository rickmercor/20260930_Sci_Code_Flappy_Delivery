"""
Fit a common exponential surface-tension curve to all films.

At a specified thickness vector, minimize the unweighted sum of squared residuals



$$

S(a,b,c)=\sum_i[a\exp(bh_i)+c-\sigma_i]^2

$$



with $a,c\in\mathbb{R}$ and $b_{\min}\leq b\leq b_{\max}<0$. Units are mN/m

for $a,c,\sigma$ and inverse angstroms for $b$. For each $b$, the linear least

squares problem in $a,c$ can be eliminated, leaving a one-dimensional objective.

Return the global minimizer, using the smaller $b$ for an exact tie. The supported

data have at least four distinct positive thicknesses and a nonconstant tension

vector; a numerically unresolved fit is rejected. Endpoints are admissible here.

Returns
-------
Return a numerical array of shape (3,) containing the fitted coefficients in the order (a, b, c).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def fit_interfacial_curve(
    thickness: np.ndarray, tension: np.ndarray, bounds: np.ndarray
) -> np.ndarray:
    r"""Fit the joint interfacial response.

    Parameters
    ----------
    thickness, tension : np.ndarray
        Shape $(m,)$, $m\geq4$, finite data, with distinct positive thicknesses
        in angstroms and nonconstant tensions in mN/m.
    bounds : np.ndarray
        Shape $(2,)$, finite strictly ordered negative bounds on $b$.

    Returns
    -------
    result : np.ndarray
        Shape $(3,)$ with fitted coefficients ordered $(a,b,c)$.

    Raises
    ------
    ValueError
        If inputs violate the real, finite, shape, distinctness, or bound
        requirements, or the exponential design is numerically rank deficient.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _fit_inputs(thickness, tension, bounds):
    thickness = _finite_array(thickness, "thickness", 1)
    tension = _finite_array(tension, "tension", 1)
    bounds = _finite_array(bounds, "bounds", 1)
    if thickness.size < 4 or tension.shape != thickness.shape:
        raise ValueError("at least four aligned observations are required")
    if np.any(thickness <= 0) or np.unique(thickness).size != thickness.size:
        raise ValueError("thicknesses must be positive and distinct")
    if np.ptp(tension) == 0:
        raise ValueError("constant tensions do not identify the decay rate")
    if bounds.shape != (2,) or not bounds[0] < bounds[1] < 0:
        raise ValueError("bounds must be two strictly ordered negative values")
    return thickness, tension, bounds


def _profile_fit(b, thickness, tension):
    exponential = np.exp(b * thickness)
    centered = exponential - exponential.mean()
    denominator = centered @ centered
    if denominator <= np.finfo(float).tiny:
        raise ValueError("the exponential design is numerically rank deficient")
    a = (centered @ (tension - tension.mean())) / denominator
    c = tension.mean() - a * exponential.mean()
    residual = a * exponential + c - tension
    slope = residual @ (a * thickness * exponential)
    return float(a), float(c), float(residual @ residual), float(slope)


def _oracle_fit_interfacial_curve(
    thickness: np.ndarray, tension: np.ndarray, bounds: np.ndarray
) -> np.ndarray:
    """Evaluate the reference numerical map."""
    thickness, tension, bounds = _fit_inputs(thickness, tension, bounds)
    grid = np.linspace(bounds[0], bounds[1], 129)
    candidates = [float(bounds[0]), float(bounds[1])]
    gradients = [_profile_fit(float(b), thickness, tension)[3] for b in grid]
    for index in range(grid.size - 1):
        left, right = float(grid[index]), float(grid[index + 1])
        if gradients[index] == 0:
            candidates.append(left)
        if gradients[index] * gradients[index + 1] < 0:
            candidates.append(
                brentq(
                    lambda b: _profile_fit(b, thickness, tension)[3],
                    left,
                    right,
                    xtol=1e-14,
                    rtol=1e-14,
                )
            )
    best = min(candidates, key=lambda b: (_profile_fit(b, thickness, tension)[2], b))
    a, c, _, _ = _profile_fit(best, thickness, tension)
    return np.array([a, best, c])

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
thickness = np.array([11., 14., 18., 23., 30., 39.])
tension = 2.76375 * d
""",
            "call": "fit_interfacial_curve(thickness.copy(), tension.copy(), bounds.copy())",
            "gold_call": "_oracle_fit_interfacial_curve(thickness.copy(), tension.copy(), bounds.copy())",
        },
        {
            "setup": """import numpy as np
from scipy.optimize import brentq
thickness = np.array([1., 3., 6., 9.])
tension = -8. * np.exp(-.4 * thickness) + 12.
bounds = np.array([-.3, -.01])
""",
            "call": "fit_interfacial_curve(thickness.copy(), tension.copy(), bounds.copy())",
            "gold_call": "_oracle_fit_interfacial_curve(thickness.copy(), tension.copy(), bounds.copy())",
        },
        {
            "setup": """import numpy as np
from scipy.optimize import brentq
thickness = np.array([20., 2., 11., 5., 31.])
tension = -2. * np.exp(-.07 * thickness) + 3.
bounds = np.array([-.3, -.01])
""",
            "call": "fit_interfacial_curve(thickness.copy(), tension.copy(), bounds.copy())",
            "gold_call": "_oracle_fit_interfacial_curve(thickness.copy(), tension.copy(), bounds.copy())",
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
thickness = np.ones(6)
tension = d.copy()

def _capture_value_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_capture_value_error(fit_interfacial_curve, thickness.copy(), tension.copy(), bounds.copy())",
            "gold_call": "_capture_value_error(_oracle_fit_interfacial_curve, thickness.copy(), tension.copy(), bounds.copy())",
        },
    ]
