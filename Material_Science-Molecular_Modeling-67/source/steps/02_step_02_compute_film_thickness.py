"""
Convert film inventory and phase densities into thermodynamic thickness.

The same vapor density describes the vapor inventory and density contrast. For

number densities in $\mathrm{\AA}^{-3}$, area in $\mathrm{\AA}^2$, and box length

in angstroms, the dividing-surface inventory is



$$

v_i=N_i/A-\rho_{g,i}L_z,\qquad h_i=\frac{v_i}{\rho_{l,i}-\rho_{g,i}}.

$$



The supported regime has $v_i>0$ and $0<h_i<L_z$. All vectors retain film order.

Returns
-------
Return a numerical array of shape (m,) containing thermodynamic film thicknesses in angstroms.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_film_thickness(
    counts: np.ndarray,
    area: float,
    box_length: float,
    vapor_density: np.ndarray,
    liquid_density: np.ndarray,
) -> np.ndarray:
    r"""Compute inventory-consistent thicknesses.

    Parameters
    ----------
    counts : np.ndarray
        Shape $(m,)$, $m\geq1$, finite positive molecule counts.
    area, box_length : float
        Positive finite area in squared angstroms and length in angstroms.
    vapor_density, liquid_density : np.ndarray
        Shape $(m,)$, nonnegative vapor density and strictly larger liquid
        density in inverse cubic angstroms.

    Returns
    -------
    result : np.ndarray
        Shape $(m,)$, film thicknesses in angstroms.

    Raises
    ------
    ValueError
        If data are nonreal, nonfinite, misaligned, or violate the stated
        positivity, density ordering, excess-inventory, or thickness regime.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_film_thickness(
    counts: np.ndarray,
    area: float,
    box_length: float,
    vapor_density: np.ndarray,
    liquid_density: np.ndarray,
) -> np.ndarray:
    """Evaluate the reference numerical map."""
    counts = _finite_array(counts, "counts", 1)
    vapor_density = _finite_array(vapor_density, "vapor_density", 1)
    liquid_density = _finite_array(liquid_density, "liquid_density", 1)
    area = _finite_scalar(area, "area")
    box_length = _finite_scalar(box_length, "box_length")
    if (
        counts.size < 1
        or vapor_density.shape != counts.shape
        or liquid_density.shape != counts.shape
    ):
        raise ValueError(
            "inventory and density vectors must have matching nonempty shapes"
        )
    if area <= 0 or box_length <= 0 or np.any(counts <= 0) or np.any(vapor_density < 0):
        raise ValueError("invalid geometry, inventory, or vapor density")
    if np.any(liquid_density <= vapor_density):
        raise ValueError("liquid density must exceed vapor density")
    excess = counts / area - vapor_density * box_length
    thickness = excess / (liquid_density - vapor_density)
    if np.any(excess <= 0) or np.any(thickness >= box_length):
        raise ValueError(
            "the film must have positive thickness strictly below box length"
        )
    return thickness

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
liquid_density = np.full(6, .03)
""",
            "call": "compute_film_thickness(counts.copy(), area, box_length, vapor_density.copy(), liquid_density.copy())",
            "gold_call": "_oracle_compute_film_thickness(counts.copy(), area, box_length, vapor_density.copy(), liquid_density.copy())",
        },
        {
            "setup": """import numpy as np
counts = np.array([1.])
vapor_density = np.zeros(1)
liquid_density = np.ones(1)
area, box_length = 2., 1.
""",
            "call": "compute_film_thickness(counts.copy(), area, box_length, vapor_density.copy(), liquid_density.copy())",
            "gold_call": "_oracle_compute_film_thickness(counts.copy(), area, box_length, vapor_density.copy(), liquid_density.copy())",
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
liquid_density = np.full(6, .028)
counts[0] = 11.00001
""",
            "call": "compute_film_thickness(counts.copy(), area, box_length, vapor_density.copy(), liquid_density.copy())",
            "gold_call": "_oracle_compute_film_thickness(counts.copy(), area, box_length, vapor_density.copy(), liquid_density.copy())",
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
liquid_density = vapor_density.copy()

def _capture_value_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_capture_value_error(compute_film_thickness, counts.copy(), area, box_length, vapor_density.copy(), liquid_density.copy())",
            "gold_call": "_capture_value_error(_oracle_compute_film_thickness, counts.copy(), area, box_length, vapor_density.copy(), liquid_density.copy())",
        },
    ]
