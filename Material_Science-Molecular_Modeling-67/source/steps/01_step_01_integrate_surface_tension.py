"""
Recover each film's single-interface surface tension from its sampled stress profile.

A free-standing film has two equivalent liquid-vapor interfaces. The mechanical

surface tension in $\mathrm{mN}/\mathrm{m}$ is



$$

\sigma_i=0.05\int [P_{zz,i}-(P_{xx,i}+P_{yy,i})/2]\,dz.

$$



Pressures are in MPa and positions in angstroms; the prefactor combines the two

interfaces and unit conversion. Use the composite trapezoidal rule on the supplied

strictly increasing, possibly nonuniform grid. The anisotropy is zero beyond it.

Returns
-------
Return a numerical array of shape (m,) containing single-interface surface tensions in mN/m.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def integrate_surface_tension(z: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    r"""Integrate the transverse stress deficit.

    Parameters
    ----------
    z : np.ndarray
        Shape $(n,)$ with $n\geq2$, finite increasing positions in angstroms.
    pressure : np.ndarray
        Shape $(m,n,3)$, $m\geq1$, finite MPa samples ordered $xx,yy,zz$.

    Returns
    -------
    result : np.ndarray
        Shape $(m,)$, signed single-interface tensions in mN/m.

    Raises
    ------
    ValueError
        If arrays are not real finite numerical data, shapes disagree, or the
        grid has fewer than two points or is not strictly increasing.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _finite_array(value, name, ndim):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    try:
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numerical") from error
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} has invalid dimensions or nonfinite values")
    return result


def _finite_scalar(value, name):
    result = _finite_array(value, name, 0)
    return float(result)


def _oracle_integrate_surface_tension(
    z: np.ndarray, pressure: np.ndarray
) -> np.ndarray:
    """Evaluate the reference numerical map."""
    z = _finite_array(z, "z", 1)
    pressure = _finite_array(pressure, "pressure", 3)
    if z.size < 2 or np.any(np.diff(z) <= 0):
        raise ValueError("z must be strictly increasing with at least two points")
    if pressure.shape != (pressure.shape[0], z.size, 3) or pressure.shape[0] < 1:
        raise ValueError("pressure must have shape (m, n, 3), m >= 1")
    anisotropy = pressure[:, :, 2] - 0.5 * (pressure[:, :, 0] + pressure[:, :, 1])
    return 0.05 * np.sum(
        0.5 * (anisotropy[:, 1:] + anisotropy[:, :-1]) * np.diff(z), axis=1
    )

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
""",
            "call": "integrate_surface_tension(z.copy(), pressure.copy())",
            "gold_call": "_oracle_integrate_surface_tension(z.copy(), pressure.copy())",
        },
        {
            "setup": """import numpy as np
z = np.array([0., 2.])
pressure = np.ones((1, 2, 3))
""",
            "call": "integrate_surface_tension(z.copy(), pressure.copy())",
            "gold_call": "_oracle_integrate_surface_tension(z.copy(), pressure.copy())",
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
pressure[:, :, 2] -= 3.
""",
            "call": "integrate_surface_tension(z.copy(), pressure.copy())",
            "gold_call": "_oracle_integrate_surface_tension(z.copy(), pressure.copy())",
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
z[1] = z[0]

def _capture_value_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_capture_value_error(integrate_surface_tension, z.copy(), pressure.copy())",
            "gold_call": "_capture_value_error(_oracle_integrate_surface_tension, z.copy(), pressure.copy())",
        },
    ]
