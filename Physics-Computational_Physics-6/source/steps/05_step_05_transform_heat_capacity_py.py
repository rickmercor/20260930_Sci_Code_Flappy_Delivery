"""
Calculate the transformed volumetric heat capacity required for transient conduction in each omnidirectional cell. Use the determinant of the recovered local Jacobian to transform the background thermal-storage property.

For transient conduction, the volumetric heat capacity transforms with the local Jacobian determinant as

$$

(\rho c)_i=\frac{\rho_0c_0}{J_i},

$$

where

$$

J_i=\det\boldsymbol{\Lambda}_i.

$$

A local transformation with $J_i>1$ decreases the volumetric heat capacity relative to the background, while $0<J_i<1$ increases it. The determinant must remain positive for the orientation-preserving transformations used here.

Returns
-------
numerical NumPy array with shape (N,), containing the transformed volumetric heat capacity of each cell
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transform_heat_capacity(
    rho_c0: float,
    determinants: "np.ndarray",
) -> "np.ndarray":
    """
    Transform the background volumetric heat capacity in each cell.

    Parameters
    ----------
    rho_c0 : float
        Positive background volumetric heat capacity.
    determinants : np.ndarray
        Positive local Jacobian determinants with shape (N,).

    Returns
    -------
    heat_capacities : np.ndarray
        Transformed volumetric heat capacities with shape (N,).

    Raises
    ------
    ValueError
        If `rho_c0` is non-finite or non-positive, or if `determinants`
        contains any non-finite or non-positive value.
    """
    return heat_capacities

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_transform_heat_capacity(
    rho_c0: float,
    determinants: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation of the transient heat-capacity transformation."""
    determinants = np.asarray(determinants, dtype=float)

    if not np.isscalar(rho_c0) or not np.isfinite(float(rho_c0)):
        raise ValueError("rho_c0 must be a finite scalar.")

    rho_c0 = float(rho_c0)

    if rho_c0 <= 0.0:
        raise ValueError("rho_c0 must be positive.")

    if determinants.ndim != 1 or determinants.size < 1:
        raise ValueError(
            "determinants must have shape (N,) with N >= 1."
        )

    if not np.all(np.isfinite(determinants)):
        raise ValueError(
            "determinants must contain only finite values."
        )

    if np.any(determinants <= 0.0):
        raise ValueError(
            "Every Jacobian determinant must be positive."
        )

    heat_capacities = rho_c0 / determinants

    return heat_capacities

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test cases for transform_heat_capacity."""
    return [
        {
            "setup": """import numpy as np

gradients = np.array([
    [0.34, 0.79],
    [0.72, 0.31],
    [0.46, -0.88],
], dtype=float)

kappa_p = np.array([0.22, 1.85, 4.60], dtype=float)
kappa_0 = 1.0
rho_c0 = 1000.0

magnitudes = np.linalg.norm(gradients, axis=1)
scale_factors = 1.0 / magnitudes

determinants = (
    scale_factors**2
    * kappa_0
    / kappa_p
)
""",
            "call": "transform_heat_capacity(rho_c0, determinants.copy())",
            "gold_call": "_oracle_transform_heat_capacity(rho_c0, determinants.copy())",
        },
        {
            "setup": """import numpy as np

rho_c0 = 1000.0
determinants = np.array([1.0, 1.0], dtype=float)
""",
            "call": "transform_heat_capacity(rho_c0, determinants.copy())",
            "gold_call": "_oracle_transform_heat_capacity(rho_c0, determinants.copy())",
        },
        {
            "setup": """import numpy as np

rho_c0 = 500.0
determinants = np.array([0.05, 20.0], dtype=float)
""",
            "call": "transform_heat_capacity(rho_c0, determinants.copy())",
            "gold_call": "_oracle_transform_heat_capacity(rho_c0, determinants.copy())",
        },
        {
            "setup": """import numpy as np

rho_c0 = 1000.0
determinants = np.array([1.0, 0.0], dtype=float)

def run_model():
    try:
        transform_heat_capacity(rho_c0, determinants.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_transform_heat_capacity(
            rho_c0,
            determinants.copy(),
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
