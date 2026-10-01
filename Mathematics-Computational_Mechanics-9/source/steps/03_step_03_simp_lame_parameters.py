"""
Interpolate the Lame parameters carried by each material point from its

pseudo-density using the solid isotropic material with penalization rule, and

return them as one array with a column for lambda and a column for mu.

Density-based topology optimization relaxes the discrete solid or void choice

into a continuous pseudo-density gamma in (0, 1], and recovers a nearly discrete

design by making intermediate densities structurally inefficient. In the

material point setting the pseudo-density is carried by the particle rather than

by a mesh element, so it is transported with the material through the

deformation. The penalization acts on the constitutive constants of the solid

phase, whose Lame parameters follow from the Young modulus E_0 and the Poisson

ratio nu of the fully solid material through the standard isotropic relations

lambda_0 = E_0 nu / ((1 + nu) (1 - 2 nu)) and mu_0 = E_0 / (2 (1 + nu)); the

plane-stress condition is imposed later in the constitutive update rather than

by modifying these constants. Raising the penalization exponent q above one

drives intermediate densities towards the void, and because the interpolation is

a single power of gamma applied to the solid constants, the stress at a material

point is proportional to gamma^q at fixed strain, which is what makes the design

derivative of the internal force a simple rescaling of the internal force

itself.

Returns
-------
np.ndarray of shape (n_points, 2), the interpolated Lame parameters as a float64 array
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def simp_lame_parameters(densities: np.ndarray, youngs_modulus: float,
                         poisson_ratio: float, penalty: float) -> np.ndarray:
    """Interpolate the per-point Lame parameters from the pseudo-densities.

    Parameters
    ----------
    densities : np.ndarray
        Pseudo-densities gamma_p in (0, 1], shape (n_points,).
    youngs_modulus : float
        Young modulus E_0 of the solid phase.
    poisson_ratio : float
        Poisson ratio nu of the solid phase.
    penalty : float
        SIMP exponent q.

    Returns
    -------
    lame : np.ndarray
        Interpolated Lame parameters, shape (n_points, 2), with lambda_p in
        column 0 and mu_p in column 1.

    Raises
    ------
    ValueError
        If `densities` is not a non-empty one-dimensional array, if any density
        lies outside (0, 1], if `youngs_modulus` is not positive, if
        `poisson_ratio` lies outside (-1, 0.5), or if `penalty` is less than 1.
    """
    return lame

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_simp_lame_parameters(densities: np.ndarray, youngs_modulus: float,
                                 poisson_ratio: float, penalty: float) -> np.ndarray:
    """Reference implementation."""
    densities = np.asarray(densities, dtype=float)
    modulus = float(youngs_modulus)
    ratio = float(poisson_ratio)
    exponent = float(penalty)
    if densities.ndim != 1 or densities.size == 0:
        raise ValueError("densities must be a non-empty one-dimensional array")
    if np.any(densities <= 0.0) or np.any(densities > 1.0):
        raise ValueError("densities must lie in (0, 1]")
    if modulus <= 0.0:
        raise ValueError("youngs_modulus must be > 0")
    if not -1.0 < ratio < 0.5:
        raise ValueError("poisson_ratio must lie in (-1, 0.5)")
    if exponent < 1.0:
        raise ValueError("penalty must be >= 1")
    lambda_solid = modulus * ratio / ((1.0 + ratio) * (1.0 - 2.0 * ratio))
    mu_solid = modulus / (2.0 * (1.0 + ratio))
    scale = densities ** exponent
    return np.stack([scale * lambda_solid, scale * mu_solid], axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

_DENSITIES = np.array([0.3, 0.5, 0.7])


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: the three density levels of the design field ---
        {
            "setup": """import numpy as np
densities = _DENSITIES.copy()
""",
            "call": "np.round(simp_lame_parameters(densities, 1.0, 0.3, 3.0), 10).tolist()",
            "gold_call": "np.round(_oracle_simp_lame_parameters(densities, 1.0, 0.3, 3.0), 10).tolist()",
        },
        # --- Boundary case: fully solid material and no penalization ---
        {
            "setup": """import numpy as np
densities = np.array([1.0])
""",
            "call": "np.round(simp_lame_parameters(densities, 12.0, 0.2, 1.0), 10).tolist()",
            "gold_call": "np.round(_oracle_simp_lame_parameters(densities, 12.0, 0.2, 1.0), 10).tolist()",
        },
        # --- Edge case: a vanishing density is outside the admissible range ---
        {
            "setup": """import numpy as np
densities = np.array([0.5, 0.0])
def run_model():
    try:
        simp_lame_parameters(densities, 1.0, 0.3, 3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_simp_lame_parameters(densities, 1.0, 0.3, 3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
