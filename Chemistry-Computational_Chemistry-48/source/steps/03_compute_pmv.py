"""
Calculate the molecular partial molar volume (PMV) indicator from the supplied direct correlation field distribution integrals.

1. **Integral Properties:** The spatial integration must reduce the multi-dimensional tracking arrays using the supplied interaction site densities and the three-dimensional grid volume elements.

2. **Compressibility Prefactor Scaling:** Incorporate the supplied isothermal compressibility multiplier to properly adjust the integrated spatial properties. The total correlation function array is not used as an input parameter during this stage.

Returns
-------
return pmv
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_pmv(
    c: np.ndarray,
    grid_spacing: float,
    rho,
    chi_kt: float,
) -> np.ndarray:
    """
    Compute the partial molar volume from the direct correlation field.

    Parameters
    ----------
    c : np.ndarray
        Direct correlation fields with shape
        (n_conformers, n_sites, nx, ny, nz).
    grid_spacing : float
        Cubic spatial grid spacing in Angstrom.
    rho : float or np.ndarray
        Solvent-site number density in Angstrom^-3. A scalar is applied
        to every site; a 1-D array must have one entry per solvent site.
    chi_kt : float
        Prefactor k_B T κ_T in Angstrom^3.

    Returns
    -------
    np.ndarray
        Partial molar volume for each conformer, shape (n_conformers,).

    Raises
    ------
    ValueError
        If c is not five-dimensional, or if numerical values are invalid.
    """
    return pmv

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_pmv(
    c,
    grid_spacing,
    rho,
    chi_kt,
):
    c = np.asarray(c, dtype=float)
    rho = np.asarray(rho, dtype=float)

    if c.ndim != 5:
        raise ValueError("c must be five-dimensional")

    if not np.all(np.isfinite(c)):
        raise ValueError("c contains non-finite values")

    if not np.isfinite(grid_spacing) or grid_spacing <= 0:
        raise ValueError("grid_spacing must be positive and finite")

    if not np.isfinite(chi_kt):
        raise ValueError("chi_kt must be finite")

    if rho.ndim == 0:
        rho = np.full(c.shape[1], float(rho), dtype=float)
    elif rho.ndim != 1 or rho.shape[0] != c.shape[1]:
        raise ValueError(
            "rho must be a scalar or a one-dimensional array "
            "with one density per solvent site"
        )

    if not np.all(np.isfinite(rho)) or np.any(rho <= 0):
        raise ValueError("rho must be positive and finite")

    dV = grid_spacing ** 3
    rho_b = rho.reshape((1, -1, 1, 1, 1))
    integral = np.sum(rho_b * c, axis=(1, 2, 3, 4)) * dV
    return chi_kt * (1.0 - integral)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np

# n_conf, n_sites, nx, ny, nz all distinct so rho cannot ride a wrong axis
rng = np.random.default_rng(17)
c = rng.normal(0.0, 0.2, size=(3, 2, 4, 5, 6))
rho = np.array([0.033327, 0.066654])

def run_model():
    return compute_pmv(c, 1.25, rho, 0.016387412)

def run_gold():
    return _oracle_compute_pmv(c, 1.25, rho, 0.016387412)
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Test case 2: Uniform field, non-unit spacing
c = np.full((1, 1, 2, 2, 2), 0.1)

def run_model():
    return compute_pmv(c, 2.0, 0.0333, 10.0)

def run_gold():
    return _oracle_compute_pmv(c, 2.0, 0.0333, 10.0)
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Test case 3: Zero direct correlation recovers the chi_kt prefactor
c = np.zeros((3, 2, 2, 2, 2))
rho = np.array([0.033327, 0.066654])

def run_model():
    return compute_pmv(c, 1.0, rho, 0.016387412)

def run_gold():
    return _oracle_compute_pmv(c, 1.0, rho, 0.016387412)
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Test case 4: Invalid dimensionality
c = np.ones((2, 2, 3, 3))

def run_model():
    try:
        compute_pmv(c, 1.0, 0.0333, 16.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_compute_pmv(c, 1.0, 0.0333, 16.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# n_conf == n_sites, but only site 1 is nonzero
c = np.zeros((2, 2, 2, 2, 2))
c[0, 1] = 0.4
rho = np.array([0.01, 10.0])

def run_model():
    return compute_pmv(c, 1.0, rho, 0.016387412)

def run_gold():
    return _oracle_compute_pmv(c, 1.0, rho, 0.016387412)
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Last spatial length equals n_sites: rho weights axis 1, not axis -1
c = np.zeros((4, 2, 1, 1, 2))
c[:, 0, :, :, :] = 0.10
c[:, 1, :, :, :] = 0.80
rho = np.array([0.01, 1.0])

def run_model():
    return compute_pmv(c, 1.25, rho, 0.016387412)

def run_gold():
    return _oracle_compute_pmv(c, 1.25, rho, 0.016387412)
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
    ]
