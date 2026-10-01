"""
Evaluate the three solvation free-energy functionals that the source paper

uses as charged-state MILC descriptors, from supplied 3D-RISM total and

direct correlation fields.

Do not solve a new 3D-RISM or closure equation. Each conformer is an

independent evaluation. Recover the integrands and the spatial integrals

from the paper.

Returns
-------
return solvation_terms
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_solvation_terms(
    h: np.ndarray,
    c: np.ndarray,
    grid_spacing: float,
    rho,
    temperature: float = 298.15,
    k_b: float = 0.00198720425864,
) -> np.ndarray:
    """
    Evaluate the paper's three solvation free-energy functionals from
    supplied 3D-RISM correlation fields.

    Parameters
    ----------
    h : np.ndarray
        Total correlation fields.
    c : np.ndarray
        Direct correlation fields with the same shape as h.
    grid_spacing : float
        Spatial grid spacing in Angstrom.
    rho : float or np.ndarray
        Solvent-site number density in Angstrom^-3.
    temperature : float, optional
        Temperature in Kelvin.
    k_b : float, optional
        Boltzmann constant in kcal/(mol·K).

    Returns
    -------
    np.ndarray
        Per-conformer values of the paper's three functionals, shape
        (n_conformers, 3), in kcal/mol.

    Raises
    ------
    ValueError
        If h and c have mismatched shapes, if either array is not
        five-dimensional, or if numerical values are invalid.
    """
    return solvation_terms

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_solvation_terms(
    h,
    c,
    grid_spacing,
    rho,
    temperature=298.15,
    k_b=0.00198720425864,
):
    h = np.asarray(h, dtype=float)
    c = np.asarray(c, dtype=float)
    rho = np.asarray(rho, dtype=float)

    if h.shape != c.shape:
        raise ValueError("h and c must have identical shapes")

    if h.ndim != 5:
        raise ValueError("h and c must be five-dimensional")

    if not np.all(np.isfinite(h)):
        raise ValueError("h contains non-finite values")

    if not np.all(np.isfinite(c)):
        raise ValueError("c contains non-finite values")

    if not np.isfinite(grid_spacing) or grid_spacing <= 0:
        raise ValueError("grid_spacing must be positive and finite")

    if rho.ndim == 0:
        rho = np.full(h.shape[1], float(rho), dtype=float)
    elif rho.ndim != 1 or rho.shape[0] != h.shape[1]:
        raise ValueError(
            "rho must be a scalar or a one-dimensional array "
            "with one density per solvent site"
        )

    if not np.all(np.isfinite(rho)) or np.any(rho <= 0):
        raise ValueError("rho must be positive and finite")

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")

    if not np.isfinite(k_b) or k_b <= 0:
        raise ValueError("k_b must be positive and finite")

    kBT = k_b * temperature
    dV = grid_spacing ** 3
    rho_b = rho.reshape((1, -1, 1, 1, 1))
    axes = (1, 2, 3, 4)

    hnc = 0.5 * h**2 - c - 0.5 * c * h
    kh = 0.5 * h**2 * (h < 0) - c - 0.5 * c * h
    gf = -c - 0.5 * c * h

    sc_hnc = kBT * np.sum(rho_b * hnc, axis=axes) * dV
    sc_kh = kBT * np.sum(rho_b * kh, axis=axes) * dV
    gf_value = kBT * np.sum(rho_b * gf, axis=axes) * dV

    return np.column_stack([sc_hnc, sc_kh, gf_value])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """
    Return differential test specifications heavily hardened with scalar-type 
    polymorphism and dynamic spatial variation to stop code shortcuts.
    """
    return [
        {
            "setup": """import numpy as np

# Test case 1: General multi-conformer calculation with site densities
rng = np.random.default_rng(101)

h = rng.normal(0.0, 0.4, size=(2, 2, 3, 3, 3))
c = rng.normal(0.0, 0.3, size=(2, 2, 3, 3, 3))

grid_spacing = 0.5
rho = np.array([0.033327, 0.066654])
temperature = 298.15
k_b = 0.00198720425864

def run_model():
    return compute_solvation_terms(
        h, c, grid_spacing, rho, temperature, k_b
    )

def run_gold():
    return _oracle_compute_solvation_terms(
        h, c, grid_spacing, rho, temperature, k_b
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Test case 2: Negative h exercises KH Heaviside behavior with fluctuations
rng = np.random.default_rng(202)
h = rng.uniform(-1.5, -0.5, size=(1, 1, 3, 3, 3))
c = rng.uniform(0.1, 0.6, size=(1, 1, 3, 3, 3))

def run_model():
    return compute_solvation_terms(h, c, 1.0, 0.0333, 298.15)

def run_gold():
    return _oracle_compute_solvation_terms(
        h, c, 1.0, 0.0333, 298.15
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Test case 3: Positive h ensures KH quadratic equals HNC with spatial variation
rng = np.random.default_rng(303)
h = rng.uniform(0.5, 1.5, size=(1, 1, 3, 3, 3))
c = rng.uniform(0.1, 0.6, size=(1, 1, 3, 3, 3))

def run_model():
    return compute_solvation_terms(h, c, 1.0, 0.0333, 298.15)

def run_gold():
    return _oracle_compute_solvation_terms(
        h, c, 1.0, 0.0333, 298.15
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Test case 4: Non-cubic asymmetric dimensions tracking
# Breaks hardcoded spatial axis size assumptions using variable lengths (2, 2, 2, 4, 3)
rng = np.random.default_rng(404)
h = rng.normal(0.1, 0.2, size=(2, 2, 2, 4, 3))
c = rng.normal(-0.1, 0.1, size=(2, 2, 2, 4, 3))
rho = np.array([0.033327, 0.066654])

def run_model():
    return compute_solvation_terms(
        h, c, 1.35, rho, temperature=310.15
    )

def run_gold():
    return _oracle_compute_solvation_terms(
        h, c, 1.35, rho, temperature=310.15
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Test case 5: Polymorphic Scalar Density Trap (Tests the float contract constraint)
# Punishes models that shortcut broadcasting with rho[:, None, None, None] without type checking
rng = np.random.default_rng(505)
h = rng.uniform(-1.0, 1.0, size=(2, 2, 3, 3, 3))
c = rng.uniform(-0.5, -0.1, size=(2, 2, 3, 3, 3))
rho_scalar = 0.033327  # Bare primitive float instead of an array object instance

def run_model():
    return compute_solvation_terms(
        h, c, 1.10, rho_scalar, temperature=298.15
    )

def run_gold():
    return _oracle_compute_solvation_terms(
        h, c, 1.10, rho_scalar, temperature=298.15
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Test case 6: Severe site density array skew checking with fluctuating entries
rng = np.random.default_rng(606)
h = rng.normal(-0.2, 0.05, size=(2, 2, 3, 3, 3))
c = rng.normal(0.1, 0.05, size=(2, 2, 3, 3, 3))
rho = np.array([0.0005, 0.9995])

def run_model():
    return compute_solvation_terms(
        h, c, 0.85, rho, temperature=280.0
    )

def run_gold():
    return _oracle_compute_solvation_terms(
        h, c, 0.85, rho, temperature=280.0
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Test case 7: Dual Scalar Density Check on asymmetrical grids
rng = np.random.default_rng(707)
h = rng.uniform(-0.5, 0.5, size=(1, 3, 2, 4, 2))
c = rng.uniform(0.1, 0.4, size=(1, 3, 2, 4, 2))
rho_scalar = 0.066654

def run_model():
    return compute_solvation_terms(h, c, 1.0, rho_scalar, 298.15)

def run_gold():
    return _oracle_compute_solvation_terms(h, c, 1.0, rho_scalar, 298.15)
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Test case 8: Invalid shape contracts exception tracking
h = np.ones((1, 1, 2, 2, 2))
c = np.ones((1, 1, 2, 2, 3))

def run_model():
    try:
        compute_solvation_terms(h, c, 1.0, 0.0333)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_compute_solvation_terms(h, c, 1.0, 0.0333)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
    ]
