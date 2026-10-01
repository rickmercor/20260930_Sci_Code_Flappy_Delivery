"""
Residual chemical potential of the reservoir the pore is in contact with.

The reservoir is the same fluid without a wall, so its residual chemical potential must come from the same free-energy expression rather than from a separate correlation, otherwise the pore and the reservoir are not on a common thermodynamic footing. In a uniform fluid the vector measures vanish and each scalar weighted density reduces to the bulk density multiplied by the integral of its weight over the sphere, which turns the functional derivative into an ordinary sum of partial derivatives against those four geometric factors. Return the dimensionless residual chemical potential, that is, the part left after the ideal-gas contribution is removed.

Returns
-------
float, the dimensionless residual chemical potential of the uniform fluid
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bulk_residual_chemical_potential(rho_bulk: float, radius: float) -> float:
    '''Reduced residual chemical potential of the uniform hard-sphere reference.

    Parameters
    ----------
    rho_bulk : float
        Reservoir number density in angstrom^-3.
    radius : float
        Effective hard-sphere radius in angstrom.

    Returns
    -------
    float
        Dimensionless residual chemical potential.

    Raises
    ------
    ValueError
        If rho_bulk is negative or radius is not positive.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_bulk_residual_chemical_potential(rho_bulk: float, radius: float) -> float:
    if rho_bulk < 0 or radius <= 0:
        raise ValueError("rho_bulk must be non-negative and radius positive")
    g0 = 1.0
    g1 = radius
    g2 = 4.0*np.pi*radius**2
    g3 = (4.0/3.0)*np.pi*radius**3
    w = np.array([[rho_bulk*g0], [rho_bulk*g1], [rho_bulk*g2], [rho_bulk*g3], [0.0], [0.0]])
    d0, d1, d2, d3, _, _ = _oracle_white_bear_partials(w)
    return float(d0[0]*g0 + d1[0]*g1 + d2[0]*g2 + d3[0]*g3)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": "bulk_residual_chemical_potential(0.010, 1.5675050365)",
            "gold_call": "_oracle_bulk_residual_chemical_potential(0.010, 1.5675050365)",
        },
        {
            "setup": "import numpy as np",
            "call": "bulk_residual_chemical_potential(0.002, 1.5675050365)",
            "gold_call": "_oracle_bulk_residual_chemical_potential(0.002, 1.5675050365)",
        },
        {
            "setup": "import numpy as np",
            "call": "bulk_residual_chemical_potential(0.020, 1.5675050365)",
            "gold_call": "_oracle_bulk_residual_chemical_potential(0.020, 1.5675050365)",
        },
        {
            "setup": "import numpy as np",
            "call": "bulk_residual_chemical_potential(0.0, 1.5675050365)",
            "gold_call": "_oracle_bulk_residual_chemical_potential(0.0, 1.5675050365)",
        },
        {
            "setup": "import numpy as np",
            "call": "bulk_residual_chemical_potential(0.010, 1.4)",
            "gold_call": "_oracle_bulk_residual_chemical_potential(0.010, 1.4)",
        },
    ]
