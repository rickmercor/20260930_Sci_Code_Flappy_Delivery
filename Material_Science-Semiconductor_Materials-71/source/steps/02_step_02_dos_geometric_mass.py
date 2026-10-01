"""
Reduce directional masses to one density-of-states mass.

Anisotropic carriers are summarized by a single mass that feeds ionization, localization, and mobility estimates when ranking wide-gap oxides.

Returns
-------
float, geometric-mean DOS mass in units of m_e
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dos_geometric_mass(directional_masses: "np.ndarray") -> float:
    """Return the geometric-mean density-of-states mass in units of m_e.

    Raises
    ------
    ValueError
        If directional_masses is not a non-empty 1D array of positive values.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_dos_geometric_mass(directional_masses: "np.ndarray") -> float:
    masses = np.asarray(directional_masses, dtype=float)
    if masses.ndim != 1 or masses.size == 0:
        raise ValueError("directional_masses must be a non-empty 1D array")
    if np.any(masses <= 0):
        raise ValueError("all directional masses must be positive")
    return float(np.prod(masses) ** (1.0 / masses.size))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\ndirectional_masses = np.array([0.22, 0.41, 0.31])\n",
            "call": "dos_geometric_mass(directional_masses)",
            "gold_call": "_oracle_dos_geometric_mass(directional_masses)",
        },
        {
            "setup": "import numpy as np\ndirectional_masses = np.array([1.0, 1.0, 1.0])\n",
            "call": "dos_geometric_mass(directional_masses)",
            "gold_call": "_oracle_dos_geometric_mass(directional_masses)",
        },
        {
            "setup": "import numpy as np\ndirectional_masses = np.array([0.2, 5.0])\n",
            "call": "dos_geometric_mass(directional_masses)",
            "gold_call": "_oracle_dos_geometric_mass(directional_masses)",
        },
    ]
