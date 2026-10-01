"""
Compute solid-state descriptors for system-dependent SCAN reparameterization.

*system-conditioned SCAN maps per-material descriptors to optimized SCAN*

 *parameters. For a zincblende AB compound the descriptors follow from the*

 *Pauling electronegativity difference and the conventional cubic lattice*

 *constant, covering bond covalency, nearest-neighbor geometry, bond strength,*

 *and atomic number density.*

Returns
-------
np.ndarray, shape (4,) material descriptors [covalency, bond_length, bond_strength, ionic_density]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_material_descriptors(
    electronegativity_a: float,
    electronegativity_b: float,
    lattice_a: float,
) -> "np.ndarray":
    """Compute [covalency, bond_length, bond_strength, ionic_density] for zincblende AB.

    Parameters
    ----------
    electronegativity_a : float
        Pauling electronegativity of species A.
    electronegativity_b : float
        Pauling electronegativity of species B.
    lattice_a : float
        Cubic lattice constant a in angstrom (must be positive).

    Returns
    -------
    descriptors : "np.ndarray"
        Shape (4,) array [covalency, bond_length, bond_strength, ionic_density].
    """
    return descriptors  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_material_descriptors(
    electronegativity_a: float,
    electronegativity_b: float,
    lattice_a: float,
) -> "np.ndarray":
    import numpy as np

    if lattice_a <= 0.0:
        raise ValueError("lattice_a must be positive")
    delta_x = electronegativity_a - electronegativity_b
    covalency = float(np.exp(-0.25 * delta_x**2))
    bond_length = float(lattice_a * np.sqrt(3.0) / 4.0)
    bond_strength = covalency / bond_length
    ionic_density = 8.0 / lattice_a**3
    return np.array(
        [covalency, bond_length, bond_strength, ionic_density], dtype=float
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nelectronegativity_a, electronegativity_b, lattice_a = 1.90, 2.55, 4.35",
            "call": "compute_material_descriptors(electronegativity_a, electronegativity_b, lattice_a)",
            "gold_call": "_oracle_compute_material_descriptors(electronegativity_a, electronegativity_b, lattice_a)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nelectronegativity_a, electronegativity_b, lattice_a = 1.81, 2.18, 5.653",
            "call": "compute_material_descriptors(electronegativity_a, electronegativity_b, lattice_a)",
            "gold_call": "_oracle_compute_material_descriptors(electronegativity_a, electronegativity_b, lattice_a)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nelectronegativity_a, electronegativity_b, lattice_a = 2.55, 2.55, 3.57",
            "call": "compute_material_descriptors(electronegativity_a, electronegativity_b, lattice_a)",
            "gold_call": "_oracle_compute_material_descriptors(electronegativity_a, electronegativity_b, lattice_a)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nelectronegativity_a, electronegativity_b, lattice_a = 0.79, 3.98, 6.2",
            "call": "compute_material_descriptors(electronegativity_a, electronegativity_b, lattice_a)",
            "gold_call": "_oracle_compute_material_descriptors(electronegativity_a, electronegativity_b, lattice_a)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nelectronegativity_a, electronegativity_b, lattice_a = 3.98, 0.79, 6.2",
            "call": "compute_material_descriptors(electronegativity_a, electronegativity_b, lattice_a)",
            "gold_call": "_oracle_compute_material_descriptors(electronegativity_a, electronegativity_b, lattice_a)",
            "tol": 1e-12,
        },
    ]
