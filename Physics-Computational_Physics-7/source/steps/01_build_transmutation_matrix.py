"""
Implement `build_transmutation_matrix` which constructs the neutron reaction matrix for a single cell.

Neutron irradiation changes fuel composition through capture and fission. The
reaction rates depend on the neutron flux and microscopic cross sections, while
fission produces a distribution of daughter nuclides. Fuel circulating between
irradiated and external regions therefore experiences different reaction
conditions.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_transmutation_matrix(flux: float,
                               sigma_gamma: "np.ndarray", sigma_f: "np.ndarray",
                               fission_yields: "np.ndarray") -> "np.ndarray":
    '''Constructs the neutron transmutation matrix for a single cell.

    Parameters
    ----------
    flux : float
        Cell-averaged scalar neutron flux [cm⁻² s⁻¹].
    sigma_gamma : np.ndarray
        Shape (m,) radiative capture cross-sections [barn].
        Capture products leave the tracked nuclide set.
    sigma_f : np.ndarray
        Shape (m,) fission cross-sections [barn] (0 if not fissile).
    fission_yields : np.ndarray
        Shape (m, m) independent fission yields. fission_yields[j, i] = yield of i from j.

    Returns
    -------
    T : np.ndarray
        Shape (m, m) reaction-rate matrix acting on the isotope number-density
        vector. Rows identify produced nuclides and columns identify parents.

    Raises
    ------
    ValueError
        If flux is negative.
    '''
    return T

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_transmutation_matrix(flux: float, sigma_gamma: "np.ndarray", sigma_f: "np.ndarray", fission_yields: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    if flux < 0:
        raise ValueError("flux must be non-negative")
    m = len(sigma_gamma)
    T = np.zeros((m, m))

    # Total absorption = capture + fission (convert barns to cm²)
    sigma_a = (sigma_gamma + sigma_f) * 1e-24

    for i in range(m):
        # Diagonal: destruction by neutron absorption
        T[i, i] = -flux * sigma_a[i]

        for j in range(m):
            if j == i:
                continue
            # Fission yield production: flux * y_{j->i} * sigma_f_j (barn -> cm²)
            T[i, j] += flux * fission_yields[j, i] * sigma_f[j] * 1e-24

    return T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    Each test case is a dict with:
        setup:      Python code to set up variables
        call:       Expression calling the model's function
        gold_call:  Expression calling the gold function
    """
    return [
        {
            # Test 1: Core cell (irradiated)
            "tol": 1e-12,
            "setup": """import copy
import numpy as np
flux = 6.0e14
# Isotope order: [U-235, I-135, Xe-135, Cs-135]
sigma_gamma = np.array([98.71, 80.03, 2.778e6, 8.302])  # barn
sigma_f = np.array([585.1, 0.0, 0.0, 0.0])
fission_yields = np.zeros((4, 4))
fission_yields[0, 1] = 0.0628  # U235 -> I135
fission_yields[0, 2] = 0.0016  # U235 -> Xe135
""",
            "call": 'build_transmutation_matrix(*copy.deepcopy((flux, sigma_gamma, sigma_f, fission_yields)))',
            "gold_call": '_oracle_build_transmutation_matrix(*copy.deepcopy((flux, sigma_gamma, sigma_f, fission_yields)))',
        },
        {
            # Test 2: Zero flux (external loop / pump bowl) — should return zeros
            "setup": """import copy
import numpy as np
flux = 0.0
sigma_gamma = np.array([98.71, 80.03, 2.778e6, 8.302])
sigma_f = np.array([585.1, 0.0, 0.0, 0.0])
fission_yields = np.zeros((4, 4))
fission_yields[0, 1] = 0.0628
fission_yields[0, 2] = 0.0016
""",
            "call": 'build_transmutation_matrix(*copy.deepcopy((flux, sigma_gamma, sigma_f, fission_yields)))',
            "gold_call": '_oracle_build_transmutation_matrix(*copy.deepcopy((flux, sigma_gamma, sigma_f, fission_yields)))',
        },
        {
            # Test 3: Single isotope with both capture and fission
            "tol": 1e-12,
            "setup": """import copy
import numpy as np
flux = 1.0e13
sigma_gamma = np.array([10.0])
sigma_f = np.array([500.0])
fission_yields = np.zeros((1, 1))
""",
            "call": 'build_transmutation_matrix(*copy.deepcopy((flux, sigma_gamma, sigma_f, fission_yields)))',
            "gold_call": '_oracle_build_transmutation_matrix(*copy.deepcopy((flux, sigma_gamma, sigma_f, fission_yields)))',
        },
    ]
