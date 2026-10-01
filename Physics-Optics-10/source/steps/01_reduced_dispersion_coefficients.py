"""
Compute the two reduced dispersion coefficients that recur throughout the analysis of a uniaxial dielectric medium with isotropic magneto-electric chirality (a gyroelectric metamaterial with constitutive tensors epsilon = diag(4, 4, eps_z), mu = I_3x3, and gamma = gamma0 * I_3x3).

These two combinations, the transverse reduced coefficient and the axial reduced coefficient appear repeatedly in the medium's dispersion relation and in every downstream criticality condition (energy-metric singularity, optical Lifshitz transition, and exceptional-point loci).

Returns
-------
np.ndarray of shape (2,): [A_perp, A_par], the transverse and axial reduced coefficients.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reduced_dispersion_coefficients(eps_z: float, gamma0: float) -> "np.ndarray":
    """Transverse and axial reduced dispersion coefficients.

    Args:
        eps_z (float): axial (out-of-plane) relative permittivity of the uniaxial
            medium. The in-plane permittivity is fixed at 4 for this model.
        gamma0 (float): isotropic chirality parameter, gamma0 >= 0.

    Raises:
        ValueError: if gamma0 < 0.

    Expected return:
        np.ndarray of shape (2,): [A_perp, A_par]. A_perp vanishes at gamma0 = 2;
        A_par vanishes at gamma0 = sqrt(eps_z).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: reduced_dispersion_coefficients
def _oracle_reduced_dispersion_coefficients(eps_z: float, gamma0: float) -> "np.ndarray":
    if gamma0 < 0.0:
        raise ValueError("gamma0 must be non-negative")
    a_perp = 4.0 - gamma0 ** 2
    a_par = eps_z - gamma0 ** 2
    return np.array([a_perp, a_par])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "v = reduced_dispersion_coefficients(9.0, 2.5)",
            "call": "np.round(v, 8)",
            "gold_call": "np.round(_oracle_reduced_dispersion_coefficients(9.0, 2.5), 8)",
            "tol": 1e-6,
        },
        {
            "setup": "v = reduced_dispersion_coefficients(-6.0, 2.5)",
            "call": "np.round(v, 8)",
            "gold_call": "np.round(_oracle_reduced_dispersion_coefficients(-6.0, 2.5), 8)",
            "tol": 1e-6,
        },
        {
            "setup": "v = reduced_dispersion_coefficients(0.0, 0.0)",
            "call": "np.round(v, 8)",
            "gold_call": "np.round(_oracle_reduced_dispersion_coefficients(0.0, 0.0), 8)",
            "tol": 1e-6,
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        reduced_dispersion_coefficients(9.0, -1.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_reduced_dispersion_coefficients(9.0, -1.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
