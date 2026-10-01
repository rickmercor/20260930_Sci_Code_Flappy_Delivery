"""
Implement final_combined_mobility_ratio, the final orchestrator that chains all eight prior sub-problems: computing the weighted MFP average, characteristic length scale, temperature-scaled bulk mobility, diameter-dependent mobility, surface-limited mobility (with direct-form consistency check), the ionized-impurity screening parameter, and the impurity-limited mobility, then combines the diameter-dependent nanowire mobility (which already includes bulk and surface scattering) with the impurity-limited mobility via a second application of Matthiessen's rule, and returns the ratio of this fully combined mobility to the temperature-scaled bulk mobility.

Predicting real device performance requires combining all significant, independent scattering mechanisms. This step assembles bulk phonon scattering, surface scattering, and ionized-impurity scattering into a single physically complete mobility prediction via two successive, independent applications of Matthiessen's rule -- the first isolating the surface contribution from the diameter-dependent relation, the second combining that size-limited result with the separately-computed impurity-limited mobility.

Returns
-------
float, the fully combined mobility ratio
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def final_combined_mobility_ratio(MFP: np.ndarray, weights: np.ndarray, kappa: float, mu_bulk_ref: float, T_ref: float, alpha: float, T: float, d: float, beta: float, C_b: float, C_i: float, N_I: float) -> float:
    '''Compute the fully combined mobility ratio (orchestrator).

    Returns
    -------
    ratio : float
        mu_total / mu_bulk(T), as a native Python float.

    Raises
    ------
    ValueError
        If any earlier step's input validity conditions are violated.
    '''
    return ratio  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_final_combined_mobility_ratio(MFP: np.ndarray, weights: np.ndarray, kappa: float, mu_bulk_ref: float, T_ref: float, alpha: float, T: float, d: float, beta: float, C_b: float, C_i: float, N_I: float) -> float:
    MFP_avg = _oracle_weighted_mfp_average(MFP, weights)
    d0 = _oracle_d0_from_mfp(MFP_avg, kappa)
    mu_bulk_T = _oracle_bulk_mobility_temperature_scaling(mu_bulk_ref, T_ref, alpha, T)
    mu_1D = _oracle_diameter_dependent_mobility(mu_bulk_T, d, d0, beta)
    mu_s = _oracle_surface_limited_mobility(mu_1D, mu_bulk_T)
    mu_s_direct = _oracle_surface_mobility_direct_check(mu_bulk_T, d, d0, beta)
    if abs(mu_s - mu_s_direct) > 1e-6 * max(abs(mu_s), 1.0):
        raise ValueError("consistency check failed: Matthiessen's-rule and direct Eq.6 surface mobilities disagree")
    b = _oracle_ionized_impurity_screening_parameter(C_b, T, N_I)
    mu_i = _oracle_ionized_impurity_mobility(C_i, T, N_I, b)
    mu_total = 1.0 / (1.0 / mu_1D + 1.0 / mu_i)
    return float(mu_total / mu_bulk_T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                "import numpy as np\n"
                "rng_mfp = np.random.default_rng(421)\n"
                "MFP = rng_mfp.uniform(20, 150, 6)\n"
                "rng_w = np.random.default_rng(433)\n"
                "weights = rng_w.uniform(0.5, 2.0, 6)\n"
                "kappa = 0.72\nmu_bulk_ref = 1090.0\nT_ref = 245.0\nalpha = 1.58\nT = 370.0\nd = 190.0\nbeta = 1.44\n"
                "C_b = 4.2e14\nC_i = 6.5e17\nN_I = 3.1e18"
            ),
            "call": "final_combined_mobility_ratio(MFP, weights, kappa, mu_bulk_ref, T_ref, alpha, T, d, beta, C_b, C_i, N_I)",
            "gold_call": "_oracle_final_combined_mobility_ratio(MFP, weights, kappa, mu_bulk_ref, T_ref, alpha, T, d, beta, C_b, C_i, N_I)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "rng_mfp = np.random.default_rng(421)\n"
                "MFP = rng_mfp.uniform(20, 150, 6)\n"
                "rng_w = np.random.default_rng(433)\n"
                "weights = rng_w.uniform(0.5, 2.0, 6)\n"
                "kappa = 0.72\nmu_bulk_ref = 1090.0\nT_ref = 245.0\nalpha = 1.58\nT = 245.0\nd = 190.0\nbeta = 1.44\n"
                "C_b = 4.2e14\nC_i = 6.5e17\nN_I = 3.1e18"
            ),
            "call": "final_combined_mobility_ratio(MFP, weights, kappa, mu_bulk_ref, T_ref, alpha, T, d, beta, C_b, C_i, N_I)",
            "gold_call": "_oracle_final_combined_mobility_ratio(MFP, weights, kappa, mu_bulk_ref, T_ref, alpha, T, d, beta, C_b, C_i, N_I)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "rng_mfp = np.random.default_rng(421)\n"
                "MFP = rng_mfp.uniform(20, 150, 6)\n"
                "rng_w = np.random.default_rng(433)\n"
                "weights = rng_w.uniform(0.5, 2.0, 6)\n"
                "kappa = 0.72\nmu_bulk_ref = 1090.0\nT_ref = 245.0\nalpha = 1.58\nT = 370.0\nd = 380.0\nbeta = 1.44\n"
                "C_b = 4.2e14\nC_i = 6.5e17\nN_I = 3.1e18"
            ),
            "call": "final_combined_mobility_ratio(MFP, weights, kappa, mu_bulk_ref, T_ref, alpha, T, d, beta, C_b, C_i, N_I)",
            "gold_call": "_oracle_final_combined_mobility_ratio(MFP, weights, kappa, mu_bulk_ref, T_ref, alpha, T, d, beta, C_b, C_i, N_I)",
        },
        {
            "setup": (
                "MFP = [1.0, 2.0]\nweights = [0.0, 0.0]\n"
                "kappa = 0.9\nmu_bulk_ref = 1400.0\nT_ref = 300.0\nalpha = 1.5\nT = 450.0\nd = 150.0\nbeta = 1.8\n"
                "C_b = 5.0e14\nC_i = 8.0e17\nN_I = 2.5e18\n"
                "def run_model():\n    try:\n        final_combined_mobility_ratio(MFP, weights, kappa, mu_bulk_ref, T_ref, alpha, T, d, beta, C_b, C_i, N_I)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
                "def run_gold():\n    try:\n        _oracle_final_combined_mobility_ratio(MFP, weights, kappa, mu_bulk_ref, T_ref, alpha, T, d, beta, C_b, C_i, N_I)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2"
            ),
            "call": "run_model()", "gold_call": "run_gold()",
        },
    ]
