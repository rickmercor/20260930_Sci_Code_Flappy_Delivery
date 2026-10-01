"""
End-to-end pipeline: fit the displaced-sheet model to a set of as-grown structures and report the barrier
height it then requires for a separately characterised as-grown sample.

Chain the earlier steps. Recover the sample's sheet density from its transport pair; fit the common sheet
separation to the structure set at the literature barrier height; then find the barrier height at which
the fitted, self-consistent displaced-sheet model reproduces the sample's measured density.

Inputs: sheet_resistance_ohm_per_sq, hall_mobility_cm2_per_Vs: transport pair of the sample al_fraction, barrier_thickness: sample composition and barrier thickness (m) set_al_fractions, set_thicknesses, set_sheet_densities: structure set (1-D arrays; m and m^-2) literature_barrier_height: float, phi_B in V used for the fit conduction_band_offset_eV, effective_mass_ratio: as in the earlier steps gan, aln: binary parameter dictionaries keyed 'a' (a-axis lattice constant, m), 'psp' (spontaneous polarization, C/m^2, negative), 'e31' and 'e33' (piezoelectric constants, C/m^2), 'c13' and 'c33' (elastic constants, Pa) and 'eps_r' (relative permittivity)

 Returns: float, required barrier height of the sample in V

 Raises: ValueError under the conditions documented for the chained steps.

Returns
-------
float, required barrier height of the sample in V
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fitted_model_barrier_height(sheet_resistance_ohm_per_sq: float, hall_mobility_cm2_per_Vs: float,
                              al_fraction: float, barrier_thickness: float,
                              set_al_fractions: "np.ndarray", set_thicknesses: "np.ndarray", set_sheet_densities: "np.ndarray",
                              literature_barrier_height: float, conduction_band_offset_eV: float,
                              effective_mass_ratio: float, gan: dict, aln: dict) -> float:
    """Barrier height in V the fitted displaced-sheet model requires for the as-grown sample.
 
    Raises ValueError under the conditions of the chained steps: a non-positive sheet resistance or mobility,
    a non-positive barrier thickness, an invalid sheet separation, a non-positive effective mass ratio, or
    structure-set arrays that are empty, of unequal length or contain a non-positive density.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq, minimize_scalar
 
def _oracle_fitted_model_barrier_height(sheet_resistance_ohm_per_sq: float, hall_mobility_cm2_per_Vs: float,
                                      al_fraction: float, barrier_thickness: float,
                                       set_al_fractions: "np.ndarray", set_thicknesses: "np.ndarray", set_sheet_densities: "np.ndarray",
                                      literature_barrier_height: float, conduction_band_offset_eV: float,
                                      effective_mass_ratio: float, gan: dict, aln: dict) -> float:
    Q = 1.602176634e-19
    EPS0 = 8.8541878128e-12
    HBAR = 1.054571817e-34
    ME = 9.1093837015e-31
    n_meas = _oracle_sheet_carrier_density(sheet_resistance_ohm_per_sq, hall_mobility_cm2_per_Vs)
    delta = _oracle_fit_sheet_separation(set_al_fractions, set_thicknesses, set_sheet_densities,
                                         literature_barrier_height, conduction_band_offset_eV,
                                         effective_mass_ratio, gan, aln)
    return _oracle_required_barrier_height(al_fraction, barrier_thickness, delta, n_meas,
                                           conduction_band_offset_eV, effective_mass_ratio, gan, aln)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    s = ("import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\nfrom scipy.linalg import eigh_tridiagonal\nGAN = {'a': 3.189e-10, 'psp': -0.034, 'e31': -0.34, 'e33': 0.67, 'c13': 106.0e9, 'c33': 398.0e9, 'eps_r': 8.9}\nALN = {'a': 3.112e-10, 'psp': -0.090, 'e31': -0.53, 'e33': 1.50, 'c13': 108.0e9, 'c33': 373.0e9, 'eps_r': 8.5}\n"
         "X = np.array([0.22, 0.26, 0.30, 0.33, 0.36])\nD = np.array([26.0, 21.0, 17.0, 14.0, 12.0]) * 1e-9\n"
         "N = np.array([8.28, 9.55, 10.6, 11.1, 11.5]) * 1e16\n")
    return [
        {"setup": s,
         "call": "round(fitted_model_barrier_height(690.0, 1500.0, 0.31, 9.5e-9, X, D, N, 1.32, 0.301, 0.22, GAN, ALN), 8)",
         "gold_call": "round(_oracle_fitted_model_barrier_height(690.0, 1500.0, 0.31, 9.5e-9, X, D, N, 1.32, 0.301, 0.22, GAN, ALN), 8)"},
        {"setup": s + "N0 = np.array([_oracle_self_consistent_sheet_density(xi, di, 0.0, 1.32, 0.301, 0.22, GAN, ALN) for xi, di in zip(X, D)])\n",
         "call": "round(fitted_model_barrier_height(690.0, 1500.0, 0.31, 9.5e-9, X, D, N0, 1.32, 0.301, 0.22, GAN, ALN), 8)",
         "gold_call": "round(_oracle_fitted_model_barrier_height(690.0, 1500.0, 0.31, 9.5e-9, X, D, N0, 1.32, 0.301, 0.22, GAN, ALN), 8)"},
        {"setup": s,
         "call": "round(fitted_model_barrier_height(600.0, 1400.0, 0.32, 18.0e-9, X, D, N, 1.32, 0.301, 0.22, GAN, ALN), 8)",
         "gold_call": "round(_oracle_fitted_model_barrier_height(600.0, 1400.0, 0.32, 18.0e-9, X, D, N, 1.32, 0.301, 0.22, GAN, ALN), 8)"},
        {"setup": s + "def run_model():\n    try:\n        fitted_model_barrier_height(690.0, 0.0, 0.31, 9.5e-9, X, D, N, 1.32, 0.301, 0.22, GAN, ALN)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle():\n    try:\n        _oracle_fitted_model_barrier_height(690.0, 0.0, 0.31, 9.5e-9, X, D, N, 1.32, 0.301, 0.22, GAN, ALN)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]
