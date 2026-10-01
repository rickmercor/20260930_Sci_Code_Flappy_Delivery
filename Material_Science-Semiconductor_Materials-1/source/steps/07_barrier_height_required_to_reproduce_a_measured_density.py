"""
Metal barrier height that the displaced-sheet model, with a given sheet separation, requires in order to
reproduce a measured zero-bias sheet density.

The self-consistent sheet density falls monotonically as the barrier height rises, so the barrier height
that reproduces a measured density is the single root of n_model(phi_B) = n_meas. Search over
0.1 V <= phi_B <= 5.0 V and return the root converged to at least 1e-12 V.

Inputs: al_fraction, barrier_thickness, sheet_separation, conduction_band_offset_eV, effective_mass_ratio:
  as in the self-consistent density step gan, aln: binary parameter dictionaries keyed 'a' (a-axis lattice constant, m), 'psp' (spontaneous polarization, C/m^2, negative), 'e31' and 'e33' (piezoelectric constants, C/m^2), 'c13' and 'c33' (elastic constants, Pa) and 'eps_r' (relative permittivity) target_sheet_density: float > 0, measured n_s in m^-2

 Returns: float, required barrier height phi_B in V

 Raises: ValueError if target_sheet_density is not positive, or if no barrier height in [0.1, 5.0] V reproduces it.

Returns
-------
float, required barrier height phi_B in V
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def required_barrier_height(al_fraction: float, barrier_thickness: float, sheet_separation: float,
                            target_sheet_density: float, conduction_band_offset_eV: float,
                            effective_mass_ratio: float, gan: dict, aln: dict) -> float:
    """Barrier height in V at which the self-consistent displaced-sheet model reproduces the target density.
 
    Raises ValueError if target_sheet_density <= 0 or no root lies in [0.1, 5.0] V.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _oracle_required_barrier_height(al_fraction: float, barrier_thickness: float, sheet_separation: float,
                                    target_sheet_density: float, conduction_band_offset_eV: float,
                                    effective_mass_ratio: float, gan: dict, aln: dict) -> float:
    Q = 1.602176634e-19
    EPS0 = 8.8541878128e-12
    HBAR = 1.054571817e-34
    ME = 9.1093837015e-31
    if not (target_sheet_density > 0.0):
        raise ValueError('target sheet density must be positive')

    def _gap(phi):
        return _oracle_self_consistent_sheet_density(al_fraction, barrier_thickness, sheet_separation, phi,
                                                     conduction_band_offset_eV, effective_mass_ratio,
                                                     gan, aln) - target_sheet_density

    lo, hi = 0.1, 5.0
    if _gap(lo) < 0.0 or _gap(hi) > 0.0:
        raise ValueError('no barrier height in [0.1, 5.0] V reproduces the target density')
    return float(brentq(_gap, lo, hi, xtol=1e-12, rtol=1e-15, maxiter=500))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    s = "import numpy as np\nfrom scipy.optimize import brentq\nfrom scipy.linalg import eigh_tridiagonal\nGAN = {'a': 3.189e-10, 'psp': -0.034, 'e31': -0.34, 'e33': 0.67, 'c13': 106.0e9, 'c33': 398.0e9, 'eps_r': 8.9}\nALN = {'a': 3.112e-10, 'psp': -0.090, 'e31': -0.53, 'e33': 1.50, 'c13': 108.0e9, 'c33': 373.0e9, 'eps_r': 8.5}\n"
    return [
        {"setup": s + "NT = _oracle_sheet_carrier_density(690.0, 1500.0)\n",
         "call": "round(required_barrier_height(0.31, 9.5e-9, 5.0e-10, NT, 0.301, 0.22, GAN, ALN), 8)",
         "gold_call": "round(_oracle_required_barrier_height(0.31, 9.5e-9, 5.0e-10, NT, 0.301, 0.22, GAN, ALN), 8)"},
        {"setup": s + "NT = _oracle_sheet_carrier_density(690.0, 1500.0)\n",
         "call": "round(required_barrier_height(0.31, 9.5e-9, 0.0, NT, 0.301, 0.22, GAN, ALN), 8)",
         "gold_call": "round(_oracle_required_barrier_height(0.31, 9.5e-9, 0.0, NT, 0.301, 0.22, GAN, ALN), 8)"},
        {"setup": s + "NT = _oracle_self_consistent_sheet_density(0.36, 12.0e-9, 6.2e-10, 2.0, 0.301, 0.22, GAN, ALN)\n",
         "call": "round(required_barrier_height(0.36, 12.0e-9, 6.2e-10, NT, 0.301, 0.22, GAN, ALN), 8)",
         "gold_call": "round(_oracle_required_barrier_height(0.36, 12.0e-9, 6.2e-10, NT, 0.301, 0.22, GAN, ALN), 8)"},
        {"setup": s + "def run_model():\n    try:\n        required_barrier_height(0.31, 9.5e-9, 5.0e-10, 0.0, 0.301, 0.22, GAN, ALN)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle():\n    try:\n        _oracle_required_barrier_height(0.31, 9.5e-9, 5.0e-10, 0.0, 0.301, 0.22, GAN, ALN)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]
