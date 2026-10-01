"""
Sheet density of the two-dimensional electron gas at zero gate bias, solved self-consistently for the
displaced-sheet barrier.

At zero bias the channel holds the charge that the barrier, acting as a capacitor over its whole thickness
d_B, supports against the gate overdrive -V_th, reduced by the Fermi level the electrons themselves occupy.
The free charges, on the metal at the surface and in the channel at the interface, do not move with the
bound sheets, so the displacement enters only through V_th:

    n_s = eps_b ( -V_th - E_F(n_s) ) / ( q d_B ),

with V_th the displaced-sheet threshold voltage and E_F(n_s) the self-consistent Schrodinger-Poisson
Fermi level of the channel in volts, from the Fermi-level step at its default grid. Because E_F grows with
n_s, this is a fixed-point equation with a single root, to be solved numerically. When -V_th does not
exceed the Fermi level of the empty channel, E_F(0), the channel is empty and the density is zero.

Inputs: al_fraction, barrier_thickness, sheet_separation, barrier_height, conduction_band_offset_eV: as in the
  threshold-voltage step effective_mass_ratio: float > 0, m* / m_e for the channel gan, aln: binary parameter dictionaries keyed 'a' (a-axis lattice constant, m), 'psp' (spontaneous polarization, C/m^2, negative), 'e31' and 'e33' (piezoelectric constants, C/m^2), 'c13' and 'c33' (elastic constants, Pa) and 'eps_r' (relative permittivity)

 Returns: float, self-consistent sheet density in m^-2 (0.0 when the channel is empty)

 Raises: ValueError under the conditions of the threshold-voltage and Fermi-level steps (non-positive thickness, invalid separation, non-positive effective mass ratio).

Returns
-------
float, self-consistent sheet density in m^-2 (0.0 when the channel is empty)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def self_consistent_sheet_density(al_fraction: float, barrier_thickness: float, sheet_separation: float,
                                  barrier_height: float, conduction_band_offset_eV: float,
                                  effective_mass_ratio: float, gan: dict, aln: dict) -> float:
    """Zero-bias 2DEG sheet density in m^-2 solved self-consistently with the Schrodinger-Poisson Fermi level.
 
    Returns 0.0 when -V_th does not exceed the empty-channel Fermi level. Raises ValueError for a non-positive
    thickness, an invalid sheet separation or a non-positive effective mass ratio.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _oracle_self_consistent_sheet_density(al_fraction: float, barrier_thickness: float, sheet_separation: float,
                                          barrier_height: float, conduction_band_offset_eV: float,
                                          effective_mass_ratio: float, gan: dict, aln: dict) -> float:
    Q = 1.602176634e-19
    EPS0 = 8.8541878128e-12
    HBAR = 1.054571817e-34
    ME = 9.1093837015e-31
    if not (effective_mass_ratio > 0.0):
        raise ValueError('effective mass ratio must be positive')
    v_th = _oracle_displaced_sheet_threshold_voltage(al_fraction, barrier_thickness, sheet_separation,
                                                     barrier_height, conduction_band_offset_eV, gan, aln)
    e_f_empty = _oracle_channel_fermi_level(0.0, effective_mass_ratio, gan)
    if not (-v_th > e_f_empty):
        return 0.0
    eps_b = (al_fraction * aln['eps_r'] + (1.0 - al_fraction) * gan['eps_r']) * EPS0
    span = barrier_thickness
    scale = eps_b / (Q * span)

    def _residual(n):
        e_f = _oracle_channel_fermi_level(n, effective_mass_ratio, gan)
        return n - scale * (-v_th - e_f)

    upper = scale * (-v_th)          # density with the Fermi level neglected; the root lies below it
    return float(brentq(_residual, 0.0, upper, xtol=1e-2, rtol=1e-13, maxiter=500))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    s = "import numpy as np\nfrom scipy.optimize import brentq\nfrom scipy.linalg import eigh_tridiagonal\nGAN = {'a': 3.189e-10, 'psp': -0.034, 'e31': -0.34, 'e33': 0.67, 'c13': 106.0e9, 'c33': 398.0e9, 'eps_r': 8.9}\nALN = {'a': 3.112e-10, 'psp': -0.090, 'e31': -0.53, 'e33': 1.50, 'c13': 108.0e9, 'c33': 373.0e9, 'eps_r': 8.5}\n"
    return [
        {"setup": s,
         "call": "round(self_consistent_sheet_density(0.31, 9.5e-9, 5.0e-10, 1.32, 0.301, 0.22, GAN, ALN) * 1e-16, 8)",
         "gold_call": "round(_oracle_self_consistent_sheet_density(0.31, 9.5e-9, 5.0e-10, 1.32, 0.301, 0.22, GAN, ALN) * 1e-16, 8)"},
        {"setup": s,
         "call": "round(self_consistent_sheet_density(0.22, 26.0e-9, 6.2e-10, 1.32, 0.301, 0.22, GAN, ALN) * 1e-16, 8)",
         "gold_call": "round(_oracle_self_consistent_sheet_density(0.22, 26.0e-9, 6.2e-10, 1.32, 0.301, 0.22, GAN, ALN) * 1e-16, 8)"},
        {"setup": s,
         "call": "round(self_consistent_sheet_density(0.31, 4.0e-9, 5.0e-10, 1.32, 0.301, 0.22, GAN, ALN) * 1e-16, 8)",
         "gold_call": "round(_oracle_self_consistent_sheet_density(0.31, 4.0e-9, 5.0e-10, 1.32, 0.301, 0.22, GAN, ALN) * 1e-16, 8)"},
        {"setup": s + "def run_model():\n    try:\n        self_consistent_sheet_density(0.31, 9.5e-9, 5.0e-10, 1.32, 0.301, 0.0, GAN, ALN)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle():\n    try:\n        _oracle_self_consistent_sheet_density(0.31, 9.5e-9, 5.0e-10, 1.32, 0.301, 0.0, GAN, ALN)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]
