"""
Pinch-off threshold voltage of a gated AlGaN/GaN barrier when the two bound polarization sheets at the
heterointerface are displaced by a separation delta, symmetrically about the interface.

With the channel empty, walking the conduction band from the gate metal to the neutral buffer gives the
threshold voltage of the displaced-sheet model,

    V_th = phi_B - dE_C + |P_sp,GaN| (delta / 2) (1 / eps_b + 1 / eps_GaN) - sigma (d_B - delta) / eps_b,

with all terms in volts. phi_B is the metal barrier height, dE_C the conduction band offset in eV (it
enters directly in volts), |P_sp,GaN| the spontaneous polarization magnitude of the relaxed GaN buffer,
sigma the net bound interface charge (total polarization of the strained barrier minus |P_sp,GaN|), d_B the
barrier thickness, eps_b the absolute permittivity of the barrier (relative permittivity linear in Al
fraction) and eps_GaN that of the buffer. The third term is the potential step across the thin slab
between the two displaced sheets; the fourth is the field of the net bound charge over the reduced span
d_B - delta. Setting delta to zero recovers the coincident-sheet result.

Inputs: al_fraction: float in [0, 1], Al fraction of the barrier barrier_thickness: float > 0, d_B in m sheet_separation: float >= 0 and < barrier_thickness, delta in m barrier_height: float, phi_B in V conduction_band_offset_eV: float, dE_C in eV gan, aln: binary parameter dictionaries keyed 'a' (a-axis lattice constant, m), 'psp' (spontaneous polarization, C/m^2, negative), 'e31' and 'e33' (piezoelectric constants, C/m^2), 'c13' and 'c33' (elastic constants, Pa) and 'eps_r' (relative permittivity)

 Returns: float, threshold voltage in V (negative for a normally-on structure)

 Raises: ValueError if barrier_thickness is not positive, sheet_separation is negative or not smaller than the barrier thickness, or al_fraction is outside [0, 1].

Returns
-------
float, threshold voltage in V (negative for a normally-on structure)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def displaced_sheet_threshold_voltage(al_fraction: float, barrier_thickness: float, sheet_separation: float,
                                      barrier_height: float, conduction_band_offset_eV: float,
                                      gan: dict, aln: dict) -> float:
    """Displaced-sheet pinch-off threshold voltage in V for the given barrier and geometry.
 
    Raises ValueError if barrier_thickness <= 0, sheet_separation < 0 or >= barrier_thickness, or al_fraction
    is outside [0, 1].
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_displaced_sheet_threshold_voltage(al_fraction: float, barrier_thickness: float, sheet_separation: float,
                                              barrier_height: float, conduction_band_offset_eV: float,
                                              gan: dict, aln: dict) -> float:
    Q = 1.602176634e-19
    EPS0 = 8.8541878128e-12
    HBAR = 1.054571817e-34
    ME = 9.1093837015e-31
    if not (barrier_thickness > 0.0):
        raise ValueError('barrier thickness must be positive')
    if not (0.0 <= sheet_separation < barrier_thickness):
        raise ValueError('sheet separation must be non-negative and smaller than the barrier thickness')
    p_barrier = _oracle_pseudomorphic_layer_polarization(al_fraction, gan, aln)
    p_buffer = _oracle_pseudomorphic_layer_polarization(0.0, gan, aln)
    sigma = p_barrier - p_buffer
    eps_b = (al_fraction * aln['eps_r'] + (1.0 - al_fraction) * gan['eps_r']) * EPS0
    eps_gan = gan['eps_r'] * EPS0
    slab = p_buffer * (sheet_separation / 2.0) * (1.0 / eps_b + 1.0 / eps_gan)
    field = sigma * (barrier_thickness - sheet_separation) / eps_b
    return barrier_height - conduction_band_offset_eV + slab - field

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    s = "import numpy as np\nGAN = {'a': 3.189e-10, 'psp': -0.034, 'e31': -0.34, 'e33': 0.67, 'c13': 106.0e9, 'c33': 398.0e9, 'eps_r': 8.9}\nALN = {'a': 3.112e-10, 'psp': -0.090, 'e31': -0.53, 'e33': 1.50, 'c13': 108.0e9, 'c33': 373.0e9, 'eps_r': 8.5}\n"
    return [
        {"setup": s,
         "call": "round(displaced_sheet_threshold_voltage(0.31, 9.5e-9, 5.0e-10, 1.32, 0.301, GAN, ALN), 9)",
         "gold_call": "round(_oracle_displaced_sheet_threshold_voltage(0.31, 9.5e-9, 5.0e-10, 1.32, 0.301, GAN, ALN), 9)"},
        {"setup": s,
         "call": "round(displaced_sheet_threshold_voltage(0.31, 9.5e-9, 0.0, 1.32, 0.301, GAN, ALN), 9)",
         "gold_call": "round(_oracle_displaced_sheet_threshold_voltage(0.31, 9.5e-9, 0.0, 1.32, 0.301, GAN, ALN), 9)"},
        {"setup": s,
         "call": "round(displaced_sheet_threshold_voltage(0.36, 12.0e-9, 6.2e-10, 1.0, 0.301, GAN, ALN), 9)",
         "gold_call": "round(_oracle_displaced_sheet_threshold_voltage(0.36, 12.0e-9, 6.2e-10, 1.0, 0.301, GAN, ALN), 9)"},
        {"setup": s + "def run_model():\n    try:\n        displaced_sheet_threshold_voltage(0.31, 9.5e-9, 20.0e-9, 1.32, 0.301, GAN, ALN)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle():\n    try:\n        _oracle_displaced_sheet_threshold_voltage(0.31, 9.5e-9, 20.0e-9, 1.32, 0.301, GAN, ALN)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]
