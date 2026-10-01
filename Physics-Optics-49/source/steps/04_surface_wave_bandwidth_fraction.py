"""
Compute the surface-wave bandwidth fraction f_SW, the fraction of the ordinary Reststrahlen band that is occupied by the allowed lossless surface-wave window, using the band edge from the previous step.

The allowed window starts at the ordinary TO frequency (where |eps_o| diverges) and ends at the finite-coupling band edge omega_*; the rest of the Reststrahlen band is a surface-wave band gap. Use the paper's own definition of the bandwidth fraction.

Returns
-------
float: f_SW, a dimensionless number in (0, 1).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def surface_wave_bandwidth_fraction(eps_inf_o: float, w_to: float, w_lo: float, eps2: float, alpha: float) -> float:
    """Fraction of the Reststrahlen band occupied by the surface-wave window.

    Args:
        eps_inf_o (float): high-frequency ordinary permittivity, > 0.
        w_to (float): ordinary TO phonon wavenumber (cm^-1).
        w_lo (float): ordinary LO phonon wavenumber (cm^-1), > w_to.
        eps2 (float): TI-side permittivity (bulk or effective), > 0.
        alpha (float): dimensionless axion interface coupling, >= 0.

    Raises:
        ValueError: under the same conditions as the band-edge step.

    Expected return:
        float: f_SW in (0, 1); larger for smaller eps2.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: surface_wave_bandwidth_fraction
def _oracle_surface_wave_bandwidth_fraction(eps_inf_o: float, w_to: float, w_lo: float, eps2: float, alpha: float) -> float:
    w_star = _oracle_finite_coupling_band_edge(eps_inf_o, w_to, w_lo, eps2, alpha)
    return float((w_star - w_to) / (w_lo - w_to))  # Eq. (37)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "v = surface_wave_bandwidth_fraction(4.87, 1360.0, 1614.0, 8.352, 7.2973525693e-3)",
            "call": "round(v, 8)",
            "gold_call": "round(_oracle_surface_wave_bandwidth_fraction(4.87, 1360.0, 1614.0, 8.352, 7.2973525693e-3), 8)",
            "tol": 1e-06,
        },
        {
            "setup": "v = surface_wave_bandwidth_fraction(4.87, 1360.0, 1614.0, 5.755, 7.2973525693e-3)",
            "call": "round(v, 8)",
            "gold_call": "round(_oracle_surface_wave_bandwidth_fraction(4.87, 1360.0, 1614.0, 5.755, 7.2973525693e-3), 8)",
            "tol": 1e-06,
        },
        {
            "setup": "v = surface_wave_bandwidth_fraction(6.6, 797.0, 973.0, 41.0, 7.2973525693e-3)",
            "call": "round(v, 8)",
            "gold_call": "round(_oracle_surface_wave_bandwidth_fraction(6.6, 797.0, 973.0, 41.0, 7.2973525693e-3), 8)",
            "tol": 1e-06,
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        surface_wave_bandwidth_fraction(4.87, 1360.0, 1360.0, 8.0, 0.007)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_surface_wave_bandwidth_fraction(4.87, 1360.0, 1360.0, 8.0, 0.007)\n"
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
