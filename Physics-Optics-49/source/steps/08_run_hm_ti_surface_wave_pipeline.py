"""
Chain the earlier steps for an hBN / thin-film Bi2Se3-on-substrate interface (tangential optical axis, axion coupling alpha = alpha_fs for a jump Delta_theta = pi): build the effective TI-side permittivity, locate the finite-coupling band edge, choose the operating wavenumber at a given fractional position inside the allowed window, solve for the effective index, and return the total surface-wave thickness. The reference implementation calls the earlier public functions by name.

The total surface-wave thickness is dominated by the largest penetration depths and is estimated in the paper by the sum of the TI-side and TE-like depths. The operating wavenumber is omega_op = omega_TO + x (omega_* - omega_TO) for a fractional window position 0 < x < 1.

Returns
-------
float: total surface-wave thickness delta_total in nm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_hm_ti_surface_wave_pipeline(d_nm: float, eps_sub: float, eps_ti: float, window_fraction: float) -> float:
    """Total thickness of the hBN / thin-film-TI surface wave.

    Args:
        d_nm (float): TI film thickness in nm (see the thin-film step).
        eps_sub (float): substrate permittivity, > 0.
        eps_ti (float): TI permittivity, > 0.
        window_fraction (float): fractional position x of the operating
            wavenumber inside the allowed window, 0 < x < 1.

    Raises:
        ValueError: if window_fraction is not strictly between 0 and 1, or if
            any earlier step rejects its inputs.

    Expected return:
        float: delta_total in nm (positive).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: run_hm_ti_surface_wave_pipeline
def _oracle_run_hm_ti_surface_wave_pipeline(d_nm: float, eps_sub: float, eps_ti: float, window_fraction: float) -> float:
    if not (0.0 < window_fraction < 1.0):
        raise ValueError("window_fraction must lie strictly between 0 and 1")
    alpha_fs, delta_theta = 7.2973525693e-3, np.pi
    alpha = alpha_fs * delta_theta / np.pi   # Eq. (1)
    eps_inf_o, w_to, w_lo = 4.87, 1360.0, 1614.0
    eps2 = _oracle_thin_film_effective_permittivity(d_nm, eps_ti, eps_sub)
    bandwidth = _oracle_surface_wave_bandwidth_fraction(eps_inf_o, w_to, w_lo, eps2, alpha)
    w_op = w_to + window_fraction * bandwidth * (w_lo - w_to)
    n = _oracle_surface_wave_effective_index(w_op, eps2, alpha)
    depths = _oracle_surface_wave_penetration_depths(n, w_op, eps2)
    return float(depths[0] + depths[1])  # Eq. (53)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "v = run_hm_ti_surface_wave_pipeline(12.0, 3.9, 41.0, 0.5)",
            "call": "round(v, 5)",
            "gold_call": "round(_oracle_run_hm_ti_surface_wave_pipeline(12.0, 3.9, 41.0, 0.5), 5)",
            "tol": 0.0001,
        },
        {
            "setup": "v = run_hm_ti_surface_wave_pipeline(5.0, 3.9, 41.0, 0.5)",
            "call": "round(v, 5)",
            "gold_call": "round(_oracle_run_hm_ti_surface_wave_pipeline(5.0, 3.9, 41.0, 0.5), 5)",
            "tol": 0.0001,
        },
        {
            "setup": "v = run_hm_ti_surface_wave_pipeline(20.0, 6.7, 41.0, 0.3)",
            "call": "round(v, 5)",
            "gold_call": "round(_oracle_run_hm_ti_surface_wave_pipeline(20.0, 6.7, 41.0, 0.3), 5)",
            "tol": 0.0001,
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        run_hm_ti_surface_wave_pipeline(12.0, 3.9, 41.0, 1.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_run_hm_ti_surface_wave_pipeline(12.0, 3.9, 41.0, 1.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        run_hm_ti_surface_wave_pipeline(120.0, 3.9, 41.0, 0.5)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_run_hm_ti_surface_wave_pipeline(120.0, 3.9, 41.0, 0.5)\n"
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
