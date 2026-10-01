"""
Compute the upper band-edge wavenumber omega_* of the lossless surface-wave window, the frequency inside the ordinary Reststrahlen band at which the localized HM/TI surface wave ceases to exist, including the finite topological (axion) interface coupling alpha.

The surface wave exists only in part of the Reststrahlen band; the band edge follows from the paper's asymptotic analysis of its dispersion function. The axion coupling shifts this edge by a small amount relative to the purely dielectric criterion. Consult the paper's own band-edge condition and its closed-form solution for omega_* for the exact way alpha enters; do not assume the non-topological criterion.

Returns
-------
float: band-edge wavenumber omega_* in cm^-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def finite_coupling_band_edge(eps_inf_o: float, w_to: float, w_lo: float, eps2: float, alpha: float) -> float:
    """Upper edge of the lossless HM/TI surface-wave window.

    Args:
        eps_inf_o (float): high-frequency ordinary permittivity, > 0.
        w_to (float): ordinary TO phonon wavenumber (cm^-1), > 0.
        w_lo (float): ordinary LO phonon wavenumber (cm^-1), > w_to.
        eps2 (float): TI-side permittivity (bulk or effective), > 0.
        alpha (float): dimensionless axion interface coupling, >= 0.

    Raises:
        ValueError: if w_lo <= w_to, w_to <= 0, eps_inf_o <= 0, eps2 <= 0,
            or alpha < 0.

    Expected return:
        float: omega_* in cm^-1, strictly inside (w_to, w_lo); it decreases
        toward w_to as eps2 grows and moves slightly downward as alpha grows.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: finite_coupling_band_edge
def _oracle_finite_coupling_band_edge(eps_inf_o: float, w_to: float, w_lo: float, eps2: float, alpha: float) -> float:
    if w_to <= 0.0 or w_lo <= w_to or eps_inf_o <= 0.0 or eps2 <= 0.0 or alpha < 0.0:
        raise ValueError("invalid band-edge inputs")
    eps2_alpha = eps2 + alpha ** 2 / 2.0  # Eq. (35)
    return float(np.sqrt((eps_inf_o * w_lo ** 2 + eps2_alpha * w_to ** 2) / (eps_inf_o + eps2_alpha)))  # Eq. (36)/(56)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "v = finite_coupling_band_edge(4.87, 1360.0, 1614.0, 8.352, 7.2973525693e-3)",
            "call": "round(v, 6)",
            "gold_call": "round(_oracle_finite_coupling_band_edge(4.87, 1360.0, 1614.0, 8.352, 7.2973525693e-3), 6)",
            "tol": 1e-09,
        },
        {
            "setup": "v = finite_coupling_band_edge(4.87, 1360.0, 1614.0, 41.0, 7.2973525693e-3)",
            "call": "round(v, 6)",
            "gold_call": "round(_oracle_finite_coupling_band_edge(4.87, 1360.0, 1614.0, 41.0, 7.2973525693e-3), 6)",
            "tol": 1e-06,
        },
        {
            "setup": "v = finite_coupling_band_edge(4.0, 820.0, 972.0, 41.0, 0.05)",
            "call": "round(v, 6)",
            "gold_call": "round(_oracle_finite_coupling_band_edge(4.0, 820.0, 972.0, 41.0, 0.05), 6)",
            "tol": 1e-06,
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        finite_coupling_band_edge(4.87, 1614.0, 1360.0, 8.0, 0.007)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_finite_coupling_band_edge(4.87, 1614.0, 1360.0, 8.0, 0.007)\n"
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
                "        finite_coupling_band_edge(4.87, 1360.0, 1614.0, 8.0, -0.1)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_finite_coupling_band_edge(4.87, 1360.0, 1614.0, 8.0, -0.1)\n"
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
