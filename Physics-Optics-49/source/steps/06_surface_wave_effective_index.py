"""
Find the real effective index n = beta/k0 of the localized lossless HM/TI surface wave (tangential optical axis, hBN hyperbolic side) at a given wavenumber, by locating the zero of the dispersion mismatch from the previous step on the localized interval.

Localization requires every transverse decay constant to be real and positive, which fixes the admissible interval of n^2. Inside the allowed window the mismatch changes sign exactly once on that interval; outside the window (in the band gap) it does not vanish. Solve to high precision.

Returns
-------
float: effective index n > sqrt(max(eps2, eps_e)).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def surface_wave_effective_index(omega_cm: float, eps2: float, alpha: float) -> float:
    """Effective index of the localized HM/TI surface wave.

    Args:
        omega_cm (float): wavenumber in cm^-1 inside the ordinary Reststrahlen
            band of hBN.
        eps2 (float): TI-side permittivity (bulk or effective), > 0.
        alpha (float): dimensionless axion interface coupling, >= 0.

    Raises:
        ValueError: if eps_o(omega) >= 0, or if no localized solution exists at
            this wavenumber (e.g. omega_cm lies in the surface-wave band gap).

    Expected return:
        float: n, the unique root on the localized interval; n grows without
        bound as omega_cm approaches the band edge from below.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: surface_wave_effective_index
def _oracle_surface_wave_effective_index(omega_cm: float, eps2: float, alpha: float) -> float:
    eps = _oracle_hbn_uniaxial_permittivities(omega_cm)
    eps_o, eps_e = float(eps[0]), float(eps[1])
    if eps_o >= 0.0:
        raise ValueError("not in the type-I hyperbolic (eps_o < 0) regime")
    n2_min = max(eps2, eps_e)  # Eq. (26)

    def g(x):
        return _oracle_axion_dispersion_mismatch(np.sqrt(n2_min + x), omega_cm, eps2, alpha)

    lo, hi = 1e-14, 1e10
    g_lo, g_hi = g(lo), g(hi)
    if not (g_lo < 0.0 < g_hi):
        raise ValueError("no localized surface-wave solution at this wavenumber")
    llo, lhi = np.log(lo), np.log(hi)
    for _ in range(300):
        mid = 0.5 * (llo + lhi)
        if g(np.exp(mid)) < 0.0:
            llo = mid
        else:
            lhi = mid
    return float(np.sqrt(n2_min + np.exp(0.5 * (llo + lhi))))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "v = surface_wave_effective_index(1409.3543485, 8.352, 7.2973525693e-3)",
            "call": "round(v, 7)",
            "gold_call": "round(_oracle_surface_wave_effective_index(1409.3543485, 8.352, 7.2973525693e-3), 7)",
            "tol": 1e-06,
        },
        {
            "setup": "v = surface_wave_effective_index(1374.5862171, 41.0, 7.2973525693e-3)",
            "call": "round(v, 7)",
            "gold_call": "round(_oracle_surface_wave_effective_index(1374.5862171, 41.0, 7.2973525693e-3), 7)",
            "tol": 1e-06,
        },
        {
            "setup": "v = surface_wave_effective_index(1420.9178616, 5.755, 7.2973525693e-3)",
            "call": "round(v, 7)",
            "gold_call": "round(_oracle_surface_wave_effective_index(1420.9178616, 5.755, 7.2973525693e-3), 7)",
            "tol": 1e-06,
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        surface_wave_effective_index(1500.0, 41.0, 0.007)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_surface_wave_effective_index(1500.0, 41.0, 0.007)\n"
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
                "        surface_wave_effective_index(1700.0, 8.0, 0.007)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_surface_wave_effective_index(1700.0, 8.0, 0.007)\n"
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
