"""
Evaluate the mismatch of the lossless HM/TI surface-wave dispersion relation for the tangential optical-axis configuration (optical axis in the interface plane, perpendicular to the propagation direction) at a trial real effective index n = beta/k0, for hBN on the hyperbolic side.

The axion term at the interface mixes TE and TM polarizations, so the dispersion relation couples the TI-side decay constant with the TE-like and TM-like decay constants of the hyperbolic medium. The appropriate decay constants depend on the optical-axis orientation. Consult the paper's own form of the dispersion relation for this configuration, written as a function whose value must equal a coupling-dependent constant; return that function minus the constant (zero at a surface-wave solution).

Returns
-------
float: dispersion mismatch, zero exactly on the surface-wave branch.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def axion_dispersion_mismatch(n: float, omega_cm: float, eps2: float, alpha: float) -> float:
    """Mismatch of the tangential-axis HM/TI dispersion relation at trial n.

    Args:
        n (float): trial real effective index beta/k0; n^2 must exceed both
            eps2 and eps_e(omega) so that every decay constant is real.
        omega_cm (float): wavenumber in cm^-1 inside the ordinary Reststrahlen
            band of hBN (eps_o < 0).
        eps2 (float): TI-side permittivity (bulk or effective), > 0.
        alpha (float): dimensionless axion interface coupling, >= 0.

    Raises:
        ValueError: if eps_o(omega) >= 0, if n^2 <= max(eps2, eps_e), or if
            alpha < 0.

    Expected return:
        float: dimensionless mismatch; negative just above the lower bound of
        n^2, positive at large n inside the allowed window, zero on the
        surface-wave branch.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: axion_dispersion_mismatch
def _oracle_axion_dispersion_mismatch(n: float, omega_cm: float, eps2: float, alpha: float) -> float:
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    eps = _oracle_hbn_uniaxial_permittivities(omega_cm)
    eps_o, eps_e = float(eps[0]), float(eps[1])
    if eps_o >= 0.0:
        raise ValueError("not in the type-I hyperbolic (eps_o < 0) regime")
    n2 = n ** 2
    if n2 <= max(eps2, eps_e):
        raise ValueError("n^2 must exceed max(eps2, eps_e) for localization")
    q = np.sqrt(n2 - eps2)            # Eq. (15), units of k0
    p1 = np.sqrt(n2 - eps_e)
    p2 = np.sqrt(n2 + abs(eps_o))
    big_f = (q + p1) * (abs(eps_o) / p2 - eps2 / q)   # Eqs. (21)-(24)
    return float(big_f - alpha ** 2)                    # Eq. (25)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "v = axion_dispersion_mismatch(3.2, 1409.3543485, 8.352, 7.2973525693e-3)",
            "call": "round(v, 8)",
            "gold_call": "round(_oracle_axion_dispersion_mismatch(3.2, 1409.3543485, 8.352, 7.2973525693e-3), 8)",
            "tol": 1e-06,
        },
        {
            "setup": "v = axion_dispersion_mismatch(5.0, 1409.3543485, 8.352, 7.2973525693e-3)",
            "call": "round(v, 8)",
            "gold_call": "round(_oracle_axion_dispersion_mismatch(5.0, 1409.3543485, 8.352, 7.2973525693e-3), 8)",
            "tol": 1e-06,
        },
        {
            "setup": "v = axion_dispersion_mismatch(9.0, 1374.5862171, 41.0, 0.0)",
            "call": "round(v, 8)",
            "gold_call": "round(_oracle_axion_dispersion_mismatch(9.0, 1374.5862171, 41.0, 0.0), 8)",
            "tol": 1e-06,
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        axion_dispersion_mismatch(3.0, 1409.3543485, 41.0, 0.007)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_axion_dispersion_mismatch(3.0, 1409.3543485, 41.0, 0.007)\n"
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
                "        axion_dispersion_mismatch(5.0, 1700.0, 8.0, 0.007)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_axion_dispersion_mismatch(5.0, 1700.0, 8.0, 0.007)\n"
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
