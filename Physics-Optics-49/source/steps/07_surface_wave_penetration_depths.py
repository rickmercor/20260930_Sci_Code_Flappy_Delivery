"""
Convert a converged effective index into the three lossless transverse penetration depths of the surface wave (tangential optical axis, hBN hyperbolic side): into the TI side (delta_TI), and the TE-like (delta_1) and TM-like (delta_2) depths into the hyperbolic medium, in nanometres.

Each penetration depth is the inverse of the corresponding real decay constant, with the vacuum wavenumber k0 set by the spectroscopic wavenumber (k0 = 2*pi*omega_cm in cm^-1). Use the decay constants appropriate to the tangential-axis configuration.

Returns
-------
np.ndarray of shape (3,): [delta_TI, delta_1, delta_2] in nm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def surface_wave_penetration_depths(n: float, omega_cm: float, eps2: float) -> "np.ndarray":
    """Lossless transverse penetration depths of the HM/TI surface wave.

    Args:
        n (float): effective index beta/k0 with n^2 > max(eps2, eps_e).
        omega_cm (float): wavenumber in cm^-1 inside the ordinary Reststrahlen
            band of hBN.
        eps2 (float): TI-side permittivity (bulk or effective), > 0.

    Raises:
        ValueError: if eps_o(omega) >= 0 or n^2 <= max(eps2, eps_e).

    Expected return:
        np.ndarray of shape (3,): [delta_TI, delta_1, delta_2] in nm, all
        positive; delta_2 is always the smallest of the three.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: surface_wave_penetration_depths
def _oracle_surface_wave_penetration_depths(n: float, omega_cm: float, eps2: float) -> "np.ndarray":
    eps = _oracle_hbn_uniaxial_permittivities(omega_cm)
    eps_o, eps_e = float(eps[0]), float(eps[1])
    if eps_o >= 0.0:
        raise ValueError("not in the type-I hyperbolic (eps_o < 0) regime")
    n2 = n ** 2
    if n2 <= max(eps2, eps_e):
        raise ValueError("n^2 must exceed max(eps2, eps_e)")
    k0_per_nm = 2.0 * np.pi * omega_cm * 1e-7   # cm^-1 -> nm^-1
    q = k0_per_nm * np.sqrt(n2 - eps2)            # Eq. (15)
    p1 = k0_per_nm * np.sqrt(n2 - eps_e)
    p2 = k0_per_nm * np.sqrt(n2 + abs(eps_o))
    return np.array([1.0 / q, 1.0 / p1, 1.0 / p2])  # Eq. (43)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nv = surface_wave_penetration_depths(3.6669037851, 1409.3543485, 8.352)",
            "call": "np.round(v, 4)",
            "gold_call": "np.round(_oracle_surface_wave_penetration_depths(3.6669037851, 1409.3543485, 8.352), 4)",
            "tol": 0.001,
        },
        {
            "setup": "import numpy as np\nv = surface_wave_penetration_depths(8.78978, 1374.5862171, 41.0)",
            "call": "np.round(v, 4)",
            "gold_call": "np.round(_oracle_surface_wave_penetration_depths(8.78978, 1374.5862171, 41.0), 4)",
            "tol": 0.001,
        },
        {
            "setup": "import numpy as np\nv = surface_wave_penetration_depths(3.0, 1450.0, 5.0)",
            "call": "np.round(v, 4)",
            "gold_call": "np.round(_oracle_surface_wave_penetration_depths(3.0, 1450.0, 5.0), 4)",
            "tol": 0.001,
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        surface_wave_penetration_depths(2.0, 1409.3543485, 8.352)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_surface_wave_penetration_depths(2.0, 1409.3543485, 8.352)\n"
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
