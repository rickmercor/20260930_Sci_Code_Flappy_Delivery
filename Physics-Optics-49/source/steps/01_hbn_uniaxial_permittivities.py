"""
Evaluate the two principal relative permittivities of hexagonal boron nitride (hBN), the ordinary (in-plane) component eps_o and the extraordinary (optical-axis) component eps_e, at a given mid-infrared wavenumber, each modelled by a single lossless Lorentz phonon resonance.

In the upper Reststrahlen band of hBN the ordinary permittivity is negative while the extraordinary permittivity stays positive, which is the type-I hyperbolic regime that supports the localized surface wave studied in this task. Parameters (cm^-1): ordinary branch eps_inf_o = 4.87, omega_TO_o = 1360, omega_LO_o = 1614; extraordinary branch eps_inf_e = 2.95, omega_TO_e = 760, omega_LO_e = 825.

Returns
-------
np.ndarray of shape (2,): [eps_o, eps_e] at the requested wavenumber.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def hbn_uniaxial_permittivities(omega_cm: float) -> "np.ndarray":
    """Ordinary and extraordinary relative permittivities of lossless hBN.

    Args:
        omega_cm (float): wavenumber in cm^-1, strictly positive and not equal
            to either transverse-optical resonance (1360 or 760 cm^-1).

    Raises:
        ValueError: if omega_cm <= 0 or omega_cm coincides with a TO resonance.

    Expected return:
        np.ndarray of shape (2,): [eps_o, eps_e]. Inside the upper Reststrahlen
        band (1360 < omega_cm < 1614) eps_o is negative and eps_e is positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: hbn_uniaxial_permittivities
def _oracle_hbn_uniaxial_permittivities(omega_cm: float) -> "np.ndarray":
    if omega_cm <= 0.0:
        raise ValueError("omega_cm must be positive")
    einf_o, wto_o, wlo_o = 4.87, 1360.0, 1614.0
    einf_e, wto_e, wlo_e = 2.95, 760.0, 825.0
    if omega_cm == wto_o or omega_cm == wto_e:
        raise ValueError("omega_cm coincides with a TO resonance")
    w2 = omega_cm ** 2
    eps_o = einf_o * (wlo_o ** 2 - w2) / (wto_o ** 2 - w2)
    eps_e = einf_e * (wlo_e ** 2 - w2) / (wto_e ** 2 - w2)
    return np.array([eps_o, eps_e])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nv = hbn_uniaxial_permittivities(1409.3543485)",
            "call": "np.round(v, 8)",
            "gold_call": "np.round(_oracle_hbn_uniaxial_permittivities(1409.3543485), 8)",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nv = hbn_uniaxial_permittivities(1500.0)",
            "call": "np.round(v, 8)",
            "gold_call": "np.round(_oracle_hbn_uniaxial_permittivities(1500.0), 8)",
            "tol": 1e-06,
        },
        {
            "setup": "import numpy as np\nv = hbn_uniaxial_permittivities(1000.0)",
            "call": "np.round(v, 8)",
            "gold_call": "np.round(_oracle_hbn_uniaxial_permittivities(1000.0), 8)",
            "tol": 1e-06,
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        hbn_uniaxial_permittivities(1360.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_hbn_uniaxial_permittivities(1360.0)\n"
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
                "        hbn_uniaxial_permittivities(-5.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_hbn_uniaxial_permittivities(-5.0)\n"
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
