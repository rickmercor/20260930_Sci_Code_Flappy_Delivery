"""
Assemble the total wavevector magnitude on the finite-frequency exceptional-point ring from its previously computed axial and transverse squared components.

Simple Euclidean combination of the two previously computed squared components; no paper-specific content beyond what the two earlier steps already encode.

Returns
-------
np.ndarray of shape (3,): [kz2, rho, k_total], where rho = kx^2 + ky^2 and k_total = sqrt(kz2 + rho) is the total wavevector magnitude on the ring.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def combine_ep_wavevector(kz2: float, rho: float) -> "np.ndarray":
    """Assemble the total EP-ring wavevector magnitude from its parts.

    Args:
        kz2 (float): squared axial wavevector component, kz2 >= 0.
        rho (float): squared transverse wavevector radius (kx^2 + ky^2), rho >= 0.

    Raises:
        ValueError: if kz2 < 0 or rho < 0.

    Expected return:
        np.ndarray of shape (3,): [kz2, rho, k_total] with
        k_total = sqrt(kz2 + rho).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: combine_ep_wavevector
def _oracle_combine_ep_wavevector(kz2: float, rho: float) -> "np.ndarray":
    if kz2 < 0.0 or rho < 0.0:
        raise ValueError("kz2 and rho must be non-negative")
    return np.array([kz2, rho, np.sqrt(kz2 + rho)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "v = combine_ep_wavevector(81.0/28.0, 33.0/14.0)",
            "call": "np.round(v, 8)",
            "gold_call": "np.round(_oracle_combine_ep_wavevector(81.0/28.0, 33.0/14.0), 8)",
            "tol": 1e-6,
        },
        {
            "setup": "v = combine_ep_wavevector(0.0, 5.25)",
            "call": "np.round(v, 8)",
            "gold_call": "np.round(_oracle_combine_ep_wavevector(0.0, 5.25), 8)",
            "tol": 1e-6,
        },
        {
            "setup": "v = combine_ep_wavevector(2.0, 2.0)",
            "call": "np.round(v, 8)",
            "gold_call": "np.round(_oracle_combine_ep_wavevector(2.0, 2.0), 8)",
            "tol": 1e-6,
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        combine_ep_wavevector(-1.0, 2.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_combine_ep_wavevector(-1.0, 2.0)\n"
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
