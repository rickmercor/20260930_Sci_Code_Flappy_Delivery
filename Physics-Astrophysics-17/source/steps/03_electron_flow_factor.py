"""
Effective electron-flow amplitude of Alfvenic fluctuations at ion scales.

Effective electron-flow amplitude of Alfvenic fluctuations at ion scales.

In a low-beta plasma with isothermal electrons, singly charged ions and perpendicular
scales far larger than the electron inertial length, the magnetic field is carried by the
electron bulk flow. Critically balanced fluctuations with k_perp rho_p << 1 are ordinary
Alfvenic fluctuations, whose flow and magnetic amplitudes are equal in velocity units. At
k_perp rho_p of order one and above they are kinetic-Alfvenic, and the proton
finite-Larmor-radius response and the electron pressure change how the effective electron
flow compares with the magnetic amplitude. This step returns that ratio for the linear
Alfvenic eigenmodes of this model, as a function of k_perp rho_p (rho_p as in step 01)
and of the electron-to-proton temperature ratio.

Returns
-------
The dimensionless effective electron-flow factor alpha_k at each k_perp rho_p.

Returns
-------
The dimensionless effective electron-flow factor alpha_k at each k_perp rho_p.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def electron_flow_factor(k_rho_p: "ArrayLike", te_over_tp: float) -> "np.ndarray":
    """Return the effective electron-flow factor alpha_k at each k_perp rho_p.

    Parameters
    ----------
    k_rho_p : float or array_like
        Perpendicular wavenumber times the proton thermal gyroradius of step 01
        (dimensionless). Non-empty, every entry finite and strictly positive.
    te_over_tp : float
        Electron-to-proton temperature ratio T_e / T_p. Finite and non-negative.

    Returns
    -------
    numpy.ndarray
        Array with the shape of ``k_rho_p`` holding alpha_k = delta u_e,k / delta b_k, the
        ratio of the effective electron bulk-flow amplitude to the perpendicular magnetic
        fluctuation amplitude in velocity units for the linear Alfvenic eigenmodes of the
        model at that k_perp rho_p (singly charged ions). alpha_k tends to 1 as
        k_rho_p -> 0.

    Raises
    ------
    ValueError
        If ``k_rho_p`` is empty or has a non-finite or non-positive entry, or if
        ``te_over_tp`` is not a finite, non-negative real scalar.
    """
    return alpha  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import i0e


def _kmb_one_minus_gamma0(x):
    """1 - I0(x) exp(-x) for x >= 0, with a series below x = 1e-3 to avoid cancellation."""
    x = np.asarray(x, dtype=float)
    series = x * (1.0 - x * (0.75 - x * (5.0 / 12.0 - x * (35.0 / 192.0 - x * 21.0 / 320.0))))
    return np.where(x < 1.0e-3, series, 1.0 - i0e(x))


def _oracle_electron_flow_factor(k_rho_p: "ArrayLike", te_over_tp: float) -> "np.ndarray":
    kr = _kmb_array("k_rho_p", k_rho_p)
    ztau = _kmb_scalar("te_over_tp", te_over_tp, strict=False)
    return kr * np.sqrt(0.5 * (ztau + 1.0 / _kmb_one_minus_gamma0(0.5 * kr * kr)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic normal, boundary, edge and invalid-input cases."""
    imports = "import numpy as np\nimport math\nfrom scipy.special import i0e, j1\n"
    probe = (
        imports
        + "def _probe_call():\n"
        "    try:\n"
        "        electron_flow_factor(np.array([0.5, 1.5]), -0.5)\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
        "def _probe_gold():\n"
        "    try:\n"
        "        _oracle_electron_flow_factor(np.array([0.5, 1.5]), -0.5)\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
    )
    return [
        # from the MHD range through the proton gyroscale to the sub-proton range
        {
            "setup": imports + "kr = np.array([0.05, 0.3, 1.0, 1.494, 3.0, 10.0])\n",
            "call": "electron_flow_factor(kr.copy(), 2.0)",
            "gold_call": "_oracle_electron_flow_factor(kr.copy(), 2.0)",
        },
        # cold electrons: no electron-pressure contribution, finite-Larmor-radius part only
        {
            "setup": imports + "kr = np.array([[0.2, 0.8], [2.5, 6.0]])\n",
            "call": "electron_flow_factor(kr.copy(), 0.0)",
            "gold_call": "_oracle_electron_flow_factor(kr.copy(), 0.0)",
        },
        # very hot electrons and scales far beyond the proton gyroradius
        {
            "setup": imports + "kr = np.array([0.08, 25.0, 80.0])\n",
            "call": "electron_flow_factor(kr.copy(), 15.0)",
            "gold_call": "_oracle_electron_flow_factor(kr.copy(), 15.0)",
        },
        # a negative temperature ratio is rejected
        {
            "setup": probe,
            "call": "_probe_call()",
            "gold_call": "_probe_gold()",
        },
    ]
