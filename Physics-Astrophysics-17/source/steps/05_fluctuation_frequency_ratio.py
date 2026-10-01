"""
How fast the fluctuations evolve, measured in proton gyrofrequencies.

How fast the fluctuations evolve, measured in proton gyrofrequencies.

Whether fluctuations at a given scale can break a proton's magnetic moment depends on how
their characteristic frequency compares with the proton gyrofrequency. In critically
balanced turbulence the fluctuations at each scale evolve at their nonlinear frequency,
which in the model of step 03 is set by the perpendicular wavenumber of that scale and the
effective electron-flow amplitude there. This step returns that frequency divided by the
proton gyrofrequency, with no order-unity constant applied.

Returns
-------
The dimensionless frequency ratio omega_k / Omega_p at each scale.

Returns
-------
The dimensionless frequency ratio omega_k / Omega_p at each scale.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fluctuation_frequency_ratio(
    k_perp: "ArrayLike",
    delta_b_kms: "ArrayLike",
    omega_p: float,
    rho_p_km: float,
    te_over_tp: float,
) -> "np.ndarray":
    """Return the ratio of the fluctuation frequency to the proton gyrofrequency.

    Parameters
    ----------
    k_perp : float or array_like
        Perpendicular wavenumber(s) in rad/km. Non-empty, every entry finite and strictly
        positive.
    delta_b_kms : float or array_like
        Perpendicular magnetic fluctuation amplitude in velocity units at the same scale(s),
        in km/s (step 02). Non-empty, every entry finite and strictly positive; must
        broadcast with ``k_perp``.
    omega_p : float
        Proton gyrofrequency in rad/s (step 01). Finite and strictly positive.
    rho_p_km : float
        Proton thermal gyroradius in km (step 01). Finite and strictly positive.
    te_over_tp : float
        Electron-to-proton temperature ratio T_e / T_p. Finite and non-negative.

    Returns
    -------
    numpy.ndarray
        Array with the broadcast shape of ``k_perp`` and ``delta_b_kms`` holding
        omega_k / Omega_p, the critically balanced nonlinear frequency of the Alfvenic
        fluctuations of step 03 at that scale divided by the proton gyrofrequency, with no
        order-unity constant applied.

    Raises
    ------
    ValueError
        If ``k_perp`` or ``delta_b_kms`` is empty or has a non-finite or non-positive entry,
        if the two do not broadcast together, if ``omega_p`` or ``rho_p_km`` is not a
        finite, strictly positive real scalar, or if ``te_over_tp`` is not a finite,
        non-negative real scalar.
    """
    return ratio  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import i0e


def _oracle_fluctuation_frequency_ratio(
    k_perp: "ArrayLike",
    delta_b_kms: "ArrayLike",
    omega_p: float,
    rho_p_km: float,
    te_over_tp: float,
) -> "np.ndarray":
    k = _kmb_array("k_perp", k_perp)
    db = _kmb_array("delta_b_kms", delta_b_kms)
    om = _kmb_scalar("omega_p", omega_p)
    rho = _kmb_scalar("rho_p_km", rho_p_km)
    try:
        k, db = np.broadcast_arrays(k, db)
    except ValueError:
        raise ValueError("k_perp and delta_b_kms must broadcast together") from None
    alpha = _oracle_electron_flow_factor(k * rho, te_over_tp)
    return alpha * k * db / om

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
        "        fluctuation_frequency_ratio(np.array([0.28]), np.array([3.2]), 0.0, 5.3, 2.0)\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
        "def _probe_gold():\n"
        "    try:\n"
        "        _oracle_fluctuation_frequency_ratio(np.array([0.28]), np.array([3.2]), 0.0, 5.3, 2.0)\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
    )
    return [
        # the benchmark plasma, across the proton gyroscale
        {
            "setup": imports + "k = np.array([0.05, 0.15, 0.28, 0.6])\n"
            "db = np.array([5.6, 3.6, 3.2, 2.0])\n",
            "call": "fluctuation_frequency_ratio(k.copy(), db.copy(), 17.2419, 5.33584, 2.0)",
            "gold_call": "_oracle_fluctuation_frequency_ratio(k.copy(), db.copy(), 17.2419, 5.33584, 2.0)",
        },
        # MHD-range scales of a 1 au plasma, where the ratio is far below one
        {
            "setup": imports + "k = np.array([[1.0e-4], [1.0e-3]])\n"
            "db = np.array([[20.0, 10.0]])\n",
            "call": "fluctuation_frequency_ratio(k.copy(), db.copy(), 0.4789, 83.5, 1.0)",
            "gold_call": "_oracle_fluctuation_frequency_ratio(k.copy(), db.copy(), 0.4789, 83.5, 1.0)",
        },
        # sub-proton scales with a large amplitude, where the ratio exceeds one
        {
            "setup": imports + "k = np.array([2.0, 9.0])\n"
            "db = np.array([15.0, 6.0])\n",
            "call": "fluctuation_frequency_ratio(k.copy(), db.copy(), 9.58, 1.2, 0.0)",
            "gold_call": "_oracle_fluctuation_frequency_ratio(k.copy(), db.copy(), 9.58, 1.2, 0.0)",
        },
        # a zero gyrofrequency is rejected
        {
            "setup": probe,
            "call": "_probe_call()",
            "gold_call": "_probe_gold()",
        },
    ]
