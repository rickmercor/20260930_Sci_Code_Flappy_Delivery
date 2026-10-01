"""
The electric-field amplitude seen by a thermal proton.

The electric-field amplitude seen by a thermal proton.

A proton's magnetic moment responds to the perpendicular electric field of a fluctuation,
measured against the proton's own perpendicular speed. For the Alfvenic fluctuations of
step 03 the relevant field is the electrostatic part, whose amplitude at a scale is the
perpendicular wavenumber times the amplitude of the electrostatic potential at that scale.
This step returns that field amplitude normalised to the mean field times the speed of a
thermal proton, taking the proton's perpendicular speed equal to v_thp, given the
perpendicular magnetic amplitude in velocity units at the same scale.

Returns
-------
The dimensionless electric amplitude parameter epsilon_k at each scale.

Returns
-------
The dimensionless electric amplitude parameter epsilon_k at each scale.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def electric_amplitude_parameter(
    k_rho_p: "ArrayLike",
    delta_b_kms: "ArrayLike",
    v_thp_kms: float,
    te_over_tp: float,
) -> "np.ndarray":
    """Return the normalised electric amplitude epsilon_k of the fluctuations at each scale.

    Parameters
    ----------
    k_rho_p : float or array_like
        Perpendicular wavenumber times the proton thermal gyroradius of step 01
        (dimensionless). Non-empty, every entry finite and strictly positive.
    delta_b_kms : float or array_like
        Perpendicular magnetic fluctuation amplitude in velocity units at the same scale(s),
        in km/s (step 02). Non-empty, every entry finite and strictly positive; must
        broadcast with ``k_rho_p``.
    v_thp_kms : float
        Proton thermal speed sqrt(2 k_B T_p / m_p) in km/s, used as the proton's
        perpendicular speed. Finite and strictly positive.
    te_over_tp : float
        Electron-to-proton temperature ratio T_e / T_p. Finite and non-negative.

    Returns
    -------
    numpy.ndarray
        Array with the broadcast shape of ``k_rho_p`` and ``delta_b_kms`` holding
        epsilon_k = E_k / (B0 v_thp), where E_k = k_perp phi_k is the amplitude of the
        electrostatic perpendicular electric field of the Alfvenic fluctuations of step 03
        at that scale.

    Raises
    ------
    ValueError
        If ``k_rho_p`` or ``delta_b_kms`` is empty or has a non-finite or non-positive
        entry, if the two do not broadcast together, if ``v_thp_kms`` is not a finite,
        strictly positive real scalar, or if ``te_over_tp`` is not a finite, non-negative
        real scalar.
    """
    return epsilon  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import i0e


def _oracle_electric_amplitude_parameter(
    k_rho_p: "ArrayLike",
    delta_b_kms: "ArrayLike",
    v_thp_kms: float,
    te_over_tp: float,
) -> "np.ndarray":
    kr = _kmb_array("k_rho_p", k_rho_p)
    db = _kmb_array("delta_b_kms", delta_b_kms)
    v_th = _kmb_scalar("v_thp_kms", v_thp_kms)
    try:
        kr, db = np.broadcast_arrays(kr, db)
    except ValueError:
        raise ValueError("k_rho_p and delta_b_kms must broadcast together") from None
    alpha = _oracle_electron_flow_factor(kr, te_over_tp)
    half_k2 = 0.5 * kr * kr
    return half_k2 / _kmb_one_minus_gamma0(half_k2) * db / (alpha * v_th)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic normal, boundary, edge and invalid-input cases."""
    imports = "import numpy as np\nimport math\nfrom scipy.special import i0e, j1\n"
    probe = (
        imports
        + "kr = np.array([0.5, 1.0, 2.0])\n"
        "db = np.array([3.0, 2.0])\n"
        "def _probe_call():\n"
        "    try:\n"
        "        electric_amplitude_parameter(kr.copy(), db.copy(), 92.0, 2.0)\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
        "def _probe_gold():\n"
        "    try:\n"
        "        _oracle_electric_amplitude_parameter(kr.copy(), db.copy(), 92.0, 2.0)\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
    )
    return [
        # a sweep through the proton gyroscale at the benchmark plasma
        {
            "setup": imports + "kr = np.array([0.1, 0.8, 1.494035, 4.0])\n"
            "db = np.array([6.1, 3.58, 3.203, 1.2])\n",
            "call": "electric_amplitude_parameter(kr.copy(), db.copy(), 92.0, 2.0)",
            "gold_call": "_oracle_electric_amplitude_parameter(kr.copy(), db.copy(), 92.0, 2.0)",
        },
        # one amplitude broadcast over a column of scales, cold electrons
        {
            "setup": imports + "kr = np.array([[0.3], [1.0], [3.0]])\n"
            "db = np.array([25.0])\n",
            "call": "electric_amplitude_parameter(kr.copy(), db.copy(), 40.0, 0.0)",
            "gold_call": "_oracle_electric_amplitude_parameter(kr.copy(), db.copy(), 40.0, 0.0)",
        },
        # deep in the kinetic range with hot electrons and a large amplitude
        {
            "setup": imports + "kr = np.array([12.0, 40.0])\n"
            "db = np.array([80.0, 30.0])\n",
            "call": "electric_amplitude_parameter(kr.copy(), db.copy(), 150.0, 6.0)",
            "gold_call": "_oracle_electric_amplitude_parameter(kr.copy(), db.copy(), 150.0, 6.0)",
        },
        # shapes that do not broadcast are rejected
        {
            "setup": probe,
            "call": "_probe_call()",
            "gold_call": "_probe_gold()",
        },
    ]
