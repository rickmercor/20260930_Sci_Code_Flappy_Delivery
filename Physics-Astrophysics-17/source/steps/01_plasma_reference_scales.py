"""
Reference scales of the proton-electron plasma.

Reference scales of the proton-electron plasma.

Every later step measures the plasma in three proton reference scales: the proton
gyrofrequency, the proton thermal gyroradius (the thermal speed divided by the
gyrofrequency, with the thermal speed defined as sqrt(2 k_B T_p / m_p)), and the Alfven
speed built from the proton mass density. The inputs are observational: the mean field in
nT, the proton density in cm^-3 and the proton thermal speed in km/s. SI constants are
e = 1.602176634e-19 C, m_p = 1.67262192369e-27 kg and mu_0 = 4 pi x 1e-7 H/m.

Returns
-------
The proton gyrofrequency (rad/s), the proton thermal gyroradius (km) and the Alfven speed
(km/s).

Returns
-------
The proton gyrofrequency (rad/s), the proton thermal gyroradius (km) and the Alfven speed (km/s).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def plasma_reference_scales(B0_nT: float, n_p_cm3: float, v_thp_kms: float) -> "np.ndarray":
    """Return the proton gyrofrequency, thermal gyroradius and Alfven speed.

    Parameters
    ----------
    B0_nT : float
        Mean magnetic field strength in nT. Finite and strictly positive.
    n_p_cm3 : float
        Proton number density in cm^-3. Finite and strictly positive.
    v_thp_kms : float
        Proton thermal speed sqrt(2 k_B T_p / m_p) in km/s. Finite and strictly positive.

    Returns
    -------
    numpy.ndarray
        Array of shape (3,) holding [Omega_p, rho_p, v_A]: the proton gyrofrequency
        Omega_p = e B0 / m_p in rad/s, the proton thermal gyroradius
        rho_p = v_thp / Omega_p in km, and the Alfven speed
        v_A = B0 / sqrt(mu_0 n_p m_p) in km/s.

    Raises
    ------
    ValueError
        If any argument is not a finite, strictly positive real scalar.
    """
    return scales  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _kmb_scalar(name, x, strict=True):
    """Finite real scalar: strictly positive when strict, otherwise non-negative."""
    if isinstance(x, (bool, np.bool_)) or np.ndim(x) != 0:
        raise ValueError(f"{name} must be a real scalar")
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number") from None
    if not math.isfinite(v) or v < 0.0 or (strict and v == 0.0):
        raise ValueError(f"{name} must be finite and {'strictly positive' if strict else 'non-negative'}")
    return v


def _kmb_si_constants():
    """SI constants (elementary charge in C, proton mass in kg, vacuum permeability in H/m)."""
    return 1.602176634e-19, 1.67262192369e-27, 4.0e-7 * math.pi


def _oracle_plasma_reference_scales(B0_nT: float, n_p_cm3: float, v_thp_kms: float) -> "np.ndarray":
    b_tesla = _kmb_scalar("B0_nT", B0_nT) * 1.0e-9
    n_si = _kmb_scalar("n_p_cm3", n_p_cm3) * 1.0e6
    v_th = _kmb_scalar("v_thp_kms", v_thp_kms)
    charge, m_p, mu_0 = _kmb_si_constants()
    omega_p = charge * b_tesla / m_p
    v_a = b_tesla / math.sqrt(mu_0 * n_si * m_p) / 1.0e3
    return np.array([omega_p, v_th / omega_p, v_a])

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
        "        plasma_reference_scales(180.0, 0.0, 92.0)\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
        "def _probe_gold():\n"
        "    try:\n"
        "        _oracle_plasma_reference_scales(180.0, 0.0, 92.0)\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
    )
    return [
        # near-Sun low-beta stream
        {
            "setup": imports,
            "call": "plasma_reference_scales(180.0, 90.0, 92.0)",
            "gold_call": "_oracle_plasma_reference_scales(180.0, 90.0, 92.0)",
        },
        # 1 au wind with proton beta of order one (v_thp comparable to v_A)
        {
            "setup": imports,
            "call": "plasma_reference_scales(5.0, 5.0, 40.0)",
            "gold_call": "_oracle_plasma_reference_scales(5.0, 5.0, 40.0)",
        },
        # strong-field, dense, hot corona: extreme magnitudes in every scale
        {
            "setup": imports,
            "call": "plasma_reference_scales(2.0e5, 1.0e9, 1.5e3)",
            "gold_call": "_oracle_plasma_reference_scales(2.0e5, 1.0e9, 1.5e3)",
        },
        # zero density is rejected
        {
            "setup": probe,
            "call": "_probe_call()",
            "gold_call": "_probe_gold()",
        },
    ]
