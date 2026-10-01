"""
The measured perpendicular fluctuation amplitude in velocity units.

The measured perpendicular fluctuation amplitude in velocity units.

The observed root-mean-square amplitude of the perpendicular magnetic fluctuations, with
both perpendicular components taken together, is a smoothly broken power law in the
perpendicular wavenumber: one power of k_perp well below a break wavenumber and a steeper
one well above it. The heating estimate works with this amplitude in velocity units,
obtained by dividing the magnetic amplitude by sqrt(mu_0 n_p m_p) with the proton mass
density alone, which is the same as scaling delta_B / B0 by the Alfven speed of step 01.

Returns
-------
The perpendicular magnetic fluctuation amplitude in velocity units (km/s) at each
requested perpendicular wavenumber.

Returns
-------
The perpendicular magnetic fluctuation amplitude in velocity units (km/s) at each requested perpendicular wavenumber.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def perpendicular_velocity_amplitude(
    k_perp: "ArrayLike",
    B0_nT: float,
    v_A_kms: float,
    dB_ref_nT: float,
    k_ref: float,
    k_break: float,
    slope_large: float,
    slope_small: float,
) -> "np.ndarray":
    """Return the perpendicular magnetic fluctuation amplitude in velocity units.

    Parameters
    ----------
    k_perp : float or array_like
        Perpendicular wavenumber(s) in rad/km. Non-empty, every entry finite and strictly
        positive.
    B0_nT : float
        Mean magnetic field strength in nT. Finite and strictly positive.
    v_A_kms : float
        Alfven speed of the proton mass density in km/s (step 01). Finite and strictly
        positive.
    dB_ref_nT : float
        Amplitude scale of the magnetic spectrum in nT. Finite and strictly positive.
    k_ref : float
        Reference wavenumber of the spectrum in rad/km. Finite and strictly positive.
    k_break : float
        Break wavenumber of the spectrum in rad/km. Finite and strictly positive.
    slope_large : float
        Power-law index of delta_B well below the break (delta_B ~ k^-slope_large). Finite.
    slope_small : float
        Power-law index of delta_B well above the break (delta_B ~ k^-slope_small). Finite.

    Returns
    -------
    numpy.ndarray
        Array with the shape of ``k_perp`` holding delta_b = delta_B(k_perp) v_A / B0 in
        km/s, where
        delta_B(k) = dB_ref_nT (k / k_ref)^(-slope_large)
        [1 + (k / k_break)^2]^(-(slope_small - slope_large) / 2) in nT is the rms amplitude
        of both perpendicular components together.

    Raises
    ------
    ValueError
        If ``k_perp`` is empty or has a non-finite or non-positive entry, if ``B0_nT``,
        ``v_A_kms``, ``dB_ref_nT``, ``k_ref`` or ``k_break`` is not a finite, strictly
        positive real scalar, or if ``slope_large`` or ``slope_small`` is not a finite real
        scalar.
    """
    return delta_b  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np
from numpy.typing import ArrayLike


def _kmb_finite(name, x):
    """Finite real scalar of either sign."""
    if isinstance(x, (bool, np.bool_)) or np.ndim(x) != 0:
        raise ValueError(f"{name} must be a real scalar")
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v


def _kmb_array(name, x):
    """Non-empty real array (a copy) whose entries are finite and strictly positive."""
    try:
        a = np.array(x, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be real") from None
    if a.size == 0 or not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError(f"{name} must be non-empty, finite and strictly positive")
    return a


def _oracle_perpendicular_velocity_amplitude(
    k_perp: "ArrayLike",
    B0_nT: float,
    v_A_kms: float,
    dB_ref_nT: float,
    k_ref: float,
    k_break: float,
    slope_large: float,
    slope_small: float,
) -> "np.ndarray":
    k = _kmb_array("k_perp", k_perp)
    b0 = _kmb_scalar("B0_nT", B0_nT)
    v_a = _kmb_scalar("v_A_kms", v_A_kms)
    amp = _kmb_scalar("dB_ref_nT", dB_ref_nT)
    k0 = _kmb_scalar("k_ref", k_ref)
    kb = _kmb_scalar("k_break", k_break)
    s_large = _kmb_finite("slope_large", slope_large)
    s_small = _kmb_finite("slope_small", slope_small)
    delta_b_nT = amp * (k / k0) ** (-s_large) * (1.0 + (k / kb) ** 2) ** (-0.5 * (s_small - s_large))
    return delta_b_nT / b0 * v_a

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic normal, boundary, edge and invalid-input cases."""
    imports = "import numpy as np\nimport math\nfrom scipy.special import i0e, j1\n"
    probe = (
        imports
        + "k = np.array([0.1, 0.0, 0.3])\n"
        "def _probe_call():\n"
        "    try:\n"
        "        perpendicular_velocity_amplitude(k.copy(), 180.0, 413.9, 12.0, 1.0e-3, 0.15, 1.0 / 3.0, 0.7)\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
        "def _probe_gold():\n"
        "    try:\n"
        "        _oracle_perpendicular_velocity_amplitude(k.copy(), 180.0, 413.9, 12.0, 1.0e-3, 0.15, 1.0 / 3.0, 0.7)\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
    )
    return [
        # broken spectrum across the inertial range, the break and the sub-proton range
        {
            "setup": imports + "k = np.array([1.0e-3, 2.0e-2, 0.15, 0.28, 2.0])\n",
            "call": "perpendicular_velocity_amplitude(k.copy(), 180.0, 413.854256, 12.0, 1.0e-3, 0.15, 1.0 / 3.0, 0.7)",
            "gold_call": "_oracle_perpendicular_velocity_amplitude(k.copy(), 180.0, 413.854256, 12.0, 1.0e-3, 0.15, 1.0 / 3.0, 0.7)",
        },
        # equal indices: a pure power law, exactly dB_ref v_A / B0 at k = k_ref
        {
            "setup": imports + "k = np.array([[5.0e-4, 5.0e-3], [5.0e-2, 0.5]])\n",
            "call": "perpendicular_velocity_amplitude(k.copy(), 6.0, 55.0, 2.0, 5.0e-3, 0.08, 0.25, 0.25)",
            "gold_call": "_oracle_perpendicular_velocity_amplitude(k.copy(), 6.0, 55.0, 2.0, 5.0e-3, 0.08, 0.25, 0.25)",
        },
        # wavenumbers eight decades apart and a very steep sub-break index
        {
            "setup": imports + "k = np.array([1.0e-6, 1.0e2])\n",
            "call": "perpendicular_velocity_amplitude(k.copy(), 30.0, 120.0, 4.0, 1.0e-2, 1.0, 0.3, 2.8)",
            "gold_call": "_oracle_perpendicular_velocity_amplitude(k.copy(), 30.0, 120.0, 4.0, 1.0e-2, 1.0, 0.3, 2.8)",
        },
        # a zero wavenumber is rejected
        {
            "setup": probe,
            "call": "_probe_call()",
            "gold_call": "_probe_gold()",
        },
    ]
