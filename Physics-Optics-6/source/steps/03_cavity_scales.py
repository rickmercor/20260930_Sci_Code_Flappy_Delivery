"""
Convert the measured properties of a microresonator resonance into the loss rate, the coupling rate, the dispersive time scale and the round-trip time that the dimensionless intracavity model needs.

A microresonator is characterised in the laboratory by quantities an optical spectrum analyser and a swept laser can reach: the wavelength of the resonance, the free spectral range, the curvature of the mode-frequency comb about that resonance, and the intrinsic and external quality factors of the line. The dynamical model that describes a pulse circulating inside the resonator instead wants rates and times. The two quality factors add as inverse rates, one for the light absorbed or scattered away and one for the light that leaves through the bus waveguide, and their sum is the loaded decay rate of the resonance.

One combination deserves attention. The balance between diffraction of the pulse in time and its decay sets a characteristic duration built from the group-velocity dispersion and the loss rate. Written with the mode-family curvature rather than the material dispersion coefficient, the refractive index and the speed of light cancel out of it entirely, so the characteristic duration is fixed by measurable cavity numbers alone. That is what makes the dimensionless model self-contained.

Returns
-------
np.ndarray of shape (4,), float64: the loaded decay rate and the external coupling rate in radians per second, then the characteristic dispersive duration and the round-trip time in seconds.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cavity_scales(lambda0: float, D1: float, D2: float, Q_int: float, Q_ext: float) -> "np.ndarray":
    '''Return the rate and time scales of one microresonator resonance.

    The four entries are, in order: the loaded (total) energy decay rate of the resonance; the
    external coupling rate alone; the characteristic dispersive duration, equal to the square
    root of D2 divided by the product of the loaded decay rate and the square of D1; and the
    cavity round-trip time, equal to two pi divided by D1.

    Parameters
    ----------
    lambda0 : float
        Vacuum wavelength of the resonance in metres.
    D1 : float
        Free spectral range as an angular frequency in radians per second.
    D2 : float
        Second-order dispersion of the mode family as an angular frequency in radians per
        second, positive for anomalous dispersion.
    Q_int : float
        Intrinsic quality factor of the resonance.
    Q_ext : float
        External coupling quality factor of the resonance.

    Returns
    -------
    scales : np.ndarray
        Float array of shape (4,) holding, in this order, the loaded decay rate in radians
        per second, the external coupling rate in radians per second, the characteristic
        dispersive duration in seconds, and the round-trip time in seconds.
    '''
    return scales  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_cavity_scales(lambda0: float, D1: float, D2: float, Q_int: float, Q_ext: float) -> "np.ndarray":
    c_light = 299792458.0
    omega0 = 2.0 * np.pi * c_light / lambda0
    kappa_ext = omega0 / Q_ext
    kappa = omega0 / Q_int + kappa_ext
    # The mode-family curvature already absorbs the index and the speed of light:
    # D2 = -c D1^2 beta2 / n0, so c|beta2|/n0 = D2/D1^2 and tau0^2 = D2/(kappa D1^2).
    tau0 = np.sqrt(D2 / (kappa * D1 * D1))
    return np.array([kappa, kappa_ext, tau0, 2.0 * np.pi / D1], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    # The four entries span seventeen orders of magnitude in SI units, so the two rates are
    # compared in inverse nanoseconds and the two times in femtoseconds. Without that the time
    # entries sit below any absolute tolerance and a constant would pass them.
    return [
        # Normal: the measured resonance of the device.
        {
            "setup": "import numpy as np\nSCALE = np.array([1e-9, 1e-9, 1e15, 1e15])\n",
            "call": "SCALE * cavity_scales(1546e-9, 2.0*np.pi*1.02e12, 2.0*np.pi*41.2e6, 6.2e6, 2.6e6)",
            "gold_call": "SCALE * _oracle_cavity_scales(1546e-9, 2.0*np.pi*1.02e12, 2.0*np.pi*41.2e6, 6.2e6, 2.6e6)",
        },
        # Boundary: critical coupling, where the two quality factors are equal.
        {
            "setup": "import numpy as np\nSCALE = np.array([1e-9, 1e-9, 1e15, 1e15])\n",
            "call": "SCALE * cavity_scales(1550e-9, 2.0*np.pi*500e9, 2.0*np.pi*5.0e6, 3.0e6, 3.0e6)",
            "gold_call": "SCALE * _oracle_cavity_scales(1550e-9, 2.0*np.pi*500e9, 2.0*np.pi*5.0e6, 3.0e6, 3.0e6)",
        },
        # Edge: a very weakly loaded, very high-Q resonance with a large free spectral range,
        # which pushes the dispersive duration to the short end of its useful range.
        {
            "setup": "import numpy as np\nSCALE = np.array([1e-9, 1e-9, 1e15, 1e15])\n",
            "call": "SCALE * cavity_scales(1064e-9, 2.0*np.pi*2.5e12, 2.0*np.pi*120.0e6, 5.0e7, 1.0e9)",
            "gold_call": "SCALE * _oracle_cavity_scales(1064e-9, 2.0*np.pi*2.5e12, 2.0*np.pi*120.0e6, 5.0e7, 1.0e9)",
        },
    ]
