"""
Scattering coefficients are amplitudes, and amplitudes are not what a modulated segment conserves or fails to conserve. Energy is. Converting the two families of amplitudes into a power budget is what turns the computation into a statement about the segment drawing energy from the modulation rather than merely redistributing it.

In the uniform rod outside the segment, a scattering order of order n is a plane wave at frequency omega + n omega_m travelling at the unmodulated wave speed, and the time-averaged power it carries past a section is proportional to the product of its wavenumber, its frequency and the square of its amplitude. Since the wavenumber of such an order is its own frequency divided by the wave speed, the power goes as the square of the frequency times the square of the amplitude, with the same constant of proportionality for every order and for the incident wave. Referring each order to the incident wave therefore weights the square of its amplitude by the square of the ratio of its frequency to the incident frequency. A down-converted order of negative frequency contributes with the square of that ratio like any other, its sign having been absorbed into the direction it travels when the exterior wavenumber was assigned.

The sum of those referred powers over both families is the total gain of the segment for that direction of incidence. A lossless and time-invariant scatterer would return exactly one; a passive but lossy one would return less, so unity is the reference for a medium that neither absorbs nor is driven rather than a property of passivity alone. A value above one is the signature of parametric amplification, the modulation having done work on the wave, and it is available here only because the modulus depends explicitly on time, so that the balance of mechanical energy carries a source term proportional to the rate of change of the modulus. The split between the transmitted and the reflected part is reported alongside the total because the two are not amplified equally and the asymmetry between them is part of what the segment does.

Returns
-------
dict holding the native floats transmitted_gain, reflected_gain and total_gain, being the referred power carried by all transmitted orders, by all reflected orders, and by both together, each relative to the power of the incident wave.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def harmonic_power_budget(
    transmission: np.ndarray,
    reflection: np.ndarray,
    omega: float,
    omega_m: float,
) -> dict:
    """Convert the scattering magnitudes into the power the segment returns, referred to the incident power.

    Parameters
    ----------
    transmission : np.ndarray
        Transmission magnitudes indexed by scattering order, odd length.
    reflection : np.ndarray
        Reflection magnitudes in the same indexing.
    omega : float
        Incident angular frequency in radians per second.
    omega_m : float
        Modulation angular frequency in radians per second.

    Returns
    -------
    dict
        Under the keys transmitted_gain, reflected_gain and total_gain, each a
        native float giving the referred power of that family relative to the
        incident power.

    Raises
    ------
    ValueError
        If the two arrays are not one-dimensional, of equal odd length, finite and
        non-negative, or if omega is not above zero or omega_m is zero.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_harmonic_power_budget(
    transmission: np.ndarray,
    reflection: np.ndarray,
    omega: float,
    omega_m: float,
) -> dict:
    transmission = np.asarray(transmission, dtype=np.float64)
    reflection = np.asarray(reflection, dtype=np.float64)
    omega = float(omega)
    omega_m = float(omega_m)
    if transmission.ndim != 1 or reflection.ndim != 1:
        raise ValueError("transmission and reflection must be one-dimensional")
    if transmission.shape != reflection.shape:
        raise ValueError("transmission and reflection must have the same shape")
    if transmission.size % 2 != 1:
        raise ValueError("transmission and reflection must have odd length")
    if not np.all(np.isfinite(transmission)) or not np.all(np.isfinite(reflection)):
        raise ValueError("transmission and reflection must be finite")
    if np.any(transmission < 0.0) or np.any(reflection < 0.0):
        raise ValueError("transmission and reflection are magnitudes and cannot be negative")
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and above zero")
    if not np.isfinite(omega_m) or omega_m == 0.0:
        raise ValueError("omega_m must be finite and not zero")

    j_order = (transmission.size - 1) // 2
    orders = np.arange(-j_order, j_order + 1, dtype=np.float64)
    weight = ((omega + orders * omega_m) / omega) ** 2
    transmitted_gain = float(np.sum(weight * transmission ** 2))
    reflected_gain = float(np.sum(weight * reflection ** 2))
    return {
        "transmitted_gain": transmitted_gain,
        "reflected_gain": reflected_gain,
        "total_gain": transmitted_gain + reflected_gain,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nT = np.array([0.05, 0.2, 2.7, 0.2, 0.02])\nR = np.array([0.04, 4.4, 0.05, 0.03, 0.01])\ndef digest(d):\n    return (round(d['transmitted_gain'], 8), round(d['reflected_gain'], 8), round(d['total_gain'], 8))\n",
            "call": 'digest(harmonic_power_budget(np.array(T, copy=True),np.array(R, copy=True), 15.0, 20.0))',
            "gold_call": 'digest(_oracle_harmonic_power_budget(np.array(T, copy=True),np.array(R, copy=True), 15.0, 20.0))',
        },
        {
            "setup": "import numpy as np\nT = np.array([1.0])\nR = np.array([0.0])\ndef digest(d):\n    return (round(d['transmitted_gain'], 12), round(d['reflected_gain'], 12), round(d['total_gain'], 12))\n",
            "call": 'digest(harmonic_power_budget(np.array(T, copy=True),np.array(R, copy=True), 7.0, 3.0))',
            "gold_call": 'digest(_oracle_harmonic_power_budget(np.array(T, copy=True),np.array(R, copy=True), 7.0, 3.0))',
        },
        {
            "setup": "import numpy as np\nT = np.array([0.0, 0.0, 0.0])\nR = np.array([1.0, 0.0, 1.0])\ndef digest(d):\n    return (round(d['transmitted_gain'], 12), round(d['reflected_gain'], 12), round(d['total_gain'], 12))\n",
            "call": 'digest(harmonic_power_budget(np.array(T, copy=True),np.array(R, copy=True), 15.0, -20.0))',
            "gold_call": 'digest(_oracle_harmonic_power_budget(np.array(T, copy=True),np.array(R, copy=True), 15.0, -20.0))',
        },
        {
            "setup": "import numpy as np\nT = np.array([0.1, 1.0, 0.1])\nR = np.array([0.1, 0.2, 0.1])\nEVEN = np.array([0.1, 1.0])\nNEG = np.array([0.1, -1.0, 0.1])\nNAN = np.array([0.1, float('nan'), 0.1])\nTWO = np.ones((3, 3))\ndef verdict(fn, t=T, r=R, w=15.0, wm=20.0):\n    try:\n        fn(np.array(t, copy=True), np.array(r, copy=True), w, wm)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": '(verdict(harmonic_power_budget, t=EVEN, r=EVEN), verdict(harmonic_power_budget, t=NEG), verdict(harmonic_power_budget, t=NAN), verdict(harmonic_power_budget, t=TWO, r=TWO), verdict(harmonic_power_budget, w=0.0), verdict(harmonic_power_budget, wm=0.0), verdict(harmonic_power_budget))',
            "gold_call": '(verdict(_oracle_harmonic_power_budget, t=EVEN, r=EVEN), verdict(_oracle_harmonic_power_budget, t=NEG), verdict(_oracle_harmonic_power_budget, t=NAN), verdict(_oracle_harmonic_power_budget, t=TWO, r=TWO), verdict(_oracle_harmonic_power_budget, w=0.0), verdict(_oracle_harmonic_power_budget, wm=0.0), verdict(_oracle_harmonic_power_budget))',
        },
    ]
