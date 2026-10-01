"""
Evaluate the delayed vibrational part of a Kerr medium's nonlinear response on a grid of time delays.

A transparent glass responds to an optical intensity in two ways. The bound electrons react essentially instantaneously, while the lattice vibrations are set ringing and keep radiating back into the field after the intensity that excited them has passed. The delayed part is what breaks the symmetry between energy flowing up and down in frequency, so a pulse short enough to resolve it is pushed steadily towards lower frequencies. Modelling that delayed part as one damped harmonic oscillator is the standard reduction: the oscillator rings at the dominant vibrational frequency of the glass and decays with the dephasing time of that mode. The response is causal, so it vanishes for non-positive delays, and it is normalised to unit area so that the fractional weight given to it elsewhere carries the whole of its strength.

Two time constants are quoted for such a mode in the literature and they differ by a factor of two pi. The convention used here is stated in the contract below and is not optional: the first constant is the vibrational PERIOD.

Returns
-------
np.ndarray with the shape of t, float64: the normalised delayed vibrational response in inverse seconds, exactly zero at every non-positive delay.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def raman_response(t: "np.ndarray", tau1: float, tau2: float) -> "np.ndarray":
    '''Evaluate the normalised delayed vibrational response at the given time delays.

    The response is that of a single damped harmonic oscillator: it oscillates at the
    vibrational frequency, decays exponentially with the vibrational lifetime, vanishes for
    every non-positive delay, and integrates to one over all positive delays.

    Parameters
    ----------
    t : np.ndarray
        Time delays in seconds. Any shape. Entries that are not strictly positive must give
        exactly zero.
    tau1 : float
        Vibrational PERIOD in seconds, so the oscillation is a sine of argument
        2 * pi * t / tau1. This is not the inverse angular frequency.
    tau2 : float
        Vibrational lifetime in seconds, the exponential decay constant of the ringing.

    Returns
    -------
    h : np.ndarray
        Float array with the shape of t, in units of inverse seconds, normalised so that its
        integral over all positive delays equals one.
    '''
    return h  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_raman_response(t: "np.ndarray", tau1: float, tau2: float) -> "np.ndarray":
    t = np.asarray(t, dtype=float)
    # tau1 is the PERIOD, so the angular frequency of the ringing is 2*pi/tau1 and the
    # oscillator time constant that enters the normalisation is a = tau1/(2*pi).
    a = tau1 / (2.0 * np.pi)
    amp = (a * a + tau2 * tau2) / (a * tau2 * tau2)
    tp = np.where(t > 0.0, t, 0.0)
    return np.where(t > 0.0, amp * np.sin(2.0 * np.pi * tp / tau1) * np.exp(-tp / tau2), 0.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    silica = """import numpy as np
tau1 = 2.0 * np.pi * 12.2e-15
tau2 = 32.0e-15
"""
    return [
        # Normal: a spread of delays across the first few ring-downs of the silica mode.
        {
            "setup": silica + "t = np.linspace(-20e-15, 250e-15, 41)\n",
            "call": "raman_response(t, tau1, tau2)",
            "gold_call": "_oracle_raman_response(t, tau1, tau2)",
        },
        # Boundary: the causal edge, where every non-positive delay must give exactly zero,
        # together with delays a quarter and three quarters through the ring, which carry real
        # signal. Sampling only the sine's own zeros would compare round-off against round-off.
        {
            "setup": silica + "t = np.array([-1e-14, -1e-30, 0.0, tau1/4.0, 3.0*tau1/4.0, 1.25*tau1])\n",
            "call": "raman_response(t, tau1, tau2)",
            "gold_call": "_oracle_raman_response(t, tau1, tau2)",
        },
        # Edge: a strongly overdamped mode on a scalar-shaped input, with unit area checked
        # on a fine grid so a wrong normalisation constant is caught.
        {
            "setup": """import numpy as np
tau1 = 40.0e-15
tau2 = 4.0e-15
grid = np.linspace(0.0, 60.0 * tau2, 200001)
def area_and_values(fn):
    h = fn(grid, tau1, tau2)
    return np.array([np.trapezoid(h, grid), h[0], h[5000], h[40000], h[160000]])
""",
            "call": "area_and_values(raman_response)",
            "gold_call": "area_and_values(_oracle_raman_response)",
        },
    ]
