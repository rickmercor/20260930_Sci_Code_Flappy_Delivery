"""
Prepare the delayed response for use inside a periodic fast-time simulation by sampling it on the lag grid and transforming it once.

Inside a resonator the field repeats every round trip, so the fast-time coordinate is periodic and the convolution of the intensity with the delayed response is a circular convolution over one round trip rather than an integral over an infinite line. That is not an approximation made for convenience: the light really does come back round, and the vibrations excited near the end of one round trip are still ringing when the pulse returns. Because the response is causal, its samples occupy the lag grid running forwards from zero; the sample at index zero is zero lag, and the response wraps onto the end of the window rather than onto negative lags.

Transforming the sampled kernel once, ahead of the integration, turns every later convolution into a multiplication. Folding the sample spacing into the transform at the same time means the product with a transformed intensity is already the discrete approximation to the continuous convolution integral, with no further scaling.

The fast time is measured in units of the cavity's characteristic dispersive duration, so the response, which is a rate in the laboratory, has to be re-expressed per unit dimensionless time.

Returns
-------
np.ndarray of shape (n_modes,), complex128: the per-mode factor that turns a transformed intensity into the transform of its circular convolution with the delayed response.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def raman_kernel_spectrum(n_modes: int, window: float, tau0: float,
                          tau1: float, tau2: float) -> "np.ndarray":
    '''Return the discrete Fourier transform of the sampled delayed response.

    The delayed response is sampled on the lag grid of n_modes points running from zero to
    window in units of tau0, with uniform spacing window / n_modes and with index zero at zero
    lag, and is expressed per unit dimensionless time rather than per second. The samples are
    taken as they fall: they are never rescaled, so on a grid too coarse to resolve the ringing
    their sum carries no particular value.

    The array returned is whatever makes the following true: for any intensity sampled on the
    same grid, multiplying the transform of that intensity by this array and inverse
    transforming gives the circular convolution of the intensity with the delayed response over
    one period of the window, as a discrete approximation to the convolution integral.

    Parameters
    ----------
    n_modes : int
        Number of fast-time samples, equal to the number of resonator modes retained.
    window : float
        Length of the periodic fast-time window in units of tau0.
    tau0 : float
        Characteristic dispersive duration of the cavity in seconds.
    tau1 : float
        Vibrational period in seconds.
    tau2 : float
        Vibrational lifetime in seconds.

    Returns
    -------
    kernel_spectrum : np.ndarray
        Complex array of shape (n_modes,), the scaled transform described above.
    '''
    return kernel_spectrum  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_raman_kernel_spectrum(n_modes: int, window: float, tau0: float,
                                  tau1: float, tau2: float) -> "np.ndarray":
    spacing = window / n_modes
    lag = np.arange(n_modes) * spacing
    # tau0 converts the response from per second to per unit dimensionless time.
    samples = _oracle_raman_response(lag * tau0, tau1, tau2) * tau0
    return np.fft.fft(samples) * spacing

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    dev = """import numpy as np
tau0 = 97.34287e-15
tau1 = 2.0 * np.pi * 12.2e-15
tau2 = 32.0e-15
window = 980.392e-15 / tau0
"""
    return [
        # Normal: the grid the benchmark uses.
        {
            "setup": dev,
            "call": "raman_kernel_spectrum(512, window, tau0, tau1, tau2)",
            "gold_call": "_oracle_raman_kernel_spectrum(512, window, tau0, tau1, tau2)",
        },
        # Boundary: a coarse grid that barely resolves the ringing, where the zero-frequency
        # entry no longer sits close to one and the sampling convention is exposed.
        {
            "setup": dev,
            "call": "raman_kernel_spectrum(16, window, tau0, tau1, tau2)",
            "gold_call": "_oracle_raman_kernel_spectrum(16, window, tau0, tau1, tau2)",
        },
        # Edge: a window far longer than the ring-down, on a different cavity scale, so the
        # kernel decays to zero well inside the window and the wrap-around is negligible.
        {
            "setup": """import numpy as np
tau0 = 51.2e-15
tau1 = 30.0e-15
tau2 = 90.0e-15
window = 4000.0e-15 / tau0
""",
            "call": "raman_kernel_spectrum(256, window, tau0, tau1, tau2)",
            "gold_call": "_oracle_raman_kernel_spectrum(256, window, tau0, tau1, tau2)",
        },
    ]
