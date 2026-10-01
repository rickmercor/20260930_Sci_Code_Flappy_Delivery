"""
Read the pulse duration and the frequency shift of a circulating pulse off its comb power spectrum.

A hyperbolic-secant pulse has a hyperbolic-secant spectrum, so the squared modulus of the mode amplitudes follows a squared hyperbolic secant in the mode offset whose width is the reciprocal of the pulse duration and whose centre is the pulse's carrier offset. Fitting that envelope, rather than taking an intensity-weighted mean of the comb, is the measurement performed here; the two differ by more than ten per cent on a real comb.

Two details are part of the measurement rather than of the analysis. The mode at zero offset is the transmitted drive, not part of the pulse, and is excluded. And the fit is restricted to the modes that carry real signal, here those within forty decibels of the strongest comb line, so that the far wings do not drag the fit.

The sign convention is worth care. A field component that advances in phase with increasing fast time belongs to a mode below the drive in optical frequency, so a positive fitted centre in the transform variable is a shift downwards in optical frequency.

Returns
-------
np.ndarray of shape (2,), float64: the pulse duration in seconds and the angular optical frequency shift in radians per second.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def soliton_observables(spectrum: "np.ndarray", window: float, tau0: float) -> "np.ndarray":
    '''Fit the comb envelope and return the pulse duration and the optical frequency shift.

    The mode offsets are the angular frequencies conjugate to the fast time on a periodic
    window of length window, in the standard transform ordering. The entry at zero offset is
    discarded. Of the remainder, the samples retained are those whose power exceeds one
    ten-thousandth of the largest power among them. The natural logarithm of the retained powers
    is fitted, in the least-squares sense, by the logarithm of an amplitude minus twice the
    logarithm of the hyperbolic cosine of pi times the offset measured from a centre, times a
    width, divided by two. The fitted width is the dimensionless pulse duration and the fitted
    centre is the carrier offset in the transform variable.

    Parameters
    ----------
    spectrum : np.ndarray
        Float array of shape (n_modes,) holding the comb power spectrum in the standard
        transform ordering. Not modified.
    window : float
        Length of the periodic fast-time window in dimensionless units.
    tau0 : float
        Characteristic dispersive duration of the cavity in seconds.

    Returns
    -------
    observables : np.ndarray
        Float array of shape (2,). Entry zero is the pulse duration in seconds, the magnitude
        of the fitted width multiplied by tau0. Entry one is the angular optical frequency
        shift in radians per second, minus the fitted centre divided by tau0, so that a pulse
        pushed to lower optical frequency gives a negative value.
    '''
    return observables  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import curve_fit


def _oracle_soliton_observables(spectrum: "np.ndarray", window: float, tau0: float) -> "np.ndarray":
    power = np.asarray(spectrum, dtype=float).copy()
    n = power.size
    power[0] = 0.0                                   # the transmitted drive is not comb light
    offsets = 2.0 * np.pi * np.fft.fftfreq(n, d=window / n)
    order = np.argsort(offsets)
    x = offsets[order]
    y = power[order]
    keep = y > y.max() * 1e-4                        # forty decibels below the strongest line
    x = x[keep]
    y = y[keep]
    mean = np.sum(x * y) / np.sum(y)
    rms = np.sqrt(np.sum(y * (x - mean) ** 2) / np.sum(y))

    def _envelope(v, log_amp, centre, width):
        arg = np.clip(np.pi * (v - centre) * width / 2.0, -300.0, 300.0)
        return log_amp - 2.0 * np.log(np.cosh(arg))

    guess = [np.log(y.max()), 0.0, 2.0 / max(rms, 1e-12)]
    fitted, _ = curve_fit(_envelope, x, np.log(y), p0=guess, maxfev=80000)
    return np.array([abs(fitted[2]) * tau0, -fitted[1] / tau0], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    guard = """
def checked(fn, arr):
    before = np.array(arr, copy=True)
    out = fn(arr, window, tau0)
    if not np.array_equal(arr, before):
        raise AssertionError("spectrum must not be modified")
    return out
"""
    # The two entries differ by twenty-six orders of magnitude in SI units, so the duration is
    # compared in femtoseconds and the angular shift in inverse nanoseconds.
    return [
        # Normal: a synthetic comb with a known width and a known offset, built directly from
        # the envelope the fit is supposed to recover.
        {
            "setup": """import numpy as np
OBS_SCALE = np.array([1e15, 1e-9])
n = 512
window = 10.07153
tau0 = 97.34287e-15
nu = 2.0 * np.pi * np.fft.fftfreq(n, d=window / n)
spectrum = 3.0 / np.cosh(np.clip(np.pi * (nu - 0.41) * 0.236 / 2.0, -300.0, 300.0)) ** 2
spectrum[0] = 50.0
""" + guard,
            "call": "OBS_SCALE * checked(soliton_observables, np.array(spectrum, copy=True))",
            "gold_call": "OBS_SCALE * checked(_oracle_soliton_observables, np.array(spectrum, copy=True))",
            "tol": 1e-4,
        },
        # Boundary: a comb displaced the other way from case 1, so the two cases pin the sign
        # of the returned shift between them. An exactly centred comb is deliberately NOT used:
        # its correct centre is exactly zero, so the graded value would be the residual at which
        # the fit happens to stop rather than a property of the spectrum.
        {
            "setup": """import numpy as np
OBS_SCALE = np.array([1e15, 1e-9])
n = 256
window = 8.0
tau0 = 50.0e-15
nu = 2.0 * np.pi * np.fft.fftfreq(n, d=window / n)
spectrum = 1.0 / np.cosh(np.clip(np.pi * (nu + 0.75) * 0.5 / 2.0, -300.0, 300.0)) ** 2
spectrum[0] = 9.0
""" + guard,
            "call": "OBS_SCALE * checked(soliton_observables, np.array(spectrum, copy=True))",
            "gold_call": "OBS_SCALE * checked(_oracle_soliton_observables, np.array(spectrum, copy=True))",
            "tol": 1e-4,
        },
        # Edge: a genuine simulated comb, so the fit meets wings, a background and the wrap of
        # the periodic window rather than a clean analytic envelope.
        {
            "setup": """import numpy as np
OBS_SCALE = np.array([1e15, 1e-9])
tau0 = 97.34287e-15
tau1 = 2.0 * np.pi * 12.2e-15
tau2 = 32.0e-15
window = 980.392e-15 / tau0
n = 128
zeta = 17.795878
dtau = 2.0e-4
kh = _oracle_raman_kernel_spectrum(n, window, tau0, tau1, tau2)
pr = _oracle_lle_propagators(zeta, n, window, dtau)
seed = _oracle_soliton_seed(zeta, n, window)
spectrum = _oracle_propagate_soliton(seed, kh, pr, 0.0217, np.sqrt(20.0), dtau, 4000)
""" + guard,
            "call": "OBS_SCALE * checked(soliton_observables, np.array(spectrum, copy=True))",
            "gold_call": "OBS_SCALE * checked(_oracle_soliton_observables, np.array(spectrum, copy=True))",
            "tol": 1e-4,
        },
    ]
