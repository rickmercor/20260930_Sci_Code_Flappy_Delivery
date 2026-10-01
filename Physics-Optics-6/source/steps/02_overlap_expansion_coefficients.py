"""
Reduce the delayed response to the two numbers that control how a long pulse feels it.

A pulse interacts with the delayed vibrational response through the overlap of that response with the pulse's own intensity profile. For a pulse much longer than the response, that overlap is dominated by the first moment of the response and the interaction collapses onto a single time constant, which is the classical result. The collapse is not exact, and the size of its first correction is what decides where it stops being usable.

Both numbers are properties of the material alone: they do not depend on the pulse. Computing them once, in closed form, replaces every later evaluation of the overlap in the long-pulse regime and fixes the duration below which that regime ends.

Returns
-------
np.ndarray of shape (2,), the coefficients c1 in seconds and c3 in seconds cubed of the long-pulse overlap expansion, in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def overlap_expansion_coefficients(f_R: float, tau1: float, tau2: float) -> "np.ndarray":
    '''Return the two leading coefficients of the long-pulse overlap expansion.

    The delayed vibrational response is the causal damped oscillator

        h(t) = ((tau1/(2*pi))^2 + tau2^2) / ((tau1/(2*pi)) * tau2^2)
               * sin(2*pi*t/tau1) * exp(-t/tau2)   for t > 0, and 0 for t <= 0,

    whose integral over the half line is one. A hyperbolic-secant pulse of duration tau_s feels
    that response through the overlap

        theta(tau_s) = f_R * integral over t from 0 to infinity of
                       h(t) * sech(t/tau_s)^2 * tanh(t/tau_s) dt.

    When tau_s is large compared with the response, theta admits the asymptotic expansion

        theta(tau_s) = c1 / tau_s  +  c3 / tau_s^3  +  O(tau_s^-5).

    Return c1 and c3, in that order, as an array of length two, in SI units, so that c1 is in
    seconds and c3 in seconds cubed. Both are exact closed-form properties of the response and
    must be returned as such rather than fitted to sampled values of theta; the expansion holds
    only asymptotically, so a fit to any finite duration carries the truncated tail with it.

    Note that c3 may be of either sign: it changes sign when the vibrational period passes
    two pi times the lifetime.

    Parameters
    ----------
    f_R : float
        Fraction of the total nonlinear response carried by the delayed vibrational channel,
        between zero and one.
    tau1 : float
        Vibrational period in seconds.
    tau2 : float
        Vibrational lifetime in seconds.

    Returns
    -------
    coefficients : np.ndarray
        Float array of shape (2,) holding c1 in seconds and c3 in seconds cubed, in that order.
    '''
    return coefficients  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_overlap_expansion_coefficients(f_R: float, tau1: float, tau2: float) -> "np.ndarray":
    # sech(u)^2 tanh(u) = u - (4/3) u^3 + O(u^5), so the expansion coefficients are the first
    # and third moments of the response, the third carrying that rational factor.
    a = tau1 / (2.0 * np.pi)
    amp = (a * a + tau2 * tau2) / (a * tau2 * tau2)
    w = 1.0 / a
    s = 1.0 / tau2
    # int_0^inf t^n exp(-s t) sin(w t) dt = Im[ n! / (s - i w)^(n+1) ]
    moment1 = amp * 2.0 * s * w / (s * s + w * w) ** 2
    moment3 = amp * 24.0 * s * w * (s * s - w * w) / (s * s + w * w) ** 4
    return np.array([f_R * moment1, -(4.0 / 3.0) * f_R * moment3], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    # c1 is of order 1e-16 s and c3 of order 1e-43 s^3, both far below any absolute comparison
    # tolerance, so the comparisons are made in femtoseconds and cubed femtoseconds.
    scale = """import numpy as np

def scaled(fn, f_R, tau1, tau2):
    c = np.asarray(fn(f_R, tau1, tau2), dtype=float)
    return np.array([c[0] * 1e15, c[1] * 1e45])
"""
    # An implementation that expands only the hyperbolic tangent, and so misses the squared
    # secant, returns a third coefficient four times too small; one that stops at first order
    # returns zero for it. Either is visible far outside this tolerance.
    identity = """
def hR(t, tau1, tau2):
    a = tau1 / (2.0 * np.pi)
    return np.where(t > 0.0,
                    (a * a + tau2 * tau2) / (a * tau2 * tau2)
                    * np.sin(2.0 * np.pi * t / tau1) * np.exp(-t / tau2), 0.0)

def identity_residual(fn):
    # the coefficients must reproduce the overlap itself in the long-pulse regime
    f_R, tau1, tau2 = 0.0217, 2.0 * np.pi * 12.2e-15, 32.0e-15
    c = np.asarray(fn(f_R, tau1, tau2), dtype=float)
    tau_s = 16.0 * tau1
    t = np.linspace(0.0, 60.0 * tau2, 200001)
    u = t / tau_s
    theta = f_R * np.trapezoid(hR(t, tau1, tau2) * (1.0 / np.cosh(u)) ** 2 * np.tanh(u), t)
    return float(abs(c[0] / tau_s + c[1] / tau_s ** 3 - theta) / abs(theta))
"""
    return [
        # Normal: the silica-clad mode of the device. Here two pi times the lifetime exceeds the
        # vibrational period, so the third coefficient comes out positive.
        {
            "setup": scale,
            "call": "scaled(overlap_expansion_coefficients, 0.0217, 2.0*np.pi*12.2e-15, 32.0e-15)",
            "gold_call": "scaled(_oracle_overlap_expansion_coefficients, 0.0217, 2.0*np.pi*12.2e-15, 32.0e-15)",
            "tol": 1e-6,
        },
        # Boundary: a long-period, short-lived mode on the other side of the sign change, with
        # the whole nonlinearity in the delayed channel.
        {
            "setup": scale,
            "call": "scaled(overlap_expansion_coefficients, 1.0, 400.0e-15, 10.0e-15)",
            "gold_call": "scaled(_oracle_overlap_expansion_coefficients, 1.0, 400.0e-15, 10.0e-15)",
            "tol": 1e-6,
        },
        # Edge: a vanishing delayed fraction must give exactly zero for both coefficients.
        {
            "setup": scale,
            "call": "scaled(overlap_expansion_coefficients, 0.0, 2.0*np.pi*12.2e-15, 32.0e-15)",
            "gold_call": "scaled(_oracle_overlap_expansion_coefficients, 0.0, 2.0*np.pi*12.2e-15, 32.0e-15)",
        },
        # Property: the coefficients are not two free numbers, they must reconstruct the overlap
        # they came from. At sixteen vibrational periods the two-term expansion is accurate to
        # about one part in a million; dropping the rational factor on the third term, or the
        # third moment's sign, shows up here as a residual orders of magnitude larger.
        {
            "setup": scale + identity,
            "call": "identity_residual(overlap_expansion_coefficients)",
            "gold_call": "identity_residual(_oracle_overlap_expansion_coefficients)",
            "tol": 1e-7,
        },
    ]
