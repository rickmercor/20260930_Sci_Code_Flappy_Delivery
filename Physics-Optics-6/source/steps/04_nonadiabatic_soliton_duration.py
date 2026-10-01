"""
Invert the delayed-response frequency shift of a circulating pulse to find the pulse duration at which a prescribed shift is produced, keeping the finite duration of the vibrational response.

The slowly varying treatment collapses the delayed interaction onto a single time constant, its weighted first moment, and predicts a shift falling as the inverse fourth power of the pulse duration. That collapse is only legitimate while the pulse is long compared with the vibrational period. Once it is not, the shift depends on how much of the vibrational ringing the pulse actually overlaps, and the single moment must be replaced by a duration-dependent overlap of the response against the pulse's own intensity profile.

Two things change together. The overlap rises, turns over near half the vibrational period and then falls away as the pulse shortens further, and a separate factor suppresses the shift once the pulse duration becomes comparable to the vibrational period, because the delayed polarisation cannot fully develop inside a pulse that has already passed. The resulting relation is not the inverse fourth power law and does not reduce to it: over part of the range it predicts a larger shift than the slowly varying estimate and over the rest a smaller one.

Reading that relation backwards turns a target shift into the pulse duration that produces it.

Returns
-------
float, the pulse duration in seconds at which the delayed-response relation delivers the requested frequency shift.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nonadiabatic_soliton_duration(omega_target: float, tau0: float, f_R: float,
                                  tau1: float, tau2: float) -> float:
    '''Return the pulse duration at which the delayed response gives omega_target.

    The delayed vibrational response is the causal damped oscillator

        h(t) = ((tau1/(2*pi))^2 + tau2^2) / ((tau1/(2*pi)) * tau2^2)
               * sin(2*pi*t/tau1) * exp(-t/tau2)   for t > 0, and 0 for t <= 0,

    whose integral over the half line is one. For a hyperbolic-secant pulse of duration tau_s the
    angular frequency shift it produces is

        omega(tau_s) = -2 * tau0^2 * (x / sinh(x)) * theta(tau_s) / tau_s^3,
        with x = pi^2 * tau_s / tau1,

    where theta is the overlap of the response with the pulse,

        theta(tau_s) = f_R * integral over t from 0 to infinity of
                       h(t) * sech(t/tau_s)^2 * tanh(t/tau_s) dt.

    That integrand is the product of a causal, oscillating, decaying response with an odd kernel,
    so it changes sign repeatedly and the value is a small residue of much larger contributions.
    Evaluate it to a relative accuracy of 1e-9 or better; a quadrature that has not converged is
    the dominant error in the returned duration.

    The magnitude of omega decreases strictly as tau_s grows, so the duration reproducing
    omega_target is unique. Return it accurate to a relative precision of 1e-10 or better.

    Parameters
    ----------
    omega_target : float
        Target angular frequency shift in radians per second. Must be negative, because the
        delayed response can only move energy to lower frequency.
    tau0 : float
        Characteristic dispersive duration of the cavity in seconds.
    f_R : float
        Fraction of the nonlinear response carried by the delayed channel.
    tau1 : float
        Vibrational period in seconds.
    tau2 : float
        Vibrational lifetime in seconds.

    Returns
    -------
    tau_s : float
        Pulse duration in seconds, as a native Python float.

    Raises
    ------
    ValueError
        If omega_target is not strictly negative, or if no duration in the searched range produces
        omega_target.
    '''
    return tau_s  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _overlap_time(tau_s: float, f_R: float, tau1: float, tau2: float) -> float:
    # The response decays on tau2, so the half line is truncated at sixty lifetimes; the grid
    # resolves both the ringing period tau1 and the pulse duration tau_s many times over.
    horizon = 60.0 * float(tau2)
    n = 50000
    t = np.linspace(0.0, horizon, n + 1)
    u = np.clip(t / float(tau_s), -300.0, 300.0)
    kernel = (1.0 / np.cosh(u)) ** 2 * np.tanh(u)
    return float(f_R * np.trapezoid(_oracle_raman_response(t, tau1, tau2) * kernel, t))


def _oracle_nonadiabatic_soliton_duration(omega_target: float, tau0: float, f_R: float,
                                          tau1: float, tau2: float) -> float:
    if not (float(omega_target) < 0.0):
        raise ValueError("omega_target must be strictly negative: the delayed response "
                         "shifts energy only towards lower frequency")
    target = float(omega_target)

    def _shift(tau_s):
        x = np.pi * np.pi * tau_s / float(tau1)
        return -2.0 * tau0 * tau0 * (x / np.sinh(x)) * _overlap_time(tau_s, f_R, tau1, tau2) / tau_s ** 3

    # the slowly varying inversion is the right order of magnitude, so it seeds the bracket
    tau_A = float(_oracle_overlap_expansion_coefficients(f_R, tau1, tau2)[0])
    seed = (8.0 * tau0 * tau0 * tau_A / (15.0 * abs(target))) ** 0.25
    lower, upper = seed / 16.0, seed * 16.0
    if not (_shift(lower) < target < _shift(upper)):
        raise ValueError("omega_target is not reachable on the bracketed range of durations")
    for _ in range(200):
        mid = 0.5 * (lower + upper)
        if _shift(mid) < target:
            lower = mid
        else:
            upper = mid
        if upper - lower < 1e-11 * upper:
            break
    return float(0.5 * (lower + upper))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    dev = """import numpy as np
tau0 = 97.34287e-15
tau1 = 2.0 * np.pi * 12.2e-15
tau2 = 32.0e-15
"""
    # Durations are of order 1e-14 s, so the comparisons are made in femtoseconds. The tolerance
    # is set by the overlap quadrature rather than by the root search. It is deliberately a
    # hundred times looser than a converged implementation needs: the oracle's own quadrature
    # sits at 5e-13 relative, and the contract asks only for 1e-9, so anything that fails this
    # has not converged its integral at all rather than having missed a knife edge. Dropping the
    # suppression factor, or using the first moment in place of the overlap, moves the answer by
    # whole femtoseconds.
    return [
        # Normal: the operating point of the benchmark, a target shift of -0.500 THz.
        {
            "setup": dev,
            "call": "1e15 * nonadiabatic_soliton_duration(-2.0*np.pi*500e9, tau0, 0.0217, tau1, tau2)",
            "gold_call": "1e15 * _oracle_nonadiabatic_soliton_duration(-2.0*np.pi*500e9, tau0, 0.0217, tau1, tau2)",
            "tol": 1e-4,
        },
        # Boundary: a far weaker target, which pushes the duration towards the range where the
        # suppression factor dominates and the slowly varying estimate is badly wrong.
        {
            "setup": dev,
            "call": "1e15 * nonadiabatic_soliton_duration(-2.0*np.pi*50e9, tau0, 0.0217, tau1, tau2)",
            "gold_call": "1e15 * _oracle_nonadiabatic_soliton_duration(-2.0*np.pi*50e9, tau0, 0.0217, tau1, tau2)",
            "tol": 1e-4,
        },
        # Edge: a non-negative target must be rejected.
        {
            "setup": dev + """
def run(fn):
    try:
        fn(0.0, tau0, 0.0217, tau1, tau2)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": "run(nonadiabatic_soliton_duration)",
            "gold_call": "run(_oracle_nonadiabatic_soliton_duration)",
        },
    ]
