"""
Determine the post-spray density of the positive periodic orbit for a single pest strain evolving on a one-dimensional boundary face. The population follows logistic growth between sprays and is instantaneously multiplied by the prescribed pulse factor at each intervention. Identify the positive fixed point of the resulting pulse-to-pulse dynamics rather than estimating it from a finite-time simulation. The calculation must reject parameter regimes in which a positive periodic orbit does not exist.

A single pest strain on a pest-only boundary face reduces to a periodically pulsed logistic population model. Between interventions, density-dependent competition limits population growth, while each spray instantaneously rescales the population by a multiplicative factor. A positive periodic orbit corresponds to a state that reproduces itself from one post-spray section to the next. For such an orbit, the integrated per-capita growth over one inter-spray interval is constrained by the logarithmic change produced by the pulse. This identity provides an exact characterization of the periodic state and serves as a consistency relation for higher-dimensional boundary calculations.

Returns
-------
the finite positive post-spray density N(0+) of the unique positive periodic orbit, returned as a single float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def single_pest_orbit(a: float, b: float, h: float, tau: float) -> float:
    """Post-spray density of the positive tau-periodic orbit of a pulsed logistic pest strain.

    Parameters
    ----------
    a : float
        Intrinsic growth rate (per week), a > 0.
    b : float
        Intraspecific competition coefficient (per week per unit density), b > 0.
    h : float
        Multiplicative spray factor, h > -1 (N -> (1 + h) N at every spray).
    tau : float
        Spray interval (weeks), tau > 0.

    Returns
    -------
    float
        N(0+): the density immediately after a spray on the unique positive tau-periodic orbit.

    Raises
    ------
    ValueError
        If a <= 0, b <= 0, tau <= 0, h <= -1, or a*tau + ln(1+h) <= 0 (no positive orbit).
    """

    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import math


def _oracle_single_pest_orbit(a: float, b: float, h: float, tau: float) -> float:
    a = float(a); b = float(b); h = float(h); tau = float(tau)
    for v in (a, b, h, tau):
        if not math.isfinite(v):
            raise ValueError("inputs must be finite")
    if a <= 0.0 or b <= 0.0 or tau <= 0.0 or h <= -1.0:
        raise ValueError("require a > 0, b > 0, tau > 0 and h > -1")
    if a * tau + math.log1p(h) <= 0.0:
        raise ValueError("no positive tau-periodic orbit: a*tau + ln(1+h) <= 0")
    E = math.exp(a * tau)
    return float((a / b) * ((1.0 + h) * E - 1.0) / (E - 1.0))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": '',
            "call": 'single_pest_orbit(1.3, 1.0, -0.4, 2.0)',
            "gold_call": '_oracle_single_pest_orbit(1.3, 1.0, -0.4, 2.0)',
        },
        {
            "setup": '',
            "call": 'single_pest_orbit(1.0, 1.0, -0.2, 2.0)',
            "gold_call": '_oracle_single_pest_orbit(1.0, 1.0, -0.2, 2.0)',
        },
        {
            "setup": '',
            "call": 'single_pest_orbit(1.2, 1.0, 0.0, 1.5)',
            "gold_call": '_oracle_single_pest_orbit(1.2, 1.0, 0.0, 1.5)',
        },
        {
            "setup": '',
            "call": 'single_pest_orbit(1.3, 1.0, -0.4, 0.45)',
            "gold_call": '_oracle_single_pest_orbit(1.3, 1.0, -0.4, 0.45)',
        },
    ]
