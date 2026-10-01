"""
Critical spray interval of the sprayed four-species community (orchestrator).

This final step assembles the whole pipeline. At the reference interval tau-zero and at both bracket ends it builds the census from the earlier steps (closed-form single-strain orbits checked against the stroboscopic map, averages identity on the pest faces checked against the quadrature, Newton-shot face orbits, within-face multipliers of the strain-parasitoid orbits, pulse-adjusted rates with the resident-zero identity, census cross-check), determines the Morse pieces, verifies that the decomposition is the same at tau-lo, tau-zero and tau-hi, and evaluates the margin through the weighted piece values; the same chain with the orbits re-shot at every tau is used inside a bracketing root search (brentq, absolute tolerance 1e-12) on [tau-lo, tau-hi], where m(tau-lo) and m(tau-hi) must have opposite signs, and the located zero tau-star is returned. The margin step is used only as an independent cross-check at tau-zero and at tau-star; a ValueError is raised on any inconsistency. Deliberately excluded: anything beyond the sufficient criterion (a negative margin is not a proof of extinction).

Spraying often knocks the parasitoid back and tilts the cycle of strain-parasitoid orbits towards attraction, spraying rarely lets it certify; the smallest interval for which the four-species community is certified permanent is the zero of the margin on [1.2, 3.0] weeks. For the frozen parameters the margin increases from -0.01445716818084 at tau = 1.2 to 0.01081570636861 at tau = 3.0, the limiting piece is the parasitoid cycle throughout and the census stays at 7 orbits and 4 pieces; the zero is tau-star = 1.6988499591 weeks. Convention errors move it outside the required six significant figures or remove it altogether: per-orbit best-single-invader margins have no zero on the bracket (they are positive everywhere), omitting the invaders' own spray term gives no zero either, and evaluating the rates on the parasitoid orbits at period-averaged densities gives 1.7070796. Variants: h(P) = -0.3 gives 1.5976489361, d = 0.45 gives 1.6371516307 and c(C) = 2.8 gives 1.6309483967.

Returns
-------
float — critical spray interval tau* in weeks (1.698849959099 for the frozen set on [1.2, 3.0])
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def critical_spray_interval(params: dict, tau0: float, tau_lo: float, tau_hi: float) -> float:
    """Critical spray interval tau* at which the permanence margin of the sprayed community vanishes.
 
    Parameters
    ----------
    params : dict
        Model parameters (keys 'a', 'B', 'e', 'c', 'om', 'd', 'h').
    tau0 : float
        Reference spray interval (weeks) at which the full boundary analysis is assembled and cross-checked.
    tau_lo, tau_hi : float
        Bracket (weeks) with m(tau_lo) and m(tau_hi) of opposite signs, 0 < tau_lo < tau_hi.
 
    Returns
    -------
    float
        The zero tau* of the permanence margin m(tau) in [tau_lo, tau_hi], located to 1e-12.
 
    Raises
    ------
    ValueError
        On invalid inputs, a bracket that does not enclose a sign change, a Morse decomposition that is not
        the same at tau_lo, tau0 and tau_hi, or any inconsistency between the assembled chain and the
        independent margin evaluation.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import itertools
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog, brentq
 
 
def _s09_chain(params, tau, pieces_ref=None):
    """Assemble the orbits, rates and margin from the earlier steps' oracles: the answer path of the pipeline."""
    P = _s08_params(params)
    a, B, e, c, om, d, h = P
    orbits, rates = _s08_census(tau, P)
    # cross-check every orbit with the individual steps
    for key, z in orbits.items():
        T = list(key)
        if len(T) == 1 and T[0] < 3:
            if abs(z[T[0]] - _oracle_single_pest_orbit(a[T[0]], B[T[0], T[0]], h[T[0]], tau)) > 1e-10:
                raise ValueError("single-strain orbit disagrees with the closed form")
        if len(T) >= 2:
            for i in T:
                if abs(_oracle_face_orbit_post_pulse(params, T, z[T], tau, i) - z[i]) > 1e-10:
                    raise ValueError("face orbit is not reproduced by the shooting step")
            if _oracle_stroboscopic_floquet_radius(params, T, z, tau) >= 1.0 and 3 in T:
                raise ValueError("a pest-parasitoid orbit does not attract within its face")
        if len(T) == 1:
            if abs(_oracle_stroboscopic_map(params, z, tau, T[0]) - z[T[0]]) > 1e-10:
                raise ValueError("single-species orbit is not a fixed point of the one-interval map")
        if all(i < 3 for i in T):
            mean = _s09_mean(z, T, tau, P)
            for i in T:
                if abs(_oracle_pest_face_mean_density(params, T, tau, i) - mean[T.index(i)]) > 1e-9:
                    raise ValueError("period average on a pest-only orbit disagrees with the averages identity")
        for i in range(4):
            if abs(_oracle_pulsed_invasion_rate(params, z, i, tau) - rates[key][i]) > 1e-10:
                raise ValueError("rate along an orbit disagrees with the invasion-rate step")
            if i in T and abs(rates[key][i]) > 1e-9:
                raise ValueError("a resident species must have zero long-term growth on its own orbit")
    if float(len(orbits)) != _oracle_boundary_census(params, tau):
        raise ValueError("boundary census mismatch")
    if pieces_ref is None:
        pieces = _s08_pieces(tau, P, orbits, rates)
    else:
        pieces = pieces_ref
        for piece in pieces:
            for key in piece:
                if key != "origin" and key not in orbits:
                    raise ValueError("an orbit of the reference decomposition is missing at this interval")
    m = math.inf
    for piece in pieces:
        val, w = _s08_piece_margin(piece, tau, P, rates)
        m = min(m, val)
    return m, pieces
 
 
def _s09_mean(z, T, tau, P):
    """Period averages of the present species along the orbit through z (quadrature of the flow)."""
    def _rhs(t, y):
        zz = y[:4]
        return np.concatenate([zz * _s08_rates(zz, P), zz])
    sol = solve_ivp(_rhs, (0.0, tau), np.concatenate([z, np.zeros(4)]), method="DOP853", rtol=1e-13, atol=1e-16)
    return sol.y[4:8, -1][T] / tau
 
 
def _oracle_critical_spray_interval(params: dict, tau0: float, tau_lo: float, tau_hi: float) -> float:
    _s08_params(params)
    tau0 = float(tau0)
    tau_lo = float(tau_lo)
    tau_hi = float(tau_hi)
    if not (math.isfinite(tau0) and math.isfinite(tau_lo) and math.isfinite(tau_hi)):
        raise ValueError("tau0, tau_lo and tau_hi must be finite")
    if tau0 <= 0.0 or not (0.0 < tau_lo < tau_hi):
        raise ValueError("require tau0 > 0 and 0 < tau_lo < tau_hi")
    m0, pieces0 = _s09_chain(params, tau0)
    if abs(m0 - _oracle_permanence_margin(params, tau0)) > 1e-9:
        raise ValueError("margin assembled from the chain disagrees with the margin step")
    m_lo, pieces_lo = _s09_chain(params, tau_lo)
    m_hi, pieces_hi = _s09_chain(params, tau_hi)
    if set(pieces_lo) != set(pieces0) or set(pieces_hi) != set(pieces0):
        raise ValueError("the Morse decomposition changes across the bracket")
    if not (m_lo * m_hi < 0.0):
        raise ValueError("bracket does not enclose a sign change of the margin")
    f = lambda t: _s09_chain(params, t, pieces_ref=pieces0)[0]
    tau_star = brentq(f, tau_lo, tau_hi, xtol=1e-12, maxiter=200)
    if abs(_oracle_permanence_margin(params, tau_star)) > 1e-8:
        raise ValueError("margin at the located threshold is not zero")
    return float(tau_star)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "name": 'normal_frozen_final_answer',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}",
            "call": 'critical_spray_interval(PZ, 2.0, 1.2, 3.0)',
            "gold_call": '_oracle_critical_spray_interval(PZ, 2.0, 1.2, 3.0)',
        },
        {
            "name": 'variant_stronger_parasitoid_spray',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}; PZv = dict(PZ, h=[-0.4, -0.3, -0.2, -0.3])",
            "call": 'critical_spray_interval(PZv, 2.0, 1.2, 3.0)',
            "gold_call": '_oracle_critical_spray_interval(PZv, 2.0, 1.2, 3.0)',
        },
        {
            "name": 'variant_higher_parasitoid_mortality',
            "setup": "PZ = {'a': [1.3, 1.2, 1.0], 'B': [[1.0, 1.4, 0.4], [0.4, 1.0, 1.6], [2.0, 0.1, 1.0]], 'e': [1.2, 1.2, 1.0], 'c': [1.5, 1.5, 3.0], 'om': [0.4, 0.15, 0.2], 'd': 0.4, 'h': [-0.4, -0.3, -0.2, -0.2]}; PZv = dict(PZ, d=0.45)",
            "call": 'critical_spray_interval(PZv, 2.0, 1.2, 3.0)',
            "gold_call": '_oracle_critical_spray_interval(PZv, 2.0, 1.2, 3.0)',
        },
    ]
