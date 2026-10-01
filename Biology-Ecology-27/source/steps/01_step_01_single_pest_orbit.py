"""
Post-spray density of the single-strain tau-periodic orbit.

Between sprays a lone pest strain follows dN/dt = N (a - b N); at every spray its density is multiplied by (1 + h), h > -1. This step returns the density N(0+) immediately after a spray on the unique positive tau-periodic orbit of that pulsed logistic population, in closed form. It validates positivity of a, b, tau, the constraint h > -1 and the existence condition a tau + ln(1+h) > 0 (a ValueError is raised otherwise). Deliberately excluded: any interaction with other species and any numerical time stepping.

For the pulsed logistic population the between-spray solution is explicit, and imposing periodicity N(0+) = (1+h) N(tau-) gives the post-spray density

N(0+) = (a/b) [ (1+h) exp(a tau) - 1 ] / [ exp(a tau) - 1 ],

which is positive exactly when a tau + ln(1+h) > 0; when h = 0 it collapses to the carrying capacity a/b. Benchmark values at tau = 2: strain A (a = 1.3, b = 1.0, h = -0.4) gives 0.738278969075; strain B (a = 1.2, h = -0.3) gives 0.8040832420454; strain C (a = 1.0, h = -0.2) gives 0.7686964714501. The pre-spray densities are larger by the factor 1/(1+h) (1.230464948 for strain A) and the unsprayed carrying capacities are 1.3, 1.2 and 1.0: the post-spray section is the one the whole pipeline uses.

Returns
-------
float — post-spray density N(0+) on the positive tau-periodic orbit (0.738278969075 for strain A at tau = 2)
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
            "name": 'normal_strain_A_frozen',
            "setup": '',
            "call": 'single_pest_orbit(1.3, 1.0, -0.4, 2.0)',
            "gold_call": '_oracle_single_pest_orbit(1.3, 1.0, -0.4, 2.0)',
        },
        {
            "name": 'variant_strain_C_frozen',
            "setup": '',
            "call": 'single_pest_orbit(1.0, 1.0, -0.2, 2.0)',
            "gold_call": '_oracle_single_pest_orbit(1.0, 1.0, -0.2, 2.0)',
        },
        {
            "name": 'edge_unsprayed_equals_carrying_capacity',
            "setup": '',
            "call": 'single_pest_orbit(1.2, 1.0, 0.0, 1.5)',
            "gold_call": '_oracle_single_pest_orbit(1.2, 1.0, 0.0, 1.5)',
        },
        {
            "name": 'edge_weak_orbit_near_existence_threshold',
            "setup": '',
            "call": 'single_pest_orbit(1.3, 1.0, -0.4, 0.45)',
            "gold_call": '_oracle_single_pest_orbit(1.3, 1.0, -0.4, 0.45)',
        },
    ]
