"""
Compute the equilibrium surviving adult offspring output at one site.

An adult target passed through egg, larval, and pupal stages and survived a possible range of adult ages. The expected adult output at equilibrium provides the adult target denominator.

Returns
-------
float: expected number of surviving adults per site at equilibrium.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def adult_denominator(nf: float, beta: float, te: int, tl: int, tp: int, ta: int, mu_a: float, mu_e: float, mu_l: float, mu_p: float) -> float:
    """Compute the equilibrium surviving adult offspring output at one site.

Parameters
----------
nf : equilibrium adult female count at the site
beta : eggs laid per female per day
te,tl,tp : fixed durations of egg, larva and pupa stages in days
ta : maximum adult age in days, covering ages 0 through ta-1
mu_a,mu_e,mu_l,mu_p : corresponding daily mortality probabilities

Returns
-------
float, expected adult output from all females at one site at a fixed sampling day.

Notes
-----
Under the paper's stationary population assumption, each possible adult age contributes surviving output. Let s_q=1-mu_q for q in {A,E,L,P}. Calculate nf*beta*s_E**te*s_L**tl*s_P**tp*sum(s_A**a for a=0,...,ta-1). The sum includes age zero and is zero only for an empty adult-age range."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_adult_denominator(nf: float, beta: float, te: int, tl: int, tp: int, ta: int, mu_a: float, mu_e: float, mu_l: float, mu_p: float) -> float:
    return float(nf*beta*(1-mu_e)**te*(1-mu_l)**tl*(1-mu_p)**tp*sum((1-mu_a)**a for a in range(ta)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"", "call":"adult_denominator(25,20,2,5,1,8,.09,.175,.554,.175)", "gold_call":"_oracle_adult_denominator(25,20,2,5,1,8,.09,.175,.554,.175)", "tol":1e-9},
        {"setup":"", "call":"adult_denominator(8,12,1,2,3,1,.2,.1,.3,.4)", "gold_call":"_oracle_adult_denominator(8,12,1,2,3,1,.2,.1,.3,.4)", "tol":1e-9},
        {"setup":"", "call":"adult_denominator(10,3,0,1,0,4,0.,0.,0.,0.)", "gold_call":"_oracle_adult_denominator(10,3,0,1,0,4,0.,0.,0.,0.)", "tol":1e-9},
    ]
