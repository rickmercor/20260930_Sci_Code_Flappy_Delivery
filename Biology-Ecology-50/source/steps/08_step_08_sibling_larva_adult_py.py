"""
Compute larva-adult full-sibling probability across sites and days.

A sampled reference larva has an unobserved laying day. A compatible adult full sibling can have been laid earlier or later, provided maternal lifetime and the adult offspring age both allow the pair.

Returns
-------
float: individual larva–adult full-sibling match probability.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sibling_larva_adult(M: "np.ndarray", x1: int, t1: int, x2: int, t2: int, nf: float, beta: float, te: int, tl: int, tp: int, ta: int, mu_a: float, mu_e: float, mu_l: float, mu_p: float) -> float:
    """Compute larva-adult full-sibling probability across sites and days.

Parameters
----------
M : adult daily movement matrix
x1,t1 : reference larva site and sampling day
x2,t2 : target adult site and sampling day
nf,beta : equilibrium females per site and female daily fecundity
te,tl,tp,ta : fixed stage durations and maximum adult age
mu_a,mu_e,mu_l,mu_p : daily stage mortalities

Returns
-------
float, probability that a target adult is a full sibling of the reference larva.

Notes
-----
Marginalize reference egg day y1 in the inclusive range t1-te-(tl-1) through t1-te, with p_L(a)=(1-mu_l)**a/sum_{r=0}^{tl-1}(1-mu_l)**r and a=t1-y1-te. For each y1 consider y2 between y1-(ta-1) and y1+(ta-1), intersected with the adult sampling support t2-te-tl-tp-(ta-1) through t2-te-tl-tp. Multiply each supported term by one half, adult maternal survival (1-mu_a)**abs(y2-y1), larva_adult_movement for these days and sites, and beta*(1-mu_e)**te*(1-mu_l)**tl*(1-mu_p)**tp*(1-mu_a)**(t2-y2-te-tl-tp). Divide the total by adult_denominator. Both offspring orders and adult age zero are included; return zero if no histories fit."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_sibling_larva_adult(M: "np.ndarray", x1: int, t1: int, x2: int, t2: int, nf: float, beta: float, te: int, tl: int, tp: int, ta: int, mu_a: float, mu_e: float, mu_l: float, mu_p: float) -> float:
    dev=te+tl+tp;sl=1-mu_l;sa=1-mu_a
    age_norm=sum(sl**a for a in range(tl))
    total=0.0
    for y1 in range(t1-te-(tl-1),t1-te+1):
        age1=t1-y1-te
        p_l=sl**age1/age_norm
        low=max(y1-(ta-1),t2-dev-(ta-1))
        high=min(y1+ta-1,t2-dev)
        for y2 in range(low,high+1):
            adult_age=t2-y2-dev
            offspring=beta*(1-mu_e)**te*sl**tl*(1-mu_p)**tp*sa**adult_age
            move=_oracle_larva_adult_movement(M,x1,x2,y1,y2,t2,te,tl,tp)
            total+=0.5*p_l*sa**abs(y2-y1)*move*offspring
    return float(total/_oracle_adult_denominator(nf,beta,te,tl,tp,ta,mu_a,mu_e,mu_l,mu_p))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup='import numpy as np\nM=np.array([[.73,.14,.08,.05],[.13,.73,.05,.09],[.1,.07,.73,.1],[.05,.1,.12,.73]],dtype=float)\n'
    return [
        {"setup":setup,"call":"sibling_larva_adult(M,0,3,2,11,25,20,2,5,1,8,.09,.175,.554,.175)","gold_call":"_oracle_sibling_larva_adult(M,0,3,2,11,25,20,2,5,1,8,.09,.175,.554,.175)","tol":1e-9},
        {"setup":setup,"call":"sibling_larva_adult(M,2,6,1,14,25,20,2,5,1,8,.09,.175,.554,.175)","gold_call":"_oracle_sibling_larva_adult(M,2,6,1,14,25,20,2,5,1,8,.09,.175,.554,.175)","tol":1e-9},
        {"setup":setup,"call":"sibling_larva_adult(M,2,0,2,70,25,20,2,5,1,8,.09,.175,.554,.175)","gold_call":"_oracle_sibling_larva_adult(M,2,0,2,70,25,20,2,5,1,8,.09,.175,.554,.175)","tol":1e-9},
    ]
