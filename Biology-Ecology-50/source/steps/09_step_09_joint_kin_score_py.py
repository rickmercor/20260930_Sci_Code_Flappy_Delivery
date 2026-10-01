"""
Combine mother-larva and larva-adult full-sibling count scores.

Independent mother and mixed-stage full-sibling count strata contribute to a joint log pseudo-likelihood. Mother counts require a group match probability; a larval reference and adult target never share an individual identity.

Returns
-------
A float: the combined mother–larva and larva–adult full-sibling log pseudo-likelihood for the supplied daily movement matrix `M`, with binomial coefficients omitted.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def joint_kin_score(M: "np.ndarray", mo_rows: list, la_rows: list, nf: float, beta: float, te: int, tl: int, tp: int, ta: int, mu_a: float, mu_e: float, mu_l: float, mu_p: float) -> float:
    """Combine mother-larva and larva-adult full-sibling count scores.

Parameters
----------
M : daily adult movement matrix
mo_rows : records (x1,t1,x2,t2,n_f_sample,n_l_sample,k_mother)
la_rows : records (x1,t1,x2,t2,n_a_sample,k_sibling)
nf,beta,te,tl,tp,ta,mu_a,mu_e,mu_l,mu_p : demographic parameters

Returns
-------
float, combined log pseudo-likelihood, excluding binomial coefficients.

Notes
-----
For each mother record evaluate individual mother_larva probability P, then use the group probability 1-(1-P)**n_f_sample in k*log(p)+(n_l_sample-k)*log(1-p). For each mixed-stage sibling record use sibling_larva_adult directly in k*log(P)+(n_a_sample-k)*log(1-P). Every listed record is used once, and the larval reference cannot coincide with a target adult, even at matching site and day. Define 0*log(0)=0 by continuity; other impossible observed counts produce negative infinity. The count coefficients independent of dispersal parameters are omitted."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math

def _mixed_binomial_log(p,n,k):
    if not (0<=k<=n and 0<=p<=1):
        raise ValueError("Invalid count or probability")
    if p==0:return 0.0 if k==0 else -math.inf
    if p==1:return 0.0 if k==n else -math.inf
    return k*math.log(p)+(n-k)*math.log1p(-p)

def _oracle_joint_kin_score(M: "np.ndarray", mo_rows: list, la_rows: list, nf: float, beta: float, te: int, tl: int, tp: int, ta: int, mu_a: float, mu_e: float, mu_l: float, mu_p: float) -> float:
    total=0.0
    for x1,t1,x2,t2,nfs,nls,k in mo_rows:
        individual=_oracle_mother_larva(M,x1,t1,x2,t2,nf,beta,te,tl,ta,mu_a,mu_e,mu_l)
        group=-math.expm1(nfs*math.log1p(-individual))
        total+=_mixed_binomial_log(group,nls,k)
    for x1,t1,x2,t2,nas,k in la_rows:
        prob=_oracle_sibling_larva_adult(M,x1,t1,x2,t2,nf,beta,te,tl,tp,ta,mu_a,mu_e,mu_l,mu_p)
        total+=_mixed_binomial_log(prob,nas,k)
    return float(total)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup="""import numpy as np
coords=np.array([[0.,0.],[0.,20.],[20.,0.],[20.,20.]],dtype=float)
mo_rows=[(0, 3, 1, 4, 5, 100, 4),
 (2, 3, 3, 4, 5, 100, 7),
 (0, 4, 2, 5, 5, 100, 1),
 (2, 4, 0, 5, 5, 100, 5),
 (0, 5, 3, 6, 5, 100, 1),
 (2, 5, 1, 6, 5, 100, 2),
 (0, 6, 0, 7, 5, 100, 11),
 (2, 6, 2, 7, 5, 100, 8),
 (0, 7, 1, 8, 5, 100, 4),
 (2, 7, 3, 8, 5, 100, 1),
 (0, 8, 2, 9, 5, 100, 3),
 (2, 8, 0, 9, 5, 100, 2)]
la_rows=[(2, 3, 0, 11, 40, 0),
 (0, 3, 1, 11, 40, 0),
 (2, 3, 2, 11, 40, 0),
 (0, 3, 3, 11, 40, 0),
 (0, 3, 0, 14, 40, 0),
 (2, 3, 1, 14, 40, 1),
 (0, 3, 2, 14, 40, 0),
 (2, 3, 3, 14, 40, 0),
 (0, 4, 0, 12, 40, 0),
 (2, 4, 1, 12, 40, 1),
 (0, 4, 2, 12, 40, 0),
 (2, 4, 3, 12, 40, 0),
 (2, 4, 0, 15, 40, 0),
 (0, 4, 1, 15, 40, 0),
 (2, 4, 2, 15, 40, 0),
 (0, 4, 3, 15, 40, 0),
 (2, 5, 0, 13, 40, 0),
 (0, 5, 1, 13, 40, 0),
 (2, 5, 2, 13, 40, 0),
 (0, 5, 3, 13, 40, 0),
 (0, 5, 0, 16, 40, 0),
 (2, 5, 1, 16, 40, 0),
 (0, 5, 2, 16, 40, 0),
 (2, 5, 3, 16, 40, 0),
 (0, 6, 0, 14, 40, 0),
 (2, 6, 1, 14, 40, 0),
 (0, 6, 2, 14, 40, 2),
 (2, 6, 3, 14, 40, 1),
 (2, 6, 0, 17, 40, 0),
 (0, 6, 1, 17, 40, 0),
 (2, 6, 2, 17, 40, 0),
 (0, 6, 3, 17, 40, 0)]
M=np.array([[.73,.14,.08,.05],[.13,.73,.05,.09],[.1,.07,.73,.1],[.05,.1,.12,.73]],dtype=float)
"""
    return [
        {"setup":setup,"call":"joint_kin_score(M,mo_rows,la_rows[:8],25,20,2,5,1,8,.09,.175,.554,.175)","gold_call":"_oracle_joint_kin_score(M,mo_rows,la_rows[:8],25,20,2,5,1,8,.09,.175,.554,.175)","tol":1e-8},
        {"setup":setup,"call":"joint_kin_score(M,mo_rows[:6],la_rows[12:20],25,20,2,5,1,8,.09,.175,.554,.175)","gold_call":"_oracle_joint_kin_score(M,mo_rows[:6],la_rows[12:20],25,20,2,5,1,8,.09,.175,.554,.175)","tol":1e-8},
        {"setup":setup,"call":"joint_kin_score(M,[],[(0,3,2,11,1,0)],25,20,2,5,1,8,.09,.175,.554,.175)","gold_call":"_oracle_joint_kin_score(M,[],[(0,3,2,11,1,0)],25,20,2,5,1,8,.09,.175,.554,.175)","tol":1e-8},
    ]
