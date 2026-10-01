"""
Estimate barrier crossing while accounting for an unknown staying rate.

The barrier parameter and daily staying probability both affect all spatial kinship probabilities. Their joint score is maximized over a closed rectangle to estimate barrier permeability.

Returns
-------
float: fitted barrier crossing factor δ.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_barrier(coords: "np.ndarray", lam: float, barrier_x: float, mo_rows: list, la_rows: list, nf: float, beta: float, te: int, tl: int, tp: int, ta: int, mu_a: float, mu_e: float, mu_l: float, mu_p: float, bounds: tuple, p0_bounds: tuple) -> float:
    """Estimate barrier crossing while accounting for an unknown staying rate.

Parameters
----------
coords : N by 2 site coordinates
lam,barrier_x : conditional dispersal scale and vertical barrier position
mo_rows,la_rows : observed mother and mixed-stage sibling strata
nf,beta,te,tl,tp,ta,mu_a,mu_e,mu_l,mu_p : fixed demographic parameters
bounds,p0_bounds : closed intervals for barrier crossing delta and staying p0

Returns
-------
float, globally maximizing barrier crossing factor delta.

Notes
-----
For each candidate (delta,p0) in the rectangular closed bounds, construct movement_kernel and evaluate joint_kin_score. Maximize that score jointly over both parameters, including endpoints, and return delta alone. A correct global solution within 1e-6 absolute error is accepted for an interior unique optimum; the choice of numerical optimizer is open. This joint fit follows the paper's pseudo-likelihood inference framework."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import differential_evolution

def _oracle_fit_barrier(coords: "np.ndarray", lam: float, barrier_x: float, mo_rows: list, la_rows: list, nf: float, beta: float, te: int, tl: int, tp: int, ta: int, mu_a: float, mu_e: float, mu_l: float, mu_p: float, bounds: tuple, p0_bounds: tuple) -> float:
    def _loss(pair):
        M=_oracle_movement_kernel(coords,lam,barrier_x,pair[0],pair[1])
        return -_oracle_joint_kin_score(M,mo_rows,la_rows,nf,beta,te,tl,tp,ta,mu_a,mu_e,mu_l,mu_p)
    result=differential_evolution(_loss,[bounds,p0_bounds],seed=7,popsize=12,tol=1e-10,maxiter=160,polish=True)
    return float(result.x[0])

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
"""
    return [
        {"setup":setup,"call":"fit_barrier(coords,1/30.,10.,mo_rows,la_rows,25,20,2,5,1,8,.09,.175,.554,.175,(.1,.9),(.05,.95))","gold_call":"_oracle_fit_barrier(coords,1/30.,10.,mo_rows,la_rows,25,20,2,5,1,8,.09,.175,.554,.175,(.1,.9),(.05,.95))","tol":1e-6},
        {"setup":setup,"call":"fit_barrier(coords,1/30.,10.,mo_rows[:10],la_rows[:24],25,20,2,5,1,8,.09,.175,.554,.175,(.1,.9),(.05,.95))","gold_call":"_oracle_fit_barrier(coords,1/30.,10.,mo_rows[:10],la_rows[:24],25,20,2,5,1,8,.09,.175,.554,.175,(.1,.9),(.05,.95))","tol":1e-6},
        {"setup":setup,"call":"fit_barrier(coords,1/30.,10.,mo_rows[2:10],la_rows[4:28],25,20,2,5,1,8,.09,.175,.554,.175,(.1,.9),(.05,.95))","gold_call":"_oracle_fit_barrier(coords,1/30.,10.,mo_rows[2:10],la_rows[4:28],25,20,2,5,1,8,.09,.175,.554,.175,(.1,.9),(.05,.95))","tol":1e-6},
    ]
