"""
Compose the full constrained-instanton branch competition and compact certificate.

The only final orchestrator combines all eight previous stages, calling the preceding public functions and combining their outputs. Quadrature uses the declared finite-space observable.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def radial_competition(coupling: float = 0.7, lower: float = 0.84, upper: float = 0.97, n: int = 160, radius: float = 16.0, power: float = 2.0, order: int = 24) -> "np.ndarray":
    """Compose the full constrained-instanton branch competition and compact certificate.

    coupling : float > 0, default 0.7
        Dimensionless lambda.
    lower, upper : floats, default 0.84, 0.97
        Integration limits in q=xi/xi_c, with 0<lower<upper<1 and both branches present.
    n : int, default 160
        Number of variable radial fields.
    radius, power : floats, default 16 and 2
        The geometry parameters defined by radial_geometry.
    order : int >= 2, default 24
        Gauss-Legendre quadrature order.
    Returns
    -------
    ndarray, length 9
        [B,K_c,xi_c,K_0(q_m),K_1(q_m),delta_s(q_m),delta_logabsD(q_m),mean_q_1,nu_anchor], with q_m=(lower+upper)/2, deltas branch 1 minus branch 0, and nu_anchor at K=-0.22. All entries are dimensionless. The public implementation must call and combine every preceding public function, including the anchor action and normal response. Geometry is restricted to the family domain in the background.

    Raises
    ------
    ValueError
        If numerical input domains or array shapes are invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import solve_banded, eigvalsh_tridiagonal
from scipy.optimize import root, brentq
from scipy.special import logsumexp

def _oracle_radial_competition(coupling: float = 0.7, lower: float = 0.84, upper: float = 0.97, n: int = 160, radius: float = 16.0, power: float = 2.0, order: int = 24) -> "np.ndarray":
    if not np.isfinite(coupling) or coupling<=0 or not 0<lower<upper<1 or not isinstance(order,(int,np.integer)) or isinstance(order,bool) or order<2:
        raise ValueError("Invalid competition inputs")
    geometry=_oracle_radial_geometry(n,radius,power)
    fold=_oracle_constraint_fold(geometry)
    # An anchor supplies a compact, independently scored spectral certificate.
    anchor=_oracle_stationary_profile(-.22,geometry)
    jet=_oracle_action_jet(anchor,-.22,geometry)
    response=_oracle_normal_response(jet,anchor,geometry)
    nodes,weights=np.polynomial.legendre.leggauss(order)
    fractions=(upper+lower)/2+(upper-lower)/2*nodes
    weights=weights*(upper-lower)/2
    logweights=[]
    for fraction in fractions:
        pair=_oracle_matched_branches(fraction,fold,geometry)
        values=_oracle_branch_logweights(pair,coupling,geometry)
        logweights.append(values[:,3])
    integrated=_oracle_integrated_suppression(fractions,weights,np.asarray(logweights),fold[1])
    middle=_oracle_matched_branches((lower+upper)/2,fold,geometry)
    values=_oracle_branch_logweights(middle,coupling,geometry)
    return np.r_[integrated[0],fold[0],fold[1],middle[:,0],
                 values[1,0]-values[0,0],values[1,2]-values[0,2],integrated[3],response[1]]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '',
      'call': 'radial_competition(coupling=0.55,lower=0.85,upper=0.96,n=128,radius=14.0,power=2.0,order=8)',
      'gold_call': '_oracle_radial_competition(coupling=0.55,lower=0.85,upper=0.96,n=128,radius=14.0,power=2.0,order=8)',
      'tol': 5e-06},
     {'setup': '',
      'call': 'radial_competition(coupling=0.95,lower=0.88,upper=0.98,n=144,radius=15.0,power=2.0,order=10)',
      'gold_call': '_oracle_radial_competition(coupling=0.95,lower=0.88,upper=0.98,n=144,radius=15.0,power=2.0,order=10)',
      'tol': 5e-06},
     {'setup': '',
      'call': 'radial_competition(coupling=0.4,lower=0.92,upper=0.99,n=176,radius=17.0,power=2.0,order=8)',
      'gold_call': '_oracle_radial_competition(coupling=0.4,lower=0.92,upper=0.99,n=176,radius=17.0,power=2.0,order=8)',
      'tol': 5e-06},
     {'setup': '',
      'call': 'radial_competition(coupling=1.2,lower=0.83,upper=0.95,n=160,radius=16.0,power=2.1,order=10)',
      'gold_call': '_oracle_radial_competition(coupling=1.2,lower=0.83,upper=0.95,n=160,radius=16.0,power=2.1,order=10)',
      'tol': 5e-06},
     {'setup': '',
      'call': 'radial_competition(order=2)',
      'gold_call': '_oracle_radial_competition(order=2)',
      'tol': 5e-06}]
