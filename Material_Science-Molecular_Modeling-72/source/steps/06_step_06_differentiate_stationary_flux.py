"""
Differentiate the reversible fixed-population likelihood optimum with respect to the molecular perturbation.

The target perturbation changes both the conditional path counts and the independently estimated populations. The requested derivative is the total local response of the optimum under C(epsilon)=C+epsilon*dC and pi(epsilon)=pi+epsilon*dpi. Observed support remains fixed in a neighborhood of zero. This is an author-derived sensitivity extension of the source likelihood.

Returns
-------
dX : np.ndarray: Float (L,L) total derivative of the fitted symmetric flux with respect to alpha, in inverse energy units. Row sums equal dpi. Derivative error may be at most 2e-7 in absolute units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def differentiate_stationary_flux(C: "np.ndarray", dC: "np.ndarray", pi: "np.ndarray", dpi: "np.ndarray", X: "np.ndarray") -> "np.ndarray":
    """Differentiate the reversible fixed-population likelihood optimum with respect to the molecular perturbation.

    Parameters
    ----------
    C : np.ndarray
        (L,L) counts meeting fit_stationary_flux preconditions.
    dC : np.ndarray
        Finite (L,L) count derivatives in inverse energy units; zero
        wherever C is zero so local two-sided feasible perturbations exist.
    pi : np.ndarray
        Strictly positive normalized (L,) equilibrium masses.
    dpi : np.ndarray
        Finite (L,) mass derivative in inverse energy units, summing to zero.
    X : np.ndarray
        (L,L) converged optimum for C and pi from fit_stationary_flux.

    Returns
    -------
    dX : np.ndarray
        Float (L,L) total derivative of the fitted symmetric flux with
        respect to alpha, in inverse energy units. Row sums equal dpi.
        Derivative error may be at most 2e-7 in absolute units."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_differentiate_stationary_flux(C: "np.ndarray", dC: "np.ndarray", pi: "np.ndarray", dpi: "np.ndarray", X: "np.ndarray") -> "np.ndarray":
    # The same fixed normalization is applied to C and its derivative.
    scale=C.sum(); C=C/scale;dC=dC/scale
    lam=np.diag(C)/np.diag(X)
    den=lam[:,None]+lam[None,:]
    S=C+C.T; dS=dC+dC.T
    R=S/den**2
    H=np.diag(R.sum(axis=1))+R
    rhs=np.sum(dS/den,axis=1)-dpi
    dlam=np.linalg.solve(H,rhs)
    return dS/den-R*(dlam[:,None]+dlam[None,:])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Independent scientific configurations, derived from the cited method."""
    return [{'setup': '# Case: coupled_count_and_population_response\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               '\n'
               'C=(lam[:,None]+lam[None,:])*X*split\n'
               'dC=np.array([[.2,-.03,.04],[.06,-.1,.02],[-.02,.03,.15]])\n'
               'dpi=np.array([.04,-.07,.03])\n',
      'call': 'differentiate_stationary_flux(C.copy(), dC.copy(), pi.copy(), dpi.copy(), X.copy())',
      'gold_call': '_oracle_differentiate_stationary_flux(C.copy(), dC.copy(), pi.copy(), dpi.copy(), '
                   'X.copy())',
      'tol': 2e-07},
     {'setup': '# Case: path_law_response_fixed_thermodynamics\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               '\n'
               'C=(lam[:,None]+lam[None,:])*X*split\n'
               'dC=np.array([[.2,-.03,.04],[.06,-.1,.02],[-.02,.03,.15]])\n'
               'dpi=np.array([.04,-.07,.03])\n'
               'dpi[:]=0.\n',
      'call': 'differentiate_stationary_flux(C.copy(), dC.copy(), pi.copy(), dpi.copy(), X.copy())',
      'gold_call': '_oracle_differentiate_stationary_flux(C.copy(), dC.copy(), pi.copy(), dpi.copy(), '
                   'X.copy())',
      'tol': 2e-07},
     {'setup': '# Case: thermodynamic_response_fixed_counts\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               '\n'
               'C=(lam[:,None]+lam[None,:])*X*split\n'
               'dC=np.array([[.2,-.03,.04],[.06,-.1,.02],[-.02,.03,.15]])\n'
               'dpi=np.array([.04,-.07,.03])\n'
               'dC[:]=0.\n',
      'call': 'differentiate_stationary_flux(C.copy(), dC.copy(), pi.copy(), dpi.copy(), X.copy())',
      'gold_call': '_oracle_differentiate_stationary_flux(C.copy(), dC.copy(), pi.copy(), dpi.copy(), '
                   'X.copy())',
      'tol': 2e-07},
     {'setup': '# Case: common_count_rescaling_null_direction\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               '\n'
               'C=(lam[:,None]+lam[None,:])*X*split\n'
               'dC=np.array([[.2,-.03,.04],[.06,-.1,.02],[-.02,.03,.15]])\n'
               'dpi=np.array([.04,-.07,.03])\n'
               'dC=3.7*C;dpi[:]=0.\n',
      'call': 'differentiate_stationary_flux(C.copy(), dC.copy(), pi.copy(), dpi.copy(), X.copy())',
      'gold_call': '_oracle_differentiate_stationary_flux(C.copy(), dC.copy(), pi.copy(), dpi.copy(), '
                   'X.copy())',
      'tol': 2e-07},
     {'setup': '# Case: diagonal_residence_response\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               '\n'
               'C=(lam[:,None]+lam[None,:])*X*split\n'
               'dC=np.array([[.2,-.03,.04],[.06,-.1,.02],[-.02,.03,.15]])\n'
               'dpi=np.array([.04,-.07,.03])\n'
               'dC=np.diag([.4,-.2,.1]);dpi[:]=0.\n',
      'call': 'differentiate_stationary_flux(C.copy(), dC.copy(), pi.copy(), dpi.copy(), X.copy())',
      'gold_call': '_oracle_differentiate_stationary_flux(C.copy(), dC.copy(), pi.copy(), dpi.copy(), '
                   'X.copy())',
      'tol': 2e-07},
     {'setup': '# Case: directed_antisymmetric_null_direction\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               '\n'
               'C=(lam[:,None]+lam[None,:])*X*split\n'
               'dC=np.array([[.2,-.03,.04],[.06,-.1,.02],[-.02,.03,.15]])\n'
               'dpi=np.array([.04,-.07,.03])\n'
               'dC=np.array([[0.,.2,-.1],[-.2,0.,.3],[.1,-.3,0.]]);dpi[:]=0.\n',
      'call': 'differentiate_stationary_flux(C.copy(), dC.copy(), pi.copy(), dpi.copy(), X.copy())',
      'gold_call': '_oracle_differentiate_stationary_flux(C.copy(), dC.copy(), pi.copy(), dpi.copy(), '
                   'X.copy())',
      'tol': 2e-07},
     {'setup': '# Case: rare_state_thermodynamic_sensitivity\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               'pi=np.array([1e-5,.39999,.6]);X=np.array([[7e-6,2e-6,1e-6],[2e-6,.349988,.05],[1e-6,.05,.549999]])\n'
               '\n'
               'C=(lam[:,None]+lam[None,:])*X*split\n'
               'dC=.13*C;dpi=np.array([2e-6,.01,-.010002])\n',
      'call': 'differentiate_stationary_flux(C.copy(), dC.copy(), pi.copy(), dpi.copy(), X.copy())',
      'gold_call': '_oracle_differentiate_stationary_flux(C.copy(), dC.copy(), pi.copy(), dpi.copy(), '
                   'X.copy())',
      'tol': 2e-07},
     {'setup': '# Case: sparse_support_preserved\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               'X[0,2]=0.;X[2,0]=0.;X[0,0]+=.02;X[2,2]+=.02\n'
               '\n'
               'C=(lam[:,None]+lam[None,:])*X*split\n'
               'dC=np.diag([.1,-.3,.2]);dpi=np.array([-.02,.05,-.03])\n',
      'call': 'differentiate_stationary_flux(C.copy(), dC.copy(), pi.copy(), dpi.copy(), X.copy())',
      'gold_call': '_oracle_differentiate_stationary_flux(C.copy(), dC.copy(), pi.copy(), dpi.copy(), '
                   'X.copy())',
      'tol': 2e-07}]
