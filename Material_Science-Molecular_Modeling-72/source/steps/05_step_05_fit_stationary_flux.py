"""
Fit a reversible transition model with independently fixed equilibrium populations.

Maximize the finite-data log likelihood sum_ij C_ij log(P_ij) over row-stochastic nonnegative P satisfying pi_i P_ij=pi_j P_ji. Return the equilibrium flux X_ij=pi_i P_ij. The observed counts are directed and need not be equilibrated. The problem has positive diagonal counts and connected undirected observed support; absent off-diagonal support carries zero fitted flux.

Returns
-------
X : np.ndarray: Dimensionless float (L,L) symmetric maximum-likelihood equilibrium flux with row sums pi. Its support is C+C.T. Common positive rescaling of every count leaves the required flux unchanged. Return the converged optimum, with absolute error at most 2e-9.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_stationary_flux(C: "np.ndarray", pi: "np.ndarray") -> "np.ndarray":
    """Fit a reversible transition model with independently fixed equilibrium populations.

    Parameters
    ----------
    C : np.ndarray
        Nonnegative finite (L,L) raw conditional path counts, L >= 2,
        positive diagonal, and connected support of C+C.T.
    pi : np.ndarray
        Strictly positive finite (L,) fixed masses summing to one.

    Returns
    -------
    X : np.ndarray
        Dimensionless float (L,L) symmetric maximum-likelihood equilibrium
        flux with row sums pi. Its support is C+C.T. Common positive
        rescaling of every count leaves the required flux unchanged.
        Return the converged optimum, with absolute error at most 2e-9."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import root

def _oracle_fit_stationary_flux(C: "np.ndarray", pi: "np.ndarray") -> "np.ndarray":
    scale=C.sum();C=C/scale
    S=C+C.T
    def _fun(z):
        lam=np.exp(z); den=lam[:,None]+lam[None,:]
        return np.sum(S/den,axis=1)/pi-1
    def _jac(z):
        lam=np.exp(z);den=lam[:,None]+lam[None,:]
        R=S/den**2
        return -(np.diag(R.sum(axis=1))+R)*lam[None,:]/pi[:,None]
    solved=root(_fun,np.log(C.sum(axis=1)/pi),jac=_jac,tol=1e-11)
    lam=np.exp(solved.x)
    X=S/(lam[:,None]+lam[None,:])
    if np.max(np.abs(X.sum(axis=1)-pi))>2e-11:raise RuntimeError('Bad constraint')
    return X

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Independent scientific configurations, derived from the cited method."""
    return [{'setup': '# Case: asymmetric_directed_sampling\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               '\n'
               'C=(lam[:,None]+lam[None,:])*X*split\n',
      'call': 'fit_stationary_flux(C.copy(), pi.copy())',
      'gold_call': '_oracle_fit_stationary_flux(C.copy(), pi.copy())',
      'tol': 2e-08},
     {'setup': '# Case: two_state_closed_form\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               'pi=np.array([.3,.7]);X=np.array([[.22,.08],[.08,.62]]);lam=np.array([1.2,3.]);split=np.array([[.5,.85],[.15,.5]])\n'
               '\n'
               'C=(lam[:,None]+lam[None,:])*X*split\n',
      'call': 'fit_stationary_flux(C.copy(), pi.copy())',
      'gold_call': '_oracle_fit_stationary_flux(C.copy(), pi.copy())',
      'tol': 2e-08},
     {'setup': '# Case: balanced_observed_flux\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               'lam[:]=1.;split[:]=.5\n'
               '\n'
               'C=(lam[:,None]+lam[None,:])*X*split\n',
      'call': 'fit_stationary_flux(C.copy(), pi.copy())',
      'gold_call': '_oracle_fit_stationary_flux(C.copy(), pi.copy())',
      'tol': 2e-08},
     {'setup': '# Case: one_way_observed_edge\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               'split[0,1]=1.;split[1,0]=0.\n'
               '\n'
               'C=(lam[:,None]+lam[None,:])*X*split\n',
      'call': 'fit_stationary_flux(C.copy(), pi.copy())',
      'gold_call': '_oracle_fit_stationary_flux(C.copy(), pi.copy())',
      'tol': 2e-08},
     {'setup': '# Case: sparse_connected_chain\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               'X[0,2]=0.;X[2,0]=0.;X[0,0]+=.02;X[2,2]+=.02\n'
               '\n'
               'C=(lam[:,None]+lam[None,:])*X*split\n',
      'call': 'fit_stationary_flux(C.copy(), pi.copy())',
      'gold_call': '_oracle_fit_stationary_flux(C.copy(), pi.copy())',
      'tol': 2e-08},
     {'setup': '# Case: rare_thermodynamic_state\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               'pi=np.array([1e-5,.39999,.6]);X=np.array([[7e-6,2e-6,1e-6],[2e-6,.349988,.05],[1e-6,.05,.549999]])\n'
               '\n'
               'C=(lam[:,None]+lam[None,:])*X*split\n',
      'call': 'fit_stationary_flux(C.copy(), pi.copy())',
      'gold_call': '_oracle_fit_stationary_flux(C.copy(), pi.copy())',
      'tol': 2e-08},
     {'setup': '# Case: weak_interbasin_bottleneck\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               'X=np.array([[.20997,.00002,.00001],[.00002,.30998,.03],[.00001,.03,.41999]])\n'
               '\n'
               'C=(lam[:,None]+lam[None,:])*X*split\n',
      'call': 'fit_stationary_flux(C.copy(), pi.copy())',
      'gold_call': '_oracle_fit_stationary_flux(C.copy(), pi.copy())',
      'tol': 2e-08},
     {'setup': '# Case: large_count_scale\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               'lam*=1e9;split[0,2]=.99;split[2,0]=.01\n'
               '\n'
               'C=(lam[:,None]+lam[None,:])*X*split\n',
      'call': 'fit_stationary_flux(C.copy(), pi.copy())',
      'gold_call': '_oracle_fit_stationary_flux(C.copy(), pi.copy())',
      'tol': 2e-08}]
