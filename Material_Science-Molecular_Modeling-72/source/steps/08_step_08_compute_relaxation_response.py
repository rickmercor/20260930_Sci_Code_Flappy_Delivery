"""
Extract the slow positive relaxation mode and its logarithmic sensitivity.

A reversible transition operator is similar to a real symmetric matrix under equilibrium weighting. Follow its largest nonstationary eigenvalue locally under the supplied operator perturbation. The derivative is taken at fixed lag duration. A simple slow eigenvalue is assumed; faster modes can have negative eigenvalues.

Returns
-------
response : np.ndarray: Float (3,): slow eigenvalue lambda_1, implied relaxation time t_1, and d(log(t_1))/dalpha, in that order. Units are 1, time, and inverse energy. Report the local simple-eigenvalue derivative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_relaxation_response(op: "np.ndarray", pi: "np.ndarray", lagtime: float) -> "np.ndarray":
    """Extract the slow positive relaxation mode and its logarithmic sensitivity.

    Parameters
    ----------
    op : np.ndarray
        Float (2,L,L), L >= 2: reversible row-stochastic P and its total
        alpha derivative dP. dP has zero row sums. Its eigenvalue 1 is
        simple, and the second-largest algebraic eigenvalue is simple
        and lies strictly between zero and one. The perturbation is a
        derivative of a reversible stochastic family.
    pi : np.ndarray
        Positive normalized float (L,) stationary vector of P.
    lagtime : float
        Positive physical duration of one transition.

    Returns
    -------
    response : np.ndarray
        Float (3,): slow eigenvalue lambda_1, implied relaxation time t_1,
        and d(log(t_1))/dalpha, in that order. Units are 1, time, and
        inverse energy. Report the local simple-eigenvalue derivative."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_relaxation_response(op: "np.ndarray", pi: "np.ndarray", lagtime: float) -> "np.ndarray":
    P,dP=op
    sp=np.sqrt(pi)
    S=sp[:,None]*P/sp[None,:]
    vals,vecs=np.linalg.eigh((S+S.T)/2)
    lam=vals[-2];u=vecs[:,-2]
    dlam=u@(sp[:,None]*dP/sp[None,:])@u
    return np.array([lam,-lagtime/np.log(lam),-dlam/(lam*np.log(lam))])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Independent scientific configurations, derived from the cited method."""
    return [{'setup': '# Case: competing_slow_modes\n'
               'import numpy as np\n'
               'P=np.array([[.86,.08,.06],[.08,.82,.1],[.06,.1,.84]])\n'
               'dP=np.array([[-.03,.01,.02],[.01,-.005,-.005],[.02,-.005,-.015]])\n'
               'pi=np.ones(3)/3;lagtime=.4\n'
               'op=np.stack([P,dP])\n',
      'call': 'compute_relaxation_response(op.copy(), pi.copy(), lagtime)',
      'gold_call': '_oracle_compute_relaxation_response(op.copy(), pi.copy(), lagtime)',
      'tol': 2e-08},
     {'setup': '# Case: two_state_mode\n'
               'import numpy as np\n'
               'P=np.array([[.86,.08,.06],[.08,.82,.1],[.06,.1,.84]])\n'
               'dP=np.array([[-.03,.01,.02],[.01,-.005,-.005],[.02,-.005,-.015]])\n'
               'pi=np.ones(3)/3;lagtime=.4\n'
               'P=np.array([[.8,.2],[.1,.9]]);dP=np.array([[-.03,.03],[.01,-.01]]);pi=np.array([1/3,2/3]);op=np.stack([P,dP])\n',
      'call': 'compute_relaxation_response(op.copy(), pi.copy(), lagtime)',
      'gold_call': '_oracle_compute_relaxation_response(op.copy(), pi.copy(), lagtime)',
      'tol': 2e-08},
     {'setup': '# Case: negative_fast_mode\n'
               'import numpy as np\n'
               'P=np.array([[.86,.08,.06],[.08,.82,.1],[.06,.1,.84]])\n'
               'dP=np.array([[-.03,.01,.02],[.01,-.005,-.005],[.02,-.005,-.015]])\n'
               'pi=np.ones(3)/3;lagtime=.4\n'
               'P=np.array([[.05,.85,.1],[.85,.05,.1],[.1,.1,.8]]);op=np.stack([P,dP])\n',
      'call': 'compute_relaxation_response(op.copy(), pi.copy(), lagtime)',
      'gold_call': '_oracle_compute_relaxation_response(op.copy(), pi.copy(), lagtime)',
      'tol': 2e-08},
     {'setup': '# Case: near_unit_slow_mode\n'
               'import numpy as np\n'
               'P=np.array([[.86,.08,.06],[.08,.82,.1],[.06,.1,.84]])\n'
               'dP=np.array([[-.03,.01,.02],[.01,-.005,-.005],[.02,-.005,-.015]])\n'
               'pi=np.ones(3)/3;lagtime=.4\n'
               'P=np.eye(3)+1e-4*(P-np.eye(3));dP*=1e-4;op=np.stack([P,dP])\n',
      'call': 'compute_relaxation_response(op.copy(), pi.copy(), lagtime)',
      'gold_call': '_oracle_compute_relaxation_response(op.copy(), pi.copy(), lagtime)',
      'tol': 2e-06},
     {'setup': '# Case: nearby_simple_slow_modes\n'
               'import numpy as np\n'
               'P=np.array([[.86,.08,.06],[.08,.82,.1],[.06,.1,.84]])\n'
               'dP=np.array([[-.03,.01,.02],[.01,-.005,-.005],[.02,-.005,-.015]])\n'
               'pi=np.ones(3)/3;lagtime=.4\n'
               'P=np.full((3,3),.04);np.fill_diagonal(P,.92);P[0,1]+=.0002;P[1,0]+=.0002;P[0,0]-=.0002;P[1,1]-=.0002;op=np.stack([P,dP])\n',
      'call': 'compute_relaxation_response(op.copy(), pi.copy(), lagtime)',
      'gold_call': '_oracle_compute_relaxation_response(op.copy(), pi.copy(), lagtime)',
      'tol': 2e-08},
     {'setup': '# Case: changing_equilibrium_similarity\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               'dX=np.diag([.1,-.06,-.04]);dpi=dX.sum(axis=1);P=X/pi[:,None];dP=dX/pi[:,None]-P*(dpi/pi)[:,None];op=np.stack([P,dP]);lagtime=.4\n',
      'call': 'compute_relaxation_response(op.copy(), pi.copy(), lagtime)',
      'gold_call': '_oracle_compute_relaxation_response(op.copy(), pi.copy(), lagtime)',
      'tol': 2e-08},
     {'setup': '# Case: frozen_kinetic_operator\n'
               'import numpy as np\n'
               'P=np.array([[.86,.08,.06],[.08,.82,.1],[.06,.1,.84]])\n'
               'dP=np.array([[-.03,.01,.02],[.01,-.005,-.005],[.02,-.005,-.015]])\n'
               'pi=np.ones(3)/3;lagtime=.4\n'
               'dP[:]=0.;op=np.stack([P,dP])\n',
      'call': 'compute_relaxation_response(op.copy(), pi.copy(), lagtime)',
      'gold_call': '_oracle_compute_relaxation_response(op.copy(), pi.copy(), lagtime)',
      'tol': 2e-08},
     {'setup': '# Case: long_lag_dimensional_time\n'
               'import numpy as np\n'
               'P=np.array([[.86,.08,.06],[.08,.82,.1],[.06,.1,.84]])\n'
               'dP=np.array([[-.03,.01,.02],[.01,-.005,-.005],[.02,-.005,-.015]])\n'
               'pi=np.ones(3)/3;lagtime=.4\n'
               'lagtime=17.;P[0,2]+=.02;P[2,0]+=.02;P[0,0]-=.02;P[2,2]-=.02;op=np.stack([P,dP])\n',
      'call': 'compute_relaxation_response(op.copy(), pi.copy(), lagtime)',
      'gold_call': '_oracle_compute_relaxation_response(op.copy(), pi.copy(), lagtime)',
      'tol': 2e-08}]
