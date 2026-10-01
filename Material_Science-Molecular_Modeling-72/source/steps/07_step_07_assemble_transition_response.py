"""
Convert equilibrium flux and its response to the kinetic transition operator and its response.

The stationary flux is the joint probability of the beginning and ending microstate under equilibrium. The transition operator conditions it on the starting state. Perturbing that condition changes the normalization as well as the flux.

Returns
-------
operators : np.ndarray: Float shape (2,L,L): row-stochastic transition matrix P followed by its total alpha derivative dP. Row indices are origins and columns are destinations. Units are 1 and inverse energy.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_transition_response(X: "np.ndarray", dX: "np.ndarray", pi: "np.ndarray", dpi: "np.ndarray") -> "np.ndarray":
    """Convert equilibrium flux and its response to the kinetic transition operator and its response.

    Parameters
    ----------
    X : np.ndarray
        Nonnegative symmetric float (L,L) equilibrium flux with row sums pi.
    dX : np.ndarray
        Symmetric float (L,L) alpha derivative, with row sums dpi.
    pi : np.ndarray
        Positive normalized float (L,) equilibrium masses, L >= 1.
    dpi : np.ndarray
        Float (L,) derivative summing to zero.

    Returns
    -------
    operators : np.ndarray
        Float shape (2,L,L): row-stochastic transition matrix P followed
        by its total alpha derivative dP. Row indices are origins and
        columns are destinations. Units are 1 and inverse energy."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_assemble_transition_response(X: "np.ndarray", dX: "np.ndarray", pi: "np.ndarray", dpi: "np.ndarray") -> "np.ndarray":
    P=X/pi[:,None]
    dP=dX/pi[:,None]-P*(dpi/pi)[:,None]
    return np.stack([P,dP])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Independent scientific configurations, derived from the cited method."""
    return [{'setup': '# Case: coupled_normalization_response\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               'dX=np.array([[.015,-.005,.01],[-.005,-.02,.015],[.01,.015,-.035]])\n'
               'dpi=dX.sum(axis=1)\n',
      'call': 'assemble_transition_response(X.copy(), dX.copy(), pi.copy(), dpi.copy())',
      'gold_call': '_oracle_assemble_transition_response(X.copy(), dX.copy(), pi.copy(), dpi.copy())',
      'tol': 2e-08},
     {'setup': '# Case: fixed_population_edge_redistribution\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               'dX=np.array([[.015,-.005,.01],[-.005,-.02,.015],[.01,.015,-.035]])\n'
               'dpi=dX.sum(axis=1)\n'
               'dX=np.array([[-.02,.01,.01],[.01,-.03,.02],[.01,.02,-.03]]);dpi=dX.sum(axis=1)\n',
      'call': 'assemble_transition_response(X.copy(), dX.copy(), pi.copy(), dpi.copy())',
      'gold_call': '_oracle_assemble_transition_response(X.copy(), dX.copy(), pi.copy(), dpi.copy())',
      'tol': 2e-08},
     {'setup': '# Case: stationary_redistribution_diagonal_flux\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               'dX=np.array([[.015,-.005,.01],[-.005,-.02,.015],[.01,.015,-.035]])\n'
               'dpi=dX.sum(axis=1)\n'
               'dX=np.diag([.1,-.06,-.04]);dpi=dX.sum(axis=1)\n',
      'call': 'assemble_transition_response(X.copy(), dX.copy(), pi.copy(), dpi.copy())',
      'gold_call': '_oracle_assemble_transition_response(X.copy(), dX.copy(), pi.copy(), dpi.copy())',
      'tol': 2e-08},
     {'setup': '# Case: single_state_identity\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               'dX=np.array([[.015,-.005,.01],[-.005,-.02,.015],[.01,.015,-.035]])\n'
               'dpi=dX.sum(axis=1)\n'
               'pi=np.ones(1);X=np.ones((1,1));dX=np.zeros((1,1));dpi=np.zeros(1)\n',
      'call': 'assemble_transition_response(X.copy(), dX.copy(), pi.copy(), dpi.copy())',
      'gold_call': '_oracle_assemble_transition_response(X.copy(), dX.copy(), pi.copy(), dpi.copy())',
      'tol': 2e-08},
     {'setup': '# Case: rare_state_amplification\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               'pi=np.array([1e-5,.39999,.6]);X=np.array([[7e-6,2e-6,1e-6],[2e-6,.349988,.05],[1e-6,.05,.549999]])\n'
               'dX=np.diag([1e-6,.02,-.020001]);dpi=dX.sum(axis=1)\n',
      'call': 'assemble_transition_response(X.copy(), dX.copy(), pi.copy(), dpi.copy())',
      'gold_call': '_oracle_assemble_transition_response(X.copy(), dX.copy(), pi.copy(), dpi.copy())',
      'tol': 2e-08},
     {'setup': '# Case: zero_flux_edge\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               'X[0,2]=0.;X[2,0]=0.;X[0,0]+=.02;X[2,2]+=.02\n'
               'dX=np.diag([.02,-.01,-.01]);dpi=dX.sum(axis=1)\n',
      'call': 'assemble_transition_response(X.copy(), dX.copy(), pi.copy(), dpi.copy())',
      'gold_call': '_oracle_assemble_transition_response(X.copy(), dX.copy(), pi.copy(), dpi.copy())',
      'tol': 2e-08},
     {'setup': '# Case: almost_absorbing_microstates\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               'dX=np.array([[.015,-.005,.01],[-.005,-.02,.015],[.01,.015,-.035]])\n'
               'dpi=dX.sum(axis=1)\n'
               'X=np.diag(pi)-1e-7*np.array([[2.,-1.,-1.],[-1.,2.,-1.],[-1.,-1.,2.]]);dX*=1e-5;dpi=dX.sum(axis=1)\n',
      'call': 'assemble_transition_response(X.copy(), dX.copy(), pi.copy(), dpi.copy())',
      'gold_call': '_oracle_assemble_transition_response(X.copy(), dX.copy(), pi.copy(), dpi.copy())',
      'tol': 2e-08},
     {'setup': '# Case: equal_stationary_masses\n'
               'import numpy as np\n'
               'pi=np.array([.21,.34,.45])\n'
               'X=np.array([[.16,.03,.02],[.03,.27,.04],[.02,.04,.39]])\n'
               'lam=np.array([2.,.7,1.4])\n'
               'split=np.array([[.5,.2,.8],[.8,.5,.35],[.2,.65,.5]])\n'
               'dX=np.array([[.015,-.005,.01],[-.005,-.02,.015],[.01,.015,-.035]])\n'
               'dpi=dX.sum(axis=1)\n'
               'pi=np.ones(3)/3;X=np.full((3,3),.03);np.fill_diagonal(X,pi-.06)\n',
      'call': 'assemble_transition_response(X.copy(), dX.copy(), pi.copy(), dpi.copy())',
      'gold_call': '_oracle_assemble_transition_response(X.copy(), dX.copy(), pi.copy(), dpi.copy())',
      'tol': 2e-08}]
