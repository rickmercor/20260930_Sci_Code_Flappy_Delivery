"""
Compute the auxiliary normal response and modified-Hessian spectral data.

The auxiliary response in Eqs. 19–21 and Appendix A controls the constrained determinant. Differentiating stationarity determines the signed constraint susceptibility. The finite-dimensional tests also cover the Appendix A identity for tridiagonal Hessians with different inertias.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def normal_response(jet: "np.ndarray", phi: "np.ndarray", geometry: "np.ndarray") -> "np.ndarray":
    """Compute the auxiliary normal response and modified-Hessian spectral data.

    jet : ndarray, shape (3+2*N,)
        Action-jet layout; only its last N entries, the diagonal of H, are used.
    phi : ndarray, shape (N,)
        Dimensionless fields; at least one entry is nonzero.
    geometry : ndarray, shape (N,4)
        Columns are radius r, positive cell weight w, outward edge c, and stiffness diagonal Ljj; rows run from origin outward. All entries are dimensionless.
    Returns
    -------
    ndarray, length 5
        [negative_count(H), nu, dxi_dK, log(abs(det(H))), dot(z,z)], where z=grad_y(xi), H*psi=z, nu=dot(z,psi)/dot(z,z), and dxi_dK is the stationary-family susceptibility. H is nonsingular with eigenvalues separated from zero by at least 1e-8. Synthetic tridiagonal fluctuation jets in the tests obey the same algebraic contract.

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

def _oracle_normal_response(jet: "np.ndarray", phi: "np.ndarray", geometry: "np.ndarray") -> "np.ndarray":
    jet=np.asarray(jet,dtype=float);phi=np.asarray(phi,dtype=float);geometry=np.asarray(geometry,dtype=float)
    if phi.ndim!=1 or len(phi)<2 or geometry.shape!=(len(phi),4) or jet.shape!=(3+2*len(phi),) or not np.all(np.isfinite(jet)) or not np.all(np.isfinite(phi)) or not np.all(np.isfinite(geometry)) or np.any(geometry[:,1]<=0) or not np.any(phi):
        raise ValueError("Invalid normal-response inputs")
    n=len(phi);w=geometry[:,1];c=geometry[:,2]
    diagonal=jet[3+n:]
    off=-c[:-1]/np.sqrt(w[:-1]*w[1:])
    band=np.zeros((3,n));band[1]=diagonal;band[0,1:]=off;band[2,:-1]=off
    zeta=3*np.sqrt(w)*np.asarray(phi)**2
    psi=solve_banded((1,1),band,zeta)
    norm=zeta@zeta;numerator=zeta@psi
    values=eigvalsh_tridiagonal(diagonal,off,lapack_driver='stebz')
    if np.min(abs(values))<1e-8:raise ValueError('Full modified Hessian must be nonsingular')
    return np.array([np.sum(values<0),numerator/norm,-numerator,
                     np.sum(np.log(abs(values))),norm])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Constructed scientific cases, not published numerical examples."""
    return [{'setup': 'import numpy as np\n'
               'jet = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 4.0], dtype=float)\n'
               'phi = np.array([1.0, 2.0, 1.0], dtype=float)\n'
               'geometry = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]], '
               'dtype=float)\n',
      'call': 'normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'gold_call': '_oracle_normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'jet = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 3.0, 5.0], dtype=float)\n'
               'phi = np.array([0.2, 2.0, 2.0], dtype=float)\n'
               'geometry = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]], '
               'dtype=float)\n',
      'call': 'normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'gold_call': '_oracle_normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'jet = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 3.0, 5.0], dtype=float)\n'
               'phi = np.array([2.0, 0.2, 0.2], dtype=float)\n'
               'geometry = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]], '
               'dtype=float)\n',
      'call': 'normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'gold_call': '_oracle_normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'jet = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, -1.0, 4.0], dtype=float)\n'
               'phi = np.array([1.0, 2.0, 1.0], dtype=float)\n'
               'geometry = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]], '
               'dtype=float)\n',
      'call': 'normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'gold_call': '_oracle_normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'jet = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, -2.0, -1.0], dtype=float)\n'
               'phi = np.array([1.0, 1.0, 1.0], dtype=float)\n'
               'geometry = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]], '
               'dtype=float)\n',
      'call': 'normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'gold_call': '_oracle_normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'jet = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0], dtype=float)\n'
               'phi = np.array([1.0, 2.0, 3.0], dtype=float)\n'
               'geometry = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]], '
               'dtype=float)\n',
      'call': 'normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'gold_call': '_oracle_normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'jet = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 4.0], dtype=float)\n'
               'phi = np.array([1.0, 2.0, 1.0], dtype=float)\n'
               'geometry = np.array([[0.0, 1.0, 0.3, 0.0], [1.0, 1.0, 0.4, 0.0], [2.0, 1.0, 0.0, 0.0]], '
               'dtype=float)\n',
      'call': 'normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'gold_call': '_oracle_normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'jet = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 2.0, 4.0], dtype=float)\n'
               'phi = np.array([1.0, 1.0, 2.0], dtype=float)\n'
               'geometry = np.array([[0.0, 1.0, 0.5, 0.0], [1.0, 1.0, 0.4, 0.0], [2.0, 1.0, 0.0, 0.0]], '
               'dtype=float)\n',
      'call': 'normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'gold_call': '_oracle_normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'jet = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 2.0, 4.0], dtype=float)\n'
               'phi = np.array([1.0, 1.0, 2.0], dtype=float)\n'
               'geometry = np.array([[0.0, 0.1, 0.22360679774997896, 0.0], [1.0, 2.0, 1.4966629547095767, 0.0], '
               '[2.0, 7.0, 0.0, 0.0]], dtype=float)\n',
      'call': 'normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'gold_call': '_oracle_normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'jet = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 3.0, 5.0], dtype=float)\n'
               'phi = np.array([1.0, 0.0, 0.0], dtype=float)\n'
               'geometry = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]], '
               'dtype=float)\n',
      'call': 'normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'gold_call': '_oracle_normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'jet = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 3.0, 5.0], dtype=float)\n'
               'phi = np.array([0.0, 1.0, 0.0], dtype=float)\n'
               'geometry = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]], '
               'dtype=float)\n',
      'call': 'normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'gold_call': '_oracle_normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'jet = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0001, 2.0, 4.0], dtype=float)\n'
               'phi = np.array([0.02, 1.0, 1.0], dtype=float)\n'
               'geometry = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]], '
               'dtype=float)\n',
      'call': 'normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'gold_call': '_oracle_normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'jet = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0001, 4.0], dtype=float)\n'
               'phi = np.array([1.0, 0.1, 1.0], dtype=float)\n'
               'geometry = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]], '
               'dtype=float)\n',
      'call': 'normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'gold_call': '_oracle_normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'jet = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 2.0, -3.0], dtype=float)\n'
               'phi = np.array([2.0, 1.0, 0.2], dtype=float)\n'
               'geometry = np.array([[0.0, 1.0, 0.3, 0.0], [1.0, 1.0, 0.4, 0.0], [2.0, 1.0, 0.0, 0.0]], '
               'dtype=float)\n',
      'call': 'normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'gold_call': '_oracle_normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'jet = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 3.0], dtype=float)\n'
               'phi = np.array([1.0, 1.0002, 0.0], dtype=float)\n'
               'geometry = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]], '
               'dtype=float)\n',
      'call': 'normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'gold_call': '_oracle_normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'jet = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 3.0], dtype=float)\n'
               'phi = np.array([1.0002, 1.0, 0.0], dtype=float)\n'
               'geometry = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]], '
               'dtype=float)\n',
      'call': 'normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'gold_call': '_oracle_normal_response(jet.copy(), phi.copy(), geometry.copy())',
      'tol': 2e-06}]
