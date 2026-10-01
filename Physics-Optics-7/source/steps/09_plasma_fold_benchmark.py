"""
Compose every preceding public function to select the fold truncation and its error certificate.

The final implementation must call and use all eight preceding public functions: fold_frame, gaussian_jet, fold_powers, fold_contractions, saddle_series, gaussian_diffraction, coherent_hybrid, and error_profile. Use the unique minimum image specified by the main problem for the regular contribution, with stationary-phase corrections through inverse-frequency order 2. The regular ray can be obtained from its stationary equation on q=alpha*exp(-x_r^T*A*x_r/2)<1/lambda_max(A).

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def plasma_fold_benchmark(precision: "np.ndarray", point: "np.ndarray", frequencies: "np.ndarray", unfoldings: "np.ndarray", weights: "np.ndarray", max_order: int = 6) -> "np.ndarray":
    """Compose every preceding public function to select the fold truncation and its error certificate.

    precision : ndarray, shape (2,2), float
        Symmetric positive definite inverse covariance A, as in fold_frame.
    point : ndarray, shape (2,), float
        Simple fold point x_c. The constructed alpha lies in (0,3].
    frequencies : ndarray, shape (F,), float
        Distinct trial frequencies in [10,180], kept in supplied order.
    unfoldings : ndarray, shape (Z,), float
        Normalized source offsets zeta. All specified source positions admit
        the unique regular minimum image on the stated q branch.
    weights : ndarray, shape (Z,), float
        Nonnegative weights with positive sum and positive reference norms.
    max_order : int, default 6
        Largest retained fold correction index, 1 through 6.
    Returns
    -------
    ndarray, shape (6,), float
        [best error percent, selected integer order, worst listed frequency,
        leading-order worst error percent, max_order worst error percent,
        runner-up minus best error in percentage points]. Minimize worst error
        across frequencies; ties select the smaller order, and ties in the
        maximizing frequency select its first supplied position. All trial
        quantities are dimensionless. The first entry is the requested scalar.
        The public implementation composes and uses every preceding public API.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from decimal import Decimal, localcontext
from math import factorial, prod
from scipy.optimize import brentq
from scipy.special import airy

def _oracle_plasma_fold_benchmark(precision: "np.ndarray", point: "np.ndarray", frequencies: "np.ndarray", unfoldings: "np.ndarray", weights: "np.ndarray", max_order: int = 6) -> "np.ndarray":
    A = np.asarray(precision, dtype=float)
    xc = np.asarray(point, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    unfoldings = np.asarray(unfoldings, dtype=float)
    frame = _oracle_fold_frame(A, xc)
    alpha, yc, V = frame[0], frame[1:3], frame[3:7].reshape(2,2)
    lam, scale = frame[7:9]
    C = _oracle_gaussian_jet(A, alpha, xc, V, max_order+3)
    P = _oracle_fold_powers(C, scale, lam, 2*max_order)
    eigA, U = np.linalg.eigh(A)
    approx = np.zeros((len(frequencies),len(unfoldings),max_order+1),complex)
    exact = np.zeros((len(frequencies),len(unfoldings)),complex)
    for i, w in enumerate(frequencies):
        for j, z in enumerate(unfoldings):
            y = yc - z/(scale*w**(2/3))*V[:,0]
            u = _oracle_fold_contractions(P,z,lam)
            yy = U.T@y
            def _ray_equation(q):
                return np.log(q/alpha)+0.5*np.sum(eigA*yy**2/(1-q*eigA)**2)
            upper = min(alpha, (1-1e-13)/eigA[-1])
            q = brentq(_ray_equation, np.finfo(float).tiny, upper, xtol=5e-15, rtol=1e-14)
            xr = np.linalg.solve(np.eye(2)-q*A,y)
            H = np.eye(2)+q*(np.outer(A@xr,A@xr)-A)
            ev, Vr = np.linalg.eigh(H)
            regular_jet = _oracle_gaussian_jet(A,alpha,xr,Vr,6)
            regular_coeff = _oracle_saddle_series(regular_jet,ev,2)
            Tc = 0.5*np.sum((xc-y)**2)+C[0,0]
            Tr = 0.5*np.sum((xr-y)**2)+q
            approx[i,j] = _oracle_coherent_hybrid(u,regular_coeff,Tc,Tr,scale,lam,ev,w)
            xy = _oracle_gaussian_diffraction(A,alpha,y,w)
            exact[i,j] = xy[0]+1j*xy[1]
    profile = _oracle_error_profile(approx,exact,weights)
    worst = np.max(profile,axis=0)
    ranking = np.argsort(worst,kind='stable')
    winner = int(ranking[0])
    second = int(ranking[1])
    frequency_index = int(np.argmax(profile[:,winner]))
    # [best percent error, retained correction order, worst frequency, leading percent,
    # maximum-order percent, runner-up percent gap]
    return np.array([100*worst[winner],winner,frequencies[frequency_index],100*worst[0],
                     100*worst[-1],100*(worst[second]-worst[winner])])

def _integration_expected():
    """Independent real-plane quadrature certificate; evaluator-only literal."""
    return np.array([12.545986320161948, 3.0, 50.0, 27.65538538343503, 15.143256888428361, 1.56621074465505],dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Eight deterministic, scientifically distinct constructed cases."""
    return [{'setup': 'import numpy as np\n'
               'A=np.array([[1, 0.18], [0.18, 0.58]]);x=np.array([0.55, 0.28]);ws=np.array([20, '
               '32]);zs=np.array([-1.4, -0.5, 0.25, 1.0]);wt=np.array([1, 2, 2, 1])',
      'call': 'plasma_fold_benchmark(A.copy(),x.copy(),ws.copy(),zs.copy(),wt.copy(),6)',
      'gold_call': '_oracle_plasma_fold_benchmark(A.copy(),x.copy(),ws.copy(),zs.copy(),wt.copy(),6)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'A=np.array([[1, 0.18], [0.18, 0.58]]);x=np.array([0.55, 0.28]);ws=np.array([40, '
               '80]);zs=np.array([-1, 0, 1]);wt=np.array([1, 2, 1])',
      'call': 'plasma_fold_benchmark(A.copy(),x.copy(),ws.copy(),zs.copy(),wt.copy(),6)',
      'gold_call': '_oracle_plasma_fold_benchmark(A.copy(),x.copy(),ws.copy(),zs.copy(),wt.copy(),6)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'A=np.array([[1, 0.18], [0.18, 0.58]]);x=np.array([0.55, 0.28]);ws=np.array([100, '
               '160]);zs=np.array([-0.8, 0.2, 0.9]);wt=np.array([1, 1, 1])',
      'call': 'plasma_fold_benchmark(A.copy(),x.copy(),ws.copy(),zs.copy(),wt.copy(),6)',
      'gold_call': '_oracle_plasma_fold_benchmark(A.copy(),x.copy(),ws.copy(),zs.copy(),wt.copy(),6)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'A=np.array([[1, 0.12], [0.12, 0.5]]);x=np.array([0.5, 0.32]);ws=np.array([35, '
               '60]);zs=np.array([-0.9, 0.2, 0.8]);wt=np.array([1, 2, 1])',
      'call': 'plasma_fold_benchmark(A.copy(),x.copy(),ws.copy(),zs.copy(),wt.copy(),4)',
      'gold_call': '_oracle_plasma_fold_benchmark(A.copy(),x.copy(),ws.copy(),zs.copy(),wt.copy(),4)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'A=np.array([[1, 0], [0, 0.35]]);x=np.array([0.45, 0.1]);ws=np.array([30, 50]);zs=np.array([-0.7, '
               '0.3, 1]);wt=np.array([1, 1, 2])',
      'call': 'plasma_fold_benchmark(A.copy(),x.copy(),ws.copy(),zs.copy(),wt.copy(),3)',
      'gold_call': '_oracle_plasma_fold_benchmark(A.copy(),x.copy(),ws.copy(),zs.copy(),wt.copy(),3)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'A=np.array([[1, 0.18], [0.18, 0.58]]);x=np.array([0.55, 0.28]);ws=np.array([50, 40, '
               '80]);zs=np.array([-1.2, -0.4, 0.3, 1.1]);wt=np.array([1, 2, 2, 1])',
      'call': 'plasma_fold_benchmark(A.copy(),x.copy(),ws.copy(),zs.copy(),wt.copy(),5)',
      'gold_call': '_oracle_plasma_fold_benchmark(A.copy(),x.copy(),ws.copy(),zs.copy(),wt.copy(),5)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'A=np.array([[1, 0.18], [0.18, 0.58]]);x=np.array([0.55, 0.28]);ws=np.array([45, '
               '75]);zs=np.array([-1.2, 0, 0.8]);wt=np.array([0, 2, 1])',
      'call': 'plasma_fold_benchmark(A.copy(),x.copy(),ws.copy(),zs.copy(),wt.copy(),4)',
      'gold_call': '_oracle_plasma_fold_benchmark(A.copy(),x.copy(),ws.copy(),zs.copy(),wt.copy(),4)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'A=np.array([[1, 0.18], [0.18, 0.58]]);x=np.array([0.55, 0.28]);ws=np.array([65, '
               '95]);zs=np.array([-1.1, -0.2, 0.4, 0.9]);wt=np.array([3, 1, 1, 2])',
      'call': 'plasma_fold_benchmark(A.copy(),x.copy(),ws.copy(),zs.copy(),wt.copy(),2)',
      'gold_call': '_oracle_plasma_fold_benchmark(A.copy(),x.copy(),ws.copy(),zs.copy(),wt.copy(),2)',
      'tol': 2e-06}]
