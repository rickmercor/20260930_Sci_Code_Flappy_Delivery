"""
Advance the PSD factor for nuclear-norm-regularized sensing.

The unconstrained factor objective is the PSD formulation in the landscape theorem. Its real-coordinate gradient and a deterministic Armijo search define the finite iterate.

Returns
-------
(d, r) complex array, the factor after one regularized Armijo update.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def regularized_backtrack_step(F: "np.ndarray", A_ops: "np.ndarray", y: "np.ndarray", lam: float) -> "np.ndarray":
    """Take one real-coordinate gradient step on regularized factored sensing.

    Parameters
    ----------
    F : np.ndarray
        Complex factor of shape (d,r).
    A_ops : np.ndarray
        Hermitian measurement operators of shape (m,d,d).
    y : np.ndarray
        Real measurement vector of shape (m,).
    lam : float
        Nonnegative coefficient of the squared Frobenius norm of F in
        one-half squared measurement residual plus lam times that norm.

    Returns
    -------
    result : np.ndarray
        Unconstrained factor after one Armijo step. Use the real-coordinate
        Euclidean gradient on real and imaginary parts of F, sufficient-decrease
        constant 1e-4, initial trial step 1, and repeated halving. Return the first
        accepted trial; do not normalize F.

    Raises
    ------
    ValueError
        If lam is negative or A_ops and y have different lengths.
    FloatingPointError
        If the line-search step underflows before acceptance.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _regularized_loss_gradient(F, A_ops, y, lam):
    rho=F@F.conj().T
    residual=np.einsum('kij,ji->k',A_ops,rho).real-y
    loss=0.5*np.dot(residual,residual)+lam*np.vdot(F,F).real
    adjoint=np.einsum('k,kij->ij',residual,A_ops)
    return float(loss),2.0*(adjoint@F+lam*F)

def _oracle_regularized_backtrack_step(F: "np.ndarray", A_ops: "np.ndarray", y: "np.ndarray", lam: float) -> "np.ndarray":
    if lam<0 or len(A_ops)!=len(y):
        raise ValueError('nonnegative lam and matching measurement lengths required')
    value,grad=_regularized_loss_gradient(F,A_ops,y,lam)
    norm2=np.vdot(grad,grad).real
    step=1.0
    while True:
        trial=F-step*grad
        trial_value,_=_regularized_loss_gradient(trial,A_ops,y,lam)
        if trial_value<=value-1e-4*step*norm2:
            return trial
        step*=0.5
        if step==0.0:
            raise FloatingPointError('Armijo trial step underflowed')

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return separate normal, boundary, and edge comparison cases."""
    return [{'setup': 'import numpy as np\n'
               'A_ops=np.array([[[1,0],[0,0]],[[0,0],[0,2]],[[0,1j],[-1j,0]]],complex)\n'
               'F=np.array([[0.8+0.2j],[0.3-0.4j]]); y=np.array([0.7,0.2,0.1]); lam=0.012',
      'call': 'regularized_backtrack_step(F.copy(),A_ops.copy(),y.copy(),lam)',
      'gold_call': '_oracle_regularized_backtrack_step(F.copy(),A_ops.copy(),y.copy(),lam)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'A_ops=np.array([[[1,0],[0,0]],[[0,0],[0,2]],[[0,1j],[-1j,0]]],complex)\n'
               'F=np.zeros((2,1),complex); y=np.array([0.7,0.2,0.1]); lam=0.3',
      'call': 'regularized_backtrack_step(F.copy(),A_ops.copy(),y.copy(),lam)',
      'gold_call': '_oracle_regularized_backtrack_step(F.copy(),A_ops.copy(),y.copy(),lam)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'A_ops=np.array([[[1,0],[0,0]],[[0,0],[0,2]],[[0,1j],[-1j,0]]],complex)\n'
               'F=np.array([[2.0+1j],[1.5-0.5j]]); y=np.array([0.1,0.2,-0.2]); lam=0.0',
      'call': 'regularized_backtrack_step(F.copy(),A_ops.copy(),y.copy(),lam)',
      'gold_call': '_oracle_regularized_backtrack_step(F.copy(),A_ops.copy(),y.copy(),lam)',
      'tol': 1e-09}]
