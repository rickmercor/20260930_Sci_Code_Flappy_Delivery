"""
Compute measurement geometry for the low-rank landscape theorem.

The upper isometry condition only concerns the target rank stratum tangent space; a lower bound on all Hermitian directions also controls admissible recovery errors.

Returns
-------
(3,) real array, full lower, tangent upper, and full upper squared-norm constants.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sensing_isometry_constants(A_ops: "np.ndarray", V_support: "np.ndarray") -> "np.ndarray":
    """Compute full-space lower and tangent-space upper squared-norm constants.

    Parameters
    ----------
    A_ops : np.ndarray
        Hermitian operators of shape (m,d,d), spanning the real Hermitian space.
    V_support : np.ndarray
        Orthonormal columns of shape (d,r_star), 1 <= r_star <= d, spanning
        the target support. The tangent space is that of rank-r_star Hermitian
        matrices at a PSD target with this support.

    Returns
    -------
    result : np.ndarray
        Real array [alpha_full,beta_tangent,beta_full], shape (3,).
        These are sharp extremal squared-norm ratios of the measurement map:
        minimum on all Hermitian matrices, maximum on the tangent space,
        and maximum on all Hermitian matrices. alpha_full always bounds
        restricted PSD recovery errors below; equality with the sharp
        restricted constant needs separate justification.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _canonical_hermitian_basis(d):
    basis=[]
    for i in range(d):
        B=np.zeros((d,d),complex); B[i,i]=1
        basis.append(B)
    for i in range(d):
        for j in range(i+1,d):
            B=np.zeros((d,d),complex); B[i,j]=B[j,i]=1/np.sqrt(2)
            basis.append(B)
            B=np.zeros((d,d),complex); B[i,j]=1j/np.sqrt(2); B[j,i]=-1j/np.sqrt(2)
            basis.append(B)
    return np.array(basis)

def _oracle_sensing_isometry_constants(A_ops: "np.ndarray", V_support: "np.ndarray") -> "np.ndarray":
    d,r_star=V_support.shape
    basis=_canonical_hermitian_basis(d)
    coefficients=np.einsum('aij,bji->ab',A_ops,basis).real
    eigenvalues=np.linalg.eigvalsh(coefficients.T@coefficients)
    Q,_=np.linalg.qr(V_support,mode='complete')
    tangent=[]
    for B in basis:
        if np.linalg.norm(B[r_star:,r_star:])<1e-14:
            tangent.append(Q@B@Q.conj().T)
    C=np.einsum('aij,bji->ab',A_ops,np.array(tangent)).real
    beta=np.linalg.eigvalsh(C.T@C)[-1]
    return np.array([eigenvalues[0],beta,eigenvalues[-1]],float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return separate normal, boundary, and edge comparison cases."""
    return [{'setup': 'import numpy as np\n'
               'B=np.array([[[1,0],[0,0]],[[0,0],[0,1]],[[0,1],[1,0]],[[0,1j],[-1j,0]]],complex)\n'
               'B[2:]/=np.sqrt(2)\n'
               'A=B*np.sqrt(np.array([1.0,9.0,1.1,1.2]))[:,None,None]; '
               'V=np.array([[1.0],[0.0]],complex)',
      'call': 'sensing_isometry_constants(A.copy(),V.copy())',
      'gold_call': '_oracle_sensing_isometry_constants(A.copy(),V.copy())',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'B=np.array([[[1,0],[0,0]],[[0,0],[0,1]],[[0,1],[1,0]],[[0,1j],[-1j,0]]],complex)\n'
               'B[2:]/=np.sqrt(2)\n'
               'A=B; V=np.eye(2,dtype=complex)',
      'call': 'sensing_isometry_constants(A.copy(),V.copy())',
      'gold_call': '_oracle_sensing_isometry_constants(A.copy(),V.copy())',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'B=np.array([[[1,0],[0,0]],[[0,0],[0,1]],[[0,1],[1,0]],[[0,1j],[-1j,0]]],complex)\n'
               'B[2:]/=np.sqrt(2)\n'
               'Q=np.array([[1,1j],[1j,1]],complex)/np.sqrt(2); A=np.array([Q@b@Q.conj().T for b in '
               'B])*np.sqrt(np.array([0.7,4.0,0.8,0.9]))[:,None,None]; V=Q[:,:1]',
      'call': 'sensing_isometry_constants(A.copy(),V.copy())',
      'gold_call': '_oracle_sensing_isometry_constants(A.copy(),V.copy())',
      'tol': 1e-09}]
