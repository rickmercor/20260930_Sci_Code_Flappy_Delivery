"""
Construct the complete, symmetrically orthogonalized intrinsic atomic orbital basis from a fixed occupied reference and a nonorthogonal minimal basis.

IAOs polarize a minimal atomic basis using the occupied and depolarized occupied projectors. The original primary basis is orthonormal. Orthogonalizing the entire polarized basis before selecting a fragment preserves the requested atom-labeled orbital definition.

Returns
-------
np.ndarray, shape (n,m), the complete symmetrically orthogonalized IAO coefficient matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def intrinsic_orbitals(F: "np.ndarray", M: "np.ndarray", nocc: int) -> "np.ndarray":
    """Construct the original polarized IAOs in the orthonormal primary basis.

    Parameters
    ----------
    F : np.ndarray
        Real symmetric (n,n) fixed reference matrix, n >= 2.
    M : np.ndarray
        Real (n,m) minimal-basis coefficients, 1 <= m <= n.
    nocc : int
        Number of occupied spatial reference orbitals: 1 <= nocc < n,
        nocc <= m. Use the lowest eigenvectors of F. The reference is fixed,
        not a fractional-occupation projector.

    Returns
    -------
    Q : np.ndarray
        (n,m) original polarized IAOs, symmetrically orthogonalized together
        in the supplied column ordering. Use positive symmetric inverse
        square roots, not an arbitrary QR basis.

    Raises
    ------
    ValueError
        If data are not convertible to real finite float arrays, shapes or
        nocc violate the above bounds (bool is not an integer here), F differs
        from F.T by more than 1e-10 in any entry, or its occupied/unoccupied
        gap is <= 1e-10. Also if any eigenvalue of the minimal-basis Gram
        matrix, projected-occupied Gram matrix, or polarized-orbital Gram
        matrix is <= 1e-12. Accepted near-symmetric F is symmetrized.
    
    """
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_intrinsic_orbitals(F: "np.ndarray", M: "np.ndarray", nocc: int) -> "np.ndarray":
    """Deterministic reference implementation."""
    try:
        if np.iscomplexobj(F) or np.iscomplexobj(M): raise ValueError('real inputs required')
        F=np.asarray(F,dtype=float); M=np.asarray(M,dtype=float)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError('real arrays required') from exc
    if F.ndim!=2 or F.shape[0]!=F.shape[1] or F.shape[0]<2 or M.ndim!=2:
        raise ValueError('shape')
    n=F.shape[0]
    if M.shape[0]!=n or not 1<=M.shape[1]<=n or not np.isfinite(F).all() or not np.isfinite(M).all(): raise ValueError('shape or finiteness')
    if isinstance(nocc,(bool,np.bool_)) or not isinstance(nocc,(int,np.integer)) or not 1<=nocc< n or nocc>M.shape[1]: raise ValueError('nocc')
    if np.max(np.abs(F-F.T))>1e-10: raise ValueError('symmetry')
    e,W=np.linalg.eigh((F+F.T)/2)
    if e[nocc]-e[nocc-1]<=1e-10: raise ValueError('reference gap')
    C=W[:,:nocc]; S=M.T@M
    if np.linalg.eigvalsh(S).min()<=1e-12: raise ValueError('minimal rank')
    X=M@np.linalg.solve(S,M.T@C)
    d,V=np.linalg.eigh(X.T@X)
    if d.min()<=1e-12: raise ValueError('projected occupied rank')
    Ct=X@((V/np.sqrt(d))@V.T)
    O=C@C.T; Ot=Ct@Ct.T; I=np.eye(n)
    R=(O@Ot+(I-O)@(I-Ot))@M
    d,V=np.linalg.eigh(R.T@R)
    if d.min()<=1e-12: raise ValueError('polarized rank')
    return R@((V/np.sqrt(d))@V.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base = """import numpy as np
F = np.array([
    [-1.6,-.45,.12,0,.18,-.08],[-.45,-.8,-.35,.16,0,.11],
    [.12,-.35,-.15,-.4,.14,0],[0,.16,-.4,.35,-.3,.17],
    [.18,0,.14,-.3,.9,-.25],[-.08,.11,0,.17,-.25,1.4]])
M = np.array([[1,.12,0,0],[.08,1,.1,0],[0,.06,1,.09],
              [.04,0,.08,1],[.3,-.2,.25,.1],[-.15,.28,.05,.32]])
U = np.array([2.4,1.7,2.1,1.3,1.9,1.5])
"""
    s1 = base
    s2 = """import numpy as np
F = np.diag([-2.0, 1.0])
M = np.array([[1.0], [0.5]])
"""
    s3 = base + """
def _exception_code(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
"""
    return [
        {"setup": s1,
         "call": 'intrinsic_orbitals(F.copy(),M.copy(),2)',
         "gold_call": '_oracle_intrinsic_orbitals(F.copy(),M.copy(),2)'},
        {"setup": s2,
         "call": 'intrinsic_orbitals(F.copy(), M.copy(), 1)',
         "gold_call": '_oracle_intrinsic_orbitals(F.copy(), M.copy(), 1)'},
        {"setup": s1,
         "call": 'intrinsic_orbitals(F.copy(),M.copy() @ np.diag([1.,.4,2.,.7]),2)',
         "gold_call": '_oracle_intrinsic_orbitals(F.copy(),M.copy() @ np.diag([1.,.4,2.,.7]),2)'},
        {"setup": s3,
         "call": '_exception_code(intrinsic_orbitals, np.diag([-1.,-1.,2.]),np.eye(3),1)',
         "gold_call": '_exception_code(_oracle_intrinsic_orbitals, np.diag([-1.,-1.,2.]),np.eye(3),1)'},
        {"setup": s3,
         "call": '_exception_code(intrinsic_orbitals, F.copy(),np.column_stack([M.copy()[:,0],M.copy()[:,0]]),1)',
         "gold_call": '_exception_code(_oracle_intrinsic_orbitals, F.copy(),np.column_stack([M.copy()[:,0],M.copy()[:,0]]),1)'},
    ]
