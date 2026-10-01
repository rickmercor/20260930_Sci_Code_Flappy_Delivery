"""
Implement eigensystem_jet. Nondegenerate eigenvectors respond to perturbations as well as eigenvalues. Their mixed normalization term is necessary for a valid orthogonal basis jet.

Nondegenerate eigenvectors respond to perturbations as well as eigenvalues. Their mixed normalization term is necessary for a valid orthogonal basis jet.

Returns
-------
Tuple (L,V), with shapes (4,n) and (4,n,n), the eigenvalue and eigenvector jets.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def eigensystem_jet(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate a simple symmetric eigensystem through mixed order.

    Parameters
    ----------
    A : np.ndarray
        Finite real symmetric jet (4,n,n), n>=1, in (value,a,b,ab) slots.
        Every slot must be symmetric to atol=1e-12, rtol=0. Baseline
        eigenvalues must have pairwise separations >1e-10. Eigenvectors
        are columns ordered by increasing eigenvalue. Fix each column's
        sign by making its largest-absolute baseline entry positive,
        choosing the lowest row index if tied. Derivatives use a smooth
        local continuation of this sign, not differentiation of argmax.

    Returns
    -------
    result : tuple[np.ndarray,np.ndarray]
        Eigenvalue jet (4,n) and normalized eigenvector jet (4,n,n).
        Include eigenvector response and mixed normalization terms.

    Raises
    ------
    ValueError
        If input is not convertible to a finite real jet of the stated
        shape and symmetry, baseline eigenvalues are separated by <=1e-10,
        the eigensolver fails, or the calculated result is nonfinite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_eigensystem_jet(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate a simple symmetric eigensystem through mixed order.

    Parameters
    ----------
    A : np.ndarray
        Finite real symmetric jet (4,n,n), n>=1, in (value,a,b,ab) slots.
        Every slot must be symmetric to atol=1e-12, rtol=0. Baseline
        eigenvalues must have pairwise separations >1e-10. Eigenvectors
        are columns ordered by increasing eigenvalue. Fix each column's
        sign by making its largest-absolute baseline entry positive,
        choosing the lowest row index if tied. Derivatives use a smooth
        local continuation of this sign, not differentiation of argmax.

    Returns
    -------
    result : tuple[np.ndarray,np.ndarray]
        Eigenvalue jet (4,n) and normalized eigenvector jet (4,n,n).
        Include eigenvector response and mixed normalization terms.

    Raises
    ------
    ValueError
        If input is not convertible to a finite real jet of the stated
        shape and symmetry, baseline eigenvalues are separated by <=1e-10,
        the eigensolver fails, or the calculated result is nonfinite.
    """
    try:
        if np.iscomplexobj(A):
            raise ValueError('complex input')
        A = np.asarray(A,float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid A') from exc
    if A.ndim != 3 or A.shape[0] != 4 or A.shape[1] < 1 or A.shape[1] != A.shape[2]:
        raise ValueError('invalid shape')
    if not np.isfinite(A).all() or not np.allclose(A,A.transpose(0,2,1),atol=1e-12,rtol=0):
        raise ValueError('nonfinite or asymmetric jet')
    try:
        lam,V = np.linalg.eigh(A[0])
    except np.linalg.LinAlgError as exc:
        raise ValueError('eigensolver failed') from exc
    if np.any(np.diff(lam)<=1e-10):
        raise ValueError('spectrum not simple')
    n = len(lam)
    for i in range(n):
        if V[np.argmax(np.abs(V[:,i])),i] < 0:
            V[:,i] *= -1
    L,R = np.zeros((4,n)), np.zeros((4,n,n))
    L[0],R[0] = lam,V
    for i in range(n):
        v = V[:,i]
        for d in (1,2):
            L[d,i] = v@A[d]@v
            for j in range(n):
                if j != i:
                    R[d,:,i] += V[:,j]*(V[:,j]@A[d]@v)/(lam[i]-lam[j])
        va,vb = R[1,:,i],R[2,:,i]
        L[3,i] = v@A[3]@v + va@A[2]@v + vb@A[1]@v
        z = A[3]@v + (A[1]-L[1,i]*np.eye(n))@vb + (A[2]-L[2,i]*np.eye(n))@va
        R[3,:,i] = -(va@vb)*v
        for j in range(n):
            if j != i:
                R[3,:,i] += V[:,j]*(V[:,j]@z)/(lam[i]-lam[j])
    if not np.isfinite(L).all() or not np.isfinite(R).all():
        raise ValueError('nonfinite eigensystem jet')
    return L,R

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
A = np.array([[[1, 0.3], [0.3, 2]], [[0.2, 0], [0, -0.4]], [[0.1, 0.2], [0.2, -0.3]], [[0.05, 0.1], [0.1, 0.02]]])


def run(fn):
    result = fn(A.copy())

    assert isinstance(result, tuple) and len(result) == 2
    n = A.shape[1]
    for value, shape in zip(result, ((4, n), (4, n, n))):
        assert isinstance(value, np.ndarray) and value.shape == shape
    return np.concatenate([value.ravel() for value in result])
""",
            "call": 'run(eigensystem_jet)',
            "gold_call": 'run(_oracle_eigensystem_jet)',
        },
        {
            "setup": """import numpy as np
A = np.array([2.0, 0.3, -0.2, 0.5]).reshape(4, 1, 1)


def run(fn):
    result = fn(A.copy())

    assert isinstance(result, tuple) and len(result) == 2
    n = A.shape[1]
    for value, shape in zip(result, ((4, n), (4, n, n))):
        assert isinstance(value, np.ndarray) and value.shape == shape
    return np.concatenate([value.ravel() for value in result])
""",
            "call": 'run(eigensystem_jet)',
            "gold_call": 'run(_oracle_eigensystem_jet)',
        },
        {
            "setup": """import numpy as np
A = np.zeros((4, 2, 2))
A[0] = np.diag([0, 0.001])
A[1] = [[0, 0.2], [0.2, 0.1]]
A[2] = [[0.1, -0.3], [-0.3, 0]]


def run(fn):
    result = fn(A.copy())

    assert isinstance(result, tuple) and len(result) == 2
    n = A.shape[1]
    for value, shape in zip(result, ((4, n), (4, n, n))):
        assert isinstance(value, np.ndarray) and value.shape == shape
    return np.concatenate([value.ravel() for value in result])
""",
            "call": 'run(eigensystem_jet)',
            "gold_call": 'run(_oracle_eigensystem_jet)',
        },
        {
            "setup": """import numpy as np
A = np.zeros((4, 2, 2))


def run(fn):
    try:
        fn(A.copy())
    except ValueError:
        return 1
    return 0
""",
            "call": 'run(eigensystem_jet)',
            "gold_call": 'run(_oracle_eigensystem_jet)',
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3705)
Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
X = rng.normal(size=(4, 5, 5))
A = 0.06 * (X + X.transpose(0, 2, 1))
A[0] = Q @ np.diag([-2.0, -0.3, 1.1, 2.4, 4.0]) @ Q.T


def run(fn):
    result = fn(A.copy())

    assert isinstance(result, tuple) and len(result) == 2
    n = A.shape[1]
    for value, shape in zip(result, ((4, n), (4, n, n))):
        assert isinstance(value, np.ndarray) and value.shape == shape
    return np.concatenate([value.ravel() for value in result])
""",
            "call": 'run(eigensystem_jet)',
            "gold_call": 'run(_oracle_eigensystem_jet)',
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3706)
Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
X = rng.normal(size=(4, 4, 4))
A = 0.06 * (X + X.transpose(0, 2, 1))
A[0] = Q @ np.diag([-1.0, 0.15, 0.17, 2.0]) @ Q.T


def run(fn):
    result = fn(A.copy())

    assert isinstance(result, tuple) and len(result) == 2
    n = A.shape[1]
    for value, shape in zip(result, ((4, n), (4, n, n))):
        assert isinstance(value, np.ndarray) and value.shape == shape
    return np.concatenate([value.ravel() for value in result])
""",
            "call": 'run(eigensystem_jet)',
            "gold_call": 'run(_oracle_eigensystem_jet)',
        },
    ]
