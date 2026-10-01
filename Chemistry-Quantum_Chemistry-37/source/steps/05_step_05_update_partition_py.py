"""
Implement update_partition. Effective eigenvalues define the new reference Hamiltonian. The physical Hamiltonian transforms in a parameter-dependent basis, and reference energies outside the selected space are retained.

Effective eigenvalues define the new reference Hamiltonian. The physical Hamiltonian transforms in a parameter-dependent basis, and reference energies outside the selected space are retained.

Returns
-------
Tuple (H_new,e_new), with shapes (4,n,n) and (4,n), the transformed partition jets.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def update_partition(H: np.ndarray, e: np.ndarray, P: np.ndarray,
                             L: np.ndarray, V: np.ndarray, frozen: int = 0
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a Hamiltonian jet and update its reference partition.

    Parameters
    ----------
    H, e : np.ndarray
        Finite real jets of shapes (4,n,n) and (4,n), n>=1. Each H slot
        is symmetric to atol=1e-12. Slots are (value,a,b,ab).
    P : np.ndarray
        Nonempty strictly increasing integer indices, all in [frozen,n).
    L, V : np.ndarray
        Finite real eigenvalue and eigenvector jets of shapes (4,p) and
        (4,p,p), p=len(P). V must be orthogonal through mixed order:
        the four jet slots of V.T@V equal (I,0,0,0) to atol=1e-9.
        Embed V on P and identity on its complement, transform H, replace
        e[:,P] with L, and retain every other reference energy unchanged.
    frozen : int
        Integer 0<=frozen<n (not bool). Preserve the prefix of this length;
        stably sort the remaining basis by updated baseline reference energy.

    Returns
    -------
    result : tuple[np.ndarray,np.ndarray]
        Transformed Hamiltonian jet (4,n,n) and reference jet (4,n), with
        one consistent permutation applied to every derivative slot.

    Raises
    ------
    ValueError
        If arrays are not convertible to finite real arrays of the stated
        shapes, H lacks the stated symmetry, P or frozen is invalid,
        V fails the stated derivative orthogonality test, or outputs
        contain nonfinite entries.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_update_partition(H: np.ndarray, e: np.ndarray, P: np.ndarray,
                             L: np.ndarray, V: np.ndarray, frozen: int = 0
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a Hamiltonian jet and update its reference partition.

    Parameters
    ----------
    H, e : np.ndarray
        Finite real jets of shapes (4,n,n) and (4,n), n>=1. Each H slot
        is symmetric to atol=1e-12. Slots are (value,a,b,ab).
    P : np.ndarray
        Nonempty strictly increasing integer indices, all in [frozen,n).
    L, V : np.ndarray
        Finite real eigenvalue and eigenvector jets of shapes (4,p) and
        (4,p,p), p=len(P). V must be orthogonal through mixed order:
        the four jet slots of V.T@V equal (I,0,0,0) to atol=1e-9.
        Embed V on P and identity on its complement, transform H, replace
        e[:,P] with L, and retain every other reference energy unchanged.
    frozen : int
        Integer 0<=frozen<n (not bool). Preserve the prefix of this length;
        stably sort the remaining basis by updated baseline reference energy.

    Returns
    -------
    result : tuple[np.ndarray,np.ndarray]
        Transformed Hamiltonian jet (4,n,n) and reference jet (4,n), with
        one consistent permutation applied to every derivative slot.

    Raises
    ------
    ValueError
        If arrays are not convertible to finite real arrays of the stated
        shapes, H lacks the stated symmetry, P or frozen is invalid,
        V fails the stated derivative orthogonality test, or outputs
        contain nonfinite entries.
    """
    try:
        if any(np.iscomplexobj(x) for x in (H,e,P,L,V)):
            raise ValueError('complex input')
        H,e,L,V = (np.asarray(x,float) for x in (H,e,L,V))
        P = np.asarray(P)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid arrays') from exc
    if H.ndim != 3 or H.shape[0] != 4 or H.shape[1] < 1 or H.shape[1] != H.shape[2]:
        raise ValueError('invalid H shape')
    n = H.shape[1]
    if isinstance(frozen,(bool,np.bool_)) or not isinstance(frozen,(int,np.integer)) or not 0<=frozen<n:
        raise ValueError('invalid frozen prefix')
    if P.ndim != 1 or P.size == 0 or P.dtype.kind not in 'iu' or np.any(P<frozen) or np.any(P>=n) or np.any(np.diff(P.astype(int))<=0):
        raise ValueError('invalid P')
    p = len(P)
    if e.shape != (4,n) or L.shape != (4,p) or V.shape != (4,p,p):
        raise ValueError('incompatible jet shapes')
    if not all(np.isfinite(x).all() for x in (H,e,L,V)) or not np.allclose(H,H.transpose(0,2,1),atol=1e-12,rtol=0):
        raise ValueError('nonfinite or asymmetric jet')
    C = np.zeros((4,4,4))
    for r,s,t in [(0,0,0),(1,1,0),(1,0,1),(2,2,0),(2,0,2),
                  (3,3,0),(3,1,2),(3,2,1),(3,0,3)]:
        C[r,s,t] = 1
    gram = np.einsum('rst,sji,tjk->rik',C,V,V)
    expected = np.zeros_like(gram); expected[0] = np.eye(p)
    if not np.allclose(gram,expected,atol=1e-9,rtol=0):
        raise ValueError('V is not an orthogonal jet')
    R = np.zeros_like(H); R[0] = np.eye(n)
    for i,ii in enumerate(P):
        for j,jj in enumerate(P):
            R[:,ii,jj] = V[:,i,j]
    HR = np.einsum('rst,sij,tjk->rik',C,H,R)
    transformed = np.einsum('rst,sji,tjk->rik',C,R,HR)
    transformed = (transformed+transformed.transpose(0,2,1))/2
    updated = e.copy(); updated[:,P] = L
    order = np.r_[np.arange(frozen), frozen+np.argsort(updated[0,frozen:],kind='stable')]
    transformed,updated = transformed[:,order,:][:,:,order],updated[:,order]
    if not np.isfinite(transformed).all() or not np.isfinite(updated).all():
        raise ValueError('nonfinite updated partition')
    return transformed,updated

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
H = np.zeros((4, 3, 3))
H[0] = [[-2, 0.1, 0.2], [0.1, 4, 0.3], [0.2, 0.3, 6]]
e = np.zeros((4, 3))
e[0] = [-2, 4, 6]
P = np.array([1, 2])
frozen = 1
L = np.zeros((4, 2))
L[0] = [1, 3]
L[1] = [0.2, -0.1]
V = np.zeros((4, 2, 2))
V[0] = np.eye(2)
V[1] = [[0, -1], [1, 0]]
V[2] = V[1]
V[3] = -np.eye(2)


def run(fn):
    result = fn(H.copy(), e.copy(), P.copy(), L.copy(), V.copy(), frozen)

    assert isinstance(result, tuple) and len(result) == 2
    n = H.shape[1]
    for value, shape in zip(result, ((4, n, n), (4, n))):
        assert isinstance(value, np.ndarray) and value.shape == shape
    return np.concatenate([value.ravel() for value in result])
""",
            "call": 'run(update_partition)',
            "gold_call": 'run(_oracle_update_partition)',
        },
        {
            "setup": """import numpy as np
H = np.array([2.0, 0.3, -0.2, 0.5]).reshape(4, 1, 1)
e = H[:, :, 0].copy()
P = np.array([0])
L = e.copy()
V = np.zeros((4, 1, 1))
V[0] = 1
frozen = 0


def run(fn):
    result = fn(H.copy(), e.copy(), P.copy(), L.copy(), V.copy(), frozen)

    assert isinstance(result, tuple) and len(result) == 2
    n = H.shape[1]
    for value, shape in zip(result, ((4, n, n), (4, n))):
        assert isinstance(value, np.ndarray) and value.shape == shape
    return np.concatenate([value.ravel() for value in result])
""",
            "call": 'run(update_partition)',
            "gold_call": 'run(_oracle_update_partition)',
        },
        {
            "setup": """import numpy as np
H = np.zeros((4, 3, 3))
H[0] = np.diag([-5, 2, 4])
e = np.zeros((4, 3))
e[0] = [-5, 2, 4]
P = np.array([2])
L = np.zeros((4, 1))
L[0] = -1
V = np.zeros((4, 1, 1))
V[0] = 1
frozen = 1


def run(fn):
    result = fn(H.copy(), e.copy(), P.copy(), L.copy(), V.copy(), frozen)

    assert isinstance(result, tuple) and len(result) == 2
    n = H.shape[1]
    for value, shape in zip(result, ((4, n, n), (4, n))):
        assert isinstance(value, np.ndarray) and value.shape == shape
    return np.concatenate([value.ravel() for value in result])
""",
            "call": 'run(update_partition)',
            "gold_call": 'run(_oracle_update_partition)',
        },
        {
            "setup": """import numpy as np
H = np.array([2.0, 0.3, -0.2, 0.5]).reshape(4, 1, 1)
e = H[:, :, 0].copy()
P = np.array([0])
L = e.copy()
V = np.zeros((4, 1, 1))
V[0] = 1
frozen = 0
V[1, 0, 0] = 1


def run(fn):
    try:
        fn(H.copy(), e.copy(), P.copy(), L.copy(), V.copy(), frozen)
    except ValueError:
        return 1
    return 0
""",
            "call": 'run(update_partition)',
            "gold_call": 'run(_oracle_update_partition)',
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3707)
X = rng.normal(size=(4, 5, 5))
H = 0.1 * (X + X.transpose(0, 2, 1))
H[0] += np.diag([-5.0, -0.2, 2.0, 4.0, 6.0])
e = 0.1 * rng.normal(size=(4, 5))
e[0] = [-6.0, 0.0, 1.0, 3.0, 7.0]
P = np.array([1, 3, 4])
L = 0.1 * rng.normal(size=(4, 3))
L[0] = [2.0, -2.0, 5.0]
X = rng.normal(size=(3, 3))
Q, _ = np.linalg.qr(X)
X = rng.normal(size=(3, 3))
S = 0.1 * (X - X.T)
X = rng.normal(size=(3, 3))
T = 0.1 * (X - X.T)
X = rng.normal(size=(3, 3))
U = 0.1 * (X - X.T)
V = np.array([Q, Q @ S, Q @ T, Q @ (S @ T + U)])


def run(fn):
    result = fn(H.copy(), e.copy(), P.copy(), L.copy(), V.copy(), frozen=1)

    assert isinstance(result, tuple) and len(result) == 2
    n = H.shape[1]
    for value, shape in zip(result, ((4, n, n), (4, n))):
        assert isinstance(value, np.ndarray) and value.shape == shape
    return np.concatenate([value.ravel() for value in result])
""",
            "call": 'run(update_partition)',
            "gold_call": 'run(_oracle_update_partition)',
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3708)
X = rng.normal(size=(4, 4, 4))
H = 0.05 * (X + X.transpose(0, 2, 1))
e = np.array([[9.0, 0.0, 2.0, 4.0], [0.1, 0.2, 0.3, 0.4], [0.2, -0.1, 0.4, -0.3], [0.3, 0.4, 0.5, 0.6]])
P = np.array([1, 3])
L = np.array([[2.0, 2.0], [0.7, 0.8], [0.9, 1.0], [1.1, 1.2]])
V = np.zeros((4, 2, 2))
V[0] = np.array([[0.8, -0.6], [0.6, 0.8]])


def run(fn):
    result = fn(H.copy(), e.copy(), P.copy(), L.copy(), V.copy(), frozen=1)

    assert isinstance(result, tuple) and len(result) == 2
    n = H.shape[1]
    for value, shape in zip(result, ((4, n, n), (4, n))):
        assert isinstance(value, np.ndarray) and value.shape == shape
    return np.concatenate([value.ravel() for value in result])
""",
            "call": 'run(update_partition)',
            "gold_call": 'run(_oracle_update_partition)',
        },
    ]
