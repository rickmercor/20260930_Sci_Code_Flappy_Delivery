"""
Reduce a purely inertial embedding to a regular auxiliary noise problem.

Zero direct velocity damping makes the first Lur'e problem singular. Curvature-based deflation exposes a regular auxiliary problem.

Returns
-------
Tuple (A1,B1,C1,D1,X). With h=q-2*d and a=q-d, the shapes are     (h,h),(h,d),(h,d),(d,d),(a,a). All arrays are real; h may be zero.     Hidden bases may differ, provided the defining identities hold
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def singular_memory_reduction(A: "np.ndarray", dimension: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Reduce a purely inertial embedding to a regular auxiliary noise problem.

    Parameters
    ----------
    A : finite real drift array (q,q), q>=2*d
    dimension : integer d>=1
        Number of leading normalized velocity coordinates.

    Contract
    --------
    Return the singular-to-regular auxiliary reduction of an inertial
    realization. Partition A = [[0, B^T], [-C, A0]] with the first d
    coordinates the normalized velocities, so B = A[:d, d:].T,
    C = -A[d:, :d] and A0 = A[d:, d:]. The leading velocity block must be
    zero within 1e-7; all drift eigenvalues have real part below -1e-10.
    The induced zero-time memory S = B^T C must be symmetric within 1e-7
    and positive definite with minimum eigenvalue above 1e-10; let S^(1/2)
    be its symmetric positive square root. Build the deflation X, of size
    (q-d, q-d), by stacking S^(-1/2) B^T on top of the transpose of an
    orthonormal basis of the null space of C^T, so that X C = [S^(1/2); 0].
    Transform the auxiliary drift as V = X A0 X^(-1) and read the outputs
    from its blocks as V = [[-D1, B1^T], [-C1, A1]], with D1 the leading
    d x d block. The symmetric part D1+D1.T must have minimum eigenvalue
    >1e-10. Any orthonormal null-space basis is accepted; the result is
    checked through X C = [S^(1/2); 0], the first d rows of X equalling
    S^(-1/2) B^T, and X A0 = V X.

    Returns
    -------
    result
        Tuple (A1,B1,C1,D1,X). With h=q-2*d and a=q-d, the shapes are
        (h,h),(h,d),(h,d),(d,d),(a,a). All arrays are real; h may be zero.
        Hidden bases may differ, provided the defining identities hold.

    Raises
    ------
    ValueError
        Complex/nonfinite A, invalid square shape or dimension, q<2*d,
        nonzero velocity block, unstable A, nonsymmetric or nonpositive S,
        singular X, or nonpositive R1 under the stated tolerances.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_singular_memory_reduction(A: "np.ndarray", dimension: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Reduce a purely inertial embedding to a regular auxiliary noise problem.

    Parameters
    ----------
    A : finite real drift array (q,q), q>=2*d
    dimension : integer d>=1
        Number of leading normalized velocity coordinates.

    Contract
    --------
    Return the source's singular-to-regular auxiliary reduction of an
    inertial realization. The leading velocity block must be zero within
    1e-7; all drift eigenvalues have real part below -1e-10. The induced
    zero-time memory must be symmetric within 1e-7 and positive definite
    with minimum eigenvalue above 1e-10. Use its symmetric positive
    square root as the coordinate convention and an orthonormal basis
    for the remaining null space. The symmetric part D1+D1.T of the
    reduced direct-damping block must have minimum eigenvalue >1e-10.
    Equivalent null-space bases are accepted through reduction identities.

    Returns
    -------
    result
        Tuple (A1,B1,C1,D1,X). With h=q-2*d and a=q-d, the shapes are
        (h,h),(h,d),(h,d),(d,d),(a,a). All arrays are real; h may be zero.
        Hidden bases may differ, provided the defining identities hold.

    Raises
    ------
    ValueError
        Complex/nonfinite A, invalid square shape or dimension, q<2*d,
        nonzero velocity block, unstable A, nonsymmetric or nonpositive S,
        singular X, or nonpositive R1 under the stated tolerances.
    """
    import numpy as np
    from scipy.linalg import null_space
    if np.iscomplexobj(A):
        raise ValueError('real drift')
    A = np.asarray(A, float)
    if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, (int, np.integer)) or dimension < 1:
        raise ValueError('dimension')
    d = dimension
    if A.ndim != 2 or A.shape[0] != A.shape[1] or len(A) < 2 * d or (not np.isfinite(A).all()):
        raise ValueError('drift shape')
    if np.max(abs(A[:d, :d])) > 1e-07 or np.max(np.linalg.eigvals(A).real) >= -1e-10:
        raise ValueError('inertial stable drift')
    B = A[:d, d:].T
    C = -A[d:, :d]
    A0 = A[d:, d:]
    S = B.T @ C
    if np.max(abs(S - S.T)) > 1e-07:
        raise ValueError('curvature symmetry')
    s, U = np.linalg.eigh((S + S.T) / 2)
    if np.min(s) <= 1e-10:
        raise ValueError('curvature positivity')
    invroot = U / np.sqrt(s) @ U.T
    X = np.vstack([invroot @ B.T, null_space(C.T, rcond=1e-12).T])
    try:
        V = np.linalg.solve(X.T, (X @ A0).T).T
    except np.linalg.LinAlgError as e:
        raise ValueError('singular deflation') from e
    D = -V[:d, :d]
    if np.min(np.linalg.eigvalsh(D + D.T)) <= 1e-10:
        raise ValueError('regular damping')
    return (V[d:, d:], V[:d, d:].T, -V[d:, :d], D, X)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    common_0 = """import copy
import numpy as np
from scipy.linalg import expm, block_diag

def near(actual, expected, atol=2e-06, rtol=2e-06):
    a = np.asarray(actual)
    b = np.asarray(expected)
    return int(a.shape == b.shape and np.isfinite(a).all() and np.allclose(a, b, atol=atol, rtol=rtol))

def noise_fixture(d=2, h=2):
    D = np.diag(np.linspace(0.6, 1.1, d))
    K = np.linalg.cholesky(D + D.T)
    S = 0.3 * np.eye(h)
    B = 0.12 * np.eye(h, d)
    L = 0.22 * np.eye(h, d)
    A1 = -(L @ L.T) / 0.6
    if h > 1:
        J = np.zeros((h, h))
        J[0, 1] = 0.13
        J[1, 0] = -0.13
        A1 += J
    C = S @ B + L @ K.T
    V = np.block([[-D, B.T], [-C, A1]])
    X = np.eye(d + h) + 0.035 * np.arange((d + h) ** 2).reshape(d + h, d + h) / (d + h)
    E = np.eye(d + h)[:, :d]
    Xi = np.linalg.inv(X)
    A = np.block([[np.zeros((d, d)), E.T @ X], [-Xi @ E, Xi @ V @ X]])
    return (A, X, S, L, K, A1, B, C, D)

def reduction_check(result, A, d):
    A1, B1, C1, D1, X = result
    q = len(A)
    h = q - 2 * d
    if A1.shape != (h, h) or B1.shape != (h, d) or C1.shape != B1.shape or (D1.shape != (d, d)) or (X.shape != (q - d, q - d)):
        return 0
    B = A[:d, d:].T
    C = -A[d:, :d]
    S = B.T @ C
    eig, U = np.linalg.eigh((S + S.T) / 2)
    root = U * np.sqrt(eig) @ U.T
    E = np.eye(q - d)[:, :d]
    V = np.block([[-D1, B1.T], [-C1, A1]])
    return int(near(X @ C, E @ root) == 1 and near(X.T @ E, B @ np.linalg.inv(root)) == 1 and (near(X @ A[d:, d:], V @ X) == 1) and (np.min(np.linalg.eigvalsh(D1 + D1.T)) > 0))
'Deterministic synthetic probe data; not measurements reported in the article.'
"""
    common_1 = """import copy
import numpy as np
from scipy.linalg import expm, block_diag

def error_code(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0

def noise_fixture(d=2, h=2):
    D = np.diag(np.linspace(0.6, 1.1, d))
    K = np.linalg.cholesky(D + D.T)
    S = 0.3 * np.eye(h)
    B = 0.12 * np.eye(h, d)
    L = 0.22 * np.eye(h, d)
    A1 = -(L @ L.T) / 0.6
    if h > 1:
        J = np.zeros((h, h))
        J[0, 1] = 0.13
        J[1, 0] = -0.13
        A1 += J
    C = S @ B + L @ K.T
    V = np.block([[-D, B.T], [-C, A1]])
    X = np.eye(d + h) + 0.035 * np.arange((d + h) ** 2).reshape(d + h, d + h) / (d + h)
    E = np.eye(d + h)[:, :d]
    Xi = np.linalg.inv(X)
    A = np.block([[np.zeros((d, d)), E.T @ X], [-Xi @ E, Xi @ V @ X]])
    return (A, X, S, L, K, A1, B, C, D)
'Deterministic synthetic probe data; not measurements reported in the article.'
d = 2
A, X, S, L, K, A1, B1, C1, D1 = noise_fixture()
"""
    return [
        {
            'setup': common_0 + """d = 2
A, X, S, L, K, A1, B1, C1, D1 = noise_fixture(d, 2)
""",
            'call': 'reduction_check(singular_memory_reduction(copy.deepcopy(A), copy.deepcopy(d)), A, d)',
            'gold_call': 'reduction_check(_oracle_singular_memory_reduction(copy.deepcopy(A), copy.deepcopy(d)), A, d)',
        },
        {
            'setup': common_0 + """d = 1
A, X, S, L, K, A1, B1, C1, D1 = noise_fixture(d, 1)
""",
            'call': 'reduction_check(singular_memory_reduction(copy.deepcopy(A), copy.deepcopy(d)), A, d)',
            'gold_call': 'reduction_check(_oracle_singular_memory_reduction(copy.deepcopy(A), copy.deepcopy(d)), A, d)',
        },
        {
            'setup': common_0 + """d = 2
A, X, S, L, K, A1, B1, C1, D1 = noise_fixture(d, 0)
""",
            'call': 'reduction_check(singular_memory_reduction(copy.deepcopy(A), copy.deepcopy(d)), A, d)',
            'gold_call': 'reduction_check(_oracle_singular_memory_reduction(copy.deepcopy(A), copy.deepcopy(d)), A, d)',
        },
        {
            'setup': common_1 + """A[0, 0] = 0.1
""",
            'call': 'error_code(singular_memory_reduction, copy.deepcopy(A), copy.deepcopy(d))',
            'gold_call': 'error_code(_oracle_singular_memory_reduction, copy.deepcopy(A), copy.deepcopy(d))',
        },
        {
            'setup': common_1,
            'call': 'error_code(singular_memory_reduction, copy.deepcopy(A.astype(complex)), copy.deepcopy(d))',
            'gold_call': 'error_code(_oracle_singular_memory_reduction, copy.deepcopy(A.astype(complex)), copy.deepcopy(d))',
        },
        {
            'setup': common_0 + """d = 2
A, X, S, L, K, A1, B1, C1, D1 = noise_fixture(d, 2)
H = np.array([[1.4,.2],[.2,.9]])
A[:d,d:] = H @ A[:d,d:]
A[d:,:d] = A[d:,:d] @ H
""",
            'call': 'reduction_check(singular_memory_reduction(copy.deepcopy(A.copy()), copy.deepcopy(d)), A, d)',
            'gold_call': 'reduction_check(_oracle_singular_memory_reduction(copy.deepcopy(A.copy()), copy.deepcopy(d)), A, d)',
            'tol': 0.0,
        },
    ]
