"""
Complete the stationary covariance and integrate the embedded noise.

The reduced covariance and shared noise factor determine a stationary continuous process. Exact noise integration preserves its equilibrium covariance at finite time steps.

Returns
-------
Tuple (Sigma,T,Q): real (q,q), (j,q,q), (j,q,q) arrays. Sigma     and Q are covariances of the dimensionless state. For j=0, the     last two arrays retain shape (0,q,q).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def complete_discrete_embedding(A: "np.ndarray", dimension: int, X: "np.ndarray", S1: "np.ndarray", L1: "np.ndarray", K1: "np.ndarray", intervals: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Complete the stationary covariance and integrate the embedded noise.

    Parameters
    ----------
    A : finite real stable drift (q,q), q>=2*d
    dimension : integer d>=1
    X : finite real invertible auxiliary transformation (q-d,q-d)
    S1, L1, K1 : finite real arrays (q-2*d,q-2*d),(q-2*d,d),(d,d)
        The covariance and noise factors from the reduced Lur'e problem.
    intervals : finite real array (j,), j>=0, each entry in (0,3]
        Time increments between successive held-out measurements.

    Contract
    --------
    Return the stationary covariance in the original state coordinates
    and exact finite-interval transition and process-noise covariances.
    Keep all coupled noise contributions. Sigma is symmetric within
    1e-7 and has minimum eigenvalue >1e-10; drift eigenvalue real parts
    are below -1e-10. The continuous stationarity residual has Frobenius
    norm at most 1e-6*max(1,||Sigma||_F). Return symmetric process-noise
    covariances with minimum eigenvalue no smaller than -1e-8.
    Exact covariance identities or exact integration are acceptable;
    no diagonal regularizer is part of the model.

    Returns
    -------
    result
        Tuple (Sigma,T,Q): real (q,q), (j,q,q), (j,q,q) arrays. Sigma
        and Q are covariances of the dimensionless state. For j=0, the
        last two arrays retain shape (0,q,q).

    Raises
    ------
    ValueError
        Complex/nonfinite input, incompatible shapes, invalid dimension
        or intervals, singular X, nonpositive/nonsymmetric covariance,
        unstable A, failed stationary Lyapunov identity, or a transition
        covariance with minimum eigenvalue below -1e-8.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_complete_discrete_embedding(A: "np.ndarray", dimension: int, X: "np.ndarray", S1: "np.ndarray", L1: "np.ndarray", K1: "np.ndarray", intervals: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Complete the stationary covariance and integrate the embedded noise.

    Parameters
    ----------
    A : finite real stable drift (q,q), q>=2*d
    dimension : integer d>=1
    X : finite real invertible auxiliary transformation (q-d,q-d)
    S1, L1, K1 : finite real arrays (q-2*d,q-2*d),(q-2*d,d),(d,d)
        The covariance and noise factors from the reduced Lur'e problem.
    intervals : finite real array (j,), j>=0, each entry in (0,3]
        Time increments between successive held-out measurements.

    Contract
    --------
    Return the stationary covariance in the original state coordinates
    and exact finite-interval transition and process-noise covariances.
    Keep all coupled noise contributions. Sigma is symmetric within
    1e-7 and has minimum eigenvalue >1e-10; drift eigenvalue real parts
    are below -1e-10. The continuous stationarity residual has Frobenius
    norm at most 1e-6*max(1,||Sigma||_F). Return symmetric process-noise
    covariances with minimum eigenvalue no smaller than -1e-8.
    Exact covariance identities or exact integration are acceptable;
    no diagonal regularizer is part of the model.

    Returns
    -------
    result
        Tuple (Sigma,T,Q): real (q,q), (j,q,q), (j,q,q) arrays. Sigma
        and Q are covariances of the dimensionless state. For j=0, the
        last two arrays retain shape (0,q,q).

    Raises
    ------
    ValueError
        Complex/nonfinite input, incompatible shapes, invalid dimension
        or intervals, singular X, nonpositive/nonsymmetric covariance,
        unstable A, failed stationary Lyapunov identity, or a transition
        covariance with minimum eigenvalue below -1e-8.
    """
    import numpy as np
    from scipy.linalg import block_diag, expm
    if any((np.iscomplexobj(x) for x in [A, X, S1, L1, K1, intervals])):
        raise ValueError('real input')
    A, X, S1, L1, K1, dt = [np.asarray(x, float) for x in [A, X, S1, L1, K1, intervals]]
    if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, (int, np.integer)) or dimension < 1:
        raise ValueError('dimension')
    d = dimension
    if A.ndim != 2 or A.shape[0] != A.shape[1] or len(A) < 2 * d:
        raise ValueError('drift')
    q = len(A)
    a = q - d
    h = q - 2 * d
    if X.shape != (a, a) or S1.shape != (h, h) or L1.shape != (h, d) or (K1.shape != (d, d)) or (dt.ndim != 1) or (not all((np.isfinite(x).all() for x in [A, X, S1, L1, K1, dt]))) or np.any(dt <= 0) or np.any(dt > 3):
        raise ValueError('input shapes or intervals')
    try:
        Xi = np.linalg.solve(X, np.eye(a))
    except np.linalg.LinAlgError as e:
        raise ValueError('singular X') from e
    Sigma = block_diag(np.eye(d), Xi @ block_diag(np.eye(d), S1) @ Xi.T)
    noise = np.vstack([np.zeros((d, d)), Xi @ np.vstack([K1, L1])])
    if np.max(abs(Sigma - Sigma.T)) > 1e-07 or np.min(np.linalg.eigvalsh((Sigma + Sigma.T) / 2)) <= 1e-10 or np.max(np.linalg.eigvals(A).real) >= -1e-10:
        raise ValueError('stationary covariance')
    Sigma = (Sigma + Sigma.T) / 2
    N = noise @ noise.T
    if np.linalg.norm(A @ Sigma + Sigma @ A.T + N) > 1e-06 * max(1.0, np.linalg.norm(Sigma)):
        raise ValueError('stationary noise')
    aug = np.block([[A, N], [np.zeros_like(A), -A.T]])
    transitions = np.empty((len(dt), q, q))
    covariances = np.empty_like(transitions)
    for k, interval in enumerate(dt):
        V = expm(interval * aug)
        T = V[:q, :q]
        Q = V[:q, q:] @ T.T
        Q = (Q + Q.T) / 2
        if np.min(np.linalg.eigvalsh(Q)) < -1e-08:
            raise ValueError('noise covariance')
        transitions[k] = T
        covariances[k] = Q
    return (Sigma, transitions, covariances)

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

def discrete_check(result, A, X, S, intervals, d):
    Sigma, T, Q = result
    Xi = np.linalg.inv(X)
    expected = block_diag(np.eye(d), Xi @ block_diag(np.eye(d), S) @ Xi.T)
    if near(Sigma, expected) != 1 or T.shape != (len(intervals), len(A), len(A)) or Q.shape != T.shape:
        return 0
    for k, h in enumerate(intervals):
        F = expm(h * A)
        if near(T[k], F) != 1 or near(Q[k], expected - F @ expected @ F.T, atol=1e-08, rtol=1e-08) != 1:
            return 0
    return 1
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
dt = np.array([0.11, 0.5, 1.2])
""",
            'call': 'discrete_check(complete_discrete_embedding(copy.deepcopy(A), copy.deepcopy(d), copy.deepcopy(X), copy.deepcopy(S), copy.deepcopy(L), copy.deepcopy(K), copy.deepcopy(dt)), A, X, S, dt, d)',
            'gold_call': 'discrete_check(_oracle_complete_discrete_embedding(copy.deepcopy(A), copy.deepcopy(d), copy.deepcopy(X), copy.deepcopy(S), copy.deepcopy(L), copy.deepcopy(K), copy.deepcopy(dt)), A, X, S, dt, d)',
        },
        {
            'setup': common_0 + """d = 1
A, X, S, L, K, A1, B1, C1, D1 = noise_fixture(d, 1)
dt = np.array([0.11, 0.5, 1.2])
""",
            'call': 'discrete_check(complete_discrete_embedding(copy.deepcopy(A), copy.deepcopy(d), copy.deepcopy(X), copy.deepcopy(S), copy.deepcopy(L), copy.deepcopy(K), copy.deepcopy(dt)), A, X, S, dt, d)',
            'gold_call': 'discrete_check(_oracle_complete_discrete_embedding(copy.deepcopy(A), copy.deepcopy(d), copy.deepcopy(X), copy.deepcopy(S), copy.deepcopy(L), copy.deepcopy(K), copy.deepcopy(dt)), A, X, S, dt, d)',
        },
        {
            'setup': common_0 + """d = 2
A, X, S, L, K, A1, B1, C1, D1 = noise_fixture(d, 0)
dt = np.array([0.11, 0.5, 1.2])
""",
            'call': 'discrete_check(complete_discrete_embedding(copy.deepcopy(A), copy.deepcopy(d), copy.deepcopy(X), copy.deepcopy(S), copy.deepcopy(L), copy.deepcopy(K), copy.deepcopy(dt)), A, X, S, dt, d)',
            'gold_call': 'discrete_check(_oracle_complete_discrete_embedding(copy.deepcopy(A), copy.deepcopy(d), copy.deepcopy(X), copy.deepcopy(S), copy.deepcopy(L), copy.deepcopy(K), copy.deepcopy(dt)), A, X, S, dt, d)',
        },
        {
            'setup': common_0 + """d = 2
A, X, S, L, K, A1, B1, C1, D1 = noise_fixture()
dt = np.array([])
""",
            'call': 'discrete_check(complete_discrete_embedding(copy.deepcopy(A), copy.deepcopy(d), copy.deepcopy(X), copy.deepcopy(S), copy.deepcopy(L), copy.deepcopy(K), copy.deepcopy(dt)), A, X, S, dt, d)',
            'gold_call': 'discrete_check(_oracle_complete_discrete_embedding(copy.deepcopy(A), copy.deepcopy(d), copy.deepcopy(X), copy.deepcopy(S), copy.deepcopy(L), copy.deepcopy(K), copy.deepcopy(dt)), A, X, S, dt, d)',
        },
        {
            'setup': common_1,
            'call': 'error_code(complete_discrete_embedding, copy.deepcopy(A), copy.deepcopy(d), copy.deepcopy(X), copy.deepcopy(S), copy.deepcopy(L), copy.deepcopy(K), copy.deepcopy(np.array([0.0])))',
            'gold_call': 'error_code(_oracle_complete_discrete_embedding, copy.deepcopy(A), copy.deepcopy(d), copy.deepcopy(X), copy.deepcopy(S), copy.deepcopy(L), copy.deepcopy(K), copy.deepcopy(np.array([0.0])))',
        },
        {
            'setup': common_1,
            'call': 'error_code(complete_discrete_embedding, copy.deepcopy(A), copy.deepcopy(d), copy.deepcopy(X), copy.deepcopy(S), copy.deepcopy(L), copy.deepcopy(2 * K), copy.deepcopy(np.array([0.1])))',
            'gold_call': 'error_code(_oracle_complete_discrete_embedding, copy.deepcopy(A), copy.deepcopy(d), copy.deepcopy(X), copy.deepcopy(S), copy.deepcopy(L), copy.deepcopy(2 * K), copy.deepcopy(np.array([0.1])))',
        },
    ]
