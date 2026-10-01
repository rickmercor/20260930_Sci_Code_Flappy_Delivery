"""
Solve the stabilizing regular Lur'e covariance completion.

Covariance completion imposes both stationarity and compatibility between colored and white noise. The stabilizing Riccati branch is required.

Returns
-------
Tuple (S,L,K): real arrays (h,h),(h,d),(d,d). S is dimensionless;     L,K have inverse-square-root-time units in the rescaled state.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def regular_lure(A0: "np.ndarray", B: "np.ndarray", C: "np.ndarray", D: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Solve the stabilizing regular Lur'e covariance completion.

    Parameters
    ----------
    A0 : finite real array (h,h), h>=0
    B, C : finite real arrays (h,d), d>=1
    D : finite real array (d,d)

    Contract
    --------
    Return the positive stationary covariance and shared-noise factors
    of the source's regular auxiliary problem, selecting its stabilizing
    covariance branch. The symmetric direct-damping part must have
    minimum eigenvalue >1e-10. K uses the lower-Cholesky convention.
    Stability means eigenvalue real parts below -1e-10. Reject Hamiltonian
    eigenvalues within 1e-10 of the imaginary axis, covariance minimum
    eigenvalue <=1e-10, or Riccati Frobenius residual greater than
    1e-7*max(1,||S||_F). For h=0 return empty S and L, together with K.

    Returns
    -------
    result
        Tuple (S,L,K): real arrays (h,h),(h,d),(d,d). S is dimensionless;
        L,K have inverse-square-root-time units in the rescaled state.

    Raises
    ------
    ValueError
        Complex/nonfinite input, incompatible shapes, nonpositive R,
        missing stable h-dimensional graph, singular U, imaginary-axis
        Hamiltonian spectrum, nonpositive covariance, unstable completed
        Riccati branch, or excessive residual under the stated thresholds.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_regular_lure(A0: "np.ndarray", B: "np.ndarray", C: "np.ndarray", D: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Solve the stabilizing regular Lur'e covariance completion.

    Parameters
    ----------
    A0 : finite real array (h,h), h>=0
    B, C : finite real arrays (h,d), d>=1
    D : finite real array (d,d)

    Contract
    --------
    Return the positive stationary covariance and shared-noise factors
    of the source's regular auxiliary problem, selecting its stabilizing
    covariance branch. The symmetric direct-damping part must have
    minimum eigenvalue >1e-10. K uses the lower-Cholesky convention.
    Stability means eigenvalue real parts below -1e-10. Reject Hamiltonian
    eigenvalues within 1e-10 of the imaginary axis, covariance minimum
    eigenvalue <=1e-10, or Riccati Frobenius residual greater than
    1e-7*max(1,||S||_F). For h=0 return empty S and L, together with K.

    Returns
    -------
    result
        Tuple (S,L,K): real arrays (h,h),(h,d),(d,d). S is dimensionless;
        L,K have inverse-square-root-time units in the rescaled state.

    Raises
    ------
    ValueError
        Complex/nonfinite input, incompatible shapes, nonpositive R,
        missing stable h-dimensional graph, singular U, imaginary-axis
        Hamiltonian spectrum, nonpositive covariance, unstable completed
        Riccati branch, or excessive residual under the stated thresholds.
    """
    import numpy as np
    from scipy.linalg import schur
    if any((np.iscomplexobj(x) for x in [A0, B, C, D])):
        raise ValueError('real matrices')
    A0, B, C, D = [np.asarray(x, float) for x in [A0, B, C, D]]
    if A0.ndim != 2 or A0.shape[0] != A0.shape[1] or D.ndim != 2 or (D.shape[0] < 1) or (D.shape[0] != D.shape[1]) or (B.shape != (len(A0), len(D))) or (C.shape != B.shape) or (not all((np.isfinite(x).all() for x in [A0, B, C, D]))):
        raise ValueError('shapes')
    R = D + D.T
    if np.min(np.linalg.eigvalsh(R)) <= 1e-10:
        raise ValueError('R positivity')
    K = np.linalg.cholesky(R)
    h = len(A0)
    if h == 0:
        return (np.empty((0, 0)), np.empty((0, len(D))), K)
    P = A0 - C @ np.linalg.solve(R, B.T)
    G = B @ np.linalg.solve(R, B.T)
    Q = C @ np.linalg.solve(R, C.T)
    H = np.block([[P.T, G], [-Q, -P]])
    if np.min(abs(np.linalg.eigvals(H).real)) <= 1e-10:
        raise ValueError('imaginary axis')
    _, U, count = schur(H, output='real', sort=lambda a, b: a < 0)
    if count != h:
        raise ValueError('stable graph dimension')
    try:
        S = np.linalg.solve(U[:h, :h].T, U[h:, :h].T).T
    except np.linalg.LinAlgError as e:
        raise ValueError('singular graph') from e
    S = (S + S.T) / 2
    if np.min(np.linalg.eigvalsh(S)) <= 1e-10 or np.max(np.linalg.eigvals(P + S @ G).real) >= -1e-10:
        raise ValueError('Riccati branch')
    if np.linalg.norm(P @ S + S @ P.T + S @ G @ S + Q) > 1e-07 * max(1.0, np.linalg.norm(S)):
        raise ValueError('Riccati residual')
    L = np.linalg.solve(K, (C - S @ B).T).T
    return (S, L, K)

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

def lure_check(result, A0, B, C, D, expected):
    S, L, K = result
    return int(near(S, expected, atol=1e-07, rtol=1e-07) == 1 and near(K, np.linalg.cholesky(D + D.T)) == 1 and (near(S @ B + L @ K.T, C) == 1) and (near(A0 @ S + S @ A0.T + L @ L.T, np.zeros_like(S)) == 1))
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
A, X, S, L, K, A1, B1, C1, D1 = noise_fixture()
"""
    return [
        {
            'setup': common_0 + """d = 2
A, X, S, L, K, A1, B1, C1, D1 = noise_fixture(d, 2)
""",
            'call': 'lure_check(regular_lure(copy.deepcopy(A1), copy.deepcopy(B1), copy.deepcopy(C1), copy.deepcopy(D1)), A1, B1, C1, D1, S)',
            'gold_call': 'lure_check(_oracle_regular_lure(copy.deepcopy(A1), copy.deepcopy(B1), copy.deepcopy(C1), copy.deepcopy(D1)), A1, B1, C1, D1, S)',
        },
        {
            'setup': common_0 + """d = 1
A, X, S, L, K, A1, B1, C1, D1 = noise_fixture(d, 1)
""",
            'call': 'lure_check(regular_lure(copy.deepcopy(A1), copy.deepcopy(B1), copy.deepcopy(C1), copy.deepcopy(D1)), A1, B1, C1, D1, S)',
            'gold_call': 'lure_check(_oracle_regular_lure(copy.deepcopy(A1), copy.deepcopy(B1), copy.deepcopy(C1), copy.deepcopy(D1)), A1, B1, C1, D1, S)',
        },
        {
            'setup': common_0 + """d = 2
A, X, S, L, K, A1, B1, C1, D1 = noise_fixture(d, 0)
""",
            'call': 'lure_check(regular_lure(copy.deepcopy(A1), copy.deepcopy(B1), copy.deepcopy(C1), copy.deepcopy(D1)), A1, B1, C1, D1, S)',
            'gold_call': 'lure_check(_oracle_regular_lure(copy.deepcopy(A1), copy.deepcopy(B1), copy.deepcopy(C1), copy.deepcopy(D1)), A1, B1, C1, D1, S)',
        },
        {
            'setup': common_1,
            'call': 'error_code(regular_lure, copy.deepcopy(A1), copy.deepcopy(B1), copy.deepcopy(C1), copy.deepcopy(-D1))',
            'gold_call': 'error_code(_oracle_regular_lure, copy.deepcopy(A1), copy.deepcopy(B1), copy.deepcopy(C1), copy.deepcopy(-D1))',
        },
        {
            'setup': common_1,
            'call': 'error_code(regular_lure, copy.deepcopy(A1.astype(complex)), copy.deepcopy(B1), copy.deepcopy(C1), copy.deepcopy(D1))',
            'gold_call': 'error_code(_oracle_regular_lure, copy.deepcopy(A1.astype(complex)), copy.deepcopy(B1), copy.deepcopy(C1), copy.deepcopy(D1))',
        },
    ]
