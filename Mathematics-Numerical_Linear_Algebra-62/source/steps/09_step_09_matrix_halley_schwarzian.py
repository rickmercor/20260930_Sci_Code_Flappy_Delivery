"""
Return a matrix-Halley correction and matrix-Schwarzian diagnostics.

Matrix Halley depends on the first and second Fréchet derivatives, while the matrix Schwarzian also contains the third derivative of the residual map. For a quadratic residual that third derivative vanishes; for a cubic residual it does not. Supporting both maps therefore requires the full matrix-valued higher-derivative construction, including ordered noncommuting products and the complete polarized Schwarzian rather than a quadratic-only specialization.

Returns
-------
tuple[np.ndarray, float, float, float], binary64 Halley correction, Halley residual norm, directional Schwarzian norm, and mixed-direction polarized Schwarzian norm
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def matrix_halley_schwarzian(
    A: np.ndarray,
    X: np.ndarray,
    H: np.ndarray,
    directions: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    power: int = 2,
) -> tuple[np.ndarray, float, float, float]:
    """Return a matrix-Halley correction and matrix-Schwarzian diagnostics.

    Parameters
    ----------
    A : np.ndarray
        Finite nonempty square target matrix.
    X : np.ndarray
        Finite compatible iterate.
    H : np.ndarray
        Compatible Newton direction.
    directions : tuple, optional
        Optional ordered triple of compatible directions for the polarized Schwarzian; otherwise use the benchmark directions.
    power : int, default=2
        Supported residual-map power, 2 or 3.

    Returns
    -------
    Y : np.ndarray
        Binary64 matrix-Halley correction.
    residual_norm : float
        Binary64 residual norm after the Halley update.
    schwarzian_norm : float
        Frobenius norm of the repeated-direction matrix Schwarzian.
    polarized_norm : float
        Frobenius norm of the selected polarized matrix Schwarzian contraction.

    Raises
    ------
    ValueError
        If the inputs violate the stated domain, `power` is unsupported, a required Fréchet operator is singular, or a diagnostic is non-finite."""
    return Y, halley_residual, directional_norm, polarized_norm

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _mh_dot64(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float64)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            s = np.float64(0.0)
            for k in range(A.shape[1]):
                s = np.float64(s + np.float64(A[i, k] * B[k, j]))
            C[i, j] = s
    return C

def _mh_fro64(A):
    A = np.asarray(A, dtype=np.float64)
    s = np.float64(0.0)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            s = np.float64(s + np.float64(A[i, j] * A[i, j]))
    return float(np.sqrt(s))

def _mh_power(X, power):
    if power == 2:
        return _mh_dot64(X, X)
    return _mh_dot64(_mh_dot64(X, X), X)

def _mh_d1(X, V, power):
    if power == 2:
        return np.asarray(_mh_dot64(X, V) + _mh_dot64(V, X), dtype=np.float64)
    X2 = _mh_dot64(X, X)
    return np.asarray(_mh_dot64(X2, V) + _mh_dot64(_mh_dot64(X, V), X) + _mh_dot64(V, X2), dtype=np.float64)

def _mh_d2(X, U, V, power):
    if power == 2:
        return np.asarray(_mh_dot64(U, V) + _mh_dot64(V, U), dtype=np.float64)
    terms = [_mh_dot64(_mh_dot64(X, U), V), _mh_dot64(_mh_dot64(X, V), U), _mh_dot64(_mh_dot64(U, X), V), _mh_dot64(_mh_dot64(U, V), X), _mh_dot64(_mh_dot64(V, X), U), _mh_dot64(_mh_dot64(V, U), X)]
    Z = np.zeros_like(X, dtype=np.float64)
    for T in terms:
        Z = np.asarray(Z + T, dtype=np.float64)
    return Z

def _mh_d3(U, V, W, power):
    if power == 2:
        return np.zeros_like(U, dtype=np.float64)
    terms = [_mh_dot64(_mh_dot64(U, V), W), _mh_dot64(_mh_dot64(U, W), V), _mh_dot64(_mh_dot64(V, U), W), _mh_dot64(_mh_dot64(V, W), U), _mh_dot64(_mh_dot64(W, U), V), _mh_dot64(_mh_dot64(W, V), U)]
    Z = np.zeros_like(U, dtype=np.float64)
    for T in terms:
        Z = np.asarray(Z + T, dtype=np.float64)
    return Z

def _mh_operator(X, power):
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    B = np.zeros((n * n, n * n), dtype=np.float64)
    for j in range(n):
        for i in range(n):
            E = np.zeros((n, n), dtype=np.float64)
            E[i, j] = 1.0
            C = _mh_d1(X, E, power)
            B[:, i + j * n] = np.array([C[a, b] for b in range(n) for a in range(n)], dtype=np.float64)
    return B

def _mh_b_operator(X, H, power):
    n = X.shape[0]
    B = np.zeros((n * n, n * n), dtype=np.float64)
    for j in range(n):
        for i in range(n):
            E = np.zeros((n, n), dtype=np.float64)
            E[i, j] = 1.0
            C = _mh_d2(X, H, E, power)
            B[:, i + j * n] = np.array([C[a, b] for b in range(n) for a in range(n)], dtype=np.float64)
    return B

def _mh_solve(A, b):
    A = np.array(A, dtype=np.float64, copy=True)
    b = np.array(b, dtype=np.float64, copy=True)
    vec = b.ndim == 1
    if vec:
        b = b[:, None]
    n = A.shape[0]
    for k in range(n):
        p = k
        best = np.float64(abs(float(A[k, k])))
        for i in range(k + 1, n):
            cand = np.float64(abs(float(A[i, k])))
            if cand > best:
                best = cand
                p = i
        if best == 0.0:
            raise ValueError('singular Frechet operator')
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
        for i in range(k + 1, n):
            fac = np.float64(A[i, k] / A[k, k])
            A[i, k] = 0.0
            for j in range(k + 1, n):
                A[i, j] = np.float64(A[i, j] - np.float64(fac * A[k, j]))
            for c in range(b.shape[1]):
                b[i, c] = np.float64(b[i, c] - np.float64(fac * b[k, c]))
    z = np.zeros_like(b)
    for i in range(n - 1, -1, -1):
        if A[i, i] == 0.0:
            raise ValueError('singular Frechet operator')
        for c in range(b.shape[1]):
            rhs = np.float64(b[i, c])
            for j in range(i + 1, n):
                rhs = np.float64(rhs - np.float64(A[i, j] * z[j, c]))
            z[i, c] = np.float64(rhs / A[i, i])
    return z[:, 0] if vec else z

def _mh_vec(M):
    n = M.shape[0]
    return np.array([M[i, j] for j in range(n) for i in range(n)], dtype=np.float64)

def _mh_inv_apply(B, M):
    n = M.shape[0]
    return _mh_solve(B, _mh_vec(M)).reshape((n, n), order='F')

def _mh_polarized_p2(B, U, V, W):
    q1 = _mh_inv_apply(B, _mh_d2(np.zeros_like(U), V, W, 2))
    z1 = _mh_inv_apply(B, _mh_d2(np.zeros_like(U), U, q1, 2))
    q2 = _mh_inv_apply(B, _mh_d2(np.zeros_like(U), U, W, 2))
    z2 = _mh_inv_apply(B, _mh_d2(np.zeros_like(U), V, q2, 2))
    q3 = _mh_inv_apply(B, _mh_d2(np.zeros_like(U), U, V, 2))
    z3 = _mh_inv_apply(B, _mh_d2(np.zeros_like(U), W, q3, 2))
    return np.asarray(-(z1 + z2 + z3), dtype=np.float64)

def _mh_schwarzian(BX, X, U, V, W, power):
    c = _mh_inv_apply(BX, _mh_d3(U, V, W, power))
    q1 = _mh_inv_apply(BX, _mh_d2(X, V, W, power))
    z1 = _mh_inv_apply(BX, _mh_d2(X, U, q1, power))
    q2 = _mh_inv_apply(BX, _mh_d2(X, U, W, power))
    z2 = _mh_inv_apply(BX, _mh_d2(X, V, q2, power))
    q3 = _mh_inv_apply(BX, _mh_d2(X, U, V, power))
    z3 = _mh_inv_apply(BX, _mh_d2(X, W, q3, power))
    return np.asarray(np.float64(2.0) * c - z1 - z2 - z3, dtype=np.float64)

def _oracle_matrix_halley_schwarzian(A: np.ndarray, X: np.ndarray, H: np.ndarray, directions: tuple[np.ndarray, np.ndarray, np.ndarray] | None=None, power: int=2) -> tuple[np.ndarray, float, float, float]:
    A = np.asarray(A)
    X = np.asarray(X)
    H = np.asarray(H)
    for M in (A, X, H):
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError('square inputs required')
    if not A.shape == X.shape == H.shape or A.shape[0] == 0:
        raise ValueError('same nonempty shape required')
    if any((not np.all(np.isfinite(M)) for M in (A, X, H))):
        raise ValueError('finite inputs required')
    if not isinstance(power, (int, np.integer)) or isinstance(power, (bool, np.bool_)) or int(power) not in (2, 3):
        raise ValueError('power must be 2 or 3')
    power = int(power)
    A64 = np.asarray(A, dtype=np.float64)
    X64 = np.asarray(X, dtype=np.float64)
    H64 = np.asarray(H, dtype=np.float64)
    n = X64.shape[0]
    R = np.asarray(A64 - _mh_power(X64, power), dtype=np.float64)
    BX = _mh_operator(X64, power)
    if power == 2:
        M = np.asarray(X64 + np.float64(0.5) * H64, dtype=np.float64)
        K = _mh_operator(M, 2)
    else:
        BH = _mh_b_operator(X64, H64, power)
        K = np.asarray(BX + np.float64(0.5) * BH, dtype=np.float64)
    Y = _mh_inv_apply(K, R)
    Xh = np.asarray(X64 + Y, dtype=np.float64)
    Rh = np.asarray(A64 - _mh_power(Xh, power), dtype=np.float64)
    hr = _mh_fro64(Rh)
    Sd = _mh_polarized_p2(BX, H64, H64, H64) if power == 2 else _mh_schwarzian(BX, X64, H64, H64, H64, power)
    if directions is None:
        U, V, W = (H64, R, np.eye(n, dtype=np.float64))
    else:
        if not isinstance(directions, tuple) or len(directions) != 3:
            raise ValueError('directions must be None or a length-three tuple')
        UVW = []
        for D in directions:
            D = np.asarray(D)
            if D.ndim != 2 or D.shape != X.shape or (not np.all(np.isfinite(D))):
                raise ValueError('directions must be finite matrices matching X')
            UVW.append(np.asarray(D, dtype=np.float64))
        U, V, W = UVW
    Sp = _mh_polarized_p2(BX, U, V, W) if power == 2 else _mh_schwarzian(BX, X64, U, V, W, power)
    dn = _mh_fro64(Sd)
    pn = _mh_fro64(Sp)
    if not (np.isfinite(hr) and np.isfinite(dn) and np.isfinite(pn)):
        raise ValueError('non-finite diagnostic')
    return (Y, float(hr), float(dn), float(pn))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return five normal and three explicit ValueError test case specifications."""
    return [
        {
            "setup": """import numpy as np; A=np.array([[4.0,1.0],[0.0,9.0]],dtype=np.float64); X=np.array([[2.0,0.2],[0.0,3.0]],dtype=np.float64); H=np.array([[0.01,-0.03],[0.02,0.04]],dtype=np.float64)""",
            "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(matrix_halley_schwarzian(A,X,H))""",
            "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(_oracle_matrix_halley_schwarzian(A,X,H))""",
        },
        {
            "setup": """import numpy as np; A=np.array([[2.25]],dtype=np.float64); X=np.array([[1.4]],dtype=np.float64); H=np.array([[0.10357142857142868]],dtype=np.float64)""",
            "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(matrix_halley_schwarzian(A,X,H))""",
            "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(_oracle_matrix_halley_schwarzian(A,X,H))""",
        },
        {
            "setup": """import numpy as np; A=np.array([[5.0,-1.0,0.5],[0.3,3.0,0.2],[-0.4,0.1,2.0]],dtype=np.float64); X=np.array([[2.0,0.1,-0.2],[0.05,1.7,0.3],[-0.1,0.07,1.3]],dtype=np.float64); H=np.array([[0.25,-0.31,0.22],[0.08,0.032,-0.19],[-0.11,0.06,0.119]],dtype=np.float64)""",
            "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(matrix_halley_schwarzian(A,X,H))""",
            "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(_oracle_matrix_halley_schwarzian(A,X,H))""",
        },
        {
            "setup": """import numpy as np; A=np.array([[6.0,-1.2,0.7],[2.1,4.5,-0.8],[-1.4,0.9,3.2]],dtype=np.float64); X=np.array([[2.1,-0.35,0.18],[0.42,1.75,-0.27],[-0.31,0.22,1.4]],dtype=np.float64); H=np.array([[0.21,-0.17,0.08],[0.14,0.19,-0.05],[-0.12,0.09,0.16]],dtype=np.float64)""",
            "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(matrix_halley_schwarzian(A,X,H))""",
            "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(_oracle_matrix_halley_schwarzian(A,X,H))""",
        },
        {
            "setup": """import numpy as np; A=np.array([[8.0,1.5,-2.0,0.3],[-0.7,5.0,1.1,-1.4],[1.2,-0.9,3.5,0.8],[-1.1,0.4,1.7,6.2]],dtype=np.float64); X=np.array([[2.4,0.3,-0.2,0.1],[-0.25,1.9,0.35,-0.15],[0.2,-0.3,1.55,0.28],[-0.18,0.12,-0.22,2.05]],dtype=np.float64); H=np.array([[0.45,-0.12,0.08,-0.03],[0.09,0.31,-0.11,0.06],[-0.07,0.13,0.22,-0.09],[0.05,-0.04,0.12,0.28]],dtype=np.float64)""",
            "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(matrix_halley_schwarzian(A,X,H))""",
            "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(_oracle_matrix_halley_schwarzian(A,X,H))""",
        },
        {
            "setup": """import numpy as np; A=np.array([[3.2,-4.1,1.7],[2.6,0.9,-3.3],[-1.8,2.2,4.7]],dtype=np.float64); X=np.array([[0.8,2.0,-0.5],[-0.3,1.1,1.7],[0.2,-0.4,1.5]],dtype=np.float64); H=np.array([[0.17,-0.23,0.11],[-0.08,0.19,-0.14],[0.13,-0.07,0.21]],dtype=np.float64)""",
            "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(matrix_halley_schwarzian(A,X,H))""",
            "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(_oracle_matrix_halley_schwarzian(A,X,H))""",
        },
        {
            "setup": """import numpy as np; A=np.array([[8.0,-3.0,2.0,-1.0],[1.5,5.5,-2.2,0.8],[-0.7,1.1,4.2,2.4],[0.9,-1.3,0.6,6.1]],dtype=np.float64); X=np.array([[1.7,1.2,-0.8,0.3],[-0.4,1.5,0.9,-0.6],[0.2,-0.7,1.3,1.1],[-0.1,0.5,-0.9,1.8]],dtype=np.float64); H=np.array([[0.31,-0.27,0.19,-0.11],[0.14,0.22,-0.18,0.09],[-0.16,0.12,0.28,-0.21],[0.07,-0.13,0.17,0.24]],dtype=np.float64)""",
            "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(matrix_halley_schwarzian(A,X,H))""",
            "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(_oracle_matrix_halley_schwarzian(A,X,H))""",
        },
        {
            "setup": """import numpy as np; A=np.array([[1.2e-4,-2.0e-5],[3.0e-5,8.0e-5]],dtype=np.float64); X=np.array([[0.011,0.004],[-0.003,0.009]],dtype=np.float64); H=np.array([[2e-4,-5e-4],[3e-4,1e-4]],dtype=np.float64)""",
            "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(matrix_halley_schwarzian(A,X,H))""",
            "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(_oracle_matrix_halley_schwarzian(A,X,H))""",
        },
        {"setup": """import numpy as np; A=np.array([[2.,-3.,4.],[5.,-6.,7.],[-8.,9.,10.]],dtype=np.float64); X=np.array([[1.,1e4,-2e4],[0.,0.5000001,3e4],[0.,0.,2.]],dtype=np.float64); H=np.array([[1e-3,-2.,3.],[-4e-4,5e-3,-6.],[7.,-8e-4,9e-3]],dtype=np.float64)""", "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(matrix_halley_schwarzian(A,X,H))""", "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(_oracle_matrix_halley_schwarzian(A,X,H))"""},
        {"setup": """import numpy as np; A=np.arange(1,26,dtype=np.float64).reshape(5,5); X=np.array([[2.,3.,-1.,4.,-2.],[-5.,1.5,6.,-3.,2.],[4.,-2.,0.75,5.,-6.],[1.,7.,-4.,2.5,3.],[-3.,2.,8.,-1.,1.25]],dtype=np.float64); H=np.array([[.1,-.2,.3,-.4,.5],[-.6,.7,-.8,.9,-1.],[1.1,-1.2,1.3,-1.4,1.5],[-1.6,1.7,-1.8,1.9,-2.],[2.1,-2.2,2.3,-2.4,2.5]],dtype=np.float64)""", "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(matrix_halley_schwarzian(A,X,H))""", "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(_oracle_matrix_halley_schwarzian(A,X,H))"""},
        {"setup": """import numpy as np; A=np.array([[1e12,-2e-8],[3e7,4e-12]],dtype=np.float64); X=np.array([[1e6,2e-6],[-3e2,4e-6]],dtype=np.float64); H=np.array([[1e-6,-2e2],[3e-8,-4e-4]],dtype=np.float64)""", "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(matrix_halley_schwarzian(A,X,H))""", "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(_oracle_matrix_halley_schwarzian(A,X,H))"""},
        {"setup": """import numpy as np; A=np.array([[3.,1.,4.,1.],[5.,9.,2.,6.],[5.,3.,5.,8.],[9.,7.,9.,3.]],dtype=np.float64); X=np.array([[1.,2.,3.,4.],[-2.,1.,-4.,3.],[3.,-4.,1.,-2.],[-4.,3.,-2.,1.]],dtype=np.float64); H=np.array([[.25,-.5,.75,-1.],[1.25,-1.5,1.75,-2.],[2.25,-2.5,2.75,-3.],[3.25,-3.5,3.75,-4.]],dtype=np.float64)""", "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(matrix_halley_schwarzian(A,X,H))""", "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(_oracle_matrix_halley_schwarzian(A,X,H))"""},
        {"setup": """import numpy as np; A=np.array([[6.,-1.,2.],[.5,4.,-1.5],[1.2,.3,3.]],dtype=np.float64); X=np.array([[2.,.4,-.2],[-.1,1.7,.5],[.3,-.2,1.4]],dtype=np.float64); H=np.array([[.2,-.1,.05],[.07,.16,-.08],[-.04,.09,.13]],dtype=np.float64); U=np.array([[1.,2.,0.],[-1.,.5,3.],[2.,-2.,1.]],dtype=np.float64); V=np.array([[.3,-1.,2.],[4.,-.2,.5],[-.7,1.5,.9]],dtype=np.float64); W=np.array([[2.,0.,-3.],[1.,1.,2.],[.5,-2.,.25]],dtype=np.float64); directions=(U,V,W)""", "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(matrix_halley_schwarzian(A,X,H,directions))""", "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(_oracle_matrix_halley_schwarzian(A,X,H,directions))"""},
        {"setup": """import numpy as np; A=np.array([[9.,2.,-1.,.5],[-.4,5.,1.2,-.7],[.8,-1.1,4.,1.5],[-.6,.3,-.9,6.]],dtype=np.float64); X=np.array([[2.5,.3,-.2,.1],[-.15,1.9,.25,-.3],[.2,-.1,1.6,.35],[-.25,.18,-.12,2.1]],dtype=np.float64); H=np.full((4,4),.03,dtype=np.float64); U=np.arange(1,17,dtype=np.float64).reshape(4,4)/7.; V=np.flipud(U.T)-.4; W=np.rot90(U)+.2; directions=(U,V,W)""", "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(matrix_halley_schwarzian(A,X,H,directions))""", "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(_oracle_matrix_halley_schwarzian(A,X,H,directions))"""},
        {
            "setup": """import numpy as np; A=np.empty((0,0),dtype=np.float64); X=np.empty((0,0),dtype=np.float64); H=np.empty((0,0),dtype=np.float64)
def run_model():
    try:
        matrix_halley_schwarzian(A,X,H); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try:
        _oracle_matrix_halley_schwarzian(A,X,H); return 0
    except ValueError: return 1
    except Exception: return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; A=np.eye(2,dtype=np.float64); X=np.eye(2,dtype=np.float64); H=np.eye(3,dtype=np.float64)
def run_model():
    try:
        matrix_halley_schwarzian(A,X,H); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try:
        _oracle_matrix_halley_schwarzian(A,X,H); return 0
    except ValueError: return 1
    except Exception: return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; A=np.eye(2,dtype=np.float64); X=np.zeros((2,2),dtype=np.float64); H=np.zeros((2,2),dtype=np.float64)
def run_model():
    try:
        matrix_halley_schwarzian(A,X,H); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try:
        _oracle_matrix_halley_schwarzian(A,X,H); return 0
    except ValueError: return 1
    except Exception: return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {"setup": """import numpy as np; A=np.eye(2); X=2*np.eye(2); H=np.eye(2); directions=(np.eye(2),np.eye(3),np.eye(2))
def run_model():
    try:
        matrix_halley_schwarzian(A,X,H,directions); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try:
        _oracle_matrix_halley_schwarzian(A,X,H,directions); return 0
    except ValueError: return 1
    except Exception: return 2""", "call": """run_model()""", "gold_call": """run_gold()"""},

        {"setup": """import numpy as np; X=np.array([[1.3,.25],[-.15,1.05]],dtype=np.float64); A=np.array([[2.8,-.3],[.5,1.7]],dtype=np.float64); H=np.array([[0.05595533019953587,-0.3582599959162474],[0.2926259004623697,0.07440950087306487]],dtype=np.float64); power=3""", "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(matrix_halley_schwarzian(A,X,H,power=power))""", "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(_oracle_matrix_halley_schwarzian(A,X,H,power=power))"""},
        {"setup": """import numpy as np; X=np.array([[1.2,.4,-.2],[-.1,1.0,.3],[.2,-.25,.85]],dtype=np.float64); A=np.array([[2.4,-.5,.7],[.2,1.6,-.3],[-.4,.15,1.2]],dtype=np.float64); H=np.array([[-0.015901943054193842,-0.4126590016215784,0.5899820130408928],[0.24425846731483658,-0.043964375327041325,-0.3330137812999214],[-0.2620859392760131,0.4699020621557808,-0.0886050841597939]],dtype=np.float64); U=np.array([[1.,2.,-1.],[.5,-.2,3.],[-2.,1.,.7]]); V=np.array([[.3,-1.,2.],[1.5,.4,-.6],[.8,-2.,1.1]]); W=np.array([[2.,.1,-.7],[-1.,1.3,.5],[.4,-.8,1.6]]); directions=(U,V,W); power=3""", "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(matrix_halley_schwarzian(A,X,H,directions,power))""", "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(_oracle_matrix_halley_schwarzian(A,X,H,directions,power))"""},
        {"setup": """import numpy as np; X=np.array([[1.1,.6,-.3,.2],[-.2,.95,.4,-.1],[.15,-.35,.8,.3],[-.1,.2,-.25,1.0]],dtype=np.float64); A=np.arange(1,17,dtype=np.float64).reshape(4,4)/9.; H=np.array([[-0.40785685664694127,-0.2204684399035743,0.054884047728650236,-0.01689491503557344],[0.18394592762898496,-0.18918756215470825,0.09168227451579954,0.19441477825220638],[0.2391320057353271,0.3903967202236644,0.24093074377950935,0.17161495615227232],[0.5332412510305641,0.3954631129219421,0.9479706824134992,0.01606080953560458]],dtype=np.float64); power=3""", "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(matrix_halley_schwarzian(A,X,H,power=power))""", "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),float(z[3])))(_oracle_matrix_halley_schwarzian(A,X,H,power=power))"""},
        {"setup": """import numpy as np; A=np.eye(2); X=np.eye(2); H=np.zeros((2,2)); power=5
def run_model():
    try: matrix_halley_schwarzian(A,X,H,power=power); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_matrix_halley_schwarzian(A,X,H,power=power); return 0
    except ValueError: return 1
    except Exception: return 2""", "call": """run_model()""", "gold_call": """run_gold()"""},
    ]
