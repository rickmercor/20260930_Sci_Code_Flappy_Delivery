"""
Lift the structured correction and evaluate refinement diagnostics.

A Schur-coordinate correction must be transported back to the original basis before it can be assessed against the frozen Fréchet equation. The associated local certificate measures both inverse sensitivity of that linearization and the size of the residual-driven Newton correction, so nonnormality can matter even when eigenvalues appear benign.

Returns
-------
tuple[np.ndarray, float] or tuple[np.ndarray, float, float, bool], lifted correction, defect ratio, and optional convergence certificate
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def lift_validate_correction(
    Q: np.ndarray,
    X0: np.ndarray,
    E: np.ndarray,
    R0: np.ndarray,
    return_certificate: bool = False,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, float, bool]:
    """Lift the structured correction and evaluate refinement diagnostics.

    Parameters
    ----------
    Q : np.ndarray
        Supplied Schur-vector factor.
    X0 : np.ndarray
        Initial square-root approximation.
    E : np.ndarray
        Schur-coordinate correction.
    R0 : np.ndarray
        Working-precision residual.
    return_certificate : bool, default=False
        Whether to include the local frozen-Fréchet convergence certificate.

    Returns
    -------
    DeltaX : np.ndarray
        Binary32 correction in the original coordinates.
    rho : float
        Binary64 normalized defect of the lifted correction.
    theta : float, optional
        Local convergence-certificate value for the frozen Fréchet map, returned only when requested.
    condition_holds : bool, optional
        Whether the certificate satisfies its prescribed convergence threshold.

    Raises
    ------
    ValueError
        If an input violates the matrix domain, `return_certificate` is not boolean, or a requested certificate is undefined."""
    return DeltaX, rho

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _dot32_left(A, B):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError("incompatible matrix product")
    C = np.empty((A.shape[0], B.shape[1]), dtype=np.float32)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            acc = np.float32(0.0)
            for k in range(A.shape[1]):
                acc = np.float32(acc + np.float32(A[i, k] * B[k, j]))
            C[i, j] = acc
    return C

def _dot64_left(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError("incompatible matrix product")
    C = np.empty((A.shape[0], B.shape[1]), dtype=np.float64)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            acc = np.float64(0.0)
            for k in range(A.shape[1]):
                acc = np.float64(acc + np.float64(A[i, k] * B[k, j]))
            C[i, j] = acc
    return C

def _fro64_row_major(A):
    A = np.asarray(A, dtype=np.float64)
    acc = np.float64(0.0)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            acc = np.float64(acc + np.float64(A[i, j] * A[i, j]))
    return float(np.sqrt(acc))

def _gauss_solve64(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.array(A, dtype=np.float64, copy=True)
    B = np.array(B, dtype=np.float64, copy=True)
    vector_rhs = B.ndim == 1
    if vector_rhs:
        B = B.reshape(-1, 1)
    n = A.shape[0]
    for k in range(n):
        p = k
        best = abs(float(A[k, k]))
        for i in range(k + 1, n):
            cand = abs(float(A[i, k]))
            if cand > best:
                best = cand
                p = i
        if A[p, k] == np.float64(0.0):
            raise ValueError("frozen Frechet operator is singular")
        if p != k:
            A[[k, p], :] = A[[p, k], :]
            B[[k, p], :] = B[[p, k], :]
        for i in range(k + 1, n):
            f = np.float64(A[i, k] / A[k, k])
            A[i, k] = np.float64(0.0)
            for j in range(k + 1, n):
                A[i, j] = np.float64(A[i, j] - np.float64(f * A[k, j]))
            for j in range(B.shape[1]):
                B[i, j] = np.float64(B[i, j] - np.float64(f * B[k, j]))
    X = np.zeros_like(B, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        if A[i, i] == np.float64(0.0):
            raise ValueError("frozen Frechet operator is singular")
        for col in range(B.shape[1]):
            rhs = np.float64(B[i, col])
            for j in range(i + 1, n):
                rhs = np.float64(rhs - np.float64(A[i, j] * X[j, col]))
            X[i, col] = np.float64(rhs / A[i, i])
    return X[:, 0] if vector_rhs else X


def _jacobi_largest_eigenvalue64(G: np.ndarray) -> float:
    G = np.array(G, dtype=np.float64, copy=True)
    n = G.shape[0]
    for _ in range(100):
        p = 0
        q = 1 if n > 1 else 0
        best = np.float64(0.0)
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = np.float64(abs(float(G[i, j])))
                if cand > best:
                    best = cand
                    p, q = i, j
        scale = np.float64(1.0)
        for i in range(n):
            d = np.float64(abs(float(G[i, i])))
            if d > scale:
                scale = d
        if n <= 1 or best <= np.float64(1e-15) * scale:
            break
        app = np.float64(G[p, p])
        aqq = np.float64(G[q, q])
        apq = np.float64(G[p, q])
        tau = np.float64((aqq - app) / np.float64(2.0 * apq))
        sign = np.float64(1.0 if tau >= np.float64(0.0) else -1.0)
        t = np.float64(sign / np.float64(abs(float(tau)) + np.sqrt(np.float64(1.0 + tau * tau))))
        c = np.float64(1.0 / np.sqrt(np.float64(1.0 + t * t)))
        s = np.float64(t * c)
        for k in range(n):
            if k == p or k == q:
                continue
            gkp = np.float64(G[k, p])
            gkq = np.float64(G[k, q])
            G[k, p] = G[p, k] = np.float64(c * gkp - s * gkq)
            G[k, q] = G[q, k] = np.float64(s * gkp + c * gkq)
        G[p, p] = np.float64(c*c*app - np.float64(2.0*s*c*apq) + s*s*aqq)
        G[q, q] = np.float64(s*s*app + np.float64(2.0*s*c*apq) + c*c*aqq)
        G[p, q] = G[q, p] = np.float64(0.0)
    largest = np.float64(G[0, 0])
    for i in range(1, n):
        if G[i, i] > largest:
            largest = np.float64(G[i, i])
    return float(largest)


def _newton_kantorovich_certificate(X0: np.ndarray, R0: np.ndarray) -> tuple[float, bool]:
    X64 = np.asarray(X0, dtype=np.float64)
    R64 = np.asarray(R0, dtype=np.float64)
    n = X64.shape[0]
    N = n * n
    B = np.zeros((N, N), dtype=np.float64)
    for j in range(n):
        for i in range(n):
            col = i + j * n
            for a in range(n):
                row = a + j * n
                B[row, col] = np.float64(B[row, col] + X64[a, i])
            for b in range(n):
                row = i + b * n
                B[row, col] = np.float64(B[row, col] + X64[j, b])
    I = np.eye(N, dtype=np.float64)
    Binv = _gauss_solve64(B, I)
    rvec = np.empty(N, dtype=np.float64)
    pos = 0
    for j in range(n):
        for i in range(n):
            rvec[pos] = R64[i, j]
            pos += 1
    step = _gauss_solve64(B, rvec)
    G = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            acc = np.float64(0.0)
            for k in range(N):
                acc = np.float64(acc + np.float64(Binv[k, i] * Binv[k, j]))
            G[i, j] = acc
    sigma_max = float(np.sqrt(np.float64(_jacobi_largest_eigenvalue64(G))))
    acc = np.float64(0.0)
    for i in range(N):
        acc = np.float64(acc + np.float64(step[i] * step[i]))
    step_norm = float(np.sqrt(acc))
    theta = float(np.float64(sigma_max * step_norm))
    return theta, bool(theta < 0.25)

def _oracle_lift_validate_correction(
    Q: np.ndarray,
    X0: np.ndarray,
    E: np.ndarray,
    R0: np.ndarray,
    return_certificate: bool = False,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, float, bool]:
    Q = np.asarray(Q)
    X0 = np.asarray(X0)
    E = np.asarray(E)
    R0 = np.asarray(R0)
    if not isinstance(return_certificate, (bool, np.bool_)):
        raise ValueError("return_certificate must be boolean")
    mats = [Q, X0, E, R0]
    for M in mats:
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError("all inputs must be square")
    if not (Q.shape == X0.shape == E.shape == R0.shape) or Q.shape[0] == 0:
        raise ValueError("all inputs must have the same nonempty shape")
    for M in mats:
        if not np.all(np.isfinite(M)):
            raise ValueError("all inputs must be finite")

    Q32 = np.asarray(Q, dtype=np.float32)
    E32 = np.asarray(E, dtype=np.float32)
    QE = _dot32_left(Q32, E32)
    DeltaX = _dot32_left(QE, Q32.T)

    X64 = np.asarray(X0, dtype=np.float64)
    D64 = np.asarray(DeltaX, dtype=np.float64)
    R64 = np.asarray(R0, dtype=np.float64)
    XD = _dot64_left(X64, D64)
    DX = _dot64_left(D64, X64)

    L = np.empty_like(R64)
    for i in range(R64.shape[0]):
        for j in range(R64.shape[1]):
            corr = np.float64(XD[i, j] + DX[i, j])
            L[i, j] = np.float64(R64[i, j] - corr)

    rnorm = _fro64_row_major(R64)
    rho = 0.0 if rnorm == 0.0 else _fro64_row_major(L) / rnorm
    if not return_certificate:
        return DeltaX, float(rho)
    theta, condition_holds = _newton_kantorovich_certificate(X64, R64)
    return DeltaX, float(rho), float(theta), bool(condition_holds)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and explicit ValueError test case specifications."""
    return [
        {
            "setup": """import numpy as np; Q=np.array([[0.6966455578804016, 0.4486113488674164, 0.5052264332771301, -0.24120382964611053], [0.21158075332641602, -0.4446275234222412, 0.4567680060863495, 0.7408797144889832], [-0.6356939673423767, 0.5600026845932007, 0.48396551609039307, 0.21924364566802979], [-0.2565384805202484, -0.5361445546150208, 0.5494421124458313, -0.5872394442558289]],dtype=np.float32); X0=np.array([[19.052352905273438, -4.507687568664551, -16.802370071411133, 2.757863998413086], [16.463993072509766, -31.918928146362305, -16.213943481445312, 32.16926193237305], [-25.94766616821289, 32.49231719970703, 28.19763946533203, -34.24215316772461], [3.4639892578125, -26.91893768310547, -3.213947296142578, 27.169265747070312]],dtype=np.float32); E=np.array([[-9.950833373295609e-06, 2.759739118118887e-06, -6.466281320172129e-06, 2.388826214883011e-05], [2.5674760308902478e-06, 1.161969885288272e-05, 6.961700273677707e-05, 0.0006165380473248661], [-1.9035020386581891e-06, -8.775080459599849e-06, -7.76720917201601e-05, -0.0028083526995033026], [7.664094141546229e-07, -3.6299400107964175e-06, -1.2480707482609432e-05, -0.00025380784063600004]],dtype=np.float32); R0=np.array([[-0.0006150363078631926, 0.0007059133731672773, 0.0005957527537248097, -0.0007854900577513035], [-0.0002648560985107906, 1.4817182091064751e-05, 0.00015251154400175437, -0.00015609611000400037], [0.0006939346785657108, -0.0005672639272233937, -0.000541794863238465, 0.0008202870012610219], [8.15210078144446e-05, -0.00018981749599333853, -8.943807915784419e-05, 0.00016918159963097423]],dtype=np.float64)""",
            "call": """(lambda z: (z[0].tolist(), float(z[1])))(lift_validate_correction(Q,X0,E,R0))""",
            "gold_call": """(lambda z: (z[0].tolist(), float(z[1])))(_oracle_lift_validate_correction(Q,X0,E,R0))""",
        },
        {
            "setup": """import numpy as np; Q=np.eye(1,dtype=np.float32); X0=np.array([[2.]],dtype=np.float32); E=np.array([[0.]],dtype=np.float32); R0=np.array([[0.]],dtype=np.float64)""",
            "call": """(lambda z: (z[0].tolist(), float(z[1])))(lift_validate_correction(Q,X0,E,R0))""",
            "gold_call": """(lambda z: (z[0].tolist(), float(z[1])))(_oracle_lift_validate_correction(Q,X0,E,R0))""",
        },
        {
            "setup": """import numpy as np; Q=np.eye(2,dtype=np.float32); X0=np.diag(np.array([2.,3.],dtype=np.float32)); E=np.array([[0.1,0.2],[0.3,0.4]],dtype=np.float32); R0=np.array([[0.4,1.0],[1.5,2.4]],dtype=np.float64)""",
            "call": """(lambda z: (z[0].tolist(), float(z[1])))(lift_validate_correction(Q,X0,E,R0))""",
            "gold_call": """(lambda z: (z[0].tolist(), float(z[1])))(_oracle_lift_validate_correction(Q,X0,E,R0))""",
        },
        {
            "setup": """import numpy as np; Q=np.array([[0.8,0.6],[-0.6,0.8]],dtype=np.float64); X0=np.array([[2.0,0.25],[0.0,3.0]],dtype=np.float64); E=np.array([[10000000.25,-19999999.25],[30000000.5,-39999999.875]],dtype=np.float64); R0=np.array([[80000000.0,-20000000.0],[10000000.0,50000000.0]],dtype=np.float64)""",
            "call": """(lambda z: (z[0].tolist(), float(z[1])))(lift_validate_correction(Q,X0,E,R0))""",
            "gold_call": """(lambda z: (z[0].tolist(), float(z[1])))(_oracle_lift_validate_correction(Q,X0,E,R0))""",
        },
        {
            "setup": """import numpy as np; Q=np.array([[0.7071067811865476,0.7071067811865475],[-0.7071067811865475,0.7071067811865476]],dtype=np.float64); X0=np.array([[2048.125,-0.75],[0.0,0.03125]],dtype=np.float64); E=np.array([[12345.678901234,-9876.543210987],[4321.123456789,-7654.987654321]],dtype=np.float64); R0=np.array([[20000000.0,-30000000.0],[15000000.0,10000000.0]],dtype=np.float64)""",
            "call": """(lambda z: (z[0].tolist(), float(z[1])))(lift_validate_correction(Q,X0,E,R0))""",
            "gold_call": """(lambda z: (z[0].tolist(), float(z[1])))(_oracle_lift_validate_correction(Q,X0,E,R0))""",
        },
        {
            "setup": """import numpy as np; Q=np.array([[0.6966455578804016,0.4486113488674164,0.5052264332771301,-0.24120382964611053],[0.21158075332641602,-0.4446275234222412,0.4567680060863495,0.7408797144889832],[-0.6356939673423767,0.5600026845932007,0.48396551609039307,0.21924364566802979],[-0.2565384805202484,-0.5361445546150208,0.5494421124458313,-0.5872394442558289]],dtype=np.float32); X0=np.array([[19.052352905273438,-4.507687568664551,-16.802370071411133,2.757863998413086],[16.463993072509766,-31.918928146362305,-16.213943481445312,32.16926193237305],[-25.94766616821289,32.49231719970703,28.19763946533203,-34.24215316772461],[3.4639892578125,-26.91893768310547,-3.213947296142578,27.169265747070312]],dtype=np.float32); E=np.zeros((4,4),dtype=np.float32); R0=np.array([[-0.0006150363078631926,0.0007059133731672773,0.0005957527537248097,-0.0007854900577513035],[-0.0002648560985107906,1.4817182091064751e-05,0.00015251154400175437,-0.00015609611000400037],[0.0006939346785657108,-0.0005672639272233937,-0.000541794863238465,0.0008202870012610219],[8.15210078144446e-05,-0.00018981749599333853,-8.943807915784419e-05,0.00016918159963097423]],dtype=np.float64)""",
            "call": """(lambda z: (z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(lift_validate_correction(Q,X0,E,R0,True))""",
            "gold_call": """(lambda z: (z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(_oracle_lift_validate_correction(Q,X0,E,R0,True))""",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np; Q=np.eye(1,dtype=np.float32); X0=np.array([[2.]],dtype=np.float64); E=np.zeros((1,1),dtype=np.float32); R0=np.array([[1.]],dtype=np.float64)""",
            "call": """(lambda z: (z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(lift_validate_correction(Q,X0,E,R0,True))""",
            "gold_call": """(lambda z: (z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(_oracle_lift_validate_correction(Q,X0,E,R0,True))""",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np; Q=np.eye(1,dtype=np.float32); X0=np.array([[2.]],dtype=np.float64); E=np.zeros((1,1),dtype=np.float32); R0=np.array([[5.]],dtype=np.float64)""",
            "call": """(lambda z: (z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(lift_validate_correction(Q,X0,E,R0,True))""",
            "gold_call": """(lambda z: (z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(_oracle_lift_validate_correction(Q,X0,E,R0,True))""",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np; Q=np.eye(2,dtype=np.float32); X0=np.array([[1.5,0.75],[0.0,4.0]],dtype=np.float64); E=np.zeros((2,2),dtype=np.float32); R0=np.array([[0.4,-1.2],[0.7,2.3]],dtype=np.float64)""",
            "call": """(lambda z: (z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(lift_validate_correction(Q,X0,E,R0,True))""",
            "gold_call": """(lambda z: (z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(_oracle_lift_validate_correction(Q,X0,E,R0,True))""",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np; Q=np.eye(3,dtype=np.float32); X0=np.array([[3.0,2.5,-1.0],[0.0,0.5,4.0],[0.0,0.0,0.125]],dtype=np.float64); E=np.zeros((3,3),dtype=np.float32); R0=np.array([[1e-4,-2e-4,3e-4],[4e-4,-5e-4,6e-4],[-7e-4,8e-4,-9e-4]],dtype=np.float64)""",
            "call": """(lambda z: (z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(lift_validate_correction(Q,X0,E,R0,True))""",
            "gold_call": """(lambda z: (z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(_oracle_lift_validate_correction(Q,X0,E,R0,True))""",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np; Q=np.eye(2,dtype=np.float32); X0=np.array([[0.2,3.0],[0.0,0.01]],dtype=np.float64); E=np.zeros((2,2),dtype=np.float32); R0=np.array([[0.02,-0.03],[0.04,0.01]],dtype=np.float64)""",
            "call": """(lambda z: (z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(lift_validate_correction(Q,X0,E,R0,True))""",
            "gold_call": """(lambda z: (z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(_oracle_lift_validate_correction(Q,X0,E,R0,True))""",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np; Q=np.eye(1,dtype=np.float32); X0=np.array([[2.]],dtype=np.float64); E=np.zeros((1,1),dtype=np.float32); R0=np.array([[1.]],dtype=np.float64)
def run_model():
    try:
        lift_validate_correction(Q,X0,E,R0,return_certificate=1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_lift_validate_correction(Q,X0,E,R0,return_certificate=1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; Q=np.eye(2,dtype=np.float32); X0=np.diag(np.array([1.0,-1.0],dtype=np.float64)); E=np.zeros((2,2),dtype=np.float32); R0=np.eye(2,dtype=np.float64)
def run_model():
    try:
        lift_validate_correction(Q,X0,E,R0,return_certificate=True)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_lift_validate_correction(Q,X0,E,R0,return_certificate=True)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; Q=np.array([[0.8,-0.6,0.0],[0.36,0.48,-0.8],[0.48,0.64,0.6]],dtype=np.float32); X0=np.array([[2.1,-0.7,0.25],[0.4,1.6,-0.35],[-0.2,0.3,1.2]],dtype=np.float64); E=np.array([[2e-4,-7e-5,3e-5],[5e-5,-1e-4,8e-5],[-4e-5,6e-5,1.5e-4]],dtype=np.float32); D=(Q@E@Q.T).astype(np.float32); R0=X0@D.astype(np.float64)+D.astype(np.float64)@X0""",
            "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(lift_validate_correction(Q,X0,E,R0,return_certificate=True))""",
            "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(_oracle_lift_validate_correction(Q,X0,E,R0,return_certificate=True))""",
        },
        {
            "setup": """import numpy as np; Q=np.eye(2,dtype=np.float32); X0=np.array([[1.0,12.0],[-0.04,0.55]],dtype=np.float64); E=np.array([[1e-5,-3e-5],[2e-5,4e-5]],dtype=np.float32); R0=np.array([[2.1e-5,-3.7e-4],[1.2e-5,6.3e-5]],dtype=np.float64)""",
            "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(lift_validate_correction(Q,X0,E,R0,return_certificate=True))""",
            "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(_oracle_lift_validate_correction(Q,X0,E,R0,return_certificate=True))""",
        },
        {
            "setup": """import numpy as np; Q=np.array([[0.70710677,-0.70710677],[0.70710677,0.70710677]],dtype=np.float32); X0=np.array([[3.0,1.2],[-0.8,1.1]],dtype=np.float64); E=np.array([[0.0,0.0],[0.0,0.0]],dtype=np.float32); R0=np.zeros((2,2),dtype=np.float64)""",
            "call": """(lambda z:(z[0].tolist(),float(z[1])))(lift_validate_correction(Q,X0,E,R0,return_certificate=False))""",
            "gold_call": """(lambda z:(z[0].tolist(),float(z[1])))(_oracle_lift_validate_correction(Q,X0,E,R0,return_certificate=False))""",
        },
        {"setup": """import numpy as np; Q=np.array([[0.5,0.5,0.5,0.5],[0.5,-0.5,0.5,-0.5],[0.5,0.5,-0.5,-0.5],[0.5,-0.5,-0.5,0.5]],dtype=np.float32); X0=np.array([[2.0,4.0,-1.0,0.5],[0.0,1.25,3.0,-2.0],[0.0,0.0,0.75,2.0],[0.0,0.0,0.0,0.4]],dtype=np.float64); E=np.array([[1e-4,-2e-4,3e-5,-4e-5],[5e-5,-6e-4,7e-4,-8e-5],[9e-6,-1e-4,11e-4,12e-4],[-13e-5,14e-5,-15e-5,16e-4]],dtype=np.float32); R0=np.array([[0.08,-0.20,0.03,-0.04],[0.05,-0.06,0.07,-0.08],[0.09,-0.10,0.11,-0.12],[0.13,-0.14,0.15,-0.16]],dtype=np.float64)""", "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(lift_validate_correction(Q,X0,E,R0,True))""", "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(_oracle_lift_validate_correction(Q,X0,E,R0,True))""", "tol":1e-12},
        {"setup": """import numpy as np; Q=np.eye(3,dtype=np.float32); X0=np.array([[1.0,5.0,-2.0],[0.0,1.1,4.0],[0.0,0.0,0.9]],dtype=np.float64); E=np.array([[1e-5,-2e-5,3e-5],[-4e-5,5e-5,-6e-5],[7e-5,-8e-5,9e-5]],dtype=np.float32); R0=np.array([[0.1,-0.2,0.3],[-0.4,0.5,-0.6],[0.7,-0.8,0.9]],dtype=np.float64)""", "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(lift_validate_correction(Q,X0,E,R0,True))""", "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(_oracle_lift_validate_correction(Q,X0,E,R0,True))""", "tol":1e-12},
        {"setup": """import numpy as np; Q=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]],dtype=np.float32); X0=np.array([[8.,-7.,6.],[5.,0.125,-4.],[3.,-2.,1.5]],dtype=np.float64); E=np.array([[0.125,-0.25,0.5],[-1.,2.,-4.],[8.,-16.,32.]],dtype=np.float32); R0=np.zeros((3,3),dtype=np.float64)""", "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(lift_validate_correction(Q,X0,E,R0,True))""", "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(_oracle_lift_validate_correction(Q,X0,E,R0,True))""", "tol":1e-12},
        {"setup": """import numpy as np; Q=np.eye(2,dtype=np.float32); X0=np.array([[1.0,50.0],[0.0,1.01]],dtype=np.float64); E=np.array([[2**-20,-2**-18],[2**-19,2**-17]],dtype=np.float32); R0=np.array([[0.125,-0.5],[0.25,0.75]],dtype=np.float64)""", "call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(lift_validate_correction(Q,X0,E,R0,True))""", "gold_call": """(lambda z:(z[0].tolist(),float(z[1]),float(z[2]),bool(z[3])))(_oracle_lift_validate_correction(Q,X0,E,R0,True))""", "tol":1e-12},
        {
            "setup": """import numpy as np; Q=np.empty((0,0),dtype=np.float32); X0=np.empty((0,0),dtype=np.float32); E=np.empty((0,0),dtype=np.float32); R0=np.empty((0,0),dtype=np.float64)
def run_model():
    try:
        lift_validate_correction(Q, X0, E, R0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_lift_validate_correction(Q, X0, E, R0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; Q=np.ones((2,3),dtype=np.float32); X0=np.ones((2,3),dtype=np.float32); E=np.ones((2,3),dtype=np.float32); R0=np.ones((2,3),dtype=np.float64)
def run_model():
    try:
        lift_validate_correction(Q, X0, E, R0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_lift_validate_correction(Q, X0, E, R0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; Q=np.eye(2,dtype=np.float32); X0=np.eye(2,dtype=np.float32); E=np.eye(3,dtype=np.float32); R0=np.eye(2,dtype=np.float64)
def run_model():
    try:
        lift_validate_correction(Q, X0, E, R0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_lift_validate_correction(Q, X0, E, R0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; Q=np.eye(2,dtype=np.float32); X0=np.array([[1.0,np.nan],[0.0,1.0]],dtype=np.float32); E=np.eye(2,dtype=np.float32); R0=np.eye(2,dtype=np.float64)
def run_model():
    try:
        lift_validate_correction(Q, X0, E, R0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_lift_validate_correction(Q, X0, E, R0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
    ]
