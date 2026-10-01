#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

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

def triangular_principal_sqrt(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T)
    if T.ndim != 2 or T.shape[0] != T.shape[1] or T.shape[0] == 0:
        raise ValueError("T must be a nonempty square matrix")
    if not np.all(np.isfinite(T)):
        raise ValueError("T must be finite")
    T = np.asarray(T, dtype=np.float32)
    if np.any(np.tril(T, -1) != np.float32(0.0)):
        raise ValueError("T must be upper triangular")
    if np.any(np.diag(T) <= np.float32(0.0)):
        raise ValueError("T must have strictly positive diagonal entries")

    n = T.shape[0]
    S = np.zeros_like(T, dtype=np.float32)
    for i in range(n):
        S[i, i] = np.float32(np.sqrt(np.float32(T[i, i])))

    for gap in range(1, n):
        for i in range(n - gap):
            j = i + gap
            rhs = np.float32(T[i, j])
            for k in range(i + 1, j):
                rhs = np.float32(rhs - np.float32(S[i, k] * S[k, j]))
            denom = np.float32(S[i, i] + S[j, j])
            if denom == np.float32(0.0):
                raise ValueError("zero triangular square-root denominator")
            S[i, j] = np.float32(rhs / denom)
    return S

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

def reconstruct_initial_sqrt(Q: np.ndarray, S: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q)
    S = np.asarray(S)
    if Q.ndim != 2 or S.ndim != 2 or Q.shape[0] != Q.shape[1] or S.shape[0] != S.shape[1]:
        raise ValueError("Q and S must be square")
    if Q.shape != S.shape or Q.shape[0] == 0:
        raise ValueError("Q and S must have the same nonempty shape")
    if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(S)):
        raise ValueError("Q and S must be finite")
    Q32 = np.asarray(Q, dtype=np.float32)
    S32 = np.asarray(S, dtype=np.float32)
    QS = _dot32_left(Q32, S32)
    return _dot32_left(QS, Q32.T)

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

def working_residual_backward_error(A: np.ndarray, X0: np.ndarray) -> tuple[np.ndarray, float]:
    A = np.asarray(A)
    X0 = np.asarray(X0)
    if A.ndim != 2 or X0.ndim != 2 or A.shape[0] != A.shape[1] or X0.shape[0] != X0.shape[1]:
        raise ValueError("A and X0 must be square")
    if A.shape != X0.shape or A.shape[0] == 0:
        raise ValueError("A and X0 must have the same nonempty shape")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(X0)):
        raise ValueError("A and X0 must be finite")
    A64 = np.asarray(A, dtype=np.float64)
    X64 = np.asarray(X0, dtype=np.float64)
    XX = _dot64_left(X64, X64)
    R0 = np.empty_like(A64)
    for i in range(A64.shape[0]):
        for j in range(A64.shape[1]):
            R0[i, j] = np.float64(A64[i, j] - XX[i, j])
    anorm = _fro64_row_major(A64)
    if anorm == 0.0:
        raise ValueError("A must have nonzero Frobenius norm")
    eta0 = _fro64_row_major(R0) / anorm
    return R0, float(eta0)

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

def schur_residual_transform(Q: np.ndarray, R0: np.ndarray) -> tuple[np.ndarray, float]:
    Q = np.asarray(Q)
    R0 = np.asarray(R0)
    if Q.ndim != 2 or R0.ndim != 2 or Q.shape[0] != Q.shape[1] or R0.shape[0] != R0.shape[1]:
        raise ValueError("Q and R0 must be square")
    if Q.shape != R0.shape or Q.shape[0] == 0:
        raise ValueError("Q and R0 must have the same nonempty shape")
    if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(R0)):
        raise ValueError("Q and R0 must be finite")
    Q64 = np.asarray(Q, dtype=np.float64)
    R64 = np.asarray(R0, dtype=np.float64)
    left = _dot64_left(Q64.T, R64)
    full = _dot64_left(left, Q64)
    Rhat = np.asarray(full, dtype=np.float32)
    return Rhat, _fro64_row_major(Rhat)

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

def _sub32(A, B):
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    if A.shape != B.shape:
        raise ValueError("shape mismatch")
    C = np.empty_like(A, dtype=np.float32)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i, j] = np.float32(A[i, j] - B[i, j])
    return C

def _is_real_schur_form(A):
    A = np.asarray(A, dtype=np.float32)
    n = A.shape[0]
    if np.any(np.tril(A, -2) != np.float32(0.0)):
        return False
    # Nonzero first-subdiagonal entries identify 2x2 diagonal blocks; they may
    # not overlap with an adjacent such entry.
    for i in range(1, n):
        if A[i, i-1] != np.float32(0.0):
            if i >= 2 and A[i-1, i-2] != np.float32(0.0):
                return False
            if i + 1 < n and A[i+1, i] != np.float32(0.0):
                return False
    return True

def _safe_half_split(A):
    n = A.shape[0]
    if n <= 1:
        return None
    p = n // 2
    if p <= 0 or p >= n:
        return None
    if A[p, p-1] == np.float32(0.0):
        return p
    cand = [q for q in (p-1, p+1) if 0 < q < n and A[q, q-1] == np.float32(0.0)]
    if not cand:
        return None
    return min(cand, key=lambda q: (abs(n - 2*q), q))

def _gauss32_solve(M, b):
    M = np.asarray(M, dtype=np.float32).copy()
    b = np.asarray(b, dtype=np.float32).copy()
    n = b.size
    for c in range(n):
        piv = c
        best = abs(float(M[c, c]))
        for r in range(c+1, n):
            val = abs(float(M[r, c]))
            if val > best:
                best = val
                piv = r
        if M[piv, c] == np.float32(0.0):
            raise ValueError("singular direct Sylvester system")
        if piv != c:
            M[[c, piv], :] = M[[piv, c], :]
            b[c], b[piv] = b[piv], b[c]
        for r in range(c+1, n):
            fac = np.float32(M[r, c] / M[c, c])
            M[r, c] = np.float32(0.0)
            for j in range(c+1, n):
                M[r, j] = np.float32(M[r, j] - np.float32(fac * M[c, j]))
            b[r] = np.float32(b[r] - np.float32(fac * b[c]))
    x = np.zeros(n, dtype=np.float32)
    for i in range(n-1, -1, -1):
        rhs = np.float32(b[i])
        for j in range(i+1, n):
            rhs = np.float32(rhs - np.float32(M[i, j] * x[j]))
        if M[i, i] == np.float32(0.0):
            raise ValueError("singular direct Sylvester system")
        x[i] = np.float32(rhs / M[i, i])
    return x

def block_recursive_sylvester(
    S: np.ndarray,
    Stilde: np.ndarray,
    R: np.ndarray,
    blks: int,
    return_profile: bool = False,
    power: int = 2,
) -> np.ndarray | tuple[np.ndarray, tuple[int, int, int, int, int]]:
    S = np.asarray(S)
    Stilde = np.asarray(Stilde)
    R = np.asarray(R)
    if isinstance(blks, bool) or not isinstance(blks, (int, np.integer)) or int(blks) <= 0:
        raise ValueError("blks must be a positive integer")
    if isinstance(power, bool) or not isinstance(power, (int, np.integer)) or int(power) not in (2,3):
        raise ValueError("power must be 2 or 3")
    power = int(power)
    if not isinstance(return_profile, (bool, np.bool_)):
        raise ValueError("return_profile must be boolean")
    blks = int(blks)
    if S.ndim != 2 or Stilde.ndim != 2 or R.ndim != 2:
        raise ValueError("inputs must be matrices")
    if S.shape[0] != S.shape[1] or Stilde.shape[0] != Stilde.shape[1]:
        raise ValueError("coefficient matrices must be square")
    m, n = S.shape[0], Stilde.shape[0]
    if m == 0 or n == 0 or R.shape != (m, n):
        raise ValueError("incompatible or empty shapes")
    if not np.all(np.isfinite(S)) or not np.all(np.isfinite(Stilde)) or not np.all(np.isfinite(R)):
        raise ValueError("inputs must be finite")
    S32 = np.asarray(S, dtype=np.float32)
    T32 = np.asarray(Stilde, dtype=np.float32)
    C32 = np.asarray(R, dtype=np.float32)
    if not _is_real_schur_form(S32) or not _is_real_schur_form(T32):
        raise ValueError("coefficients must be real quasi-upper triangular Schur form")

    profile = [0, 0, 0, 0, 0]  # row, col, both, direct, max_depth

    def _direct(A, B, C):
        mm, nn = C.shape
        # Preserve the scalar triangular recurrence exactly for the benchmark path.
        if np.all(np.diag(A, k=-1) == np.float32(0.0)) and np.all(np.diag(B, k=-1) == np.float32(0.0)):
            Y = np.zeros((mm, nn), dtype=np.float32)
            for i in range(mm - 1, -1, -1):
                for j in range(nn):
                    rhs = np.float32(C[i, j])
                    for k in range(i + 1, mm):
                        rhs = np.float32(rhs - np.float32(A[i, k] * Y[k, j]))
                    for k in range(j):
                        rhs = np.float32(rhs - np.float32(Y[i, k] * B[k, j]))
                    denom = np.float32(A[i, i] + B[j, j])
                    if denom == np.float32(0.0):
                        raise ValueError("singular direct Sylvester system")
                    Y[i, j] = np.float32(rhs / denom)
            return Y
        N = mm * nn
        K = np.zeros((N, N), dtype=np.float32)
        b = np.asarray(C, dtype=np.float32).reshape(N).copy()
        for i in range(mm):
            for j in range(nn):
                row = i * nn + j
                for k in range(mm):
                    col = k * nn + j
                    K[row, col] = np.float32(K[row, col] + A[i, k])
                for k in range(nn):
                    col = i * nn + k
                    K[row, col] = np.float32(K[row, col] + B[k, j])
        return _gauss32_solve(K, b).reshape(mm, nn)

    def _solve(A, B, C, depth):
        profile[4] = max(profile[4], depth)
        mm, nn = C.shape
        ps = _safe_half_split(A)
        qs = _safe_half_split(B)
        # Algorithm 1 switches to the direct solver only when BOTH dimensions
        # are at most blks. Real-Schur block safety can make an individual side
        # structurally unsplittable even above that threshold.
        if mm <= blks and nn <= blks:
            profile[3] += 1
            return _direct(A, B, C)

        if nn <= mm // 2:
            desired = 'row'
        elif mm <= nn // 2:
            desired = 'col'
        else:
            desired = 'both'

        if desired == 'row':
            if ps is not None:
                branch = 'row'
            elif qs is not None:
                branch = 'col'
            else:
                profile[3] += 1
                return _direct(A, B, C)
        elif desired == 'col':
            if qs is not None:
                branch = 'col'
            elif ps is not None:
                branch = 'row'
            else:
                profile[3] += 1
                return _direct(A, B, C)
        else:
            if ps is not None and qs is not None:
                branch = 'both'
            elif ps is not None:
                branch = 'row'
            elif qs is not None:
                branch = 'col'
            else:
                profile[3] += 1
                return _direct(A, B, C)

        if branch == 'row':
            profile[0] += 1
            p = ps
            A11, A12, A22 = A[:p, :p], A[:p, p:], A[p:, p:]
            C1, C2 = C[:p, :].copy(), C[p:, :].copy()
            Y2 = _solve(A22, B, C2, depth+1)
            C1 = _sub32(C1, _dot32_left(A12, Y2))
            Y1 = _solve(A11, B, C1, depth+1)
            return np.vstack((Y1, Y2)).astype(np.float32, copy=False)
        if branch == 'col':
            profile[1] += 1
            q = qs
            B11, B12, B22 = B[:q, :q], B[:q, q:], B[q:, q:]
            C1, C2 = C[:, :q].copy(), C[:, q:].copy()
            Y1 = _solve(A, B11, C1, depth+1)
            C2 = _sub32(C2, _dot32_left(Y1, B12))
            Y2 = _solve(A, B22, C2, depth+1)
            return np.hstack((Y1, Y2)).astype(np.float32, copy=False)

        profile[2] += 1
        p, q = ps, qs
        A11, A12, A22 = A[:p, :p], A[:p, p:], A[p:, p:]
        B11, B12, B22 = B[:q, :q], B[:q, q:], B[q:, q:]
        C11, C12 = C[:p, :q].copy(), C[:p, q:].copy()
        C21, C22 = C[p:, :q].copy(), C[p:, q:].copy()
        Y21 = _solve(A22, B11, C21, depth+1)
        C11 = _sub32(C11, _dot32_left(A12, Y21))
        C22 = _sub32(C22, _dot32_left(Y21, B12))
        Y11 = _solve(A11, B11, C11, depth+1)
        Y22 = _solve(A22, B22, C22, depth+1)
        C12 = _sub32(C12, _dot32_left(A12, Y22))
        C12 = _sub32(C12, _dot32_left(Y11, B12))
        Y12 = _solve(A11, B22, C12, depth+1)
        return np.vstack((np.hstack((Y11, Y12)), np.hstack((Y21, Y22)))).astype(np.float32, copy=False)

    def _direct_p3(A,B,C):
        mm,nn=C.shape; N=mm*nn
        A2=_dot32_left(A,A); B2=_dot32_left(B,B)
        K=np.zeros((N,N),dtype=np.float32); b=np.asarray(C,dtype=np.float32).reshape(N).copy()
        for i in range(mm):
            for j in range(nn):
                row=i*nn+j
                for k in range(mm):
                    K[row,k*nn+j]=np.float32(K[row,k*nn+j]+A2[i,k])
                for a in range(mm):
                    for bb in range(nn):
                        K[row,a*nn+bb]=np.float32(K[row,a*nn+bb]+np.float32(A[i,a]*B[bb,j]))
                for k in range(nn):
                    K[row,i*nn+k]=np.float32(K[row,i*nn+k]+B2[k,j])
        return _gauss32_solve(K,b).reshape(mm,nn)

    def _solve_p3(A,B,C,depth):
        profile[4]=max(profile[4],depth); mm,nn=C.shape
        ps=_safe_half_split(A); qs=_safe_half_split(B)
        if mm<=blks and nn<=blks:
            profile[3]+=1; return _direct_p3(A,B,C)
        if nn<=mm//2: desired='row'
        elif mm<=nn//2: desired='col'
        else: desired='both'
        if desired=='row': branch='row' if ps is not None else ('col' if qs is not None else 'direct')
        elif desired=='col': branch='col' if qs is not None else ('row' if ps is not None else 'direct')
        else: branch='both' if ps is not None and qs is not None else ('row' if ps is not None else ('col' if qs is not None else 'direct'))
        if branch=='direct': profile[3]+=1; return _direct_p3(A,B,C)
        # For generalized p=3 recursion, compute solved trailing/leading blocks in the
        # dependency order induced by the four block equations. Updates are evaluated
        # in binary32 with the task-wide block products.
        if branch=='row':
            profile[0]+=1; p=ps; A11,A12,A22=A[:p,:p],A[:p,p:],A[p:,p:]; C1,C2=C[:p,:].copy(),C[p:,:].copy()
            Y2=_solve_p3(A22,B,C2,depth+1)
            cross=np.asarray(_dot32_left(A11,A12)+_dot32_left(A12,A22),dtype=np.float32)
            C1=_sub32(C1,_dot32_left(cross,Y2)); C1=_sub32(C1,_dot32_left(_dot32_left(A12,Y2),B))
            Y1=_solve_p3(A11,B,C1,depth+1); return np.vstack((Y1,Y2)).astype(np.float32)
        if branch=='col':
            profile[1]+=1; q=qs; B11,B12,B22=B[:q,:q],B[:q,q:],B[q:,q:]; C1,C2=C[:,:q].copy(),C[:,q:].copy()
            Y1=_solve_p3(A,B11,C1,depth+1)
            cross=np.asarray(_dot32_left(B11,B12)+_dot32_left(B12,B22),dtype=np.float32)
            C2=_sub32(C2,_dot32_left(Y1,cross)); C2=_sub32(C2,_dot32_left(_dot32_left(A,Y1),B12))
            Y2=_solve_p3(A,B22,C2,depth+1); return np.hstack((Y1,Y2)).astype(np.float32)
        profile[2]+=1; p,q=ps,qs
        A11,A12,A22=A[:p,:p],A[:p,p:],A[p:,p:]; B11,B12,B22=B[:q,:q],B[:q,q:],B[q:,q:]
        C11,C12=C[:p,:q].copy(),C[:p,q:].copy(); C21,C22=C[p:,:q].copy(),C[p:,q:].copy()
        Y21=_solve_p3(A22,B11,C21,depth+1)
        Across=np.asarray(_dot32_left(A11,A12)+_dot32_left(A12,A22),dtype=np.float32)
        Bcross=np.asarray(_dot32_left(B11,B12)+_dot32_left(B12,B22),dtype=np.float32)
        C11=_sub32(C11,_dot32_left(Across,Y21)); C11=_sub32(C11,_dot32_left(_dot32_left(A12,Y21),B11))
        C22=_sub32(C22,_dot32_left(Y21,Bcross)); C22=_sub32(C22,_dot32_left(_dot32_left(A22,Y21),B12))
        Y11=_solve_p3(A11,B11,C11,depth+1); Y22=_solve_p3(A22,B22,C22,depth+1)
        C12=_sub32(C12,_dot32_left(Across,Y22)); C12=_sub32(C12,_dot32_left(_dot32_left(A12,Y22),B22))
        C12=_sub32(C12,_dot32_left(Y11,Bcross)); C12=_sub32(C12,_dot32_left(_dot32_left(A11,Y11),B12))
        C12=_sub32(C12,_dot32_left(_dot32_left(A12,Y21),B12))
        Y12=_solve_p3(A11,B22,C12,depth+1)
        return np.vstack((np.hstack((Y11,Y12)),np.hstack((Y21,Y22)))).astype(np.float32)

    Y = _solve(S32, T32, C32, 0) if power==2 else _solve_p3(S32,T32,C32,0)
    if return_profile:
        return Y, tuple(int(v) for v in profile)
    return Y

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

def lift_validate_correction(
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

import numpy as np

def _dot64_left(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError('incompatible matrix product')
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

def working_precision_update(A: np.ndarray, Xk: np.ndarray, DeltaXk: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    A = np.asarray(A)
    Xk = np.asarray(Xk)
    DeltaXk = np.asarray(DeltaXk)
    for M in (A, Xk, DeltaXk):
        if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] == 0:
            raise ValueError('inputs must be nonempty square matrices')
        if not np.all(np.isfinite(M)):
            raise ValueError('inputs must be finite')
    if not A.shape == Xk.shape == DeltaXk.shape:
        raise ValueError('shape mismatch')
    A64 = np.asarray(A, dtype=np.float64)
    X64 = np.asarray(Xk, dtype=np.float64)
    D64 = np.asarray(DeltaXk, dtype=np.float64)
    Xnext = np.empty_like(X64)
    for i in range(X64.shape[0]):
        for j in range(X64.shape[1]):
            Xnext[i, j] = np.float64(np.float64(X64[i, j]) + np.float64(D64[i, j]))
    P = _dot64_left(Xnext, Xnext)
    R = np.empty_like(A64)
    for i in range(A64.shape[0]):
        for j in range(A64.shape[1]):
            R[i, j] = np.float64(np.float64(A64[i, j]) - np.float64(P[i, j]))
    return (Xnext, R, _fro64_row_major(R))

import numpy as np

def _halley_gauss_solve64(A, b):
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
            raise ValueError('singular Frechet derivative')
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
        for i in range(k + 1, n):
            f = np.float64(A[i, k] / A[k, k])
            A[i, k] = 0.0
            for j in range(k + 1, n):
                A[i, j] = np.float64(A[i, j] - np.float64(f * A[k, j]))
            for j in range(b.shape[1]):
                b[i, j] = np.float64(b[i, j] - np.float64(f * b[k, j]))
    z = np.zeros_like(b)
    for i in range(n - 1, -1, -1):
        if A[i, i] == 0.0:
            raise ValueError('singular Frechet derivative')
        for c in range(b.shape[1]):
            rhs = np.float64(b[i, c])
            for j in range(i + 1, n):
                rhs = np.float64(rhs - np.float64(A[i, j] * z[j, c]))
            z[i, c] = np.float64(rhs / A[i, i])
    return z[:, 0] if vec else z

def _halley_dot64(A, B):
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

def _halley_power64(X, power):
    X = np.asarray(X, dtype=np.float64)
    if power == 2:
        return _halley_dot64(X, X)
    return _halley_dot64(_halley_dot64(X, X), X)

def _halley_frechet64(X, V, power):
    X = np.asarray(X, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if power == 2:
        return np.asarray(_halley_dot64(X, V) + _halley_dot64(V, X), dtype=np.float64)
    X2 = _halley_dot64(X, X)
    return np.asarray(_halley_dot64(X2, V) + _halley_dot64(_halley_dot64(X, V), X) + _halley_dot64(V, X2), dtype=np.float64)

def _halley_operator_matrix64(X, power):
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    N = n * n
    B = np.zeros((N, N), dtype=np.float64)
    for j in range(n):
        for i in range(n):
            E = np.zeros((n, n), dtype=np.float64)
            E[i, j] = 1.0
            C = _halley_frechet64(X, E, power)
            B[:, i + j * n] = np.array([C[a, b] for b in range(n) for a in range(n)], dtype=np.float64)
    return B

def halley_newton_direction(A: np.ndarray, X: np.ndarray, power: int=2) -> np.ndarray:
    A = np.asarray(A)
    X = np.asarray(X)
    for M in (A, X):
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError('square inputs required')
    if A.shape != X.shape or A.shape[0] == 0:
        raise ValueError('same nonempty shape required')
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(X)):
        raise ValueError('finite inputs required')
    if not isinstance(power, (int, np.integer)) or isinstance(power, (bool, np.bool_)) or int(power) not in (2, 3):
        raise ValueError('power must be 2 or 3')
    power = int(power)
    X64 = np.asarray(X, dtype=np.float64)
    A64 = np.asarray(A, dtype=np.float64)
    R = np.asarray(A64 - _halley_power64(X64, power), dtype=np.float64)
    B = _halley_operator_matrix64(X64, power)
    r = np.array([R[i, j] for j in range(X64.shape[0]) for i in range(X64.shape[0])], dtype=np.float64)
    h = _halley_gauss_solve64(B, r)
    return h.reshape(X64.shape, order='F')

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

def matrix_halley_schwarzian(A: np.ndarray, X: np.ndarray, H: np.ndarray, directions: tuple[np.ndarray, np.ndarray, np.ndarray] | None=None, power: int=2) -> tuple[np.ndarray, float, float, float]:
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

import numpy as np

def integrated_halley_advantage(A: np.ndarray, T: np.ndarray, Q: np.ndarray) -> tuple[float, float, float, float]:
    A = np.asarray(A)
    T = np.asarray(T)
    Q = np.asarray(Q)
    S = triangular_principal_sqrt(T)
    X0 = reconstruct_initial_sqrt(Q, S)
    R0, _ = working_residual_backward_error(A, X0)
    Rhat, _ = schur_residual_transform(Q, R0)
    E = block_recursive_sylvester(S, S, Rhat, 1)
    D, _ = lift_validate_correction(Q, X0, E, R0)
    _, _, mixed_r = working_precision_update(A, X0, D)
    H = halley_newton_direction(A, X0)
    _, halley_r, sn, pn = matrix_halley_schwarzian(A, X0, H)
    if halley_r == 0.0:
        raise ValueError('zero Halley residual makes advantage undefined')
    return (float(np.float64(mixed_r / halley_r)), float(sn), float(pn), float(halley_r))
SCICODE_GOLD_EOF
