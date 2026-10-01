"""
Solve the triangular Fréchet correction by real-Schur block recursion.

The mixed-precision root framework extends from square roots to matrix pth roots through the Fréchet derivative of the matrix power map. The cubic-root correction is a coupled generalized Sylvester problem whose block dependencies must be derived from that derivative while preserving real-Schur blocks and deterministic binary32 recursion.

Returns
-------
np.ndarray | tuple[np.ndarray, tuple[int, int, int, int, int]]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def block_recursive_sylvester(
    S: np.ndarray,
    Stilde: np.ndarray,
    R: np.ndarray,
    blks: int,
    return_profile: bool = False,
    power: int = 2,
) -> np.ndarray | tuple[np.ndarray, tuple[int, int, int, int, int]]:
    """Solve the triangular Fréchet correction by real-Schur block recursion.

    Parameters
    ----------
    S : np.ndarray
        Finite square real Schur-form left coefficient with nonoverlapping 1x1 and 2x2 diagonal blocks.
    Stilde : np.ndarray
        Finite square real Schur-form right coefficient with the same structural convention.
    R : np.ndarray
        Finite compatible right-hand side.
    blks : int
        Positive minimal block size for the recursive solver. The direct solver is
        used only when both dimensions are at most `blks`. Otherwise a side is
        partitioned at its midpoint, except that a boundary which would cut a 2x2
        real-Schur diagonal block moves to the nearest legal boundary, with an exact
        distance tie resolved to the lower index. If the side selected by the
        recursion has no legal boundary, the other side is partitioned instead; if
        neither side has one, that subproblem is solved directly.
    power : int, default=2
        Root power selecting the paper-specific Fréchet correction; supported values are 2 and 3.
    return_profile : bool, optional
        Whether to return recursion-profile diagnostics with the correction.

    Returns
    -------
    Y : np.ndarray
        Binary32 solution of the selected triangular Fréchet correction under the prescribed real-Schur block-recursive method and task-wide deterministic arithmetic convention.
    profile : tuple, optional
        Returned only when `return_profile=True`. Five integers giving, in this
        order, the number of row splits, the number of column splits, the number of
        combined row-and-column splits, the number of direct solves performed, and
        the maximum recursion depth reached.

    Raises
    ------
    ValueError
        If an argument violates the stated domain or the required direct Fréchet subproblem is singular."""
    return Y

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

def _oracle_block_recursive_sylvester(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and explicit ValueError test case specifications."""
    return [
        {
            "setup": """import numpy as np; S=np.array([[32.0,-5.6349077224731445,2.8753058910369873,-46.20516586303711],[0.0,8.000008583068848,-4.598959445953369,68.10249328613281],[0.0,0.0,1.999853491783142,-22.72388458251953],[0.0,0.0,0.0,0.5004583597183228]],dtype=np.float32); R=np.array([[-0.0006922060856595635,0.00024347663566004485,-0.0003000994911417365,0.0017492688493803144],[0.00016364757902920246,-3.540382022038102e-05,0.00015734766202513129,-3.785319859161973e-05],[-8.213458932004869e-05,5.462803983391495e-06,7.82765528128948e-06,1.1011018159479136e-06],[2.490865699655842e-05,-3.517483128234744e-05,-1.2308053555898368e-05,-0.0002530503843445331]],dtype=np.float32)""",
            "call": """block_recursive_sylvester(S,S,R,1).tolist()""",
            "gold_call": """_oracle_block_recursive_sylvester(S,S,R,1).tolist()""",
        },
        {
            "setup": """import numpy as np; S=np.array([[2.]],dtype=np.float32); T=np.array([[3.]],dtype=np.float32); R=np.array([[1.]],dtype=np.float32)""",
            "call": """block_recursive_sylvester(S,T,R,1).tolist()""",
            "gold_call": """_oracle_block_recursive_sylvester(S,T,R,1).tolist()""",
        },
        {
            "setup": """import numpy as np; S=np.array([[2.,0.5,-0.25,0.125],[0.,3.,0.75,-0.5],[0.,0.,4.,0.25],[0.,0.,0.,5.]],dtype=np.float32); T=np.array([[1.5]],dtype=np.float32); R=np.array([[1.],[2.],[-1.],[0.5]],dtype=np.float32)""",
            "call": """block_recursive_sylvester(S,T,R,1).tolist()""",
            "gold_call": """_oracle_block_recursive_sylvester(S,T,R,1).tolist()""",
        },
        {
            "setup": """import numpy as np; S=np.array([[2.]],dtype=np.float32); T=np.array([[1.5,0.2,-0.1,0.05],[0.,2.5,0.3,-0.2],[0.,0.,3.5,0.4],[0.,0.,0.,4.5]],dtype=np.float32); R=np.array([[1.,-2.,0.5,3.]],dtype=np.float32)""",
            "call": """block_recursive_sylvester(S,T,R,1).tolist()""",
            "gold_call": """_oracle_block_recursive_sylvester(S,T,R,1).tolist()""",
        },
        {
            "setup": """import numpy as np; S=np.array([[2.,0.5,-0.25,0.125],[0.,3.,0.75,-0.5],[0.,0.,4.,0.25],[0.,0.,0.,5.]],dtype=np.float32); T=np.array([[1.25,-0.4,0.2,-0.1],[0.,2.25,0.35,0.15],[0.,0.,3.25,-0.45],[0.,0.,0.,4.25]],dtype=np.float32); R=np.array([[1.,-2.,3.,-4.],[0.5,1.5,-2.5,3.5],[-1.25,0.75,2.25,-3.25],[4.5,-3.5,2.5,-1.5]],dtype=np.float32)""",
            "call": """block_recursive_sylvester(S,T,R,1).tolist()""",
            "gold_call": """_oracle_block_recursive_sylvester(S,T,R,1).tolist()""",
        },
        {
            "setup": """import numpy as np; S=np.array([[2.,0.5],[0.,3.]],dtype=np.float32); T=np.array([[1.5,-0.25],[0.,2.5]],dtype=np.float32); R=np.array([[1.,2.],[-1.,0.5]],dtype=np.float32)""",
            "call": """block_recursive_sylvester(S,T,R,2).tolist()""",
            "gold_call": """_oracle_block_recursive_sylvester(S,T,R,2).tolist()""",
        },
        {
            "setup": """import numpy as np; S=np.array([[1.75,-0.5476887822151184,0.6472963094711304,0.4840942919254303,-0.08635523170232773,0.23171599209308624,0.30215296149253845,-0.26513350009918213,-0.15947496891021729,0.48651424050331116],[0.0,2.2300000190734863,0.11141631007194519,0.5069504976272583,-0.23181386291980743,0.5738078951835632,0.24727554619312286,-0.6389654874801636,-0.26891806721687317,-0.14532488584518433],[0.0,0.0,2.7100000381469727,0.48986032605171204,-0.35671019554138184,-0.23628348112106323,0.17477069795131683,0.25266844034194946,0.05760499835014343,0.38257068395614624],[0.0,0.0,0.0,2.859999895095825,-0.0658932477235794,-0.0034122848883271217,-0.40489813685417175,-0.49106642603874207,0.577375590801239,-0.5957116484642029],[0.0,0.0,0.0,0.0,3.3399999141693115,0.20856986939907074,-0.31330257654190063,-0.5742366313934326,0.10909886658191681,0.6493996977806091],[0.0,0.0,0.0,0.0,0.0,3.819999933242798,-0.08699362725019455,-0.011230244301259518,-0.5510382652282715,0.35228511691093445],[0.0,0.0,0.0,0.0,0.0,0.0,3.9700000286102295,-0.6491404175758362,-0.20304425060749054,-0.10961788892745972],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.449999809265137,-0.3608510196208954,0.6261024475097656],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.929999828338623,-0.23277510702610016],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,5.079999923706055]],dtype=np.float32); T=np.array([[2.200000047683716,0.1049313023686409,-0.2690006196498871],[0.0,2.680000066757202,-0.45873016119003296],[0.0,0.0,3.1600000858306885]],dtype=np.float32); R=np.array([[-1.1015925407409668,1.2961809635162354,-0.958315372467041],[-0.23619705438613892,-1.6999733448028564,-2.1093225479125977],[1.181613564491272,-2.140307664871216,0.17146269977092743],[-1.0701769590377808,-2.445169448852539,-1.8600653409957886],[1.1822470426559448,1.6451971530914307,1.1213274002075195],[-1.63606595993042,-2.096529722213745,-1.6198945045471191],[1.7716633081436157,-1.1637077331542969,-1.161063313484192],[-2.2429308891296387,-0.6962889432907104,1.4529048204421997],[2.413417339324951,-0.08931871503591537,1.775968074798584],[2.1889185905456543,0.6982285976409912,-0.6296772956848145]],dtype=np.float32)""",
            "call": """block_recursive_sylvester(S,T,R,1).tolist()""",
            "gold_call": """_oracle_block_recursive_sylvester(S,T,R,1).tolist()""",
        },
        {
            "setup": """import numpy as np; S=np.array([[1.899999976158142,-0.01564614661037922,-0.08642522990703583],[0.0,2.380000114440918,-0.508642315864563],[0.0,0.0,2.859999895095825]],dtype=np.float32); T=np.array([[1.5499999523162842,-0.4416293203830719,-0.5495997667312622,-0.13836321234703064,0.6273057460784912,0.19018542766571045,-0.0514005683362484,0.2933192253112793,0.35061347484588623,0.10260419547557831],[0.0,2.0299999713897705,0.3235713541507721,-0.6412179470062256,0.19800321757793427,0.47489380836486816,0.470605731010437,-0.5248532891273499,0.5890662670135498,-0.26707515120506287],[0.0,0.0,2.509999990463257,0.6148937344551086,-0.3597637414932251,-0.02118632011115551,-0.2043759822845459,0.478229820728302,0.28451937437057495,-0.4862826466560364],[0.0,0.0,0.0,2.6600000858306885,0.6049089431762695,0.40141430497169495,-0.5476971864700317,-0.17289093136787415,-0.24992714822292328,-0.506114661693573],[0.0,0.0,0.0,0.0,3.140000104904175,0.3386887013912201,0.4694070518016815,0.44216063618659973,0.3914896547794342,-0.36109447479248047],[0.0,0.0,0.0,0.0,0.0,3.619999885559082,-0.024751313030719757,0.5338707566261292,0.40119582414627075,0.437540739774704],[0.0,0.0,0.0,0.0,0.0,0.0,3.7699999809265137,-0.6324575543403625,-0.24650557339191437,0.3388233184814453],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.25,-0.4249870181083679,0.04012199118733406],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.730000019073486,0.5008672475814819],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.880000114440918]],dtype=np.float32); R=np.array([[-1.3622642755508423,-1.2455500364303589,-0.06225818395614624,1.1534602642059326,-2.120269775390625,-1.277160406112671,1.56657874584198,1.3427586555480957,-0.4969804584980011,0.7350289225578308],[-2.3984687328338623,1.7522974014282227,-1.0418387651443481,-0.022784452885389328,2.4801764488220215,1.344723105430603,-2.107222557067871,1.6378182172775269,2.037989377975464,-0.8533186316490173],[-0.9887574911117554,-0.27005669474601746,0.8042901158332825,-0.44581350684165955,1.5101784467697144,0.7916272282600403,1.7968770265579224,1.2946187257766724,-0.3803021311759949,-1.0526021718978882]],dtype=np.float32)""",
            "call": """block_recursive_sylvester(S,T,R,1).tolist()""",
            "gold_call": """_oracle_block_recursive_sylvester(S,T,R,1).tolist()""",
        },
        {
            "setup": """import numpy as np; S=np.array([[2.049999952316284,0.27466556429862976,0.21280279755592346,0.3707174062728882,-0.20543953776359558,-0.11378571391105652,-0.5729430913925171,0.10628576576709747,0.386366069316864],[0.0,2.5299999713897705,0.07666278630495071,0.05698003992438316,-0.47882890701293945,0.48720794916152954,-0.2646786570549011,-0.5860773921012878,-0.582369863986969],[0.0,0.0,3.009999990463257,-0.6488746404647827,-0.48264846205711365,0.1724986582994461,-0.48267731070518494,-0.37322384119033813,0.5082414746284485],[0.0,0.0,0.0,3.1600000858306885,0.048693086951971054,-0.607214093208313,-0.273158460855484,0.5864311456680298,0.11641765385866165],[0.0,0.0,0.0,0.0,3.640000104904175,0.574130117893219,-0.01632075197994709,0.27064216136932373,-0.2955159544944763],[0.0,0.0,0.0,0.0,0.0,4.119999885559082,0.07858990877866745,0.4176456332206726,-0.4572107195854187],[0.0,0.0,0.0,0.0,0.0,0.0,4.269999980926514,-0.36757200956344604,-0.5429508090019226],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.75,-0.27132704854011536],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,5.230000019073486]],dtype=np.float32); T=np.array([[1.649999976158142,0.29743388295173645,-0.3678254187107086,-0.19318093359470367,0.5535237193107605,-0.394182950258255,0.19274163246154785],[0.0,2.130000114440918,0.04787071794271469,-0.08247730135917664,0.5413141846656799,-0.1518942415714264,0.21716497838497162],[0.0,0.0,2.609999895095825,0.42744702100753784,-0.33741626143455505,0.25475677847862244,0.630094587802887],[0.0,0.0,0.0,2.759999990463257,0.3283791244029999,0.18145430088043213,-0.2114899456501007],[0.0,0.0,0.0,0.0,3.240000009536743,0.042161740362644196,0.6136667728424072],[0.0,0.0,0.0,0.0,0.0,3.7200000286102295,0.08925901353359222],[0.0,0.0,0.0,0.0,0.0,0.0,3.869999885559082]],dtype=np.float32); R=np.array([[-0.46990305185317993,0.15762130916118622,-0.14415277540683746,-0.33417847752571106,2.198154926300049,2.005228281021118,-1.2012505531311035],[-0.22533997893333435,-0.6247860789299011,-1.275482416152954,-2.4686825275421143,-0.1975466012954712,-2.164938449859619,-1.860123634338379],[0.7592403292655945,-2.4396772384643555,-1.5972084999084473,0.9809824228286743,-0.16530810296535492,1.28664231300354,-0.8282751441001892],[-1.7524242401123047,0.6157729029655457,0.9585070610046387,0.6966427564620972,-0.7830951809883118,0.3525508642196655,-1.6669660806655884],[0.05246699973940849,-1.2030341625213623,-0.7354111075401306,-1.8331578969955444,-0.5777220726013184,0.40040552616119385,1.8246291875839233],[-1.884787917137146,2.026543378829956,2.3673412799835205,2.365288496017456,0.5837588310241699,0.6522024273872375,-0.38033437728881836],[-1.930582880973816,1.3299914598464966,-2.0592241287231445,1.1780405044555664,-1.0183817148208618,0.2417355477809906,-0.27039286494255066],[-1.8666561841964722,-1.4563919305801392,0.08707696199417114,-1.0897246599197388,-0.5683491230010986,-1.340749979019165,0.17989493906497955],[-2.3851096630096436,-1.3455736637115479,-2.457257032394409,2.3549695014953613,-1.4844253063201904,0.811797022819519,-0.11261232942342758]],dtype=np.float32)""",
            "call": """block_recursive_sylvester(S,T,R,2).tolist()""",
            "gold_call": """_oracle_block_recursive_sylvester(S,T,R,2).tolist()""",
        },
        {
            "setup": """import numpy as np
m,n=17,13
S=np.zeros((m,m),dtype=np.float32); T=np.zeros((n,n),dtype=np.float32); R=np.empty((m,n),dtype=np.float32)
for i in range(m):
    S[i,i]=np.float32(1.25+0.125*i)
    for j in range(i+1,m): S[i,j]=np.float32((((17*i+13*j)%19)-9)/16.0)
for i in range(n):
    T[i,i]=np.float32(0.875+0.1*i)
    for j in range(i+1,n): T[i,j]=np.float32((((7*i+11*j)%17)-8)/20.0)
for i in range(m):
    for j in range(n): R[i,j]=np.float32((((11*i+7*j+3)%23)-11)/8.0)""",
            "call": """block_recursive_sylvester(S,T,R,2).tolist()""",
            "gold_call": """_oracle_block_recursive_sylvester(S,T,R,2).tolist()""",
        },
        {
            "setup": """import numpy as np
m,n=17,5
S=np.zeros((m,m),dtype=np.float32); T=np.zeros((n,n),dtype=np.float32); R=np.empty((m,n),dtype=np.float32)
for i in range(m):
    S[i,i]=np.float32(1.5+0.0625*i)
    for j in range(i+1,m): S[i,j]=np.float32((((5*i+9*j)%13)-6)/14.0)
for i in range(n):
    T[i,i]=np.float32(0.75+0.2*i)
    for j in range(i+1,n): T[i,j]=np.float32((((3*i+8*j)%11)-5)/12.0)
for i in range(m):
    for j in range(n): R[i,j]=np.float32((((13*i+5*j+1)%29)-14)/9.0)""",
            "call": """block_recursive_sylvester(S,T,R,2).tolist()""",
            "gold_call": """_oracle_block_recursive_sylvester(S,T,R,2).tolist()""",
        },
        {
            "setup": """import numpy as np
m,n=5,17
S=np.zeros((m,m),dtype=np.float32); T=np.zeros((n,n),dtype=np.float32); R=np.empty((m,n),dtype=np.float32)
for i in range(m):
    S[i,i]=np.float32(1.125+0.25*i)
    for j in range(i+1,m): S[i,j]=np.float32((((4*i+7*j)%11)-5)/10.0)
for i in range(n):
    T[i,i]=np.float32(0.9+0.075*i)
    for j in range(i+1,n): T[i,j]=np.float32((((9*i+5*j)%19)-9)/18.0)
for i in range(m):
    for j in range(n): R[i,j]=np.float32((((7*i+15*j+2)%31)-15)/10.0)""",
            "call": """block_recursive_sylvester(S,T,R,2).tolist()""",
            "gold_call": """_oracle_block_recursive_sylvester(S,T,R,2).tolist()""",
        },
        {
            "setup": """import numpy as np
m,n=15,11
S=np.zeros((m,m),dtype=np.float32); T=np.zeros((n,n),dtype=np.float32); R=np.empty((m,n),dtype=np.float32)
for i in range(m):
    S[i,i]=np.float32(1.375+0.09375*i)
    for j in range(i+1,m): S[i,j]=np.float32((((12*i+5*j)%23)-11)/21.0)
for i in range(n):
    T[i,i]=np.float32(0.8125+0.125*i)
    for j in range(i+1,n): T[i,j]=np.float32((((8*i+3*j)%13)-6)/15.0)
for i in range(m):
    for j in range(n): R[i,j]=np.float32((((9*i+4*j+5)%27)-13)/7.0)""",
            "call": """block_recursive_sylvester(S,T,R,3).tolist()""",
            "gold_call": """_oracle_block_recursive_sylvester(S,T,R,3).tolist()""",
        },
        {
            "setup": """import numpy as np; S=np.array([[1.0,4.0,0.3,0.0],[-0.25,1.0,-0.2,0.4],[0.0,0.0,2.5,-3.0],[0.0,0.0,0.5,2.5]],dtype=np.float32); T=np.array([[0.75,-2.0,0.1],[0.5,0.75,1.5],[0.0,0.0,3.25]],dtype=np.float32); R=np.array([[1.0,-2.0,0.5],[-1.5,0.25,2.0],[0.75,-0.5,1.25],[2.0,1.0,-1.0]],dtype=np.float32)""",
            "call": """block_recursive_sylvester(S,T,R,1).tolist()""",
            "gold_call": """_oracle_block_recursive_sylvester(S,T,R,1).tolist()""",
        },
        {
            "setup": """import numpy as np; S=np.array([[1.0,5.0],[-0.2,1.0]],dtype=np.float32); T=np.array([[1.0002,-4.0],[0.25,1.0002]],dtype=np.float32); R=np.array([[0.125,-0.75],[1.5,0.25]],dtype=np.float32)""",
            "call": """block_recursive_sylvester(S,T,R,1).tolist()""",
            "gold_call": """_oracle_block_recursive_sylvester(S,T,R,1).tolist()""",
        },
        {
            "setup": """import numpy as np; S=np.array([[1.25,0.4,-0.2,0.1,0.0],[0.0,1.75,0.3,-0.5,0.2],[0.0,0.0,2.25,0.6,-0.4],[0.0,0.0,0.0,2.75,0.7],[0.0,0.0,0.0,0.0,3.25]],dtype=np.float32); T=np.array([[0.9,-0.3,0.2,0.1],[0.0,1.4,0.5,-0.2],[0.0,0.0,1.9,0.4],[0.0,0.0,0.0,2.4]],dtype=np.float32); R=np.arange(1,21,dtype=np.float32).reshape(5,4)/np.float32(7.0)""",
            "call": """block_recursive_sylvester(S,T,R,2).tolist()""",
            "gold_call": """_oracle_block_recursive_sylvester(S,T,R,2).tolist()""",
        },
        {"setup": """import numpy as np; S=np.array([[2.,1.],[0.,3.]],dtype=np.float32); Stilde=np.array([[1.,2.],[0.,4.]],dtype=np.float32); Y0=np.array([[1.,-2.],[3.,1.]],dtype=np.float32); R=(S@S@Y0 + S@Y0@Stilde + Y0@Stilde@Stilde).astype(np.float32); blks=1""", "call": """block_recursive_sylvester(S,Stilde,R,blks,power=3).tolist()""", "gold_call": """_oracle_block_recursive_sylvester(S,Stilde,R,blks,power=3).tolist()"""},
        {"setup": """import numpy as np; S=np.array([[2.,1.,-1.],[0.,3.,2.],[0.,0.,5.]],dtype=np.float32); Stilde=np.array([[1.,-2.,1.],[0.,4.,3.],[0.,0.,6.]],dtype=np.float32); Y0=np.array([[1.,2.,-1.],[0.,-2.,3.],[2.,1.,1.]],dtype=np.float32); R=(S@S@Y0 + S@Y0@Stilde + Y0@Stilde@Stilde).astype(np.float32); blks=1""", "call": """block_recursive_sylvester(S,Stilde,R,blks,power=3).tolist()""", "gold_call": """_oracle_block_recursive_sylvester(S,Stilde,R,blks,power=3).tolist()"""},
        {"setup": """import numpy as np; S=np.array([[3.,1.,0.,-1.],[0.,2.,2.,1.],[0.,0.,4.,-2.],[0.,0.,0.,5.]],dtype=np.float32); Stilde=np.array([[2.,-1.,1.],[0.,3.,2.],[0.,0.,6.]],dtype=np.float32); Y0=np.array([[1.,0.,-1.],[2.,1.,3.],[-2.,1.,0.],[1.,-1.,2.]],dtype=np.float32); R=(S@S@Y0 + S@Y0@Stilde + Y0@Stilde@Stilde).astype(np.float32); blks=1""", "call": """(lambda z:(z[0].tolist(),z[1]))(block_recursive_sylvester(S,Stilde,R,blks,power=3,return_profile=True))""", "gold_call": """(lambda z:(z[0].tolist(),z[1]))(_oracle_block_recursive_sylvester(S,Stilde,R,blks,power=3,return_profile=True))"""},
        {
            "setup": """import numpy as np; S=np.eye(2,dtype=np.float32); Stilde=np.eye(2,dtype=np.float32); R=np.eye(2,dtype=np.float32); blks=0
def run_model():
    try:
        block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; S=np.eye(2,dtype=np.float32); Stilde=np.eye(2,dtype=np.float32); R=np.eye(2,dtype=np.float32); blks=1.5
def run_model():
    try:
        block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; S=np.array([1.0,2.0],dtype=np.float32); Stilde=np.eye(2,dtype=np.float32); R=np.ones((2,2),dtype=np.float32); blks=1
def run_model():
    try:
        block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; S=np.ones((2,3),dtype=np.float32); Stilde=np.eye(2,dtype=np.float32); R=np.ones((2,2),dtype=np.float32); blks=1
def run_model():
    try:
        block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {"setup": """import numpy as np; S=np.array([[1.0,2.0,0.5,0.0,0.0,0.0],[-3.0,1.0,1.25,0.0,0.0,0.0],[0.0,0.0,1.0009765625,-2.5,0.75,0.0],[0.0,0.0,2.0,1.0009765625,-1.5,0.25],[0.0,0.0,0.0,0.0,4.0,3.0],[0.0,0.0,0.0,0.0,-2.0,4.0]],dtype=np.float32); St=np.array([[0.5,-1.0,0.25,0.0,0.0],[2.0,0.5,-0.75,0.0,0.0],[0.0,0.0,2.0,1.0,-0.5],[0.0,0.0,0.0,3.0,4.0],[0.0,0.0,0.0,-1.0,3.0]],dtype=np.float32); R=np.arange(1,31,dtype=np.float32).reshape(6,5)/7; blks=1""", "call": """(block_recursive_sylvester(S,St,R,blks,True)[0].tolist(),block_recursive_sylvester(S,St,R,blks,True)[1])""", "gold_call": """(_oracle_block_recursive_sylvester(S,St,R,blks,True)[0].tolist(),_oracle_block_recursive_sylvester(S,St,R,blks,True)[1])"""},
        {"setup": """import numpy as np; S=np.array([[1e-2,4.0,0.0,0.0],[-3.0,1e-2,2.0,0.0],[0.0,0.0,8.0,-5.0],[0.0,0.0,2.0,8.0]],dtype=np.float32); St=np.array([[1.1e-2,-2.0,0.5,0.0],[3.5,1.1e-2,-1.0,0.25],[0.0,0.0,7.5,6.0],[0.0,0.0,-1.5,7.5]],dtype=np.float32); R=np.array([[1e8,-1e-4,3.0,-7.0],[-2e8,5.0,1e-5,9.0],[4.0,-6.0,7e7,2.0],[8.0,1e-6,-3e7,11.0]],dtype=np.float32); blks=1""", "call": """block_recursive_sylvester(S,St,R,blks).tolist()""", "gold_call": """_oracle_block_recursive_sylvester(S,St,R,blks).tolist()"""},
        {"setup": """import numpy as np; S=np.diag(np.array([1.,2.,3.,4.,5.,6.,7.],dtype=np.float32)); S[0,1]=9.; S[1,2]=-8.; S[2,3]=7.; S[3,4]=-6.; S[4,5]=5.; S[5,6]=-4.; St=np.diag(np.array([0.5,1.5,2.5],dtype=np.float32)); St[0,1]=-3.; St[1,2]=2.; R=np.arange(21,dtype=np.float32).reshape(7,3)-10.; blks=2""", "call": """(block_recursive_sylvester(S,St,R,blks,True)[0].tolist(),block_recursive_sylvester(S,St,R,blks,True)[1])""", "gold_call": """(_oracle_block_recursive_sylvester(S,St,R,blks,True)[0].tolist(),_oracle_block_recursive_sylvester(S,St,R,blks,True)[1])"""},
        {"setup": """import numpy as np; S=np.array([[2.,5.,0.,0.,0.],[0.,3.,4.,0.,0.],[0.,0.,4.,3.,0.],[0.,0.,0.,5.,2.],[0.,0.,0.,0.,6.]],dtype=np.float32); St=np.array([[1.,7.,0.,0.,0.,0.,0.],[0.,2.,6.,0.,0.,0.,0.],[0.,0.,3.,5.,0.,0.,0.],[0.,0.,0.,4.,4.,0.,0.],[0.,0.,0.,0.,5.,3.,0.],[0.,0.,0.,0.,0.,6.,2.],[0.,0.,0.,0.,0.,0.,7.]],dtype=np.float32); R=np.linspace(-3,4,35,dtype=np.float32).reshape(5,7); blks=2""", "call": """(block_recursive_sylvester(S,St,R,blks,True)[0].tolist(),block_recursive_sylvester(S,St,R,blks,True)[1])""", "gold_call": """(_oracle_block_recursive_sylvester(S,St,R,blks,True)[0].tolist(),_oracle_block_recursive_sylvester(S,St,R,blks,True)[1])"""},
        {
            "setup": """import numpy as np; S=np.empty((0,0),dtype=np.float32); Stilde=np.empty((0,0),dtype=np.float32); R=np.empty((0,0),dtype=np.float32); blks=1
def run_model():
    try:
        block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; S=np.eye(2,dtype=np.float32); Stilde=np.eye(3,dtype=np.float32); R=np.ones((2,2),dtype=np.float32); blks=1
def run_model():
    try:
        block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; S=np.eye(2,dtype=np.float32); Stilde=np.eye(2,dtype=np.float32); R=np.array([[1.0,np.inf],[0.0,1.0]],dtype=np.float32); blks=1
def run_model():
    try:
        block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; S=np.array([[1.0,0.2,0.0],[0.0,2.0,0.3],[0.1,0.0,3.0]],dtype=np.float32); Stilde=np.eye(3,dtype=np.float32); R=np.eye(3,dtype=np.float32); blks=1
def run_model():
    try:
        block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; S=np.array([[1.0]],dtype=np.float32); Stilde=np.array([[-1.0]],dtype=np.float32); R=np.array([[1.0]],dtype=np.float32); blks=1
def run_model():
    try:
        block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np
S=np.array([[2.0,0.8,0.2,-0.1,0.05,0.0],[-0.6,2.4,0.3,0.1,-0.2,0.04],[0.0,0.0,3.0,-0.7,0.25,0.1],[0.0,0.0,0.5,3.3,-0.15,0.2],[0.0,0.0,0.0,0.0,4.0,0.35],[0.0,0.0,0.0,0.0,0.0,4.5]],dtype=np.float32)
T=np.array([[1.5,-0.5,0.2,0.0,0.1],[0.4,1.8,-0.1,0.3,0.0],[0.0,0.0,2.5,0.6,-0.2],[0.0,0.0,-0.45,2.9,0.25],[0.0,0.0,0.0,0.0,3.7]],dtype=np.float32)
R=np.array([[0.25,-0.5,0.75,-1.0,1.25],[-0.2,0.4,-0.6,0.8,-1.0],[0.15,-0.3,0.45,-0.6,0.75],[-0.1,0.2,-0.3,0.4,-0.5],[0.05,-0.1,0.15,-0.2,0.25],[-0.04,0.08,-0.12,0.16,-0.2]],dtype=np.float32)""",
            "call": """(block_recursive_sylvester(S,T,R,1,return_profile=True)[0].tolist(), block_recursive_sylvester(S,T,R,1,return_profile=True)[1])""",
            "gold_call": """(_oracle_block_recursive_sylvester(S,T,R,1,return_profile=True)[0].tolist(), _oracle_block_recursive_sylvester(S,T,R,1,return_profile=True)[1])""",
        },
        {
            "setup": """import numpy as np
S=np.array([[1.8,0.7,0.1,0.0,0.2,0.0,0.0,0.05],[-0.5,2.1,0.2,-0.1,0.0,0.1,0.0,0.0],[0.0,0.0,2.7,0.4,0.1,0.0,-0.1,0.0],[0.0,0.0,0.0,3.1,-0.6,0.2,0.0,0.1],[0.0,0.0,0.0,0.45,3.4,-0.2,0.1,0.0],[0.0,0.0,0.0,0.0,0.0,4.0,0.3,-0.15],[0.0,0.0,0.0,0.0,0.0,0.0,4.4,0.25],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.9]],dtype=np.float32)
T=np.array([[1.2,0.35,-0.1],[0.0,1.7,0.2],[0.0,0.0,2.3]],dtype=np.float32)
R=np.arange(24,dtype=np.float32).reshape(8,3)/np.float32(17.0)-np.float32(0.5)""",
            "call": """(block_recursive_sylvester(S,T,R,1,return_profile=True)[0].tolist(), block_recursive_sylvester(S,T,R,1,return_profile=True)[1])""",
            "gold_call": """(_oracle_block_recursive_sylvester(S,T,R,1,return_profile=True)[0].tolist(), _oracle_block_recursive_sylvester(S,T,R,1,return_profile=True)[1])""",
        },
        {
            "setup": """import numpy as np
S=np.array([[1.4,0.25,-0.1],[0.0,1.9,0.3],[0.0,0.0,2.6]],dtype=np.float32)
T=np.array([[1.1,0.6,0.1,0.0,-0.1,0.0,0.0,0.05],[-0.45,1.5,-0.2,0.1,0.0,0.1,0.0,0.0],[0.0,0.0,2.0,0.3,-0.15,0.0,0.1,0.0],[0.0,0.0,0.0,2.5,0.55,0.2,0.0,-0.1],[0.0,0.0,0.0,-0.35,2.8,-0.2,0.15,0.0],[0.0,0.0,0.0,0.0,0.0,3.4,0.25,-0.1],[0.0,0.0,0.0,0.0,0.0,0.0,3.9,0.2],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.3]],dtype=np.float32)
R=np.arange(24,dtype=np.float32).reshape(3,8)/np.float32(19.0)-np.float32(0.4)""",
            "call": """(block_recursive_sylvester(S,T,R,1,return_profile=True)[0].tolist(), block_recursive_sylvester(S,T,R,1,return_profile=True)[1])""",
            "gold_call": """(_oracle_block_recursive_sylvester(S,T,R,1,return_profile=True)[0].tolist(), _oracle_block_recursive_sylvester(S,T,R,1,return_profile=True)[1])""",
        },
        {
            "setup": """import numpy as np
S=np.array([[2.0,1.0],[-0.75,2.5]],dtype=np.float32); T=np.array([[1.2,-0.4],[0.6,1.7]],dtype=np.float32); R=np.array([[1.0,-0.5],[0.25,0.75]],dtype=np.float32)""",
            "call": """(block_recursive_sylvester(S,T,R,1,return_profile=True)[0].tolist(), block_recursive_sylvester(S,T,R,1,return_profile=True)[1])""",
            "gold_call": """(_oracle_block_recursive_sylvester(S,T,R,1,return_profile=True)[0].tolist(), _oracle_block_recursive_sylvester(S,T,R,1,return_profile=True)[1])""",
        },
        {
            "setup": """import numpy as np
m,n=9,7
S=np.zeros((m,m),dtype=np.float32); T=np.zeros((n,n),dtype=np.float32); R=np.empty((m,n),dtype=np.float32)
for i in range(m):
    S[i,i]=np.float32(1.0+0.2*i)
    for j in range(i+1,m): S[i,j]=np.float32((((5*i+7*j)%13)-6)/15.0)
for i in range(n):
    T[i,i]=np.float32(0.8+0.15*i)
    for j in range(i+1,n): T[i,j]=np.float32((((3*i+11*j)%17)-8)/18.0)
for i in range(m):
    for j in range(n): R[i,j]=np.float32((((9*i+4*j+2)%19)-9)/11.0)""",
            "call": """block_recursive_sylvester(S,T,R,2,return_profile=True)[1]""",
            "gold_call": """_oracle_block_recursive_sylvester(S,T,R,2,return_profile=True)[1]""",
        },
        {
            "setup": """import numpy as np; S=np.array([[1.0,0.1,0.0],[0.2,2.0,0.3],[0.0,0.4,3.0]],dtype=np.float32); Stilde=np.eye(3,dtype=np.float32); R=np.eye(3,dtype=np.float32); blks=1
def run_model():
    try:
        block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_block_recursive_sylvester(S, Stilde, R, blks)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; S=np.eye(2,dtype=np.float32); Stilde=np.eye(2,dtype=np.float32); R=np.eye(2,dtype=np.float32); blks=1; return_profile=1
def run_model():
    try:
        block_recursive_sylvester(S, Stilde, R, blks, return_profile=return_profile)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_block_recursive_sylvester(S, Stilde, R, blks, return_profile=return_profile)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },

        {"setup": """import numpy as np; S=np.eye(2,dtype=np.float32); Stilde=np.eye(2,dtype=np.float32); R=np.eye(2,dtype=np.float32); blks=1
def run_model():
    try:
        block_recursive_sylvester(S,Stilde,R,blks,power=4); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try:
        _oracle_block_recursive_sylvester(S,Stilde,R,blks,power=4); return 0
    except ValueError: return 1
    except Exception: return 2""", "call": """run_model()""", "gold_call": """run_gold()"""},
    ]
