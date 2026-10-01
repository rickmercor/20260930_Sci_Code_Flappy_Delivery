"""
Compute one directional interface-continuous Fourier representation of a six-component electric-magnetic operator. Each local operator has shape (6s,6s), ordered component-major with s old harmonics per component; columns are (E1,E2,E3,H1,H2,H3) and rows are (D1,D2,D3,B1,B2,B3). Group components axis and axis+3 as a and the other components in increasing order as b. For each local P write A=Paa, B=Pab, C=Pba and D=Pbb, then Q=A^{-1}, R=QB, L=CQ and W=D-CQB. For the supplied k by k indicator T, lift each block pair by H(U)=kron(T,U_in-U_out)+kron(I_k,U_out). Reconstruct V=H(Q)^{-1}, Paa=V, Pab=V H(R), Pba=H(L)V and Pbb=H(W)+H(L)V H(R). Each block initially has order (new harmonic, grouped component, old harmonic); undo grouping and return (original component, new harmonic, old harmonic). Products are not conjugated. Accept finite entries of magnitude at most 1e6, 1<=s<=25, 1<=k<=9 and 6sk<=150. axis is an integer in [0,2], not boolean. T must be Hermitian to maximum-entry tolerance 1e-10 and its Hermitian part's eigenvalues must lie in [-1e-10,1+1e-10]. Required inverses must exist and all computed arrays must be finite. Raise ValueError otherwise; do not impose a condition-number cutoff on invertible matrices. The mathematical construction, not a particular floating-point evaluation order, defines the output. Return each entry to relative tolerance 1e-7 plus absolute tolerance 1e-9 for the supplied binary64 inputs, including normal pivots of order 1e-20 with nearly common off-diagonal blocks whose Schur terms cancel. Use a cancellation-resistant reformulation or adequate working precision; these valid cases must not be refused merely for conditioning.

Normal electric and magnetic fluxes remain continuous across a material interface even when their fields jump. A joint normal block retains electric-magnetic coupling during Fourier factorization. The Schur terms are essential for general anisotropic tensors, and the second directional application acts on full matrix-valued harmonic blocks rather than independent scalar entries.

Returns
-------
complex ndarray, directional constitutive Fourier operator
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def directional_factorization(p_in: "np.ndarray | list | tuple", p_out: "np.ndarray | list | tuple", indicator: "np.ndarray | list | tuple", axis: int) -> "np.ndarray":
    """Factorize a two-material directional Fourier operator.

    Parameters
    ----------
    p_in, p_out : array_like
        Same-shaped complex (6s,6s) arrays, 1<=s<=25, entries at most 1e6.
    indicator : array_like
        Complex Hermitian contractive (k,k) array, 1<=k<=9, entries at most 1e6.
        Absolute Hermiticity and spectral slack are 1e-10; 6*s*k<=150.
    axis : int
        Normal direction 0, 1 or 2, excluding booleans.

    Returns
    -------
    ndarray
        Complex (6*s*k,6*s*k) matrix in original component order, then new
        harmonic and old harmonic. Original order is (E1,E2,E3,H1,H2,H3).
        Entrywise accuracy is rtol=1e-7, atol=1e-9, including finite cancellation
        cases with pivots of order 1e-20. Inputs mean their binary64 values.

    Raises
    ------
    ValueError
        For invalid shapes, bounds, axis, indicator or singular inverses,
        or nonfinite intermediates/results. No condition-number cutoff applies.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import decimal
from decimal import Decimal
from fractions import Fraction

import numpy as np


def _df34_mat(x):
    return (
        [[Decimal.from_float(float(z.real)) for z in row] for row in x],
        [[Decimal.from_float(float(z.imag)) for z in row] for row in x],
    )


def _df34_sub(x, y):
    return (
        [[a-b for a, b in zip(ar, br)] for ar, br in zip(x[0], y[0])],
        [[a-b for a, b in zip(ai, bi)] for ai, bi in zip(x[1], y[1])],
    )


def _df34_add(x, y):
    return (
        [[a+b for a, b in zip(ar, br)] for ar, br in zip(x[0], y[0])],
        [[a+b for a, b in zip(ai, bi)] for ai, bi in zip(x[1], y[1])],
    )


def _df34_mm(x, y):
    xr, xi = x
    yr, yi = y
    rows, inner, cols = len(xr), len(yr), len(yr[0])
    zero = Decimal(0)
    zr = [[zero for _ in range(cols)] for _ in range(rows)]
    zi = [[zero for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        zri, zii = zr[i], zi[i]
        for h in range(inner):
            ar, ai = xr[i][h], xi[i][h]
            if not ar and not ai:
                continue
            yrh, yih = yr[h], yi[h]
            for j in range(cols):
                br, bi = yrh[j], yih[j]
                zri[j] += ar*br-ai*bi
                zii[j] += ar*bi+ai*br
    return zr, zi


def _df34_norm(x):
    return max(sum(abs(a)+abs(b) for a, b in zip(ar, ai))
               for ar, ai in zip(x[0], x[1]))


def _df34_inv(x, which, need_more):
    """Partial-pivoted complex Decimal LU inverse."""
    ar = [row[:] for row in x[0]]
    ai = [row[:] for row in x[1]]
    n = len(ar)
    order = list(range(n))
    for h in range(n):
        pivot = max(range(h, n), key=lambda i: abs(ar[i][h])+abs(ai[i][h]))
        if not ar[pivot][h] and not ai[pivot][h]:
            raise need_more(which)
        if pivot != h:
            ar[h], ar[pivot] = ar[pivot], ar[h]
            ai[h], ai[pivot] = ai[pivot], ai[h]
            order[h], order[pivot] = order[pivot], order[h]
        pr, pi = ar[h][h], ai[h][h]
        den = pr*pr+pi*pi
        for i in range(h+1, n):
            xr, xi = ar[i][h], ai[i][h]
            fr = (xr*pr+xi*pi)/den
            fi = (xi*pr-xr*pi)/den
            ar[i][h], ai[i][h] = fr, fi
            for j in range(h+1, n):
                br, bi = ar[h][j], ai[h][j]
                ar[i][j] -= fr*br-fi*bi
                ai[i][j] -= fr*bi+fi*br
    zero, one = Decimal(0), Decimal(1)
    rr = [[zero for _ in range(n)] for _ in range(n)]
    ri = [[zero for _ in range(n)] for _ in range(n)]
    for i, j in enumerate(order):
        rr[i][j] = one
    for i in range(n):
        for h in range(i):
            lr, li = ar[i][h], ai[i][h]
            if not lr and not li:
                continue
            for j in range(n):
                br, bi = rr[h][j], ri[h][j]
                rr[i][j] -= lr*br-li*bi
                ri[i][j] -= lr*bi+li*br
    for i in range(n-1, -1, -1):
        for h in range(i+1, n):
            ur, ui = ar[i][h], ai[i][h]
            if not ur and not ui:
                continue
            for j in range(n):
                br, bi = rr[h][j], ri[h][j]
                rr[i][j] -= ur*br-ui*bi
                ri[i][j] -= ur*bi+ui*br
        pr, pi = ar[i][i], ai[i][i]
        den = pr*pr+pi*pi
        if not den:
            raise need_more(which)
        for j in range(n):
            xr, xi = rr[i][j], ri[i][j]
            rr[i][j] = (xr*pr+xi*pi)/den
            ri[i][j] = (xi*pr-xr*pi)/den
    return rr, ri


def _df34_lift(t, inside, outside):
    """T kron (inside-outside) + I kron outside, new index first."""
    tr, ti = t
    dr, di = _df34_sub(inside, outside)
    vr, vi = outside
    k, rows, cols = len(tr), len(dr), len(dr[0])
    zero = Decimal(0)
    rr = [[zero for _ in range(k*cols)] for _ in range(k*rows)]
    ri = [[zero for _ in range(k*cols)] for _ in range(k*rows)]
    for n in range(k):
        for m in range(k):
            ar, ai = tr[n][m], ti[n][m]
            for i in range(rows):
                row = n*rows+i
                for j in range(cols):
                    br, bi = dr[i][j], di[i][j]
                    rr[row][m*cols+j] = ar*br-ai*bi+(vr[i][j] if n == m else zero)
                    ri[row][m*cols+j] = ar*bi+ai*br+(vi[i][j] if n == m else zero)
    return rr, ri


def _df34_common(x, k):
    """I_k kron x."""
    xr, xi = x
    rows, cols = len(xr), len(xr[0])
    zero = Decimal(0)
    rr = [[zero for _ in range(k*cols)] for _ in range(k*rows)]
    ri = [[zero for _ in range(k*cols)] for _ in range(k*rows)]
    for n in range(k):
        for i in range(rows):
            rr[n*rows+i][n*cols:n*cols+cols] = xr[i][:]
            ri[n*rows+i][n*cols:n*cols+cols] = xi[i][:]
    return rr, ri


def _df34_digits(x):
    return 0 if not x else max(0, x.adjusted()+1)


def _oracle_directional_factorization(p_in: "np.ndarray | list | tuple", p_out: "np.ndarray | list | tuple", indicator: "np.ndarray | list | tuple", axis: int) -> "np.ndarray":
    class _NeedMorePrecision(ArithmeticError):
        def __init__(self, which):
            self.which = which

    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)) or axis not in (0, 1, 2):
        raise ValueError('axis')
    try:
        pi, po, t = [np.asarray(x, dtype=complex) for x in (p_in, p_out, indicator)]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('matrix') from exc
    if any(x.ndim != 2 or x.shape[0] != x.shape[1] or x.shape[0] == 0 or not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e6 for x in (pi, po, t)):
        raise ValueError('matrix')
    if pi.shape != po.shape or len(pi) % 6:
        raise ValueError('shape')
    s, k = len(pi)//6, len(t)
    if not 1 <= s <= 25 or not 1 <= k <= 9 or 6*s*k > 150:
        raise ValueError('size')
    if np.max(np.abs(t-t.conj().T)) > 1e-10:
        raise ValueError('Hermitian')
    try:
        eigenvalues = np.linalg.eigvalsh((t+t.conj().T)/2)
    except np.linalg.LinAlgError as exc:
        raise ValueError('indicator') from exc
    if np.min(eigenvalues) < -1e-10 or np.max(eigenvalues) > 1+1e-10:
        raise ValueError('indicator')

    a = [c*s+h for c in (axis, axis+3) for h in range(s)]
    b = [c*s+h for c in range(6) if c not in (axis, axis+3) for h in range(s)]
    local = [[p[np.ix_(i, j)] for i, j in ((a,a), (a,b), (b,a), (b,b))]
             for p in (pi, po)]

    def _exact_singular(which):
        """Certify singularity over the exact binary64 inputs."""
        def _pair(z):
            return Fraction(float(z.real)), Fraction(float(z.imag))
        def _plus(x, y):
            return x[0]+y[0], x[1]+y[1]
        def _times(x, y):
            return x[0]*y[0]-x[1]*y[1], x[0]*y[1]+x[1]*y[0]
        normal = [[[_pair(z) for z in row] for row in p[0]] for p in local]
        if which < 2:
            matrix = normal[which]
        else:
            inside, outside = normal
            d = len(a)
            matrix = []
            for n in range(k):
                for i in range(d):
                    row = []
                    for m in range(k):
                        tm = _pair(t[n,m])
                        for j in range(d):
                            delta = (outside[i][j][0]-inside[i][j][0],
                                     outside[i][j][1]-inside[i][j][1])
                            row.append(_plus(inside[i][j] if n == m else (Fraction(0), Fraction(0)),
                                             _times(tm, delta)))
                    matrix.append(row)
        d = len(matrix)
        real = [[matrix[i][j][0] for j in range(d)]+[-matrix[i][j][1] for j in range(d)] for i in range(d)]
        real += [[matrix[i][j][1] for j in range(d)]+[matrix[i][j][0] for j in range(d)] for i in range(d)]
        for col in range(2*d):
            pivot = next((r for r in range(col, 2*d) if real[r][col]), None)
            if pivot is None:
                return True
            real[col], real[pivot] = real[pivot], real[col]
            for row in range(col+1, 2*d):
                if real[row][col]:
                    scale = real[row][col]/real[col][col]
                    for j in range(col+1, 2*d):
                        real[row][j] -= scale*real[col][j]
                    real[row][col] = Fraction(0)
        return False

    precision = 64
    previous = None
    rank_checked = False
    while True:
        retry = False
        needed = 2*precision
        with decimal.localcontext() as context:
            context.prec = precision
            largest = Decimal(1)
            condition = Decimal(1)

            def _record(x):
                nonlocal largest
                largest = max(largest, _df34_norm(x))
                return x

            def _inverse(x, which):
                nonlocal condition
                result = _df34_inv(x, which, _NeedMorePrecision)
                _record(result)
                condition = max(condition, _df34_norm(x)*_df34_norm(result))
                return result

            try:
                br, cr = _df34_mat(local[1][1]), _df34_mat(local[1][2])
                parts = []
                for which, arrays in enumerate(local):
                    aa, bb, cc, dd = map(_df34_mat, arrays)
                    q = _inverse(aa, which)
                    db, dc = _df34_sub(bb, br), _df34_sub(cc, cr)
                    r = _record(_df34_mm(q, db))
                    l = _record(_df34_mm(dc, q))
                    w = _record(_df34_sub(dd, _df34_mm(l, db)))
                    parts.append((q, r, l, w))
                mt = _df34_mat(t)
                q, r, l, w = [_record(_df34_lift(mt, u, v)) for u, v in zip(*parts)]
                v = _inverse(q, 2)
                vr = _record(_df34_mm(v, r))
                lv = _record(_df34_mm(l, v))
                bb = _record(_df34_add(w, _df34_mm(lv, r)))
                ab = _df34_add(_df34_common(br, k), vr)
                ba = _df34_add(_df34_common(cr, k), lv)
            except _NeedMorePrecision as exc:
                if _exact_singular(exc.which):
                    raise ValueError('singular') from exc
                retry = True

            if not retry:
                needed = max(64, _df34_digits(largest)+_df34_digits(condition)+40)
                if needed > precision:
                    if not rank_checked:
                        if any(_exact_singular(which) for which in range(3)):
                            raise ValueError('singular')
                        rank_checked = True
                    retry = True
                else:
                    ai = [c*k*s+n*s+h for n in range(k) for c in (axis, axis+3) for h in range(s)]
                    bi = [c*k*s+n*s+h for n in range(k) for c in range(6) if c not in (axis, axis+3) for h in range(s)]
                    out = np.empty((6*s*k, 6*s*k), dtype=complex)
                    for block, rows, cols in ((v,ai,ai), (ab,ai,bi), (ba,bi,ai), (bb,bi,bi)):
                        converted = np.array([[complex(float(x), float(y)) for x, y in zip(xr, xi)]
                                              for xr, xi in zip(block[0], block[1])], dtype=complex)
                        out[np.ix_(rows, cols)] = converted
        if retry:
            precision = max(2*precision, needed)
            continue
        if not np.all(np.isfinite(out)):
            raise ValueError('nonfinite')
        if previous is not None and np.all(np.abs(out-previous) <= 1e-12+1e-10*np.abs(out)):
            return out
        previous = out
        precision *= 2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    cases=[]
    for s,k,axis in [(1,3,0),(3,3,1),(2,2,2),(1,1,0)]:
        setup=f'import numpy as np\nrng=np.random.default_rng({100+s+k+axis})\ns={s};k={k}\nx=rng.normal(size=(6*s,6*s))+1j*rng.normal(size=(6*s,6*s))\np=x@x.conj().T/(6*s)+np.eye(6*s)\nx=rng.normal(size=(6*s,6*s))+1j*rng.normal(size=(6*s,6*s))\nb=x@x.conj().T/(6*s)+2*np.eye(6*s)\nu,_=np.linalg.qr(rng.normal(size=(k,k))+1j*rng.normal(size=(k,k)))\nt=u@np.diag(np.linspace(.08,.91,k))@u.conj().T\n'
        args=f'p,b,t,{axis}'
        cases.append({'setup':setup,'call':f'directional_factorization(p.copy(),b.copy(),t.copy(),{axis})','gold_call':f'_oracle_directional_factorization({args})'})
    for t in ('np.zeros((3,3))','np.eye(3)'):
        cases.append({'setup':'import numpy as np','call':f'directional_factorization(2*np.eye(6),np.eye(6),{t},1)','gold_call':f'_oracle_directional_factorization(2*np.eye(6),np.eye(6),{t},1)'})
    setup='import numpy as np\np=np.eye(6,dtype=complex);p[0,0]=1e-18\np_model=p.copy()'
    cases.append({'setup':setup,'call':'directional_factorization(p_model,p_model,np.eye(1),0)','gold_call':'_oracle_directional_factorization(p,p,np.eye(1),0)'})
    for small,axis in [(1e-12,0),(1e-16,1),(1e-20,2)]:
        setup=f'import numpy as np\np=np.eye(6,dtype=complex);b=2*np.eye(6,dtype=complex)\nj={axis};h=(j+1)%3\np[j,j]={small};b[j,j]=2*{small}\np[j,h]=p[h,j]=.7;b[j,h]=b[h,j]=.7000000001\np[j+3,h+3]=p[h+3,j+3]=.2+.1j;b[j+3,h+3]=b[h+3,j+3]=-.1+.2j\nt=np.array([[.5,.15j],[-.15j,.5]])'
        cases.append({'setup':setup,'call':f'directional_factorization(p.copy(),b.copy(),t.copy(),{axis})','gold_call':f'_oracle_directional_factorization(p,b,t,{axis})'})
    setup='import numpy as np\np=np.eye(6,dtype=complex);b=2*np.eye(6,dtype=complex)\np[0,0]=1e-20;b[0,0]=2e-20\np[0,1]=p[1,0]=.7;b[0,1]=b[1,0]=.7000000001\nt=np.array([[.5,.15j],[-.15j,.5]])'
    cases.append({'setup':setup,'call':'abs(directional_factorization(p.copy(),b.copy(),t.copy(),0)[2,2]-1.346801321449919)<=1e-9+1e-7*1.346801321449919','gold_call':'abs(_oracle_directional_factorization(p,b,t,0)[2,2]-1.346801321449919)<=1e-9+1e-7*1.346801321449919'})
    for a in ('np.eye(5),np.eye(5),np.eye(1),0','np.eye(6),np.eye(6),np.eye(1),True','np.eye(6),np.eye(6),2*np.eye(1),0','np.zeros((6,6)),np.eye(6),np.eye(1),0','np.eye(6),-np.eye(6),.5*np.eye(1),0','np.full((6,6),np.nan),np.eye(6),np.eye(1),0'):
        setup='import numpy as np\n'
        for name,fn in [('run_model','directional_factorization'),('run_gold','_oracle_directional_factorization')]:
            setup+=f'def {name}():\n    try:\n        {fn}({a})\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()'})
    for case in cases[:10]:
        case['tol']=1e-7
    regression_setups = [
        (
            'import numpy as np\n'
            'p_out=np.diag([1e-20,1.23456789,1.,1e-20,1.,1.]).astype(complex)\n'
            'p_out[0,1]=p_out[1,0]=.7\n'
            'p_in=p_out.copy()\n'
            'p_in[0,1]=p_in[1,0]=.7001\n'
            't=np.eye(1)\n'
        ),
        (
            'import numpy as np\n'
            'p_out=np.diag([1e-20,1.,1.,1e-20,1.,1.]).astype(complex)\n'
            'p_in=p_out.copy()\n'
            'p_in[0,1]=p_in[1,0]=.5\n'
            't=np.array([[.36,.48],[.48,.64]])\n'
        ),
    ]
    for setup in regression_setups:
        cases.append({
            'setup': setup,
            'call': 'directional_factorization(p_in.copy(),p_out.copy(),t.copy(),0)',
            'gold_call': '_oracle_directional_factorization(p_in,p_out,t,0)',
            'tol': 1e-7,
        })
    return cases
