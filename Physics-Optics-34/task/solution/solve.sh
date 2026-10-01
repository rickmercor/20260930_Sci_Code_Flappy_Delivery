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


def material_operator(epsilon: "np.ndarray | list | tuple", mu: "np.ndarray | list | tuple", xi: "np.ndarray | list | tuple", eta: float) -> "np.ndarray":
    try:
        arrays=[np.asarray(x,dtype=complex) for x in (epsilon,mu,xi)]
        loss=np.asarray(eta)
        if loss.ndim or np.iscomplexobj(loss):raise ValueError('loss')
        loss=float(loss)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError('invalid input') from exc
    if not np.isfinite(loss) or not 0<=loss<=1:raise ValueError('loss')
    if any(x.shape!=(3,3) or not np.all(np.isfinite(x)) or np.max(np.abs(x))>1e6 for x in arrays):raise ValueError('tensor')
    e,m,x=arrays
    return np.block([[e,x],[x.conj().T,m]])+1j*loss*np.eye(6)

import numpy as np


def indicator_matrix(order: int, fraction: float, center: float) -> "np.ndarray":
    if isinstance(order,(bool,np.bool_)) or not isinstance(order,(int,np.integer)) or not 0<=order<=4:raise ValueError('order')
    try:
        values=[]
        for x in (fraction,center):
            a=np.asarray(x)
            if a.ndim or np.iscomplexobj(a):raise ValueError('real scalar')
            values.append(float(a))
        f,c=values
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError('scalar') from exc
    if not np.isfinite(f) or not np.isfinite(c) or not 0<=f<=1 or not -1<=c<=1:raise ValueError('range')
    h=np.arange(-order,order+1);d=h[:,None]-h[None,:]
    if f==0:return np.zeros(d.shape,dtype=complex)
    if f==1:return np.eye(len(h),dtype=complex)
    return f*np.sinc(d*f)*np.exp(-2j*np.pi*d*c)

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


def directional_factorization(p_in: "np.ndarray | list | tuple", p_out: "np.ndarray | list | tuple", indicator: "np.ndarray | list | tuple", axis: int) -> "np.ndarray":
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

import numpy as np


def rectangle_operator(p_in: "np.ndarray | list | tuple", p_out: "np.ndarray | list | tuple", tx: "np.ndarray | list | tuple", ty: "np.ndarray | list | tuple") -> "np.ndarray":
    try:pi,po,x,y=[np.asarray(a,dtype=complex) for a in (p_in,p_out,tx,ty)]
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError('array') from exc
    if pi.shape!=(6,6) or po.shape!=(6,6):raise ValueError('material shape')
    if any(a.ndim!=2 or a.shape[0]!=a.shape[1] or len(a)%2!=1 or not 1<=len(a)<=9 for a in (x,y)):raise ValueError('indicator shape')
    if len(x)*len(y)>25:raise ValueError('harmonic count')
    first=directional_factorization(pi,po,x,0)
    return directional_factorization(first,np.kron(po,np.eye(len(x))),y,1)

import numpy as np


def maxwell_operator(p: "np.ndarray | list | tuple", mx: int, my: int, period_ratio: float, frequency: float, qx: float, qy: float) -> "np.ndarray":
    if any(isinstance(a,(bool,np.bool_)) or not isinstance(a,(int,np.integer)) or not 0<=a<=4 for a in (mx,my)):raise ValueError('order')
    ng=(2*mx+1)*(2*my+1)
    if ng>25:raise ValueError('size')
    try:
        p=np.asarray(p,dtype=complex)
        vals=[]
        for a in (period_ratio,frequency,qx,qy):
            x=np.asarray(a)
            if x.ndim or np.iscomplexobj(x):raise ValueError('scalar')
            vals.append(float(x))
        r,f,qx,qy=vals
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError('input') from exc
    if not all(np.isfinite(vals)) or not .25<=r<=4 or not .25<=f<=20 or not -2<=qx<=2 or not -2<=qy<=2:raise ValueError('range')
    if p.shape!=(6*ng,6*ng) or not np.all(np.isfinite(p)) or np.max(np.abs(p))>1e6:raise ValueError('matrix')
    ids=np.arange(6*ng).reshape(6,ng);normal=ids[[2,5]].ravel();tang=ids[[0,1,3,4]].ravel()
    ms=np.tile(np.arange(-mx,mx+1),2*my+1);ns=np.repeat(np.arange(-my,my+1),2*mx+1)
    x=np.diag(qx+2*np.pi*ms/f);y=np.diag(qy+2*np.pi*ns/(r*f));zero=np.zeros((ng,ng))
    cz=np.block([[zero,zero,y,-x],[-y,x,zero,zero]])
    try:
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            z=np.linalg.solve(p[np.ix_(normal,normal)],cz-p[np.ix_(normal,tang)])
            u=p[np.ix_(tang,tang)]+p[np.ix_(tang,normal)]@z
            dx,dy,bx,by=np.split(u,4,axis=0);ez,hz=np.split(z,2,axis=0)
            out=np.vstack([by+x@ez,-bx+y@ez,-dy+x@hz,dx+y@hz])
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(out)):raise ValueError('nonfinite')
    except np.linalg.LinAlgError as exc:raise ValueError('singular') from exc
    return out

import numpy as np


def spectral_abscissa(matrix: "np.ndarray | list | tuple") -> float:
    try:a=np.asarray(matrix,dtype=complex)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError('matrix') from exc
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not 1<=len(a)<=100 or not np.all(np.isfinite(a)) or np.max(np.abs(a))>1e6:raise ValueError('matrix')
    try:v=np.linalg.eigvals(a)
    except np.linalg.LinAlgError as exc:raise ValueError('eigensolver') from exc
    if not np.all(np.isfinite(v)):raise ValueError('spectrum')
    return float(np.max(np.real(v)))

import numpy as np


def run_pipeline(mx: int, my: int, fx: float, fy: float) -> float:
    e1=np.array([[9.,1.3,.7],[1.3,7.,-.8],[.7,-.8,6.]])
    m1=np.array([[1.7,.18,-.12],[.18,1.3,.15],[-.12,.15,1.5]])
    x1=np.array([[1.2j,.55+.25j,.2-.3j],[-.25+.4j,.9j,.45+.15j],[.35+.1j,-.3+.2j,1.1j]])
    e0=np.array([[2.4,.15,-.05],[.15,2.1,.12],[-.05,.12,2.7]])
    m0=np.array([[1.1,.04,0],[.04,1.2,-.03],[0,-.03,1.05]])
    x0=np.array([[.15j,.08,.03j],[-.04,.2j,.06],[.02j,-.05,.1j]])
    p1=material_operator(e1,m1,x1,.025)
    p0=material_operator(e0,m0,x0,.025)
    tx=indicator_matrix(mx,fx,.11);ty=indicator_matrix(my,fy,-.08)
    p=rectangle_operator(p1,p0,tx,ty)
    mat=maxwell_operator(p,mx,my,1.3,3.3,.21,-.17)
    return spectral_abscissa(mat)

def _final_answer_case():
    return {'setup':'import numpy as np','gold_call':'run_pipeline(1,1,.43,.37)','extract':'round(result,6)'}
SCICODE_GOLD_EOF
