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
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def bhz_layer(coords: ArrayLike, edges: ArrayLike, mass: ArrayLike, potential: ArrayLike, t1: float, delta: float) -> np.ndarray:
    def _checked_coords(coords, distinct=False):
        r = _checked_numeric(coords, 'coords')
        if r.ndim != 2 or r.shape[1] != 2 or len(r) == 0:
            raise ValueError('coords must have shape (N,2), N>0')
        if distinct and len(np.unique(r, axis=0)) != len(r):
            raise ValueError('vertex coordinates must be distinct')
        return r

    def _checked_edges(edges, n):
        try:
            e = np.asarray(edges)
        except (TypeError, ValueError) as exc:
            raise ValueError('edges must be an integer array of shape (B,2)') from exc
        if e.dtype.kind not in 'iu' or e.ndim != 2 or e.shape[1] != 2:
            raise ValueError('edges must be an integer array of shape (B,2), including B=0')
        if np.any(e < 0) or np.any(e >= n) or np.any(e[:, 0] == e[:, 1]):
            raise ValueError('edge endpoints must be distinct valid vertex indices')
        if len(np.unique(np.sort(e, axis=1), axis=0)) != len(e):
            raise ValueError('each undirected edge must occur once')
        return e.astype(int)

    def _checked_numeric(value, name, real=True):
        try:
            a = np.asarray(value)
            allowed = 'iuf' if real else 'iufc'
            if a.dtype.kind not in allowed or not np.isfinite(a).all():
                raise ValueError(name + ' must contain finite numeric values of the declared type')
            with np.errstate(over='ignore', invalid='ignore'):
                a = a.astype(float if real else complex)
            if not np.isfinite(a).all():
                raise ValueError(name + ' exceeds the supported floating-point range')
            return a
        except (TypeError, OverflowError) as exc:
            raise ValueError(name + ' must be a numeric scalar or array') from exc

    def _checked_scalar(value, name, minimum=None, strict=False, integer=False):
        a = _checked_numeric(value, name)
        if a.shape != ():
            raise ValueError(name + ' must be a scalar')
        x = float(a)
        if integer and (np.asarray(value).dtype.kind not in 'iu' or x != np.floor(x)):
            raise ValueError(name + ' must have integer type')
        if minimum is not None and (x < minimum or (strict and x == minimum)):
            raise ValueError(name + ' is outside its stated domain')
        return int(x) if integer else x

    def _checked_vector(value, n, name, scalar=False):
        a = _checked_numeric(value, name)
        if scalar and a.shape == ():
            return np.full(n, float(a))
        if a.shape != (n,):
            raise ValueError(name + ' has an incompatible shape')
        return a

    r=_checked_coords(coords, distinct=True);n=len(r);edges=_checked_edges(edges,n)
    mass=_checked_vector(mass,n,'mass',scalar=True);potential=_checked_vector(potential,n,'potential',scalar=True)
    t1=_checked_scalar(t1,'t1');delta=_checked_scalar(delta,'delta')
    sx=np.array([[0,1],[1,0]],complex);sy=np.array([[0,-1j],[1j,0]]);sz=np.diag([1.,-1.])
    h=np.zeros((2*n,2*n),complex)
    for j in range(n):h[2*j:2*j+2,2*j:2*j+2]=mass[j]*sz+potential[j]*np.eye(2)
    for j,k in edges:
        d=r[k]-r[j]; d=d/np.linalg.norm(d)
        b=t1*sz+.5j*delta*(d[0]*sx+d[1]*sy)
        h[2*j:2*j+2,2*k:2*k+2]=b;h[2*k:2*k+2,2*j:2*j+2]=b.conj().T
    return h

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def aii_pencil(layer: ArrayLike, coords: ArrayLike, profile: ArrayLike, probe: ArrayLike, kappa: float) -> np.ndarray:
    def _checked_coords(coords, distinct=False):
        r = _checked_numeric(coords, 'coords')
        if r.ndim != 2 or r.shape[1] != 2 or len(r) == 0:
            raise ValueError('coords must have shape (N,2), N>0')
        if distinct and len(np.unique(r, axis=0)) != len(r):
            raise ValueError('vertex coordinates must be distinct')
        return r

    def _checked_numeric(value, name, real=True):
        try:
            a = np.asarray(value)
            allowed = 'iuf' if real else 'iufc'
            if a.dtype.kind not in allowed or not np.isfinite(a).all():
                raise ValueError(name + ' must contain finite numeric values of the declared type')
            with np.errstate(over='ignore', invalid='ignore'):
                a = a.astype(float if real else complex)
            if not np.isfinite(a).all():
                raise ValueError(name + ' exceeds the supported floating-point range')
            return a
        except (TypeError, OverflowError) as exc:
            raise ValueError(name + ' must be a numeric scalar or array') from exc

    def _checked_scalar(value, name, minimum=None, strict=False, integer=False):
        a = _checked_numeric(value, name)
        if a.shape != ():
            raise ValueError(name + ' must be a scalar')
        x = float(a)
        if integer and (np.asarray(value).dtype.kind not in 'iu' or x != np.floor(x)):
            raise ValueError(name + ' must have integer type')
        if minimum is not None and (x < minimum or (strict and x == minimum)):
            raise ValueError(name + ' is outside its stated domain')
        return int(x) if integer else x

    def _checked_vector(value, n, name, scalar=False):
        a = _checked_numeric(value, name)
        if scalar and a.shape == ():
            return np.full(n, float(a))
        if a.shape != (n,):
            raise ValueError(name + ' has an incompatible shape')
        return a

    r=_checked_coords(coords);n=len(r);a=_checked_numeric(layer,'layer',real=False)
    profile=_checked_vector(profile,n,'profile');probe=_checked_vector(probe,3,'probe')
    kappa=_checked_scalar(kappa,'kappa',minimum=0)
    if a.shape!=(2*n,2*n):raise ValueError('layer must have shape (2N,2N)')
    scale=float(np.max(abs(a)))
    if scale and np.max(abs(a/scale-a.conj().T/scale))>1e-12:
        raise ValueError('layer must be Hermitian to relative max-entry tolerance 1e-12')
    m=2*n;z=np.zeros_like(a)
    sx=np.array([[0,1],[1,0]],complex);sy=np.array([[0,-1j],[1j,0]]);sz=np.diag([1.,-1.])
    h0=np.block([[a,z],[z,a.conj()]]);c=np.kron(np.diag(profile),sy)
    h1=np.block([[z,c],[c.conj().T,z]])
    j=np.block([[np.zeros((m,m)),np.eye(m)],[-np.eye(m),np.zeros((m,m))]])
    f=np.kron(j,1j*sy);q=(np.eye(4*m)-1j*f)/np.sqrt(2)
    x=np.tile(np.repeat(r[:,0],2),2);y=np.tile(np.repeat(r[:,1],2),2)
    l0=np.kron(h0-probe[2]*np.eye(2*m),sz)+kappa*np.kron(np.diag(x-probe[0]),sx)+kappa*np.kron(np.diag(y-probe[1]),sy)
    result=np.array([1j*q.conj().T@l@q for l in (l0,np.kron(h1,sz))]).real
    return (result-result.transpose(0,2,1))/2

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def paired_ordering(skew: ArrayLike) -> np.ndarray:
    def _checked_numeric(value, name, real=True):
        try:
            a = np.asarray(value)
            allowed = 'iuf' if real else 'iufc'
            if a.dtype.kind not in allowed or not np.isfinite(a).all():
                raise ValueError(name + ' must contain finite numeric values of the declared type')
            with np.errstate(over='ignore', invalid='ignore'):
                a = a.astype(float if real else complex)
            if not np.isfinite(a).all():
                raise ValueError(name + ' exceeds the supported floating-point range')
            return a
        except (TypeError, OverflowError) as exc:
            raise ValueError(name + ' must be a numeric scalar or array') from exc

    def _checked_skew(skew):
        a = _checked_numeric(skew, 'skew')
        if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) == 0 or len(a) % 2:
            raise ValueError('skew must be square with positive even size')
        scale = float(np.max(abs(a)))
        if scale and np.max(abs(a / scale + a.T / scale)) > 1e-12:
            raise ValueError('matrix must be skew-symmetric to relative max-entry tolerance 1e-12')
        return a

    a=_checked_skew(skew)
    # Maximum-cardinality graph matching by Edmonds alternating forests.
    n=len(a);graph=[list(np.flatnonzero(a[i]!=0)) for i in range(n)]
    match=[-1]*n
    for root in range(n):
        if match[root]!=-1:continue
        parent=[-1]*n;base=list(range(n));used=[False]*n;queue=[root];used[root]=True;found=-1
        def _lca(v,w):
            seen=[False]*n
            while True:
                v=base[v];seen[v]=True
                if match[v]==-1:break
                v=parent[match[v]]
            while not seen[base[w]]:w=parent[match[base[w]]]
            return base[w]
        def _mark(v,b,child,blossom):
            while base[v]!=b:
                blossom[base[v]]=blossom[base[match[v]]]=True
                parent[v]=child;child=match[v];v=parent[match[v]]
        for v in queue:
            for w in graph[v]:
                if base[v]==base[w] or match[v]==w:continue
                if w==root or (match[w]!=-1 and parent[match[w]]!=-1):
                    b=_lca(v,w);blossom=[False]*n
                    _mark(v,b,w,blossom);_mark(w,b,v,blossom)
                    for i in range(n):
                        if blossom[base[i]]:
                            base[i]=b
                            if not used[i]:used[i]=True;queue.append(i)
                elif parent[w]==-1:
                    parent[w]=v
                    if match[w]==-1:found=w;break
                    w=match[w];used[w]=True;queue.append(w)
            if found!=-1:break
        while found!=-1:
            v=parent[found];nxt=match[v] if v!=-1 else -1
            match[found]=v
            if v!=-1:match[v]=found
            found=nxt
    pairs=[(i,match[i]) for i in range(n) if match[i]>i]
    free=[i for i in range(n) if match[i]==-1]
    pairs+=list(zip(free[::2],free[1::2]));pairs=sorted(pairs)
    m=len(pairs);adj=[set() for _ in range(m)]
    for i in range(m):
        for j in range(i+1,m):
            if np.any(a[np.ix_(pairs[i],pairs[j])]!=0):adj[i].add(j);adj[j].add(i)
    remaining=set(range(m));order=[]
    while remaining:
        k=min(remaining,key=lambda j:(len(adj[j]&remaining),pairs[j]))
        nbr=adj[k]&remaining
        for j in nbr:adj[j].update(nbr-{j})
        remaining.remove(k);order.extend(pairs[k])
    return np.asarray(order,int)

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def skew_factor(skew: ArrayLike, ordering: ArrayLike) -> np.ndarray:
    def _checked_numeric(value, name, real=True):
        try:
            a = np.asarray(value)
            allowed = 'iuf' if real else 'iufc'
            if a.dtype.kind not in allowed or not np.isfinite(a).all():
                raise ValueError(name + ' must contain finite numeric values of the declared type')
            with np.errstate(over='ignore', invalid='ignore'):
                a = a.astype(float if real else complex)
            if not np.isfinite(a).all():
                raise ValueError(name + ' exceeds the supported floating-point range')
            return a
        except (TypeError, OverflowError) as exc:
            raise ValueError(name + ' must be a numeric scalar or array') from exc

    def _checked_permutation(ordering, n, packed=False):
        a = _checked_numeric(ordering, 'permutation')
        if a.shape != (n,) or (not packed and np.asarray(ordering).dtype.kind not in 'iu'):
            raise ValueError('permutation must have the declared shape and integer type')
        if np.any(a != np.floor(a)) or np.any(a < 0) or np.any(a >= n):
            raise ValueError('permutation entries must be integer indices from 0 through n-1')
        p = a.astype(int)
        if not np.array_equal(np.sort(p), np.arange(n)):
            raise ValueError('permutation must contain each index exactly once')
        return p

    def _checked_skew(skew):
        a = _checked_numeric(skew, 'skew')
        if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) == 0 or len(a) % 2:
            raise ValueError('skew must be square with positive even size')
        scale = float(np.max(abs(a)))
        if scale and np.max(abs(a / scale + a.T / scale)) > 1e-12:
            raise ValueError('matrix must be skew-symmetric to relative max-entry tolerance 1e-12')
        return a

    s=_checked_skew(skew);n=len(s);p=_checked_permutation(ordering,n).copy();scale=float(np.max(abs(s)))
    l=np.eye(n);d=np.zeros(n//2)
    if scale==0:
        out=np.zeros((n+2,n));out[:n]=l;out[n]=p;return out
    a=s[np.ix_(p,p)]/scale
    def _swap(i,j,k):
        if i==j:return
        a[[i,j],:]=a[[j,i],:];a[:,[i,j]]=a[:,[j,i]]
        l[[i,j],:k]=l[[j,i],:k];p[i],p[j]=p[j],p[i]
    for k in range(0,n,2):
        tail=abs(np.triu(a[k:,k:],1));ind=np.unravel_index(np.argmax(tail),tail.shape)
        peak=tail[ind]
        if peak==0:break
        if abs(a[k,k+1])<.1*peak:
            i,j=k+ind[0],k+ind[1];_swap(k,i,k)
            if j==k:j=i
            _swap(k+1,j,k)
        pivot=a[k,k+1];d[k//2]=pivot
        if k+2<n:
            cross=a[k+2:,k:k+2].copy()
            multipliers=np.column_stack((cross[:,1]/pivot,-cross[:,0]/pivot))
            l[k+2:,k:k+2]=multipliers
            t=np.array([[0,pivot],[-pivot,0.]])
            b=a[k+2:,k+2:]-multipliers@t@multipliers.T
            a[k+2:,k+2:]=(b-b.T)/2
    out=np.zeros((n+2,n));out[:n]=l;out[n]=p;out[n+1,:n//2]=d;out[n+1,n//2]=scale
    return out

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def pfaffian_certificate(skew: ArrayLike, factor: ArrayLike, orientation: int = 1) -> np.ndarray:
    def _checked_factor(factor, s):
        n = len(s)
        f = _checked_numeric(factor, 'factor')
        if f.shape != (n + 2, n):
            raise ValueError('factor must have shape (n+2,n)')
        p = _checked_permutation(f[n], n, packed=True)
        l = f[:n]
        if np.max(abs(np.triu(l, 1))) > 1e-12 or np.max(abs(np.diag(l) - 1)) > 1e-12:
            raise ValueError('factor L must be unit lower triangular within 1e-12')
        d, scale = f[n + 1, :n // 2], f[n + 1, n // 2]
        if scale < 0 or np.any(f[n + 1, n // 2 + 1:] != 0):
            raise ValueError('factor scale must be nonnegative and reserved entries must be zero')
        norm = float(np.max(abs(s)))
        if scale == 0 and (norm != 0 or np.any(d != 0)):
            raise ValueError('zero scale is valid only for a zero matrix and zero pivots')
        t = np.zeros((n, n))
        for i, pivot in enumerate(d):
            t[2*i, 2*i+1] = pivot
            t[2*i+1, 2*i] = -pivot
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            expected = s[np.ix_(p, p)] / (norm if norm else 1.)
            rebuilt = l @ (((scale / norm) if norm else scale) * t) @ l.T
        if not np.isfinite(rebuilt).all() or np.max(abs(rebuilt - expected)) > 1e-9:
            raise ValueError('factor must reconstruct the supplied matrix within relative max-entry tolerance 1e-9')
        return f, p

    def _checked_numeric(value, name, real=True):
        try:
            a = np.asarray(value)
            allowed = 'iuf' if real else 'iufc'
            if a.dtype.kind not in allowed or not np.isfinite(a).all():
                raise ValueError(name + ' must contain finite numeric values of the declared type')
            with np.errstate(over='ignore', invalid='ignore'):
                a = a.astype(float if real else complex)
            if not np.isfinite(a).all():
                raise ValueError(name + ' exceeds the supported floating-point range')
            return a
        except (TypeError, OverflowError) as exc:
            raise ValueError(name + ' must be a numeric scalar or array') from exc

    def _checked_orientation(orientation):
        sign = _checked_scalar(orientation, 'orientation', integer=True)
        if sign not in (-1, 1):
            raise ValueError('orientation must be integer +1 or -1')
        return sign

    def _checked_permutation(ordering, n, packed=False):
        a = _checked_numeric(ordering, 'permutation')
        if a.shape != (n,) or (not packed and np.asarray(ordering).dtype.kind not in 'iu'):
            raise ValueError('permutation must have the declared shape and integer type')
        if np.any(a != np.floor(a)) or np.any(a < 0) or np.any(a >= n):
            raise ValueError('permutation entries must be integer indices from 0 through n-1')
        p = a.astype(int)
        if not np.array_equal(np.sort(p), np.arange(n)):
            raise ValueError('permutation must contain each index exactly once')
        return p

    def _checked_scalar(value, name, minimum=None, strict=False, integer=False):
        a = _checked_numeric(value, name)
        if a.shape != ():
            raise ValueError(name + ' must be a scalar')
        x = float(a)
        if integer and (np.asarray(value).dtype.kind not in 'iu' or x != np.floor(x)):
            raise ValueError(name + ' must have integer type')
        if minimum is not None and (x < minimum or (strict and x == minimum)):
            raise ValueError(name + ' is outside its stated domain')
        return int(x) if integer else x

    def _checked_skew(skew):
        a = _checked_numeric(skew, 'skew')
        if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) == 0 or len(a) % 2:
            raise ValueError('skew must be square with positive even size')
        scale = float(np.max(abs(a)))
        if scale and np.max(abs(a / scale + a.T / scale)) > 1e-12:
            raise ValueError('matrix must be skew-symmetric to relative max-entry tolerance 1e-12')
        return a

    s=_checked_skew(skew);n=len(s);f,p=_checked_factor(factor,s);orientation=_checked_orientation(orientation)
    piv=f[n+1,:n//2];scale=f[n+1,n//2]
    if scale==0 or np.any(piv==0):return np.array([0.,0.,0.])
    parity=(-1)**sum(np.count_nonzero(p[i+1:]<p[i]) for i in range(n))
    sg=float(orientation*parity*np.prod(np.sign(piv)))
    logabs=float(np.sum(np.log(abs(piv)))+(n//2)*np.log(scale))
    # Scaling protects this diagnostic under uniform underflow/overflow regimes.
    gap=float(np.min(abs(eigvalsh(1j*(s/scale))))*scale)
    return np.array([sg,logabs,gap])

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def pencil_crossings(pencil: ArrayLike, lower: float, upper: float) -> np.ndarray:
    def _checked_interval(lower, upper):
        lo = _checked_scalar(lower, 'lower')
        hi = _checked_scalar(upper, 'upper')
        if lo >= hi:
            raise ValueError('bounds must be strictly increasing')
        return lo, hi

    def _checked_numeric(value, name, real=True):
        try:
            a = np.asarray(value)
            allowed = 'iuf' if real else 'iufc'
            if a.dtype.kind not in allowed or not np.isfinite(a).all():
                raise ValueError(name + ' must contain finite numeric values of the declared type')
            with np.errstate(over='ignore', invalid='ignore'):
                a = a.astype(float if real else complex)
            if not np.isfinite(a).all():
                raise ValueError(name + ' exceeds the supported floating-point range')
            return a
        except (TypeError, OverflowError) as exc:
            raise ValueError(name + ' must be a numeric scalar or array') from exc

    def _checked_pencil(pencil):
        p = _checked_numeric(pencil, 'pencil')
        if p.ndim != 3 or p.shape[0] != 2:
            raise ValueError('pencil must have shape (2,n,n)')
        for a in p:
            _checked_skew(a)
        return p

    def _checked_scalar(value, name, minimum=None, strict=False, integer=False):
        a = _checked_numeric(value, name)
        if a.shape != ():
            raise ValueError(name + ' must be a scalar')
        x = float(a)
        if integer and (np.asarray(value).dtype.kind not in 'iu' or x != np.floor(x)):
            raise ValueError(name + ' must have integer type')
        if minimum is not None and (x < minimum or (strict and x == minimum)):
            raise ValueError(name + ' is outside its stated domain')
        return int(x) if integer else x

    def _checked_skew(skew):
        a = _checked_numeric(skew, 'skew')
        if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) == 0 or len(a) % 2:
            raise ValueError('skew must be square with positive even size')
        scale = float(np.max(abs(a)))
        if scale and np.max(abs(a / scale + a.T / scale)) > 1e-12:
            raise ValueError('matrix must be skew-symmetric to relative max-entry tolerance 1e-12')
        return a

    def _exact_fallback(a, b, lower, upper):
        from decimal import Decimal, localcontext
        from fractions import Fraction
        from math import cos, pi, sin

        def _trim(poly):
            while len(poly) > 1 and poly[-1] == 0:
                poly.pop()
            return poly

        def _divide(numerator, denominator):
            remainder = numerator.copy()
            quotient = [Fraction(0)] * max(1, len(numerator) - len(denominator) + 1)
            while any(remainder) and len(remainder) >= len(denominator):
                shift = len(remainder) - len(denominator)
                coefficient = remainder[-1] / denominator[-1]
                quotient[shift] = coefficient
                for index, value in enumerate(denominator):
                    remainder[index + shift] -= coefficient * value
                _trim(remainder)
            return _trim(quotient), _trim(remainder)

        def _pfaffian(matrix):
            matrix = [row.copy() for row in matrix]
            result = Fraction(1)
            for k in range(0, len(matrix), 2):
                partner = next((j for j in range(k + 1, len(matrix)) if matrix[k][j]), None)
                if partner is None:
                    return Fraction(0)
                if partner != k + 1:
                    matrix[k + 1], matrix[partner] = matrix[partner], matrix[k + 1]
                    for row in matrix:
                        row[k + 1], row[partner] = row[partner], row[k + 1]
                    result = -result
                pivot = matrix[k][k + 1]
                result *= pivot
                for i in range(k + 2, len(matrix)):
                    for j in range(i + 1, len(matrix)):
                        matrix[i][j] -= (
                            matrix[k][i] * matrix[k + 1][j]
                            - matrix[k][j] * matrix[k + 1][i]
                        ) / pivot
                        matrix[j][i] = -matrix[i][j]
            return result

        def _add(z, w):
            return z[0] + w[0], z[1] + w[1]

        def _subtract(z, w):
            return z[0] - w[0], z[1] - w[1]

        def _multiply(z, w):
            return z[0] * w[0] - z[1] * w[1], z[0] * w[1] + z[1] * w[0]

        def _quotient(z, w):
            norm = w[0] * w[0] + w[1] * w[1]
            return (z[0] * w[0] + z[1] * w[1]) / norm, (z[1] * w[0] - z[0] * w[1]) / norm

        def _norm(z):
            return abs(z[0]) + abs(z[1])

        def _decimal(value):
            return Decimal(value.numerator) / Decimal(value.denominator)

        def _simple_roots(poly, precision):
            coefficients = [_decimal(value / poly[-1]) for value in poly]
            degree = len(coefficients) - 1
            zero, one = Decimal(0), Decimal(1)
            if degree == 1:
                return [(-coefficients[0], zero)]
            if degree == 2:
                # Determine the discriminant sign exactly, including tiny pairs.
                discriminant = poly[1] ** 2 - 4 * poly[0] * poly[2]
                middle = -_decimal(poly[1] / (2 * poly[2]))
                offset = _decimal(abs(discriminant)).sqrt() / (2 * abs(_decimal(poly[2])))
                if discriminant < 0:
                    return [(middle, offset), (middle, -offset)]
                return [(middle - offset, zero), (middle + offset, zero)]

            magnitude = max(abs(value) for value in poly)
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                approximate = np.roots([float(value / magnitude) for value in poly[::-1]])
            if len(approximate) == degree and np.isfinite(approximate).all():
                roots = [(Decimal(float(value.real)), Decimal(float(value.imag))) for value in approximate]
            else:
                radius = one + max(abs(value) for value in coefficients[:-1])
                roots = [
                    (radius * Decimal(cos(2 * pi * (i + .25) / degree)),
                     radius * Decimal(sin(2 * pi * (i + .25) / degree)))
                    for i in range(degree)
                ]
            for i in range(degree):
                if roots[i] in roots[:i]:
                    roots[i] = _add(roots[i], (Decimal('1e-8') * (i + 1), Decimal('1e-8')))

            tolerance = Decimal(10) ** (-(precision // 2))
            # Aberth's simultaneous correction keeps nearby simple roots distinct.
            for _ in range(400 + 20 * degree):
                largest = zero
                for i, root in enumerate(roots):
                    value, derivative = (one, zero), (zero, zero)
                    for coefficient in coefficients[-2::-1]:
                        derivative = _add(_multiply(derivative, root), value)
                        value = _add(_multiply(value, root), (coefficient, zero))
                    if _norm(value) == 0:
                        continue
                    if _norm(derivative) == 0:
                        return None
                    newton = _quotient(value, derivative)
                    repulsion = (zero, zero)
                    for j, other in enumerate(roots):
                        if i != j:
                            difference = _subtract(root, other)
                            if _norm(difference) == 0:
                                return None
                            repulsion = _add(repulsion, _quotient((one, zero), difference))
                    denominator = _subtract((one, zero), _multiply(newton, repulsion))
                    if _norm(denominator) == 0:
                        return None
                    correction = _quotient(newton, denominator)
                    roots[i] = _subtract(root, correction)
                    largest = max(largest, _norm(correction))
                if largest <= tolerance:
                    # Weierstrass corrections also detect missing/duplicated roots.
                    residual = zero
                    for i, root in enumerate(roots):
                        value = (one, zero)
                        denominator = (one, zero)
                        for coefficient in coefficients[-2::-1]:
                            value = _add(_multiply(value, root), (coefficient, zero))
                        for j, other in enumerate(roots):
                            if i != j:
                                denominator = _multiply(denominator, _subtract(root, other))
                        if _norm(denominator) == 0:
                            return None
                        residual = max(residual, _norm(_quotient(value, denominator)))
                    if residual <= tolerance * degree:
                        return roots
            return None

        def _classify(roots, center, radius):
            equality, separation = Decimal('1e-10'), Decimal('1e-5')
            lo, hi = Decimal(lower), Decimal(upper)
            interior = []
            for real, imaginary in roots:
                real, imaginary = center + radius * real, radius * imaginary
                if abs(imaginary) > equality:
                    if abs(imaginary) <= separation:
                        return 'complex roots must be separated from the real axis by more than 1e-5', []
                    continue
                if any(equality < abs(real - endpoint) <= separation for endpoint in (lo, hi)):
                    return 'non-endpoint roots must be separated from endpoints by more than 1e-5', []
                if lo + equality < real < hi - equality:
                    interior.append(real)
            interior.sort()
            groups = []
            for root in interior:
                if groups and root - groups[-1][0] <= equality:
                    groups[-1].append(root)
                else:
                    groups.append([root])
            unique = [sum(group) / len(group) for group in groups]
            if any(right - left <= separation for left, right in zip(unique, unique[1:])):
                return 'distinct real crossings must be separated by more than 1e-5', []
            return None, [float(value) for value in unique]

        n = len(a)
        center = (Fraction(lower) + Fraction(upper)) / 2
        radius = (Fraction(upper) - Fraction(lower)) / 2
        # Remove only the roundoff-scale symmetric components permitted by input validation.
        exact_a = [[(Fraction(float(a[i, j])) - Fraction(float(a[j, i]))) / 2 for j in range(n)] for i in range(n)]
        exact_b = [[(Fraction(float(b[i, j])) - Fraction(float(b[j, i]))) / 2 for j in range(n)] for i in range(n)]
        differences = [
            _pfaffian([[exact_a[i][j] + (center + radius * x) * exact_b[i][j] for j in range(n)] for i in range(n)])
            for x in range(n // 2 + 1)
        ]
        polynomial = [Fraction(0)] * (n // 2 + 1)
        basis = [Fraction(1)]
        for order in range(n // 2 + 1):
            for index, coefficient in enumerate(basis):
                polynomial[index] += differences[0] * coefficient
            differences = [right - left for left, right in zip(differences, differences[1:])]
            next_basis = [Fraction(0)] * (len(basis) + 1)
            for index, coefficient in enumerate(basis):
                next_basis[index] -= order * coefficient / (order + 1)
                next_basis[index + 1] += coefficient / (order + 1)
            basis = next_basis
        _trim(polynomial)
        if not any(polynomial):
            raise ValueError('pencil must be regular, with determinant not identically zero')
        if len(polynomial) == 1:
            return []
        derivative = [index * value for index, value in enumerate(polynomial)][1:]
        first, second = polynomial.copy(), derivative
        while any(second):
            _, remainder = _divide(first, second)
            first, second = second, remainder
        square_free, _ = _divide(polynomial, [value / first[-1] for value in first])

        previous = None
        precision = 96
        while True:
            with localcontext() as context:
                context.prec = precision
                roots = _simple_roots(square_free, precision)
                if roots is not None:
                    classified = _classify(roots, _decimal(center), _decimal(radius))
                    if previous is not None and classified[0] == previous[0] and len(classified[1]) == len(previous[1]):
                        if all(abs(x - y) <= 1e-12 for x, y in zip(classified[1], previous[1])):
                            if classified[0] is not None:
                                raise ValueError(classified[0])
                            return classified[1]
                    previous = classified
            precision = precision * 2

    p = _checked_pencil(pencil)
    lower, upper = _checked_interval(lower, upper)
    n = p.shape[1]
    a, b = p
    values = eigvals(-a, b, homogeneous_eigvals=True)
    ambiguous = bool(np.any((values[0] == 0) & (values[1] == 0)))
    equality = 1e-10
    scale = max(float(np.max(abs(a))), float(np.max(abs(b))), 1e-300)
    roots = []
    determinant_groups = []
    for alpha, beta in zip(*values):
        if abs(beta) == 0:
            continue
        root = alpha / beta
        if not np.isfinite(root):
            continue
        for group in determinant_groups:
            if abs(root - group[0]) <= equality:
                group.append(root)
                break
        else:
            determinant_groups.append([root])
        if abs(root.imag) > equality:
            if abs(root.imag) <= 1e-5:
                ambiguous = True
            continue
        root = float(root.real)
        if any(equality < abs(root - endpoint) <= 1e-5 for endpoint in (lower, upper)):
            ambiguous = True
        if lower + equality < root < upper - equality:
            residual = np.min(abs(eigvalsh(1j * (a + root * b) / scale)))
            if residual < 1e-7 * max(1., abs(root)):
                roots.append(root)
            else:
                ambiguous = True
    # Every finite determinant root of a regular skew pencil has even
    # algebraic multiplicity. Lost partners expose larger QZ root splitting,
    # including tangencies whose apparent imaginary parts exceed 1e-5.
    if any(len(group) % 2 for group in determinant_groups):
        ambiguous = True
    roots.sort()
    groups = []
    for root in roots:
        if groups and root - groups[-1][0] <= equality:
            groups[-1].append(root)
        else:
            groups.append([root])
    unique = [sum(group) / len(group) for group in groups]
    if np.any(np.diff(unique) <= 1e-5) or len(unique) > n // 2:
        ambiguous = True
    if ambiguous:
        unique = _exact_fallback(a, b, lower, upper)
    output = np.zeros(n // 2 + 1)
    output[0] = len(unique)
    output[1:len(unique) + 1] = unique
    return output

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def phase_integrals(pencil: ArrayLike, crossings: ArrayLike, lower: float, upper: float, orientation: int = 1, order: int = 64) -> np.ndarray:
    def _checked_interval(lower, upper):
        lo = _checked_scalar(lower, 'lower')
        hi = _checked_scalar(upper, 'upper')
        if lo >= hi:
            raise ValueError('bounds must be strictly increasing')
        return lo, hi

    def _checked_numeric(value, name, real=True):
        try:
            a = np.asarray(value)
            allowed = 'iuf' if real else 'iufc'
            if a.dtype.kind not in allowed or not np.isfinite(a).all():
                raise ValueError(name + ' must contain finite numeric values of the declared type')
            with np.errstate(over='ignore', invalid='ignore'):
                a = a.astype(float if real else complex)
            if not np.isfinite(a).all():
                raise ValueError(name + ' exceeds the supported floating-point range')
            return a
        except (TypeError, OverflowError) as exc:
            raise ValueError(name + ' must be a numeric scalar or array') from exc

    def _checked_orientation(orientation):
        sign = _checked_scalar(orientation, 'orientation', integer=True)
        if sign not in (-1, 1):
            raise ValueError('orientation must be integer +1 or -1')
        return sign

    def _checked_pencil(pencil):
        p = _checked_numeric(pencil, 'pencil')
        if p.ndim != 3 or p.shape[0] != 2:
            raise ValueError('pencil must have shape (2,n,n)')
        for a in p:
            _checked_skew(a)
        return p

    def _checked_scalar(value, name, minimum=None, strict=False, integer=False):
        a = _checked_numeric(value, name)
        if a.shape != ():
            raise ValueError(name + ' must be a scalar')
        x = float(a)
        if integer and (np.asarray(value).dtype.kind not in 'iu' or x != np.floor(x)):
            raise ValueError(name + ' must have integer type')
        if minimum is not None and (x < minimum or (strict and x == minimum)):
            raise ValueError(name + ' is outside its stated domain')
        return int(x) if integer else x

    def _checked_skew(skew):
        a = _checked_numeric(skew, 'skew')
        if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) == 0 or len(a) % 2:
            raise ValueError('skew must be square with positive even size')
        scale = float(np.max(abs(a)))
        if scale and np.max(abs(a / scale + a.T / scale)) > 1e-12:
            raise ValueError('matrix must be skew-symmetric to relative max-entry tolerance 1e-12')
        return a

    p=_checked_pencil(pencil);lower,upper=_checked_interval(lower,upper)
    orientation=_checked_orientation(orientation);order=_checked_scalar(order,'order',minimum=1,integer=True)
    crossings=_checked_numeric(crossings,'crossings');n=p.shape[1]
    if crossings.shape!=(n//2+1,):raise ValueError('crossings must have shape (n/2+1,)')
    count=crossings[0]
    if count!=np.floor(count) or count<0 or count>n//2:raise ValueError('crossing count must be an integer from 0 through n/2')
    count=int(count);roots=crossings[1:count+1]
    if np.any(crossings[count+1:]!=0) or np.any(roots<=lower) or np.any(roots>=upper) or np.any(np.diff(roots)<=0):
        raise ValueError('crossings must be increasing interior locations followed by zero padding')
    expected=pencil_crossings(p,lower,upper)
    if count!=int(expected[0]) or np.any(abs(roots-expected[1:count+1])>1e-6):
        raise ValueError('crossing record must contain all distinct interior closures within 1e-6')
    knots=np.r_[lower,roots,upper]
    nodes,weights=np.polynomial.legendre.leggauss(int(order));out=np.zeros((count+1,4))
    for k,(lo,hi) in enumerate(zip(knots[:-1],knots[1:])):
        mid=(lo+hi)/2;s=p[0]+mid*p[1]
        perm=paired_ordering(s);f=skew_factor(s,perm)
        sg=pfaffian_certificate(s,f,orientation)[0]
        val=0.
        if sg<0:
            for node,weight in zip(nodes,weights):
                c=mid+(hi-lo)*node/2;s=p[0]+c*p[1]
                cert=pfaffian_certificate(s,skew_factor(s,perm),orientation)
                val+=weight*cert[2]
            val*=(hi-lo)/2
        out[k]=[lo,hi,sg,val]
    return out

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def topological_exposure(coords: ArrayLike, edges: ArrayLike, mass: ArrayLike, potential: ArrayLike, profile: ArrayLike, t1: float, delta: float, probe: ArrayLike, kappa: float, cmax: float, order: int = 64) -> float:
    def _checked_numeric(value, name, real=True):
        try:
            a = np.asarray(value)
            allowed = 'iuf' if real else 'iufc'
            if a.dtype.kind not in allowed or not np.isfinite(a).all():
                raise ValueError(name + ' must contain finite numeric values of the declared type')
            with np.errstate(over='ignore', invalid='ignore'):
                a = a.astype(float if real else complex)
            if not np.isfinite(a).all():
                raise ValueError(name + ' exceeds the supported floating-point range')
            return a
        except (TypeError, OverflowError) as exc:
            raise ValueError(name + ' must be a numeric scalar or array') from exc

    def _checked_scalar(value, name, minimum=None, strict=False, integer=False):
        a = _checked_numeric(value, name)
        if a.shape != ():
            raise ValueError(name + ' must be a scalar')
        x = float(a)
        if integer and (np.asarray(value).dtype.kind not in 'iu' or x != np.floor(x)):
            raise ValueError(name + ' must have integer type')
        if minimum is not None and (x < minimum or (strict and x == minimum)):
            raise ValueError(name + ' is outside its stated domain')
        return int(x) if integer else x

    cmax=_checked_scalar(cmax,'cmax',minimum=0,strict=True)
    order=_checked_scalar(order,'order',minimum=1,integer=True)
    h=bhz_layer(coords,edges,mass,potential,t1,delta)
    p=aii_pencil(h,coords,profile,probe,kappa)
    atom=(abs(probe[2])+1)*np.eye(len(h))
    ref=aii_pencil(atom,coords,np.zeros(len(coords)),probe,kappa)[0]
    perm=paired_ordering(ref);fac=skew_factor(ref,perm)
    orientation=int(pfaffian_certificate(ref,fac)[0])
    crossing=pencil_crossings(p,0.,cmax)
    rows=phase_integrals(p,crossing,0.,cmax,orientation,order)
    return float(np.sum(rows[:,3]))
SCICODE_GOLD_EOF
