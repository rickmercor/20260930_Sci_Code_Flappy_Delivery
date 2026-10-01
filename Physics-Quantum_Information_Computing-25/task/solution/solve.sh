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
def build_phenomenological_check_matrix(
    h1: "np.ndarray",
    h2: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    """Reference construction of the hypergraph product with outcome-flip columns."""
    import numpy as np

    def _binary_matrix(value, name):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a 2-D binary array") from None
        if array.ndim != 2 or array.size == 0:
            raise ValueError(f"{name} must be a nonempty 2-D array")
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError(f"{name} must hold real 0/1 entries")
        if not np.all((array == 0) | (array == 1)):
            raise ValueError(f"{name} entries must be 0 or 1")
        return array.astype(np.int64)

    a = _binary_matrix(h1, "h1")
    b = _binary_matrix(h2, "h2")
    m1, n1 = a.shape
    m2, n2 = b.shape
    hx = np.hstack([
        np.kron(a, np.eye(n2, dtype=np.int64)),
        np.kron(np.eye(m1, dtype=np.int64), b.T),
    ]) % 2
    hz = np.hstack([
        np.kron(np.eye(n1, dtype=np.int64), b),
        np.kron(a.T, np.eye(m2, dtype=np.int64)),
    ]) % 2
    d_matrix = np.hstack([hx, np.eye(m1 * n2, dtype=np.int64)])
    return d_matrix.astype(np.int64), hz.astype(np.int64)

import numpy as np
def decouple_hypergraph_product(h1: "np.ndarray", h2: "np.ndarray") -> "np.ndarray":
    """Reference split: A = h1 kron I, blocks (I | h2^T) with K = m1."""
    import numpy as np

    def _shape(value, name):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a 2-D binary array") from None
        if array.ndim != 2 or array.size == 0:
            raise ValueError(f"{name} must be a nonempty 2-D array")
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError(f"{name} must hold real 0/1 entries")
        if not np.all((array == 0) | (array == 1)):
            raise ValueError(f"{name} entries must be 0 or 1")
        return array.shape

    m1, n1 = _shape(h1, "h1")
    m2, n2 = _shape(h2, "h2")
    n_data = n1 * n2 + m1 * m2
    order = []
    for block in range(m1):
        rows = block * n2 + np.arange(n2)
        order.extend((n_data + rows).tolist())
        order.extend((n1 * n2 + block * m2 + np.arange(m2)).tolist())
    order.extend(range(n1 * n2))
    return np.asarray(order, dtype=np.int64)

import numpy as np
def compute_decoupling_transform(
    d_matrix: "np.ndarray",
    perm: "np.ndarray",
    n_blocks: int,
    block_cols: int,
) -> "np.ndarray":
    """Reference transform: invert the identity-position columns over GF(2)."""
    import numpy as np

    def _is_int(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    d = np.asarray(d_matrix)
    if d.ndim != 2 or d.size == 0 or not np.issubdtype(d.dtype, np.number):
        raise ValueError("d_matrix must be a nonempty 2-D binary array")
    if not np.all((d == 0) | (d == 1)):
        raise ValueError("d_matrix entries must be 0 or 1")
    d = d.astype(np.int64)
    m, n = d.shape
    p = np.asarray(perm)
    if p.shape != (n,) or not np.issubdtype(p.dtype, np.integer) or not np.array_equal(np.sort(p), np.arange(n)):
        raise ValueError("perm must be a permutation of range(n)")
    if not (_is_int(n_blocks) and n_blocks > 0 and m % n_blocks == 0):
        raise ValueError("n_blocks must be a positive divisor of m")
    rows = m // n_blocks
    if not (_is_int(block_cols) and rows <= block_cols and n_blocks * block_cols <= n):
        raise ValueError("block_cols must satisfy m_D <= block_cols and n_blocks*block_cols <= n")
    permuted = d[:, p]
    pivots = np.concatenate([i * block_cols + np.arange(rows) for i in range(n_blocks)])
    # Gauss-Jordan inversion of the identity-position columns over GF(2).
    work = np.hstack([permuted[:, pivots], np.eye(m, dtype=np.int64)])
    for col in range(m):
        hits = np.nonzero(work[col:, col])[0]
        if hits.size == 0:
            raise ValueError("the identity-position columns are singular over GF(2)")
        pivot = col + hits[0]
        if pivot != col:
            work[[col, pivot]] = work[[pivot, col]]
        others = np.nonzero(work[:, col])[0]
        others = others[others != col]
        work[others] ^= work[col]
    t_matrix = work[:, m:]
    decoupled = (t_matrix @ permuted) % 2
    for i in range(n_blocks):
        cols = slice(i * block_cols, (i + 1) * block_cols)
        inside = decoupled[i * rows:(i + 1) * rows, cols]
        outside = np.delete(decoupled[:, cols], np.s_[i * rows:(i + 1) * rows], axis=0)
        if outside.any() or not np.array_equal(inside[:, :rows], np.eye(rows, dtype=np.int64)):
            raise ValueError("no row transformation decouples the matrix under this permutation")
    return t_matrix.astype(np.int64)

import numpy as np
from fractions import Fraction
from bisect import bisect_right

def _gain_bounds(interval):
    try:
        pairs = tuple(tuple(pair) for pair in interval)
        if len(pairs) != 2 or any(len(pair) != 2 for pair in pairs):
            raise ValueError
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               for pair in pairs for v in pair):
            raise ValueError
        if any(pair[1] <= 0 for pair in pairs):
            raise ValueError
        lo, hi = (Fraction(int(a), int(b)) for a, b in pairs)
        if not 0 < lo < hi:
            raise ValueError
    except (TypeError, ValueError, ZeroDivisionError):
        raise ValueError("gain_interval must hold two increasing positive rational endpoints") from None
    return lo, hi

def _affine_array(values, rows):
    a = np.asarray(values)
    if a.shape != (rows, 2) or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("weights must be integer arrays with columns (intercept, slope)")
    return a.copy()

def _profile_cost(coefficients, error):
    return tuple(sum(int(coefficients[j, k]) for j in np.flatnonzero(error)) for k in range(2))

def _right_key(cost, x):
    return cost[0] * x.denominator + cost[1] * x.numerator, cost[1]

def _crossing(x, end, lesser, greater):
    slope = lesser[1] - greater[1]
    if slope > 0:
        root = Fraction(greater[0] - lesser[0], slope)
        if x < root < end:
            end = root
    return end

def _profile_append(edges, errors, end, error):
    if errors and np.array_equal(errors[-1], error):
        edges[-1] = end
    else:
        errors.append(error.copy())
        edges.append(end)

def _profile_result(edges, errors):
    return tuple((int(x.numerator), int(x.denominator)) for x in edges), np.asarray(errors, dtype=np.int64)


def greedy_guess_block(
    b_matrix: "np.ndarray",
    syndrome: "np.ndarray",
    weight_f: "np.ndarray",
    weight_g: "np.ndarray",
    max_iter: int,
    gain_interval: tuple,
) -> tuple:
    b = np.asarray(b_matrix)
    if b.ndim != 2 or min(b.shape) < 1 or not np.all((b == 0) | (b == 1)):
        raise ValueError("b_matrix must be a nonempty binary matrix")
    b = b.astype(np.int64)
    rows, cols = b.shape
    s = np.asarray(syndrome)
    if s.shape != (rows,) or not np.all((s == 0) | (s == 1)):
        raise ValueError("syndrome must be a binary vector matching B")
    s = s.astype(np.int64)
    coeff = np.vstack((_affine_array(weight_f, rows), _affine_array(weight_g, cols)))
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError("max_iter must be a non-negative integer")
    lo, hi = _gain_bounds(gain_interval)
    def complete(g):
        e = np.concatenate(((b @ g + s) % 2, g))
        return e, _profile_cost(coeff, e)
    edges, errors, x = [lo], [], lo
    while x < hi:
        g = np.zeros(cols, dtype=np.int64)
        e, incumbent = complete(g)
        end = hi
        for _ in range(int(max_iter)):
            candidates = []
            for j in np.flatnonzero(g == 0):
                candidate = g.copy(); candidate[j] = 1
                ce, cost = complete(candidate)
                candidates.append((cost, int(j), candidate, ce))
            if not candidates:
                break
            best = min(candidates, key=lambda item: (_right_key(item[0], x), item[1]))
            cost, _, candidate, ce = best
            for other, _, _, _ in candidates:
                end = _crossing(x, end, cost, other)
            if _right_key(cost, x) < _right_key(incumbent, x):
                end = _crossing(x, end, cost, incumbent)
                g, e, incumbent = candidate, ce, cost
            else:
                end = _crossing(x, end, incumbent, cost)
                break
        _profile_append(edges, errors, end, e)
        x = end
    return _profile_result(edges, errors)

import numpy as np
def compute_left_syndromes(
    syndrome: "np.ndarray",
    t_matrix: "np.ndarray",
    a_matrix: "np.ndarray",
    right_error: "np.ndarray",
    n_blocks: int,
) -> "np.ndarray":
    """Reference left-part syndrome: (T s + A r) mod 2, split by block rows."""
    import numpy as np

    def _binary(value, shape, name):
        array = np.asarray(value)
        if array.shape != shape or not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"{name} must be a binary array of shape {shape}")
        if not np.all((array == 0) | (array == 1)):
            raise ValueError(f"{name} entries must be 0 or 1")
        return array.astype(np.int64)

    s_raw = np.asarray(syndrome)
    if s_raw.ndim != 1 or s_raw.size == 0:
        raise ValueError("syndrome must be a nonempty vector")
    m = s_raw.size
    s = _binary(s_raw, (m,), "syndrome")
    t = _binary(t_matrix, (m, m), "t_matrix")
    a_raw = np.asarray(a_matrix)
    if a_raw.ndim != 2 or a_raw.shape[0] != m:
        raise ValueError("a_matrix must have m rows")
    a = _binary(a_raw, a_raw.shape, "a_matrix")
    r = _binary(right_error, (a.shape[1],), "right_error")
    if isinstance(n_blocks, bool) or not isinstance(n_blocks, (int, np.integer)) or n_blocks < 1 or m % n_blocks:
        raise ValueError("n_blocks must be a positive integer dividing m")
    left = (t @ s + a @ r) % 2
    return left.reshape(int(n_blocks), m // int(n_blocks)).astype(np.int64)

import numpy as np
from fractions import Fraction
from bisect import bisect_right

def hierarchical_decode(
    syndrome: "np.ndarray",
    d_matrix: "np.ndarray",
    perm: "np.ndarray",
    t_matrix: "np.ndarray",
    n_blocks: int,
    block_cols: int,
    weights: "np.ndarray",
    max_iter: int,
    gain_interval: tuple,
) -> tuple:
    d = np.asarray(d_matrix)
    if d.ndim != 2 or min(d.shape) < 1 or not np.all((d == 0) | (d == 1)):
        raise ValueError("D must be a nonempty binary matrix")
    d = d.astype(np.int64);m,n = d.shape
    s = np.asarray(syndrome)
    if s.shape != (m,) or not np.all((s == 0) | (s == 1)):
        raise ValueError("syndrome must be binary with m entries")
    s = s.astype(np.int64)
    p = np.asarray(perm);t = np.asarray(t_matrix)
    if p.shape != (n,) or not np.issubdtype(p.dtype,np.integer) or not np.array_equal(np.sort(p),np.arange(n)):
        raise ValueError("perm must be a permutation")
    if t.shape != (m,m) or not np.all((t == 0) | (t == 1)):
        raise ValueError("T must be binary with shape (m,m)")
    if isinstance(n_blocks,(bool,np.bool_)) or not isinstance(n_blocks,(int,np.integer)) or n_blocks < 1 or m%n_blocks:
        raise ValueError("invalid block count")
    rows=m//int(n_blocks)
    if isinstance(block_cols,(bool,np.bool_)) or not isinstance(block_cols,(int,np.integer)) or block_cols <= rows or n_blocks*block_cols > n:
        raise ValueError("invalid block width")
    if isinstance(max_iter,(bool,np.bool_)) or not isinstance(max_iter,(int,np.integer)) or max_iter < 0:
        raise ValueError("invalid cap")
    w=_affine_array(weights,n);lo,hi=_gain_bounds(gain_interval)
    k,width=int(n_blocks),int(block_cols)
    dp=(t.astype(np.int64)@d[:,p])%2;wp=w[p]
    for i in range(k):
        part=dp[:,i*width:(i+1)*width]
        if np.delete(part,np.s_[i*rows:(i+1)*rows],axis=0).any() or not np.array_equal(part[i*rows:(i+1)*rows,:rows],np.eye(rows,dtype=int)):
            raise ValueError("matrix is not decoupled in the stated layout")
    a=dp[:,k*width:];cache={}
    def complete(r,x):
        residuals=compute_left_syndromes(s,t,a,r,k)
        pieces=[];end=hi
        for i,block_s in enumerate(residuals):
            key=(i,tuple(int(v) for v in block_s))
            if key not in cache:
                offset=i*width
                bp,es=greedy_guess_block(dp[i*rows:(i+1)*rows,offset+rows:offset+width],block_s,wp[offset:offset+rows],wp[offset+rows:offset+width],max_iter,gain_interval)
                cache[key]=([Fraction(*pair) for pair in bp],es)
            edges,es=cache[key];idx=bisect_right(edges,x)-1
            pieces.append(es[idx]);end=min(end,edges[idx+1])
        error=np.concatenate(pieces+[r])
        return error,_profile_cost(wp,error),end
    edges,errors,x=[lo],[],lo
    while x < hi:
        r=np.zeros(a.shape[1],dtype=np.int64);e,incumbent,end=complete(r,x)
        for _ in range(int(max_iter)):
            candidates=[]
            for j in np.flatnonzero(r==0):
                cr=r.copy();cr[j]=1
                ce,cost,ce_end=complete(cr,x);end=min(end,ce_end)
                candidates.append((cost,int(j),cr,ce))
            if not candidates:break
            cost,_,cr,ce=min(candidates,key=lambda item:(_right_key(item[0],x),item[1]))
            for other,_,_,_ in candidates:end=_crossing(x,end,cost,other)
            if _right_key(cost,x) < _right_key(incumbent,x):
                end=_crossing(x,end,cost,incumbent);r,e,incumbent=cr,ce,cost
            else:
                end=_crossing(x,end,incumbent,cost);break
        original=np.empty(n,dtype=np.int64);original[p]=e
        _profile_append(edges,errors,end,original);x=end
    return _profile_result(edges,errors)

import numpy as np
from fractions import Fraction

def truncated_failure_probability(
    d_matrix: "np.ndarray",
    hz: "np.ndarray",
    probabilities: "np.ndarray",
    decode_fn: "Callable[[np.ndarray], tuple]",
    max_faults: int,
    gain_interval: tuple,
) -> tuple:
    import itertools
    import math
    d=np.asarray(d_matrix);h=np.asarray(hz)
    if d.ndim!=2 or min(d.shape)<1 or not np.all((d==0)|(d==1)):
        raise ValueError("invalid fault matrix")
    d=d.astype(np.int64);m,n=d.shape
    if h.ndim!=2 or not 1<=h.shape[1]<=n or not np.all((h==0)|(h==1)):
        raise ValueError("invalid stabilizer matrix")
    nd=h.shape[1];p=np.asarray(probabilities,dtype=float)
    if p.shape!=(n,) or not np.all((p>=0)&(p<=1)):
        raise ValueError("invalid probabilities")
    if isinstance(max_faults,(bool,np.bool_)) or not isinstance(max_faults,(int,np.integer)) or not 0<=max_faults<=n or not callable(decode_fn):
        raise ValueError("invalid truncation or decoder")
    lo,hi=_gain_bounds(gain_interval)
    basis={}
    def mask(v):return sum(int(bit)<<i for i,bit in enumerate(v))
    for row in h:
        v=mask(row)
        while v:
            pivot=v.bit_length()-1
            if pivot in basis:v^=basis[pivot]
            else:basis[pivot]=v;break
    def remainder(v):
        for pivot in sorted(basis,reverse=True):
            if v>>pivot&1:v^=basis[pivot]
        return v
    cache={};counts=[[] for _ in range(max_faults+1)];terms=[[] for _ in counts]
    for order in range(max_faults+1):
        for support in itertools.combinations(range(n),order):
            e=np.zeros(n,dtype=np.int64);e[list(support)]=1;s=(d@e)%2;key=tuple(s)
            if key not in cache:
                try:
                    raw,corrections=decode_fn(s.copy())
                    if any(len(pair)!=2 or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in pair) or pair[1]<=0 for pair in raw):raise ValueError
                    edges=[Fraction(int(a),int(b)) for a,b in raw]
                    corrections=np.asarray(corrections)
                    if len(edges)<2 or edges[0]!=lo or edges[-1]!=hi or any(a>=b for a,b in zip(edges,edges[1:])):raise ValueError
                    if corrections.shape!=(len(edges)-1,n) or not np.all((corrections==0)|(corrections==1)):raise ValueError
                    corrections=corrections.astype(np.int64)
                    if not np.all((corrections@d.T)%2==s):raise ValueError
                except (TypeError,ValueError,ZeroDivisionError):
                    raise ValueError("invalid decoder profile") from None
                cache[key]=[(b-a,remainder(mask(c[:nd]))) for a,b,c in zip(edges,edges[1:],corrections)]
            residual_class=remainder(mask(e[:nd]))
            fraction=sum((length for length,cls in cache[key] if cls!=residual_class),Fraction(0))/(hi-lo)
            mass=float(np.prod(np.where(e,p,1-p)))
            counts[order].append(float(fraction));terms[order].append(float(fraction)*mass)
    return np.array([math.fsum(x) for x in counts]),np.array([math.fsum(x) for x in terms])

import numpy as np

def estimate_decoder_failure_probability(
    ring_lengths: tuple = (3, 4),
    data_prob: tuple = (0.010, 0.001),
    meas_prob: tuple = (0.006, 0.0005),
    max_iter: int = 3,
    max_faults: int = 3,
    gain_interval: tuple = ((3, 4), (5, 4)),
    weight_scale: int = 1000000,
) -> float:
    import math
    try:
        l1,l2=ring_lengths;p0,dp=data_prob;q0,dq=meas_prob
    except (TypeError,ValueError):raise ValueError("lengths and probability parameters must be pairs") from None
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or v<2 for v in (l1,l2)):
        raise ValueError("ring lengths must be integers at least 2")
    if isinstance(weight_scale,(bool,np.bool_)) or not isinstance(weight_scale,(int,np.integer)) or weight_scale<=0:
        raise ValueError("weight_scale must be a positive integer")
    _gain_bounds(gain_interval)
    def ring(length):
        h=np.eye(length,dtype=np.int64);h[np.arange(length),(np.arange(length)+1)%length]=1
        return h
    h1,h2=ring(int(l1)),ring(int(l2))
    d,hz=build_phenomenological_check_matrix(h1,h2);m,n=d.shape;nd=hz.shape[1]
    p=np.concatenate((p0+dp*np.arange(nd),q0+dq*np.arange(m)))
    if not np.all((p>0)&(p<1)):raise ValueError("all fault probabilities must lie in (0,1)")
    nominal=np.array([round(int(weight_scale)*math.log((1-float(x))/float(x))) for x in p],dtype=np.int64)
    w=np.zeros((n,2),dtype=np.int64);w[:nd,1]=nominal[:nd];w[nd:,0]=nominal[nd:]
    perm=decouple_hypergraph_product(h1,h2);k=int(l1);width=2*int(l2)
    t=compute_decoupling_transform(d,perm,k,width)
    def decode(s):return hierarchical_decode(s,d,perm,t,k,width,w,max_iter,gain_interval)
    counts,contributions=truncated_failure_probability(d,hz,p,decode,max_faults,gain_interval)
    return float(math.fsum(contributions))
SCICODE_GOLD_EOF
