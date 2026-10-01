#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

from fractions import Fraction as _Fraction
import numpy as np

def _f3r_fraction(pair):
    if (not isinstance(pair, (tuple, list)) or len(pair) != 2
            or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in pair)
            or int(pair[1]) <= 0):
        raise ValueError("Expected an integer numerator and positive integer denominator")
    return _Fraction(int(pair[0]), int(pair[1]))

def _f3r_pair(value):
    value = _Fraction(value)
    return (value.numerator, value.denominator)

def _f3r_diagonal(diagonal):
    try:
        raw = np.asarray(diagonal)
        if raw.dtype.kind not in 'fiu':
            raise ValueError("diagonal must be real numeric data")
        a = np.asarray(raw, dtype=np.float64)
    except (TypeError, OverflowError) as exc:
        raise ValueError("Invalid diagonal") from exc
    if a.ndim != 1 or a.size == 0 or a.size % 4 or not np.isfinite(a).all() or np.any(a < 1) or np.any(a > 2):
        raise ValueError("diagonal must have length 4*b and entries in [1,2]")
    return a

def _f3r_half_exact(value):
    # The seed conversion locates neighbours; exact distances decide rounding.
    value = _Fraction(value)
    seed = np.float16(float(value))
    candidates = (np.nextafter(seed, np.float16(-np.inf), dtype=np.float16),
                  seed, np.nextafter(seed, np.float16(np.inf), dtype=np.float16))
    distances = [abs(value - _Fraction(float(x))) for x in candidates]
    best = min(distances)
    tied = [x for x, dist in zip(candidates, distances) if dist == best]
    return next((x for x in tied if int(np.asarray(x).view(np.uint16)) % 2 == 0), tied[0])

def build_block_factors(diagonal: "np.ndarray", multiplier: tuple) -> tuple:
    a = _f3r_diagonal(diagonal)
    lam = _f3r_fraction(multiplier)
    if not 1 <= lam <= 2:
        raise ValueError("multiplier must lie in [1,2]")
    lower = np.zeros((a.size // 4, 4), dtype=np.float16)
    upper = np.empty_like(lower)
    for block in range(a.size // 4):
        previous = None
        for k in range(4):
            pivot = lam * _Fraction(float(a[4 * block + k]))
            if k:
                sub = _Fraction(-3, 16) / previous
                lower[block, k] = _f3r_half_exact(sub)
                pivot -= sub * _Fraction(-7, 16)
            upper[block, k] = _f3r_half_exact(pivot)
            previous = pivot
    return lower, upper

from fractions import Fraction as _Fraction
from functools import cmp_to_key as _cmp_to_key
from math import gcd as _integer_gcd
import numpy as np


def _f3r_ptrim(poly):
    """Ascending-power exact polynomial coefficients; zero is (0,)."""
    out = [_Fraction(c) for c in poly]
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return tuple(out) if out else (_Fraction(0),)


def _f3r_pscale(poly, value):
    return _f3r_ptrim(c * value for c in poly)


def _f3r_psub(left, right):
    n = max(len(left), len(right))
    return _f3r_ptrim((left[i] if i < len(left) else 0)
                     - (right[i] if i < len(right) else 0) for i in range(n))


def _f3r_pdivrem(dividend, divisor):
    a, b = list(_f3r_ptrim(dividend)), _f3r_ptrim(divisor)
    if b == (0,):
        raise ValueError("Polynomial divisor is zero")
    quotient = [_Fraction(0)] * max(1, len(a) - len(b) + 1)
    while len(a) >= len(b) and a != [0]:
        k, c = len(a) - len(b), a[-1] / b[-1]
        quotient[k] += c
        for j in range(len(b)):
            a[k + j] -= c * b[j]
        a = list(_f3r_ptrim(a))
    return _f3r_ptrim(quotient), tuple(a)


def _f3r_pgcd(left, right):
    a, b = _f3r_ptrim(left), _f3r_ptrim(right)
    while b != (0,):
        a, b = b, _f3r_pdivrem(a, b)[1]
    return _f3r_pscale(a, 1 / a[-1]) if a != (0,) else a


def _f3r_primitive(poly):
    """Primitive integer coefficients in descending-power order."""
    poly = _f3r_ptrim(poly)
    den = 1
    for c in poly:
        den = den // _integer_gcd(den, c.denominator) * c.denominator
    integers = [int(c * den) for c in reversed(poly)]
    content = 0
    for c in integers:
        content = _integer_gcd(content, abs(c))
    if not content:
        return (0,)
    divisor = content if integers[0] > 0 else -content
    return tuple(c // divisor for c in integers)


def _f3r_peval(poly, value):
    result = _Fraction(0)
    for c in reversed(poly):
        result = result * value + c
    return result


def _f3r_psign(coefficients, value):
    """Exact sign at a rational, using integer homogeneous Horner evaluation."""
    num, den = value.numerator, value.denominator
    total, scale = int(coefficients[0]), den
    for c in coefficients[1:]:
        total = total * num + int(c) * scale
        scale *= den
    return (total > 0) - (total < 0)


def _f3r_root_count(poly, left, right):
    """Count distinct real roots in a CLOSED rational interval by Sturm."""
    p = _f3r_ptrim(poly)
    if left > right or len(p) <= 1:
        return 0
    if left == right:
        return int(_f3r_peval(p, left) == 0)
    derivative = tuple(i * p[i] for i in range(1, len(p)))
    common = _f3r_pgcd(p, derivative)
    p = _f3r_pdivrem(p, common)[0]  # Square-free part, without changing roots.
    derivative = tuple(i * p[i] for i in range(1, len(p)))
    chain = [p, derivative]
    while chain[-1] != (0,):
        rem = _f3r_pdivrem(chain[-2], chain[-1])[1]
        if rem == (0,):
            break
        rem = _f3r_pscale(rem, -1)
        # Positive scaling limits coefficient growth without changing signs.
        chain.append(_f3r_pscale(rem, 1 / abs(rem[-1])))
    def _variations(x):
        signs = []
        for f in chain:
            v = _f3r_peval(f, x)
            if v:
                signs.append(1 if v > 0 else -1)
        return sum(a != b for a, b in zip(signs, signs[1:]))
    # V(left)-V(right) includes a right-endpoint root, not a left one.
    return _variations(left) - _variations(right) + int(_f3r_peval(p, left) == 0)


def _f3r_endpoint(value):
    return ((value.denominator, -value.numerator), _f3r_pair(value), _f3r_pair(value))


def _f3r_isolate_monotone(coeff, left, right):
    """Isolate the sole root of a monotone factor-midpoint equation."""
    sl, sr = _f3r_psign(coeff, left), _f3r_psign(coeff, right)
    if sl == 0:
        return (left.denominator, -left.numerator), left, left
    if sr == 0:
        return (right.denominator, -right.numerator), right, right
    if sl == sr:
        raise ValueError("Internal midpoint root is not bracketed")
    if len(coeff) == 2:
        root = _Fraction(-coeff[1], coeff[0])
        return (root.denominator, -root.numerator), root, root
    # A rational root of a primitive integer polynomial has denominator <= B.
    # This width makes it uniquely recoverable by rational reconstruction.
    B = abs(coeff[0])
    target = min(_Fraction(1, 10**30), _Fraction(1, 4 * B * B))
    lo, hi = left, right
    while hi - lo >= target:
        middle = (lo + hi) / 2
        sm = _f3r_psign(coeff, middle)
        if sm == 0:
            return (middle.denominator, -middle.numerator), middle, middle
        if sm == sl:
            lo = middle
        else:
            hi = middle
    candidate = ((lo + hi) / 2).limit_denominator(B)
    if lo <= candidate <= hi and _f3r_psign(coeff, candidate) == 0:
        return (candidate.denominator, -candidate.numerator), candidate, candidate
    return coeff, lo, hi


def _f3r_refine_event(event):
    if event['lo'] == event['hi']:
        return
    lo, hi, coeff = event['lo'], event['hi'], event['coeff']
    mid = (lo + hi) / 2
    sm = _f3r_psign(coeff, mid)
    if sm == 0:
        event['lo'] = event['hi'] = mid
        event['coeff'] = (mid.denominator, -mid.numerator)
    elif sm == _f3r_psign(coeff, lo):
        event['lo'] = mid
    else:
        event['hi'] = mid


def _f3r_compare_events(a, b):
    if a is b:
        return 0
    while True:
        if a['hi'] < b['lo']:
            return -1
        if b['hi'] < a['lo']:
            return 1
        common = _f3r_pgcd(tuple(reversed(a['coeff'])), tuple(reversed(b['coeff'])))
        lo, hi = max(a['lo'], b['lo']), min(a['hi'], b['hi'])
        if len(common) > 1 and _f3r_root_count(common, lo, hi) == 1:
            return 0
        _f3r_refine_event(a)
        _f3r_refine_event(b)


def factor_rounding_regions(diagonal: "np.ndarray", domain: tuple) -> tuple:
    a = _f3r_diagonal(diagonal)
    if not isinstance(domain, (tuple, list)) or len(domain) != 2:
        raise ValueError("domain must contain two rational endpoints")
    left, right = map(_f3r_fraction, domain)
    if not 1 <= left <= right <= 2:
        raise ValueError("Require 1 <= left <= right <= 2")
    if left == right:
        low, up = build_block_factors(a, _f3r_pair(left))
        return ((_f3r_endpoint(left),), np.array([[0, 0, 0]], dtype=np.int64), low[None], up[None])
    unique_blocks = {}
    for block in range(a.size // 4):
        key = tuple(float(x) for x in a[4 * block:4 * block + 4])
        unique_blocks.setdefault(key, []).append(block)
    events = {}
    for diagonal_block, block_ids in unique_blocks.items():
        # Consecutive leading principal determinants P_k give u_k=P_k/P_(k-1).
        # With a_i and lambda in [1,2], all pivots stay positive and all
        # nonconstant factor entries are strictly increasing on this domain.
        previous_previous = (_Fraction(0),)
        previous = (_Fraction(1),)
        functions = []
        for k, avalue in enumerate(diagonal_block):
            determinant = (_Fraction(0),) + _f3r_pscale(previous, _Fraction(avalue))
            if k:
                functions.append((0, k, _f3r_pscale(previous_previous, _Fraction(-3, 16)), previous))
                determinant = _f3r_psub(determinant, _f3r_pscale(previous_previous, _Fraction(21, 256)))
            functions.append((1, k, determinant, previous))
            previous_previous, previous = previous, determinant
        for kind, k, numerator, denominator in functions:
            low = _f3r_peval(numerator, left) / _f3r_peval(denominator, left)
            high = _f3r_peval(numerator, right) / _f3r_peval(denominator, right)
            h = np.nextafter(_f3r_half_exact(low), np.float16(-np.inf), dtype=np.float16)
            while True:
                nxt = np.nextafter(h, np.float16(np.inf), dtype=np.float16)
                midpoint = (_Fraction(float(h)) + _Fraction(float(nxt))) / 2
                if midpoint > high:
                    break
                if midpoint >= low:
                    equation = _f3r_psub(numerator, _f3r_pscale(denominator, midpoint))
                    coeff = _f3r_primitive(equation)
                    coeff, lo, hi = _f3r_isolate_monotone(coeff, left, right)
                    if coeff in events:
                        event = events[coeff]
                        event['lo'] = max(event['lo'], lo)
                        event['hi'] = min(event['hi'], hi)
                    else:
                        event = events[coeff] = dict(lo=lo, hi=hi, coeff=coeff, changes=[])
                    tie = h if int(np.asarray(h).view(np.uint16)) % 2 == 0 else nxt
                    for block in block_ids:
                        event['changes'].append((kind, block, k, float(tie)))
                h = nxt
    interior = [e for e in events.values() if not (e['lo'] == e['hi'] and e['lo'] in (left, right))]
    interior.sort(key=_cmp_to_key(_f3r_compare_events))
    distinct = []
    for event in interior:
        if distinct and _f3r_compare_events(distinct[-1], event) == 0:
            old = distinct[-1]
            common = _f3r_pgcd(tuple(reversed(old['coeff'])), tuple(reversed(event['coeff'])))
            old['coeff'] = _f3r_primitive(common)
            old['lo'], old['hi'] = max(old['lo'], event['lo']), min(old['hi'], event['hi'])
            old['changes'].extend(event['changes'])
        else:
            distinct.append(event)
    bounds = [dict(lo=left, hi=left, coeff=(left.denominator, -left.numerator), changes=[])]
    bounds += distinct
    bounds += [dict(lo=right, hi=right, coeff=(right.denominator, -right.numerator), changes=[])]
    # Isolating brackets must be strictly disjoint, also from domain endpoints.
    for e, f in zip(bounds, bounds[1:]):
        while e['hi'] >= f['lo']:
            _f3r_refine_event(e)
            _f3r_refine_event(f)
    states, state_ids, strata = [], {}, []
    def _record(lower, upper, li, ri):
        key = lower.tobytes() + upper.tobytes()
        if key not in state_ids:
            state_ids[key] = len(states)
            states.append((lower.copy(), upper.copy()))
        strata.append((li, ri, state_ids[key]))
    for i, bound in enumerate(bounds):
        middle = (bound['lo'] + bound['hi']) / 2
        lower, upper = build_block_factors(a, _f3r_pair(middle))
        # At an algebraic boundary, every changing factor is set from its
        # exact midpoint tie; the representative alone would be one-sided.
        for kind, block, k, tie in bound['changes']:
            (lower if kind == 0 else upper)[block, k] = np.float16(tie)
        _record(lower, upper, i, i)
        if i + 1 < len(bounds):
            middle = (bound['hi'] + bounds[i + 1]['lo']) / 2
            lower, upper = build_block_factors(a, _f3r_pair(middle))
            _record(lower, upper, i, i + 1)
    numeric_bounds = tuple((e['coeff'], _f3r_pair(e['lo']), _f3r_pair(e['hi'])) for e in bounds)
    return (numeric_bounds, np.asarray(strata, dtype=np.int64),
            np.stack([s[0] for s in states]), np.stack([s[1] for s in states]))

from functools import lru_cache as _lru_cache
import numpy as np

def _f3r_real_array(value, name):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in 'fiu':
            raise ValueError(name + " must contain real numbers")
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, OverflowError) as exc:
        raise ValueError(name + " must contain real numbers") from exc
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite")
    return result

def _f3r_factor_arrays(lower, upper):
    l, u = _f3r_real_array(lower, 'lower'), _f3r_real_array(upper, 'upper')
    if l.ndim != 2 or l.shape[1] != 4 or l.shape[0] == 0 or l.shape != u.shape:
        raise ValueError("Factor arrays must have matching shape (b,4)")
    with np.errstate(over='ignore'):
        l16, u16 = l.astype(np.float16), u.astype(np.float16)
    if (not np.array_equal(l, l16.astype(np.float64)) or not np.array_equal(u, u16.astype(np.float64))
            or np.any(l[:, 0] != 0) or np.any(u == 0)):
        raise ValueError("Factors must be finite binary16 values with valid diagonals")
    return l16, u16

def _f3r_vector(value, size, dtype, name):
    arr = _f3r_real_array(value, name)
    if arr.shape != (size,):
        raise ValueError(name + " has an incompatible shape")
    with np.errstate(over='ignore'):
        arr = arr.astype(dtype)
    if not np.isfinite(arr).all():
        raise ValueError(name + " overflows its storage format")
    return arr

def _f3r_integer(value, name, low=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < low:
        raise ValueError(name + " is outside its integer domain")
    return int(value)

def _f3r_finite(value):
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite arithmetic result")
    return value

def _f3r_dot(a, b, dtype):
    return np.cumsum(np.multiply(a, b, dtype=dtype), dtype=dtype)[-1]

def _f3r_norm(a, dtype):
    return dtype(np.sqrt(_f3r_dot(a, a, dtype)))

@_lru_cache(maxsize=16)
def _f3r_sparse_data(shape, raw):
    A = np.frombuffer(raw, dtype=np.float64).reshape(shape)
    n = A.shape[0]
    widths = np.count_nonzero(A, axis=1)
    width = max(1, int(widths.max()))
    cols = np.full((n, width), n, dtype=np.int64)
    vals = np.zeros((n, width), dtype=np.float64)
    for i in range(n):
        js = np.flatnonzero(A[i])
        cols[i, :len(js)] = js
        vals[i, :len(js)] = A[i, js]
    return cols, (vals, vals.astype(np.float32), vals.astype(np.float16))

def _f3r_matrix(A, n):
    a = _f3r_real_array(A, 'A')
    if a.shape != (n, n):
        raise ValueError("A has an incompatible shape")
    with np.errstate(over='ignore'):
        if not np.isfinite(a.astype(np.float16)).all():
            raise ValueError("A must have a finite binary16 representation")
    return a

def _f3r_mv(A, v, level, dtype):
    cols, vals = _f3r_sparse_data(A.shape, A.tobytes())
    vpad = np.append(np.asarray(v, dtype=dtype), dtype(0))
    pro = np.multiply(vals[level], vpad[cols], dtype=dtype)
    return np.cumsum(pro, axis=1, dtype=dtype)[:, -1]

def _f3r_dot_many(a, b, dtype):
    # The last axis is one vector; never reduce across independent solves.
    return np.cumsum(np.multiply(a, b, dtype=dtype), axis=-1, dtype=dtype)[..., -1]


def _f3r_norm_many(a, dtype):
    return np.sqrt(_f3r_dot_many(a, a, dtype)).astype(dtype, copy=False)


def _f3r_mv_many(A, v, level, dtype):
    cols, vals = _f3r_sparse_data(A.shape, A.tobytes())
    vpad = np.concatenate((np.asarray(v, dtype=dtype), np.zeros((len(v), 1), dtype=dtype)), axis=1)
    products = np.multiply(vals[level][None, :, :], vpad[:, cols], dtype=dtype)
    return np.cumsum(products, axis=2, dtype=dtype)[:, :, -1]


def _f3rapply_primary_many(lower, upper, rhs, precision):
    """Internal numerical kernel for already validated independent factor/vector rows."""
    dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    r = _f3r_finite(np.asarray(rhs, dtype=dtype)).reshape(lower.shape)
    l, u = lower.astype(dtype, copy=False), upper.astype(dtype, copy=False)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        y = r.copy()
        for k in range(1, 4):
            y[:, :, k] = np.subtract(r[:, :, k], np.multiply(l[:, :, k], y[:, :, k-1], dtype=dtype), dtype=dtype)
        z = np.zeros_like(y)
        z[:, :, 3] = np.divide(y[:, :, 3], u[:, :, 3], dtype=dtype)
        for k in range(2, -1, -1):
            term = np.multiply(dtype(-7/16), z[:, :, k+1], dtype=dtype)
            z[:, :, k] = np.divide(np.subtract(y[:, :, k], term, dtype=dtype), u[:, :, k], dtype=dtype)
    return _f3r_finite(z.reshape((len(lower), -1)))


def apply_primary(lower: "np.ndarray", upper: "np.ndarray", rhs: "np.ndarray", precision: int) -> "np.ndarray":
    if precision not in (16, 32, 64) or isinstance(precision, (bool, np.bool_)):
        raise ValueError("precision must be 16, 32, or 64")
    dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    lower, upper = _f3r_factor_arrays(lower, upper)
    r = _f3r_vector(rhs, lower.size, dtype, 'rhs')
    return _f3rapply_primary_many(lower[None, :, :], upper[None, :, :], r[None, :], precision)[0]

import numpy as np

def _f3r_state(weights, counter, interval, adaptive):
    w = _f3r_vector(weights, 2, np.float16, 'weights')
    if not np.array_equal(_f3r_real_array(weights, 'weights'), w.astype(np.float64)):
        raise ValueError("weights must be stored binary16 values")
    counter = _f3r_integer(counter, 'counter')
    interval = _f3r_integer(interval, 'interval')
    if not isinstance(adaptive, (bool, np.bool_)):
        raise ValueError("adaptive must be Boolean")
    if counter == 1:
        w = np.ones(2, dtype=np.float16)
    return w.copy(), counter, interval

def _f3rrichardson_many(A, rhs, lower, upper, weights, counter, interval, adaptive=True):
    """Independent Richardson invocations with a common call number, never shared weights."""
    v = _f3r_finite(np.asarray(rhs, dtype=np.float16)).copy()
    omega = np.asarray(weights, dtype=np.float16).copy()
    if counter == 1:
        omega.fill(np.float16(1))
    z = np.zeros_like(v)
    scheduled = bool(adaptive) and counter % interval == 0
    for k in range(2):
        r = v.copy() if k == 0 else np.subtract(v, _f3r_mv_many(A, z, 2, np.float16), dtype=np.float16)
        correction = _f3rapply_primary_many(lower, upper, r, 16)
        if scheduled:
            r32 = r.astype(np.float32)
            q = _f3r_mv_many(A, _f3rapply_primary_many(lower, upper, r32, 32), 2, np.float32)
            denominator = _f3r_dot_many(q, q, np.float32)
            if np.any(denominator == 0) or not np.isfinite(denominator).all():
                raise ValueError("Invalid local-weight denominator")
            gamma = np.divide(_f3r_dot_many(r32, q, np.float32), denominator, dtype=np.float32)
            ell = counter // interval
            weighted = np.multiply(np.float32(ell), omega[:, k].astype(np.float32), dtype=np.float32)
            avg = np.divide(np.add(weighted, gamma, dtype=np.float32), np.float32(ell + 1), dtype=np.float32)
            omega[:, k] = avg.astype(np.float16)
            z = np.add(z.astype(np.float32), np.multiply(gamma[:, None], correction.astype(np.float32), dtype=np.float32), dtype=np.float32).astype(np.float16)
        else:
            weight = omega[:, k, None] if adaptive else np.float16(1)
            z = np.add(z, np.multiply(weight, correction, dtype=np.float16), dtype=np.float16)
        _f3r_finite(z)
        _f3r_finite(omega)
    return z, omega, counter + 1


def richardson_step(A: "np.ndarray", rhs: "np.ndarray", lower: "np.ndarray", upper: "np.ndarray",
                           weights: "np.ndarray", counter: int, interval: int, adaptive: bool = True) -> tuple:
    lower, upper = _f3r_factor_arrays(lower, upper)
    A = _f3r_matrix(A, lower.size)
    v = _f3r_vector(rhs, lower.size, np.float16, 'rhs')
    omega, counter, interval = _f3r_state(weights, counter, interval, adaptive)
    z, omega, counter = _f3rrichardson_many(
        A, v[None, :], lower[None, :, :], upper[None, :, :], omega[None, :], counter, interval, adaptive)
    return z[0], omega[0], counter

import numpy as np


def _f3rnested_fgmres_many(A, rhs, lower, upper, iterations, interval, weights, counter,
                              level=0, initial=None, adaptive=True):
    """Vectorize independent solves, keeping every within-solve operation in its original order."""
    dtype = np.float64 if level == 0 else np.float32
    b = _f3r_finite(np.asarray(rhs, dtype=dtype)).copy()
    omega = np.asarray(weights, dtype=np.float16).copy()
    if counter == 1:
        omega.fill(np.float16(1))
    if initial is None:
        x, r = np.zeros_like(b), b.copy()
    else:
        x = _f3r_finite(np.asarray(initial, dtype=dtype)).copy()
        r = np.subtract(b, _f3r_mv_many(A, x, level, dtype), dtype=dtype)
    beta = _f3r_norm_many(r, dtype)
    if np.any(beta == 0) or not np.isfinite(beta).all():
        raise ValueError("Invalid initial residual norm")
    m = iterations[level]
    size, n = b.shape
    V = np.zeros((m + 1, size, n), dtype=dtype)
    Z = np.zeros((m, size, n), dtype=dtype)
    H = np.zeros((size, m + 1, m), dtype=dtype)
    g = np.zeros((size, m + 1), dtype=dtype)
    cs, sn = np.zeros((size, m), dtype=dtype), np.zeros((size, m), dtype=dtype)
    g[:, 0], V[0] = beta, np.divide(r, beta[:, None], dtype=dtype)
    for j in range(m):
        if level == 2:
            z, omega, counter = _f3rrichardson_many(A, V[j], lower, upper, omega, counter, interval, adaptive)
        else:
            z, omega, counter = _f3rnested_fgmres_many(A, V[j], lower, upper, iterations, interval, omega, counter, level + 1, None, adaptive)
        Z[j] = np.asarray(z, dtype=dtype)
        w = _f3r_mv_many(A, Z[j], level, dtype)
        for i in range(j + 1):
            H[:, i, j] = _f3r_dot_many(V[i], w, dtype)
        projection = np.zeros((size, n), dtype=dtype)
        for i in range(j + 1):
            projection = np.add(projection, np.multiply(H[:, i, j, None], V[i], dtype=dtype), dtype=dtype)
        w = np.subtract(w, projection, dtype=dtype)
        norm = _f3r_norm_many(w, dtype)
        if np.any(norm == 0) or not np.isfinite(norm).all():
            raise ValueError("Arnoldi breakdown")
        H[:, j + 1, j], V[j + 1] = norm, np.divide(w, norm[:, None], dtype=dtype)
        for i in range(j):
            aa, bb = H[:, i, j].copy(), H[:, i + 1, j].copy()
            H[:, i, j] = np.add(np.multiply(cs[:, i], aa, dtype=dtype), np.multiply(sn[:, i], bb, dtype=dtype), dtype=dtype)
            H[:, i + 1, j] = np.add(np.multiply(-sn[:, i], aa, dtype=dtype), np.multiply(cs[:, i], bb, dtype=dtype), dtype=dtype)
        aa, bb = H[:, j, j].copy(), H[:, j + 1, j].copy()
        rho = np.sqrt(np.add(np.multiply(aa, aa, dtype=dtype), np.multiply(bb, bb, dtype=dtype), dtype=dtype)).astype(dtype, copy=False)
        if np.any(rho == 0) or not np.isfinite(rho).all():
            raise ValueError("Invalid Givens denominator")
        cs[:, j], sn[:, j] = np.divide(aa, rho, dtype=dtype), np.divide(bb, rho, dtype=dtype)
        H[:, j, j], H[:, j + 1, j] = rho, dtype(0)
        aa, bb = g[:, j].copy(), g[:, j + 1].copy()
        g[:, j] = np.add(np.multiply(cs[:, j], aa, dtype=dtype), np.multiply(sn[:, j], bb, dtype=dtype), dtype=dtype)
        g[:, j + 1] = np.add(np.multiply(-sn[:, j], aa, dtype=dtype), np.multiply(cs[:, j], bb, dtype=dtype), dtype=dtype)
    y = np.zeros((size, m), dtype=dtype)
    for i in range(m - 1, -1, -1):
        if np.any(H[:, i, i] == 0):
            raise ValueError("Zero reduced triangular pivot")
        total = np.zeros(size, dtype=dtype)
        for j in range(i + 1, m):
            total = np.add(total, np.multiply(H[:, i, j], y[:, j], dtype=dtype), dtype=dtype)
        y[:, i] = np.divide(np.subtract(g[:, i], total, dtype=dtype), H[:, i, i], dtype=dtype)
    correction = np.zeros((size, n), dtype=dtype)
    for j in range(m):
        correction = np.add(correction, np.multiply(Z[j], y[:, j, None], dtype=dtype), dtype=dtype)
    return _f3r_finite(np.add(x, correction, dtype=dtype)), omega, counter


def nested_fgmres_cycle(A: "np.ndarray", rhs: "np.ndarray", lower: "np.ndarray", upper: "np.ndarray",
                                iterations: tuple, interval: int, weights: "np.ndarray", counter: int,
                                level: int = 0, initial: "np.ndarray | None" = None, adaptive: bool = True) -> tuple:
    lower, upper = _f3r_factor_arrays(lower, upper)
    n = lower.size
    A = _f3r_matrix(A, n)
    if not isinstance(iterations, (tuple, list)) or len(iterations) != 3:
        raise ValueError("iterations must contain three integers")
    ms = tuple(_f3r_integer(m, 'iteration count') for m in iterations)
    if any(m > n for m in ms):
        raise ValueError("Iteration counts cannot exceed n")
    level = _f3r_integer(level, 'level', 0)
    if level not in (0, 1, 2):
        raise ValueError("level must be 0,1,2")
    omega, counter, interval = _f3r_state(weights, counter, interval, adaptive)
    dtype = np.float64 if level == 0 else np.float32
    b = _f3r_vector(rhs, n, dtype, 'rhs')
    x = None if initial is None else _f3r_vector(initial, n, dtype, 'initial')[None, :]
    x, omega, counter = _f3rnested_fgmres_many(
        A, b[None, :], lower[None, :, :], upper[None, :, :], ms, interval,
        omega[None, :], counter, level, x, adaptive)
    return x[0], omega[0], counter

import itertools as _itertools
import numpy as np


def _f3rbatch_residuals_many(A, B, lower, upper, theta, outer_iterations=2, restarts=2, fixed_by_rhs=None):
    """Internal batch kernel: independent factor states and ordered prefixes, no shared adaptive state."""
    m2, m3, interval = theta
    ms = (outer_iterations, m2, m3)
    nstates, nrhs = len(lower), len(B)
    norms = _f3r_norm_many(B, np.float64)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("Invalid right-hand-side norm")
    def _solve(j, w, count, adaptive):
        width = len(j)
        rhs = np.tile(B[j], (nstates, 1))
        ll = np.repeat(lower, width, axis=0)
        uu = np.repeat(upper, width, axis=0)
        x = None
        for cycle in range(restarts):
            x, w, count = _f3rnested_fgmres_many(A, rhs, ll, uu, ms, interval, w, count, 0, x, adaptive)
        residual = np.subtract(rhs, _f3r_mv_many(A, x, 0, np.float64), dtype=np.float64)
        ratios = np.divide(_f3r_norm_many(residual, np.float64), np.tile(norms[j], nstates), dtype=np.float64)
        if not np.isfinite(ratios).all():
            raise ValueError("Nonfinite true residual")
        return ratios.reshape((nstates, width)), w.reshape((nstates, width, 2)), count
    if fixed_by_rhs is None:
        fixed_by_rhs = _solve(list(range(nrhs)), np.ones((nstates * nrhs, 2), dtype=np.float16), 1, False)[0]
    # Distinct ordered prefixes retain distinct state rows. Only identical prefixes are reused.
    previous = {(): 0}
    previous_weights = np.ones((nstates, 1, 2), dtype=np.float16)
    count = 1
    residuals = {}
    for length in range(1, nrhs + 1):
        prefixes = tuple(_itertools.permutations(range(nrhs), length))
        parent_indices = [previous[prefix[:-1]] for prefix in prefixes]
        w = previous_weights[:, parent_indices, :].reshape((-1, 2)).copy()
        rr, ww, count = _solve([prefix[-1] for prefix in prefixes], w, count, True)
        for i, prefix in enumerate(prefixes):
            residuals[prefix] = rr[:, i].copy()
        previous, previous_weights = {prefix: i for i, prefix in enumerate(prefixes)}, ww
    permutations = tuple(_itertools.permutations(range(nrhs)))
    orders = np.asarray(permutations, dtype=np.int64)
    fixed = np.asarray(fixed_by_rhs[:, orders], dtype=np.float64)
    adaptive = np.stack([np.stack([residuals[order[:t+1]] for t in range(nrhs)], axis=1) for order in permutations], axis=1)
    return fixed, adaptive, previous_weights.copy(), np.full((nstates, len(permutations)), count, dtype=np.int64)


def batch_residuals(A: "np.ndarray", right_hand_sides: "np.ndarray", lower: "np.ndarray", upper: "np.ndarray",
                            theta: tuple, outer_iterations: int = 2, restarts: int = 2) -> tuple:
    lower, upper = _f3r_factor_arrays(lower, upper)
    n = lower.size
    A = _f3r_matrix(A, n)
    B = _f3r_real_array(right_hand_sides, 'right_hand_sides')
    if B.ndim != 2 or B.shape[1] != n or not 1 <= B.shape[0] <= 3 or np.any(np.all(B == 0, axis=1)):
        raise ValueError("right_hand_sides must have 1 to 3 nonzero rows of length n")
    if not isinstance(theta, (tuple, list)) or len(theta) != 3:
        raise ValueError("theta must contain three positive integers")
    m2, m3, interval = (_f3r_integer(v, 'theta entry') for v in theta)
    m1 = _f3r_integer(outer_iterations, 'outer_iterations')
    restarts = _f3r_integer(restarts, 'restarts')
    if max(m1, m2, m3) > n:
        raise ValueError("FGMRES lengths cannot exceed n")
    result = _f3rbatch_residuals_many(A, B, lower[None, :, :], upper[None, :, :], (m2, m3, interval), m1, restarts)
    return tuple(value[0] for value in result)

import numpy as np

def _f3r_outward(value, direction):
    v = float(value)
    rational_v = _Fraction(v)
    if direction < 0 and rational_v > value:
        v = float(np.nextafter(v, -np.inf))
    elif direction > 0 and rational_v < value:
        v = float(np.nextafter(v, np.inf))
    return v

def safe_component_margins(boundaries: tuple, strata: "np.ndarray", gains: "np.ndarray",
                                  threshold: float = 2.0) -> "np.ndarray":
    if not isinstance(boundaries, (tuple, list)) or len(boundaries) == 0:
        raise ValueError("At least one boundary is required")
    brackets = []
    for desc in boundaries:
        if not isinstance(desc, (tuple, list)) or len(desc) != 3:
            raise ValueError("Invalid boundary descriptor")
        coeff, lp, hp = desc
        if (not isinstance(coeff, (tuple, list)) or len(coeff) < 2
                or any(isinstance(c, bool) or not isinstance(c, (int, np.integer)) for c in coeff)
                or coeff[0] <= 0):
            raise ValueError("Invalid integer boundary polynomial")
        lo, hi = _f3r_fraction(lp), _f3r_fraction(hp)
        if not 1 <= lo <= hi <= 2 or hi - lo > _Fraction(1, 10**24):
            raise ValueError("Invalid boundary bracket")
        if brackets and brackets[-1][1] >= lo:
            raise ValueError("Boundary brackets must be disjoint and ordered")
        brackets.append((lo, hi))
    st = np.asarray(strata)
    g = _f3r_real_array(gains, 'gains')
    nb = len(brackets)
    if st.dtype.kind not in 'iu' or st.shape != (2*nb-1,3):
        raise ValueError("Invalid stratum structure")
    expected = [(i,i) if j % 2 == 0 else (i,i+1) for j in range(2*nb-1) for i in [j//2]]
    if any(tuple(row[:2]) != exp for row, exp in zip(st, expected)):
        raise ValueError("Strata must alternate all points and open intervals")
    if g.ndim != 2 or g.shape[0] < 1 or g.shape[1] < 1 or np.any(g < 0) or np.any(st[:,2] < 0) or np.any(st[:,2] >= g.shape[0]):
        raise ValueError("Invalid gain table or state IDs")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("threshold must be positive and finite") from exc
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    result = []
    def _append(configuration, start, stop):
        li, ri = int(st[start,0]), int(st[stop,1])
        lc, rc = int(start % 2 == 0), int(stop % 2 == 0)
        if li == ri:
            qlow = qhigh = _Fraction(0)
        else:
            al, au = brackets[li]
            bl, bu = brackets[ri]
            qlow = 10**6 * (bl-au)/(bl+au)
            qhigh = 10**6 * (bu-al)/(bu+al)
        result.append((configuration,li,ri,lc,rc,_f3r_outward(qlow,-1),_f3r_outward(qhigh,1)))
    for j in range(g.shape[1]):
        first = None
        for i, row in enumerate(st):
            safe = g[int(row[2]),j] >= threshold
            if safe and first is None:
                first = i
            elif not safe and first is not None:
                _append(j,first,i-1)
                first = None
        if first is not None:
            _append(j,first,len(st)-1)
    return np.asarray(result, dtype=np.float64).reshape(-1,7)

import numpy as np

def _f3r_benchmark_data():
    n = 48
    i = np.arange(n)
    A = np.zeros((n,n), dtype=np.float64)
    A[i,i] = 1 + (1+i%3)/4096
    for offset,value in ((1,-7/16),(-1,-3/16),(8,-1/4),(-8,-1/8)):
        A[i,(i+offset)%n] = value
    B = np.stack((1+(-1.0)**i/4+(i%7)/32,
                  (-1.0)**i*(1+(i%5)/16),
                  (((3*i)%11)-5)/8+(i%2)/32))
    return A,B

def solve_order_robust_margin(domain: tuple = ((287, 256), (289, 256)),
                                     configurations: tuple | None = None, threshold: float = 2.0) -> float:
    if not isinstance(domain, (tuple,list)) or len(domain) != 2:
        raise ValueError("domain must contain two rational endpoints")
    a,b = map(_f3r_fraction,domain)
    if not _Fraction(287,256) <= a <= b <= _Fraction(289,256):
        raise ValueError("domain is outside the benchmark interval")
    allowed = tuple((u,v,c) for u,v in ((3,4),(4,3),(6,2)) for c in (5,11,17,23,31))
    if configurations is None:
        configs = allowed
    else:
        if not isinstance(configurations,(tuple,list)) or not configurations:
            raise ValueError("configurations must be nonempty")
        configs = []
        for row in configurations:
            if not isinstance(row,(tuple,list)) or len(row) != 3:
                raise ValueError("Invalid configuration")
            row = tuple(_f3r_integer(v,'configuration entry') for v in row)
            if row not in allowed or row in configs:
                raise ValueError("Invalid or duplicate configuration")
            configs.append(row)
        configs = tuple(configs)
    try:
        threshold = float(threshold)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError("threshold must be positive and finite") from exc
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("threshold must be positive and finite")
    A,B = _f3r_benchmark_data()
    boundaries,strata,lower,upper = factor_rounding_regions(np.diag(A),domain)
    gains = np.empty((len(lower), len(configs)), dtype=np.float64)
    # Batch independent factor states. Each row keeps its own adaptive history.
    # Reuse only fixed-weight controls with the same nesting, matrix and factors.
    # Bound temporary memory while batching independent factor states.
    for start in range(0, len(lower), 64):
        stop = min(start + 64, len(lower))
        fixed_controls = {}
        for j, theta in enumerate(configs):
            nesting = theta[:2]
            fixed, adaptive, weights, counters = _f3rbatch_residuals_many(
                A, B, lower[start:stop], upper[start:stop], theta, 2, 2,
                fixed_controls.get(nesting))
            if nesting not in fixed_controls:
                fixed_controls[nesting] = fixed[:, 0, :].copy()
            if np.any(adaptive <= 0):
                raise ValueError("Adaptive residual denominator must be positive")
            ratios = np.divide(fixed, adaptive, dtype=np.float64)
            gains[start:stop, j] = np.min(ratios, axis=(1, 2))
    components = safe_component_margins(boundaries,strata,gains,threshold)
    if len(components) == 0:
        raise ValueError("No safe multiplier exists")
    low,high = float(components[:,5].max()),float(components[:,6].max())
    return float(low + (high-low)/2)
SCICODE_GOLD_EOF
