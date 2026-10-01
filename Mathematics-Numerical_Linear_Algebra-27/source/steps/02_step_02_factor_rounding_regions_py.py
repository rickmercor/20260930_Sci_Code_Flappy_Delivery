"""
Partition the closed multiplier domain into all factor-rounding point and open-interval states.

Each exact block-factor entry is a monotone rational function of the multiplier on [1, 2]. Its binary16 value changes only at midpoints of adjacent representable values. Exact root isolation and ties-to-even distinguish transition points from the neighbouring open intervals. The returned numeric descriptors preserve endpoint states without a parameter grid.



For each exact nonconstant factor entry f(lambda), a candidate transition satisfies f(lambda) = (h_minus + h_plus)/2, where h_minus and h_plus are adjacent fp16 values. A boundary record specifies an exact root; a stratum is either one boundary point or one open interval.

Returns
-------
tuple: numeric algebraic boundary descriptors, int64 stratum rows, and two binary16 factor-state arrays.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def factor_rounding_regions(diagonal: "np.ndarray", domain: tuple) -> tuple:
    """Enumerate every stored factor state, including isolated boundaries.

    Parameters
    ----------
    diagonal : np.ndarray
        Real finite vector of length 4*b, b >= 1, with entries in [1, 2].
        Block interpretation and rounding are those of build_block_factors.
    domain : tuple
        Two integer rational pairs (left, right), defining a closed interval
        1 <= left <= right <= 2. A singleton interval is permitted.

    Returns
    -------
    boundaries, strata, lower_states, upper_states : tuple
        boundaries is a tuple in increasing root order. Each entry is
        (coefficients, low, high): coefficients are the primitive integer
        coefficients in descending degree, with positive leading coefficient,
        of a polynomial specifying the boundary; low and high are integer
        rational pairs isolating its unique root. Rational boundaries have
        low == high and a linear polynomial. Each bracket has width <= 1e-24.
        Only domain endpoints and genuine internal factor transitions occur.
        strata is an int64 array of shape (2*len(boundaries)-1, 3), in spatial
        order: (left_boundary_id, right_boundary_id, state_id). Equal endpoint
        IDs denote a singleton; consecutive IDs denote an OPEN interval.
        Endpoints use their exact ties-to-even states, not one-sided limits.
        lower_states and upper_states are binary16 arrays of shape (s, b, 4).
        State IDs are assigned at first occurrence in strata, with duplicate
        full-factor states sharing an ID. All returned values are numeric;
        no symbolic expressions or strings occur in the result.

    Raises
    ------
    ValueError
        If diagonal violates its stated domain, domain is not two integer
        rational pairs with positive denominators, or its bounds do not
        satisfy 1 <= left <= right <= 2.
    """
    return boundaries, strata, lower_states, upper_states  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_factor_rounding_regions(diagonal: "np.ndarray", domain: tuple) -> tuple:
    a = _f3r_diagonal(diagonal)
    if not isinstance(domain, (tuple, list)) or len(domain) != 2:
        raise ValueError("domain must contain two rational endpoints")
    left, right = map(_f3r_fraction, domain)
    if not 1 <= left <= right <= 2:
        raise ValueError("Require 1 <= left <= right <= 2")
    if left == right:
        low, up = _oracle_build_block_factors(a, _f3r_pair(left))
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
        lower, upper = _oracle_build_block_factors(a, _f3r_pair(middle))
        # At an algebraic boundary, every changing factor is set from its
        # exact midpoint tie; the representative alone would be one-sided.
        for kind, block, k, tie in bound['changes']:
            (lower if kind == 0 else upper)[block, k] = np.float16(tie)
        _record(lower, upper, i, i)
        if i + 1 < len(bounds):
            middle = (bound['hi'] + bounds[i + 1]['lo']) / 2
            lower, upper = _oracle_build_block_factors(a, _f3r_pair(middle))
            _record(lower, upper, i, i + 1)
    numeric_bounds = tuple((e['coeff'], _f3r_pair(e['lo']), _f3r_pair(e['hi'])) for e in bounds)
    return (numeric_bounds, np.asarray(strata, dtype=np.int64),
            np.stack([s[0] for s in states]), np.stack([s[1] for s in states]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the original cases with exact standard-library root checks."""
    return [{'setup': '\n'
               'diagonal=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64).reshape((4,))\n'
               'domain=((1, 1), (1025, 1024))\n'
               'expected=((((1, -1), (1, 1), (1, 1)), ((431744, -393216, -70833, 32256), (1090146623750316, '
               '1090024699249033), (1572934040181553, 1572758119627178)), ((4096, -3761, -336), '
               '(4211853896379171, 4210903766464244), (4029177057086333, 4028268136276481)), ((1048576, -955136, '
               '-172032, 78351), (1478450331706598, 1478115507896945), (1299627031759451, 1299332705961303)), '
               '((3071, -3072), (3072, 3071), (3072, 3071)), ((285440, -262144, -23415), (921782511898189, '
               '921427480987031), (1316498087721729, 1315991029375982)), ((524288, -477312, -129024, 78309, '
               '3528), (1185037026657097, 1184522933122921), (1548517011833060, 1547845233175147)), ((2048, '
               '-2049), (2049, 2048), (2049, 2048)), ((431488, -393216, -70791, 32256), (2116643470152731, '
               '2115372649827284), (1793759250434953, 1792682287901321)), ((1048576, -955648, -172032, 78393), '
               '(1919880949508464, 1918598714647907), (1167060907130363, 1166281459748535)), ((4096, -3763, '
               '-336), (1558787660147621, 1557733183787474), (884702171761346, 884103695426343)), ((524288, '
               '-477568, -129024, 78351, 3528), (567116398156401, 566621035894426), (2703188923946476, '
               '2700827754732775)), ((855808, -786432, -70203), (3925412654324005, 3921909846175559), '
               '(282985583087789, 282733063341810)), ((1024, -1025), (1025, 1024), (1025, 1024))), np.array([[0, '
               '0, 0], [0, 1, 0], [1, 1, 1], [1, 2, 1], [2, 2, 1], [2, 3, 2], [3, 3, 3], [3, 4, 3], [4, 4, 3], '
               '[4, 5, 4], [5, 5, 5], [5, 6, 5], [6, 6, 5], [6, 7, 6], [7, 7, 6], [7, 8, 7], [8, 8, 7], [8, 9, '
               '8], [9, 9, 8], [9, 10, 9], [10, 10, 10], [10, 11, 10], [11, 11, 11], [11, 12, 11], [12, 12, 11], '
               '[12, 13, 12], [13, 13, 12]], dtype=np.int64).reshape((27, 3)), np.array([[[0.0, -0.1875, '
               '-0.2042236328125, -0.2059326171875]], [[0.0, -0.1875, -0.2042236328125, -0.205810546875]], '
               '[[0.0, -0.1875, -0.2042236328125, -0.205810546875]], [[0.0, -0.1875, -0.2042236328125, '
               '-0.205810546875]], [[0.0, -0.1873779296875, -0.2042236328125, -0.205810546875]], [[0.0, '
               '-0.1873779296875, -0.2041015625, -0.205810546875]], [[0.0, -0.1873779296875, -0.2041015625, '
               '-0.205810546875]], [[0.0, -0.1873779296875, -0.2041015625, -0.205810546875]], [[0.0, '
               '-0.1873779296875, -0.2041015625, -0.2056884765625]], [[0.0, -0.1873779296875, -0.2041015625, '
               '-0.2056884765625]], [[0.0, -0.1873779296875, -0.2041015625, -0.2056884765625]], [[0.0, '
               '-0.1873779296875, -0.2041015625, -0.2056884765625]], [[0.0, -0.1873779296875, -0.2039794921875, '
               '-0.2056884765625]]], dtype=np.float16).reshape((13, 1, 4)), np.array([[[1.0, 0.91796875, '
               '0.91064453125, 0.91015625]], [[1.0, 0.91796875, 0.91064453125, 0.91015625]], [[1.0, '
               '0.91845703125, 0.91064453125, 0.91015625]], [[1.0, 0.91845703125, 0.9111328125, 0.91015625]], '
               '[[1.0, 0.91845703125, 0.9111328125, 0.91015625]], [[1.0, 0.91845703125, 0.9111328125, '
               '0.91015625]], [[1.0, 0.91845703125, 0.9111328125, 0.91064453125]], [[1.0009765625, '
               '0.91845703125, 0.9111328125, 0.91064453125]], [[1.0009765625, 0.91845703125, 0.9111328125, '
               '0.91064453125]], [[1.0009765625, 0.91845703125, 0.91162109375, 0.91064453125]], [[1.0009765625, '
               '0.9189453125, 0.91162109375, 0.91064453125]], [[1.0009765625, 0.9189453125, 0.91162109375, '
               '0.9111328125]], [[1.0009765625, 0.9189453125, 0.91162109375, 0.9111328125]]], '
               'dtype=np.float16).reshape((13, 1, 4)))\n'
               'import numpy as np\n'
               'from fractions import Fraction as _CaseFraction\n'
               'from math import gcd as _case_integer_gcd\n'
               '\n'
               '\n'
               'def _case_trim(p):\n'
               '    p = [_CaseFraction(int(c) if isinstance(c, np.integer) else c) for c in p]\n'
               '    while len(p) > 1 and p[0] == 0:\n'
               '        p.pop(0)\n'
               '    return tuple(p) if p else (_CaseFraction(0),)\n'
               '\n'
               '\n'
               'def _case_poly_value(p, x):\n'
               '    result = _CaseFraction(0)\n'
               '    for c in p:\n'
               '        result = result * x + c\n'
               '    return result\n'
               '\n'
               '\n'
               'def _case_divrem(p, q):\n'
               '    p, q = list(_case_trim(p)), _case_trim(q)\n'
               '    if q == (0,):\n'
               '        raise ValueError("Zero polynomial divisor")\n'
               '    quotient = [_CaseFraction(0)] * max(1, len(p) - len(q) + 1)\n'
               '    degree = len(quotient) - 1\n'
               '    while p != [0] and len(p) >= len(q):\n'
               '        offset = len(p) - len(q)\n'
               '        c = p[0] / q[0]\n'
               '        quotient[degree-offset] += c\n'
               '        for i in range(len(q)):\n'
               '            p[i] -= c*q[i]\n'
               '        p = list(_case_trim(p))\n'
               '    return _case_trim(quotient), tuple(p)\n'
               '\n'
               '\n'
               'def _case_gcd(p, q):\n'
               '    p, q = _case_trim(p), _case_trim(q)\n'
               '    while q != (0,):\n'
               '        p, q = q, _case_divrem(p, q)[1]\n'
               '    return tuple(c / p[0] for c in p) if p != (0,) else p\n'
               '\n'
               '\n'
               'def _case_count_roots_closed(p, a, b):\n'
               '    p = _case_trim(p)\n'
               '    if a > b or len(p) <= 1:\n'
               '        return 0\n'
               '    if a == b:\n'
               '        return int(_case_poly_value(p, a) == 0)\n'
               '    dp = tuple((len(p)-1-i)*p[i] for i in range(len(p)-1))\n'
               '    p = _case_divrem(p, _case_gcd(p, dp))[0]\n'
               '    dp = tuple((len(p)-1-i)*p[i] for i in range(len(p)-1))\n'
               '    chain = [p, dp]\n'
               '    while True:\n'
               '        r = _case_divrem(chain[-2], chain[-1])[1]\n'
               '        if r == (0,):\n'
               '            break\n'
               '        chain.append(tuple(-c / abs(r[0]) for c in r))\n'
               '    def _changes(x):\n'
               '        values = [_case_poly_value(s, x) for s in chain]\n'
               '        signs = [1 if v > 0 else -1 for v in values if v]\n'
               '        return sum(s != t for s, t in zip(signs, signs[1:]))\n'
               '    return _changes(a)-_changes(b)+int(_case_poly_value(p,a)==0)\n'
               '\n'
               '\n'
               'def _case_region_check(actual):\n'
               '    try:\n'
               '        if not isinstance(actual, tuple) or len(actual) != 4:\n'
               '            return 0\n'
               '        b, st, l, u = actual\n'
               '        eb, es, el, eu = expected\n'
               '        if not isinstance(b, tuple) or len(b) != len(eb):\n'
               '            return 0\n'
               '        if not isinstance(st, np.ndarray) or st.dtype != np.int64 or not np.array_equal(st, '
               'es):\n'
               '            return 0\n'
               '        for a, e in ((l, el), (u, eu)):\n'
               '            if not isinstance(a, np.ndarray) or a.dtype != np.float16 or not np.array_equal(a, '
               'e):\n'
               '                return 0\n'
               '        last = None\n'
               '        for (coeff, lo, hi), (ec, elo, ehi) in zip(b, eb):\n'
               '            if not isinstance(coeff, tuple) or len(coeff) < 2 or any(not isinstance(v, (int, '
               'np.integer)) for v in coeff):\n'
               '                return 0\n'
               '            aa = _CaseFraction(*(int(v) if isinstance(v, np.integer) else v for v in lo))\n'
               '            bb = _CaseFraction(*(int(v) if isinstance(v, np.integer) else v for v in hi))\n'
               '            al, bl = _CaseFraction(*elo), _CaseFraction(*ehi)\n'
               '            if aa > bb or bb-aa > _CaseFraction(1, 10**24):\n'
               '                return 0\n'
               '            if last is not None and last >= aa:\n'
               '                return 0\n'
               '            last = bb\n'
               '            content = 0\n'
               '            for c in coeff:\n'
               '                content = _case_integer_gcd(content, abs(int(c)))\n'
               '            if coeff[0] <= 0 or content != 1:\n'
               '                return 0\n'
               '            common = _case_gcd(coeff, ec)\n'
               '            lower, upper = max(aa, al), min(bb, bl)\n'
               '            if _case_count_roots_closed(common, lower, upper) != 1:\n'
               '                return 0\n'
               '        return 1\n'
               '    except (TypeError, ValueError, OverflowError, IndexError, AttributeError, '
               'ZeroDivisionError):\n'
               '        return 0\n',
      'call': '_case_region_check(factor_rounding_regions(diagonal,domain))',
      'gold_call': '_case_region_check(_oracle_factor_rounding_regions(diagonal,domain))'},
     {'setup': '\n'
               'diagonal=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64).reshape((4,))\n'
               'domain=((2049, 2048), (2049, 2048))\n'
               'expected=((((2048, -2049), (2049, 2048), (2049, 2048)),), np.array([[0, 0, 0]], '
               'dtype=np.int64).reshape((1, 3)), np.array([[[0.0, -0.1873779296875, -0.2041015625, '
               '-0.205810546875]]], dtype=np.float16).reshape((1, 1, 4)), np.array([[[1.0, 0.91845703125, '
               '0.9111328125, 0.91064453125]]], dtype=np.float16).reshape((1, 1, 4)))\n'
               'import numpy as np\n'
               'from fractions import Fraction as _CaseFraction\n'
               'from math import gcd as _case_integer_gcd\n'
               '\n'
               '\n'
               'def _case_trim(p):\n'
               '    p = [_CaseFraction(int(c) if isinstance(c, np.integer) else c) for c in p]\n'
               '    while len(p) > 1 and p[0] == 0:\n'
               '        p.pop(0)\n'
               '    return tuple(p) if p else (_CaseFraction(0),)\n'
               '\n'
               '\n'
               'def _case_poly_value(p, x):\n'
               '    result = _CaseFraction(0)\n'
               '    for c in p:\n'
               '        result = result * x + c\n'
               '    return result\n'
               '\n'
               '\n'
               'def _case_divrem(p, q):\n'
               '    p, q = list(_case_trim(p)), _case_trim(q)\n'
               '    if q == (0,):\n'
               '        raise ValueError("Zero polynomial divisor")\n'
               '    quotient = [_CaseFraction(0)] * max(1, len(p) - len(q) + 1)\n'
               '    degree = len(quotient) - 1\n'
               '    while p != [0] and len(p) >= len(q):\n'
               '        offset = len(p) - len(q)\n'
               '        c = p[0] / q[0]\n'
               '        quotient[degree-offset] += c\n'
               '        for i in range(len(q)):\n'
               '            p[i] -= c*q[i]\n'
               '        p = list(_case_trim(p))\n'
               '    return _case_trim(quotient), tuple(p)\n'
               '\n'
               '\n'
               'def _case_gcd(p, q):\n'
               '    p, q = _case_trim(p), _case_trim(q)\n'
               '    while q != (0,):\n'
               '        p, q = q, _case_divrem(p, q)[1]\n'
               '    return tuple(c / p[0] for c in p) if p != (0,) else p\n'
               '\n'
               '\n'
               'def _case_count_roots_closed(p, a, b):\n'
               '    p = _case_trim(p)\n'
               '    if a > b or len(p) <= 1:\n'
               '        return 0\n'
               '    if a == b:\n'
               '        return int(_case_poly_value(p, a) == 0)\n'
               '    dp = tuple((len(p)-1-i)*p[i] for i in range(len(p)-1))\n'
               '    p = _case_divrem(p, _case_gcd(p, dp))[0]\n'
               '    dp = tuple((len(p)-1-i)*p[i] for i in range(len(p)-1))\n'
               '    chain = [p, dp]\n'
               '    while True:\n'
               '        r = _case_divrem(chain[-2], chain[-1])[1]\n'
               '        if r == (0,):\n'
               '            break\n'
               '        chain.append(tuple(-c / abs(r[0]) for c in r))\n'
               '    def _changes(x):\n'
               '        values = [_case_poly_value(s, x) for s in chain]\n'
               '        signs = [1 if v > 0 else -1 for v in values if v]\n'
               '        return sum(s != t for s, t in zip(signs, signs[1:]))\n'
               '    return _changes(a)-_changes(b)+int(_case_poly_value(p,a)==0)\n'
               '\n'
               '\n'
               'def _case_region_check(actual):\n'
               '    try:\n'
               '        if not isinstance(actual, tuple) or len(actual) != 4:\n'
               '            return 0\n'
               '        b, st, l, u = actual\n'
               '        eb, es, el, eu = expected\n'
               '        if not isinstance(b, tuple) or len(b) != len(eb):\n'
               '            return 0\n'
               '        if not isinstance(st, np.ndarray) or st.dtype != np.int64 or not np.array_equal(st, '
               'es):\n'
               '            return 0\n'
               '        for a, e in ((l, el), (u, eu)):\n'
               '            if not isinstance(a, np.ndarray) or a.dtype != np.float16 or not np.array_equal(a, '
               'e):\n'
               '                return 0\n'
               '        last = None\n'
               '        for (coeff, lo, hi), (ec, elo, ehi) in zip(b, eb):\n'
               '            if not isinstance(coeff, tuple) or len(coeff) < 2 or any(not isinstance(v, (int, '
               'np.integer)) for v in coeff):\n'
               '                return 0\n'
               '            aa = _CaseFraction(*(int(v) if isinstance(v, np.integer) else v for v in lo))\n'
               '            bb = _CaseFraction(*(int(v) if isinstance(v, np.integer) else v for v in hi))\n'
               '            al, bl = _CaseFraction(*elo), _CaseFraction(*ehi)\n'
               '            if aa > bb or bb-aa > _CaseFraction(1, 10**24):\n'
               '                return 0\n'
               '            if last is not None and last >= aa:\n'
               '                return 0\n'
               '            last = bb\n'
               '            content = 0\n'
               '            for c in coeff:\n'
               '                content = _case_integer_gcd(content, abs(int(c)))\n'
               '            if coeff[0] <= 0 or content != 1:\n'
               '                return 0\n'
               '            common = _case_gcd(coeff, ec)\n'
               '            lower, upper = max(aa, al), min(bb, bl)\n'
               '            if _case_count_roots_closed(common, lower, upper) != 1:\n'
               '                return 0\n'
               '        return 1\n'
               '    except (TypeError, ValueError, OverflowError, IndexError, AttributeError, '
               'ZeroDivisionError):\n'
               '        return 0\n',
      'call': '_case_region_check(factor_rounding_regions(diagonal,domain))',
      'gold_call': '_case_region_check(_oracle_factor_rounding_regions(diagonal,domain))'},
     {'setup': '\n'
               'diagonal=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64).reshape((4,))\n'
               'domain=((2049, 2048), (1025, 1024))\n'
               'expected=((((2048, -2049), (2049, 2048), (2049, 2048)), ((431488, -393216, -70791, 32256), '
               '(2116643470152731, 2115372649827284), (1793759250434953, 1792682287901321)), ((1048576, -955648, '
               '-172032, 78393), (1919880949508464, 1918598714647907), (1167060907130363, 1166281459748535)), '
               '((4096, -3763, -336), (1558787660147621, 1557733183787474), (884702171761346, 884103695426343)), '
               '((524288, -477568, -129024, 78351, 3528), (567116398156401, 566621035894426), (2703188923946476, '
               '2700827754732775)), ((855808, -786432, -70203), (3925412654324005, 3921909846175559), '
               '(282985583087789, 282733063341810)), ((1024, -1025), (1025, 1024), (1025, 1024))), np.array([[0, '
               '0, 0], [0, 1, 1], [1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 3, 3], [3, 3, 4], [3, 4, 4], [4, 4, 5], '
               '[4, 5, 5], [5, 5, 5], [5, 6, 6], [6, 6, 6]], dtype=np.int64).reshape((13, 3)), np.array([[[0.0, '
               '-0.1873779296875, -0.2041015625, -0.205810546875]], [[0.0, -0.1873779296875, -0.2041015625, '
               '-0.205810546875]], [[0.0, -0.1873779296875, -0.2041015625, -0.2056884765625]], [[0.0, '
               '-0.1873779296875, -0.2041015625, -0.2056884765625]], [[0.0, -0.1873779296875, -0.2041015625, '
               '-0.2056884765625]], [[0.0, -0.1873779296875, -0.2041015625, -0.2056884765625]], [[0.0, '
               '-0.1873779296875, -0.2039794921875, -0.2056884765625]]], dtype=np.float16).reshape((7, 1, 4)), '
               'np.array([[[1.0, 0.91845703125, 0.9111328125, 0.91064453125]], [[1.0009765625, 0.91845703125, '
               '0.9111328125, 0.91064453125]], [[1.0009765625, 0.91845703125, 0.9111328125, 0.91064453125]], '
               '[[1.0009765625, 0.91845703125, 0.91162109375, 0.91064453125]], [[1.0009765625, 0.9189453125, '
               '0.91162109375, 0.91064453125]], [[1.0009765625, 0.9189453125, 0.91162109375, 0.9111328125]], '
               '[[1.0009765625, 0.9189453125, 0.91162109375, 0.9111328125]]], dtype=np.float16).reshape((7, 1, '
               '4)))\n'
               'import numpy as np\n'
               'from fractions import Fraction as _CaseFraction\n'
               'from math import gcd as _case_integer_gcd\n'
               '\n'
               '\n'
               'def _case_trim(p):\n'
               '    p = [_CaseFraction(int(c) if isinstance(c, np.integer) else c) for c in p]\n'
               '    while len(p) > 1 and p[0] == 0:\n'
               '        p.pop(0)\n'
               '    return tuple(p) if p else (_CaseFraction(0),)\n'
               '\n'
               '\n'
               'def _case_poly_value(p, x):\n'
               '    result = _CaseFraction(0)\n'
               '    for c in p:\n'
               '        result = result * x + c\n'
               '    return result\n'
               '\n'
               '\n'
               'def _case_divrem(p, q):\n'
               '    p, q = list(_case_trim(p)), _case_trim(q)\n'
               '    if q == (0,):\n'
               '        raise ValueError("Zero polynomial divisor")\n'
               '    quotient = [_CaseFraction(0)] * max(1, len(p) - len(q) + 1)\n'
               '    degree = len(quotient) - 1\n'
               '    while p != [0] and len(p) >= len(q):\n'
               '        offset = len(p) - len(q)\n'
               '        c = p[0] / q[0]\n'
               '        quotient[degree-offset] += c\n'
               '        for i in range(len(q)):\n'
               '            p[i] -= c*q[i]\n'
               '        p = list(_case_trim(p))\n'
               '    return _case_trim(quotient), tuple(p)\n'
               '\n'
               '\n'
               'def _case_gcd(p, q):\n'
               '    p, q = _case_trim(p), _case_trim(q)\n'
               '    while q != (0,):\n'
               '        p, q = q, _case_divrem(p, q)[1]\n'
               '    return tuple(c / p[0] for c in p) if p != (0,) else p\n'
               '\n'
               '\n'
               'def _case_count_roots_closed(p, a, b):\n'
               '    p = _case_trim(p)\n'
               '    if a > b or len(p) <= 1:\n'
               '        return 0\n'
               '    if a == b:\n'
               '        return int(_case_poly_value(p, a) == 0)\n'
               '    dp = tuple((len(p)-1-i)*p[i] for i in range(len(p)-1))\n'
               '    p = _case_divrem(p, _case_gcd(p, dp))[0]\n'
               '    dp = tuple((len(p)-1-i)*p[i] for i in range(len(p)-1))\n'
               '    chain = [p, dp]\n'
               '    while True:\n'
               '        r = _case_divrem(chain[-2], chain[-1])[1]\n'
               '        if r == (0,):\n'
               '            break\n'
               '        chain.append(tuple(-c / abs(r[0]) for c in r))\n'
               '    def _changes(x):\n'
               '        values = [_case_poly_value(s, x) for s in chain]\n'
               '        signs = [1 if v > 0 else -1 for v in values if v]\n'
               '        return sum(s != t for s, t in zip(signs, signs[1:]))\n'
               '    return _changes(a)-_changes(b)+int(_case_poly_value(p,a)==0)\n'
               '\n'
               '\n'
               'def _case_region_check(actual):\n'
               '    try:\n'
               '        if not isinstance(actual, tuple) or len(actual) != 4:\n'
               '            return 0\n'
               '        b, st, l, u = actual\n'
               '        eb, es, el, eu = expected\n'
               '        if not isinstance(b, tuple) or len(b) != len(eb):\n'
               '            return 0\n'
               '        if not isinstance(st, np.ndarray) or st.dtype != np.int64 or not np.array_equal(st, '
               'es):\n'
               '            return 0\n'
               '        for a, e in ((l, el), (u, eu)):\n'
               '            if not isinstance(a, np.ndarray) or a.dtype != np.float16 or not np.array_equal(a, '
               'e):\n'
               '                return 0\n'
               '        last = None\n'
               '        for (coeff, lo, hi), (ec, elo, ehi) in zip(b, eb):\n'
               '            if not isinstance(coeff, tuple) or len(coeff) < 2 or any(not isinstance(v, (int, '
               'np.integer)) for v in coeff):\n'
               '                return 0\n'
               '            aa = _CaseFraction(*(int(v) if isinstance(v, np.integer) else v for v in lo))\n'
               '            bb = _CaseFraction(*(int(v) if isinstance(v, np.integer) else v for v in hi))\n'
               '            al, bl = _CaseFraction(*elo), _CaseFraction(*ehi)\n'
               '            if aa > bb or bb-aa > _CaseFraction(1, 10**24):\n'
               '                return 0\n'
               '            if last is not None and last >= aa:\n'
               '                return 0\n'
               '            last = bb\n'
               '            content = 0\n'
               '            for c in coeff:\n'
               '                content = _case_integer_gcd(content, abs(int(c)))\n'
               '            if coeff[0] <= 0 or content != 1:\n'
               '                return 0\n'
               '            common = _case_gcd(coeff, ec)\n'
               '            lower, upper = max(aa, al), min(bb, bl)\n'
               '            if _case_count_roots_closed(common, lower, upper) != 1:\n'
               '                return 0\n'
               '        return 1\n'
               '    except (TypeError, ValueError, OverflowError, IndexError, AttributeError, '
               'ZeroDivisionError):\n'
               '        return 0\n',
      'call': '_case_region_check(factor_rounding_regions(diagonal,domain))',
      'gold_call': '_case_region_check(_oracle_factor_rounding_regions(diagonal,domain))'},
     {'setup': '\n'
               'diagonal=np.array([1.000244140625, 1.00048828125, 1.000732421875, 1.000244140625, 1.00048828125, '
               '1.000732421875, 1.000244140625, 1.00048828125, 1.000732421875, 1.000244140625, 1.00048828125, '
               '1.000732421875, 1.000244140625, 1.00048828125, 1.000732421875, 1.000244140625, 1.00048828125, '
               '1.000732421875, 1.000244140625, 1.00048828125, 1.000732421875, 1.000244140625, 1.00048828125, '
               '1.000732421875, 1.000244140625, 1.00048828125, 1.000732421875, 1.000244140625, 1.00048828125, '
               '1.000732421875, 1.000244140625, 1.00048828125, 1.000732421875, 1.000244140625, 1.00048828125, '
               '1.000732421875, 1.000244140625, 1.00048828125, 1.000732421875, 1.000244140625, 1.00048828125, '
               '1.000732421875, 1.000244140625, 1.00048828125, 1.000732421875, 1.000244140625, 1.00048828125, '
               '1.000732421875], dtype=np.float64).reshape((48,))\n'
               'domain=((11227, 10000), (11228, 10000))\n'
               'expected=((((10000, -11227), (11227, 10000), (11227, 10000)), ((4099, -4602), (4602, 4099), '
               '(4602, 4099)), ((8184884175, -8592031744, -670924800), (1669279304540508, 1486773464455345), '
               '(881117552199841, 784783104968268)), ((2798251, -2937549, -229376), (2151187090871018, '
               '1915934704219731), (2996183994845095, 2668523309902858)), ((2500, -2807), (2807, 2500), (2807, '
               '2500))), np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 2, 1], [2, 2, 2], [2, 3, 2], [3, 3, 3], '
               '[3, 4, 3], [4, 4, 3]], dtype=np.int64).reshape((9, 3)), np.array([[[0.0, -0.1669921875, '
               '-0.1785888671875, -0.1793212890625], [0.0, -0.1668701171875, -0.178466796875, -0.179443359375], '
               '[0.0, -0.1668701171875, -0.1785888671875, -0.179443359375], [0.0, -0.1669921875, '
               '-0.1785888671875, -0.1793212890625], [0.0, -0.1668701171875, -0.178466796875, -0.179443359375], '
               '[0.0, -0.1668701171875, -0.1785888671875, -0.179443359375], [0.0, -0.1669921875, '
               '-0.1785888671875, -0.1793212890625], [0.0, -0.1668701171875, -0.178466796875, -0.179443359375], '
               '[0.0, -0.1668701171875, -0.1785888671875, -0.179443359375], [0.0, -0.1669921875, '
               '-0.1785888671875, -0.1793212890625], [0.0, -0.1668701171875, -0.178466796875, -0.179443359375], '
               '[0.0, -0.1668701171875, -0.1785888671875, -0.179443359375]], [[0.0, -0.1669921875, '
               '-0.1785888671875, -0.1793212890625], [0.0, -0.1668701171875, -0.178466796875, -0.179443359375], '
               '[0.0, -0.1668701171875, -0.1785888671875, -0.179443359375], [0.0, -0.1669921875, '
               '-0.1785888671875, -0.1793212890625], [0.0, -0.1668701171875, -0.178466796875, -0.179443359375], '
               '[0.0, -0.1668701171875, -0.1785888671875, -0.179443359375], [0.0, -0.1669921875, '
               '-0.1785888671875, -0.1793212890625], [0.0, -0.1668701171875, -0.178466796875, -0.179443359375], '
               '[0.0, -0.1668701171875, -0.1785888671875, -0.179443359375], [0.0, -0.1669921875, '
               '-0.1785888671875, -0.1793212890625], [0.0, -0.1668701171875, -0.178466796875, -0.179443359375], '
               '[0.0, -0.1668701171875, -0.1785888671875, -0.179443359375]], [[0.0, -0.1669921875, '
               '-0.178466796875, -0.1793212890625], [0.0, -0.1668701171875, -0.178466796875, -0.179443359375], '
               '[0.0, -0.1668701171875, -0.1785888671875, -0.179443359375], [0.0, -0.1669921875, '
               '-0.178466796875, -0.1793212890625], [0.0, -0.1668701171875, -0.178466796875, -0.179443359375], '
               '[0.0, -0.1668701171875, -0.1785888671875, -0.179443359375], [0.0, -0.1669921875, '
               '-0.178466796875, -0.1793212890625], [0.0, -0.1668701171875, -0.178466796875, -0.179443359375], '
               '[0.0, -0.1668701171875, -0.1785888671875, -0.179443359375], [0.0, -0.1669921875, '
               '-0.178466796875, -0.1793212890625], [0.0, -0.1668701171875, -0.178466796875, -0.179443359375], '
               '[0.0, -0.1668701171875, -0.1785888671875, -0.179443359375]], [[0.0, -0.1669921875, '
               '-0.178466796875, -0.1793212890625], [0.0, -0.1668701171875, -0.178466796875, -0.179443359375], '
               '[0.0, -0.1668701171875, -0.1785888671875, -0.179443359375], [0.0, -0.1669921875, '
               '-0.178466796875, -0.1793212890625], [0.0, -0.1668701171875, -0.178466796875, -0.179443359375], '
               '[0.0, -0.1668701171875, -0.1785888671875, -0.179443359375], [0.0, -0.1669921875, '
               '-0.178466796875, -0.1793212890625], [0.0, -0.1668701171875, -0.178466796875, -0.179443359375], '
               '[0.0, -0.1668701171875, -0.1785888671875, -0.179443359375], [0.0, -0.1669921875, '
               '-0.178466796875, -0.1793212890625], [0.0, -0.1668701171875, -0.178466796875, -0.179443359375], '
               '[0.0, -0.1668701171875, -0.1785888671875, -0.179443359375]]], dtype=np.float16).reshape((4, 12, '
               '4)), np.array([[[1.123046875, 1.0498046875, 1.0458984375, 1.044921875], [1.123046875, '
               '1.05078125, 1.044921875, 1.044921875], [1.123046875, 1.0498046875, 1.044921875, 1.044921875], '
               '[1.123046875, 1.0498046875, 1.0458984375, 1.044921875], [1.123046875, 1.05078125, 1.044921875, '
               '1.044921875], [1.123046875, 1.0498046875, 1.044921875, 1.044921875], [1.123046875, 1.0498046875, '
               '1.0458984375, 1.044921875], [1.123046875, 1.05078125, 1.044921875, 1.044921875], [1.123046875, '
               '1.0498046875, 1.044921875, 1.044921875], [1.123046875, 1.0498046875, 1.0458984375, 1.044921875], '
               '[1.123046875, 1.05078125, 1.044921875, 1.044921875], [1.123046875, 1.0498046875, 1.044921875, '
               '1.044921875]], [[1.123046875, 1.0498046875, 1.0458984375, 1.044921875], [1.123046875, '
               '1.05078125, 1.044921875, 1.044921875], [1.1240234375, 1.0498046875, 1.044921875, 1.044921875], '
               '[1.123046875, 1.0498046875, 1.0458984375, 1.044921875], [1.123046875, 1.05078125, 1.044921875, '
               '1.044921875], [1.1240234375, 1.0498046875, 1.044921875, 1.044921875], [1.123046875, '
               '1.0498046875, 1.0458984375, 1.044921875], [1.123046875, 1.05078125, 1.044921875, 1.044921875], '
               '[1.1240234375, 1.0498046875, 1.044921875, 1.044921875], [1.123046875, 1.0498046875, '
               '1.0458984375, 1.044921875], [1.123046875, 1.05078125, 1.044921875, 1.044921875], [1.1240234375, '
               '1.0498046875, 1.044921875, 1.044921875]], [[1.123046875, 1.0498046875, 1.0458984375, '
               '1.044921875], [1.123046875, 1.05078125, 1.044921875, 1.044921875], [1.1240234375, 1.0498046875, '
               '1.044921875, 1.044921875], [1.123046875, 1.0498046875, 1.0458984375, 1.044921875], [1.123046875, '
               '1.05078125, 1.044921875, 1.044921875], [1.1240234375, 1.0498046875, 1.044921875, 1.044921875], '
               '[1.123046875, 1.0498046875, 1.0458984375, 1.044921875], [1.123046875, 1.05078125, 1.044921875, '
               '1.044921875], [1.1240234375, 1.0498046875, 1.044921875, 1.044921875], [1.123046875, '
               '1.0498046875, 1.0458984375, 1.044921875], [1.123046875, 1.05078125, 1.044921875, 1.044921875], '
               '[1.1240234375, 1.0498046875, 1.044921875, 1.044921875]], [[1.123046875, 1.05078125, '
               '1.0458984375, 1.044921875], [1.123046875, 1.05078125, 1.044921875, 1.044921875], [1.1240234375, '
               '1.0498046875, 1.044921875, 1.044921875], [1.123046875, 1.05078125, 1.0458984375, 1.044921875], '
               '[1.123046875, 1.05078125, 1.044921875, 1.044921875], [1.1240234375, 1.0498046875, 1.044921875, '
               '1.044921875], [1.123046875, 1.05078125, 1.0458984375, 1.044921875], [1.123046875, 1.05078125, '
               '1.044921875, 1.044921875], [1.1240234375, 1.0498046875, 1.044921875, 1.044921875], [1.123046875, '
               '1.05078125, 1.0458984375, 1.044921875], [1.123046875, 1.05078125, 1.044921875, 1.044921875], '
               '[1.1240234375, 1.0498046875, 1.044921875, 1.044921875]]], dtype=np.float16).reshape((4, 12, '
               '4)))\n'
               'import numpy as np\n'
               'from fractions import Fraction as _CaseFraction\n'
               'from math import gcd as _case_integer_gcd\n'
               '\n'
               '\n'
               'def _case_trim(p):\n'
               '    p = [_CaseFraction(int(c) if isinstance(c, np.integer) else c) for c in p]\n'
               '    while len(p) > 1 and p[0] == 0:\n'
               '        p.pop(0)\n'
               '    return tuple(p) if p else (_CaseFraction(0),)\n'
               '\n'
               '\n'
               'def _case_poly_value(p, x):\n'
               '    result = _CaseFraction(0)\n'
               '    for c in p:\n'
               '        result = result * x + c\n'
               '    return result\n'
               '\n'
               '\n'
               'def _case_divrem(p, q):\n'
               '    p, q = list(_case_trim(p)), _case_trim(q)\n'
               '    if q == (0,):\n'
               '        raise ValueError("Zero polynomial divisor")\n'
               '    quotient = [_CaseFraction(0)] * max(1, len(p) - len(q) + 1)\n'
               '    degree = len(quotient) - 1\n'
               '    while p != [0] and len(p) >= len(q):\n'
               '        offset = len(p) - len(q)\n'
               '        c = p[0] / q[0]\n'
               '        quotient[degree-offset] += c\n'
               '        for i in range(len(q)):\n'
               '            p[i] -= c*q[i]\n'
               '        p = list(_case_trim(p))\n'
               '    return _case_trim(quotient), tuple(p)\n'
               '\n'
               '\n'
               'def _case_gcd(p, q):\n'
               '    p, q = _case_trim(p), _case_trim(q)\n'
               '    while q != (0,):\n'
               '        p, q = q, _case_divrem(p, q)[1]\n'
               '    return tuple(c / p[0] for c in p) if p != (0,) else p\n'
               '\n'
               '\n'
               'def _case_count_roots_closed(p, a, b):\n'
               '    p = _case_trim(p)\n'
               '    if a > b or len(p) <= 1:\n'
               '        return 0\n'
               '    if a == b:\n'
               '        return int(_case_poly_value(p, a) == 0)\n'
               '    dp = tuple((len(p)-1-i)*p[i] for i in range(len(p)-1))\n'
               '    p = _case_divrem(p, _case_gcd(p, dp))[0]\n'
               '    dp = tuple((len(p)-1-i)*p[i] for i in range(len(p)-1))\n'
               '    chain = [p, dp]\n'
               '    while True:\n'
               '        r = _case_divrem(chain[-2], chain[-1])[1]\n'
               '        if r == (0,):\n'
               '            break\n'
               '        chain.append(tuple(-c / abs(r[0]) for c in r))\n'
               '    def _changes(x):\n'
               '        values = [_case_poly_value(s, x) for s in chain]\n'
               '        signs = [1 if v > 0 else -1 for v in values if v]\n'
               '        return sum(s != t for s, t in zip(signs, signs[1:]))\n'
               '    return _changes(a)-_changes(b)+int(_case_poly_value(p,a)==0)\n'
               '\n'
               '\n'
               'def _case_region_check(actual):\n'
               '    try:\n'
               '        if not isinstance(actual, tuple) or len(actual) != 4:\n'
               '            return 0\n'
               '        b, st, l, u = actual\n'
               '        eb, es, el, eu = expected\n'
               '        if not isinstance(b, tuple) or len(b) != len(eb):\n'
               '            return 0\n'
               '        if not isinstance(st, np.ndarray) or st.dtype != np.int64 or not np.array_equal(st, '
               'es):\n'
               '            return 0\n'
               '        for a, e in ((l, el), (u, eu)):\n'
               '            if not isinstance(a, np.ndarray) or a.dtype != np.float16 or not np.array_equal(a, '
               'e):\n'
               '                return 0\n'
               '        last = None\n'
               '        for (coeff, lo, hi), (ec, elo, ehi) in zip(b, eb):\n'
               '            if not isinstance(coeff, tuple) or len(coeff) < 2 or any(not isinstance(v, (int, '
               'np.integer)) for v in coeff):\n'
               '                return 0\n'
               '            aa = _CaseFraction(*(int(v) if isinstance(v, np.integer) else v for v in lo))\n'
               '            bb = _CaseFraction(*(int(v) if isinstance(v, np.integer) else v for v in hi))\n'
               '            al, bl = _CaseFraction(*elo), _CaseFraction(*ehi)\n'
               '            if aa > bb or bb-aa > _CaseFraction(1, 10**24):\n'
               '                return 0\n'
               '            if last is not None and last >= aa:\n'
               '                return 0\n'
               '            last = bb\n'
               '            content = 0\n'
               '            for c in coeff:\n'
               '                content = _case_integer_gcd(content, abs(int(c)))\n'
               '            if coeff[0] <= 0 or content != 1:\n'
               '                return 0\n'
               '            common = _case_gcd(coeff, ec)\n'
               '            lower, upper = max(aa, al), min(bb, bl)\n'
               '            if _case_count_roots_closed(common, lower, upper) != 1:\n'
               '                return 0\n'
               '        return 1\n'
               '    except (TypeError, ValueError, OverflowError, IndexError, AttributeError, '
               'ZeroDivisionError):\n'
               '        return 0\n',
      'call': '_case_region_check(factor_rounding_regions(diagonal,domain))',
      'gold_call': '_case_region_check(_oracle_factor_rounding_regions(diagonal,domain))'},
     {'setup': '\n'
               'diagonal=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64).reshape((4,))\n'
               'domain=((9, 8), (1, 1))\n'
               'def _case_raise(fn):\n'
               '    try:\n'
               '        fn()\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2',
      'call': '_case_raise(lambda: factor_rounding_regions(diagonal,domain))',
      'gold_call': '_case_raise(lambda: _oracle_factor_rounding_regions(diagonal,domain))'}]
