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

def _c_assemble_qqbar_sector_jets__sector_components(sector, points):
    import numpy as np
    names = ('I1', 'I2', 'I3', 'I4', 'I5', 'II1', 'II2')
    if not isinstance(sector, str) or sector not in names:
        raise ValueError('one of the seven named sectors')
    x = np.asarray(points)
    if x.dtype.kind not in 'iuf':
        raise ValueError('real numeric points required')
    x = x.astype(np.result_type(x.dtype, float))
    if x.ndim != 2 or x.shape[1] != 5 or (not np.all(np.isfinite(x))) or np.any(x[:, :4] < 0) or np.any(x[:, :4] >= 1) or np.any(x[:, 4] <= 0) or np.any(x[:, 4] >= 1):
        raise ValueError('finite unit-cube points with permitted zero faces')
    t, a, b, r, chi = x.T
    result = np.zeros((len(x), 5), dtype=x.dtype)
    if sector in names[:5]:
        if sector == 'I1':
            u, v, y, sign = (a / 2, b / 2, r, 1)
            delta, sv, active = (1 - b / 2, np.sqrt(b) / np.sqrt(2.0), b > 0)
        elif sector == 'I2':
            u, v, y, sign = (b / 2, a / 2, a * r, -1)
            delta, sv, active = (1 - a / 2, np.sqrt(a) / np.sqrt(2.0), a > 0)
        elif sector == 'I3':
            u, v, y, sign = (b / 2, a * r / 2, r, -1)
            delta, sv, active = (1 - a * r / 2, np.sqrt(a) * np.sqrt(r) / np.sqrt(2.0), (a > 0) & (r > 0))
        elif sector == 'I4':
            u, v, y, sign = (a / 2, 1 - b / 2, r, 1)
            delta, sv, active = (b / 2, np.sqrt(v), np.ones(len(x), dtype=bool))
        else:
            u, v, y, sign = (b / 2, 1 - a / 2, r, -1)
            delta, sv, active = (a / 2, np.sqrt(v), np.ones(len(x), dtype=bool))
        A, B = (np.ones(len(x)), v) if sign == 1 else (v, np.ones(len(x)))
        angular = 1 + v - 2 * u * v
        remainder = (1 - u) * (1 - u * v)
        root = sv * np.sqrt(remainder)
        sg = np.hypot(delta / np.sqrt(angular + 2 * root), 2 * np.sqrt(chi) * np.sqrt(root))
        c = (delta / sg) ** 2
        d1 = 1 - t
        dy = 1 - a + a * (1 - r) if sector == 'I2' else 1 - r
        d2 = d1 + t * dy
        D = d1 * d2
        L = d1 + t * d2 + t * t * y * u * c
        z = D / L
        if sector in names[:3]:
            ell = -2 * np.log(z) + 2 * np.log(L) + 4 * np.log(sg) - np.log(remainder) - np.log(chi) - np.log1p(-chi) - np.log(2.0) - 2 * np.log(delta)
            multiplier = v / delta
        else:
            ell = -2 * np.log(z) + 2 * np.log(L) + 4 * np.log(sg) - np.log(v * remainder) - np.log(chi) - np.log1p(-chi)
            multiplier = np.ones(len(x))
        E = A * d2 + y * B * d1 + t * y * c
        F = (1 - u * A) * d2 + y * (1 - u * B) * d1 + t * y * u * c
        if sector == 'I2':
            yE = r / (d2 / 2 + r * d1 + t * r * c)
        elif sector == 'I3':
            yE = 1 / (a * d2 / 2 + d1 + t * c)
        else:
            yE = y / E
        base = 1 - a if sector in ('I1', 'I4') else 1 - b
        J = base * dy + 2 * u * delta * (d2 if sign == -1 else -y * d1)
        radial = d1 / L * (d2 / L) / np.pi
        scale = multiplier * radial
        spin = scale * (yE / F * (sg * D - t * sign * (delta / sg) * J / 2)) ** 2
        qq = scale * yE / F * (-D + y * t * t * u * c / 2) + spin
    else:
        cap = np.full(len(x), 0.5) if sector == 'II1' else np.minimum(r, 0.5)
        p1, p2, q1, q2 = (a * cap, 1 - b / 2, 1 - a * cap, b / 2)
        y = a * r / 2 if sector == 'II1' else r
        active = (a > 0) & (b > 0)
        if sector == 'II2':
            active &= r > 0
        delta = (1 - b) / 2 + (0.5 - cap) + cap * (1 - a)
        angular = p1 * q2 + p2 * q1
        small_root = np.sqrt(a) * np.sqrt(cap) * np.sqrt(b) / np.sqrt(2.0)
        root = small_root * np.sqrt(q1 * p2)
        sg = np.hypot(delta / np.sqrt(angular + 2 * root), 2 * np.sqrt(chi) * np.sqrt(root))
        h = (delta / sg) ** 2
        d1 = 1 - t
        d2 = 1 - t * y if sector == 'II1' else d1 + t * (1 - r)
        D = d1 * d2
        L = d1 + t * d2 + t * t * y * h
        z = D / L
        multiplier = small_root / delta * small_root
        ell = -2 * np.log(z) + 2 * np.log(L) + 4 * np.log(sg) - np.log(q1 * p2) - 2 * np.log(delta) - np.log(chi) - np.log1p(-chi)
        if sector == 'II1':
            yE = r / (d2 + r * p2 * d1 + t * r * h)
        else:
            ratio = np.divide(0.5, r, out=np.ones(len(x), dtype=x.dtype), where=r > 0.5)
            yE = 1 / (a * ratio * d2 + p2 * d1 + t * h)
            ell += -3 * np.log(2.0) - np.log(ratio)
        F = q1 * d2 + y * q2 * d1 + t * y * h
        J = d2 * (1 - a + a * (1 - 2 * cap)) + y * d1 * (1 - b)
        radial = d1 / L * (d2 / L) / np.pi
        scale = multiplier * radial
        spin = scale * (yE / F * (sg * D + t * (delta / sg) * J / 2)) ** 2
        qq = scale * yE / F * (-D + y * t * t * h / 2) + spin
    result[:, 4] = ell
    result[active, :2] = np.column_stack([qq, spin])[active]
    return result

def _c_assemble_qqbar_sector_jets__gamma_jet(ell):
    import numpy as np
    from scipy.special import zeta
    z2, z3, z4 = (np.pi ** 2 / 6, float(zeta(3.0, 1.0)), np.pi ** 4 / 90)
    return np.column_stack([np.ones(len(ell)), ell, ell ** 2 / 2 - 2 * z2, ell ** 3 / 6 - 2 * z2 * ell - 8 * z3 / 3, ell ** 4 / 24 - z2 * ell ** 2 - 8 * z3 * ell / 3 + z4])

def assemble_qqbar_sector_jets(sector: str, points: np.ndarray) -> np.ndarray:
    components = _c_assemble_qqbar_sector_jets__sector_components(sector, points)
    gamma = _c_assemble_qqbar_sector_jets__gamma_jet(components[:, 4])
    return components[:, :2, None] * gamma[:, None, :]

import numpy as np

def _c_assemble_nonabelian_sector_jets__sector_components(sector, points):
    import numpy as np
    names = ('I1', 'I2', 'I3', 'I4', 'I5', 'II1', 'II2')
    if not isinstance(sector, str) or sector not in names:
        raise ValueError('one of the seven named sectors')
    x = np.asarray(points)
    if x.dtype.kind not in 'iuf':
        raise ValueError('real numeric points required')
    x = x.astype(np.result_type(x.dtype, float))
    if x.ndim != 2 or x.shape[1] != 5 or (not np.all(np.isfinite(x))) or np.any(x[:, :4] < 0) or np.any(x[:, :4] >= 1) or np.any(x[:, 4] <= 0) or np.any(x[:, 4] >= 1):
        raise ValueError('finite unit-cube points with permitted zero faces')
    t, a, b, r, chi = x.T
    result = np.zeros((len(x), 5), dtype=x.dtype)
    if sector in names[:5]:
        if sector == 'I1':
            u, v, y, sign = (a / 2, b / 2, r, 1)
            delta, sv, active = (1 - b / 2, np.sqrt(b) / np.sqrt(2.0), b > 0)
        elif sector == 'I2':
            u, v, y, sign = (b / 2, a / 2, a * r, -1)
            delta, sv, active = (1 - a / 2, np.sqrt(a) / np.sqrt(2.0), a > 0)
        elif sector == 'I3':
            u, v, y, sign = (b / 2, a * r / 2, r, -1)
            delta, sv, active = (1 - a * r / 2, np.sqrt(a) * np.sqrt(r) / np.sqrt(2.0), (a > 0) & (r > 0))
        elif sector == 'I4':
            u, v, y, sign = (a / 2, 1 - b / 2, r, 1)
            delta, sv, active = (b / 2, np.sqrt(v), np.ones(len(x), dtype=bool))
        else:
            u, v, y, sign = (b / 2, 1 - a / 2, r, -1)
            delta, sv, active = (a / 2, np.sqrt(v), np.ones(len(x), dtype=bool))
        A, B = (np.ones(len(x)), v) if sign == 1 else (v, np.ones(len(x)))
        angular = 1 + v - 2 * u * v
        remainder = (1 - u) * (1 - u * v)
        root = sv * np.sqrt(remainder)
        sg = np.hypot(delta / np.sqrt(angular + 2 * root), 2 * np.sqrt(chi) * np.sqrt(root))
        c = (delta / sg) ** 2
        d1 = 1 - t
        dy = 1 - a + a * (1 - r) if sector == 'I2' else 1 - r
        d2 = d1 + t * dy
        D = d1 * d2
        L = d1 + t * d2 + t * t * y * u * c
        z = D / L
        if sector in names[:3]:
            ell = -2 * np.log(z) + 2 * np.log(L) + 4 * np.log(sg) - np.log(remainder) - np.log(chi) - np.log1p(-chi) - np.log(2.0) - 2 * np.log(delta)
        else:
            ell = -2 * np.log(z) + 2 * np.log(L) + 4 * np.log(sg) - np.log(v * remainder) - np.log(chi) - np.log1p(-chi)
        E = A * d2 + y * B * d1 + t * y * c
        F = (1 - u * A) * d2 + y * (1 - u * B) * d1 + t * y * u * c
        if sector == 'I2':
            yE = r / (d2 / 2 + r * d1 + t * r * c)
        elif sector == 'I3':
            yE = 1 / (a * d2 / 2 + d1 + t * c)
        else:
            yE = y / E
        base = 1 - a if sector in ('I1', 'I4') else 1 - b
        radial = d1 / L * (d2 / L) / np.pi
        if sector in names[:3]:
            cosine = 2 * angular * (np.sqrt(chi) / sg) ** 2 - c / (angular + 2 * root)
            ordered = 2 * radial / delta * sv / np.sqrt(remainder) * cosine
        else:
            ordered = radial * (angular - c) / (v * remainder)
        b1 = D - t * (y * d1 * u * B + d2 * (1 - u * A)) - t * t * y * u * c / 2
        b2 = D - t * (y * d1 * (1 - u * B) + d2 * u * A) - t * t * y * u * c / 2
        correction = yE / F * (A * (1 - u * B) * b1 + B * (1 - u * A) * b2)
        correction += t * c * yE / 2 + t * y * u * c / (2 * F) - t * t * y * u * c * c * yE / (2 * F)
    else:
        cap = np.full(len(x), 0.5) if sector == 'II1' else np.minimum(r, 0.5)
        p1, p2, q1, q2 = (a * cap, 1 - b / 2, 1 - a * cap, b / 2)
        y = a * r / 2 if sector == 'II1' else r
        active = (a > 0) & (b > 0)
        if sector == 'II2':
            active &= r > 0
        delta = (1 - b) / 2 + (0.5 - cap) + cap * (1 - a)
        angular = p1 * q2 + p2 * q1
        small_root = np.sqrt(a) * np.sqrt(cap) * np.sqrt(b) / np.sqrt(2.0)
        root = small_root * np.sqrt(q1 * p2)
        sg = np.hypot(delta / np.sqrt(angular + 2 * root), 2 * np.sqrt(chi) * np.sqrt(root))
        h = (delta / sg) ** 2
        d1 = 1 - t
        d2 = 1 - t * y if sector == 'II1' else d1 + t * (1 - r)
        D = d1 * d2
        L = d1 + t * d2 + t * t * y * h
        z = D / L
        ell = -2 * np.log(z) + 2 * np.log(L) + 4 * np.log(sg) - np.log(q1 * p2) - 2 * np.log(delta) - np.log(chi) - np.log1p(-chi)
        if sector == 'II1':
            yE = r / (d2 + r * p2 * d1 + t * r * h)
        else:
            ratio = np.divide(0.5, r, out=np.ones(len(x), dtype=x.dtype), where=r > 0.5)
            yE = 1 / (a * ratio * d2 + p2 * d1 + t * h)
            ell += -3 * np.log(2.0) - np.log(ratio)
        F = q1 * d2 + y * q2 * d1 + t * y * h
        radial = d1 / L * (d2 / L) / np.pi
        cosine = 2 * angular * (np.sqrt(chi) / sg) ** 2 - h / (angular + 2 * root)
        ordered = 2 * radial * (small_root / delta) / np.sqrt(q1 * p2) * cosine
        b1 = D - t * (y * d1 * p2 + d2 * q1) - t * t * y * h / 2
        b2 = D - t * (y * d1 * q2 + d2 * p1) - t * t * y * h / 2
        correction = yE / F * (p1 * q2 * b1 + p2 * q1 * b2)
        correction += t * h * yE / 2 + t * y * h / (2 * F) - t * t * y * h * h * yE / (2 * F)
    result[:, 4] = ell
    result[active, 2:4] = np.column_stack([ordered, correction])[active]
    return result

def _c_assemble_nonabelian_sector_jets__gamma_jet(ell):
    import numpy as np
    from scipy.special import zeta
    z2, z3, z4 = (np.pi ** 2 / 6, float(zeta(3.0, 1.0)), np.pi ** 4 / 90)
    return np.column_stack([np.ones(len(ell)), ell, ell ** 2 / 2 - 2 * z2, ell ** 3 / 6 - 2 * z2 * ell - 8 * z3 / 3, ell ** 4 / 24 - z2 * ell ** 2 - 8 * z3 * ell / 3 + z4])

def assemble_nonabelian_sector_jets(sector: str, points: np.ndarray, qqbar_jets: np.ndarray, moments: tuple = (0, 0, 0, 0, 0)) -> np.ndarray:
    import numpy as np
    components = _c_assemble_nonabelian_sector_jets__sector_components(sector, points)
    q = np.asarray(qqbar_jets)
    if q.dtype.kind not in 'iuf':
        raise ValueError('real numeric jets required')
    dtype = np.result_type(components.dtype, q.dtype, float)
    x, q, powers = (np.asarray(points, dtype=dtype), q.astype(dtype), np.asarray(moments))
    if q.shape != (len(x), 2, 5) or not np.all(np.isfinite(q)):
        raise ValueError('finite aligned quark-pair and polarization jets')
    if powers.shape != (5,) or powers.dtype.kind not in 'iu' or any((isinstance(v, (bool, np.bool_)) for v in moments)) or np.any(powers < 0) or np.any(powers > 2):
        raise ValueError('five integer moment powers from zero through two')
    gamma = _c_assemble_nonabelian_sector_jets__gamma_jet(components[:, 4])
    jets = (components[:, 2] * (1 - components[:, 3] / 2))[:, None] * gamma
    jets += 2 * q[:, 0, :] - q[:, 1, :]
    jets[:, 1:] -= q[:, 1, :-1]
    return jets * np.prod(x ** powers, axis=1)[:, None]

import numpy as np

def assemble_laurent_sector_rule(sector: str, order: int, start: int = 0, stop: int | None = None) -> np.ndarray:
    import math
    import numpy as np
    rates = {'I1': (4,2,1,2), 'I2': (4,3,2,2), 'I3': (4,1,2,3), 'I4': (4,2,2,2), 'I5': (4,2,2,2), 'II1': (4,3,1,2), 'II2': (4,1,1,3)}
    _integer = lambda x: isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))
    if not isinstance(sector, str) or sector not in rates or not _integer(order) or not 1 <= order <= 16:
        raise ValueError('named sector and integer quadrature order from one through sixteen')
    shape = (order+1, order+1, order+1, 2*order+1 if sector == 'II2' else order+1, order)
    count = math.prod(shape)
    stop = count if stop is None else stop
    if not _integer(start) or not _integer(stop) or not 0 <= start <= stop <= count:
        raise ValueError('integer bounds 0 <= start <= stop <= row count')
    gl, wg = np.polynomial.legendre.leggauss(order)
    s, w = (gl.astype(np.float64)+1)/2, wg.astype(np.float64)/2
    denom = s**3+(1-s)**3
    x, v = s**3/denom, w*3*s**2*(1-s)**2/denom**2
    nodes, axes = [], []
    for axis, rate in enumerate(rates[sector]):
        xx, vv = (np.r_[x/2, (1+x)/2], np.r_[v/2, v/2]) if sector == 'II2' and axis == 3 else (x, v)
        nodes.append(np.r_[np.float64(0), xx])
        coeff = np.zeros((len(xx)+1, 6), dtype=np.float64)
        coeff[0,0] = -np.float64(1)/rate
        for k in range(5):
            coeff[1:,k+1] = vv/xx*(-rate*np.log(xx))**k/math.factorial(k)
            coeff[0,k+1] = -np.sum(coeff[1:,k+1], dtype=np.float64)
        axes.append(coeff)
    u = s**3*(10-15*s+6*s**2)
    nodes.append(np.sin(np.pi*u/2)**2)
    angular = np.pi*w*30*s**2*(1-s)**2
    indices = np.unravel_index(np.arange(start, stop), shape)
    result = np.empty((stop-start, 30), dtype=np.float64)
    for j in range(5):
        result[:,j] = nodes[j][indices[j]]
    for j in range(4):
        result[:,5+6*j:11+6*j] = axes[j][indices[j]]
    result[:,29] = angular[indices[4]]
    return result

import numpy as np

def combine_laurent_sector_coefficients(rule: np.ndarray, jets: np.ndarray) -> np.ndarray:
    import numpy as np
    rule, jets = np.asarray(rule), np.asarray(jets)
    if rule.dtype.kind not in 'iuf' or jets.dtype.kind not in 'iuf':
        raise ValueError('real numeric rule and jets required')
    dtype = np.result_type(rule.dtype, jets.dtype, float)
    rule, jets = rule.astype(dtype), jets.astype(dtype)
    if rule.ndim != 2 or rule.shape[1] != 30 or jets.shape != (len(rule),5) or not np.all(np.isfinite(rule)) or not np.all(np.isfinite(jets)):
        raise ValueError('aligned finite rule and epsilon jets')
    product = np.zeros((len(rule),5), dtype=dtype)
    product[:,0] = 1
    for start in (5,11,17,23):
        updated = np.zeros_like(product)
        for degree in range(5):
            for j in range(degree+1):
                updated[:,degree] += product[:,j]*rule[:,start+degree-j]
        product = updated
    terms = np.zeros_like(product)
    for degree in range(5):
        for j in range(degree+1):
            terms[:,degree] += product[:,j]*jets[:,degree-j]
    return np.sum(terms*rule[:,29,None], axis=0, dtype=dtype)

import numpy as np

def assemble_opposite_hemisphere_coefficients(order: int = 16, moments: tuple = (0, 0, 0, 0, 0)) -> np.ndarray:
    import numpy as np
    coefficients = []
    for sector in ('II1','II2'):
        assemble_laurent_sector_rule(sector, order, 0, 0)
        count = (order+1)**3*(2*order+1 if sector == 'II2' else order+1)*order
        parts = []
        for start in range(0,count,16384):
            rule = assemble_laurent_sector_rule(sector, order, start, min(start+16384,count))
            qqbar = assemble_qqbar_sector_jets(sector, rule[:,:5])
            jets = assemble_nonabelian_sector_jets(sector, rule[:,:5], qqbar, moments)
            parts.append(combine_laurent_sector_coefficients(rule,jets))
        coefficients.append(np.sum(parts,axis=0,dtype=np.longdouble))
    return np.asarray(coefficients, dtype=np.float64)

import numpy as np

def combine_sector_finite_parts(same_hemisphere: np.ndarray, opposite: np.ndarray) -> float:
    import numpy as np
    same = np.asarray(same_hemisphere)
    opp = np.asarray(opposite)
    if np.iscomplexobj(same) or np.iscomplexobj(opp):
        raise ValueError('complex dtype is not accepted')
    if same.dtype == bool or opp.dtype == bool:
        raise ValueError('bool entries are not accepted')
    same = same.astype(np.longdouble)
    opp = opp.astype(np.longdouble)
    if same.ndim != 1 or same.size != 5:
        raise ValueError('same_hemisphere must hold the five I-sector finite coefficients')
    if opp.ndim != 2 or opp.shape != (2,5):
        raise ValueError('opposite must be the two II rows of five coefficients')
    if not (np.all(np.isfinite(same)) and np.all(np.isfinite(opp))):
        raise ValueError('inputs must be finite')
    return float(4*np.sum(np.concatenate([same, opp[:,4]]),dtype=np.longdouble))

import numpy as np

def compute_nonabelian_integrated_finite_part(order: int = 16, moments: tuple = (0, 0, 0, 0, 0)) -> float:
    import numpy as np
    assemble_laurent_sector_rule('I1',order,0,0)
    count = (order+1)**4*order
    same = []
    for sector in ('I1','I2','I3','I4','I5'):
        parts = []
        for start in range(0,count,16384):
            rule = assemble_laurent_sector_rule(sector, order, start, min(start+16384,count))
            qqbar = assemble_qqbar_sector_jets(sector, rule[:,:5])
            jets = assemble_nonabelian_sector_jets(sector, rule[:,:5], qqbar, moments)
            parts.append(combine_laurent_sector_coefficients(rule,jets)[4])
        same.append(np.sum(parts,dtype=np.longdouble))
    opposite = assemble_opposite_hemisphere_coefficients(order,moments)
    return combine_sector_finite_parts(same, opposite)
SCICODE_GOLD_EOF
