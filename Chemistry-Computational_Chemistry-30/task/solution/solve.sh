#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def prepare_inputs(source, kappa=1.0):
    import numpy as np
    try:
        if np.iscomplexobj(source) or np.iscomplexobj(kappa):
            raise ValueError("real inputs required")
        s = np.asarray(source, dtype=float)
        k = float(kappa)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("real numeric inputs required") from exc
    if s.shape != (8,) or not np.all(np.isfinite(s)) or not np.isfinite(k):
        raise ValueError("finite source shape (8,) and finite kappa required")
    if np.any(s[:6] <= 0) or k <= 0 or s[6] < s[7]:
        raise ValueError("invalid lifetimes, rates, thermal energy or gap")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        out = np.array([s[0]*1e-9, s[1], s[2], s[3], s[4]*1e-3,
                        1000.0/s[5], s[6]-s[7], k], dtype=float)
    if not np.all(np.isfinite(out)) or out[5] <= k:
        raise ValueError("finite converted inputs with lambda>kappa required")
    return out

def population_generator(tau_s, i, u, c, z, d):
    import numpy as np
    try:
        raw = [tau_s, i, u, c, z, d]
        if np.iscomplexobj(raw):
            raise ValueError("real rates required")
        v = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("real scalar inputs required") from exc
    if v.shape != (6,) or not np.all(np.isfinite(v)):
        raise ValueError("finite scalar inputs required")
    if np.any(v[:4] <= 0) or np.any(v[4:] < 0):
        raise ValueError("invalid rates")
    tau_s, i, u, c, z, d = v
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        m = np.array([[-1.0/tau_s-i, u, u*z],
                      [i, -u-c, c*z],
                      [0.0, c, -d-(u+c)*z]], dtype=float)
    if not np.all(np.isfinite(m)):
        raise ValueError("matrix overflow")
    return m

def tail_coefficients(tau_s, i, u, c, decay):
    import numpy as np
    try:
        raw = [tau_s, i, u, c, decay]
        if np.iscomplexobj(raw):
            raise ValueError("real inputs required")
        v = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("real scalar inputs required") from exc
    if v.shape != (5,) or not np.all(np.isfinite(v)) or np.any(v <= 0):
        raise ValueError("finite positive inputs required")
    tau_s, i, u, c, decay = v
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        a = 1.0/tau_s+i
        b = u+c
        B = (a-decay)*(b-decay)-i*u
    if not np.all(np.isfinite([a,b,B])) or a <= decay or b <= decay or B <= 0:
        raise ValueError("target must lie below fast-block poles")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        H = b-c*((a-decay)*c+i*u)/B
    if not np.isfinite(H) or H <= 0:
        raise ValueError("finite positive H required")
    return np.array([a, B, H], dtype=float)

def admissible_point(H, decay, kappa, theta, gap, alpha):
    import numpy as np
    try:
        raw = [H, decay, kappa, theta, gap, alpha]
        if np.iscomplexobj(raw):
            raise ValueError("real inputs required")
        v = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("real scalar inputs required") from exc
    if v.shape != (6,) or not np.all(np.isfinite(v)):
        raise ValueError("finite scalars required")
    H, decay, kappa, theta, gap, alpha = v
    if min(H,kappa,theta) <= 0 or decay < kappa or min(gap,alpha) < 0:
        raise ValueError("invalid parameter domain")
    with np.errstate(over="ignore", divide="ignore", under="ignore", invalid="ignore"):
        z = alpha*np.exp(-gap/theta)
        d = decay-H*z
        zmax = (decay-kappa)/H
    if not np.all(np.isfinite([z,d,zmax])):
        raise ValueError("nonfinite family evaluation")
    tol = 64*np.finfo(float).eps*max(1.0,decay,kappa)
    if abs(d-kappa) <= tol:
        d = kappa
    return np.array([z,d,zmax,float(d >= kappa)], dtype=float)

def slow_mode(matrix):
    import numpy as np
    try:
        if np.iscomplexobj(matrix):
            raise ValueError("real generator required")
        m = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("numeric generator required") from exc
    if m.shape != (3,3) or not np.all(np.isfinite(m)):
        raise ValueError("finite (3,3) matrix required")
    off = m.copy()
    np.fill_diagonal(off, 0.0)
    tol = 64*np.finfo(float).eps*max(1.0,float(np.max(np.abs(m))))
    if np.any(off < 0) or np.any(m.sum(axis=0) > tol):
        raise ValueError("matrix is not a loss-only population generator")
    try:
        values, vectors = np.linalg.eig(m)
        j = int(np.argmax(values.real))
        pole = values[j]
        etol = 1e-10*max(1.0,abs(pole))
        others = np.delete(values,j)
        if pole.real >= 0 or abs(pole.imag) > etol or np.any(abs(others-pole) <= etol):
            raise ValueError("stable simple real dominant pole required")
        amplitude = vectors[2,j]*np.linalg.inv(vectors)[j,0]
    except np.linalg.LinAlgError as exc:
        raise ValueError("eigendecomposition requires a nonsingular basis") from exc
    if abs(amplitude.imag) > 1e-10*max(1.0,abs(amplitude.real)):
        raise ValueError("nonreal tail amplitude")
    result = np.array([-pole.real, amplitude.real], dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite spectral result")
    return result

def kinetic_bounds(H, decay, kappa, theta, gap):
    import numpy as np
    try:
        raw = [H,decay,kappa,theta,gap]
        if np.iscomplexobj(raw):
            raise ValueError("real inputs required")
        v = np.asarray(raw,dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("real scalars required") from exc
    if v.shape != (5,) or not np.all(np.isfinite(v)):
        raise ValueError("finite scalars required")
    H,decay,kappa,theta,gap = v
    if min(H,kappa,theta) <= 0 or decay <= kappa or gap < 0:
        raise ValueError("invalid bound domain")
    with np.errstate(over="ignore", divide="ignore", under="ignore", invalid="ignore"):
        zmax = (decay-kappa)/H
        logz = np.log(zmax)
        minimum = max(0.0,float(-theta*logz))
        maximum = np.exp(logz+gap/theta)
    result = np.array([zmax,minimum,maximum],dtype=float)
    if zmax <= 0 or not np.all(np.isfinite(result)):
        raise ValueError("nonfinite or unrepresentable bounds")
    return result

def thermal_identification(theta,z):
    import numpy as np
    try:
        if np.iscomplexobj(theta) or np.iscomplexobj(z):
            raise ValueError("real inputs required")
        t = np.asarray(theta,dtype=float)
        f = np.asarray(z,dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("real arrays required") from exc
    if t.shape != (2,) or f.shape != (2,):
        raise ValueError("two temperatures and factors required")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(f)) or np.any(t<=0) or np.any(f<=0) or t[0]==t[1]:
        raise ValueError("finite positive factors and distinct temperatures required")
    with np.errstate(over="ignore",divide="ignore",under="ignore",invalid="ignore"):
        logs = np.log(f)
        gap = (logs[1]-logs[0])/(1.0/t[0]-1.0/t[1])
        alpha = np.exp(logs[0]+gap/t[0])
    if not np.isfinite(gap) or not np.isfinite(alpha) or gap<0 or alpha<=0:
        raise ValueError("nonnegative finite gap and positive finite alpha required")
    return np.array([gap,alpha],dtype=float)

def solve_triplet_bound(source=None,kappa=1.0):
    import numpy as np
    if source is None:
        source = np.array([6.0,6.0e7,7.5e5,30.0,25.6,120.0,2.953,2.912])
    tau_s,i,u,c,theta,decay,gap,k = prepare_inputs(source,kappa)
    a,B,H = tail_coefficients(tau_s,i,u,c,decay)
    zmax,minimum_gap,maximum_alpha = kinetic_bounds(H,decay,k,theta,gap)
    point = admissible_point(H,decay,k,theta,gap,maximum_alpha)
    if point[3] != 1.0:
        raise ValueError("constructed boundary is not physically admissible")
    matrix = population_generator(tau_s,i,u,c,point[0],point[1])
    rate,amplitude = slow_mode(matrix)
    if abs(rate-decay) > 1e-6*decay or amplitude <= 0:
        raise ValueError("target is not an observable dominant tail")
    return float(maximum_alpha)
SCICODE_GOLD_EOF
