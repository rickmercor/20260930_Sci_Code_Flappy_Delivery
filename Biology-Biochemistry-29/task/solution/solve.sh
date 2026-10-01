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

def _checked_array(value, shape, name, positive=False):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must have shape {shape} and be finite")
    if positive and np.any(a <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return a

def compute_equilibrium_coefficients(rate_constants):
    import numpy as np

    k = _checked_array(rate_constants, (14,), "rate_constants", True)

    with np.errstate(all="ignore"):
        D = (
            k[8]*k[10]*(k[12]+k[13])
            + k[0]*k[2]*k[11]*k[13]*(k[9]+k[10])
            / (k[1]*(k[3]+k[4]))
        )

        c = np.array([
            k[7]*(k[3]+k[4])/(k[2]*k[4]),
            k[1]*k[7]*(k[3]+k[4])/(k[0]*k[2]*k[4]),
            k[7]*k[8]*(k[12]+k[13])/D,
            k[7]/k[4],
            k[0]*k[2]*k[7]*k[11]*(k[9]+k[10])
            / (k[1]*(k[3]+k[4])*D),
            (k[6]+k[7])/k[5],
            k[0]*k[2]*k[4]*(k[9]+k[10])*(k[12]+k[13])
            / (k[1]*(k[3]+k[4])*D)
        ])

    return _checked_array(c, (7,), "coefficients", True)

import numpy as np

def _checked_array(value, shape, name, positive=False):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must have shape {shape} and be finite")
    if positive and np.any(a <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return a

def solve_positive_equilibrium(coeffs, T1, T2):
    import numpy as np

    c = _checked_array(coeffs, (7,), "coeffs", True)
    t1, t2 = float(T1), float(T2)

    if not np.all(np.isfinite([t1, t2])) or t1 <= 0 or t2 <= 0:
        raise ValueError("totals must be finite and positive")

    S = 1 + np.sum(c[:5])
    C = 1 + c[2] + c[4]
    beta, yp = c[5:]
    y0 = t1 + t2 - yp

    if y0 <= 0:
        raise ValueError("no positive equilibrium")

    h = S*y0 + beta + t1*C
    disc = h*h - 4*S*C*t1*y0

    if disc < 0:
        raise ValueError("no real equilibrium")

    q = 2*t1*y0/(h + np.sqrt(disc))
    y = y0 - C*q
    xp = beta*q/y

    state = np.array([q, y, *(c[:5]*q), xp, yp])
    return _checked_array(state, (9,), "state", True)

import numpy as np

def _checked_array(value, shape, name, positive=False):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must have shape {shape} and be finite")
    if positive and np.any(a <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return a

def compute_fraction_sensitivity(coeffs, T1, T2):
    import numpy as np

    # Ensure earlier oracle functions can resolve shared names
    # when Studio uses separate module namespaces.
    _context = {
        "np": np,
        "_checked_array": _checked_array,
        "solve_positive_equilibrium":
            solve_positive_equilibrium
    }
    solve_positive_equilibrium.__globals__.update(_context)

    c = _checked_array(coeffs, (7,), "coeffs", True)
    t1 = float(T1)
    state = solve_positive_equilibrium(c, t1, T2)

    q, y = state[:2]
    xp = state[7]
    S = 1 + np.sum(c[:5])
    B = np.sum(c[[0, 1, 2, 3]])
    beta, yp = c[5:]
    y0 = t1 + float(T2) - yp

    F = (xp + q + state[6])/t1
    G = -B*xp/(t1*(S + beta*y0/y**2))

    return _checked_array(np.array([F, G]), (2,), "observables")

import numpy as np

def _checked_array(value, shape, name, positive=False):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must have shape {shape} and be finite")
    if positive and np.any(a <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return a

def construct_inverse_constraints(observations):
    import numpy as np

    o = _checked_array(observations, (4,), "observations")
    yp, f1, f2, g = o

    if (
        yp <= 0 or yp >= 1.8
        or not (0 < f1 < 1 and 0 < f2 < 1)
        or g >= 0
    ):
        raise ValueError("invalid calibration observations")

    x1, x2 = 1-f1, 1-f2
    y1, y2 = 1.8-yp, 2.1-yp

    return _checked_array(
        np.array([
            -(y1/x1-y2/x2),
            -0.8,
            -(y1-y2),
            x1-1.8*x2
        ]),
        (4,),
        "inverse relation"
    )

import numpy as np

def _checked_array(value, shape, name, positive=False):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must have shape {shape} and be finite")
    if positive and np.any(a <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return a

def reconstruct_kinetic_parameters(normalized, Yp, fixed_rates):
    import numpy as np

    s, v, r = _checked_array(normalized, (3,), "normalized")
    k = _checked_array(fixed_rates, (14,), "fixed_rates", True)
    yp = float(Yp)

    if (
        not np.isfinite(yp) or yp <= 0
        or s <= 1 or v <= 0 or r <= 0
    ):
        raise ValueError("invalid inverse parameters")

    d0 = k[8]*k[10]*(k[12]+k[13])
    d1 = (
        k[0]*k[13]*(k[9]+k[10])
        / (k[1]*(k[3]+k[4]))
    )
    H = (
        k[0]*k[4]*(k[9]+k[10])*(k[12]+k[13])
        / (k[1]*(k[3]+k[4]))
    )

    a = k[8]*(k[12]+k[13])*yp/H
    a0 = (k[3]+k[4])/k[4]*(1+k[1]/k[0])
    e = yp/(k[4]*(k[12]+k[13]))

    delta = 1-s+v
    if delta <= 0 or s-v <= 0:
        raise ValueError("invalid inverse invariants")

    J = k[4]*(1-(1+a0/a)*delta)
    if J <= 0:
        raise ValueError("invalid inverse invariants")

    u = a*J/delta
    w = (H/yp-d0/u)/d1

    den = s-1-e*J*w
    if den <= 0:
        raise ValueError("invalid inverse invariants")

    B = 1/den
    q = J*B
    k6 = (k[6]+q)/(r*B)

    return _checked_array(
        np.array([u, k6, q, w]),
        (4,),
        "reconstructed rates",
        True
    )

import numpy as np
from numpy.polynomial import Polynomial

def _checked_array(value, shape, name, positive=False):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must have shape {shape} and be finite")
    if positive and np.any(a <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return a

def _assemble_rates(base, unknown):
    k = np.asarray(base, dtype=float).copy()
    k[[2, 5, 7, 11]] = unknown
    return k

def _predict_calibration(k):
    c = compute_equilibrium_coefficients(k)
    return np.array([
        c[6],
        compute_fraction_sensitivity(c, 1.0, 0.8)[0],
        compute_fraction_sensitivity(c, 1.8, 0.3)[0],
        compute_fraction_sensitivity(c, 1.4, 1.1)[1]
    ])

def _inverse_polynomial(observations, relation):
    yp, f1, f2, g_signed = observations
    p0, p1, q0, q1 = relation

    v = Polynomial([0.0, 1.0])
    den = Polynomial([q0, q1])
    sn = Polynomial([p0, p1])

    x1 = 1.0 - f1
    y1 = 1.8 - yp
    rn = (den / x1 - sn) * (y1 - v * x1)

    tc, yc = 1.4, 2.5 - yp
    g = -g_signed

    a = tc * v * sn
    b = -(yc * sn + tc * v * den + rn)

    poly = (
        yc * (rn + 2*g*a)**2 * den
        - g*b*b*(rn + g*a)
    )

    if poly.degree() > 12 or not np.any(poly.coef):
        raise ValueError("invalid inverse polynomial")

    return poly

def calibrate_kinetic_parameters(
    fixed_rates, observations, bounds
):
    import numpy as np
    from numpy.polynomial import Polynomial

    _context = {
        "np": np,
        "Polynomial": Polynomial,
        "_checked_array": _checked_array,
        "_assemble_rates": _assemble_rates,
        "_predict_calibration": _predict_calibration,
        "_inverse_polynomial": _inverse_polynomial,
        "compute_equilibrium_coefficients":
            compute_equilibrium_coefficients,
        "solve_positive_equilibrium":
            solve_positive_equilibrium,
        "compute_fraction_sensitivity":
            compute_fraction_sensitivity,
        "construct_inverse_constraints":
            construct_inverse_constraints,
        "reconstruct_kinetic_parameters":
            reconstruct_kinetic_parameters
    }

    for fn in (
        compute_equilibrium_coefficients,
        solve_positive_equilibrium,
        compute_fraction_sensitivity,
        construct_inverse_constraints,
        reconstruct_kinetic_parameters,
        _assemble_rates,
        _predict_calibration,
        _inverse_polynomial
    ):
        fn.__globals__.update(_context)

    k = _checked_array(fixed_rates, (14,), "fixed_rates", True)
    o = _checked_array(observations, (4,), "observations")
    b = _checked_array(bounds, (4, 2), "bounds", True)

    if np.any(b[:, 0] >= b[:, 1]):
        raise ValueError("invalid bounds")

    relation = construct_inverse_constraints(o)
    polynomial = _inverse_polynomial(o, relation)
    candidates = []

    for root in polynomial.roots():
        if abs(root.imag) > 1e-7:
            continue

        v = float(root.real)
        if v <= 0:
            continue

        p0, p1, q0, q1 = relation
        den = q0 + q1*v

        if abs(den) < 1e-12:
            continue

        s = (p0 + p1*v)/den
        x1 = 1.0 - o[1]
        y1 = 1.8 - o[0]
        r = (1.0/x1 - s)*(y1 - v*x1)

        if (
            not np.all(np.isfinite([s, v, r]))
            or s <= 1
            or r <= 0
            or not 0 < s-v < 1
        ):
            continue

        try:
            unknown = reconstruct_kinetic_parameters(
                [s, v, r], o[0], k
            )

            if (
                np.any(unknown < b[:, 0] - 1e-8)
                or np.any(unknown > b[:, 1] + 1e-8)
            ):
                continue

            unknown = np.clip(unknown, b[:, 0], b[:, 1])

            residual = (
                _predict_calibration(_assemble_rates(k, unknown))
                - o
            )

            if np.max(np.abs(residual)) > 1e-8:
                continue

            if not any(
                np.allclose(
                    unknown, old, rtol=1e-5, atol=1e-8
                )
                for old in candidates
            ):
                candidates.append(unknown)

        except (
            ValueError,
            OverflowError,
            ZeroDivisionError,
            FloatingPointError
        ):
            continue

    if len(candidates) != 1:
        raise ValueError("expected exactly one admissible inverse solution")

    return candidates[0]

import numpy as np

def _checked_array(value, shape, name, positive=False):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must have shape {shape} and be finite")
    if positive and np.any(a <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return a

def optimize_sensitivity(
    coeffs, T1, T2_lower, T2_upper
):
    import numpy as np

    _context = {
        "np": np,
        "_checked_array": _checked_array,
        "solve_positive_equilibrium":
            solve_positive_equilibrium,
        "compute_fraction_sensitivity":
            compute_fraction_sensitivity
    }
    for _fn in (
        solve_positive_equilibrium,
        compute_fraction_sensitivity
    ):
        _fn.__globals__.update(_context)

    c = _checked_array(coeffs, (7,), "coeffs", True)
    t1,lo,hi = map(float,(T1,T2_lower,T2_upper))

    if (
        not np.all(np.isfinite([t1,lo,hi]))
        or t1 <= 0 or lo <= 0 or hi <= lo
    ):
        raise ValueError("invalid optimization interval")

    S = 1+np.sum(c[:5])
    B = np.sum(c[[0,1,2,3]])
    C = 1+c[2]+c[4]
    beta,yp = c[5:]

    s,v,r = S/B,C/B,beta/B
    a = v*t1/r
    z = 1/(1+np.sqrt(1+a))
    x = (1-z)/s

    q = t1*x/B
    y = beta*q/(t1*z)
    t2star = y+C*q+yp-t1

    candidates = [lo,hi]
    if lo < t2star < hi:
        candidates.append(t2star)

    results = []
    for t2 in candidates:
        try:
            F,G = compute_fraction_sensitivity(c,t1,t2)
            results.append((abs(G),t2,F))
        except ValueError:
            continue

    if not results:
        raise ValueError("no admissible equilibrium in interval")

    value,t2,F = max(results,key=lambda item:(item[0],-item[1]))
    return np.array([t2,F,value])

import numpy as np
from numpy.polynomial import Polynomial

def _assemble_rates(base, unknown):
    k = np.asarray(base, dtype=float).copy()
    k[[2, 5, 7, 11]] = unknown
    return k

def compute_optimized_fraction(
    fixed_rates, observations, bounds,
    T1=1.35, T2_lower=0.2, T2_upper=2.0
):
    import numpy as np
    from numpy.polynomial import Polynomial

    _context = {
        "np": np,
        "Polynomial": Polynomial,
        "_assemble_rates": _assemble_rates,
        "compute_equilibrium_coefficients":
            compute_equilibrium_coefficients,
        "solve_positive_equilibrium":
            solve_positive_equilibrium,
        "compute_fraction_sensitivity":
            compute_fraction_sensitivity,
        "construct_inverse_constraints":
            construct_inverse_constraints,
        "reconstruct_kinetic_parameters":
            reconstruct_kinetic_parameters,
        "calibrate_kinetic_parameters":
            calibrate_kinetic_parameters,
        "optimize_sensitivity":
            optimize_sensitivity
    }

    for fn in (
        compute_equilibrium_coefficients,
        solve_positive_equilibrium,
        compute_fraction_sensitivity,
        construct_inverse_constraints,
        reconstruct_kinetic_parameters,
        calibrate_kinetic_parameters,
        optimize_sensitivity
    ):
        fn.__globals__.update(_context)

    unknown = calibrate_kinetic_parameters(
        fixed_rates, observations, bounds
    )

    k = _assemble_rates(fixed_rates, unknown)
    c = compute_equilibrium_coefficients(k)

    return float(
        optimize_sensitivity(
            c, T1, T2_lower, T2_upper
        )[1]
    )
SCICODE_GOLD_EOF
