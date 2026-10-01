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

def effective_ionization_parameters(a_n: float, b_n: float, a_p: float, b_p: float) -> np.ndarray:
    for _v in (a_n, b_n, a_p, b_p):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('a_n, b_n, a_p, b_p must be finite and strictly positive')
    return np.array([np.sqrt(a_n*a_p), 0.5*(b_n + b_p)], dtype=float)

import numpy as np

def ionization_dimensionless_group(a_eff: float, b_eff: float, eps_r: float, doping: float) -> float:
    for _v in (a_eff, b_eff, eps_r, doping):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('a_eff, b_eff, eps_r, doping must be finite and strictly positive')
    q = 1.602e-19; eps0 = 8.854e-14
    return float(a_eff*b_eff*eps_r*eps0/(q*doping))

import numpy as np
from scipy.special import exp1
from scipy.optimize import brentq

def avalanche_zeta(phi: float) -> float:
    for _v in (phi,):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('phi must be finite and strictly positive')
    f = lambda z: np.exp(-z)/z - exp1(z) - 1.0/phi
    return float(brentq(f, 1e-10, 400.0, xtol=1e-15, rtol=8.9e-16, maxiter=300))

import numpy as np

def critical_field(b_eff: float, zeta: float) -> float:
    for _v in (b_eff, zeta):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('b_eff, zeta must be finite and strictly positive')
    return float(b_eff/zeta)

import numpy as np
from scipy.special import exp1

def punchthrough_field_ratio(zeta: float) -> float:
    for _v in (zeta,):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('zeta must be finite and strictly positive')
    return float((2.0/3.0)*zeta*np.exp(zeta)*exp1(zeta) - 1.0/3.0)

import numpy as np

def punchthrough_depletion_width(eps_r: float, doping: float, e_crit: float, eta: float) -> float:
    for _v in (eps_r, doping, e_crit):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('eps_r, doping, e_crit must be finite and strictly positive')
    if not np.isfinite(eta) or not (0.0 <= eta < 1.0):
        raise ValueError('eta must be finite and lie in [0, 1)')
    q = 1.602e-19; eps0 = 8.854e-14
    return float((1.0 - eta)*(eps_r*eps0/(q*doping))*e_crit)

import numpy as np

def punchthrough_breakdown_voltage(eps_r: float, doping: float, e_crit: float, eta: float) -> float:
    for _v in (eps_r, doping, e_crit):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('eps_r, doping, e_crit must be finite and strictly positive')
    if not np.isfinite(eta) or not (0.0 <= eta < 1.0):
        raise ValueError('eta must be finite and lie in [0, 1)')
    q = 1.602e-19; eps0 = 8.854e-14
    return float(0.5*(1.0 - eta**2)*(eps_r*eps0/(q*doping))*e_crit**2)

import numpy as np
from scipy.optimize import brentq

def field_ratio_at_rating(eps_r: float, doping: float, e_crit: float, bv_target: float) -> float:
    for _v in (eps_r, doping, e_crit, bv_target):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('eps_r, doping, e_crit, bv_target must be finite and strictly positive')
    f = lambda eta: punchthrough_breakdown_voltage(eps_r, doping, e_crit, eta) - bv_target
    if f(0.0) < 0.0:
        raise ValueError('this doping cannot reach bv_target at any punch-through field ratio')
    return float(brentq(f, 0.0, 1.0 - 1.0e-12, xtol=1.0e-15, rtol=8.9e-16, maxiter=300))

import numpy as np
from scipy.optimize import brentq

def free_electron_density(doping: float, n_c: float, degeneracy: float, ea_shallow: float, ea_deep: float, kt: float) -> float:
    for _v in (doping, n_c, degeneracy, kt):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('doping, n_c, degeneracy, kt must be finite and strictly positive')
    for _v in (ea_shallow, ea_deep):
        if not np.isfinite(_v) or _v < 0.0:
            raise ValueError('activation energies must be finite and non-negative')
    bs = (degeneracy/n_c)*np.exp(ea_shallow/kt)
    bd = (degeneracy/n_c)*np.exp(ea_deep/kt)
    f = lambda n: doping*(0.5/(1.0 + bs*n) + 0.5/(1.0 + bd*n)) - n
    return float(brentq(f, 1.0e5, doping, xtol=1e-8, rtol=8.9e-16, maxiter=300))

import numpy as np

def drift_electron_mobility(doping: float) -> float:
    for _v in (doping,):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('doping must be finite and strictly positive')
    return float(40.0 + 910.0/(1.0 + (doping/2.0e17)**0.61))

import numpy as np
from scipy.special import exp1
from scipy.integrate import quad
from scipy.optimize import brentq

def _ionisation_antiderivative(a: float, b: float, e: float) -> float:
    import numpy as np
    from scipy.special import exp1
    return a*(e*np.exp(-b/e) - b*exp1(b/e))

def two_carrier_critical_field(a_n: float, b_n: float, a_p: float, b_p: float, eps_r: float, doping: float) -> float:
    import numpy as np
    from scipy.integrate import quad
    from scipy.optimize import brentq
    for _v in (a_n, b_n, a_p, b_p, eps_r, doping):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('a_n, b_n, a_p, b_p, eps_r, doping must be finite and strictly positive')
    q = 1.602e-19
    k = eps_r*8.854e-14/(q*doping)

    def _condition(e_cr: float) -> float:
        fn_top = _ionisation_antiderivative(a_n, b_n, e_cr)
        fp_top = _ionisation_antiderivative(a_p, b_p, e_cr)

        def _integrand(e: float) -> float:
            inner = k*((fn_top - _ionisation_antiderivative(a_n, b_n, e)) - (fp_top - _ionisation_antiderivative(a_p, b_p, e)))
            return k*a_n*np.exp(-b_n/e)*np.exp(-inner)

        val = quad(_integrand, 1.0e-6*e_cr, e_cr, limit=400, epsabs=0.0, epsrel=1.0e-13)[0]
        return val - 1.0

    lo, hi = 1.0e4, 1.0e5
    while _condition(hi) < 0.0:
        lo, hi = hi, 2.0*hi
        if hi > 1.0e9:
            raise ValueError('no breakdown field below 1e9 V/cm')
    return float(brentq(_condition, lo, hi, xtol=1.0e-13*hi, rtol=1.0e-15, maxiter=300))

import numpy as np
from scipy.optimize import brentq, minimize_scalar

def minimum_specific_on_resistance(a_n: float, b_n: float, a_p: float, b_p: float, eps_r: float, bv_target: float,
                                           n_c: float, degeneracy: float, ea_shallow: float, ea_deep: float, kt: float) -> float:
    for _v in (a_n, b_n, a_p, b_p, eps_r, bv_target, n_c, degeneracy, kt):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('a_n, b_n, a_p, b_p, eps_r, bv_target, n_c, degeneracy, kt must be finite and strictly positive')
    for _v in (ea_shallow, ea_deep):
        if not np.isfinite(_v) or _v < 0.0:
            raise ValueError('activation energies must be finite and non-negative')
    q = 1.602e-19
    a_eff, b_eff = effective_ionization_parameters(a_n, b_n, a_p, b_p)

    def _design_resistance(nd: float, ec: float) -> float:
        try:
            eta = field_ratio_at_rating(eps_r, nd, ec, bv_target)
        except ValueError:
            return np.inf
        w = punchthrough_depletion_width(eps_r, nd, ec, eta)
        n = free_electron_density(nd, n_c, degeneracy, ea_shallow, ea_deep, kt)
        return 1.0e3*w/(q*drift_electron_mobility(nd)*n)

    def _collapsed_ecr(nd: float) -> float:
        return critical_field(b_eff, avalanche_zeta(
            ionization_dimensionless_group(a_eff, b_eff, eps_r, nd)))

    def _r_collapsed(nd: float) -> float:
        return _design_resistance(nd, _collapsed_ecr(nd))

    def _r_two_carrier(nd: float) -> float:
        return _design_resistance(nd, two_carrier_critical_field(a_n, b_n, a_p, b_p, eps_r, nd))

    def _bv_pub(nd: float) -> float:
        z = avalanche_zeta(ionization_dimensionless_group(a_eff, b_eff, eps_r, nd))
        eta = punchthrough_field_ratio(z)
        if not (0.0 <= eta < 1.0):
            return np.inf
        return punchthrough_breakdown_voltage(eps_r, nd, critical_field(b_eff, z), eta) - bv_target

    def _scan_minimum(fun, centre: float):
        grid = centre*np.logspace(np.log10(0.5), np.log10(1.5), 61)
        vals = np.array([fun(x) for x in grid])
        if not np.any(np.isfinite(vals)):
            raise ValueError('no punch-through design reaches bv_target near the starting doping')
        i = int(np.argmin(np.where(np.isfinite(vals), vals, np.inf)))
        a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        res = minimize_scalar(fun, bounds=(a, b), method='bounded', options=dict(xatol=1.0e-9*grid[i]))
        return (res.x, fun(res.x)) if fun(res.x) <= vals[i] else (grid[i], vals[i])

    lo, hi = 1.0e12, 1.0e19
    while hi > lo and not np.isfinite(_bv_pub(hi)):
        hi *= 0.5
    if hi <= lo or _bv_pub(lo)*_bv_pub(hi) > 0.0:
        raise ValueError('no punch-through design reaches bv_target in the physical doping range')
    nd_pub = brentq(_bv_pub, lo, hi, xtol=1.0e2, rtol=8.9e-16, maxiter=300)
    nd_collapsed, _ = _scan_minimum(_r_collapsed, nd_pub)
    _, r_min = _scan_minimum(_r_two_carrier, nd_collapsed)
    return float(r_min)
SCICODE_GOLD_EOF
