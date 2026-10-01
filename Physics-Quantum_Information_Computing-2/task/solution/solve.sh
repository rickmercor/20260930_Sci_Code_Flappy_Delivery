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
from scipy.integrate import quad
from scipy.special import betaln
from scipy.optimize import least_squares, linprog

def damped_resource(a: float, b: float) -> np.ndarray:
    if not np.isfinite([a, b]).all() or not (0 <= a <= 1 and 0 <= b <= 1):
        raise ValueError('Damping parameters must lie in [0,1].')
    c = np.sqrt((1 - a) * (1 - b))
    return np.array([[1 + a * b, 0, 0, c], [0, a * (1 - b), 0, 0], [0, 0, (1 - a) * b, 0], [c, 0, 0, (1 - a) * (1 - b)]]) / 2

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln
from scipy.optimize import least_squares, linprog

def conditional_fidelity_law(rho: np.ndarray, mode: int=0) -> np.ndarray:
    if mode not in (0, 1):
        raise ValueError('Mode must be 0 (recorded) or 1 (erased).')
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (4, 4) or not np.isfinite(rho).all():
        raise ValueError('Expected a finite 4 by 4 density matrix.')
    if not np.allclose(rho, rho.conj().T, atol=1e-11) or abs(np.trace(rho) - 1) > 1e-10 or np.linalg.eigvalsh(rho)[0] < -1e-10:
        raise ValueError('Expected a physical density matrix.')
    a = float((rho[0, 0] + rho[1, 1] - rho[2, 2] - rho[3, 3]).real)
    b = float((rho[0, 0] - rho[1, 1] + rho[2, 2] - rho[3, 3]).real)
    c = float(2 * rho[0, 3].real)
    w = float((rho[0, 0] - rho[1, 1] - rho[2, 2] + rho[3, 3]).real)
    if not np.allclose(rho, damped_resource(a, b), atol=1e-10, rtol=0):
        raise ValueError('The state must be in the declared amplitude-damped Bell family.')
    if mode == 1:
        return np.tile([(1 + c) / 8, 0.0, (w - c) / 8, 0.25, 0.0], (4, 1))
    return np.array([[(1 + c) / 8, s * (a + b) / 8, (w - c) / 8, 0.25, s * a / 4] for s in (1, 1, -1, -1)])

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln
from scipy.optimize import least_squares, linprog

def fidelity_preimages(law: np.ndarray, threshold: float) -> np.ndarray:
    law = np.asarray(law, dtype=float)
    if law.shape != (4, 5) or not np.isfinite(law).all() or (not np.isfinite(threshold)):
        raise ValueError('Expected a finite law of shape (4,5) and finite threshold.')

    def _validate_physical_law(law):
        """Validate the declared R/E family with a coefficient tolerance of 1e-10."""
        if law.shape != (4, 5) or not np.isfinite(law).all():
            raise ValueError('Expected a finite physical law of shape (4,5).')
        tol = 1e-10
        c = 8 * law[0, 0] - 1
        w = 8 * law[0, 2] + c
        a = 4 * law[0, 4]
        b = 8 * law[0, 1] - a
        if c >= -tol and -tol <= a <= 1 + tol and (-tol <= b <= 1 + tol):
            expected = np.array([[(1 + c) / 8, s * (a + b) / 8, (w - c) / 8, 0.25, s * a / 4] for s in (1, 1, -1, -1)])
            if abs(c * c - (1 - a) * (1 - b)) <= tol and abs(w - (1 - a - b + 2 * a * b)) <= tol and np.allclose(law, expected, rtol=0, atol=tol):
                return
        product = w - c * c
        total = 1 + product - c * c
        discriminant = total * total - 4 * product
        if c >= -tol and discriminant >= -tol:
            root = np.sqrt(max(0.0, discriminant))
            (a, b) = ((total + root) / 2, (total - root) / 2)
            expected = np.tile([(1 + c) / 8, 0.0, (w - c) / 8, 0.25, 0.0], (4, 1))
            if -tol <= a <= 1 + tol and -tol <= b <= 1 + tol and np.allclose(law, expected, rtol=0, atol=tol):
                return
        raise ValueError('Law is outside the declared recorded/erased physical family.')
    _validate_physical_law(law)
    out = np.zeros((4, 2, 2))
    for (j, (q0, q1, q2, p0, p1)) in enumerate(law):
        coeff = np.array([q2, q1 - threshold * p1, q0 - threshold * p0])
        scale = max(abs(coeff))
        if scale == 0:
            out[j, 0] = [-1, 1]
            continue
        coeff = coeff / scale
        if abs(coeff[0]) < 2e-14:
            roots = [] if abs(coeff[1]) < 2e-14 else [-coeff[2] / coeff[1]]
        else:
            roots = np.roots(coeff)
            roots = [r.real for r in roots if abs(r.imag) < 1e-10]
        bounds = sorted(set([-1.0, 1.0] + [float(r) for r in roots if -1 < r < 1]))
        parts = []
        for (lo, hi) in zip(bounds[:-1], bounds[1:]):
            if np.polyval(coeff, (lo + hi) / 2) <= 0:
                if parts and abs(parts[-1][1] - lo) < 1e-12:
                    parts[-1][1] = hi
                else:
                    parts.append([lo, hi])
        if parts:
            out[j, :len(parts)] = parts
    return out

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln
from scipy.optimize import least_squares, linprog

def fidelity_cdf(law: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    law = np.asarray(law, dtype=float)
    if law.shape != (4, 5) or not np.isfinite(law).all():
        raise ValueError('Expected a finite law of shape (4,5).')

    def _validate_physical_law(law):
        """Validate the declared R/E family with a coefficient tolerance of 1e-10."""
        if law.shape != (4, 5) or not np.isfinite(law).all():
            raise ValueError('Expected a finite physical law of shape (4,5).')
        tol = 1e-10
        c = 8 * law[0, 0] - 1
        w = 8 * law[0, 2] + c
        a = 4 * law[0, 4]
        b = 8 * law[0, 1] - a
        if c >= -tol and -tol <= a <= 1 + tol and (-tol <= b <= 1 + tol):
            expected = np.array([[(1 + c) / 8, s * (a + b) / 8, (w - c) / 8, 0.25, s * a / 4] for s in (1, 1, -1, -1)])
            if abs(c * c - (1 - a) * (1 - b)) <= tol and abs(w - (1 - a - b + 2 * a * b)) <= tol and np.allclose(law, expected, rtol=0, atol=tol):
                return
        product = w - c * c
        total = 1 + product - c * c
        discriminant = total * total - 4 * product
        if c >= -tol and discriminant >= -tol:
            root = np.sqrt(max(0.0, discriminant))
            (a, b) = ((total + root) / 2, (total - root) / 2)
            expected = np.tile([(1 + c) / 8, 0.0, (w - c) / 8, 0.25, 0.0], (4, 1))
            if -tol <= a <= 1 + tol and -tol <= b <= 1 + tol and np.allclose(law, expected, rtol=0, atol=tol):
                return
        raise ValueError('Law is outside the declared recorded/erased physical family.')
    _validate_physical_law(law)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or not np.isfinite(thresholds).all():
        raise ValueError('Thresholds must be a finite one-dimensional array.')
    out = []
    for t in thresholds:
        spans = fidelity_preimages(law, t)
        mass = 0.0
        for (j, edges) in enumerate(spans):
            (p0, p1) = law[j, 3:]
            for (lo, hi) in edges:
                mass += p0 * (hi - lo) / 2 + p1 * (hi * hi - lo * lo) / 4
        out.append(mass)
    return np.array(out)

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln
from scipy.optimize import least_squares, linprog

def fidelity_moments(law: np.ndarray, orders: np.ndarray) -> np.ndarray:
    law = np.asarray(law, dtype=float)
    orders = np.asarray(orders)
    if law.shape != (4, 5) or not np.isfinite(law).all() or orders.ndim != 1 or (not np.isfinite(orders).all()) or np.any(orders < 0) or np.any(orders != np.floor(orders)):
        raise ValueError('Expected a finite law and nonnegative integer orders.')

    def _validate_physical_law(law):
        """Validate the declared R/E family with a coefficient tolerance of 1e-10."""
        if law.shape != (4, 5) or not np.isfinite(law).all():
            raise ValueError('Expected a finite physical law of shape (4,5).')
        tol = 1e-10
        c = 8 * law[0, 0] - 1
        w = 8 * law[0, 2] + c
        a = 4 * law[0, 4]
        b = 8 * law[0, 1] - a
        if c >= -tol and -tol <= a <= 1 + tol and (-tol <= b <= 1 + tol):
            expected = np.array([[(1 + c) / 8, s * (a + b) / 8, (w - c) / 8, 0.25, s * a / 4] for s in (1, 1, -1, -1)])
            if abs(c * c - (1 - a) * (1 - b)) <= tol and abs(w - (1 - a - b + 2 * a * b)) <= tol and np.allclose(law, expected, rtol=0, atol=tol):
                return
        product = w - c * c
        total = 1 + product - c * c
        discriminant = total * total - 4 * product
        if c >= -tol and discriminant >= -tol:
            root = np.sqrt(max(0.0, discriminant))
            (a, b) = ((total + root) / 2, (total - root) / 2)
            expected = np.tile([(1 + c) / 8, 0.0, (w - c) / 8, 0.25, 0.0], (4, 1))
            if -tol <= a <= 1 + tol and -tol <= b <= 1 + tol and np.allclose(law, expected, rtol=0, atol=tol):
                return
        raise ValueError('Law is outside the declared recorded/erased physical family.')
    _validate_physical_law(law)
    result = []
    for order in orders:
        total = 0.0
        for (q0, q1, q2, p0, p1) in law:

            def _fun(z):
                p = p0 + p1 * z
                f = np.clip((q0 + q1 * z + q2 * z * z) / p, 0, 1)
                return 0.5 * p * f ** order
            total += quad(_fun, -1, 1, epsabs=2e-12, epsrel=2e-12)[0]
        result.append(total)
    return np.array(result)

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln
from scipy.optimize import least_squares, linprog

def importance_expectations(law: np.ndarray, priors: np.ndarray) -> np.ndarray:
    law = np.asarray(law, dtype=float)
    priors = np.asarray(priors, dtype=float)
    if law.shape != (4, 5) or not np.isfinite(law).all() or priors.ndim != 2 or (priors.shape[1] != 2) or (not np.isfinite(priors).all()) or np.any(priors < 1):
        raise ValueError('Expected a finite law and beta shapes alpha,beta >= 1.')

    def _validate_physical_law(law):
        """Validate the declared R/E family with a coefficient tolerance of 1e-10."""
        if law.shape != (4, 5) or not np.isfinite(law).all():
            raise ValueError('Expected a finite physical law of shape (4,5).')
        tol = 1e-10
        c = 8 * law[0, 0] - 1
        w = 8 * law[0, 2] + c
        a = 4 * law[0, 4]
        b = 8 * law[0, 1] - a
        if c >= -tol and -tol <= a <= 1 + tol and (-tol <= b <= 1 + tol):
            expected = np.array([[(1 + c) / 8, s * (a + b) / 8, (w - c) / 8, 0.25, s * a / 4] for s in (1, 1, -1, -1)])
            if abs(c * c - (1 - a) * (1 - b)) <= tol and abs(w - (1 - a - b + 2 * a * b)) <= tol and np.allclose(law, expected, rtol=0, atol=tol):
                return
        product = w - c * c
        total = 1 + product - c * c
        discriminant = total * total - 4 * product
        if c >= -tol and discriminant >= -tol:
            root = np.sqrt(max(0.0, discriminant))
            (a, b) = ((total + root) / 2, (total - root) / 2)
            expected = np.tile([(1 + c) / 8, 0.0, (w - c) / 8, 0.25, 0.0], (4, 1))
            if -tol <= a <= 1 + tol and -tol <= b <= 1 + tol and np.allclose(law, expected, rtol=0, atol=tol):
                return
        raise ValueError('Law is outside the declared recorded/erased physical family.')
    _validate_physical_law(law)
    result = []
    for (alpha, beta) in priors:
        total = 0.0
        norm = np.exp(-betaln(alpha, beta))
        for (q0, q1, q2, p0, p1) in law:

            def _fun(z):
                p = p0 + p1 * z
                f = np.clip((q0 + q1 * z + q2 * z * z) / p, 0, 1)
                return 0.5 * p * norm * f ** (alpha - 1) * (1 - f) ** (beta - 1)
            total += quad(_fun, -1, 1, epsabs=2e-11, epsrel=2e-11)[0]
        result.append(total)
    return np.array(result)

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln
from scipy.optimize import least_squares, linprog

def classical_reference(priors: np.ndarray, thresholds: np.ndarray, mode: int=0) -> np.ndarray:
    if mode not in (0, 1):
        raise ValueError('Mode must be 0 (recorded) or 1 (erased).')
    priors = np.asarray(priors, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if priors.ndim != 2 or priors.shape[1] != 2 or (not np.isfinite(priors).all()) or np.any(priors < 1) or (thresholds.ndim != 1) or (not np.isfinite(thresholds).all()):
        raise ValueError('Expected beta shapes >= 1 and finite thresholds.')
    if mode == 0:
        return np.r_[2 * priors[:, 0] / priors.sum(axis=1), np.clip(thresholds, 0, 1) ** 2]
    expectations = []
    for (a, b) in priors:

        def _integrand(z):
            f = (1 + z * z) / 2
            return np.exp(-betaln(a, b)) * f ** (a - 1) * (1 - f) ** (b - 1)
        expectations.append(quad(_integrand, 0, 1, epsabs=2e-11, epsrel=2e-11)[0])
    return np.r_[expectations, np.sqrt(np.clip(2 * thresholds - 1, 0, 1))]

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln
from scipy.optimize import least_squares, linprog

def infer_damping(observations: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or observations.shape != (2 + len(thresholds),) or (not np.isfinite(observations).all()) or (not np.isfinite(thresholds).all()):
        raise ValueError('Expected [E F,E F^2,CDF values] and their thresholds.')

    def _residual(x):
        L = conditional_fidelity_law(damped_resource(*x))
        return np.r_[fidelity_moments(L, [1, 2]), fidelity_cdf(L, thresholds)] - observations
    solutions = []
    for a in [0.1, 0.35, 0.65, 0.88]:
        for b in [0.1, 0.35, 0.65, 0.88]:
            fit = least_squares(_residual, [a, b], bounds=([0.01, 0.01], [0.96, 0.96]), xtol=2e-12, ftol=2e-12, gtol=2e-12, max_nfev=250)
            solutions.append((float(np.linalg.norm(fit.fun)), fit.x))
    (error, answer) = min(solutions, key=lambda item: item[0])
    if error > 1e-08:
        raise ValueError('Calibration is inconsistent with the physical model.')
    return answer.copy()

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln
from scipy.optimize import least_squares, linprog

def optimize_policy(quality: np.ndarray, tails: np.ndarray, costs: np.ndarray, budget: float, tail_min: float) -> np.ndarray:
    quality = np.asarray(quality, dtype=float)
    tails = np.asarray(tails, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if quality.ndim != 2 or tails.ndim != 2 or costs.ndim != 1 or (quality.shape[1] != len(costs)) or (tails.shape[1] != len(costs)) or (not len(costs)) or (not len(quality)) or (not len(tails)):
        raise ValueError('Incompatible nonempty quality, tail, and cost arrays.')
    if not all((np.isfinite(v).all() for v in [quality, tails, costs, np.array([budget, tail_min])])) or np.any((tails < 0) | (tails > 1)) or np.any(costs < 0) or (budget < 0) or (not 0 <= tail_min <= 1):
        raise ValueError('Invalid finite budget or probability data.')
    n = len(costs)
    objective = np.r_[np.zeros(n), -1.0]
    A = np.vstack([np.c_[-quality, np.ones(len(quality))], np.c_[-tails, np.zeros(len(tails))], np.r_[costs, 0.0]])
    b = np.r_[np.zeros(len(quality)), -np.full(len(tails), tail_min), budget]
    equality = np.array([np.r_[np.ones(n), 0.0]])
    eq_values = np.array([1.0])
    bounds = [(0, 1)] * n + [(None, None)]
    opts = {'dual_feasibility_tolerance': 1e-09, 'primal_feasibility_tolerance': 1e-09}
    sol = linprog(objective, A_ub=A, b_ub=b, A_eq=equality, b_eq=eq_values, bounds=bounds, method='highs', options=opts)
    if not sol.success:
        raise ValueError('No feasible policy.')
    equality = np.vstack([equality, np.r_[np.zeros(n), 1.0]])
    eq_values = np.r_[eq_values, sol.x[-1]]
    for k in range(n - 1):
        obj = np.zeros(n + 1)
        obj[k] = 1
        trial = linprog(obj, A_ub=A, b_ub=b, A_eq=equality, b_eq=eq_values, bounds=bounds, method='highs', options=opts)
        if not trial.success:
            raise ValueError('Optimal-face refinement failed.')
        sol = trial
        equality = np.vstack([equality, obj])
        eq_values = np.r_[eq_values, sol.x[k]]
    x = np.maximum(sol.x[:n], 0)
    return np.r_[np.min(quality @ x), x, np.min(tails @ x), costs @ x]

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln
from scipy.optimize import least_squares, linprog

def teleportation_benchmark(observations: np.ndarray, thresholds: np.ndarray, profiles: np.ndarray, scenarios: np.ndarray, priors: np.ndarray, budget: float, target: float, tail_min: float) -> float:
    profiles = np.asarray(profiles, dtype=float)
    scenarios = np.asarray(scenarios, dtype=float)
    priors = np.asarray(priors, dtype=float)
    if profiles.ndim != 2 or profiles.shape[1] != 4 or scenarios.ndim != 2 or (scenarios.shape[1] != 2) or (not len(profiles)) or (not len(scenarios)):
        raise ValueError('Expected nonempty profile (C,4) and scenario (S,2) arrays.')
    if not np.isfinite(profiles).all() or not np.isfinite(scenarios).all() or np.any(profiles[:, 1:3] <= 0) or np.any(scenarios <= 0) or np.any(~np.isin(profiles[:, 0], [0, 1])) or (not np.isfinite(target)):
        raise ValueError('Invalid orientations, exposure multipliers, or target.')
    pair = infer_damping(observations, thresholds)
    scores = []
    for mode in (0, 1):
        base = classical_reference(priors, np.array([], dtype=float), mode)
        quality = []
        tails = []
        for exposure in scenarios:
            q = []
            t = []
            for (orientation, ea, eb, _) in profiles:
                damp = 1 - (1 - pair) ** (np.array([ea, eb]) * exposure)
                if orientation:
                    damp = damp[::-1]
                rho = damped_resource(*damp)
                L = conditional_fidelity_law(rho, mode)
                q.append(importance_expectations(L, priors) - base)
                t.append(1 - fidelity_cdf(L, np.array([target]))[0])
            quality.extend(np.array(q).T)
            tails.append(t)
        result = optimize_policy(np.array(quality), np.array(tails), profiles[:, 3], budget, tail_min)
        scores.append(result[0])
    return float(scores[0] - scores[1])
SCICODE_GOLD_EOF
