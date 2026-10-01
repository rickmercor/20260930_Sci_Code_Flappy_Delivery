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


def curvature_spline_values(x_1: float, x_end: float, theta, x):
    """Reference implementation."""
    theta = np.asarray(theta, dtype=float)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if theta.ndim != 1 or theta.size < 3:
        raise ValueError("theta must be one dimensional with at least three entries")
    if x.ndim != 1:
        raise ValueError("x must be one dimensional")
    if not float(x_end) > float(x_1):
        raise ValueError("x_end must be strictly greater than x_1")

    d = float(np.logaddexp(0.0, theta[0]))
    c = np.logaddexp(0.0, theta[1:])
    n_c = c.size
    h = (float(x_end) - float(x_1)) / (n_c - 1)

    # Slope and value accumulated at the interpolation sites. The quadrature is
    # exact because f'' is piecewise linear and f' piecewise quadratic.
    slope_at = np.zeros(n_c)
    value_at = np.zeros(n_c)
    for j in range(1, n_c):
        slope_at[j] = slope_at[j - 1] + 0.5 * h * (c[j - 1] + c[j])
        value_at[j] = (value_at[j - 1] + h * (d + slope_at[j - 1])
                       + 0.5 * c[j - 1] * h * h + (c[j] - c[j - 1]) * h * h / 6.0)

    f = np.empty_like(x)
    f_prime = np.empty_like(x)
    f_second = np.empty_like(x)
    for k, xk in enumerate(x):
        if xk < x_1:
            u = xk - x_1
            f_second[k] = c[0]
            f_prime[k] = d + c[0] * u
            f[k] = d * u + 0.5 * c[0] * u * u
        elif xk > x_end:
            u = xk - x_end
            slope_end = d + slope_at[-1]
            f_second[k] = c[-1]
            f_prime[k] = slope_end + c[-1] * u
            f[k] = value_at[-1] + slope_end * u + 0.5 * c[-1] * u * u
        else:
            j = min(int((xk - x_1) / h), n_c - 2)
            u = xk - (x_1 + j * h)
            ramp = (c[j + 1] - c[j]) / h
            f_second[k] = c[j] + ramp * u
            f_prime[k] = d + slope_at[j] + c[j] * u + 0.5 * ramp * u * u
            f[k] = (value_at[j] + (d + slope_at[j]) * u + 0.5 * c[j] * u * u
                    + ramp * u ** 3 / 6.0)
    return f, f_prime, f_second

import numpy as np


def uniaxial_kinematics(stretch: float, cv_inverse):
    """Reference implementation."""
    internal = np.asarray(cv_inverse, dtype=float)
    if internal.shape != (3,):
        raise ValueError("cv_inverse must hold exactly three entries")
    if not np.all(internal > 0.0):
        raise ValueError("cv_inverse entries must be strictly positive")
    if not float(stretch) > 0.0:
        raise ValueError("stretch must be strictly positive")

    lam = float(stretch)
    stretch_squared = np.array([lam ** 2, 1.0 / lam, 1.0 / lam])
    beta_trial = stretch_squared * internal
    return beta_trial, stretch_squared

import numpy as np

_SQRT3 = np.sqrt(3.0)


def isochoric_invariants(principal_values):
    """Reference implementation."""
    b = np.asarray(principal_values, dtype=float)
    if b.shape != (3,):
        raise ValueError("principal_values must hold exactly three entries")
    if not np.all(b > 0.0):
        raise ValueError("principal values must be strictly positive")

    determinant = float(b[0] * b[1] * b[2])
    unimodular = b * determinant ** (-1.0 / 3.0)
    I1 = float(unimodular.sum())
    w = float(unimodular[0] * unimodular[1] + unimodular[0] * unimodular[2]
              + unimodular[1] * unimodular[2])
    I2 = float(w ** 1.5 - 3.0 * _SQRT3)
    return I1, I2

import numpy as np

_SQRT3 = np.sqrt(3.0)


def _spline_slope(spline, argument):
    """First derivative of a curvature-based spline at one argument."""
    x_1, x_end, theta = spline
    theta = np.asarray(theta, dtype=float)
    d = float(np.logaddexp(0.0, theta[0]))
    c = np.logaddexp(0.0, theta[1:])
    n_c = c.size
    h = (float(x_end) - float(x_1)) / (n_c - 1)
    slope_at = np.zeros(n_c)
    for j in range(1, n_c):
        slope_at[j] = slope_at[j - 1] + 0.5 * h * (c[j - 1] + c[j])
    if argument < x_1:
        return d + c[0] * (argument - x_1)
    if argument > x_end:
        return d + slope_at[-1] + c[-1] * (argument - x_end)
    j = min(int((argument - x_1) / h), n_c - 2)
    u = argument - (x_1 + j * h)
    return d + slope_at[j] + c[j] * u + 0.5 * (c[j + 1] - c[j]) / h * u * u


def branch_kirchhoff_stress(beta_e, psi1_spline, psi2_spline):
    """Reference implementation."""
    beta = np.asarray(beta_e, dtype=float)
    if beta.shape != (3,):
        raise ValueError("beta_e must hold exactly three entries")
    if not np.all(beta > 0.0):
        raise ValueError("beta_e entries must be strictly positive")
    for spline in (psi1_spline, psi2_spline):
        if len(spline) != 3:
            raise ValueError("a spline descriptor must be (x_1, x_end, theta)")

    determinant = float(beta[0] * beta[1] * beta[2])
    unimodular = beta * determinant ** (-1.0 / 3.0)
    I1 = float(unimodular.sum())
    w = float(unimodular[0] * unimodular[1] + unimodular[0] * unimodular[2]
              + unimodular[1] * unimodular[2])
    I2 = float(w ** 1.5 - 3.0 * _SQRT3)

    slope_1 = _spline_slope(psi1_spline, I1)
    slope_2 = _spline_slope(psi2_spline, I2)
    tau_full = 2.0 * beta * (slope_1 + slope_2 * 1.5 * np.sqrt(w) * (I1 - beta))
    tau_neq = tau_full - tau_full.mean()
    J_tau = 1.5 * float(np.dot(tau_neq, tau_neq))
    return tau_neq, J_tau

import numpy as np


def _spline_slope(spline, argument):
    """First derivative of a curvature-based spline at one argument."""
    x_1, x_end, theta = spline
    theta = np.asarray(theta, dtype=float)
    d = float(np.logaddexp(0.0, theta[0]))
    c = np.logaddexp(0.0, theta[1:])
    n_c = c.size
    h = (float(x_end) - float(x_1)) / (n_c - 1)
    slope_at = np.zeros(n_c)
    for j in range(1, n_c):
        slope_at[j] = slope_at[j - 1] + 0.5 * h * (c[j - 1] + c[j])
    if argument < x_1:
        return d + c[0] * (argument - x_1)
    if argument > x_end:
        return d + slope_at[-1] + c[-1] * (argument - x_end)
    j = min(int((argument - x_1) / h), n_c - 2)
    u = argument - (x_1 + j * h)
    return d + slope_at[j] + c[j] * u + 0.5 * (c[j + 1] - c[j]) / h * u * u


def viscous_flow_rate(tau_neq, J_tau: float, phi_spline):
    """Reference implementation."""
    tau = np.asarray(tau_neq, dtype=float)
    if tau.shape != (3,):
        raise ValueError("tau_neq must hold exactly three entries")
    if not float(J_tau) >= 0.0:
        raise ValueError("J_tau must be non-negative")
    if len(phi_spline) != 3:
        raise ValueError("phi_spline must be (x_1, x_end, theta)")

    return 3.0 * _spline_slope(phi_spline, float(J_tau)) * tau

import numpy as np

_SQRT3 = np.sqrt(3.0)
_FD_STEP = 1e-7


def _spline_slope(spline, argument):
    """First derivative of a curvature-based spline at one argument."""
    x_1, x_end, theta = spline
    theta = np.asarray(theta, dtype=float)
    d = float(np.logaddexp(0.0, theta[0]))
    c = np.logaddexp(0.0, theta[1:])
    n_c = c.size
    h = (float(x_end) - float(x_1)) / (n_c - 1)
    slope_at = np.zeros(n_c)
    for j in range(1, n_c):
        slope_at[j] = slope_at[j - 1] + 0.5 * h * (c[j - 1] + c[j])
    if argument < x_1:
        return d + c[0] * (argument - x_1)
    if argument > x_end:
        return d + slope_at[-1] + c[-1] * (argument - x_end)
    j = min(int((argument - x_1) / h), n_c - 2)
    u = argument - (x_1 + j * h)
    return d + slope_at[j] + c[j] * u + 0.5 * (c[j + 1] - c[j]) / h * u * u


def _viscous_rate(beta, psi1_spline, psi2_spline, phi_spline):
    """Viscous deformation-type rate at an elastic state."""
    determinant = float(beta[0] * beta[1] * beta[2])
    unimodular = beta * determinant ** (-1.0 / 3.0)
    I1 = float(unimodular.sum())
    w = float(unimodular[0] * unimodular[1] + unimodular[0] * unimodular[2]
              + unimodular[1] * unimodular[2])
    I2 = float(w ** 1.5 - 3.0 * _SQRT3)
    slope_1 = _spline_slope(psi1_spline, I1)
    slope_2 = _spline_slope(psi2_spline, I2)
    tau_full = 2.0 * beta * (slope_1 + slope_2 * 1.5 * np.sqrt(w) * (I1 - beta))
    tau_neq = tau_full - tau_full.mean()
    J_tau = 1.5 * float(np.dot(tau_neq, tau_neq))
    return 3.0 * _spline_slope(phi_spline, J_tau) * tau_neq


def local_branch_update(beta_trial, dt: float, psi1_spline, psi2_spline,
                                phi_spline, tol: float = 1e-12, max_iter: int = 50):
    """Reference implementation."""
    beta_trial = np.asarray(beta_trial, dtype=float)
    if beta_trial.shape != (3,):
        raise ValueError("beta_trial must hold exactly three entries")
    if not np.all(beta_trial > 0.0):
        raise ValueError("beta_trial entries must be strictly positive")
    if not float(dt) > 0.0:
        raise ValueError("dt must be strictly positive")
    if not (isinstance(max_iter, int) and max_iter > 0):
        raise ValueError("max_iter must be a positive integer")

    eps_trial = 0.5 * np.log(beta_trial)

    def residual(eps):
        rate = _viscous_rate(np.exp(2.0 * eps), psi1_spline, psi2_spline,
                             phi_spline)
        return eps - eps_trial + float(dt) * rate

    eps = eps_trial.copy()
    for _ in range(max_iter):
        r = residual(eps)
        if np.max(np.abs(r)) < float(tol):
            break
        jacobian = np.empty((3, 3))
        for j in range(3):
            forward = eps.copy()
            backward = eps.copy()
            forward[j] += _FD_STEP
            backward[j] -= _FD_STEP
            jacobian[:, j] = (residual(forward) - residual(backward)) / (2.0 * _FD_STEP)
        eps = eps - np.linalg.solve(jacobian, r)

    return np.exp(2.0 * eps)

import numpy as np


def adapt_domain_endpoint(x_end: float, samples, alpha_smooth: float,
                                  eta: float) -> float:
    """Reference implementation."""
    values = np.atleast_1d(np.asarray(samples, dtype=float))
    if values.ndim != 1 or values.size == 0:
        raise ValueError("samples must be one dimensional and non-empty")
    if not float(alpha_smooth) > 0.0:
        raise ValueError("alpha_smooth must be strictly positive")
    if not 0.0 < float(eta) <= 1.0:
        raise ValueError("eta must lie in (0, 1]")

    shift = float(values.max())
    x_act = shift + float(np.log(np.mean(np.exp(float(alpha_smooth)
                                                * (values - shift))))) / float(alpha_smooth)
    return float((1.0 - float(eta)) * float(x_end) + float(eta) * x_act)

import numpy as np
from scipy.optimize import nnls

_COEFFICIENT_FLOOR = 1e-12


def _integrated_basis(x_1, x_end, n_c, x):
    """Integrals of the degree-one basis functions from x_1 up to each sample."""
    h = (float(x_end) - float(x_1)) / (n_c - 1)
    columns = np.empty((x.size, n_c))
    for i in range(n_c):
        c = np.zeros(n_c)
        c[i] = 1.0
        accumulated = np.zeros(n_c)
        for j in range(1, n_c):
            accumulated[j] = accumulated[j - 1] + 0.5 * h * (c[j - 1] + c[j])
        for k, xk in enumerate(x):
            if xk < x_1:
                columns[k, i] = c[0] * (xk - x_1)
            elif xk > x_end:
                columns[k, i] = accumulated[-1] + c[-1] * (xk - x_end)
            else:
                j = min(int((xk - x_1) / h), n_c - 2)
                u = xk - (x_1 + j * h)
                columns[k, i] = (accumulated[j] + c[j] * u
                                 + 0.5 * (c[j + 1] - c[j]) / h * u * u)
    return columns


def _spline_slopes(x_1, x_end, theta, x):
    """First derivative of a curvature-based spline at every sample."""
    theta = np.asarray(theta, dtype=float)
    d = float(np.logaddexp(0.0, theta[0]))
    c = np.logaddexp(0.0, theta[1:])
    n_c = c.size
    h = (float(x_end) - float(x_1)) / (n_c - 1)
    accumulated = np.zeros(n_c)
    for j in range(1, n_c):
        accumulated[j] = accumulated[j - 1] + 0.5 * h * (c[j - 1] + c[j])
    out = np.empty_like(x)
    for k, xk in enumerate(x):
        if xk < x_1:
            out[k] = d + c[0] * (xk - x_1)
        elif xk > x_end:
            out[k] = d + accumulated[-1] + c[-1] * (xk - x_end)
        else:
            j = min(int((xk - x_1) / h), n_c - 2)
            u = xk - (x_1 + j * h)
            out[k] = d + accumulated[j] + c[j] * u + 0.5 * (c[j + 1] - c[j]) / h * u * u
    return out


def project_spline_parameters(x_1: float, x_end_old: float, theta_old,
                                      x_end_new: float, samples):
    """Reference implementation."""
    theta_old = np.asarray(theta_old, dtype=float)
    values = np.atleast_1d(np.asarray(samples, dtype=float))
    if theta_old.ndim != 1 or theta_old.size < 3:
        raise ValueError("theta_old must be one dimensional with at least three entries")
    if values.size == 0:
        raise ValueError("samples must be non-empty")
    if not (float(x_end_old) > float(x_1) and float(x_end_new) > float(x_1)):
        raise ValueError("both upper endpoints must be strictly greater than x_1")

    n_c = theta_old.size - 1
    target = _spline_slopes(float(x_1), float(x_end_old), theta_old, values)
    design = np.empty((values.size, n_c + 1))
    design[:, 0] = 1.0
    design[:, 1:] = _integrated_basis(float(x_1), float(x_end_new), n_c, values)
    physical, _ = nnls(design, target)
    return np.log(np.expm1(np.maximum(physical, _COEFFICIENT_FLOOR)))

import numpy as np

_PSI1_INITIAL = (3.0, 3.02, (-1.0, -2.0, -2.5, -3.0))
_PSI2_INITIAL = (0.0, 0.05, (-2.0, -3.0, -3.5, -4.0))
_PHI_INITIAL = (0.0, 0.10, (-4.0, -5.0, -5.5, -6.0))
_ADMISSIBILITY_TOLERANCE = 1e-12


def run_full_pipeline(n_outer: int = 3, alpha_smooth: float = 5.0,
                              eta: float = 0.5, dt: float = 1.0,
                              stretch_rate: float = 0.05,
                              stretch_max: float = 2.0) -> float:
    """Reference implementation chaining every earlier step."""
    if not (isinstance(n_outer, int) and not isinstance(n_outer, bool)
            and n_outer >= 0):
        raise ValueError("n_outer must be a non-negative integer")
    if not float(stretch_max) > 1.0:
        raise ValueError("stretch_max must be greater than one")

    n_half = int(round((float(stretch_max) - 1.0) / (float(stretch_rate) * float(dt))))
    ramp = np.arange(1, n_half + 1)
    stretch_history = np.concatenate([1.0 + float(stretch_rate) * float(dt) * ramp,
                                      float(stretch_max)
                                      - float(stretch_rate) * float(dt) * ramp])

    functions = [list(_PSI1_INITIAL), list(_PSI2_INITIAL), list(_PHI_INITIAL)]
    for _ in range(n_outer):
        psi1 = tuple(functions[0])
        psi2 = tuple(functions[1])
        phi = tuple(functions[2])
        cv_inverse = np.ones(3)
        samples = [np.empty(stretch_history.size) for _ in range(3)]
        dissipation = 0.0
        for index, stretch in enumerate(stretch_history):
            beta_trial, stretch_squared = uniaxial_kinematics(
                float(stretch), cv_inverse)
            beta_e = local_branch_update(beta_trial, float(dt), psi1, psi2,
                                                 phi)
            cv_inverse = beta_e / stretch_squared
            first, second = isochoric_invariants(beta_e)
            tau_neq, J_tau = branch_kirchhoff_stress(beta_e, psi1, psi2)
            d_e = viscous_flow_rate(tau_neq, J_tau, phi)
            dissipation += float(dt) * float(np.dot(tau_neq, d_e))
            stored = float(
                curvature_spline_values(psi1[0], psi1[1], psi1[2],
                                                [first])[0][0]
                + curvature_spline_values(psi2[0], psi2[1], psi2[2],
                                                  [second])[0][0])
            if stored < -_ADMISSIBILITY_TOLERANCE:
                raise ValueError("the branch stores a negative free energy")
            samples[0][index] = first
            samples[1][index] = second
            samples[2][index] = J_tau
        if dissipation < -_ADMISSIBILITY_TOLERANCE:
            raise ValueError("the accumulated reduced dissipation is negative")

        moved = [adapt_domain_endpoint(functions[i][1], samples[i],
                                               float(alpha_smooth), float(eta))
                 for i in range(3)]
        for i in range(3):
            functions[i][2] = project_spline_parameters(
                functions[i][0], functions[i][1], functions[i][2], moved[i],
                samples[i])
            functions[i][1] = moved[i]
    return float(functions[2][1])
SCICODE_GOLD_EOF
