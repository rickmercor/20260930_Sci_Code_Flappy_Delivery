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


def _diabat_params(params: "np.ndarray") -> tuple:
    """Validated model parameters (omega_l, omega_r, x0, eps, V01) as floats."""
    p = np.asarray(params, dtype=float).ravel()
    if p.size != 5 or not np.all(np.isfinite(p)):
        raise ValueError("params must hold five finite values [omega_l, omega_r, x0, eps, V01]")
    omega_l, omega_r, x0, eps, v01 = (float(v) for v in p)
    if omega_l <= 0.0 or omega_r <= 0.0 or v01 <= 0.0:
        raise ValueError("omega_l, omega_r and V01 must be positive")
    if x0 == 0.0:
        raise ValueError("x0 must be nonzero")
    return omega_l, omega_r, x0, eps, v01


def two_diabat_potential(x: "np.ndarray", params: "np.ndarray") -> "np.ndarray":
    omega_l, omega_r, x0, eps, v01 = _diabat_params(params)
    x = np.asarray(x, dtype=float).ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError("positions must be finite")
    k_l, k_r = omega_l * omega_l, omega_r * omega_r
    v00 = 0.5 * k_l * (x + x0) ** 2 - eps
    v11 = 0.5 * k_r * (x - x0) ** 2
    gap = v00 - v11
    gap_slope = k_l * (x + x0) - k_r * (x - x0)
    root = np.sqrt(gap * gap + 4.0 * v01 * v01)
    v = 0.5 * (v00 + v11) - 0.5 * root
    dv = 0.5 * (k_l * (x + x0) + k_r * (x - x0)) - 0.5 * gap * gap_slope / root
    d2v = (0.5 * (k_l + k_r) - 0.5 * gap * (k_l - k_r) / root
           - 2.0 * v01 * v01 * gap_slope * gap_slope / root ** 3)
    return np.vstack([v, dv, d2v])

import numpy as np
from scipy.optimize import brentq


def _is_mirror_symmetric(params: "np.ndarray") -> bool:
    """True when the surface is symmetric under x -> -x (degenerate minima)."""
    omega_l, omega_r, _, eps, _ = _diabat_params(params)
    return omega_l == omega_r and eps == 0.0


def locate_stationary_points(params: "np.ndarray") -> "np.ndarray":
    x0 = _diabat_params(params)[2]
    # every stationary point of the lower adiabat lies between the two diabatic minima
    grid = np.linspace(-abs(x0), abs(x0), 20001)
    slope = two_diabat_potential(grid, params)[1]

    def _force(z):
        return float(two_diabat_potential(z, params)[1, 0])

    def _roots(brackets):
        return [brentq(_force, grid[i], grid[i + 1], xtol=1e-15, rtol=1e-15, maxiter=200)
                for i in np.flatnonzero(brackets)]

    minima = _roots((slope[:-1] < 0.0) & (slope[1:] >= 0.0))
    maxima = _roots((slope[:-1] > 0.0) & (slope[1:] <= 0.0))
    if len(minima) != 2:
        raise ValueError("the surface does not have exactly two minima")
    x_top = [x for x in maxima if minima[0] < x < minima[1]][0]
    v_minima = two_diabat_potential(np.array(minima), params)[0]
    if _is_mirror_symmetric(params) or v_minima[0] <= v_minima[1]:
        x_low, x_high = minima
    else:
        x_high, x_low = minima
    v, _, d2v = two_diabat_potential(np.array([x_low, x_top, x_high]), params)
    return np.array([x_low, x_top, x_high, v[0], v[1], v[2], np.sqrt(d2v[0]), np.sqrt(d2v[2])])

import numpy as np


def _ring_step(beta: float, n_beads: int) -> float:
    """Validated imaginary-time step beta/N between neighbouring ring beads (hbar = 1)."""
    if int(n_beads) != n_beads or n_beads < 4 or int(n_beads) % 2:
        raise ValueError("n_beads must be an even integer >= 4")
    if not (np.isfinite(beta) and beta > 0.0):
        raise ValueError("beta must be positive")
    return float(beta) / int(n_beads)


def _half_ring_beads(beads: "np.ndarray", beta: float, n_beads: int) -> tuple:
    """Validated half-ring positions and the imaginary-time step beta/N."""
    tau = _ring_step(beta, n_beads)
    y = np.asarray(beads, dtype=float).ravel()
    if y.size != int(n_beads) // 2 + 1:
        raise ValueError("beads must hold n_beads/2 + 1 positions")
    return y, tau


def half_ring_action(beads: "np.ndarray", params: "np.ndarray", beta: float, n_beads: int) -> tuple:
    y, tau = _half_ring_beads(beads, beta, n_beads)
    v, dv, d2v = two_diabat_potential(y, params)
    # the turning-point beads occur once on the ring, all other beads twice
    weight = np.ones(y.size)
    weight[0] = weight[-1] = 0.5
    stretch = np.diff(y)
    action = 0.5 * float(np.dot(stretch, stretch)) / tau + tau * float(np.dot(weight, v))
    gradient = tau * weight * dv
    gradient[:-1] -= stretch / tau
    gradient[1:] += stretch / tau
    hessian_diag = tau * weight * d2v + 2.0 / tau
    hessian_diag[0] -= 1.0 / tau
    hessian_diag[-1] -= 1.0 / tau
    hessian_offdiag = np.full(y.size - 1, -1.0 / tau)
    return action, gradient, hessian_diag, hessian_offdiag

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh_tridiagonal, solve_banded
from scipy.optimize import brentq


def _tridiagonal_solve(diag: "np.ndarray", off: "np.ndarray", rhs: "np.ndarray") -> "np.ndarray":
    """Solve the symmetric tridiagonal system with diagonal diag and off-diagonal off."""
    band = np.zeros((3, diag.size))
    band[0, 1:] = off
    band[1] = diag
    band[2, :-1] = off
    return solve_banded((1, 1), band, rhs)


def _inverted_orbit(params: "np.ndarray", start: float, speed: float, times: "np.ndarray", target: float) -> "np.ndarray":
    """Path of x'' = V'(x) (m = 1) leaving start towards target with the given speed, sampled
    at times; samples after the path stops approaching target are placed at target."""
    direction = float(np.sign(target - start))

    def _rhs(_t, u):
        return [u[1], float(two_diabat_potential(u[0], params)[1, 0])]

    def _passes_target(_t, u):
        return direction * (u[0] - target)
    _passes_target.terminal = True

    def _turns_back(_t, u):
        return direction * u[1] if _t > 0.0 else 1.0
    _turns_back.terminal = True
    _turns_back.direction = -1

    solution = solve_ivp(_rhs, (0.0, float(times[-1]) + 1.0), [start, direction * speed],
                         method="DOP853", rtol=1e-12, atol=1e-13, dense_output=True,
                         events=(_passes_target, _turns_back))
    path = np.full(times.size, target)
    reached = times <= solution.t[-1]
    path[reached] = solution.sol(times[reached])[0]
    return path


def optimize_instanton(params: "np.ndarray", beta: float, n_beads: int) -> "np.ndarray":
    tau = _ring_step(beta, n_beads)
    x_low, x_top, x_high, v_low, v_top, v_high = locate_stationary_points(params)[:6]
    last = int(n_beads) // 2
    index = np.arange(last + 1)
    symmetric = _is_mirror_symmetric(params)
    free = index[: (last + 1) // 2]
    if symmetric:
        # orbit at the energy of the minima, centred on the middle of the half ring
        right = index >= 0.5 * last
        y = np.empty(last + 1)
        y[right] = _inverted_orbit(params, 0.0, np.sqrt(2.0 * (v_top - v_high)),
                                   (index[right] - 0.5 * last) * tau, x_high)
        y[free] = -y[last - free]
    else:
        # orbit at the energy of the higher minimum, starting at rest at its turning point
        # inside the lower well
        x_turn = brentq(lambda z: float(two_diabat_potential(z, params)[0, 0]) - v_high,
                        x_low, x_top, xtol=1e-15, rtol=1e-15, maxiter=500)
        y = _inverted_orbit(params, x_turn, 0.0, index * tau, x_high)
    gtol = max(1e-12, 16.0 * np.finfo(float).eps * float(np.max(np.abs(y))) / tau)
    for _ in range(100):
        if symmetric:
            y[last - free] = -y[free]
            if last % 2 == 0:
                y[last // 2] = 0.0
        _, grad, hess_diag, hess_off = half_ring_action(y, params, beta, n_beads)
        converged = float(np.max(np.abs(grad))) < gtol
        if symmetric:
            if converged:
                break
            # Newton step within the mirror-symmetric subspace (the two halves move oppositely)
            diag = hess_diag[: free.size].copy()
            if last % 2:
                diag[-1] -= hess_off[0]
            step = np.zeros(last + 1)
            step[free] = _tridiagonal_solve(diag, hess_off[: free.size - 1], -grad[free])
        else:
            lowest = eigh_tridiagonal(hess_diag, hess_off, eigvals_only=True, select="i",
                                      select_range=(0, 1))
            if converged and lowest[0] < 0.0 < lowest[1]:
                break
            step = _tridiagonal_solve(hess_diag, hess_off, -grad)
        largest = float(np.max(np.abs(step)))
        if largest > 0.2:
            step *= 0.2 / largest
        y = y + step
    else:
        raise ValueError("the instanton search did not converge")
    if not ((y[0] - x_top) * (x_low - x_top) > 0.0 and (y[-1] - x_top) * (x_high - x_top) > 0.0):
        raise ValueError("the stationary point found does not connect the two wells")
    return y

import numpy as np


def select_dividing_surface(beads: "np.ndarray", params: "np.ndarray", beta: float, n_beads: int) -> "np.ndarray":
    y, tau = _half_ring_beads(beads, beta, n_beads)
    potential = two_diabat_potential(y, params)[0]
    k = 1 + int(np.argmax(potential[1:-1]))
    # the arc through y_0 runs from x_{N-k} over x_0 to x_k
    n_low = 2 * k
    n_high = int(n_beads) - n_low
    qdot = abs(y[k + 1] - y[k - 1]) / (2.0 * tau)
    return np.array([k, y[k], n_low, n_high, n_low * tau, n_high * tau, qdot], dtype=float)

import numpy as np


def _dividing_bead(k: int, n_beads: int) -> int:
    """Validated dividing-surface bead index 1 <= k <= N/2 - 1."""
    if int(k) != k or not 1 <= int(k) <= int(n_beads) // 2 - 1:
        raise ValueError("k must be an interior bead of the half ring")
    return int(k)


def _logdet_open_chain(diag: "np.ndarray") -> float:
    """log|det| of the symmetric tridiagonal matrix with diagonal diag and off-diagonals -1."""
    log_det, pivot = 0.0, None
    for a in diag:
        pivot = a if pivot is None else a - 1.0 / pivot
        log_det += np.log(abs(pivot))
    return float(log_det)


def _logdet_collapsed_ring(n: int, c: float) -> float:
    """log det of the n x n cyclic matrix with diagonal 2 + c and neighbour couplings -1."""
    theta = n * float(np.arccosh(1.0 + 0.5 * c))
    return theta + 2.0 * float(np.log1p(-np.exp(-theta)))


def fluctuation_factor(beads: "np.ndarray", params: "np.ndarray", beta: float, n_beads: int, k: int) -> "np.ndarray":
    y, tau = _half_ring_beads(beads, beta, n_beads)
    k = _dividing_bead(k, n_beads)
    last = y.size - 1
    omega_low, omega_high = locate_stationary_points(params)[6:]
    diag = 2.0 + tau * tau * two_diabat_potential(y, params)[2]
    # removing x_k and x_{N-k} leaves two open chains: one through y_0, one through y_{N/2}
    low_chain = np.concatenate([diag[k - 1:0:-1], diag[:k]])
    high_chain = np.concatenate([diag[k + 1:], diag[last - 1:k:-1]])
    logdet_pin = _logdet_open_chain(low_chain) + _logdet_open_chain(high_chain)
    logdet_low = _logdet_collapsed_ring(2 * k, (tau * omega_low) ** 2)
    logdet_high = _logdet_collapsed_ring(int(n_beads) - 2 * k, (tau * omega_high) ** 2)
    phi = np.sqrt(tau) * np.exp(0.25 * (logdet_pin - logdet_low - logdet_high))
    return np.array([logdet_pin, logdet_low, logdet_high, phi])

import numpy as np


def instanton_exponent(beads: "np.ndarray", params: "np.ndarray", beta: float, n_beads: int, k: int) -> float:
    action = half_ring_action(beads, params, beta, n_beads)[0]
    k = _dividing_bead(k, n_beads)
    tau = _ring_step(beta, n_beads)
    v_low, v_high = locate_stationary_points(params)[[3, 5]]
    tau_low = 2 * k * tau
    tau_high = (int(n_beads) - 2 * k) * tau
    return float(action - 0.5 * tau_low * v_low - 0.5 * tau_high * v_high)

import numpy as np


def tunnelling_frequency(params: "np.ndarray", beta: float, n_beads: int) -> float:
    beads = optimize_instanton(params, beta, n_beads)
    surface = select_dividing_surface(beads, params, beta, n_beads)
    k, qdot = int(surface[0]), float(surface[6])
    phi = float(fluctuation_factor(beads, params, beta, n_beads, k)[3])
    exponent = instanton_exponent(beads, params, beta, n_beads, k)
    return float(qdot / (phi * np.sqrt(2.0 * np.pi)) * np.exp(-exponent))
SCICODE_GOLD_EOF
