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
from scipy.optimize import brentq

def _epi_require_scalar(value: object, name: str, low: float, strict: bool) -> float:
    """Return ``value`` as a finite float that respects the lower bound ``low``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + " must be a real number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(name + " must be finite")
    if strict and not number > low:
        raise ValueError(name + " must be greater than " + repr(low))
    if not strict and number < low:
        raise ValueError(name + " must be at least " + repr(low))
    return number

def _epi_require_vector(values: object, name: str) -> "np.ndarray":
    """Return ``values`` as a non-empty one-dimensional array of finite floats."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(name + " must be an array of real numbers") from error
    if array.ndim != 1 or array.size == 0 or not bool(np.all(np.isfinite(array))):
        raise ValueError(name + " must be a non-empty 1-D array of finite numbers")
    return array

def _epi_require_count(value: object, name: str, low: int) -> int:
    """Return ``value`` as a Python integer of at least ``low`` (booleans are rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    count = int(value)
    if count < low:
        raise ValueError(name + " must be at least " + str(low))
    return count

def solve_epigenetic_equilibria(alpha: float, beta: float, c: float,
                                        scan_points: int = 4001) -> "np.ndarray":
    """Reference implementation using monotonic intervals and Brent refinement."""
    import numpy as np

    alpha = _epi_require_scalar(alpha, "alpha", 0.0, False)
    beta = _epi_require_scalar(beta, "beta", 0.0, True)
    c = _epi_require_scalar(c, "c", -np.inf, False)
    samples = _epi_require_count(scan_points, "scan_points", 3)

    def _residual(value):
        return alpha * np.tanh(beta * (value + c)) - value

    span = alpha + 1.0  # every solution obeys |theta| <= alpha
    grid = np.linspace(-span, span, samples)
    gain = alpha * beta
    if gain > 1.0:
        displacement = np.arccosh(np.sqrt(gain)) / beta
        stationary = np.array([-c - displacement, -c + displacement], dtype=float)
        stationary = stationary[(stationary >= -span) & (stationary <= span)]
        grid = np.unique(np.concatenate((grid, stationary)))
    values = _residual(grid)

    roots = [float(node) for node, value in zip(grid, values) if value == 0.0]
    for index in np.nonzero(values[:-1] * values[1:] < 0.0)[0]:
        roots.append(float(brentq(_residual, grid[index], grid[index + 1],
                                  xtol=1e-15, rtol=4.0 * float(np.finfo(float).eps))))
    if not roots:
        raise ValueError("the scan brackets no equilibrium; increase scan_points")

    marks = np.array(sorted(roots), dtype=float)
    if marks.size > 1:
        keep = np.concatenate(([True], np.diff(marks) > 1e-10))
        marks = marks[keep]
    rates = alpha * beta / np.cosh(beta * (marks + c)) ** 2 - 1.0
    return np.column_stack([marks, rates])

import numpy as np

def settle_starting_marks(starting: "np.ndarray", equilibria: "np.ndarray",
                                  c: float) -> "np.ndarray":
    """Reference implementation (watershed search and basin counting)."""
    import numpy as np

    marks = _epi_require_vector(starting, "starting")
    c = _epi_require_scalar(c, "c", -np.inf, False)
    try:
        table = np.asarray(equilibria, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("equilibria must be an array of real numbers") from error
    if (table.ndim != 2 or table.shape[1] != 2 or table.shape[0] == 0
            or not bool(np.all(np.isfinite(table)))):
        raise ValueError("equilibria must be a non-empty finite (M,2) array")
    if table.shape[0] > 1 and not bool(np.all(np.diff(table[:, 0]) > 0.0)):
        raise ValueError("equilibria marks must be strictly increasing")
    rates = table[:, 1]
    tolerance = 1e-12
    stable = rates < -tolerance
    repelling = rates > tolerance
    marginal = ~(stable | repelling)

    if table.shape[0] == 1 and (stable[0] or marginal[0]):
        terminal = table[:, 0]
        basin = np.zeros(marks.size, dtype=int)
    elif (table.shape[0] == 2 and int(np.sum(stable)) == 1
          and int(np.sum(marginal)) == 1 and not bool(np.any(repelling))):
        terminal = table[:, 0]
        marginal_index = int(np.nonzero(marginal)[0][0])
        fold = float(terminal[marginal_index])
        if marginal_index == 0:
            basin = np.where(marks <= fold, 0, 1)
        else:
            basin = np.where(marks >= fold, 1, 0)
    else:
        standard = (table.shape[0] % 2 == 1
                    and bool(np.all(stable[0::2]))
                    and bool(np.all(repelling[1::2])))
        if not standard:
            raise ValueError("equilibrium rates do not form an admissible attractor/watershed pattern")
        terminal = table[0::2, 0]
        watersheds = table[1::2, 0]
        if bool(np.any(np.isin(marks, watersheds))):
            raise ValueError("a starting mark sits exactly on a watershed")
        basin = np.searchsorted(watersheds, marks)

    counts = np.bincount(basin, minlength=terminal.size)
    rows = [[float(terminal[k]) + c, counts[k] / float(marks.size)]
            for k in range(terminal.size) if counts[k] > 0]
    return np.array(sorted(rows), dtype=float)

import numpy as np

def _epi_normal_nodes(nodes: object) -> tuple:
    """Return equally spaced standard-normal abscissae on [-12, 12] and their weights."""
    count = _epi_require_count(nodes, "nodes", 3)
    abscissae = np.linspace(-12.0, 12.0, count)
    step = 24.0 / (count - 1)
    return abscissae, step * np.exp(-0.5 * abscissae ** 2) / np.sqrt(2.0 * np.pi)

def _epi_require_ensemble(fields: object, weights: object) -> tuple:
    """Return the constant inputs and their population shares as validated float arrays."""
    values = _epi_require_vector(fields, "fields")
    shares = _epi_require_vector(weights, "weights")
    if shares.size != values.size:
        raise ValueError("fields and weights must have the same length")
    if bool(np.any(shares < 0.0)):
        raise ValueError("weights must be nonnegative")
    if abs(float(np.sum(shares)) - 1.0) > 1e-9:
        raise ValueError("weights must sum to one within 1e-9")
    return values, shares

def _epi_require_lags(deltas: object, delta_zero: object) -> tuple:
    """Return the lag values and the equal-time correlation after range checks."""
    lags = _epi_require_vector(deltas, "deltas")
    equal_time = _epi_require_scalar(delta_zero, "delta_zero", 0.0, False)
    if bool(np.any(np.abs(lags) > equal_time)):
        raise ValueError("every lag value must lie in [-delta_zero, delta_zero]")
    return lags, equal_time

def _epi_pair_profile(lags: "np.ndarray", equal_time: float, fields: "np.ndarray",
                      weights: "np.ndarray", nodes: object, kernel) -> "np.ndarray":
    """Return the population-weighted three-Gaussian pair average of ``kernel`` per lag."""
    abscissae, quad = _epi_normal_nodes(nodes)
    result = np.empty(lags.size)
    for start in range(0, lags.size, 8):
        lag = lags[start:start + 8]
        magnitude = np.abs(lag)
        spread = np.sqrt(equal_time - magnitude)
        shared = np.sqrt(magnitude)[:, None, None] * abscissae[None, :, None]
        centre_left = shared + fields[None, None, :]
        signs = np.where(lag < 0.0, -1.0, 1.0)[:, None, None]
        centre_right = signs * shared + fields[None, None, :]
        private = spread[:, None, None, None] * abscissae[None, :, None, None]
        values_left = kernel(private + centre_left[:, None, :, :])
        inner_left = np.tensordot(quad, values_left, axes=(0, 1))
        values_right = kernel(private + centre_right[:, None, :, :])
        inner_right = np.tensordot(quad, values_right, axes=(0, 1))
        result[start:start + 8] = np.einsum(
            "j,mjk,mjk,k->m", quad, inner_left, inner_right, weights)
    return result

def average_correlation_overlap(deltas: "np.ndarray", delta_zero: float,
                                        fields: "np.ndarray", weights: "np.ndarray",
                                        beta: float, nodes: int = 241) -> "np.ndarray":
    """Reference implementation (trapezoidal Gaussian averages, vectorised over lags)."""
    import numpy as np

    lags, equal_time = _epi_require_lags(deltas, delta_zero)
    values, shares = _epi_require_ensemble(fields, weights)
    beta = _epi_require_scalar(beta, "beta", 0.0, True)

    def _kernel(argument):
        return np.tanh(beta * argument)

    return _epi_pair_profile(lags, equal_time, values, shares, nodes, _kernel)

import numpy as np

def average_gain_overlap(deltas: "np.ndarray", delta_zero: float, fields: "np.ndarray",
                                 weights: "np.ndarray", beta: float,
                                 nodes: int = 241) -> "np.ndarray":
    """Reference implementation (same rule as the correlation overlap)."""
    import numpy as np

    lags, equal_time = _epi_require_lags(deltas, delta_zero)
    values, shares = _epi_require_ensemble(fields, weights)
    beta = _epi_require_scalar(beta, "beta", 0.0, True)

    def _kernel(argument):
        return beta / np.cosh(beta * argument) ** 2

    return _epi_pair_profile(lags, equal_time, values, shares, nodes, _kernel)

import numpy as np

def average_potential_overlap(deltas: "np.ndarray", delta_zero: float,
                                      fields: "np.ndarray", weights: "np.ndarray",
                                      beta: float, nodes: int = 241) -> "np.ndarray":
    """Reference implementation (same rule as the correlation overlap)."""
    import numpy as np

    lags, equal_time = _epi_require_lags(deltas, delta_zero)
    values, shares = _epi_require_ensemble(fields, weights)
    beta = _epi_require_scalar(beta, "beta", 0.0, True)

    def _kernel(argument):
        scaled = beta * argument
        return (np.logaddexp(scaled, -scaled) - np.log(2.0)) / beta  # overflow-safe ln cosh

    return _epi_pair_profile(lags, equal_time, values, shares, nodes, _kernel)

import numpy as np
from scipy.optimize import brentq

def solve_frozen_correlation(fields: "np.ndarray", weights: "np.ndarray", beta: float,
                                     lower: float, upper: float, nodes: int = 241) -> float:
    """Reference implementation (Brent's method on the equal-time residual)."""
    import numpy as np

    low = _epi_require_scalar(lower, "lower", 0.0, False)
    high = _epi_require_scalar(upper, "upper", -np.inf, False)
    if not high > low:
        raise ValueError("upper must be greater than lower")

    def _residual(value):
        overlap = average_correlation_overlap(np.array([value]), value, fields,
                                                      weights, beta, nodes)
        return float(overlap[0]) - value

    low_value = _residual(low)
    high_value = _residual(high)
    if low_value * high_value > 0.0:
        raise ValueError("the residual does not change sign across the bracket")
    if low_value == 0.0 or high_value == 0.0:
        return low if low_value == 0.0 else high
    return float(brentq(_residual, low, high, xtol=1e-15,
                        rtol=4.0 * float(np.finfo(float).eps)))

import numpy as np
from scipy.optimize import brentq

def locate_correlation_hilltop(delta_zero: float, fields: "np.ndarray",
                                       weights: "np.ndarray", beta: float, nodes: int = 241,
                                       scan_points: int = 17) -> float:
    """Reference implementation (vectorised slope scan, then Brent refinement)."""
    import numpy as np

    equal_time = _epi_require_scalar(delta_zero, "delta_zero", 0.0, True)
    samples = _epi_require_count(scan_points, "scan_points", 3)

    def _slope(value):
        return float(average_correlation_overlap(np.array([value]), equal_time, fields,
                                                         weights, beta, nodes)[0]) - value

    lags = np.linspace(0.0, equal_time, samples)
    lags[-1] = equal_time
    # The scan uses the same scalar evaluation as the refinement, so the bracket signs agree.
    slopes = [_slope(float(lag)) for lag in lags]
    for index in range(samples - 1):
        if slopes[index] > 0.0 >= slopes[index + 1]:
            # Brent's method returns an endpoint whose slope is exactly zero as the root.
            return float(brentq(_slope, lags[index], lags[index + 1], xtol=1e-15,
                                rtol=4.0 * float(np.finfo(float).eps)))
    return float("nan")

import numpy as np
from scipy.optimize import brentq

def solve_decaying_correlation(fields: "np.ndarray", weights: "np.ndarray", beta: float,
                                       lower: float, upper: float, nodes: int = 241,
                                       scan_points: int = 17) -> "np.ndarray":
    """Reference implementation (Brent's method on the energy gap)."""
    import numpy as np

    low = _epi_require_scalar(lower, "lower", 0.0, True)
    high = _epi_require_scalar(upper, "upper", -np.inf, False)
    if not high > low:
        raise ValueError("upper must be greater than lower")

    def _gap(equal_time):
        top = locate_correlation_hilltop(equal_time, fields, weights, beta, nodes,
                                                 scan_points)
        if np.isnan(top):
            return 1.0
        lags = np.array([top, equal_time])
        potential = (average_potential_overlap(lags, equal_time, fields, weights, beta,
                                                       nodes) - 0.5 * lags ** 2)
        return float(potential[1] - potential[0])

    if not (_gap(low) > 0.0 > _gap(high)):
        raise ValueError("the energy gap must be positive at lower and negative at upper")
    equal_time = float(brentq(_gap, low, high, xtol=1e-15,
                              rtol=4.0 * float(np.finfo(float).eps)))
    top = locate_correlation_hilltop(equal_time, fields, weights, beta, nodes,
                                             scan_points)
    return np.array([equal_time, top])

import numpy as np
from numpy.polynomial import chebyshev
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

def _epi_chebyshev_fit(function, lower: float, upper: float, degree: int) -> "np.ndarray":
    """Return Chebyshev coefficients interpolating a vectorised function on [lower, upper]."""
    half, middle = 0.5 * (upper - lower), 0.5 * (upper + lower)
    return chebyshev.chebinterpolate(lambda t: function(half * np.asarray(t) + middle), degree)

def _epi_chebyshev_eval(coefficients: "np.ndarray", values, lower: float, upper: float):
    """Evaluate Chebyshev coefficients fitted on [lower, upper] at the given values."""
    return chebyshev.chebval((2.0 * np.asarray(values) - lower - upper) / (upper - lower),
                             coefficients)

def trace_decay_trajectory(delta_zero: float, delta_inf: float, fields: "np.ndarray",
                                   weights: "np.ndarray", beta: float, t_max: float,
                                   n_tau: int, nodes: int = 241,
                                   degree: int = 64) -> "np.ndarray":
    """Reference implementation (backward integration along the unstable manifold)."""
    import numpy as np

    top = _epi_require_scalar(delta_inf, "delta_inf", 0.0, False)
    start = _epi_require_scalar(delta_zero, "delta_zero", top, True)
    horizon = _epi_require_scalar(t_max, "t_max", 0.0, True)
    samples = _epi_require_count(n_tau, "n_tau", 2)
    order = _epi_require_count(degree, "degree", 4)

    def _slope(values):
        lags = np.atleast_1d(np.asarray(values, dtype=float))
        return average_correlation_overlap(lags, start, fields, weights, beta,
                                                   nodes) - lags

    coefficients = _epi_chebyshev_fit(_slope, top, start, order)
    width = start - top

    def _fitted(value):
        return float(_epi_chebyshev_eval(coefficients, value, top, start))

    # Start on the interpolant's own hilltop, so the release is consistent with the force used.
    low, high = top - 1e-3 * width, top + 1e-3 * width
    if not (_fitted(low) > 0.0 > _fitted(high)):
        raise ValueError("the interpolated slope does not fall through zero near delta_inf")
    summit = float(brentq(_fitted, low, high, xtol=1e-15, rtol=4.0 * float(np.finfo(float).eps)))
    fluctuation = 1.0 - float(average_gain_overlap(np.array([max(summit, 0.0)]), start,
                                                           fields, weights, beta, nodes)[0])
    if not fluctuation > 0.0:
        raise ValueError("delta_inf is not a hilltop: the fluctuation potential is not positive")
    kappa = np.sqrt(fluctuation)
    eps = 1e-11 * width

    def _rest(s, state):
        return state[1]
    _rest.terminal, _rest.direction = True, -1
    solution = solve_ivp(lambda s, state: [state[1], -_fitted(state[0])], (0.0, 1.0e4),
                         [summit + eps, kappa * eps], method="DOP853", rtol=1e-12, atol=1e-15,
                         events=_rest, dense_output=True)
    if solution.t_events[0].size == 0:
        raise ValueError("the backward integration never came to rest")
    if float(solution.y_events[0][0][0]) < top + 0.5 * width:
        raise ValueError("the motion came to rest before approaching delta_zero")
    s_turn = float(solution.t_events[0][0])

    tau = np.linspace(0.0, horizon, samples)
    reversed_time = s_turn - tau
    profile = summit + eps * np.exp(kappa * np.minimum(reversed_time, 0.0))
    inside = reversed_time >= 0.0
    profile[inside] = solution.sol(reversed_time[inside])[0]
    return profile

import numpy as np
from scipy.integrate import solve_bvp
from scipy.linalg import eigh_tridiagonal

def solve_fluctuation_ground_state(half_profile: "np.ndarray", spacing: float) -> float:
    """Solve the continuous piecewise-linear well with an exact Robin tail."""
    import numpy as np

    samples = _epi_require_vector(half_profile, "half_profile")
    if samples.size < 3:
        raise ValueError("half_profile must hold at least 3 samples")
    step = _epi_require_scalar(spacing, "spacing", 0.0, True)

    plateau = float(samples[-1])
    floor = float(np.min(samples))
    if not floor < plateau:
        raise ValueError("the potential has no well below its long-lag plateau")
    # An exactly constant suffix is analytically equivalent to the Robin tail.
    # Removing it avoids an arbitrarily long numerical interval, without
    # approximating any varying part of the supplied potential.
    last_change = int(np.flatnonzero(samples != plateau)[-1])
    values = samples[:last_change + 2]
    times = np.arange(values.size, dtype=float) * step

    # The positive finite-difference ground mode is only an initial guess.
    # Adaptive collocation below solves the continuous, interpolated potential.
    # Include a constant tail in the initial-guess eigenproblem: a Dirichlet
    # wall at a short sampling window can otherwise erase a shallow bound mode.
    padding = max(8, int(np.ceil(12.0 / (step * np.sqrt(plateau - floor)))))
    padded = np.concatenate([values, np.full(padding, plateau)])
    window = np.concatenate([padded[:0:-1], padded])
    coupling = 1.0 / step ** 2
    energies, modes = eigh_tridiagonal(
        window + 2.0 * coupling, np.full(window.size - 1, -coupling),
        select="i", select_range=(0, 0))
    estimate = min(float(energies[0]), plateau - 1e-5)
    mode = modes[padded.size - 1:padded.size - 1 + values.size, 0]
    mode = mode / mode[0]
    initial = np.vstack((mode, np.gradient(mode, times)))

    # Parameterizing plateau-E by exp(p) keeps the exact decay rate real.
    def _equation(tau, state, parameter):
        energy = plateau - np.exp(parameter[0])
        potential = np.interp(tau, times, values)
        return np.vstack((state[1], (potential - energy) * state[0]))

    def _boundary(left, right, parameter):
        decay = np.exp(0.5 * parameter[0])
        return np.array([left[0] - 1.0, left[1], right[1] + decay * right[0]])

    solution = solve_bvp(
        _equation, _boundary, times, initial,
        p=np.array([np.log(plateau - estimate)]), tol=1e-10,
        max_nodes=max(100000, 4 * times.size))
    if not solution.success:
        raise RuntimeError("the continuous bound-state solve did not converge")
    energy = float(plateau - np.exp(solution.p[0]))
    amplitude = solution.y[0]
    if (not floor < energy < plateau
            or float(np.min(amplitude)) < -1e-8 * float(np.max(np.abs(amplitude)))):
        raise RuntimeError("the converged eigenfunction is not a nodeless bound state")
    return energy

import numpy as np

def compute_settled_lyapunov_exponent(
    inputs: tuple = (-0.32, -0.1, 0.04, 0.11, 0.35),
    fractions: tuple = (0.14, 0.22, 0.27, 0.21, 0.16),
    alpha: float = 0.5,
    beta: float = 4.0,
    n_marks: int = 1000,
    mark_span: float = 2.5,
    nodes: int = 241,
    t_max: float = 40.0,
    n_tau: int = 4001,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    classes = _epi_require_vector(inputs, "inputs")
    shares = _epi_require_vector(fractions, "fractions")
    if shares.size != classes.size:
        raise ValueError("inputs and fractions must have the same length")
    if bool(np.any(shares <= 0.0)) or abs(float(np.sum(shares)) - 1.0) > 1e-9:
        raise ValueError("fractions must be positive and sum to one within 1e-9")
    span = _epi_require_scalar(mark_span, "mark_span", 0.0, True)
    horizon = _epi_require_scalar(t_max, "t_max", 0.0, True)
    marks = _epi_require_count(n_marks, "n_marks", 2)
    coarse = _epi_require_count(n_tau, "n_tau", 3)

    starting = np.linspace(-span, span, marks)
    fields, weights = [], []
    for external, share in zip(classes, shares):
        table = solve_epigenetic_equilibria(alpha, beta, float(external))
        for value, basin in settle_starting_marks(starting, table, float(external)):
            fields.append(float(value))
            weights.append(float(share) * float(basin))
    fields = np.asarray(fields, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / float(np.sum(weights))  # removes accumulated rounding only

    grid = np.linspace(0.0, 1.0, 41)
    residual = np.array([float(average_correlation_overlap(
        np.array([node]), node, fields, weights, beta, nodes)[0]) - node for node in grid])
    crossings = np.nonzero(residual[:-1] * residual[1:] < 0.0)[0]
    if crossings.size > 0:
        last = int(crossings[-1])
        d_star = solve_frozen_correlation(fields, weights, beta, float(grid[last]),
                                                  float(grid[last + 1]), nodes)
    else:
        exact = np.nonzero(residual == 0.0)[0]
        if exact.size == 0:
            raise ValueError("no time-independent state on [0, 1]")
        d_star = float(grid[int(exact[-1])])

    static = 1.0 - float(average_gain_overlap(np.array([d_star]), d_star, fields,
                                                      weights, beta, nodes)[0])
    if static >= 0.0:
        # A stable frozen state: constant fluctuation potential, lowest eigenvalue W itself.
        return float(-1.0 + np.sqrt(1.0 - static))

    delta_zero, delta_inf = solve_decaying_correlation(
        fields, weights, beta, 0.5 * d_star, d_star * (1.0 - 1e-9), nodes)
    checked_top = locate_correlation_hilltop(delta_zero, fields, weights, beta, nodes)
    if not np.isfinite(checked_top) or abs(float(checked_top) - float(delta_inf)) > 5e-10:
        raise ValueError("the decaying solution does not end on the selected hilltop")
    endpoints = np.array([delta_inf, delta_zero])
    endpoint_potential = (average_potential_overlap(
        endpoints, delta_zero, fields, weights, beta, nodes) - 0.5 * endpoints ** 2)
    if abs(float(endpoint_potential[1] - endpoint_potential[0])) > 5e-10:
        raise ValueError("the decaying solution does not conserve endpoint energy")
    fluctuation = _epi_chebyshev_fit(
        lambda lags: 1.0 - average_gain_overlap(np.atleast_1d(lags), delta_zero, fields,
                                                        weights, beta, nodes),
        delta_inf, delta_zero, 64)
    energies = []
    for points in (coarse, 2 * coarse - 1):
        profile = trace_decay_trajectory(delta_zero, delta_inf, fields, weights, beta,
                                                 horizon, points, nodes)
        potential = _epi_chebyshev_eval(fluctuation, profile, delta_inf, delta_zero)
        energies.append(solve_fluctuation_ground_state(potential, horizon / (points - 1)))
    ground = (4.0 * energies[1] - energies[0]) / 3.0  # leading O(h^2) interpolation error
    if ground > 1.0:
        raise ValueError("the fluctuation operator admits no real Lyapunov exponent")
    return float(-1.0 + np.sqrt(1.0 - ground))
SCICODE_GOLD_EOF
