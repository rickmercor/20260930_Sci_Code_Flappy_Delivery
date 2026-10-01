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


def _real_array(value, name):
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must be real")
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _real_scalar(value, name):
    result = _real_array(value, name)
    if result.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    return float(result)


def _positive_integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _checked_mixture(probabilities, shapes, rates):
    arrays = [
        _real_array(x, name)
        for x, name in zip(
            (probabilities, shapes, rates), ("probabilities", "shapes", "rates")
        )
    ]
    if any(x.ndim != 1 or x.size == 0 or np.any(x <= 0) for x in arrays):
        raise ValueError("mixture inputs must be positive nonempty vectors")
    if len({x.shape for x in arrays}) != 1:
        raise ValueError("mixture vectors must have equal shapes")
    if abs(np.sum(arrays[0]) - 1.0) > 1e-12:
        raise ValueError("probabilities must sum to one")
    return arrays


def _checked_market(market):
    market = _real_array(market, "market")
    if market.shape != (5,):
        raise ValueError("market must contain kappa, baseline, initial, beta, rate")
    if market[0] <= 0 or np.any(market[1:] < 0):
        raise ValueError("kappa must be positive; other market entries nonnegative")
    return market


def tilt_mark_distribution(
    probabilities: np.ndarray,
    shapes: np.ndarray,
    rates: np.ndarray,
    theta: float,
) -> np.ndarray:
    p, k, b = _checked_mixture(probabilities, shapes, rates)
    theta = _real_scalar(theta, "theta")
    if theta >= np.min(b):
        raise ValueError("theta must be below every rate")
    log_weights = np.log(p) + k * (np.log(b) - np.log(b - theta))
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    if np.any(weights <= 0):
        raise ValueError("tilted probabilities underflow")
    return np.column_stack((weights, k, b - theta))

import numpy as np
from scipy.optimize import brentq


def calibrate_mark_tilt(
    probabilities: np.ndarray,
    shapes: np.ndarray,
    rates: np.ndarray,
    market: np.ndarray,
    swap_maturity: float,
    swap_price: float,
    bracket: np.ndarray,
) -> float:
    market = _checked_market(market)
    if market[1] + market[2] <= 0:
        raise ValueError("calibration requires positive arrival activity")
    maturity = _real_scalar(swap_maturity, "swap_maturity")
    quote = _real_scalar(swap_price, "swap_price")
    bounds = _real_array(bracket, "bracket")
    if maturity <= 0 or quote < 0 or bounds.shape != (2,) or bounds[0] >= bounds[1]:
        raise ValueError("invalid maturity, quote, or bracket")

    def _residual(theta):
        mixture = tilt_mark_distribution(probabilities, shapes, rates, theta)
        mean = np.sum(mixture[:, 0] * mixture[:, 1] / mixture[:, 2])
        kappa, baseline, initial, beta, rate = market
        decay = kappa - beta * mean
        if decay <= 0:
            raise ValueError("calibration bracket must be subcritical")
        equilibrium = kappa * baseline / decay
        integrated = (
            equilibrium * maturity
            + (initial - equilibrium) * (-np.expm1(-decay * maturity)) / decay
        )
        return float(np.exp(-rate * maturity) * mean * integrated - quote)

    left, right = _residual(bounds[0]), _residual(bounds[1])
    if left == 0:
        return float(bounds[0])
    if right == 0:
        return float(bounds[1])
    if left * right > 0:
        raise ValueError("swap quote is outside bracket")
    return float(brentq(_residual, bounds[0], bounds[1], xtol=1e-13, rtol=1e-14))

import numpy as np
from scipy.special import gammaln, roots_genlaguerre


def build_mark_quadrature(
    mixture: np.ndarray,
    order: int,
) -> np.ndarray:
    mixture = _real_array(mixture, "mixture")
    order = _positive_integer(order, "order")
    if mixture.ndim != 2 or mixture.shape[1] != 3 or order > 128:
        raise ValueError("invalid mixture shape or quadrature order")
    p, k, b = _checked_mixture(mixture[:, 0], mixture[:, 1], mixture[:, 2])
    result = np.empty((len(p), order, 2))
    for m, (probability, shape, rate) in enumerate(zip(p, k, b)):
        nodes, weights = roots_genlaguerre(order, shape - 1)
        result[m, :, 0] = nodes / rate
        result[m, :, 1] = probability * np.exp(np.log(weights) - gammaln(shape))
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError(
            "quadrature cannot be represented with positive finite entries"
        )
    return result

import numpy as np


def _uniform_intensity_grid(grid):
    grid = _real_array(grid, "grid")
    if grid.ndim != 1 or grid.size < 3 or grid[0] != 0:
        raise ValueError("grid must start at zero and contain at least three nodes")
    differences = np.diff(grid)
    if np.any(differences <= 0) or not np.allclose(
        differences, differences[0], rtol=1e-12, atol=1e-14
    ):
        raise ValueError("grid must be increasing and uniform")
    return grid


def build_backward_drift(
    grid: np.ndarray,
    market: np.ndarray,
    dt: float,
) -> np.ndarray:
    grid = _uniform_intensity_grid(grid)
    market = _checked_market(market)
    dt = _real_scalar(dt, "dt")
    if dt <= 0 or market[1] > grid[-1]:
        raise ValueError("positive dt and baseline inside grid required")
    drift = market[0] * (market[1] - grid)
    spacing = grid[1] - grid[0]
    bands = np.zeros((3, grid.size))
    bands[1] = 1 + dt * (np.abs(drift) / spacing + market[4])
    bands[0, 1:] = -dt * np.maximum(drift[:-1], 0) / spacing
    bands[2, :-1] = dt * np.minimum(drift[1:], 0) / spacing
    return bands

import numpy as np


def build_jump_transfer(
    grid: np.ndarray,
    rule: np.ndarray,
    beta: float,
    frequencies: np.ndarray,
    delta: float,
) -> np.ndarray:
    grid = _uniform_intensity_grid(grid)
    rule = _real_array(rule, "rule")
    beta = _real_scalar(beta, "beta")
    delta = _real_scalar(delta, "delta")
    frequencies = _real_array(frequencies, "frequencies")
    if (
        rule.ndim != 3
        or rule.shape[2] != 2
        or min(rule.shape[:2]) < 1
        or np.any(rule <= 0)
    ):
        raise ValueError("rule must contain positive nodes and weights")
    if abs(np.sum(rule[:, :, 1]) - 1) > 1e-12:
        raise ValueError("quadrature weights must sum to one")
    if beta < 0 or delta < 0 or frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("invalid excitation, shift, or frequencies")
    if frequencies[0] != 0 or np.any(np.diff(frequencies) <= 0):
        raise ValueError("frequencies must increase from zero")
    nodes = rule[:, :, 0].ravel()
    weights = rule[:, :, 1].ravel()
    if np.max(delta * nodes) > 700:
        raise ValueError("modal exponent exceeds supported finite range")
    eta = delta + 1j * frequencies
    result = np.zeros((frequencies.size, grid.size, grid.size), dtype=complex)
    rows = np.arange(grid.size)
    spacing = grid[1] - grid[0]
    for node, weight in zip(nodes, weights):
        query = np.minimum(grid + beta * node, grid[-1])
        left = np.minimum(np.floor(query / spacing).astype(int), grid.size - 2)
        fraction = (query - grid[left]) / spacing
        factor = weight * np.exp(eta * node)
        result[:, rows, left] += factor[:, None] * (1 - fraction)
        result[:, rows, left + 1] += factor[:, None] * fraction
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite modal transfer")
    return result

import numpy as np
from scipy.linalg import solve_banded


def evolve_modal_values(
    grid: np.ndarray,
    bands: np.ndarray,
    transfer: np.ndarray,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    grid = _uniform_intensity_grid(grid)
    bands = _real_array(bands, "bands")
    try:
        transfer = np.asarray(transfer, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError("transfer must be numerical") from exc
    dt = _real_scalar(dt, "dt")
    n_steps = _positive_integer(n_steps, "n_steps")
    size = grid.size
    if (
        bands.shape != (3, size)
        or transfer.ndim != 3
        or transfer.shape[0] < 1
        or transfer.shape[1:] != (size, size)
    ):
        raise ValueError("incompatible modal dimensions")
    if dt <= 0 or not np.all(np.isfinite(transfer)) or np.any(bands[[0, 2]] > 0):
        raise ValueError("invalid matrix, transfer, or time step")
    if bands[0, 0] != 0 or bands[2, -1] != 0:
        raise ValueError("unused band corners must be zero")
    row_sum = bands[1].copy()
    row_sum[:-1] += bands[0, 1:]
    row_sum[1:] += bands[2, :-1]
    if np.any(row_sum <= 0):
        raise ValueError("implicit matrix must be strictly row dominant")
    cfl = dt * np.max(grid[None, :] * (1 + np.sum(np.abs(transfer), axis=2)))
    if cfl >= 1:
        raise ValueError("explicit jump stability factor must be below one")
    values = np.ones((size, transfer.shape[0]), dtype=complex)
    for _ in range(n_steps):
        gain = np.einsum("fij,jf->if", transfer, values, optimize=False)
        rhs = values + dt * grid[:, None] * (gain - values)
        values = solve_banded((1, 1), bands, rhs, check_finite=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("modal evolution is non-finite")
    return values

import numpy as np


def invert_capped_payoff(
    grid: np.ndarray,
    modes: np.ndarray,
    frequencies: np.ndarray,
    delta: float,
    initial_intensity: float,
    initial_loss: float,
    strike: float,
    cap: float,
) -> float:
    grid = _uniform_intensity_grid(grid)
    frequencies = _real_array(frequencies, "frequencies")
    try:
        modes = np.asarray(modes, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError("modes must be numerical") from exc
    delta = _real_scalar(delta, "delta")
    initial_intensity = _real_scalar(initial_intensity, "initial_intensity")
    initial_loss = _real_scalar(initial_loss, "initial_loss")
    strike = _real_scalar(strike, "strike")
    cap = _real_scalar(cap, "cap")
    if frequencies.ndim != 1 or frequencies.size < 3 or frequencies.size % 2 != 1:
        raise ValueError("an odd frequency count of at least three is required")
    dy = np.diff(frequencies)
    if (
        frequencies[0] != 0
        or np.any(dy <= 0)
        or not np.allclose(dy, dy[0], rtol=1e-12, atol=1e-14)
    ):
        raise ValueError("frequencies must be uniform and start at zero")
    if modes.shape != (grid.size, frequencies.size) or not np.all(np.isfinite(modes)):
        raise ValueError("invalid modes")
    if (
        delta <= 0
        or not 0 <= initial_intensity <= grid[-1]
        or initial_loss < 0
        or strike < 0
        or cap <= 0
    ):
        raise ValueError("invalid payoff or initial state")
    left = min(
        int(np.searchsorted(grid, initial_intensity, side="right") - 1), grid.size - 2
    )
    fraction = (initial_intensity - grid[left]) / (grid[left + 1] - grid[left])
    field = (1 - fraction) * modes[left] + fraction * modes[left + 1]
    eta = delta + 1j * frequencies
    transform = np.exp(-eta * strike) * (-np.expm1(-eta * cap)) / eta**2
    integrand = np.real(transform * field * np.exp(1j * frequencies * initial_loss))
    coefficients = np.ones(frequencies.size)
    coefficients[1:-1:2] = 4
    coefficients[2:-1:2] = 2
    value = (
        np.exp(delta * initial_loss)
        * dy[0]
        / (3 * np.pi)
        * np.dot(coefficients, integrand)
    )
    if not np.isfinite(value):
        raise ValueError("inversion is non-finite")
    return float(value)

import numpy as np


def price_calibrated_claim(
    probabilities: np.ndarray,
    shapes: np.ndarray,
    rates: np.ndarray,
    market: np.ndarray,
    swap: np.ndarray,
    contract: np.ndarray,
    resolution: np.ndarray,
) -> float:
    market = _checked_market(market)
    swap = _real_array(swap, "swap")
    contract = _real_array(contract, "contract")
    resolution = _real_array(resolution, "resolution")
    if swap.shape != (4,) or contract.shape != (4,) or resolution.shape != (7,):
        raise ValueError("swap, contract, and resolution have invalid shapes")
    if contract[0] <= 0 or contract[1] < 0 or contract[2] < 0 or contract[3] <= 0:
        raise ValueError("invalid claim contract")
    maximum, intervals, n_steps, order, delta, ymax, freq_intervals = resolution
    counts = (intervals, n_steps, order, freq_intervals)
    if any(value != int(value) or value < 1 for value in counts):
        raise ValueError("resolution counts must be positive integers")
    intervals, n_steps, order, freq_intervals = map(int, counts)
    if intervals < 2 or freq_intervals < 2 or freq_intervals % 2 or order > 128:
        raise ValueError("unsupported grid resolution")
    if maximum <= 0 or ymax <= 0 or delta <= 0 or max(market[1:3]) > maximum:
        raise ValueError("invalid contour or intensity domain")
    theta = calibrate_mark_tilt(
        probabilities, shapes, rates, market, swap[0], swap[1], swap[2:]
    )
    mixture = tilt_mark_distribution(probabilities, shapes, rates, theta)
    if delta >= np.min(mixture[:, 2]):
        raise ValueError("contour exceeds tilted mark moment domain")
    rule = build_mark_quadrature(mixture, order)
    grid = np.linspace(0, maximum, intervals + 1)
    frequencies = np.linspace(0, ymax, freq_intervals + 1)
    dt = contract[0] / n_steps
    bands = build_backward_drift(grid, market, dt)
    transfer = build_jump_transfer(grid, rule, market[3], frequencies, delta)
    modes = evolve_modal_values(grid, bands, transfer, dt, n_steps)
    return invert_capped_payoff(
        grid,
        modes,
        frequencies,
        delta,
        market[2],
        contract[1],
        contract[2],
        contract[3],
    )
SCICODE_GOLD_EOF
