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


def _validated_endpoints(beta_low, beta_high):
    """Validate a pair of endpoints and return them with the crossover potential."""
    if any(not np.isrealobj(b) for b in (beta_low, beta_high)):
        raise ValueError("endpoints must be real")
    try:
        low = np.asarray(beta_low, dtype=float)
        high = np.asarray(beta_high, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("endpoints must be finite real arrays") from exc
    if low.shape != (3,) or high.shape != (3,):
        raise ValueError("each endpoint must have shape (3,)")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("endpoints must be finite")
    if not np.all(low > 0.0) or not np.all(high > 0.0):
        raise ValueError("endpoint values must be positive")
    if not np.all(low < high):
        raise ValueError("every low endpoint entry must lie below its high counterpart")
    (mu_l, n_l, p_l), (mu_h, n_h, p_h) = low, high
    denominator = mu_l * n_h - mu_h * n_l
    numerator = mu_l * mu_h * (mu_h * n_h - mu_l * n_l - 2.0 * (p_h - p_l))
    if denominator <= 0.0 or numerator <= 0.0:
        raise ValueError("the endpoints do not enclose a non-empty allowed region")
    crossover = np.sqrt(numerator / denominator)
    if not np.isfinite(crossover) or not mu_l < crossover < mu_h:
        raise ValueError("the crossover chemical potential must lie between the endpoints")
    return low, high, crossover


def allowed_density_bounds(chemical_potential: float, beta_low: "np.ndarray",
                                   beta_high: "np.ndarray") -> "np.ndarray":
    low, high, crossover = _validated_endpoints(beta_low, beta_high)
    if not np.isscalar(chemical_potential) or not np.isrealobj(chemical_potential):
        raise ValueError("the chemical potential must be a real scalar")
    mu = float(chemical_potential)
    if not np.isfinite(mu) or not low[0] <= mu <= high[0]:
        raise ValueError("the chemical potential must lie within the endpoint interval")
    (mu_l, n_l, p_l), (mu_h, n_h, p_h) = low, high
    gap = p_h - p_l
    if mu <= crossover:
        lower = n_l * mu / mu_l
    else:
        lower = (mu**3 * n_h - mu * mu_h * (mu_h * n_h - 2.0 * gap)) / ((mu**2 - mu_l**2) * mu_h)
    if mu < crossover:
        upper = (mu**3 * n_l - mu_l * mu * (mu_l * n_l + 2.0 * gap)) / ((mu**2 - mu_h**2) * mu_l)
    else:
        upper = n_h * mu / mu_h
    bounds = np.array([lower, upper], dtype=float)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("the evaluated density bounds must be finite")
    return bounds

import numpy as np


def allowed_pressure_bounds(chemical_potential: float, density: float,
                                    beta_low: "np.ndarray",
                                    beta_high: "np.ndarray") -> "np.ndarray":
    span = allowed_density_bounds(chemical_potential, beta_low, beta_high)
    low, high, _ = _validated_endpoints(beta_low, beta_high)
    if not np.isscalar(density) or not np.isrealobj(density):
        raise ValueError("the density must be a real scalar")
    n = float(density)
    if not np.isfinite(n) or not span[0] <= n <= span[1]:
        raise ValueError("the density must lie within its allowed bounds")
    mu = float(chemical_potential)
    (mu_l, n_l, p_l), (mu_h, n_h, p_h) = low, high
    stiff = (mu**2 - mu_l**2) / (2.0 * mu)
    crossover_density = allowed_density_bounds(mu_l, beta_low, beta_high)[1] * mu / mu_l
    lower = p_l + stiff * span[0]
    if n <= crossover_density:
        upper = p_l + stiff * n
    else:
        upper = p_h - (mu_h**2 - mu**2) / (2.0 * mu) * n
    bounds = np.array([lower, upper], dtype=float)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("the evaluated pressure bounds must be finite")
    return bounds

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def _marginal_density(mu, low, high, crossover):
    """Unnormalised marginal density of the uniform measure over the allowed region."""
    mu_l, mu_h = low[0], high[0]
    if mu < crossover:
        scale = (mu_h**2 - crossover**2) / (crossover**2 - mu_l**2)
        return mu * (mu**2 - mu_l**2) / (mu_h**2 - mu**2) * scale
    scale = (crossover**2 - mu_l**2) / (mu_h**2 - crossover**2)
    return mu * (mu_h**2 - mu**2) / (mu**2 - mu_l**2) * scale


def _marginal_mass(mu, low, high, crossover):
    """Cumulative mass of the marginal density from the low endpoint up to mu."""
    extra = (low, high, crossover)
    if mu <= crossover:
        return quad(_marginal_density, low[0], mu, args=extra, limit=200)[0]
    head = quad(_marginal_density, low[0], crossover, args=extra, limit=200)[0]
    return head + quad(_marginal_density, crossover, mu, args=extra, limit=200)[0]


def _quantile_shortfall(mu, low, high, crossover, target):
    """Cumulative mass up to mu, less the mass the requested quantile calls for."""
    return _marginal_mass(mu, low, high, crossover) - target


def chemical_potential_quantile(quantile: float, beta_low: "np.ndarray",
                                        beta_high: "np.ndarray") -> float:
    low, high, crossover = _validated_endpoints(beta_low, beta_high)
    if not np.isscalar(quantile) or not np.isrealobj(quantile):
        raise ValueError("the quantile must be a real scalar")
    q = float(quantile)
    if not np.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("the quantile must lie in [0, 1]")
    if q == 0.0:
        return float(low[0])
    if q == 1.0:
        return float(high[0])
    total = _marginal_mass(high[0], low, high, crossover)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("the marginal distribution must carry positive mass")
    target = q * total

    value = brentq(_quantile_shortfall, low[0], high[0],
                   args=(low, high, crossover, target), xtol=1e-12, rtol=1e-14, maxiter=200)
    if not np.isfinite(value):
        raise ValueError("the quantile inversion must return a finite value")
    return float(value)

import numpy as np


def quantile_point_in_volume(quantiles: "np.ndarray", beta_low: "np.ndarray",
                                     beta_high: "np.ndarray") -> "np.ndarray":
    if not np.isrealobj(quantiles):
        raise ValueError("the quantiles must be real")
    try:
        q = np.asarray(quantiles, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("the quantiles must be a finite real array") from exc
    if q.shape != (3,) or not np.all(np.isfinite(q)) or np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("the quantiles must be a shape (3,) array inside [0, 1]")
    mu = chemical_potential_quantile(q[0], beta_low, beta_high)
    span = allowed_density_bounds(mu, beta_low, beta_high)
    lo, hi = float(span[0]), float(span[1])
    if q[0] == 0.0 or q[0] == 1.0 or hi <= lo:
        n = 0.5 * (lo + hi)
    elif q[1] == 0.0:
        n = lo
    elif q[1] == 1.0:
        n = hi
    else:
        # The pressure width is a triangular density. Its mode is where the
        # two upper pressure edges meet; invert its normalized CDF exactly.
        low, high, _ = _validated_endpoints(beta_low, beta_high)
        mode = 2.0 * mu * (high[2] - low[2]) / ((high[0] - low[0]) * (high[0] + low[0]))
        mode = float(np.clip(mode, lo, hi))
        width = hi - lo
        if q[1] <= (mode - lo) / width:
            n = lo + np.sqrt(q[1] * width * (mode - lo))
        else:
            n = hi - np.sqrt((1.0 - q[1]) * width * (hi - mode))
    edges = allowed_pressure_bounds(mu, n, beta_low, beta_high)
    point = np.array([mu, n, edges[0] + q[2] * (edges[1] - edges[0])], dtype=float)
    if not np.all(np.isfinite(point)):
        raise ValueError("the selected state must be finite")
    return point

import numpy as np


def _refine_interval(beta_low, beta_high, depth, quantiles):
    """Return the interior states of one interval, ordered by increasing density."""
    if depth == 0:
        return []
    middle = quantile_point_in_volume(quantiles, beta_low, beta_high)
    left = _refine_interval(beta_low, middle, depth - 1, quantiles)
    right = _refine_interval(middle, beta_high, depth - 1, quantiles)
    return left + [middle] + right


def self_similar_refine(beta_low: "np.ndarray", beta_high: "np.ndarray", depth: int,
                                quantiles: "np.ndarray") -> "np.ndarray":
    low, high, _ = _validated_endpoints(beta_low, beta_high)
    if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)):
        raise ValueError("depth must be an integer")
    levels = int(depth)
    if not 0 <= levels <= 12:
        raise ValueError("depth must lie in [0, 12]")
    interior = _refine_interval(low, high, levels, quantiles)
    states = np.array([low] + [np.asarray(b, dtype=float) for b in interior] + [high],
                      dtype=float)
    if not np.all(np.isfinite(states)):
        raise ValueError("the constructed bridge must be finite")
    if not np.all(np.diff(states[:, 1]) > 0.0):
        raise ValueError("the bridge must be strictly increasing in density")
    return states

import numpy as np


def _validated_density_grid(density_grid):
    """Validate a density grid and return it as a float array."""
    if not np.isrealobj(density_grid):
        raise ValueError("the density grid must be real")
    try:
        grid = np.asarray(density_grid, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("the density grid must be a finite real array") from exc
    if grid.ndim != 1 or grid.size < 3:
        raise ValueError("the density grid must be one-dimensional with at least three points")
    if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
        raise ValueError("the density grid must be finite and strictly positive")
    if not np.all(np.diff(grid) > 0.0):
        raise ValueError("the density grid must be strictly increasing")
    return grid


def diffusion_profile(density_grid: "np.ndarray",
                              correlation_fraction: float) -> "np.ndarray":
    grid = _validated_density_grid(density_grid)
    if not np.isscalar(correlation_fraction) or not np.isrealobj(correlation_fraction):
        raise ValueError("the correlation fraction must be a real scalar")
    fraction = float(correlation_fraction)
    if not np.isfinite(fraction) or fraction <= 0.0:
        raise ValueError("the correlation fraction must be finite and positive")
    coefficient = grid**2 * fraction**2 / 4.0
    first = grid * fraction**2 / 2.0
    second = np.full_like(grid, fraction**2 / 2.0)
    profile = np.stack((coefficient, first, second - first / grid))
    if not np.all(np.isfinite(profile)):
        raise ValueError("the evaluated diffusion profile must be finite")
    return profile

import numpy as np


def diffuse_chemical_potential(density_grid: "np.ndarray",
                                       chemical_potential: "np.ndarray",
                                       correlation_fraction: float,
                                       n_tau_steps: int) -> "np.ndarray":
    profile = diffusion_profile(density_grid, correlation_fraction)
    grid = _validated_density_grid(density_grid)
    if not np.isrealobj(chemical_potential):
        raise ValueError("the chemical potential must be real")
    try:
        mu = np.array(chemical_potential, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("the chemical potential must be a finite real array") from exc
    if mu.shape != grid.shape or not np.all(np.isfinite(mu)):
        raise ValueError("the chemical potential must be finite and match the density grid")
    spacing = grid[1] - grid[0]
    if not np.allclose(np.diff(grid), spacing, rtol=0.0, atol=1e-12):
        raise ValueError("the density grid must be uniformly spaced")
    if isinstance(n_tau_steps, bool) or not isinstance(n_tau_steps, (int, np.integer)):
        raise ValueError("n_tau_steps must be an integer")
    steps = int(n_tau_steps)
    if steps < 1:
        raise ValueError("n_tau_steps must be at least 1")
    half = 0.5 * (profile[0][1:] + profile[0][:-1])
    step = 1.0 / steps
    accumulated = 0.0
    for _ in range(steps):
        flux = half * np.diff(mu) / spacing
        accumulated += step * (flux[-1] - flux[0])
        mu[1:-1] += step * np.diff(flux) / spacing
    result = np.concatenate(([accumulated], mu))
    if not np.all(np.isfinite(result)):
        raise ValueError("the diffused result must be finite")
    return result

import numpy as np


def sound_speed_squared(density_grid: "np.ndarray",
                                chemical_potential: "np.ndarray") -> "np.ndarray":
    grid = _validated_density_grid(density_grid)
    if not np.isrealobj(chemical_potential):
        raise ValueError("the chemical potential must be real")
    try:
        mu = np.asarray(chemical_potential, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("the chemical potential must be a finite real array") from exc
    if mu.shape != grid.shape or not np.all(np.isfinite(mu)) or np.any(mu <= 0.0):
        raise ValueError("the chemical potential must be finite, positive and match the grid")
    speeds = grid / mu * np.gradient(mu, grid, edge_order=2)
    if not np.all(np.isfinite(speeds)):
        raise ValueError("the evaluated sound speeds must be finite")
    return speeds

import numpy as np


def bridge_boundary_flux(beta_low: "np.ndarray", beta_high: "np.ndarray", depth: int,
                                 quantiles: "np.ndarray", correlation_fraction: float,
                                 n_grid_points: int, n_tau_steps: int) -> "np.ndarray":
    states = self_similar_refine(beta_low, beta_high, depth, quantiles)
    if isinstance(n_grid_points, bool) or not isinstance(n_grid_points, (int, np.integer)):
        raise ValueError("n_grid_points must be an integer")
    points = int(n_grid_points)
    if points < 3:
        raise ValueError("n_grid_points must be at least 3")
    grid = np.linspace(states[0, 1], states[-1, 1], points)
    initial = np.interp(grid, states[:, 1], states[:, 0])
    evolved = diffuse_chemical_potential(grid, initial, correlation_fraction,
                                                 n_tau_steps)
    speeds = sound_speed_squared(grid, evolved[1:])
    summary = np.array([evolved[0], float(np.max(speeds[1:-1]))], dtype=float)
    if not np.all(np.isfinite(summary)):
        raise ValueError("the construction summary must be finite")
    return summary
SCICODE_GOLD_EOF
