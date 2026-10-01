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

def compute_link_free_energy(x_bar: "np.ndarray", y_bar: float, n_segments: int, bond_energy: float) -> "np.ndarray":
    """Reference implementation using stable log-space radial quadrature."""
    import numpy as np

    x = np.asarray(x_bar, dtype=float)
    if (
        isinstance(n_segments, bool)
        or not isinstance(n_segments, (int, np.integer))
        or n_segments < 2
        or not bond_energy > 0.0
        or not y_bar >= 0.0
        or x.ndim != 1
        or x.size == 0
        or not np.all(x > 0.0)
    ):
        raise ValueError(
            "need integer n_segments >= 2, bond_energy > 0, "
            "y_bar >= 0 and positive 1-D x_bar"
        )

    m = int(n_segments) - 1
    y = float(y_bar)
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(16)
    drops = np.array([
        0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0,
        14.0, 19.0, 25.0, 32.0, 40.0, 50.0
    ])

    def _rigid_energy(r):
        gap = (m - r) / m
        e = r / m
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            force = e * (3.0 - e * e) / (gap * (2.0 - gap))
            small = force < 1e-4
            safe = np.where(small, 1.0, force)
            h = np.where(
                small,
                1.0 + force * force / 6.0,
                np.log(2.0 * safe)
                + 2.0 * safe / np.expm1(2.0 * safe)
                - np.log1p(-np.exp(-2.0 * safe)),
            )
        return np.where(gap > 0.0, m * h, np.inf)

    def _log_weight(r):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                np.where(
                    r > 0.0,
                    np.log(np.where(r > 0.0, r, 1.0)),
                    -np.inf,
                )
                - _rigid_energy(r)
            )

    probe = np.linspace(0.0, m, 801)[1:-1]
    r_mode = probe[int(np.argmax(_log_weight(probe)))]

    def _log_integral(lo, hi):
        peak_r = np.clip(r_mode, lo, hi)
        target = _log_weight(peak_r)[:, None] - drops[None, :]
        up_in = np.repeat(peak_r[:, None], drops.size, 1)
        up_out = np.repeat(hi[:, None], drops.size, 1)
        down_out = np.repeat(lo[:, None], drops.size, 1)
        down_in = np.repeat(peak_r[:, None], drops.size, 1)

        for _ in range(14):
            mid = 0.5 * (up_in + up_out)
            keep = _log_weight(mid) >= target
            up_in, up_out = (
                np.where(keep, mid, up_in),
                np.where(keep, up_out, mid),
            )
            mid = 0.5 * (down_out + down_in)
            keep = _log_weight(mid) >= target
            down_out, down_in = (
                np.where(keep, down_out, mid),
                np.where(keep, mid, down_in),
            )

        edges = np.sort(
            np.concatenate(
                [lo[:, None], down_in, peak_r[:, None], up_in, hi[:, None]],
                axis=1,
            ),
            axis=1,
        )
        left, right = edges[:, :-1], edges[:, 1:]
        nodes = (
            0.5 * (left + right)[..., None]
            + 0.5 * (right - left)[..., None] * nodes_1d
        )
        weights = 0.5 * (right - left)[..., None] * weights_1d
        values = _log_weight(nodes)
        top = np.max(
            np.where(weights > 0.0, values, -np.inf),
            axis=(1, 2),
        )
        with np.errstate(invalid="ignore", over="ignore"):
            total = np.sum(
                np.where(
                    weights > 0.0,
                    weights * np.exp(values - top[:, None, None]),
                    0.0,
                ),
                axis=(1, 2),
            )
        return top + np.log(total)

    bond = bond_energy * (x ** -12.0 - 2.0 * x ** -6.0)
    out = np.full(x.shape, np.inf)

    if y == 0.0:
        reach = x < m
        out[reach] = (
            bond[reach]
            - 2.0 * np.log(x[reach])
            - np.log(2.0)
            + _rigid_energy(x[reach])
        )
        return out

    reach = np.abs(y - x) < m
    if np.any(reach):
        xs = x[reach]
        lo = np.abs(y - xs)
        hi = np.minimum(xs + y, m)
        collapsed = hi <= lo
        log_angular = np.empty_like(xs)

        if np.any(~collapsed):
            log_angular[~collapsed] = (
                _log_integral(lo[~collapsed], hi[~collapsed])
                - np.log(xs[~collapsed] * y)
            )

        if np.any(collapsed):
            log_angular[collapsed] = (
                np.log(2.0) - _rigid_energy(xs[collapsed])
            )

        out[reach] = bond[reach] - 2.0 * np.log(xs) - log_angular

    return out

import numpy as np

def locate_free_energy_extrema(y_bar: float, n_segments: int, bond_energy: float) -> "np.ndarray":
    """Locate stationary points using Richardson-extrapolated slopes."""
    import numpy as np
    from scipy.optimize import brentq, minimize_scalar

    if (
        isinstance(n_segments, bool)
        or not isinstance(n_segments, (int, np.integer))
        or n_segments < 10
    ):
        raise ValueError("n_segments must be an integer of at least 10")
    if not 0.0 <= y_bar < n_segments + 1:
        raise ValueError("y_bar must lie in [0, n_segments + 1)")
    if not 20.0 <= bond_energy <= 150.0:
        raise ValueError("bond_energy must lie in [20, 150]")

    m = int(n_segments) - 1
    y = float(y_bar)

    def _profile(x):
        return compute_link_free_energy(
            np.atleast_1d(np.asarray(x, dtype=float)),
            y,
            n_segments,
            bond_energy,
        )

    def _slope(x):
        def _difference(step):
            values = _profile(np.array([
                x - 2.0 * step,
                x - step,
                x + step,
                x + 2.0 * step,
            ]))
            return float(
                (values[0] - 8.0 * values[1]
                 + 8.0 * values[2] - values[3])
                / (12.0 * step)
            )

        step = 1e-3 * x
        return (
            16.0 * _difference(step / 2.0) - _difference(step)
        ) / 15.0

    def _root(lo, hi):
        while _slope(lo) * _slope(hi) > 0.0:
            lo, hi = (
                lo - 0.25 * (hi - lo),
                hi + 0.25 * (hi - lo),
            )
        return brentq(_slope, lo, hi, xtol=1e-12)

    x_low = max(0.8, y - m + 0.01)
    grid = np.concatenate([
        np.linspace(x_low, 3.0, 161),
        np.geomspace(3.0, y + m - 0.01, 81)[1:],
    ])
    rise = np.diff(_profile(grid)) > 0.0
    bonded = grid[1:-1] <= 3.0
    minima = np.where(~rise[:-1] & rise[1:])[0] + 1
    maxima = np.where(rise[:-1] & ~rise[1:])[0] + 1
    first_min = minima[bonded[minima - 1]] if minima.size else minima

    result = np.full(3, np.nan)
    k = None
    x_intact = None
    x_transition = None

    if first_min.size:
        i = first_min[0]
        x_intact = _root(grid[i - 1], grid[i + 1])
        following_maxima = maxima[maxima > i]
        if following_maxima.size:
            k = following_maxima[0]
            x_transition = _root(grid[k - 1], grid[k + 1])
    else:
        cells = np.where(grid[1:] <= 3.0)[0]
        slopes = np.diff(_profile(grid))[cells] / np.diff(grid)[cells]
        j = cells[int(np.argmax(slopes))]
        lo, hi = grid[max(j - 1, 0)], grid[j + 2]
        best = minimize_scalar(
            lambda x: -_slope(x),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-11},
        )
        if -best.fun > 0.0:
            x_intact = brentq(_slope, lo, best.x, xtol=1e-12)
            x_transition = brentq(_slope, best.x, hi, xtol=1e-12)
            k = int(np.searchsorted(grid, x_transition))

    if k is not None:
        later = minima[minima > k]
        if later.size:
            x_broken = _root(
                grid[later[0] - 1],
                grid[later[0] + 1],
            )
            result = np.array([x_intact, x_transition, x_broken])

    return result

import numpy as np
def compute_intact_chain_tension(y_bar: float, x_bar: float, n_segments: int) -> float:
    """Reference implementation (Richardson-extrapolated five-point derivative)."""
    import numpy as np

    if not (y_bar >= 0.0 and x_bar > 0.0):
        raise ValueError("y_bar must be nonnegative and x_bar must be positive")
    if isinstance(n_segments, bool) or not isinstance(n_segments, (int, np.integer)) or n_segments < 2:
        raise ValueError("n_segments must be an integer of at least 2")
    if not abs(y_bar - x_bar) < n_segments - 1.01:
        raise ValueError("the rigid segments must be able to close the chain")
    if y_bar == 0.0:
        return 0.0
    # The bond term does not depend on y_bar, so any positive well depth gives the same derivative.
    step = min(1e-3, 0.2 * y_bar)
    offsets = np.array([-2.0, -1.0, 1.0, 2.0])

    def _derivative(h):
        values = np.array([
            compute_link_free_energy(np.array([float(x_bar)]), float(y_bar + d * h), n_segments, 1.0)[0]
            for d in offsets
        ])
        return (values[0] - 8.0 * values[1] + 8.0 * values[2] - values[3]) / (12.0 * h)

    fine, coarse = _derivative(step), _derivative(2.0 * step)
    return float((16.0 * fine - coarse) / 15.0)

import numpy as np
def infer_segment_count_and_bond_energy(y_bar_hold: float, nu_tau_intact: float, nu_tau_broken: float, n_min: int, n_max: int) -> "np.ndarray":
    """Reference implementation (bond energy by root finding, segment count by bisection on the mismatch)."""
    import numpy as np
    from scipy.optimize import brentq

    if not (nu_tau_intact > 0.0 and nu_tau_broken > 0.0):
        raise ValueError("dwell times must be positive")
    for value in (n_min, n_max):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("segment-count bounds must be integers")
    if n_min < 10 or n_max < n_min or n_max > 10000 or n_max - n_min > 100:
        raise ValueError("need 10 <= n_min <= n_max <= 10000 and at most 101 candidates")

    def _barriers(n, depth):
        points = locate_free_energy_extrema(y_bar_hold, n, depth)
        if np.isnan(points[0]):
            return -np.inf, np.inf
        energy = compute_link_free_energy(points, y_bar_hold, n, depth)
        return energy[1] - energy[0], energy[1] - energy[2]

    cache = {}

    def _fit(n):
        # Returns the fitted depth and the signed log mismatch of the broken dwell time.
        if n not in cache:
            target = np.log(n * nu_tau_intact)
            depth = brentq(lambda d: _barriers(n, d)[0] - target, 20.0, 150.0, xtol=1e-11)
            cache[n] = (depth, _barriers(n, depth)[1] - np.log(nu_tau_broken))
        return cache[n]

    # Compare every admitted integer; the mismatch is not globally monotone.
    best = min(range(int(n_min), int(n_max) + 1), key=lambda n: (abs(_fit(n)[1]), n))
    return np.array([float(best), _fit(best)[0]])

import numpy as np
def integrate_intact_probability(y_grid: "np.ndarray", scission_barrier: "np.ndarray", healing_barrier: "np.ndarray", n_segments: int, loading_rate: float) -> "np.ndarray":
    """Reference implementation (exponential update with a linearly moving equilibrium)."""
    import numpy as np

    y = np.asarray(y_grid, dtype=float)
    e_s = np.asarray(scission_barrier, dtype=float)
    e_h = np.asarray(healing_barrier, dtype=float)
    if y.ndim != 1 or y.size < 2 or not np.all(np.diff(y) > 0.0):
        raise ValueError("y_grid must be strictly increasing with at least two nodes")
    if e_s.shape != y.shape or e_h.shape != y.shape:
        raise ValueError("barrier arrays must match y_grid")
    if isinstance(n_segments, bool) or not isinstance(n_segments, (int, np.integer)) or n_segments < 1:
        raise ValueError("n_segments must be a positive integer")
    if not loading_rate > 0.0:
        raise ValueError("loading_rate must be positive")
    k_s = n_segments * np.exp(-e_s)
    k_h = np.exp(-e_h)
    p_eq = k_h / (k_s + k_h)
    width = np.diff(y)
    decay_length = (np.sqrt(k_s[:-1] * k_s[1:]) + np.sqrt(k_h[:-1] * k_h[1:])) * width / loading_rate
    survive = np.exp(-decay_length)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Averaging weight of a linearly moving target, (1 - exp(-L)) / L, with its small-L limit.
        lag = np.where(decay_length > 1e-8, -np.expm1(-decay_length) / decay_length, 1.0 - 0.5 * decay_length)
    p = np.empty_like(y)
    p[0] = 1.0
    for i in range(width.size):
        p[i + 1] = p_eq[i + 1] + (p[i] - p_eq[i]) * survive[i] - (p_eq[i + 1] - p_eq[i]) * lag[i]
    return p

import numpy as np
def compute_final_scission_statistics(y_grid: "np.ndarray", intact_probability: "np.ndarray", scission_barrier: "np.ndarray", healing_barrier: "np.ndarray", tension: "np.ndarray", n_segments: int, loading_rate: float) -> "np.ndarray":
    """Reference implementation (trapezoidal survival exposure and moments)."""
    import numpy as np

    y = np.asarray(y_grid, dtype=float)
    arrays = [np.asarray(a, dtype=float) for a in (intact_probability, scission_barrier, healing_barrier, tension)]
    if y.ndim != 1 or y.size < 2 or not np.all(np.diff(y) > 0.0):
        raise ValueError("y_grid must be strictly increasing with at least two nodes")
    if any(a.shape != y.shape for a in arrays):
        raise ValueError("all arrays must match y_grid")
    if isinstance(n_segments, bool) or not isinstance(n_segments, (int, np.integer)) or n_segments < 1:
        raise ValueError("n_segments must be a positive integer")
    if not loading_rate > 0.0:
        raise ValueError("loading_rate must be positive")
    p, e_s, e_h, force = arrays
    width = np.diff(y)

    def _trapezoid(values):
        return float(np.sum(0.5 * (values[:-1] + values[1:]) * width))

    scission_density = p * n_segments * np.exp(-e_s) / loading_rate
    healing_density = np.exp(-e_h) / loading_rate
    # Re-formation exposure from each node to the end of the grid.
    pieces = 0.5 * (healing_density[:-1] + healing_density[1:]) * width
    exposure = np.concatenate([np.cumsum(pieces[::-1])[::-1], [0.0]])
    final_density = scission_density * np.exp(-exposure)
    tail = p[-1]
    return np.array([
        _trapezoid(force * final_density) + tail * force[-1],
        _trapezoid(y * final_density) + tail * y[-1],
        _trapezoid(final_density) + tail,
        _trapezoid(scission_density) + tail,
    ])

import numpy as np
def predict_mean_final_rupture_force(hold_extension_nm: float, intact_dwell_s: float, broken_dwell_s: float, pulling_speed_nm_s: float, kuhn_length_nm: float, temperature_k: float, attempt_frequency_hz: float, n_min: int, n_max: int) -> float:
    """Reference implementation (end-to-end chain of the earlier oracles)."""
    import numpy as np
    from scipy.interpolate import CubicSpline

    if not (pulling_speed_nm_s > 0.0 and kuhn_length_nm > 0.0 and temperature_k > 0.0 and attempt_frequency_hz > 0.0):
        raise ValueError("speed, Kuhn length, temperature and attempt frequency must be positive")
    y_hold = hold_extension_nm / kuhn_length_nm
    rate = pulling_speed_nm_s / (attempt_frequency_hz * kuhn_length_nm)
    estimate = infer_segment_count_and_bond_energy(
        y_hold, attempt_frequency_hz * intact_dwell_s, attempt_frequency_hz * broken_dwell_s, n_min, n_max)
    n_segments, depth = int(round(estimate[0])), float(estimate[1])

    nodes, scission, healing, tension = [], [], [], []
    node = 0
    while True:
        y = 0.5 * node
        points = locate_free_energy_extrema(y, n_segments, depth)
        if np.isnan(points[0]):
            break
        energy = compute_link_free_energy(points, y, n_segments, depth)
        nodes.append(y)
        scission.append(energy[1] - energy[0])
        healing.append(energy[1] - energy[2])
        tension.append(compute_intact_chain_tension(y, points[0], n_segments))
        node += 1

    nodes = np.array(nodes)
    grid = 0.001 * np.arange(int(round(nodes[-1] / 0.001)) + 1)
    scission_f = CubicSpline(nodes, scission)(grid)
    healing_f = CubicSpline(nodes, healing)(grid)
    tension_f = CubicSpline(nodes, tension)(grid)
    intact = integrate_intact_probability(grid, scission_f, healing_f, n_segments, rate)
    statistics = compute_final_scission_statistics(grid, intact, scission_f, healing_f, tension_f, n_segments, rate)
    force_unit_pn = 1.380649e-23 * temperature_k / (kuhn_length_nm * 1e-9) * 1e12
    return float(statistics[0] * force_unit_pn)
SCICODE_GOLD_EOF
