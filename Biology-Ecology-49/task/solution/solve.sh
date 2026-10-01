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


def _check_run(run):
    arr = np.asarray(run, dtype=float)
    if arr.shape != (6,) or not np.all(np.isfinite(arr)):
        raise ValueError("run must be a shape (6,) array of finite values")
    season, peak, up, down, cv, pull = arr
    if season <= 0.0 or peak <= 0.0 or cv <= 0.0 or pull <= 0.0:
        raise ValueError("season, peak, cv and pull must be positive")
    if up <= 1.0 or down <= 1.0:
        raise ValueError("shape_up and shape_down must be greater than 1")
    return season, peak, up, down, cv, pull


def _count_scale(run):
    """Amplitude p0 with mean(t) = p0 * th**up * (1-th)**down."""
    season, peak, up, down, cv, pull = _check_run(run)
    th_pk = up / (up + down)
    return peak / (th_pk ** up * (1.0 - th_pk) ** down)


def _count_mean(times, run):
    season, peak, up, down, cv, pull = _check_run(run)
    p0 = _count_scale(run)
    th = np.clip(np.asarray(times, dtype=float) / season, 0.0, 1.0)
    inside = (np.asarray(times, dtype=float) > 0.0) & (np.asarray(times, dtype=float) < season)
    return np.where(inside, p0 * th ** up * (1.0 - th) ** down, 0.0)


def migration_rate_coefficients(times: "np.ndarray", run: "np.ndarray") -> "np.ndarray":
    season, peak, up, down, cv, pull = _check_run(run)
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size == 0 or not np.all(np.isfinite(t)):
        raise ValueError("times must be a non-empty one-dimensional array of finite values")
    if np.any(t <= 0.0) or np.any(t >= season):
        raise ValueError("times must lie strictly inside the season")
    p0 = _count_scale(run)
    q0 = (cv * p0) ** 2
    q_up, q_down = 2.0 * up, 2.0 * down
    th = t / season
    one = 1.0 - th
    # mean(t) = p0 th^up (1-th)^down solves  d mean/dt = a - pull*mean/(season-t)
    a = (p0 / season) * th ** (up - 1.0) * one ** (down - 1.0) * ((pull - up - down) * th + up)
    # var(t) = q0 th^q_up (1-th)^q_down solves  d var/dt = -2 pull var/(season-t) + D*mean
    d = (q0 / (p0 * season)) * th ** (q_up - up - 1.0) * one ** (q_down - down - 1.0) \
        * ((2.0 * pull - q_up - q_down) * th + q_up)
    return np.vstack([a, d])

import numpy as np


def _check_reach(reach):
    arr = np.asarray(reach, dtype=float)
    if arr.shape != (5,) or not np.all(np.isfinite(arr)):
        raise ValueError("reach must be a shape (5,) array of finite values")
    length, flow, ground, decay, noise = arr
    if length <= 0.0 or flow <= 0.0 or ground <= 0.0 or decay <= 0.0:
        raise ValueError("reach length, flow speed, ground speed and decay rate must be positive")
    if noise < 0.0:
        raise ValueError("noise intensity must not be negative")
    return length, flow, ground, decay, noise


def _gauss(lo, hi, n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    return mid + half * nodes, half * weights


def _panel_nodes(lo, hi, breaks, n):
    edges = [lo] + sorted(b for b in breaks if lo < b < hi) + [hi]
    xs, ws = [], []
    for k in range(len(edges) - 1):
        a, b = edges[k], edges[k + 1]
        if b <= a:
            continue
        x, w = _gauss(a, b, n)
        xs.append(x)
        ws.append(w)
    if not xs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(xs), np.concatenate(ws)


def mean_sampled_concentration(day: float, site_start: float, window_length: float,
                                       reach: "np.ndarray", run: "np.ndarray",
                                       shedding_rate: float) -> float:
    length, flow, ground, decay, noise = _check_reach(reach)
    season = _check_run(run)[0]
    d = float(day)
    if not np.isfinite(d) or d <= 0.0 or d > season:
        raise ValueError("day must be finite and inside the season")
    x0, wd = float(site_start), float(window_length)
    if not np.isfinite(x0) or not np.isfinite(wd) or x0 < 0.0 or wd <= 0.0 or x0 + wd > length:
        raise ValueError("the sampled window must lie inside the reach")
    g = float(shedding_rate)
    if not np.isfinite(g) or g <= 0.0:
        raise ValueError("shedding_rate must be finite and positive")

    slope = 1.0 + flow / ground
    xs, xw = _panel_nodes(x0, x0 + wd, [length - flow * d], 64)
    total = 0.0
    for x, w in zip(xs, xw):
        xi_max = min(d, (length - x) / flow)
        if xi_max <= 0.0:
            continue
        zero_at = (d - x / ground) / slope
        season_at = (d - x / ground - season) / slope
        xi, xiw = _panel_nodes(0.0, xi_max, [zero_at, season_at], 64)
        src = _count_mean(d - xi - (x + flow * xi) / ground, run)
        total += w * float(np.sum(xiw * np.exp(-decay * xi) * src))
    return float(g * total / wd)

import numpy as np


def sample_memory_horizon(site_start: float, reach: "np.ndarray") -> float:
    length, flow, ground, decay, noise = _check_reach(reach)
    x0 = float(site_start)
    if not np.isfinite(x0) or x0 < 0.0 or x0 >= length:
        raise ValueError("site_start must be finite, at least 0 and less than the reach length")
    return float(length / flow + length / ground - x0 / flow)

import numpy as np


def washout_exponent(backward_times: "np.ndarray", assayed_volume: float,
                             window_length: float, reach: "np.ndarray") -> "np.ndarray":
    length, flow, ground, decay, noise = _check_reach(reach)
    s = np.asarray(backward_times, dtype=float)
    if s.ndim != 1 or s.size == 0 or not np.all(np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("backward_times must be a non-empty 1-D array of finite nonnegative values")
    vol, wd = float(assayed_volume), float(window_length)
    if not np.isfinite(vol) or vol <= 0.0 or not np.isfinite(wd) or wd <= 0.0:
        raise ValueError("assayed_volume and window_length must be finite and positive")
    lam0 = vol / wd
    if noise == 0.0:
        return -lam0 * np.exp(-decay * s)
    half = noise * noise / (2.0 * decay)
    return 1.0 / (np.exp(decay * s) * (-1.0 / lam0 - half) + half)

import numpy as np


def sample_weight_kernel(backward_times: "np.ndarray", site_starts: "np.ndarray",
                                 assayed_volumes: "np.ndarray", window_length: float,
                                 reach: "np.ndarray", shedding_rate: float) -> "np.ndarray":
    length, flow, ground, decay, noise = _check_reach(reach)
    s = np.asarray(backward_times, dtype=float)
    if s.ndim != 1 or s.size == 0 or not np.all(np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("backward_times must be a non-empty 1-D array of finite nonnegative values")
    starts = np.asarray(site_starts, dtype=float)
    vols = np.asarray(assayed_volumes, dtype=float)
    if starts.ndim != 1 or vols.ndim != 1 or starts.size == 0 or starts.size != vols.size:
        raise ValueError("site_starts and assayed_volumes must be 1-D arrays of the same non-zero length")
    wd, g = float(window_length), float(shedding_rate)
    if not np.isfinite(wd) or wd <= 0.0 or not np.isfinite(g) or g <= 0.0:
        raise ValueError("window_length and shedding_rate must be finite and positive")
    if not np.all(np.isfinite(starts)) or np.any(starts < 0.0) or np.any(starts + wd > length):
        raise ValueError("every sampled window must lie inside the reach")
    if not np.all(np.isfinite(vols)) or np.any(vols <= 0.0):
        raise ValueError("every assayed volume must be finite and positive")

    nodes, weights = np.polynomial.legendre.leggauss(64)
    slope = 1.0 + flow / ground
    out = np.zeros(s.shape)
    for start, vol in zip(starts, vols):
        horizon = sample_memory_horizon(float(start), reach)
        lo_x = (start + flow * s) / slope
        hi_x = np.minimum(np.minimum((start + wd + flow * s) / slope, length), ground * s)
        live = (hi_x > lo_x) & (s <= horizon) & (s > 0.0)
        for j in np.nonzero(live)[0]:
            xi_hi = s[j] - lo_x[j] / ground
            xi_lo = s[j] - hi_x[j] / ground
            mid, half = 0.5 * (xi_lo + xi_hi), 0.5 * (xi_hi - xi_lo)
            xi = mid + half * nodes
            r_vals = washout_exponent(xi, float(vol), wd, reach)
            out[j] += g * ground * half * float(np.sum(weights * r_vals))
    return out

import numpy as np


def _simpson_weights(n_intervals, step):
    w = np.ones(n_intervals + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w * step / 3.0


def nondetection_probability(day: float, site_starts: "np.ndarray",
                                     assayed_volumes: "np.ndarray", window_length: float,
                                     reach: "np.ndarray", run: "np.ndarray",
                                     shedding_rate: float) -> float:
    season, peak, up, down, cv, pull = _check_run(run)
    d = float(day)
    if not np.isfinite(d) or d <= 0.0 or d > season:
        raise ValueError("day must be finite and inside the season")
    steps_per_day = 160
    t_end = min(season, d)
    n_full = 2 * int(np.ceil(steps_per_day * t_end / 2.0))
    step = t_end / n_full
    grid = np.arange(2 * n_full + 1) * (0.5 * step)
    kernel = sample_weight_kernel(d - grid, site_starts, assayed_volumes,
                                          window_length, reach, shedding_rate)
    coef = np.zeros((2, grid.size))
    inner = (grid > 0.0) & (grid < season)
    coef[:, inner] = migration_rate_coefficients(grid[inner], run)
    drift, diffusion = coef[0], coef[1]
    pull_rate = pull / np.maximum(season - grid, 1e-300)

    def _slope(index, value):
        return pull_rate[index] * value - 0.5 * diffusion[index] * value * value - kernel[index]

    beta = np.zeros(2 * n_full + 1)
    current = 0.0
    for k in range(2 * n_full, 0, -2):
        s1 = _slope(k, current)
        s2 = _slope(k - 1, current - 0.5 * step * s1)
        s3 = _slope(k - 1, current - 0.5 * step * s2)
        s4 = _slope(k - 2, current - step * s3)
        current = current - (step / 6.0) * (s1 + 2.0 * s2 + 2.0 * s3 + s4)
        beta[k - 2] = current
    quad = _simpson_weights(n_full, step)
    return float(np.exp(np.sum(quad * beta[0::2] * drift[0::2])))

import numpy as np


def survey_window_days(site_start: float, assayed_volume: float, window_length: float,
                               reach: "np.ndarray", run: "np.ndarray", shedding_rate: float,
                               failure_target: float, first_day: int, last_day: int) -> "np.ndarray":
    season = _check_run(run)[0]
    target = float(failure_target)
    if not np.isfinite(target) or target <= 0.0 or target >= 1.0:
        raise ValueError("failure_target must be finite, greater than 0 and less than 1")
    for value in (first_day, last_day):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("first_day and last_day must be integers")
    if first_day < 1 or last_day < first_day or last_day > season:
        raise ValueError("the search range must satisfy 1 <= first_day <= last_day <= season")
    starts = np.array([float(site_start)])
    volumes = np.array([float(assayed_volume)])
    good = []
    for day in range(int(first_day), int(last_day) + 1):
        value = nondetection_probability(float(day), starts, volumes, window_length,
                                                 reach, run, shedding_rate)
        if value <= target:
            good.append(float(day))
    if not good:
        raise ValueError("no whole day in the search range meets the failure target")
    return np.array([good[0], good[-1]])

import math

import numpy as np


def planned_replicates(day: float, site_start: float, window_length: float,
                               reach: "np.ndarray", run: "np.ndarray", shedding_rate: float,
                               capture_volume: float, copy_target: float) -> int:
    volume, target = float(capture_volume), float(copy_target)
    if not np.isfinite(volume) or volume <= 0.0 or not np.isfinite(target) or target <= 0.0:
        raise ValueError("capture_volume and copy_target must be finite and positive")
    concentration = mean_sampled_concentration(day, site_start, window_length,
                                                       reach, run, shedding_rate)
    if concentration <= 0.0:
        raise ValueError("the expected concentration of the sampled water is zero")
    return int(math.ceil(target / (volume * concentration)))

import numpy as np


def survey_failure_probability(reach: "np.ndarray", run: "np.ndarray",
                                       shedding_rate: float, window_length: float,
                                       station_start: float, station_replicates: int,
                                       upstream_start: float, capture_volume: float,
                                       failure_target: float, copy_target: float,
                                       first_day: int, last_day: int) -> float:
    if isinstance(station_replicates, bool) or not isinstance(station_replicates, (int, np.integer)) \
            or station_replicates < 1:
        raise ValueError("station_replicates must be an integer of at least 1")
    volume = float(capture_volume)
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("capture_volume must be finite and positive")
    station_volume = float(station_replicates) * volume
    window = survey_window_days(station_start, station_volume, window_length,
                                        reach, run, shedding_rate, failure_target,
                                        first_day, last_day)
    opening = float(window[0])
    upstream_replicates = planned_replicates(opening, upstream_start, window_length,
                                                     reach, run, shedding_rate, volume,
                                                     copy_target)
    starts = np.array([float(station_start), float(upstream_start)])
    volumes = np.array([station_volume, float(upstream_replicates) * volume])
    return nondetection_probability(opening, starts, volumes, window_length,
                                            reach, run, shedding_rate)
SCICODE_GOLD_EOF
