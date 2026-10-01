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


def restore_opes_bias(
    raw_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    events: 'np.ndarray',
    barriers: 'np.ndarray',
    dt: float,
    temperature: float,
) -> 'np.ndarray':
    try:
        raw = np.asarray(raw_bias)
        lens = np.asarray(lengths)
        ev = np.asarray(events)
        barr = np.asarray(barriers)
        step = float(dt)
        temp = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("archive inputs must be numeric") from exc
    if (
        raw.ndim != 3 or raw.shape[0] < 3 or raw.shape[1] == 0 or raw.shape[2] < 2
        or lens.shape != raw.shape[:2] or ev.shape != raw.shape[:2]
        or barr.shape != (raw.shape[0],)
    ):
        raise ValueError("incompatible archive shapes")
    if (
        not np.issubdtype(raw.dtype, np.number)
        or not np.issubdtype(lens.dtype, np.integer)
        or not np.issubdtype(ev.dtype, np.number)
        or not np.issubdtype(barr.dtype, np.number)
        or not np.isrealobj(raw) or not np.isrealobj(ev) or not np.isrealobj(barr)
    ):
        raise ValueError("archive arrays must be real numeric data")
    try:
        raw = raw.astype(float, copy=False)
        evf = ev.astype(float, copy=False)
        barr = barr.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("archive arrays must fit float64") from exc
    if (
        np.any(lens < 2) or np.any(lens > raw.shape[2])
        or np.any(~np.isfinite(evf)) or np.any((evf != 0.0) & (evf != 1.0))
        or np.any(~np.isfinite(barr)) or len(np.unique(barr)) != len(barr)
        or not np.isfinite(step) or step <= 0.0
        or not np.isfinite(temp) or temp <= 0.0
    ):
        raise ValueError("invalid lengths, events, barriers, dt, or temperature")
    active = np.arange(raw.shape[2])[None, None, :] < lens[:, :, None]
    if np.any(~np.isfinite(raw[active])):
        raise ValueError("active bias samples must be finite")
    restored = np.zeros(raw.shape, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = raw + barr[:, None, None]
    restored[active] = shifted[active]
    if np.any(~np.isfinite(restored)):
        raise ValueError("restored bias exceeded the float64 range")
    return restored

import numpy as np


def censored_observed_log_rates(
    lengths: 'np.ndarray',
    events: 'np.ndarray',
    dt: float,
) -> 'np.ndarray':
    try:
        lens = np.asarray(lengths)
        ev = np.asarray(events)
        step = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("lengths, events, and dt must be numeric") from exc
    if lens.ndim != 2 or lens.size == 0 or ev.shape != lens.shape:
        raise ValueError("lengths and events must be nonempty matching matrices")
    if not np.issubdtype(lens.dtype, np.integer) or not np.issubdtype(ev.dtype, np.number):
        raise ValueError("lengths must be integers and events numeric")
    ev = ev.astype(float, copy=False)
    if (
        np.any(lens <= 0) or np.any(~np.isfinite(ev))
        or np.any((ev != 0.0) & (ev != 1.0))
        or not np.isfinite(step) or step <= 0.0
    ):
        raise ValueError("invalid lengths, events, or dt")
    counts = np.sum(ev, axis=1)
    if np.any(counts <= 0.0):
        raise ValueError("every set must contain at least one transition")
    exposure = np.sum(lens.astype(float), axis=1) * step
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = np.log(counts / exposure)
    if np.any(~np.isfinite(result)):
        raise ValueError("observed log-rate is not finite")
    return result

import numpy as np


def _eatr_logmeanexp(values: 'np.ndarray') -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.mean(np.exp(values - maximum))))


def ensemble_time_log_acceleration(
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    temperature: float,
    gamma: float,
) -> 'np.ndarray':
    try:
        bias = np.asarray(restored_bias)
        lens = np.asarray(lengths)
        temp = float(temperature)
        quality = float(gamma)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("acceleration inputs must be numeric") from exc
    if bias.ndim != 3 or lens.shape != bias.shape[:2] or bias.shape[0] == 0:
        raise ValueError("incompatible bias and length shapes")
    if (
        not np.issubdtype(bias.dtype, np.number) or not np.isrealobj(bias)
        or not np.issubdtype(lens.dtype, np.integer)
    ):
        raise ValueError("bias must be real numeric and lengths integral")
    bias = bias.astype(float, copy=False)
    if (
        np.any(lens < 1) or np.any(lens > bias.shape[2])
        or not np.isfinite(temp) or temp <= 0.0
        or not np.isfinite(quality) or not 0.0 <= quality <= 1.0
    ):
        raise ValueError("invalid lengths, temperature, or gamma")
    beta = 1.0 / (0.008314462618 * temp)
    output = np.empty(bias.shape[0], dtype=float)
    for set_index in range(bias.shape[0]):
        frame_logs = []
        for frame in range(int(np.max(lens[set_index]))):
            running = lens[set_index] > frame
            active = bias[set_index, running, frame]
            if np.any(~np.isfinite(active)):
                raise ValueError("active restored bias samples must be finite")
            frame_logs.append(_eatr_logmeanexp(beta * quality * active))
        output[set_index] = _eatr_logmeanexp(np.asarray(frame_logs))
    if np.any(~np.isfinite(output)):
        raise ValueError("log acceleration is not finite")
    return output

import numpy as np
from scipy.optimize import minimize_scalar


def _eatr_prefix_variance(
    gamma: float,
    logs: 'np.ndarray',
    bias: 'np.ndarray',
    lens: 'np.ndarray',
    temp: float,
    chosen: 'np.ndarray',
) -> float:
    acceleration = ensemble_time_log_acceleration(
        bias[chosen], lens[chosen], temp, float(gamma)
    )
    corrected = logs[chosen] - acceleration
    return float(np.var(corrected))


def fit_eatr_flooding(
    log_k_observed: 'np.ndarray',
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    temperature: float,
    indices: 'np.ndarray',
    gamma_min: float = 0.0,
    gamma_max: float = 1.0,
) -> 'np.ndarray':
    try:
        logs = np.asarray(log_k_observed)
        bias = np.asarray(restored_bias)
        lens = np.asarray(lengths)
        chosen = np.asarray(indices)
        lower = float(gamma_min)
        upper = float(gamma_max)
        temp = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("fit inputs must be numeric") from exc
    if (
        bias.ndim != 3 or logs.shape != (bias.shape[0],)
        or lens.shape != bias.shape[:2] or chosen.ndim != 1 or chosen.size < 3
        or not np.issubdtype(chosen.dtype, np.integer)
    ):
        raise ValueError("incompatible fit shapes or selection")
    if (
        not np.issubdtype(logs.dtype, np.number) or not np.isrealobj(logs)
        or np.any(~np.isfinite(logs.astype(float, copy=False)))
        or np.any(chosen < 0) or np.any(chosen >= bias.shape[0])
        or len(np.unique(chosen)) != len(chosen)
        or not np.isfinite(temp) or temp <= 0.0
        or not np.isfinite(lower) or not np.isfinite(upper)
        or not 0.0 <= lower < upper <= 1.0
    ):
        raise ValueError("invalid fit data, indices, temperature, or bounds")
    logs = logs.astype(float, copy=False)
    optimum = minimize_scalar(
        _eatr_prefix_variance,
        args=(logs, bias, lens, temp, chosen),
        bounds=(lower, upper),
        options={'xatol': 1e-12, 'maxiter': 10000},
        method="bounded",
    )
    gamma = float(optimum.x)
    corrected = logs[chosen] - ensemble_time_log_acceleration(
        bias[chosen], lens[chosen], temp, gamma
    )
    result = np.array([gamma, np.mean(corrected), np.var(corrected)], dtype=float)
    if not bool(optimum.success) or np.any(~np.isfinite(result)):
        raise ValueError("bounded flooding fit did not produce a finite optimum")
    return result

import numpy as np


def scan_eatr_prefixes(
    log_k_observed: 'np.ndarray',
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    temperature: float,
    order: 'np.ndarray',
    min_sets: int = 3,
) -> 'np.ndarray':
    try:
        bias = np.asarray(restored_bias)
        permutation = np.asarray(order)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("prefix inputs must be arrays") from exc
    if bias.ndim != 3 or permutation.shape != (bias.shape[0],):
        raise ValueError("order must cover every set")
    if not np.issubdtype(permutation.dtype, np.integer):
        raise ValueError("order must contain integer indices")
    if sorted(permutation.tolist()) != list(range(bias.shape[0])):
        raise ValueError("order must be a permutation of the set indices")
    if not isinstance(min_sets, (int, np.integer)) or isinstance(min_sets, (bool, np.bool_)):
        raise ValueError("min_sets must be an integer")
    if not 3 <= int(min_sets) <= bias.shape[0]:
        raise ValueError("min_sets must lie between three and the set count")
    rows = []
    for count in range(int(min_sets), bias.shape[0] + 1):
        fitted = fit_eatr_flooding(
            log_k_observed, bias, lengths, temperature, permutation[:count]
        )
        rows.append([float(count), fitted[0], fitted[1], fitted[2]])
    result = np.asarray(rows, dtype=float)
    if np.any(~np.isfinite(result)):
        raise ValueError("prefix scan produced a nonfinite result")
    return result

import numpy as np


def select_consistent_prefix(
    prefix_table: 'np.ndarray',
    variance_limit: float,
    gamma_margin: float = 0.02,
) -> 'np.ndarray':
    try:
        table = np.asarray(prefix_table)
        limit = float(variance_limit)
        margin = float(gamma_margin)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("prefix diagnostics must be numeric") from exc
    if (
        table.ndim != 2 or table.shape[1] != 4 or table.shape[0] == 0
        or not np.issubdtype(table.dtype, np.number) or not np.isrealobj(table)
    ):
        raise ValueError("prefix_table must have shape (n,4)")
    table = table.astype(float, copy=False)
    if (
        np.any(~np.isfinite(table)) or np.any(table[:, 3] < 0.0)
        or not np.isfinite(limit) or limit < 0.0
        or not np.isfinite(margin) or not 0.0 <= margin < 0.5
    ):
        raise ValueError("invalid prefix values or selection controls")
    counts = table[:, 0]
    if np.any(counts != np.rint(counts)) or np.any(np.diff(counts) != 1.0) or counts[0] < 3.0:
        raise ValueError("prefix counts must be consecutive integers from at least three")
    eligible = (
        (table[:, 3] <= limit)
        & (table[:, 1] >= margin)
        & (table[:, 1] <= 1.0 - margin)
    )
    if not np.any(eligible):
        raise ValueError("no prefix satisfies variance and interior-gamma checks")
    return table[np.flatnonzero(eligible)[-1]].copy()

import numpy as np
from scipy.optimize import curve_fit


def opesf_cdf_log_rate(
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    events: 'np.ndarray',
    dt: float,
    temperature: float,
    indices: 'np.ndarray',
) -> float:
    try:
        bias = np.asarray(restored_bias)
        lens = np.asarray(lengths)
        ev = np.asarray(events)
        chosen = np.asarray(indices)
        step = float(dt)
        temp = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("OPES-f inputs must be numeric") from exc
    if (
        bias.ndim != 3 or lens.shape != bias.shape[:2] or ev.shape != lens.shape
        or chosen.ndim != 1 or chosen.size < 1
        or not np.issubdtype(chosen.dtype, np.integer)
    ):
        raise ValueError("incompatible OPES-f input shapes")
    if (
        not np.issubdtype(bias.dtype, np.number) or not np.isrealobj(bias)
        or not np.issubdtype(lens.dtype, np.integer)
        or not np.issubdtype(ev.dtype, np.number)
        or np.any(chosen < 0) or np.any(chosen >= bias.shape[0])
        or len(np.unique(chosen)) != len(chosen)
        or not np.isfinite(step) or step <= 0.0
        or not np.isfinite(temp) or temp <= 0.0
    ):
        raise ValueError("invalid OPES-f arrays or controls")
    bias = bias.astype(float, copy=False)
    ev = ev.astype(float, copy=False)
    if np.any((ev != 0.0) & (ev != 1.0)) or np.any(lens < 1) or np.any(lens > bias.shape[2]):
        raise ValueError("events must be binary and lengths in range")
    beta = 1.0 / (0.008314462618 * temp)
    log_times = []
    pooled_events = []
    for set_index in chosen:
        for trajectory in range(bias.shape[1]):
            count = int(lens[set_index, trajectory])
            active = bias[set_index, trajectory, :count]
            if np.any(~np.isfinite(active)):
                raise ValueError("active bias samples must be finite")
            log_acceleration = _eatr_logmeanexp(beta * active)
            log_times.append(np.log(step * count) + log_acceleration)
            pooled_events.append(bool(ev[set_index, trajectory]))
    log_times = np.asarray(log_times, dtype=float)
    pooled_events = np.asarray(pooled_events, dtype=bool)
    if np.count_nonzero(pooled_events) < 2:
        raise ValueError("CDF fitting requires at least two transitioned trajectories")
    if np.max(log_times) >= np.log(np.finfo(float).max):
        raise ValueError("rescaled time exceeded the float64 range")
    times = np.exp(log_times)
    event_times = np.sort(times[pooled_events])
    ordinates = np.arange(1, len(event_times) + 1, dtype=float) / len(times)
    initial = float(np.count_nonzero(pooled_events) / np.sum(times))
    try:
        parameters, _ = curve_fit(
            lambda time, rate: 1.0 - np.exp(-rate * time),
            event_times,
            ordinates,
            p0=initial,
            maxfev=100000,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
        raise ValueError("OPES-f CDF fit failed") from exc
    rate = float(parameters[0])
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("OPES-f CDF fit returned a nonpositive rate")
    result = float(np.log(rate))
    if not np.isfinite(result):
        raise ValueError("OPES-f log-rate is not finite")
    return result

import numpy as np


def leave_one_set_out_log_rates(
    log_k_observed: 'np.ndarray',
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    temperature: float,
    indices: 'np.ndarray',
) -> 'np.ndarray':
    try:
        chosen = np.asarray(indices)
        bias = np.asarray(restored_bias)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("leave-one-out inputs must be arrays") from exc
    if (
        bias.ndim != 3 or chosen.ndim != 1 or chosen.size < 4
        or not np.issubdtype(chosen.dtype, np.integer)
        or np.any(chosen < 0) or np.any(chosen >= bias.shape[0])
        or len(np.unique(chosen)) != len(chosen)
    ):
        raise ValueError("at least four distinct valid set indices are required")
    values = []
    for position in range(chosen.size):
        keep = np.delete(chosen, position)
        values.append(
            fit_eatr_flooding(
                log_k_observed, restored_bias, lengths, temperature, keep
            )[1]
        )
    result = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(result)):
        raise ValueError("leave-one-out refits are not finite")
    return result

import numpy as np


def _flooding_rate_snapshot(seed: int) -> tuple:
    rng = np.random.default_rng(int(seed))
    barriers = np.array([3.0, 4.5, 6.0, 7.5, 9.0, 11.0, 13.5])
    lengths = np.array([
        [18, 18, 17, 18, 16, 18, 18, 15, 18],
        [18, 15, 14, 18, 13, 16, 18, 12, 17],
        [16, 14, 15, 13, 14, 12, 16, 13, 13],
        [14, 13, 12, 14, 11, 13, 14, 12, 12],
        [13, 12, 11, 13, 10, 12, 13, 10, 11],
        [8, 7, 10, 6, 7, 5, 9, 6, 5],
        [7, 6, 8, 5, 6, 4, 7, 5, 4],
    ], dtype=int)
    events = np.array([
        [0, 0, 1, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 1, 1, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 1, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=int)
    raw_bias = np.full((7, 9, 18), np.nan)
    for set_index in range(7):
        for trajectory in range(9):
            count = int(lengths[set_index, trajectory])
            coordinate = np.linspace(0.0, 1.0, count)
            restored = (
                0.18 + 0.76 * barriers[set_index]
                + 0.18 * np.sin(2.0 * np.pi * (coordinate + 0.07 * trajectory))
                + 0.06 * trajectory
                + rng.normal(scale=0.055, size=count)
            )
            raw_bias[set_index, trajectory, :count] = restored - barriers[set_index]
    return raw_bias, lengths, events, barriers, 0.08, 300.0


def run_flooding_rate_audit(
    seed: int,
    variance_limit: float = 0.003,
    influence_limit: float = 0.2,
) -> float:
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    try:
        variance_cutoff = float(variance_limit)
        influence_cutoff = float(influence_limit)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("audit controls must be numeric") from exc
    if (
        not np.isfinite(variance_cutoff) or variance_cutoff < 0.0
        or not np.isfinite(influence_cutoff) or influence_cutoff <= 0.0
    ):
        raise ValueError("invalid variance or influence limit")
    raw, lengths, events, barriers, dt, temperature = _flooding_rate_snapshot(int(seed))
    restored = restore_opes_bias(
        raw, lengths, events, barriers, dt, temperature
    )
    observed = censored_observed_log_rates(lengths, events, dt)
    full_acceleration = ensemble_time_log_acceleration(
        restored, lengths, temperature, 1.0
    )
    order = np.argsort(full_acceleration, kind="stable")
    prefixes = scan_eatr_prefixes(
        observed, restored, lengths, temperature, order, 3
    )
    selected_row = select_consistent_prefix(
        prefixes, variance_cutoff, 0.02
    )
    selected_count = int(round(float(selected_row[0])))
    selected = order[:selected_count]
    fitted = fit_eatr_flooding(
        observed, restored, lengths, temperature, selected
    )
    if not np.allclose(fitted, selected_row[1:], rtol=0.0, atol=5e-10):
        raise ValueError("selected prefix and direct refit disagree")
    opesf_log_rate = opesf_cdf_log_rate(
        restored, lengths, events, dt, temperature, selected
    )
    deleted = leave_one_set_out_log_rates(
        observed, restored, lengths, temperature, selected
    )
    influence = float(np.max(np.abs(deleted - fitted[1])))
    if not np.isfinite(influence) or influence > influence_cutoff:
        raise ValueError("selected flooding estimate fails the influence check")
    with np.errstate(over="ignore", invalid="ignore"):
        result = 100.0 * np.expm1(float(fitted[1] - opesf_log_rate))
    if not np.isfinite(result):
        raise ValueError("signed rate difference is not finite")
    return float(result)
SCICODE_GOLD_EOF
