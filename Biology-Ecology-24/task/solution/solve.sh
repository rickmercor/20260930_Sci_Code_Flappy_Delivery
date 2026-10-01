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


def development_rate(
    temperature: "numpy.typing.ArrayLike",
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
) -> "numpy.ndarray":
    """Reference implementation for development_rate."""
    T = np.asarray(temperature, dtype=float)
    if not np.all(np.isfinite(T)):
        raise ValueError("temperature entries must be finite")
    Tk = T + 273.15
    if np.any(Tk <= 0.0):
        raise ValueError("temperature must be above -273.15 degrees Celsius")
    rho25 = float(Rho25) / 1e7
    dha = float(DHA) * 1e3
    dhh = float(DHH) * 1e3
    return (rho25 * (Tk / 298.0) * np.exp((dha / 8.314472) * (1.0 / 298.0 - 1.0 / Tk))) / (
        1.0 + np.exp((dhh / 8.314472) * (1.0 / np.abs(float(T12H)) - 1.0 / Tk))
    )

import numpy as np


def embryo_growth(
    times: "numpy.typing.ArrayLike",
    temperature: "numpy.typing.ArrayLike",
    hatchling_length: float,
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    query_times: "numpy.typing.ArrayLike | None" = None,
) -> "numpy.ndarray":
    """Reference implementation for embryo_growth."""
    t = np.asarray(times, dtype=float)
    T = np.asarray(temperature, dtype=float)
    if t.ndim != 1 or T.ndim != 1 or t.size != T.size or t.size < 2:
        raise ValueError("times and temperature must be 1-D arrays of at least two readings")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(T))):
        raise ValueError("times and temperature must be finite")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("times must strictly increase")
    if not (float(hatchling_length) > 0.0):
        raise ValueError("hatchling_length must be positive")
    if not (float(asymptotic_ratio) > 0.0) or not (0.0 < float(start_size)):
        raise ValueError("start_size and asymptotic_ratio must be positive")
    K = float(asymptotic_ratio) * float(hatchling_length)
    if not (float(start_size) < K):
        raise ValueError("start_size must lie below the asymptote")
    rate = development_rate(T, DHA, DHH, T12H, Rho25)
    u = np.empty(t.size, dtype=float)
    u[0] = np.log(K / float(start_size))
    u[1:] = u[0] * np.exp(-np.cumsum(rate[:-1] * np.diff(t)))
    if query_times is None:
        return K * np.exp(-u)
    q = np.asarray(query_times, dtype=float)
    if not np.all(np.isfinite(q)):
        raise ValueError("query_times must be finite")
    flat = np.atleast_1d(q)
    k = np.clip(np.searchsorted(t, flat, side="right") - 1, 0, t.size - 2)
    sizes = K * np.exp(-u[k] * np.exp(-rate[k] * (flat - t[k])))
    return sizes.reshape(q.shape) if q.ndim else sizes

import numpy as np


def pipping_time(
    times: "numpy.typing.ArrayLike",
    temperature: "numpy.typing.ArrayLike",
    hatchling_length: float,
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    target_size: "float | None" = None,
) -> float:
    """Reference implementation for pipping_time."""
    t = np.asarray(times, dtype=float)
    target = float(hatchling_length) if target_size is None else float(target_size)
    if not (target > 0.0):
        raise ValueError("target_size must be positive")
    sizes = embryo_growth(t, temperature, hatchling_length, DHA, DHH, T12H, Rho25,
                                  start_size, asymptotic_ratio)
    if np.all(sizes <= target):
        return float(t[-1])
    if sizes[0] > target:
        return float(t[0])
    kb = int(np.argmax(sizes > target)) - 1
    K = float(asymptotic_ratio) * float(hatchling_length)
    rate = development_rate(np.asarray(temperature, dtype=float)[kb], DHA, DHH, T12H, Rho25)
    u_kb = np.log(K / sizes[kb])
    return float(t[kb] + np.log(u_kb / np.log(K / target)) / rate)

import numpy as np


def window_limits(
    times: "numpy.typing.ArrayLike",
    temperature: "numpy.typing.ArrayLike",
    hatchling_length: float,
    end_time: float,
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    frac_begin: float = 0.250554,
    frac_end: float = 0.811197,
) -> "numpy.ndarray":
    """Reference implementation for window_limits."""
    t = np.asarray(times, dtype=float)
    if not np.isfinite(float(end_time)):
        raise ValueError("end_time must be finite")
    if not (float(t[0]) <= float(end_time) <= float(t[-1])):
        raise ValueError("end_time must lie inside the record")
    if not (0.0 < float(frac_begin) < float(frac_end)):
        raise ValueError("the stage fractions must be positive and strictly increasing")
    size_end = embryo_growth(t, temperature, hatchling_length, DHA, DHH, T12H, Rho25,
                                     start_size, asymptotic_ratio, query_times=[float(end_time)])[0]
    begin = pipping_time(t, temperature, hatchling_length, DHA, DHH, T12H, Rho25,
                                 start_size, asymptotic_ratio, target_size=float(frac_begin) * size_end)
    end = pipping_time(t, temperature, hatchling_length, DHA, DHH, T12H, Rho25,
                               start_size, asymptotic_ratio, target_size=float(frac_end) * size_end)
    return np.array([float(size_end), begin, end], dtype=float)

import numpy as np
from scipy.special import betainc


def sexualization_weight(
    temperature: "numpy.typing.ArrayLike",
    standardised_edges: "numpy.typing.ArrayLike",
    DHA: float = -719.576256,
    DHH: float = 685.203574,
    T12H: float = 552.204465,
    shape1: float = 1.725323,
    shape2: float = 0.802045,
    Rho25: float = 100.0,
) -> "numpy.ndarray":
    """Reference implementation for sexualization_weight."""
    T = np.asarray(temperature, dtype=float)
    edges = np.asarray(standardised_edges, dtype=float)
    if T.ndim != 1 or edges.ndim != 1 or edges.size != T.size + 1:
        raise ValueError("standardised_edges must be one longer than temperature")
    if not (np.all(np.isfinite(T)) and np.all(np.isfinite(edges))):
        raise ValueError("temperature and standardised_edges must be finite")
    if edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("standardised_edges must strictly increase")
    if not (0.0 <= float(edges[0]) and float(edges[-1]) <= 1.0):
        raise ValueError("standardised_edges must lie inside the closed unit interval")
    if not (float(shape1) > 0.0 and float(shape2) > 0.0 and float(Rho25) > 0.0):
        raise ValueError("shape1, shape2 and Rho25 must be positive")
    norm = development_rate(T, DHA, DHH, T12H, Rho25)
    norm = np.where(norm <= 0.0, 0.001, norm)
    mass = betainc(float(shape1), float(shape2), edges[1:]) - betainc(float(shape1), float(shape2), edges[:-1])
    return norm * mass

import numpy as np

def constant_temperature_equivalent(
    times: "numpy.typing.ArrayLike",
    temperature: "numpy.typing.ArrayLike",
    hatchling_length: float,
    window_start: float,
    window_end: float,
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    DHA_s: float = -719.576256,
    DHH_s: float = 685.203574,
    T12H_s: float = 552.204465,
    shape1: float = 1.725323,
    shape2: float = 0.802045,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    Rho25_s: float = 100.0,
) -> float:
    """Reference implementation for constant_temperature_equivalent."""
    t = np.asarray(times, dtype=float)
    T = np.asarray(temperature, dtype=float)
    if not (np.isfinite(float(window_start)) and np.isfinite(float(window_end))):
        raise ValueError("the window limits must be finite")
    if not (float(window_start) < float(window_end)):
        raise ValueError("window_start must lie below window_end")
    if not (float(t[0]) <= float(window_start) and float(window_end) <= float(t[-1])):
        raise ValueError("the window must lie inside the record")
    edges = np.concatenate(([float(window_start)], t[(t > float(window_start)) & (t < float(window_end))],
                            [float(window_end)]))
    sizes = embryo_growth(t, T, hatchling_length, DHA, DHH, T12H, Rho25,
                                  start_size, asymptotic_ratio, query_times=edges)
    span = (sizes - sizes[0]) / (sizes[-1] - sizes[0])
    left = np.clip(np.searchsorted(t, edges[:-1], side="right") - 1, 0, t.size - 1)
    piece_temperature = T[left]
    weight = sexualization_weight(piece_temperature, span, DHA_s, DHH_s, T12H_s,
                                          shape1, shape2, Rho25_s)
    return float((piece_temperature * weight).sum() / weight.sum())

import numpy as np


def male_fraction(
    temperature: float,
    P: float = 29.349664,
    SL: float = 2.238029,
    SH: float = 2.080107,
    l: float = 0.05,
) -> float:
    """Reference implementation for male_fraction."""
    if not np.isfinite(float(temperature)):
        raise ValueError("temperature must be finite")
    if not np.isfinite(float(P)):
        raise ValueError("P must be finite")
    if not (float(SL) > 0.0 and float(SH) > 0.0):
        raise ValueError("SL and SH must be positive")
    if not (0.0 < float(l) < 0.5):
        raise ValueError("l must lie strictly between zero and one half")
    span = float(SL) if float(temperature) < float(P) else float(SH)
    return float(1.0 / (1.0 + np.exp((-np.log((1.0 - float(l)) / float(l)) / span)
                                     * (float(P) - float(temperature)))))

import numpy as np


def expected_males(
    records: "list",
    hatchling_lengths: "numpy.typing.ArrayLike",
    sexed_counts: "numpy.typing.ArrayLike",
    end_times: "numpy.typing.ArrayLike",
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    P: float = 29.349664,
    SL: float = 2.238029,
    SH: float = 2.080107,
    DHA_s: float = -719.576256,
    DHH_s: float = 685.203574,
    T12H_s: float = 552.204465,
    shape1: float = 1.725323,
    shape2: float = 0.802045,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    frac_begin: float = 0.250554,
    frac_end: float = 0.811197,
    Rho25_s: float = 100.0,
) -> float:
    """Reference implementation for expected_males."""
    recs = list(records)
    hatchlings = np.asarray(hatchling_lengths, dtype=float)
    sexed = np.asarray(sexed_counts, dtype=float)
    ends = np.asarray(end_times, dtype=float)
    if not recs:
        raise ValueError("records must not be empty")
    if not (hatchlings.ndim == 1 and sexed.ndim == 1 and ends.ndim == 1
            and hatchlings.size == sexed.size == ends.size == len(recs)):
        raise ValueError("the per-nest inputs must be one-dimensional and equally long")
    if not (np.all(np.isfinite(hatchlings)) and np.all(np.isfinite(sexed)) and np.all(np.isfinite(ends))):
        raise ValueError("the per-nest inputs must be finite")
    if np.any(hatchlings <= 0.0):
        raise ValueError("hatchling lengths must be positive")
    if np.any(sexed < 0.0):
        raise ValueError("sexed counts must be non-negative")
    total = 0.0
    for record, hatchling, count, end in zip(recs, hatchlings, sexed, ends):
        r = np.asarray(record, dtype=float)
        if r.ndim != 2 or r.shape[1] != 2 or r.shape[0] < 2:
            raise ValueError("each record must be a 2-D array with two columns and at least two readings")
        times, temperature = r[:, 0], r[:, 1]
        limits = window_limits(times, temperature, float(hatchling), float(end), DHA, DHH,
                                       T12H, Rho25, start_size, asymptotic_ratio, frac_begin, frac_end)
        equivalent = constant_temperature_equivalent(
            times, temperature, float(hatchling), float(limits[1]), float(limits[2]), DHA, DHH,
            T12H, Rho25, DHA_s, DHH_s, T12H_s, shape1, shape2, start_size, asymptotic_ratio, Rho25_s)
        total += float(count) * male_fraction(equivalent, P, SL, SH)
    return float(total)

import numpy as np


def emergence_effect(
    records: "list",
    hatchling_lengths: "numpy.typing.ArrayLike",
    sexed_counts: "numpy.typing.ArrayLike",
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
    P: float = 29.349664,
    SL: float = 2.238029,
    SH: float = 2.080107,
    DHA_s: float = -719.576256,
    DHH_s: float = 685.203574,
    T12H_s: float = 552.204465,
    shape1: float = 1.725323,
    shape2: float = 0.802045,
    start_size: float = 0.347089,
    asymptotic_ratio: float = 1.208968,
    frac_begin: float = 0.250554,
    frac_end: float = 0.811197,
    Rho25_s: float = 100.0,
) -> float:
    """Reference implementation for emergence_effect."""
    recs = list(records)
    hatchlings = np.asarray(hatchling_lengths, dtype=float)
    sexed = np.asarray(sexed_counts, dtype=float)
    if not recs:
        raise ValueError("records must not be empty")
    if not (hatchlings.ndim == 1 and sexed.ndim == 1
            and hatchlings.size == sexed.size == len(recs)):
        raise ValueError("the per-nest inputs must be one-dimensional and equally long")
    if not (np.all(np.isfinite(hatchlings)) and np.all(np.isfinite(sexed))):
        raise ValueError("the per-nest inputs must be finite")
    if np.any(hatchlings <= 0.0):
        raise ValueError("hatchling lengths must be positive")
    if np.any(sexed < 0.0):
        raise ValueError("sexed counts must be non-negative")
    pipping = np.empty(len(recs), dtype=float)
    emergence = np.empty(len(recs), dtype=float)
    for i, (record, hatchling) in enumerate(zip(recs, hatchlings)):
        r = np.asarray(record, dtype=float)
        if r.ndim != 2 or r.shape[1] != 2 or r.shape[0] < 2:
            raise ValueError("each record must be a 2-D array with two columns and at least two readings")
        pipping[i] = pipping_time(r[:, 0], r[:, 1], float(hatchling), DHA, DHH, T12H,
                                          Rho25, start_size, asymptotic_ratio)
        emergence[i] = float(r[-1, 0])
    under_pipping = expected_males(recs, hatchlings, sexed, pipping, DHA, DHH, T12H, Rho25,
                                           P, SL, SH, DHA_s, DHH_s, T12H_s, shape1, shape2,
                                           start_size, asymptotic_ratio, frac_begin, frac_end, Rho25_s)
    under_emergence = expected_males(recs, hatchlings, sexed, emergence, DHA, DHH, T12H, Rho25,
                                             P, SL, SH, DHA_s, DHH_s, T12H_s, shape1, shape2,
                                             start_size, asymptotic_ratio, frac_begin, frac_end, Rho25_s)
    return float(under_pipping - under_emergence)
SCICODE_GOLD_EOF
