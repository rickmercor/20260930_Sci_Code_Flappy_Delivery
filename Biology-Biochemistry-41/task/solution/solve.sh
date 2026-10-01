#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def decode_series(
    amp_counts: "numpy.typing.ArrayLike",
    fwhm_counts: "numpy.typing.ArrayLike",
    dt_ms: float = 0.0154,
    amp_unit: float = 1.0e-4,
    fwhm_unit: float = 1.0e-3,
) -> "tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]":
    """Reference implementation for decode_series."""
    import numpy as np

    for name, value in (("dt_ms", dt_ms), ("amp_unit", amp_unit),
                        ("fwhm_unit", fwhm_unit)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")

    a = np.asarray(amp_counts, dtype=float)
    f = np.asarray(fwhm_counts, dtype=float)
    if a.ndim != 1 or a.shape != f.shape or a.size == 0:
        raise ValueError("amp_counts and fwhm_counts must share a non-empty 1-D shape")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(f))):
        raise ValueError("the counts must be finite")
    if np.any(a != np.round(a)) or np.any(f != np.round(f)):
        raise ValueError("the counts must be integer-valued")
    if np.any(a < 0) or np.any(f < 0):
        raise ValueError("the counts must be non-negative; a zero marks a failed fit")

    amp = np.round(a * float(amp_unit), 4)
    fwhm = np.round(f * float(fwhm_unit), 3)
    t = np.arange(a.size, dtype=float) * float(dt_ms)
    valid = (a > 0) & (f > 0)
    return t, amp, fwhm, valid

def signal_mass_interval(
    amp: "numpy.ndarray",
    fwhm: "numpy.ndarray",
    valid: "numpy.ndarray",
    baseline: float = 1.0,
    peak_fraction: float = 0.20,
    tail_ms: float = 1.0,
    dt_ms: float = 0.0154,
) -> "tuple[int, int]":
    """Reference implementation for signal_mass_interval."""
    import numpy as np

    a = np.asarray(amp, dtype=float)
    f = np.asarray(fwhm, dtype=float)
    v = np.asarray(valid, dtype=bool)
    if a.ndim != 1 or a.size == 0 or a.shape != f.shape or a.shape != v.shape:
        raise ValueError("amp, fwhm and valid must share a non-empty 1-D shape")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(f))):
        raise ValueError("amp and fwhm must be finite")
    if not np.isfinite(baseline):
        raise ValueError("baseline must be finite")
    if not np.isfinite(peak_fraction) or not (0.0 < peak_fraction < 1.0):
        raise ValueError("peak_fraction must lie strictly between 0 and 1")
    if not np.isfinite(tail_ms) or tail_ms <= 0:
        raise ValueError("tail_ms must be positive and finite")
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("dt_ms must be positive and finite")
    if not v.any():
        raise ValueError("the record has no converged samples")

    signal = a - float(baseline)
    peak_idx = int(np.argmax(signal))
    if not v[peak_idx]:
        raise ValueError("the peak sample is not a converged measurement")
    peak_signal = float(signal[peak_idx])
    if peak_signal <= 0.0:
        raise ValueError("the peak amplitude must rise above the baseline")

    threshold = float(peak_fraction) * peak_signal
    candidates = np.flatnonzero(v & (signal >= threshold))
    if candidates.size == 0:
        raise ValueError("no converged sample reaches the criterion (a) threshold")
    ts1_a = int(candidates[0])

    failures = np.flatnonzero(~v[: peak_idx + 1])
    ts1 = ts1_a if failures.size == 0 else max(ts1_a, int(failures[-1]) + 1)
    if ts1 > peak_idx:
        raise ValueError("criterion (b) leaves no converged interval before the peak")

    ts2 = peak_idx + int(round(float(tail_ms) / float(dt_ms)))
    if ts2 >= a.size:
        raise ValueError("the record does not extend tail_ms past the peak sample")
    return ts1, ts2

def signal_mass_curve(
    amp: "numpy.ndarray",
    fwhm: "numpy.ndarray",
    valid: "numpy.ndarray",
    ts1: int,
    ts2: int,
    baseline: float = 1.0,
    k: float = 1.206,
) -> "tuple[numpy.ndarray, bool]":
    """Reference implementation for signal_mass_curve."""
    import numpy as np

    a = np.asarray(amp, dtype=float)
    f = np.asarray(fwhm, dtype=float)
    v = np.asarray(valid, dtype=bool)
    if a.ndim != 1 or a.size == 0 or a.shape != f.shape or a.shape != v.shape:
        raise ValueError("amp, fwhm and valid must share a non-empty 1-D shape")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(f))):
        raise ValueError("amp and fwhm must be finite")
    if not np.isfinite(baseline):
        raise ValueError("baseline must be finite")
    if not np.isfinite(k) or k <= 0:
        raise ValueError("k must be positive and finite")
    if int(ts1) != ts1 or int(ts2) != ts2:
        raise ValueError("ts1 and ts2 must be integer sample indices")
    ts1, ts2 = int(ts1), int(ts2)
    if ts1 < 0 or ts2 >= a.size or ts1 > ts2:
        raise ValueError("the interval must lie inside the record and not be reversed")

    sm = float(k) * (a[ts1: ts2 + 1] - float(baseline)) * f[ts1: ts2 + 1] ** 3
    keep = v[ts1: ts2 + 1]
    sm = np.where(keep, sm, np.nan)
    continuous = bool(np.all(keep))
    return sm, continuous

def interval_statistics(
    amp: "numpy.ndarray",
    fwhm: "numpy.ndarray",
    valid: "numpy.ndarray",
    ts1: int,
    ts2: int,
    baseline: float = 1.0,
    k: float = 1.206,
) -> "tuple[int, float, float, float]":
    """Reference implementation for interval_statistics."""
    import numpy as np

    curves = signal_mass_curve(amp, fwhm, valid, ts1, ts2, baseline, k)
    sm, _continuous = curves
    a = np.asarray(amp, dtype=float)
    f = np.asarray(fwhm, dtype=float)
    v = np.asarray(valid, dtype=bool)
    ts1, ts2 = int(ts1), int(ts2)

    keep = v[ts1: ts2 + 1]
    if not keep.any():
        raise ValueError("the interval holds no converged sample")
    if not bool(keep[0]):
        raise ValueError("the interval's first sample is not a converged measurement")

    n_samples = ts2 - ts1 + 1
    mean_fwhm = float(np.mean(f[ts1: ts2 + 1][keep]))
    sm_at_start = float(sm[0])
    sm_max = float(np.nanmax(sm))
    return n_samples, mean_fwhm, sm_at_start, sm_max

def reference_bin_width(
    sigma_background: float,
    n_active: int,
    tested: "numpy.typing.ArrayLike" = (0.015, 0.020, 0.025),
) -> "tuple[float, float]":
    """Reference implementation for reference_bin_width."""
    import numpy as np

    if not np.isfinite(sigma_background) or sigma_background <= 0:
        raise ValueError("sigma_background must be positive and finite")
    if int(n_active) != n_active or n_active <= 0:
        raise ValueError("n_active must be a positive integer")
    grid = np.asarray(tested, dtype=float)
    if grid.size == 0:
        raise ValueError("tested must not be empty")
    if not np.all(np.isfinite(grid)) or np.any(grid <= 0):
        raise ValueError("tested bin widths must be positive and finite")

    scott_raw = float(3.49 * float(sigma_background)
                      * float(n_active) ** (-1.0 / 3.0))
    order = np.lexsort((grid, np.abs(grid - scott_raw)))
    adopted = float(grid[order[0]])
    return scott_raw, adopted

def level_histogram(
    amp_act: "numpy.ndarray",
    bin_width: float = 0.015,
    anchor: float = 1.0,
) -> "tuple[numpy.ndarray, numpy.ndarray]":
    """Reference implementation for level_histogram."""
    import numpy as np

    a = np.asarray(amp_act, dtype=float)
    if a.size == 0:
        raise ValueError("the activation phase must not be empty")
    if not np.all(np.isfinite(a)):
        raise ValueError("the activation phase must be finite")
    for name, value in (("bin_width", bin_width), ("anchor", anchor)):
        if not np.isfinite(value) or (name == "bin_width" and value <= 0):
            raise ValueError(f"{name} must be finite" + (" and positive" if name == "bin_width" else ""))

    # Edges anchored at the baseline so the alignment is noise-independent.
    lo = float(anchor) + np.floor((float(np.min(a)) - float(anchor)) / bin_width) * bin_width
    hi = float(anchor) + np.ceil((float(np.max(a)) - float(anchor)) / bin_width) * bin_width
    edges = lo + bin_width * np.arange(int(round((hi - lo) / bin_width)) + 1)
    counts, edges = np.histogram(a, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts, centers

def resolve_step_levels(
    counts: "numpy.ndarray",
    centers: "numpy.ndarray",
    bin_width: float = 0.015,
    prominence: float = 5.0,
) -> "numpy.ndarray":
    """Reference implementation for resolve_step_levels."""
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks

    counts = np.asarray(counts, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if counts.shape != centers.shape or counts.size == 0:
        raise ValueError("counts and centers must share a non-empty shape")
    if not (np.all(np.isfinite(counts)) and np.all(np.isfinite(centers))):
        raise ValueError("counts and centers must be finite")
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be positive and finite")
    if not np.isfinite(prominence) or prominence < 0:
        raise ValueError("prominence must be non-negative and finite")

    def _gauss_sum(x, *p):
        n = len(p) // 3
        y = np.zeros_like(x, dtype=float)
        for i in range(n):
            a, mu, s = p[3 * i: 3 * i + 3]
            y = y + a * np.exp(-0.5 * ((x - mu) / s) ** 2)
        return y

    pks, _ = find_peaks(counts, prominence=prominence)
    if pks.size == 0:
        pks = np.array([int(np.argmax(counts))])
    pks = np.sort(pks)
    k = pks.size
    p0 = []
    lo = []
    hi = []
    for pk in pks:
        p0.extend([float(counts[pk]), float(centers[pk]), float(bin_width)])
        lo.extend([0.0, float(centers[pk]) - bin_width, bin_width * 0.3])
        hi.extend([float(counts.max()) * 5.0, float(centers[pk]) + bin_width,
                   bin_width * 4.0])
    try:
        popt, _ = curve_fit(_gauss_sum, centers, counts, p0=p0,
                            bounds=(lo, hi), maxfev=60000)
        return np.sort(np.array([popt[3 * i + 1] for i in range(k)], dtype=float))
    except Exception:
        return np.sort(np.array([float(centers[i]) for i in pks], dtype=float))

def apply_merge_correction(
    levels: "numpy.ndarray",
    factor: float = 1.5,
) -> "tuple[int, float]":
    """Reference implementation for apply_merge_correction."""
    import numpy as np

    lv = np.asarray(levels, dtype=float)
    if lv.size == 0:
        raise ValueError("at least one level is required")
    if not np.all(np.isfinite(lv)):
        raise ValueError("levels must be finite")
    if not np.isfinite(factor) or factor <= 1.0:
        raise ValueError("factor must exceed 1")

    if lv.size == 1:
        return int(lv.size), 0.0
    gaps = np.diff(np.sort(lv))
    m = float(np.median(gaps))
    corrected = []
    for g in gaps:
        if g > factor * m:
            corrected.extend([g / 2.0, g / 2.0])
        else:
            corrected.append(float(g))
    corrected = np.asarray(corrected)
    # count = levels + one extra per split gap
    step_count = int(lv.size + sum(1 for g in gaps if g > factor * m))
    step_size = float(np.mean(corrected))
    return step_count, step_size

def signal_mass_increment(
    amp_counts: "numpy.typing.ArrayLike",
    fwhm_counts: "numpy.typing.ArrayLike",
    dt_ms: float = 0.0154,
    sigma_background: float = 0.0346,
) -> float:
    """Reference implementation for signal_mass_increment."""
    import numpy as np

    _t, amp, fwhm, valid = decode_series(amp_counts, fwhm_counts, dt_ms)
    ts1, ts2 = signal_mass_interval(amp, fwhm, valid, dt_ms=dt_ms)
    _n, mean_fwhm, _sm_start, _sm_max = interval_statistics(
        amp, fwhm, valid, ts1, ts2)

    peak_idx = int(np.argmax(amp))
    n_analyzed = int(valid[: peak_idx + 1].sum())
    _raw, bin_width = reference_bin_width(sigma_background, n_analyzed)
    hist, centers = level_histogram(
        amp[: peak_idx + 1][valid[: peak_idx + 1]], bin_width)
    levels = resolve_step_levels(hist, centers, bin_width, 5.0)
    _step_count, q = apply_merge_correction(levels, 1.5)
    return float(1.206 * q * mean_fwhm ** 3)
SCICODE_GOLD_EOF
