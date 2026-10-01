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


import math


def waples_ne_estimate(r2_mean: float, sample_size: int) -> float:
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or not 30 <= sample_size <= 100000:
        raise ValueError("sample_size must be an integer from 30 to 100000")
    r2 = float(r2_mean)
    if not math.isfinite(r2) or not 0.0 <= r2 <= 1.0:
        raise ValueError("r2_mean must be a finite number between 0 and 1")
    s = float(sample_size)
    x = r2 - (1.0 / s + 3.19 / (s * s))
    if x < 1e-6:
        raise ValueError("the drift part of r2_mean must be at least 1e-6")
    disc = 1.0 / 9.0 - 2.76 * x
    if disc < 0.0:
        raise ValueError("the drift part of r2_mean exceeds 1/(9 * 2.76)")
    return float((1.0 / 3.0 + math.sqrt(disc)) / (2.0 * x))

import numpy as np


import math


def expected_unlinked_r2(ne: float, sample_size: int) -> float:
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or not 30 <= sample_size <= 100000:
        raise ValueError("sample_size must be an integer from 30 to 100000")
    n = float(ne)
    if not math.isfinite(n) or not 5.0 <= n <= 300000.0:
        raise ValueError("ne must be a finite number from 5 to 300000")
    s = float(sample_size)
    return float(1.0 / s + 3.19 / (s * s) + 1.0 / (3.0 * n) - 0.69 / (n * n))

import numpy as np


def pseudo_replication_rho(raw_same_chromosome_r2: "np.ndarray", n_snps: "np.ndarray", sample_size: int) -> float:
    raw = np.asarray(raw_same_chromosome_r2, dtype=float)
    L = np.asarray(n_snps)
    if raw.ndim != 1 or L.ndim != 1 or raw.shape != L.shape or L.size < 2:
        raise ValueError("raw_same_chromosome_r2 and n_snps must be 1-D arrays of the same length C >= 2")
    if not np.issubdtype(L.dtype, np.integer) or np.any(L < 2) or np.any(L > 10 ** 6):
        raise ValueError("n_snps must hold integers from 2 to 10**6")
    if isinstance(sample_size, bool) or not isinstance(sample_size, (int, np.integer)) \
            or not 30 <= int(sample_size) <= 100000:
        raise ValueError("sample_size must be an integer from 30 to 100000")
    s = int(sample_size)
    e_samp = 1.0 / s + 3.19 / (s * s)
    if not np.all(np.isfinite(raw)) or np.any(raw < e_samp) or np.any(raw > 1.0):
        raise ValueError("each raw mean must be finite, at least the sampling expectation and at most 1")
    Lf = L.astype(float)
    A = (raw - e_samp) * Lf * (Lf - 1.0) / 2.0        # drift-induced r^2 summed over the pairs on each chromosome
    total_pairs = 0.0
    corr_sum = 0.0
    C = L.size
    for a in range(C):
        for b in range(a + 1, C):
            total_pairs += Lf[a] * Lf[b]
            # pairs of unlinked (a, b) pairs sharing a SNP on a or on b, then pairs sharing none
            corr_sum += Lf[a] * A[b] + Lf[b] * A[a] + 2.0 * A[a] * A[b]
    return float(corr_sum / total_pairs)

import math
import numpy as np


def r2_confidence_limits(r2_mean: float, rho: float, n_snps: "np.ndarray", z: float) -> "np.ndarray":
    r2 = float(r2_mean)
    if not math.isfinite(r2) or not 0.0 < r2 <= 1.0:
        raise ValueError("r2_mean must be a finite number in (0, 1]")
    r = float(rho)
    if not math.isfinite(r) or r < 0.0:
        raise ValueError("rho must be a finite number that is at least 0")
    zz = float(z)
    if not math.isfinite(zz) or not 0.5 <= zz <= 5.0:
        raise ValueError("z must be a finite number from 0.5 to 5")
    L = np.asarray(n_snps)
    if L.ndim != 1 or L.size < 2 or not np.issubdtype(L.dtype, np.integer) \
            or np.any(L < 2) or np.any(L > 10 ** 6):
        raise ValueError("n_snps must be a 1-D array of at least two integers from 2 to 10**6")
    Lf = L.astype(float)
    total_pairs = float(sum(Lf[a] * Lf[b] for a in range(Lf.size) for b in range(a + 1, Lf.size)))
    half_width = zz * math.sqrt(2.0 * (1.0 + 2.0 * r) / total_pairs)
    upper = r2 / (1.0 - half_width) if half_width < 1.0 else math.inf
    return np.array([min(max(r2 / (1.0 + half_width), 0.0), 1.0), min(max(upper, 0.0), 1.0)])

import numpy as np


def ne_confidence_limits(r2_mean: float, sample_size: int, rho: float, n_snps: "np.ndarray", z: float) -> "np.ndarray":
    r2_limits = r2_confidence_limits(r2_mean, rho, n_snps, z)
    ne_low = waples_ne_estimate(float(r2_limits[1]), sample_size)    # the upper r^2 limit bounds Ne from below
    ne_high = waples_ne_estimate(float(r2_limits[0]), sample_size)
    return np.array([ne_low, ne_high])

import math
import numpy as np
from scipy import stats


def certification_probability(ne: float, ne_threshold: float, sample_size: int, rho: float, n_snps: "np.ndarray", z: float) -> float:
    n = float(ne)
    if not math.isfinite(n) or not 5.0 <= n <= 300000.0:
        raise ValueError("ne must be a finite number from 5 to 300000")
    t = float(ne_threshold)
    if not math.isfinite(t) or not 5.0 <= t <= n:
        raise ValueError("ne_threshold must be a finite number from 5 to ne")
    if isinstance(sample_size, bool) or not isinstance(sample_size, (int, np.integer)) \
            or not 30 <= int(sample_size) <= 100000:
        raise ValueError("sample_size must be an integer from 30 to 100000")
    s = int(sample_size)
    r = float(rho)
    if not math.isfinite(r) or r < 0.0:
        raise ValueError("rho must be a finite number that is at least 0")
    zz = float(z)
    if not math.isfinite(zz) or not 0.5 <= zz <= 5.0:
        raise ValueError("z must be a finite number from 0.5 to 5")
    L = np.asarray(n_snps)
    if L.ndim != 1 or L.size < 2 or not np.issubdtype(L.dtype, np.integer) \
            or np.any(L < 2) or np.any(L > 10 ** 6):
        raise ValueError("n_snps must be a 1-D array of at least two integers from 2 to 10**6")
    Lf = L.astype(float)
    total_pairs = float(sum(Lf[a] * Lf[b] for a in range(Lf.size) for b in range(a + 1, Lf.size)))
    relative_sd = math.sqrt(2.0 * (1.0 + 2.0 * r) / total_pairs)
    half_width = zz * relative_sd
    if half_width >= 1.0:
        raise ValueError("z sqrt(2 (1 + 2 rho) / N) must be below 1")
    mean = expected_unlinked_r2(n, s)
    # Waples' estimate falls as r^2 rises, so the lower limit clears the threshold exactly while the sample's
    # own mean stays below the mean expected at the threshold, shrunk by the interval's factor 1 - w
    critical = (1.0 - half_width) * expected_unlinked_r2(t, s)
    return float(stats.norm.cdf((critical - mean) / (mean * relative_sd)))

import numpy as np


import math


def smallest_assured_sample(ne: float, ne_threshold: float, assurance: float, rho: float, n_snps: "np.ndarray", z: float, max_sample: int) -> int:
    a = float(assurance)
    if not math.isfinite(a) or not 0.5 <= a <= 0.999:
        raise ValueError("assurance must be a finite number from 0.5 to 0.999")
    if isinstance(max_sample, bool) or not isinstance(max_sample, int) or not 30 <= max_sample <= 100000:
        raise ValueError("max_sample must be an integer from 30 to 100000")
    for s in range(30, max_sample + 1):
        if certification_probability(ne, ne_threshold, s, rho, n_snps, z) >= a:
            return s
    raise ValueError("no sample size up to max_sample qualifies")

import numpy as np


def planned_upper_ne_limit(pilot_r2_mean: float, pilot_size: int, raw_same_chromosome_r2: "np.ndarray", n_snps: "np.ndarray", ne_threshold: float, assurance: float, z: float, max_sample: int) -> float:
    ne_hat = waples_ne_estimate(pilot_r2_mean, pilot_size)
    rho = pseudo_replication_rho(raw_same_chromosome_r2, n_snps, pilot_size)
    s_assured = smallest_assured_sample(ne_hat, ne_threshold, assurance, rho, n_snps, z, max_sample)
    r2_assured = expected_unlinked_r2(ne_hat, s_assured)
    limits = ne_confidence_limits(r2_assured, s_assured, rho, n_snps, z)
    return float(limits[1])
SCICODE_GOLD_EOF
