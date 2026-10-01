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


def _harmonic(m: int) -> float:
    """a_m = sum over j = 1 .. m - 1 of 1/j, with a_1 = 0."""
    return float(np.sum(1.0 / np.arange(1, m))) if m > 1 else 0.0


def linked_site_spectrum_components(n: int, i: int) -> "np.ndarray":
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 2:
        raise ValueError("n must be an integer >= 2")
    if isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 1 <= int(i) <= int(n) - 1:
        raise ValueError("i must be an integer from 1 to n - 1")
    n, i = int(n), int(i)
    a = np.array([_harmonic(m) for m in range(n + 2)])            # a[m] = a_m
    b = np.zeros(n + 2)                                              # b[j] = beta_n(j), Fu's second-moment coefficient
    for j in range(1, n):
        b[j] = 2.0 * n * (a[n + 1] - a[j]) / ((n - j + 1.0) * (n - j)) - 2.0 / (n - j)
    out = np.zeros((5, n - 1))
    for k in range(1, n):
        if k < i:
            out[0, k - 1] = i * (b[k] - b[k + 1]) / 2.0
        elif k == i:
            out[1, k - 1] = i * b[i]
        else:
            out[2, k - 1] = i * (b[i] - b[i + 1]) / 2.0
        if k == n - i:
            out[3, k - 1] = i * ((a[n] - a[k]) / (n - k) + (a[n] - a[i]) / (n - i) - (b[k] + b[i]) / 2.0)
        if k + i < n:
            out[4, k - 1] = 1.0 / k - i * (b[k] - b[k + 1] + b[i] - b[i + 1]) / 2.0
    return out

import numpy as np


def indel_null_spectrum(n: int, carriers: int, absence_derived: bool) -> "np.ndarray":
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 3:
        raise ValueError("n must be an integer >= 3")
    if isinstance(carriers, bool) or not isinstance(carriers, (int, np.integer)) or not 2 <= int(carriers) <= int(n) - 1:
        raise ValueError("carriers must be an integer from 2 to n - 1")
    if not isinstance(absence_derived, (bool, np.bool_)):
        raise ValueError("absence_derived must be a bool")
    n, c = int(n), int(carriers)
    k = np.arange(1, c)
    if absence_derived:
        i = n - c                                                    # sample count of the deletion, the derived state
        comp = linked_site_spectrum_components(n, i)
        # a site polymorphic among the survivors is either disjoint from the deletion (count k in the whole
        # sample) or encloses it (count k + i in the whole sample, its copies on the deleted background unseen)
        return comp[4, k - 1] + comp[2, k + i - 1]
    comp = linked_site_spectrum_components(n, c)             # the insertion itself is the focal variant
    return comp[0, k - 1]                                            # every site inside it is strictly nested

import numpy as np


def fold_spectrum(spectrum: "np.ndarray", m: int) -> "np.ndarray":
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or int(m) < 2:
        raise ValueError("m must be an integer >= 2")
    m = int(m)
    x = np.asarray(spectrum, dtype=float) if not isinstance(spectrum, (str, bytes)) else None
    if x is None or x.ndim != 1 or x.shape[0] != m - 1 or not np.all(np.isfinite(x)):
        raise ValueError("spectrum must be a one-dimensional array of finite numbers of length m - 1")
    out = np.zeros(m // 2)
    for j in range(1, m // 2 + 1):
        out[j - 1] = x[j - 1] if 2 * j == m else x[j - 1] + x[m - j - 1]
    return out

import numpy as np


def spectrum_statistics(counts: "np.ndarray", m: int) -> "np.ndarray":
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or int(m) < 4:
        raise ValueError("m must be an integer >= 4")
    m = int(m)
    x = np.asarray(counts, dtype=float) if not isinstance(counts, (str, bytes)) else None
    if x is None or x.ndim != 1 or x.shape[0] != m - 1 or not np.all(np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("counts must be a one-dimensional array of finite numbers >= 0 of length m - 1")
    k = np.arange(1, m, dtype=float)
    s_total = float(np.sum(x))
    if s_total < 2.0:
        raise ValueError("Tajima's D needs at least 2 sites")
    a1 = float(np.sum(1.0 / k))
    a2 = float(np.sum(1.0 / k ** 2))
    theta_w = s_total / a1
    pi = float(np.sum(2.0 * k * (m - k) / (m * (m - 1.0)) * x))
    theta_h = float(np.sum(2.0 * k * k / (m * (m - 1.0)) * x))
    b1 = (m + 1.0) / (3.0 * (m - 1.0))
    b2 = 2.0 * (m * m + m + 3.0) / (9.0 * m * (m - 1.0))
    c1 = b1 - 1.0 / a1
    c2 = b2 - (m + 2.0) / (a1 * m) + a2 / a1 ** 2
    e1, e2 = c1 / a1, c2 / (a1 ** 2 + a2)
    d = (pi - theta_w) / np.sqrt(e1 * s_total + e2 * s_total * (s_total - 1.0))
    return np.array([s_total, theta_w, pi, theta_h, d])

import numpy as np


def polarity_log_likelihoods(folded_counts: "np.ndarray", n: int, carriers: int, theta_per_site: float,
                                     segment_length: float) -> "np.ndarray":
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    if isinstance(carriers, bool) or not isinstance(carriers, (int, np.integer)):
        raise ValueError("carriers must be an integer from 2 to n - 1")
    c = int(carriers)
    eta = np.asarray(folded_counts, dtype=float) if not isinstance(folded_counts, (str, bytes)) else None
    if eta is None or eta.ndim != 1 or eta.shape[0] != c // 2 or not np.all(np.isfinite(eta)) or np.any(eta < 0.0):
        raise ValueError("folded_counts must be a one-dimensional array of finite numbers >= 0 of length carriers // 2")
    if not _num(theta_per_site) or float(theta_per_site) <= 0.0 or not _num(segment_length) or float(segment_length) <= 0.0:
        raise ValueError("theta_per_site and segment_length must be finite numbers > 0")
    scale = float(theta_per_site) * float(segment_length)
    out = np.zeros(2)
    for idx, absence_derived in enumerate((True, False)):
        lam = scale * fold_spectrum(indel_null_spectrum(n, c, absence_derived), c)
        out[idx] = float(np.sum(eta * np.log(lam) - lam))
    return out

import numpy as np


def sv_aware_theta_folded(folded_counts: "np.ndarray", n: int, carriers: int, absence_derived: bool,
                                  segment_length: float, weighting: str, correction: int) -> float:
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    if isinstance(carriers, bool) or not isinstance(carriers, (int, np.integer)):
        raise ValueError("carriers must be an integer from 2 to n - 1")
    c = int(carriers)
    eta = np.asarray(folded_counts, dtype=float) if not isinstance(folded_counts, (str, bytes)) else None
    if eta is None or eta.ndim != 1 or eta.shape[0] != c // 2 or not np.all(np.isfinite(eta)) or np.any(eta < 0.0):
        raise ValueError("folded_counts must be a one-dimensional array of finite numbers >= 0 of length carriers // 2")
    if not _num(segment_length) or float(segment_length) <= 0.0:
        raise ValueError("segment_length must be a finite number > 0")
    if weighting not in ("watterson", "pairwise"):
        raise ValueError('weighting must be "watterson" or "pairwise"')
    if isinstance(correction, bool) or correction not in (1, 2):
        raise ValueError("correction must be 1 or 2")
    length = float(segment_length)
    null = indel_null_spectrum(n, c, absence_derived)       # per unit theta L, derived counts 1 .. c - 1
    k = np.arange(1, c, dtype=float)
    if weighting == "watterson":
        w = 1.0 / (k * float(np.sum(1.0 / k)))
    else:
        w = 2.0 * (c - k) / (c * (c - 1.0))
    if correction == 1:
        w_folded = fold_spectrum(w, c)
        null_folded = fold_spectrum(null, c)
        return float(np.sum(w_folded * eta / (length * null_folded)))
    coef = w * k                                                     # symmetric under k <-> c - k for both weightings
    standard = float(np.sum(coef[:c // 2] * eta)) / length           # the standard estimator from the folded classes
    return float(standard / np.sum(coef * null))

import numpy as np


def sv_aware_relative_diversity(n: int, carriers: int, folded_counts: "np.ndarray", segment_length: float,
                                        flank_counts: "np.ndarray", flank_length: float) -> float:
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    if not _num(flank_length) or float(flank_length) <= 0.0:
        raise ValueError("flank_length must be a finite number > 0")
    theta_flank = float(spectrum_statistics(flank_counts, n)[1]) / float(flank_length)
    log_likelihoods = polarity_log_likelihoods(folded_counts, n, carriers, theta_flank, segment_length)
    absence_derived = bool(log_likelihoods[0] >= log_likelihoods[1])
    theta_segment = sv_aware_theta_folded(folded_counts, n, carriers, absence_derived, segment_length, "pairwise", 1)
    return float(theta_segment / theta_flank)
SCICODE_GOLD_EOF
