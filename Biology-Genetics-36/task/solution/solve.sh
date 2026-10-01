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


def _check_sizes(relative_sizes) -> "np.ndarray":
    """Validate a size vector and return it as a float array."""
    v = np.asarray(relative_sizes)
    if v.ndim != 1 or v.size < 1:
        raise ValueError("relative_sizes must be a one-dimensional array with at least one entry")
    if not np.issubdtype(v.dtype, np.number) or np.issubdtype(v.dtype, np.bool_):
        raise ValueError("relative_sizes must hold finite numbers > 0")
    v = v.astype(float)
    if not np.all(np.isfinite(v)) or np.any(v <= 0.0):
        raise ValueError("relative_sizes must hold finite numbers > 0")
    return v


def tree_length_moments(relative_sizes: "np.ndarray", max_order: int) -> "np.ndarray":
    v = _check_sizes(relative_sizes)
    if isinstance(max_order, bool) or not isinstance(max_order, (int, np.integer)) or int(max_order) < 0:
        raise ValueError("max_order must be an integer >= 0")
    m_max = int(max_order)
    # L is a sum of independent exponential variables k t_k with rates (k - 1) / v_{k-1};
    # its cumulants are kappa_m = (m - 1)! sum_k (v_{k-1} / (k - 1))^m, and the power
    # moments follow from the cumulants by the standard recursion
    scales = v / np.arange(1, v.size + 1, dtype=float)          # v_{k-1} / (k - 1), k = 2..n
    moments = np.empty(m_max + 1)
    moments[0] = 1.0
    for m in range(1, m_max + 1):
        total = 0.0
        coeff = 1.0                                              # (m - 1)_{i-1} = (m-1)(m-2)...(m-i+1)
        for i in range(1, m + 1):
            total += coeff * float(np.sum(scales ** i)) * moments[m - i]
            coeff *= (m - i)
        moments[m] = total
    return moments

import numpy as np


def _compositions(total: int, parts: int) -> int:
    """Number of ways to write total as an ordered sum of `parts` positive integers."""
    import math
    if parts < 0 or total < 0:
        return 0
    if parts == 0:
        return 1 if total == 0 else 0
    if total < parts:
        return 0
    return math.comb(total - 1, parts - 1)


def lineage_size_pair_counts(sample_size: int) -> "np.ndarray":
    if isinstance(sample_size, bool) or not isinstance(sample_size, (int, np.integer)) or int(sample_size) < 2:
        raise ValueError("sample_size must be an integer >= 2")
    n = int(sample_size)
    counts = np.zeros((n - 1, n - 1, n - 1, n - 1))
    # in a state with k lineages the sizes are uniform over the ordered compositions of n into k parts
    for k in range(2, n + 1):
        ck = _compositions(n, k)
        for i in range(1, n):
            for j in range(1, n):
                value = k * (k - 1) * _compositions(n - i - j, k - 2) / ck
                if i == j:
                    value += k * _compositions(n - i, k - 1) / ck
                counts[k - 2, k - 2, i - 1, j - 1] = value
    # two states k < k': the grouping of the k' lineages into their k ancestors is independent of the
    # sizes at k'; an ancestor has m descendant lineages at k' with the size probability of a sample of k'
    for k in range(2, n + 1):
        for kp in range(k + 1, n + 1):
            ckp = _compositions(n, kp)
            for i in range(1, n):
                for j in range(1, n):
                    total = 0.0
                    for m in range(1, kp - k + 2):
                        pm = _compositions(kp - m, k - 1) / _compositions(kp, k)
                        # the lineage of size j at k' descends from the lineage of size i at k
                        total += k * m * pm * _compositions(i - j, m - 1) * _compositions(n - i, kp - m) / ckp
                        # the lineage of size j at k' descends from another ancestor
                        total += k * (kp - m) * pm * _compositions(i, m) * _compositions(n - i - j, kp - m - 1) / ckp
                    counts[k - 2, kp - 2, i - 1, j - 1] = total
                    counts[kp - 2, k - 2, j - 1, i - 1] = total
    return counts

import numpy as np


def _check_count(value, n: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1 <= int(value) <= n - 1:
        raise ValueError(f"{name} must be an integer from 1 to n - 1")
    return int(value)


def _check_theta(theta) -> float:
    if isinstance(theta, bool) or not isinstance(theta, (int, float, np.integer, np.floating)):
        raise ValueError("theta must be a finite number > 0")
    if not np.isfinite(float(theta)) or float(theta) <= 0.0:
        raise ValueError("theta must be a finite number > 0")
    return float(theta)


def _pair_time_factors(v: "np.ndarray", theta: float) -> "np.ndarray":
    """T[a, b] = E[t_k t_k' exp(-theta L)] for the states k = a + 2, k' = b + 2, with t_k the waiting time with k lineages."""
    n = v.size + 1
    k = np.arange(2, n + 1, dtype=float)
    lam = (k - 1.0) / v                                  # rates of the scaled times k t_k
    fac = lam / (lam + theta)                            # E[exp(-theta k t_k)]
    allprod = float(np.prod(fac))
    T = np.zeros((n - 1, n - 1))
    for a in range(n - 1):
        la, ka = lam[a], k[a]
        T[a, a] = 2.0 * la / (la + theta) ** 3 * allprod / fac[a] / ka ** 2
        for b in range(a + 1, n - 1):
            lb, kb = lam[b], k[b]
            T[a, b] = T[b, a] = (la / (la + theta) ** 2) * (lb / (lb + theta) ** 2) * allprod / fac[a] / fac[b] / (ka * kb)
    return T


def two_site_pattern_probability(relative_sizes: "np.ndarray", theta: float, count_a: int, count_b: int) -> float:
    v = _check_sizes(relative_sizes)
    th = _check_theta(theta)
    n = v.size + 1
    i = _check_count(count_a, n, "count_a")
    j = _check_count(count_b, n, "count_b")
    counts = lineage_size_pair_counts(n)
    T = _pair_time_factors(v, th)
    # E[L_i L_j exp(-theta L)] = sum over pairs of states of E[l_k(i) l_k'(j)] E[t_k t_k' exp(-theta L)]
    mixed = float(np.sum(counts[:, :, i - 1, j - 1] * T))
    # two sites of the same count are two mutations of one Poisson class: (theta L_i)^2 / 2
    return th ** 2 * mixed / (2.0 if i == j else 1.0)

import numpy as np


def _site_count_distribution(v: "np.ndarray", theta: float, max_sites: int) -> "np.ndarray":
    """P(K = 0), ..., P(K = max_sites): the numbers of mutations in the states are independent geometric variables."""
    k = np.arange(2, v.size + 2, dtype=float)
    lam = (k - 1.0) / v
    dist = np.zeros(max_sites + 1)
    dist[0] = 1.0
    for rate in lam:
        q = theta / (rate + theta)
        geo = (1.0 - q) * q ** np.arange(max_sites + 1)
        dist = np.convolve(dist, geo)[: max_sites + 1]
    return dist


def single_site_probability(relative_sizes: "np.ndarray", theta: float, count: int) -> "np.ndarray":
    v = _check_sizes(relative_sizes)
    th = _check_theta(theta)
    n = v.size + 1
    i = _check_count(count, n, "count")
    k = np.arange(2, n + 1, dtype=float)
    lam = (k - 1.0) / v
    fac = lam / (lam + th)
    allprod = float(np.prod(fac))
    # theta E[L_i exp(-theta L)]: a random lineage in the state with k lineages has size i with
    # probability C(n-i-1, k-2)/C(n-1, k-1) (the compositions of n into k parts), and there are k of them
    total = 0.0
    for a in range(n - 1):
        kk = a + 2
        p_size = _compositions(n - i, kk - 1) / _compositions(n, kk)
        total += p_size * lam[a] / (lam[a] + th) ** 2 * allprod / fac[a]
    p_count = th * total
    p_one = float(_site_count_distribution(v, th, 1)[1])
    return np.array([p_count, p_one])

import numpy as np


def _stirling_second(m: int, k: int) -> int:
    """Stirling number of the second kind S(m, k)."""
    import math
    return sum((-1) ** (k - r) * math.comb(k, r) * r ** m for r in range(k + 1)) // math.factorial(k)


def site_count_statistics(relative_sizes: "np.ndarray", theta: float, n_sites: int, tolerance: float,
                                  moment_order: int) -> "np.ndarray":
    import math
    v = _check_sizes(relative_sizes)
    th = _check_theta(theta)
    if isinstance(n_sites, bool) or not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 0:
        raise ValueError("n_sites must be an integer >= 0")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float, np.integer, np.floating)):
        raise ValueError("tolerance must be a finite number in (0, 1)")
    if not np.isfinite(float(tolerance)) or not 0.0 < float(tolerance) < 1.0:
        raise ValueError("tolerance must be a finite number in (0, 1)")
    if isinstance(moment_order, bool) or not isinstance(moment_order, (int, np.integer)) or int(moment_order) < 0:
        raise ValueError("moment_order must be an integer >= 0")
    K, tol, order = int(n_sites), float(tolerance), int(moment_order)
    exact = float(_site_count_distribution(v, th, K)[K])
    # the series: P(K) = theta^K / K! sum_h (-1)^h theta^h mu_{h+K} / h!
    h_cap = 150
    moments = tree_length_moments(v, h_cap + K)
    prefactor = th ** K / math.factorial(K)
    partial, h_min, value = 0.0, -1, 0.0
    for h in range(h_cap + 1):
        partial += (-1) ** h * th ** h / math.factorial(h) * moments[h + K]
        value = prefactor * partial
        if abs(value - exact) < tol * exact:
            h_min = h
            break
    if h_min < 0:
        raise ValueError("the series partial sums have not reached the tolerance by index 150")
    # power moments of K by conditioning on the length (K | L is Poisson with mean theta L), then the
    # central moment by the binomial expansion about E[K]
    power = [sum(_stirling_second(m, r) * moments[r] * th ** r for r in range(m + 1)) for m in range(order + 1)]
    mean = th * moments[1]
    central = sum((-1) ** r * math.comb(order, r) * mean ** r * power[order - r] for r in range(order + 1))
    return np.array([exact, float(h_min), value, float(central)])

import numpy as np


def history_decision(sizes_null: "np.ndarray", sizes_alt: "np.ndarray", theta_a: float, count_a1: int,
                             count_a2: int, theta_b: float, count_b: int, theta_c: float, count_c: int) -> "np.ndarray":
    v0 = _check_sizes(sizes_null)
    v1 = _check_sizes(sizes_alt)
    if v0.size != v1.size:
        raise ValueError("sizes_null and sizes_alt must have the same length")
    n = v0.size + 1
    _check_count(count_b, n, "count_b")
    _check_count(count_c, n, "count_c")
    joint = []
    for v in (v0, v1):
        pattern = two_site_pattern_probability(v, theta_a, count_a1, count_a2)
        single_b = single_site_probability(v, theta_b, count_b)[0]
        single_c = single_site_probability(v, theta_c, count_c)[0]
        joint.append(pattern * single_b * single_c)       # independent loci: the probabilities multiply
    decision = 1.0 if joint[1] > joint[0] else 0.0
    return np.array([joint[0], joint[1], decision])

import numpy as np


def configuration_probability_given_site_counts(sample_size: int, reduced_fraction: float, reduced_from: int,
                                                        reduced_to: int, theta_per_site: float, length_a: float,
                                                        count_a1: int, count_a2: int, length_b: float, count_b: int,
                                                        length_c: float, count_c: int) -> float:
    if isinstance(sample_size, bool) or not isinstance(sample_size, (int, np.integer)) or int(sample_size) < 3:
        raise ValueError("sample_size must be an integer >= 3")
    n = int(sample_size)
    for name, value in (("reduced_fraction", reduced_fraction), ("theta_per_site", theta_per_site),
                        ("length_a", length_a), ("length_b", length_b), ("length_c", length_c)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a finite number > 0")
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite number > 0")
    for name, value in (("reduced_from", reduced_from), ("reduced_to", reduced_to)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("reduced_from and reduced_to must be integers with 2 <= reduced_from <= reduced_to <= sample_size")
    k_lo, k_hi = int(reduced_from), int(reduced_to)
    if not 2 <= k_lo <= k_hi <= n:
        raise ValueError("reduced_from and reduced_to must be integers with 2 <= reduced_from <= reduced_to <= sample_size")
    theta_a = float(theta_per_site) * float(length_a)
    theta_b = float(theta_per_site) * float(length_b)
    theta_c = float(theta_per_site) * float(length_c)
    sizes_null = np.ones(n - 1)
    sizes_alt = np.ones(n - 1)
    sizes_alt[k_lo - 2:k_hi - 1] = float(reduced_fraction)     # entry k - 2 holds the state with k lineages
    decision = history_decision(sizes_null, sizes_alt, theta_a, count_a1, count_a2, theta_b, count_b,
                                        theta_c, count_c)
    sizes = sizes_alt if decision[2] > 0.5 else sizes_null
    pattern = two_site_pattern_probability(sizes, theta_a, count_a1, count_a2)
    # the site-count step also reports the series accuracy at one per cent and the fifth central moment,
    # which the reasoning asks for; only the exact probability enters the deliverable
    two_sites = site_count_statistics(sizes, theta_a, 2, 0.01, 5)[0]
    single_b = single_site_probability(sizes, theta_b, count_b)
    single_c = single_site_probability(sizes, theta_c, count_c)
    return float(pattern / two_sites * single_b[0] / single_b[1] * single_c[0] / single_c[1])
SCICODE_GOLD_EOF
