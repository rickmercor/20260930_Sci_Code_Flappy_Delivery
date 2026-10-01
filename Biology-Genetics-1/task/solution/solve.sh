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


def flank_survival_and_density(x: "np.ndarray", t: int, break_rate: float, marker_spacing: float,
                                       heterozygosity: float) -> "np.ndarray":
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    xx = np.asarray(x, dtype=float) if not isinstance(x, (str, bytes)) else None
    if xx is None or xx.ndim != 1 or xx.size < 1 or not np.all(np.isfinite(xx)) or np.any(xx < 0.0):
        raise ValueError("x must be a one-dimensional array of finite numbers >= 0 with at least one entry")
    if isinstance(t, bool) or not isinstance(t, (int, np.integer)) or int(t) < 1:
        raise ValueError("t must be an integer >= 1")
    if not _num(break_rate) or float(break_rate) < 0.0:
        raise ValueError("break_rate must be a finite number >= 0")
    if not _num(heterozygosity) or not 0.0 < float(heterozygosity) <= 1.0:
        raise ValueError("heterozygosity must be a finite number in (0, 1]")
    if not _num(marker_spacing) or not 0.0 <= float(marker_spacing) < float(heterozygosity):
        raise ValueError("marker_spacing must be a finite number in [0, heterozygosity)")
    m, d, H = float(break_rate), float(marker_spacing), float(heterozygosity)
    n_lineage_generations = 2 * int(t)          # two lineages, t generations each
    decay = np.exp(-m * xx)                     # mutation and conversion breaks, observed where they occur
    if d == 0.0:
        p = np.exp(-xx) * decay
        dp = -(1.0 + m) * np.exp(-xx) * decay
    else:
        # a recombination breakpoint is observed at the first heterozygous marker beyond it: the
        # observable distance is the sum of an exponential(1) and an exponential(H/d) variable
        r = H / d
        c1 = H / (H - d)
        c2 = d / (H - d)
        rec = c1 * np.exp(-xx) - c2 * np.exp(-r * xx)
        drec = -c1 * np.exp(-xx) + c2 * r * np.exp(-r * xx)
        p = rec * decay
        dp = (drec - m * rec) * decay
    survival = p ** n_lineage_generations
    density = -n_lineage_generations * p ** (n_lineage_generations - 1) * dp
    return np.array([survival, density])

import numpy as np


def class_mass_given_time(lower: float, upper: float, t: "np.ndarray", break_rate: float,
                                  marker_spacing: float, heterozygosity: float) -> "np.ndarray":
    from numpy.polynomial.legendre import leggauss

    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating))

    if not _num(lower) or not np.isfinite(float(lower)) or float(lower) < 0.0:
        raise ValueError("lower must be a finite number >= 0")
    if not _num(upper) or np.isnan(float(upper)) or not float(upper) > float(lower):
        raise ValueError("upper must be a number > lower (infinite allowed)")
    tt = np.asarray(t) if not isinstance(t, (str, bytes)) else None
    if (tt is None or tt.ndim != 1 or tt.size < 1 or tt.dtype.kind == "b"
            or not np.all(np.isfinite(tt.astype(float))) or np.any(tt.astype(float) != np.floor(tt.astype(float)))
            or np.any(tt.astype(float) < 1)):
        raise ValueError("t must be a one-dimensional array of integers >= 1 with at least one entry")
    a, b = float(lower), float(upper)
    # the two sides are independent, so with T(x) = P(L + R >= x) = S(x) + int_0^x f(u) S(x - u) du,
    # a sum of non-negative terms, the class probability is T(lower) - T(upper), with T(inf) = 0
    nodes, weights = leggauss(128)

    def _tail_grid(x):
        u = 0.5 * x * (nodes + 1.0)
        return u, 0.5 * x * weights

    ua, wa = _tail_grid(a)
    finite = np.isfinite(b)
    if finite:
        ub, wb = _tail_grid(b)
        grid = np.concatenate([[a], ua, a - ua, [b], ub, b - ub])
    else:
        grid = np.concatenate([[a], ua, a - ua])
    n = ua.size
    out = np.empty(tt.size)
    for j, tj in enumerate(tt.astype(float)):
        sd = flank_survival_and_density(grid, int(tj), break_rate, marker_spacing, heterozygosity)
        tail_a = sd[0, 0] + float((sd[1, 1:n + 1] * sd[0, n + 1:2 * n + 1]) @ wa)
        if finite:
            off = 2 * n + 1
            tail_b = sd[0, off] + float((sd[1, off + 1:off + n + 1] * sd[0, off + n + 1:off + 2 * n + 1]) @ wb)
        else:
            tail_b = 0.0
        out[j] = tail_a - tail_b
    return out

import numpy as np


def fitness_variance_under_ratchet(census: float, mutation_rate: float, selection: float,
                                           chromosome_length: float) -> float:
    from numpy.polynomial.legendre import leggauss

    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    if not _num(census) or float(census) < 1.0:
        raise ValueError("census must be a finite number >= 1")
    if not _num(mutation_rate) or float(mutation_rate) <= 0.0:
        raise ValueError("mutation_rate must be a finite number > 0")
    if not _num(selection) or not 0.0 < float(selection) < 1.0:
        raise ValueError("selection must be a finite number in (0, 1)")
    if not _num(chromosome_length) or float(chromosome_length) <= 0.0:
        raise ValueError("chromosome_length must be a finite number > 0")
    big_n, u, sel, length = float(census), float(mutation_rate), float(selection), float(chromosome_length)
    # quadrature over the map distances to half the chromosome, split near x = 0 where the integrand varies on the
    # scale of the renewal fraction
    nodes, weights = leggauss(96)
    half = 0.5 * length
    split = min(0.05, half)
    x1, w1 = 0.5 * split * (nodes + 1.0), 0.5 * split * weights
    x2, w2 = split + 0.5 * (half - split) * (nodes + 1.0), 0.5 * (half - split) * weights
    x = np.concatenate([x1, x2])
    w = np.concatenate([w1, w2])
    recomb = 0.5 * (1.0 - np.exp(-2.0 * x))                # Haldane's map function

    def _variance(n):
        """V_w = U s - s/T at the asymptotic haploid number n, with s/T written so that large 2 s n cannot overflow."""
        return u * sel - 2.0 * u * sel * sel * n * np.exp(-2.0 * sel * n) / (-np.expm1(-2.0 * sel * n))

    def _asymptotic(n):
        """2 N exp(-V_w Qbar) for the variance that n implies."""
        variance = _variance(n)
        renewal = u * sel * sel / variance
        q = 1.0 / (1.0 - (1.0 - recomb) * (1.0 - renewal))
        return 2.0 * big_n * np.exp(-variance * (2.0 / length) * ((q * q) @ w))

    # the map n -> 2 N exp(-V_w Qbar) is strictly decreasing, so g(n) = map(n) - n has one root in (0, 2 N):
    # bisection to far below the required precision
    lo, hi = 1.0e-9, 2.0 * big_n
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _asymptotic(mid) > mid:
            lo = mid
        else:
            hi = mid
    return float(_variance(0.5 * (lo + hi)))

import numpy as np


def selection_reduction_factor(generations: "np.ndarray", mutation_rate: float, selection: float,
                                       chromosome_length: float, census: float) -> "np.ndarray":
    from numpy.polynomial.legendre import leggauss

    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    gg = np.asarray(generations) if not isinstance(generations, (str, bytes)) else None
    if (gg is None or gg.ndim != 1 or gg.size < 1 or gg.dtype.kind == "b"
            or not np.all(np.isfinite(gg.astype(float))) or np.any(gg.astype(float) != np.floor(gg.astype(float)))
            or np.any(gg.astype(float) < 1)):
        raise ValueError("generations must be a one-dimensional array of integers >= 1 with at least one entry")
    if not _num(mutation_rate) or float(mutation_rate) <= 0.0:
        raise ValueError("mutation_rate must be a finite number > 0")
    if not _num(selection) or not 0.0 < float(selection) < 1.0:
        raise ValueError("selection must be a finite number in (0, 1)")
    if not _num(chromosome_length) or float(chromosome_length) <= 0.0:
        raise ValueError("chromosome_length must be a finite number > 0")
    if not _num(census) or float(census) < 1.0:
        raise ValueError("census must be a finite number >= 1")
    u, sel, length = float(mutation_rate), float(selection), float(chromosome_length)
    variance = fitness_variance_under_ratchet(census, u, sel, length)   # V_w for this census
    renewal = u * sel * sel / variance                     # V_M / V_w
    # the integrand varies on the scale of the selection coefficient near x = 0, so the range is split there
    nodes, weights = leggauss(96)
    half = 0.5 * length
    split = min(0.05, half)
    x1, w1 = 0.5 * split * (nodes + 1.0), 0.5 * split * weights
    x2, w2 = split + 0.5 * (half - split) * (nodes + 1.0), 0.5 * (half - split) * weights
    x = np.concatenate([x1, x2])
    w = np.concatenate([w1, w2])
    recomb = 0.5 * (1.0 - np.exp(-2.0 * x))                # Haldane's map function
    ratio = (1.0 - recomb) * (1.0 - renewal)               # the per-generation survival of the association, < 1
    g = gg.astype(float)[:, None]
    q = (1.0 - ratio ** g) / (1.0 - ratio)                 # sum of the geometric series with g terms
    exponent = (2.0 / length) * variance * ((q * q) @ w)
    return np.exp(-exponent)

import numpy as np


def coalescence_weights(n_ancestral: float, n_bottleneck: float, n_recent: float, bottleneck_length: int,
                                since_end: int, mutation_rate: float, selection: float, chromosome_length: float, horizon: int) -> "np.ndarray":
    def _size(v):
        return ((not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating))
                and bool(np.isfinite(float(v))) and float(v) >= 1.0)

    def _count(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, np.integer))

    if not (_size(n_ancestral) and _size(n_bottleneck) and _size(n_recent)):
        raise ValueError("n_ancestral, n_bottleneck and n_recent must be finite numbers >= 1")
    if not (_count(bottleneck_length) and int(bottleneck_length) >= 0 and _count(since_end) and int(since_end) >= 0):
        raise ValueError("bottleneck_length and since_end must be integers >= 0")
    if not (_count(horizon) and int(horizon) >= 1):
        raise ValueError("horizon must be an integer >= 1")
    g = np.arange(1, int(horizon) + 1)
    e, dd = int(since_end), int(bottleneck_length)
    size = np.where(g <= e, float(n_recent), np.where(g <= e + dd, float(n_bottleneck), float(n_ancestral)))
    factor = np.empty(g.size)
    for census in (float(n_recent), float(n_bottleneck), float(n_ancestral)):   # each epoch's census sets its own variance
        in_epoch = size == census
        if np.any(in_epoch):
            factor[in_epoch] = selection_reduction_factor(g[in_epoch], mutation_rate, selection, chromosome_length, census)
    rate = 1.0 / (2.0 * size * factor)
    # probability of no coalescence in the generations more recent than g, then coalescence at g
    not_yet = np.concatenate([[1.0], np.cumprod(1.0 - rate)[:-1]])
    return rate * not_yet

import numpy as np


def class_coverage_from_reported_size(reported_size: float, lower: float, upper: float, break_rate: float,
                                              marker_spacing: float, heterozygosity: float) -> float:
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating))

    if not _num(reported_size) or not np.isfinite(float(reported_size)) or float(reported_size) < 1.0:
        raise ValueError("reported_size must be a finite number >= 1")
    if not _num(lower) or not np.isfinite(float(lower)) or float(lower) <= 0.0:
        raise ValueError("lower must be a finite number > 0")
    if not _num(upper) or np.isnan(float(upper)) or not float(upper) > float(lower):
        raise ValueError("upper must be a number > lower (infinite allowed)")
    if not _num(break_rate) or not np.isfinite(float(break_rate)) or float(break_rate) < 0.0:
        raise ValueError("break_rate must be a finite number >= 0")
    if not _num(heterozygosity) or not np.isfinite(float(heterozygosity)) or not 0.0 < float(heterozygosity) <= 1.0:
        raise ValueError("heterozygosity must be a finite number in (0, 1]")
    if not _num(marker_spacing) or not np.isfinite(float(marker_spacing)) or not 0.0 <= float(marker_spacing) < float(heterozygosity):
        raise ValueError("marker_spacing must be a finite number in [0, heterozygosity)")
    n, m, d, h = float(reported_size), float(break_rate), float(marker_spacing), float(heterozygosity)
    a, b = float(lower), float(upper)
    # the source's steady-state density of coverage by runs of length x under a constant size:
    # 4 x (1 + m)^2 / (N (2 x (1 + m) + 1/(2N) - 4 d/H)^3), the displacement entering through -4 d/H
    beta = 2.0 * (1.0 + m)
    shift = 1.0 / (2.0 * n) - 4.0 * d / h
    if beta * a + shift <= 0.0:
        raise ValueError("the expression does not apply to the class: lower is not large enough relative to the mean displacement")
    def _antiderivative(u):
        return -1.0 / u + shift / (2.0 * u * u)
    upper_term = 0.0 if np.isinf(b) else _antiderivative(beta * b + shift)
    return float((upper_term - _antiderivative(beta * a + shift)) / n)

import numpy as np


def bottleneck_end_from_class_coverage(observed: float, lower: float, upper: float, n_ancestral: float,
                                               n_bottleneck: float, n_recent: float, bottleneck_length: int, mutation_rate: float, selection: float, chromosome_length: float, break_rate: float,
                                               marker_spacing: float, heterozygosity: float, horizon: int,
                                               max_since_end: int) -> int:
    if (isinstance(observed, bool) or not isinstance(observed, (int, float, np.integer, np.floating))
            or not np.isfinite(float(observed)) or not 0.0 <= float(observed) <= 1.0):
        raise ValueError("observed must be a finite number in [0, 1]")
    if isinstance(max_since_end, bool) or not isinstance(max_since_end, (int, np.integer)) or int(max_since_end) < 1:
        raise ValueError("max_since_end must be an integer >= 1")
    generations = np.arange(1, int(horizon) + 1) if (isinstance(horizon, (int, np.integer)) and not isinstance(horizon, bool)
                                                    and int(horizon) >= 1) else None
    if generations is None:
        raise ValueError("horizon must be an integer >= 1")
    # the class probabilities given the coalescence generation do not depend on the candidate,
    # so they are evaluated once; the coalescence weights are evaluated per candidate
    mass = class_mass_given_time(lower, upper, generations, break_rate, marker_spacing, heterozygosity)
    best, best_gap = None, None
    for candidate in range(1, int(max_since_end) + 1):
        weights = coalescence_weights(n_ancestral, n_bottleneck, n_recent, bottleneck_length, candidate,
                                              mutation_rate, selection, chromosome_length, horizon)
        gap = abs(float(weights @ mass) - float(observed))
        if best is None or gap < best_gap:
            best, best_gap = candidate, gap
    return int(best)

import numpy as np


def sweep_frequency_trajectory(census: float, advantage: float, generations: int) -> "np.ndarray":
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    if not _num(census) or float(census) < 1.0:
        raise ValueError("census must be a finite number >= 1")
    if not _num(advantage) or float(advantage) <= 0.0:
        raise ValueError("advantage must be a finite number > 0")
    if isinstance(generations, bool) or not isinstance(generations, (int, np.integer)) or int(generations) < 0:
        raise ValueError("generations must be an integer >= 0")
    n, a = float(census), float(advantage)
    one_copy = 1.0 / (2.0 * n)
    threshold = 1.0 / (2.0 * n * a)          # the frequency at which selection overtakes the branching phase
    q = np.empty(int(generations) + 1)
    q[0] = one_copy
    for k in range(int(generations)):
        x = q[k]
        if x < threshold:
            q[k + 1] = x + one_copy            # branching phase: one more copy per generation
        else:
            q[k + 1] = x * (1.0 + a + a * x) / (1.0 + 2.0 * a * x)   # deterministic selection
        if not q[k + 1] < 1.0:
            raise ValueError("the expected frequency reaches 1 within the requested generations")
    return q

import numpy as np


def sweep_size_reduction(frequencies: "np.ndarray", census: float, recombination: float) -> "np.ndarray":
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    q = np.asarray(frequencies, dtype=float) if not isinstance(frequencies, (str, bytes)) else None
    if q is None or q.ndim != 1 or q.size < 2 or not np.all(np.isfinite(q)) or np.any(q <= 0.0) or np.any(q >= 1.0):
        raise ValueError("frequencies must be a one-dimensional array of at least two finite numbers in (0, 1)")
    if not _num(census) or float(census) < 1.0:
        raise ValueError("census must be a finite number >= 1")
    if not _num(recombination) or not 0.0 <= float(recombination) <= 0.5:
        raise ValueError("recombination must be a finite number in [0, 0.5]")
    r, two_n = float(recombination), 2.0 * float(census)
    a_now = q.size - 1
    q_now = q[a_now]
    factor = np.empty(a_now)
    for k in range(a_now):
        y, w = 1.0, 1.0                          # linked copies of generation k, among the carriers
        for t in range(k, a_now):
            y_next = y * (1.0 - (1.0 - q[t]) * r)
            w = w * (1.0 - 2.0 * (1.0 - q[t]) * r) + (y - w) / (two_n * q[t])
            y = y_next
        variance = w * q_now * q_now / q[k] + (1.0 - 2.0 * y * q_now + w * q_now * q_now) / (1.0 - q[k]) - 1.0
        factor[a_now - k - 1] = 1.0 / (1.0 + variance)     # g = A - k generations before the present
    return factor

import numpy as np


def focal_coalescence_weights(n_ancestral: float, n_bottleneck: float, n_recent: float, bottleneck_length: int,
                                      since_end: int, mutation_rate: float, selection: float, chromosome_length: float,
                                      advantage: float, recombination: float, sweep_age: int, horizon: int) -> "np.ndarray":
    def _size(v):
        return ((not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating))
                and bool(np.isfinite(float(v))) and float(v) >= 1.0)

    def _count(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, np.integer))

    if not (_size(n_ancestral) and _size(n_bottleneck) and _size(n_recent)):
        raise ValueError("n_ancestral, n_bottleneck and n_recent must be finite numbers >= 1")
    if not (_count(bottleneck_length) and int(bottleneck_length) >= 0):
        raise ValueError("bottleneck_length must be an integer >= 0")
    if not (_count(since_end) and int(since_end) >= 1):
        raise ValueError("since_end must be an integer >= 1")
    if not (_count(sweep_age) and 1 <= int(sweep_age) <= int(since_end)):
        raise ValueError("sweep_age must be an integer in [1, since_end]")
    if not (_count(horizon) and int(horizon) >= 1):
        raise ValueError("horizon must be an integer >= 1")
    g = np.arange(1, int(horizon) + 1)
    e, dd, a = int(since_end), int(bottleneck_length), int(sweep_age)
    size = np.where(g <= e, float(n_recent), np.where(g <= e + dd, float(n_bottleneck), float(n_ancestral)))
    background = np.empty(g.size)
    for census in (float(n_recent), float(n_bottleneck), float(n_ancestral)):   # each epoch's census sets its variance
        in_epoch = size == census
        if np.any(in_epoch):
            background[in_epoch] = selection_reduction_factor(g[in_epoch], mutation_rate, selection,
                                                                      chromosome_length, census)
    trajectory = sweep_frequency_trajectory(n_recent, advantage, a)
    sweep = np.ones(g.size)
    reduction = sweep_size_reduction(trajectory, n_recent, recombination)
    reach = min(a, g.size)
    sweep[:reach] = reduction[:reach]
    rate = 1.0 / (2.0 * size * background * sweep)
    if np.any(rate > 1.0):
        raise ValueError("the coalescence probability of some generation exceeds 1")
    # probability of no coalescence in the generations more recent than g, then coalescence at g
    not_yet = np.concatenate([[1.0], np.cumprod(1.0 - rate)[:-1]])
    return rate * not_yet

import numpy as np


def far_side_class_profile(lower: float, upper: float, n_ancestral: float, n_bottleneck: float, n_recent: float,
                                   bottleneck_length: int, since_end: int, mutation_rate: float, selection: float,
                                   chromosome_length: float, advantage: float, recombination: float, sweep_age: int,
                                   break_rate: float, marker_spacing: float, heterozygosity: float,
                                   horizon: int) -> "np.ndarray":
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and not np.isnan(float(v))

    if not (_num(lower) and _num(upper) and np.isfinite(float(lower)) and 0.0 <= float(lower) < float(upper)):
        raise ValueError("lower and upper must satisfy 0 <= lower < upper with lower finite")
    weights = focal_coalescence_weights(n_ancestral, n_bottleneck, n_recent, bottleneck_length, since_end,
                                                mutation_rate, selection, chromosome_length, advantage, recombination,
                                                sweep_age, horizon)
    lo, hi = float(lower), float(upper)
    open_class = not np.isfinite(hi)
    ends = np.array([lo]) if open_class else np.array([lo, hi])
    generations = np.arange(1, int(horizon) + 1)
    # the one-side survival after g generations is the survival after one generation to the power g, since the
    # observable breaks of different generations are independent
    one_generation = flank_survival_and_density(ends, 1, break_rate, marker_spacing, heterozygosity)[0]
    survival = one_generation[None, :] ** generations[:, None]
    mass = survival[:, 0] - (0.0 if open_class else survival[:, 1])
    contributions = weights * mass
    probability = float(contributions.sum())
    if probability <= 0.0:
        raise ValueError("the probability of the class within the horizon is zero")
    mean_generation = float((generations * contributions).sum() / probability)
    during = float(contributions[generations <= int(sweep_age)].sum() / probability)
    return np.array([probability, mean_generation, during])

import numpy as np


def sweep_age_from_far_side(observed: float, lower: float, upper: float, n_ancestral: float, n_bottleneck: float,
                                    n_recent: float, bottleneck_length: int, since_end: int, mutation_rate: float,
                                    selection: float, chromosome_length: float, advantage: float, recombination: float,
                                    break_rate: float, marker_spacing: float, heterozygosity: float, horizon: int) -> int:
    if (isinstance(observed, bool) or not isinstance(observed, (int, float, np.integer, np.floating))
            or not np.isfinite(float(observed)) or not 0.0 <= float(observed) <= 1.0):
        raise ValueError("observed must be a finite number in [0, 1]")
    if isinstance(since_end, bool) or not isinstance(since_end, (int, np.integer)) or int(since_end) < 1:
        raise ValueError("since_end must be an integer >= 1")
    best, best_gap = None, None
    for candidate in range(1, int(since_end) + 1):
        probability = far_side_class_profile(lower, upper, n_ancestral, n_bottleneck, n_recent,
                                                     bottleneck_length, since_end, mutation_rate, selection,
                                                     chromosome_length, advantage, recombination, candidate,
                                                     break_rate, marker_spacing, heterozygosity, horizon)[0]
        gap = abs(float(probability) - float(observed))
        if best is None or gap < best_gap:
            best, best_gap = candidate, gap
    return int(best)

import numpy as np


def swept_focal_mean_coalescence_time(n_ancestral: float, n_bottleneck: float, n_recent: float, bottleneck_length: int,
                                              mutation_rate_per_chromosome: float, selection_coefficient: float,
                                              chromosome_length_morgans: float,
                                              marker_spacing_kb: float, map_cM_per_Mb: float, heterozygosity: float,
                                              mutation_rate_per_bp: float, conversion_rate_per_bp: float,
                                              observed_class_cM: "np.ndarray", reported_size: float,
                                              advantage: float, recombination_fraction: float,
                                              far_observed_class_cM: "np.ndarray", far_observed_probability: float,
                                              far_report_class_cM: "np.ndarray", max_since_end: int, horizon: int) -> float:
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    def _class(c):
        cc = np.asarray(c, dtype=float) if not isinstance(c, (str, bytes)) else None
        if cc is None or cc.shape != (2,) or np.isnan(cc).any() or not np.isfinite(cc[0]) or not 0.0 <= cc[0] < cc[1]:
            raise ValueError("a length class must be a pair of numbers with 0 <= lower < upper")
        return float(cc[0]) / 100.0, float(cc[1]) / 100.0

    if not _num(marker_spacing_kb) or float(marker_spacing_kb) < 0.0:
        raise ValueError("marker_spacing_kb must be a finite number >= 0")
    if not _num(map_cM_per_Mb) or float(map_cM_per_Mb) <= 0.0:
        raise ValueError("map_cM_per_Mb must be a finite number > 0")
    if not _num(mutation_rate_per_bp) or float(mutation_rate_per_bp) < 0.0 or not _num(conversion_rate_per_bp) \
            or float(conversion_rate_per_bp) < 0.0:
        raise ValueError("mutation_rate_per_bp and conversion_rate_per_bp must be finite numbers >= 0")
    lo_obs, hi_obs = _class(observed_class_cM)
    lo_far, hi_far = _class(far_observed_class_cM)
    lo_rep, hi_rep = _class(far_report_class_cM)
    # laboratory units to map units: one Morgan is 100 cM, and map_cM_per_Mb cM span one megabase
    bp_per_morgan = 1.0e8 / float(map_cM_per_Mb)
    marker_spacing = float(marker_spacing_kb) * 1.0e3 / bp_per_morgan
    break_rate = (float(mutation_rate_per_bp) + float(conversion_rate_per_bp)) * bp_per_morgan
    observed_coverage = class_coverage_from_reported_size(reported_size, lo_obs, hi_obs, break_rate, marker_spacing,
                                                                  heterozygosity)
    since_end = bottleneck_end_from_class_coverage(observed_coverage, lo_obs, hi_obs, n_ancestral, n_bottleneck,
                                                           n_recent, bottleneck_length, mutation_rate_per_chromosome,
                                                           selection_coefficient, chromosome_length_morgans, break_rate,
                                                           marker_spacing, heterozygosity, horizon, max_since_end)
    sweep_age = sweep_age_from_far_side(far_observed_probability, lo_far, hi_far, n_ancestral, n_bottleneck,
                                                n_recent, bottleneck_length, since_end, mutation_rate_per_chromosome,
                                                selection_coefficient, chromosome_length_morgans, advantage,
                                                recombination_fraction, break_rate, marker_spacing, heterozygosity,
                                                horizon)
    profile = far_side_class_profile(lo_rep, hi_rep, n_ancestral, n_bottleneck, n_recent, bottleneck_length,
                                             since_end, mutation_rate_per_chromosome, selection_coefficient,
                                             chromosome_length_morgans, advantage, recombination_fraction, sweep_age,
                                             break_rate, marker_spacing, heterozygosity, horizon)
    return float(profile[1])
SCICODE_GOLD_EOF
