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


def compose_detector_response(
    redistribution: np.ndarray,
    resolution: np.ndarray,
) -> np.ndarray:
    d = np.asarray(redistribution, dtype=float)
    g = np.asarray(resolution, dtype=float)

    for name, m in (("redistribution", d), ("resolution", g)):
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError(f"{name} must be a square 2D array")
        if m.shape[0] == 0:
            raise ValueError(f"{name} must act on at least one bin")
        if not np.all(np.isfinite(m)):
            raise ValueError(f"{name} must be finite")
        if np.any(m < 0.0):
            raise ValueError(f"{name} must be non-negative")
    if np.any(d.sum(axis=0) <= 0.0) or np.any(d.sum(axis=0) > 1.0 + 1e-6):
        raise ValueError("every column of redistribution must sum to an efficiency in (0, 1]")
    if not np.allclose(g.sum(axis=0), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("every column of resolution must sum to one")
    if d.shape != g.shape:
        raise ValueError("redistribution and resolution must share one grid")

    # Deposition happens before the deposited energy is broadened, so the
    # redistribution acts first and the resolution acts on its output.
    return g @ d

import numpy as np


def build_background_reference(
    off_counts: np.ndarray,
    prior_shape: float,
) -> np.ndarray:
    n_off = np.asarray(off_counts, dtype=float)
    a0 = float(prior_shape)

    if n_off.ndim != 1 or n_off.size == 0:
        raise ValueError("off_counts must be a non-empty 1D array")
    if not np.all(np.isfinite(n_off)):
        raise ValueError("off_counts must be finite")
    if np.any(n_off < 0.0) or np.any(n_off != np.round(n_off)):
        raise ValueError("off_counts must hold non-negative integer values")
    if not np.any(n_off > 0.0):
        raise ValueError("the background run must hold at least one count")
    if not np.isfinite(a0) or a0 <= 0.0:
        raise ValueError("prior_shape must be a finite positive number")

    # The Gamma rate is set so that the prior mean equals the average OFF count.
    rate = a0 / float(np.mean(n_off))
    # Gamma-Poisson conjugacy: the posterior of one bin is Gamma(a0 + n_off, rate + 1),
    # whose mean is the reference used inside the iteration.
    return (a0 + n_off) / (rate + 1.0)

import numpy as np


def run_richardson_lucy_iterates(
    on_counts: np.ndarray,
    response: np.ndarray,
    background_reference: np.ndarray,
    n_iterations: int,
    guard: float,
) -> np.ndarray:
    n_on = np.asarray(on_counts, dtype=float)
    r = np.asarray(response, dtype=float)
    b_ref = np.asarray(background_reference, dtype=float)

    if n_on.ndim != 1 or n_on.size == 0:
        raise ValueError("on_counts must be a non-empty 1D array")
    n_bins = n_on.size
    if r.ndim != 2 or r.shape != (n_bins, n_bins):
        raise ValueError("response must have shape (J, J) matching on_counts")
    if b_ref.ndim != 1 or b_ref.size != n_bins:
        raise ValueError("background_reference must have shape (J,) matching on_counts")
    for name, arr in (("on_counts", n_on), ("response", r), ("background_reference", b_ref)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite")
        if np.any(arr < 0.0):
            raise ValueError(f"{name} must be non-negative")
    if np.any(n_on != np.round(n_on)):
        raise ValueError("on_counts must hold integer values")
    sensitivity = r.sum(axis=0)
    if np.any(sensitivity <= 0.0) or np.any(sensitivity > 1.0 + 1e-6):
        raise ValueError("every column of response must sum to an efficiency in (0, 1]")
    if int(n_iterations) != n_iterations or n_iterations < 0:
        raise ValueError("n_iterations must be a non-negative integer")
    eps = float(guard)
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("guard must be a finite non-negative number")

    net = float(n_on.sum() - b_ref.sum())
    if net <= 0.0:
        raise ValueError("the signal run must hold more counts than the background reference")

    n_it = int(n_iterations)
    iterates = np.empty((n_it + 1, n_bins), dtype=float)
    estimate = np.full(n_bins, net / n_bins, dtype=float)
    iterates[0] = estimate
    for t in range(1, n_it + 1):
        predicted = r @ estimate + b_ref
        estimate = estimate / sensitivity * (r.T @ (n_on / (predicted + eps)))
        iterates[t] = estimate
    return iterates

import numpy as np


def select_reference_iteration(
    on_counts: np.ndarray,
    resampled_counts: np.ndarray,
    response: np.ndarray,
    resolution: np.ndarray,
    background_reference: np.ndarray,
    window: int,
    ratio_threshold: float,
    run_length: int,
    max_iterations: int,
    guard: float,
) -> int:
    n_on = np.asarray(on_counts, dtype=float)
    res = np.asarray(resampled_counts, dtype=float)
    g = np.asarray(resolution, dtype=float)

    if n_on.ndim != 1 or n_on.size == 0:
        raise ValueError("on_counts must be a non-empty 1D array")
    n_bins = n_on.size
    if res.ndim != 2 or res.shape[1] != n_bins:
        raise ValueError("resampled_counts must have shape (K, J) matching on_counts")
    if res.shape[0] < 2:
        raise ValueError("at least two resamples are needed for a sample variance")
    if not np.all(np.isfinite(res)) or np.any(res < 0.0) or np.any(res != np.round(res)):
        raise ValueError("resampled_counts must hold non-negative integer values")
    if g.ndim != 2 or g.shape != (n_bins, n_bins):
        raise ValueError("resolution must have shape (J, J) matching on_counts")
    if not np.all(np.isfinite(g)) or np.any(g < 0.0):
        raise ValueError("resolution must be finite and non-negative")
    for name, val in (("window", window), ("run_length", run_length),
                      ("max_iterations", max_iterations)):
        if int(val) != val or val < 1:
            raise ValueError(f"{name} must be a positive integer")
    tau = float(ratio_threshold)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("ratio_threshold must be a finite positive number")
    eps = float(guard)
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("guard must be a finite non-negative number")

    w, run, t_max = int(window), int(run_length), int(max_iterations)

    # The nominal run, monitored in the resolution-limited space.
    iterates = run_richardson_lucy_iterates(
        n_on, response, background_reference, t_max, eps
    )
    eta = iterates @ g.T
    change = np.full(t_max + 1, np.nan)
    for t in range(w, t_max + 1):
        change[t] = np.linalg.norm(eta[t] - eta[t - w]) / (np.linalg.norm(eta[t]) + eps)

    # The resampled runs, whose spread at each iteration is the noise level.
    eta_res = np.empty((res.shape[0], t_max + 1, n_bins), dtype=float)
    for k in range(res.shape[0]):
        eta_res[k] = run_richardson_lucy_iterates(
            res[k], response, background_reference, t_max, eps
        ) @ g.T
    mean_eta = eta_res.mean(axis=0)
    var_eta = eta_res.var(axis=0, ddof=1)
    noise = np.sqrt(var_eta.sum(axis=1)) / (np.linalg.norm(mean_eta, axis=1) + eps)
    ratio = change / (noise + eps)

    # Earliest candidate that starts a full run of qualifying iterations.
    for t in range(w, t_max - run + 2):
        block = ratio[t:t + run]
        if np.all(np.isfinite(block)) and np.all(block <= tau):
            return int(t)
    return int(t_max)

import numpy as np


def build_prior_centre_and_width(
    reference_iterate: np.ndarray,
    resolution: np.ndarray,
    floor: float,
    sigma_min: float,
    sigma_max: float,
    count_scale: float,
) -> np.ndarray:
    mu_raw = np.asarray(reference_iterate, dtype=float)
    g = np.asarray(resolution, dtype=float)
    floor_v, s_min, s_max, c_ref = map(float, (floor, sigma_min, sigma_max, count_scale))

    if mu_raw.ndim != 1 or mu_raw.size == 0:
        raise ValueError("reference_iterate must be a non-empty 1D array")
    n_bins = mu_raw.size
    if g.ndim != 2 or g.shape != (n_bins, n_bins):
        raise ValueError("resolution must have shape (J, J) matching reference_iterate")
    for name, arr in (("reference_iterate", mu_raw), ("resolution", g)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite")
        if np.any(arr < 0.0):
            raise ValueError(f"{name} must be non-negative")
    if not np.allclose(g.sum(axis=0), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("every column of resolution must sum to one")
    if not np.isfinite(floor_v) or floor_v <= 0.0:
        raise ValueError("floor must be a finite positive number")
    if not np.isfinite(s_min) or s_min <= 0.0:
        raise ValueError("sigma_min must be a finite positive number")
    if not np.isfinite(s_max) or s_max < s_min:
        raise ValueError("sigma_max must be finite and at least sigma_min")
    if not np.isfinite(c_ref) or c_ref <= 0.0:
        raise ValueError("count_scale must be a finite positive number")

    # The floor is applied first, and the resolution-limited reference is built
    # from the floored centre.
    centre = np.maximum(mu_raw, floor_v)
    eta_ref = g @ centre
    eta_mean = float(np.mean(eta_ref))

    # Shape-dependent schedule, narrow where the reference is prominent.
    sigma_shape = s_min + (s_max - s_min) / (1.0 + eta_ref / eta_mean)
    # Local activation by absolute count level, half activated at count_scale.
    activation = eta_ref / (eta_ref + c_ref)
    sigma = (1.0 - activation) * s_max + activation * sigma_shape
    return np.vstack([centre, sigma])

import numpy as np


def draw_prior_spectra(
    centre: np.ndarray,
    width: np.ndarray,
    gamma_shape: float,
    normal_draws: np.ndarray,
    uniform_draws: np.ndarray,
    resolution: np.ndarray,
) -> np.ndarray:
    from scipy.special import gammaincinv

    mu_c = np.asarray(centre, dtype=float)
    sig = np.asarray(width, dtype=float)
    z = np.asarray(normal_draws, dtype=float)
    u = np.asarray(uniform_draws, dtype=float)
    g = np.asarray(resolution, dtype=float)
    alpha = float(gamma_shape)

    if mu_c.ndim != 1 or mu_c.size == 0:
        raise ValueError("centre must be a non-empty 1D array")
    n_bins = mu_c.size
    if sig.shape != (n_bins,):
        raise ValueError("width must have shape (J,) matching centre")
    if z.ndim != 2 or z.shape[1] != n_bins or z.shape[0] == 0:
        raise ValueError("normal_draws must have shape (N, J) with at least one draw")
    if u.shape != z.shape:
        raise ValueError("uniform_draws must have the same shape as normal_draws")
    if g.ndim != 2 or g.shape != (n_bins, n_bins):
        raise ValueError("resolution must have shape (J, J) matching centre")
    if not np.all(np.isfinite(mu_c)) or np.any(mu_c <= 0.0):
        raise ValueError("centre must be finite and strictly positive")
    if not np.all(np.isfinite(sig)) or np.any(sig <= 0.0):
        raise ValueError("width must be finite and strictly positive")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("gamma_shape must be a finite positive number")
    if not np.all(np.isfinite(z)):
        raise ValueError("normal_draws must be finite")
    if not np.all(np.isfinite(u)) or np.any(u <= 0.0) or np.any(u >= 1.0):
        raise ValueError("uniform_draws must lie strictly inside the unit interval")
    if not np.all(np.isfinite(g)) or np.any(g < 0.0):
        raise ValueError("resolution must be finite and non-negative")
    if not np.allclose(g.sum(axis=0), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("every column of resolution must sum to one")

    # Non-centred lognormal layer, mean-preserving around the centre.
    log_m = np.log(mu_c) - 0.5 * sig ** 2 + sig * z
    latent_mean = np.exp(log_m)
    # Conditional Gamma(alpha, alpha / m) by inverse-CDF at the supplied uniform.
    emitted = latent_mean * gammaincinv(alpha, u) / alpha
    # Report in the resolution-limited space.
    return emitted @ g.T

import numpy as np


def _midranks(values):
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="mergesort")
    sorted_v = v[order]
    ranks = np.empty(v.size, dtype=float)
    start = 0
    while start < v.size:
        stop = start + 1
        while stop < v.size and sorted_v[stop] == sorted_v[start]:
            stop += 1
        # Tied values share the mean of the one-based ranks they span.
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def build_global_rank_envelope(
    draws: np.ndarray,
    mass: float,
) -> np.ndarray:
    x = np.asarray(draws, dtype=float)
    m = float(mass)

    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("draws must have shape (N, J) with at least two draws")
    if not np.all(np.isfinite(x)):
        raise ValueError("draws must be finite")
    if not np.isfinite(m) or m <= 0.0 or m > 1.0:
        raise ValueError("mass must lie in (0, 1]")

    n_draws, n_bins = x.shape
    two_sided = np.empty((n_draws, n_bins), dtype=float)
    for j in range(n_bins):
        r = _midranks(x[:, j])
        two_sided[:, j] = np.minimum(r, (n_draws + 1) - r)

    # Extreme rank length: sort each curve's two-sided ranks, then order the
    # curves lexicographically, the smallest sorted vector being the most
    # extreme. lexsort takes its primary key last.
    sorted_ranks = np.sort(two_sided, axis=1)
    keys = tuple(sorted_ranks[:, c] for c in range(n_bins - 1, -1, -1))
    order = np.lexsort(keys)

    n_keep = int(np.ceil(m * n_draws))
    n_keep = max(1, min(n_draws, n_keep))
    kept = x[order[n_draws - n_keep:], :]

    centre = x.mean(axis=0)
    return np.vstack([centre, kept.min(axis=0), kept.max(axis=0)])

import numpy as np


def orchestrate_prior_diagnostic(
    redistribution: np.ndarray,
    resolution: np.ndarray,
    on_counts: np.ndarray,
    off_counts: np.ndarray,
    resampled_counts: np.ndarray,
    emitted_truth: np.ndarray,
    normal_draws: np.ndarray,
    uniform_draws: np.ndarray,
    background_prior_shape: float,
    window: int,
    ratio_threshold: float,
    run_length: int,
    max_iterations: int,
    floor: float,
    sigma_min: float,
    sigma_max: float,
    count_scale: float,
    gamma_shape: float,
    mass: float,
    guard: float,
) -> np.ndarray:
    g = np.asarray(resolution, dtype=float)
    mu_true = np.asarray(emitted_truth, dtype=float)
    eps = float(guard)

    response = compose_detector_response(redistribution, resolution)
    n_bins = response.shape[0]
    if mu_true.shape != (n_bins,):
        raise ValueError("emitted_truth must have shape (J,) matching the operators")
    if not np.all(np.isfinite(mu_true)) or np.any(mu_true < 0.0):
        raise ValueError("emitted_truth must be finite and non-negative")

    background_reference = build_background_reference(
        off_counts, background_prior_shape
    )
    selected = select_reference_iteration(
        on_counts, resampled_counts, response, resolution, background_reference,
        window, ratio_threshold, run_length, max_iterations, guard,
    )
    iterates = run_richardson_lucy_iterates(
        on_counts, response, background_reference, selected, guard
    )
    centre_and_width = build_prior_centre_and_width(
        iterates[selected], resolution, floor, sigma_min, sigma_max, count_scale
    )
    draws = draw_prior_spectra(
        centre_and_width[0], centre_and_width[1], gamma_shape,
        normal_draws, uniform_draws, resolution,
    )
    envelope = build_global_rank_envelope(draws, mass)

    # The truth is compared in the resolution-limited space.
    eta_true = g @ mu_true
    half_width = np.maximum(0.5 * (envelope[2] - envelope[1]), eps)
    scaled = (envelope[0] - eta_true) / half_width

    out = np.array([
        float(np.mean(np.abs(scaled))),
        float(selected),
        float(np.max(np.abs(scaled))),
    ], dtype=float)
    if not np.all(np.isfinite(out)):
        raise ValueError("computed diagnostics must be finite")
    return out
SCICODE_GOLD_EOF
