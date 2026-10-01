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


def length_to_weight(lengths: "numpy.ndarray", intercept: float = 1.02,
                             slope: float = 3.02) -> "numpy.ndarray":
    """Length-weight relation of Table 1: log10(weight in kg) = intercept + slope * log10(length in m)."""
    z = np.asarray(lengths, dtype=np.float64)
    if z.ndim != 1 or z.size < 1:
        raise ValueError("lengths must be a non-empty 1-D array")
    if not np.all(np.isfinite(z)) or np.any(z <= 0.0):
        raise ValueError("lengths must be finite and strictly positive")
    if not (np.isfinite(intercept) and np.isfinite(slope)):
        raise ValueError("intercept and slope must be finite")
    # log10, not ln: the source's own check "10 adult fish ~ 68 kg" only holds for log10
    return 10.0 ** (float(intercept) + float(slope) * np.log10(z))

import numpy as np


def survival_probability(lengths: "numpy.ndarray", slope: float = 2.7,
                                 exponent: float = -0.315) -> "numpy.ndarray":
    """Annual survival s(z) = exp(-slope * w^exponent) with w the body weight in grams."""
    z = np.asarray(lengths, dtype=np.float64)
    if z.ndim != 1 or z.size < 1:
        raise ValueError("lengths must be a non-empty 1-D array")
    if not np.all(np.isfinite(z)) or np.any(z <= 0.0):
        raise ValueError("lengths must be finite and strictly positive")
    if not (np.isfinite(slope) and np.isfinite(exponent)) or slope <= 0.0:
        raise ValueError("slope must be positive and finite, exponent finite")
    # Lorenzen-type mortality is parameterised in GRAMS; the length-weight relation returns kg
    grams = 1000.0 * length_to_weight(z)
    return np.exp(-float(slope) * grams ** float(exponent))

import numpy as np


def maturity_probability(lengths: "numpy.ndarray", intercept: float = -7.41,
                                 slope: float = 24.98) -> "numpy.ndarray":
    """Logistic maturity-at-length m(z) = 1 / (1 + exp(-(intercept + slope * z))), increasing in z."""
    z = np.asarray(lengths, dtype=np.float64)
    if z.ndim != 1 or z.size < 1:
        raise ValueError("lengths must be a non-empty 1-D array")
    if not np.all(np.isfinite(z)) or np.any(z < 0.0):
        raise ValueError("lengths must be finite and nonnegative")
    if not (np.isfinite(intercept) and np.isfinite(slope)):
        raise ValueError("intercept and slope must be finite")
    # increasing with length: 50% maturity at z = -intercept/slope = 0.2966 m. Table 1 prints the
    # logistic with the argument sign flipped, which would make large fish immature; the authors'
    # code and every result in the paper use the increasing form.
    return 1.0 / (1.0 + np.exp(-(float(intercept) + float(slope) * z)))

import numpy as np


def midpoint_mesh(z_min: float, z_max: float, n_bins: int) -> "numpy.ndarray":
    """Midpoint-rule mesh on [z_min, z_max]: row 0 the bin midpoints, row 1 the (constant) bin widths."""
    if not (np.isfinite(z_min) and np.isfinite(z_max)) or not z_max > z_min:
        raise ValueError("z_max must exceed z_min and both must be finite")
    if int(n_bins) != n_bins or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    n = int(n_bins)
    edges = np.linspace(float(z_min), float(z_max), n + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    widths = np.full(n, (float(z_max) - float(z_min)) / n)
    return np.vstack([mids, widths])

import numpy as np


def growth_kernel(midpoints: "numpy.ndarray", width: float, mu0: float = 0.194,
                          mu1: float = 0.841, sd: float = 0.1) -> "numpy.ndarray":
    """Midpoint-rule growth kernel G[i, j] = N(z_i; mu0 + mu1 z_j, sd) * width, no renormalisation."""
    z = np.asarray(midpoints, dtype=np.float64)
    if z.ndim != 1 or z.size < 1:
        raise ValueError("midpoints must be a non-empty 1-D array")
    if not np.all(np.isfinite(z)):
        raise ValueError("midpoints must be finite")
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("width must be positive and finite")
    if not (np.isfinite(mu0) and np.isfinite(mu1) and np.isfinite(sd)) or sd <= 0.0:
        raise ValueError("mu0 and mu1 must be finite, sd positive and finite")
    mean = float(mu0) + float(mu1) * z[None, :]                 # column j: source length z_j
    pdf = np.exp(-0.5 * ((z[:, None] - mean) / float(sd)) ** 2) / (float(sd) * np.sqrt(2.0 * np.pi))
    # the density times the bin width IS the midpoint rule for int G(z', z) f(z) dz; mass that the
    # normal places outside the domain is lost (no per-column renormalisation)
    return pdf * float(width)

import numpy as np


def recruit_length_density(midpoints: "numpy.ndarray", mean: float = 0.319,
                                   sd: float = 0.040) -> "numpy.ndarray":
    """Age-1 recruit length density C1(z) = N(z; mean, sd) evaluated at the midpoints."""
    z = np.asarray(midpoints, dtype=np.float64)
    if z.ndim != 1 or z.size < 1:
        raise ValueError("midpoints must be a non-empty 1-D array")
    if not np.all(np.isfinite(z)):
        raise ValueError("midpoints must be finite")
    if not (np.isfinite(mean) and np.isfinite(sd)) or sd <= 0.0:
        raise ValueError("mean must be finite, sd positive and finite")
    return np.exp(-0.5 * ((z - float(mean)) / float(sd)) ** 2) / (float(sd) * np.sqrt(2.0 * np.pi))

import numpy as np


def ricker_allee_target(biomass: float, gamma: float, allee_threshold: float,
                                carrying_capacity: float) -> float:
    """Eq 2 on biomass: b_{t+1} = b exp( gamma (1 - b/K) (b - C)/K )."""
    b, g, C, K = float(biomass), float(gamma), float(allee_threshold), float(carrying_capacity)
    if not all(np.isfinite([b, g, C, K])):
        raise ValueError("all inputs must be finite")
    if b < 0.0:
        raise ValueError("biomass must be nonnegative")
    if not (0.0 < C < K):
        raise ValueError("the Allee threshold must satisfy 0 < C < K")
    if g <= 0.0:
        raise ValueError("gamma must be positive")
    # BOTH factors are scaled by K: (1 - b/K) and (b - C)/K. The authors' code writes
    # gamma*(1 - b/K)*(b - C) with gamma = 1.837/K, which is the same expression.
    return b * np.exp(g * (1.0 - b / K) * ((b - C) / K))

import numpy as np


def spawning_proportion(density: "numpy.ndarray", mesh: "numpy.ndarray", growth: "numpy.ndarray",
                                survival: "numpy.ndarray", weights: "numpy.ndarray",
                                maturity: "numpy.ndarray", recruit_density: "numpy.ndarray",
                                recruits_per_spawner: int, target_biomass: float) -> float:
    """Spawning proportion A_t of the biomass balance b_{t+1} = s_{t+1} + A_t m_t b_R, clamped to [0, 1]."""
    n = np.asarray(density, dtype=np.float64)
    mesh = np.asarray(mesh, dtype=np.float64)
    G = np.asarray(growth, dtype=np.float64)
    arrs = [np.asarray(a, dtype=np.float64) for a in (survival, weights, maturity, recruit_density)]
    if n.ndim != 1 or n.size < 1:
        raise ValueError("density must be a non-empty 1-D array")
    m = n.size
    if mesh.shape != (2, m) or G.shape != (m, m) or any(a.shape != (m,) for a in arrs):
        raise ValueError("mesh, growth and the vital-rate arrays must match the density length")
    if not all(np.all(np.isfinite(a)) for a in (n, mesh, G, *arrs)):
        raise ValueError("all arrays must be finite")
    if np.any(n < 0.0) or np.any(mesh[1] <= 0.0) or np.any(G < 0.0) or any(np.any(a < 0.0) for a in arrs):
        raise ValueError("density, bin widths, growth and vital rates must be nonnegative")
    if int(recruits_per_spawner) != recruits_per_spawner or recruits_per_spawner < 1:
        raise ValueError("recruits_per_spawner must be a positive integer")
    if not np.isfinite(target_biomass) or target_biomass < 0.0:
        raise ValueError("target_biomass must be finite and nonnegative")
    s, W, mat, C1 = arrs
    w = mesh[1]
    # s_{t+1}: the survivors are weighed at the NEXT census, i.e. after growth (integrated over z')
    surviving_biomass = float(np.sum(W * (G @ (s * n)) * w))
    mature_total = float(np.sum(mat * n * w))                          # m_t
    recruit_biomass = int(recruits_per_spawner) * float(np.sum(W * C1 * w))   # b_R = int W(z') k C1(z') dz'
    deficit = float(target_biomass) - surviving_biomass               # recruit deficit, Fig 1 phase 2
    potential = mature_total * recruit_biomass                        # maximum recruit biomass m_t b_R
    if deficit < 0.0 or potential <= 0.0:
        return 0.0                                                    # survivors already exceed the target, or nobody can spawn
    if deficit > potential:
        return 1.0                                                    # even universal spawning cannot bridge it
    return deficit / potential

import numpy as np


def expected_trajectory(n0: float, mean_len0: float, sd_len0: float, n_years: int, n_bins: int,
                                gamma: float, allee_threshold: float, carrying_capacity: float,
                                recruits_per_spawner: int, recruit_mean: float, recruit_sd: float,
                                z_min: float = 0.01, z_max: float = 1.5) -> "numpy.ndarray":
    """Project the expected female length density n_years censuses forward with Eq 4; row t is census t."""
    if not all(np.isfinite([n0, mean_len0, sd_len0, gamma, allee_threshold, carrying_capacity,
                            recruit_mean, recruit_sd, z_min, z_max])):
        raise ValueError("all real-valued inputs must be finite")
    if n0 <= 0.0 or sd_len0 <= 0.0:
        raise ValueError("n0 and sd_len0 must be positive")
    if int(n_years) != n_years or n_years < 1:
        raise ValueError("n_years must be a positive integer")
    if int(recruits_per_spawner) != recruits_per_spawner or recruits_per_spawner < 1:
        raise ValueError("recruits_per_spawner must be a positive integer")
    mesh = midpoint_mesh(z_min, z_max, n_bins)
    z, w = mesh[0], mesh[1]
    weights = length_to_weight(z)                             # kg at the midpoints
    survival = survival_probability(z)
    maturity = maturity_probability(z)
    growth = growth_kernel(z, float(w[0]))                     # includes the bin width
    recruit = recruit_length_density(z, recruit_mean, recruit_sd)
    k = int(recruits_per_spawner)
    # introduced cohort: n0 females with lengths ~ N(mean_len0, sd_len0), as a density on the mesh
    n = float(n0) * np.exp(-0.5 * ((z - float(mean_len0)) / float(sd_len0)) ** 2) \
        / (float(sd_len0) * np.sqrt(2.0 * np.pi))
    out = np.empty((int(n_years) + 1, z.size))
    out[0] = n
    for t in range(int(n_years)):
        biomass_t = float(np.sum(weights * n * w))                    # b_t
        target = ricker_allee_target(biomass_t, gamma, allee_threshold, carrying_capacity)
        a_t = spawning_proportion(n, mesh, growth, survival, weights, maturity, recruit, k, target)
        mature_t = float(np.sum(maturity * n * w))                    # m_t
        n = growth @ (survival * n) + k * recruit * (mature_t * a_t)  # Eq 4: survival-growth + recruitment
        out[t + 1] = n
    return out

import numpy as np


def mature_count_probability(n0: float, mean_len0: float, sd_len0: float, n_years: int, n_bins: int,
                                     gamma: float, allee_threshold: float, carrying_capacity: float,
                                     recruits_per_spawner: int, recruit_mean: float, recruit_sd: float,
                                     threshold: int, z_min: float = 0.01, z_max: float = 1.5) -> float:
    """ORCHESTRATOR: exact probability that at least `threshold` mature females are present at census n_years
    in the source's discrete-individual branching process (Eq 3 outcomes; spawning proportions of the expected
    trajectory, Eq 4; founders = expected numbers per bin with the source's stochastic rounding of the
    fractional individual), by probability generating functions evaluated on the unit circle and inverted
    with the FFT."""
    if int(threshold) != threshold or threshold < 0:
        raise ValueError("threshold must be a nonnegative integer")
    traj = expected_trajectory(n0, mean_len0, sd_len0, n_years, n_bins, gamma, allee_threshold,
                                       carrying_capacity, recruits_per_spawner, recruit_mean, recruit_sd,
                                       z_min, z_max)
    mesh = midpoint_mesh(z_min, z_max, n_bins)
    z, w = mesh[0], mesh[1]
    weights = length_to_weight(z)
    survival = survival_probability(z)
    maturity = maturity_probability(z)
    growth = growth_kernel(z, float(w[0]))
    recruit = recruit_length_density(z, recruit_mean, recruit_sd)
    k = int(recruits_per_spawner)
    n_years = int(n_years)
    # spawning proportion of each year, frozen at its expected-trajectory value
    a = np.empty(n_years)
    for t in range(n_years):
        target = ricker_allee_target(float(np.sum(weights * traj[t] * w)), gamma,
                                             allee_threshold, carrying_capacity)
        a[t] = spawning_proportion(traj[t], mesh, growth, survival, weights, maturity, recruit, k, target)
    c = recruit * w                                                   # bin probabilities of a recruit's length
    lost = np.maximum(0.0, 1.0 - growth.sum(axis=0))                  # survivor mass leaving the mesh: no descendants
    # number of evaluation points on the unit circle: a power of two far beyond any count with non-negligible probability
    n_pts = 1 << int(np.ceil(np.log2(max(1024.0, 8.0 * float(np.sum(traj[n_years] * w)) + 256.0))))
    u = np.exp(2j * np.pi * np.arange(n_pts) / n_pts)
    # F_t(u; z): generating function of the number of mature descendants (herself included) at the last census
    # of one female of length z present at census t; at the last census maturity is a Bernoulli event
    F = (1.0 - maturity)[:, None] + maturity[:, None] * u[None, :]
    for t in range(n_years - 1, -1, -1):
        p = maturity * a[t]                                           # probability of reproducing this year
        GF = growth.T @ F + lost[:, None]                             # survivor: average over her next length
        CF = c @ F                                                    # one recruit: average over its length
        # survival and reproduction are independent (Eq 3), the k recruit lengths are iid: the pgf factorises
        F = (1.0 - survival[:, None] + survival[:, None] * GF) * (1.0 - p[:, None] + p[:, None] * CF[None, :] ** k)
    # founders: the expected number in each bin split into whole individuals and a fractional individual that is
    # present with probability equal to the fraction (the source's stochastic rounding); independent lineages
    expected = traj[0] * w
    whole = np.floor(expected)
    frac = expected - whole
    phi = np.ones(n_pts, dtype=complex)
    for j in range(z.size):
        if whole[j] > 0.0 or frac[j] > 0.0:
            phi *= F[j] ** int(whole[j]) * (1.0 - frac[j] + frac[j] * F[j])
    probs = np.fft.fft(phi).real / n_pts                              # P(count = n) for n = 0 .. n_pts - 1
    return float(np.sum(probs[int(threshold):]))
SCICODE_GOLD_EOF
