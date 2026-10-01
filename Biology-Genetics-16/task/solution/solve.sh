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

def _standard_normal_quantile(prob: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF, vectorised, without a SciPy dependency."""
    from math import sqrt
    try:
        from scipy.special import ndtri
        return np.asarray(ndtri(prob), dtype=float)
    except Exception:
        from math import erf
        p = np.asarray(prob, dtype=float)
        lo, hi = np.full(p.shape, -40.0), np.full(p.shape, 40.0)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            cdf = 0.5 * (1.0 + np.vectorize(erf)(mid / sqrt(2.0)))
            hi = np.where(cdf > p, mid, hi)
            lo = np.where(cdf > p, lo, mid)
        return 0.5 * (lo + hi)


# ORACLE FUNCTION

def simulate_base_population(n_loci: int, n_individuals: int, rho: float,
                                     p_low: float, p_high: float,
                                     seed: int) -> np.ndarray:
    if not (isinstance(n_loci, (int, np.integer)) and int(n_loci) >= 2):
        raise ValueError("n_loci must be an integer >= 2")
    if not (isinstance(n_individuals, (int, np.integer)) and int(n_individuals) >= 2):
        raise ValueError("n_individuals must be an integer >= 2")
    if isinstance(rho, bool) or not isinstance(rho, (int, float, np.floating, np.integer)):
        raise ValueError("rho must be a real number in [0, 1)")
    if not (0.0 <= float(rho) < 1.0):
        raise ValueError("rho must be in [0, 1)")
    for name, val in (("p_low", p_low), ("p_high", p_high)):
        if isinstance(val, bool) or not isinstance(val, (int, float, np.floating, np.integer)):
            raise ValueError(f"{name} must be a real number in (0, 1)")
        if not (0.0 < float(val) < 1.0):
            raise ValueError(f"{name} must be in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    n_loci, n_individuals = int(n_loci), int(n_individuals)
    rho = float(rho)
    rng = np.random.default_rng(int(seed))
    Z = rng.standard_normal((2 * n_individuals, n_loci))
    X = np.empty_like(Z)
    X[:, 0] = Z[:, 0]
    scale = np.sqrt(1.0 - rho * rho)
    for i in range(1, n_loci):
        X[:, i] = rho * X[:, i - 1] + scale * Z[:, i]
    targets = np.linspace(float(p_low), float(p_high), n_loci)
    thresholds = _standard_normal_quantile(1.0 - targets)
    return (X > thresholds[None, :]).astype(np.int64)

import numpy as np


def individual_diversity_matrix(haplotypes: np.ndarray) -> np.ndarray:
    H = np.asarray(haplotypes)

    if H.ndim != 2:
        raise ValueError("haplotypes must be a 2D array")
    if H.shape[0] < 4:
        raise ValueError(
            "haplotypes must contain at least 2 individuals (4 rows)"
        )
    if H.shape[0] % 2 != 0:
        raise ValueError("haplotypes must have an even number of rows")
    if H.shape[1] < 1:
        raise ValueError("haplotypes must have at least one locus")

    Hf = np.asarray(H, dtype=float)

    if not np.all((Hf == 0.0) | (Hf == 1.0)):
        raise ValueError("haplotypes must contain only 0 and 1")

    C = 0.5 * (Hf[0::2] + Hf[1::2])
    Cc = C - C.mean(axis=0, keepdims=True)

    return (Cc.T @ Cc) / C.shape[0]

import numpy as np


def pairwise_recombination_matrix(interval_rates: np.ndarray) -> np.ndarray:
    r = np.asarray(interval_rates, dtype=float)
    if r.ndim != 1:
        raise ValueError("interval_rates must be a 1D array")
    if r.size < 1:
        raise ValueError("interval_rates must contain at least one interval")
    if not np.all(np.isfinite(r)):
        raise ValueError("interval_rates must be finite")
    if np.any(r < 0.0) or np.any(r >= 0.5):
        raise ValueError("every interval rate must lie in [0, 0.5)")

    d = np.concatenate([[0.0], np.cumsum(-0.5 * np.log1p(-2.0 * r))])
    M = np.abs(d[:, None] - d[None, :])
    return 0.5 * (1.0 - np.exp(-2.0 * M))

import numpy as np


# HELPER FUNCTIONS

def _phase_components(haplotypes: np.ndarray):
    """Split the allele-content covariance into gametic and non-gametic parts.

    With a and b the two gamete sets of the population, both centred on the
    POOLED reference-allele frequency (the labels a and b are arbitrary, so
    centring them separately would leak a finite-sample artefact into the split):

        L'  = (1/4) [Cov(a_i, a_j) + Cov(b_i, b_j)]
        L'' = (1/4) [Cov(a_i, b_j) + Cov(b_i, a_j)]

    so that L' + L'' is exactly the covariance of the allele proportions, and
    L'[i, i] = p_i q_i / 2 exactly.
    """
    Hf = np.asarray(haplotypes, dtype=float)
    p = Hf.mean(axis=0, keepdims=True)
    a, b = Hf[0::2] - p, Hf[1::2] - p
    n = a.shape[0]
    gametic = 0.25 * (a.T @ a + b.T @ b) / n
    cross = 0.25 * (a.T @ b + b.T @ a) / n
    return gametic, cross


# ORACLE FUNCTION

def weighted_base_diversity_matrix(haplotypes: np.ndarray,
                                           recombination: np.ndarray) -> np.ndarray:
    H = np.asarray(haplotypes)
    if H.ndim != 2 or H.shape[0] < 4 or H.shape[0] % 2 != 0:
        raise ValueError("haplotypes must be 2D with an even number of rows >= 4")
    Hf = np.asarray(H, dtype=float)
    if not np.all((Hf == 0.0) | (Hf == 1.0)):
        raise ValueError("haplotypes must contain only 0 and 1")
    R = np.asarray(recombination, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("recombination must be a square 2D array")
    if R.shape[0] != Hf.shape[1]:
        raise ValueError("recombination size must match the number of loci")
    if not np.allclose(R, R.T, rtol=0.0, atol=1e-12):
        raise ValueError("recombination must be symmetric within atol=1e-12")
    if not np.all(np.isfinite(R)) or np.any(R < 0.0) or np.any(R >= 0.5):
        raise ValueError("recombination entries must lie in [0, 0.5)")

    gametic, cross = _phase_components(Hf)
    return gametic + (R / (1.0 - R)) * cross

import numpy as np


def variance_effective_size(census_size: int, offspring_variance: float) -> float:
    if isinstance(census_size, bool) or not isinstance(census_size, (int, np.integer)):
        raise ValueError("census_size must be an integer >= 1")
    if int(census_size) < 1:
        raise ValueError("census_size must be an integer >= 1")
    if isinstance(offspring_variance, bool) or not isinstance(
            offspring_variance, (int, float, np.floating, np.integer)):
        raise ValueError("offspring_variance must be a finite real number >= 0")
    v = float(offspring_variance)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError("offspring_variance must be a finite real number >= 0")
    return float(4.0 * int(census_size) / (2.0 + v))

import numpy as np


def expected_change_operator(base_matrix: np.ndarray, recombination: np.ndarray,
                                     n_generations: int, founding_effective_size: float,
                                     inbreeding_effective_size: float) -> np.ndarray:
    L = np.asarray(base_matrix, dtype=float)
    R = np.asarray(recombination, dtype=float)
    for name, A in (("base_matrix", L), ("recombination", R)):
        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
            raise ValueError(f"{name} must be a square 2D array")
        if not np.allclose(A, A.T, rtol=0.0, atol=1e-12):
            raise ValueError(f"{name} must be symmetric within atol=1e-12")
    if L.shape != R.shape:
        raise ValueError("base_matrix and recombination must have the same shape")
    if not np.all(np.isfinite(R)) or np.any(R < 0.0) or np.any(R >= 0.5):
        raise ValueError("recombination entries must lie in [0, 0.5)")
    if isinstance(n_generations, bool) or not isinstance(n_generations, (int, np.integer)):
        raise ValueError("n_generations must be an integer >= 1")
    if int(n_generations) < 1:
        raise ValueError("n_generations must be an integer >= 1")
    sizes = []
    for name, value in (("founding_effective_size", founding_effective_size),
                        ("inbreeding_effective_size", inbreeding_effective_size)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
            raise ValueError(f"{name} must be a finite number > 0.5")
        if not np.isfinite(float(value)) or float(value) <= 0.5:
            raise ValueError(f"{name} must be a finite number > 0.5")
        sizes.append(float(value))
    ne0, ne = sizes

    # The window opens in the founding generation, one round of mating after the
    # base, so it spans t = 1 .. n_generations; the prediction N_t o L_tilde of
    # the paper's Eq. 7 holds only for t > 0. Its drift factor is a product over
    # the rounds k < t, and round k = 0 is the founding bottleneck.
    founding = 1.0 - 1.0 / (2.0 * ne0)
    drift = 1.0 - 1.0 / (2.0 * ne)
    out = np.zeros_like(L)
    for t in range(1, int(n_generations) + 1):
        out = out + ((1.0 - R) ** t) * founding * (drift ** (t - 1)) * L
    return out

import numpy as np

def drift_covariance_matrix(base_matrix: np.ndarray, recombination: np.ndarray,
                                    n_generations: int, founding_effective_size: float,
                                    inbreeding_effective_size: float,
                                    n_effective: float) -> np.ndarray:
    L = np.asarray(base_matrix, dtype=float)
    R = np.asarray(recombination, dtype=float)
    for name, A in (("base_matrix", L), ("recombination", R)):
        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
            raise ValueError(f"{name} must be a square 2D array")
        if not np.allclose(A, A.T, rtol=0.0, atol=1e-12):
            raise ValueError(f"{name} must be symmetric within atol=1e-12")
    if L.shape != R.shape:
        raise ValueError("base_matrix and recombination must have the same shape")
    if not np.all(np.isfinite(R)) or np.any(R < 0.0) or np.any(R >= 0.5):
        raise ValueError("recombination entries must lie in [0, 0.5)")
    if isinstance(n_generations, bool) or not isinstance(n_generations, (int, np.integer)):
        raise ValueError("n_generations must be an integer >= 1")
    if int(n_generations) < 1:
        raise ValueError("n_generations must be an integer >= 1")
    sizes = []
    for name, value in (("founding_effective_size", founding_effective_size),
                        ("inbreeding_effective_size", inbreeding_effective_size)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
            raise ValueError(f"{name} must be a finite number > 0.5")
        if not np.isfinite(float(value)) or float(value) <= 0.5:
            raise ValueError(f"{name} must be a finite number > 0.5")
        sizes.append(float(value))
    ne0, ne = sizes
    if isinstance(n_effective, bool) or not isinstance(
            n_effective, (int, float, np.floating, np.integer)):
        raise ValueError("n_effective must be a finite number > 0")
    nE = float(n_effective)
    if not np.isfinite(nE) or nE <= 0.0:
        raise ValueError("n_effective must be a finite number > 0")

    # Same window and drift factor as expected_change_operator: generations
    # t = 1 .. n_generations, founding bottleneck in round k = 0.
    founding = 1.0 - 1.0 / (2.0 * ne0)
    drift = 1.0 - 1.0 / (2.0 * ne)
    total = np.zeros_like(L)
    for t in range(1, int(n_generations) + 1):
        total = total + ((1.0 - R) / nE) * ((1.0 - R) ** t) * founding * (drift ** (t - 1))
    return L * total

import numpy as np

def genic_variance_from_change(delta_p: np.ndarray,
                                       diversity_matrix: np.ndarray) -> float:
    dp = np.asarray(delta_p, dtype=float)
    L = np.asarray(diversity_matrix, dtype=float)
    if dp.ndim != 1 or dp.size < 1:
        raise ValueError("delta_p must be a non-empty 1D array")
    if not np.all(np.isfinite(dp)):
        raise ValueError("delta_p must be finite")
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("diversity_matrix must be a square 2D array")
    if L.shape[0] != dp.size:
        raise ValueError("diversity_matrix size must match delta_p")
    d = np.diag(L)
    if np.any(d < 0.0):
        raise ValueError("diagonal entries of diversity_matrix must be >= 0")
    zero = d == 0.0
    if np.any(zero & (dp != 0.0)):
        raise ValueError("a monomorphic locus cannot have a non-zero delta_p")
    keep = ~zero
    return float(np.sum(dp[keep] ** 2 / d[keep]))

import numpy as np

def average_effect_scale(delta_p_by_replicate: np.ndarray,
                                 operators: list, drift_covariances: list,
                                 contrast: np.ndarray) -> float:
    Y = np.asarray(delta_p_by_replicate, dtype=float)
    if Y.ndim != 2 or Y.shape[0] < 1 or Y.shape[1] < 1:
        raise ValueError("delta_p_by_replicate must be a 2D array with >= 1 row")
    if not np.all(np.isfinite(Y)):
        raise ValueError("delta_p_by_replicate must be finite")
    m_rep, n_loci = Y.shape
    if not isinstance(operators, list) or len(operators) != m_rep:
        raise ValueError("operators must be a list with one matrix per replicate")
    if not isinstance(drift_covariances, list) or len(drift_covariances) != m_rep:
        raise ValueError("drift_covariances must be a list with one matrix per replicate")
    x = np.asarray(contrast, dtype=float)
    if x.ndim != 1 or x.size != n_loci or not np.all(np.isfinite(x)):
        raise ValueError("contrast must be a finite 1D array with one entry per locus")

    numerator = 0.0
    information = 0.0
    for m in range(m_rep):
        Lm = np.asarray(operators[m], dtype=float)
        Dm = np.asarray(drift_covariances[m], dtype=float)
        for name, A in (("operator", Lm), ("drift covariance", Dm)):
            if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] != n_loci:
                raise ValueError(f"every {name} must be square with size n_loci")
            if not np.all(np.isfinite(A)):
                raise ValueError(f"every {name} must be finite")
        if not np.allclose(Dm, Dm.T, rtol=0.0, atol=1e-12):
            raise ValueError("every drift covariance must be symmetric within atol=1e-12")
        # LAPACK's Cholesky accepts a numerically singular matrix whose smallest
        # factor diagonal merely underflows, which would silently return a finite
        # but meaningless solve, so test conditioning directly and scale-invariantly.
        eig = np.linalg.eigvalsh(Dm)
        if eig[-1] <= 0.0 or eig[0] <= 1e-12 * eig[-1]:
            raise ValueError("every drift covariance must be positive definite "
                             "with condition number below 1e12")
        chol = np.linalg.cholesky(Dm)
        z = Lm @ x
        w = np.linalg.solve(chol.T, np.linalg.solve(chol, np.column_stack([z, Y[m]])))
        numerator += float(z @ w[:, 1])
        information += float(z @ w[:, 0])
    if information == 0.0:
        raise ValueError("accumulated information is zero; the coefficient is not identified")
    return float(numerator / information)

import numpy as np

def additive_genetic_variance_for_fitness(delta_p_by_replicate: np.ndarray,
                                                  census_sizes: list,
                                                  founding_effective_size: float,
                                                  interval_rates: np.ndarray,
                                                  n_generations: int,
                                                  offspring_variance: float,
                                                  population_spec: dict) -> float:
    _SPEC_KEYS = {"n_loci", "n_individuals", "rho", "p_low", "p_high", "seed"}
    Y = np.asarray(delta_p_by_replicate, dtype=float)
    if Y.ndim != 2 or Y.shape[0] < 1 or Y.shape[1] < 1:
        raise ValueError("delta_p_by_replicate must be a 2D array with >= 1 row")
    if not np.all(np.isfinite(Y)):
        raise ValueError("delta_p_by_replicate must be finite")
    if not isinstance(census_sizes, list) or len(census_sizes) != Y.shape[0]:
        raise ValueError("census_sizes must be a list with one entry per replicate")
    if isinstance(founding_effective_size, bool) or not isinstance(
            founding_effective_size, (int, float, np.floating, np.integer)):
        raise ValueError("founding_effective_size must be a finite number > 0.5")
    if not np.isfinite(float(founding_effective_size)) or float(founding_effective_size) <= 0.5:
        raise ValueError("founding_effective_size must be a finite number > 0.5")
    if not isinstance(population_spec, dict) or set(population_spec) != _SPEC_KEYS:
        raise ValueError(f"population_spec must be a dict with exactly the keys {sorted(_SPEC_KEYS)}")
    if int(population_spec["n_loci"]) != Y.shape[1]:
        raise ValueError("population_spec['n_loci'] must match the number of loci")
    rates = np.asarray(interval_rates, dtype=float)
    if rates.ndim != 1 or rates.size != Y.shape[1] - 1:
        raise ValueError("interval_rates must have length n_loci - 1")

    haplotypes = simulate_base_population(
        population_spec["n_loci"], population_spec["n_individuals"],
        population_spec["rho"], population_spec["p_low"],
        population_spec["p_high"], population_spec["seed"])

    base_diversity = individual_diversity_matrix(haplotypes)
    recombination = pairwise_recombination_matrix(rates)
    weighted_base = weighted_base_diversity_matrix(haplotypes, recombination)

    # Diagnostic-only cross-check: the additive GENIC variance (step 08) ignores
    # linkage disequilibrium and is not the target quantity, but a broken step 08
    # must not be able to pass this orchestrator silently just because its output
    # is never touched.
    replicate_mean_change = Y.mean(axis=0)
    genic_variance = genic_variance_from_change(replicate_mean_change, base_diversity)
    if not np.isfinite(genic_variance) or genic_variance < 0.0:
        raise ValueError("genic-variance diagnostic (step 08) returned a non-finite or negative value")

    operators, covariances = [], []
    for census in census_sizes:
        n_eff = variance_effective_size(census, offspring_variance)
        operators.append(expected_change_operator(
            weighted_base, recombination, n_generations,
            float(founding_effective_size), float(census)))
        covariances.append(drift_covariance_matrix(
            weighted_base, recombination, n_generations,
            float(founding_effective_size), float(census), n_eff))

    frequency = np.asarray(haplotypes, dtype=float).mean(axis=0)
    contrast = 2.0 * frequency - 1.0

    coefficient = average_effect_scale(Y, operators, covariances, contrast)

    information = 0.0
    for Lm, Dm in zip(operators, covariances):
        z = Lm @ contrast
        information += float(z @ np.linalg.solve(Dm, z))
    sampling_variance = 1.0 / information

    scale = float(contrast @ base_diversity @ contrast)
    return float(coefficient ** 2 * scale - scale * sampling_variance)
SCICODE_GOLD_EOF
