#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

from statistics import NormalDist  # noqa: E402

import numpy as np  # noqa: E402, F811


def compute_fixed_margins(
    alt_counts: np.ndarray, n_haplotypes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation of the fixed-margin transformation."""
    counts = np.asarray(alt_counts, dtype=float)
    if counts.ndim != 1 or counts.size < 2 or not np.isfinite(counts).all():
        raise ValueError("alt_counts must be a finite one-dimensional vector with at least two entries")
    if not isinstance(n_haplotypes, (int, np.integer)) or int(n_haplotypes) < 1:
        raise ValueError("n_haplotypes must be a positive integer")
    if np.any(counts < 0.0) or np.any(counts > int(n_haplotypes)):
        raise ValueError("alt_counts must lie between zero and n_haplotypes")
    if not np.all(counts == np.floor(counts)):
        raise ValueError("alt_counts must contain integer values")

    corrected_frequencies = (counts + 0.5) / (int(n_haplotypes) + 1.0)
    normal = NormalDist()
    thresholds = np.asarray(
        [normal.inv_cdf(1.0 - float(value)) for value in corrected_frequencies],
        dtype=float,
    )
    return corrected_frequencies, thresholds

import numpy as np  # noqa: E402, F811


def map_factor_parameters(
    working_loadings: np.ndarray, thresholds: np.ndarray, psi_min: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation of the constrained parameter map."""
    coefficients = np.asarray(working_loadings, dtype=float)
    tau = np.asarray(thresholds, dtype=float)
    if coefficients.ndim != 1 or coefficients.size < 2 or not np.isfinite(coefficients).all():
        raise ValueError("working_loadings must be a finite vector with at least two entries")
    if tau.shape != coefficients.shape or not np.isfinite(tau).all():
        raise ValueError("thresholds must be finite and match working_loadings")
    if not isinstance(psi_min, (int, float, np.integer, np.floating)):
        raise ValueError("psi_min must be numeric")
    if not 0.0 < float(psi_min) < 1.0:
        raise ValueError("psi_min must lie in (0, 1)")
    if coefficients[0] == 0.0:
        raise ValueError("the lead working loading must be nonzero")

    uniqueness = 1.0 / (1.0 + coefficients * coefficients)
    if np.any(uniqueness < float(psi_min) - 1e-12):
        raise ValueError("working_loadings violate the uniqueness floor")
    loadings = coefficients * np.sqrt(uniqueness)
    if loadings[0] < 0.0:
        loadings = -loadings
        coefficients = -coefficients
    intercepts = -tau / np.sqrt(uniqueness)
    return loadings, uniqueness, intercepts

import math  # noqa: E402

import numpy as np  # noqa: E402, F811


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    """Evaluate the standard-normal CDF without an external dependency."""
    flat = np.asarray(values, dtype=float).ravel()
    result = np.fromiter(
        (0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))) for value in flat),
        dtype=float,
        count=flat.size,
    )
    return result.reshape(np.asarray(values).shape)


def build_conditional_mixture(
    loadings: np.ndarray,
    uniqueness: np.ndarray,
    thresholds: np.ndarray,
    lead_state: int,
    quadrature_order: int,
    factor_bound: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation of fixed lead-conditioned quadrature."""
    b = np.asarray(loadings, dtype=float)
    psi = np.asarray(uniqueness, dtype=float)
    tau = np.asarray(thresholds, dtype=float)
    if b.ndim != 1 or b.size < 2 or not np.isfinite(b).all():
        raise ValueError("loadings must be a finite vector with at least two entries")
    if psi.shape != b.shape or tau.shape != b.shape:
        raise ValueError("uniqueness and thresholds must match loadings")
    if not np.isfinite(psi).all() or np.any(psi <= 0.0) or not np.isfinite(tau).all():
        raise ValueError("uniqueness must be positive and all parameters finite")
    if lead_state not in (0, 1):
        raise ValueError("lead_state must be zero or one")
    if not isinstance(quadrature_order, (int, np.integer)) or int(quadrature_order) < 8:
        raise ValueError("quadrature_order must be an integer of at least eight")
    if not isinstance(factor_bound, (int, float, np.integer, np.floating)):
        raise ValueError("factor_bound must be numeric")
    if not np.isfinite(float(factor_bound)) or float(factor_bound) <= 0.0:
        raise ValueError("factor_bound must be positive and finite")

    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    factor_nodes = float(factor_bound) * nodes
    base = (
        float(factor_bound)
        * weights
        * np.exp(-0.5 * factor_nodes * factor_nodes)
        / math.sqrt(2.0 * math.pi)
    )
    lead_eta = (b[0] * factor_nodes - tau[0]) / math.sqrt(psi[0])
    lead_probability = _normal_cdf(lead_eta)
    base *= lead_probability if lead_state == 1 else (1.0 - lead_probability)
    normalizer = float(np.sum(base))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("lead-conditioned quadrature has zero mass")
    mixture_weights = base / normalizer
    partner_eta = (
        factor_nodes[:, None] * b[None, 1:] - tau[None, 1:]
    ) / np.sqrt(psi)[None, 1:]
    partner_probabilities = _normal_cdf(partner_eta)
    return factor_nodes, mixture_weights, partner_probabilities

import numpy as np  # noqa: E402, F811


def enumerate_factor_modes(
    partner_loadings: np.ndarray, partner_thresholds: np.ndarray
) -> np.ndarray:
    """Reference implementation of one-factor breakpoint enumeration."""
    loadings = np.asarray(partner_loadings, dtype=float)
    thresholds = np.asarray(partner_thresholds, dtype=float)
    if loadings.ndim != 1 or loadings.size < 1 or not np.isfinite(loadings).all():
        raise ValueError("partner_loadings must be a nonempty finite vector")
    if thresholds.shape != loadings.shape or not np.isfinite(thresholds).all():
        raise ValueError("partner_thresholds must be finite and match partner_loadings")

    active = loadings != 0.0
    if np.any(active):
        breakpoints = np.unique(np.sort(thresholds[active] / loadings[active]))
        representatives = [float(breakpoints[0] - 1.0)]
        representatives.extend(
            float(0.5 * (left + right))
            for left, right in zip(breakpoints[:-1], breakpoints[1:])
        )
        representatives.append(float(breakpoints[-1] + 1.0))
    else:
        representatives = [0.0]
    rows = [
        (loadings * factor > thresholds).astype(np.uint8)
        for factor in representatives
    ]
    return np.unique(np.asarray(rows, dtype=np.uint8), axis=0)

import numpy as np  # noqa: E402, F811


def score_conditional_candidates(
    configurations: np.ndarray,
    mixture_weights: np.ndarray,
    partner_probabilities: np.ndarray,
) -> np.ndarray:
    """Reference implementation of fixed-mixture candidate scoring."""
    candidates = np.asarray(configurations)
    weights = np.asarray(mixture_weights, dtype=float)
    probabilities = np.asarray(partner_probabilities, dtype=float)
    if candidates.ndim != 2 or candidates.shape[0] < 1 or candidates.shape[1] < 1:
        raise ValueError("configurations must be a nonempty two-dimensional array")
    if not np.isin(candidates, (0, 1)).all():
        raise ValueError("configurations must be binary")
    if weights.ndim != 1 or weights.size < 1 or not np.isfinite(weights).all():
        raise ValueError("mixture_weights must be a nonempty finite vector")
    if probabilities.shape != (weights.size, candidates.shape[1]):
        raise ValueError("partner_probabilities shape does not match weights and configurations")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("partner_probabilities must lie in [0, 1]")
    if np.any(weights < 0.0) or not np.isclose(np.sum(weights), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("mixture_weights must be nonnegative and sum to one")

    candidates = candidates.astype(np.uint8)
    conditional_probabilities = np.empty(candidates.shape[0], dtype=float)
    for index, candidate in enumerate(candidates):
        component = np.prod(
            np.where(candidate[None, :] == 1, probabilities, 1.0 - probabilities),
            axis=1,
        )
        conditional_probabilities[index] = float(np.sum(weights * component))
    return conditional_probabilities

import numpy as np  # noqa: E402, F811

def expand_mode_candidates(
    mode_candidates: np.ndarray, mode_probabilities: np.ndarray
) -> np.ndarray:
    """Reference implementation of the one-allele neighborhood expansion."""
    candidates = np.asarray(mode_candidates)
    probabilities = np.asarray(mode_probabilities, dtype=float)
    if candidates.ndim != 2 or candidates.shape[0] < 1 or candidates.shape[1] < 1:
        raise ValueError("mode_candidates must be a nonempty two-dimensional array")
    if not np.isin(candidates, (0, 1)).all():
        raise ValueError("mode_candidates must be binary")
    if probabilities.shape != (candidates.shape[0],) or not np.isfinite(probabilities).all():
        raise ValueError("mode_probabilities must be finite and match the candidate rows")
    if np.any(probabilities < 0.0):
        raise ValueError("mode_probabilities must be nonnegative")

    candidates = candidates.astype(np.uint8)
    leading_index = min(
        range(candidates.shape[0]),
        key=lambda index: (-float(probabilities[index]), tuple(int(v) for v in candidates[index])),
    )
    leading = candidates[leading_index]
    neighbors = np.repeat(leading[None, :], leading.size, axis=0)
    neighbors[np.arange(leading.size), np.arange(leading.size)] ^= 1
    return np.unique(np.vstack([candidates, neighbors]), axis=0).astype(np.uint8)

import numpy as np  # noqa: E402, F811


def summarize_top_burden(
    configurations: np.ndarray, conditional_probabilities: np.ndarray, top_l: int
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Reference implementation of top-L ranking and burden averaging."""
    candidates = np.asarray(configurations)
    probabilities = np.asarray(conditional_probabilities, dtype=float)
    if candidates.ndim != 2 or candidates.shape[0] < 1 or candidates.shape[1] < 1:
        raise ValueError("configurations must be a nonempty two-dimensional array")
    if not np.isin(candidates, (0, 1)).all():
        raise ValueError("configurations must be binary")
    if probabilities.shape != (candidates.shape[0],) or not np.isfinite(probabilities).all():
        raise ValueError("conditional_probabilities must be finite and match candidate rows")
    if np.any(probabilities < 0.0):
        raise ValueError("conditional_probabilities must be nonnegative")
    if not isinstance(top_l, (int, np.integer)) or not 1 <= int(top_l) <= candidates.shape[0]:
        raise ValueError("top_l must be between one and the number of candidates")

    candidates = candidates.astype(np.uint8)
    order = sorted(
        range(candidates.shape[0]),
        key=lambda index: (-float(probabilities[index]), tuple(int(v) for v in candidates[index])),
    )[: int(top_l)]
    top_configurations = candidates[order]
    top_probabilities = probabilities[order]
    top_mass = float(np.sum(top_probabilities))
    if top_mass <= 0.0:
        raise ValueError("the retained candidates have zero probability")
    burdens = np.sum(top_configurations, axis=1, dtype=float)
    mean_burden = float(np.sum(top_probabilities * burdens) / top_mass)
    return mean_burden, top_configurations, top_probabilities, top_mass

import numpy as np  # noqa: E402, F811


def run_haploperturb_benchmark(
    alt_counts: np.ndarray,
    n_haplotypes: int,
    working_loadings: np.ndarray,
    psi_min: float,
    quadrature_order: int,
    factor_bound: float,
    top_l: int,
) -> float:
    """Reference implementation chaining every earlier benchmark step."""
    _, thresholds = compute_fixed_margins(  # noqa: F821
        alt_counts, n_haplotypes
    )
    loadings, uniqueness, _ = map_factor_parameters(  # noqa: F821
        working_loadings, thresholds, psi_min
    )
    modes = enumerate_factor_modes(  # noqa: F821
        loadings[1:], thresholds[1:]
    )
    burdens = []
    for lead_state in (0, 1):
        _, mixture_weights, partner_probabilities = build_conditional_mixture(  # noqa: F821
            loadings,
            uniqueness,
            thresholds,
            lead_state,
            quadrature_order,
            factor_bound,
        )
        mode_probabilities = score_conditional_candidates(  # noqa: F821
            modes, mixture_weights, partner_probabilities
        )
        expanded = expand_mode_candidates(  # noqa: F821
            modes, mode_probabilities
        )
        expanded_probabilities = score_conditional_candidates(  # noqa: F821
            expanded, mixture_weights, partner_probabilities
        )
        mean_burden, _, _, _ = summarize_top_burden(  # noqa: F821
            expanded, expanded_probabilities, top_l
        )
        burdens.append(mean_burden)
    return float(burdens[1] - burdens[0])
SCICODE_GOLD_EOF
