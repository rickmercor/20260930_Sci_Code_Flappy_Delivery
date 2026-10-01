"""
### Compute the observed PNcp/NNcp curves; derive the exhaustive regional-pool null means and sample standard deviations from the exact pool-null certificate; and compute the displacement-null moments by applying the certified regional boundary to ordered two-dimensional Halton candidates (bases 2 and 3, indices 1..n_candidates), retaining f(x)>=0 in order, forming consecutive groups of the community richness, and discarding an incomplete final group. Standardize both nulls with the supplied critical magnitude, classify with strict thresholds, and mark every statistic-by-threshold cell where the two classifications differ.

The full-community analysis is the first point at which the observed arrangement and both ecological null expectations are integrated. Returning the complete evidence tensor makes the cross-null comparison auditable while preserving the multi-scale structure of PNcp and NNcp.

Returns
-------
np.ndarray of shape (2, 10, len(r_values)) containing the full dual-null evidence tensor.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def analyze_full_community_dual_null(
    pool_coords: "np.ndarray",
    observed_indices: "np.ndarray",
    r_values: "np.ndarray",
    nu: float = 0.05,
    n_candidates: int = 300,
    critical: float = 2.58
) -> "np.ndarray":
    """Build the complete full-community dual-null evidence tensor.

    Parameters
    ----------
    pool_coords : np.ndarray
        Finite regional-pool coordinates with shape (n_species, 2)
        and at least four species.
    observed_indices : np.ndarray
        One-dimensional array of unique valid pool indices defining
        an observed community with at least three species and richness
        smaller than the regional pool.
    r_values : np.ndarray
        Non-empty one-dimensional array of finite, non-negative thresholds.
        Input order and repeated values are preserved.
    nu : float, default=0.05
        OCSVM parameter strictly between 0 and 1. The boundary is fitted from
        the unchanged regional pool.
    n_candidates : int, default=300
        Positive number of deterministic Halton candidates.
    critical : float, default=2.58
        Positive finite SES critical magnitude; equality to either
        boundary remains classified as random.

    Returns
    -------
    evidence : np.ndarray
        Float array of shape (2, 10, len(r_values)). Axis 1 is ordered as
        observed value, pool mean, pool sample SD, pool SES, pool class,
        displacement mean, displacement sample SD, displacement SES,
        displacement class, disagreement indicator.

    Raises
    ------
    ValueError
        If any input is invalid or a downstream null-model requirement fails.
    """
    return evidence

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _dual_null_radical_inverse(index: int, base: int) -> float:
    result, factor, value = 0.0, 1.0 / float(base), int(index)
    while value > 0:
        value, digit = divmod(value, base)
        result += digit * factor
        factor /= float(base)
    return float(result)


def _dual_null_pool_moments(certificate, community_size, n_thresholds):
    total = int(certificate[0, 0])
    sums = np.array([int(x) for x in certificate[0, 1:]], dtype=object)
    squares = np.array(
        [[int(x) for x in row] for row in certificate[1:, 1:]],
        dtype=object
    )

    centred = total * np.diagonal(squares) - sums * sums
    variance = np.array([float(x) for x in centred]) / (
        total * (total - 1.0)
    )
    mean = np.array([float(x) / total for x in sums])

    scale = np.r_[
        np.full(
            n_thresholds,
            1.0 / (community_size * (community_size - 1))
        ),
        np.full(n_thresholds, 1.0 / community_size),
    ]

    mean = (mean * scale).reshape(2, n_thresholds)
    sd = (np.sqrt(variance) * scale).reshape(2, n_thresholds)

    if np.any(~np.isfinite(sd)) or np.any(sd <= 0.0):
        raise ValueError("Pool null requires positive sample SD.")

    return mean, sd


def _dual_null_displacement_moments(
    pool_coords,
    boundary,
    community_size,
    r_values,
    n_candidates
):
    n = pool_coords.shape[0]
    sigma = boundary[0]
    alpha = boundary[1:n + 1]
    rho = boundary[n + 1]

    candidates = np.array(
        [
            [
                _dual_null_radical_inverse(t, 2),
                _dual_null_radical_inverse(t, 3)
            ]
            for t in range(1, int(n_candidates) + 1)
        ],
        dtype=float,
    )

    squared = np.sum(
        (candidates[:, None, :] - pool_coords[None, :, :]) ** 2,
        axis=2
    )

    accepted = candidates[
        np.exp(-sigma * squared) @ alpha - rho >= 0.0
    ]

    k = int(community_size)
    n_complete = accepted.shape[0] // k

    if n_complete < 2:
        raise ValueError(
            "At least two complete displacement communities are required."
        )

    curves = np.stack([
        _oracle_compute_observed_arrangement_curves(group, r_values)
        for group in accepted[: n_complete * k].reshape(
            n_complete, k, 2
        )
    ])

    mean = np.mean(curves, axis=0)
    sd = np.std(curves, axis=0, ddof=1)

    if np.any(~np.isfinite(sd)) or np.any(sd <= 0.0):
        raise ValueError("Displacement null requires positive sample SD.")

    return mean, sd


def _oracle_analyze_full_community_dual_null(
    pool_coords: "np.ndarray",
    observed_indices: "np.ndarray",
    r_values: "np.ndarray",
    nu: float = 0.05,
    n_candidates: int = 300,
    critical: float = 2.58
) -> "np.ndarray":
    pool_coords = np.asarray(pool_coords, dtype=float)
    r_values = np.asarray(r_values, dtype=float)
    indices = np.asarray(observed_indices)

    if (
        pool_coords.ndim != 2
        or pool_coords.shape[0] < 4
        or pool_coords.shape[1] != 2
        or not np.all(np.isfinite(pool_coords))
    ):
        raise ValueError(
            "pool_coords must contain at least four finite two-dimensional species."
        )

    if (
        indices.ndim != 1
        or indices.size < 3
        or indices.size >= pool_coords.shape[0]
    ):
        raise ValueError("observed_indices has invalid size.")

    if (
        not np.issubdtype(indices.dtype, np.integer)
        and not np.all(indices == np.floor(indices))
    ):
        raise ValueError("observed_indices must contain integers.")

    indices = indices.astype(int)

    if (
        np.any(indices < 0)
        or np.any(indices >= pool_coords.shape[0])
        or np.unique(indices).size != indices.size
    ):
        raise ValueError(
            "observed_indices must contain unique valid pool indices."
        )

    if (
        r_values.ndim != 1
        or r_values.size < 1
        or not np.all(np.isfinite(r_values))
        or np.any(r_values < 0.0)
    ):
        raise ValueError(
            "r_values must contain finite non-negative thresholds."
        )

    if not np.isfinite(critical) or critical <= 0.0:
        raise ValueError("critical must be positive and finite.")

    k = int(indices.size)
    t_count = int(r_values.size)

    observed = _oracle_compute_observed_arrangement_curves(
        pool_coords[indices],
        r_values
    )

    certificate = _oracle_compute_exact_pool_null_certificate(
        pool_coords,
        k,
        r_values
    )

    pool_mean, pool_sd = _dual_null_pool_moments(
        certificate,
        k,
        t_count
    )

    boundary = _oracle_fit_regional_boundary_certificate(
        pool_coords,
        nu
    )

    disp_mean, disp_sd = _dual_null_displacement_moments(
        pool_coords,
        boundary,
        k,
        r_values,
        n_candidates
    )

    def _classify(ses):
        out = np.zeros_like(ses, dtype=float)
        out[ses > critical] = 1.0
        out[ses < -critical] = -1.0
        return out

    pool_ses = (observed - pool_mean) / pool_sd
    disp_ses = (observed - disp_mean) / disp_sd

    pool_class = _classify(pool_ses)
    disp_class = _classify(disp_ses)

    return np.stack(
        (
            observed,
            pool_mean,
            pool_sd,
            pool_ses,
            pool_class,
            disp_mean,
            disp_sd,
            disp_ses,
            disp_class,
            (pool_class != disp_class).astype(float)
        ),
        axis=1,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    pool = """np.array([
    [0.96,0.04],[0.25,0.36],[0.57,0.46],[0.94,0.19],[0.31,0.08],
    [0.18,0.60],[0.05,0.94],[0.76,0.09],[0.79,0.12],[0.52,0.21],[0.17,0.53]
], dtype=float)"""

    def _case(indices, r_values):
        return f"""import numpy as np
pool_candidate = {pool}
pool_oracle = pool_candidate.copy()
observed_candidate = np.array({indices}, dtype=int)
observed_oracle = observed_candidate.copy()
r_candidate = np.array({r_values}, dtype=float)
r_oracle = r_candidate.copy()
nu_candidate = 0.05
nu_oracle = 0.05
n_candidate = 300
n_oracle = 300
critical_candidate = 2.58
critical_oracle = 2.58"""

    return [
        {
            "setup": _case("[0,3,5,7,8]", "[0.08,0.18,0.28,0.36,0.44,0.52,0.56]"),
            "call": "analyze_full_community_dual_null(pool_candidate, observed_candidate, r_candidate, nu_candidate, n_candidate, critical_candidate)",
            "gold_call": "_oracle_analyze_full_community_dual_null(pool_oracle, observed_oracle, r_oracle, nu_oracle, n_oracle, critical_oracle)",
        },
        {
            "setup": _case("[0,1,2,3,4]", "[0.56,0.18,0.44,0.18,0.28]"),
            "call": "analyze_full_community_dual_null(pool_candidate, observed_candidate, r_candidate, nu_candidate, n_candidate, critical_candidate)",
            "gold_call": "_oracle_analyze_full_community_dual_null(pool_oracle, observed_oracle, r_oracle, nu_oracle, n_oracle, critical_oracle)",
        },
        {
            "setup": _case("[0,1,2,5,10]", "[0.08,0.18,0.28,0.36,0.44,0.52,0.56]"),
            "call": "analyze_full_community_dual_null(pool_candidate, observed_candidate, r_candidate, nu_candidate, n_candidate, critical_candidate)",
            "gold_call": "_oracle_analyze_full_community_dual_null(pool_oracle, observed_oracle, r_oracle, nu_oracle, n_oracle, critical_oracle)",
        },
    ]
