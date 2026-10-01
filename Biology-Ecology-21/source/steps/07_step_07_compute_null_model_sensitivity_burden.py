"""
Compute the final null-model sensitivity burden from the raw regional pool, observed-community definition, distance thresholds, and regional-pool-fitted OCSVM boundary.

Use the preceding task components to construct the full-community evidence and the batched leave-one-out analysis while preserving the fixed regional reference structures. Verify that independently assembled intermediate outputs agree, then sum the ordered full-plus-leave-one-out disagreement profile and return one integer burden. Do not hard-code intermediate classifications, disagreement counts, profiles, or final values.

The final burden is an end-to-end robustness summary. It accumulates every statistic-by-threshold cell in which ecological classification depends on null-model choice across the original community and all one-species perturbations, while retaining the same regional pool and fitted functional-space boundary.

Returns
-------
int representing the complete null-model sensitivity burden summed across the full community and all leave-one-out communities.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_null_model_sensitivity_burden(
    pool_coords: "np.ndarray",
    observed_indices: "np.ndarray",
    r_values: "np.ndarray",
    nu: float = 0.05,
    n_candidates: int = 300,
    critical: float = 2.58
) -> int:
    """Compute the full and leave-one-out null-model sensitivity burden.

    Parameters
    ----------
    pool_coords : np.ndarray
        Finite regional-pool coordinates with shape (n_species, 2).
    observed_indices : np.ndarray
        One-dimensional array of unique valid pool indices defining an
        observed community with at least four species.
    r_values : np.ndarray
        Non-empty one-dimensional array of finite, non-negative thresholds.
    nu : float, default=0.05
        OCSVM parameter strictly between 0 and 1. The boundary reference is
        fitted from the unchanged regional pool.
    n_candidates : int, default=300
        Positive number of deterministic Halton candidates.
    critical : float, default=2.58
        Positive finite SES critical magnitude.

    Returns
    -------
    sensitivity_burden : int
        Sum of cross-null disagreement counts for the full community and
        every leave-one-out community in observed_indices order.

    Raises
    ------
    ValueError
        If an input is invalid or a downstream null-model requirement fails.
    RuntimeError
        If independently assembled intermediate results are inconsistent.
    """
    return sensitivity_burden

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_null_model_sensitivity_burden(
    pool_coords: "np.ndarray",
    observed_indices: "np.ndarray",
    r_values: "np.ndarray",
    nu: float = 0.05,
    n_candidates: int = 300,
    critical: float = 2.58
) -> int:
    pool_coords = np.asarray(pool_coords, dtype=float)
    observed_indices = np.asarray(observed_indices, dtype=int)
    r_values = np.asarray(r_values, dtype=float)

    n = pool_coords.shape[0]
    k = observed_indices.size
    t_count = r_values.size

    observed = _oracle_compute_observed_arrangement_curves(
        pool_coords[observed_indices],
        r_values
    )

    certificate = _oracle_compute_exact_pool_null_certificate(
        pool_coords,
        k,
        r_values
    )

    boundary = _oracle_fit_regional_boundary_certificate(
        pool_coords,
        nu
    )

    full_evidence = _oracle_analyze_full_community_dual_null(
        pool_coords,
        observed_indices,
        r_values,
        nu,
        n_candidates,
        critical
    )

    loo_curves = _oracle_compute_all_leave_one_out_observed_curves(
        pool_coords,
        observed_indices,
        r_values
    )

    profile = _oracle_compute_leave_one_out_disagreement_profile(
        pool_coords,
        observed_indices,
        r_values,
        nu,
        n_candidates,
        critical
    )

    if int(certificate[0, 0]) <= 1:
        raise RuntimeError(
            "Pool-null certificate inconsistency."
        )

    certificate_mean = (
        np.array(
            [int(x) for x in certificate[0, 1:]],
            dtype=float
        )
        / float(certificate[0, 0])
    )

    certificate_mean = certificate_mean * np.r_[
        np.full(
            t_count,
            1.0 / (k * (k - 1))
        ),
        np.full(
            t_count,
            1.0 / k
        )
    ]

    if not np.allclose(
        full_evidence[:, 1, :].ravel(),
        certificate_mean,
        atol=1e-12,
        rtol=0.0
    ):
        raise RuntimeError(
            "Pool-null inconsistency."
        )

    if not np.allclose(
        full_evidence[:, 0, :],
        observed,
        atol=1e-12,
        rtol=0.0
    ):
        raise RuntimeError(
            "Observed-curve inconsistency."
        )

    if not (
        boundary[n + 2]
        <= float(nu)
        <= boundary[n + 3]
    ):
        raise RuntimeError(
            "Requested nu lies outside the certified boundary interval."
        )

    if loo_curves.shape[0] != k:
        raise RuntimeError(
            "Leave-one-out curve-count inconsistency."
        )

    if int(profile[0]) != int(
        np.sum(full_evidence[:, 9, :])
    ):
        raise RuntimeError(
            "Full-community disagreement-count inconsistency."
        )

    return int(np.sum(profile))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    pool = """np.array([
    [0.96, 0.04],
    [0.25, 0.36],
    [0.57, 0.46],
    [0.94, 0.19],
    [0.31, 0.08],
    [0.18, 0.60],
    [0.05, 0.94],
    [0.76, 0.09],
    [0.79, 0.12],
    [0.52, 0.21],
    [0.17, 0.53]
], dtype=float)"""

    def _benchmark_case(indices):
        return f"""import numpy as np
pool_candidate = {pool}
pool_oracle = pool_candidate.copy()
observed_candidate = np.array({indices}, dtype=int)
observed_oracle = observed_candidate.copy()
r_candidate = np.array([0.08,0.18,0.28,0.36,0.44,0.52,0.56], dtype=float)
r_oracle = r_candidate.copy()
nu_candidate = 0.05
nu_oracle = 0.05
n_candidate = 300
n_oracle = 300
critical_candidate = 2.58
critical_oracle = 2.58"""

    return [
        {
            "setup": _benchmark_case("[0,1,2,3,4]"),
            "call": "compute_null_model_sensitivity_burden(pool_candidate, observed_candidate, r_candidate, nu_candidate, n_candidate, critical_candidate)",
            "gold_call": "_oracle_compute_null_model_sensitivity_burden(pool_oracle, observed_oracle, r_oracle, nu_oracle, n_oracle, critical_oracle)",
        },
        {
            "setup": _benchmark_case("[0,1,2,5,10]"),
            "call": "compute_null_model_sensitivity_burden(pool_candidate, observed_candidate, r_candidate, nu_candidate, n_candidate, critical_candidate)",
            "gold_call": "_oracle_compute_null_model_sensitivity_burden(pool_oracle, observed_oracle, r_oracle, nu_oracle, n_oracle, critical_oracle)",
        },
        {
            "setup": """import numpy as np
index_candidate = np.arange(12, dtype=float)
pool_candidate = np.column_stack((((index_candidate * 37 + 11) % 101) / 101, ((index_candidate * 53 + 7) % 103) / 103))
pool_oracle = pool_candidate.copy()
observed_candidate = np.array([0,2,4,6,8], dtype=int)
observed_oracle = observed_candidate.copy()
r_candidate = np.array([0.31,0.18,0.27,0.14,0.33,0.22,0.18], dtype=float)
r_oracle = r_candidate.copy()
nu_candidate = 0.60
nu_oracle = 0.60
n_candidate = 300
n_oracle = 300
critical_candidate = 2.58
critical_oracle = 2.58""",
            "call": "compute_null_model_sensitivity_burden(pool_candidate, observed_candidate, r_candidate, nu_candidate, n_candidate, critical_candidate)",
            "gold_call": "_oracle_compute_null_model_sensitivity_burden(pool_oracle, observed_oracle, r_oracle, nu_oracle, n_oracle, critical_oracle)",
        },
    ]
