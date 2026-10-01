"""
Compute the complete cross-null disagreement profile for the full community and every leave-one-out perturbation while reusing richness-dependent null quantities.



Analyze the full observed community once. For the leave-one-out communities, first obtain every reduced observed PNcp/NNcp curve in one batched calculation. Because every leave-one-out community has the same richness and the regional pool and fitted OCSVM boundary remain fixed, compute the fixed-richness pool-null moments and displacement-null moments once for richness k-1, then standardize and classify all leave-one-out observed curves against those shared null moments.



Return the full-community disagreement count followed by one count for each omission in observed_indices order.

The disagreement profile separates composition-dependent observed arrangement from richness-dependent null expectations. All leave-one-out communities share richness k-1, the unchanged regional pool, and the same fitted functional-space boundary, so their null distributions are shared even though their observed PNcp and NNcp curves differ.

Returns
-------
np.ndarray of integers with length 1 + len(observed_indices); element 0 is the full-community disagreement count and subsequent elements are leave-one-out disagreement counts in observed_indices order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_leave_one_out_disagreement_profile(
    pool_coords: "np.ndarray",
    observed_indices: "np.ndarray",
    r_values: "np.ndarray",
    nu: float = 0.05,
    n_candidates: int = 300,
    critical: float = 2.58
) -> "np.ndarray":
    """Return full-community and ordered leave-one-out disagreement counts.

    Parameters
    ----------
    pool_coords : np.ndarray
        Finite regional-pool coordinates with shape (n_species, 2).
    observed_indices : np.ndarray
        One-dimensional array of unique valid pool indices defining an
        observed community with at least four species and richness smaller
        than the regional pool.
    r_values : np.ndarray
        Non-empty one-dimensional array of finite, non-negative thresholds.
    nu : float, default=0.05
        OCSVM parameter strictly between 0 and 1. The boundary reference is
        determined from the unchanged regional pool.
    n_candidates : int, default=300
        Positive number of deterministic Halton candidates.
    critical : float, default=2.58
        Positive finite SES critical magnitude.

    Returns
    -------
    profile : np.ndarray
        Integer array of length 1 + len(observed_indices). Element 0 is the
        full-community disagreement count; subsequent elements correspond
        to omissions in observed_indices order.

    Raises
    ------
    ValueError
        If an input is invalid or a downstream null-model requirement fails.
    """
    return profile

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_leave_one_out_disagreement_profile(
    pool_coords: "np.ndarray",
    observed_indices: "np.ndarray",
    r_values: "np.ndarray",
    nu: float = 0.05,
    n_candidates: int = 300,
    critical: float = 2.58
) -> "np.ndarray":
    pool_coords = np.asarray(pool_coords, dtype=float)
    r_values = np.asarray(r_values, dtype=float)
    indices = np.asarray(observed_indices).astype(int)

    if indices.ndim != 1 or indices.size < 4:
        raise ValueError(
            "observed_indices must define at least four species."
        )

    if not np.isfinite(critical) or critical <= 0.0:
        raise ValueError("critical must be positive and finite.")

    full_evidence = _oracle_analyze_full_community_dual_null(
        pool_coords,
        indices,
        r_values,
        nu,
        n_candidates,
        critical
    )

    loo_curves = _oracle_compute_all_leave_one_out_observed_curves(
        pool_coords,
        indices,
        r_values
    )

    shared = _oracle_analyze_full_community_dual_null(
        pool_coords,
        np.delete(indices, 0),
        r_values,
        nu,
        n_candidates,
        critical
    )

    def _classify(ses):
        out = np.zeros_like(ses, dtype=np.int8)
        out[ses > critical] = 1
        out[ses < -critical] = -1
        return out

    pool_class = _classify(
        (loo_curves - shared[None, :, 1, :])
        / shared[None, :, 2, :]
    )

    disp_class = _classify(
        (loo_curves - shared[None, :, 5, :])
        / shared[None, :, 6, :]
    )

    loo_counts = np.sum(
        pool_class != disp_class,
        axis=(1, 2),
        dtype=np.int64
    )

    return np.concatenate(
        (
            np.array([
                int(np.sum(full_evidence[:, 9, :]))
            ]),
            loo_counts
        )
    ).astype(int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    pool = """np.array([
    [0.96,0.04],[0.25,0.36],[0.57,0.46],[0.94,0.19],[0.31,0.08],
    [0.18,0.60],[0.05,0.94],[0.76,0.09],[0.79,0.12],[0.52,0.21],[0.17,0.53]
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
            "setup": _benchmark_case("[0,3,5,7,8]"),
            "call": "compute_leave_one_out_disagreement_profile(pool_candidate, observed_candidate, r_candidate, nu_candidate, n_candidate, critical_candidate)",
            "gold_call": "_oracle_compute_leave_one_out_disagreement_profile(pool_oracle, observed_oracle, r_oracle, nu_oracle, n_oracle, critical_oracle)",
        },
        {
            "setup": _benchmark_case("[0,1,2,3,4]"),
            "call": "compute_leave_one_out_disagreement_profile(pool_candidate, observed_candidate, r_candidate, nu_candidate, n_candidate, critical_candidate)",
            "gold_call": "_oracle_compute_leave_one_out_disagreement_profile(pool_oracle, observed_oracle, r_oracle, nu_oracle, n_oracle, critical_oracle)",
        },
        {
            "setup": _benchmark_case("[0,1,2,5,10]"),
            "call": "compute_leave_one_out_disagreement_profile(pool_candidate, observed_candidate, r_candidate, nu_candidate, n_candidate, critical_candidate)",
            "gold_call": "_oracle_compute_leave_one_out_disagreement_profile(pool_oracle, observed_oracle, r_oracle, nu_oracle, n_oracle, critical_oracle)",
        },
    ]
