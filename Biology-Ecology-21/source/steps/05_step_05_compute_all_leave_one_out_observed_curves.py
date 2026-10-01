"""
Compute the observed PNcp and NNcp curves for every leave-one-out community in one batched calculation.



Start from the observed community selected from the regional pool and return the two observed arrangement curves after omitting each observed species in turn, preserving observed_indices order. Reuse the full pairwise-distance structure rather than recomputing a complete pairwise analysis separately for every omission. The implementation must remain exact for repeated or unsorted thresholds and for tied nearest-neighbour distances.



The required running-time target is quadratic in observed-community richness for constructing the distance structure, plus work proportional to the returned richness-by-threshold output. A cubic leave-one-out recomputation is not acceptable.

Leave-one-out perturbations change both pairwise and nearest-neighbour summaries. Pairwise counts can be updated by removing the omitted species' incident edges. Nearest-neighbour curves require more care because deleting a species changes another species' nearest neighbour only when the deleted species was its unique closest neighbour; tied nearest neighbours must therefore be handled explicitly.

Returns
-------
np.ndarray of shape (len(observed_indices), 2, len(r_values)); axis 0 follows omission order and axis 1 contains PNcp then NNcp.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_all_leave_one_out_observed_curves(
    pool_coords: "np.ndarray",
    observed_indices: "np.ndarray",
    r_values: "np.ndarray"
) -> "np.ndarray":
    """Compute all leave-one-out observed arrangement curves efficiently.

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
        Input order and repeated values are preserved.

    Returns
    -------
    curves : np.ndarray
        Float array of shape (k, 2, len(r_values)), where k is the observed
        richness. curves[j] contains PNcp then NNcp after omitting the
        j-th entry of observed_indices.

    Raises
    ------
    ValueError
        If the pool, observed indices, or thresholds are invalid.
    """
    return curves

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_all_leave_one_out_observed_curves(
    pool_coords: "np.ndarray",
    observed_indices: "np.ndarray",
    r_values: "np.ndarray"
) -> "np.ndarray":
    pool_coords = np.asarray(pool_coords, dtype=float)
    r_values = np.asarray(r_values, dtype=float)

    if (
        pool_coords.ndim != 2
        or pool_coords.shape[0] < 5
        or pool_coords.shape[1] != 2
        or not np.all(np.isfinite(pool_coords))
    ):
        raise ValueError(
            "pool_coords must contain at least five finite two-dimensional species."
        )

    indices = np.asarray(observed_indices)
    if (
        indices.ndim != 1
        or indices.size < 4
        or indices.size >= pool_coords.shape[0]
    ):
        raise ValueError(
            "observed_indices has invalid size."
        )
    if not np.issubdtype(indices.dtype, np.integer):
        if not np.all(np.equal(indices, np.floor(indices))):
            raise ValueError(
                "observed_indices must contain integers."
            )
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

    coords = pool_coords[indices]
    k = int(coords.shape[0])
    reduced_n = k - 1

    delta = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    np.fill_diagonal(distances, np.inf)

    # ---------- PNcp for all omissions ----------
    # Each row contains the k-1 finite incident distances. Searchsorted on
    # sorted rows gives the threshold-specific incident directed degree.
    sorted_rows = np.sort(distances, axis=1)[:, : k - 1]
    degree = np.empty((k, r_values.size), dtype=np.int64)
    for i in range(k):
        degree[i] = np.searchsorted(
            sorted_rows[i],
            r_values,
            side="right",
        )

    full_directed = np.sum(degree, axis=0, dtype=np.int64)
    reduced_directed = full_directed[None, :] - 2 * degree
    pn = reduced_directed.astype(float) / float(
        reduced_n * (reduced_n - 1)
    )

    # ---------- NNcp for all omissions ----------
    # d1/d2 are the first two order statistics in each row. If d2 == d1,
    # the nearest neighbour is tied and deleting one tied neighbour does not
    # alter that species' nearest-neighbour distance.
    d1 = sorted_rows[:, 0]
    d2 = sorted_rows[:, 1]
    nearest_index = np.argmin(distances, axis=1)
    unique_nearest = d2 > d1

    d1_active = d1[:, None] <= r_values[None, :]
    full_active = np.sum(d1_active, axis=0, dtype=np.int64)

    # When species j itself is omitted, remove its original NN indicator.
    nn_count = (
        full_active[None, :] - d1_active.astype(np.int64)
    )

    # If j was the unique nearest neighbour of i, omission j changes i from
    # d1_i to d2_i. The correction is -1 exactly where d1_i <= r < d2_i.
    affected_rows = np.nonzero(unique_nearest)[0]
    for i in affected_rows:
        j = int(nearest_index[i])
        correction = (
            (d2[i] <= r_values).astype(np.int64)
            - (d1[i] <= r_values).astype(np.int64)
        )
        nn_count[j] += correction

    nn = nn_count.astype(float) / float(reduced_n)

    return np.stack((pn, nn), axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    pool = """np.array([
    [0.96,0.04],[0.25,0.36],[0.57,0.46],[0.94,0.19],[0.31,0.08],
    [0.18,0.60],[0.05,0.94],[0.76,0.09],[0.79,0.12],[0.52,0.21],[0.17,0.53]
], dtype=float)"""

    return [
        {
            "setup": f"""import numpy as np
pool_candidate = {pool}
pool_oracle = pool_candidate.copy()
observed_candidate = np.array([0,3,5,7,8], dtype=int)
observed_oracle = observed_candidate.copy()
r_candidate = np.array([0.08,0.18,0.28,0.36,0.44,0.52,0.56], dtype=float)
r_oracle = r_candidate.copy()""",
            "call": "compute_all_leave_one_out_observed_curves(pool_candidate, observed_candidate, r_candidate)",
            "gold_call": "_oracle_compute_all_leave_one_out_observed_curves(pool_oracle, observed_oracle, r_oracle)",
        },
        {
            # Symmetric geometry creates exact nearest-neighbour ties; thresholds
            # are deliberately repeated and unsorted.
            "setup": """import numpy as np
pool_candidate = np.array([
    [0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0],
    [0.5,0.5],[2.0,0.0],[2.0,1.0]
], dtype=float)
pool_oracle = pool_candidate.copy()
observed_candidate = np.array([0,1,2,3,4,5], dtype=int)
observed_oracle = observed_candidate.copy()
r_candidate = np.array([1.0,0.71,0.50,1.0,1.42,0.71], dtype=float)
r_oracle = r_candidate.copy()""",
            "call": "compute_all_leave_one_out_observed_curves(pool_candidate, observed_candidate, r_candidate)",
            "gold_call": "_oracle_compute_all_leave_one_out_observed_curves(pool_oracle, observed_oracle, r_oracle)",
        },
        {
            # Performance case: a cubic strategy that rebuilds the complete
            # pairwise analysis for each omission is intentionally impractical.
            "setup": """import numpy as np
n_obs = 1200
idx = np.arange(n_obs + 1, dtype=np.int64)
pool_candidate = np.column_stack((
    ((idx * 37 + 11) % 2003) / 2003.0,
    ((idx * 991 + 17) % 2017) / 2017.0,
)).astype(float)
pool_oracle = pool_candidate.copy()
observed_candidate = np.arange(n_obs, dtype=int)
observed_oracle = observed_candidate.copy()
base_r = np.linspace(0.002, 0.72, 127)
r_candidate = np.concatenate((base_r[::-1], np.array([0.10,0.10,0.25,0.25], dtype=float)))
r_oracle = r_candidate.copy()""",
            "call": "compute_all_leave_one_out_observed_curves(pool_candidate, observed_candidate, r_candidate)",
            "gold_call": "_oracle_compute_all_leave_one_out_observed_curves(pool_oracle, observed_oracle, r_oracle)",
        },
    ]
