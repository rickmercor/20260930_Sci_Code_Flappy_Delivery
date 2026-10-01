"""
Compute an exact, exhaustive second-moment certificate of the fixed-richness regional-pool null for the complete PNcp/NNcp threshold battery.



Let S range over every unique community_size-species subset of the regional pool. For threshold r_t, let X_t(S) be the number of ordered pairs of distinct species in S whose Euclidean distance is <= r_t (the PNcp numerator), and let Y_t(S) be the number of species in S whose nearest-neighbour distance within S is <= r_t (the NNcp numerator). With z(S) = (1, X_1(S), ..., X_T(S), Y_1(S), ..., Y_T(S)), return the integer matrix M = sum over all S of z(S) z(S)^T.



M must be exact. Do not enumerate, sample, or materialize subsets; the running time must be polynomial in the pool size, and inputs with more than 10^8 subsets must be handled. Thresholds keep their input order, repeated thresholds are allowed, and distances exactly equal to a threshold (including zero distances between duplicated coordinates) count as within the threshold.

The exhaustive regional-pool null is a finite distribution, so its moments are exact rather than simulated. The per-threshold SES tests used downstream are not independent: the same species pairs and nearest-neighbour events contribute to several thresholds and to both PNcp and NNcp. The full second-moment matrix fixes the null means and sample standard deviations used for standardization, and also the exact covariance among all statistic-by-threshold tests, which quantifies the multiple-testing dependence discussed in the source study.

Returns
-------
np.ndarray of dtype int64 and shape (1 + 2*len(r_values), 1 + 2*len(r_values)); M[0,0] is the number of subsets, M[0,1:] are the summed counts, and M[1:,1:] are the summed cross-products, ordered PNcp thresholds then NNcp thresholds.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_exact_pool_null_certificate(
    pool_coords: "np.ndarray",
    community_size: int,
    r_values: "np.ndarray"
) -> "np.ndarray":
    """Return the exact exhaustive pool-null second-moment certificate.

    Parameters
    ----------
    pool_coords : np.ndarray
        Finite regional-pool coordinates with shape (n_species, 2) and at
        least three species. Duplicated coordinates are allowed.
    community_size : int
        Fixed richness k of every null community; 2 <= k < n_species.
    r_values : np.ndarray
        Non-empty one-dimensional array of finite, non-negative thresholds.
        Input order and repeated values are preserved.

    Returns
    -------
    certificate : np.ndarray
        int64 array M of shape (1 + 2T, 1 + 2T), M = sum_S z(S) z(S)^T with
        z(S) = (1, X_1..X_T, Y_1..Y_T) as defined in the description.

    Raises
    ------
    ValueError
        If an input is invalid or an entry cannot be represented exactly.
    """
    return certificate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import comb


def _certificate_count(n_pool: int, k: int, n_in: int, n_out: int) -> int:
    if k - n_in < 0 or n_pool - n_in - n_out < k - n_in:
        return 0
    return comb(n_pool - n_in - n_out, k - n_in)


def _oracle_compute_exact_pool_null_certificate(
    pool_coords: "np.ndarray",
    community_size: int,
    r_values: "np.ndarray"
) -> "np.ndarray":
    pool_coords = np.asarray(pool_coords, dtype=float)
    r_values = np.asarray(r_values, dtype=float)

    if (
        pool_coords.ndim != 2
        or pool_coords.shape[0] < 3
        or pool_coords.shape[1] != 2
        or not np.all(np.isfinite(pool_coords))
    ):
        raise ValueError("pool_coords must contain at least three finite two-dimensional species.")
    if (
        not isinstance(community_size, (int, np.integer))
        or int(community_size) < 2
        or int(community_size) >= pool_coords.shape[0]
    ):
        raise ValueError("community_size must be an integer in [2, n_species).")
    if (
        r_values.ndim != 1
        or r_values.size < 1
        or not np.all(np.isfinite(r_values))
        or np.any(r_values < 0.0)
    ):
        raise ValueError("r_values must contain finite non-negative thresholds.")

    n = int(pool_coords.shape[0])
    k = int(community_size)
    t_count = int(r_values.size)
    cnt = lambda n_in, n_out: _certificate_count(n, k, n_in, n_out)

    delta = pool_coords[:, None, :] - pool_coords[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    off_diagonal = ~np.eye(n, dtype=bool)
    adjacency = ((distances[None] <= r_values[:, None, None]) & off_diagonal[None]).astype(np.int64)
    non_adjacency = ((distances[None] > r_values[:, None, None]) & off_diagonal[None]).astype(np.int64)
    degree = adjacency.sum(axis=2)
    n_edges = adjacency.sum(axis=(1, 2)) // 2
    upper_i, upper_j = np.triu_indices(n, 1)

    m = np.zeros((1 + 2 * t_count, 1 + 2 * t_count), dtype=object)
    m[0, 0] = cnt(0, 0)

    for t in range(t_count):
        sum_x = 2 * int(n_edges[t]) * cnt(2, 0)
        sum_y = sum(cnt(1, 0) - cnt(1, int(degree[t, i])) for i in range(n))
        m[0, 1 + t] = m[1 + t, 0] = sum_x
        m[0, 1 + t_count + t] = m[1 + t_count + t, 0] = sum_y

    for t in range(t_count):
        for u in range(t_count):
            same_edges = int(np.sum(adjacency[t] & adjacency[u])) // 2
            one_shared = int(np.sum(degree[t] * degree[u])) - 2 * same_edges
            disjoint = int(n_edges[t]) * int(n_edges[u]) - same_edges - one_shared
            m[1 + t, 1 + u] = 4 * (
                same_edges * cnt(2, 0) + one_shared * cnt(3, 0) + disjoint * cnt(4, 0)
            )

            mask = adjacency[t][upper_i, upper_j] == 1
            ea, eb = upper_i[mask], upper_j[mask]
            total = 0
            if ea.size:
                for a, b in ((ea, eb), (eb, ea)):
                    covered = distances[a, b] <= r_values[u]
                    total += int(ea.size) * cnt(2, 0)
                    total -= sum(cnt(2, int(degree[u, x])) for x in a[~covered])
                weights = np.array([cnt(3, int(d)) for d in degree[u]], dtype=object)
                total += int(ea.size) * (n - 2) * cnt(3, 0)
                total -= int(np.sum(
                    (non_adjacency[u][ea].astype(object) * weights[None, :]) * non_adjacency[u][eb]
                ))
            m[1 + t, 1 + t_count + u] = m[1 + t_count + u, 1 + t] = 2 * total

            if u < t:
                continue
            nested = np.minimum(degree[t], degree[u])
            diagonal_part = sum(cnt(1, 0) - cnt(1, int(d)) for d in nested)
            overlap = adjacency[t] @ adjacency[u].T
            cross_part = 0
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    value = cnt(2, 0)
                    i_open = distances[i, j] > r_values[t]
                    j_open = distances[i, j] > r_values[u]
                    if i_open:
                        value -= cnt(2, int(degree[t, i]))
                    if j_open:
                        value -= cnt(2, int(degree[u, j]))
                    if i_open and j_open:
                        value += cnt(2, int(degree[t, i] + degree[u, j] - overlap[i, j]))
                    cross_part += value
            m[1 + t_count + t, 1 + t_count + u] = diagonal_part + cross_part
            m[1 + t_count + u, 1 + t_count + t] = diagonal_part + cross_part

    if max(abs(int(x)) for x in m.ravel()) >= 2 ** 53:
        raise ValueError("Certificate entries exceed the exactly representable integer range.")
    return np.array([[int(x) for x in row] for row in m], dtype=np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    pool = """np.array([
    [0.96,0.04],[0.25,0.36],[0.57,0.46],[0.94,0.19],[0.31,0.08],
    [0.18,0.60],[0.05,0.94],[0.76,0.09],[0.79,0.12],[0.52,0.21],[0.17,0.53]
], dtype=float)"""
    r_bench = "np.array([0.08,0.18,0.28,0.36,0.44,0.52,0.56], dtype=float)"
    return [
        {
            "setup": f"""import numpy as np
pool_candidate = {pool}
pool_oracle = pool_candidate.copy()
k_candidate = 5
k_oracle = 5
r_candidate = {r_bench}
r_oracle = r_candidate.copy()""",
            "call": "compute_exact_pool_null_certificate(pool_candidate, k_candidate, r_candidate)",
            "gold_call": "_oracle_compute_exact_pool_null_certificate(pool_oracle, k_oracle, r_oracle)",
        },
        {
            "setup": """import numpy as np
pool_candidate = np.array([
    [0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0],
    [0.5,0.5],[2.0,0.0],[2.0,1.0],[1.0,1.0]
], dtype=float)
pool_oracle = pool_candidate.copy()
k_candidate = 4
k_oracle = 4
r_candidate = np.array([1.0,0.71,0.0,1.0,1.42,0.71], dtype=float)
r_oracle = r_candidate.copy()""",
            "call": "compute_exact_pool_null_certificate(pool_candidate, k_candidate, r_candidate)",
            "gold_call": "_oracle_compute_exact_pool_null_certificate(pool_oracle, k_oracle, r_oracle)",
        },
        {
            "setup": f"""import numpy as np
idx = np.arange(30)
pool_candidate = np.column_stack((((idx*37+11) % 101)/101, ((idx*53+7) % 103)/103)).astype(float)
pool_oracle = pool_candidate.copy()
k_candidate = 15
k_oracle = 15
r_candidate = {r_bench}
r_oracle = r_candidate.copy()""",
            "call": "compute_exact_pool_null_certificate(pool_candidate, k_candidate, r_candidate)",
            "gold_call": "_oracle_compute_exact_pool_null_certificate(pool_oracle, k_oracle, r_oracle)",
        },
    ]
