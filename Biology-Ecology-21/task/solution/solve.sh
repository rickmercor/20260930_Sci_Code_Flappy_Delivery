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


def compute_observed_arrangement_curves(
    coords: "np.ndarray",
    r_values: "np.ndarray"
) -> "np.ndarray":
    coords = np.asarray(coords, dtype=float)
    r_values = np.asarray(r_values, dtype=float)

    if (
        coords.ndim != 2
        or coords.shape[0] < 2
        or coords.shape[1] < 1
    ):
        raise ValueError(
            "coords must contain at least two species and one dimension."
        )

    if not np.all(np.isfinite(coords)):
        raise ValueError(
            "coords must contain only finite values."
        )

    if r_values.ndim != 1 or r_values.size < 1:
        raise ValueError(
            "r_values must be a non-empty one-dimensional array."
        )

    if (
        not np.all(np.isfinite(r_values))
        or np.any(r_values < 0.0)
    ):
        raise ValueError(
            "r_values must contain finite non-negative values."
        )

    differences = (
        coords[:, np.newaxis, :]
        - coords[np.newaxis, :, :]
    )

    distances = np.sqrt(
        np.sum(differences ** 2, axis=2)
    )

    n_species = coords.shape[0]

    off_diagonal = distances[
        ~np.eye(n_species, dtype=bool)
    ]

    nearest_matrix = distances.copy()
    np.fill_diagonal(nearest_matrix, np.inf)

    nearest_distances = np.min(
        nearest_matrix,
        axis=1,
    )

    pncp = np.array(
        [
            np.mean(off_diagonal <= r)
            for r in r_values
        ],
        dtype=float,
    )

    nncp = np.array(
        [
            np.mean(nearest_distances <= r)
            for r in r_values
        ],
        dtype=float,
    )

    return np.vstack((pncp, nncp))

import numpy as np
from math import comb


def _certificate_count(n_pool: int, k: int, n_in: int, n_out: int) -> int:
    if k - n_in < 0 or n_pool - n_in - n_out < k - n_in:
        return 0
    return comb(n_pool - n_in - n_out, k - n_in)


def compute_exact_pool_null_certificate(
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

import numpy as np
from scipy.optimize import minimize


def _boundary_partition_solution(gram, free, upper):
    n_free = free.size
    system = np.zeros((n_free + 1, n_free + 1))
    system[:n_free, :n_free] = gram[np.ix_(free, free)]
    system[:n_free, n_free] = -1.0
    system[n_free, :n_free] = 1.0
    rhs_constant = np.zeros(n_free + 1)
    rhs_constant[n_free] = 1.0
    rhs_slope = np.zeros(n_free + 1)
    rhs_slope[:n_free] = -gram[np.ix_(free, upper)].sum(axis=1)
    rhs_slope[n_free] = -float(upper.size)
    return np.linalg.solve(system, rhs_constant), np.linalg.solve(system, rhs_slope)


def _boundary_kkt_certified(gram, alpha, rho, upper_bound):
    tol = 1e-10
    margin = gram @ alpha - rho
    zero = alpha <= tol
    bounded = alpha >= upper_bound - tol
    free = ~zero & ~bounded
    return (
        abs(alpha.sum() - 1.0) < 1e-12
        and np.all(alpha >= -tol)
        and np.all(alpha <= upper_bound + tol)
        and np.all(margin[zero] >= -1e-11)
        and np.all(margin[bounded] <= 1e-11)
        and np.all(np.abs(margin[free]) <= 1e-11)
    )


def fit_regional_boundary_certificate(
    pool_coords: "np.ndarray",
    nu: float = 0.05
) -> "np.ndarray":
    pool_coords = np.asarray(pool_coords, dtype=float)
    if (
        pool_coords.ndim != 2
        or pool_coords.shape[0] < 3
        or pool_coords.shape[1] != 2
        or not np.all(np.isfinite(pool_coords))
    ):
        raise ValueError("pool_coords must contain at least three finite two-dimensional species.")
    if not np.isfinite(nu) or float(nu) <= 0.0 or float(nu) >= 1.0:
        raise ValueError("nu must be finite and strictly between 0 and 1.")

    n = int(pool_coords.shape[0])
    squared = np.sum((pool_coords[:, None, :] - pool_coords[None, :, :]) ** 2, axis=2)
    median_distance = float(np.median(np.sqrt(squared)[np.triu_indices(n, 1)]))
    if not np.isfinite(median_distance) or median_distance <= 0.0:
        raise ValueError("Median pairwise distance must be positive and finite.")
    sigma = 1.0 / median_distance
    gram = np.exp(-sigma * squared)
    upper_bound = 1.0 / (float(nu) * n)

    start = minimize(
        fun=lambda a: 0.5 * float(a @ gram @ a),
        x0=np.full(n, 1.0 / n),
        jac=lambda a: gram @ a,
        bounds=[(0.0, upper_bound)] * n,
        constraints=[{"type": "eq", "fun": lambda a: float(a.sum() - 1.0),
                      "jac": lambda a: np.ones(n)}],
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 10000, "disp": False},
    ).x
    proposal_tol = 1e-6 * max(1.0, upper_bound)
    free = np.nonzero((start > proposal_tol) & (start < upper_bound - proposal_tol))[0]
    upper = np.nonzero(start >= upper_bound - proposal_tol)[0]

    for _ in range(4 * n):
        if free.size == 0:
            raise ValueError("OCSVM fit requires at least one free support species.")
        constant, slope = _boundary_partition_solution(gram, free, upper)
        alpha = np.zeros(n)
        alpha[upper] = upper_bound
        alpha[free] = constant[:-1] + slope[:-1] * upper_bound
        rho = float(constant[-1] + slope[-1] * upper_bound)
        if _boundary_kkt_certified(gram, alpha, rho, upper_bound):
            break

        margin = gram @ alpha - rho
        in_free = np.zeros(n, dtype=bool)
        in_free[free] = True
        in_upper = np.zeros(n, dtype=bool)
        in_upper[upper] = True

        violation = (
            np.where(in_free & (alpha < 0.0), -alpha, 0.0)
            + np.where(in_free & (alpha > upper_bound), alpha - upper_bound, 0.0)
            + np.where(~in_free & ~in_upper & (margin < 0.0), -margin, 0.0)
            + np.where(in_upper & (margin > 0.0), margin, 0.0)
        )

        worst = int(np.argmax(violation))
        if in_free[worst]:
            free = free[free != worst]
            if alpha[worst] > upper_bound:
                upper = np.sort(np.r_[upper, worst])
        elif in_upper[worst]:
            upper = upper[upper != worst]
            free = np.sort(np.r_[free, worst])
        else:
            free = np.sort(np.r_[free, worst])
    else:
        raise ValueError("OCSVM KKT conditions could not be certified.")

    zero = np.setdiff1d(np.arange(n), np.r_[free, upper])
    alpha_constant = np.zeros(n)
    alpha_constant[free] = constant[:-1]
    alpha_slope = np.zeros(n)
    alpha_slope[free] = slope[:-1]
    alpha_slope[upper] = 1.0
    margin_constant = gram @ alpha_constant - constant[-1]
    margin_slope = gram @ alpha_slope - slope[-1]

    c_low, c_high = 1.0 / n, np.inf
    for p_values, q_values in (
        (constant[:-1], slope[:-1]),
        (-constant[:-1], 1.0 - slope[:-1]),
        (margin_constant[zero], margin_slope[zero]),
        (-margin_constant[upper], -margin_slope[upper]),
    ):
        for p, q in zip(np.atleast_1d(p_values), np.atleast_1d(q_values)):
            if abs(q) <= 1e-13:
                continue
            if q > 0.0:
                c_low = max(c_low, -p / q)
            else:
                c_high = min(c_high, -p / q)

    nu_low = 0.0 if not np.isfinite(c_high) else 1.0 / (n * c_high)
    nu_high = min(1.0, 1.0 / (n * c_low))

    return np.concatenate(([sigma], alpha, [rho, nu_low, nu_high]))

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
        compute_observed_arrangement_curves(group, r_values)
        for group in accepted[: n_complete * k].reshape(
            n_complete, k, 2
        )
    ])

    mean = np.mean(curves, axis=0)
    sd = np.std(curves, axis=0, ddof=1)

    if np.any(~np.isfinite(sd)) or np.any(sd <= 0.0):
        raise ValueError("Displacement null requires positive sample SD.")

    return mean, sd


def analyze_full_community_dual_null(
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

    observed = compute_observed_arrangement_curves(
        pool_coords[indices],
        r_values
    )

    certificate = compute_exact_pool_null_certificate(
        pool_coords,
        k,
        r_values
    )

    pool_mean, pool_sd = _dual_null_pool_moments(
        certificate,
        k,
        t_count
    )

    boundary = fit_regional_boundary_certificate(
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

import numpy as np


def compute_all_leave_one_out_observed_curves(
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

import numpy as np


def compute_leave_one_out_disagreement_profile(
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

    full_evidence = analyze_full_community_dual_null(
        pool_coords,
        indices,
        r_values,
        nu,
        n_candidates,
        critical
    )

    loo_curves = compute_all_leave_one_out_observed_curves(
        pool_coords,
        indices,
        r_values
    )

    shared = analyze_full_community_dual_null(
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

import numpy as np


def compute_null_model_sensitivity_burden(
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

    observed = compute_observed_arrangement_curves(
        pool_coords[observed_indices],
        r_values
    )

    certificate = compute_exact_pool_null_certificate(
        pool_coords,
        k,
        r_values
    )

    boundary = fit_regional_boundary_certificate(
        pool_coords,
        nu
    )

    full_evidence = analyze_full_community_dual_null(
        pool_coords,
        observed_indices,
        r_values,
        nu,
        n_candidates,
        critical
    )

    loo_curves = compute_all_leave_one_out_observed_curves(
        pool_coords,
        observed_indices,
        r_values
    )

    profile = compute_leave_one_out_disagreement_profile(
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
SCICODE_GOLD_EOF
