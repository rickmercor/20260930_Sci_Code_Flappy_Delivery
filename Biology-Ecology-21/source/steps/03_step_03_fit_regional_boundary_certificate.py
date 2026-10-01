"""
Fit the regional functional-space boundary and certify it.



Set sigma to the reciprocal of the median Euclidean distance over all unique regional-pool species pairs and use K(x,s)=exp(-sigma*||x-s||^2). With c = 1/(nu*N), alpha is the exact minimizer of 0.5*alpha^T K alpha subject to sum(alpha)=1 and 0<=alpha_i<=c; rho is the mean of (K alpha)_i over free support species (0<alpha_i<c). The returned alpha and rho must satisfy the KKT optimality conditions to within 1e-10; a solution accurate only to optimizer tolerance is not sufficient.



The optimal solution partitions the species into zero, free, and upper-bounded dual coefficients. Return also the exact endpoints nu_lo <= nu <= nu_hi of the maximal closed interval of nu on which this partition remains optimal (use nu_lo = 0 if the partition persists as nu -> 0, and cap nu_hi at 1).



Return [sigma, alpha_1, ..., alpha_N, rho, nu_lo, nu_hi].

The source study treats the OCSVM parameter nu as a context-dependent choice (conservative values such as 0.01 or 0.001 versus 0.05-0.1 when extreme trait values are present). Because the displacement null is conditioned on the fitted boundary, the interval of nu over which the boundary's support structure is unchanged is a direct, exact statement of how sensitive the downstream null-model inference is to that choice.

Returns
-------
np.ndarray of float and shape (N + 4,): sigma, the N dual coefficients in regional-pool order, rho, nu_lo, nu_hi.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_regional_boundary_certificate(
    pool_coords: "np.ndarray",
    nu: float = 0.05
) -> "np.ndarray":
    """Fit and certify the regional OCSVM boundary.

    Parameters
    ----------
    pool_coords : np.ndarray
        Finite regional-pool coordinates with shape (n_species, 2) and at
        least three species.
    nu : float, default=0.05
        OCSVM parameter strictly between 0 and 1; the dual upper bound is
        1/(nu*n_species).

    Returns
    -------
    certificate : np.ndarray
        Float array [sigma, alpha_1..alpha_N, rho, nu_lo, nu_hi].

    Raises
    ------
    ValueError
        If an input is invalid, the median pairwise distance is not positive,
        the optimum has no free support species, or the KKT conditions cannot
        be certified.
    """
    return certificate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_fit_regional_boundary_certificate(
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
nu_candidate = 0.05
nu_oracle = 0.05""",
            "call": "fit_regional_boundary_certificate(pool_candidate, nu_candidate)",
            "gold_call": "_oracle_fit_regional_boundary_certificate(pool_oracle, nu_oracle)",
        },
        {
            "setup": f"""import numpy as np
pool_candidate = {pool}
pool_oracle = pool_candidate.copy()
nu_candidate = 0.30
nu_oracle = 0.30""",
            "call": "fit_regional_boundary_certificate(pool_candidate, nu_candidate)",
            "gold_call": "_oracle_fit_regional_boundary_certificate(pool_oracle, nu_oracle)",
        },
        {
            "setup": """import numpy as np
pool_candidate = np.array([
    [0.05,0.10],[0.20,0.80],[0.42,0.35],
    [0.65,0.90],[0.80,0.15],[0.95,0.55]
], dtype=float)
pool_oracle = pool_candidate.copy()
nu_candidate = 0.60
nu_oracle = 0.60""",
            "call": "fit_regional_boundary_certificate(pool_candidate, nu_candidate)",
            "gold_call": "_oracle_fit_regional_boundary_certificate(pool_oracle, nu_oracle)",
        },
    ]
