"""
Allocates quadrature points across the multilevel hierarchy for a given finest level big_l and quadrature tolerance eps_quad, using the algebraic quadrature-error model with the supplied fitted constants (level-zero prefactor a0 and smoothness index s0_idx; first-correction prefactor a1 and common correction index s_idx), the per-evaluation cost exponent beta, and the empirical rate p for prefactor propagation across correction levels. Returns the total number of quadrature points (level-zero plus all corrections) as a float. This step validates the closed-form work-optimal allocation and its integer rounding. Deliberately excluded: any integrand evaluation.

Splitting the integrand into a level-zero term plus level differences lets each term be integrated with its own number of nodes; minimizing total work subject to an algebraic quadrature-error budget gives closed-form real-valued node counts, which are then rounded to integers so that the error budget is preserved. Level-zero and correction budgets are handled separately, correction prefactors at finer levels are propagated from the first correction level with the empirical rate, and the cost of one correction evaluation reflects solves on both of its adjacent grids. For the frozen benchmark configuration (big_l = 4, eps_quad = 1.25e-4, a0 = 12.0, s0_idx = 8.0, a1 = 1.4, s_idx = 7.0, beta = 2.0, p = 1.66): the allocation is 21 level-zero points, corrections (42, 24, 14, 8), total 109.

Returns
-------
float — the total allocated quadrature-node count over level zero and all corrections (an integer value returned as a float).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def allocate_points_total(big_l: int, eps_quad: float, a0: float,
                          s0_idx: float, a1: float, s_idx: float,
                          beta: float, p: float, big_t: float,
                          m0: int) -> float:
    """Total quadrature points allocated across the hierarchy.

    Parameters
    ----------
    big_l : int
        Finest correction level, big_l >= 1.
    eps_quad : float
        Quadrature-error tolerance, eps_quad > 0.
    a0 : float
        Fitted algebraic prefactor of the level-zero integrand, a0 > 0.
    s0_idx : float
        Fitted smoothness index of the level-zero integrand, s0_idx >= 1.
    a1 : float
        Fitted algebraic prefactor of the first level difference, a1 > 0.
    s_idx : float
        Common fitted smoothness index of the level differences, s_idx >= 1.
    beta : float
        Cost exponent of one integrand evaluation at level ell
        (work proportional to dt_ell^(-beta)), beta > 0.
    p : float
        Empirical rate used to propagate correction prefactors, p > 0.
    big_t : float
        Maturity, big_t > 0.
    m0 : int
        Level-zero step count, m0 >= 1.

    Returns
    -------
    float
        Total node count: level-zero points plus all correction points
        (an integer value returned as float).

    Raises
    ------
    ValueError
        If any parameter is outside its admissible range.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _s08_alloc_vector(big_l, eps_quad, a0, s0_idx, a1, s_idx, beta, p,
                    big_t, m0):
    """Allocation of quadrature points: level-zero and correction counts.

    One half of eps_quad is assigned to the level-zero term and one half to
    the corrections.  Correction prefactors are propagated from the first
    correction level, A_l = a1 * (dt_l / dt_1)^p.  Per-point correction cost
    is c_l = W_l + W_{l-1} with W_l = dt_l^{-beta}.  Real-valued minimisers
    are rounded up (ceiling)."""
    import numpy as np
    from math import ceil
    dt = np.array([big_t / (m0 * 2 ** l) for l in range(big_l + 1)])
    w_cost = dt ** (-beta)
    n0_star = (2.0 * a0 / eps_quad) ** (2.0 / s0_idx)
    a_l = np.array([a1 * (dt[l] / dt[1]) ** p for l in range(1, big_l + 1)])
    c_l = np.array([w_cost[l] + w_cost[l - 1] for l in range(1, big_l + 1)])
    ssum = float(np.sum(a_l ** (2.0 / (s_idx + 2.0))
                        * c_l ** (s_idx / (s_idx + 2.0))))
    n_star = ((a_l / c_l) ** (2.0 / (s_idx + 2.0))
              * ((2.0 / eps_quad) * ssum) ** (2.0 / s_idx))
    n0 = int(ceil(n0_star - 1e-12))
    nl = [max(1, int(ceil(v - 1e-12))) for v in n_star]
    return n0, nl


def _oracle_allocate_points_total(big_l, eps_quad, a0, s0_idx, a1, s_idx,
                                  beta, p, big_t, m0):
    big_l = int(big_l)
    if big_l < 1:
        raise ValueError("big_l must be >= 1")
    if eps_quad <= 0.0:
        raise ValueError("eps_quad must be positive")
    if a0 <= 0.0 or a1 <= 0.0:
        raise ValueError("a0 and a1 must be positive")
    if s0_idx < 1.0 or s_idx < 1.0:
        raise ValueError("s0_idx and s_idx must be >= 1")
    if beta <= 0.0 or p <= 0.0:
        raise ValueError("beta and p must be positive")
    if big_t <= 0.0 or int(m0) < 1:
        raise ValueError("big_t must be > 0 and m0 >= 1")
    n0, nl = _s08_alloc_vector(big_l, eps_quad, a0, s0_idx, a1, s_idx,
                               beta, p, big_t, int(m0))
    return float(n0 + sum(nl))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "allocate_points_total(4, 1.25e-4, 12.0, 8.0, 1.4, 7.0, 2.0, 1.66, 2.0, 32)",
            "gold_call": "_oracle_allocate_points_total(4, 1.25e-4, 12.0, 8.0, 1.4, 7.0, 2.0, 1.66, 2.0, 32)",
        },
        {
            "setup": "",
            "call": "allocate_points_total(2, 5.0e-4, 12.0, 8.0, 1.4, 7.0, 2.0, 1.66, 2.0, 32)",
            "gold_call": "_oracle_allocate_points_total(2, 5.0e-4, 12.0, 8.0, 1.4, 7.0, 2.0, 1.66, 2.0, 32)",
        },
        {
            "setup": "",
            "call": "allocate_points_total(1, 5.0e-2, 12.0, 8.0, 1.4, 7.0, 2.0, 1.66, 2.0, 32)",
            "gold_call": "_oracle_allocate_points_total(1, 5.0e-2, 12.0, 8.0, 1.4, 7.0, 2.0, 1.66, 2.0, 32)",
        },
    ]
