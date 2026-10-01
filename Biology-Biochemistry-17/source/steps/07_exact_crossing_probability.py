"""
Compute, without sampling, the natural logarithm of the exact probability that a path of the lattice Metropolis walk which leaves A and crosses x = lam_ref goes on to reach B, at inverse temperature beta, from the transition probabilities of the walk and its equilibrium distribution. The submitted function must import inside itself whatever it uses.

The lattice, free energy, proposal rule and states are those of the problem statement; each of the four moves is proposed with probability 1/4 and an off-lattice proposal leaves the walker in place. A path leaving A is the equilibrium trajectory from the last frame in A before the walker leaves A up to its first entry into A or B, and it crosses x = lam_ref when a frame has x > lam_ref. When no site outside A and B lies beyond lam_ref, every path that crosses it has reached B and the result is 0.

Returns
-------
return log_p
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def exact_crossing_probability(beta: float, lam_ref: float) -> float:
    """Exact log probability that a path leaving A which crosses x = lam_ref reaches B.

    Args:
        beta: inverse temperature (non-negative).
        lam_ref: the conditioning value of x, with -0.9 <= lam_ref < 0.9.

    Returns:
        float: natural logarithm of the conditional probability.
    """
    return log_p

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_exact_crossing_probability(beta: float, lam_ref: float) -> float:
    """Exact log probability that a path leaving A which crosses x = lam_ref reaches B, for the lattice chain."""
    import numpy as np
    nx, ny = 30, 20
    x = -1.45 + 0.1 * np.arange(nx)
    y = -0.95 + 0.1 * np.arange(ny)
    u = ((x[:, None] ** 2 - 1.0) ** 2 + y[None, :] ** 2).ravel()
    ii = np.repeat(np.arange(nx), ny)
    jj = np.tile(np.arange(ny), nx)
    size = nx * ny
    kmat = np.zeros((size, size))
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = ii + di, jj + dj
        ok = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)
        src = np.nonzero(ok)[0]
        dst = ni[ok] * ny + nj[ok]
        kmat[src, dst] += 0.25 * np.minimum(1.0, np.exp(-beta * (u[dst] - u[src])))
    kmat[np.arange(size), np.arange(size)] += 1.0 - kmat.sum(axis=1)
    in_a = x[ii] < -0.9
    hits = []
    for target in (x[ii] > 0.9, (x[ii] > lam_ref) & ~in_a):
        free = ~(in_a | target)
        h = target.astype(float)
        h[free] = np.linalg.solve(np.eye(free.sum()) - kmat[np.ix_(free, free)], kmat[np.ix_(free, target)].sum(axis=1))
        hits.append(h)
    flux = np.exp(-beta * u[in_a])[:, None] * kmat[in_a] * (~in_a)[None, :]
    return float(np.log((flux @ hits[0]).sum() / (flux @ hits[1]).sum()))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {   # 7.1 the benchmark temperature and conditioning interface
            "setup": "",
            "call": "exact_crossing_probability(10.0, -0.8)",
            "gold_call": "_oracle_exact_crossing_probability(10.0, -0.8)",
        },
        {   # 7.2 a higher temperature, checked against direct simulation (2.442e-2 +- 0.04e-2)
            "setup": "",
            "call": "exact_crossing_probability(2.5, -0.8)",
            "gold_call": "_oracle_exact_crossing_probability(2.5, -0.8)",
        },
        {   # 7.3 a conditioning interface further out
            "setup": "",
            "call": "exact_crossing_probability(5.0, -0.3)",
            "gold_call": "_oracle_exact_crossing_probability(5.0, -0.3)",
        },
        {   # 7.4 boundary: beta = 0, every on-lattice proposal accepted
            "setup": "",
            "call": "exact_crossing_probability(0.0, -0.8)",
            "gold_call": "_oracle_exact_crossing_probability(0.0, -0.8)",
        },
        {   # 7.5 edge: no site outside A and B lies beyond lam_ref, so the probability is 1
            "setup": "",
            "call": "exact_crossing_probability(10.0, 0.88)",
            "gold_call": "_oracle_exact_crossing_probability(10.0, 0.88)",
        },
    ]
