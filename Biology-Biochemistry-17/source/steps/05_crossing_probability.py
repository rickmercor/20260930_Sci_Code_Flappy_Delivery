"""
Given the pooled paths, interfaces and sample counts of step 4 and the log partition sums of all the ensembles, reweight every pooled path as the joint maximum-likelihood reweighting prescribes and return, for each value lam in lams, the natural logarithm of the reweighted probability that a path which crosses lambda1 = x = lam_ref also crosses x = lam. The submitted function must import inside itself whatever it uses.

Column 0 of maxima is the maximum of lambda1 = x over each path, and the inputs have the layout of step 4. The probability is conditional: numerator and denominator are total reweighted weights of pooled paths, the denominator over the paths whose maximum of x exceeds lam_ref and the numerator over those whose maximum exceeds both lam_ref and lam, so the result is 0 for every lam not above lam_ref. A path reaches B exactly when its maximum of x exceeds 0.9.

Returns
-------
return log_p
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def crossing_probability(maxima: "np.ndarray", interfaces: list, counts: list, log_z: "np.ndarray",
                         lam_ref: float, lams: "np.ndarray") -> "np.ndarray":
    """Log of the reweighted probability of crossing each x = lam, given that x = lam_ref is crossed.

    Args:
        maxima: float array of shape (n_paths, n_sets), as in step 4; column 0 is lambda1 = x.
        interfaces: list of n_sets increasing float arrays, as in step 4.
        counts: list of n_sets arrays of sample counts, as in step 4.
        log_z: float array of the log partition sums, ordered as the output of step 4.
        lam_ref: the conditioning value of x.
        lams: float array of values of x.

    Returns:
        np.ndarray: float array of len(lams), the natural logarithms of the conditional probabilities.
    """
    return log_p

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_crossing_probability(maxima: "np.ndarray", interfaces: list, counts: list, log_z: "np.ndarray",
                                 lam_ref: float, lams: "np.ndarray") -> "np.ndarray":
    """Log of the reweighted probability that a path crossing lam_ref along lambda1 = x also crosses each lam."""
    import numpy as np
    maxima = np.asarray(maxima, dtype=float)
    log_z = np.asarray(log_z, dtype=float)
    denom = np.zeros(len(maxima))
    start = 0
    for i in range(len(interfaces)):
        size = len(interfaces[i])
        levels = np.searchsorted(np.asarray(interfaces[i], dtype=float), maxima[:, i], side="left")
        cum = np.concatenate([[0.0], np.cumsum(np.asarray(counts[i], dtype=float) * np.exp(-log_z[start:start + size]))])
        denom += cum[levels]
        start += size
    w = 1.0 / denom
    w /= w.sum()
    ref = maxima[:, 0] > lam_ref
    total = w[ref].sum()
    return np.log(np.array([w[ref & (maxima[:, 0] > lam)].sum() / total for lam in np.atleast_1d(lams)]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = """import numpy as np
IF = [np.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2]),
      np.array([-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2]),
      np.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2])]
def pooled(sets, n, seed):
    rng = np.random.default_rng(seed)
    rows, counts = [], []
    for s in sets:
        c = []
        for k, lam in enumerate(IF[s]):
            m = n + 7 * k
            top = lam + rng.exponential(0.3, m)
            mx = top[:, None] + rng.normal(0.0, 0.04, (m, 3))
            mx[:, s] = top
            rows.append(mx[:, sets])
            c.append(m)
        counts.append(np.array(c))
    return np.concatenate(rows), [IF[s] for s in sets], counts
LAMS = np.array([-0.9, -0.8, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9])
"""
    return [
        {   # 5.1 three sets pooled, conditioned on the first interface of set 1
            "setup": common + "mx, ifs, cs = pooled([0, 1, 2], 30, 5)\nlz = -np.linspace(0.2, 6.0, 29)\n",
            "call": "crossing_probability(mx, ifs, cs, lz, -0.8, LAMS)",
            "gold_call": "_oracle_crossing_probability(mx, ifs, cs, lz, -0.8, LAMS)",
        },
        {   # 5.2 conditioning on x = -0.5, which only part of the pooled weight crosses
            "setup": common + "mx, ifs, cs = pooled([2, 1], 25, 6)\nlz = -np.linspace(0.1, 5.0, 19)\n",
            "call": "crossing_probability(mx, ifs, cs, lz, -0.5, LAMS)",
            "gold_call": "_oracle_crossing_probability(mx, ifs, cs, lz, -0.5, LAMS)",
        },
        {   # 5.3 analytic: one set, two interfaces, partition sums 1 and 9/37 give P(-0.7 | -0.8) = 9/37
            "setup": common + """mx = np.array([[-0.75]] * 28 + [[-0.65]] * 9 + [[-0.6]] * 23)
def check(out):
    return np.isclose(out[0], np.log(9 / 37), atol=1e-10), np.asarray(out, dtype=float)
""",
            "call": "check(crossing_probability(mx, [np.array([-0.8, -0.7])], [np.array([37, 23])], np.array([0.0, np.log(9 / 37)]), -0.8, np.array([-0.7])))",
            "gold_call": "check(_oracle_crossing_probability(mx, [np.array([-0.8, -0.7])], [np.array([37, 23])], np.array([0.0, np.log(9 / 37)]), -0.8, np.array([-0.7])))",
        },
    ]
