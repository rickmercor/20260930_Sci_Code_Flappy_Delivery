"""
Reweight each interface set on its own, with the maximum-likelihood reweighting of step 4 restricted to that set's ensembles and order parameter, rescale the weights of each set so that the mean weight of its paths reaching B is one, pool the rescaled sets, and return the natural logarithm of the probability that a path which crosses x = lam_ref reaches B in the pooled ensemble. The submitted function must import inside itself whatever it uses.

maxima_by_set[s] holds the stored paths of set s only, as rows of the maxima of lambda1, lambda2 and lambda3 in the layout of step 3, and interfaces[s] and counts[s] are that set's interfaces and sample counts; set s uses column s for its own reweighting. A path reaches B exactly when its maximum of x (column 0) exceeds 0.9. The probability is conditional, total pooled weight of the paths that cross x = lam_ref and reach B over total pooled weight of the paths that cross x = lam_ref. A set without any path that reaches B cannot be rescaled and raises ValueError.

Returns
-------
return log_p
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rescaled_crossing_probability(maxima_by_set: list, interfaces: list, counts: list,
                                  lam_ref: float) -> float:
    """Log crossing probability into B from independently reweighted sets rescaled to unit reactive weight.

    Args:
        maxima_by_set: list of float arrays of shape (n_paths_s, 3), the stored paths of each set.
        interfaces: list of increasing float arrays, the interfaces of each set.
        counts: list of arrays of sample counts per ensemble of each set.
        lam_ref: the conditioning value of x.

    Returns:
        float: natural logarithm of the pooled conditional probability of reaching B.

    Raises:
        ValueError: if a set has no path that reaches B.
    """
    return log_p

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_rescaled_crossing_probability(maxima_by_set: list, interfaces: list, counts: list,
                                          lam_ref: float) -> float:
    """Log crossing probability into B from independently reweighted sets rescaled to unit reactive weight."""
    import numpy as np
    weights, pooled = [], []
    for i in range(len(interfaces)):
        mx = np.asarray(maxima_by_set[i], dtype=float)
        log_z = _oracle_joint_log_partition_sums(mx[:, i:i + 1], [interfaces[i]], [counts[i]])
        levels = np.searchsorted(np.asarray(interfaces[i], dtype=float), mx[:, i], side="left")
        cum = np.concatenate([[0.0], np.cumsum(np.asarray(counts[i], dtype=float) * np.exp(-log_z))])
        w = 1.0 / cum[levels]
        reactive = mx[:, 0] > 0.9
        if not reactive.any():
            raise ValueError(f"set {i + 1} has no path that reaches B")
        weights.append(w / w[reactive].mean())
        pooled.append(mx)
    w = np.concatenate(weights)
    mx = np.concatenate(pooled)
    ref = mx[:, 0] > lam_ref
    return float(np.log(w[ref & (mx[:, 0] > 0.9)].sum() / w[ref].sum()))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = """import numpy as np
IF = [np.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2]),
      np.array([-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2]),
      np.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2])]
def by_set(n, seed):
    rng = np.random.default_rng(seed)
    out, counts = [], []
    for s in range(3):
        rows, c = [], []
        for k, lam in enumerate(IF[s]):
            m = n + 5 * k
            top = lam + rng.exponential(0.3, m)
            mx = top[:, None] + rng.normal(0.0, 0.04, (m, 3))
            mx[:, s] = top
            rows.append(mx)
            c.append(m)
        out.append(np.concatenate(rows))
        counts.append(np.array(c))
    return out, IF, counts
def raises(f, *args):
    try:
        f(*args)
        return 0
    except ValueError:
        return 1
"""
    return [
        {   # 6.1 three synthetic sets, conditioned on x = -0.8
            "setup": common + "mb, ifs, cs = by_set(30, 7)\n",
            "call": "rescaled_crossing_probability(mb, ifs, cs, -0.8)",
            "gold_call": "_oracle_rescaled_crossing_probability(mb, ifs, cs, -0.8)",
        },
        {   # 6.2 analytic: one set, two interfaces; P(B | -0.8) = (9/37) (8/32)
            "setup": common + """col = np.array([-0.75] * 28 + [-0.65] * 7 + [0.95] * 2 + [-0.6] * 17 + [0.95] * 6)
mb = [np.stack([col, col, col], axis=1)]
def check(out):
    return np.isclose(out, np.log(9 / 37 * 8 / 32), atol=1e-10), float(out)
""",
            "call": "check(rescaled_crossing_probability(mb, [np.array([-0.8, -0.7])], [np.array([37, 23])], -0.8))",
            "gold_call": "check(_oracle_rescaled_crossing_probability(mb, [np.array([-0.8, -0.7])], [np.array([37, 23])], -0.8))",
        },
        {   # 6.3 conditioning on x = -0.5, which part of the pooled paths never cross
            "setup": common + "mb, ifs, cs = by_set(25, 9)\n",
            "call": "rescaled_crossing_probability(mb, ifs, cs, -0.5)",
            "gold_call": "_oracle_rescaled_crossing_probability(mb, ifs, cs, -0.5)",
        },
        {   # 6.4 edge: a set without any path reaching B raises ValueError
            "setup": common + "mb, ifs, cs = by_set(20, 8)\nmb[1] = np.minimum(mb[1], 0.85)\n",
            "call": "raises(rescaled_crossing_probability, mb, ifs, cs, -0.8)",
            "gold_call": "raises(_oracle_rescaled_crossing_probability, mb, ifs, cs, -0.8)",
        },
    ]
