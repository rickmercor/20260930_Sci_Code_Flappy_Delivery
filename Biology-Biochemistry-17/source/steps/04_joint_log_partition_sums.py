"""
Given pooled paths from interface ensembles of one or more interface sets, described by the maxima of the sets' order parameters over each path, together with the interfaces of each set and the number of samples drawn in each ensemble, return the natural logarithms of the maximum-likelihood (multistate Bennett acceptance ratio) estimates of the conditional partition sums of all the ensembles, with every pooled path used in the joint estimate and the reweighted weights of the pooled paths normalised to sum to one. The submitted function must import inside itself whatever it uses.

Column s of maxima belongs to the s-th set passed in interfaces; a path crosses interface k of that set when its maximum is greater than the interface value. Each ensemble is a sample of the equilibrium ensemble of paths leaving A, selected by the condition that its paths cross its interface; the partition sums are expressed relative to the total reweighted weight of the pooled paths, normalised to one. counts[s][k] is the number of samples drawn in ensemble (s, k), repeats included, and the pooled paths may appear in any order. The output lists the ensembles set by set, each set in increasing interface order, and must be accurate to 1e-10 in the logarithms. Pooled data that do not fix the partition sums, such as a path that crosses no interface of any set, raise ValueError.

Returns
-------
return log_z
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def joint_log_partition_sums(maxima: "np.ndarray", interfaces: list, counts: list) -> "np.ndarray":
    """Log conditional partition sums of all interface ensembles from the jointly reweighted pooled paths.

    Args:
        maxima: float array of shape (n_paths, n_sets); entry (p, s) is the maximum of the order
            parameter of set s over pooled path p.
        interfaces: list of n_sets increasing float arrays, the interfaces of each set.
        counts: list of n_sets arrays; counts[s][k] is the number of samples drawn in ensemble (s, k).

    Returns:
        np.ndarray: float array of length sum of len(interfaces[s]), the log partition sums ordered set by
            set and interface by interface, for path weights normalised to sum to one.

    Raises:
        ValueError: if a pooled path crosses no interface of any set, or the self-consistent solution
            does not converge because the pooled data do not determine it.
    """
    return log_z

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_joint_log_partition_sums(maxima: "np.ndarray", interfaces: list, counts: list) -> "np.ndarray":
    """Log conditional partition sums of every interface ensemble of every set, from all pooled paths."""
    import numpy as np
    maxima = np.asarray(maxima, dtype=float)
    m = len(interfaces)
    levels = np.stack([np.searchsorted(np.asarray(interfaces[i], dtype=float), maxima[:, i], side="left")
                       for i in range(m)], axis=1)
    if np.any(levels.max(axis=1) == 0):
        raise ValueError("a pooled path crosses no interface of any set")
    cells, mult = np.unique(levels, axis=0, return_counts=True)
    n = [np.asarray(c, dtype=float) for c in counts]
    z = [np.ones(len(c)) for c in n]
    for _ in range(100000):
        denom = np.zeros(len(cells))
        for i in range(m):
            denom += np.concatenate([[0.0], np.cumsum(n[i] / z[i])])[cells[:, i]]
        w = mult / denom
        w /= w.sum()
        new = [np.array([w[cells[:, i] >= kk].sum() for kk in range(1, len(n[i]) + 1)]) for i in range(m)]
        change = max(np.max(np.abs(a / b - 1.0)) for a, b in zip(new, z))
        z = new
        if change < 1e-13:
            return np.log(np.concatenate(z))
    raise ValueError("the self-consistent equations did not converge")

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
            top = lam + rng.exponential(0.15, m)
            mx = top[:, None] + rng.normal(0.0, 0.04, (m, 3))
            mx[:, s] = top
            rows.append(mx[:, sets])
            c.append(m)
        counts.append(np.array(c))
    return np.concatenate(rows), [IF[s] for s in sets], counts
def raises(f, *args):
    try:
        f(*args)
        return 0
    except ValueError:
        return 1
"""
    return [
        {   # 4.1 three sets pooled, unequal sample counts
            "setup": common,
            "call": "joint_log_partition_sums(*pooled([0, 1, 2], 40, 1))",
            "gold_call": "_oracle_joint_log_partition_sums(*pooled([0, 1, 2], 40, 1))",
        },
        {   # 4.2 analytic: one set, two interfaces; 9 of the 37 paths of ensemble 1 cross interface 2
            "setup": common + """mx = np.array([[-0.75]] * 28 + [[-0.65]] * 9 + [[-0.6]] * 23)
def ratio(out):
    return np.isclose(out[1] - out[0], np.log(9 / 37), atol=1e-10), np.asarray(out, dtype=float)
""",
            "call": "ratio(joint_log_partition_sums(mx, [np.array([-0.8, -0.7])], [np.array([37, 23])]))",
            "gold_call": "ratio(_oracle_joint_log_partition_sums(mx, [np.array([-0.8, -0.7])], [np.array([37, 23])]))",
        },
        {   # 4.3 two identical sets sampled separately give identical partition sums
            "setup": common + """a, _, ca = pooled([0], 40, 2)
b, _, cb = pooled([0], 25, 3)
mx = np.repeat(np.concatenate([a, b]), 2, axis=1)
def same(out):
    return np.allclose(out[:10], out[10:], rtol=0, atol=1e-10), np.asarray(out, dtype=float)
""",
            "call": "same(joint_log_partition_sums(mx, [IF[0], IF[0]], [ca[0], cb[0]]))",
            "gold_call": "same(_oracle_joint_log_partition_sums(mx, [IF[0], IF[0]], [ca[0], cb[0]]))",
        },
        {   # 4.4 boundary: a single interface, crossed by every path, has log partition sum 0
            "setup": common + "mx = np.array([[-0.3], [0.1], [-0.55]])\n",
            "call": "joint_log_partition_sums(mx, [np.array([-0.6])], [np.array([3])])",
            "gold_call": "_oracle_joint_log_partition_sums(mx, [np.array([-0.6])], [np.array([3])])",
        },
        {   # 4.5 edge: a pooled path that crosses no interface of any set raises ValueError
            "setup": common + """mx, ifs, cs = pooled([0, 1], 30, 4)
mx[0] = [-0.85, -0.75]
""",
            "call": "raises(joint_log_partition_sums, mx, ifs, cs)",
            "gold_call": "raises(_oracle_joint_log_partition_sums, mx, ifs, cs)",
        },
    ]
