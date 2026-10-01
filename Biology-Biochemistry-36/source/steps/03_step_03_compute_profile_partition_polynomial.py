"""
Compute the exact multivariate mutation-rate polynomial of a stacked RNA grammar averaged over a grouped independent substitution profile.

The recursion must preserve both kinds of dependence simultaneously: multidimensional substitution-order coefficients across position groups and pair context shared between an inner emission and its adjacent stacking factor.

Returns
-------
np.ndarray: exact grouped partition coefficients, shape (n_0 + 1, ..., n_(G-1) + 1).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_profile_partition_polynomial(
    profile_polynomials: "np.ndarray",
    position_groups: "np.ndarray",
    rule_weights: "np.ndarray",
    left_emission: "np.ndarray",
    right_emission: "np.ndarray",
    pair_emission: "np.ndarray",
    stacking_factors: "np.ndarray",
    min_loop: int,
) -> "np.ndarray":
    """Return the exact grouped profile-mean partition polynomial.

    For ``G`` contiguous position groups, ``profile_polynomials`` has shape
    ``(L, 4) + (2,) * G`` and follows the grouped affine contract of
    ``build_mutation_profile_polynomials``. Site ``i`` may depend only on the
    variable selected by ``position_groups[i]``: only its all-zero coefficient
    and its degree-one coefficient on that group axis may be nonzero. Its four
    nucleotide polynomials sum identically to one and are nonnegative at both
    endpoints of their active variable.

    The grammar and stack classes are those of
    ``compute_stacked_inside_weights``. A pair may enclose at least
    ``min_loop`` nucleotides, and a stack factor applies only when the interior
    derivation is directly pair-topped. Since an inner pair's nucleotide
    identities determine both its emission and the adjacent stack, those
    shared endpoint probabilities must be combined jointly. Retaining both
    endpoint identities is exact; an algebraically equivalent sufficient
    context is also valid.

    Polynomial products are multidimensional convolutions. If group ``g``
    contains ``n_g`` positions, return a coefficient grid of shape
    ``(n_0 + 1, ..., n_{G-1} + 1)``. Entry ``r`` is the coefficient of
    ``prod_g mu_g**r_g``. No coefficient may be discarded within those bounds.

    Parameters
    ----------
    profile_polynomials : np.ndarray
        Finite grouped coefficients with shape ``(L, 4) + (2,) * G``, where
        ``L >= 1`` and ``1 <= G <= 4``.
    position_groups : np.ndarray
        Contiguous integer labels ``0..G-1`` of shape ``(L,)``, with every
        label present.
    rule_weights : np.ndarray
        Finite nonnegative ``(t_P, t_L, t_R, t_B, t_E)``, shape ``(5,)``.
    left_emission, right_emission : np.ndarray
        Finite nonnegative factors indexed by nucleotide, shape ``(4,)``.
    pair_emission : np.ndarray
        Finite nonnegative factors indexed ``[5' nucleotide, 3' nucleotide]``,
        shape ``(4, 4)``.
    stacking_factors : np.ndarray
        Finite nonnegative values for WC/WC, WC/GU, GU/WC and GU/GU,
        shape ``(4,)``; a neutral pair on either side gives factor one.
    min_loop : int
        Nonnegative minimum number of nucleotides enclosed by a pair.

    Returns
    -------
    np.ndarray
        Multivariate float coefficient grid with shape determined by the group
        population counts.

    Raises
    ------
    ValueError
        If an input violates any condition above.
    """
    return coefficients

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.signal import convolve as _scipy_nd_convolve


def _multivar_factor_array(value, shape, name):
    """Return a finite nonnegative float array of the requested shape."""
    import numpy as np

    try:
        array = np.array(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric") from None
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must hold finite nonnegative values")
    return array


def _multivar_stack_table(stacking_factors):
    """Expand four ordered WC/GU factors to endpoint identities."""
    import numpy as np

    kind = np.full((4, 4), -1, dtype=int)
    for a, b in ((0, 3), (3, 0), (2, 1), (1, 2)):
        kind[a, b] = 0
    for a, b in ((2, 3), (3, 2)):
        kind[a, b] = 1
    table = np.ones((4, 4, 4, 4), dtype=float)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    if kind[a, b] >= 0 and kind[c, d] >= 0:
                        table[a, b, c, d] = stacking_factors[2 * kind[a, b] + kind[c, d]]
    return table


def _multivar_profile_and_groups(profile_polynomials, position_groups):
    """Validate a grouped site-probability tensor and return it with labels."""
    import numpy as np

    try:
        profile = np.array(profile_polynomials, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("profile_polynomials must be numeric") from None
    if profile.ndim < 3 or profile.ndim > 6 or profile.shape[0] == 0 or profile.shape[1] != 4:
        raise ValueError("profile_polynomials must have shape (L, 4) + (2,) * G")
    group_count = profile.ndim - 2
    if group_count < 1 or group_count > 4 or profile.shape[2:] != (2,) * group_count:
        raise ValueError("profile coefficient axes must all have length two with 1 <= G <= 4")
    if not np.all(np.isfinite(profile)):
        raise ValueError("profile_polynomials must be finite")

    raw = np.asarray(position_groups)
    if raw.ndim != 1 or raw.shape != (profile.shape[0],) or raw.dtype == bool:
        raise ValueError("position_groups must be a one-dimensional length-L integer array")
    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError):
        raise ValueError("position_groups must hold integer labels") from None
    if not np.all(np.isfinite(numeric)) or np.any(numeric != np.round(numeric)):
        raise ValueError("position_groups must hold integer labels")
    groups = numeric.astype(int)
    if np.any(groups < 0) or not np.array_equal(np.unique(groups), np.arange(group_count)):
        raise ValueError("position_groups must use every contiguous label 0..G-1")

    zero = (0,) * group_count
    for i, group in enumerate(groups):
        linear = [0] * group_count
        linear[int(group)] = 1
        allowed = np.zeros((2,) * group_count, dtype=bool)
        allowed[zero] = True
        allowed[tuple(linear)] = True
        if np.any(np.abs(profile[i][:, ~allowed]) > 1e-12):
            raise ValueError("each site may depend only on its assigned group variable")
        constant = profile[(i, slice(None)) + zero]
        slope = profile[(i, slice(None)) + tuple(linear)]
        if np.any(constant < -1e-12) or np.any(constant + slope < -1e-12):
            raise ValueError("site probabilities must be nonnegative at both active endpoints")
        if not np.isclose(np.sum(constant), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("site probabilities at the zero endpoint must sum to one")
        if not np.isclose(np.sum(slope), 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("site probability polynomials must sum identically to one")
    return profile, groups


def _trim_multivar(poly, counts):
    """Return the coefficient box supported by a position-count vector."""
    return poly[tuple(slice(0, int(count) + 1) for count in counts)]


def _accumulate_multivar_product(destination, factors, count_vectors, scale=1.0):
    """Accumulate a bounded direct multidimensional convolution."""
    import numpy as np

    dimension = destination.ndim
    product = np.ones((1,) * dimension, dtype=float)
    total_counts = np.zeros(dimension, dtype=int)
    for factor, counts in zip(factors, count_vectors):
        counts = np.asarray(counts, dtype=int)
        product = _scipy_nd_convolve(
            product, _trim_multivar(factor, counts), mode="full", method="direct"
        )
        total_counts += counts
    destination[tuple(slice(0, int(count) + 1) for count in total_counts)] += float(scale) * product


def _oracle_compute_profile_partition_polynomial(
    profile_polynomials: "np.ndarray",
    position_groups: "np.ndarray",
    rule_weights: "np.ndarray",
    left_emission: "np.ndarray",
    right_emission: "np.ndarray",
    pair_emission: "np.ndarray",
    stacking_factors: "np.ndarray",
    min_loop: int,
) -> "np.ndarray":
    """Reference endpoint-conditioned multivariate inside recursion."""
    import numpy as np

    profile, groups = _multivar_profile_and_groups(profile_polynomials, position_groups)
    n = profile.shape[0]
    group_count = profile.ndim - 2
    group_sizes = np.bincount(groups, minlength=group_count).astype(int)
    coefficient_shape = tuple(int(size) + 1 for size in group_sizes)
    t_p, t_l, t_r, t_b, t_e = _multivar_factor_array(rule_weights, (5,), "rule_weights")
    e_l = _multivar_factor_array(left_emission, (4,), "left_emission")
    e_r = _multivar_factor_array(right_emission, (4,), "right_emission")
    e_p = _multivar_factor_array(pair_emission, (4, 4), "pair_emission")
    stack = _multivar_stack_table(
        _multivar_factor_array(stacking_factors, (4,), "stacking_factors")
    )
    if isinstance(min_loop, bool) or not isinstance(min_loop, (int, np.integer)) or min_loop < 0:
        raise ValueError("min_loop must be a nonnegative integer")

    prefix = np.zeros((n + 1, group_count), dtype=int)
    site_counts = np.zeros((n, group_count), dtype=int)
    for position, group in enumerate(groups):
        site_counts[position, int(group)] = 1
        prefix[position + 1] = prefix[position] + site_counts[position]

    total = np.zeros((n + 2, n + 1) + coefficient_shape, dtype=float)
    pair_mean = np.zeros_like(total)
    pair_context = np.zeros((n + 2, n + 1, 4, 4) + coefficient_shape, dtype=float)
    zero = (0,) * group_count
    for i in range(1, n + 2):
        total[(i, i - 1) + zero] = t_e

    active_pairs = [(a, b) for a in range(4) for b in range(4) if e_p[a, b] != 0.0]
    for span in range(1, n + 1):
        for i in range(1, n - span + 2):
            j = i + span - 1
            current_counts = prefix[j] - prefix[i - 1]
            left_inner_counts = prefix[j] - prefix[i]
            right_inner_counts = prefix[j - 1] - prefix[i - 1]

            if j - i - 1 >= min_loop:
                inner_counts = prefix[j - 1] - prefix[i]
                nonpair_inner = total[i + 1, j - 1] - pair_mean[i + 1, j - 1]
                for a, b in active_pairs:
                    conditioned = nonpair_inner.copy()
                    if span >= 4:
                        deep_counts = inner_counts - site_counts[i] - site_counts[j - 2]
                        for c, d in active_pairs:
                            _accumulate_multivar_product(
                                conditioned,
                                (
                                    profile[i, c],
                                    profile[j - 2, d],
                                    pair_context[i + 1, j - 1, c, d],
                                ),
                                (site_counts[i], site_counts[j - 2], deep_counts),
                                stack[a, b, c, d],
                            )
                    pair_context[i, j, a, b] = t_p * e_p[a, b] * conditioned
                    _accumulate_multivar_product(
                        pair_mean[i, j],
                        (profile[i - 1, a], profile[j - 1, b], pair_context[i, j, a, b]),
                        (site_counts[i - 1], site_counts[j - 1], inner_counts),
                    )

            value = pair_mean[i, j].copy()
            for nucleotide in range(4):
                _accumulate_multivar_product(
                    value,
                    (profile[i - 1, nucleotide], total[i + 1, j]),
                    (site_counts[i - 1], left_inner_counts),
                    t_l * e_l[nucleotide],
                )
                _accumulate_multivar_product(
                    value,
                    (profile[j - 1, nucleotide], total[i, j - 1]),
                    (site_counts[j - 1], right_inner_counts),
                    t_r * e_r[nucleotide],
                )
            if span > 1:
                for split in range(i, j):
                    left_counts = prefix[split] - prefix[i - 1]
                    right_counts = prefix[j] - prefix[split]
                    _accumulate_multivar_product(
                        value,
                        (total[i, split], total[split + 1, j]),
                        (left_counts, right_counts),
                        t_b,
                    )
            total[i, j] = value
    return total[1, n].copy()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return multivariate normal, boundary, edge and validation cases."""
    base = (
        "import numpy as np\n"
        "W = np.array([0.41, 0.27, 0.16, 0.12, 0.80])\n"
        "EL = np.array([0.28, 0.24, 0.19, 0.33])\n"
        "ER = np.array([0.22, 0.31, 0.26, 0.18])\n"
        "EP = np.zeros((4, 4))\n"
        "for a, b, v in [(0, 3, .90), (3, 0, 1.30), (2, 1, 1.60), (1, 2, 1.10), (2, 3, .45), (3, 2, .70)]:\n"
        "    EP[a, b] = v\n"
        "ST = np.array([1.9, 1.4, 1.2, .7])\n"
        "K = np.ones((4, 4)) / 3.0\n"
        "np.fill_diagonal(K, 0.0)\n"
        "def _profile(text, groups, kernel=K):\n"
        "    x = np.array(['ACGU'.index(ch) for ch in text])\n"
        "    groups = np.asarray(groups, dtype=int)\n"
        "    gcount = int(groups.max()) + 1\n"
        "    p = np.zeros((len(x), 4) + (2,) * gcount)\n"
        "    zero = (0,) * gcount\n"
        "    for i, code in enumerate(x):\n"
        "        linear = [0] * gcount; linear[groups[i]] = 1\n"
        "        p[(i, code) + zero] = 1.0\n"
        "        p[(i, slice(None)) + tuple(linear)] = kernel[code]\n"
        "        p[(i, code) + tuple(linear)] -= 1.0\n"
        "    return p\n"
        "def _sig(c):\n"
        "    c = np.asarray(c, dtype=float)\n"
        "    k = np.arange(c.size, dtype=float).reshape(c.shape)\n"
        "    return float(sum((q + 1) * s for q, s in enumerate(c.shape)) + np.sum(c * np.cos(k + .4)) + np.sum(np.abs(c)) / 11.0)\n"
    )
    status = (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    return [
        {
            "setup": base + "G = np.array([0,1,2,3,0,1,2,3,0,1]); P = _profile('GGCAUAGCCA', G)\n",
            "call": "_sig(compute_profile_partition_polynomial(P.copy(), G.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 3))",
            "gold_call": "_sig(_oracle_compute_profile_partition_polynomial(P.copy(), G.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 3))",
        },
        {
            "setup": base + "G = np.array([0,0,1,2,1,2,0,1,2]); P = _profile('UGAACGUCA', G)\n",
            "call": "_sig(compute_profile_partition_polynomial(P.copy(), G.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 0))",
            "gold_call": "_sig(_oracle_compute_profile_partition_polynomial(P.copy(), G.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 0))",
        },
        {
            "setup": base + "G = np.array([0]); P = _profile('G', G)\n",
            "call": "_sig(compute_profile_partition_polynomial(P.copy(), G.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 3))",
            "gold_call": "_sig(_oracle_compute_profile_partition_polynomial(P.copy(), G.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 3))",
        },
        {
            "setup": base + "G = np.array([0,1,0,1,0,1,0,1]); K2 = np.array([[0,.2,.3,.5],[.1,0,.6,.3],[.4,.2,0,.4],[.7,.1,.2,0]]); P = _profile('AACGUUGC', G, K2); EP[0,1] = .37\n",
            "call": "_sig(compute_profile_partition_polynomial(P.copy(), G.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), np.array([2.5,.6,1.8,.2]), 0))",
            "gold_call": "_sig(_oracle_compute_profile_partition_polynomial(P.copy(), G.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), np.array([2.5,.6,1.8,.2]), 0))",
        },
        {
            "setup": base + status + "G = np.array([0,1,0,1]); P = _profile('GCAU', G); P[0,0,1,1] = 1e-3\n",
            "call": "_status(lambda: compute_profile_partition_polynomial(P, G, W, EL, ER, EP, ST, 0))",
            "gold_call": "_status(lambda: _oracle_compute_profile_partition_polynomial(P, G, W, EL, ER, EP, ST, 0))",
        },
        {
            "setup": base + status + "G = np.array([0,1,0,1]); P = _profile('GCAU', G)\n",
            "call": "_status(lambda: compute_profile_partition_polynomial(P, G, W, EL, ER, EP, ST, True))",
            "gold_call": "_status(lambda: _oracle_compute_profile_partition_polynomial(P, G, W, EL, ER, EP, ST, True))",
        },
    ]
