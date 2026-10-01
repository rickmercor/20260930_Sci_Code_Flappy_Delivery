"""
Compute the total and pair-class-resolved inside weights of every subsequence of an RNA under a five-rule stochastic grammar with a minimum enclosed loop and nearest-neighbour stacking.

Separating neutral, Watson–Crick and G–U pair-topped derivations makes the shared pair context explicit: the outer pair applies a different stack multiplier to each directly enclosed pair class, while non-pair-topped interior derivations receive no stack multiplier.

Returns
-------
np.ndarray: [total, neutral-pair, WC-pair, GU-pair] inside weights, shape (4, L + 2, L + 1).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_stacked_inside_weights(
    sequence: "np.ndarray",
    rule_weights: "np.ndarray",
    left_emission: "np.ndarray",
    right_emission: "np.ndarray",
    pair_emission: "np.ndarray",
    stacking_factors: "np.ndarray",
    min_loop: int,
) -> "np.ndarray":
    """Return total and pair-class-resolved inside weights of all subsequences.

    Nucleotides are coded A = 0, C = 1, G = 2, U = 3 and positions are
    1-based. ``rule_weights`` is ``(t_P, t_L, t_R, t_B, t_E)``. The grammar
    has one nonterminal ``S`` and five rules: ``S -> a S b`` emits paired
    endpoints with factor ``t_P * pair_emission[a, b]`` and is allowed only
    when at least ``min_loop`` nucleotides lie between them; ``S -> a S`` and
    ``S -> S a`` emit one unpaired endpoint with factors
    ``t_L * left_emission[a]`` and ``t_R * right_emission[a]``;
    ``S -> S S`` splits into two nonempty strings with factor ``t_B``; and
    ``S -> epsilon`` is the sole empty derivation with factor ``t_E``.
    Distinct parse trees count separately.

    Pair-topped states have three classes: neutral/other, Watson–Crick (A-U,
    U-A, G-C, C-G), and G–U wobble (G-U, U-G). A neutral outer or inner pair
    contributes stack factor 1. Otherwise use ``stacking_factors[k]``, where
    ``k = 0, 1, 2, 3`` for ordered outer/inner classes WC/WC, WC/GU, GU/WC,
    and GU/GU. Only a directly pair-topped interior receives this factor.

    Channel 0 of the result is the total inside weight. Channels 1, 2 and 3
    are respectively the neutral, Watson–Crick and G–U pair-topped weights.
    Entry ``[0, i, j]`` covers ``x_i ... x_j`` for ``1 <= i <= L + 1`` and
    ``i - 1 <= j <= L``; ``j = i - 1`` is empty. Every other entry is zero,
    and ``result[0, 1, L]`` is the partition function. Consequently, for a
    nonempty span, channel 0 includes the sum of channels 1 through 3 plus
    all non-pair-rooted derivations.

    Parameters
    ----------
    sequence : np.ndarray
        Integer nucleotide codes, shape ``(L,)`` with ``L >= 1``.
    rule_weights : np.ndarray
        Finite nonnegative ``(t_P, t_L, t_R, t_B, t_E)``, shape ``(5,)``.
    left_emission, right_emission : np.ndarray
        Finite nonnegative factors indexed by nucleotide, shape ``(4,)``.
    pair_emission : np.ndarray
        Finite nonnegative factors indexed ``[5' nucleotide, 3' nucleotide]``,
        shape ``(4, 4)``; noncanonical entries may be nonzero.
    stacking_factors : np.ndarray
        Finite nonnegative stacking factors, shape ``(4,)``.
    min_loop : int
        Nonnegative minimum number of nucleotides enclosed by a pair.

    Returns
    -------
    np.ndarray
        Float array with shape ``(4, L + 2, L + 1)``.

    Raises
    ------
    ValueError
        If ``sequence`` is not a nonempty one-dimensional array of integer
        codes in ``0..3``, if a factor array has the wrong shape or holds a
        negative or non-finite value, or if ``min_loop`` is not a nonnegative
        integer (booleans are rejected).
    """
    return weights

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _fixed_factor_array(value, shape, name):
    """Return a finite nonnegative float array of the given shape."""
    import numpy as np

    try:
        array = np.array(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric") from None
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not (np.all(np.isfinite(array)) and np.all(array >= 0.0)):
        raise ValueError(f"{name} must hold finite nonnegative values")
    return array


def _fixed_pair_channel(five, three):
    """Return 1 for neutral, 2 for Watson–Crick and 3 for G–U."""
    if (five, three) in ((0, 3), (3, 0), (2, 1), (1, 2)):
        return 2
    if (five, three) in ((2, 3), (3, 2)):
        return 3
    return 1


def _oracle_compute_stacked_inside_weights(
    sequence: "np.ndarray",
    rule_weights: "np.ndarray",
    left_emission: "np.ndarray",
    right_emission: "np.ndarray",
    pair_emission: "np.ndarray",
    stacking_factors: "np.ndarray",
    min_loop: int,
) -> "np.ndarray":
    """Reference class-resolved inside recursion in increasing span length."""
    import numpy as np

    raw = np.asarray(sequence)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype == bool:
        raise ValueError("sequence must be a nonempty one-dimensional code array")
    try:
        codes = raw.astype(float)
    except (TypeError, ValueError):
        raise ValueError("sequence must hold integer codes") from None
    if not np.all(np.isfinite(codes)) or np.any(codes != np.round(codes)):
        raise ValueError("sequence must hold integer codes")
    if np.any(codes < 0) or np.any(codes > 3):
        raise ValueError("sequence codes must lie in 0..3")
    x = codes.astype(int)
    t_p, t_l, t_r, t_b, t_e = _fixed_factor_array(rule_weights, (5,), "rule_weights")
    e_l = _fixed_factor_array(left_emission, (4,), "left_emission")
    e_r = _fixed_factor_array(right_emission, (4,), "right_emission")
    e_p = _fixed_factor_array(pair_emission, (4, 4), "pair_emission")
    stack = _fixed_factor_array(stacking_factors, (4,), "stacking_factors")
    if isinstance(min_loop, bool) or not isinstance(min_loop, (int, np.integer)) or min_loop < 0:
        raise ValueError("min_loop must be a nonnegative integer")

    n = x.size
    weights = np.zeros((4, n + 2, n + 1), dtype=float)
    for i in range(1, n + 2):
        weights[0, i, i - 1] = t_e

    for span in range(1, n + 1):
        for i in range(1, n - span + 2):
            j = i + span - 1
            total = t_l * e_l[x[i - 1]] * weights[0, i + 1, j]
            total += t_r * e_r[x[j - 1]] * weights[0, i, j - 1]
            if span > 1:
                total += t_b * float(np.dot(weights[0, i, i:j], weights[0, i + 1:j + 1, j]))

            if j - i - 1 >= min_loop:
                outer = _fixed_pair_channel(int(x[i - 1]), int(x[j - 1]))
                inner_neutral = weights[1, i + 1, j - 1]
                inner_wc = weights[2, i + 1, j - 1]
                inner_gu = weights[3, i + 1, j - 1]
                nonpair_inner = weights[0, i + 1, j - 1] - inner_neutral - inner_wc - inner_gu
                if outer == 1:
                    paired_inner = inner_neutral + inner_wc + inner_gu
                elif outer == 2:
                    paired_inner = inner_neutral + stack[0] * inner_wc + stack[1] * inner_gu
                else:
                    paired_inner = inner_neutral + stack[2] * inner_wc + stack[3] * inner_gu
                weights[outer, i, j] = (
                    t_p * e_p[x[i - 1], x[j - 1]] * (nonpair_inner + paired_inner)
                )
            weights[0, i, j] = total + float(np.sum(weights[1:, i, j]))
    return weights

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return class-sensitive normal, boundary, edge and validation cases."""
    base = (
        "import numpy as np\n"
        "W = np.array([0.41, 0.27, 0.16, 0.12, 0.80])\n"
        "EL = np.array([0.28, 0.24, 0.19, 0.33])\n"
        "ER = np.array([0.22, 0.31, 0.26, 0.18])\n"
        "EP = np.zeros((4, 4))\n"
        "for a, b, v in [(0, 3, 0.90), (3, 0, 1.30), (2, 1, 1.60), (1, 2, 1.10), (2, 3, 0.45), (3, 2, 0.70)]:\n"
        "    EP[a, b] = v\n"
        "ST = np.array([1.9, 1.4, 1.2, 0.7])\n"
        "def _code(text):\n"
        "    return np.array(['ACGU'.index(ch) for ch in text])\n"
        "def _sig(t, n):\n"
        "    t = np.asarray(t, dtype=float)\n"
        "    if t.shape != (4, n + 2, n + 1):\n"
        "        return -1.0\n"
        "    z = t[0, 1, n]\n"
        "    if not np.isfinite(z) or z <= 0.0:\n"
        "        return -2.0\n"
        "    k = np.arange(t.size, dtype=float).reshape(t.shape)\n"
        "    return float(np.log(z) / 10.0 + np.sum(np.abs(t) / z * (1.0 + k % 7)) / t.size + np.sum(t / z * np.cos(k + .17)) / 9.0)\n"
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
            "setup": base + "S = _code('GGCAUAGCCA')\n",
            "call": "_sig(compute_stacked_inside_weights(S.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 3), 10)",
            "gold_call": "_sig(_oracle_compute_stacked_inside_weights(S.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 3), 10)",
        },
        {
            "setup": base + "S = _code('UGAACGUCA')\n",
            "call": "_sig(compute_stacked_inside_weights(S.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 0), 9)",
            "gold_call": "_sig(_oracle_compute_stacked_inside_weights(S.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 0), 9)",
        },
        {
            "setup": base + "S = _code('GUGCAAAGCAC')\n",
            "call": "_sig(compute_stacked_inside_weights(S.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 1), 11)",
            "gold_call": "_sig(_oracle_compute_stacked_inside_weights(S.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 1), 11)",
        },
        {
            "setup": base + "S = _code('GGCC')\n",
            "call": "float(1e3 * compute_stacked_inside_weights(S.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 0)[2, 1, 4])",
            "gold_call": "float(1e3 * _oracle_compute_stacked_inside_weights(S.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 0)[2, 1, 4])",
        },
        {
            "setup": base + "S = _code('AC'); EP[0, 1] = .73\n",
            "call": "float(1e3 * compute_stacked_inside_weights(S.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 0)[1, 1, 2])",
            "gold_call": "float(1e3 * _oracle_compute_stacked_inside_weights(S.copy(), W.copy(), EL.copy(), ER.copy(), EP.copy(), ST.copy(), 0)[1, 1, 2])",
        },
        {
            "setup": base + "S = _code('CAGUUCAGGUAC')\nEP2 = EP.copy(); EP2[0, 0] = 0.3\n",
            "call": "_sig(compute_stacked_inside_weights(S.copy(), W.copy(), EL.copy(), ER.copy(), EP2.copy(), np.array([2.5, 0.6, 1.8, 0.2]), 2), 12)",
            "gold_call": "_sig(_oracle_compute_stacked_inside_weights(S.copy(), W.copy(), EL.copy(), ER.copy(), EP2.copy(), np.array([2.5, 0.6, 1.8, 0.2]), 2), 12)",
        },
        {
            "setup": base + status + "S = _code('GCAU')\n",
            "call": "_status(lambda: compute_stacked_inside_weights(S, W, EL, ER, EP, np.array([1.0, 1.0, 1.0]), 3))",
            "gold_call": "_status(lambda: _oracle_compute_stacked_inside_weights(S, W, EL, ER, EP, np.array([1.0, 1.0, 1.0]), 3))",
        },
        {
            "setup": base + status + "S = _code('GCAU')\n",
            "call": "_status(lambda: compute_stacked_inside_weights(S, W, EL, ER, EP, ST, True))",
            "gold_call": "_status(lambda: _oracle_compute_stacked_inside_weights(S, W, EL, ER, EP, ST, True))",
        },
    ]
