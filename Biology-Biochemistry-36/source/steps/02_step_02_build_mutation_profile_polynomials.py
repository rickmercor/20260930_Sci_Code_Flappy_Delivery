"""
Lift an independent RNA substitution library into a multivariate affine probability profile whose variables label disjoint position groups.

The grouped lift retains which subset classes contribute to each mixed replacement order. Restricting all group variables to one common mutation rate later recovers the original one-parameter library exactly.

Returns
-------
np.ndarray: grouped multivariate coefficients with shape (L, 4) + (2,) * G.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_mutation_profile_polynomials(
    sequence: "np.ndarray",
    mutation_kernel: "np.ndarray",
    position_groups: "np.ndarray",
) -> "np.ndarray":
    """Return grouped multivariate nucleotide-probability coefficients.

    Nucleotides are coded A = 0, C = 1, G = 2 and U = 3. Row ``x`` of
    ``mutation_kernel`` is the conditional replacement distribution when a
    reference nucleotide ``x`` mutates; its diagonal is zero and each row
    sums to one. ``position_groups[i]`` assigns position ``i`` to one of
    ``G`` mutation variables. Group labels must be contiguous ``0..G-1``,
    every label must occur, and ``1 <= G <= 4``.

    With ``g = position_groups[i]``, site ``i`` has nucleotide ``a`` with
    probability

    ``P[i,a](mu_0,...,mu_{G-1}) = (1-mu_g) 1[a=sequence[i]]
                                      + mu_g mutation_kernel[sequence[i],a]``.

    Return the coefficient tensor in increasing degree along every mutation
    variable. Its shape is ``(L, 4) + (2,) * G``. For each site, only the
    all-zero coefficient index and the index with degree one on its assigned
    group axis may be nonzero. Setting every ``mu_g = mu`` recovers the
    original independent one-rate mutation library.

    Parameters
    ----------
    sequence : np.ndarray
        Nonempty one-dimensional integer array with entries in ``0..3``.
    mutation_kernel : np.ndarray
        Finite nonnegative row-stochastic array of shape ``(4, 4)`` with a
        zero diagonal.
    position_groups : np.ndarray
        One-dimensional integer group labels of shape ``(L,)`` satisfying
        the contiguous-label conditions above; booleans are rejected.

    Returns
    -------
    np.ndarray
        Float coefficient tensor of shape ``(L, 4) + (2,) * G``.

    Raises
    ------
    ValueError
        If the sequence, mutation kernel or position-group assignment violates
        the conditions above.
    """
    return profile

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _grouped_validated_codes(sequence):
    """Return validated integer nucleotide codes."""
    import numpy as np

    raw = np.asarray(sequence)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype == bool:
        raise ValueError("sequence must be a nonempty one-dimensional code array")
    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError):
        raise ValueError("sequence must hold integer codes") from None
    if not np.all(np.isfinite(numeric)) or np.any(numeric != np.round(numeric)):
        raise ValueError("sequence must hold integer codes")
    if np.any(numeric < 0) or np.any(numeric > 3):
        raise ValueError("sequence codes must lie in 0..3")
    return numeric.astype(int)


def _grouped_validated_kernel(mutation_kernel):
    """Return a valid conditional replacement kernel."""
    import numpy as np

    try:
        kernel = np.array(mutation_kernel, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("mutation_kernel must be numeric") from None
    if kernel.shape != (4, 4):
        raise ValueError("mutation_kernel must have shape (4, 4)")
    if not np.all(np.isfinite(kernel)) or np.any(kernel < 0.0):
        raise ValueError("mutation_kernel must be finite and nonnegative")
    if not np.allclose(np.diag(kernel), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("mutation_kernel must have a zero diagonal")
    if not np.allclose(np.sum(kernel, axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("mutation_kernel rows must sum to one")
    return kernel


def _grouped_validated_labels(position_groups, length):
    """Return contiguous integer group labels for all positions."""
    import numpy as np

    raw = np.asarray(position_groups)
    if raw.ndim != 1 or raw.shape != (length,) or raw.dtype == bool:
        raise ValueError("position_groups must be a one-dimensional length-L integer array")
    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError):
        raise ValueError("position_groups must hold integer labels") from None
    if not np.all(np.isfinite(numeric)) or np.any(numeric != np.round(numeric)):
        raise ValueError("position_groups must hold integer labels")
    groups = numeric.astype(int)
    if np.any(groups < 0):
        raise ValueError("position_groups must be nonnegative")
    unique = np.unique(groups)
    if unique.size == 0 or unique.size > 4 or not np.array_equal(unique, np.arange(unique.size)):
        raise ValueError("group labels must be contiguous 0..G-1 with 1 <= G <= 4")
    return groups


def _oracle_build_mutation_profile_polynomials(
    sequence: "np.ndarray",
    mutation_kernel: "np.ndarray",
    position_groups: "np.ndarray",
) -> "np.ndarray":
    """Reference construction of grouped affine site probabilities."""
    import numpy as np

    codes = _grouped_validated_codes(sequence)
    kernel = _grouped_validated_kernel(mutation_kernel)
    groups = _grouped_validated_labels(position_groups, codes.size)
    group_count = int(np.max(groups)) + 1
    profile = np.zeros((codes.size, 4) + (2,) * group_count, dtype=float)
    zero = (0,) * group_count
    for i, code in enumerate(codes):
        linear = [0] * group_count
        linear[int(groups[i])] = 1
        profile[(i, int(code)) + zero] = 1.0
        profile[(i, slice(None)) + tuple(linear)] = kernel[int(code)]
        profile[(i, int(code)) + tuple(linear)] -= 1.0
    return profile

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return grouped-profile normal, boundary, edge and validation cases."""
    base = (
        "import numpy as np\n"
        "K = np.ones((4, 4), dtype=float) / 3.0\n"
        "np.fill_diagonal(K, 0.0)\n"
        "def _sig(a):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    k = np.arange(a.size, dtype=float).reshape(a.shape)\n"
        "    shape_code = sum((q + 1) * s for q, s in enumerate(a.shape))\n"
        "    return float(shape_code + np.sum(a * np.cos(k + .25)) + np.sum(a * a) / 7.0)\n"
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
            "setup": base + "S = np.array([2, 2, 0, 1, 3]); G = np.array([0, 1, 2, 0, 1])\n",
            "call": "_sig(build_mutation_profile_polynomials(S.copy(), K.copy(), G.copy()))",
            "gold_call": "_sig(_oracle_build_mutation_profile_polynomials(S.copy(), K.copy(), G.copy()))",
        },
        {
            "setup": base + "S = np.array([0]); G = np.array([0]); K2 = np.array([[0, .2, .3, .5], [.1, 0, .6, .3], [.4, .2, 0, .4], [.7, .1, .2, 0]])\n",
            "call": "_sig(build_mutation_profile_polynomials(S.copy(), K2.copy(), G.copy()))",
            "gold_call": "_sig(_oracle_build_mutation_profile_polynomials(S.copy(), K2.copy(), G.copy()))",
        },
        {
            "setup": base + "S = np.array([3, 2, 1, 0]); G = np.array([3, 2, 1, 0])\n",
            "call": "_sig(build_mutation_profile_polynomials(S.copy(), K.copy(), G.copy()))",
            "gold_call": "_sig(_oracle_build_mutation_profile_polynomials(S.copy(), K.copy(), G.copy()))",
        },
        {
            "setup": base + status + "S = np.array([0, 4]); G = np.array([0, 1])\n",
            "call": "_status(lambda: build_mutation_profile_polynomials(S, K, G))",
            "gold_call": "_status(lambda: _oracle_build_mutation_profile_polynomials(S, K, G))",
        },
        {
            "setup": base + status + "S = np.array([0, 1]); G = np.array([0, 1]); K[2] *= .9\n",
            "call": "_status(lambda: build_mutation_profile_polynomials(S, K, G))",
            "gold_call": "_status(lambda: _oracle_build_mutation_profile_polynomials(S, K, G))",
        },
        {
            "setup": base + status + "S = np.array([0, 1, 2]); G = np.array([0, 2, 2])\n",
            "call": "_status(lambda: build_mutation_profile_polynomials(S, K, G))",
            "gold_call": "_status(lambda: _oracle_build_mutation_profile_polynomials(S, K, G))",
        },
    ]
