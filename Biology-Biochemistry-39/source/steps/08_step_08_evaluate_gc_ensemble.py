"""
Evaluate the expected G+C fraction of GC-weighted distinct designs and the mean number of proposals an exact rejection sampler spends per accepted design.

Weighting designs by an exponential of their G+C content tunes composition, and a sampler that draws through overlapping residue assignments must reject part of its proposals to stay exact.

Returns
-------
np.ndarray: float [expected G+C fraction, expected proposals per accepted design].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_gc_ensemble(
    distinct_counts: "np.ndarray",
    assignment_counts: "np.ndarray",
    length: int,
    weight: float,
) -> "np.ndarray":
    """Return the expected G+C fraction and the proposals per accepted design.

    ``distinct_counts[g]`` is the number of distinct designs with ``g`` G-C
    or C-G pairs; ``assignment_counts[s, g]`` is the number of those designs
    counted in assignment row ``s``, where a design may be counted in several
    rows and every design is counted in at least one. Unpaired positions are
    A, so a design with ``g`` strong pairs holds ``2 g`` G and C nucleotides
    among ``length`` positions.

    Give each distinct design the probability proportional to
    ``exp(weight * N)``, with ``N`` its number of G and C nucleotides, and
    return ``[fraction, proposals]``:

    * ``fraction`` is the expected value of ``N / length``;
    * ``proposals`` is the expected number of proposals per accepted design
      of the sampler that picks a row ``s`` with probability proportional to
      the ``exp(weight * N)``-weighted count of row ``s``, draws a design
      counted in row ``s`` with probability proportional to
      ``exp(weight * N)``, accepts it with probability one over the number
      of rows counting it, and otherwise starts again.

    Both values must hold to a relative ``1e-12`` for every ``|weight|`` up
    to 50, where single Boltzmann factors exceed double precision.

    Parameters
    ----------
    distinct_counts : np.ndarray
        Non-negative counts of length ``k + 1``, not all zero.
    assignment_counts : np.ndarray
        Non-negative counts of shape ``(rows, k + 1)``.
    length : int
        Sequence length, at least ``max(1, 2 k)``.
    weight : float
        Boltzmann weight per G or C nucleotide, with ``|weight| <= 50``.

    Returns
    -------
    np.ndarray
        Float array ``[fraction, proposals]``.

    Raises
    ------
    ValueError
        If ``distinct_counts`` is not a non-empty one-dimensional array of
        finite non-negative numbers with a positive entry, if
        ``assignment_counts`` is not a two-dimensional array of finite
        non-negative numbers with ``k + 1`` columns, if a column of
        ``assignment_counts`` sums to less than the matching entry of
        ``distinct_counts``, if ``length`` is not an integer of at least
        ``max(1, 2 k)``, or if ``weight`` is not a finite real number with
        ``|weight| <= 50`` (booleans are rejected).
    """
    return summary

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_evaluate_gc_ensemble(
    distinct_counts: "np.ndarray",
    assignment_counts: "np.ndarray",
    length: int,
    weight: float,
) -> "np.ndarray":
    """Reference implementation (log-domain Boltzmann sums)."""
    import math

    import numpy as np

    distinct = np.asarray(distinct_counts, dtype=float)
    table = np.asarray(assignment_counts, dtype=float)
    if (distinct.ndim != 1 or distinct.size == 0 or not np.all(np.isfinite(distinct))
            or np.any(distinct < 0) or not np.any(distinct > 0)):
        raise ValueError("distinct_counts must be non-negative, finite and not all zero")
    if (table.ndim != 2 or table.shape[1] != distinct.size or not np.all(np.isfinite(table))
            or np.any(table < 0)):
        raise ValueError("assignment_counts must be a non-negative table with k + 1 columns")
    if np.any(table.sum(axis=0) < distinct):
        raise ValueError("every distinct design must be counted in some assignment row")
    minimum_length = max(1, 2 * (distinct.size - 1))
    if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or int(length) < minimum_length:
        raise ValueError("length must be an integer of at least max(1, 2 k)")
    if (isinstance(weight, bool) or not isinstance(weight, (int, float, np.integer, np.floating))
            or not math.isfinite(float(weight)) or abs(float(weight)) > 50.0):
        raise ValueError("weight must be a finite real number with |weight| <= 50")
    beta = float(weight)
    strong = np.arange(distinct.size, dtype=float)

    def _log_weighted_total(counts):
        # log of sum over entries of count * exp(2 * weight * g), without overflow
        grid = np.broadcast_to(strong, counts.shape)
        positive = counts > 0
        exponents = np.log(counts[positive]) + 2.0 * beta * grid[positive]
        top = float(np.max(exponents))
        return top + math.log(float(np.sum(np.exp(exponents - top))))

    log_designs = _log_weighted_total(distinct)
    positive_numerator = (distinct > 0) & (strong > 0)
    if np.any(positive_numerator):
        numerator_exponents = (
            np.log(distinct[positive_numerator])
            + np.log(2.0 * strong[positive_numerator])
            + 2.0 * beta * strong[positive_numerator]
        )
        numerator_top = float(np.max(numerator_exponents))
        log_nucleotides = numerator_top + math.log(
            float(np.sum(np.exp(numerator_exponents - numerator_top)))
        )
        fraction = math.exp(log_nucleotides - log_designs) / int(length)
    else:
        fraction = 0.0
    proposals = math.exp(_log_weighted_total(table) - log_designs)
    return np.array([fraction, proposals], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    data = (
        "import math\n"
        "import numpy as np\n"
        "small = np.array([0, 0, 4, 10, 6, 2])\n"
        "small_rows = np.array([[0, 0, 4, 10, 6, 2], [0, 0, 0, 3, 6, 2], [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])\n"
        "wide = np.array([math.comb(30, g) for g in range(31)])\n"
        "wide_rows = np.vstack([wide, np.array([math.comb(30, g) if g % 3 == 0 else 0 for g in range(31)])])\n"
        "huge = np.array([1e308, 1e308])\n"
        "huge_rows = huge.reshape(1, -1)\n"
        "def _pair(a):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (2,):\n"
        "        return -1.0\n"
        "    return float(a[0] + 0.37 * a[1])\n"
    )
    status = data + (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    return [
        {
            "setup": data,
            "call": "_pair(evaluate_gc_ensemble(small.copy(), small_rows.copy(), 14, 0.3))",
            "gold_call": "_pair(_oracle_evaluate_gc_ensemble(small.copy(), small_rows.copy(), 14, 0.3))",
        },
        {
            "setup": data,
            "call": "_pair(evaluate_gc_ensemble(small.copy(), small_rows.copy(), 20, 0))",
            "gold_call": "_pair(_oracle_evaluate_gc_ensemble(small.copy(), small_rows.copy(), 20, 0))",
        },
        {
            "setup": data,
            "call": "_pair(evaluate_gc_ensemble(wide.copy(), wide_rows.copy(), 80, 45.0))",
            "gold_call": "_pair(_oracle_evaluate_gc_ensemble(wide.copy(), wide_rows.copy(), 80, 45.0))",
        },
        {
            "setup": data,
            "call": "_pair(evaluate_gc_ensemble(wide.copy(), wide_rows.copy(), 64, -38.5))",
            "gold_call": "_pair(_oracle_evaluate_gc_ensemble(wide.copy(), wide_rows.copy(), 64, -38.5))",
        },
        {
            "setup": data,
            "call": "_pair(evaluate_gc_ensemble(wide.copy(), wide_rows.copy(), 70, -0.21))",
            "gold_call": "_pair(_oracle_evaluate_gc_ensemble(wide.copy(), wide_rows.copy(), 70, -0.21))",
        },
        {
            "setup": data,
            "call": "_pair(evaluate_gc_ensemble(np.array([7]), np.array([[7], [7], [0]]), 9, 2.0))",
            "gold_call": "_pair(_oracle_evaluate_gc_ensemble(np.array([7]), np.array([[7], [7], [0]]), 9, 2.0))",
        },
        {
            "setup": data,
            "call": "_pair(evaluate_gc_ensemble(huge.copy(), huge_rows.copy(), 2, 0.0))",
            "gold_call": "_pair(_oracle_evaluate_gc_ensemble(huge.copy(), huge_rows.copy(), 2, 0.0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: evaluate_gc_ensemble(small.copy(), small_rows.copy(), 14, 50.5))",
            "gold_call": "_status(lambda: _oracle_evaluate_gc_ensemble(small.copy(), small_rows.copy(), 14, 50.5))",
        },
        {
            "setup": status,
            "call": "_status(lambda: evaluate_gc_ensemble(small.copy(), small_rows[1:].copy(), 14, 0.3))",
            "gold_call": "_status(lambda: _oracle_evaluate_gc_ensemble(small.copy(), small_rows[1:].copy(), 14, 0.3))",
        },
    ]
