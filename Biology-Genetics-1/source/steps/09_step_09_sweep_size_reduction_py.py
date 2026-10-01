"""
Factor by which a sweeping favourable mutation reduces the effective size of a linked neutral position in each past generation.

A neutral position close to a sweeping favourable mutation drifts faster than an unlinked

one: the neutral copies that happen to sit on the chromosomes carrying the favourable

allele are carried up with it, so the long-term contributions of neutral copies become

very unequal. Recombination between the two sites dissolves the association, and drift

among the carriers randomises which neutral copies remain linked. The variance of the

expected long-term contributions, projected from a past generation to the present, sets

the effective size that the neutral position experienced in that generation. This step

computes, for each past generation of a sweep, the factor by which the sweep reduces that

effective size.

Returns
-------
np.ndarray, shape (A,): factor[g - 1] = 1 / (1 + V_{A-g}) for g = 1 to A, the factor for the generation g generations before the present; each entry is positive
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sweep_size_reduction(frequencies: "np.ndarray", census: float, recombination: float) -> "np.ndarray":
    '''Factor by which a sweeping favourable mutation reduces the effective size of a linked neutral position in each past generation.

    frequencies[k] is the expected frequency q_k of a favourable mutation k
    generations after it arose, for k = 0 to A, where generation A is the
    present; census is the number N of diploid individuals, constant over the
    sweep, and recombination is the recombination fraction r between the
    selected site and the neutral position. For each generation k from 0 to
    A - 1, consider the neutral copies that are linked to the favourable allele
    in generation k. Let y_t be their expected frequency, and w_t the expected
    square of their frequency, among the chromosomes that carry the favourable
    allele in generation t >= k, with y_k = w_k = 1. For t = k to A - 1,
    y_{t+1} = y_t * (1 - (1 - q_t) * r) and
    w_{t+1} = w_t * (1 - 2 * (1 - q_t) * r) + (y_t - w_t) / (2 * N * q_t),
    the last term being the random sampling of copies among the carriers. The
    variance of the expected long-term contributions of the neutral copies of
    generation k to the present is
    V_k = w_A * q_A^2 / q_k + (1 - 2 * y_A * q_A + w_A * q_A^2) / (1 - q_k) - 1,
    and the sweep multiplies the effective size of the neutral position in
    generation k by 1 / (1 + V_k). Return these factors by the number of
    generations before the present, g = A - k, for g = 1 to A.

    Parameters
    ----------
    frequencies : np.ndarray
        Shape (A + 1,) with A >= 1: the expected frequencies q_0 to q_A, each
        in (0, 1).
    census : float
        Number of breeding diploid individuals, >= 1.
    recombination : float
        Recombination fraction between the selected site and the neutral
        position, in [0, 0.5].

    Returns
    -------
    factor : np.ndarray
        Shape (A,): factor[g - 1] = 1 / (1 + V_{A-g}) for g = 1 to A, the
        factor for the generation g generations before the present; each entry
        is positive.

    Raises
    ------
    ValueError
        If frequencies is not a one-dimensional array of at least two finite
        numbers in (0, 1), if census is not a finite number >= 1, or if
        recombination is not a finite number in [0, 0.5].
    '''
    return factor  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sweep_size_reduction(frequencies: "np.ndarray", census: float, recombination: float) -> "np.ndarray":
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating)) and bool(np.isfinite(float(v)))

    q = np.asarray(frequencies, dtype=float) if not isinstance(frequencies, (str, bytes)) else None
    if q is None or q.ndim != 1 or q.size < 2 or not np.all(np.isfinite(q)) or np.any(q <= 0.0) or np.any(q >= 1.0):
        raise ValueError("frequencies must be a one-dimensional array of at least two finite numbers in (0, 1)")
    if not _num(census) or float(census) < 1.0:
        raise ValueError("census must be a finite number >= 1")
    if not _num(recombination) or not 0.0 <= float(recombination) <= 0.5:
        raise ValueError("recombination must be a finite number in [0, 0.5]")
    r, two_n = float(recombination), 2.0 * float(census)
    a_now = q.size - 1
    q_now = q[a_now]
    factor = np.empty(a_now)
    for k in range(a_now):
        y, w = 1.0, 1.0                          # linked copies of generation k, among the carriers
        for t in range(k, a_now):
            y_next = y * (1.0 - (1.0 - q[t]) * r)
            w = w * (1.0 - 2.0 * (1.0 - q[t]) * r) + (y - w) / (two_n * q[t])
            y = y_next
        variance = w * q_now * q_now / q[k] + (1.0 - 2.0 * y * q_now + w * q_now * q_now) / (1.0 - q[k]) - 1.0
        factor[a_now - k - 1] = 1.0 / (1.0 + variance)     # g = A - k generations before the present
    return factor

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: a rising trajectory at the recent census, the shipped recombination fraction ---
        {
            "setup": "import numpy as np\nq = np.array([1/1200, 2/1200, 3/1200, 4/1200, 0.0043, 0.0056, 0.0073, 0.0095, 0.0123, 0.016, 0.0207, 0.0267, 0.0343, 0.044, 0.056, 0.071, 0.089, 0.11, 0.135, 0.165, 0.2, 0.24])\n",
            "call": "sweep_size_reduction(q, 600, 0.01)",
            "gold_call": "_oracle_sweep_size_reduction(q, 600, 0.01)",
        },
        # --- boundary: complete linkage, where only the sampling among carriers erodes the association ---
        {
            "setup": "import numpy as np\nq = np.array([0.001, 0.003, 0.01, 0.03, 0.08, 0.2, 0.4, 0.6])\n",
            "call": "sweep_size_reduction(q, 500, 0.0)",
            "gold_call": "_oracle_sweep_size_reduction(q, 500, 0.0)",
        },
        # --- edge: one generation of sweep with loose linkage in a small population ---
        {
            "setup": "import numpy as np\nq = np.array([0.02, 0.05])\n",
            "call": "sweep_size_reduction(q, 25, 0.3)",
            "gold_call": "_oracle_sweep_size_reduction(q, 25, 0.3)",
        },
        # --- edge: a sweep carried almost to fixation, with strong recombination ---
        {
            "setup": "import numpy as np\nq = np.array([0.005, 0.01, 0.03, 0.09, 0.25, 0.55, 0.82, 0.95, 0.99])\n",
            "call": "sweep_size_reduction(q, 100, 0.2)",
            "gold_call": "_oracle_sweep_size_reduction(q, 100, 0.2)",
        },
    ]
