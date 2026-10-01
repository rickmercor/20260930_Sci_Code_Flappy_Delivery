"""
Recover, from a table of joint binomial moments of transcript and protein counts, the probability of a given transcript count and the mean and Fano factor of the protein count in the cells that hold exactly that many transcripts.

Joint binomial moments are the Taylor coefficients of the joint probability generating function about unit arguments, so statistics of a fixed-count slice of the joint distribution can be read back from them.

Returns
-------
np.ndarray: float array [P(M1 = m), E[M2 | M1 = m], Var(M2 | M1 = m) / E[M2 | M1 = m]].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_transcript_conditioned_protein_statistics(moment_table: np.ndarray, mrna_count: int) -> np.ndarray:
    """Return transcript-count probability and conditional protein mean and Fano factor.

    ``moment_table`` has shape ``(L + 1, L + 1)``; for ``p + q <= L`` its
    entry ``[p, q]`` is ``E[C(M1, p) * C(M2, q)]`` for a joint distribution of
    nonnegative integer counts ``M1`` (transcripts) and ``M2`` (proteins),
    where ``C`` is the binomial coefficient. Entries with ``p + q > L`` are
    ignored. With ``m = mrna_count``, return ``P(M1 = m)``, the mean of
    ``M2`` given ``M1 = m``, and the variance of ``M2`` given ``M1 = m``
    divided by that mean. Every quantity is evaluated from the table alone,
    with each sum over the transcript order ``p`` cut at ``p + q <= L`` for
    the protein order ``q`` it involves, so a table with small ``L`` yields
    the correspondingly truncated values.

    Parameters
    ----------
    moment_table : np.ndarray
        Square two-dimensional array of finite floats.
    mrna_count : int
        Transcript count ``m``; ``0 <= m <= L - 2``.

    Returns
    -------
    np.ndarray
        Float array ``[probability, conditional_mean, conditional_fano]``.

    Raises
    ------
    ValueError
        If ``moment_table`` is not a square two-dimensional array of finite
        numbers, if ``mrna_count`` is not an integer with
        ``0 <= mrna_count <= L - 2`` (booleans are rejected), or if the
        evaluated ``P(M1 = m)`` or ``E[M2 ; M1 = m]`` is not positive.
    """
    return statistics

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_transcript_conditioned_protein_statistics(moment_table: np.ndarray, mrna_count: int) -> np.ndarray:
    """Reference implementation (alternating binomial-moment inversion in the transcript index)."""
    import math
    import numpy as np

    table = np.asarray(moment_table, dtype=float)
    if table.ndim != 2 or table.shape[0] != table.shape[1] or not np.all(np.isfinite(table)):
        raise ValueError("moment_table must be a square two-dimensional array of finite numbers")
    top = table.shape[0] - 1
    if isinstance(mrna_count, bool) or not isinstance(mrna_count, (int, np.integer)):
        raise ValueError("mrna_count must be an integer")
    m = int(mrna_count)
    if m < 0 or m > top - 2:
        raise ValueError("mrna_count must satisfy 0 <= mrna_count <= L - 2")
    slice_moments = []
    for q in range(3):
        terms = []
        for p in range(m, top - q + 1):
            value = table[p, q]
            if value == 0.0:
                continue
            # Binomial weights are formed in the log domain to keep large orders finite.
            log_weight = math.lgamma(p + 1) - math.lgamma(m + 1) - math.lgamma(p - m + 1)
            sign = -1.0 if (p - m) % 2 else 1.0
            terms.append(sign * math.copysign(math.exp(log_weight + math.log(abs(value))), value))
        slice_moments.append(math.fsum(terms))
    probability, first, second = slice_moments
    if not (probability > 0.0 and first > 0.0):
        raise ValueError("the transcript slice has no probability or no protein")
    mean = first / probability
    fano = (2.0 * second / probability + mean - mean * mean) / mean
    return np.array([probability, mean, fano], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    tables = (
        "import numpy as np\n"
        "from math import comb\n"
        "def _pmf_table(top, seed):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    pmf = rng.random((5, 7)) ** 2\n"
        "    pmf /= pmf.sum()\n"
        "    table = np.zeros((top + 1, top + 1))\n"
        "    for p in range(top + 1):\n"
        "        for q in range(top + 1 - p):\n"
        "            table[p, q] = sum(comb(i, p) * comb(j, q) * pmf[i, j] for i in range(5) for j in range(7))\n"
        "    return table\n"
        "def _poisson_table(top, a, b):\n"
        "    from math import factorial\n"
        "    table = np.zeros((top + 1, top + 1))\n"
        "    for p in range(top + 1):\n"
        "        for q in range(top + 1 - p):\n"
        "            table[p, q] = a ** p / factorial(p) * b ** q / factorial(q)\n"
        "    return table\n"
        "def _digest(x):\n"
        "    s = np.asarray(x, dtype=float)\n"
        "    if s.shape != (3,):\n"
        "        return -1.0\n"
        "    return float(3.0 + 1.0e3 * s[0] + 7.0e1 * s[1] + 3.0e2 * s[2])\n"
    )
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        {
            "setup": tables,
            "call": "_digest(compute_transcript_conditioned_protein_statistics(_pmf_table(12, 3), 0))",
            "gold_call": "_digest(_oracle_compute_transcript_conditioned_protein_statistics(_pmf_table(12, 3), 0))",
        },
        {
            "setup": tables,
            "call": "_digest(compute_transcript_conditioned_protein_statistics(_pmf_table(12, 3), 2))",
            "gold_call": "_digest(_oracle_compute_transcript_conditioned_protein_statistics(_pmf_table(12, 3), 2))",
        },
        {
            "setup": tables,
            "call": "_digest(compute_transcript_conditioned_protein_statistics(_poisson_table(40, 1.3, 4.0), 0))",
            "gold_call": "_digest(_oracle_compute_transcript_conditioned_protein_statistics(_poisson_table(40, 1.3, 4.0), 0))",
        },
        {
            "setup": tables,
            "call": "_digest(compute_transcript_conditioned_protein_statistics(_pmf_table(5, 11), 1))",
            "gold_call": "_digest(_oracle_compute_transcript_conditioned_protein_statistics(_pmf_table(5, 11), 1))",
        },
        {
            "setup": tables,
            "call": "_digest(compute_transcript_conditioned_protein_statistics(_poisson_table(9, 0.6, 1.5), 7))",
            "gold_call": "_digest(_oracle_compute_transcript_conditioned_protein_statistics(_poisson_table(9, 0.6, 1.5), 7))",
        },
        {
            "setup": tables + status,
            "call": "_status(lambda: compute_transcript_conditioned_protein_statistics(_pmf_table(6, 3), 5))",
            "gold_call": "_status(lambda: _oracle_compute_transcript_conditioned_protein_statistics(_pmf_table(6, 3), 5))",
        },
        {
            "setup": tables + status,
            "call": "_status(lambda: compute_transcript_conditioned_protein_statistics(_pmf_table(12, 3), 6))",
            "gold_call": "_status(lambda: _oracle_compute_transcript_conditioned_protein_statistics(_pmf_table(12, 3), 6))",
        },
    ]
