"""
Compute every stationary joint binomial moment of the transcript and protein copy numbers up to a given total order for a gene with a multi-state promoter.

For a reaction network whose propensities are at most linear in the molecule counts, the steady-state joint binomial moments of transcript and protein numbers are close order by order, which gives exact access to their joint statistics without truncating the copy-number space.

Returns
-------
np.ndarray: (L + 1, L + 1) table of stationary E[C(M1, p) C(M2, q)], zero where p + q > L.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_binomial_moment_table(
    switch_rates: np.ndarray,
    transcription_rates: np.ndarray,
    mrna_decay: float,
    translation: float,
    protein_decay: float,
    max_layer: int,
) -> np.ndarray:
    """Return the stationary joint binomial moments of transcript and protein numbers.

    ``M1`` and ``M2`` are the transcript and protein copy numbers of the full
    model of ``compute_protein_mean_and_fano``, with the promoter state summed
    over. With ``L = max_layer``, entry ``[p, q]`` of the returned table is
    the stationary expectation of ``C(M1, p) * C(M2, q)``, where ``C`` is the
    binomial coefficient, for every ``p + q <= L``; every entry with
    ``p + q > L`` is exactly zero. Entries are exact up to floating-point
    rounding.

    Parameters
    ----------
    switch_rates : np.ndarray
        ``(N, N)`` silent switching rates with a zero diagonal.
    transcription_rates : np.ndarray
        ``(N, N)`` rates of transcription events from state ``i`` to ``j``.
    mrna_decay : float
        Positive transcript degradation rate.
    translation : float
        Positive translation rate per transcript.
    protein_decay : float
        Positive protein degradation rate.
    max_layer : int
        Largest total order ``p + q``; at least 0.

    Returns
    -------
    np.ndarray
        Float array of shape ``(L + 1, L + 1)``.

    Raises
    ------
    ValueError
        If the rate arrays violate the contract of
        ``compute_promoter_occupancy``, if a scalar rate is not a positive
        finite number, or if ``max_layer`` is not an integer of at least 0
        (booleans are rejected).
    """
    return table

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_binomial_moment_table(
    switch_rates: np.ndarray,
    transcription_rates: np.ndarray,
    mrna_decay: float,
    translation: float,
    protein_decay: float,
    max_layer: int,
) -> np.ndarray:
    """Reference implementation (layer-by-layer matrix hierarchy, coarse-grained)."""
    import numpy as np

    for value in (mrna_decay, translation, protein_decay):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("scalar rates must be real numbers")
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError("scalar rates must be positive and finite")
    if isinstance(max_layer, bool) or not isinstance(max_layer, (int, np.integer)) or max_layer < 0:
        raise ValueError("max_layer must be an integer of at least 0")
    u, v, delta, top = float(mrna_decay), float(translation), float(protein_decay), int(max_layer)
    occupancy = _oracle_compute_promoter_occupancy(switch_rates, transcription_rates)
    firing = np.asarray(transcription_rates, dtype=float)
    generator = np.asarray(switch_rates, dtype=float) + firing
    np.fill_diagonal(generator, 0.0)
    np.fill_diagonal(generator, -generator.sum(axis=1))
    n = generator.shape[0]
    identity = np.eye(n)
    ones = np.ones(n)
    table = np.zeros((top + 1, top + 1))
    table[0, 0] = 1.0
    previous = [identity]  # matrix-form moments of the layer below, indexed by p
    for layer in range(1, top + 1):
        current = [None] * (layer + 1)
        # Within a layer the moments are solved from the largest p downwards.
        for p in range(layer, -1, -1):
            q = layer - p
            rhs = np.zeros((n, n))
            if p >= 1:
                rhs -= previous[p - 1] @ firing
            if q >= 1:
                rhs -= v * p * previous[p] + v * (p + 1) * current[p + 1]
            system = generator - (u * p + delta * q) * identity
            current[p] = np.linalg.solve(system.T, rhs.T).T
            table[p, q] = occupancy @ current[p] @ ones
        previous = current
    return table

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    digest = (
        "import numpy as np\n"
        "def _digest(t):\n"
        "    a = np.asarray(t, dtype=float)\n"
        "    if a.ndim != 2 or a.shape[0] != a.shape[1]:\n"
        "        return -1.0\n"
        "    top = a.shape[0] - 1\n"
        "    p, q = np.indices(a.shape)\n"
        "    inside = (p + q) <= top\n"
        "    if np.any(a[~inside] != 0.0) or np.any(a[inside] <= 0.0):\n"
        "        return -2.0\n"
        "    logs = np.log(a[inside])\n"
        "    w1 = (1.0 + p[inside]) / (1.0 + top)\n"
        "    w2 = np.sqrt(1.0 + q[inside])\n"
        "    return float(a.shape[0] + 100.0 + np.sum(w1 * logs) / 10.0 + np.sum(w2 * logs) / 30.0)\n"
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
            "setup": digest,
            "call": "_digest(compute_binomial_moment_table(np.array([[0.0, 0.3, 0.0], [0.7, 0.0, 1.2], [0.2, 0.9, 0.0]]), np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.8, 0.0, 2.6]]), 1.1, 6.0, 0.2, 12))",
            "gold_call": "_digest(_oracle_compute_binomial_moment_table(np.array([[0.0, 0.3, 0.0], [0.7, 0.0, 1.2], [0.2, 0.9, 0.0]]), np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.8, 0.0, 2.6]]), 1.1, 6.0, 0.2, 12))",
        },
        {
            "setup": digest,
            "call": "_digest(compute_binomial_moment_table(np.array([[0.0, 0.05, 0.0, 0.0], [0.4, 0.0, 0.6, 0.0], [0.0, 0.3, 0.0, 0.8], [0.0, 0.0, 0.25, 0.0]]), np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0], [2.0, 0.0, 1.5, 4.0]]), 2.3, 15.0, 0.12, 24))",
            "gold_call": "_digest(_oracle_compute_binomial_moment_table(np.array([[0.0, 0.05, 0.0, 0.0], [0.4, 0.0, 0.6, 0.0], [0.0, 0.3, 0.0, 0.8], [0.0, 0.0, 0.25, 0.0]]), np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0], [2.0, 0.0, 1.5, 4.0]]), 2.3, 15.0, 0.12, 24))",
        },
        {
            "setup": digest,
            "call": "_digest(compute_binomial_moment_table(np.array([[0.0]]), np.array([[2.0]]), 0.5, 3.0, 0.1, 2))",
            "gold_call": "_digest(_oracle_compute_binomial_moment_table(np.array([[0.0]]), np.array([[2.0]]), 0.5, 3.0, 0.1, 2))",
        },
        {
            "setup": digest,
            "call": "_digest(compute_binomial_moment_table(np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.2, 0.0], [1.5, 0.5]]), 0.7, 4.0, 0.7, 0))",
            "gold_call": "_digest(_oracle_compute_binomial_moment_table(np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.2, 0.0], [1.5, 0.5]]), 0.7, 4.0, 0.7, 0))",
        },
        {
            "setup": digest,
            "call": "_digest(compute_binomial_moment_table(np.zeros((3, 3)), np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.5], [0.7, 0.0, 3.0]]), 0.05, 0.8, 2.0, 16))",
            "gold_call": "_digest(_oracle_compute_binomial_moment_table(np.zeros((3, 3)), np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.5], [0.7, 0.0, 3.0]]), 0.05, 0.8, 2.0, 16))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(compute_binomial_moment_table(np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.2, 0.0], [1.5, 0.5]]), 0.7, 4.0, 0.7, 5)[2, 3])",
            "gold_call": "float(_oracle_compute_binomial_moment_table(np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.2, 0.0], [1.5, 0.5]]), 0.7, 4.0, 0.7, 5)[2, 3])",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_binomial_moment_table(np.array([[0.0]]), np.array([[2.0]]), 0.5, 3.0, 0.1, -1))",
            "gold_call": "_status(lambda: _oracle_compute_binomial_moment_table(np.array([[0.0]]), np.array([[2.0]]), 0.5, 3.0, 0.1, -1))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_binomial_moment_table(np.array([[0.0]]), np.array([[2.0]]), 0.0, 3.0, 0.1, 4))",
            "gold_call": "_status(lambda: _oracle_compute_binomial_moment_table(np.array([[0.0]]), np.array([[2.0]]), 0.0, 3.0, 0.1, 4))",
        },
    ]
