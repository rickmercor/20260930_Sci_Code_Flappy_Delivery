"""
Compute the stationary protein Fano factor of the reduced gene expression model in which every transcription event releases an instantaneous geometric burst of proteins.

Because transcripts are usually far shorter-lived than proteins, each transcript's output is often collapsed into an instantaneous burst whose size follows from the race between translation and transcript decay.

Returns
-------
float: stationary protein Fano factor of the instantaneous geometric-burst model.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_burst_limit_fano(
    switch_rates: np.ndarray,
    transcription_rates: np.ndarray,
    burst_size: float,
    protein_decay: float,
) -> float:
    """Return the stationary protein Fano factor of the instantaneous-burst model.

    The promoter follows ``switch_rates`` and ``transcription_rates`` exactly
    as in ``compute_promoter_occupancy``, including the promoter transition of
    every transcription event. Transcripts are not tracked: each transcription
    event instantly adds ``R`` proteins, independently of all other events,
    with ``P(R = r) = (1 - theta) * theta**r`` for ``r = 0, 1, 2, ...``, where
    ``theta / (1 - theta) = burst_size``. Each protein is degraded at rate
    ``protein_decay``. The Fano factor is the stationary variance of the
    protein copy number divided by its mean.

    Parameters
    ----------
    switch_rates : np.ndarray
        ``(N, N)`` silent switching rates with a zero diagonal.
    transcription_rates : np.ndarray
        ``(N, N)`` rates of transcription events from state ``i`` to ``j``.
    burst_size : float
        Positive mean number of proteins released per transcription event.
    protein_decay : float
        Positive protein degradation rate.

    Returns
    -------
    float
        Stationary protein Fano factor of the instantaneous-burst model.

    Raises
    ------
    ValueError
        If the rate arrays violate the contract of
        ``compute_promoter_occupancy``, if ``burst_size`` or
        ``protein_decay`` is not a positive finite number (booleans are
        rejected), or if the stationary transcription rate is zero.
    """
    return fano

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_burst_limit_fano(
    switch_rates: np.ndarray,
    transcription_rates: np.ndarray,
    burst_size: float,
    protein_decay: float,
) -> float:
    """Reference implementation (first two protein binomial moments of the burst model)."""
    import numpy as np

    for value in (burst_size, protein_decay):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("burst_size and protein_decay must be real numbers")
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError("burst_size and protein_decay must be positive and finite")
    beta, delta = float(burst_size), float(protein_decay)
    occupancy = _oracle_compute_promoter_occupancy(switch_rates, transcription_rates)
    silent = np.asarray(switch_rates, dtype=float)
    firing = np.asarray(transcription_rates, dtype=float)
    generator = silent + firing
    np.fill_diagonal(generator, 0.0)
    np.fill_diagonal(generator, -generator.sum(axis=1))
    n = generator.shape[0]
    output = firing @ np.ones(n)
    s1 = float(occupancy @ output)
    if s1 <= 0.0:
        raise ValueError("the promoter never transcribes at stationarity")
    # Protein memory of the promoter state enters through the resolvent at the protein decay rate.
    t_delta = float(occupancy @ firing @ np.linalg.solve(delta * np.eye(n) - generator, output))
    b1 = beta * s1 / delta
    b2 = beta**2 / (2.0 * delta) * (s1 + t_delta)
    return float(1.0 + 2.0 * b2 / b1 - b1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    wrap = (
        "import numpy as np\n"
        "def _wrap(x):\n"
        "    if not isinstance(x, (float, np.floating)):\n"
        "        return -1.0\n"
        "    return float(1.0 + 1.0e3 * x)\n"
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
            "setup": wrap,
            "call": "_wrap(compute_burst_limit_fano(np.array([[0.0, 0.3, 0.0], [0.7, 0.0, 1.2], [0.2, 0.9, 0.0]]), np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.8, 0.0, 2.6]]), 5.45, 0.2))",
            "gold_call": "_wrap(_oracle_compute_burst_limit_fano(np.array([[0.0, 0.3, 0.0], [0.7, 0.0, 1.2], [0.2, 0.9, 0.0]]), np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.8, 0.0, 2.6]]), 5.45, 0.2))",
        },
        {
            "setup": wrap,
            "call": "_wrap(compute_burst_limit_fano(np.array([[0.0, 0.05, 0.0, 0.0], [0.4, 0.0, 0.6, 0.0], [0.0, 0.3, 0.0, 0.8], [0.0, 0.0, 0.25, 0.0]]), np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0], [2.0, 0.0, 1.5, 4.0]]), 12.0, 0.12))",
            "gold_call": "_wrap(_oracle_compute_burst_limit_fano(np.array([[0.0, 0.05, 0.0, 0.0], [0.4, 0.0, 0.6, 0.0], [0.0, 0.3, 0.0, 0.8], [0.0, 0.0, 0.25, 0.0]]), np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0], [2.0, 0.0, 1.5, 4.0]]), 12.0, 0.12))",
        },
        {
            "setup": wrap,
            "call": "_wrap(compute_burst_limit_fano(np.array([[0.0]]), np.array([[2.0]]), 6.0, 0.1))",
            "gold_call": "_wrap(_oracle_compute_burst_limit_fano(np.array([[0.0]]), np.array([[2.0]]), 6.0, 0.1))",
        },
        {
            "setup": wrap,
            "call": "_wrap(compute_burst_limit_fano(np.zeros((3, 3)), np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.5], [0.7, 0.0, 3.0]]), 0.02, 2.0))",
            "gold_call": "_wrap(_oracle_compute_burst_limit_fano(np.zeros((3, 3)), np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.5], [0.7, 0.0, 3.0]]), 0.02, 2.0))",
        },
        {
            "setup": wrap,
            "call": "_wrap(compute_burst_limit_fano(np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.2, 0.0], [1.5, 0.5]]), 3.0, 0.03))",
            "gold_call": "_wrap(_oracle_compute_burst_limit_fano(np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.2, 0.0], [1.5, 0.5]]), 3.0, 0.03))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_burst_limit_fano(np.array([[0.0]]), np.array([[2.0]]), 0.0, 0.1))",
            "gold_call": "_status(lambda: _oracle_compute_burst_limit_fano(np.array([[0.0]]), np.array([[2.0]]), 0.0, 0.1))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_burst_limit_fano(np.array([[0.0, 1.0], [1.0, 0.0]]), np.zeros((2, 2)), 4.0, 0.5))",
            "gold_call": "_status(lambda: _oracle_compute_burst_limit_fano(np.array([[0.0, 1.0], [1.0, 0.0]]), np.zeros((2, 2)), 4.0, 0.5))",
        },
    ]
