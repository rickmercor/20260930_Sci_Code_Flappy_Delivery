"""
Compute the exact stationary mean and Fano factor of the protein copy number for a gene with a multi-state promoter, finite transcript lifetime and first-order translation.

Protein fluctuations inherit noise from promoter switching, from the random timing of transcription and from each transcript's random lifetime of translation, and all three contributions enter the steady-state second moment.

Returns
-------
np.ndarray: float array [mean, fano] of the stationary protein copy number.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_protein_mean_and_fano(
    switch_rates: np.ndarray,
    transcription_rates: np.ndarray,
    mrna_decay: float,
    translation: float,
    protein_decay: float,
) -> np.ndarray:
    """Return the stationary protein mean and Fano factor of the full gene expression model.

    The promoter follows ``switch_rates`` and ``transcription_rates`` exactly
    as in ``compute_promoter_occupancy``, and every transcription event adds
    one transcript. Each transcript is degraded at rate ``mrna_decay`` and,
    while it exists, produces one protein at a time at rate ``translation``.
    Each protein is degraded at rate ``protein_decay``. All events are
    memoryless and molecules act independently. The Fano factor is the
    stationary variance of the protein copy number divided by its mean.

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

    Returns
    -------
    np.ndarray
        Float array ``[mean, fano]`` of the stationary protein copy number.

    Raises
    ------
    ValueError
        If the rate arrays violate the contract of
        ``compute_promoter_occupancy``, if a scalar rate is not a positive
        finite number (booleans are rejected), or if the stationary
        transcription rate is zero.
    """
    return moments

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_protein_mean_and_fano(
    switch_rates: np.ndarray,
    transcription_rates: np.ndarray,
    mrna_decay: float,
    translation: float,
    protein_decay: float,
) -> np.ndarray:
    """Reference implementation (first two coarse-grained protein binomial moments)."""
    import numpy as np

    for value in (mrna_decay, translation, protein_decay):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("scalar rates must be real numbers")
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError("scalar rates must be positive and finite")
    u, v, delta = float(mrna_decay), float(translation), float(protein_decay)
    occupancy = _oracle_compute_promoter_occupancy(switch_rates, transcription_rates)
    silent = np.asarray(switch_rates, dtype=float)
    firing = np.asarray(transcription_rates, dtype=float)
    generator = silent + firing
    np.fill_diagonal(generator, 0.0)
    np.fill_diagonal(generator, -generator.sum(axis=1))
    n = generator.shape[0]
    identity = np.eye(n)
    output = firing @ np.ones(n)
    s1 = float(occupancy @ output)
    if s1 <= 0.0:
        raise ValueError("the promoter never transcribes at stationarity")
    # Resolvents of the promoter generator act on the per-state transcription output.
    after_u = np.linalg.solve(u * identity - generator, output)
    after_ud = np.linalg.solve(u * identity - generator, np.linalg.solve(delta * identity - generator, output))
    t_u = float(occupancy @ firing @ after_u)
    t_ud = float(occupancy @ firing @ after_ud)
    b01 = v * s1 / (u * delta)
    b02 = (v**2 / (2.0 * u * delta * (u + delta)) * s1
           + v**2 / (2.0 * delta * (u + delta)) * t_ud
           + v**2 / (2.0 * u * delta * (u + delta)) * t_u)
    fano = 1.0 + 2.0 * b02 / b01 - b01
    return np.array([b01, fano], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    digest = (
        "import numpy as np\n"
        "def _digest(x):\n"
        "    a = np.asarray(x, dtype=float)\n"
        "    if a.shape != (2,):\n"
        "        return -1.0\n"
        "    return float(2.0 + 1.0e3 * a[0] + 7.0e3 * a[1])\n"
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
            "call": "_digest(compute_protein_mean_and_fano(np.array([[0.0, 0.3, 0.0], [0.7, 0.0, 1.2], [0.2, 0.9, 0.0]]), np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.8, 0.0, 2.6]]), 1.1, 6.0, 0.2))",
            "gold_call": "_digest(_oracle_compute_protein_mean_and_fano(np.array([[0.0, 0.3, 0.0], [0.7, 0.0, 1.2], [0.2, 0.9, 0.0]]), np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.8, 0.0, 2.6]]), 1.1, 6.0, 0.2))",
        },
        {
            "setup": digest,
            "call": "_digest(compute_protein_mean_and_fano(np.array([[0.0, 0.05, 0.0, 0.0], [0.4, 0.0, 0.6, 0.0], [0.0, 0.3, 0.0, 0.8], [0.0, 0.0, 0.25, 0.0]]), np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0], [2.0, 0.0, 1.5, 4.0]]), 2.3, 15.0, 0.12))",
            "gold_call": "_digest(_oracle_compute_protein_mean_and_fano(np.array([[0.0, 0.05, 0.0, 0.0], [0.4, 0.0, 0.6, 0.0], [0.0, 0.3, 0.0, 0.8], [0.0, 0.0, 0.25, 0.0]]), np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0], [2.0, 0.0, 1.5, 4.0]]), 2.3, 15.0, 0.12))",
        },
        {
            "setup": digest,
            "call": "_digest(compute_protein_mean_and_fano(np.array([[0.0]]), np.array([[2.0]]), 0.5, 3.0, 0.1))",
            "gold_call": "_digest(_oracle_compute_protein_mean_and_fano(np.array([[0.0]]), np.array([[2.0]]), 0.5, 3.0, 0.1))",
        },
        {
            "setup": digest,
            "call": "_digest(compute_protein_mean_and_fano(np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.2, 0.0], [1.5, 0.5]]), 0.7, 4.0, 0.7))",
            "gold_call": "_digest(_oracle_compute_protein_mean_and_fano(np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.2, 0.0], [1.5, 0.5]]), 0.7, 4.0, 0.7))",
        },
        {
            "setup": digest,
            "call": "_digest(compute_protein_mean_and_fano(np.zeros((3, 3)), np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.5], [0.7, 0.0, 3.0]]), 0.05, 0.8, 2.0))",
            "gold_call": "_digest(_oracle_compute_protein_mean_and_fano(np.zeros((3, 3)), np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.5], [0.7, 0.0, 3.0]]), 0.05, 0.8, 2.0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_protein_mean_and_fano(np.array([[0.0, 1.0], [1.0, 0.0]]), np.zeros((2, 2)), 1.0, 2.0, 0.5))",
            "gold_call": "_status(lambda: _oracle_compute_protein_mean_and_fano(np.array([[0.0, 1.0], [1.0, 0.0]]), np.zeros((2, 2)), 1.0, 2.0, 0.5))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_protein_mean_and_fano(np.array([[0.0]]), np.array([[2.0]]), 0.5, 0.0, 0.1))",
            "gold_call": "_status(lambda: _oracle_compute_protein_mean_and_fano(np.array([[0.0]]), np.array([[2.0]]), 0.5, 0.0, 0.1))",
        },
    ]
