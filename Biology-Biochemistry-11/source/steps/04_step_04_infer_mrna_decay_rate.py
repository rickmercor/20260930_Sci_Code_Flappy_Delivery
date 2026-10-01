"""
Infer the transcript decay rate that reproduces a measured stationary protein Fano factor when the mean number of proteins made per transcript is already fixed.

A protein snapshot fixes the ratio of translation to transcript decay through its mean, while the finite lifetime of each transcript leaves a separate imprint on the protein noise that can be inverted for the decay rate itself.

Returns
-------
float: transcript decay rate u reproducing the protein Fano factor at fixed burst size.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def infer_mrna_decay_rate(
    switch_rates: np.ndarray,
    transcription_rates: np.ndarray,
    protein_decay: float,
    burst_size: float,
    protein_fano: float,
) -> float:
    """Return the transcript decay rate that reproduces a stationary protein Fano factor.

    Consider the full model of ``compute_protein_mean_and_fano`` with the
    given promoter rates and ``protein_decay``, a transcript decay rate
    ``u > 0`` and translation rate ``burst_size * u``, so that the mean number
    of proteins made per transcript stays ``burst_size``. Return the value of
    ``u`` at which its stationary protein Fano factor equals
    ``protein_fano``. At most one such ``u`` exists; locate it to a relative
    accuracy of at least ``1e-10``.

    Parameters
    ----------
    switch_rates : np.ndarray
        ``(N, N)`` silent switching rates with a zero diagonal.
    transcription_rates : np.ndarray
        ``(N, N)`` rates of transcription events from state ``i`` to ``j``.
    protein_decay : float
        Positive protein degradation rate.
    burst_size : float
        Positive ratio of translation rate to transcript decay rate.
    protein_fano : float
        Target stationary protein Fano factor.

    Returns
    -------
    float
        The transcript decay rate ``u``.

    Raises
    ------
    ValueError
        If the rate arrays violate the contract of
        ``compute_promoter_occupancy``, if ``protein_decay`` or
        ``burst_size`` is not a positive finite number or ``protein_fano`` is
        not a finite number (booleans are rejected), if the stationary
        transcription rate is zero, if ``protein_fano`` is not greater than
        1, or if no ``u > 0`` reproduces ``protein_fano``.
    """
    return decay_rate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_infer_mrna_decay_rate(
    switch_rates: np.ndarray,
    transcription_rates: np.ndarray,
    protein_decay: float,
    burst_size: float,
    protein_fano: float,
) -> float:
    """Reference implementation (bracketed root of the exact Fano factor in log u)."""
    import numpy as np
    from scipy.optimize import brentq

    if isinstance(protein_fano, bool) or not isinstance(protein_fano, (int, float, np.integer, np.floating)):
        raise ValueError("protein_fano must be a real number")
    if not np.isfinite(protein_fano):
        raise ValueError("protein_fano must be finite")
    target = float(protein_fano)
    # The burst model is the infinite-decay limit at fixed burst size and bounds the
    # attainable Fano factor from above; an arbitrarily stable transcript approaches 1.
    ceiling = _oracle_compute_burst_limit_fano(switch_rates, transcription_rates, burst_size, protein_decay)
    if not (1.0 < target < ceiling):
        raise ValueError("no positive transcript decay rate reproduces protein_fano")
    beta, delta = float(burst_size), float(protein_decay)

    def _fano_gap(log_u):
        u = float(np.exp(log_u))
        moments = _oracle_compute_protein_mean_and_fano(
            switch_rates, transcription_rates, u, beta * u, delta)
        return float(moments[1]) - target

    low = high = float(np.log(delta))
    for _ in range(200):
        if _fano_gap(low) < 0.0:
            break
        low -= np.log(4.0)
    else:
        raise ValueError("no positive transcript decay rate reproduces protein_fano")
    for _ in range(200):
        if _fano_gap(high) > 0.0:
            break
        high += np.log(4.0)
    else:
        raise ValueError("no positive transcript decay rate reproduces protein_fano")
    root = brentq(_fano_gap, low, high, xtol=1.0e-14, rtol=1.0e-15, maxiter=500)
    return float(np.exp(root))

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
            "call": "_wrap(infer_mrna_decay_rate(np.array([[0.0, 0.3, 0.0], [0.7, 0.0, 1.2], [0.2, 0.9, 0.0]]), np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.8, 0.0, 2.6]]), 0.2, 5.45, 8.75))",
            "gold_call": "_wrap(_oracle_infer_mrna_decay_rate(np.array([[0.0, 0.3, 0.0], [0.7, 0.0, 1.2], [0.2, 0.9, 0.0]]), np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.8, 0.0, 2.6]]), 0.2, 5.45, 8.75))",
        },
        {
            "setup": wrap,
            "call": "_wrap(infer_mrna_decay_rate(np.array([[0.0, 0.05, 0.0, 0.0], [0.4, 0.0, 0.6, 0.0], [0.0, 0.3, 0.0, 0.8], [0.0, 0.0, 0.25, 0.0]]), np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0], [2.0, 0.0, 1.5, 4.0]]), 0.12, 12.0, 30.0))",
            "gold_call": "_wrap(_oracle_infer_mrna_decay_rate(np.array([[0.0, 0.05, 0.0, 0.0], [0.4, 0.0, 0.6, 0.0], [0.0, 0.3, 0.0, 0.8], [0.0, 0.0, 0.25, 0.0]]), np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0], [2.0, 0.0, 1.5, 4.0]]), 0.12, 12.0, 30.0))",
        },
        {
            "setup": wrap,
            "call": "_wrap(infer_mrna_decay_rate(np.array([[0.0]]), np.array([[2.0]]), 0.1, 6.0, 5.5))",
            "gold_call": "_wrap(_oracle_infer_mrna_decay_rate(np.array([[0.0]]), np.array([[2.0]]), 0.1, 6.0, 5.5))",
        },
        {
            "setup": wrap,
            "call": "_wrap(infer_mrna_decay_rate(np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.2, 0.0], [1.5, 0.5]]), 0.03, 3.0, 1.02))",
            "gold_call": "_wrap(_oracle_infer_mrna_decay_rate(np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.2, 0.0], [1.5, 0.5]]), 0.03, 3.0, 1.02))",
        },
        {
            "setup": wrap,
            "call": "_wrap(infer_mrna_decay_rate(np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.2, 0.0], [1.5, 0.5]]), 0.03, 3.0, 3.9))",
            "gold_call": "_wrap(_oracle_infer_mrna_decay_rate(np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.2, 0.0], [1.5, 0.5]]), 0.03, 3.0, 3.9))",
        },
        {
            "setup": status,
            "call": "_status(lambda: infer_mrna_decay_rate(np.array([[0.0]]), np.array([[2.0]]), 0.1, 6.0, 7.5))",
            "gold_call": "_status(lambda: _oracle_infer_mrna_decay_rate(np.array([[0.0]]), np.array([[2.0]]), 0.1, 6.0, 7.5))",
        },
        {
            "setup": status,
            "call": "_status(lambda: infer_mrna_decay_rate(np.array([[0.0, 0.3, 0.0], [0.7, 0.0, 1.2], [0.2, 0.9, 0.0]]), np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.8, 0.0, 2.6]]), 0.2, 5.45, 0.9))",
            "gold_call": "_status(lambda: _oracle_infer_mrna_decay_rate(np.array([[0.0, 0.3, 0.0], [0.7, 0.0, 1.2], [0.2, 0.9, 0.0]]), np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.8, 0.0, 2.6]]), 0.2, 5.45, 0.9))",
        },
    ]
