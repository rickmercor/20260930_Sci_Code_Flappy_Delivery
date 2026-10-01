"""
Combine a calibrated protein mean of a stable reporter with an instantaneous-burst summary of a destabilized reporter of the same gene to predict the mean stable-reporter protein count in cells that hold no transcript. Orchestrator: It computes the promoter occupancy (compute_promoter_occupancy), recovers the Fano factor behind the burst summary (compute_burst_limit_fano), infers the transcript decay rate (infer_mrna_decay_rate), checks the calibrated model against that Fano factor (compute_protein_mean_and_fano), builds the joint binomial moment table (compute_binomial_moment_table) and reads off the transcript-free protein mean (compute_transcript_conditioned_protein_statistics), consuming each output rather than reimplementing any step.

Reporters that share a promoter and transcript but differ in protein stability sample the same transcriptional noise over different averaging times, so a noise summary of one constrains the transcript lifetime that shapes the joint transcript and protein statistics of the other.

Returns
-------
float: stationary mean stable-reporter protein count among cells with no transcript.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def predict_transcript_free_protein_mean(
    switch_rates: np.ndarray,
    transcription_rates: np.ndarray,
    stable_half_life: float,
    stable_mean: float,
    destabilized_half_life: float,
    reported_burst_size: float,
    max_layer: int,
) -> float:
    """Return the stable reporter's mean protein count among cells with no transcript.

    Two reporters share the promoter rates, one transcript decay rate and one
    translation rate of the full model of ``compute_protein_mean_and_fano``
    and differ only in protein half-life. The stable reporter (half-life
    ``stable_half_life``) has stationary protein mean ``stable_mean``. For the
    destabilized reporter (half-life ``destabilized_half_life``),
    ``reported_burst_size`` is the ``burst_size`` at which the instantaneous
    burst model of ``compute_burst_limit_fano`` reproduces its stationary
    protein Fano factor. Infer the two transcript rates from these data and
    return, for the stable reporter, the stationary mean protein copy number
    given that the transcript count is zero, evaluated from the joint binomial
    moment table with total order up to ``max_layer`` as in
    ``compute_transcript_conditioned_protein_statistics``.

    Parameters
    ----------
    switch_rates : np.ndarray
        ``(N, N)`` silent switching rates with a zero diagonal.
    transcription_rates : np.ndarray
        ``(N, N)`` rates of transcription events from state ``i`` to ``j``.
    stable_half_life : float
        Positive protein half-life of the stable reporter.
    stable_mean : float
        Positive stationary protein mean of the stable reporter.
    destabilized_half_life : float
        Positive protein half-life of the destabilized reporter.
    reported_burst_size : float
        Positive burst size of the instantaneous burst summary.
    max_layer : int
        Largest total order of the binomial moment table; at least 2.

    Returns
    -------
    float
        Mean stable-reporter protein count in cells with no transcript.

    Raises
    ------
    ValueError
        If the rate arrays violate the contract of
        ``compute_promoter_occupancy``, if a half-life, ``stable_mean`` or
        ``reported_burst_size`` is not a positive finite number, if
        ``max_layer`` is not an integer of at least 2 (booleans are rejected),
        if the stationary transcription rate is zero, or if no transcript
        decay rate is consistent with the data.
    """
    return transcript_free_mean

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_predict_transcript_free_protein_mean(
    switch_rates: np.ndarray,
    transcription_rates: np.ndarray,
    stable_half_life: float,
    stable_mean: float,
    destabilized_half_life: float,
    reported_burst_size: float,
    max_layer: int,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    for value in (stable_half_life, stable_mean, destabilized_half_life, reported_burst_size):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("half-lives, stable_mean and reported_burst_size must be real numbers")
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError("half-lives, stable_mean and reported_burst_size must be positive and finite")
    if isinstance(max_layer, bool) or not isinstance(max_layer, (int, np.integer)) or max_layer < 2:
        raise ValueError("max_layer must be an integer of at least 2")
    stable_decay = float(np.log(2.0) / stable_half_life)
    destabilized_decay = float(np.log(2.0) / destabilized_half_life)
    occupancy = _oracle_compute_promoter_occupancy(switch_rates, transcription_rates)
    firing = np.asarray(transcription_rates, dtype=float)
    transcript_output = float(occupancy @ firing @ np.ones(firing.shape[0]))
    if transcript_output <= 0.0:
        raise ValueError("the promoter never transcribes at stationarity")
    # The protein mean fixes the proteins made per transcript, identically in both descriptions.
    burst_size = float(stable_mean) * stable_decay / transcript_output
    matched_fano = _oracle_compute_burst_limit_fano(
        switch_rates, transcription_rates, float(reported_burst_size), destabilized_decay)
    decay = _oracle_infer_mrna_decay_rate(
        switch_rates, transcription_rates, destabilized_decay, burst_size, matched_fano)
    translation = burst_size * decay
    check = _oracle_compute_protein_mean_and_fano(
        switch_rates, transcription_rates, decay, translation, destabilized_decay)
    if not np.isclose(check[1], matched_fano, rtol=1.0e-9, atol=0.0):
        raise ValueError("the calibrated rates do not reproduce the matched Fano factor")
    table = _oracle_compute_binomial_moment_table(
        switch_rates, transcription_rates, decay, translation, stable_decay, max_layer)
    statistics = _oracle_compute_transcript_conditioned_protein_statistics(table, 0)
    return float(statistics[1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
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
            "setup": "import numpy as np\n",
            "call": "predict_transcript_free_protein_mean(np.array([[0.0, 0.3, 0.0], [0.7, 0.0, 1.2], [0.2, 0.9, 0.0]]), np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.8, 0.0, 2.6]]), 5.0, 40.0, 1.2, 3.5, 60)",
            "gold_call": "_oracle_predict_transcript_free_protein_mean(np.array([[0.0, 0.3, 0.0], [0.7, 0.0, 1.2], [0.2, 0.9, 0.0]]), np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.8, 0.0, 2.6]]), 5.0, 40.0, 1.2, 3.5, 60)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "predict_transcript_free_protein_mean(np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.2, 0.0], [1.5, 0.5]]), 8.0, 120.0, 2.0, 10.0, 40)",
            "gold_call": "_oracle_predict_transcript_free_protein_mean(np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.2, 0.0], [1.5, 0.5]]), 8.0, 120.0, 2.0, 10.0, 40)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "predict_transcript_free_protein_mean(np.array([[0.0]]), np.array([[2.0]]), 6.0, 90.0, 1.0, 4.5, 30)",
            "gold_call": "_oracle_predict_transcript_free_protein_mean(np.array([[0.0]]), np.array([[2.0]]), 6.0, 90.0, 1.0, 4.5, 30)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "predict_transcript_free_protein_mean(np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.2, 0.0], [1.5, 0.5]]), 8.0, 120.0, 2.0, 10.0, 6)",
            "gold_call": "_oracle_predict_transcript_free_protein_mean(np.array([[0.0, 0.4], [0.0, 0.0]]), np.array([[0.2, 0.0], [1.5, 0.5]]), 8.0, 120.0, 2.0, 10.0, 6)",
        },
        {
            "setup": status,
            "call": "_status(lambda: predict_transcript_free_protein_mean(np.array([[0.0]]), np.array([[2.0]]), 6.0, 90.0, 1.0, 40.0, 30))",
            "gold_call": "_status(lambda: _oracle_predict_transcript_free_protein_mean(np.array([[0.0]]), np.array([[2.0]]), 6.0, 90.0, 1.0, 40.0, 30))",
        },
        {
            "setup": status,
            "call": "_status(lambda: predict_transcript_free_protein_mean(np.array([[0.0]]), np.array([[2.0]]), -6.0, 90.0, 1.0, 1.5, 30))",
            "gold_call": "_status(lambda: _oracle_predict_transcript_free_protein_mean(np.array([[0.0]]), np.array([[2.0]]), -6.0, 90.0, 1.0, 1.5, 30))",
        },
    ]
