# Biology-Biochemistry-11

## Background

Genetically identical cells differ in their transcript and protein contents because promoters switch among regulatory states, every synthesis and decay event happens at a random time, and in many genes the act of transcription itself changes the state of the promoter. Single-cell measurements often capture only one molecular species at a time, so predicting how the two species co-vary within a cell requires a mechanistic model of their joint stochastic dynamics.

## Problem

The promoter of a reporter gene in my cells moves among a repressed state R, a poised state P and an active state A, switching R→P at 0.15 h⁻¹, P→R at 0.40 h⁻¹, P→A at 0.90 h⁻¹ and A→P at 0.50 h⁻¹. The active promoter initiates transcription at 9.0 h⁻¹, and each initiation releases one transcript and, with probability 0.35, returns the promoter to R in that same event; the poised promoter releases single transcripts at 0.60 h⁻¹ without changing state. Every transcript is degraded and translated through independent memoryless events at two constant rates I do not know, and every protein is degraded independently. I have two versions of the reporter that differ only in protein stability: for the stable version, with a protein half-life of 6.0 h, my calibrated single-cell counts at steady state give a protein mean of 75.79, while for the destabilized version, with a half-life of 1.5 h, all I have is a colleague's summary. Using the promoter scheme above and the 1.5 h half-life, she described each transcription event, together with its promoter transition, as releasing its proteins instantly in a geometrically distributed number that may be zero; she chose the mean of that number so that this description reproduced the destabilized reporter's steady-state protein Fano factor exactly, and reported 19.66 proteins per event. Treat the supplied decimal observations 75.79 and 19.66 as exact nominal benchmark inputs. What mean protein copy number should I expect for the stable reporter among the cells that contain no transcript at all? Report it to at least six significant figures, and tell me what these data imply about the two unknown rates, about how common transcript-free cells are, and about my colleague's number, giving the relations your inference rests on.

## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_compute_promoter_occupancy

Goal
----
Compute the stationary occupancy of each promoter state of a gene whose promoter switches silently and can also change state in the same event that releases a transcript.

```python
def compute_promoter_occupancy(switch_rates: np.ndarray, transcription_rates: np.ndarray) -> np.ndarray:
    """Return the long-time probability of each promoter state.

    The promoter has ``N`` states labelled ``0..N-1``. For ``i != j``,
    ``switch_rates[i, j]`` is the rate of a transition from state ``i`` to
    state ``j`` that releases no transcript. For every ``i`` and ``j``,
    ``transcription_rates[i, j]`` is the rate at which a promoter in state
    ``i`` releases one transcript in an event that leaves it in state ``j``;
    the diagonal entry ``j = i`` is transcription without a change of state.
    All events are memoryless.

    Parameters
    ----------
    switch_rates : np.ndarray
        Array of shape ``(N, N)``, ``N >= 1``, with nonnegative finite
        entries and a zero diagonal.
    transcription_rates : np.ndarray
        Array of shape ``(N, N)`` with nonnegative finite entries.

    Returns
    -------
    np.ndarray
        Float array of shape ``(N,)`` holding the stationary probabilities of
        the promoter states; its entries sum to 1.

    Raises
    ------
    ValueError
        If either array is not square of the same shape ``(N, N)`` with
        ``N >= 1``, if an entry is negative or not finite, if
        ``switch_rates`` has a nonzero diagonal entry, or if the promoter
        chain does not have a unique stationary distribution.
    """
    return occupancy
```

### Step 2

02_compute_burst_limit_fano

Goal
----
Compute the stationary protein Fano factor of the reduced gene expression model in which every transcription event releases an instantaneous geometric burst of proteins.

```python
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
```

### Step 3

03_compute_protein_mean_and_fano

Goal
----
Compute the exact stationary mean and Fano factor of the protein copy number for a gene with a multi-state promoter, finite transcript lifetime and first-order translation.

```python
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
```

### Step 4

04_infer_mrna_decay_rate

Goal
----
Infer the transcript decay rate that reproduces a measured stationary protein Fano factor when the mean number of proteins made per transcript is already fixed.

```python
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
```

### Step 5

05_compute_binomial_moment_table

Goal
----
Compute every stationary joint binomial moment of the transcript and protein copy numbers up to a given total order for a gene with a multi-state promoter.

```python
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
```

### Step 6

06_compute_transcript_conditioned_protein_statistics

Goal
----
Recover, from a table of joint binomial moments of transcript and protein counts, the probability of a given transcript count and the mean and Fano factor of the protein count in the cells that hold exactly that many transcripts.

```python
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
```

### Step 7

07_predict_transcript_free_protein_mean

Goal
----
Combine a calibrated protein mean of a stable reporter with an instantaneous-burst summary of a destabilized reporter of the same gene to predict the mean stable-reporter protein count in cells that hold no transcript. Orchestrator: It computes the promoter occupancy (compute_promoter_occupancy), recovers the Fano factor behind the burst summary (compute_burst_limit_fano), infers the transcript decay rate (infer_mrna_decay_rate), checks the calibrated model against that Fano factor (compute_protein_mean_and_fano), builds the joint binomial moment table (compute_binomial_moment_table) and reads off the transcript-free protein mean (compute_transcript_conditioned_protein_statistics), consuming each output rather than reimplementing any step.

```python
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
```
