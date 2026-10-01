# Endpoint bias of a time-discretized delay approximation in an auto-inhibitory gene circuit

## Background

Stochastic models of intracellular reaction networks normally treat every reaction event as instantaneous and memoryless, which makes the copy-number vector a Markov process and admits well-established exact and approximate simulators. Transcription breaks that assumption: an initiated transcript takes a non-negligible and variable time to mature, and its rate of maturing can depend both on how long that individual transcript has been elongating and on how much mature and immature mRNA is competing for a finite pool of processing machinery. A delay of this kind is not reducible to a fixed lag or to a lag that depends only on the clock, so simulating the network exactly requires carrying per-event history that a Markovian simulator discards, while the faster approximate schemes that discretize time trade part of that history away. Whether the trade is acceptable is judged by how far an approximate ensemble average of an observable sits from the exact one.

## Problem

I am studying a gene circuit in which transcription is a delayed reaction. Initiation is repressed by protein through a Hill response; each initiated transcript matures only after a stochastic delay whose completion rate depends on that transcript's own elongation age and slows as mature and immature mRNA compete for a finite pool of processing machinery; mature mRNA is translated into protein; and mRNA and protein each degrade in one step. I want to know what I give up at a coarse time step by replacing an exact event-driven simulation of this circuit with the time-discretized alternative that advances every channel by a Poisson firing count over each step and carries all transcripts initiated within one step as a single delayed group.

Use the initial state `(M, P, M*) = (0, 0, 0)` and the delayed auto-inhibition parameter set `beta_m=10`, `beta_mstar=0.175`, `beta_p=1`, `gamma_m=0.08`, `gamma_p=0.05`, `alpha=2.5`, `M_a=10`, `v=0.5`, `h=1.5`, `P_a=5`. Estimate the mean protein copy number at time 60 from 80 paths of each simulator, with a step of 2.5 for the approximation, and report `100 * (approximate mean - exact mean) / exact mean`.

So that both estimates are reproducible, drive the exact path `r` with `numpy.random.default_rng(314159 + 2*r)` and approximate path `r` with `numpy.random.default_rng(314160 + 2*r)` for zero-based `r`. Index the four channels other than transcription completion as transcription initiation, translation, mRNA degradation and protein degradation, and take every draw in that order; hold concurrent transcription delays in initiation order; where one firing both refreshes its own channel clock and opens a new delay, refresh the channel clock first. Truncate any draw that would carry a population below zero, and let an event falling exactly at time 60 take effect before the endpoint is read.

In `<reasoning>`, the quantities I need you to state are: the two ensemble mean protein copy numbers; the Monte Carlo uncertainty carried by the reported percentage at this ensemble size; the published empirical weak and strong convergence orders of this approximation in the step size; the published comparison of the two simulators' mean protein peak heights and of their peak times at a step of 2.5; the value of the delay shape parameter at which this circuit loses its history dependence; and the per-transcript bookkeeping an exact simulation of this circuit has to carry that a Markovian simulator does not. Those are the quantities that fix the reported number or place it against the published benchmarks, so stating them is what the output requirements below call for; what those requirements exclude is restating the supplied parameters and any per-path or per-step tables.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_pack_gene_parameters

Goal
----
Validate and pack the delayed auto-inhibition parameters.

```python
def pack_gene_parameters(beta_m: float, beta_mstar: float, beta_p: float,
                         gamma_m: float, gamma_p: float, alpha: float,
                         m_a: float, resource_v: float, hill_h: float,
                         p_a: float) -> "np.ndarray":
    """Return parameters in the task's canonical order.

    Returns
    -------
    np.ndarray
        ``[beta_m,beta_mstar,beta_p,gamma_m,gamma_p,alpha,M_a,v,h,P_a]``.

    Raises
    ------
    ValueError
        If a parameter is non-finite, a rate/scale/exponent is non-positive,
        or ``resource_v`` is negative.
    """
    return parameters  # placeholder
```

### Step 2

02_gene_channel_propensities

Goal
----
Evaluate initiation, translation, and degradation rates.

```python
def gene_channel_propensities(state: "np.ndarray", parameters: "np.ndarray") -> "np.ndarray":
    """Return propensities in initiation, translation, mRNA-loss, protein-loss order.

    Parameters
    ----------
    state : array_like
        Non-negative integer-like ``[M, P, Mstar]``.
    parameters : array_like
        Length-10 vector from step 01.

    Raises
    ------
    ValueError
        If either vector has the wrong shape or contains inadmissible values.
    """
    return propensities  # placeholder
```

### Step 3

03_completion_hazard_increment

Goal
----
Integrate pending completion hazards over a fixed-state interval.

```python
def completion_hazard_increment(ages: "np.ndarray", dt: float,
                                state: "np.ndarray",
                                parameters: "np.ndarray") -> "np.ndarray":
    """Integrate every pending transcript's completion hazard over ``dt``.

    The biochemical state is frozen over the interval, but each transcript's
    age advances continuously.

    Raises
    ------
    ValueError
        If ages, state, parameters, or ``dt`` are non-finite or inadmissible.
    """
    return increments  # placeholder
```

### Step 4

04_invert_completion_waits

Goal
----
Convert residual unit clocks into physical completion waits.

```python
def invert_completion_waits(ages: "np.ndarray", residuals: "np.ndarray",
                            state: "np.ndarray",
                            parameters: "np.ndarray") -> "np.ndarray":
    """Return physical waiting times that exhaust the pending residual clocks.

    Raises
    ------
    ValueError
        If arrays have inconsistent shapes, ages are negative, residuals are
        non-positive, or the state and delay parameters are inadmissible.
    """
    return waiting_times  # placeholder
```

### Step 5

05_advance_remaining_clocks

Goal
----
Advance all residual clocks to the next exact event.

```python
def advance_remaining_clocks(reaction_residuals: "np.ndarray",
                             ages: "np.ndarray", completion_residuals: "np.ndarray",
                             dt: float, state: "np.ndarray",
                             parameters: "np.ndarray") -> "np.ndarray":
    """Deduct accumulated hazards and advance delay ages over ``dt``.

    Returns the four reaction residuals, then completion residuals, then ages.

    Raises
    ------
    ValueError
        If clock arrays are inconsistent or contain inadmissible values.
    """
    return packed_clocks  # placeholder
```

### Step 6

06_simulate_exact_gene_path

Goal
----
Simulate one exact remaining clock gene-expression path.

```python
def simulate_exact_gene_path(parameters: "np.ndarray", final_time: float,
                             seed: int, max_events: int = 2000000) -> "np.ndarray":
    """Simulate one exact path and return ``[M,P,Mstar,event_count]``.

    Raises
    ------
    ValueError
        If inputs are inadmissible or the event limit is reached.
    """
    return endpoint  # placeholder
```

### Step 7

07_compute_group_completion_means

Goal
----
Compute aggregate completion means for existing delay groups.

```python
def compute_group_completion_means(group_ages: "np.ndarray",
                                   group_sizes: "np.ndarray", dt: float,
                                   state: "np.ndarray",
                                   parameters: "np.ndarray") -> "np.ndarray":
    """Return Poisson means for pre-existing delay groups over one leap.

    Raises
    ------
    ValueError
        If group arrays disagree, group sizes are not positive counts, or
        the hazard inputs are inadmissible.
    """
    return means  # placeholder
```

### Step 8

08_simulate_tau_gene_path

Goal
----
Simulate one grouped-delay non-Markovian tau-leap path.

```python
def simulate_tau_gene_path(parameters: "np.ndarray", final_time: float,
                           tau: float, seed: int) -> "np.ndarray":
    """Simulate one approximate path and return ``[M,P,Mstar,step_count]``.

    Raises
    ------
    ValueError
        If parameters, times, or seed are inadmissible.
    """
    return endpoint  # placeholder
```

### Step 9

09_estimate_endpoint_means

Goal
----
Estimate endpoint protein means for both simulator laws.

```python
def estimate_endpoint_means(parameters: "np.ndarray", final_time: float,
                            tau: float, n_paths: int,
                            base_seed: int) -> "np.ndarray":
    """Return exact and approximate mean protein counts and standard errors.

    Both standard errors are over the paths, with one degree of freedom
    removed, so at least two paths are required.

    Raises
    ------
    ValueError
        If ``n_paths`` is not an integer of at least two, if ``base_seed`` is
        not a non-negative integer, or if the simulation arguments are
        inadmissible.
    """
    return estimates  # placeholder
```

### Step 10

10_run_nonmarkovian_gene_pipeline

Goal
----
Chain steps 01-09 and return the signed protein-mean bias.

Orchestrator: yes - packs the parameters (pack_gene_parameters), then evaluates the channel rates (gene_channel_propensities), the integrated completion hazard (completion_hazard_increment), its inversion (invert_completion_waits), the clock advance (advance_remaining_clocks) and the delay-group means (compute_group_completion_means) on the initial state, before running both ensembles (estimate_endpoint_means, which drives simulate_exact_gene_path and simulate_tau_gene_path) and forming the signed ratio; it consumes each step's output rather than reimplementing any of them.

```python
def run_nonmarkovian_gene_pipeline(final_time: float = 60.0,
                                    tau: float = 2.5,
                                    n_paths: int = 80,
                                    base_seed: int = 314159) -> float:
    """Return the signed percent bias in approximate mean protein abundance.

    Raises
    ------
    ValueError
        If ``final_time`` or ``tau`` is not positive and finite, if
        ``n_paths`` is not an integer of at least two, if ``base_seed`` is not
        a non-negative integer, or if the exact ensemble mean is not positive.
    """
    return signed_percent_bias  # placeholder
```
