# Material_Science-Molecular_Modeling-73

## Background

Molecular dynamics resolves motion at the femtosecond scale, while the processes that decide chemical and biological function, a ligand leaving its binding site, an ion pair separating in water, a channel opening, happen on scales longer by nine or twelve orders of magnitude. The gap cannot be closed by waiting, and it is not closed by the biasing methods that work so well for free energies, because metadynamics, umbrella sampling and their relatives distort the dynamics itself, whereas a rate constant is a property of the dynamics and not only of the equilibrium distribution. Path sampling takes a different route: it leaves the dynamics unbiased and changes what is sampled, partitioning the space between reactant and product with a series of interfaces along a chosen order parameter and performing Monte Carlo moves in the space of trajectories rather than of configurations.

The dissociation of a simple salt in water is a standard testbed for this machinery, and it is more interesting than its stoichiometry suggests. Separating a potassium and a chloride ion is not a single barrier crossing but a sequence of rearrangements of the surrounding water, so the interionic distance is a natural order parameter without being the slow one. That makes the process diffusive rather than ballistic across much of its range, and it is why the total elapsed time of a dissociation event is distributed very unevenly along the coordinate rather than being concentrated at a single barrier top.

The reward for reconstructing that distribution is not only a rate constant. A rate is a single number and says nothing about mechanism; it cannot distinguish a process limited by a sharp barrier from one limited by slow diffusion across a broad flat region, and it cannot say which part of the coordinate consumes the time. An analysis that tracks where the sampled trajectories sit along the order parameter can answer that, separating the waiting inherent to a rare event from the transit that follows commitment, and locating the region that actually limits the transit a separation that a free-energy profile alone does not make.

## Problem

A partial-path interface-sampling simulation of K$^{+}$–Cl$^{-}$ dissociation in explicit water, using the interionic distance $\lambda$ as the order parameter with sixteen interfaces spaced every 2 Å from 4 to 34 Å, the first and last delimiting the bound and the dissociated state. Every sampled path is confined between the two interfaces neighbouring the one its ensemble is centred on, so no sampled path spans more than two interface spacings and the run never observes a dissociation event from start to finish. For each ensemble I recorded how many paths arrived from each side and departed to each side, and the mean duration of each such type split into the part before that path first crosses the interface its ensemble is centred on, the part between that first crossing and the last one, and the part after the last crossing. I also sampled the excursions that drop below 4 Å and return, whose mean duration was $0$, $43.7$ and $0$ phase points in those same three parts. The order parameter was stored every 20 fs and all durations below are mean numbers of stored phase points.

From these measurements alone, work out the mean time, in picoseconds, that one dissociation event - from the moment the pair first enters the bound state until the moment it first reaches the dissociated state - accumulates in the sampled segments centred on the single interface above 4 Å carrying the largest such time. Set out the reconstruction you use and why its choices are forced by the way the paths were sampled, and give the scalars that fix the number: how many distinct states the reconstruction distinguishes, the mean duration of one event, the share of that duration accumulated at the interface you report, and the accumulated time at the runner-up interface above 4 Å.

| centred on (Å) | arrived from | departed to | paths | before first crossing | between crossings | after last crossing |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| 4 | below | below | 3 820 | 0.0 | 47.3 | 0.0 |
| 4 | below | above | 844 | 0.0 | 0.0 | 26.8 |
| 4 | above | below | 1 970 | 24.1 | 0.0 | 0.0 |
| 6 | below | below | 2 538 | 18.4 | 11.4 | 18.0 |
| 6 | below | above | 1 845 | 18.7 | 8.5 | 24.5 |
| 6 | above | below | 1 477 | 25.2 | 9.3 | 19.2 |
| 6 | above | above | 2 421 | 25.7 | 12.3 | 25.4 |
| 8 | below | below | 2 446 | 23.1 | 108.5 | 22.7 |
| 8 | below | above | 1 915 | 23.4 | 80.8 | 29.2 |
| 8 | above | below | 1 412 | 29.9 | 88.6 | 23.9 |
| 8 | above | above | 2 365 | 30.4 | 117.5 | 30.1 |
| 10 | below | below | 2 455 | 27.6 | 52.0 | 27.2 |
| 10 | below | above | 2 061 | 27.9 | 38.7 | 33.7 |
| 10 | above | below | 1 370 | 34.4 | 42.5 | 28.4 |
| 10 | above | above | 2 271 | 34.9 | 56.3 | 34.6 |
| 12 | below | below | 2 432 | 32.0 | 45.7 | 31.6 |
| 12 | below | above | 2 180 | 32.3 | 34.0 | 38.1 |
| 12 | above | below | 1 453 | 38.8 | 37.3 | 32.8 |
| 12 | above | above | 2 333 | 39.3 | 49.4 | 39.0 |
| 14 | below | below | 2 226 | 36.4 | 40.1 | 36.0 |
| 14 | below | above | 2 125 | 36.7 | 29.9 | 42.5 |
| 14 | above | below | 1 536 | 43.2 | 32.8 | 37.2 |
| 14 | above | above | 2 354 | 43.7 | 43.4 | 43.4 |
| 16 | below | below | 2 278 | 41.1 | 35.2 | 40.7 |
| 16 | below | above | 2 305 | 41.4 | 26.2 | 47.2 |
| 16 | above | below | 1 449 | 47.9 | 28.8 | 41.9 |
| 16 | above | above | 2 089 | 48.4 | 38.1 | 48.1 |
| 18 | below | below | 2 216 | 45.9 | 30.9 | 45.5 |
| 18 | below | above | 2 370 | 46.2 | 23.0 | 52.0 |
| 18 | above | below | 1 456 | 52.7 | 25.2 | 46.7 |
| 18 | above | above | 1 969 | 53.2 | 33.5 | 52.9 |
| 20 | below | below | 2 021 | 49.9 | 27.1 | 49.5 |
| 20 | below | above | 2 279 | 50.2 | 20.2 | 56.0 |
| 20 | above | below | 1 522 | 56.7 | 22.2 | 50.7 |
| 20 | above | above | 2 171 | 57.2 | 29.4 | 56.9 |
| 22 | below | below | 2 022 | 54.7 | 23.8 | 54.3 |
| 22 | below | above | 2 395 | 55.0 | 17.7 | 60.8 |
| 22 | above | below | 1 597 | 61.5 | 19.5 | 55.5 |
| 22 | above | above | 2 396 | 62.0 | 25.8 | 61.7 |
| 24 | below | below | 1 916 | 59.5 | 20.9 | 59.1 |
| 24 | below | above | 2 375 | 59.8 | 15.6 | 65.6 |
| 24 | above | below | 1 501 | 66.3 | 17.1 | 60.3 |
| 24 | above | above | 2 363 | 66.8 | 22.7 | 66.5 |
| 26 | below | below | 2 037 | 64.1 | 18.4 | 63.7 |
| 26 | below | above | 2 634 | 64.4 | 13.7 | 70.2 |
| 26 | above | below | 1 498 | 70.9 | 15.0 | 64.9 |
| 26 | above | above | 2 466 | 71.4 | 19.9 | 71.1 |
| 28 | below | below | 1 950 | 68.6 | 16.1 | 68.2 |
| 28 | below | above | 2 624 | 68.9 | 12.0 | 74.7 |
| 28 | above | below | 1 294 | 75.4 | 13.2 | 69.4 |
| 28 | above | above | 2 220 | 75.9 | 17.5 | 75.6 |
| 30 | below | below | 1 900 | 72.9 | 14.2 | 72.5 |
| 30 | below | above | 2 650 | 73.2 | 10.5 | 79.0 |
| 30 | above | below | 1 289 | 79.7 | 11.6 | 73.7 |
| 30 | above | above | 2 297 | 80.2 | 15.3 | 79.9 |
| 32 | below | below | 1 832 | 77.5 | 12.4 | 77.1 |
| 32 | below | above | 2 643 | 77.8 | 9.3 | 83.6 |
| 32 | above | below | 1 401 | 84.3 | 10.2 | 78.3 |
| 32 | above | above | 2 585 | 84.8 | 13.5 | 84.5 |

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

Implement **all 11 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_compute_local_crossing_probabilities

Goal
----
Reduce the raw partial-path type counts of every ensemble to the local crossing probabilities that drive the chain.

```python
import numpy as np


def compute_local_crossing_probabilities(path_counts) -> np.ndarray:
    """Convert sampled path-type populations into local crossing probabilities.

    Parameters
    ----------
    path_counts : array_like
        Numeric array of shape (n_ensembles, 4) whose entries are whole numbers
        of sampled paths; any numeric dtype is accepted, including float. Row j
        holds the sampled populations of ensemble j in the fixed order (arrive
        below / depart below, arrive below / depart above, arrive above / depart
        below, arrive above / depart above). At least two ensembles are
        required. Entries must be non-negative, and the two populations sharing
        an arrival side must not both vanish.

    Returns
    -------
    probabilities : numpy.ndarray
        Float array of shape (n_ensembles, 2, 2). Element [j, a, b] is the
        probability that a segment of ensemble j which arrived from the side
        indexed by a departs towards the side indexed by b, where index 0 means
        below and index 1 means above.

    Raises
    ------
    ValueError
        If any sampled population is negative or if the two populations
        sharing either arrival side both vanish.
    """
    return np.zeros((0, 2, 2))
```

### Step 2

02_build_state_space

Goal
----
Enumerate the segment types that a long trajectory visits and fix the ordering used by every array downstream.

```python
import numpy as np


def build_state_space(n_interfaces: int) -> np.ndarray:
    """Enumerate the segment types of the partial-path chain.

    Parameters
    ----------
    n_interfaces : int
        Total number of interfaces lambda_0 ... lambda_N, so that
        N = n_interfaces - 1. Must be at least 4.

    Returns
    -------
    states : numpy.ndarray
        Integer array of shape (n_states, 3). Row s holds the labels
        (i, k, l) of state s: i is the index of the interface the segment is
        centred on, k is -1 if the segment arrived from below and +1 if it
        arrived from above, and l is -1 if it departed below and +1 if it
        departed above. The reactant excursion is labelled (-1, +1, +1) and the
        product state is labelled (N, 0, 0).

    Raises
    ------
    ValueError
        If n_interfaces is not an integer or is smaller than 4.
    """
    return np.zeros((0, 3), dtype=int)
```

### Step 3

03_build_transition_matrix

Goal
----
Wire the enumerated segment types into the stochastic matrix of the chain, including the two boundary conventions.

```python
import numpy as np


def build_transition_matrix(states, probabilities) -> np.ndarray:
    """Assemble the stochastic matrix of the partial-path chain.

    Parameters
    ----------
    states : array_like
        Integer array of shape (n_states, 3) holding the (i, k, l) labels of
        every segment type, with the reactant excursion labelled (-1, +1, +1)
        and the product state labelled (N, 0, 0).
    probabilities : array_like
        Float array of shape (N, 2, 2). Element [j, a, b] is the probability
        that a segment of the ensemble centred on lambda_j which arrived from
        the side indexed by a departs towards the side indexed by b, index 0
        meaning below and index 1 meaning above.

    Returns
    -------
    transition_matrix : numpy.ndarray
        Float array of shape (n_states, n_states) whose rows sum to one. Entry
        [s, t] is the probability that the segment following state s is of type
        t. The product state is absorbing.

    Raises
    ------
    ValueError
        If the probability table does not match the interface count implied by
        states or assigns non-zero probability to a segment type absent from
        the straddling ensemble.
    """
    return np.zeros((0, 0))
```

### Step 4

04_compute_crossing_probability

Goal
----
Solve the hitting-probability system of the chain to obtain the probability of reaching the product state before falling back.

```python
import numpy as np


def compute_crossing_probability(transition_matrix, states) -> float:
    """Return the probability of reaching the product state before falling back.

    Parameters
    ----------
    transition_matrix : array_like
        Float array of shape (n_states, n_states) whose rows sum to one, giving
        the probability that one segment type follows another.
    states : array_like
        Integer array of shape (n_states, 3) holding the (i, k, l) labels of
        every segment type, with the reactant excursion labelled (-1, +1, +1)
        and the product state labelled (N, 0, 0).

    Returns
    -------
    crossing_probability : float
        Probability that a trajectory leaving the reactant excursion reaches the
        product state before returning to the reactant state, as a native Python
        float in (0, 1].

    Raises
    ------
    ValueError
        If transition_matrix is not row-stochastic or if the product state is
        unreachable, so the crossing probability is not in (0, 1].
    """
    return 0.0
```

### Step 5

05_compute_overlap_free_times

Goal
----
Turn the measured three-part segment lengths into the time each state contributes to a stitched trajectory without double counting.

```python
import numpy as np


def compute_overlap_free_times(states, reactant_parts, ensemble_parts) -> np.ndarray:
    """Assemble the non-overlapping time contributed by every segment type.

    Parameters
    ----------
    states : array_like
        Integer array of shape (n_states, 3) holding the (i, k, l) labels of
        every segment type, with the reactant excursion labelled (-1, +1, +1)
        and the product state labelled (N, 0, 0).
    reactant_parts : array_like
        Float array of shape (3,) holding the mean leading, middle and trailing
        piece of the reactant excursion, in phase points.
    ensemble_parts : array_like
        Float array of shape (N, 4, 3). Element [j, t, :] holds the mean
        leading, middle and trailing piece of path type t of the ensemble
        centred on lambda_j, in phase points. Types are ordered as (arrive
        below / depart below, arrive below / depart above, arrive above /
        depart below, arrive above / depart above).

    Returns
    -------
    overlap_free_times : numpy.ndarray
        Float array of shape (n_states,) holding the time each state adds to a
        stitched trajectory, in phase points. The product state contributes
        zero because the measurement stops on arrival there.

    Raises
    ------
    ValueError
        If a measured path piece is negative or if ensemble_parts does not
        supply one entry per ensemble implied by states.
    """
    return np.zeros(0)
```

### Step 6

06_compute_visit_counts

Goal
----
Count how often each segment type is visited on the way from the reactant state to the product state.

```python
import numpy as np


def compute_visit_counts(transition_matrix, states) -> np.ndarray:
    """Return the expected number of visits to every segment type per passage.

    Parameters
    ----------
    transition_matrix : array_like
        Float array of shape (n_states, n_states) whose rows sum to one, giving
        the probability that one segment type follows another.
    states : array_like
        Integer array of shape (n_states, 3) holding the (i, k, l) labels of
        every segment type, with the reactant excursion labelled (-1, +1, +1)
        and the product state labelled (N, 0, 0).

    Returns
    -------
    visit_counts : numpy.ndarray
        Float array of shape (n_states,) holding the expected number of visits
        to each state during one passage that starts in the reactant excursion
        and stops on first arrival in the product state. The initial visit is
        counted and the product state entry is zero.

    Raises
    ------
    ValueError
        If the product state is unreachable from the transient states or if
        states and transition_matrix disagree on the state count.
    """
    return np.zeros(0)
```

### Step 7

07_compute_interface_dwell_times

Goal
----
Attribute the accumulated time of one passage to the interface each contribution was measured on.

```python
import numpy as np


def compute_interface_dwell_times(states, visit_counts, overlap_free_times) -> np.ndarray:
    """Group the accumulated time of one passage by middle interface.

    Parameters
    ----------
    states : array_like
        Integer array of shape (n_states, 3) holding the (i, k, l) labels of
        every segment type, with the reactant excursion labelled (-1, +1, +1)
        and the product state labelled (N, 0, 0).
    visit_counts : array_like
        Float array of shape (n_states,) holding the expected number of visits
        to each state during one passage. Entries must be non-negative.
    overlap_free_times : array_like
        Float array of shape (n_states,) holding the non-overlapping time each
        state adds to a stitched trajectory, in phase points. Entries must be
        non-negative.

    Returns
    -------
    dwell_times : numpy.ndarray
        Float array of shape (N,) holding the mean time one passage accumulates
        in segments centred on lambda_0 ... lambda_(N-1), in phase points.

    Raises
    ------
    ValueError
        If visit_counts contains a negative entry or does not have one entry
        per state.
    """
    return np.zeros(0)
```

### Step 8

08_compute_conditional_passage_time

Goal
----
Recover the mean length of a full excursion out of the reactant state, which the partial-path sampling never observes directly.

```python
import numpy as np


def compute_conditional_passage_time(transition_matrix, states, overlap_free_times,
                                     reactant_middle_time: float) -> float:
    """Return the mean duration of a full excursion above the reactant boundary.

    Parameters
    ----------
    transition_matrix : array_like
        Float array of shape (n_states, n_states) whose rows sum to one, giving
        the probability that one segment type follows another.
    states : array_like
        Integer array of shape (n_states, 3) holding the (i, k, l) labels of
        every segment type, with the reactant excursion labelled (-1, +1, +1)
        and the product state labelled (N, 0, 0).
    overlap_free_times : array_like
        Float array of shape (n_states,) holding the non-overlapping time each
        state adds to a stitched trajectory, in phase points.
    reactant_middle_time : float
        Mean time the reactant excursion spends below the reactant boundary
        between its first and last crossing of it, in phase points. Must be
        non-negative.

    Returns
    -------
    conditional_passage_time : float
        Mean duration of one excursion that leaves the reactant boundary and
        ends on returning to the reactant state or on reaching the product
        state, in phase points, as a native Python float.

    Raises
    ------
    ValueError
        If overlap_free_times contains a negative entry or if
        reactant_middle_time is at least the reconstructed stopped time.
    """
    return 0.0
```

### Step 9

09_compute_rate_constant

Goal
----
Combine the reconstructed excursion length with the crossing probability into the transition rate.

```python
import numpy as np


def compute_rate_constant(reactant_time: float, conditional_passage_time: float,
                          crossing_probability: float, time_step: float) -> float:
    """Combine the reconstructed excursion length and the crossing probability.

    Parameters
    ----------
    reactant_time : float
        Mean time spent per excursion below the reactant boundary, in phase
        points. Must be non-negative.
    conditional_passage_time : float
        Mean duration of one full excursion above the reactant boundary, in
        phase points. Must be positive.
    crossing_probability : float
        Probability that an upward crossing of the reactant boundary reaches the
        product state before falling back. Must lie in (0, 1].
    time_step : float
        Physical duration of one phase point, expressed in the time unit whose
        reciprocal the rate is reported in. Must be positive.

    Returns
    -------
    rate_constant : float
        Rate constant of the transition into the product state, in the
        reciprocal of the time unit of time_step, as a native Python float.

    Raises
    ------
    ValueError
        If crossing_probability is not in (0, 1] or if time_step is not
        positive.
    """
    return 0.0
```

### Step 10

10_compute_bottleneck_dwell

Goal
----
Reduce the accumulated-time profile to the dwell time of the interface that limits the transition.

```python
import numpy as np


def compute_bottleneck_dwell(dwell_times, rate_constant: float) -> float:
    """Return the dwell time of the slowest interface above the reactant boundary.

    Parameters
    ----------
    dwell_times : array_like
        Float array of shape (N,) holding the mean time one passage accumulates
        in segments centred on lambda_0 ... lambda_(N-1), in any consistent
        unit. Entries must be non-negative and at least two are required.
    rate_constant : float
        Rate constant of the transition into the product state, in the
        reciprocal of the time unit in which the answer is wanted. Must be
        positive.

    Returns
    -------
    bottleneck_dwell : float
        Mean time one passage accumulates in segments centred on the interface
        above the reactant boundary that carries the largest such time,
        expressed in the time unit implied by rate_constant, as a native Python
        float.

    Raises
    ------
    ValueError
        If no positive dwell time is accumulated above the reactant boundary
        or if rate_constant is not positive.
    """
    return 0.0
```

### Step 11

11_run_kcl_dissociation_pipeline

Goal
----
Chain the sub-problem functions 01-10 end to end on the measured KCl dissociation data set and return the dwell time of the interface that limits the transit.

```python
import numpy as np

def run_kcl_dissociation_pipeline(time_step: float = 0.02,
                                  well_scale: float = 1.0) -> float:
    """Run the full dwell-time measurement on the ion-pair data set.

    The sixteen interfaces sit every 2 Angstrom from 4 to 34 Angstrom along the
    interionic distance, the first and last delimiting the bound and the
    dissociated state. The sampled path-type populations and the mean leading,
    middle and trailing pieces of every path type are the measured input and are
    embedded in this step.

    Parameters
    ----------
    time_step : float
        Physical duration of one phase point, in picoseconds
        (time_step > 0).
    well_scale : float
        Multiplier applied to the middle piece of every path type of the
        ensemble centred on 8 Angstrom, which deepens or flattens the
        solvent-separated well (well_scale > 0).

    Returns
    -------
    bottleneck_dwell : float
        Mean time one dissociation event accumulates in segments centred on the
        interface above the bound-state boundary that carries the largest such
        time, in picoseconds, as a native Python float.

    Raises
    ------
    ValueError
        If time_step or well_scale is not positive.
    """
    return 0.0
```
