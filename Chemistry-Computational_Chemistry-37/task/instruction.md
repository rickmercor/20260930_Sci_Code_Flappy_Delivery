# Chemistry-Computational_Chemistry-37

## Background

Chemical events that matter most are usually the ones that almost never happen. A ligand unbinds, a molecule crosses a lipid bilayer, a bond breaks: on the timescale of a molecular dynamics integration step these are astronomically rare, so a brute-force trajectory spends essentially all of its time rattling around inside a stable basin and almost none of it on the transition that the chemistry depends on. Path sampling methods attack this by treating whole trajectories, rather than individual configurations, as the objects being sampled with Monte Carlo. Interfaces are laid down along a progress coordinate between the reactant and product states, and each interface defines an ensemble of trajectories that penetrate at least that far before falling back. The rate constant then factorises into a flux out of the reactant state and a product of conditional probabilities of advancing from one interface to the next, each of which is an ordinary, well-sampled quantity even when their product is fantastically small.

Two developments drive the current state of the art. The first is the recognition that the trajectories collected in this way carry far more information than the rate alone. Once each sampled path is assigned a correct statistical weight, the same data set yields activation energies, committor functions, reaction-prediction metrics and free energy surfaces, with no additional simulation. Extracting these requires a reweighting scheme, because the ensembles deliberately oversample the barrier region and must be unbiased and stitched back together before any physical average is meaningful.

The second is parallel efficiency. Classical replica exchange between interface ensembles forces the workers handling short trajectories to idle while those handling long ones finish, so the wall-clock cost is set by the slowest ensemble. A recent generation of algorithms removes that synchronisation by letting workers shoot asynchronously and by replacing explicit swaps with their infinite-swap limit, computing analytically the fraction of time a given trajectory would spend in each ensemble under unlimited exchange. The bookkeeping consequence is substantial. A trajectory is no longer the property of one ensemble but is associated with several at once, and the aggressive shooting moves used alongside this draw from a path distribution that is deliberately not the physical one. Both features put the older analysis on a different footing, and settling what a correct statistical weight now means for such a record is an active question in the field.

A quantity that becomes accessible in this framework, and that behaves in a way many practitioners find surprising, is the history-dependent or conditional free energy. Instead of histogramming configurations from an equilibrium simulation, one histograms the phase points lying along trajectories that belong to a particular path ensemble, for instance those that most recently visited the reactant state. The resulting profile is not purely thermodynamic. Because membership of a path ensemble depends on where the trajectory has been, the profile responds to kinetic parameters such as particle masses and the friction coefficient of the thermostat, which leave an ordinary free energy completely untouched. It also retains information about barriers that a poorly chosen reaction coordinate would otherwise hide: when a projection makes a blocked channel look accessible, the unconditional profile reports a modest barrier while the conditional one correctly reports a prohibitive one. Conditional profiles computed from the forward and backward processes can be rescaled by the ratio of the two rate constants and recombined to recover the ordinary free energy, so the conditional description generalises rather than replaces the thermodynamic picture.

One further subtlety belongs to systems whose progress coordinate is unbounded on the reactant side, membrane permeation being the standard example. There the reactant basin has no natural wall and an extra interface has to be placed beyond it to keep the ensemble definitions finite. A symmetry that the bounded case takes for granted, namely that an infinitely long trajectory produces equal numbers of segments on either side of the first interface, no longer holds once that outer interface is present, and an analysis that assumes it will misweight one family of trajectories against the other. Membrane permeation is where this was first confronted in earnest.

## Problem

Rare transitions between long-lived molecular states are studied by sampling whole trajectories rather than single configurations, using a family of path ensembles indexed by interfaces placed along a progress coordinate. Recent implementations advance those ensembles asynchronously across many workers and emulate an unlimited number of replica exchanges among the ones left idle, so the record such a run produces lists, for every trajectory and every ensemble, a visitation count that need not be a whole number together with the acceptance weight that the shooting move carried in that ensemble. Your task is to turn one such record into a history-dependent free energy profile along the progress coordinate and report a single number.

The record holds trajectories of two kinds. Reactant-side trajectories stay below the first interface and are the only ones that populate the region behind it; the progress coordinate is unbounded on that side, so a further interface is placed beyond the reactant basin and each of these trajectories terminates at one of the two. Barrier-side trajectories begin and end at the first interface unless they reach the last one, and each is characterised by the furthest point it reached along the coordinate.

Reconstruct the free energy conditioned on the trajectories that most recently visited the reactant state, expressed in units of the thermal energy on a scale whose minimum is zero, and report its value in the highest bin.

Use exactly this configuration:

- interfaces: lambda_0 through lambda_4 at -1.0, -0.4, 0.2, 0.7, 1.0; the reactant state is lambda < -1.0 and the product state is lambda > 1.0
- outer reactant interface: lambda_-1 = -1.6
- 12 barrier-side trajectories, indexed j = 0..11, with furthest progress lambda_max = [-0.95, -0.72, -0.40, -0.18, 0.05, 0.20, 0.36, 0.52, 0.70, 0.84, 0.97, 1.12]
- their length parameters L = [11, 7, 10, 6, 9, 12, 8, 11, 7, 10, 6, 9]; a trajectory of length parameter L holds L+1 phase points in total
- visitation count of trajectory j in the ensemble associated with interface k (k = 0..3): mu = ((j + 2k + 1) mod 4) + 1 whenever lambda_k < lambda_max of that trajectory, and 0 otherwise
- acceptance weight in that ensemble, defined only where the visitation count is non-zero and equal to 1 elsewhere: w = 1 for k = 0, and w = 1 + 0.75 ((j + 3k) mod 4) otherwise
- 6 reactant-side trajectories with length parameters [8, 5, 7, 9, 6, 8] and visitation counts [1, 2, 3, 1, 2, 3]; the third and the fifth terminate at lambda_-1, the other four terminate at lambda_0
- interior phase points: of the L+1 phase points of a trajectory, L-1 are interior. Writing p = floor((L-2)/2), a barrier-side trajectory that does not reach the product state places its m-th interior point (m = 0, 1, ...) at lambda_0 + (lambda_max - lambda_0) (m+1)/(p+1) for m <= p and at lambda_0 + (lambda_max - lambda_0) (L-1-m)/(L-1-p) for m > p. One that reaches the product state rises monotonically instead, its m-th interior point sitting at lambda_0 + (lambda_4 - lambda_0) (m+1)/L. A reactant-side trajectory follows the same rise-and-fall rule with lambda_-1 in place of lambda_max.
- histogram: 8 equal-width bins spanning lambda_-1 to lambda_4, with a point on a shared edge assigned to the upper bin
- reduced units, so that the thermal energy is 1

Your final answer must be a single number: the conditional free energy of the highest bin. Name your intermediate checkpoints as you go rather than reporting the final value alone: the per-ensemble totals you derive from the record, the ensemble label you assign each barrier-side trajectory, the crossing probability of each interface, the weight you assign each trajectory in both families together with any factors entering those weights, the bin populations of the density, and the reference value that sets the zero of the profile. The instruction below against pasting tables refers to the supplied input data, not to these checkpoints.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.
You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you
are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that
determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration
paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 13 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

sampling_record

Goal
----
Build the per-ensemble bookkeeping of the simulation record. A trajectory belongs to the ensemble associated with interface k only when its furthest progress along the coordinate lies strictly beyond that interface, so the visitation counts form a staircase pattern that is non-zero only for the low-lying ensembles of trajectories that did not travel far. Return both the visitation counts and the acceptance weights of the biased shooting moves, stacked into one array, using the construction rules given in the problem.

```python
import numpy as np


def sampling_record(lam_max: np.ndarray, interfaces: np.ndarray) -> np.ndarray:
    '''Assemble visitation counts and acceptance weights for every trajectory and ensemble.

    Parameters
    ----------
    lam_max : np.ndarray
        (P,) furthest progress along the order parameter reached by each trajectory.
    interfaces : np.ndarray
        (n+1,) strictly increasing interface positions, the first being the reactant
        boundary and the last the product boundary.

    Returns
    -------
    record : np.ndarray
        (2, P, n) array whose first slab holds the visitation counts and whose second
        slab holds the acceptance weights.
    
    Raises
    ------
    ValueError
        lam_max must be a 1D array with at least one entry.
        interfaces must be a 1D array with at least three entries.
        interfaces must be strictly increasing.
        lam_max must be finite.
    '''
    return record
```

### Step 2

ensemble_path_totals

Goal
----
Reduce the visitation counts to one total per ensemble. Because the counts are fractional, these totals play the role that the plain number of sampled paths plays in a synchronous simulation, and they set the scale against which every later weight is normalised.

```python
import numpy as np


def ensemble_path_totals(mu: np.ndarray) -> np.ndarray:
    '''Total fractional number of trajectories contributed by each ensemble.

    Parameters
    ----------
    mu : np.ndarray
        (P, n) visitation count of each trajectory in each ensemble.

    Returns
    -------
    eta : np.ndarray
        (n,) total fractional trajectory count per ensemble.
    
    Raises
    ------
    ValueError
        mu must be a non-empty 2D array.
        mu entries must be finite.
        mu entries must be non-negative.
    '''
    return eta
```

### Step 3

unbiased_sampling_weights

Goal
----
Remove the sampling bias and place the ensembles on a common scale. Each trajectory is counted inversely to the acceptance weight it carried in a given ensemble, and the resulting column is then rescaled so that it sums back to that ensemble's own total visitation count. The second operation is not cosmetic: the combination rule that follows compares ensembles against one another, so their weights are only meaningful once fixed to a shared scale.

```python
import numpy as np


def unbiased_sampling_weights(mu: np.ndarray, w: np.ndarray) -> np.ndarray:
    '''Unbias the visitation counts and rescale them onto a common per-ensemble scale.

    Parameters
    ----------
    mu : np.ndarray
        (P, n) visitation count of each trajectory in each ensemble.
    w : np.ndarray
        (P, n) acceptance weight of each trajectory in each ensemble.

    Returns
    -------
    t : np.ndarray
        (P, n) unbiased sampling weights, each column summing to that ensemble's
        total visitation count.
    
    Raises
    ------
    ValueError
        mu must be a non-empty 2D array.
        mu and w must have the same shape.
        mu and w must be finite.
        mu entries must be non-negative.
        w must be strictly positive wherever mu is non-zero.
    '''
    return t
```

### Step 4

highest_ensemble_index

Goal
----
Label each trajectory with the highest-indexed ensemble it belongs to. The label is fixed by where the trajectory's furthest progress falls relative to the interfaces, using an interval that excludes its lower interface and includes its upper one, with everything beyond the last sampled interface collapsed onto that final index. Trajectories whose maximum coincides exactly with an interface therefore take the lower of the two candidate labels.

```python
import numpy as np


def highest_ensemble_index(lam_max: np.ndarray, interfaces: np.ndarray) -> np.ndarray:
    '''Highest ensemble index that each trajectory still belongs to.

    Parameters
    ----------
    lam_max : np.ndarray
        (P,) furthest progress reached by each trajectory.
    interfaces : np.ndarray
        (n+1,) strictly increasing interface positions.

    Returns
    -------
    index : np.ndarray
        (P,) integer ensemble index in the range 0 to n-1.
    
    Raises
    ------
    ValueError
        lam_max must be a 1D array with at least one entry.
        interfaces must be a 1D array with at least three entries.
        interfaces must be strictly increasing.
        lam_max must be finite.
        every trajectory must pass the first interface.
    '''
    return index
```

### Step 5

interface_crossing_totals

Goal
----
For one nominated interface, accumulate per ensemble the unbiased sampling weight carried by the trajectories that advanced beyond it. Trajectories that fell back before reaching the interface contribute nothing. These restricted totals are the numerators of the crossing probability that follows.

```python
import numpy as np


def interface_crossing_totals(t: np.ndarray, lam_max: np.ndarray,
                              interfaces: np.ndarray, level: int) -> np.ndarray:
    '''Unbiased weight per ensemble carried by trajectories passing one interface.

    Parameters
    ----------
    t : np.ndarray
        (P, n) unbiased sampling weights.
    lam_max : np.ndarray
        (P,) furthest progress reached by each trajectory.
    interfaces : np.ndarray
        (n+1,) strictly increasing interface positions.
    level : int
        Index of the interface being crossed.

    Returns
    -------
    totals : np.ndarray
        (n,) restricted weight contributed by each ensemble.
    
    Raises
    ------
    ValueError
        t must be a non-empty 2D array.
        t must carry one row per trajectory.
        interfaces must be a 1D array with at least three entries.
        interfaces must be strictly increasing.
        level must be an integer.
        level must index one of the interfaces.
    '''
    return totals
```

### Step 6

crossing_probabilities

Goal
----
Run the forward recursion that turns the per-interface restricted totals into the probability of advancing from the reactant boundary to each successive interface. Two features of the recursion decide the result: only the ensembles whose index lies strictly below the target interface contribute to its estimate, and the summed restricted totals are scaled by the normaliser belonging to the previous step of the recursion rather than the current one. Return the full ladder of probabilities, the first of which is unity.

```python
import numpy as np


def crossing_probabilities(crossing_totals: np.ndarray,
                           ensemble_totals: np.ndarray) -> np.ndarray:
    '''Ladder of crossing probabilities from the per-interface restricted totals.

    Parameters
    ----------
    crossing_totals : np.ndarray
        (n, n) restricted totals. Row i-1 holds, per ensemble, the unbiased weight
        carried by the trajectories that advanced beyond interface i, for i = 1..n.
    ensemble_totals : np.ndarray
        (n,) total visitation count of each ensemble.

    Returns
    -------
    probs : np.ndarray
        (n+1,) crossing probabilities, with probs[0] equal to 1.0.
    
    Raises
    ------
    ValueError
        ensemble_totals must be a non-empty 1D array.
        crossing_totals must be square with one row per interface.
        totals must be non-negative.
        the lowest ensemble must carry non-zero weight.
        crossing probability vanished before the last ensemble.
    '''
    return probs
```

### Step 7

wham_normalisers

Goal
----
Form the per-index normalisers of the multi-ensemble combination. The normaliser at index i is the reciprocal of a cumulative sum running over ensembles 0 through i, in which each ensemble total is divided by the crossing probability of its own interface. The lowest normaliser is therefore the reciprocal of the lowest ensemble total.

```python
import numpy as np


def wham_normalisers(eta: np.ndarray, crossing_probs: np.ndarray) -> np.ndarray:
    '''Cumulative weighted-histogram normalisers, one per ensemble index.

    Parameters
    ----------
    eta : np.ndarray
        (n,) total fractional trajectory count per ensemble.
    crossing_probs : np.ndarray
        (n+1,) crossing probabilities, the first entry equal to 1.

    Returns
    -------
    norm : np.ndarray
        (n,) normaliser for each ensemble index.
    
    Raises
    ------
    ValueError
        eta must be a 1D array with at least one entry.
        crossing_probs must hold exactly one more entry than eta.
        eta and crossing_probs must be finite.
        eta entries must be non-negative.
        crossing probabilities must be strictly positive.
        cumulative normalising sum must be strictly positive.
    '''
    return norm
```

### Step 8

plus_path_weights

Goal
----
Collapse the ensemble bookkeeping into one weight per barrier-side trajectory: the normaliser evaluated at that trajectory's highest ensemble index, multiplied by the sum of its unbiased sampling weights taken over every ensemble. The sum extends over all ensembles rather than stopping at the trajectory's own index, which is harmless because the weights vanish above it.

```python
import numpy as np


def plus_path_weights(t: np.ndarray, ensemble_index: np.ndarray,
                      normalisers: np.ndarray) -> np.ndarray:
    '''Single reweighting weight for each barrier-side trajectory.

    Parameters
    ----------
    t : np.ndarray
        (P, n) unbiased sampling weights.
    ensemble_index : np.ndarray
        (P,) highest ensemble index of each trajectory.
    normalisers : np.ndarray
        (n,) cumulative weighted-histogram normalisers.

    Returns
    -------
    weights : np.ndarray
        (P,) weight of each trajectory, summing to 1 for a consistent record.
    
    Raises
    ------
    ValueError
        t must be a non-empty 2D array.
        ensemble_index must carry one entry per trajectory.
        ensemble_index must hold integers.
        normalisers must carry one entry per ensemble.
        ensemble_index entries must address the normalisers.
        t and normalisers must be finite.
    '''
    return weights
```

### Step 9

minus_path_weights

Goal
----
Weight the reactant-side trajectories. This ensemble does not overlap any of the barrier-side ensembles, so no multi-ensemble combination is involved and no acceptance bias has to be removed: the weights are the visitation counts normalised to sum to unity.

```python
import numpy as np


def minus_path_weights(minus_multiplicities: np.ndarray) -> np.ndarray:
    '''Normalised weights of the reactant-side trajectories.

    Parameters
    ----------
    minus_multiplicities : np.ndarray
        (M,) visitation count of each reactant-side trajectory.

    Returns
    -------
    weights : np.ndarray
        (M,) weights summing to 1.
    
    Raises
    ------
    ValueError
        minus_multiplicities must be a 1D array with at least one entry.
        minus_multiplicities must be finite.
        minus_multiplicities must be non-negative.
        the reactant-side ensemble must carry non-zero weight.
    '''
    return weights
```

### Step 10

end_interface_fraction

Goal
----
Compute the weighted fraction of reactant-side trajectories that terminate at the inner interface rather than at the outer one. This is a weighted average of an indicator over the reactant-side ensemble, so each trajectory enters through its own normalised weight and not through a plain head count.

```python
import numpy as np


def end_interface_fraction(minus_weights: np.ndarray,
                           ends_at_first_interface: np.ndarray) -> float:
    '''Weighted fraction of reactant-side trajectories ending at the inner interface.

    Parameters
    ----------
    minus_weights : np.ndarray
        (M,) normalised weights of the reactant-side trajectories.
    ends_at_first_interface : np.ndarray
        (M,) boolean flags, true where a trajectory terminates at the inner interface.

    Returns
    -------
    fraction : float
        Weighted fraction between 0 and 1.
    
    Raises
    ------
    ValueError
        minus_weights must be a 1D array with at least one entry.
        ends_at_first_interface must match minus_weights in length.
        minus_weights must be finite.
        minus_weights must be non-negative.
    '''
    return fraction
```

### Step 11

path_slice_values

Goal
----
Place the interior phase points of one trajectory along the order parameter, following the rise-and-fall profile specified in the problem. A trajectory of length parameter L holds L+1 phase points, of which the L-1 interior ones are kept and the two end points discarded. A trajectory that reaches the product state rises monotonically instead, and a reactant-side trajectory uses the same rise-and-fall rule with the outer interface as its turning point.

```python
import numpy as np


def path_slice_values(lam_max_value: float, path_length: int, interfaces: np.ndarray,
                      lower_turning_point: float = None) -> np.ndarray:
    '''Order-parameter values of the interior phase points of one trajectory.

    Parameters
    ----------
    lam_max_value : float
        Furthest progress reached by the trajectory.
    path_length : int
        Length parameter of the trajectory, which holds L+1 phase points
        in total and L-1 interior ones.
    interfaces : np.ndarray
        (n+1,) strictly increasing interface positions.
    lower_turning_point : float, optional
        Turning point for a reactant-side trajectory. When given it replaces
        lam_max_value as the extreme of the profile.

    Returns
    -------
    values : np.ndarray
        (path_length - 1,) order-parameter values of the interior phase points.
    
    Raises
    ------
    ValueError
        interfaces must be a 1D array with at least three entries.
        interfaces must be strictly increasing.
        path_length must be an integer.
        path_length must be at least 3.
        lam_max_value must be finite.
        lower_turning_point must be finite.
    '''
    return values
```

### Step 12

conditional_density_histogram

Goal
----
Accumulate the weighted phase points of both families into a histogram along the order parameter. Each family arrives as a flat list of interior positions together with the weight of the trajectory each position came from. Reactant-side points enter with their weight unchanged, while barrier-side points enter with theirs scaled by the fraction of reactant-side trajectories terminating at the inner interface. Bins are of equal width and span the stated bounds, with a point lying on a shared edge assigned to the upper bin.

```python
import numpy as np


def conditional_density_histogram(plus_values: np.ndarray, plus_weights: np.ndarray,
                                  minus_values: np.ndarray, minus_weights: np.ndarray,
                                  end_fraction: float, lower_bound: float,
                                  upper_bound: float, n_bins: int) -> np.ndarray:
    '''Unnormalised conditional density along the order parameter.

    Parameters
    ----------
    plus_values : np.ndarray
        (A,) order-parameter positions of every barrier-side interior phase point.
    plus_weights : np.ndarray
        (A,) weight of the trajectory each barrier-side position came from.
    minus_values : np.ndarray
        (B,) order-parameter positions of every reactant-side interior phase point.
    minus_weights : np.ndarray
        (B,) weight of the trajectory each reactant-side position came from.
    end_fraction : float
        Fraction of reactant-side trajectories terminating at the inner interface.
    lower_bound : float
        Lower edge of the histogram.
    upper_bound : float
        Upper edge of the histogram.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    density : np.ndarray
        (n_bins,) accumulated weight per bin.
    
    Raises
    ------
    ValueError
        the barrier-side positions and weights must be 1D and equal in length.
        the reactant-side positions and weights must be 1D and equal in length.
        n_bins must be a positive integer.
        the histogram bounds must be finite.
        lower_bound must lie below upper_bound.
        end_fraction must be finite and non-negative.
    '''
    return density
```

### Step 13

conditional_free_energy_barrier

Goal
----
Run the whole analysis end to end and return the requested scalar. Thread the record through every earlier step in order: build the visitation counts and acceptance weights, reduce them to per-ensemble totals, unbias and rescale them, label each trajectory with its highest ensemble, collect the restricted totals at each interface, run the forward recursion for the crossing probabilities, form the normalisers, collapse the bookkeeping into one weight per trajectory on each side, obtain the fraction of reactant-side trajectories terminating at the inner interface, place the interior phase points of every trajectory along the coordinate, accumulate them into the conditional density, and convert that density to a free energy on a scale whose minimum is zero. Return the free energy of the highest bin.

```python
import numpy as np


def conditional_free_energy_barrier(interfaces: np.ndarray, lower_bound: float,
                                    lam_max: np.ndarray, plus_lengths: np.ndarray,
                                    minus_lengths: np.ndarray,
                                    minus_multiplicities: np.ndarray,
                                    ends_at_first_interface: np.ndarray,
                                    n_bins: int) -> float:
    '''Conditional free energy of the highest bin, in units of the thermal energy.

    Parameters
    ----------
    interfaces : np.ndarray
        (n+1,) strictly increasing interface positions.
    lower_bound : float
        Outer reactant interface, the lower edge of the histogram.
    lam_max : np.ndarray
        (P,) furthest progress of each barrier-side trajectory.
    plus_lengths : np.ndarray
        (P,) length parameter of each barrier-side trajectory.
    minus_lengths : np.ndarray
        (M,) length parameter of each reactant-side trajectory.
    minus_multiplicities : np.ndarray
        (M,) visitation count of each reactant-side trajectory.
    ends_at_first_interface : np.ndarray
        (M,) boolean flags, true where a reactant-side trajectory ends at the inner interface.
    n_bins : int
        Number of equal-width histogram bins.

    Returns
    -------
    barrier : float
        Conditional free energy of the highest bin.
    
    Raises
    ------
    ValueError
        interfaces must be a 1D array with at least three entries.
        interfaces must be strictly increasing.
        lam_max and plus_lengths must have equal length.
        the reactant-side arrays must have equal length.
        n_bins must be a positive integer.
        lower_bound must be finite and below the first interface.
        every barrier-side trajectory must pass the first interface.
        every trajectory must hold at least three phase points.
        every histogram bin must receive weight.
    '''
    return barrier
```
