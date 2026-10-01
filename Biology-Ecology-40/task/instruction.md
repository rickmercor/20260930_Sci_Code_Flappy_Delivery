# Biology-Ecology-40

## Background

Microbes facing several resources adopt one of three broad strategies: specialists use a single resource, hierarchical (diauxic) utilisers consume resources largely one after another, and co-utilisers consume them simultaneously. Proteome allocation couples these strategies to a trade-off, because enzyme committed to a resource that is not the best one available lowers the current growth rate but shortens the lag when the best resource runs out. Which strategy evolution selects in a boom-and-bust environment whose supply ratio fluctuates from cycle to cycle is not settled by the trade-off alone.

## Problem

Microbial strains compete in a serial-dilution environment on two substitutable resources of equal quality. Each cycle lasts 24 hours: at its start one unit of resource, in the units of biomass it yields (yield one on either resource), is supplied as the two resources in proportions drawn independently each cycle from the symmetric Dirichlet distribution $\mathrm{Dir}(\alpha, \alpha)$ with $\alpha = 0.03$; at its end the culture, biomass only, is diluted 100-fold. Both resources are present at the start of every cycle however small the drawn amount, every cycle begins with no strain in a lag, and resources are far above half-saturation, so a growing strain grows exponentially at a constant rate.

Every strain has a fixed preference order over the two resources and one heritable trait, the secondary allocation $\phi \in [0, 1]$: of its metabolic proteome it allocates $\phi/n$ to each of the $n$ resources currently present and the remaining $1 - \phi$ in addition to the more preferred of them, and it grows on each present resource in proportion to the enzyme it holds for it, with potential growth rates, for the whole metabolic proteome devoted to one resource, of $0.8$ per hour on its preferred resource and $0.4$ per hour on the other. When a resource that a strain is consuming runs out, the strain stops growing while it rebuilds its metabolic proteome around the resource still present: the fraction of its metabolic proteome devoted to that resource grows in proportion to itself with timescale $\tau_0 = 0.3$ h, and growth resumes, at the full rate on that resource, once the fraction reaches one. The community is founded by two specialists ($\phi = 0$), one preferring each resource, and $\phi$ evolves by rare mutations that keep the parent's preference order and must invade the resident community. Compute the evolutionarily stable secondary allocation $\phi^{*}$ in this environment, the value to which $\phi$ evolves and which, once resident, no rare mutant can invade.

Report $\phi^{*}$ to three decimal places. The answer is graded within $\pm 0.03$.

Report also, inside the reasoning: the composition of the resident community at $\phi^{*}$; the lag a resident strain suffers when its preferred resource runs out and when its other resource runs out; its growth rate while both resources are present; the quantity you used as the invasion fitness of a rare mutant and how you evaluated it; the value of that invasion fitness, per cycle, for a rare specialist ($\phi = 0$) arising in the resident community at $\phi^{*}$; the evidence that $\phi^{*}$ is both approached by evolution and uninvadable; the strategy class, specialist, hierarchical utiliser or co-utiliser, to which $\phi^{*}$ belongs and the class boundaries you used; and the mechanism by which the amplitude of the supply-ratio fluctuations sets $\phi^{*}$.

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

Implement **all 8 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

step_01_proteome_allocation

Goal
----
Given the secondary allocation phi of a strain, its preference order over the resources and a Boolean vector marking which resources are currently present, return the metabolic proteome allocation over all resources, the potential growth rates, the per-resource uptake rates per unit biomass (allocation times potential rate) and the total growth rate (their sum). When no resource is present the allocation, the uptake and the growth rate are all zero.

```python
def proteome_allocation(phi: float, preference: np.ndarray, present: np.ndarray,
                        max_rate: float = 0.8, other_factor: float = 0.5) -> dict:
    """Metabolic proteome allocation, uptake and growth rate of a strain.

    Parameters
    ----------
    phi : float
        Secondary allocation in [0, 1].
    preference : np.ndarray
        Permutation of the resource indices, most preferred first, shape (n,).
    present : np.ndarray
        Boolean presence of each resource, shape (n,).
    max_rate : float
        Potential growth rate on the most preferred resource, per hour.
    other_factor : float
        Potential rate on any other resource relative to max_rate.

    Returns
    -------
    dict
        Under the keys allocation, potential_rates, uptake and growth_rate.

    Raises
    ------
    ValueError
        When phi lies outside [0, 1], the preference is not an integer-typed permutation of at least
        two indices, the presence vector is not a Boolean array matching it, or a rate parameter is
        invalid.
    """
    return
```

### Step 2

step_02_reallocation_lag

Goal
----
Given a strain's secondary allocation phi and preference order, the index of the resource that has just been exhausted and the Boolean presence vector just after the exhaustion, return the reallocation lag in hours and the fraction f0 of the metabolic proteome that was allocated, just before the exhaustion, to the resources still present. Use the allocation rule of the preceding step with the exhausted resource counted as present. Return a lag of zero, with f0 equal to one, when the strain was not consuming the exhausted resource or when no resource remains; return an infinite lag when f0 is zero.

```python
def reallocation_lag(phi: float, preference: np.ndarray, depleted: int, present_after: np.ndarray,
                     lag_scale: float = 0.3) -> dict:
    """Lag after a consumed resource is exhausted, from autocatalytic proteome reallocation.

    Parameters
    ----------
    phi : float
        Secondary allocation in [0, 1].
    preference : np.ndarray
        Permutation of the resource indices, most preferred first.
    depleted : int
        Index of the resource just exhausted.
    present_after : np.ndarray
        Boolean presence of each resource just after the exhaustion.
    lag_scale : float
        Autocatalytic timescale tau0, hours.

    Returns
    -------
    dict
        Under the keys lag and retained_fraction.

    Raises
    ------
    ValueError
        When phi lies outside [0, 1], the preference is not an integer-typed permutation, the
        presence vector is not a Boolean array, the exhausted index is not an integer, is out of
        range or is still marked present, or lag_scale is not positive.
    """
    return
```

### Step 3

step_03_first_exhaustion

Goal
----
Given the remaining stock of every resource, the biomass, growth rate and per-resource uptake rates of every strain, and a positive horizon, return, resolved to near machine precision, the earliest time within the horizon at which a resource with positive stock is exhausted and the index of that resource. If no resource runs out within the horizon, return the horizon itself and the index -1. Resources with zero stock are already gone and are ignored. The growth rate of each strain must equal the sum of its uptake rates.

```python
def first_exhaustion(stock: np.ndarray, biomass: np.ndarray, rates: np.ndarray, uptake: np.ndarray,
                     horizon: float) -> dict:
    """Earliest exhaustion of a resource under exponential growth within a horizon.

    Parameters
    ----------
    stock : np.ndarray
        Remaining resource amounts, shape (n,).
    biomass : np.ndarray
        Strain biomasses, shape (m,).
    rates : np.ndarray
        Strain growth rates per hour, shape (m,).
    uptake : np.ndarray
        Per-biomass uptake rates, shape (m, n).
    horizon : float
        Length of the interval, hours.

    Returns
    -------
    dict
        Under the keys time and resource.

    Raises
    ------
    ValueError
        When an array has the wrong shape, holds a negative or non-finite value, an uptake row
        does not sum to its growth rate, or the horizon is not positive and finite.
    """
    return
```

### Step 4

step_04_grow_one_cycle

Goal
----
Given the secondary allocations, preference orders and starting biomasses of a community of strains and the amounts of the two resources supplied, integrate one cycle and return the final biomass of every strain, the time at which each resource was exhausted (infinity if it was not exhausted within the cycle), the index of the resource exhausted first (-1 if none) and the lag each strain incurred during the cycle (zero if none, infinity for a strain that can never switch). A resource is exhausted when the root finder of the preceding step returns it or, at the end of any interval, when its remaining stock has fallen to one part in 10^12 of the amount supplied or below; resources exhausted at the same instant are exhausted together, and when none remains no lag is incurred. The first exhausted index is the one returned by the root finder when there is one.

```python
def grow_one_cycle(phi: np.ndarray, preference: np.ndarray, biomass: np.ndarray, supply: np.ndarray,
                   lag_scale: float = 0.3, cycle_length: float = 24.0) -> dict:
    """Event-driven integration of one growth cycle on two substitutable resources.

    Parameters
    ----------
    phi : np.ndarray
        Secondary allocations, shape (m,).
    preference : np.ndarray
        Preference orders, shape (m, 2).
    biomass : np.ndarray
        Starting biomasses, shape (m,).
    supply : np.ndarray
        Amounts of the two resources supplied, shape (2,).
    lag_scale : float
        Autocatalytic timescale tau0, hours.
    cycle_length : float
        Cycle duration, hours.

    Returns
    -------
    dict
        Under the keys biomass, exhaustion_times, first_exhausted and lags.

    Raises
    ------
    ValueError
        When the arrays disagree in shape, a trait lies outside [0, 1], a preference row is not
        an integer permutation of (0, 1), a biomass or a supply is not positive and finite, or a
        time parameter is not positive and finite.
    """
    return
```

### Step 5

step_05_invasion_growth_rates

Goal
----
Given the resident trait phi, an array of mutant trait values and a sequence of per-cycle supplies of the two resources, integrate the symmetric resident pair through the sequence (resident 0 prefers resource 0, resident 1 prefers resource 1; after each cycle every biomass is divided by the dilution factor). In every cycle, replay each mutant, for both preference orders, against the residents' exhaustion schedule, and return the long-run invasion growth rates averaged over the cycles after the burn-in. Also return the same average for the residents themselves, obtained by the replay, the largest absolute difference, over all retained cycles and both residents, between a resident's replayed and integrated log fold changes, and the mean durations over the retained cycles of the primary temporal niche (from the start of the cycle to the first exhaustion, capped at the cycle length) and of the secondary temporal niche (from the first exhaustion to the second, both capped at the cycle length).

```python
def invasion_growth_rates(phi_resident: float, phi_mutants: np.ndarray, supply: np.ndarray, burn_in: int,
                          dilution: float = 100.0, lag_scale: float = 0.3, cycle_length: float = 24.0) -> dict:
    """Long-run invasion growth rates of rare mutants into the symmetric resident pair.

    Parameters
    ----------
    phi_resident : float
        Resident secondary allocation.
    phi_mutants : np.ndarray
        Mutant secondary allocations, shape (k,).
    supply : np.ndarray
        Per-cycle supplies of the two resources, shape (n_cycles, 2).
    burn_in : int
        Cycles discarded before averaging.
    dilution : float
        Dilution factor.
    lag_scale : float
        Autocatalytic timescale tau0, hours.
    cycle_length : float
        Cycle duration, hours.

    Returns
    -------
    dict
        Under the keys invasion_rates, resident_rates, replay_error, mean_share, primary_niche
        and secondary_niche.

    Raises
    ------
    ValueError
        When a trait lies outside its range, the supply is not a positive finite (n_cycles, 2)
        array, the burn-in leaves no cycle, the dilution factor does not exceed one, or a time
        parameter is not positive and finite.
    """
    return
```

### Step 6

step_06_selection_gradient

Goal
----
Given the resident trait phi, the supply sequence, the burn-in and a step h in ln phi, compute the invasion growth rates of mutants with traits phi exp(-h), phi and phi exp(h), for both preference orders, against the symmetric resident pair, and return the selection gradient (the central difference of the rates divided by 2h) and the curvature (the second central difference divided by h squared), each averaged over the two preference orders, together with the three-by-two table of rates. The largest mutant trait must not exceed one.

```python
def selection_gradient(phi: float, supply: np.ndarray, burn_in: int, step: float = 0.05,
                       dilution: float = 100.0, lag_scale: float = 0.3, cycle_length: float = 24.0) -> dict:
    """Selection gradient and curvature of the invasion growth rate in ln phi.

    Parameters
    ----------
    phi : float
        Resident secondary allocation.
    supply : np.ndarray
        Per-cycle supplies of the two resources, shape (n_cycles, 2).
    burn_in : int
        Cycles discarded before averaging.
    step : float
        Central-difference step in ln phi.
    dilution : float
        Dilution factor.
    lag_scale : float
        Autocatalytic timescale tau0, hours.
    cycle_length : float
        Cycle duration, hours.

    Returns
    -------
    dict
        Under the keys gradient, curvature and rates.

    Raises
    ------
    ValueError
        When the step lies outside (0, 0.5], phi lies outside (0, 1] or phi exp(step) exceeds
        one, or the supply and burn-in are invalid.
    """
    return
```

### Step 7

step_07_singular_strategy

Goal
----
Given the supply sequence, the burn-in and a bracket (lower, upper) in phi at which the selection gradient of the preceding step is respectively positive and negative, locate the singular strategy phi* by the Illinois method in ln phi, stopping when the bracket or the step in ln phi falls below the tolerance, and return phi*, the gradients at the two ends of the initial bracket, the gradient and the curvature at phi*, the slope of the gradient with respect to ln phi at phi*, taken as the central difference of the gradient between phi* exp(-0.02) and phi* exp(0.02) divided by 0.04, and the number of gradient evaluations made inside the bracket.

```python
def singular_strategy(supply: np.ndarray, burn_in: int, lower: float = 0.01, upper: float = 0.9,
                      tolerance: float = 1e-9) -> dict:
    """Singular strategy of the secondary allocation by bracketed root finding in ln phi.

    Parameters
    ----------
    supply : np.ndarray
        Per-cycle supplies of the two resources, shape (n_cycles, 2).
    burn_in : int
        Cycles discarded before averaging.
    lower : float
        Lower bracket end, where the gradient is positive.
    upper : float
        Upper bracket end, where the gradient is negative.
    tolerance : float
        Convergence tolerance in ln phi.

    Returns
    -------
    dict
        Under the keys phi_star, gradient_lower, gradient_upper, gradient_star, curvature,
        convergence_slope and evaluations.

    Raises
    ------
    ValueError
        When the bracket is invalid, the gradient does not change sign across it, or the
        tolerance is not positive.
    """
    return
```

### Step 8

step_08_evolutionarily_stable_allocation

Goal
----
Given the Dirichlet concentration alpha, the number of cycles, the burn-in and a seed, draw the supply sequence as an (n_cycles, 2) array of gamma variates of shape alpha and unit scale in a single call to numpy.random.default_rng(seed).gamma, floor every variate at 1e-300 and divide each row by its sum; locate the singular strategy phi* with the preceding step using its default bracket and tolerance, and evaluate the invasion growth rates against the symmetric resident pair at phi* of mutants with traits phi* exp(x) for 25 values of x evenly spaced from -4 to ln(1 / phi*), together with a pure specialist, phi = 0. Return phi*, the largest invasion growth rate on that grid minus the resident's own rate (the largest over both preference orders), the trait at which it occurs, the curvature and the convergence slope at phi*, the mean share of resident biomass held by the resident preferring resource 0, and the replay error of the resident pair. Return also the two-resource growth rate of a resident at phi* and its lags after the exhaustion of its preferred and of its other resource; the selection gradient at phi* exp(-0.5) and at the smaller of phi* exp(0.5) and exp(-0.05); the mean durations of the primary and secondary temporal niches over the retained cycles; and, for one cycle of the resident pair at phi* started from the total biomass 1 / (D - 1), with D = 100, split between the two residents in the mean resident proportion just returned, under a supply of 0.5 of each resource, the time of the first exhaustion, found directly by the root finder of step 3, and the duration of the secondary niche from the cycle integration of step 4.

```python
def evolutionarily_stable_allocation(alpha: float, n_cycles: int, burn_in: int, seed: int) -> dict:
    """Evolutionarily stable secondary allocation under Dirichlet-fluctuating supply ratios.

    Parameters
    ----------
    alpha : float
        Dirichlet concentration of the supply split.
    n_cycles : int
        Number of serial-dilution cycles; an integer at least 100.
    burn_in : int
        Cycles discarded before averaging; an integer satisfying 0 <= burn_in < n_cycles / 2.
    seed : int
        Seed of the supply draws.

    Returns
    -------
    dict
        Under the keys phi_star, invasion_excess, excess_trait, curvature, convergence_slope,
        resident_share, replay_error, growth_rate, lag_primary, lag_secondary, gradient_below,
        gradient_above, primary_niche, secondary_niche, balanced_primary_niche and
        balanced_secondary_niche.

    Raises
    ------
    ValueError
        When alpha is not finite or is below 0.01, the cycle count, burn-in or seed is not of
        integer type, the cycle count or burn-in is out of range, or the seed is negative.
    """
    return
```
