# Physics-Computational_Physics-24

## Background

A reactor at very low power is not well described by the populations it holds on average, because those populations are small enough that the discreteness of individual fissions, captures and decays is visible in the behaviour of the system. Start-up and approach-to-critical studies therefore have to be posed stochastically, and two families of description compete. One writes down the exact continuous-time Markov process on integer populations, in which every neutron event is a jump, and either solves probability balances for it or simulates it by analog Monte Carlo. The other keeps only its diffusive limit, a small system of stochastic differential equations whose drift is the familiar point-kinetic system and whose noise is a Brownian term whose amplitude is read off from the variance of the underlying jumps. The second is far cheaper and is what almost all published low-population studies use; the price is that a diffusive limit is an approximation whose fidelity has to be checked against the process it approximates, and the check is usually made only on the neutron population.

Circulating-fuel reactors, in which the fuel itself is the coolant and flows around a loop, complicate this considerably. Delayed-neutron precursors are born in the core but are swept out of it before many of them decay, so a fraction of the delayed neutrons is simply lost to the chain, and the reactor needs a positive static reactivity merely to hold criticality. The classical way of writing this introduces a retarded term representing plug flow around the loop, which destroys the Markov property and so blocks any equivalent jump process outright. Replacing the loop by a second perfectly mixed volume removes the memory at the cost of doubling the number of precursor equations, and it is what makes a stochastic treatment of a circulating-fuel system possible at all. The resulting state carries a neutron population together with an in-core and an ex-core inventory for each delayed group, coupled through the two mean residence times, with only in-core decays feeding neutrons back into the chain.

What this exposes is a structural asymmetry between the two descriptions. The exact jump process gives every precursor population its own noise: precursors are created in discrete bursts by fission, destroyed one at a time by decay, and shuttled between the regions one at a time by the flow, and several of those events move two components of the state at once and so correlate them. The diffusive description, as it is conventionally derived, retains only a single scalar Brownian term on the neutron equation and leaves the precursor equations deterministic, on the argument that the precursor noise disappears in the diffusive scaling. Whether that omission is harmless is a quantitative question about a particular operating point. In this affine stationary model, changing the source strength rescales the mean and both covariance matrices, so it changes relative population fluctuations but not the dispersion ratio between the two descriptions. The discrepancy instead grows as the system is taken deeper subcritical and as both residence times are increased: slower circulation changes the core/ex-core partition and the transfer-event rates while weakening the drift-mediated propagation of neutron noise into the precursor variables.

Both descriptions happen to be tractable without any sampling. Every event rate of the jump process is affine in the state, and the drift of the diffusive system is linear, so in both cases the equations for the second moments close exactly on the stationary mean and the stationary covariance is fixed by a single linear matrix equation. The discrepancy between the two can therefore be computed rather than estimated, to whatever precision the arithmetic allows, and the comparison becomes a deterministic question about two matrices instead of a Monte Carlo study.

## Problem

A zero-power circulating-fuel reactor is idealised as two perfectly mixed volumes, an active core and an ex-core loop, joined so that the outlet of each is the inlet of the other. Delayed-neutron precursors are carried between the two with constant mean residence times $\tau_c = 7.5$ s in the core and $\tau_e = 12.5$ s in the loop, they decay wherever they happen to be, and a decay contributes a neutron to the chain only when it occurs inside the core. The fuel has a prompt neutron generation time $\Lambda = 1.0\times10^{-3}$ s and a total delayed-neutron fraction $\beta = 0.0065$ distributed over six groups,

| group $j$ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| $\beta_j/\beta$ (-) | 0.033 | 0.219 | 0.196 | 0.395 | 0.115 | 0.042 |
| $\lambda_j$ (s$^{-1}$) | 0.0124 | 0.0305 | 0.111 | 0.301 | 1.14 | 3.01 |

and the number of prompt neutrons released by a fission takes the values $0, 1, 2, 3, 4, 5$ with probabilities $0.027$, $0.158$, $0.339$, $0.305$, $0.133$, $0.038$ respectively. For each fission event, take the total delayed-neutron precursor yield $\nu_d$ to be independent of the tabulated prompt multiplicity $\nu_p$ and, conditional on $\nu_d$, assign each precursor independently to group $j$ with probability $\beta_j/\beta$. An external source emits $S = 8800$ neutrons per second into the core, and the control state holds the reactor $2500$ pcm below the reactivity at which this circulating configuration is exactly critical.

I want to compare two stochastic descriptions of this same stationary system. The first is the exact continuous-time Markov jump process, in which each capture-or-leakage, each fission, each source emission, each precursor decay and each transfer of a precursor between the two regions is an elementary event carrying its own rate. The second is the Itô stochastic point-kinetic description, in which the precursor equations carry no stochastic forcing at all and the whole of the noise is a single scalar Brownian term on the neutron balance. Your final answer must be a single number, the dimensionless dispersion ratio

$$R = \frac{\sigma_{\mathrm{jump}}(C_e)}{\sigma_{\mathrm{diff}}(C_e)}, \qquad C_e = \sum_j C_{e,j},$$

in which $C_e$ is the total ex-core precursor inventory and $\sigma$ denotes its stationary standard deviation under each description, quoted to at least six significant figures and obtained exactly rather than by sampling trajectories. State, in one line each, the balance that locates the critical reactivity of the circulating configuration, the rule that fixes the per-neutron fission and capture-and-leakage rates from the point-kinetic constants, the amplitude of the Brownian term of the diffusive description, and the relation from which each stationary covariance follows; then report the reactivity that circulation removes in pcm, the stationary neutron population, the margin by which the source rate exceeds the threshold that keeps the diffusive sample paths strictly positive, the same dispersion ratio evaluated for the neutron population and for the total in-core precursor inventory, the values $R$ takes at $500$, $1000$, $5000$ and $10000$ pcm subcriticality with everything else held fixed, the values it takes when both residence times are multiplied by $2$, $4$ and $10$, recomputing the circulating critical reactivity for each scaled pair and placing each scaled configuration $2500$ pcm below its own critical point while holding all other inputs fixed, and what those two trends say about the regime in which the diffusive description can be trusted. Those are the short report this task wants: they are the few scalars that determine the final number and the few that establish it was computed rather than estimated, and nothing beyond them is required.

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

01_evaluate_multiplicity_moments

Goal
----
Reduce a tabulated prompt-fission multiplicity distribution to the two moments that the point-kinetic and stochastic descriptions of the neutron balance both need.

```python
import numpy as np


def evaluate_multiplicity_moments(multiplicity: np.ndarray,
                                  abundance: np.ndarray) -> np.ndarray:
    """Reduce a prompt-fission multiplicity table to its two required moments.

    Parameters
    ----------
    multiplicity : np.ndarray
        One-dimensional array of the distinct prompt-neutron multiplicities of a
        fission event, each a non-negative integer value.
    abundance : np.ndarray
        One-dimensional array of the same length holding the probability of each
        multiplicity; the entries are non-negative and sum to one.

    Returns
    -------
    moments : np.ndarray
        Array of shape (2,), holding in order the mean prompt multiplicity and
        the mean square of the net prompt-neutron gain of a fission, both
        dimensionless.

    Raises
    ------
    ValueError
        If either input is not a one-dimensional, non-empty array of finite
        entries, if the two arrays differ in length, if any ``multiplicity``
        entry is negative or is not an integer value, if any ``abundance`` entry
        is negative, or if the ``abundance`` entries do not sum to one within
        1e-10.
    """
    return moments  # placeholder
```

### Step 2

02_compute_neutron_event_rates

Goal
----
Convert the point-kinetic constants and the fission multiplicity moments into the per-neutron event rates of the equivalent stochastic process and into the amplitude of the neutron noise.

```python
import numpy as np


def compute_neutron_event_rates(reactivity: float, delayed_fraction: float,
                                generation_time: float, multiplicity_moments: np.ndarray
                                ) -> np.ndarray:
    """Return the per-neutron event rates and the neutron noise amplitude.

    Parameters
    ----------
    reactivity : float
        Dimensionless reactivity of the configuration; may be negative.
    delayed_fraction : float
        Total delayed-neutron fraction, strictly between zero and one.
    generation_time : float
        Prompt neutron generation time in seconds (generation_time > 0).
    multiplicity_moments : np.ndarray
        Array of shape (2,) holding the mean prompt fission multiplicity and the
        mean square net prompt gain of a fission, as returned by sub-problem 01.
        The mean multiplicity must be strictly positive.

    Returns
    -------
    rates : np.ndarray
        Array of shape (4,) holding, in order, the per-neutron fission rate in
        inverse seconds, the per-neutron capture-and-leakage rate in inverse
        seconds, the dimensionless mean delayed-neutron yield of a fission, and
        the scalar coefficient in inverse square-root seconds that multiplies
        the square root of the neutron population in the diffusion term of the
        neutron equation.

    Raises
    ------
    ValueError
        If ``reactivity`` is not a finite number; if ``delayed_fraction`` is not
        a finite number strictly between zero and one; if ``generation_time`` is
        not a finite number greater than zero; if ``multiplicity_moments`` is not
        a finite array of shape (2,) whose first entry is strictly positive or
        whose second entry is negative; or if the resulting capture-and-leakage
        rate is negative, which places the configuration above prompt critical
        and leaves the stochastic process without a stationary state.
    """
    return rates  # placeholder
```

### Step 3

03_build_kinetics_matrix

Goal
----
Assemble the drift matrix of the two-region perfectly-mixed point-kinetic model, whose state carries the neutron population together with one in-core and one ex-core precursor population per delayed group.

```python
import numpy as np


def build_kinetics_matrix(reactivity: float, delayed_fraction: float,
                          group_fractions: np.ndarray, decay_constants: np.ndarray,
                          generation_time: float, tau_core: float,
                          tau_excore: float) -> np.ndarray:
    """Assemble the drift matrix of the two-region point-kinetic model.

    The state vector is ordered with the neutron population first, then the
    in-core precursor population of every delayed group in the order supplied,
    then the ex-core precursor population of every delayed group in the same
    order.

    Parameters
    ----------
    reactivity : float
        Dimensionless reactivity of the configuration; may be negative.
    delayed_fraction : float
        Total delayed-neutron fraction, strictly between zero and one.
    group_fractions : np.ndarray
        One-dimensional array of the delayed-group shares of the total delayed
        fraction; entries are strictly positive and sum to one.
    decay_constants : np.ndarray
        One-dimensional array of the same length holding the decay constant of
        each delayed group in inverse seconds; entries are strictly positive.
    generation_time : float
        Prompt neutron generation time in seconds (generation_time > 0).
    tau_core : float
        Mean residence time of a precursor in the core in seconds
        (tau_core > 0).
    tau_excore : float
        Mean residence time of a precursor in the ex-core loop in seconds
        (tau_excore > 0).

    Returns
    -------
    kinetics_matrix : np.ndarray
        Array of shape (1 + 2 * n_groups, 1 + 2 * n_groups) in inverse seconds,
        the drift matrix acting on the state vector described above.

    Raises
    ------
    ValueError
        If ``reactivity``, ``generation_time``, ``tau_core`` or ``tau_excore``
        is not a finite number, if ``generation_time``, ``tau_core`` or
        ``tau_excore`` is not greater than zero, if ``delayed_fraction`` is not
        a finite number strictly between zero and one, if ``group_fractions``
        and ``decay_constants`` are not non-empty one-dimensional finite arrays
        of equal length, if any entry of either is not strictly positive, or if
        the ``group_fractions`` entries do not sum to one within 1e-10.
    """
    return kinetics_matrix  # placeholder
```

### Step 4

04_compute_drift_reactivity_loss

Goal
----
Determine the reactivity that circulation of the fuel removes from the two-region system, which is the reactivity at which the circulating configuration is exactly critical.

```python
import numpy as np


def compute_drift_reactivity_loss(kinetics_matrix: np.ndarray, delayed_fraction: float,
                                  decay_constants: np.ndarray,
                                  generation_time: float) -> float:
    """Return the reactivity removed by the circulation of the fuel.

    Parameters
    ----------
    kinetics_matrix : np.ndarray
        Square drift matrix of shape (1 + 2 * n_groups, 1 + 2 * n_groups) with
        the state ordering of sub-problem 03. Its neutron row and column may
        have been assembled at any reactivity.
    delayed_fraction : float
        Total delayed-neutron fraction, strictly between zero and one.
    decay_constants : np.ndarray
        One-dimensional array of the decay constants of the delayed groups in
        inverse seconds, in the same group order used to assemble the matrix.
    generation_time : float
        Prompt neutron generation time in seconds (generation_time > 0).

    Returns
    -------
    reactivity_loss : float
        Dimensionless reactivity removed by circulation, as a native Python
        float.

    Raises
    ------
    ValueError
        If ``kinetics_matrix`` is not a finite two-dimensional square array
        whose size is one more than twice the number of delay groups, if
        ``decay_constants`` is not a non-empty one-dimensional array of finite
        strictly positive entries, if ``delayed_fraction`` is not a finite
        number strictly between zero and one, if ``generation_time`` is not a
        finite number greater than zero, if the precursor block of the matrix is
        singular, or if the resulting per-neutron precursor inventory is not
        strictly positive in every group and region.
    """
    return reactivity_loss  # placeholder
```

### Step 5

05_solve_steady_state_populations

Goal
----
Solve the source-driven stationary state of the two-region point-kinetic model for the neutron population and for the in-core and ex-core precursor inventory of every delayed group.

```python
import numpy as np


def solve_steady_state_populations(kinetics_matrix: np.ndarray,
                                   source_rate: float) -> np.ndarray:
    """Solve for the source-driven stationary populations of the two-region model.

    Parameters
    ----------
    kinetics_matrix : np.ndarray
        Square drift matrix of shape (1 + 2 * n_groups, 1 + 2 * n_groups) with
        the state ordering of sub-problem 03.
    source_rate : float
        Rate at which the external source emits neutrons into the core, in
        neutrons per second (source_rate > 0).

    Returns
    -------
    state : np.ndarray
        Array of shape (1 + 2 * n_groups,) holding the stationary neutron
        population followed by the in-core and then the ex-core precursor
        populations, all dimensionless counts.

    Raises
    ------
    ValueError
        If ``kinetics_matrix`` is not a finite two-dimensional square array of
        odd size at least three, if ``source_rate`` is not a finite number
        greater than zero, if any eigenvalue of ``kinetics_matrix`` has a
        non-negative real part so that no stationary state exists, or if the
        resulting state is not strictly positive in every component.
    """
    return state  # placeholder
```

### Step 6

06_build_jump_diffusion_matrix

Goal
----
Assemble the diffusion matrix of the exact continuous-time Markov jump process at a given state, accumulating over every elementary event the product of its rate with the outer product of its state increment.

```python
import numpy as np


def build_jump_diffusion_matrix(state: np.ndarray, event_rates: np.ndarray,
                                group_fractions: np.ndarray, decay_constants: np.ndarray,
                                multiplicity_moments: np.ndarray, tau_core: float,
                                tau_excore: float, source_rate: float) -> np.ndarray:
    """Assemble the diffusion matrix of the exact jump process at a given state.

    Parameters
    ----------
    state : np.ndarray
        Array of shape (1 + 2 * n_groups,) holding the neutron population, the
        in-core precursor populations and the ex-core precursor populations in
        the ordering of sub-problem 03; every entry is non-negative.
    event_rates : np.ndarray
        Array of shape (4,) as returned by sub-problem 02: the per-neutron
        fission rate, the per-neutron capture-and-leakage rate, the mean delayed
        yield of a fission and the neutron noise coefficient. The last entry is
        not used here.
    group_fractions : np.ndarray
        One-dimensional array of the delayed-group shares of the total delayed
        fraction, summing to one.
    decay_constants : np.ndarray
        One-dimensional array of the decay constants of the delayed groups in
        inverse seconds, in the same group order.
    multiplicity_moments : np.ndarray
        Array of shape (2,) holding the mean prompt fission multiplicity and the
        mean square net prompt gain of a fission, as returned by sub-problem 01.
    tau_core : float
        Mean residence time of a precursor in the core in seconds
        (tau_core > 0).
    tau_excore : float
        Mean residence time of a precursor in the ex-core loop in seconds
        (tau_excore > 0).
    source_rate : float
        Rate at which the external source emits neutrons into the core, in
        neutrons per second (source_rate > 0).

    Returns
    -------
    diffusion_matrix : np.ndarray
        Symmetric positive semi-definite array of shape
        (1 + 2 * n_groups, 1 + 2 * n_groups) in inverse seconds, the
        instantaneous covariance rate of the jump process at the supplied state.

    Raises
    ------
    ValueError
        If ``state`` is not a one-dimensional finite array of non-negative
        entries whose length is one more than twice the number of delay groups,
        if ``event_rates`` is not a finite array of shape (4,) whose first three
        entries are non-negative, if ``multiplicity_moments`` is not a finite
        array of shape (2,), if ``group_fractions`` and ``decay_constants`` are
        not non-empty one-dimensional finite arrays of equal length with
        strictly positive entries, if the ``group_fractions`` entries do not sum
        to one within 1e-10, or if ``tau_core``, ``tau_excore`` or
        ``source_rate`` is not a finite number greater than zero.
    """
    return diffusion_matrix  # placeholder
```

### Step 7

07_build_diffusive_noise_matrix

Goal
----
Assemble the diffusion matrix of the Itô stochastic point-kinetic description, in which the whole stochastic forcing is a single scalar Brownian term attached to the neutron balance.

```python
import numpy as np


def build_diffusive_noise_matrix(state: np.ndarray,
                                 noise_coefficient: float) -> np.ndarray:
    """Assemble the diffusion matrix of the Itô point-kinetic description.

    Parameters
    ----------
    state : np.ndarray
        Array of shape (1 + 2 * n_groups,) holding the neutron population, the
        in-core precursor populations and the ex-core precursor populations in
        the ordering of sub-problem 03; every entry is non-negative and the
        length is odd and at least three.
    noise_coefficient : float
        The scalar coefficient in inverse square-root seconds that multiplies
        the square root of the neutron population in the diffusion term of the
        neutron equation, the fourth entry returned by sub-problem 02
        (noise_coefficient >= 0).

    Returns
    -------
    diffusion_matrix : np.ndarray
        Symmetric positive semi-definite array of shape
        (1 + 2 * n_groups, 1 + 2 * n_groups) in inverse seconds, the
        instantaneous covariance rate of the diffusive description at the
        supplied state.

    Raises
    ------
    ValueError
        If ``state`` is not a one-dimensional finite array of non-negative
        entries whose length is odd and at least three, or if
        ``noise_coefficient`` is not a finite non-negative number.
    """
    return diffusion_matrix  # placeholder
```

### Step 8

08_solve_stationary_covariance

Goal
----
Obtain the stationary covariance matrix of a linear stochastic system from its drift matrix and its diffusion matrix, without sampling any trajectory.

```python
import numpy as np


def solve_stationary_covariance(drift_matrix: np.ndarray,
                                diffusion_matrix: np.ndarray) -> np.ndarray:
    """Solve for the stationary covariance of a linear stochastic system.

    Parameters
    ----------
    drift_matrix : np.ndarray
        Square array of shape (n, n) in inverse seconds; the linear part of the
        deterministic drift.
    diffusion_matrix : np.ndarray
        Symmetric positive semi-definite array of the same shape in inverse
        seconds; the instantaneous covariance rate of the stochastic forcing.

    Returns
    -------
    covariance : np.ndarray
        Symmetric array of shape (n, n) holding the stationary covariance of the
        state, in squared counts.

    Raises
    ------
    ValueError
        If either input is not a finite two-dimensional square array, if the two
        do not have the same shape, if ``diffusion_matrix`` is not symmetric
        within 1e-9 of its largest absolute entry, if ``diffusion_matrix`` has an
        eigenvalue below -1e-9 times its largest absolute eigenvalue and so is
        not positive semi-definite, if any eigenvalue of ``drift_matrix`` has a
        non-negative real part, or if the resulting covariance fails the same
        positive semi-definiteness test.
    """
    return covariance  # placeholder
```

### Step 9

09_compute_inventory_dispersion_ratio

Goal
----
Reduce two stationary covariance matrices to the ratio of the standard deviations they predict for one weighted sum of the state components.

```python
import numpy as np


def compute_inventory_dispersion_ratio(reference_covariance: np.ndarray,
                                       comparison_covariance: np.ndarray,
                                       weights: np.ndarray) -> float:
    """Return the ratio of the standard deviations two covariances give an aggregate.

    Parameters
    ----------
    reference_covariance : np.ndarray
        Symmetric array of shape (n, n) whose standard deviation is the
        numerator of the ratio.
    comparison_covariance : np.ndarray
        Symmetric array of the same shape whose standard deviation is the
        denominator of the ratio.
    weights : np.ndarray
        One-dimensional array of length n giving the coefficients of the state
        components in the aggregate; at least one entry is non-zero.

    Returns
    -------
    dispersion_ratio : float
        The reference standard deviation of the aggregate divided by the
        comparison standard deviation of the same aggregate, dimensionless, as a
        native Python float.

    Raises
    ------
    ValueError
        If either covariance is not a finite two-dimensional square array, if
        the two do not have the same shape, if either is not symmetric within
        1e-9 of its largest absolute entry, if ``weights`` is not a
        one-dimensional finite array of matching length with at least one
        non-zero entry, if either quadratic form is negative, or if the
        comparison quadratic form is zero so that the ratio is undefined.
    """
    return dispersion_ratio  # placeholder
```

### Step 10

10_run_precursor_dispersion_pipeline

Goal
----
Chain the sub-problem functions 01-09 end to end on the circulating-fuel testbed and return the ratio of the stationary standard deviations that the exact jump process and the diffusive description predict for the selected inventory.

```python
import numpy as np


def run_precursor_dispersion_pipeline(subcriticality: float = 0.025,
                                      tau_core: float = 7.5,
                                      tau_excore: float = 12.5,
                                      source_rate: float = 8800.0,
                                      generation_time: float = 1.0e-3,
                                      delayed_fraction: float = 0.0065,
                                      region: str = "excore") -> float:
    """Run the full dispersion comparison on the circulating-fuel testbed.

    The delayed-group shares are ``(0.033, 0.219, 0.196, 0.395, 0.115, 0.042)``
    with decay constants ``(0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01)`` inverse
    seconds, and the prompt fission multiplicity takes the values zero to five
    with abundances ``(0.027, 0.158, 0.339, 0.305, 0.133, 0.038)``.

    Parameters
    ----------
    subcriticality : float
        Dimensionless distance below the reactivity at which the circulating
        configuration is critical (subcriticality > 0).
    tau_core : float
        Mean residence time of a precursor in the core in seconds
        (tau_core > 0).
    tau_excore : float
        Mean residence time of a precursor in the ex-core loop in seconds
        (tau_excore > 0).
    source_rate : float
        Rate at which the external source emits neutrons into the core, in
        neutrons per second (source_rate > 0).
    generation_time : float
        Prompt neutron generation time in seconds (generation_time > 0).
    delayed_fraction : float
        Total delayed-neutron fraction, strictly between zero and one.
    region : str
        Which aggregate the dispersion is reported for: ``"excore"`` for the
        total ex-core precursor inventory, ``"core"`` for the total in-core
        precursor inventory, or ``"neutron"`` for the neutron population.

    Returns
    -------
    dispersion_ratio : float
        The stationary standard deviation the exact jump process predicts for
        the selected aggregate, divided by the standard deviation the diffusive
        description predicts for it, dimensionless, as a native Python float.

    Raises
    ------
    ValueError
        If ``subcriticality``, ``tau_core``, ``tau_excore``, ``source_rate`` or
        ``generation_time`` is not a finite number greater than zero, if
        ``delayed_fraction`` is not a finite number strictly between zero and
        one, or if ``region`` is not one of ``"excore"``, ``"core"`` or
        ``"neutron"``.
    """
    return dispersion_ratio  # placeholder
```
