# Physics-Astrophysics-14

## Background

Cosmological simulations represent matter with particles whose positions, velocities, accelerations and gravitational potentials are known at one instant. A useful halo boundary should identify material confined to the same potential well without imposing spherical symmetry or ignoring the gravity of nearby structures. Removing a locally averaged acceleration places the candidate in a free-falling frame, while a quadratic turn-around contribution guarantees a finite outer escape contour.

A shallow embedded minimum may represent a moving substructure rather than an independent host boundary. The source's graph exploration distinguishes such a feature from a genuine escape contour, while its energetic assignment accounts for both individual particles and coherent substructures.

The finite graph uses particle-count means, strict comparisons and smaller node ID for equal priorities. Include the true saddle but not its deeper basin in the completed well. The candidate seed includes its absorbed contour node, which counts toward an inclusive three-particle threshold. Process only candidates recorded by the host fill, in discovery order; local re-analysis does not extend that list. Each recorded candidate supplies all its members, regardless of their host flags, as the seed for one local re-analysis on the full graph. Its seed means initialize the preliminary field; the lowest seed particle defines the recomputed spatial center. The recomputed local well, not the seed alone, supplies the local energy-pass velocity mean. Only recorded branch members may enter the locally bound subset. Define that subset's rebound surface as all full-graph neighbors outside the subset, even outside the recorded branch. A failed bulk-object decision removes the locally bound subset from the current host flags; a passing decision leaves existing flags unchanged.

The explicitly stated mean populations, graph surface and finite-instance conventions govern their respective operations; they are not claims of verbatim agreement between article and code. Energies are reported in the common input code units, with the additive potential zero at the host minimum.

## Problem

Dark-matter haloes are often assigned by density thresholds or by testing each particle against the self-potential of a provisional group, but both choices can misclassify material near a distorted boundary or an infalling substructure. A potential-based assignment instead asks which particles are confined by the first escape contour connecting a candidate well to a genuinely deeper valley. The inputs below are a finite particle graph with instantaneous cosmological phase-space data, and the output is the fraction of the completed host well that remains bound.

Use the post-August-2025 potential-based particle assignment that treats cosmological confinement and coherent substructure in an accelerated frame. This benchmark uses the printed journal equations for energies but adopts minimum-centered recomputation before graph filling from the paper-linked reference implementation, overriding the seed-mean topology reading of the text. Specifically, seed position, potential and acceleration means define a preliminary field only for selecting the lowest seed particle; recompute the field about that particle, retaining the seed acceleration mean, before descent and filling. Apply the same convention to local re-analysis. The host energy pass uses the mean canonical velocity over all particles in the completed host well. Each bulk-subgroup decision instead uses the mean over host particles flagged bound immediately before that decision. The Scientific background fixes the finite-graph conventions. In short reasoning, identify the potential transformation, below-contour exit test and physical energy rules used. Report only these checkpoints: true saddle ID, completed-well size, stationary saddle energy relative to the host-minimum potential zero, individual host-bound count, each candidate's locally bound subset size and bulk retain-or-reject decision in discovery order, and final count over the completed-well denominator; exhaustive node lists are not requested.

All particles have equal unit mass, positions and canonical velocities have one spatial component, no periodic wrapping is used and arrays are ordered by node ID from 0 through 14; use the supplied velocity array as the source's canonical kinematic variable and set $a=1$, $H=1$, $\Omega_m=1$, $\Omega_\Lambda=0$ and $\widetilde\delta=4.55$. Position is $[10.0,9.7,10.4,9.4,8.5,10.8,11.0,11.1,10.9,11.5,12.5,12.7,13.0,9.1,8.9]$; canonical velocity is $[-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,2.7,2.7,2.7,0.0,-2.0,0.0,0.0,-2.0,0.0]$; acceleration is $[-0.3,-0.5,-0.4,-0.2,-3/5,0.0,-0.1,0.0,0.1,0.0,0.0,0.0,0.0,0.0,0.0]$; raw potential is $[-20.0,-19.417625,-18.758,-18.3305,-14.740625,-15.752,-17.7625,-17.183625,-17.618625,-10.840625,-7.890625,-8.627625,-18.5625,-14.438625,-21.063625]$; seed IDs are $[0,1,2,3,4]$; and the undirected edges are $(0,1),(0,2),(1,3),(3,4),(4,13),(13,14),(2,5),(5,6),(6,7),(8,6),(8,9),(2,10),(10,11),(11,12)$. Your final answer must be a single number: within the pre-energy completed host well, the final flagged mass divided by that same completed-well mass, rounded to six decimal places.

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

01_boosted_turnaround_potential

Goal
----
Compute the boosted turn-around potential for every node in a one-dimensional particle system. Position, acceleration and raw potential must be finite one-dimensional numeric arrays of equal length with at least three entries. Seed IDs must form a nonempty one-dimensional sequence of distinct native or NumPy integers in range. First set the reference position and potential to the particle-count means over the seed IDs and evaluate the displayed boosted expression for each seed. Choose the seed minimizing `$(boosted_value[id], id)$` as the reference center, then evaluate every node relative to that center. With `$q_i = position_i - position_c$` and `$mean_grad = -mean(acceleration[j] for j in seed_ids)$`, return ``raw_potential_i - raw_potential_c - mean_grad * q_i - 0.25 * hubble_rate**2 * omega_m * (scale_factor * q_i)**2 * turnaround_overdensity`` as native Python floats. Preserve a finite representable result even when a scalar power overflows before multiplication by zero or compensating factors. Invalid input or a boosted result outside the finite native-float range raises ValueError.

```python
import numpy as np


def boosted_turnaround_potential(position, acceleration, raw_potential, seed_ids,
                                 scale_factor, hubble_rate, omega_m,
                                 turnaround_overdensity):
    """Return boosted turn-around potential values for all nodes.

    Parameters
    ----------
    position : array_like
        Finite one-dimensional positions with at least three entries.
    acceleration : array_like
        Finite one-dimensional accelerations, one per position.
    raw_potential : array_like
        Finite one-dimensional raw potentials, one per position.
    seed_ids : array_like of int
        Nonempty distinct valid native or NumPy integer node IDs.
    scale_factor : float
        Positive finite cosmological scale factor.
    hubble_rate : float
        Nonnegative finite Hubble rate.
    omega_m : float
        Nonnegative finite matter density parameter.
    turnaround_overdensity : float
        Nonnegative finite turn-around overdensity.

    Returns
    -------
    list of float
        Boosted turn-around potential for every node.

    Raises
    ------
    ValueError
        If an input violates the contract or a boosted result is not representable
        as a finite native float; intermediate overflow alone is not a refusal.
    """
    return None
```

### Step 2

02_descend_to_minimum

Goal
----
Follow strictly decreasing graph links from the lowest-potential seed to a local minimum. The potential input is a finite one-dimensional array of at least three pairwise-distinct real values. Adjacency rows are sorted unique native or NumPy integral IDs and encode a connected simple undirected graph. Seed IDs are a nonempty one-dimensional sequence of valid integral IDs. Select the seed minimizing ``(potential[id], id)``, repeatedly move to the lower-potential neighbor minimizing that pair and return the terminal node as a native Python int. Invalid input raises ValueError.

```python
import numpy as np


def descend_to_minimum(potential, adjacency, seed_ids):
    """Return the local-minimum node reached by graph descent.

    Parameters
    ----------
    potential : array_like
        Finite one-dimensional pairwise-distinct potential values.
    adjacency : sequence of sequence of int
        Sorted unique neighbors of a connected simple undirected graph.
    seed_ids : array_like of int
        Nonempty valid native or NumPy integer seed IDs.

    Returns
    -------
    int
        Native Python integer ID of the terminal local minimum.

    Raises
    ------
    ValueError
        If the potentials, graph or IDs violate the stated contract.
    """
    return None
```

### Step 3

03_locate_host_well

Goal
----
Grow a discrete potential well from a supplied minimum until its first contour reaches a deeper basin. Potential values must be finite, one-dimensional and pairwise distinct. Adjacency rows must be sorted unique native or NumPy integral IDs for a connected simple undirected graph. The start ID must be valid and the subgroup threshold must be a positive integral scalar. Initialize the internal set with the start and its surface with external neighbors. Always select the surface node minimizing ``(potential[id], id)``. Admit it ordinarily when every strictly lower neighbor is internal. Otherwise explore its external connected branch through nodes strictly below its contour. If that branch reaches a node strictly below the starting potential, add only the contour node and return it as the true saddle. If not, the false-saddle candidate is the contour node together with the external nodes reached strictly below its contour. Include the contour node when testing subgroup_min_size. Absorb this inclusive candidate, merge its exterior frontier into the surface and record its sorted IDs when its inclusive size meets the threshold. Return sorted well IDs, a native-int saddle ID and candidate branches in discovery order. Invalid input or surface exhaustion raises ValueError.

```python
import heapq
import numpy as np


def locate_host_well(potential, adjacency, minimum_id, subgroup_min_size):
    """Return the completed well, its true saddle and recorded branches.

    Parameters
    ----------
    potential : array_like
        Finite one-dimensional pairwise-distinct potential values.
    adjacency : sequence of sequence of int
        Sorted unique neighbors of a connected simple undirected graph.
    minimum_id : int
        Valid native or NumPy integer ID of the starting minimum.
    subgroup_min_size : int
        Positive integral threshold on candidate size, including its contour node.

    Returns
    -------
    tuple
        Sorted native-int well IDs, native-int saddle ID and sorted native-int
        candidate branches in discovery order, each including its false saddle.

    Raises
    ------
    ValueError
        If an input violates the contract or no deeper exit is reachable.
    """
    return None
```

### Step 4

04_host_energy_mask

Goal
----
Classify particles in a completed one-dimensional host well with the physical-frame energy rule. Position, canonical velocity and boosted potential are finite one-dimensional arrays of equal length with at least three entries. Well IDs are a nonempty one-dimensional sequence of distinct valid native or NumPy integers, and minimum and saddle IDs are valid integral members of the well. Recenter with `$q_i = position_i - position_minimum$` and `$u_i = canonical_velocity_i - mean(canonical_velocity[j] for j in well_ids)$`. The physical potential is ``Phi_i = boosted_potential_i + 0.25 * hubble_rate**2 * (omega_m * (1.0 + turnaround_overdensity) - 2.0 * omega_lambda) * (scale_factor * q_i)**2``. The energy is ``E_i = Phi_i + 0.5 * (scale_factor * hubble_rate * q_i + u_i / scale_factor)**2`` and the escape energy is `$Phi_saddle$`. Return a zero-one mask that marks only well members satisfying strict `$E_i < E_escape$`, plus all energies and physical potentials as native floats and the native-float escape energy. Preserve finite representable physical potentials and energies despite overflow in an intermediate power or product; invalid input or an unrepresentable returned potential or energy raises ValueError.

```python
import numpy as np


def host_energy_mask(position, canonical_velocity, boosted_potential, well_ids,
                     saddle_id, minimum_id, scale_factor, hubble_rate, omega_m,
                     omega_lambda, turnaround_overdensity):
    """Return host binding flags, escape energy and physical potentials.

    Parameters
    ----------
    position : array_like
        Finite one-dimensional positions.
    canonical_velocity : array_like
        Finite one-dimensional canonical velocities.
    boosted_potential : array_like
        Finite one-dimensional boosted turn-around potentials.
    well_ids : array_like of int
        Nonempty distinct valid native or NumPy integer IDs in the completed well.
    saddle_id : int
        Valid integral saddle ID belonging to the well.
    minimum_id : int
        Valid integral minimum ID belonging to the well.
    scale_factor : float
        Positive finite scale factor.
    hubble_rate : float
        Nonnegative finite Hubble rate.
    omega_m : float
        Nonnegative finite matter density parameter.
    omega_lambda : float
        Nonnegative finite dark-energy density parameter.
    turnaround_overdensity : float
        Nonnegative finite turn-around overdensity.

    Returns
    -------
    tuple
        Native-int zero-one mask, native-float escape energy, native-float
        physical potentials and native-float particle energies.

    Raises
    ------
    ValueError
        If an input violates the contract or a returned potential or energy is not
        representable as a finite native float; avoidable intermediate overflow
        does not invalidate a representable result.
    """
    return None
```

### Step 5

05_rebound_subgroup

Goal
----
Re-analyse a recorded subgroup in its own accelerated frame and return its locally bound members and graph surface. All particle arrays are finite, one-dimensional and equally long with at least three entries. The graph is connected, simple and undirected with sorted unique native or NumPy integral adjacency IDs. Subgroup IDs are sorted, unique, integral and in range, and their count meets the positive integral candidate threshold. Build the local boosted potential using the entire subgroup as the seed, choose the subgroup member minimizing `$(local_potential[id], id)$`, run the full-graph well traversal with the unchanged threshold and apply the host-energy rule to that local well. Return the sorted subgroup IDs whose local mask is one, followed by the sorted union of all full-graph neighbors of those bound members that are not locally bound. Invalid input raises ValueError.

```python
import numpy as np


def rebound_subgroup(position, canonical_velocity, acceleration, raw_potential,
                     adjacency, subgroup_ids, scale_factor, hubble_rate, omega_m,
                     omega_lambda, turnaround_overdensity, candidate_threshold):
    """Return locally rebound candidate members and their full-graph surface.

    Parameters
    ----------
    position : array_like
        Finite one-dimensional positions.
    canonical_velocity : array_like
        Finite one-dimensional canonical velocities.
    acceleration : array_like
        Finite one-dimensional accelerations.
    raw_potential : array_like
        Finite one-dimensional raw potentials.
    adjacency : sequence of sequence of int
        Sorted unique neighbors of a connected simple undirected graph.
    subgroup_ids : array_like of int
        Sorted unique valid candidate IDs meeting ``candidate_threshold``.
    scale_factor : float
        Positive finite scale factor.
    hubble_rate : float
        Nonnegative finite Hubble rate.
    omega_m : float
        Nonnegative finite matter density parameter.
    omega_lambda : float
        Nonnegative finite dark-energy density parameter.
    turnaround_overdensity : float
        Nonnegative finite turn-around overdensity.
    candidate_threshold : int
        Positive native or NumPy integral branch-size threshold.

    Returns
    -------
    tuple of list of int
        Sorted native-int locally bound IDs and sorted native-int surface IDs.

    Raises
    ------
    ValueError
        If an array, graph, ID sequence or scalar violates the contract, or if
        the local traversal has no deeper exit.
    """
    return None
```

### Step 6

06_bulk_binding_mask

Goal
----
Apply the all-or-nothing host binding decision to a locally rebound subgroup. Position and canonical velocity must be finite, nonempty one-dimensional arrays of equal length, host physical potential must be a finite one-dimensional array of that length and the host individual mask must contain exactly zero or one. Locally bound subgroup IDs and surface IDs are distinct one-dimensional native or NumPy integral sequences in range. Copy the input mask. An empty subgroup is a no-op. Otherwise let `$H$` be IDs marked one in the original mask, `$v_host = mean(v_i for i in H)$`, `$q_mean = mean(position_i - position_minimum for i in S)$`, `$u_mean = mean(canonical_velocity_i for i in S) - v_host$` and `$v_bulk = scale_factor * hubble_rate * q_mean + u_mean / scale_factor$`. Form ``E_proxy = max(host_physical_potential[j] for j on the local surface) + 0.5 * v_bulk**2``. Set every locally bound subgroup ID to zero only when strict `$E_proxy > host_escape$`; equality remains bound. Return native Python ints. Preserve a finite representable proxy despite overflow in an intermediate power or product. Invalid input or a proxy outside the finite native-float range raises ValueError.

```python
import numpy as np


def bulk_binding_mask(position, canonical_velocity, host_physical_potential,
                      host_individual_mask, subgroup_bound_ids,
                      subgroup_surface_ids, host_escape, minimum_id,
                      scale_factor, hubble_rate):
    """Return the host mask after one subgroup bulk decision.

    Parameters
    ----------
    position : array_like
        Finite nonempty one-dimensional positions.
    canonical_velocity : array_like
        Finite one-dimensional canonical velocities of the same length.
    host_physical_potential : array_like
        Finite one-dimensional host-frame physical potentials.
    host_individual_mask : array_like of int
        One-dimensional zero-one host binding mask.
    subgroup_bound_ids : array_like of int
        Distinct valid locally bound subgroup IDs.
    subgroup_surface_ids : array_like of int
        Distinct valid graph-surface IDs disjoint from a nonempty subgroup.
    host_escape : float
        Finite host escape energy.
    minimum_id : int
        Valid native or NumPy integral host minimum ID.
    scale_factor : float
        Positive finite scale factor.
    hubble_rate : float
        Nonnegative finite Hubble rate.

    Returns
    -------
    list of int
        Copied zero-one host mask after the strict bulk rejection decision.

    Raises
    ------
    ValueError
        If an input violates the contract or a nonempty subgroup's proxy is not
        representable as a finite native float; preserve representable proxies.
    """
    return None
```

### Step 7

07_bound_mass_fraction

Goal
----
Compose accelerated-frame topology, individual binding, subgroup rebound and bulk rejection into one bound mass fraction. All particle arrays are finite one-dimensional sequences of equal length with at least three entries. The supplied adjacency is a connected simple undirected graph with sorted unique native or NumPy integral IDs, seed IDs are nonempty distinct valid integral IDs and the subgroup size threshold is a positive integral scalar. Build the host boosted potential, descend from the seeds, locate the completed well and true saddle, apply strict individual binding, then process recorded candidate branches in discovery order by local rebinding and all-or-nothing host bulk tests. With unit particle masses, return the number of final mask entries equal to one divided by the completed well size before energetic unbinding. Invalid input or an upstream potential or energy outside the finite native-float range raises ValueError. Representable upstream results remain valid even when a naive intermediate power would overflow.

```python
import numpy as np


def bound_mass_fraction(position, canonical_velocity, acceleration,
                        raw_potential, adjacency, seed_ids, scale_factor,
                        hubble_rate, omega_m, omega_lambda,
                        turnaround_overdensity, subgroup_min_size):
    """Return the final unit-mass bound fraction of the completed host well.

    Parameters
    ----------
    position : array_like
        Finite one-dimensional positions.
    canonical_velocity : array_like
        Finite one-dimensional canonical velocities.
    acceleration : array_like
        Finite one-dimensional accelerations.
    raw_potential : array_like
        Finite one-dimensional raw potentials.
    adjacency : sequence of sequence of int
        Sorted unique neighbors of a connected simple undirected graph.
    seed_ids : array_like of int
        Nonempty distinct valid native or NumPy integer seed IDs.
    scale_factor : float
        Positive finite scale factor.
    hubble_rate : float
        Nonnegative finite Hubble rate.
    omega_m : float
        Nonnegative finite matter density parameter.
    omega_lambda : float
        Nonnegative finite dark-energy density parameter.
    turnaround_overdensity : float
        Nonnegative finite turn-around overdensity.
    subgroup_min_size : int
        Positive native or NumPy integral candidate-recording threshold.

    Returns
    -------
    float
        Final unit-mass bound count divided by the pre-unbinding well size.

    Raises
    ------
    ValueError
        If an upstream input or derived-value representability contract fails,
        or no true saddle is found. Upstream ValueError propagates unchanged.
    """
    return None
```
