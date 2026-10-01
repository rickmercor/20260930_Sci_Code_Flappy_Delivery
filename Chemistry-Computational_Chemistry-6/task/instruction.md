# Chemistry-Computational_Chemistry-6

## Problem

Macroscopic hydrogen transport codes couple two materials across an interface by imposing continuity of chemical potential, a purely algebraic constraint that gives the interface no timescale, no resistance, and a carrier species fixed before the simulation is run; replacing it with reversible reaction channels obeying mass action removes all three assumptions, thermodynamic consistency fixing each channel's ratio of rate constants from the same solubility data so that only the magnitude is a new input, and local equilibrium reappearing as the fast-kinetics limit of a single channel. Consider a one-dimensional two-slab system in nondimensional units, slab A of thickness 0.37 and diffusivity 0.57 joined at x = 0.37 to slab B of thickness 0.53 and diffusivity 0.14, with concentrations held at 1.83 on the outer face of A and 0.71 on the outer face of B, and with no volumetric source anywhere. Three interface chemistries are posed on that same geometry: first, a single first-order exchange channel carrying an atom across without changing its chemical identity, with exchange velocity 0.31 and with the dissolved species having solubility 0.83 in A and 2.41 in B. Second, a channel in which the atomic species of A recombines into a diatomic molecule that dissolves physically in B, with forward rate constant 1.29, dissociative solubility constant 1.07 on the A side and molecular solubility constant 0.89 on the B side, the reverse constant following from thermodynamic consistency; and third, that recombination channel together with a competing first-order channel that oxidises the atom to a second, more soluble carrier, of forward rate constant 0.29 and reverse constant half its forward value, in a medium of lumped oxidising activity 1.97; the second carrier diffuses at a quarter of the molecular carrier's diffusivity, and both carriers are removed at the outer face of B. Finally, admit a second, heavier isotope into slab A, holding the lighter isotope at the interfacial concentration that two-channel solution returned and placing the heavier one at a quarter of it, so that no new bulk problem is solved on the metal side. Let recombination populate all three isotopologues from the pair of constants used for the light homonuclear channel, the mixed channel's forward constant carrying the statistical degeneracy the source assigns it and its reverse constant left equal to the like-atom one, let the oxidation channel act on the heavier isotope with the same forward and reverse constants and the same activity, and give every molecular isotopologue the molecular carrier's diffusivity and every fluoride the fluoride carrier's. Each of those carriers is removed at the outer face of B, which is not the interface: every channel here stays reversible and each carrier sits at the interface at whatever concentration its own transport leaves it. Report the single indicator the source defines for how badly a local-equilibrium interface law describes the two-channel interface of the second chemistry, the one whose throughput is split between the recombination and oxidation channels; both of the indicator's coordinates belong to that interface. State the conventions you adopted at each point where the framework leaves a choice open, justifying each from the source literature, and in your reasoning also report six quantities from the same system: the interfacial concentration on the B side of the first-order problem, the donor-side interfacial concentration of the two-channel solution, the dimensionless group of the second-order channel on the donor side, the branching ratio the source's map of interface behaviour is indexed by, formed from the same two net channel rate laws as the two-channel solve above, each still limited by its own carrier's transport, but evaluated at the loading held on the outer face of A rather than at the loading that solve returns, the dimensionless group of the first-order channel that decides for this asymmetric pair of slabs, and the total atomic budget of the heavier isotope leaving slab A.

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
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
```

## Your task

Implement **all 14 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

slab_resistances

Goal
----
Return the bulk transport resistance contributed by each of the two slabs in a steady one-dimensional permeation problem, in the form the source's interface model adds them in. The two slabs measure concentration on scales that differ by the partition, and the returned pair must already account for that. The source fixes a convention here that the natural reading does not; follow the source.

```python
def slab_resistances(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float) -> "np.ndarray":
    """Return the bulk transport resistance contributed by each of the two slabs in a steady one-dimensional permeation problem, in the form the source's interface model adds them in. The two slabs measure concentration on scales that differ by the partition, and the returned pair must already account for that. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    length_a : float
        Thickness of slab A. Positive.
    diffusivity_a : float
        Diffusivity of the mobile species in slab A. Positive.
    length_b : float
        Thickness of slab B. Positive.
    diffusivity_b : float
        Diffusivity of the mobile species in slab B. Positive.
    partition : float
        Ratio of the forward to the reverse rate constant of the first-order channel. Positive.

    Returns
    -------
    ndarray of shape (2,): the resistance of slab A, then the resistance of slab B.

    Raises
    ------
    ValueError: if either slab length is not positive, either diffusivity is not positive, or partition is not positive.
    """
    return result
```

### Step 2

first_order_rate_ratio

Goal
----
Return the ratio of the forward to the reverse rate constant of a first-order exchange channel that moves an atom across an interface without changing its chemical identity, given the solubility of that atom in each of the two materials. Thermodynamic consistency fixes this ratio completely. The source fixes a convention here that the natural reading does not; follow the source.

```python
def first_order_rate_ratio(solubility_a: float, solubility_b: float) -> float:
    """Return the ratio of the forward to the reverse rate constant of a first-order exchange channel that moves an atom across an interface without changing its chemical identity, given the solubility of that atom in each of the two materials. Thermodynamic consistency fixes this ratio completely. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    solubility_a : float
        Solubility of the dissolved species in slab A. Positive.
    solubility_b : float
        Solubility of the dissolved species in slab B. Positive.

    Returns
    -------
    float, the ratio of the forward rate constant to the reverse one.

    Raises
    ------
    ValueError: if either solubility is not positive.
    """
    return result
```

### Step 3

linear_channel_flux

Goal
----
Return the steady flux crossing a two-slab system whose shared interface carries a single first-order exchange channel, with the concentration held fixed on each outer face. The interface contributes a resistance of its own, set by the exchange velocity. The source fixes a convention here that the natural reading does not; follow the source.

```python
def linear_channel_flux(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float, exchange_velocity: float, c_outer_a: float, c_outer_b: float) -> float:
    """Return the steady flux crossing a two-slab system whose shared interface carries a single first-order exchange channel, with the concentration held fixed on each outer face. The interface contributes a resistance of its own, set by the exchange velocity. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    length_a : float
        Thickness of slab A. Positive.
    diffusivity_a : float
        Diffusivity of the mobile species in slab A. Positive.
    length_b : float
        Thickness of slab B. Positive.
    diffusivity_b : float
        Diffusivity of the mobile species in slab B. Positive.
    partition : float
        Ratio of the forward to the reverse rate constant of the first-order channel. Positive.
    exchange_velocity : float
        Forward rate constant of the first-order channel. Positive.
    c_outer_a : float
        Concentration held on the outer face of slab A.
    c_outer_b : float
        Concentration held on the outer face of slab B.

    Returns
    -------
    float, the steady flux crossing the system.

    Raises
    ------
    ValueError: if exchange_velocity is not positive, or any value it forwards to an earlier step is invalid.
    """
    return result
```

### Step 4

interfacial_traces

Goal
----
Return the two concentrations the solution takes at the shared interface of the first-order two-slab problem, each reported in the units of the side it sits on. Each profile is linear, so each trace follows from the outer value and the flux the slab carries. The source fixes a convention here that the natural reading does not; follow the source.

```python
def interfacial_traces(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float, exchange_velocity: float, c_outer_a: float, c_outer_b: float) -> "np.ndarray":
    """Return the two concentrations the solution takes at the shared interface of the first-order two-slab problem, each reported in the units of the side it sits on. Each profile is linear, so each trace follows from the outer value and the flux the slab carries. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    length_a : float
        Thickness of slab A. Positive.
    diffusivity_a : float
        Diffusivity of the mobile species in slab A. Positive.
    length_b : float
        Thickness of slab B. Positive.
    diffusivity_b : float
        Diffusivity of the mobile species in slab B. Positive.
    partition : float
        Ratio of the forward to the reverse rate constant of the first-order channel. Positive.
    exchange_velocity : float
        Forward rate constant of the first-order channel. Positive.
    c_outer_a : float
        Concentration held on the outer face of slab A.
    c_outer_b : float
        Concentration held on the outer face of slab B.

    Returns
    -------
    ndarray of shape (2,): the trace on side A, then the trace on side B.

    Raises
    ------
    ValueError: if exchange_velocity is not positive, or any value it forwards to an earlier step is invalid.
    """
    return result
```

### Step 5

two_sided_damkohler

Goal
----
Return the dimensionless group that decides whether a first-order interface may be treated as locally equilibrated, together with the relative error that treatment makes on the flux. The group compares bulk transport against interfacial exchange. The source fixes a convention here that the natural reading does not; follow the source.

```python
def two_sided_damkohler(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float, exchange_velocity: float) -> "np.ndarray":
    """Return the dimensionless group that decides whether a first-order interface may be treated as locally equilibrated, together with the relative error that treatment makes on the flux. The group compares bulk transport against interfacial exchange. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    length_a : float
        Thickness of slab A. Positive.
    diffusivity_a : float
        Diffusivity of the mobile species in slab A. Positive.
    length_b : float
        Thickness of slab B. Positive.
    diffusivity_b : float
        Diffusivity of the mobile species in slab B. Positive.
    partition : float
        Ratio of the forward to the reverse rate constant of the first-order channel. Positive.
    exchange_velocity : float
        Forward rate constant of the first-order channel. Positive.

    Returns
    -------
    ndarray of shape (2,): the dimensionless group, then the relative error on the flux.

    Raises
    ------
    ValueError: if exchange_velocity is not positive, or any value it forwards to an earlier step is invalid.
    """
    return result
```

### Step 6

recombination_rate_ratio

Goal
----
Return the ratio of the forward to the reverse rate constant of a channel in which two atoms dissolved in the first material combine into one molecule that dissolves physically in the second. The two solubility constants describe dissociative dissolution on one side and molecular dissolution on the other. The source fixes a convention here that the natural reading does not; follow the source.

```python
def recombination_rate_ratio(sieverts_constant: float, henry_constant: float) -> float:
    """Return the ratio of the forward to the reverse rate constant of a channel in which two atoms dissolved in the first material combine into one molecule that dissolves physically in the second. The two solubility constants describe dissociative dissolution on one side and molecular dissolution on the other. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    sieverts_constant : float
        Dissociative solubility constant on the donor side. Positive.
    henry_constant : float
        Molecular solubility constant on the receiving side. Positive.

    Returns
    -------
    float, the ratio of the forward rate constant to the reverse one.

    Raises
    ------
    ValueError: if either solubility constant is not positive.
    """
    return result
```

### Step 7

recombination_interface

Goal
----
Solve the steady two-slab problem whose interface carries the recombination channel of the previous step, returning the channel rate and the two interfacial traces. Atoms live in the first slab and molecules in the second, so the two slabs do not carry the same quantity. The source fixes a convention here that the natural reading does not; follow the source.

```python
def recombination_interface(length_m: float, diffusivity_m: float, length_s: float, diffusivity_s: float, forward_rate: float, sieverts_constant: float, henry_constant: float, c_outer_m: float, c_outer_s: float) -> "np.ndarray":
    """Solve the steady two-slab problem whose interface carries the recombination channel of the previous step, returning the channel rate and the two interfacial traces. Atoms live in the first slab and molecules in the second, so the two slabs do not carry the same quantity. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    length_m : float
        Thickness of the donor slab. Positive.
    diffusivity_m : float
        Diffusivity in the donor slab. Positive.
    length_s : float
        Thickness of the receiving slab. Positive.
    diffusivity_s : float
        Diffusivity in the receiving slab. Positive.
    forward_rate : float
        Forward rate constant of the recombination channel. Positive.
    sieverts_constant : float
        Dissociative solubility constant on the donor side. Positive.
    henry_constant : float
        Molecular solubility constant on the receiving side. Positive.
    c_outer_m : float
        Concentration held on the outer face of the donor slab.
    c_outer_s : float
        Concentration held on the outer face of the receiving slab.

    Returns
    -------
    ndarray of shape (3,): the channel rate, the trace in the first slab, then the trace in the second.

    Raises
    ------
    ValueError: if forward_rate is not positive, either slab length is not positive, or either solubility constant is not positive.
    """
    return result
```

### Step 8

quadratic_channel_damkohler

Goal
----
Return the dimensionless group for a channel that is second order in the donor material. Such a channel has no exchange velocity of its own, so one has to be constructed from the rate constant and a concentration, and the source is specific about which concentration that is. The source fixes a convention here that the natural reading does not; follow the source.

```python
def quadratic_channel_damkohler(length_m: float, diffusivity_m: float, forward_rate: float, reference_concentration: float) -> float:
    """Return the dimensionless group for a channel that is second order in the donor material. Such a channel has no exchange velocity of its own, so one has to be constructed from the rate constant and a concentration, and the source is specific about which concentration that is. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    length_m : float
        Thickness of the donor slab. Positive.
    diffusivity_m : float
        Diffusivity in the donor slab. Positive.
    forward_rate : float
        Forward rate constant of the recombination channel. Positive.
    reference_concentration : float
        Donor loading the ratio is evaluated at. Positive.

    Returns
    -------
    float, the dimensionless group for the second-order channel.

    Raises
    ------
    ValueError: if forward_rate or reference_concentration is not positive, or length_m or diffusivity_m is not positive.
    """
    return result
```

### Step 9

competing_channel_traces

Goal
----
Solve the steady problem in which the same interface carries two channels at once: the recombination channel of the earlier step and a first-order channel that oxidises the atom to a second, more soluble carrier. Both carriers are transported in the second slab with different diffusivities, and each is removed at its outer face. The source fixes a convention here that the natural reading does not; follow the source.

```python
def competing_channel_traces(length_m: float, diffusivity_m: float, length_s: float, diff_molecular: float, diff_fluoride: float, recomb_forward: float, recomb_reverse: float, fluor_forward: float, fluor_reverse: float, fluoride_activity: float, c_outer_m: float) -> "np.ndarray":
    """Solve the steady problem in which the same interface carries two channels at once: the recombination channel of the earlier step and a first-order channel that oxidises the atom to a second, more soluble carrier. Both carriers are transported in the second slab with different diffusivities, and each is removed at its outer face. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    length_m : float
        Thickness of the donor slab. Positive.
    diffusivity_m : float
        Diffusivity in the donor slab. Positive.
    length_s : float
        Thickness of the receiving slab. Positive.
    diff_molecular : float
        Diffusivity of the molecular carrier in the receiving slab. Positive.
    diff_fluoride : float
        Diffusivity of the fluoride carrier in the receiving slab. Positive.
    recomb_forward : float
        Forward rate constant of the recombination channel. Positive.
    recomb_reverse : float
        Reverse rate constant of the recombination channel.
    fluor_forward : float
        Forward rate constant of the oxidation channel. Positive.
    fluor_reverse : float
        Reverse rate constant of the oxidation channel.
    fluoride_activity : float
        Lumped activity of the oxidising medium. Not negative.
    c_outer_m : float
        Concentration held on the outer face of the donor slab.

    Returns
    -------
    ndarray of shape (5,): the donor trace, the recombination rate, the oxidation rate, then the two carrier traces in the order molecular, fluoride.

    Raises
    ------
    ValueError: if either forward rate constant is not positive, or fluoride_activity is negative.
    """
    return result
```

### Step 10

branching_ratio_and_exponent

Goal
----
Return the ratio measuring how the interface divides its throughput between the two competing channels, and the apparent power law an experiment would report for that interface. The second follows from the first in closed form and is bounded between the two single-channel limits. The source fixes a convention here that the natural reading does not; follow the source.

```python
def branching_ratio_and_exponent(recombination_rate: float, fluorination_rate: float) -> "np.ndarray":
    """Return the ratio measuring how the interface divides its throughput between the two competing channels, and the apparent power law an experiment would report for that interface. The second follows from the first in closed form and is bounded between the two single-channel limits. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    recombination_rate : float
        Net rate of the recombination channel. Positive.
    fluorination_rate : float
        Net rate of the oxidation channel. Not negative.

    Returns
    -------
    ndarray of shape (2,): the branching ratio, then the apparent exponent.

    Raises
    ------
    ValueError: if recombination_rate is not positive, or fluorination_rate is negative.
    """
    return result
```

### Step 11

isotopologue_fluxes

Goal
----
With two isotopes present in the donor material, recombination populates three molecules rather than one. Return the two like-atom and mixed channel rates, the total atomic budget of the heavier isotope leaving the donor, and the share of that budget carried by the mixed molecule measured across the RECOMBINATION channels only, with the oxidation channel excluded from that share. Every channel here is reversible and each carrier's own interfacial concentration is supplied. A single pair of constants parameterises all three recombination channels. The source fixes a convention here that the natural reading does not; follow the source.

```python
def isotopologue_fluxes(k_homonuclear_forward: float, k_homonuclear_reverse: float, k_fluorination_forward: float, k_fluorination_reverse: float, c_metal_h: float, c_metal_t: float, c_salt_hh: float, c_salt_ht: float, c_salt_tt: float, c_salt_tf: float, fluoride_activity: float) -> "np.ndarray":
    """With two isotopes present in the donor material, recombination populates three molecules rather than one. Return the two like-atom and mixed channel rates, the total atomic budget of the heavier isotope leaving the donor, and the share of that budget carried by the mixed molecule measured across the RECOMBINATION channels only, with the oxidation channel excluded from that share. Every channel here is reversible and each carrier's own interfacial concentration is supplied. A single pair of constants parameterises all three recombination channels. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    k_homonuclear_forward : float
        Forward rate constant of a like-atom recombination channel. Positive.
    k_homonuclear_reverse : float
        Reverse rate constant shared by the recombination channels. Positive.
    k_fluorination_forward : float
        Forward rate constant of an oxidation channel. Positive.
    k_fluorination_reverse : float
        Reverse rate constant of an oxidation channel. Positive.
    c_metal_h : float
        Interfacial concentration of the light isotope in the donor slab.
    c_metal_t : float
        Interfacial concentration of the heavy isotope in the donor slab.
    c_salt_hh : float
        Interfacial concentration of the light homonuclear molecule.
    c_salt_ht : float
        Interfacial concentration of the mixed molecule.
    c_salt_tt : float
        Interfacial concentration of the heavy homonuclear molecule.
    c_salt_tf : float
        Interfacial concentration of the heavy fluoride.
    fluoride_activity : float
        Lumped activity of the oxidising medium. Not negative.

    Returns
    -------
    ndarray of shape (4,): the like-atom rate for the light isotope, the mixed rate, the total atomic budget of the heavy isotope, then the mixed molecule's share of the recombination-only budget, w_HT/(w_HT + 2 w_TT).

    Raises
    ------
    ValueError: if either homonuclear rate constant is not positive, either fluorination rate constant is not positive, or the recombination channels carry no heavy isotope so the mixed share is undefined.
    """
    return result
```

### Step 12

governing_damkohler

Goal
----
Return the dimensionless group evaluated separately for each side of an asymmetric two-slab system, each against that side's bulk resistance as the earlier resistance step measures it, together with the single one of them that decides whether the interface may be treated as equilibrated. The source fixes a convention here that the natural reading does not; follow the source.

```python
def governing_damkohler(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, partition: float, exchange_velocity: float) -> "np.ndarray":
    """Return the dimensionless group evaluated separately for each side of an asymmetric two-slab system, each against that side's bulk resistance as the earlier resistance step measures it, together with the single one of them that decides whether the interface may be treated as equilibrated. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    length_a : float
        Thickness of slab A. Positive.
    diffusivity_a : float
        Diffusivity of the mobile species in slab A. Positive.
    length_b : float
        Thickness of slab B. Positive.
    diffusivity_b : float
        Diffusivity of the mobile species in slab B. Positive.
    partition : float
        Ratio of the forward to the reverse rate constant of the first-order channel. Positive.
    exchange_velocity : float
        Forward rate constant of the first-order channel. Positive.

    Returns
    -------
    ndarray of shape (3,): the group for side A, the group for side B, then the one that decides.

    Raises
    ------
    ValueError: if exchange_velocity is not positive, either slab length is not positive, either diffusivity is not positive, or partition is not positive.
    """
    return result
```

### Step 13

reference_branching_ratio

Goal
----
Return the branching ratio that indexes the source's map of interface behaviour, formed from the two channel rates evaluated at a supplied reference concentration rather than at a concentration any solve produces. Both carriers are removed at the far face of the second slab, as in the earlier two-channel step. The source fixes a convention here that the natural reading does not; follow the source.

```python
def reference_branching_ratio(length_s: float, diff_molecular: float, diff_fluoride: float, recomb_forward: float, recomb_reverse: float, fluor_forward: float, fluor_reverse: float, fluoride_activity: float, reference_concentration: float) -> float:
    """Return the branching ratio that indexes the source's map of interface behaviour, formed from the two channel rates evaluated at a supplied reference concentration rather than at a concentration any solve produces. Both carriers are removed at the far face of the second slab, as in the earlier two-channel step. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    length_s : float
        Thickness of the receiving slab. Positive.
    diff_molecular : float
        Diffusivity of the molecular carrier in the receiving slab. Positive.
    diff_fluoride : float
        Diffusivity of the fluoride carrier in the receiving slab. Positive.
    recomb_forward : float
        Forward rate constant of the recombination channel. Positive.
    recomb_reverse : float
        Reverse rate constant of the recombination channel.
    fluor_forward : float
        Forward rate constant of the oxidation channel. Positive.
    fluor_reverse : float
        Reverse rate constant of the oxidation channel.
    fluoride_activity : float
        Lumped activity of the oxidising medium. Not negative.
    reference_concentration : float
        Donor loading the ratio is evaluated at. Positive.

    Returns
    -------
    float, the branching ratio at the supplied reference concentration.

    Raises
    ------
    ValueError: if either forward rate constant is not positive, reference_concentration is not positive, or fluoride_activity is negative.
    """
    return result
```

### Step 14

interface_regime_report

Goal
----
Run the whole pipeline on one interface and report it. Build the first-order problem from the two solubilities and solve it; build the second-order problem from the two solubility constants and solve it; open both channels together, taking the fluoride diffusivity as a quarter of the molecular one, the oxidation reverse constant as half its forward one, and the heavier isotope at a quarter of the lighter, with every carrier removed at its outer face; then combine the two failure modes into the single indicator the source defines. Report as well the group that decides for this asymmetric pair and the branching ratio at the reference concentration the source builds its map on. The source fixes a convention here that the natural reading does not; follow the source.

```python
def interface_regime_report(length_a: float, diffusivity_a: float, length_b: float, diffusivity_b: float, solubility_a: float, solubility_b: float, exchange_velocity: float, c_outer_a: float, c_outer_b: float, sieverts_constant: float, henry_constant: float, recomb_forward: float, fluor_forward: float, fluoride_activity: float) -> "np.ndarray":
    """Run the whole pipeline on one interface and report it. Build the first-order problem from the two solubilities and solve it; build the second-order problem from the two solubility constants and solve it; open both channels together, taking the fluoride diffusivity as a quarter of the molecular one, the oxidation reverse constant as half its forward one, and the heavier isotope at a quarter of the lighter, with every carrier removed at its outer face; then combine the two failure modes into the single indicator the source defines. Report as well the group that decides for this asymmetric pair and the branching ratio at the reference concentration the source builds its map on. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    length_a : float
        Thickness of slab A. Positive.
    diffusivity_a : float
        Diffusivity of the mobile species in slab A. Positive.
    length_b : float
        Thickness of slab B. Positive.
    diffusivity_b : float
        Diffusivity of the mobile species in slab B. Positive.
    solubility_a : float
        Solubility of the dissolved species in slab A. Positive.
    solubility_b : float
        Solubility of the dissolved species in slab B. Positive.
    exchange_velocity : float
        Forward rate constant of the first-order channel. Positive.
    c_outer_a : float
        Concentration held on the outer face of slab A.
    c_outer_b : float
        Concentration held on the outer face of slab B.
    sieverts_constant : float
        Dissociative solubility constant on the donor side. Positive.
    henry_constant : float
        Molecular solubility constant on the receiving side. Positive.
    recomb_forward : float
        Forward rate constant of the recombination channel. Positive.
    fluor_forward : float
        Forward rate constant of the oxidation channel. Positive.
    fluoride_activity : float
        Lumped activity of the oxidising medium. Not negative.

    Returns
    -------
    ndarray of shape (11,): the first-order flux, its two interfacial traces, the relative flux error, the second-order dimensionless group, the apparent exponent, the heavy isotope's atomic budget, the failure indicator, the second-order departure, the group that decides for the asymmetric system, then the branching ratio at the reference concentration.

    Raises
    ------
    ValueError: if any of the three rate constants is not positive, or any value it forwards to an earlier step is invalid.
    """
    return result
```
