# Material_Science-Semiconductor_Materials-7

## Background

Group-V acceptors are the leading route to stable p-type doping of cadmium telluride and related II-VI absorbers for thin-film photovoltaics. Melt-grown bulk crystals can reach high acceptor activation, yet polycrystalline films a few micrometers thick, deposited by high-throughput vapor methods, often convert only a small fraction of the incorporated dopant into free holes. Native donors such as cation interstitials, dopant atoms on the wrong sublattice, and dopant-donor complexes are the usual suspects for this compensation.

Which of these compensating species survive to room temperature depends on the thermal history of the sample and on how far defects must travel to reach grain boundaries, surfaces and other sources and sinks. Conventional defect calculations bracket this behavior with two limits: full equilibrium maintained down to room temperature, or an instantaneous quench that locks in the high-temperature populations. Real growth and processing conditions lie between these limits, and because every charged defect is coupled to every other through charge neutrality, the outcome is not simply an interpolation between them.

## Problem

A recent defect-chemistry framework predicts the room-temperature defect populations of doped semiconductors cooled at a finite rate from their defect formation and migration energies. Apply it to the acceptor-doped polycrystalline II-VI absorber film described below.

The film has 3.8 micrometer grains whose boundaries are the only sources and sinks of point defects. It is in full defect equilibrium at 1123 K and is then cooled at a constant 0.73 K/s down to 296 K. The total dopant concentration, summed over all dopant-containing defects, is 4.3 x 10^16 cm^-3 and is conserved throughout. Host chemical potentials are fixed and already included in the tabulated formation energies.

The band gap is 1.583 eV at 0 K and narrows with temperature according to the Varshni law with coefficient 3.4 x 10^-4 eV/K and temperature parameter 128 K. The framework's band-gap partition parameter is 0.78. The conduction- and valence-band effective densities of states are 7.9 x 10^17 and 1.69 x 10^19 cm^-3 at 296 K, both scaling with the three-halves power of temperature. The host's representative vibrational quantum is 11.9 meV.

| Defect | Charge states | Neutral formation energy (eV) | Transition levels (eV) | Migration energy (eV) | Site density (cm^-3) |
| --- | --- | --- | --- | --- | --- |
| Cation interstitial | 2+, 1+, 0 | 3.15 | (2+/1+) 1.37; (1+/0) 1.49 | 0.97 | 1.484 x 10^22 |
| Cation vacancy | 0, 1-, 2- | 2.61 | (0/1-) 0.31; (1-/2-) 0.74 | 1.43 | 1.484 x 10^22 |
| Dopant on an anion site | 0, 1- | 1.21 | (0/1-) 0.13 | 1.62 | 1.484 x 10^22 |
| Dopant on a cation site | 1+, 0 | 1.30 | (1+/0) 1.29 | 1.77 | 1.484 x 10^22 |
| Anion-site dopant bound to a cation interstitial | 1+, 0 | 1.84 | (1+/0) 1.18 | 1.86 | 5.936 x 10^22 |

Formation energies are given at a Fermi level equal to the valence-band maximum, which is also the zero of the transition levels, and at the reference dopant chemical potential. Each defect's charge states share that defect's site density and migration energy, and every defect's Arrhenius diffusivity has the prefactor 0.25 cm^2/s. Where the framework allows either whole defects or individual charge states to freeze, freeze only each defect's total concentration; its charge-state populations keep following the Fermi level.

Report the dopant activation ratio of the film at 296 K, defined as the hole density divided by the total dopant concentration, as a single number.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

Charge-state formation energies

Goal
----
Compute the formation energy of every listed charge state of one point defect at a Fermi level of zero, from the neutral-state formation energy and the thermodynamic transition levels.

```python
def charge_state_formation_energies(e_neutral: float, charges: "np.ndarray", levels: "np.ndarray") -> "np.ndarray":
    '''Formation energies of the listed charge states at a Fermi level of zero.

    Parameters
    ----------
    e_neutral : float
        Formation energy of the neutral charge state (eV) at a Fermi level of zero.
    charges : np.ndarray
        One-dimensional integer array of charge states, strictly decreasing in steps of one,
        that contains the neutral state 0.
    levels : np.ndarray
        One-dimensional array of transition levels (eV, on the same energy scale as the Fermi
        level) with length len(charges) - 1. Entry i is the level between charges[i] and
        charges[i + 1], i.e. the Fermi level at which those two states have equal formation
        energy.

    Returns
    -------
    energies : np.ndarray
        Float array with the same length and order as charges: the formation energy (eV) of
        each charge state at a Fermi level of zero.

    Raises
    ------
    ValueError
        If charges does not contain 0, is not strictly decreasing in unit steps, or levels
        does not have length len(charges) - 1.
    '''
    return energies
```

### Step 2

Band edges and densities of states

Goal
----
Return the conduction- and valence-band edge energies and the effective densities of states of the host at a given temperature, on the fixed electron-energy scale used by the defect calculation.

```python
def band_edges_and_densities(temperature: float, host: dict) -> "np.ndarray":
    '''Band edges and effective densities of states at one temperature.

    Parameters
    ----------
    temperature : float
        Absolute temperature (K), strictly positive.
    host : dict
        Host description with keys
        "eg0" : band gap at 0 K (eV);
        "varshni_alpha" : Varshni coefficient (eV/K) of the gap temperature dependence;
        "varshni_beta" : Varshni temperature parameter (K);
        "f_cb" : fraction, between 0 and 1, of the band-gap change relative to 0 K that is
        carried by the conduction band, in the sense of the source framework's band-edge
        convention;
        "nc_ref", "nv_ref" : conduction- and valence-band effective densities of states
        (cm^-3) at the reference temperature;
        "t_ref" : reference temperature (K) for nc_ref and nv_ref.
        Other keys are ignored.

    Returns
    -------
    edges : np.ndarray
        Float array [Ec, Ev, Nc, Nv]: conduction- and valence-band edge energies (eV) on the
        framework's fixed energy scale, whose zero is the valence-band maximum at 0 K, and the
        effective densities of states (cm^-3), which scale with the three-halves power of
        temperature from their reference values.

    Raises
    ------
    ValueError
        If temperature is not positive or f_cb lies outside [0, 1].
    '''
    return edges
```

### Step 3

Vibrational free-energy shift

Goal
----
Return the vibrational free-energy contribution to the formation free energy of each defect at a given temperature, from the net number of atoms the defect adds to or removes from the crystal.

```python
def vibrational_free_energy_shift(temperature: float, hw0: float, n_added: "np.ndarray") -> "np.ndarray":
    '''Vibrational free-energy term added to each defect's formation energy.

    Parameters
    ----------
    temperature : float
        Absolute temperature (K), strictly positive.
    hw0 : float
        Representative vibrational quantum of the host lattice (eV), strictly positive.
    n_added : np.ndarray
        One-dimensional integer array: net number of host or impurity atoms each defect adds
        to the crystal (positive for interstitial-type defects, negative for vacancies, zero
        for substitutionals and antisites).

    Returns
    -------
    shift : np.ndarray
        Float array with the same length as n_added: the vibrational contribution (eV) to each
        defect's formation free energy under the source framework's average mode-counting
        treatment. The same value applies to every charge state of a defect; a negative value
        lowers the formation free energy.
        Temperatures convert to energies with k_B = 8.617333262e-5 eV/K (CODATA 2018).

    Raises
    ------
    ValueError
        If temperature or hw0 is not positive.
    '''
    return shift
```

### Step 4

Open charge-state concentrations

Goal
----
Compute the equilibrium concentration of every charge state of every defect at a given temperature, Fermi level and dopant chemical potential, for defects that are free to exchange atoms with their surroundings.

```python
def open_charge_state_concentrations(temperature: float, fermi_level: float, mu_dopant: float, host: dict, defects: dict) -> "np.ndarray":
    '''Dilute-limit equilibrium concentrations of all charge states.

    Parameters
    ----------
    temperature : float
        Absolute temperature (K), strictly positive.
    fermi_level : float
        Fermi level (eV) on the fixed energy scale whose zero is the valence-band maximum at
        0 K.
    mu_dopant : float
        Dopant chemical potential (eV) relative to the reference used for the tabulated
        formation energies.
    host : dict
        Host description; only "hw0", the representative vibrational quantum of the lattice
        (eV), is used.
    defects : dict
        Defect table with per-defect arrays of length n_def:
        "site" : site density (cm^-3), shared by all charge states of the defect;
        "n_added" : net number of atoms the defect adds to the crystal (integer);
        "n_dopant" : number of dopant atoms the defect contains (integer, 0 or 1);
        and per-charge-state arrays of length n_cs:
        "cs_def" : index of the defect each charge state belongs to;
        "cs_q" : charge of the state (units of the elementary charge);
        "cs_e" : formation energy (eV) at a Fermi level of zero and dopant chemical potential
        zero, with host chemical potentials already included.
        Other keys are ignored.

    Returns
    -------
    concentrations : np.ndarray
        Float array of length n_cs: concentration (cm^-3) of each charge state, including the
        vibrational free-energy contribution of the source framework for the atoms each defect
        adds or removes.
        Temperatures convert to energies with k_B = 8.617333262e-5 eV/K (CODATA 2018).

    Raises
    ------
    ValueError
        If temperature is not positive or the per-charge-state arrays have unequal lengths.
    '''
    return concentrations
```

### Step 5

Remaining diffusion length

Goal
----
Return the diffusion length a defect can still accumulate while the sample cools at a constant rate from a given temperature down to the end of the cooling path.

```python
def remaining_diffusion_length(temperature: float, em: float, d0: float, gamma: float, t_min: float) -> float:
    '''Diffusion length still available during linear cooling from temperature to t_min.

    Parameters
    ----------
    temperature : float
        Current temperature (K), with temperature >= t_min.
    em : float
        Migration energy (eV) of the defect's Arrhenius diffusivity, strictly positive.
    d0 : float
        Arrhenius diffusion prefactor (cm^2/s), strictly positive.
    gamma : float
        Constant cooling rate (K/s), strictly positive; the temperature falls linearly in time.
    t_min : float
        Final temperature of the cooling path (K), strictly positive.

    Returns
    -------
    length : float
        Root-mean-square diffusion length (cm), as defined in the source framework,
        accumulated between the current temperature and t_min. It is zero when
        temperature equals t_min.
        Temperatures convert to energies with k_B = 8.617333262e-5 eV/K (CODATA 2018).

    Raises
    ------
    ValueError
        If em, d0, gamma or t_min is not positive, or temperature < t_min.
    '''
    return length
```

### Step 6

Freeze-in temperature

Goal
----
Return the temperature at which a defect species stops equilibrating during linear cooling of a polycrystalline film, given its migration energy, its diffusion prefactor, the cooling rate and the grain size.

```python
def freeze_in_temperature(em: float, d0: float, gamma: float, t_max: float, t_min: float, grain_size: float) -> float:
    '''Freeze-in temperature of one defect species under linear cooling.

    Parameters
    ----------
    em : float
        Migration energy (eV) of the defect, strictly positive.
    d0 : float
        Arrhenius diffusion prefactor (cm^2/s), strictly positive.
    gamma : float
        Constant cooling rate (K/s), strictly positive.
    t_max : float
        Starting temperature of the cooling path (K).
    t_min : float
        Final temperature of the cooling path (K), with 0 < t_min < t_max.
    grain_size : float
        Grain size of the film (cm), strictly positive.

    Returns
    -------
    t_freeze : float
        Freeze-in temperature (K) under the source framework's freeze-in criterion, using its
        convention relating the source/sink distance to the grain size. The value lies in
        [t_min, t_max]; it equals t_max when the criterion is already met at the start of cooling.
        Temperatures convert to energies with k_B = 8.617333262e-5 eV/K (CODATA 2018).

    Raises
    ------
    ValueError
        If em, d0, gamma or grain_size is not positive, or the temperatures do not satisfy
        0 < t_min < t_max.
    '''
    return t_freeze
```

### Step 7

Partial-equilibrium state

Goal
----
Solve for the Fermi level, free-carrier densities and defect totals of a doped semiconductor in partial equilibrium at one temperature, where some defect species are frozen at fixed total concentrations and the rest are still free to equilibrate.

```python
def solve_partial_equilibrium(temperature: float, frozen_totals: "np.ndarray", dopant_total: float, host: dict, defects: dict) -> "np.ndarray":
    '''Charge-neutral partial-equilibrium state at one temperature.

    Parameters
    ----------
    temperature : float
        Absolute temperature (K), strictly positive.
    frozen_totals : np.ndarray
        Float array of length n_def. A finite entry is the fixed total concentration (cm^-3,
        summed over charge states) of a frozen defect; NaN marks a defect that is open and
        equilibrates at this temperature. The array is not modified.
    dopant_total : float
        Total dopant concentration (cm^-3) summed over every dopant-containing defect, open or
        frozen.
    host : dict
        Host description with the keys used by band_edges_and_densities ("eg0",
        "varshni_alpha", "varshni_beta", "f_cb", "nc_ref", "nv_ref", "t_ref") and "hw0", the
        representative vibrational quantum of the lattice (eV).
    defects : dict
        Defect table with per-defect arrays "site", "n_added", "n_dopant" (0 or 1) and "em",
        and per-charge-state arrays "cs_def", "cs_q", "cs_e", as documented for
        open_charge_state_concentrations. The tabulated energies are referenced to a dopant
        chemical potential of zero.

    Returns
    -------
    state : np.ndarray
        Float array of length 3 + n_def: [Fermi level (eV) on the fixed energy scale whose
        zero is the 0 K valence-band maximum, electron density (cm^-3), hole density (cm^-3),
        then the total concentration (cm^-3) of each defect in table order]. Free carriers are
        non-degenerate. Open defects follow the dilute-limit equilibrium populations, with the
        dopant chemical potential set so that the dopant content of all defects equals
        dopant_total whenever an open dopant-containing defect exists. A frozen defect keeps
        its total while its charge-state populations follow the Fermi level. The state is
        electrically neutral.
        Temperatures convert to energies with k_B = 8.617333262e-5 eV/K (CODATA 2018).

    Raises
    ------
    ValueError
        If temperature is not positive, frozen_totals has the wrong length or a negative
        entry, or an open dopant-containing defect exists while frozen species already hold
        all of dopant_total.
    '''
    return state
```

### Step 8

Sequential freeze-in totals

Goal
----
Return the total concentration of every defect species retained after a doped polycrystalline film is cooled at a constant rate, when each species freezes in at its own temperature.

```python
def sequential_freeze_in_totals(process: dict, host: dict, defects: dict) -> "np.ndarray":
    '''Frozen-in defect totals after linear cooling with species-by-species freeze-in.

    Parameters
    ----------
    process : dict
        Cooling and doping conditions with keys
        "t_max" : starting temperature (K), where every species that has not frozen is in
        equilibrium;
        "t_min" : final temperature of the cooling path (K), 0 < t_min < t_max;
        "gamma" : constant cooling rate (K/s);
        "grain_size" : grain size of the film (cm);
        "d0" : Arrhenius diffusion prefactor (cm^2/s), shared by all defects;
        "dopant_total" : total dopant concentration (cm^-3), fixed throughout.
    host : dict
        Host description as documented for solve_partial_equilibrium.
    defects : dict
        Defect table as documented for solve_partial_equilibrium, including the per-defect
        migration energies "em" (eV).

    Returns
    -------
    totals : np.ndarray
        Float array of length n_def: the total concentration (cm^-3) of each defect, in table
        order, retained at the end of the cooling path under the source framework's sequential
        freeze-in model, in which whole defects (all charge states together) freeze in.
        Temperatures convert to energies with k_B = 8.617333262e-5 eV/K (CODATA 2018).

    Raises
    ------
    ValueError
        If the process parameters are invalid (see freeze_in_temperature).
    '''
    return totals
```

### Step 9

Dopant activation ratio (orchestrator)

Goal
----
Return the room-temperature dopant activation ratio, the hole density divided by the total dopant concentration, of an acceptor-doped polycrystalline film after linear cooling with species-by-species defect freeze-in.

```python
def dopant_activation_ratio(process: dict, host: dict, defects: dict) -> float:
    '''Hole density at the end of cooling divided by the total dopant concentration.

    Parameters
    ----------
    process : dict
        Cooling and doping conditions as documented for sequential_freeze_in_totals; the hole
        density is evaluated at "t_min".
    host : dict
        Host description as documented for solve_partial_equilibrium.
    defects : dict
        Defect table in the form the problem gives it, with per-defect entries of length n_def:
        "charges" (charge states, strictly decreasing in unit steps and containing 0),
        "e_neutral" (neutral formation energy, eV), "levels" (transition levels between
        consecutive charge states, eV), and "site", "n_added", "n_dopant" and "em" as documented
        for solve_partial_equilibrium.

    Returns
    -------
    ratio : float
        Dimensionless activation ratio p / dopant_total at t_min, where the defect totals are
        those retained by sequential freeze-in, their charge states follow the Fermi level at
        t_min, and the state is electrically neutral.
        Temperatures convert to energies with k_B = 8.617333262e-5 eV/K (CODATA 2018).

    Raises
    ------
    ValueError
        If the process parameters are invalid (see freeze_in_temperature).
    '''
    return ratio
```
