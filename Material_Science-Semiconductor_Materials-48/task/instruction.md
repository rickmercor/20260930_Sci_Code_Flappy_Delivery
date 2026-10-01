# Material_Science-Semiconductor_Materials-48

## Background

## Excitons in atomically thin semiconductors

Monolayers of transition-metal dichalcogenides such as MoS2, MoSe2, WS2 and WSe2 are direct-gap
semiconductors only three atoms thick. Because the electric field lines between an electron and a
hole leave the sheet and run through the much weaker dielectric around it, the Coulomb attraction is
screened far less than in a bulk crystal, and the resulting excitons are bound by hundreds of meV and
have radii of only a few lattice constants. They dominate the optical response even at room
temperature, and their properties can be tuned by the choice of substrate, encapsulation and stacking.
The screening itself is unusual: it depends on the distance between the charges, so the familiar
hydrogen-like scaling of binding energy and radius with the dielectric constant does not hold, and
the exciton has to be computed for the actual environment rather than scaled from a textbook model.

## Exciton-polaritons in microcavities

When such a monolayer is placed inside an optical microcavity, the excitons can exchange energy with
the confined light field faster than either decays, and the two hybridise into exciton-polaritons.
These quasiparticles inherit a tiny effective mass and fast propagation from their photonic part and
the ability to interact from their excitonic part. Because the excitons of these materials are so
robust, the strong-coupling regime survives up to room temperature, which makes monolayer
semiconductors attractive for polariton lasers, all-optical switches and integrated polaritonic
circuits. Every one of these applications relies on nonlinearity, that is, on the energy of the
polaritons changing with their own density.

## Where the nonlinearity comes from

An exciton is a composite boson built from two fermions. At low density this substructure is
invisible, but as the density grows the Pauli principle starts to act between the electrons and holes
of different excitons. It shows up in two ways: as an exchange interaction between excitons, and as a
loss of the phase space available for creating new electron-hole pairs, which weakens the coupling to
light. Both effects are transmitted to polaritons only through their excitonic content, so the
nonlinear response depends on how excitonic the populated polariton states are, and that in turn
depends on the detuning between cavity and exciton and on the temperature.

## Thermal populations and measured blueshifts

Polaritons in real samples do not all sit in the lowest state. Relaxation through phonons and
carrier scattering spreads them over the dispersion, which contains a small, strongly curved,
photon-like region near normal incidence and a vast, nearly flat, exciton-like reservoir at larger
momenta. Experiments usually report the density-dependent energy shifts of the lower and upper
polariton at normal incidence and convert them into an effective interaction constant. Interpreting
those numbers, and designing devices around them, requires connecting the observed shifts back to
the microscopic interactions and to the way the population is distributed, which is where
phenomenological models and microscopic theories can disagree.

## Problem

When exciton-polaritons in a monolayer semiconductor inside a microcavity are driven to high density, both branches shift, and such data are usually read with a phenomenological picture in which the lower and upper polaritons move rigidly by an exciton-exciton interaction constant times the density. A microscopic description instead builds the nonlinear response from the fermionic substructure of the excitons, treating the exchange of identical carriers and the phase-space filling (Pauli blocking) of the light-matter coupling separately for each branch and populating the whole polariton dispersion thermally, so that it takes the carrier masses, the screened electron-hole interaction, the cavity mode, the light-matter coupling, the temperature and the density as input and returns the density-induced energy shift of each polariton branch at normal incidence.

Consider an hBN-encapsulated MoS2 monolayer carrying a single bright 1s exciton species (one valley and one spin configuration, with no dark or excited excitonic states), whose relative motion obeys the effective-mass Wannier equation with electron and hole masses 0.47 m0 and 0.54 m0 and the screened interaction V(q) = e^2/(2 eps0 q (kappa + r0 q)) with kappa = 4.5 and r0 = 4.15 nm, solved to convergence, and whose dispersion is E_X(Q) = 1.900 eV + hbar^2 Q^2/(2M) with M the sum of the carrier masses. The Fabry-Perot mode has E_C(Q) = sqrt(E_C0^2 + (hbar c Q/n_c)^2) with n_c = 1.8 and E_C0 = E_X(0) - 30 meV, and at each in-plane momentum it mixes with the exciton through the 2x2 Hamiltonian whose off-diagonal element is a real, positive, momentum-independent coupling g = 20 meV, where g is the interband electron-photon matrix element (taken momentum-independent) multiplied by the sum over relative momenta of the normalised 1s amplitude, chosen positive at the origin.

Treat both interactions at the mean-field level of the carrier equations of motion, transformed into the polariton basis while keeping only branch-diagonal polariton occupations (no interaction-induced coupling between branches): the fermionic exchange between the carriers of two 1s excitons in its long-wavelength limit, with its zero-momentum value used for all momenta, and the phase-space filling of the electron-photon coupling with the full momentum dependence of its matrix element. Populate both branches with a Boltzmann distribution over the low-density polariton dispersion, continuous in two-dimensional momentum and normalised to a total polariton density of 1.0 x 10^12 cm^-2, at 50 K, using hbar^2/(2 m0) = 0.0380998 eV nm^2, e^2/(4 pi eps0) = 1.439964 eV nm, hbar c = 197.32698 eV nm and k_B = 8.617333 x 10^-5 eV/K. Report as the final answer, in meV, the density-induced change of the normal-incidence splitting between the upper (UP) and lower (LP) polariton, Delta E_UP - Delta E_LP. The binding energy of the 1s exciton, the exchange constant in eV nm^2 (compared with the value published for monolayer MoS2 with this microscopic approach), the zero-momentum saturation density at which the linearised phase-space filling would bleach the exciton-photon coupling (in cm^-2), the exciton fraction of each branch at normal incidence, the density-weighted mean exciton fraction of the thermal gas, the LP and UP normal-incidence shifts each split into their exchange and phase-space-filling parts, and the same splitting change at 40 K are among the scalars that determine the final number, so set them out in the reasoning, each to at least four significant figures, together with the relations you used for each channel in the polariton basis.

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

01_screened_gaussian_moments

Goal
----
Step 01 - Gaussian moments of the screened electron-hole interaction in a monolayer.

```python
import numpy as np
import numpy.typing as npt


def screened_gaussian_moments(s: npt.ArrayLike, kappa: float, r0: float) -> np.ndarray:
    '''Gaussian moments of the repulsive Rytova-Keldysh interaction.

    Parameters
    ----------
    s : array_like
        One-dimensional array of Gaussian exponents, each finite and > 0, in nm^-2.
    kappa : float
        Mean dielectric constant of the surrounding media, finite and > 0.
    r0 : float
        Screening length of the sheet in nm, finite and >= 0; r0 = 0 is the
        Coulomb limit.

    Returns
    -------
    moments : numpy.ndarray
        U(s) = integral over the plane of V(r) exp(-s r^2) for each s, in eV nm^2,
        where V(q) = e^2/(2 eps0 q (kappa + r0 q)) and e^2/(4 pi eps0) = 1.439964
        eV nm. Same shape as s.

    Raises
    ------
    ValueError
        If s is not a non-empty one-dimensional array of finite positive values,
        if kappa is not finite and positive, or if r0 is negative or not finite.
    '''
    return moments
```

### Step 2

02_exciton_ground_state

Goal
----
Step 02 - Lowest bound state of the screened Wannier problem in a Gaussian basis.

```python
import numpy as np
import numpy.typing as npt


def exciton_ground_state(exponents: npt.ArrayLike, m_e: float, m_h: float, kappa: float, r0: float) -> np.ndarray:
    '''Lowest eigenstate of the screened Wannier equation in a Gaussian basis.

    Parameters
    ----------
    exponents : array_like
        One-dimensional array of distinct, finite, positive exponents a_i of the
        basis functions exp(-a_i r^2), in nm^-2.
    m_e : float
        Electron effective mass in units of m0, finite and > 0.
    m_h : float
        Hole effective mass in units of m0, finite and > 0.
    kappa : float
        Mean dielectric constant of the surrounding media, finite and > 0.
    r0 : float
        Screening length of the sheet in nm, finite and >= 0.

    Returns
    -------
    state : numpy.ndarray
        Array of length N + 1: the lowest eigenvalue E_1s in eV (negative for a
        bound state), followed by the N coefficients c_i of
        psi(r) = sum_i c_i exp(-a_i r^2), normalised so that the integral of psi^2
        over the plane is one and signed so that sum_i c_i > 0.

    Raises
    ------
    ValueError
        If exponents is not a non-empty one-dimensional array of distinct, finite,
        positive values, if a mass is not finite and positive, if kappa or r0 is
        invalid as in step 01, or if the basis is numerically linearly dependent.
    '''
    return state
```

### Step 3

03_exchange_constant

Goal
----
Step 03 - Long-wavelength fermionic exchange constant of two 1s excitons.

```python
import numpy as np
import numpy.typing as npt


def exchange_constant(exponents: npt.ArrayLike, coefficients: npt.ArrayLike, kappa: float, r0: float) -> float:
    '''Long-wavelength exchange constant W of two identical 1s excitons.

    Parameters
    ----------
    exponents : array_like
        One-dimensional array of finite positive Gaussian exponents a_i, nm^-2.
    coefficients : array_like
        Coefficients c_i of the real amplitude psi(r) = sum_i c_i exp(-a_i r^2),
        same length as exponents, taken as given (not renormalised).
    kappa : float
        Mean dielectric constant of the surrounding media, finite and > 0.
    r0 : float
        Screening length of the sheet in nm, finite and >= 0.

    Returns
    -------
    W : float
        The exchange constant in eV nm^2, defined so that a gas of density n of
        identical 1s excitons shifts each of them by W n.

    Raises
    ------
    ValueError
        If exponents and coefficients are not non-empty one-dimensional arrays of
        equal length with finite entries and positive exponents, or if kappa or
        r0 is invalid as in step 01.
    '''
    return W
```

### Step 4

04_saturation_overlap

Goal
----
Step 04 - Phase-space-filling overlap of a zero-momentum pair state with an exciton at momentum Q.

```python
import numpy as np
import numpy.typing as npt


def saturation_overlap(exponents: npt.ArrayLike, coefficients: npt.ArrayLike, m_e: float, m_h: float, momenta: npt.ArrayLike) -> np.ndarray:
    '''Phase-space-filling overlap I(Q) for a zero-momentum probe and an exciton at Q.

    Parameters
    ----------
    exponents : array_like
        One-dimensional array of finite positive Gaussian exponents a_i, nm^-2.
    coefficients : array_like
        Coefficients c_i of the real amplitude psi(r) = sum_i c_i exp(-a_i r^2),
        same length as exponents, taken as given (not renormalised).
    m_e : float
        Electron effective mass in units of m0, finite and > 0.
    m_h : float
        Hole effective mass in units of m0, finite and > 0.
    momenta : array_like
        One-dimensional array of centre-of-mass momenta Q >= 0 of the occupied
        exciton, nm^-1.

    Returns
    -------
    overlap : numpy.ndarray
        I(Q) in nm for each momentum, same shape as momenta.

    Raises
    ------
    ValueError
        If exponents and coefficients are not non-empty one-dimensional arrays of
        equal length with finite entries and positive exponents, if a mass is not
        finite and positive, or if momenta is not a non-empty one-dimensional
        array of finite non-negative values.
    '''
    return overlap
```

### Step 5

05_polariton_branches

Goal
----
Step 05 - Lower and upper polariton dispersions and Hopfield amplitudes.

```python
import numpy as np
import numpy.typing as npt


def polariton_branches(momenta: npt.ArrayLike, exciton_energy: float, total_mass: float, cavity_detuning: float, cavity_index: float, coupling: float) -> np.ndarray:
    '''Polariton energies and Hopfield amplitudes of the 2x2 exciton-photon problem.

    Parameters
    ----------
    momenta : array_like
        One-dimensional array of in-plane momenta Q >= 0, nm^-1.
    exciton_energy : float
        Exciton energy at Q = 0, E_X(0), in eV, > 0.
    total_mass : float
        Total exciton mass M = m_e + m_h in units of m0, > 0.
    cavity_detuning : float
        E_C(0) - E_X(0) in eV (negative when the cavity lies below the exciton);
        E_C(0) must be positive.
    cavity_index : float
        Effective refractive index n_c of the cavity mode, > 0.
    coupling : float
        Real exciton-photon coupling g in eV, > 0.

    Returns
    -------
    branches : numpy.ndarray
        Array of shape (6, len(momenta)) with rows E_LP and E_UP in eV, then the
        amplitudes X_LP, C_LP, X_UP, C_UP of the normalised eigenvectors, each
        signed so that its photon amplitude is non-negative.

    Raises
    ------
    ValueError
        If momenta is not a non-empty one-dimensional array of finite
        non-negative values, if a scalar input is not finite, or if E_X(0),
        E_C(0), total_mass, cavity_index or coupling is not positive.
    '''
    return branches
```

### Step 6

06_thermal_exciton_fraction

Goal
----
Step 06 - Mean exciton fraction of a thermal polariton gas.

```python
import numpy as np
import numpy.typing as npt


def thermal_exciton_fraction(exciton_energy: float, total_mass: float, cavity_detuning: float, cavity_index: float, coupling: float, temperature: float) -> float:
    '''Density-weighted mean exciton fraction of a Boltzmann polariton gas.

    Parameters
    ----------
    exciton_energy : float
        Exciton energy at Q = 0 in eV, > 0.
    total_mass : float
        Total exciton mass in units of m0, > 0.
    cavity_detuning : float
        E_C(0) - E_X(0) in eV; E_C(0) must be positive.
    cavity_index : float
        Effective refractive index of the cavity mode, > 0.
    coupling : float
        Real exciton-photon coupling g in eV, > 0.
    temperature : float
        Temperature in K, finite and > 0.

    Returns
    -------
    fraction : float
        <X^2>, the Boltzmann-weighted mean of X_nu(Q)^2 over both branches and the
        whole momentum plane, between 0 and 1.

    Raises
    ------
    ValueError
        If temperature is not finite and positive, or if the dispersion
        parameters are invalid as in step 05.
    '''
    return fraction
```

### Step 7

07_normal_incidence_shifts

Goal
----
Step 07 - Density-induced shifts of the lower and upper polariton at normal incidence.

```python
import numpy as np
import numpy.typing as npt


def normal_incidence_shifts(exponents: npt.ArrayLike, coefficients: npt.ArrayLike, m_e: float, m_h: float, kappa: float, r0: float, exciton_energy: float, cavity_detuning: float, cavity_index: float, coupling: float, temperature: float, density: float) -> np.ndarray:
    '''Normal-incidence LP and UP shifts produced by a thermal polariton gas.

    Parameters
    ----------
    exponents : array_like
        One-dimensional array of finite positive Gaussian exponents a_i, nm^-2.
    coefficients : array_like
        Coefficients c_i of the 1s amplitude psi(r) = sum_i c_i exp(-a_i r^2),
        same length as exponents, with sum_i c_i > 0.
    m_e : float
        Electron effective mass in units of m0, > 0.
    m_h : float
        Hole effective mass in units of m0, > 0.
    kappa : float
        Mean dielectric constant of the surrounding media, > 0.
    r0 : float
        Screening length of the sheet in nm, >= 0.
    exciton_energy : float
        Exciton energy at Q = 0 in eV, > 0.
    cavity_detuning : float
        E_C(0) - E_X(0) in eV; E_C(0) must be positive.
    cavity_index : float
        Effective refractive index of the cavity mode, > 0.
    coupling : float
        Real exciton-photon coupling g in eV, > 0.
    temperature : float
        Temperature in K, > 0.
    density : float
        Total polariton density in nm^-2, > 0 (1 nm^-2 = 1e14 cm^-2).

    Returns
    -------
    shifts : numpy.ndarray
        Array (Delta E_LP, Delta E_UP) of the normal-incidence shifts in meV,
        each the sum of the exchange and phase-space-filling contributions.

    Raises
    ------
    ValueError
        If temperature or density is not finite and positive, if sum_i c_i is not
        positive, or if any input is invalid as in steps 03 to 06.
    '''
    return shifts
```

### Step 8

08_polariton_splitting_change

Goal
----
Step 08 - Density-induced change of the normal-incidence LP-UP splitting (orchestrator).

```python
import numpy as np
import numpy.typing as npt


def polariton_splitting_change(exponents: npt.ArrayLike, m_e: float, m_h: float, kappa: float, r0: float, exciton_energy: float, cavity_detuning: float, cavity_index: float, coupling: float, temperature: float, density: float) -> float:
    '''Density-induced change of the normal-incidence splitting, Delta E_UP - Delta E_LP.

    Parameters
    ----------
    exponents : array_like
        One-dimensional array of distinct, finite, positive exponents of the
        Gaussian basis used for the 1s exciton, nm^-2.
    m_e : float
        Electron effective mass in units of m0, > 0.
    m_h : float
        Hole effective mass in units of m0, > 0.
    kappa : float
        Mean dielectric constant of the surrounding media, > 0.
    r0 : float
        Screening length of the sheet in nm, >= 0.
    exciton_energy : float
        Exciton energy at Q = 0 in eV, > 0.
    cavity_detuning : float
        E_C(0) - E_X(0) in eV; E_C(0) must be positive.
    cavity_index : float
        Effective refractive index of the cavity mode, > 0.
    coupling : float
        Real exciton-photon coupling g in eV, > 0.
    temperature : float
        Temperature in K, > 0.
    density : float
        Total polariton density in nm^-2, > 0.

    Returns
    -------
    splitting_change : float
        Delta E_UP - Delta E_LP at normal incidence, in meV.

    Raises
    ------
    ValueError
        If the lowest eigenvalue of step 02 is not negative (no bound 1s state),
        or if any input is invalid as in steps 02 to 07.
    '''
    return splitting_change
```
