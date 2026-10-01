# Material_Science-Semiconductor_Materials-55

## Background

Point defects in wide-bandgap semiconductors can behave as isolated atoms trapped in a solid. A
handful of them, most famously the nitrogen-vacancy centre in diamond, carry an electronic spin that
can be initialised optically, manipulated with microwaves, and read out through the intensity of the
light they emit. Silicon carbide is an attractive host for the same physics because it already has an
industrial supply chain: high-quality boules, controlled n-type and p-type doping, and mature device
processing. That makes it possible to imagine embedding a spin qubit inside a working diode rather
than in a bespoke laboratory sample.

What separates a useful colour centre from a curiosity is where its excitation goes. An electronic
transition in a defect does not produce a single sharp line, because the lattice around the defect
relaxes when the electronic state changes. Some fraction of the emission comes out at the pure
electronic energy, the zero-phonon line, and the rest is spread over a broad sideband in which the
photon has given up part of its energy to local vibrations. Worse, if the electronic energy is low
enough that a manageable number of vibrational quanta can absorb all of it, the excited state can
relax without emitting any photon at all, and the defect simply goes dark.

Both effects are controlled by the same small set of numbers: how far the atoms move when the
electronic state changes, how stiff the local vibration is, and how large the electronic transition
energy is. Collapsing the full multidimensional lattice relaxation onto one effective vibrational
mode makes these tractable, and the resulting configuration coordinate diagram, two parabolas of
equal curvature offset in position and energy, has been the workhorse of defect photophysics for
decades.

The practical obstacle has been cost rather than concept. Screening a large library of candidate
defects means running expensive hybrid-functional calculations for each one, and the vibronic
quantities have traditionally demanded more than the electronic ones: a fitted vibrational potential
sampled at many displaced geometries, and a nonradiative rate assembled as a sum over vibrational
levels in which an energy-conserving delta function has to be smeared into a Gaussian of arbitrary
width. Recent work removes both obstacles, pinning the vibrational potential with a few total
energies and carrying out the nonradiative sum analytically, so that no broadening parameter ever
enters.

Screening of this kind is usually done for a cold lattice, yet the devices these centres are meant
for run warm, and warming can switch on vibrational decay that is negligible at cryogenic
temperature. Whether a
centre that looks bright in a cryostat is still bright at the temperature of a working device is
therefore a separate question from its low-temperature ranking, and it is the one this task asks.

## Problem

Optically addressable point defects in silicon carbide are strong candidates for solid-state spin qubits and telecom-band single-photon sources, but a centre is only useful if it keeps emitting light at the temperature it is operated at. High-throughput screening of such centres has become affordable because the vibronic ingredients, the effective vibrational quantum and the multiphonon decay rate, can now be obtained from a handful of total energies and a closed-form rate expression. Consider one impurity-vacancy centre in 4H-SiC, an antimony atom substituting a silicon site next to a carbon vacancy in the 1+ charge state, and determine how warm its lattice can become before half of the light it emits when cold has been lost to vibrational decay.

Treat the two electronic states as harmonic wells of equal curvature that share a single effective vibrational mode in the mass-weighted configuration coordinate, offset from one another along that coordinate and separated in energy. The total radiative rate is the electric-dipole rate of the bare electronic transition reduced in proportion to the mean energy of the emitted photon relative to the zero-phonon line energy. The nonradiative rate is a Fermi golden rule sum over the vibrational ladder of the electronic ground state, in which the mass-weighted configuration coordinate itself, with its origin at the relaxed ground-state geometry, connects a vibrational level of the excited state to each ground-state level and a delta function enforces energy conservation. Evaluate that sum by continuing the ground-state level index from the integers to the real line, with factorials of that index replaced by the gamma function, so that the sum collapses at a single energy-conserving level without introducing any artificial broadening. For the finite-temperature extension, derive the coordinate matrix element from the stated ground-origin Q operator and continue the resulting generalized-Laguerre coefficients polynomially in the real ground-state index; use the paper's printed zero-temperature expression only as a literature comparison if it conflicts with that operator algebra. Vibrational relaxation inside the excited electronic state is fast compared with both decay channels, so at a finite lattice temperature the excited-state vibrational levels are populated according to a Boltzmann distribution and the nonradiative rate is the corresponding thermal average.

Use the configuration below, in which the relaxed excited-state geometry differs from the relaxed ground-state geometry by the displacements listed and every other atom in the supercell is unmoved.

Displacements in Angstrom, given as the relaxed excited-state position minus the relaxed ground-state position:

    Sb   0.0258  -0.0179   0.0409
    Si  -0.0394   0.0280  -0.0320
    Si   0.0318  -0.0421  -0.0289
    Si   0.0093   0.0370   0.0438
    C    0.0546  -0.0258   0.0211
    C   -0.0471   0.0393  -0.0179
    C    0.0248   0.0511  -0.0344
    C   -0.0180  -0.0458   0.0401
    C    0.0382   0.0222   0.0499
    C   -0.0526  -0.0169  -0.0256
    C    0.0154  -0.0554   0.0196
    C   -0.0324   0.0234   0.0453
    C    0.0430  -0.0346  -0.0185

Atomic masses: Sb 121.760 amu, Si 28.0855 amu, C 12.011 amu.

Total energies: relaxed ground state -8912.475160 eV; relaxed excited state -8911.713724 eV; ground-state electronic configuration evaluated at the relaxed excited-state geometry -8912.352817 eV.

Transition dipole moment 2.85 D. Refractive index of the host at the emission wavelength 2.5732. Ratio of the effective electric field at the defect to the field in the bulk 1.0. Electron-phonon coupling matrix element 0.128 eV amu^-1/2 Angstrom^-1. Configurational degeneracy of the defect 4.

The radiative quantum efficiency is the fraction of all decay events out of the excited electronic state that emit a photon. Find the lattice temperature, in kelvin, at which the radiative quantum efficiency of this centre has fallen to one half of its zero-temperature value. Along the way report the zero-temperature radiative quantum efficiency, the radiative quantum efficiency at 100 K and at 200 K, and the squared configuration-coordinate matrix element that connects the first excited vibrational level of the excited state to its energy-conserving ground-state level, together with a compact list of the other intermediate quantities you used and of the literature values and conventions the calculation relies on.

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

mass_weighted_displacement

Goal
----
Reduce the displaced defect cluster to the single mass-weighted coordinate offset that couples the electronic transition to the lattice.

```python
def mass_weighted_displacement(masses: np.ndarray, displacements: np.ndarray) -> float:
    '''Mass-weighted configuration coordinate offset between two geometries.

    Parameters
    ----------
    masses : array_like, shape (N,)
        Atomic mass of each moving atom in atomic mass units. Must be positive.
    displacements : array_like, shape (N, 3)
        Cartesian displacement of each atom in Angstrom, taken as the relaxed
        excited-state position minus the relaxed ground-state position.

    Returns
    -------
    delta_q : float
        The mass-weighted configuration coordinate offset in
        sqrt(amu) Angstrom. Raises ValueError on non-finite input, on a
        non-positive mass, or when the two arrays disagree in length or the
        displacements are not three-dimensional.
    '''
    return delta_q
```

### Step 2

configuration_coordinate_parameters

Goal
----
Turn three total energies and the coordinate offset into the zero-phonon line energy, the relaxation energy and the effective vibrational quantum.

```python
def configuration_coordinate_parameters(e_ground_relaxed: float, e_excited_relaxed: float,
                                        e_ground_at_excited: float, delta_q: float) -> np.ndarray:
    '''Zero-phonon line energy, effective phonon quantum and relaxation energy.

    Parameters
    ----------
    e_ground_relaxed : float
        Total energy of the relaxed electronic ground state in eV.
    e_excited_relaxed : float
        Total energy of the relaxed electronic excited state in eV.
    e_ground_at_excited : float
        Total energy of the ground-state electronic configuration evaluated at
        the relaxed excited-state geometry, in eV.
    delta_q : float
        Mass-weighted coordinate offset between the two relaxed geometries in
        sqrt(amu) Angstrom. Must be positive.

    Returns
    -------
    parameters : numpy.ndarray, shape (3,)
        [E_ZPL, hbar_Omega, E_rel] in eV, where E_ZPL is the zero-phonon line
        energy, hbar_Omega the effective vibrational quantum obtained from the
        harmonic ground-state curve, and E_rel the relaxation energy released
        along that curve. Raises ValueError on non-finite input, on a
        non-positive delta_q, or when the energies do not describe an excited
        state above the ground state with a positive relaxation energy.
    '''
    return parameters
```

### Step 3

electron_phonon_coupling

Goal
----
Convert the coordinate offset and the vibrational quantum into the dimensionless electron-phonon coupling strength and the Debye-Waller factor.

```python
def electron_phonon_coupling(delta_q: float, hbar_omega: float) -> np.ndarray:
    '''Dimensionless coupling strength and Debye-Waller factor.

    Parameters
    ----------
    delta_q : float
        Mass-weighted coordinate offset in sqrt(amu) Angstrom. Must be positive.
    hbar_omega : float
        Effective vibrational quantum in eV. Must be positive.

    Returns
    -------
    coupling : numpy.ndarray, shape (2,)
        [S, W], the dimensionless electron-phonon coupling strength and the
        Debye-Waller factor exp(-S). Raises ValueError on non-finite input or
        on a non-positive delta_q or hbar_omega.
    '''
    return coupling
```

### Step 4

dipole_transition_rate

Goal
----
Evaluate the purely electronic electric-dipole emission rate of the defect radiating inside the dielectric host.

```python
def dipole_transition_rate(dipole_debye: float, refractive_index: float,
                           e_zpl: float, field_ratio: float = 1.0) -> float:
    '''Purely electronic electric-dipole emission rate of a point defect.

    Parameters
    ----------
    dipole_debye : float
        Magnitude of the transition dipole moment in debye. Must be positive.
    refractive_index : float
        Refractive index of the host at the emission wavelength. Must be positive.
    e_zpl : float
        Zero-phonon line energy in eV. Must be positive.
    field_ratio : float, optional
        Ratio of the effective electric field at the defect to the field in the
        bulk. Defaults to unity.

    Returns
    -------
    gamma_r0 : float
        The purely electronic radiative transition rate in s^-1. Raises
        ValueError on non-finite input or on a non-positive dipole moment,
        refractive index or transition energy.
    '''
    return gamma_r0
```

### Step 5

radiative_rate_partition

Goal
----
Split the electronic dipole rate into the phonon-corrected total radiative rate, the zero-phonon line rate and the sideband rate.

```python
def radiative_rate_partition(gamma_r0: float, huang_rhys: float,
                             hbar_omega: float, e_zpl: float) -> np.ndarray:
    '''Split the electronic dipole rate into total, zero-phonon and sideband parts.

    Parameters
    ----------
    gamma_r0 : float
        Purely electronic electric-dipole emission rate in s^-1. Must be positive.
    huang_rhys : float
        Dimensionless electron-phonon coupling strength. Must be positive.
    hbar_omega : float
        Effective vibrational quantum in eV. Must be positive.
    e_zpl : float
        Zero-phonon line energy in eV. Must be positive.

    Returns
    -------
    rates : numpy.ndarray, shape (3,)
        [Gamma_R, Gamma_ZPL, Gamma_PSB] in s^-1: the phonon-corrected total
        radiative rate, the zero-phonon line rate, and the sideband rate.
        Raises ValueError on non-finite input, on a non-positive argument, or
        when the mean vibrational energy released reaches the transition energy,
        or when the corrected total rate would be smaller than the zero-phonon
        line rate and therefore imply a negative sideband rate.
    '''
    return rates
```

### Step 6

coordinate_matrix_element

Goal
----
Evaluate the squared generalised configuration-coordinate matrix element out of each thermally populated excited-state vibrational level into the energy-conserving ground-state level.

```python
def coordinate_matrix_element(delta_q: float, huang_rhys: float,
                              excited_levels: np.ndarray, level_offset: float) -> np.ndarray:
    '''Squared coordinate matrix element from excited levels m into ground levels m + p.

    Parameters
    ----------
    delta_q : float
        Mass-weighted coordinate offset in sqrt(amu) Angstrom. Must be positive.
    huang_rhys : float
        Dimensionless electron-phonon coupling strength S. Must be positive.
    excited_levels : numpy.ndarray of int
        Vibrational level indices m of the excited electronic state, each a
        non-negative integer.
    level_offset : float
        Real, non-negative offset p, so that each excited level m is paired with
        the ground-state level n = m + p continued to real values.

    Returns
    -------
    element_sq : numpy.ndarray
        |<chi_e,m| Q |chi_g,m+p>|^2 in amu Angstrom^2 for every m, with the same
        shape as excited_levels. The coordinate Q is measured from the relaxed
        ground-state geometry. Raises ValueError on non-finite input, on a
        non-positive delta_q or huang_rhys, on a negative level_offset, or when
        excited_levels holds anything other than non-negative integers.
    '''
    return element_sq
```

### Step 7

multiphonon_rate

Goal
----
Collapse the thermally averaged golden-rule sum at the energy-conserving levels and return the nonradiative multiphonon transition rate at a given lattice temperature.

```python
def multiphonon_rate(delta_q: float, huang_rhys: float, hbar_omega: float,
                     e_zpl: float, coupling: float, degeneracy: float,
                     temperature: float) -> float:
    '''Thermally averaged nonradiative multiphonon transition rate.

    Parameters
    ----------
    delta_q : float
        Mass-weighted coordinate offset in sqrt(amu) Angstrom. Must be positive.
    huang_rhys : float
        Dimensionless electron-phonon coupling strength S. Must be positive.
    hbar_omega : float
        Effective vibrational quantum in eV. Must be positive.
    e_zpl : float
        Zero-phonon line energy in eV. Must be positive.
    coupling : float
        Electron-phonon coupling matrix element in eV amu^-1/2 Angstrom^-1.
        Must be positive.
    degeneracy : float
        Configurational degeneracy of the point defect. Must be positive.
    temperature : float
        Lattice temperature in K. Must lie between 0 and 3000 K, inclusive;
        zero selects the zero-point level alone.

    Returns
    -------
    gamma_nr : float
        The nonradiative multiphonon transition rate in s^-1. Raises ValueError
        on non-finite input, on a non-positive delta_q, huang_rhys, hbar_omega,
        e_zpl, coupling or degeneracy, or when temperature lies outside
        the supported interval from 0 to 3000 K.
    '''
    return gamma_nr
```

### Step 8

radiative_efficiency

Goal
----
Run the whole vibronic chain at a given lattice temperature and report the radiative quantum efficiency of the centre as a percentage.

```python
def radiative_efficiency(masses: np.ndarray, displacements: np.ndarray,
                         e_ground_relaxed: float, e_excited_relaxed: float,
                         e_ground_at_excited: float, dipole_debye: float,
                         refractive_index: float, field_ratio: float,
                         coupling: float, degeneracy: float,
                         temperature: float) -> float:
    '''Radiative quantum efficiency of a colour centre at a lattice temperature, as a percentage.

    Parameters
    ----------
    masses : numpy.ndarray, shape (N,)
        Atomic masses of the moving atoms in amu.
    displacements : numpy.ndarray, shape (N, 3)
        Displacement of each atom in Angstrom, relaxed excited-state position
        minus relaxed ground-state position.
    e_ground_relaxed : float
        Total energy of the relaxed ground state in eV.
    e_excited_relaxed : float
        Total energy of the relaxed excited state in eV.
    e_ground_at_excited : float
        Total energy of the ground-state electronic configuration at the
        relaxed excited-state geometry, in eV.
    dipole_debye : float
        Transition dipole moment in debye.
    refractive_index : float
        Refractive index of the host at the emission wavelength.
    field_ratio : float
        Ratio of the effective field at the defect to the bulk field.
    coupling : float
        Electron-phonon coupling matrix element in eV amu^-1/2 Angstrom^-1.
    degeneracy : float
        Configurational degeneracy of the point defect.
    temperature : float
        Lattice temperature in K. Must be non-negative.

    Returns
    -------
    eta_rad : float
        The radiative quantum efficiency as a percentage. Raises ValueError
        whenever any stage of the chain receives invalid input, including a
        negative temperature.
    '''
    return eta_rad
```

### Step 9

thermal_quenching_temperature

Goal
----
Find the lattice temperature at which the radiative quantum efficiency of the centre has fallen to a given fraction of its zero-temperature value.

```python
def thermal_quenching_temperature(masses: np.ndarray, displacements: np.ndarray,
                                  e_ground_relaxed: float, e_excited_relaxed: float,
                                  e_ground_at_excited: float, dipole_debye: float,
                                  refractive_index: float, field_ratio: float,
                                  coupling: float, degeneracy: float,
                                  fraction: float) -> float:
    '''Lowest temperature at which the radiative efficiency falls to a fraction of its zero-temperature value.

    Parameters
    ----------
    masses : numpy.ndarray, shape (N,)
        Atomic masses of the moving atoms in amu.
    displacements : numpy.ndarray, shape (N, 3)
        Displacement of each atom in Angstrom, relaxed excited-state position
        minus relaxed ground-state position.
    e_ground_relaxed : float
        Total energy of the relaxed ground state in eV.
    e_excited_relaxed : float
        Total energy of the relaxed excited state in eV.
    e_ground_at_excited : float
        Total energy of the ground-state electronic configuration at the
        relaxed excited-state geometry, in eV.
    dipole_debye : float
        Transition dipole moment in debye.
    refractive_index : float
        Refractive index of the host at the emission wavelength.
    field_ratio : float
        Ratio of the effective field at the defect to the bulk field.
    coupling : float
        Electron-phonon coupling matrix element in eV amu^-1/2 Angstrom^-1.
    degeneracy : float
        Configurational degeneracy of the point defect.
    fraction : float
        Target efficiency as a fraction of the zero-temperature efficiency.
        Must lie strictly between 0 and 1.

    Returns
    -------
    t_quench : float
        The quenching temperature in K. Raises ValueError on a fraction outside
        (0, 1), when any stage of the chain receives invalid input, or when the
        efficiency does not fall to the target fraction below 3000 K.
    '''
    return t_quench
```
