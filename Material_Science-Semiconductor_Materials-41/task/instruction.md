# What a constant-cross-section DLTS analysis gets wrong

## Background

Deep-level transient spectroscopy measures the thermal emission rate of carriers from a deep
level as a function of temperature. The rate is the product of a capture cross section, a carrier
thermal velocity, the effective density of states of the band and the Boltzmann factor of the
thermodynamic level, so the level and the cross section are extracted together from the slope and
the intercept of a plot of the logarithm of the rate, scaled by the square of the temperature,
against reciprocal temperature. The scaling by the square of the temperature is there because the
thermal velocity and the density of states between them carry two powers of temperature.

Two readings of that plot are in common use and they disagree. The original one assumes the
capture cross section does not depend on temperature, so the plot is a straight line whose slope
is the level and whose intercept is the cross section. The second treats capture as a
nonradiative multiphonon transition and places a classical capture barrier, read off the crossing
of the two configuration coordinate parabolas, into an activated exponential, which adds the
barrier to the slope and leaves one algebraic power of temperature in the intercept. Published
values of levels and cross sections depend on which reading was used, and the two can disagree
substantially.

Both rest on approximations that nonradiative multiphonon theory has not supported for decades.
Factorising the electronic and vibrational parts of the transition matrix element, so that only
the vibrational overlap survives, is one of them; assuming high temperature and strong coupling
is another. Without those approximations the transition rate is a thermally weighted sum over
initial phonon states of a squared phonon matrix element, with the final phonon state fixed by
energy conservation, and its temperature dependence is governed not by a fixed barrier but by an
averaged phonon energy that the transition itself selects. The selection is the point: phonon
states whose wavefunctions overlap the final state well are promoted in the average, and for a
strongly relaxing defect those lie far above the thermal average energy, while for a weakly
relaxing one they lie close to it. A model carrying a single barrier cannot express that
difference at all, and so assigns the same temperature dependence to two defects whose lattice
relaxation differs by a factor of four.

The model system here is a pair of deep levels in one wide-band-gap host that share an effective
phonon energy, an electronic transition energy and a thermodynamic level, and differ only in
lattice relaxation. Their classical barriers are nearly equal by construction, so the activated
model cannot tell them apart, while the rigorous treatment separates them by many orders of
magnitude in the temperature dependence of the capture cross section.

## Problem

Deep-level transient spectroscopy is the standard way to measure a deep level in a
semiconductor: the thermal emission rate of carriers from the level is recorded against
temperature, and a model is fitted to it to extract the thermodynamic level and the carrier
capture cross section. The oldest and still the most widely used reading assumes the capture
cross section is temperature independent. It is not, because capture is a nonradiative
multiphonon transition, and the temperature dependence it really has is not the activated form
that the other common model assumes either.

Your task is to generate synthetic emission rates from the rigorous nonradiative multiphonon
model on the configuration below, analyse them exactly as an experimentalist would with the
constant-cross-section model, and report how far the extracted capture cross section lands from
the truth.

## The configuration

Two deep levels in one wide-band-gap host. They share an effective phonon energy, an electronic
transition energy and a thermodynamic level, and differ only in the lattice relaxation that
accompanies the change of charge state:

    effective phonon energy               hw    = 0.038 eV
    electronic transition energy          dE    = 1.02 eV
    thermodynamic level E_C - E_T         level = 0.62 eV
    lattice relaxation, level A           dQ_A  = 4.90 amu^(1/2) Angstrom
    lattice relaxation, level B           dQ_B  = 1.20 amu^(1/2) Angstrom
    electron effective mass               m*    = 0.29 m_e
    degeneracy ratio of the charge states g     = 2.0
    true capture cross section at 300 K   sigma(T_ref) = 1.0e-15 cm^2

Work in the mass-weighted configuration coordinate, in which the two charge states are harmonic
surfaces of equal curvature displaced by the lattice relaxation and offset in energy by the
electronic transition energy, and in which the kinetic operator carries no mass. Use
k_B = 8.617333262e-05 eV/K, h = 4.135667696e-15 eV s, an electron mass of
5.6095886e-32 eV s^2 / Angstrom^2, and hbar^2 = 4.18005956e-03 eV amu Angstrom^2, which is the
constant that turns a phonon energy in eV into the squared oscillator length of that coordinate in
amu Angstrom^2. One square Angstrom is 1e-16 cm^2. Where a dimensionless displacement between the
two surfaces is wanted, define it so that its square is the Huang-Rhys factor.

## The construction

Follow the source for every definition it fixes. In particular the source fixes which operator
stands between the initial-state and final-state phonon states in the transition rate, and how
the final phonon state is selected; the averaged initial-state phonon energy and what weights
that average; the average oscillator energy it is compared against; the temperature derivative
of the cross section and the closed form obtained by integrating it, including its algebraic
prefactor; the form of the older model's cross section in terms of the classical barrier; the
carrier thermal velocity including its numerical factor; and the ordinate that the
constant-cross-section model predicts to be linear in reciprocal temperature, together with
what its slope and its intercept mean. None of those are supplied here.

These are settings of this calculation rather than claims of the source, so they are given:

* Phonon quantum numbers 0 to 220 are retained for the initial state, and the phonon matrix
  elements are integrated on 47501 uniform nodes of the reduced coordinate over [-35, 60].
* The energy-conserving delta is replaced by a normalised Gaussian of width 0.012 eV in energy,
  summed over final phonon states out to 5.0 widths and normalised over the states retained.
* The thermal occupation of the initial-state oscillator is its normalised Boltzmann weight.
* The effective density of states of the band is two times (2 pi m* k_B T / h^2) to the power
  three halves, per cubic Angstrom.
* The temperature grid runs from 60 K to 620 K on 1121 uniform nodes; the reference temperature
  300 K is a node of it, and the cross section is anchored there without interpolation. Carry
  the integral of the temperature derivative with a cumulative Simpson rule on that grid.
* Two fit windows are used, in reciprocal temperature measured in inverse kilokelvin:
  [2.0, 3.5] and [6.0, 10.0]. Each is sampled at 29 equally spaced reciprocal temperatures, and
  the quantity interpolated on the temperature grid is the fitted ordinate itself, not the rate.
  Interpolate it linearly against reciprocal temperature, so that a genuinely constant cross
  section is recovered exactly.

## What to report

For each of the two lattice relaxations and each of the two fit windows, fit the
constant-cross-section model to the rigorous emission rates and read off the capture cross
section its intercept implies. Report the sum, over all four combinations, of the base-ten
logarithm of the ratio between that extracted cross section and the true rigorous cross section
at 300 K. Give the sum to six significant figures.

Report as well, for each of the four combinations, the apparent thermodynamic level in eV and
the decades of error in the extracted cross section; the Huang-Rhys factor and the classical
barrier of each level; the averaged initial-state phonon energy and the average oscillator
energy at 300 K for each level; and the decades of error that the older activated model would
have produced in the same four analyses. Say which of the two levels the older model cannot
distinguish from the other and why, and say what the comparison between the two fit windows
implies for a measurement that covers only one of them.

The quantities listed above are the reported result, not intermediate bulk output. Give them as
a compact table or a short list of labelled values inside the reasoning section, and state each
convention in a sentence or two. A short table of exactly those values is not the kind of
per-iteration or per-candidate output the format note below asks you to leave out, and a
response that reports them compactly is both complete and within the length the note asks for.

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

configuration_coordinate

Goal
----
Reduce one deep level to its configuration coordinate diagram. From the lattice relaxation that accompanies the charge-state change, the effective phonon energy and the electronic transition energy, return the oscillator length of the mass-weighted coordinate, the Huang-Rhys factor, the relaxation energy, the classical capture barrier of the diagram, and the dimensionless displacement between the two parabolas, the last of these defined so that its square is the Huang-Rhys factor. Work in the mass-weighted coordinate in which the kinetic operator carries no mass, so that a relaxation quoted in amu^(1/2) Angstrom and a phonon energy quoted in eV fix the oscillator length with no further input.

```python
import numpy as np


def configuration_coordinate(delta_q, hw, delta_e):
    """Reduce one deep level to its configuration coordinate diagram. A (5,) float64 array
    holding [oscillator length in amu^(1/2) Angstrom, Huang-Rhys factor, relaxation energy
    in eV, classical capture barrier in eV, dimensionless displacement whose square is the
    Huang-Rhys factor]."""
    return np.zeros(5)
```

### Step 2

oscillator_basis

Goal
----
Tabulate the harmonic oscillator eigenfunctions that the phonon matrix elements are built from, on a uniform grid of the reduced coordinate, for every quantum number from zero up to the retained cut-off. Normalise each eigenfunction so that the grid quadrature reproduces orthonormality, and generate them by the three-term recurrence in the normalised functions rather than by evaluating Hermite polynomials and a Gaussian separately, which overflows well before the cut-off this calculation needs. Return the grid in the first row and the eigenfunctions in the rows below it.

```python
import numpy as np


def oscillator_basis(n_phonon, u_lo, u_hi, n_quad):
    """Tabulate the harmonic oscillator eigenfunctions that the phonon matrix elements are
    built from, on a uniform grid of the reduced coordinate, for every quantum number from
    zero up to the retained cut-off. A (n_phonon + 2, n_quad) float64 array whose first row
    is the reduced coordinate grid and whose row k + 1 is the normalised eigenfunction of
    quantum number k."""
    return np.zeros((n_phonon + 2, n_quad))
```

### Step 3

franck_condon_overlaps

Goal
----
Build the matrix of overlaps between the phonon states of the two charge states, the initial-state oscillator centred at the origin of the mass-weighted coordinate and the final-state oscillator centred at the lattice relaxation. Integrate on the grid of the previous step and use the same grid spacing as the quadrature weight. Index the matrix with the initial quantum number first.

```python
import numpy as np


def franck_condon_overlaps(delta_q, hw, n_phonon, u_lo, u_hi, n_quad):
    """Build the matrix of overlaps between the phonon states of the two charge states, the
    initial-state oscillator centred at the origin of the mass-weighted coordinate and the
    final-state oscillator centred at the lattice relaxation. An (n_phonon + 1, n_phonon +
    1) float64 array whose entry (m, n) is the overlap of the initial-state phonon state m
    with the final-state phonon state n."""
    return np.zeros((n_phonon + 1, n_phonon + 1))
```

### Step 4

coordinate_matrix_elements

Goal
----
Build the matrix of phonon matrix elements that the transition rate of the source is actually made of. The source writes the rate with one particular operator between the initial-state and final-state phonon states, and it is not the identity; the whole difference from the older model rests on which operator it is. Return the matrix in the mass-weighted coordinate measured from the centre of the initial-state surface, indexed with the initial quantum number first, on the same grid as the previous steps.

```python
import numpy as np


def coordinate_matrix_elements(delta_q, hw, n_phonon, u_lo, u_hi, n_quad):
    """Build the matrix of phonon matrix elements that the transition rate of the source is
    actually made of. An (n_phonon + 1, n_phonon + 1) float64 array of phonon matrix
    elements in amu^(1/2) Angstrom, entry (m, n) pairing initial-state phonon state m with
    final-state phonon state n."""
    return np.zeros((n_phonon + 1, n_phonon + 1))
```

### Step 5

lineshape_function

Goal
----
Evaluate, at each requested temperature, the lineshape function of the transition rate and the two energies whose difference governs its temperature derivative. The lineshape function is the thermally weighted sum over initial phonon states of the squared matrix element of the previous step, with the final phonon state fixed by energy conservation; replace the energy-conserving delta by a normalised Gaussian of the given width in energy, summed over final states out to the given multiple of that width, and normalise the Gaussian weights over the retained final states. A final state is retained when its energy lies within that multiple of the width of the energy-conserving target, so the retained states are those whose quantum number differs from the target by at most the given multiple of the width divided by the phonon energy. Then return the averaged initial-state phonon energy that the source defines, whose weight is the product of the thermal occupation and the squared matrix element, and the ordinary quantum-statistical average energy of one oscillator at that temperature. Take the thermal occupation to be the normalised Boltzmann weight of the initial-state oscillator.

```python
import numpy as np


def lineshape_function(q_elements, hw, delta_e, temperature, broaden, n_sigma):
    """Evaluate, at each requested temperature, the lineshape function of the transition rate
    and the two energies whose difference governs its temperature derivative. An
    (n_temperature, 3) float64 array whose columns are [lineshape function, averaged
    initial-state phonon energy in eV, quantum-statistical average oscillator energy in eV]."""
    return np.zeros((len(np.atleast_1d(temperature)), 3))
```

### Step 6

nmp_cross_section

Goal
----
Propagate the capture cross section across the temperature grid from its value at the reference temperature, using the closed-form result the source obtains by integrating its own expression for the temperature derivative. The integral runs from the reference temperature to each grid temperature and its integrand is built from the two energies of the previous step; carry it out with a cumulative Simpson rule on the given grid. The source's result also carries an algebraic prefactor in temperature that does not come from the integral; include it. Require the reference temperature to be a node of the grid and raise if it is not, so that the integral is anchored without interpolation.

```python
import numpy as np


def nmp_cross_section(q_elements, hw, delta_e, t_grid, broaden, n_sigma, t_ref, sigma_ref):
    """Propagate the capture cross section across the temperature grid from its value at the
    reference temperature, using the closed-form result the source obtains by integrating
    its own expression for the temperature derivative. A (n_temperature,) float64 array of
    capture cross sections in the same unit as the reference value."""
    return np.zeros(len(t_grid))
```

### Step 7

henry_lang_cross_section

Goal
----
Evaluate the older model's capture cross section on the same grid, anchored to the same value at the same reference temperature. That model puts the classical barrier of the configuration coordinate diagram into an activated exponential and carries one algebraic power of temperature in front of it; the source writes both and states which power it is. Return the curve normalised so that it agrees with the reference value at the reference temperature.

```python
import numpy as np


def henry_lang_cross_section(barrier, t_grid, t_ref, sigma_ref):
    """Evaluate the older model's capture cross section on the same grid, anchored to the same
    value at the same reference temperature. A (n_temperature,) float64 array of capture
    cross sections in the same unit as the reference value."""
    return np.zeros(len(t_grid))
```

### Step 8

emission_rate

Goal
----
Convert a capture cross section into the thermal emission rate that a transient spectroscopy experiment measures, using detailed balance: the rate is the cross section times the carrier thermal velocity times the effective density of states of the band, divided by the ratio of the degeneracies of the two charge states, times the Boltzmann factor of the thermodynamic level. Take the effective density of states to be two times the usual (2 pi m* k_B T / h^2) raised to the three halves. The thermal velocity is the one the source writes, including its numerical factor.

```python
import numpy as np


def emission_rate(sigma, t_grid, level, mstar, degeneracy):
    """Convert a capture cross section into the thermal emission rate that a transient
    spectroscopy experiment measures, using detailed balance: the rate is the cross section
    times the carrier thermal velocity times the effective density of states of the band,
    divided by the ratio of the degeneracies of the two charge states, times the Boltzmann
    factor of the thermodynamic level. A (n_temperature,) float64 array of emission rates in
    inverse seconds, for a cross section supplied in square Angstrom."""
    return np.zeros(len(t_grid))
```

### Step 9

arrhenius_signature

Goal
----
Carry out the analysis an experimentalist performs. Form the ordinate that the constant-cross-section model predicts to be a straight line against the reciprocal temperature in units of inverse kilokelvin, sample it at equally spaced reciprocal temperatures across the given window by linear interpolation of that ordinate on the supplied grid, and fit a straight line by least squares. Return the apparent thermodynamic level that the slope implies, the apparent capture cross section that the intercept implies once the temperature-independent prefactor built from the thermal velocity and the band density of states is divided out, and the root-mean-square residual of the fit. Interpolate the ordinate itself rather than the rate, so that a genuinely constant cross section is recovered exactly.

```python
import numpy as np


def arrhenius_signature(rate, t_grid, window, n_fit, mstar, degeneracy):
    """Carry out the analysis an experimentalist performs. A (3,) float64 array holding
    [apparent thermodynamic level in eV, apparent capture cross section in square Angstrom,
    root-mean-square fit residual]."""
    return np.zeros(3)
```

### Step 10

dlts_audit

Goal
----
Run the whole analysis on the configuration and report it. Build the temperature grid, and for each lattice relaxation in turn reduce the level to its configuration coordinate diagram, build the phonon matrix elements, propagate the rigorous cross section and the older model's cross section from the same reference value, convert both to emission rates, and then for each fit window apply the constant-cross-section analysis to both. Accumulate, over every relaxation and every window, the base-ten logarithm of the ratio between the cross section the constant-cross-section analysis extracts from the rigorous rates and the true rigorous cross section at the reference temperature; that running total is the reported value. Record one row per relaxation and window, with the run-level quantities in a leading row.

```python
import numpy as np


def dlts_audit(hw, delta_e, level, dq_values, mstar, degeneracy, sigma_ref, n_phonon, u_lo, u_hi, n_quad, broaden, n_sigma, t_lo, t_hi, n_t, t_ref, windows, n_fit):
    """Run the whole analysis on the configuration and report it. A (n_relaxation * n_window +
    1, 9) float64 array. Row zero holds [reported total, oscillator length, ratio of the
    transition energy to the phonon energy, averaged initial-state phonon energy of the
    first relaxation at the reference temperature, quantum-statistical average oscillator
    energy at the reference temperature, lineshape function of the first relaxation at the
    reference temperature, squared Franck-Condon factor out of the initial ground state into
    the energy-conserving final state of the first relaxation, quadrature spacing of the
    reduced coordinate grid, 0]. Each later row holds [lattice relaxation, lower edge of the
    window in inverse kilokelvin, upper edge, apparent thermodynamic level in eV, decades of
    error in the extracted cross section, apparent level minus the true level in eV, base-
    ten logarithm of the apparent cross section in square centimetre, decades of error the
    older model would give, Huang-Rhys factor]."""
    return np.zeros((len(dq_values)*len(windows) + 1, 9))
```
