# Physics-Condensed_Matter_Physics-10

## Background

A conventional superconductor below its transition temperature is described by a complex order parameter whose magnitude is the energy gap that opens at the Fermi surface. In the BCS picture that gap is fixed by a self-consistency condition: the sum over the band of the pairing weight, one minus twice the Fermi occupation of the quasiparticle energy divided by twice that energy, must equal the reciprocal of the pairing coupling. With a constant density of states the sum becomes an integral over band energy across a symmetric cutoff, the density of states cancels against the coupling, and both the equilibrium gap at any temperature and the critical temperature above which no positive root survives follow from one dimensionless product and one cutoff. That reduction reproduces the fitted gap and transition temperature of thin niobium nitride films, which is the material the pulse parameters in this task were measured on.

Driving such a film with an intense terahertz pulse whose frequency is below twice the gap does not break pairs one by one. Instead the vector potential couples to the condensate through the gauge-covariant gradient, which for a spatially uniform sample reduces to a term proportional to the square of the vector potential multiplying the gap. That coupling pushes the amplitude of the order parameter away from equilibrium and it rings back, the collective amplitude oscillation often called the Higgs mode of the superconductor. The equation governing that motion is a driven nonlinear ordinary differential equation for the gap alone, with a restoring force that vanishes exactly at the equilibrium gap, a linear damping term, and an inertial term carrying the second time derivative.

Near the transition the standard time-dependent Ginzburg-Landau description drops the inertial term and the gap simply relaxes. Well below the transition the inertia cannot be dropped, and the coefficient in front of the second derivative, together with the coefficient in front of the squared vector potential, is what decides how far the pulse drives the gap down before it recovers. Both coefficients are integrals over the quasiparticle spectrum that depend on the gap and the temperature, and how they are defined, how they scale with temperature and how the drive amplitude is normalized against them is the substance the solver must establish. The graded number is the deepest value the gap magnitude reaches during the drive, divided by its equilibrium value at the simulation temperature, so it is a dimensionless suppression between zero and one.

## Problem

A terahertz pulse driving a superconductor does not simply heat it: if the drive is short and strong enough the amplitude of the order parameter is set into motion and rings, and the gap that reopens afterwards carries the memory of that motion. Describing this requires an equation of motion for the gap itself, and what multiplies each of its time derivatives is what decides how far the gap is driven down.

This calculation takes a homogeneous s-wave superconductor with a constant density of states, drives it with a single multi-cycle terahertz pulse, and returns one number describing how deeply the gap is suppressed at the worst moment of the drive. Because the sample is uniform there is no spatial variation and no phase winding, so the order parameter stays real and positive and the whole problem reduces to a single driven nonlinear ordinary differential equation in time, with the uniform vector potential entering only through the gauge-covariant coupling as a term proportional to its square.

Measure energy in millielectronvolts and time in picoseconds throughout, and keep the reduced Planck constant at its physical value of 0.6582119569 millielectronvolt picoseconds, inserting whatever power of it each term needs so that every term of the equation of motion carries units of energy. Take a constant density of states, so every momentum sum becomes an integral over band energy from minus the cutoff to plus the cutoff, with the pairing entering only through the dimensionless product of the coupling constant and that density of states; that product is the value named coupling below, and the density of states itself cancels from every term and is never needed numerically. Determine the equilibrium gap at any temperature as the strictly positive root of the condition that the integral over band energy from minus the cutoff to plus the cutoff of one minus twice the Fermi occupation of the quasiparticle energy, divided by twice that quasiparticle energy, equals the reciprocal of coupling, where the quasiparticle energy is the square root of the sum of the squares of the band energy and the gap and the Fermi occupation uses Boltzmann constant 0.08617333262 millielectronvolts per kelvin; the critical temperature is the temperature above which no such positive root exists. Identifying the coefficients that multiply the two time derivatives, and how the drive amplitude is normalized, is the substance of this problem.

The configuration is

cutoff = 2.6
coupling = 1.11
T_sim = 0.5 * T_c
T_ref = 0.8 * T_c
drive_strength = 0.4
damping = 0.03
frequency = 0.6
sigma_before = 1.341
sigma_after = 5.065
phase = -6.33
t_center = 4.76
t_max = 40.0

The vector potential is a Gaussian envelope times a carrier, exp of minus the squared time offset from t_center divided by sigma squared, multiplied by the cosine of two pi times frequency times the time plus the phase, where sigma is sigma_before for times before t_center and sigma_after afterwards. That whole quantity is the vector potential itself, so the drive term in the equation of motion involves its square, while the damping value is a time in picoseconds and multiplies the first time derivative of the gap directly, on the same side of the equation as the inertia term. The drive strength fixes the squared pulse amplitude in units of the stiffness evaluated at T_ref while the dynamics runs at T_sim. Every coefficient that depends on the gap is evaluated once, at the equilibrium gap of its own temperature, and then held fixed for the whole integration rather than being recomputed as the gap evolves. Start the gap at its equilibrium value at T_sim with zero time derivative, integrate from time zero to t_max, and take the smallest gap magnitude anywhere in that window.

Your final answer must be a single number: the minimum gap magnitude reached during the pulse divided by the equilibrium gap magnitude at T_sim.
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

01_fermi_occupation

Goal
----
Evaluate the Fermi-Dirac occupation for an array of quasiparticle energies at a given temperature. The routine takes energies in millielectronvolts and a temperature in kelvin and returns occupations in the closed interval from zero to one, with the exponent clipped so that large ratios saturate instead of overflowing. Invalid input raises ValueError: the temperature must be a finite non-negative scalar and the energies must all be finite.

```python
import numpy as np


def fermi_occupation(energies, temperature):
    """Return the Fermi-Dirac occupation of the given quasiparticle energies.

    Parameters
    ----------
    energies : array_like
        Quasiparticle energies in millielectronvolts.
    temperature : float
        Temperature in kelvin. Zero selects the step-function limit.

    Returns
    -------
    numpy.ndarray
        Occupations in [0, 1] with the same shape as ``energies``.

    Raises
    ------
    ValueError
        If ``temperature`` is not a finite non-negative scalar, or if any entry
        of ``energies`` is not finite.
    """
    return None
```

### Step 2

02_gap_kernel

Goal
----
Evaluate the reduced BCS gap kernel for a positive gap, a non-negative temperature and a positive energy cutoff. The routine takes the gap magnitude and cutoff in millielectronvolts and the temperature in kelvin, then returns the constant-density-of-states momentum sum divided by that density of states as a single float. Invalid input raises ValueError: the gap and cutoff must be finite scalars strictly greater than zero and the temperature must be a finite non-negative scalar.

```python
import numpy as np
from scipy.integrate import quad


def gap_kernel(gap, temperature, cutoff):
    """Return the reduced BCS gap kernel within the symmetric energy cutoff.

    Parameters
    ----------
    gap : float
        Positive quasiparticle gap magnitude in millielectronvolts.
    temperature : float
        Temperature in kelvin, including the zero-temperature limit.
    cutoff : float
        Positive symmetric energy cutoff in millielectronvolts.

    Returns
    -------
    float
        Momentum integral divided by the constant density of states.

    Raises
    ------
    ValueError
        If any input is not a finite scalar, if ``gap`` or ``cutoff`` is not
        strictly positive, or if ``temperature`` is negative.
    """
    return None
```

### Step 3

03_equilibrium_gap

Goal
----
Solve the reduced BCS self-consistency condition for the positive equilibrium superconducting gap at a given temperature. The routine takes temperature in kelvin, a positive energy cutoff in millielectronvolts and a positive dimensionless coupling, then returns the gap in millielectronvolts as a single float. A collapsed superconducting solution returns zero rather than raising. Invalid input raises ValueError: temperature must be a finite non-negative scalar, while cutoff and coupling must be finite scalars strictly greater than zero.

```python
import numpy as np
from scipy.optimize import brentq


def equilibrium_gap(temperature, cutoff, coupling):
    """Return the positive equilibrium superconducting gap or zero in the normal state.

    Parameters
    ----------
    temperature : float
        Temperature in kelvin, including the zero-temperature limit.
    cutoff : float
        Positive symmetric energy cutoff in millielectronvolts.
    coupling : float
        Positive dimensionless product of pairing strength and density of states.

    Returns
    -------
    float
        Equilibrium gap in millielectronvolts, or zero if no positive root exists.

    Raises
    ------
    ValueError
        If any input is not a finite scalar, if ``temperature`` is negative, or
        if ``cutoff`` or ``coupling`` is not strictly positive.
    """
    return None
```

### Step 4

04_critical_temperature

Goal
----
Locate the critical temperature by bisecting the equilibrium superconducting gap over temperature. The routine takes a positive energy cutoff in millielectronvolts and a positive dimensionless coupling, then returns in kelvin the temperature where the gap reaches a threshold of 1e-6 millielectronvolts within the fixed search window from 0.01 to 100.0 kelvin. Invalid input raises ValueError: cutoff and coupling must be finite scalars strictly greater than zero, and the search bracket must contain a sign change.

```python
import numpy as np
from scipy.optimize import brentq


def critical_temperature(cutoff, coupling):
    """Return the temperature where the equilibrium superconducting gap collapses.

    Parameters
    ----------
    cutoff : float
        Positive symmetric energy cutoff in millielectronvolts.
    coupling : float
        Positive dimensionless product of pairing strength and density of states.

    Returns
    -------
    float
        Critical temperature in kelvin.

    Raises
    ------
    ValueError
        If either input is not a finite scalar strictly greater than zero, or if
        no transition is found from 0.01 to 100.0 kelvin.
    """
    return None
```

### Step 5

05_inertia_coefficient

Goal
----
Evaluate the reduced inertia coefficient for a positive gap, a non-negative temperature and a positive energy cutoff. The routine takes the gap magnitude and cutoff in millielectronvolts and the temperature in kelvin, then returns the coefficient divided by the constant density of states as a single float obtained by numerical quadrature over the band. Invalid input raises ValueError: the gap and cutoff must be finite scalars strictly greater than zero and the temperature must be a finite non-negative scalar.

```python
import numpy as np
from scipy.integrate import quad


def inertia_coefficient(gap, temperature, cutoff):
    """Return the reduced inertia coefficient within the symmetric energy cutoff.

    Parameters
    ----------
    gap : float
        Positive quasiparticle gap magnitude in millielectronvolts.
    temperature : float
        Temperature in kelvin, including the zero-temperature limit.
    cutoff : float
        Positive symmetric energy cutoff in millielectronvolts.

    Returns
    -------
    float
        Inertia coefficient divided by the constant density of states.

    Raises
    ------
    ValueError
        If any input is not a finite scalar, if ``gap`` or ``cutoff`` is not
        strictly positive, or if ``temperature`` is negative.
    """
    return None
```

### Step 6

06_pulse_envelope

Goal
----
Evaluate a normalized asymmetric multi-cycle terahertz vector potential at an array of times. The routine takes times and pulse widths in picoseconds, frequency in terahertz and phase in radians, then returns a dimensionless NumPy array with the same shape as the times. Invalid input raises ValueError: the pulse center, both widths, frequency and phase must be finite scalars, both widths must be strictly greater than zero and every time must be finite.

```python
import numpy as np


def pulse_envelope(times, t_center, sigma_before, sigma_after, frequency, phase):
    """Return the normalized asymmetric terahertz vector potential.

    Parameters
    ----------
    times : array_like
        Times in picoseconds.
    t_center : float
        Pulse center in picoseconds.
    sigma_before : float
        Positive envelope width before the pulse center in picoseconds.
    sigma_after : float
        Positive envelope width at and after the pulse center in picoseconds.
    frequency : float
        Carrier frequency in terahertz.
    phase : float
        Carrier phase in radians.

    Returns
    -------
    numpy.ndarray
        Dimensionless normalized vector potential with the same shape as ``times``.

    Raises
    ------
    ValueError
        If any scalar parameter is not finite, either width is not strictly
        positive, or any entry of ``times`` is not finite.
    """
    return None
```

### Step 7

07_drive_coefficient

Goal
----
Evaluate the dimensionless coefficient multiplying the squared pulse in the gap equation of motion. The routine takes a finite non-negative drive strength, positive simulation and reference temperatures in kelvin, a positive cutoff in millielectronvolts and a positive dimensionless coupling, then returns the coefficient as a single float. Invalid input raises ValueError: drive strength must be a finite non-negative scalar, both temperatures, cutoff and coupling must be finite scalars strictly greater than zero and the equilibrium gap must not vanish at either temperature.

```python
import numpy as np


def drive_coefficient(drive_strength, t_sim, t_ref, cutoff, coupling):
    """Return the dimensionless coefficient multiplying the squared pulse.

    Parameters
    ----------
    drive_strength : float
        Finite non-negative squared-pulse strength.
    t_sim : float
        Positive simulation temperature in kelvin.
    t_ref : float
        Positive reference temperature in kelvin.
    cutoff : float
        Positive symmetric energy cutoff in millielectronvolts.
    coupling : float
        Positive dimensionless product of pairing strength and density of states.

    Returns
    -------
    float
        Dimensionless coefficient multiplying the squared normalized pulse.

    Raises
    ------
    ValueError
        If any input is not a finite scalar, if ``drive_strength`` is negative,
        if another input is not strictly positive, or if either equilibrium gap
        vanishes.
    """
    return None
```

### Step 8

08_run_pipeline

Goal
----
Chain every preceding step into the graded number by integrating the driven gap dynamics and returning the deepest suppression the pulse produces. The routine determines the critical temperature from the pairing parameters, places the simulation and reference temperatures at the requested fractions of it, builds the equilibrium gap and the inertia coefficient there, forms the drive coefficient, then integrates the equation of motion from equilibrium at rest and reports the minimum gap magnitude divided by its equilibrium value. Invalid input raises ValueError: every scalar argument must be finite, the cutoff, coupling, damping and t_max must be strictly positive, the drive strength must be non-negative, and the two temperature fractions must lie strictly between zero and one.

```python
import numpy as np
from scipy.integrate import solve_ivp


def run_pipeline(cutoff, coupling, sim_fraction, ref_fraction, drive_strength,
                 damping, frequency, sigma_before, sigma_after, phase,
                 t_center, t_max):
    """Return the minimum gap magnitude during the pulse divided by its equilibrium value.

    Parameters
    ----------
    cutoff : float
        Symmetric band-energy cutoff in millielectronvolts.
    coupling : float
        Dimensionless product of the pairing constant with the density of states.
    sim_fraction : float
        Simulation temperature as a fraction of the critical temperature.
    ref_fraction : float
        Drive-normalization reference temperature as a fraction of the critical temperature.
    drive_strength : float
        Dimensionless pump strength fixing the squared pulse amplitude.
    damping : float
        Damping coefficient of the first time derivative, in picoseconds.
    frequency : float
        Pulse carrier frequency in terahertz.
    sigma_before, sigma_after : float
        Gaussian widths in picoseconds before and after the pulse centre.
    phase : float
        Carrier phase in radians.
    t_center : float
        Pulse centre in picoseconds.
    t_max : float
        End of the integration window in picoseconds.

    Returns
    -------
    float
        Minimum gap magnitude reached during the window divided by the
        equilibrium gap magnitude at the simulation temperature.

    Raises
    ------
    ValueError
        If any argument is not a finite scalar, if ``cutoff``, ``coupling``,
        ``damping`` or ``t_max`` is not strictly positive, if ``drive_strength``
        is negative, or if either temperature fraction is outside the open
        interval from zero to one.
    """
    return None
```
