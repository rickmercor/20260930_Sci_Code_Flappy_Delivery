# Physics-Astrophysics-37

## Background

Low-frequency Alfvenic fluctuations carry most of the wave energy in the solar corona and
solar wind, far below the ion gyrofrequency where cyclotron resonance is unavailable. Ions can
nevertheless be heated by such waves once their motion turns chaotic, and the physical picture
of that transition is the breakdown of the first adiabatic invariant: when the field lines bend
sharply enough on the scale of a gyro-orbit the magnetic moment is no longer conserved and the
ion is scattered from one field line to another.

The task concerns a spectrum of finite-amplitude oblique modes riding on a uniform background,
all sharing one propagation direction, with frequencies spread over a narrow band and
amplitudes falling with frequency. Because several modes are superposed, the field geometry is
much richer than for a single wave: the fluctuations can nearly cancel the background at some
phase combinations and reinforce it at others, and the point in the space of mode phases where
the field bends most violently relative to the ion gyration is not at any symmetric
configuration.

All quantities are dimensionless. Magnetic fields are normalised to the background strength,
lengths to the Alfven speed divided by the ion gyrofrequency, and the ion speed is set to the
Alfven speed. The reported quantity is a single dimensionless number characterising the most
dangerous point of the specified field for magnetic-moment conservation.

## Problem

A population of singly charged ions moves through a uniform background magnetic field of
strength B0 along z-hat that also carries eleven finite-amplitude oblique Alfvenic magnetic
fluctuations. Report the smallest value that the effective relative curvature radius, the
dimensionless chaos-onset criterion used for this system in the recent literature, attains
anywhere in the space of the eleven mode phases for the field specified below.

Lengths are normalised to the Alfven speed divided by the ion gyrofrequency, magnetic fields to
B0, and the ion speed is one. The eleven modes share one propagation direction lying in the x-z
plane at angle alpha to z-hat with tan(alpha) = 4.4, each satisfies the Alfven dispersion
relation for this background, and their dimensionless frequencies are the eleven equally spaced
values covering 0.11 to 0.19 inclusive. Mode k contributes the Cartesian fluctuation B_k times
(-cos(alpha) sin(psi_k), cos(psi_k), sin(alpha) sin(psi_k)), where psi_k is that mode's phase,
and the mode amplitudes satisfy B_k squared proportional to (omega_k / omega_1) raised to the
power -1.667, that decimal exactly rather than five thirds, normalised so that the eleven
squared amplitudes sum to 0.19.

Work in the frame in which the fluctuation pattern is stationary, so the field is a fixed
function of the eleven phases and the criterion follows from the local field and its spatial
derivatives alone at unit ion speed. The phase-space search must be global and demonstrably
converged, since the minimiser is not at an obvious symmetric point.

Report that global minimum over the eleven mode phases.

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

mode_spectrum

Goal
----
Deterministic construction of the oblique Alfvenic mode spectrum.

```python
import math

import numpy as np


def mode_spectrum(
    num_modes: int = 11,
    bw2: float = 0.19,
    omega1: float = 0.11,
    q: float = 1.667,
    tan_alpha: float = 4.4,
    band_width: float = 0.08,
) -> np.ndarray:
    """Return the per-mode spectrum of the benchmark wave field.

    Parameters
    ----------
    num_modes : int
        Number of wave modes in the spectrum. Must be at least 1.
    bw2 : float
        Total dimensionless wave power, the sum of the squared mode amplitudes.
        Must be positive.
    omega1 : float
        Lowest dimensionless mode frequency. Must be positive.
    q : float
        Exponent of the amplitude power law in frequency.
    tan_alpha : float
        Tangent of the common propagation angle measured from the background
        field direction. Must be positive.
    band_width : float
        Width of the dimensionless frequency band above ``omega1``.
        Must be positive.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(4, num_modes)`` holding, in order, the mode
        frequencies, the mode amplitudes, the perpendicular wavenumber
        components and the parallel wavenumber components.

    Raises
    ------
    ValueError
        If ``num_modes`` is below one, or if ``bw2``, ``omega1``, ``tan_alpha``
        or ``band_width`` is not strictly positive.
    """
    return
```

### Step 2

total_field

Goal
----
Total magnetic field of the multimode wave packet at one phase point.

```python
import math

import numpy as np


def total_field(phases: np.ndarray, amplitude: np.ndarray, tan_alpha: float = 4.4) -> np.ndarray:
    """Return the total dimensionless magnetic field at one phase vector.

    Parameters
    ----------
    phases : numpy.ndarray
        Mode phases, one per mode.
    amplitude : numpy.ndarray
        Mode amplitudes, same length as ``phases``.
    tan_alpha : float
        Tangent of the common propagation angle. Must be positive.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(3,)`` with the Cartesian components of the field,
        normalised to the background field strength.

    Raises
    ------
    ValueError
        If ``phases`` and ``amplitude`` do not have the same shape, if
        ``phases`` is not a non-empty one-dimensional array, or if
        ``tan_alpha`` is not strictly positive.
    """
    return
```

### Step 3

field_gradient_tensor

Goal
----
Spatial gradient tensor of the multimode field.

```python
import math

import numpy as np


def field_gradient_tensor(
    phases: np.ndarray,
    omega: np.ndarray,
    amplitude: np.ndarray,
    tan_alpha: float = 4.4,
) -> np.ndarray:
    """Return the spatial gradient tensor of the total field at one phase vector.

    Parameters
    ----------
    phases : numpy.ndarray
        Mode phases, one per mode.
    omega : numpy.ndarray
        Mode frequencies, same length as ``phases``.
    amplitude : numpy.ndarray
        Mode amplitudes, same length as ``phases``.
    tan_alpha : float
        Tangent of the common propagation angle. Must be positive.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(3, 3)`` whose entry in row ``j`` and column ``i`` is the
        derivative of field component ``i`` with respect to coordinate ``j``.

    Raises
    ------
    ValueError
        If ``phases``, ``omega`` and ``amplitude`` do not share one shape, if
        ``phases`` is not a non-empty one-dimensional array, or if
        ``tan_alpha`` is not strictly positive.
    """
    return
```

### Step 4

curvature_radius

Goal
----
Local field-line curvature radius of the multimode field.

```python
import numpy as np


def curvature_radius(field: np.ndarray, gradient: np.ndarray) -> float:
    """Return the local field-line curvature radius.

    The radius is ``|B|**2 / |(B . grad) B|`` with the FULL directional
    derivative ``(B . grad) B``, whose component ``i`` is
    ``sum_j field[j] * gradient[j, i]``. It is not projected perpendicular to
    ``B`` before its magnitude is taken.

    Parameters
    ----------
    field : numpy.ndarray
        Total magnetic field at the point, shape ``(3,)``.
    gradient : numpy.ndarray
        Spatial gradient tensor at the same point, shape ``(3, 3)``, with the
        derivative index first.

    Returns
    -------
    float
        The curvature radius ``|B|**2 / |(B . grad) B|``.

    Raises
    ------
    ValueError
        If ``field`` does not have shape ``(3,)``, if ``gradient`` does not
        have shape ``(3, 3)``, if the field magnitude is zero, or if
        ``(B . grad) B`` vanishes so that the field line is locally straight.
    """
    return 0.0
```

### Step 5

gradient_anisotropy_ratio

Goal
----
Anisotropy of the field gradient about the background field direction.

```python
import numpy as np


def gradient_anisotropy_ratio(gradient: np.ndarray) -> float:
    """Return the ratio of full to cross-field gradient magnitude.

    Both magnitudes are Frobenius norms of the corresponding tensors, the
    cross-field part being the gradient with the derivative along the
    background direction (the z axis) projected out.

    Parameters
    ----------
    gradient : numpy.ndarray
        Spatial gradient tensor at the same point, shape ``(3, 3)``, with the
        derivative index first.

    Returns
    -------
    float
        The dimensionless ratio, never smaller than one.

    Raises
    ------
    ValueError
        If ``gradient`` does not have shape ``(3, 3)``, if the gradient
        vanishes, or if its cross-field part vanishes.
    """
    return 0.0
```

### Step 6

effective_curvature_parameter

Goal
----
The dimensionless curvature parameter at one phase point.

```python
import numpy as np


def effective_curvature_parameter(
    phases: np.ndarray,
    spectrum: np.ndarray,
    tan_alpha: float = 4.4,
    speed: float = 1.0,
) -> float:
    """Return the dimensionless curvature parameter at one phase vector.

    Parameters
    ----------
    phases : numpy.ndarray
        Mode phases, one per mode.
    spectrum : numpy.ndarray
        Spectrum array of shape ``(4, num_modes)`` as produced by the first step.
    tan_alpha : float
        Tangent of the common propagation angle. Must be positive.
    speed : float
        Dimensionless ion speed entering the gyroradius. Must be positive.

    Returns
    -------
    float
        The parameter value at that phase point.

    Raises
    ------
    ValueError
        If ``spectrum`` does not have shape ``(4, num_modes)``, if ``phases``
        does not hold exactly one entry per mode, or if ``speed`` is not
        strictly positive.
    """
    return 0.0
```

### Step 7

phase_space_minimum

Goal
----
Global minimum of the curvature parameter over the phase torus.

```python
import math

import numpy as np


def phase_space_minimum(
    spectrum: np.ndarray,
    tan_alpha: float = 4.4,
    speed: float = 1.0,
    num_starts: int = 4000,
    seed: int = 20260909,
    num_iterations: int = 4000,
) -> np.ndarray:
    """Locate the global minimum of the curvature parameter over all mode phases.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Spectrum array of shape ``(4, num_modes)`` as produced by the first step.
    tan_alpha : float
        Tangent of the common propagation angle. Must be positive.
    speed : float
        Dimensionless ion speed entering the gyroradius. Must be positive.
    num_starts : int
        Number of independent starting phase vectors. Must be at least 1.
    seed : int
        Seed of the generator that draws the starting phase vectors.
    num_iterations : int
        Number of descent iterations applied to every start. Must be at least 1.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(num_modes + 1,)`` holding the minimum value found
        followed by the mode phases that attain it, each reduced to the
        interval ``[0, 2*pi)``. The value equals the curvature parameter
        evaluated at the returned phases and is converged with respect to the
        number of starts. Any minimising phase vector is acceptable, since the
        minimiser is not unique.

    Raises
    ------
    ValueError
        If ``spectrum`` does not have shape ``(4, num_modes)``, if ``tan_alpha``
        or ``speed`` is not strictly positive, or if ``num_starts`` or
        ``num_iterations`` is below one.
    """
    return
```

### Step 8

chaos_onset_curvature_minimum

Goal
----
End-to-end benchmark value (final orchestrator step).

```python
import numpy as np


def chaos_onset_curvature_minimum(
    num_modes: int = 11,
    bw2: float = 0.19,
    omega1: float = 0.11,
    q: float = 1.667,
    tan_alpha: float = 4.4,
    speed: float = 1.0,
    num_starts: int = 4000,
    seed: int = 20260909,
    num_iterations: int = 4000,
) -> float:
    """Return the benchmark phase-space minimum of the curvature parameter.

    Parameters
    ----------
    num_modes : int
        Number of wave modes in the spectrum.
    bw2 : float
        Total dimensionless wave power.
    omega1 : float
        Lowest dimensionless mode frequency.
    q : float
        Exponent of the amplitude power law in frequency.
    tan_alpha : float
        Tangent of the common propagation angle.
    speed : float
        Dimensionless ion speed entering the gyroradius.
    num_starts : int
        Number of independent starting phase vectors for the search.
    seed : int
        Seed of the generator that draws the starting phase vectors.
    num_iterations : int
        Number of descent iterations applied to every start.

    Returns
    -------
    float
        The benchmark value.

    Raises
    ------
    ValueError
        If ``speed`` is not strictly positive, if ``num_starts`` or
        ``num_iterations`` is below one, or if the spectrum parameters are
        invalid (``num_modes`` below one; ``bw2``, ``omega1`` or ``tan_alpha``
        not strictly positive).
    """
    return 0.0
```
