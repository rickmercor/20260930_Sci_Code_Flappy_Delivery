# Physics-Astrophysics-13

## Background

Magnetic reconnection changes the connectivity of magnetic field lines and converts magnetic energy into kinetic and thermal energy of the plasma. In two dimensions the process is anchored at an isolated X-point, and the familiar description - an inflow region, an outflow region, and a rate quoted as a single dimensionless number - is built around that point.

Reconnection in the solar corona, in planetary magnetospheres, and in laboratory devices is not generally two-dimensional. When the field carries a component along the direction about which the reconnecting components are organised, the process is not confined to a single point but is distributed along an extended curve, and the quantities that define a rate have to be evaluated with reference to that curve rather than at one location.

The configuration used here is an analytic model of the kind employed to test methods that operate on a field given at arbitrary points rather than on simulation output. Nothing is advanced in time and nothing is solved for: the magnetic field, the electric field and the mass density are all prescribed in closed form, so each of them, and any derivative of them, can be evaluated exactly at any position. The transverse structure varies quadratically in the second and third coordinates and linearly in the first and the third, it is displaced from the coordinate origin along the second axis, and it is carried by a uniform third component. The density is stratified about the same displaced axis.

Gaussian (CGS) units are used throughout: lengths in cm, magnetic field in Gauss, electric field in statvolt per cm, mass density in g per cm^3, and the speed of light c = 2.99792458e10 cm/s. The reported quantity is a dimensionless ratio, so it is independent of the overall scale of the field but not of the unit system in which the normalisation is formed.

## Problem

The analytic magnetic configuration specified below admits a single distinguished curve along which magnetic reconnection proceeds; identify that curve and report one dimensionless number R, defined in full in the third paragraph.

All quantities are in Gaussian (CGS) units, with lengths in cm, magnetic field in Gauss, electric field in statvolt/cm, mass density in g/cm^3 and c = 2.99792458e10 cm/s. The magnetic field is B(x,y,z) = ( y^2 - 4y + 3 + z^2 + 0.4x , -x - 0.4y + 0.2z , 0.5 ). The electric field is E(x,y,z) = 7.5e-6 * ( 0 , 2z , 3 - 2y ). The mass density is rho(x,y,z) = 1.0e-14 * ( 1 + 0.6*(y^2 - 4y + 4 + z^2) ).

Sample that curve at 81 points whose z coordinates are equally spaced over the closed interval [-0.8, 0.8], the remaining two coordinates being those of the curve at that z. Throughout, the curve average of a quantity means its trapezoidal integral in arclength, taken with the straight-line distances between consecutive sample points, divided by the total sampled arclength; let R_0 be the curve average of E_par = (E.B)/|B|. Displace every sample point by plus and by minus 0.05 along the unit vector (0,1,0); for each displacement sign form the curve average of the vector B and of rho over the displaced points, then average the two signs. Let B_in be the magnitude of the resulting mean vector and rho_in the resulting mean density. Then form V_A = B_in / sqrt(4*pi*rho_in) and report R = R_0 / ( B_in * V_A / c ).

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

field_samples

Goal
----
Step 01: evaluate the three prescribed fields at supplied positions.

```python
import numpy as np


def field_samples(points, which):
    """Return the requested field evaluated at every supplied position.

    Parameters
    ----------
    points : array_like
        Shape (N, 3), N >= 1, all finite.
    which : str
        "B", "E" or "rho".

    Returns
    -------
    numpy.ndarray
        (N, 3) for "B" and "E"; (N,) for "rho", in units of 1.0e-14 g/cm^3.

    Raises
    ------
    ValueError
        If points is not a finite (N, 3) array with N >= 1, or if which is not
        one of the three accepted selectors.
    """
    return values  # placeholder
```

### Step 2

gradient_tensor

Goal
----
Step 02: the gradient tensor of the magnetic field.

```python
import numpy as np


def gradient_tensor(points):
    """Return the (N, 3, 3) gradient tensor of the magnetic field.

    Parameters
    ----------
    points : array_like
        Shape (N, 3), N >= 1, all finite.

    Returns
    -------
    numpy.ndarray
        Shape (N, 3, 3), dtype float64, with the derivative index second.

    Raises
    ------
    ValueError
        If points is not a finite (N, 3) array with N >= 1.
    """
    return tensors  # placeholder
```

### Step 3

alignment_residual

Goal
----
Step 03: alignment residual and in-plane discriminant.

```python
import numpy as np


def alignment_residual(points):
    """Return the (N, 4) residual-and-discriminant array.

    Parameters
    ----------
    points : array_like
        Shape (N, 3), N >= 1, all finite.

    Returns
    -------
    numpy.ndarray
        Shape (N, 4), dtype float64: three residual components followed by the
        discriminant.

    Raises
    ------
    ValueError
        If points is not a finite (N, 3) array with N >= 1.
    """
    return out  # placeholder
```

### Step 4

candidate_line

Goal
----
Step 04: the candidate curve on which the step-03 residual vanishes.

```python
import numpy as np


def candidate_line(z_values):
    """Return the (N, 3) sample points of the candidate curve.

    Of the two points at which the step-03 vector residual vanishes at a given
    height, this returns the one carrying a strictly positive step-03
    discriminant.

    Parameters
    ----------
    z_values : array_like
        One-dimensional finite heights, length >= 1, each admissible in the
        sense of the module docstring.

    Returns
    -------
    numpy.ndarray
        Shape (N, 3), dtype float64.

    Raises
    ------
    ValueError
        If z_values is not a one-dimensional finite array of length >= 1, or if
        the sought point does not exist at some supplied height.
    """
    return points  # placeholder
```

### Step 5

seed_point

Goal
----
Step 05: the distinguished point of the step-04 candidate curve.

```python
import numpy as np


def seed_point(z_lo, z_hi):
    """Return the point of the step-04 curve of largest step-03 discriminant.

    Parameters
    ----------
    z_lo, z_hi : float
        Finite reals with z_lo < z_hi, the closed interval to search.

    Returns
    -------
    numpy.ndarray
        Shape (3,), dtype float64, ordered (x, y, z).

    Raises
    ------
    ValueError
        If z_lo or z_hi is not a finite real, if z_lo >= z_hi, or if the
        step-04 curve does not exist somewhere on the closed interval.
    """
    return point  # placeholder
```

### Step 6

quasi_x_line

Goal
----
Step 06: the magnetic field line through a supplied point.

```python
import numpy as np


def quasi_x_line(z_values, seed):
    """Return the field line through seed, sampled at the requested heights.

    Parameters
    ----------
    z_values : array_like
        One-dimensional finite heights, length >= 1.
    seed : array_like
        Shape (3,), finite; a point of the curve.

    Returns
    -------
    numpy.ndarray
        Shape (N, 3), dtype float64.

    Raises
    ------
    ValueError
        If z_values is not a one-dimensional finite array of length >= 1, or
        if seed is not a finite array of shape (3,).
    """
    return points  # placeholder
```

### Step 7

parallel_electric_field

Goal
----
Step 07: the field-aligned electric field at supplied positions.

```python
import numpy as np


def parallel_electric_field(points):
    """Return the field-aligned electric field at every supplied position.

    Parameters
    ----------
    points : array_like
        Shape (N, 3), N >= 1, all finite.

    Returns
    -------
    numpy.ndarray
        Shape (N,), dtype float64, in statvolt/cm.

    Raises
    ------
    ValueError
        If points is not a finite (N, 3) array with N >= 1, or if the magnetic
        field magnitude vanishes at some supplied position.
    """
    return values  # placeholder
```

### Step 8

line_average

Goal
----
Step 08: the curve average of a sampled quantity.

```python
import numpy as np


def line_average(points, values):
    """Return the arclength-weighted average and the total sampled arclength.

    Parameters
    ----------
    points : array_like
        Shape (N, 3), N >= 2, all finite.
    values : array_like
        Shape (N,) or (N, 3), all finite.

    Returns
    -------
    average : float or numpy.ndarray
        Float for scalar values; shape (3,) for vector values.
    total : float
        The total sampled arclength.

    Raises
    ------
    ValueError
        If points is not a finite (N, 3) array with N >= 2, if values is not a
        finite array of shape (N,) or (N, 3), or if the total sampled
        arclength is zero.
    """
    return average, total  # placeholder
```

### Step 9

inflow_quantities

Goal
----
Step 09: inflow field strength, inflow density and Alfven speed.

```python
import numpy as np


def inflow_quantities(points, delta):
    """Return the inflow field strength, scaled inflow density and Alfven speed.

    Parameters
    ----------
    points : array_like
        Shape (N, 3), N >= 2, all finite.
    delta : float
        Strictly positive finite displacement along the unit vector (0, 1, 0).

    Returns
    -------
    B_in : float
        Magnitude of the sign-averaged curve average of the magnetic field.
    rho_s : float
        Sign-averaged curve average of the scaled density.
    V_A : float
        B_in divided by the square root of 4 * pi * rho_s * 1.0e-14.

    Raises
    ------
    ValueError
        If points is not a finite (N, 3) array with N >= 2, if delta is not a
        positive finite real, or if the resulting scaled density is not
        positive.
    """
    return B_in, rho_s, V_A  # placeholder
```

### Step 10

reconnection_rate

Goal
----
Step 10: the normalized reconnection rate (final orchestrator step).

```python
import numpy as np


def reconnection_rate(z_lo=-0.8, z_hi=0.8, n_points=81, delta=0.05):
    """Return the dimensionless normalized reconnection rate.

    Parameters
    ----------
    z_lo, z_hi : float
        Height interval; z_lo < z_hi.
    n_points : int
        Number of equally spaced sample heights, at least 2.
    delta : float
        Strictly positive finite inflow displacement.

    Returns
    -------
    float
        The dimensionless rate.

    Raises
    ------
    ValueError
        If the interval is not valid, if n_points is not an integer of at
        least 2, or if delta is not a positive finite real.
    """
    return rate  # placeholder
```
