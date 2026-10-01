# Physics-Astrophysics-3

## Background

The solar wind close to the Sun is threaded by Alfvenic fluctuations of very
large amplitude. Spacecraft crossing this region encounter intervals in which
the magnetic field swings far from the ambient direction - by tens of degrees,
and in the most extreme cases through a full reversal - while the strength of
the field stays close to the value it has on either side of the excursion.
These structures dominate the fluctuation energy in the young solar wind, and
their nature bears on how momentum and energy are carried outward from the
corona.

What makes them awkward to model is that they are simultaneously
large-amplitude and localised. At the amplitudes actually observed the
fluctuation is no longer a small perturbation of the background: the excursion
is comparable in size to the ambient field itself, and it occupies only a
finite region of the domain.

The setting for this task is a single such packet on a periodic cubic grid in
dimensionless units: the background field has unit magnitude and points along
the x axis, the cube has unit side and unit period in each direction, and the
packet is centred in it. All arithmetic is in double precision. The reported
quantity is an angle in degrees between the local field direction and the
ambient x direction, measured over every point of the grid.

## Problem

Large-amplitude Alfvenic fluctuations in the near-Sun solar wind appear as localised, strongly deflected excursions of the magnetic field that nonetheless preserve a nearly uniform field strength. Your task is to construct an admissible three-dimensional wave packet of this kind on the periodic unit cube, at the configuration specified below, and report a single number: the maximum deflection angle, in degrees, of the resulting magnetic field from the uniform background direction.

Start from the seed field, sampled at x_i = i/N for i = 0, ..., N-1 and likewise in y and z, given by

F0 = (1, 0, 0) + A [cos(phi) yhat + sin(phi) zhat] exp(-dr^2 / (2 sigma^2)),

where phi = 2 pi kx x and dr is the distance from the cube centre (0.5, 0.5, 0.5). Use N = 64, A = 20.0, sigma = 1/30 and kx = 4, in float64 throughout.

Run exactly 200 construction cycles starting from that seed, then evaluate the deflection angle theta = arccos(Bx / |B|) in degrees at every grid point of the field the 200th cycle produces, and report the largest value. Your answer must be within 0.5 degrees of the reference. Your reasoning must also report the numerical checks by which you verified the result; those checks count as values that determine the final number, and the brevity rules below are not a reason to omit them.

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

seed_field

Goal
----
Step 01: build the seed vector field F0 that the rest of the pipeline starts

from.

```python
import numpy as np

def seed_field(N, A, sigma, kx):
    """Build the (3, N, N, N) seed field F0 on the periodic unit cube.

    The field is a uniform background of unit strength along x plus a
    circularly polarised transverse perturbation of peak amplitude A, confined
    by a Gaussian envelope of width sigma centred at (0.5, 0.5, 0.5) and
    rotating with axial wavenumber kx:

        F0[0] = 1.0
        F0[1] = cos(phi) * env
        F0[2] = sin(phi) * env
        phi   = 2 * pi * kx * X
        env   = A * exp(-dr2 / (2 * sigma**2))
        dr2   = (X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2

    Coordinates are x_i = i/N for i = 0..N-1 on every axis, combined with
    np.meshgrid(coord, coord, coord, indexing="ij").

    Parameters
    ----------
    N : int
        Grid points per axis. Must be an integer with N >= 2.
    A : float
        Peak amplitude of the transverse perturbation. Must be finite.
    sigma : float
        Gaussian envelope width in box-length units. Must be > 0.
    kx : int
        Axial wavenumber in cycles per box length. Must be a non-negative
        integer.

    Returns
    -------
    F0 : numpy.ndarray
        Array of shape (3, N, N, N) and dtype float64 holding the seed field,
        component-first.

    Raises
    ------
    ValueError
        If N is not an integer >= 2, if sigma is not strictly positive and
        finite, if A is not finite, or if kx is not a non-negative integer.
    """
    return F0  # placeholder
```

### Step 2

wavevector_grid

Goal
----
Discrete wavevector grid for the periodic unit cube.

```python
import numpy as np

def wavevector_grid(N):
    """Build the discrete wavevector components for the periodic unit cube.

    Parameters
    ----------
    N : int
        Number of grid points along each axis of the cube. Must be an integer
        greater than or equal to 2.

    Returns
    -------
    numpy.ndarray
        Array ``K`` of shape (3, N, N, N) and dtype float64 with
        ``K[0] = KX``, ``K[1] = KY`` and ``K[2] = KZ``. Each component is the
        physical wavenumber, i.e. 2 * pi times the integer FFT frequency, laid
        out in numpy ``fftn`` storage order and broadcast over the three axes
        with ``indexing="ij"``.

    Raises
    ------
    ValueError
        If ``N`` is not an integer or if ``N`` is smaller than 2.
    """
    return K  # placeholder
```

### Step 3

solenoidal_projection

Goal
----
Implement solenoidal_projection, the operator that returns the divergence-free

part of a vector field on the periodic unit cube.

```python
import numpy as np

def solenoidal_projection(F: np.ndarray) -> np.ndarray:
    '''Return the solenoidal (divergence-free) part of a periodic vector field.

    The separation is made spectrally, on the wavevector grid of step 02 and
    the coefficients of np.fft.fftn taken over the three spatial axes. The
    returned field is real-valued and is required to be divergence-free to
    floating-point round-off when measured by the spectral diagnostic of
    step 06.

    Parameters
    ----------
    F : np.ndarray
        (3, N, N, N) float array, component-first vector field on the periodic
        unit cube with grid x_i = i/N, and N >= 2.

    Returns
    -------
    G : np.ndarray
        (3, N, N, N) float array, the solenoidal part of F.

    Raises
    ------
    ValueError
        If F is not a (3, N, N, N) float array with equal trailing
        dimensions and N >= 2.
    '''
    return G  # placeholder
```

### Step 4

unit_normalise

Goal
----
Step 04: pointwise unit-magnitude normalisation of a vector field.

```python
import numpy as np

def unit_normalise(G):
    """Divide a vector field pointwise by its magnitude.

    Parameters
    ----------
    G : numpy.ndarray
        Vector field of shape (3, N, N, N), component-first, on the periodic unit
        cube.  Sites whose magnitude is exactly zero are left unchanged.

    Returns
    -------
    numpy.ndarray
        Normalised field of shape (3, N, N, N), float64.

    Raises
    ------
    ValueError
        If the input is not a (3, N, N, N) array with equal trailing
        dimensions, or if N < 1.  The operation is pointwise and couples no
        neighbouring samples, so N = 1 is admissible here even though the
        spectral steps require N >= 2.
    """
    return normalised  # placeholder
```

### Step 5

deflection_angles

Goal
----
Step 05: the deflection angle of the local field direction from the background.

```python
import numpy as np

def deflection_angles(B: np.ndarray) -> np.ndarray:
    '''Deflection angle in degrees between the local field and xhat.

    Parameters
    ----------
    B : np.ndarray
        (3, N, N, N) component-first vector field on the periodic unit-cube
        grid. The shape is validated as in step 03: four dimensions, a leading
        axis of length 3, and three equal trailing axes of length N >= 1.
        Every grid site must have strictly positive magnitude, because the
        field direction, and hence the angle, is undefined at a site where the
        vector vanishes.

    Returns
    -------
    theta : np.ndarray
        (N, N, N) array of deflection angles in DEGREES, computed elementwise
        as np.degrees(np.arccos(np.clip(B[0] / mag, -1.0, 1.0))) with
        mag the pointwise magnitude. The clip is a floating-point guard. The
        ratio B[0] / mag reaches exactly +/-1 at grid sites where the field is
        exactly axial, which is common here because the transverse envelope
        vanishes over most of the cube, so arccos is evaluated with no margin
        at the edge of its domain. Any floating-point excursion a few ulp
        beyond 1 in absolute value, from a differently ordered or
        differently rounded magnitude, makes np.arccos return NaN, and a single
        NaN would poison the maximum over the cube. Clipping folds such an
        excursion back onto the endpoint and leaves every interior ratio
        untouched.

    Raises
    ------
    ValueError
        If B does not have shape (3, N, N, N) with N >= 1, or if any grid site
        has exactly zero magnitude.
        N = 1 is admissible: the angle is evaluated pointwise and couples
        no neighbouring samples, unlike the spectral steps, which require
        N >= 2.
    '''
    return theta  # placeholder
```

### Step 6

max_divergence

Goal
----
Implement max_divergence, the spectral divergence diagnostic of a periodic

vector field.

```python
import numpy as np

def max_divergence(B: np.ndarray) -> float:
    '''Return the maximum absolute divergence of a periodic vector field.

    The divergence is evaluated spectrally, from the coefficients of
    np.fft.fftn taken over the three spatial axes together with the wavevector
    grid of step 02, with the real part of the inverse transform taken before
    the maximum. A grid finite-difference estimate is a different operator and
    is not interchangeable with it.

    Parameters
    ----------
    B : np.ndarray
        (3, N, N, N) float array, component-first vector field on the periodic
        unit cube with grid x_i = i/N (likewise y_j, z_k), with N >= 2 and a
        cubic grid.

    Returns
    -------
    float
        The maximum over all grid points of the absolute value of div B.

    Raises
    ------
    ValueError
        If B is not a (3, N, N, N) float array on a cubic grid with
        N >= 2.
    '''
    return max_div  # placeholder
```

### Step 7

magnitude_defect

Goal
----
Implement magnitude_defect, the uniformity diagnostic of the field magnitude.

```python
import numpy as np

def magnitude_defect(B: np.ndarray) -> float:
    '''Return the spread of the pointwise field magnitude over the grid.

    Parameters
    ----------
    B : np.ndarray
        Vector field of shape (3, N, N, N) on the periodic unit cube, stored
        component-first with the spatial axes ordered (x, y, z).

    Returns
    -------
    defect : float
        Population standard deviation (ddof = 0) of |B| over all N**3 grid
        sites, in the units of B. It is 0.0 for any field of uniform
        magnitude, whether or not that common magnitude equals one.

    Raises
    ------
    ValueError
        If B is not a (3, N, N, N) float array with equal trailing
        dimensions and N >= 2.
    '''
    return defect  # placeholder
```

### Step 8

packet_max_deflection

Goal
----
Step 08: the end-to-end driver for the construction and its maximum deflection

angle.

```python
import numpy as np

def packet_max_deflection(N, A, sigma, kx, n_iter):
    """Run the construction end to end and report its peak deflection.

    The seed of step 01 is built from ``N``, ``A``, ``sigma`` and ``kx`` and
    relaxed by ``n_iter`` cycles, each applying one solenoidal projection
    (step 03) and one pointwise normalisation (step 04). The returned value is
    the maximum over the spatial grid of the step-05 deflection angle of the
    delivered field, in degrees.

    Parameters
    ----------
    N : int
        Number of grid points along each axis of the periodic unit cube. Must be
        an integer greater than or equal to 2.
    A : float
        Amplitude of the Gaussian envelope of the seed field.
    sigma : float
        Width of the Gaussian envelope of the seed field. Must be positive.
    kx : int
        Integer number of transverse rotations of the seed along x.
    n_iter : int
        Number of projection/normalisation cycles. Must be an integer greater
        than or equal to 1.

    Returns
    -------
    float
        Maximum deflection angle in degrees over all grid sites of the
        delivered field.

    Raises
    ------
    ValueError
        If ``n_iter`` is not an integer or is smaller than 1, or if any of
        ``N``, ``A``, ``sigma`` or ``kx`` fails the seed-field validation.
    """
    return theta_max  # placeholder
```
