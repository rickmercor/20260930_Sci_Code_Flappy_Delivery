# Physics-Optics-37

## Background

Transverse Fourier collocation turns scalar Helmholtz propagation in a longitudinally invariant dielectric into a matrix-function problem. The forward branch is fixed by homogeneous-medium dispersion after removal of the vacuum carrier. Squaring a sinusoidally modulated refractive index generates both first- and second-harmonic Fourier couplings, including additive collisions on a periodic grid.

The longitudinal propagator is approximated in its dimensionless operator argument, rather than by asserting an order of accuracy in the longitudinal step size. Auxiliary resolvent solves and other evaluations of the same rational matrix function represent equivalent numerical methods. The requested reference keeps the discretization and rational order fixed, including all collocation modes during propagation.

At the output, the stipulated vacuum angular-spectrum convention admits only Cartesian samples at or below the vacuum wavenumber. Radial annular powers and densities per unit polar angle are different quantities because equal radial-bin widths map to unequal angular widths. The final observable is a dimensionless power fraction; it is not a peak-normalized angular-density value, an azimuthal angle, or a dielectric-interface transmission coefficient.

## Problem

Determine the wide-angle fraction of the output angular spectrum of a normally incident scalar Gaussian beam after exactly z=1e-3 m of forward propagation through the longitudinally invariant dielectric n(x,y)=1.4+0.1*cos(k_mod*x), neglecting backscatter and removing the free-space carrier exp(i*k0*z).

Use f0=1e14 Hz, c=299792458 m/s, k0=2*pi*f0/c, and U0(x,y)=exp(-(x**2+y**2)/tp**2) with tp=15e-6 m on the half-open periodic square x_i=y_i=(i-N/2)*L/N, i=0,...,N-1, where L=768e-6 m, N=1024, and the first array axis is x.

The Fourier convention is the unnormalized sum with negative exponential and centered coordinate and angular-wavenumber arrays, k_i=2*pi*(i-N/2)/L; set k_mod=m*2*pi/L, where m is the nearest integer to 0.75*k0*L/(2*pi), with ties to even, and Fourier-index shifts are periodic with coincident couplings added.

The numerical target uses the normalized degree-(4,4) rational approximation to the forward Helmholtz-envelope exponential, expanded about zero dimensionless Helmholtz perturbation and matched through perturbation degree eight, with denominator value one at the expansion point; equivalent evaluations of this same discrete rational operator are admissible.

The nominal longitudinal increment is h=L/N, and the endpoint policy is floor(z/h) full increments followed by the positive remainder, if any, using coefficients appropriate to each increment; retain all collocation modes during propagation without adding damping or interface-transmission factors.

For the output diagnostic retain individual Cartesian samples with k_perp=sqrt(kx**2+ky**2)<=k0 before accumulation, assign spectral power abs(U_hat)**2*delta_kx*delta_ky to N uniform radial bins with edges e_j=j*k0/N, and use left-closed, right-open bins except that the last bin also contains k0.

The polar angles are measured from the positive z axis in radians, with edges alpha_j=arcsin(e_j/k0) and centers theta_j=arcsin((e_j+e_(j+1))/(2*k0)); when reporting an intermediate angular density, use annular power divided by the radial-bin width, normalized by the largest such radial value, and then divided by alpha_(j+1)-alpha_j, with zero density for zero retained power.

Return the dimensionless fraction F of total retained annular power in bins whose center angle is at least pi/6, taking F=0 for zero retained power, as one finite decimal accurate to an absolute tolerance of 1e-6.

This is the specified finite-grid rational-propagation diagnostic, not a continuum limit or transmitted-interface power; justify the forward operator, material couplings, approximation variable and its propagation limits, endpoint treatment, and power measure, and state the wide-angle and total annular-power sums on the common scale where the largest retained annulus has power one, each accurate to absolute tolerance 1e-6, in the short reasoning accompanying the scalar answer.

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

PadeCoeff

Goal
----
Construct the partial-fraction coefficients of the normalized [P/P] approximation to the forward-envelope exponential for dimensionless t=k0*delta_z. Return arrays b and d with shapes (P,) and (P+1,). The represented rational function, not a particular ordering of its matched terms, defines the answer. Include the identity limit at t=0; invalid inputs or a numerically unresolved finite representation raise ValueError as specified in the Signature.

```python
import numpy as np


def PadeCoeff(P, t):
    """Construct normalized [P/P] Pade partial fractions at x=0.

    Approximate f(x)=exp(1j*t*(sqrt(1+x)-1)), with Q(0)=1 and
    Q(x)*f(x)-N(x)=O(x**(2*P+1)). The input t=k0*h is dimensionless.
    P must be a positive integer and t a finite real or complex scalar.
    Return finite complex arrays b and d of shapes (P,) and (P+1,), with
    R(x)=d[0]+sum(d[j+1]/(1+b[j]*x)). Simultaneously permuting b and d[1:]
    is equivalent. At t=0 return b=0, d[0]=1, and d[1:]=0.
    For nonzero t the tested regimes have simple denominator roots and a
    finite partial-fraction representation. Raise ValueError for invalid
    inputs or a numerically unresolved representation. Internal higher
    precision is permitted; return ordinary NumPy complex arrays.
    The reference uses NumPy and Python's standard library only.
    """
    return np.zeros(P, dtype=complex), np.zeros(P + 1, dtype=complex)
```

### Step 2

build_modulation_matrix

Goal
----
Construct the (ny,ny) CSC matrix representing multiplication by the positive index-squared profile (C+B*cos(k_mod*x))**2 on the periodic Fourier grid. Here mx is the integer Fourier shift associated with k_mod. Include the diagonal and shifts +/-mx and +/-2*mx, wrapping indices modulo ny and adding all coincident contributions. The Signature is a neutral sparse-matrix stub; the computation belongs in Gold.

```python
import numpy as np
from scipy.sparse import csc_matrix


def build_modulation_matrix(ny, mx, B, C):
    """Return the (ny, ny) CSC matrix for multiplication by
    (C+B*cos(mx*2*pi*x/L))**2 on a periodic Fourier collocation grid.
    ny is a positive integer, mx is an integer Fourier-index shift, and
    B and C are finite real scalars. Add C**2+B**2/2 on the diagonal,
    B*C at shifts +/-mx, and B**2/4 at shifts +/-2*mx. Accumulate all
    collisions modulo ny, including shifts that wrap to the diagonal.
    """
    return csc_matrix((ny, ny), dtype=complex)
```

### Step 3

wjsolver

Goal
----
For each second-axis wavenumber, solve (I+bj*X_column)@Wj[:,column]=U[:,column], where X_column=M_tilde-I-diag((kx**2+ky[column]**2)/k0**2) and M_tilde is the positive index-squared multiplication matrix from Step 2. Return the complex auxiliary field and one reusable factor per column. Reuse a supplied cache only when the pole, grid, and material are unchanged; the right-hand side may change. Use a new cache when any operator input changes.

```python
import numpy as np


def wjsolver(kx, ky, mx, B, C, bj, k0, U, lu_factors=None):
    """Solve one auxiliary Fourier-space resolvent for every column.

    Solve (I + bj*X_column) @ Wj[:, column] = U[:, column], where
    X_column = M_tilde - I - diag((kx**2 + ky[column]**2) / k0**2).
    M_tilde is the positive index-squared matrix from Step 2.
    kx and ky have shape (ny,) in centered order; U has shape (ny, ny).
    k0 is positive, mx is an integer, B and C are real, and bj is complex.
    Return (Wj, lu_factors), with one reusable factor per column.
    A supplied cache must match the pole, grid, and material, but U may
    differ from the right-hand side used to construct the cache.
    """
    return np.zeros_like(U, dtype=complex), []
```

### Step 4

apply_pade

Goal
----
```
Apply one rational propagation increment to the centered Fourier envelope: d[0]*U plus the sum of d[j+1] times the auxiliary field for pole b[j]. Every auxiliary solve uses the same original U. Return the updated field and one factor list per pole in the current pole order. Reuse a cache only for unchanged ordered poles, grid, and material; the right-hand side may change. Pass None after changing the operator inputs or permuting coefficients.
```

```python
import numpy as np


def apply_pade(kx, ky, mx, B, C, k0, U, b, d, lu_cache=None):
    """Apply one partial-fraction Pade step to the centered field U.

    U2 = d[0]*U + sum_j d[j+1]*(I + b[j]*X)^(-1) @ U, with
    X = M_tilde - I - diag((kx**2 + ky[column]**2) / k0**2)
    for each column. Every auxiliary solve uses the original U.
    U is square; kx and ky match its dimensions; k0 is positive.
    mx is an integer shift; B and C are real. The coefficient arrays
    b and d have shapes (P,) and (P+1,), with matched pole-residue pairs.
    Return (U2, lu_cache), with one factor list per ordered pole.
    Reuse a cache only for unchanged b, grid, and material; U may change.
    Pass None after changing operator inputs or permuting coefficients.
    """
    return np.zeros_like(U, dtype=complex), []
```

### Step 5

radial_angular_power

Goal
----
Convert the centered Fourier field into radial-bin centers, peak-radial-PSD-normalized density per polar angle, and polar bin-center angles. Use uniform radial bins on [0,K], where K=min(max(abs(kx)),max(abs(ky)),k0), and discard samples above K before accumulating their powers. Bins are left-closed and right-open except that the final edge is included. Return three numeric arrays of length nbins, with zero density for a zero retained field.

```python
import numpy as np


def radial_angular_power(U, kx, ky, k0, nbins=None):
    """Reduce a centered Fourier field to a polar angular density.

    U has shape (len(kx), len(ky)); both axes are finite, increasing,
    uniformly spaced one-dimensional arrays with at least two samples.
    k0 is finite and positive. nbins is a positive integer, defaulting
    to min(len(kx), len(ky)). Let K=min(max(abs(kx)), max(abs(ky)), k0).
    Use nbins uniform radial bins on [0, K], left-closed/right-open
    except that the final bin includes K. Retain only individual samples
    with hypot(kx, ky) <= K and <= k0 before summing abs(U)**2*dkx*dky.
    Divide these sums by the common radial width, normalize that radial
    PSD by its peak, then divide by each angular-bin width. Angular edges
    are arcsin(radial_edges/k0); angles use radial-bin midpoints.
    Return (k_perp, angular_power_density, Pangle), three float arrays of
    length nbins. Density is zero if all retained power is zero. Polar
    angles are measured from z in radians; density has units rad**(-1).
    """
    bin_count = nbins if nbins is not None else min(len(kx), len(ky))
    return np.zeros(bin_count), np.zeros(bin_count), np.zeros(bin_count)
```

### Step 6

initial_fourier_transform

Goal
----
Transform a square centered physical field into its unnormalized centered two-dimensional Fourier field and the matching angular-wavenumber axes. Use fftshift(fft2(ifftshift(U0))), with spacings dx and dy assigned to array axes zero and one. Return (U,kx,ky), preserving complex amplitudes and the center-origin convention.

```python
import numpy as np


def initial_fourier_transform(U0, dx, dy):
    """Return (U, kx, ky) for a square centered physical field U0.

    U = fftshift(fft2(ifftshift(U0))) is an unnormalized centered FFT.
    With size = U0.shape[0], kx = 2*pi*fftshift(fftfreq(size, d=dx))
    and ky = 2*pi*fftshift(fftfreq(size, d=dy)). Positive dx and dy
    measure spacings on axes 0 and 1. The physical origin is at
    (size//2, size//2). Preserve complex amplitudes without additional
    normalization or spatial-cell-area factors.
    """
    size = np.shape(U0)[0]
    return np.zeros_like(U0, dtype=complex), np.zeros(size), np.zeros(size)
```

### Step 7

propagate_and_angular_power

Goal
----
Run the field-propagation and angular-density pipeline to exactly zlength. Form the centered initial transform, take floor(zlength/h) nominal increments, and apply any positive remainder using its own coefficients and fresh factors. Zero distance returns the initial Fourier field without marching. Return the seven outputs in the Signature: Fourier field, Cartesian peak-normalized power, both frequency axes, radial centers, angular density, and polar angles. Leave plotting disabled unless explicitly requested. This step is not the final orchestrator of the scalar task.

```python
import numpy as np


def propagate_and_angular_power(
    U0, P, k0, h, zlength, dx, dy, mx, B, C, nbins=None, plot=False
):
    """Propagate the centered square physical field U0 to zlength.

    P is a positive integer; k0, h, dx, and dy are finite and positive.
    zlength is finite and nonnegative; mx is an integer Fourier shift.
    B and C are finite real modulation parameters. nbins defaults to
    the transverse size. plot=False suppresses plotting.
    Transform once using the convention of initial_fourier_transform.
    Take floor(zlength/h) full steps with t=k0*h, then any positive
    remainder with its own coefficients and fresh factors. Zero distance
    returns the initial FFT without propagation. Use X=M_tilde-I-D and
    the Pade function specified in Step 1.
    Return (U, normalized_power, kx, ky, k_perp, angular_density, Pangle).
    U is the final centered Fourier field. normalized_power is abs(U)**2
    divided by its Cartesian peak, or zero for a zero field. The last
    three arrays follow radial_angular_power, not Cartesian normalization.
    """
    size = np.shape(U0)[0]
    bin_count = nbins if nbins is not None else size
    spectrum = np.zeros_like(U0, dtype=complex)
    power = np.zeros_like(U0, dtype=float)
    kx = np.zeros(size)
    ky = np.zeros(size)
    centers = np.zeros(bin_count)
    density = np.zeros(bin_count)
    angles = np.zeros(bin_count)
    return spectrum, power, kx, ky, centers, density, angles
```

### Step 8

wide_angle_power_fraction

Goal
----
Run the complete propagation and angular-density pipeline with plotting disabled, then return one dimensionless float: the fraction of retained annular power in bins whose polar center angle is at least theta_min. Recover common-scale annular powers by multiplying the angular density by each angular-bin width. Divide the selected-bin sum by the total retained-bin sum. Return 0.0 when retained power is zero. The cutoff defaults to pi/6 and must be a finite real scalar in [0,pi/2]. This is the final orchestrator of the scalar task.

```python
import numpy as np


def wide_angle_power_fraction(
    U0, P, k0, h, zlength, dx, dy, mx, B, C,
    nbins=None, theta_min=np.pi / 6
):
    """Return the retained annular-power fraction above a polar cutoff.

    Propagation inputs follow propagate_and_angular_power. Run that
    pipeline with plot=False; nbins defaults to the transverse size.
    Multiply angular density by angular-bin widths to recover relative
    annular powers. Derive radial edges from the uniform radial centers
    and angular edges from arcsin(radial_edges/k0).
    Select whole bins whose center angle is at least theta_min.
    theta_min is a finite real scalar in [0, pi/2], measured in radians.
    Return one float in [0, 1], or 0.0 when retained power is zero.

    Raises
    ------
    ValueError
        If theta_min is non-scalar, non-real, Boolean, non-finite,
        or outside [0, pi/2].
    """
    return 0.0
```
