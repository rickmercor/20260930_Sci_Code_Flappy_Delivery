# Physics-Optics-11

## Background

Magnetodielectric gratings have spatially varying electric and magnetic material properties. Their transmitted diffraction patterns describe how electromagnetic power is distributed among outgoing directions, rather than only how much power passes through the structure.

This problem compares two arrangements with the same material profiles, individual layer thicknesses, and illumination, differing only in the order of the layers. The material coefficients and layer-reversal diagnostic define a constructed numerical benchmark, not an experimental measurement. The requested quantity is the dimensionless contrast under the approximation and normalization specified in the problem statement.

The source framework describes constitutive TE/TM scattering using a dynamical transfer matrix. Its ordered Dyson expansion organizes the low-frequency contributions while retaining finite material contrast, providing the theoretical connection to the optical-thickness expansion used in this benchmark.

## Problem

Determine the layer-reversal contrast in nonspecular transmission for the following two-layer magnetodielectric grating. The system is invariant in $z$, vacuum surrounds $0<x<\ell$, time dependence is $e^{-i\omega t}$, TE denotes $E_z=\psi$, TM denotes $H_z=\psi$, and unit-amplitude order-zero illumination arrives from the left with no incoming wave from the right.

In layer $L$, write $\epsilon_L(y)=\sum_{h=0}^2\epsilon_{Lh}e^{ihKy}$ and $\mu_L(y)=\sum_{h=0}^2\mu_{Lh}e^{ihKy}$, with the dimensionless relative constitutive coefficients below and vacuum wavenumber $k=\omega/c$:

| Layer | $(\epsilon_{L0},\epsilon_{L1},\epsilon_{L2})$ | $(\mu_{L0},\mu_{L1},\mu_{L2})$ |
| --- | --- | --- |
| $A$ | $(2.4+0.45i,\;0.35,\;0.04i)$ | $(1.2+0.14i,\;0.10e^{0.3i},\;0.015)$ |
| $B$ | $(3.1+0.38i,\;0.30e^{0.8i},\;-0.025i)$ | $(1.6+0.15i,\;0.12e^{-0.55i},\;0.012i)$ |

Use $f_A=0.37$, $f_B=0.63$, $\eta=k\ell=0.08$, $s_0=\sin\theta_0=-0.6$, and $K/k=0.39$, where forward order is $A$ on $0<x<f_A\ell$ followed by $B$ and reverse order exchanges the two layers together with their thicknesses, without conjugating or translating either transverse profile. For each polarization and ordering, retain the Taylor polynomial through $\eta^2$ of every complex transmission amplitude at fixed $s_0$, $K/k$, fractions, and constitutive coefficients, using the global-coordinate convention $\psi(x,y)=\sum_j t_j(\eta)e^{ik(s_jy+c_jx)}$ for $x>\ell$. Here $j$ is the integer diffraction order, $s_j=s_0+jK/k$, and $c_j=\sqrt{1-s_j^2}$ is positive real for an open channel and positive imaginary for an evanescent channel.

Define $t_{p,j}^{[2]}=t_{p,j}^{(0)}+\eta t_{p,j}^{(1)}+\eta^2t_{p,j}^{(2)}$ and, for $O\in\{AB,BA\}$, compute the equal incoherent polarization average
$$
Q_O=\frac12\sum_{p\in\{\mathrm{TE},\mathrm{TM}\}}\sum_{\substack{j\geq1\\ |s_j|<1}}\frac{c_j}{c_0}\left|t_{p,j}^{[2]}\right|^2,\qquad
\Delta=\frac{Q_{AB}-Q_{BA}}{Q_{AB}+Q_{BA}},
$$
with $\Delta=0$ for an exactly zero denominator. Square the evaluated amplitude polynomials without Taylor-truncating the powers again, make no expansion in material contrast, and do not replace this prescribed diagnostic by exact finite-thickness scattering or renormalized efficiencies. Report $\Delta$ to absolute error at most $10^{-8}$ and give $Q_{AB}$ and $Q_{BA}$ to absolute error at most $10^{-12}$ in the short reasoning, with a concise derivation and citations to the scientific sources used.

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

grating_channels

Goal
----
Construct the causal diffraction orders and outgoing vacuum branches.

```python
import numpy as np


def grating_channels(incident_sine, reciprocal_ratio, order_count):
    """Return (sines, cosines), each with shape (order_count,).

    Orders are j=0,...,order_count-1, with s_j=incident_sine+j*K/k
    and reciprocal_ratio=K/k. The incident sine is finite, real, and
    strictly between -1 and 1; reciprocal_ratio is finite and positive.
    order_count is a positive integer, not Boolean. The outgoing branch
    c_j=sqrt(1-s_j**2) is positive real for propagating orders and positive
    imaginary for evanescent orders. Raise ValueError for invalid inputs,
    nonfinite generated values, or abs(1-s_j**2)<=1e-12 (grazing).
    Do not reorder channels or replace an evanescent cosine by a real one.
    """
    return np.zeros(order_count), np.zeros(order_count, dtype=complex)
```

### Step 2

reciprocal_fourier

Goal
----
Resolve the one-sided Fourier coefficients of a reciprocal material.

```python
import numpy as np


def reciprocal_fourier(coefficients, order_count):
    """Return coefficients 0,...,order_count-1 of 1/a(y).

    coefficients is a nonempty finite numeric one-dimensional array for
    a(y)=sum(coefficients[h]*exp(1j*h*K*y)), h>=0. Require
    abs(coefficients[0])>sum(abs(coefficients[1:])) so the reciprocal has
    a convergent one-sided expansion. order_count is a positive non-Boolean
    integer. Return a finite complex array of shape (order_count,) without
    changing the input. Raise ValueError for invalid inputs or nonfinite
    output. Reciprocal coefficients beyond the material's polynomial
    degree generally do not vanish.
    """
    return np.zeros(order_count, dtype=complex)
```

### Step 3

bergmann_generator

Goal
----
Build the dimensionless first-order TE/TM layer generator.

```python
import numpy as np


def bergmann_generator(sines, alpha_coefficients, beta_coefficients):
    """Return the dimensionless state generator with shape (2*J, 2*J).

    Use time dependence exp(-1j*omega*t), state (psi, chi), and
    chi=alpha**(-1)*partial_x(psi)/(1j*k). For the equation
    div(alpha**(-1)*grad(psi))+k**2*beta*psi=0, derive the matrix G
    satisfying partial_(k*x)(state)=1j*G@state. TE has (alpha,beta)
    =(mu,epsilon); TM has (epsilon,mu). The input coefficients already
    have this mapping. A positive harmonic h couples order j to j+h.
    sines is a nonempty finite real vector of length J. Both profiles
    are nonempty finite numeric coefficient vectors, and alpha's constant
    strictly dominates its harmonic tail. Use reciprocal_fourier for the
    reciprocal alpha coefficients, including harmonics through J-1.
    Return complex values, preserve inputs, and raise ValueError for
    invalid inputs. Evanescent transverse sines are allowed.

    Raise ValueError if the computation produces nonfinite coefficients
    or efficiencies; do not return nonfinite results.
    """
    return np.zeros((2 * len(sines), 2 * len(sines)), dtype=complex)
```

### Step 4

ordered_transfer_series

Goal
----
Compose an ordered layer stack through second order in total thickness.

```python
import numpy as np


def ordered_transfer_series(generators, fractions):
    """Return the state-transfer coefficients [T0,T1,T2].

    generators has finite numeric shape (layers, dimension, dimension).
    fractions is a finite real nonnegative vector with one entry per layer,
    summing to one within absolute tolerance 1e-12. Inputs are ordered from
    the incident face to the exit face. The state transfer is the ordered
    product exp(1j*eta*f_last*G_last) ... exp(1j*eta*f_first*G_first).
    Return a complex array of shape (3, dimension, dimension) whose entries
    multiply eta**0, eta**1, eta**2, respectively; these are polynomial
    coefficients, not unscaled derivatives. T0 is identity. Zero-thickness
    layers are allowed. Raise ValueError for invalid inputs and preserve
    all inputs. Do not commute generators or normalize supplied fractions.

    Raise ValueError if the computation produces nonfinite coefficients
    or efficiencies; do not return nonfinite results.
    """
    return np.zeros(
        (
            3,
            np.asarray(generators).shape[-1],
            np.asarray(generators).shape[-1],
        ),
        dtype=complex,
    )
```

### Step 5

global_transfer_series

Goal
----
Convert state-transfer coefficients into global plane-wave coordinates.

```python
import numpy as np


def global_transfer_series(state_series, cosines):
    """Return [M0,M1,M2] for globally referenced wave amplitudes.

    state_series has finite numeric shape (3,2*J,2*J), with its constant
    term equal to identity within absolute tolerance 1e-12. cosines is a
    finite vector of length J with each entry positive real or positive
    purely imaginary, and modulus greater than 1e-12. The state at a plane
    with coordinate x is the sum of [I;C] A exp(1j*k*C*x) and
    [I;-C] B exp(-1j*k*C*x), with C=diag(cosines). Both exterior media
    are vacuum, the entry face is x=0, and the exit face is k*x=eta.
    Derive M mapping [A_left,B_left] to [A_right,B_right] through eta**2.
    Coefficients are not derivatives. Remove the exit-face vacuum phase
    consistently for both propagation directions. Preserve inputs and
    raise ValueError for invalid shape, constant term, or branches.

    Raise ValueError if the computation produces nonfinite coefficients
    or efficiencies; do not return nonfinite results.
    """
    return np.zeros((3, 2 * len(cosines), 2 * len(cosines)), dtype=complex)
```

### Step 6

scattering_series

Goal
----
Apply outgoing boundary conditions to a second-order transfer series.

```python
import numpy as np


def scattering_series(global_series):
    """Return reflection/transmission amplitude coefficients, shape (2,3,J).

    global_series has finite numeric shape (3,2*J,2*J), J>=1, and constant
    coefficient identity within absolute tolerance 1e-12. It maps
    [A_left,B_left] to [A_right,B_right] in global plane-wave coordinates.
    Incidence is from the left in order zero, A_left=e_0; there is no
    incidence from the right, B_right=0. Solve these boundary conditions
    as a series through eta**2. The first output index is reflected=0 or
    transmitted=1, the second is polynomial degree 0,1,2, and the third
    is diffraction order. Return coefficients, not amplitudes evaluated
    at a chosen thickness. Preserve inputs and raise ValueError for
    invalid values, shapes, or constant coefficient.

    Raise ValueError if the computation produces nonfinite coefficients
    or efficiencies; do not return nonfinite results.
    """
    return np.zeros(
        (2, 3, np.asarray(global_series).shape[-1] // 2), dtype=complex
    )
```

### Step 7

diffraction_efficiencies

Goal
----
Evaluate the specified amplitude polynomial and its vacuum flux weights.

```python
import numpy as np


def diffraction_efficiencies(amplitude_series, cosines, thickness):
    """Return reflected/transmitted power efficiencies, shape (2,J).

    amplitude_series has finite numeric shape (2,3,J), with polynomial
    coefficients through second order. thickness=eta=k*total_length is
    finite, real, nonnegative, and not Boolean. Evaluate the quadratic
    amplitude first, then take its squared modulus; do not Taylor-truncate
    the resulting power polynomial. The incident and exit media are vacuum
    with unit-amplitude incidence in order zero. Weight each propagating
    order by its normal vacuum Poynting flux relative to order zero.
    cosines is a finite vector: c_0 is positive real, and every entry is
    positive real or positive purely imaginary, of modulus above 1e-12.
    Evanescent orders carry zero normal far-field power. Do not renormalize
    the efficiencies to sum to one or clip absorption/gain effects.
    Preserve inputs and raise ValueError for invalid inputs/nonfinite output.
    """
    return np.zeros((2, len(cosines)))
```

### Step 8

layer_reversal_contrast

Goal
----
Compose the source-derived TE/TM layer-order contrast diagnostic.

```python
import numpy as np


def layer_reversal_contrast(
    materials,
    fractions,
    incident_sine,
    reciprocal_ratio,
    thickness,
    order_count,
):
    """Return the scalar contrast of nonspecular transmitted efficiencies.

    materials is a finite numeric array with shape (2,2,H), H>=1. Axis 0
    lists the two layers in forward order; axis 1 is epsilon then mu;
    axis 2 lists coefficients of exp(1j*h*K*y), h=0,...,H-1. Each profile
    has a constant coefficient strictly dominating its harmonic tail.
    fractions has shape (2,), is real and nonnegative, and sums to one
    within 1e-12. Reversing the stack reverses both layers and their
    associated fractions, without conjugating or translating harmonics.
    Use grating_channels to construct orders 0,...,order_count-1. All open
    nonnegative orders must be included: the next sine must exceed one.
    Extra evanescent orders are allowed; grazing is excluded as in Step 1.
    For each stack order use bergmann_generator (which uses
    reciprocal_fourier), ordered_transfer_series, global_transfer_series,
    scattering_series, and diffraction_efficiencies. For TE set
    (alpha,beta)=(mu,epsilon); for TM set (epsilon,mu). For each ordering
    let Q be the equal incoherent average of TE and TM transmitted power
    in j>=1. Return (Q_forward-Q_reverse)/(Q_forward+Q_reverse), or 0.0
    when the denominator is exactly zero. thickness and channel parameters
    satisfy the earlier contracts. Raise ValueError for invalid inputs,
    preserve input arrays, and return a finite Python float.

    Raise ValueError if the computation produces nonfinite coefficients
    or efficiencies; do not return nonfinite results.
    """
    return 0.0
```
