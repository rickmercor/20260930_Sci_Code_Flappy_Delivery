# Physics-Optics-9

## Background

Optical resonances organize the response of an open electromagnetic system, but the analytic structure of its scattering matrix also depends on the basis used to describe incoming and outgoing waves. A channel normalization can make a modal expansion possible while introducing additional mathematical poles. Distinguishing those poles from resonant states matters when constructing reduced descriptions of a scatterer's spectrum.

## Problem

Quantify the error caused by omitting channel poles from a finite optical resonance expansion for a homogeneous, lossless, nonmagnetic sphere of refractive index \(n=2.7\) in vacuum, using the transverse-magnetic channel of angular momentum \(l=3\), time dependence \(e^{-i\omega t}\), and dimensionless wavenumber \(x=kR\), where \(R\) is the sphere radius. Let \(S(x)\) be the outgoing-to-incoming scattering amplitude in the usual spherical Hankel basis, and let \(T(x)\) be its derivative-normalized TM regularization, denoted R3 in the source's supplementary treatment of channel rescaling.

Use the first-order, zero-subtracted Mittag-Leffler expansion of \(T\), truncated to the set \(\Omega_p\) of all physical poles in \(|\Re x|<18.5\), \(-4<\Im x<0\), including both signs of the real part. Construct \(T_p\) using only \(\Omega_p\), and \(T_{p+c}\) using \(\Omega_p\) together with every channel pole of R3, retaining the same exact subtraction constant \(T(0)\) in both approximants. Each pole contributes its residue with respect to \(x\); keep this finite truncation fixed, without fitting an additional analytic background or renormalizing either approximant to enforce unitarity.

For either approximant \(U\), define the dimensionless weighted spectral error
\[
E[U]=\int_{0.35}^{7.4}x^2\left|T(x)-U(x)\right|^2\,dx.
\]
Report the single scalar \(\log_{10}(E[T_p]/E[T_{p+c}])\) with absolute error at most \(10^{-7}\). In brief reasoning, give the physical- and channel-pole counts and the two errors to absolute error \(10^{-5}\), establish the channel normalization, the two residue prescriptions and the subtraction constant, and explain the physical meaning of the upper-half-plane channel poles.

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

riccati_wave_data

Goal
----
Evaluate Riccati spherical Bessel and incoming/outgoing Hankel data.

```python
def riccati_wave_data(ell: int, x: "np.ndarray") -> "np.ndarray":
    """Evaluate radial waves and two argument derivatives.

    Parameters
    ----------
    ell : int
        Angular momentum, 1 through 4.
    x : ndarray
        Complex evaluation points; 1e-8 <= abs(x) <= 100 and
        abs(Im(x)) <= 20. A scalar array or any array shape is allowed.

    Returns
    -------
    ndarray
        Complex128 array of shape x.shape + (9,), ordered as
        psi, psi', psi'', xi_+, xi_+', xi_+'', xi_-, xi_-', xi_-''.

    Raises
    ------
    ValueError
        If ell is outside 1 through 4 or any x is zero.

    """
    return None
```

### Step 2

tm_boundary_data

Goal
----
Form the TM scattering numerator, outgoing determinant and its derivative.

```python
def tm_boundary_data(ell: int, n: float, x: "np.ndarray") -> "np.ndarray":
    """Evaluate the TM boundary determinant data.

    Parameters
    ----------
    ell : int
        Angular momentum, 1 through 4.
    n : float
        Real refractive index, 2 <= n <= 3.
    x : ndarray
        Nonzero complex points such that x and n*x satisfy the preceding
        radial-wave domain. Any array shape, including scalar, is allowed.

    Returns
    -------
    ndarray
        Complex128 array of shape x.shape + (3,), ordered N, D, D'.
        The amplitude N/D is not evaluated at its poles here.

    Raises
    ------
    ValueError
        If n is outside [2, 3], ell is unsupported, or an argument is zero.

    """
    return None
```

### Step 3

physical_tm_poles

Goal
----
Find the complete finite set of outgoing TM resonances in a rectangle.

```python
def physical_tm_poles(ell: int, n: float, x_cut: float) -> "np.ndarray":
    """Find every simple physical pole in the prescribed open rectangle.

    Parameters
    ----------
    ell : int
        Angular momentum, 1 through 4.
    n : float
        Real refractive index in [2, 3].
    x_cut : float
        Positive real-part cutoff in [8, 22]. The rectangle has depth 4.
        Supported instances have simple poles at least 1e-5 from its edges.

    Returns
    -------
    ndarray
        One-dimensional complex128 array of all poles, sorted first by real
        part and then by imaginary part. Absolute root error at most 1e-9
        is sufficient. Real parts smaller than 1e-9 in magnitude are zero.

    Raises
    ------
    ValueError
        If ell, n or x_cut is outside the stated interval, or a complete
        simple-pole set cannot be resolved numerically.

    """
    return None
```

### Step 4

tm_channel_data

Goal
----
Find all derivative-normalization channel poles and their residues.

```python
def tm_channel_data(ell: int, n: float) -> "np.ndarray":
    """Return all TM channel poles with dimensionless residues.

    Parameters
    ----------
    ell : int
        Angular momentum, 1 through 4.
    n : float
        Real refractive index in [2, 3].

    Returns
    -------
    ndarray
        Complex128 array of shape (ell + 1, 2). Each row is (c, residue).
        Sort by Re(c), then Im(c); set real parts smaller than 1e-9 in
        magnitude to zero. Poles are independent of n; residues are not.

    Raises
    ------
    ValueError
        If ell or n is outside its supported interval.

    """
    return None
```

### Step 5

tm_physical_residues

Goal
----
Evaluate the derivative-normalized residues of supplied physical poles.

```python
def tm_physical_residues(
    ell: int, n: float, poles: "np.ndarray"
) -> "np.ndarray":
    """Return one transformed residue for each supplied physical pole.

    Parameters
    ----------
    ell : int
        Angular momentum, 1 through 4.
    n : float
        Real refractive index in [2, 3].
    poles : ndarray
        One-dimensional complex array of simple physical poles, at least
        1e-8 from zero, with |Re(p)| <= 22 and -4 < Im(p) < 0. An empty
        array is allowed. Pole approximations accurate to 1e-10 suffice.

    Returns
    -------
    ndarray
        Complex128 array with the same one-dimensional shape and order
        as poles, containing residues of T with respect to x.

    Raises
    ------
    ValueError
        If ell or n is unsupported, or a supplied pole is zero.

    """
    return None
```

### Step 6

subtracted_pole_sum

Goal
----
Evaluate a finite first-order pole expansion with a fixed subtraction value.

```python
def subtracted_pole_sum(
    x: "np.ndarray",
    poles: "np.ndarray",
    residues: "np.ndarray",
    at_zero: complex,
) -> "np.ndarray":
    """Evaluate the normalized finite modal sum.

    Parameters
    ----------
    x : ndarray
        Finite real or complex grid with arbitrary shape, including a
        scalar or empty grid. Evaluation points do not coincide with poles.
    poles : ndarray
        One-dimensional nonzero complex poles with moderate finite values.
    residues : ndarray
        Complex residues, same one-dimensional shape as poles.
    at_zero : complex
        Fixed subtraction value, equal to -1 for the TM audit.

    Returns
    -------
    ndarray
        Complex128 array with exactly x.shape. An empty pole set gives
        at_zero at every evaluation point.

    Raises
    ------
    ValueError
        If pole and residue shapes differ, or a pole is zero.

    """
    return None
```

### Step 7

tm_spectral_errors

Goal
----
Integrate the two weighted spectral errors for supplied modal data.

```python
def tm_spectral_errors(
    ell: int,
    n: float,
    poles: "np.ndarray",
    residues: "np.ndarray",
    channels: "np.ndarray",
    left: float,
    right: float,
    order: int,
) -> "np.ndarray":
    """Compute weighted squared errors with the specified quadrature.

    Parameters
    ----------
    ell : int
        Angular momentum, 1 through 4.
    n : float
        Real refractive index in [2, 3].
    poles : ndarray
        One-dimensional complex physical-pole data; may be empty.
    residues : ndarray
        Complex residue data with the same shape as poles.
    channels : ndarray
        Complex array of shape (number_of_channel_poles, 2), with each
        row containing a channel pole and its residue. May have zero rows.
    left, right : float
        Positive real band endpoints, 0.1 <= left < right <= 8.
    order : int
        Number of Gauss-Legendre nodes, an integer from 16 through 800.

    Returns
    -------
    ndarray
        Float64 array of shape (2,), physical-only error followed by the
        error including the supplied channel data.

    Raises
    ------
    ValueError
        If the band or order is outside its range, ell or n is unsupported,
        or the supplied pole/residue data violates the preceding sum contract.

    """
    return None
```

### Step 8

scattering_pole_audit

Goal
----
Run the complete TM channel-pole audit and return its diagnostic scalar.

```python
def scattering_pole_audit(n: float, ell: int, x_cut: float) -> "np.ndarray":
    """Compute the complete derivative-normalized TM modal audit.

    Parameters
    ----------
    n : float
        Real refractive index in [2, 3].
    ell : int
        Angular momentum, 1 through 4.
    x_cut : float
        Real-part pole cutoff in [8, 22]; poles are simple and at least
        1e-5 from the boundary, as in the physical-pole contract.

    Returns
    -------
    ndarray
        Float64 array of shape (5,), ordered as log10 of the physical-only
        error divided by the complete error, physical-only error, complete
        error, physical-pole count and channel-pole count. Both errors use
        the fixed 400-node rule specified in the scientific background.

    Raises
    ------
    ValueError
        If an input lies outside the supported domain or the physical-pole
        set cannot be resolved numerically.

    """
    return None
```
