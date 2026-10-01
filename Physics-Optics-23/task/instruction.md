# Physics-Optics-23

## Background

A partially coherent monochromatic field is described by a set of mutually incoherent coherent modes, each carrying a share of the optical power. Propagating every mode through an optical element as a two-dimensional complex wavefront dominates the cost of repeated transport calculations in imaging, lithography, and beamline design.

Many of those modes are nearly separable into a horizontal factor and a vertical factor, while aberrations, apertures, and bilinear phase leave a smaller coupled remainder. A compression that keeps the dominant separable piece, represents the remainders in one shared low-rank subspace, and decouples the element into separable terms can transport the separable parts with one-dimensional operators and rebuild the output intensity without propagating every full wavefront.

The calculation requested here uses one prescribed realization of that transport. The grid, the mode shapes, the modal eigenvalues, the element, the propagation convention, and the two retention thresholds are fixed by the problem statement.

## Problem

Partially coherent optical transport is expensive when every coherent mode is propagated as a full two-dimensional field. A 2026 journal method replaces each mode by its leading separable singular component together with a shared low-rank coupled residual, decouples the optical element into separable terms so that the separable parts can be transported with one-dimensional operators, and rebuilds the partially coherent intensity at the output plane.

Construct five modes on a grid of 32 rows and 40 columns whose coordinates are the unit-spaced pixel offsets from the grid center, x = j − 19.5 for column j = 0, …, 39 and y = i − 15.5 for row i = 0, …, 31, in the xy convention, so a column index advances the horizontal coordinate. Mode n has envelope width σ_n, carrier tilt (a_n, b_n), and bilinear coupling c_n:

σ = (3.2, 4.6, 2.4, 5.5, 3.9),
(a, b) = ((0.12, -0.05), (-0.16, 0.10), (0.05, 0.18), (-0.08, -0.14), (0.20, 0.03)),
c = (0.020, -0.035, 0.050, -0.015, 0.040).

The unnormalized field is exp[-(x^2 + y^2) / (2 σ_n^2)] exp[i(a_n x + b_n y + c_n x y)]. Orthonormalize the C-order columns by modified Gram-Schmidt, using the inner product that conjugates its first argument. Phase each finished column so that its largest-magnitude sample is real and nonnegative, breaking ties by the smallest flat index. The coherent-mode eigenvalues, in the same order, are (0.42, 0.25, 0.16, 0.10, 0.07).

Keep the leading separable component of each mode, including its singular value. Weight each coupled residual by its coherent-mode eigenvalue and stack those weighted fields as row-major rows. Restore a compressed residual by dividing it by that same eigenvalue. Retain the smallest number of shared residual components whose nominal cumulative compression score is at least 0.95. This score applies the journal's cumulative-energy expression to the eigenvalue-weighted residual stack prescribed here; it is an operational rank-selection score, not the physical power fraction of the restored fields.

The modes pass through a thin element in their own plane. In the rotated coordinates x' = x cos θ + y sin θ and y' = −x sin θ + y cos θ with θ = 0.35 rad, its complex transmission is exp[−(x'/9)^4 − (y'/6)^4] exp(i 0.025 x' y'), sampled on the same grid. Decompose the element into separable terms ordered by decreasing singular value and keep the smallest number of leading terms whose cumulative share of the sum of squared singular values is at least 0.99.

After the element the fields propagate paraxially over a propagation parameter λz = 12 in units of the squared pixel pitch: the discrete Fourier transform of an unpadded field is multiplied by exp[−iπ λz (f_x^2 + f_y^2)], where f_x and f_y are the discrete Fourier frequencies of the two axes at unit pitch. In the compressed transport the leading separable component of each mode passes through the retained element terms and is transported along the two axes separately, while the restored coupled residual of each mode passes through the full element and is propagated as a two-dimensional field; the two transported parts add. The exact reference propagates each original mode multiplied by the full element. A partially coherent intensity is the eigenvalue-weighted sum of the squared moduli of the transported modes.

Report one hundred times the relative Euclidean error between the compressed and the exact output-plane intensities.

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

orthonormal_source_modes

Goal
----
Build the orthonormal coherent modes of the benchmark source on a pixel grid.

```python
def orthonormal_source_modes(
    shape: tuple,
    sigmas: "np.ndarray",
    tilts: "np.ndarray",
    coupling: "np.ndarray",
) -> "np.ndarray":
    """Return the orthonormal coherent modes on the requested grid.

    Column ``n`` is the Gaussian envelope of width ``sigmas[n]``, carrier
    tilt ``tilts[n]``, and bilinear coupling ``coupling[n]``. Coordinates
    are the unit-spaced pixel offsets from the grid center,
    ``x = j - (nx - 1) / 2`` for column ``j`` and ``y = i - (ny - 1) / 2``
    for row ``i``, with ``indexing="xy"``. The vectorized columns, in C
    order, are orthonormalized by modified Gram-Schmidt. Each finished
    column is phased so that its largest-magnitude sample is real and
    nonnegative, with ties broken by the smallest flat index. The inner
    product conjugates its first argument.

    Parameters
    ----------
    shape : tuple
        ``(ny, nx)``, both integers at least 2.
    sigmas : np.ndarray
        Positive finite envelope widths, shape ``(n_modes,)``.
    tilts : np.ndarray
        Finite carrier coefficients, shape ``(n_modes, 2)``.
    coupling : np.ndarray
        Finite bilinear coefficients, shape ``(n_modes,)``.

    Returns
    -------
    modes : np.ndarray
        Complex128 array of shape ``(n_modes, ny, nx)``.

    Raises
    ------
    ValueError
        If a shape is wrong, a width is not positive, or a coefficient
        is not finite.
    """
    return modes
```

### Step 2

coupled_residuals

Goal
----
Remove the leading separable component of each coherent mode and return the coupled remainder.

```python
def coupled_residuals(modes: "np.ndarray") -> "np.ndarray":
    """Return the coupled residual of every coherent mode.

    Parameters
    ----------
    modes : np.ndarray
        Complex array of shape ``(n_modes, ny, nx)`` with both spatial
        sides at least 2.

    Returns
    -------
    residuals : np.ndarray
        Complex128 array with the same shape as ``modes``.

    Raises
    ------
    ValueError
        If ``modes`` does not have that shape.
    """
    return residuals
```

### Step 3

residual_stack

Goal
----
Stack the eigenvalue-weighted coupled residuals as the rows of one matrix.

```python
def residual_stack(
    residuals: "np.ndarray",
    eigenvalues: "np.ndarray",
) -> "np.ndarray":
    """Return one row-major row for each eigenvalue-weighted residual.

    Parameters
    ----------
    residuals : np.ndarray
        Complex array of shape ``(n_modes, ny, nx)``.
    eigenvalues : np.ndarray
        Positive finite coherent-mode eigenvalues, shape ``(n_modes,)``.

    Returns
    -------
    stack : np.ndarray
        Complex128 array of shape ``(n_modes, ny * nx)``.

    Raises
    ------
    ValueError
        If the arrays do not match or an eigenvalue is not positive.
    """
    return stack
```

### Step 4

compressed_residuals

Goal
----
Return the shared low-rank compression of the residual stack at a requested component count, as one field per mode.

```python
def compressed_residuals(
    stack: "np.ndarray",
    n_keep: int,
    field_shape: tuple,
) -> "np.ndarray":
    """Return the compressed residual fields for the requested component count.

    A request for zero components returns an array of zeros. Each row is
    reshaped in row-major order to ``field_shape``.

    Parameters
    ----------
    stack : np.ndarray
        Complex array of shape ``(n_modes, ny * nx)``.
    n_keep : int
        Number of residual components, from zero through the stack rank.
    field_shape : tuple
        ``(ny, nx)`` matching the row length of ``stack``.

    Returns
    -------
    compressed : np.ndarray
        Complex128 array of shape ``(n_modes, ny, nx)``.

    Raises
    ------
    ValueError
        If ``field_shape`` disagrees with ``stack`` or ``n_keep`` is
        outside the stack rank.
    """
    return compressed
```

### Step 5

cumulative_energy_ratio

Goal
----
Evaluate the nominal cumulative compression score at a requested number of shared residual components.

```python
def cumulative_energy_ratio(
    modes: "np.ndarray",
    eigenvalues: "np.ndarray",
    n_keep: int,
) -> float:
    """Return the nominal cumulative compression score at ``n_keep``.

    Every mode must have unit Euclidean norm. Zero retained components
    returns the eigenvalue-weighted separable-energy ratio.

    Parameters
    ----------
    modes : np.ndarray
        Complex unit-norm modes of shape ``(n_modes, ny, nx)``.
    eigenvalues : np.ndarray
        Positive finite eigenvalues, shape ``(n_modes,)``.
    n_keep : int
        Number of compressed residual components, from zero through the
        residual-stack rank.

    Returns
    -------
    ratio : float
        Native Python float in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the eigenvalues do not match, a mode is not unit-norm, or
        ``n_keep`` is outside the residual-stack rank.
    """
    return ratio
```

### Step 6

element_transmission

Goal
----
Sample the complex transmission of the rotated super-Gaussian phase element on the pixel grid.

```python
def element_transmission(
    shape: tuple,
    widths: tuple,
    angle: float,
    phase_coupling: float,
) -> "np.ndarray":
    """Return the sampled complex transmission of the element.

    Parameters
    ----------
    shape : tuple
        ``(ny, nx)``, both integers at least 2.
    widths : tuple
        Positive finite half-widths ``(w_a, w_b)`` along the rotated axes.
    angle : float
        Rotation angle of the element axes in radians, finite.
    phase_coupling : float
        Finite coefficient of the bilinear phase in the rotated coordinates.

    Returns
    -------
    transmission : np.ndarray
        Complex128 array of shape ``(ny, nx)``.

    Raises
    ------
    ValueError
        If a shape is wrong, a width is not positive, or a coefficient
        is not finite.
    """
    return transmission
```

### Step 7

element_terms

Goal
----
Decouple the element transmission into the separable terms that carry a requested fraction of its occupation.

```python
def element_terms(
    transmission: "np.ndarray",
    occupation: float,
) -> "np.ndarray":
    """Return the leading separable terms of the element transmission.

    Parameters
    ----------
    transmission : np.ndarray
        Complex array of shape ``(ny, nx)`` with both sides at least 2.
    occupation : float
        Target cumulative share of the sum of squared singular values,
        in ``(0, 1]``.

    Returns
    -------
    terms : np.ndarray
        Complex128 array of shape ``(K, ny, nx)``; term ``k`` is the
        ``k``-th separable term including its singular value.

    Raises
    ------
    ValueError
        If ``transmission`` is not a two-dimensional field or
        ``occupation`` is outside ``(0, 1]``.
    """
    return terms
```

### Step 8

separable_transport

Goal
----
Transport the leading separable component of every mode through the retained element terms to the output plane.

```python
def separable_transport(
    modes: "np.ndarray",
    terms: "np.ndarray",
    fresnel_parameter: float,
) -> "np.ndarray":
    """Return the transported leading separable component of each mode.

    Parameters
    ----------
    modes : np.ndarray
        Complex modes of shape ``(n_modes, ny, nx)``.
    terms : np.ndarray
        Complex separable element terms of shape ``(K, ny, nx)``, each
        including its singular value.
    fresnel_parameter : float
        Finite propagation parameter in units of the squared pixel pitch.

    Returns
    -------
    transported : np.ndarray
        Complex128 array with the same shape as ``modes``.

    Raises
    ------
    ValueError
        If the spatial shapes of ``modes`` and ``terms`` differ or the
        propagation parameter is not finite.
    """
    return transported
```

### Step 9

coupled_transport

Goal
----
Transport the restored coupled residual of every mode through the full element to the output plane.

```python
def coupled_transport(
    modes: "np.ndarray",
    eigenvalues: "np.ndarray",
    n_keep: int,
    transmission: "np.ndarray",
    fresnel_parameter: float,
) -> "np.ndarray":
    """Return the transported restored coupled residual of each mode.

    Parameters
    ----------
    modes : np.ndarray
        Complex unit-norm modes of shape ``(n_modes, ny, nx)``.
    eigenvalues : np.ndarray
        Positive finite eigenvalues, shape ``(n_modes,)``.
    n_keep : int
        Number of compressed residual components, from zero through the
        residual-stack rank.
    transmission : np.ndarray
        Complex element transmission of shape ``(ny, nx)``.
    fresnel_parameter : float
        Finite propagation parameter in units of the squared pixel pitch.

    Returns
    -------
    transported : np.ndarray
        Complex128 array with the same shape as ``modes``.

    Raises
    ------
    ValueError
        If the eigenvalues or the transmission do not match the modes,
        ``n_keep`` is outside the residual-stack rank, or the propagation
        parameter is not finite.
    """
    return transported
```

### Step 10

transport_intensity_error_percent

Goal
----
Run the compressed transport of the partially coherent source through the element and report the percent intensity error at the output plane.

```python
def transport_intensity_error_percent(
    shape: tuple,
    sigmas: "np.ndarray",
    tilts: "np.ndarray",
    coupling: "np.ndarray",
    eigenvalues: "np.ndarray",
    energy_threshold: float,
    widths: tuple,
    angle: float,
    phase_coupling: float,
    occupation: float,
    fresnel_parameter: float,
) -> float:
    """Return the percent intensity error of the compressed transport.

    Parameters
    ----------
    shape : tuple
        ``(ny, nx)``, both integers at least 2.
    sigmas : np.ndarray
        Positive finite envelope widths, shape ``(n_modes,)``.
    tilts : np.ndarray
        Finite carrier coefficients, shape ``(n_modes, 2)``.
    coupling : np.ndarray
        Finite bilinear coefficients of the modes, shape ``(n_modes,)``.
    eigenvalues : np.ndarray
        Positive finite coherent-mode eigenvalues, shape ``(n_modes,)``.
    energy_threshold : float
        Target cumulative retained-energy ratio in ``(0, 1]``.
    widths : tuple
        Positive finite half-widths ``(w_a, w_b)`` of the element.
    angle : float
        Rotation angle of the element axes in radians.
    phase_coupling : float
        Coefficient of the bilinear phase of the element.
    occupation : float
        Target cumulative occupation of the element terms in ``(0, 1]``.
    fresnel_parameter : float
        Finite propagation parameter in units of the squared pixel pitch.

    Returns
    -------
    error_percent : float
        One hundred times the relative Euclidean error between the
        compressed and exact output-plane intensities, as a native Python
        float.

    Raises
    ------
    ValueError
        If any input is invalid for the underlying steps or a threshold is
        outside ``(0, 1]``.
    """
    return error_percent
```
