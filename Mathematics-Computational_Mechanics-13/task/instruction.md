# Mathematics-Computational_Mechanics-13

## Background

High-order time integration and residual-based variational multiscale stabilization both influence the spectral behavior of numerical flow solvers. In incompressible flow, enforcing the algebraic divergence constraint at Runge--Kutta stages leads naturally to differential-algebraic time integrators, while multiscale stabilization represents unresolved velocity content through a residual-driven fine scale. The order in which temporal and spatial modeling operations are applied need not commute, so two schemes with the same nominal temporal order can have different amplification, dissipation, dispersion, and stability characteristics. Fourier analysis of a periodic advection--diffusion surrogate provides a controlled way to compare those effects across wavenumber, Courant number, and nondimensional diffusivity. A path built from dimensionless spectral discrepancies summarizes how the ordering effect evolves from moderately resolved to high-wavenumber convection-dominated configurations.

## Problem

Residual-based variational multiscale discretizations of incompressible flow can behave differently when high-order Runge--Kutta time integration is applied before the spatial multiscale reduction than when the multiscale formulation is formed first and then advanced in time. For the source's one-dimensional advection--diffusion Fourier model with periodic boundary conditions and C1-continuous quadratic B-splines, quantify this ordering effect for the source's second-, third-, and fourth-order explicit Runge--Kutta schemes. Use the source's time-first shifted Runge--Kutta/VMS amplification construction with the dimensionless fine-scale parameter \(\tau^*=0.37\), and compare it with the source's space-first VMS/Runge--Kutta construction using its own stabilization parameter.

Evaluate the six configurations below, whose columns are \((p,K^*,a^*,\kappa^*)\), with \(s=p\):
\[
\begin{bmatrix}
2&0.75&0.12&0.060\\
3&1.10&0.22&0.050\\
4&1.45&0.32&0.040\\
2&1.80&0.40&0.030\\
3&2.15&0.48&0.025\\
4&2.50&0.56&0.020
\end{bmatrix}.
\]

Before evaluating either source-dependent amplification factor, report for all six rows the source-independent normalization quantities
\[
A_j=\kappa^*(K^*)^2,\qquad B_j=-a^*K^*,\qquad \mathrm{Pe}_j=\frac{a^*}{2\kappa^*}.
\]
These quantities are determined entirely by the supplied configurations and are used only as normalization and regime checks; they do not replace the source-specific Fourier constructions.

For each row, let \(\zeta_R\) be the source's time-first amplification factor and \(\zeta_V\) the source's space-first amplification factor. Recover the time-first construction from the source's shifted-RK Fourier recurrence (Eqs. (5.9)--(5.11) and Table 2) and the space-first construction from its stabilization scale, Fourier symbol, and explicit-RK amplification polynomial (Eq. (5.5) and Table 3). From any amplification factor \(\zeta\), define the dimensionless spectral diagnostics
\[
D(\zeta)=-\frac{\ln|\zeta|}{\kappa^*(K^*)^2},\qquad
F(\zeta)=\frac{\operatorname{Arg}(\zeta)}{-a^*K^*},
\]
where \(\operatorname{Arg}\in(-\pi,\pi]\) is the principal complex argument. Define the ordering-mismatch vector
\[
q=(D(\zeta_R)-D(\zeta_V),\;F(\zeta_R)-F(\zeta_V),\;|\zeta_R|-|\zeta_V|).
\]

Treat the six mismatch vectors in the listed order as a path in \(\mathbb R^3\), and compute
\[
L=\sum_{j=2}^{6}\|q_j-q_{j-1}\|_2.
\]


Use the exact source formulas for the shifted Runge--Kutta coefficients and the two Fourier constructions, perform no intermediate rounding, and report \(L\) rounded to exactly 12 digits after the decimal point. Output (<final_answer>) immediately, followed by \(<reasoning>\); the final-answer tag must contain exactly one finite decimal value, while the reasoning should state the normalization quantities, the source-specific constructions, representative Fourier quantities, the six mismatch vectors, and the five path increments that justify the scalar.

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

01_shifted_rk_matrix.py

Goal
----
Construct the shifted explicit Runge--Kutta coefficient matrix used in the source's unified stage notation.

```python
def shifted_rk_matrix(order: int) -> "np.ndarray":
    """Return the source's shifted RK coefficient matrix for a supported order.

    Parameters
    ----------
    order : int
        Runge--Kutta order and stage count; supported values are 2, 3, and 4.

    Returns
    -------
    alpha : np.ndarray
        Square float array of shape (order, order) containing the shifted
        stage coefficients and final weights.

    Raises
    ------
    ValueError
        If order is not 2, 3, or 4.
    """
    return alpha
```

### Step 2

02_rothe_vms_fourier_coefficients.py

Goal
----
Evaluate the three complex Fourier coefficients of the source's time-first RK/VMS construction.

```python
def rothe_vms_fourier_coefficients(K_star: float, a_star: float, kappa_star: float, tau_star: float) -> "np.ndarray":
    """Return the three complex Fourier coefficients for the time-first scheme.

    Parameters
    ----------
    K_star : float
        Dimensionless wavenumber, in radians.
    a_star : float
        Dimensionless advective speed (Courant number).
    kappa_star : float
        Positive nondimensional diffusivity.
    tau_star : float
        Dimensionless fine-scale parameter in the open interval (0, 1).

    Returns
    -------
    lambdas : np.ndarray
        Complex array of shape (3,) ordered as (lambda1, lambda2, lambda3).
    """
    return lambdas
```

### Step 3

03_rothe_vms_amplification.py

Goal
----
Compute the source's time-first RK/VMS one-step amplification factor.

```python
def rothe_vms_amplification(order: int, K_star: float, a_star: float, kappa_star: float, tau_star: float) -> complex:
    """Return the one-step complex amplification factor of the time-first scheme.

    Parameters
    ----------
    order : int
        Supported RK order/stage count 2, 3, or 4.
    K_star : float
        Dimensionless wavenumber.
    a_star : float
        Dimensionless advective speed.
    kappa_star : float
        Positive nondimensional diffusivity.
    tau_star : float
        Dimensionless time-first fine-scale parameter.

    Returns
    -------
    zeta : complex
        One-step Fourier amplification factor.
    """
    return zeta
```

### Step 4

04_vertical_vms_fourier_symbol.py

Goal
----
Evaluate the stabilization scale and complex Fourier symbol of the source's space-first VMS/RK construction.

```python
def vertical_vms_fourier_symbol(K_star: float, a_star: float, kappa_star: float) -> "np.ndarray":
    """Return the dimensionless stabilization scale and complex Fourier symbol.

    Parameters
    ----------
    K_star : float
        Dimensionless wavenumber.
    a_star : float
        Dimensionless advective speed.
    kappa_star : float
        Positive nondimensional diffusivity.

    Returns
    -------
    symbol : np.ndarray
        Float array ``[tau_diamond_star, Re(gamma), Im(gamma)]``.
    """
    return symbol
```

### Step 5

05_vertical_vms_amplification.py

Goal
----
Advance the source's space-first Fourier symbol with the selected explicit RK scheme.

```python
def vertical_vms_amplification(order: int, K_star: float, a_star: float, kappa_star: float) -> complex:
    """Return the one-step complex amplification factor of the space-first scheme.

    Parameters
    ----------
    order : int
        Supported RK order/stage count 2, 3, or 4.
    K_star : float
        Dimensionless wavenumber.
    a_star : float
        Dimensionless advective speed.
    kappa_star : float
        Positive nondimensional diffusivity.

    Returns
    -------
    zeta : complex
        One-step Fourier amplification factor.
    """
    return zeta
```

### Step 6

06_spectral_diagnostics.py

Goal
----
Convert a complex amplification factor into dimensionless amplitude, damping, and frequency diagnostics.

```python
def spectral_diagnostics(zeta: complex, K_star: float, a_star: float, kappa_star: float) -> "np.ndarray":
    """Return amplitude and normalized damping/frequency diagnostics.

    Parameters
    ----------
    zeta : complex
        Nonzero one-step Fourier amplification factor.
    K_star : float
        Positive dimensionless wavenumber.
    a_star : float
        Positive dimensionless advective speed.
    kappa_star : float
        Positive nondimensional diffusivity.

    Returns
    -------
    diagnostics : np.ndarray
        Float array ``[abs(zeta), damping_ratio, frequency_ratio]`` using the
        principal complex argument in (-pi, pi].
    """
    return diagnostics
```

### Step 7

07_spectral_ordering_mismatch.py

Goal
----
Measure the three-component spectral discrepancy caused by reversing VMS and RK ordering.

```python
def spectral_ordering_mismatch(order: int, K_star: float, a_star: float, kappa_star: float, tau_star: float) -> "np.ndarray":
    """Return the spectral mismatch vector between time-first and space-first schemes.

    Parameters
    ----------
    order : int
        Supported RK order/stage count 2, 3, or 4.
    K_star : float
        Dimensionless wavenumber.
    a_star : float
        Positive dimensionless advective speed.
    kappa_star : float
        Positive nondimensional diffusivity.
    tau_star : float
        Time-first dimensionless fine-scale parameter.

    Returns
    -------
    mismatch : np.ndarray
        Float array ``[Delta damping ratio, Delta frequency ratio,
        Delta amplification magnitude]``, with time-first minus space-first.
    """
    return mismatch
```

### Step 8

08_cumulative_spectral_ordering_path.py

Goal
----
Accumulate the Euclidean path length traced by ordering-mismatch vectors over an ordered configuration sequence.

```python
def cumulative_spectral_ordering_path(configurations: "np.ndarray", tau_star: float) -> float:
    """Return the cumulative Euclidean path length of ordering mismatches.

    Parameters
    ----------
    configurations : np.ndarray
        Float array of shape (n, 4), n >= 2, with rows
        ``[order, K_star, a_star, kappa_star]`` in path order. Supported orders
        are 2, 3, and 4.
    tau_star : float
        Time-first dimensionless fine-scale parameter shared by all rows.

    Returns
    -------
    path_length : float
        Sum of Euclidean distances between consecutive three-component
        spectral ordering mismatch vectors.
    """
    return path_length
```
