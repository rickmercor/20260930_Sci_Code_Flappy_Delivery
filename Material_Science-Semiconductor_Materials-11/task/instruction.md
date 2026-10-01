# Material_Science-Semiconductor_Materials-11

## Background

Long-range periodic electrostatics are normally evaluated by splitting the Coulomb kernel into short-range and reciprocal-space parts. The source method replaces the conventional smooth window by a normalized zeroth-order prolate spheroidal wave function, exploiting its simultaneous concentration in finite real- and Fourier-space intervals, and ties the prolate bandwidth to the requested force accuracy.

## Problem

A recent prolate-Ewald formulation reduces the reciprocal bandwidth needed for long-range Coulomb interactions by using the zeroth-order prolate spheroidal wave function (PSWF) as the splitting window; in this benchmark, the forward spectral calculation is embedded in an inverse problem: recover the prolate bandwidth parameter, the real-space cutoff and a cell shear from three reciprocal-space observables, then report the force-error tolerance to which the recovered bandwidth corresponds.
Use the source paper's zeroth-order prolate splitting construction, its normalized finite-Fourier window, its reciprocal spectral kernel, and its operative rule for choosing the prolate bandwidth parameter at a prescribed force-error tolerance (the rule itself, not its asymptotic small-error form); the inverse calculation must retain the paper's normalized Coulomb convention 1/(4πr), with no additional electrostatic prefactor.
Wherever the prolate function itself, rather than the unit-integral splitting window, is evaluated, use the L2 normalization ∫_{-1}^{1} ψ_0^c(x)^2 dx = 1 with ψ_0^c(0) > 0.
The unknown physical parameters are
- the prolate bandwidth c,
- the real-space cutoff r_c,
- a volume-preserving shear gamma.
The periodic direct-lattice matrix is
H(gamma) = [[L, gamma*L, 0],
            [0, L,         0],
            [0, 0,         L]],
whose columns are the cell vectors; particle coordinates below are fractional coordinates f_j, so Cartesian positions are r_j = H(gamma) f_j, and because the same cell is used for direct and reciprocal coordinates, the reciprocal phase is 2π k · f_j.
Use the centered even reciprocal index cube with n_f = 10, i.e. each integer component is in {-5,-4,...,4}, excluding k=(0,0,0); use 32-point Gauss-Legendre quadrature for the numerical PSWF.
Fractional coordinates:
f = [[-0.23, -0.08,  0.05],
     [ 0.16, -0.18, -0.06],
     [ 0.25,  0.14,  0.11],
     [-0.07,  0.23, -0.15]]
Charges:
q = [1.00, -1.00, 0.75, -0.75]
Cubic scale:
L = 4.8
For the inverse benchmark, match these three target observables:
1. spectral filter factor at the reference wavenumber kappa_ref = 4.0, i.e. |xi|^2 times the Fourier-space long-range kernel evaluated at |xi| = kappa_ref:
   T_target = 0.49415969767112566
2. truncated reciprocal spectral energy:
   E_target = 0.24195924758540227
3. x component of the reciprocal spectral force on particle 0:
   Fx_target = 0.01899987567836107
Use residual scales
sigma = [0.05, 0.20, 0.02].
The admissible parameter bounds are
c       in [3.0, 11.0]
r_c     in [0.45, 1.15]
gamma   in [-0.35, 0.35]
and the initial physical guess is
[c, r_c, gamma] = [5.0, 0.60, -0.10].
Use the analytic PSWF bandwidth sensitivity obtained from the symmetric Nyström eigenproblem, propagate that sensitivity through the window transform and the reciprocal spectral observables, and solve the bounded three-parameter nonlinear inverse problem to a maximum scaled residual of 1e-12.
After recovering c, convert it to the force-error tolerance epsilon with the source paper's operative bandwidth rule.

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

01_pswf_bandwidth_sensitivity

Goal
----
Construct the unit-integral prolate window samples and their analytic bandwidth sensitivity.

```python
def pswf_bandwidth_sensitivity(c: float, quadrature_order: int) -> "np.ndarray":
    """Return the unit-integral window samples and their derivative with respect to c.

    Parameters
    ----------
    c : float
        Positive finite prolate bandwidth.
    quadrature_order : int
        Integer Gauss-Legendre order n, at least 8.

    Returns
    -------
    packed : np.ndarray
        One-dimensional array of length 4*n+2. Blocks are Gauss-Legendre nodes,
        weights, the unit-integral window samples chi = psi_0^c / int_{-1}^{1} psi_0^c
        (so that sum_j w_j chi_j = 1), dchi/dc, then the leading
        eigenvalue lambda_0 and dlambda_0/dc, where lambda_0 is the leading
        eigenvalue of the even finite-Fourier (cosine-kernel) operator,
        int_{-1}^{1} cos(c*x*t) chi(t) dt = lambda_0 * chi(x).

    Raises
    ------
    ValueError
        If c is not positive and finite or quadrature_order is not an integer at least 8.
    RuntimeError
        If the leading discrete eigenvalue is not numerically simple.
    """
    return packed
```

### Step 2

02_prolate_transform_sensitivity

Goal
----
Evaluate the normalized prolate-window transform and its analytic derivatives.

```python
def prolate_transform_sensitivity(c: float, quadrature_order: int, s_values: "np.ndarray") -> "np.ndarray":
    """Return chi-hat(s), partial chi-hat/partial c, and partial chi-hat/partial s.

    Parameters
    ----------
    c : float
        Positive finite prolate bandwidth.
    quadrature_order : int
        Integer Gauss-Legendre order n, at least 8.
    s_values : np.ndarray
        Nonempty finite one-dimensional array of real transform arguments.

    Returns
    -------
    values : np.ndarray
        Real array of shape (M,3). Columns are chi-hat(s), the derivative with
        respect to c at fixed s, and the derivative with respect to s at fixed c.
        For |s| <= c use the finite-Fourier eigenfunction identity with a
        degree-(n-1) Legendre reconstruction; for |s| > c use direct quadrature.

    Raises
    ------
    ValueError
        If s_values is not a nonempty finite vector or an upstream PSWF input is invalid.
    """
    return values
```

### Step 3

03_centered_reciprocal_indices

Goal
----
Enumerate the centered even reciprocal index cube used by the truncated spectral sum.

```python
def centered_reciprocal_indices(mode_count: int) -> "np.ndarray":
    """Return all nonzero integer reciprocal indices in the centered even cube.

    Parameters
    ----------
    mode_count : int
        Even integer n_f at least 4.

    Returns
    -------
    indices : np.ndarray
        Integer array with shape (mode_count**3 - 1, 3), in deterministic
        meshgrid order, excluding only [0,0,0].

    Raises
    ------
    ValueError
        If mode_count is not an even integer at least 4.
    """
    return indices
```

### Step 4

04_sheared_reciprocal_geometry

Goal
----
Construct reciprocal vectors for a volume-preserving simple shear and differentiate them analytically.

```python
def sheared_reciprocal_geometry(indices: "np.ndarray", box_length: float, shear: float) -> "np.ndarray":
    """Return reciprocal vectors and their shear derivatives.

    Parameters
    ----------
    indices : np.ndarray
        Nonempty integer-valued array of shape (M,3), with no zero row.
    box_length : float
        Positive finite cubic scale L.
    shear : float
        Finite simple-shear parameter gamma satisfying |gamma| < 0.5.

    Returns
    -------
    geometry : np.ndarray
        Real array with shape (M,8). Columns 0:3 are xi_k, columns 3:6 are
        dxi_k/dgamma, column 6 is |xi_k|, and column 7 is d|xi_k|/dgamma.

    Raises
    ------
    ValueError
        If indices, box_length, or shear violate the stated domain.
    """
    return geometry
```

### Step 5

05_fractional_structure_factors

Goal
----
Evaluate reciprocal charge structure factors from fractional particle coordinates.

```python
def fractional_structure_factors(fractional_positions: "np.ndarray", charges: "np.ndarray", indices: "np.ndarray") -> "np.ndarray":
    """Return S_k = sum_j q_j exp(i 2*pi*k dot f_j) for each reciprocal index.

    Parameters
    ----------
    fractional_positions : np.ndarray
        Finite array of particle fractional coordinates with shape (N,3).
    charges : np.ndarray
        Finite one-dimensional real charge vector of length N.
    indices : np.ndarray
        Finite integer-valued reciprocal indices with shape (M,3).

    Returns
    -------
    factors : np.ndarray
        Complex vector of length M containing the charge structure factors.

    Raises
    ------
    ValueError
        If the array shapes, finiteness, or integer-index requirement are violated.
    """
    return factors
```

### Step 6

06_scaled_inverse_residual

Goal
----
Assemble the bounded inverse residual and its full analytic Jacobian.

```python
def scaled_inverse_residual(y: "np.ndarray", target_values: "np.ndarray", target_scales: "np.ndarray", bounds: "np.ndarray", fractional_positions: "np.ndarray", charges: "np.ndarray", indices: "np.ndarray", box_length: float, quadrature_order: int, reference_frequency: float) -> "np.ndarray":
    """Return scaled residuals, the Jacobian in logistic coordinates, and physical parameters.

    Parameters
    ----------
    y : np.ndarray
        Finite unconstrained vector of length 3.
    target_values : np.ndarray
        Finite target vector [T_ref, E_s, F_0x] of length 3.
    target_scales : np.ndarray
        Positive finite residual scales of length 3.
    bounds : np.ndarray
        Finite array of shape (3,2), one strict lower/upper bound pair for
        [c, cutoff, shear].
    fractional_positions : np.ndarray
        Finite fractional particle coordinates with shape (N,3).
    charges : np.ndarray
        Finite neutral charge vector of length N, summing to zero within 1e-12.
    indices : np.ndarray
        Nonempty integer reciprocal index array of shape (M,3), excluding zero.
    box_length : float
        Positive finite cubic scale L.
    quadrature_order : int
        Integer Gauss-Legendre order at least 8.
    reference_frequency : float
        Positive finite reference wavenumber kappa_ref. The diagnostic
        observable T_ref is the spectral filter factor |xi|^2 S_hat(xi) at
        |xi| = kappa_ref, where S_hat is the source's Fourier-space long-range kernel.

    Returns
    -------
    packed : np.ndarray
        One-dimensional array of length 15. Entries 0:3 are scaled residuals;
        entries 3:12 are the 3x3 Jacobian dR/dy in row-major order; entries
        12:15 are the physical parameters [c, cutoff, shear].

    Raises
    ------
    ValueError
        If dimensions, finiteness, neutrality, scales, bounds, or scalar domains are invalid.
    """
    return packed
```

### Step 7

07_damped_lm_iteration

Goal
----
Perform one deterministic damped least-squares iteration in bounded logistic coordinates.

```python
def damped_lm_iteration(y: "np.ndarray", damping: float, trust_radius: float, target_values: "np.ndarray", target_scales: "np.ndarray", bounds: "np.ndarray", fractional_positions: "np.ndarray", charges: "np.ndarray", indices: "np.ndarray", box_length: float, quadrature_order: int, reference_frequency: float) -> "np.ndarray":
    """Return one deterministic bounded Levenberg-Marquardt iteration.

    Parameters
    ----------
    y : np.ndarray
        Finite unconstrained parameter vector of length 3.
    damping : float
        Positive finite Levenberg-Marquardt damping mu.
    trust_radius : float
        Positive finite Euclidean trust radius in y-space.
    target_values, target_scales, bounds, fractional_positions, charges, indices,
    box_length, quadrature_order, reference_frequency :
        Inputs passed unchanged to scaled_inverse_residual.

    Returns
    -------
    packed : np.ndarray
        Real vector of length 8: updated y[0:3], updated damping, updated trust
        radius, accepted flag (1 or 0), old objective, and new objective,
        where objective = 0.5*sum(r_i**2) over the scaled residual r.
        The SVD step is -V diag(s/(s^2+mu)) U^T r, scaled to the trust radius.
        Try alpha=1,1/2,... for at most 12 trials and accept the first strict
        objective decrease. If accepted, use rho=actual/predicted reduction:
        rho>0.75 halves damping and doubles trust up to 4; rho<0.25 multiplies
        damping by 4 and halves trust down to 1e-6. If no trial is accepted,
        keep y, multiply damping by 10, and halve trust down to 1e-8.

    Raises
    ------
    ValueError
        If y, damping, trust_radius, or downstream residual inputs are invalid.
    """
    return packed
```

### Step 8

08_solve_inverse_precision

Goal
----
Solve the complete bounded inverse prolate-Ewald problem and return the recovered force-error parameter.

```python
def solve_inverse_precision(fractional_positions: "np.ndarray", charges: "np.ndarray", box_length: float, mode_count: int, quadrature_order: int, reference_frequency: float, target_values: "np.ndarray", target_scales: "np.ndarray", bounds: "np.ndarray", initial_parameters: "np.ndarray", tolerance: float, max_iterations: int) -> float:
    """Return the force-error tolerance epsilon for the recovered c after solving the inverse problem.

    Parameters
    ----------
    fractional_positions : np.ndarray
        Finite fractional coordinates with shape (N,3).
    charges : np.ndarray
        Finite neutral charge vector of length N.
    box_length : float
        Positive finite cubic scale L.
    mode_count : int
        Even centered reciprocal mode count at least 4.
    quadrature_order : int
        Integer Gauss-Legendre order at least 8.
    reference_frequency : float
        Positive finite reference wavenumber kappa_ref of the filter-factor
        diagnostic |xi|^2 S_hat(xi) at |xi| = kappa_ref.
    target_values : np.ndarray
        Target vector [T_ref,E_s,F_0x] of length 3.
    target_scales : np.ndarray
        Positive residual scales of length 3.
    bounds : np.ndarray
        Ordered physical bounds of shape (3,2) for [c,cutoff,shear].
    initial_parameters : np.ndarray
        Strictly interior physical initial guess of length 3.
    tolerance : float
        Positive convergence tolerance for max(abs(scaled residual)).
    max_iterations : int
        Positive integer iteration limit.

    Returns
    -------
    epsilon : float
        Force-error tolerance corresponding to the converged recovered bandwidth c
        under the source paper's operative bandwidth rule. Wherever the PSWF
        itself is evaluated, it is normalized so that
        int_{-1}^{1} psi_0^c(x)^2 dx = 1 and psi_0^c(0) > 0.

    Raises
    ------
    ValueError
        If solver inputs or the initial point violate the stated domain.
    RuntimeError
        If the scaled residual does not converge within max_iterations.
    """
    return epsilon
```
