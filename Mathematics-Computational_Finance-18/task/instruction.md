# Mathematics-Computational_Finance-18

## Background

Rough volatility models replace the Markovian variance dynamics of classical stochastic volatility with a Volterra convolution against a slowly decaying kernel. That single change reproduces both the roughness of realised volatility and the steepness of the short-maturity smile, which classical models fit only by adding parameters. When the same kernel is also used to propagate jump impulses, its behaviour at zero lag stops being a technicality: the response to a jump arriving now is the kernel evaluated at zero lag, and for the benchmark fractional kernel that quantity is unbounded. A model of this kind therefore has to say what it does about the singularity before it can be simulated at all.

A kernel of the right class can be written as a continuous mixture of decaying exponentials. Truncating that mixture to a finite positive quadrature replaces the infinite-dimensional Volterra dynamics with a finite system of Ornstein-Uhlenbeck factors, which is what makes simulation and pricing tractable. How faithful that replacement is depends on the quadrature, and it is not adequately judged by eye.

The one-step variance of the weighted aggregate of the factors, set against the one-step variance of the exact Volterra noise it stands in for, is the natural diagnostic for whether the finite-dimensional realization is faithful.

The source's model also carries a self-exciting jump block: variance jumps arrive with an intensity that every jump, of either type, raises for a while, and the same kernel propagates each variance jump into the variance. Under the lift the factors and the two intensities together form an affine jump-diffusion, so their first and second moments obey a closed linear system that can be solved exactly rather than simulated. How the variance and the jump intensity move together at a horizon depends on every coupling of that system: how the factors are driven, what each kind of jump does to each coordinate, and how the jump marks are distributed.

## Problem

A rough-volatility model with clustered variance jumps drives both its diffusive memory and
its jump propagation through a single weakly singular fractional kernel. That kernel cannot
be used directly in the jump channel, because a jump would produce an unbounded response at
zero lag, so the model replaces it with a finite-resolution regularization at a stated
resolution scale. The regularized kernel is then written as a finite positive sum of
decaying exponentials, which turns the Volterra dynamics into a finite system of
Ornstein-Uhlenbeck factors.

Adopt the regularization and the node-and-weight construction that the source prescribes for
this model. The construction partitions the positive frequency axis into consecutive blocks
whose breakpoint is set by an integer index m, and applies a quadrature rule of its own
choosing to each block, using n_quad points per block. On the unbounded block, apply the
standard Gauss-Laguerre rule shifted to start at the block's lower end, that is nodes equal
to one plus the Gauss-Laguerre abscissae, with that rule's own unit-rate weight function
exp(-(x - 1)).

Work at alpha = 0.35, resolution scale delta_star = 0.05, breakpoint index m = 3 and
n_quad = 24 points per block. First check the lift over a single time step of length
dt = 0.25: compute the one-step variance of the weighted aggregate of the Ornstein-Uhlenbeck
factors, compute the one-step variance of the exact Volterra noise those factors stand in
for as the integral of the squared regularized kernel across the step by the composite
Simpson rule on a uniform grid of 200001 points spanning zero to dt inclusive, and form the
lift-fidelity error: the ratio of the first to the second, minus one, all multiplied by ten
thousand, in basis points. Treat the lift as admissible only if that error lies within 100 basis points
of zero.

For the model take the source's TSLA calibration: mean-reversion speed kappa = 4.924 and
base level theta = 0.288 per year, volatility of variance xi = 0.275, initial variance
V0 = 0.1081, baseline jump intensities 0.523 per year for price jumps and 0.687 per year for
variance jumps, mean variance-jump size mu_V = 0.0865, immediate excitations eta_SS = 0.30,
eta_SV = 0.15, eta_VS = 0.20 and eta_VV = 0.25, and decay rates beta_S = 3.0 and
beta_V = 2.5 per year. Every factor starts at zero and both intensities start at their
baselines. The leverage correlation and the price-jump marks do not enter this calculation.
Treat the variance as nonnegative throughout, so that the positive-part truncation in the
coefficients is inactive.

Under the lifted dynamics the state, the factors together with the two intensities, is an
affine jump-diffusion, so its first and second moments obey a closed linear system that can
be solved exactly. Build that system for the lifted model with every coupling the source's
model implies, including how the factors are driven, what a jump of each type does to every
coordinate of the state, and how the variance-jump marks are distributed, and solve it to
the horizon T = 0.25 without simulation. Report the correlation coefficient between the
variance and the variance-jump intensity at the horizon.

State the conventions you adopted and justify each from the source. In the reasoning also
report the lift-fidelity error in basis points, the number of exponential-sum nodes and the
sum of their weights against the kernel's value at zero lag, the spectral radius of the
normalized excitation matrix and the stationary mean intensities of the two jump types, the
expected variance and its standard deviation at the horizon, the expected variance-jump
intensity and its standard deviation at the horizon, their covariance, and the standard
deviation of the variance at the horizon that remains when the volatility of variance is set
to zero. Quote the correlation to at least five decimal places and each of these
values to at least four significant figures.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> concise (a few hundred words): state the conventions you adopted and report the values the problem asks for, including the scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> concise (a few hundred words): state the conventions you adopted and report the values the problem asks for, including the scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
```

## Your task

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

regularized_kernel

Goal
----
Evaluate the regularized fractional kernel of Eq (19) on a 1-D grid of nonnegative lags. Apply the regularization the source prescribes at its stated resolution scale; it must leave the kernel finite at zero lag and preserve the fractional decay at resolved lags. Raise ValueError if alpha is outside (0, 1/2), if delta_star is not positive, if t is not a non-empty 1-D array, or if any lag is negative.

```python
def regularized_kernel(alpha: float, delta_star: float, t: "np.ndarray") -> "np.ndarray":
    """Evaluate the regularized fractional kernel of Eq (19) on a 1-D grid of nonnegative lags.

    Parameters
    ----------
    alpha : float
        Roughness exponent of the fractional kernel, 0 < alpha < 1/2.
    delta_star : float
        Resolution scale of the regularization, positive.
    t : np.ndarray
        Non-empty 1-D array of nonnegative lags.

    Returns
    -------
    result : np.ndarray
        ndarray of float64, the kernel evaluated elementwise on t, same shape as t.

    Raises
    ------
    ValueError
        If alpha is outside (0, 1/2).
        If delta_star is not positive.
        If t is not a non-empty 1-D array.
        If any lag is negative.
    """
    return result  # placeholder
```

### Step 2

bernstein_density

Goal
----
Evaluate the density of the representing measure of the SAME regularized kernel used in the previous step, at strictly positive frequencies x. This is the density whose mixture of decaying exponentials reproduces that kernel; its form follows from the regularization the source fixed. Raise ValueError if alpha is outside (0, 1/2), if delta_star is not positive, if x is not a non-empty 1-D array, or if any x is not strictly positive.

```python
def bernstein_density(alpha: float, delta_star: float, x: "np.ndarray") -> "np.ndarray":
    """Evaluate the density of the representing measure of the regularized kernel of the
    previous step at strictly positive frequencies.

    Parameters
    ----------
    alpha : float
        Roughness exponent of the fractional kernel, 0 < alpha < 1/2.
    delta_star : float
        Resolution scale of the regularization, positive.
    x : np.ndarray
        Non-empty 1-D array of strictly positive frequencies.

    Returns
    -------
    result : np.ndarray
        ndarray of float64, the density evaluated elementwise on x, same shape as x.

    Raises
    ------
    ValueError
        If alpha is outside (0, 1/2).
        If delta_star is not positive.
        If x is not a non-empty 1-D array.
        If any x is not strictly positive.
    """
    return result  # placeholder
```

### Step 3

block_quadrature_nodes

Goal
----
Build the quadrature nodes and weights of Eq (25) for the frequency axis. Use the partition of the positive axis that the source prescribes, with the quadrature rule it assigns to each block, at n_quad points per block; the integer m sets the prescribed breakpoint. Return the blocks in increasing order of frequency, with the nodes of each block in ascending order. Weights are returned RELATIVE TO THE REPRESENTING DENSITY, so that multiplying them elementwise by that density at these nodes gives the exponential-sum weights. Raise ValueError if alpha is outside (0, 1/2), if m is not a nonnegative integer, or if n_quad is not a positive integer.

```python
def block_quadrature_nodes(alpha: float, m: int, n_quad: int) -> tuple:
    """Build the quadrature nodes and density-relative weights of Eq (25) for the frequency
    axis.

    Parameters
    ----------
    alpha : float
        Roughness exponent of the fractional kernel, 0 < alpha < 1/2.
    m : int
        Nonnegative integer that sets the block breakpoint.
    n_quad : int
        Positive number of quadrature points per block.

    Returns
    -------
    result : tuple
        tuple (x, q) of two 1-D float64 ndarrays, each of length 3*n_quad.

    Raises
    ------
    ValueError
        If alpha is outside (0, 1/2).
        If m is not a nonnegative integer.
        If n_quad is not a positive integer.
    """
    return result  # placeholder
```

### Step 4

lift_fidelity_error_bp

Goal
----
Build the quadrature nodes and density-relative weights (previous step), evaluate the representing density at those nodes and form the exponential-sum weights omega = q * w(x). Evaluate the regularized kernel on a uniform grid of n_grid points spanning zero to dt inclusive and integrate its square over the step by the composite Simpson rule. Compute the one-step variance, over a step of length dt, of the omega-weighted aggregate of the Ornstein-Uhlenbeck factor increments that the lift uses in place of the Volterra noise, per unit instantaneous variance of the driving noise. Return the ratio of that one-step variance to the integrated square, minus one, all multiplied by ten thousand. Call the earlier step functions rather than reimplementing them. Raise ValueError if dt is not positive, or if n_grid is not an odd integer of at least three.

```python
def lift_fidelity_error_bp(alpha: float, delta_star: float, m: int, n_quad: int, dt: float, n_grid: int) -> float:
    """Compute the one-step lift-fidelity error of the exponential-sum lift, in basis points.

    Parameters
    ----------
    alpha : float
        Roughness exponent of the fractional kernel, 0 < alpha < 1/2.
    delta_star : float
        Resolution scale of the regularization, positive.
    m : int
        Nonnegative integer that sets the block breakpoint.
    n_quad : int
        Positive number of quadrature points per block.
    dt : float
        Length of the time step, positive.
    n_grid : int
        Odd number, at least three, of uniform grid points spanning zero to dt inclusive for
        the composite Simpson rule.

    Returns
    -------
    result : float
        float, the lift-fidelity error in basis points.

    Raises
    ------
    ValueError
        If dt is not positive.
        If n_grid is not an odd integer of at least three.
        If alpha, delta_star, m or n_quad is invalid, as in the earlier steps.
    """
    return result  # placeholder
```

### Step 5

lifted_moment_blocks

Goal
----
Build the affine drift and the jump blocks of the lifted state z = (U_1, ..., U_N, lam_S, lam_V), types ordered (price jumps S, variance jumps V), for the source's lifted model with variance V = v0 + sum_k omega_k U_k and the positive-part truncation inactive. Return (A, c, j_s, j_v, Q_s, Q_v): the matrix A and vector c of the continuous drift A z + c; for each jump type k the mean jump vector j_k, whose entry i is the expected change of z_i at a type-k jump; and the matrix Q_k, whose entry (i, j) is the expected product of the changes of z_i and z_j at a type-k jump. Raise ValueError if x and omega are not non-empty 1-D arrays of equal length, if any rate is not finite and strictly positive, if kappa, theta, v0 or mu_v is not finite and nonnegative, if lam_inf, eta or beta has the wrong shape or a non-finite entry, if a baseline or excitation is negative, or if a decay rate is not positive.

```python
def lifted_moment_blocks(x: "np.ndarray", omega: "np.ndarray", kappa: float, theta: float, v0: float, mu_v: float, lam_inf: "np.ndarray", eta: "np.ndarray", beta: "np.ndarray") -> tuple:
    """Build the affine drift and the jump blocks of the lifted state (U_1, ..., U_N, lam_S,
    lam_V).

    Parameters
    ----------
    x : np.ndarray
        Exponential-sum nodes (factor decay rates), a non-empty 1-D array of strictly
        positive values.
    omega : np.ndarray
        Exponential-sum weights, a 1-D array of the same length as the nodes.
    kappa : float
        Mean-reversion speed of the variance, finite and nonnegative.
    theta : float
        Base level of the variance, finite and nonnegative.
    v0 : float
        Initial variance, finite and nonnegative.
    mu_v : float
        Mean variance-jump size, finite and nonnegative.
    lam_inf : np.ndarray
        Baseline jump intensities, shape (2,), ordered (price jumps, variance jumps).
    eta : np.ndarray
        Immediate excitations, shape (2, 2); column k holds the increments of the two
        intensities caused by a jump of type k, types ordered (price jumps, variance jumps).
    beta : np.ndarray
        Decay rates of the two intensities, shape (2,).

    Returns
    -------
    result : tuple
        tuple (A, c, j_s, j_v, Q_s, Q_v) of float64 arrays with shapes (n, n), (n,), (n,),
        (n,), (n, n), (n, n), n = len(x) + 2.

    Raises
    ------
    ValueError
        If x and omega are not non-empty 1-D arrays of equal length.
        If any rate is not finite and strictly positive.
        If kappa, theta, v0 or mu_v is not finite and nonnegative.
        If lam_inf, eta or beta has the wrong shape or a non-finite entry.
        If a baseline or an excitation is negative, or a decay rate is not positive.
    """
    return result  # placeholder
```

### Step 6

lifted_moment_generator

Goal
----
Build the generator G of the closed linear system for the first and second moments of the lifted state from the drift and jump blocks of the previous step, the volatility of variance xi, the exponential-sum weights omega and v0. Coordinates of the moment vector y: the second moments S_ij = E[z_i z_j] for i <= j in row-major order (n(n+1)/2 entries), then the first moments m_i = E[z_i] (n entries), then the constant 1; return G with dy/dt = G y, shape (n(n+1)/2 + n + 1) square, where the price-jump and variance-jump intensities are the coordinates z_N and z_{N+1}. Raise ValueError if the block shapes are inconsistent with n = len(omega) + 2, if any block is not finite, or if xi or v0 is not finite and nonnegative.

```python
def lifted_moment_generator(A: "np.ndarray", c: "np.ndarray", xi: float, omega: "np.ndarray", v0: float, j_s: "np.ndarray", j_v: "np.ndarray", Q_s: "np.ndarray", Q_v: "np.ndarray") -> "np.ndarray":
    """Build the generator G of the closed linear system for the first and second moments of
    the lifted state.

    Parameters
    ----------
    A : np.ndarray
        Drift matrix of the lifted state, shape (n, n).
    c : np.ndarray
        Constant drift vector of the lifted state, shape (n,).
    xi : float
        Volatility of variance, finite and nonnegative.
    omega : np.ndarray
        Exponential-sum weights, shape (N,) with n = N + 2.
    v0 : float
        Initial variance, finite and nonnegative.
    j_s : np.ndarray
        Mean jump vector of a price jump, shape (n,).
    j_v : np.ndarray
        Mean jump vector of a variance jump, shape (n,).
    Q_s : np.ndarray
        Second-moment jump products of a price jump, shape (n, n).
    Q_v : np.ndarray
        Second-moment jump products of a variance jump, shape (n, n).

    Returns
    -------
    result : np.ndarray
        ndarray of float64, square of size n(n+1)/2 + n + 1 with n = len(omega) + 2, the
        moment generator G.

    Raises
    ------
    ValueError
        If the block shapes are inconsistent with n = len(omega) + 2.
        If any block is not finite.
        If xi or v0 is not finite and nonnegative.
    """
    return result  # placeholder
```

### Step 7

variance_intensity_correlation

Goal
----
Orchestrator. Build the nodes and density-relative weights and form the exponential-sum weights (steps 3 and 2). Compute the lift-fidelity error over dt on the n_grid grid (step 4) and raise ValueError if it is not finite or exceeds 100 basis points in magnitude; evaluate the kernel at zero lag (step 1) and raise ValueError if the weights do not sum to it within one percent. Build the drift and jump blocks (step 5) and the moment generator (step 6). Start every factor at zero and both intensities at their baselines, so the initial second moments are the outer product of the initial means and the constant coordinate is one. Propagate the moment vector to the horizon with the matrix exponential of G times the horizon, recover the covariance matrix of the state as S - m m^T, and return the correlation between the variance V = v0 + omega . U and the variance-jump intensity: their covariance divided by the product of their standard deviations. Call the earlier step functions rather than reimplementing any of them. Raise ValueError if the horizon is not a positive finite number or if either variance at the horizon is not positive.

```python
def variance_intensity_correlation(alpha: float, delta_star: float, m: int, n_quad: int, dt: float, n_grid: int, kappa: float, theta: float, v0: float, xi: float, mu_v: float, lam_inf: "np.ndarray", eta: "np.ndarray", beta: "np.ndarray", horizon: float) -> float:
    """Orchestrator: the correlation between the variance and the variance-jump intensity at
    the horizon under the lifted model.

    Parameters
    ----------
    alpha : float
        Roughness exponent of the fractional kernel, 0 < alpha < 1/2.
    delta_star : float
        Resolution scale of the regularization, positive.
    m : int
        Nonnegative integer that sets the block breakpoint.
    n_quad : int
        Positive number of quadrature points per block.
    dt : float
        Length of the time step, positive.
    n_grid : int
        Odd number, at least three, of uniform grid points spanning zero to dt inclusive for
        the composite Simpson rule.
    kappa : float
        Mean-reversion speed of the variance, finite and nonnegative.
    theta : float
        Base level of the variance, finite and nonnegative.
    v0 : float
        Initial variance, finite and nonnegative.
    xi : float
        Volatility of variance, finite and nonnegative.
    mu_v : float
        Mean variance-jump size, finite and nonnegative.
    lam_inf : np.ndarray
        Baseline jump intensities, shape (2,), ordered (price jumps, variance jumps).
    eta : np.ndarray
        Immediate excitations, shape (2, 2); column k holds the increments of the two
        intensities caused by a jump of type k, types ordered (price jumps, variance jumps).
    beta : np.ndarray
        Decay rates of the two intensities, shape (2,).
    horizon : float
        Horizon T at which the correlation is evaluated, positive and finite.

    Returns
    -------
    result : float
        float, the correlation between the variance and the variance-jump intensity at the
        horizon.

    Raises
    ------
    ValueError
        If the horizon is not a positive finite number.
        If the lift-fidelity error is not finite or exceeds 100 basis points in magnitude.
        If the exponential-sum weights do not sum to the kernel at zero lag within one
        percent.
        If either variance at the horizon is not positive.
        If an input is invalid for an earlier step.
    """
    return result  # placeholder
```
