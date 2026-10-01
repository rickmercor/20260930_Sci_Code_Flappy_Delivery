# Mathematics-Computational_Finance-6

## Background

Rough volatility has become the dominant empirical description of equity-index variance: realized-volatility series show a power-law decay of autocorrelation that classical one-factor stochastic-volatility models cannot reproduce, and fractional Brownian motion with a small Hurst exponent captures it. The cost is that rough models are non-Markovian, so exact simulation is slow and standard Euler schemes need very fine time steps to control the discretization bias of the variance process. A productive response is to approximate the fractional kernel by a finite sum of exponentials, which turns the rough dynamics into a Markovian system of variance factors whose weighted sum converges to the target variance as the number of factors grows. The lifted Heston model is the resulting family, and it inherits the rough model's term structure while remaining amenable to Monte Carlo. The remaining difficulty is the large-step regime: when the step exceeds the fastest mean-reversion scale, naive discretizations distort the conditional moments of the variance increment and can drive the variance negative. The constrained linear projection scheme addresses this by matching the conditional first two moments of the increment and enforcing non-negativity through a constrained fallback, so that long-dated options and path-dependent payoffs can be priced with few time steps and no loss of moment accuracy.

## Problem

Rough volatility models reproduce the observed power-law decay of the volatility autocorrelation, but their fractional dynamics are not Markovian and are expensive to simulate at the large time steps practitioners need. The lifted Heston model approximates the fractional kernel by a finite sum of exponential factors, turning the rough dynamics into a Markovian system of N variance factors whose weighted sum recovers the variance process, and simulating that system accurately over long horizons requires a step that stays stable and preserves the conditional moments of the variance increment even when the step is far larger than the fastest mean-reversion scale. This task asks you to implement the constrained linear projection (C-LP) step for the lifted Heston variance factors and to run a deterministic ten-step march, reporting the terminal integrated variance.

The lifted Heston variance is V_t = g0(t) + ω·U_t, where U_t ∈ R^N solves dU^n_t = (−x_n U^n_t − λ V_t) dt + ν √(V_t) dW_t and the deterministic initial curve is g0(t) = v0 + λθ Σ_n (ω_n/x_n)(1 − e^{−x_n t}); the lift weights ω_n > 0 and mean-reversion speeds x_n > 0, for n = 1, …, N, are the node weights and decay rates of the specific finite sum of N exponential factors that reproduces the fractional kernel of Hurst exponent H at refinement ratio r_N = 1 + 10N^{−0.9}, with their closed form intentionally left unstated, and the factor dynamics are governed by the state matrix A = −λ 1_N ω − diag(x). Over a step [s, t], define the factor conditional-mean vector μ = A^{−1}(e^{AΔt} − I)U_s + ξ_t, where ξ solves dξ/du = Aξ − λ G0(s, u) 1_N from ξ_s = 0 and G0(s, u) = ∫_s^u g0(r) dr; then the conditional mean of the integrated variance is α = ω·μ + G0(s, t); define also the per-factor cross-moment vector κ = ν(I₁ − A^{−1}(e^{AΔt} − I)1_N ω)A^{−1}U_s + ν∫_s^t e^{A(t−u)}(1_N(ω·ξ_u) + G0(s, u)1_N) du, where I₁ = ∫_s^t e^{A(t−u)} 1_N ω e^{A(u−s)} du, which must be evaluated from this defining integral rather than from any closed form quoted elsewhere, so the scalar conditional cross-moment needed by the projection is EXZ = E_s[X_{s,t} Z_{s,t}] = ω·κ. Here μ^n and κ^n denote the nth components of these vectors, so the state update below is componentwise and unambiguous.

Each step forms the optimal conditional projection of the variance increment onto the integrated variance: the projection slope matches the conditional first two moments of the increment, and the updated variance must remain non-negative for every realization of the inverse-Gaussian draw, which forces a constrained fallback whenever the unconstrained slope would violate that requirement; the unconstrained slope β is admissible only when it is positive, does not exceed the slope ceiling βᴸ, and satisfies C(0, β) ≥ 0, with both feasibility boundaries inclusive and no tolerance applied. The constrained fallback β^C is the admissible slope at which the intercept condition binds exactly, i.e. C(0, β^C) = 0. Let the selected slope be β̃ = β when those conditions all hold and β̃ = β^C otherwise; draw the variance increment from the inverse-Gaussian family with mean α and shape γ = (α/β̃)², and then advance the state as X̂ⁿ = μⁿ + (κⁿ/EXZ)(X̂ − α), Ẑ = (X̂ − α)/β̃, U ← U − x⊙X̂ⁿ − λX̂ + νẐ, and V = ω·U + g0(t).

Set up the simulation with exactly the following configuration:

  - Model: lifted Heston with H = 0.3, N = 5, λ = 0.25, ν = 0.1, v0 = 0.02, θ = 0.5; initial state U₀ = 0 ∈ R⁵.
  - Horizon and grid: T = 5, M = 10 equal steps of length Δt = 0.5 on [0, T].
  - ODE convention: the ξ equation and the cross-moment time integral are evaluated by classical fixed-step RK4 on a uniform 4001-point grid over each step [s, t].
  - Random draws: with rng = numpy.random.default_rng(12345), draw z = rng.standard_normal(10) and u = rng.random(10) once at the start; step k (k = 0, …, 9) consumes z[k] and u[k].
  - Inverse-Gaussian draw: use the full Michael–Schucany–Haas transform with its root selection — writing its scalar mean as m = α, with V = z² and X₁ = m + m²V/(2γ) − (m/(2γ))√(4mγV + m²V²), the sample is X̂ = X₁ when u ≤ m/(m + X₁) and X̂ = m²/X₁ otherwise.
  - Reporting: report X_T = Σ_{k=1}^{10} X̂_k, the terminal integrated variance, rounded to 4 significant figures (round-to-nearest).

Derive the lift parametrization, the state matrix, the conditional moments, the feasibility boundaries, and the constrained slope from the definitions above, whose closed forms are intentionally left unstated, and in the reasoning block explain the steps you took to get the final answer. Report the first-step conditional mean, cross-moment, unconstrained slope and feasibility ceiling, the value of the positivity constraint that decides the branch, the constrained slope actually used at the first and final steps, the realized first-step variance increment, a compact summary covering the branch decision at all ten steps, the range that the ratio of the unconstrained slope to its ceiling spans across the march, and the terminal integrated variance; one march under the pinned configuration determines the requested quantity, so the result is fully deterministic.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
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

01_lift_parameters

Goal
----
Builds the discrete lift weights and mean-reversion speeds that approximate the rough-volatility kernel of the lifted Heston model from the Hurst exponent and the number of lift factors.

```python
def lift_parameters(H: float, N: int) -> tuple:
    """Build the lift weights and mean-reversion speeds of the lifted Heston model.

    Parameters
    ----------
    H : float
        Hurst exponent of the rough-volatility kernel (must satisfy
        0 < H < 0.5).
    N : int
        Number of lift factors (must be an integer >= 1).

    Returns
    -------
    omega : (N,) float array
        Lift weights of the N factors, in the order of increasing index.
    x : (N,) float array
        Mean-reversion speeds of the N factors, in the order of increasing
        index (all strictly positive).

    Raises
    ------
    ValueError
        If H is not a real number with 0 < H < 0.5, or if N is not an
        integer >= 1.
    """
    return omega, x
```

### Step 2

02_state_matrix

Goal
----
Assembles the linear state matrix that governs the joint evolution of the lifted variance factors from the mean-reversion strength, the lift weights, and the mean-reversion speeds.

```python
def state_matrix(lam: float, omega: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Assemble the state matrix of the lifted variance dynamics.

    Parameters
    ----------
    lam : float
        Mean-reversion strength of the variance in the lifted model (must be
        a real finite number >= 0).
    omega : (N,) float array
        Lift weights of the N factors.
    x : (N,) float array
        Mean-reversion speeds of the N factors.

    Returns
    -------
    A : (N, N) float array
        State matrix -lam * 1_N omega - diag(x) of the lifted system.

    Raises
    ------
    ValueError
        If lam is not a real finite number >= 0, or if omega and x are not
        one-dimensional arrays of the same length >= 1.
    """
    return A
```

### Step 3

03_conditional_mean

Goal
----
Computes the conditional mean of the integrated variance over a time step and the per-factor conditional mean vector of the lifted state, by integrating the linear forcing equation and exponentiating the state matrix.

```python
import numpy as np

def conditional_mean(lam: float, v0: float, theta: float, omega: np.ndarray, x: np.ndarray, A: np.ndarray, s: float, t: float, U_s: np.ndarray, n_grid: int = 4001) -> tuple:
    """Compute the conditional mean of the integrated variance over a step.


    Definitions
    -----------
    g0(u)=v0+lam*theta*sum(omega/x*(1-exp(-x*u))) and
    G0(s,u)=integral_s^u g0(r) dr. Let P(h)=A^{-1}(exp(A*h)-I).
    xi'=A*xi-lam*G0(s,u)*ones, xi(s)=0.
    mu=P(t-s)*U_s+xi(t); alpha=omega@mu+G0(s,t).

    Numerical domain
    ----------------
    Support short positive steps down to 1e-8 and positive factor speeds down
    to 1e-10, including nonzero U_s and clustered speeds. Return the unscaled
    moments. Tests also compare alpha and mu divided by (t-s), or EXZ and
    kappa divided by (t-s)**2, to check the moment densities in these regimes.
    The formulas denote real mathematical quantities; use numerically stable
    equivalent evaluations when differences of exponentials lose precision.
    The forcing equations use classical RK4 on n_grid uniform points. Evaluate
    the homogeneous matrix-integral contribution to numerical precision.

    Parameters
    ----------
    lam : float
        Mean-reversion strength of the variance in the lifted model.
    v0 : float
        Long-run variance level of the initial curve.
    theta : float
        Long-run mean of the variance.
    omega : (N,) float array
        Lift weights of the N factors.
    x : (N,) float array
        Mean-reversion speeds of the N factors.
    A : (N, N) float array
        State matrix of the lifted system.
    s : float
        Start time of the step.
    t : float
        End time of the step (must satisfy t > s).
    U_s : (N,) float array
        Factor state at the start of the step.
    n_grid : int, optional
        Number of uniform grid points used by the fixed-step integrator
        (must be an integer >= 2; default 4001).

    Returns
    -------
    alpha : float
        Conditional mean of the integrated variance over the step.
    mu : (N,) float array
        Per-factor conditional mean vector of the lifted state over the step.

    Raises
    ------
    ValueError
        If t <= s, if n_grid < 2, if lam, v0, or theta is not a real finite
        number, or if the array shapes are inconsistent.
    """
    return alpha, mu
```

### Step 4

04_cross_moment

Goal
----
Computes the conditional cross-moment between the integrated variance and the driving noise over a time step, together with the per-factor cross-moment vector, from the defining covariance integral.

```python
import numpy as np

def cross_moment(lam: float, nu: float, v0: float, theta: float, omega: np.ndarray, x: np.ndarray, A: np.ndarray, s: float, t: float, U_s: np.ndarray, n_grid: int = 4001) -> tuple:
    """Compute the conditional cross-moment of the integrated variance.


    Definitions
    -----------
    g0(u)=v0+lam*theta*sum(omega/x*(1-exp(-x*u))) and
    G0(s,u)=integral_s^u g0(r) dr. Let P(h)=A^{-1}(exp(A*h)-I).
    xi'=A*xi-lam*G0(s,u)*ones, xi(s)=0.
    I1=integral_s^t exp(A*(t-u))*outer(ones,omega)*exp(A*(u-s)) du.
    psi'=A*psi+nu*(omega@xi+G0(s,u))*ones, psi(s)=0.
    kappa=nu*(I1-P(t-s)*outer(ones,omega))*A^{-1}*U_s+psi(t);
    EXZ=omega@kappa. Integrate xi and psi jointly with RK4.

    Numerical domain
    ----------------
    Support short positive steps down to 1e-8 and positive factor speeds down
    to 1e-10, including nonzero U_s and clustered speeds. Return the unscaled
    moments. Tests also compare alpha and mu divided by (t-s), or EXZ and
    kappa divided by (t-s)**2, to check the moment densities in these regimes.
    The formulas denote real mathematical quantities; use numerically stable
    equivalent evaluations when differences of exponentials lose precision.
    The forcing equations use classical RK4 on n_grid uniform points. Evaluate
    the homogeneous matrix-integral contribution to numerical precision.

    Parameters
    ----------
    lam : float
        Mean-reversion strength of the variance in the lifted model.
    nu : float
        Volatility-of-variance coefficient.
    v0 : float
        Long-run variance level of the initial curve.
    theta : float
        Long-run mean of the variance.
    omega : (N,) float array
        Lift weights of the N factors.
    x : (N,) float array
        Mean-reversion speeds of the N factors.
    A : (N, N) float array
        State matrix of the lifted system.
    s : float
        Start time of the step.
    t : float
        End time of the step (must satisfy t > s).
    U_s : (N,) float array
        Factor state at the start of the step.
    n_grid : int, optional
        Number of uniform grid points used by the fixed-step integrator
        (must be an integer >= 2; default 4001).

    Returns
    -------
    EXZ : float
        Conditional cross-moment of the integrated variance with the noise
        over the step.
    kappa : (N,) float array
        Per-factor cross-moment vector whose weighted sum equals EXZ.

    Raises
    ------
    ValueError
        If t <= s, if n_grid < 2, if a scalar argument is not a real finite
        number, if the array shapes are inconsistent, or if the resulting
        cross-moment is not finite and non-zero.
    """
    return EXZ, kappa
```

### Step 5

05_projection_slope

Goal
----
Forms the unconstrained conditional projection slope together with the feasibility metrics that decide whether the positivity requirement forces a constrained fallback.

```python
def projection_slope(lam: float, nu: float, v0: float, theta: float, omega: np.ndarray, x: np.ndarray, U_s: np.ndarray, s: float, t: float, mu: np.ndarray, alpha: float, kappa: np.ndarray, EXZ: float) -> tuple:
    """Form the unconstrained projection slope and its feasibility metrics.

    Parameters
    ----------
    lam : float
        Mean-reversion strength of the variance in the lifted model.
    nu : float
        Volatility-of-variance coefficient.
    v0 : float
        Long-run variance level of the initial curve.
    theta : float
        Long-run mean of the variance.
    omega : (N,) float array
        Lift weights of the N factors.
    x : (N,) float array
        Mean-reversion speeds of the N factors.
    U_s : (N,) float array
        Factor state at the start of the step.
    s : float
        Start time of the step.
    t : float
        End time of the step.
    mu : (N,) float array
        Per-factor conditional mean vector over the step.
    alpha : float
        Conditional mean of the integrated variance (must be non-zero).
    kappa : (N,) float array
        Per-factor cross-moment vector over the step.
    EXZ : float
        Conditional cross-moment of the integrated variance (must be
        non-zero).

    Returns
    -------
    beta : float
        Unconstrained projection slope.
    betaL : float
        Upper admissible slope implied by the positivity requirement.
    C0 : float
        Value of the positivity constraint at the intercept.
    c : float
        Intercept of the projection constraint.
    feasible : bool
        True when the unconstrained slope satisfies every positivity
        condition.

    Raises
    ------
    ValueError
        If alpha is zero, if EXZ is zero, if a scalar argument is not a real
        finite number, or if the array shapes are inconsistent.
    """
    return beta, betaL, C0, c, feasible
```

### Step 6

06_constrained_slope

Goal
----
Selects the slope actually used for the inverse-Gaussian draw, returning the unconstrained projection slope when it satisfies the positivity requirement and the constrained fallback otherwise.

```python
def constrained_slope(nu: float, omega: np.ndarray, alpha: float, c: float, beta: float, betaL: float, C0: float) -> float:
    """Select the slope actually used for the inverse-Gaussian draw.

    Parameters
    ----------
    nu : float
        Volatility-of-variance coefficient (must be > 0).
    omega : (N,) float array
        Lift weights of the N factors.
    alpha : float
        Conditional mean of the variance increment (must be > 0).
    c : float
        Intercept of the positivity constraint (must be non-zero).
    beta : float
        Unconstrained projection slope.
    betaL : float
        Upper admissible slope implied by the positivity requirement.
    C0 : float
        Value of the positivity constraint evaluated at zero slope.

    Returns
    -------
    beta_tilde : float
        The slope used for the draw: the unconstrained slope when it is
        feasible, otherwise the constrained fallback.

    Raises
    ------
    ValueError
        If alpha is not > 0, if c is zero, if nu is not > 0, or if omega is
        not a one-dimensional array with at least one entry.
    """
    return beta_tilde
```

### Step 7

07_simulate_step

Goal
----
Advances the lifted-Heston state by one time step: draws the variance increment from the inverse-Gaussian family with the full two-root transform, then updates the factor vector and the variance level.

```python
def simulate_step(alpha: float, beta_tilde: float, mu: np.ndarray, kappa: np.ndarray, EXZ: float, x: np.ndarray, omega: np.ndarray, lam: float, nu: float, U_s: np.ndarray, z_k: float, u_k: float, g0_t: float) -> tuple:
    """Advance the lifted-Heston state by one time step.

    Parameters
    ----------
    alpha : float
        Conditional mean of the variance increment (must be > 0).
    beta_tilde : float
        Slope used for the draw (must be > 0).
    mu : (N,) float array
        Conditional mean vector of the factor state.
    kappa : (N,) float array
        Conditional cross-moment vector of the factor state.
    EXZ : float
        Conditional cross-moment scalar (must be non-zero).
    x : (N,) float array
        Mean-reversion speeds of the N factors.
    omega : (N,) float array
        Lift weights of the N factors.
    lam : float
        Mean-reversion speed of the variance level.
    nu : float
        Volatility-of-variance coefficient.
    U_s : (N,) float array
        Factor state at the start of the step.
    z_k : float
        Standard normal draw for this step.
    u_k : float
        Uniform draw for this step.
    g0_t : float
        Initial variance curve evaluated at the end of the step.

    Returns
    -------
    Xhat : float
        Realized variance increment for this step.
    Zhat : float
        Standardized increment used in the state update.
    U_next : (N,) float array
        Factor state at the end of the step.
    V_next : float
        Variance level at the end of the step.

    Raises
    ------
    ValueError
        If alpha is not > 0, if beta_tilde is not > 0, if EXZ is zero, or if the
        vector inputs are not one-dimensional arrays of equal length.
    """
    return Xhat, Zhat, U_next, V_next
```

### Step 8

08_run_clp_march

Goal
----
Chains the lift construction, state matrix, conditional mean, cross moment, projection slope, constrained slope, and one-step simulation of the preceding steps into the complete lifted-Heston march at the requested configuration, and returns the accumulated variance increment over the horizon.

```python
def run_clp_march(H: float = 0.3, N: int = 5, lam: float = 0.25, nu: float = 0.1, v0: float = 0.02, theta: float = 0.5, T: float = 5.0, M: int = 10, seed: int = 12345, n_grid: int = 4001) -> float:
    """Run the full lifted-Heston march and return the accumulated variance increment.

    Parameters
    ----------
    H : float
        Hurst exponent of the rough-volatility kernel (must satisfy
        0 < H < 0.5).
    N : int
        Number of lift factors (must be an integer >= 1).
    lam : float
        Mean-reversion speed of the variance level (must be > 0).
    nu : float
        Volatility-of-variance coefficient (must be > 0).
    v0 : float
        Initial variance level (must be > 0).
    theta : float
        Long-run variance level (must be > 0).
    T : float
        Length of the horizon (must be > 0).
    M : int
        Number of time steps in the march (must be an integer >= 1).
    seed : int
        Seed of the generator that produces the pinned draws (must be an
        integer).
    n_grid : int
        Number of uniform grid points used by the fixed-step integrator on
        each step (must be an integer >= 2).

    Returns
    -------
    X_T : float
        The accumulated variance increment over the horizon, the sum of the
        per-step realized increments.

    Raises
    ------
    ValueError
        If M is not an integer >= 1, if seed is not an integer, if T is not
        > 0, or if n_grid is not an integer >= 2.
    """
    return X_T
```
