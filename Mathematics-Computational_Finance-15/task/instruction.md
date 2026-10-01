# Mathematics-Computational_Finance-15

## Background

Under the Black-Scholes model the price of a European option solves a linear parabolic partial differential equation in the asset price and time, with the option payoff as its terminal (or, in time to maturity, initial) condition. Closed-form prices exist for plain calls and puts, which makes them the standard test problems for numerical methods that are later applied to contracts without closed forms. Finite-difference methods discretise the equation on a truncated asset domain with boundary conditions taken from the asymptotic behaviour of the option.

Two difficulties dominate. The payoff of a call or put has a kink at the strike, where its second derivative is a Dirac mass, so the numerical error concentrates near the strike at early times and the observed convergence order falls below the formal order of the scheme. Replacing the kink by a smooth function on a narrow band around the strike restores regularity at the cost of a small, controllable perturbation of the initial data. A scheme started from smoothed data approximates the solution of the smoothed problem, so its discretisation error is measured against that solution, while the difference between the smoothed and the original problem is a separate smoothing error. The second difficulty is that standard central and upwind schemes can produce negative prices or spurious oscillations when the convection term dominates or when the time step is large, unless step sizes are restricted.

Non-standard finite-difference methods address the second difficulty by building discretisations from exact solutions of simpler equations. An exact finite-difference scheme reproduces the solution of a differential equation exactly at the grid points; it exists for linear constant-coefficient equations and for equidimensional (Cauchy-Euler) equations, where it can be obtained from the equation's fundamental solutions. Such schemes use non-trivial denominator functions of the step sizes in place of the step sizes themselves. Combining exact schemes of the sub-equations of a larger operator, and choosing where each term is evaluated in time, can yield implicit schemes whose matrix is an M-matrix, so that a discrete minimum principle, positivity and unconditional stability follow.

American options may be exercised at any time before maturity, so their value is the solution of a free-boundary (obstacle) problem: it never falls below the exercise payoff, satisfies the Black-Scholes equation where holding is optimal, and a variational inequality couples the two regions. After an implicit time discretisation, each time level becomes a linear complementarity problem. When the discrete operator is an M-matrix this problem has a unique solution.

The accuracy of a scheme is commonly reported by the maximum nodal error against a reference solution over the whole space-time grid, and by the observed order of convergence estimated from that error on successively refined grids.

## Problem

Finite-difference solvers for the Black-Scholes equation lose accuracy near the strike, where the option payoff has a kink, and standard schemes can produce negative or oscillating prices unless the time step is restricted. A recent non-standard finite-difference (NSFD) approach addresses both problems: it smooths the payoff with a polynomial of high continuity on a narrow band around the kink, and builds an implicit scheme by combining the exact finite-difference schemes of two sub-equations of the Black-Scholes operator, giving an M-matrix system that preserves positivity for every step size. The method was presented and validated for European calls. Your task is to use it for an American put, where the M-matrix structure makes the early-exercise problem at each time level uniquely solvable: from the contract, market and grid parameters, compute the NSFD American put value at the strike, together with the European NSFD put on the same grid and its discretisation error against the exact solution of the smoothed-payoff problem.

With t the time to maturity, the European value u(s, t) solves u_t = (sigma^2/2) s^2 u_ss + r s u_s - r u, with the payoff as initial condition. The NSFD scheme combines the exact scheme of the Cauchy-Euler equation (sigma^2/2) s^2 u'' + r s u' - r u = 0, whose independent solutions are s and s^(-alpha) with alpha = 2r/sigma^2, with the exact scheme of the decay equation u_t = -r u, and it is implicit in time. The initial condition is the payoff smoothed with the method's ninth-degree C^4 polynomial on the band of half-width eps around the kink. For the American put, each implicit time level becomes a discrete linear complementarity problem: at every interior node the scheme's time difference minus its spatial operator applied to the new level is non-negative, the new value is at least the exercise value, and at least one of these two inequalities holds with equality.

Use the following configuration:

- strike K = 1, risk-free rate r = 0.1, volatility sigma = 0.3, maturity T = 1
- truncated asset domain [0, S_max] with S_max = 4, smoothing half-width eps = 0.05 (a band that spans grid nodes, so the smoothed initial data are resolved)
- one uniform grid with M = 128 asset intervals, s_m = m S_max / M, and N = 128 time steps, t_k = k T / N
- initial row for both options: the smoothed put payoff Psi(K - s_m) at every node m = 0, ..., M, where Psi is the method's smoothed version of max(x, 0)
- European Dirichlet data for k >= 1: u(0, t_k) = K exp(-r t_k) and u(S_max, t_k) = 0
- American Dirichlet data for k >= 1: u(0, t_k) = K and u(S_max, t_k) = 0; the exercise value at every interior node and every time level is the smoothed payoff Psi(K - s_m), and each time level is the exact solution of its complementarity problem
- evaluate the Cauchy-Euler exact-difference weights of interior node m at the node's own index m (the source's printed index m - 1 is singular at m = 1, and index m reproduces the source's published error tables)
- maximum nodal error of the European solution E = max over k = 0, ..., N and m = 0, ..., M of |v_m^k - u(s_m, t_k)|, where u is the exact solution of the Black-Scholes equation with the smoothed payoff as initial condition, u(s, t) = exp(-r t) E[Psi(K - S_t) | S_0 = s] under risk-neutral geometric Brownian motion, so that E measures the discretisation error alone; the initial row and the boundary nodes are included

Your final answer is the American NSFD put value at s = K and t = T.

In <reasoning>, state the smoothing polynomial Psi on [-eps, eps], the Cauchy-Euler exact-difference weights, the non-standard time-step function, the implicit NSFD equation with its left, centre and right coefficients, and the complementarity conditions of the American step, as formulas. Then report, each to at least seven significant figures, alpha, the non-standard time-step function psi that replaces dt, the left, centre and right coefficients of the implicit equation at interior node m = 1, scaled so that the previous-level value enters its right-hand side as -v_1^{k-1} / (r psi), the European maximum nodal error E, the European NSFD value at s = K and t = T, the exact smoothed-payoff solution u(K, T), the American NSFD value at s = K and t = T, and the early-exercise premium (American minus European NSFD value at s = K and t = T). Report the American value to at least ten decimal places; it is graded to an absolute tolerance of 1e-8.

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

smooth_put_payoff

Goal
----
Evaluate the smoothed European put payoff used as the initial condition of the non-standard finite-difference scheme. Given asset prices, the strike and the half-width eps of the smoothing band, return phi(s) = Psi(K - s), where Psi replaces the kink of max(x, 0) at x = 0 by a polynomial on the band [-eps, eps].

```python
def smooth_put_payoff(s: "np.ndarray", strike: float, eps: float) -> "np.ndarray":
    """Smoothed put payoff phi(s) = Psi(K - s).

    Parameters
    ----------
    s : np.ndarray
        Asset prices, any shape, all finite.
    strike : float
        Strike K, finite and strictly positive.
    eps : float
        Half-width of the smoothing band, finite and strictly positive.

    Returns
    -------
    phi : np.ndarray
        Float64 array with the shape of s: 0 where K - s <= -eps, K - s where
        K - s >= eps, and the C^4 ninth-degree matching polynomial of x = K - s
        on -eps < x < eps.

    Raises
    ------
    ValueError
        If s contains non-finite values, strike is not finite and positive,
        or eps is not finite and positive.
    """
    return phi
```

### Step 2

exact_difference_weights

Goal
----
Compute the exact finite-difference weights of the Cauchy-Euler part of the Black-Scholes operator on a uniform asset grid. Given the number of asset intervals, the risk-free rate and the volatility, return the three weight arrays A1, A2 and A3 of the interior grid nodes m = 1, ..., M-1.

```python
def exact_difference_weights(n_intervals: int, rate: float, volatility: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Cauchy-Euler exact-difference weights A1, A2, A3 at m = 1, ..., M-1.

    Parameters
    ----------
    n_intervals : int
        Number M of asset intervals, an integer of at least 2.
    rate : float
        Risk-free rate r, finite and strictly positive.
    volatility : float
        Volatility sigma, finite and strictly positive.

    Returns
    -------
    A1, A2, A3 : np.ndarray
        Float64 arrays of shape (M-1,), entry m-1 holding the weight at index m
        as defined in the step background.

    Raises
    ------
    ValueError
        If n_intervals is not an integer of at least 2, or rate or volatility
        is not finite and strictly positive.
    """
    return A1, A2, A3
```

### Step 3

nsfd_coefficients

Goal
----
Assemble the coefficients of the implicit non-standard finite-difference (NSFD) scheme for the Black-Scholes equation. Given the number of asset intervals, the time step, the rate and the volatility, return the left, centre and right coefficients of the tridiagonal system at every interior node and the scaled time denominator psi1.

```python
def nsfd_coefficients(n_intervals: int, time_step: float, rate: float, volatility: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray, float]":
    """Tridiagonal NSFD coefficients at m = 1, ..., M-1 and psi1.

    Parameters
    ----------
    n_intervals : int
        Number M of asset intervals, an integer of at least 2.
    time_step : float
        Time step dt, finite and strictly positive.
    rate : float
        Risk-free rate r, finite and strictly positive.
    volatility : float
        Volatility sigma, finite and strictly positive.

    Returns
    -------
    lower, centre, upper : np.ndarray
        Float64 arrays of shape (M-1,) holding lower_m, centre_m and upper_m
        at the interior nodes m = 1, ..., M-1.
    psi1 : float
        rate times the exact decay-scheme denominator phi(time_step), as a
        Python float.

    Raises
    ------
    ValueError
        If n_intervals is not an integer of at least 2, or time_step, rate or
        volatility is not finite and strictly positive.
    """
    return lower, centre, upper, psi1
```

### Step 4

smoothed_put_exact

Goal
----
Evaluate the exact solution of the Black-Scholes equation whose initial condition is the smoothed put payoff. Given asset prices, the strike, the rate, the volatility, the time to maturity and the smoothing half-width, return u(s, tau) = exp(-r tau) E[Psi(K - S_tau) | S_0 = s], where S follows risk-neutral geometric Brownian motion and Psi is the C^4 smoothed version of max(x, 0) on the band [-eps, eps].

```python
def smoothed_put_exact(s: "np.ndarray", strike: float, rate: float, volatility: float, tau: float, eps: float) -> "np.ndarray":
    """Exact Black-Scholes solution with the smoothed put payoff as initial data.

    Parameters
    ----------
    s : np.ndarray
        Asset prices, any shape, finite and non-negative.
    strike : float
        Strike K, finite and strictly positive.
    rate : float
        Risk-free rate r, finite.
    volatility : float
        Volatility sigma, finite and strictly positive.
    tau : float
        Time to maturity, finite and non-negative.
    eps : float
        Half-width of the payoff smoothing band, finite, positive and below K.

    Returns
    -------
    price : np.ndarray
        Float64 array with the shape of s: exp(-rate tau) E[Psi(K - S_tau)]
        for s > 0 and tau > 0, K exp(-rate tau) at s = 0, and Psi(K - s) at
        tau = 0, with absolute error at most 1e-12.

    Raises
    ------
    ValueError
        If s is not finite and non-negative, strike is not finite and positive,
        rate is not finite, volatility is not finite and positive, tau is not
        finite and non-negative, or eps is not finite with 0 < eps < strike.
    """
    return price
```

### Step 5

nsfd_time_step

Goal
----
Advance the implicit NSFD scheme by one time step. Given the solution on the full asset grid at the previous time level, the interior tridiagonal coefficients, the time denominator psi1 and the two Dirichlet boundary values at the new time level, return the solution on the full grid at the new level.

```python
def nsfd_time_step(v_prev: "np.ndarray", lower: "np.ndarray", centre: "np.ndarray", upper: "np.ndarray", psi1: float, left_value: float, right_value: float) -> "np.ndarray":
    """One implicit NSFD step on the full asset grid.

    Parameters
    ----------
    v_prev : np.ndarray
        Solution at the previous time level, shape (M+1,), finite, M >= 2.
    lower, centre, upper : np.ndarray
        Coefficients Bl_m, Bc_m, Br_m at m = 1, ..., M-1, each of shape (M-1,).
    psi1 : float
        Time denominator exp(r dt) - 1, finite and strictly positive.
    left_value : float
        Dirichlet value v_0 at the new time level.
    right_value : float
        Dirichlet value v_M at the new time level.

    Returns
    -------
    v_new : np.ndarray
        Float64 array of shape (M+1,): left_value, the M-1 interior values
        solving the tridiagonal system, and right_value.

    Raises
    ------
    ValueError
        If v_prev is not one-dimensional with at least 3 finite entries, any
        coefficient array does not have shape (M-1,), psi1 is not finite and
        positive, or a boundary value is not finite.
    """
    return v_new
```

### Step 6

solve_nsfd_put

Goal
----
Solve the Black-Scholes equation for a European put with the implicit NSFD scheme on a uniform grid. Given the contract and market parameters, the truncated asset domain, the smoothing half-width and the numbers of asset intervals and time steps, return the numerical put value at every grid node and every time level.

```python
def solve_nsfd_put(strike: float, rate: float, volatility: float, maturity: float, s_max: float, eps: float, n_intervals: int, n_steps: int) -> "np.ndarray":
    """NSFD put values on the full space-time grid.

    Parameters
    ----------
    strike : float
        Strike K, finite and strictly positive, with K + eps < s_max.
    rate : float
        Risk-free rate r, finite and strictly positive.
    volatility : float
        Volatility sigma, finite and strictly positive.
    maturity : float
        Maturity T, finite and strictly positive.
    s_max : float
        Right end S_max of the asset domain, finite and larger than K + eps.
    eps : float
        Half-width of the payoff smoothing band, finite, positive and below K.
    n_intervals : int
        Number M of asset intervals, an integer of at least 2.
    n_steps : int
        Number N of time steps, an integer of at least 1.

    Returns
    -------
    grid : np.ndarray
        Float64 array of shape (N+1, M+1); grid[k, m] approximates the put
        value at time to maturity k T / N and asset price m S_max / M.

    Raises
    ------
    ValueError
        If any parameter violates the conditions above.
    """
    return grid
```

### Step 7

max_nodal_error

Goal
----
Measure the discretisation error of a numerical European put solution by its maximum nodal error. Given the full space-time grid of numerical values and the parameters that produced it, return the largest absolute difference between the numerical value and the exact solution of the smoothed-payoff Black-Scholes problem over every node of the grid.

```python
def max_nodal_error(grid: "np.ndarray", strike: float, rate: float, volatility: float, maturity: float, s_max: float, eps: float) -> float:
    """Maximum absolute error of a put grid against the exact smoothed-payoff solution.

    Parameters
    ----------
    grid : np.ndarray
        Finite array of shape (N+1, M+1) with N >= 1 and M >= 1; grid[k, m]
        is the numerical value at time to maturity k T / N and asset price
        m S_max / M.
    strike : float
        Strike K, finite and strictly positive.
    rate : float
        Risk-free rate r, finite.
    volatility : float
        Volatility sigma, finite and strictly positive.
    maturity : float
        Maturity T, finite and strictly positive.
    s_max : float
        Right end S_max of the asset domain, finite and strictly positive.
    eps : float
        Half-width of the payoff smoothing band, finite with 0 < eps < K.

    Returns
    -------
    error : float
        max over all nodes of |grid[k, m] - u(s_m, t_k)|, as a Python float.

    Raises
    ------
    ValueError
        If grid is not a finite two-dimensional array with at least two rows
        and two columns, or a scalar parameter violates the conditions above.
    """
    return error
```

### Step 8

nsfd_american_step

Goal
----
Advance the NSFD scheme for an American put by one implicit time step. Given the solution at the previous time level, the interior tridiagonal coefficients, the time denominator psi1, the exercise payoff on the grid and the two Dirichlet boundary values at the new level, return the exact solution of the discrete linear complementarity problem on the full grid.

```python
def nsfd_american_step(v_prev: "np.ndarray", lower: "np.ndarray", centre: "np.ndarray", upper: "np.ndarray", psi1: float, obstacle: "np.ndarray", left_value: float, right_value: float) -> "np.ndarray":
    """One implicit NSFD step of an American put (exact complementarity solve).

    Parameters
    ----------
    v_prev : np.ndarray
        Solution at the previous time level, shape (M+1,), finite, M >= 2.
    lower, centre, upper : np.ndarray
        Coefficients Bl_m, Bc_m, Br_m at m = 1, ..., M-1, each of shape (M-1,).
    psi1 : float
        Time denominator exp(r dt) - 1, finite and strictly positive.
    obstacle : np.ndarray
        Exercise payoff g on the full grid, shape (M+1,), finite; only the
        interior entries g_1, ..., g_{M-1} enter the complementarity problem.
    left_value : float
        Dirichlet value v_0 at the new time level.
    right_value : float
        Dirichlet value v_M at the new time level.

    Returns
    -------
    v_new : np.ndarray
        Float64 array of shape (M+1,): left_value, the M-1 interior values
        solving the linear complementarity problem, and right_value.

    Raises
    ------
    ValueError
        If v_prev is not one-dimensional with at least 3 finite entries, any
        coefficient array does not have shape (M-1,), obstacle does not have
        shape (M+1,) or is not finite, psi1 is not finite and positive, or a
        boundary value is not finite.
    """
    return v_new
```

### Step 9

solve_nsfd_american_put

Goal
----
Solve the American put problem with the implicit NSFD scheme on a uniform grid. Given the contract and market parameters, the truncated asset domain, the smoothing half-width and the numbers of asset intervals and time steps, return the numerical American put value at every grid node and every time level.

```python
def solve_nsfd_american_put(strike: float, rate: float, volatility: float, maturity: float, s_max: float, eps: float, n_intervals: int, n_steps: int) -> "np.ndarray":
    """NSFD American put values on the full space-time grid.

    Parameters
    ----------
    strike : float
        Strike K, finite and strictly positive, with K + eps < s_max.
    rate : float
        Risk-free rate r, finite and strictly positive.
    volatility : float
        Volatility sigma, finite and strictly positive.
    maturity : float
        Maturity T, finite and strictly positive.
    s_max : float
        Right end S_max of the asset domain, finite and larger than K + eps.
    eps : float
        Half-width of the payoff smoothing band, finite, positive and below K.
    n_intervals : int
        Number M of asset intervals, an integer of at least 2.
    n_steps : int
        Number N of time steps, an integer of at least 1.

    Returns
    -------
    grid : np.ndarray
        Float64 array of shape (N+1, M+1); grid[k, m] is the American put value
        at time to maturity k T / N and asset price m S_max / M.

    Raises
    ------
    ValueError
        If any parameter violates the conditions above.
    """
    return grid
```

### Step 10

nsfd_american_put_study

Goal
----
Price an American put and its European counterpart with the NSFD scheme on one grid and report the accuracy of the European solution and the early-exercise premium. Given the contract and market parameters, the asset domain, the smoothing half-width and the number of asset intervals M (with N = M time steps), return the maximum nodal error of the European NSFD solution against the exact solution of the smoothed-payoff Black-Scholes problem, the European and American NSFD values at the strike at maturity, and their difference.

```python
def nsfd_american_put_study(strike: float, rate: float, volatility: float, maturity: float, s_max: float, eps: float, n_intervals: int) -> "np.ndarray":
    """European error, European and American values at the strike, and the premium.

    Parameters
    ----------
    strike : float
        Strike K, finite and strictly positive, with K + eps < s_max.
    rate : float
        Risk-free rate r, finite and strictly positive.
    volatility : float
        Volatility sigma, finite and strictly positive.
    maturity : float
        Maturity T, finite and strictly positive.
    s_max : float
        Right end S_max of the asset domain, finite and larger than K + eps.
    eps : float
        Half-width of the payoff smoothing band, finite, positive and below K.
    n_intervals : int
        Number M of asset intervals and of time steps, an integer of at least 2.

    Returns
    -------
    result : np.ndarray
        Float64 array of shape (4,): [E, european, american, premium], where E
        is the maximum nodal error of the European M x M solution against the
        exact smoothed-payoff solution, european and american are the European and American
        M x M values at time to maturity T linearly interpolated at s = K, and
        premium = american - european.

    Raises
    ------
    ValueError
        If any parameter violates the conditions above.
    """
    return result
```
