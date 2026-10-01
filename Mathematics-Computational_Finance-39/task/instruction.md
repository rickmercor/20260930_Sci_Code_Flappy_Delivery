# Mathematics-Computational_Finance-39

## Background

Geometric Asian options settle on the geometric average of the underlying over a set of fixing dates. Under stochastic volatility driven by a memory kernel, the average is not a Markov functional of any finite-dimensional state, and pricing and hedging rely on the affine transform of the joint law of the terminal log-price and the log-geometric average, on contour representations of the payoff, and on quadratic hedging in the incomplete stock-and-cash market.

## Problem

Asian options settle on an average of the underlying over a set of fixing dates, and under stochastic volatility with a memory kernel the average is not a Markov functional of any finite state. A recent source derives semi-closed transform formulas for discretely monitored geometric Asian options in the Volterra-Heston model, by feeding the monitoring measure into the affine transform as a measure-valued input, and derives from the same transform the variance-optimal strategy in the stock-and-cash market, where the option cannot be replicated. Recover that construction from the source. The load-bearing choices are the source's and are not derivable from the statement below: how the discrete monitoring measure enters the first component of the Riccati-Volterra equation and how the already observed spot fixing is treated, what the forward variance curve solves, the form of the Riccati function, how the payoff is represented as a contour integral and against which measure, and how the variance-optimal holding is represented and what separates it from the delta of the price.

The model is the Volterra-Heston model: $dS_t = \sqrt{\nu_t}\,S_t\,dW^S_t$ at zero rate, $\nu_t = \nu_0 + \int_0^t K(t-s)\,\kappa(\theta - \nu_s)\,ds + \int_0^t K(t-s)\,\sigma\sqrt{\nu_s}\,dW^\nu_s$, $d\langle W^S, W^\nu\rangle_t = \rho\,dt$, with the completely monotone kernel $K(t) = (1 - e^{-\lambda t})/(\lambda t)$, $K(0) = 1$, whose Bernstein measure is uniform on $[0, \lambda]$. The contract is a fixed-strike call on the geometric average $G_T = \exp\big(\frac{1}{N+1}\sum_{j=0}^{N} \log S_{t_j}\big)$ of the $N + 1$ fixings $t_j = jT/N$, the spot fixing at $t_0 = 0$ included, with payoff $(G_T - K)^+$. Implement eight functions on a uniform time grid of $n = Nm$ cells on $[0, T]$, with these conventions: every Volterra integral on the grid is a product-trapezoid sum with the kernel evaluated at node differences; the forward variance is implicit at the new node; a Riccati-Volterra step uses the cell's own monitoring coefficient at both ends of the cell, a predictor that repeats the left-end integrand at the right end, and one corrector pass; the time integrals in the exponent of the transform are cellwise trapezoid sums in the integration variable with the coefficients at the reflected time; the contour is the vertical line $\mathrm{Re}\,z = R$ with $\mathrm{Im}\,z \in [-Y, Y]$ split into equal panels carrying Gauss-Legendre nodes, integrated against the Lebesgue measure of the imaginary part. Transform arguments $s$ are passed as separate real and imaginary arrays, and complex results are returned as stacked real and imaginary parts.

`vha_monitoring_mass(T, N, m)` returns, on each cell, the value the source's measure-valued input takes on the interior of that cell for the equal-weight monitoring measure. `vha_forward_variance(nu0, kappa, theta, lam, T, n)` returns the forward variance curve $E[\nu_t]$ on the nodes, from the linear Volterra equation the variance dynamics imply. `vha_riccati_volterra(s_re, s_im, w, mass, kappa, sigma, rho, lam, T)` returns the Riccati-Volterra solution on the nodes, shape $(2, M, n+1)$, for the piecewise-constant first component built from the monitoring input and the arguments $(s, w)$. `vha_transform(s_re, s_im, w, S0, r, T, N, mass, xi, psi2, sigma, rho)` returns $E[G_T^s S_T^w]$ at time zero, shape $(2, M)$, by the source's transform formula. `vha_contour(R, Y, panels, nq)` returns the imaginary parts of the nodes and the weights, shape $(2, \mathrm{panels}\cdot nq)$, in increasing order. `vha_price(K, T, r, R, y, wq, h1, hz)` takes the contour arrays and the transform values at the argument pair $(1, 0)$ and at $(z, 0)$ for the nodes $z = R + iy$, and returns the call price, $E[G_T]$ and $E[\min(G_T, K)]$, the last two undiscounted, through the source's contour representation of the payoff. `vha_hedge(S0, K, N, sigma, rho, R, y, wq, h1, hz, psi2_1, psi2_z)` takes in addition the Riccati-Volterra solutions for the same argument pairs and returns, at zero rate, the variance-optimal initial holding in the stock by the source's representation, and the partial derivative of the price with respect to the spot with the spot fixing held fixed. `vha_audit(S0, K, T, nu0, kappa, theta, sigma, rho, lam, N, m, R, Y, panels, nq)`, the orchestrator, must call the earlier functions rather than reimplementing them and returns seven values: the forward variance at maturity, the real part of the Riccati-Volterra solution at maturity for the argument pair $(1, 0)$, $E[G_T]$, $E[\min(G_T, K)]$, the price, the held-fixing delta, and the variance-optimal initial holding.

All outputs are float64, finite and deterministic: two runs on identical inputs must agree exactly. Validate inputs and raise `ValueError` on non-finite values, on arrays of the wrong length or shape, on non-positive $S_0$, $K$, $T$, $\kappa$, $\theta$, $\sigma$, $\lambda$ or $Y$, on negative $r$ or $\nu_0$, on $\rho$ outside $[-1, 1]$, on $R$ outside $(0, 1)$, and on non-integral or non-positive counts.

Evaluate the audit at $S_0 = 100$, $K = 105$, $T = 1$, $\nu_0 = 0.03$, $\kappa = 2.5$, $\theta = 0.06$, $\sigma = 0.5$, $\rho = -0.8$, $\lambda = 2$, $N = 8$, $m = 50$, $R = 0.5$, $Y = 120$, $120$ panels of $32$ nodes.

In your reasoning report the conventions you used, and justify each from the source: how the discrete monitoring measure enters the first component of the Riccati-Volterra equation, which interval convention that component takes, and how the already observed spot fixing enters the transform; what the forward variance curve solves under this kernel; the form of the Riccati function; how the payoff is written as a contour integral of power functions of the average and against which measure on the line; how the variance-optimal initial holding is represented in transform coordinates and what separates it from the delta of the price; the condition on the transform arguments under which the transform exists; and the source's representation of the minimal mean-squared hedging error.

Report numerically, as evidence that the chain was executed: the forward variance at maturity; the real part of the Riccati-Volterra solution at maturity for the argument pair $(1, 0)$; the expectation of the geometric average; the expectation of the minimum of the average and the strike; the price; the held-fixing delta; and the variance-optimal initial holding. These are the scalars that determine the final number.

As the final answer, report the variance-optimal initial holding in the stock, to six significant figures.

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
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
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

vha_monitoring_mass

Goal
----
Builds the measure-valued monitoring input of the transform on the fixed grid for an equal-weight discrete monitoring measure that includes the spot fixing.

```python
def vha_monitoring_mass(T, N, m):
    r"""T: positive float, maturity. N: positive integer, number of monitoring intervals; the fixings
    are $t_j = jT/N$, $j = 0, \dots, N$, the spot fixing at $t_0 = 0$ included. m: positive integer,
    grid cells per monitoring interval.

    Returns a numpy float64 array of shape $(N m,)$: on each cell of the uniform grid with $Nm$ cells
    on $[0, T]$, the value that the source's measure-valued input to the Riccati-Volterra equation
    takes on the interior of that cell for the equal-weight monitoring measure, namely the
    monitoring mass on the closed interval $[T - t, T]$ for $t$ inside the cell.

    Raises:
        ValueError: on non-finite or non-positive T, or non-integral or non-positive N or m.
    """
    return None
```

### Step 2

vha_forward_variance

Goal
----
Solves the linear Volterra equation for the forward variance curve of the model on the fixed grid.

```python
def vha_forward_variance(nu0, kappa, theta, lam, T, n):
    r"""nu0: non-negative float, spot variance. kappa, theta: positive floats, mean-reversion speed and
    level. lam: positive float, kernel parameter of $K(t) = (1 - e^{-\lambda t})/(\lambda t)$, $K(0) = 1$.
    T: positive float. n: positive integer, number of grid cells on $[0, T]$.

    Returns a numpy float64 array of shape $(n + 1,)$: the forward variance curve $\xi_0(t) = E[\nu_t]$
    of the Volterra-Heston variance dynamics on the nodes $t_i = iT/n$, obtained from the linear
    Volterra equation the dynamics imply, by the product-trapezoid rule on the grid with the kernel
    at node differences, implicit at the new node.

    Raises:
        ValueError: on a negative or non-finite nu0, non-finite or non-positive kappa, theta, lam or T,
            or non-integral or non-positive n.
    """
    return None
```

### Step 3

vha_riccati_volterra

Goal
----
Solves the Riccati-Volterra equation of the model for a piecewise-constant, measure-valued first component and a vector of complex transform arguments.

```python
def vha_riccati_volterra(s_re, s_im, w, mass, kappa, sigma, rho, lam, T):
    r"""s_re, s_im: float arrays of equal length $M$, real and imaginary parts of the transform
    arguments $s$. w: real float, the second transform argument. mass: $(n,)$ array, the cell values
    from the monitoring step. kappa, sigma: positive floats; rho: float in $[-1, 1]$; lam: positive
    float, kernel parameter; T: positive float.

    Returns a numpy float64 array of shape $(2, M, n + 1)$: real and imaginary parts of the solution
    $\psi_2$ of the source's Riccati-Volterra equation on the nodes $t_i = iT/n$, for the coefficient
    $\psi_1 = s\,\mathrm{mass} + w$ on each cell, with the kernel $K(t) = (1 - e^{-\lambda t})/(\lambda t)$.
    The convolution is the product-trapezoid rule on the grid with the kernel at node differences and
    the cell's own coefficient at both ends of each cell; the value at a new node is obtained from a
    predictor that repeats the left-end integrand at the right end, followed by one corrector pass.

    Raises:
        ValueError: on s_re and s_im of different or zero length, non-finite entries, a mass array that
            is empty or not finite, non-finite or non-positive kappa, sigma, lam or T, or rho outside
            $[-1, 1]$.
    """
    return None
```

### Step 4

vha_transform

Goal
----
Evaluates the joint transform of the geometric average and the terminal price at time zero from the Riccati-Volterra solution and the forward variance curve.

```python
def vha_transform(s_re, s_im, w, S0, r, T, N, mass, xi, psi2, sigma, rho):
    r"""s_re, s_im, w: as in the Riccati step. S0: positive float, spot. r: non-negative float, rate.
    T: positive float. N: positive integer, monitoring intervals. mass: $(n,)$ array from the
    monitoring step. xi: $(n + 1,)$ array, the forward variance curve. psi2: $(2, M, n + 1)$ array from
    the Riccati step. sigma: positive float; rho: float in $[-1, 1]$.

    Returns a numpy float64 array of shape $(2, M)$: real and imaginary parts of
    $H_0(s, w) = E[G_T^s S_T^w]$ at time zero, where $G_T$ is the geometric average over all $N + 1$
    fixings, the spot fixing included, by the source's transform formula for the model. The two
    time integrals in the exponent are cellwise trapezoid sums in the integration variable, with the
    coefficients evaluated at the reflected time $T - x$.

    Raises:
        ValueError: on mismatched or non-finite array shapes, non-finite or non-positive S0, T or
            sigma, a negative r, a non-integral or non-positive N, or rho outside $[-1, 1]$.
    """
    return None
```

### Step 5

vha_contour

Goal
----
Builds the composite Gauss-Legendre nodes and weights on the vertical contour used by the payoff representation.

```python
def vha_contour(R, Y, panels, nq):
    r"""R: float in $(0, 1)$, real part of the contour. Y: positive float, half-length of the imaginary
    range. panels, nq: positive integers, number of equal panels on $[-Y, Y]$ and Gauss-Legendre nodes
    per panel.

    Returns a numpy float64 array of shape $(2, \mathrm{panels}\cdot nq)$: row 0 the imaginary parts
    $y$ of the nodes $z = R + iy$ on the vertical line, row 1 the quadrature weights for integration
    against the Lebesgue measure of the imaginary part, both in increasing order of $y$.

    Raises:
        ValueError: on R outside $(0, 1)$, non-finite or non-positive Y, or non-integral or
            non-positive panels or nq.
    """
    return None
```

### Step 6

vha_price

Goal
----
Assembles the fixed-strike geometric Asian call price from the transform values on the contour through the source's representation of the payoff.

```python
def vha_price(K, T, r, R, y, wq, h1, hz):
    r"""K: positive float, strike. T: positive float. r: non-negative float, rate. R: float in $(0, 1)$,
    real part of the contour. y, wq: $(M,)$ arrays, the imaginary parts of the contour nodes and the
    weights from the contour step. h1: $(2, 1)$ array, the transform at the argument pair $(1, 0)$.
    hz: $(2, M)$ array, the transform at the argument pairs $(z, 0)$ for the nodes $z = R + iy$.

    Returns a numpy float64 array of shape $(3,)$: the time-zero price of the fixed-strike discretely
    monitored geometric Asian call with strike K, the expectation $E[G_T]$ of the geometric average,
    and the expectation $E[\min(G_T, K)]$, the last two undiscounted, through the source's contour
    representation of the payoff.

    Raises:
        ValueError: on non-finite or non-positive K or T, negative r, R outside $(0, 1)$, arrays that
            are not finite, or y, wq and hz of inconsistent length.
    """
    return None
```

### Step 7

vha_hedge

Goal
----
Assembles the variance-optimal initial stock holding for the geometric Asian call from the transform values and the Riccati-Volterra solutions on the contour, together with the held-fixing delta.

```python
def vha_hedge(S0, K, N, sigma, rho, R, y, wq, h1, hz, psi2_1, psi2_z):
    r"""S0: positive float, spot. K: positive float, strike. N: positive integer, monitoring intervals.
    sigma: positive float; rho: float in $[-1, 1]$. R: float in $(0, 1)$, real part of the contour.
    y, wq: $(M,)$ contour arrays. h1: $(2, 1)$ and hz: $(2, M)$, the transform at $(1, 0)$ and at
    $(z, 0)$ for the nodes $z = R + iy$. psi2_1:
    $(2, 1, n+1)$ and psi2_z: $(2, M, n+1)$, the Riccati-Volterra solutions for the same arguments,
    at zero rate.

    Returns a numpy float64 array of shape $(2,)$: the variance-optimal initial holding in the stock
    for the fixed-strike discretely monitored geometric Asian call, by the source's representation of
    the variance-optimal strategy in transform coordinates, and the partial derivative of the price
    with respect to the spot with the spot fixing held fixed.

    Raises:
        ValueError: on non-finite or non-positive S0, K or sigma, rho outside $[-1, 1]$, R outside
            $(0, 1)$, a non-integral or non-positive N, arrays that are not finite, or inconsistent shapes.
    """
    return None
```

### Step 8

vha_audit

Goal
----
Runs the whole chain from the monitoring measure to the variance-optimal holding and reports every intermediate the construction depends on.

```python
def vha_audit(S0, K, T, nu0, kappa, theta, sigma, rho, lam, N, m, R, Y, panels, nq):
    r"""Parameters as in the hedging step, at zero rate.

    The orchestrator. It must call the earlier functions rather than reimplementing them.
    Returns a numpy float64 array of shape $(7,)$: the forward variance at maturity, the real part
    of the Riccati-Volterra solution at maturity for the argument pair $(1, 0)$, the expectation of
    the geometric average, the expectation of the minimum of the average and the strike, the price of
    the call, the held-fixing delta, and the variance-optimal initial holding.

    Raises:
        ValueError: whenever any of the functions it calls would raise.
    """
    return None
```
