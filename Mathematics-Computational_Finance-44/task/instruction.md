# Level-hierarchy Fourier pricing under rough stochastic variance

## Background

## Scientific background

Rough-volatility models capture short-maturity volatility behavior through nonlocal fractional dynamics, but this makes transform pricing substantially more expensive than in classical Markovian stochastic-volatility models. Under rough Heston dynamics, evaluating the log-price characteristic function at one Fourier node requires solving a nonlinear fractional Riccati-Volterra equation over the full time history.

The source paper develops scaled Gauss-Laguerre rules and a multilevel Fourier hierarchy that balance the fractional time-discretization error against quadrature error. The construction estimates the transformed integrand's exponential decay, chooses an admissible damping contour, extrapolates the required time-grid level, and allocates quadrature nodes across the coarse term and successive level corrections to meet a prescribed error budget at reduced computational cost.

## Problem

Fourier inversion values a European call from the characteristic function of the log-price, but under a rough stochastic-variance model that function is available only through the numerical solution of a nonlinear fractional integral equation, one solve per quadrature node, at a cost that grows like the square of the number of time steps. Take the variance parameters $\alpha=0.62$, $\gamma=0.1$, $\nu=0.331$, $\rho=-0.681$, $V_0=0.0392$, $\theta=0.3156$, the at-the-money contract $S_0=K=1000$, $r=0$, $T=2$, the payoff $\max(e^x-K,0)$, and the log-price started at $\log S_0$.

Report the call value produced by the level-hierarchy quadrature construction that a recent source develops for exactly this model, at total absolute tolerance $\varepsilon=8\times10^{-4}$ split evenly between its time-discretization part and its quadrature part. Fix the coarsest time grid at $32$ uniform steps over $[0,T]$ and halve the step at each further level, and use the source's own quantitative choices throughout: the exponential decay rate it derives for the transformed integrand together with the rescaling of the quadrature rule to that rate, its constrained contour-shift rule (report the shift rounded to the three decimals the source tabulates), the price-error index $p=1+\alpha$ and the cost exponent $\beta=2$ it measures for this solver, the pilot quadrature order $16$ behind its level indicator, and the smoothness indices $s_0=8$ and $s=7$ it reports for this parameter set.

Estimate the two error prefactors as the smallest algebraic covers over the fixed pilot grid $N\in\{2,4,6,8,10,12,14,16\}$, each measured against a $64$-node rule at the same rate and shift: with $Q_N$ the resulting quadrature functional, $g_0$ the level-zero integrand and $\Delta g_1=g_1-g_0$ the first level difference, take $A_0=\max_N |Q_{64}[g_0]-Q_N[g_0]|\,N^{s_0/2}$ and $A_1=\max_N |Q_{64}[\Delta g_1]-Q_N[\Delta g_1]|\,N^{s/2}$. Propagate $A_1$ to the finer correction levels with the source's own level-scaling rule, and round every allocated node count up to the next integer.

Report in the reasoning the decay rate, the contour shift, the first level difference, the selected level index, both prefactors, and the number of quadrature nodes assigned to every level. Your final answer must be a single number: the option value, to at least ten significant digits.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags.
Enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary.
A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_payoff_fourier_transform

Goal
----
Evaluate the generalized Fourier transform of a European call payoff. The transform is required on a horizontal line of the complex plane whose imaginary part is strongly negative, where its magnitude can be smaller than 1e-300 while the intermediate factors are not.

```python
import numpy as np

def payoff_fourier_transform(xi_real, xi_imag, strike):
    """Return the payoff transform on a horizontal line of the complex plane.

    For a European call with the given strike the transform is

        P(xi) = -strike ** (1 - 1j * xi) / (xi ** 2 + 1j * xi),

    evaluated at ``xi = xi_real + 1j * xi_imag``.  The value must remain
    accurate when the line lies deep in the lower half plane, for example
    ``strike = 1000`` with ``xi_imag = -100``, so the complex power must not be
    routed through an intermediate quantity that leaves the double-precision
    range.

    Parameters
    ----------
    xi_real : array_like
        Real parts of the evaluation points, interpreted as a one-dimensional
        sequence of length ``n``.  All entries must be finite.
    xi_imag : array_like
        Imaginary parts, same shape as ``xi_real``.  All entries must be finite.
    strike : float
        Strictly positive strike.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``(n,)`` holding ``P(xi)``.

    Raises
    ------
    ValueError
        If ``xi_real`` and ``xi_imag`` have different shapes, if either holds a
        non-finite entry, if ``strike`` is not finite and strictly positive, or
        if any evaluation point is a zero of ``xi ** 2 + 1j * xi``.
    """
    return result
```

### Step 2

02_riccati_volterra_nodes

Goal
----
Solve the nonlinear fractional integral equation that carries the variance dynamics, and return its values on a uniform grid. The equation has a weakly singular power-law kernel, so the solution's entire history affects every grid value.

```python
import numpy as np


def riccati_volterra_nodes(xi_real, xi_imag, model, horizon, steps):
    """Return the prescribed discrete variance-equation history.

    Interpret the two coordinate arrays as one complex Fourier line and apply
    the source's history-dependent discretization of the nonlinear fractional
    variance equation on a uniform grid over ``[0, horizon]``.  Preserve the
    complete prior history at every update and evaluate all frequencies in
    complex arithmetic.  The initial row corresponds to time zero and must
    vanish exactly.

    Parameters
    ----------
    xi_real : array_like
        Real parts of the evaluation points, a one-dimensional sequence of
        length ``n`` with finite entries.
    xi_imag : array_like
        Imaginary parts, same shape as ``xi_real``, finite.
    model : sequence of float
        Six finite numbers ``(alpha, gamma, nu, rho, v0, theta)`` with
        ``0 < alpha <= 1``, ``gamma > 0``, ``nu > 0``, ``abs(rho) < 1``,
        ``v0 > 0`` and ``theta > 0``.  Only the first four are used here.
    horizon : float
        Strictly positive end of the time interval.
    steps : int
        Strictly positive number of uniform grid intervals.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``(steps + 1, n)`` containing the prescribed
        nodal history in increasing time order.

    Raises
    ------
    ValueError
        If ``model`` violates any stated bound, if ``horizon`` is not finite and
        strictly positive, if ``steps`` is not a positive integer, or if
        ``xi_real`` and ``xi_imag`` disagree in shape or hold a non-finite entry.
    """
    return result
```

### Step 3

03_characteristic_exponent

Goal
----
Assemble the fully discrete log-characteristic function of the log-price from nodal values of the variance equation. Only the grid values already available are used; no continuous reconstruction of the underlying trajectory is performed.

```python
import numpy as np

def characteristic_exponent(xi_real, xi_imag, h_nodes, model, market):
    """Return the discrete log-characteristic function on a complex line.

    Write ``xi = xi_real + 1j * xi_imag``, ``alpha = model[0]``,
    ``gamma = model[1]``, ``nu = model[2]``, ``rho = model[3]``,
    ``v0 = model[4]``, ``theta = model[5]``, ``spot = market[0]``,
    ``rate = market[2]``, ``horizon = market[3]``, and

        F(xi, h) = -(xi ** 2 + 1j * xi) / 2
                   + gamma * (1j * xi * rho * nu - 1) * h
                   + (gamma * nu) ** 2 * h ** 2 / 2.

    Let ``steps = h_nodes.shape[0] - 1`` and ``dt = horizon / steps``.  With

        J_j = theta * gamma * h_nodes[j] + v0 * F(xi, h_nodes[j]),

    formed at every grid index ``j = 0, ..., steps``, return

        G(xi) = 1j * xi * (log(spot) + rate * horizon)
                + dt * ( J_0 / 2 + sum_{j=1}^{steps-1} J_j + J_steps / 2 ).

    The index ``j = 0`` contributes even though ``h_nodes[0]`` vanishes.

    Parameters
    ----------
    xi_real : array_like
        Real parts of the evaluation points, one-dimensional of length ``n``.
    xi_imag : array_like
        Imaginary parts, same shape as ``xi_real``.
    h_nodes : numpy.ndarray
        Complex array of shape ``(steps + 1, n)`` holding the nodal values of
        the variance equation on the same uniform grid, with ``steps >= 1``.
    model : sequence of float
        Six finite numbers ``(alpha, gamma, nu, rho, v0, theta)`` with
        ``0 < alpha <= 1``, ``gamma > 0``, ``nu > 0``, ``abs(rho) < 1``,
        ``v0 > 0`` and ``theta > 0``.
    market : sequence of float
        Four finite numbers ``(spot, strike, rate, horizon)`` with
        ``spot > 0``, ``strike > 0`` and ``horizon > 0``.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``(n,)`` holding ``G(xi)``.

    Raises
    ------
    ValueError
        If ``model`` or ``market`` violates a stated bound, if ``xi_real`` and
        ``xi_imag`` disagree in shape or hold a non-finite entry, or if
        ``h_nodes`` is not a two-dimensional array with ``n`` columns and at
        least two rows.
    """
    return result
```

### Step 4

04_integrand_decay_rate

Goal
----
Compute the exponential decay rate that the damped Fourier integrand inherits from the variance dynamics at a given maturity. The rate combines a long-run variance contribution with a maturity term whose exponent is set by the roughness of the variance path.

```python
import numpy as np


def integrand_decay_rate(model, horizon):
    """Return the prescribed exponential decay rate at one maturity.

    Use the source's rough-variance decay estimate for the supplied parameter
    ordering.  The estimate combines correlation, volatility of volatility,
    accumulated mean variance, and the rough initial-variance contribution at
    ``horizon``.  Evaluate the expression in double precision, including its
    finite limiting case when the roughness index equals one.

    Parameters
    ----------
    model : sequence of float
        Six finite numbers ``(alpha, gamma, nu, rho, v0, theta)`` with
        ``0 < alpha <= 1``, ``gamma > 0``, ``nu > 0``, ``abs(rho) < 1``,
        ``v0 > 0`` and ``theta > 0``.
    horizon : float
        Strictly positive maturity.

    Returns
    -------
    float
        Strictly positive finite rate for the exponential quadrature weight.

    Raises
    ------
    ValueError
        If ``model`` violates a stated bound or ``horizon`` is not finite and
        strictly positive.
    """
    return result
```

### Step 5

05_exponential_weight_rule

Goal
----
Build the quadrature rule adapted to an exponentially decaying weight on the positive half line. The rate of the weight is a free parameter, so the rule must follow the weight rather than assume the unit rate.

```python
import numpy as np

def exponential_weight_rule(order, scale):
    """Return nodes and weights of the maximal-degree rule for a decaying weight.

    Return one-dimensional arrays ``u`` and ``w`` of length ``order`` such that

        sum_{n} w[n] * P(u[n]) = integral_0^infinity exp(-scale * x) P(x) dx

    holds exactly for every polynomial ``P`` of degree at most
    ``2 * order - 1``.  The nodes must be returned in increasing order.  A
    consequence that can be used as a self-check is
    ``sum(w) == 1 / scale``.

    Parameters
    ----------
    order : int
        Strictly positive number of nodes.
    scale : float
        Strictly positive decay rate of the weight.

    Returns
    -------
    tuple of numpy.ndarray
        ``(u, w)``, each a real array of shape ``(order,)`` with strictly
        positive entries and increasing ``u``.

    Raises
    ------
    ValueError
        If ``order`` is not a positive integer or ``scale`` is not finite and
        strictly positive.
    """
    return result
```

### Step 6

06_damped_fourier_integrand

Goal
----
Assemble the real integrand whose integral over the positive half line prices a European call, with the contour shifted into the lower half plane. The integrand is built from the discrete pieces already available, so its value depends on the resolution of the time grid used to obtain them.

```python
import numpy as np

def damped_fourier_integrand(u, damping, steps, model, market):
    """Return the real damped Fourier integrand at a given time resolution.

    With ``rate = market[2]``, ``horizon = market[3]`` and
    ``xi = u + 1j * damping``, the integrand is

        g(u) = exp(-rate * horizon) / (2 * pi)
               * real( exp(G(xi)) * P(xi) ),

    where ``P`` is the European-call payoff transform for strike ``market[1]``
    and ``G`` is the discrete log-characteristic function assembled from the
    nodal values of the variance equation on a uniform grid of ``steps``
    intervals over ``[0, horizon]``.

    Implementation requirement: obtain ``P``, the nodal values and ``G`` by
    calling the previously defined public functions rather than reproducing
    their formulas here.

    Parameters
    ----------
    u : array_like
        Real evaluation points, interpreted as a one-dimensional sequence of
        length ``n`` with finite entries.
    damping : float
        Imaginary part of the contour; must satisfy ``damping < -1``.
    steps : int
        Strictly positive number of uniform time intervals.
    model : sequence of float
        Six finite numbers ``(alpha, gamma, nu, rho, v0, theta)`` with
        ``0 < alpha <= 1``, ``gamma > 0``, ``nu > 0``, ``abs(rho) < 1``,
        ``v0 > 0`` and ``theta > 0``.
    market : sequence of float
        Four finite numbers ``(spot, strike, rate, horizon)`` with
        ``spot > 0``, ``strike > 0`` and ``horizon > 0``.

    Returns
    -------
    numpy.ndarray
        Real array of shape ``(n,)`` holding ``g(u)``.

    Raises
    ------
    ValueError
        If ``model`` or ``market`` violates a stated bound, if ``damping`` is
        not strictly below ``-1``, if ``steps`` is not a positive integer, or if
        ``u`` holds a non-finite entry.
    """
    return result
```

### Step 7

07_select_damping_parameter

Goal
----
Choose the contour shift used by the damped Fourier representation. The shift is not free: it must keep the payoff transform integrable, and among the admissible values one is singled out by the size of the integrand at the origin.

```python
import numpy as np


def select_damping_parameter(model, market, steps):
    """Return the prescribed contour shift at one time resolution.

    Apply the source's constrained selection rule to the damped Fourier
    integrand at the origin using ``steps`` uniform time intervals.  Search the
    closed admissible interval ``[-40, -1.5]`` accurately enough that the
    selected shift is stable when rounded to three decimals.  Reuse the earlier
    public integrand function when evaluating the selection criterion.

    Parameters
    ----------
    model : sequence of float
        Six finite numbers ``(alpha, gamma, nu, rho, v0, theta)`` with
        ``0 < alpha <= 1``, ``gamma > 0``, ``nu > 0``, ``abs(rho) < 1``,
        ``v0 > 0`` and ``theta > 0``.
    market : sequence of float
        Four finite numbers ``(spot, strike, rate, horizon)`` with
        ``spot > 0``, ``strike > 0`` and ``horizon > 0``.
    steps : int
        Strictly positive number of uniform time intervals.

    Returns
    -------
    float
        Selected shift in ``[-40, -1.5]``, rounded to three decimals.

    Raises
    ------
    ValueError
        If ``model`` or ``market`` violates a stated bound, or if ``steps`` is
        not a positive integer.
    """
    return result
```

### Step 8

08_select_discretization_level

Goal
----
Decide how far the time grid has to be refined before the remaining bias sits inside a prescribed budget. The decision is taken from a single measured gap between the two coarsest grids and an assumed algebraic decay of that gap.

```python
import numpy as np


def select_discretization_level(model, market, base_steps, damping, scale,
                                convergence_order, eps_disc, pilot_order):
    """Return the prescribed refinement index for a bias budget.

    Measure the first gap between the two coarsest time grids with one shared
    ``pilot_order`` exponential-weight quadrature rule.  Apply the source's
    single-gap bias extrapolation using ``convergence_order`` and ``eps_disc``;
    level zero has ``base_steps`` intervals and every subsequent level doubles
    that count.  Return level one when the measured gap vanishes.  Reuse the
    earlier public quadrature and integrand functions.

    Parameters
    ----------
    model : sequence of float
        Six finite numbers ``(alpha, gamma, nu, rho, v0, theta)`` with
        ``0 < alpha <= 1``, ``gamma > 0``, ``nu > 0``, ``abs(rho) < 1``,
        ``v0 > 0`` and ``theta > 0``.
    market : sequence of float
        Four finite numbers ``(spot, strike, rate, horizon)`` with
        ``spot > 0``, ``strike > 0`` and ``horizon > 0``.
    base_steps : int
        Strictly positive number of time intervals at level zero.
    damping : float
        Contour shift, strictly below ``-1``.
    scale : float
        Strictly positive decay rate of the quadrature weight.
    convergence_order : float
        Strictly positive assumed algebraic order of the bias in the step size.
    eps_disc : float
        Strictly positive bias budget.
    pilot_order : int
        Strictly positive number of quadrature nodes used for the measurement.

    Returns
    -------
    int
        Refinement index, at least one.

    Raises
    ------
    ValueError
        If ``model`` or ``market`` violates a stated bound, if ``damping`` is
        not strictly below ``-1``, if ``base_steps`` or ``pilot_order`` is not a
        positive integer, or if ``scale``, ``convergence_order`` or ``eps_disc``
        is not strictly positive.
    """
    return result
```

### Step 9

09_hierarchical_fourier_price

Goal
----
Run the complete adaptive pricing experiment and return the option value. Every free choice is fixed by the inputs: the contour shift, the refinement index, the two error prefactors, and how many quadrature nodes each refinement level receives.

```python
import numpy as np


def hierarchical_fourier_price(model, market, tolerance, base_steps,
                               level_zero_index, correction_index,
                               cost_exponent, pilot_order, reference_order):
    """Return the option value produced by the prescribed level hierarchy.

    Split ``tolerance`` evenly between time bias and quadrature error.  Reuse
    the earlier public functions to select the decay rate, contour shift, and
    refinement depth.  Estimate the level-zero and first-correction algebraic
    error covers on the fixed pilot orders ``(2, 4, 6, 8, 10, 12, 14, 16)``
    against ``reference_order``.  Propagate the correction cover through the
    remaining levels with the prescribed time-step scaling.

    Allocate integer quadrature orders with the source's work-minimizing rule,
    using ``level_zero_index``, ``correction_index``, and ``cost_exponent``.
    Assemble the resulting telescoping price with a shared quadrature rule for
    both terms of each level difference.  Use the supplied parameters and the
    preceding public functions for every intermediate choice.

    Parameters
    ----------
    model : sequence of float
        Six finite numbers ``(alpha, gamma, nu, rho, v0, theta)`` with
        ``0 < alpha <= 1``, ``gamma > 0``, ``nu > 0``, ``abs(rho) < 1``,
        ``v0 > 0`` and ``theta > 0``.
    market : sequence of float
        Four finite numbers ``(spot, strike, rate, horizon)`` with
        ``spot > 0``, ``strike > 0`` and ``horizon > 0``.
    tolerance : float
        Strictly positive total error budget.
    base_steps : int
        Strictly positive number of time intervals at level zero.
    level_zero_index : float
        Strictly positive algebraic index of the level-zero quadrature error.
    correction_index : float
        Strictly positive algebraic index shared by the level corrections.
    cost_exponent : float
        Strictly positive exponent of the per-level cost weights.
    pilot_order : int
        Strictly positive number of nodes used to measure the first bias gap.
    reference_order : int
        Strictly positive number of nodes used as the quadrature reference in
        the two prefactors.

    Returns
    -------
    float
        Finite call value produced by the complete level hierarchy.

    Raises
    ------
    ValueError
        If ``model`` or ``market`` violates a stated bound, if ``tolerance``,
        ``level_zero_index``, ``correction_index`` or ``cost_exponent`` is not
        strictly positive, or if ``base_steps``, ``pilot_order`` or
        ``reference_order`` is not a positive integer.
    """
    return result
```
