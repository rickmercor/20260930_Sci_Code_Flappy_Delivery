# Mathematics-Computational_Finance-34

## Background

Fourier methods are the workhorse of option pricing and calibration in stochastic volatility models: when the characteristic function of the log price is available, valuation becomes the numerical evaluation of a deterministic contour integral of an analytic, possibly oscillatory function. In classical affine models the characteristic function is a closed-form expression, so the quadrature is essentially free. Rough volatility models break this economy. Empirical work on implied volatility surfaces — in particular the steep at-the-money skew at short maturities — motivates driving the variance process by a fractional kernel, and in the rough Heston model the characteristic function is only semi-explicit: it is the exponential of a functional of the solution of a fractional Riccati equation, a nonlinear Volterra equation with a weakly singular power kernel. That equation has no closed form and must be solved numerically afresh at every Fourier argument. Because the kernel carries memory, a direct time-stepping scheme evaluates a history sum over all previous grid points at every step, so one solve on a grid with M steps costs O(M^2) operations, and the solve is repeated at every quadrature node.

This creates a genuine two-error budget problem. The computed price carries a time-discretization error, controlled by the Riccati grid resolution, and a Fourier quadrature error, controlled by the number of nodes; the cost of one node evaluation depends on the grid, so the two choices are coupled through the total work. Choosing the grid too fine wastes O(M^2) solves on accuracy the quadrature cannot see; choosing too many nodes wastes expensive solves on a discretization floor. A principled method must therefore split a single prescribed tolerance between the two error sources and pick both resolutions from computable quantities — neither the exact price nor the exact discretization error is available, so practical selection has to rest on extrapolation from computable indicators and on empirically fitted error models.

The placement of quadrature nodes is a second genuine difficulty. The damped Fourier integrand of the call decays essentially exponentially in the integration variable, but the effective decay rate varies by orders of magnitude with the model parameters, the damping, and especially the maturity. A quadrature rule whose weight is fixed once and for all can place almost all of its nodes where the integrand is negligible, or too few where it carries mass, and this loss is documented to be severe away from moderate maturities. Adapting the rule's weight to the estimated decay of the specific integrand at hand restores the expected rapid convergence without changing the rule's family.

Finally, hierarchies help when a fine grid is unavoidable. Refining the Riccati grid changes the integrand only slightly; the difference between consecutive levels is small and gets smaller with depth. Splitting the integrand accordingly lets the expensive fine-grid solves be performed at only a few nodes, while the bulk of the nodes are spent where solves are cheap, with substantial measured savings over evaluating everything at the finest grid. What the correct split is, how many nodes each term deserves, and which computable quantities drive those choices are precisely what the source specifies and what this benchmark asks to be reproduced exactly.

The benchmark quantity is the deterministic output of the frozen pipeline at the prescribed tolerance — the number the method itself commits to — so it is reproducible to all reported digits, and it intentionally differs from the exact rough Heston price by an amount within, but not far below, the tolerance. Independent high-accuracy pricing methods therefore do not reproduce the requested scalar.

## Problem

A recent line of work develops a hierarchical Fourier pricing framework with joint error control for the rough Heston model, in which the characteristic function is available only through the numerical solution of a fractional Riccati equation. The framework couples the Riccati time discretization with an adaptively scaled half-line quadrature rule, selects the discretization depth and the numbers of quadrature nodes from one prescribed error tolerance, and evaluates the Fourier integrand through a hierarchy whose terms are integrated separately. Reproduce that framework's multilevel pricing pipeline on the bespoke frozen configuration below and return the multilevel benchmark price it commits to.

Determine from the source, and implement exactly: (i) how the fully discrete characteristic-function exponent of a hierarchy level is assembled from the nodal Riccati values on that level's time grid; (ii) which quadrature family is used for the half-line Fourier integral and how the rule's weight is adapted to the model parameters and the maturity; (iii) how the finest discretization level is selected from the prescribed tolerance via a computable pilot indicator, including the empirical convergence-rate convention the source adopts for its tested configurations; (iv) how the integrand is decomposed across the hierarchy and how each term of the decomposition is evaluated; (v) how the quadrature points are allocated over the hierarchy under the algebraic quadrature-error model — including how the correction-level model constants are constructed from the first-correction constants, what one evaluation of a correction term costs, and the per-evaluation cost exponent implied by the source's direct implementation of the fractional Riccati solver; and (vi) how the real-valued allocation becomes integer node counts.

Frozen configuration. The asset follows the rough Heston model under the risk-neutral measure,

dS_t = S_t ( r dt + sqrt(V_t) dW^1_t ),
V_t = V0 + (1/Gamma(alpha)) * integral_0^t (t-s)^(alpha-1) * [ gamma*(theta - V_s) ds + nu*gamma*sqrt(V_s) dW^2_s ],

with d<W^1, W^2>_t = rho dt and log price X_t = ln S_t. Price the European call with payoff max(e^{X_T} - K, 0). The configuration is:

alpha = 0.66        # roughness index of the fractional kernel (= H + 1/2)
gamma = 0.50        # 1/year, mean-reversion speed
nu    = 0.12        # dimensionless vol-of-vol scale (the vol-of-vol is gamma*nu)
rho   = -0.55       # spot-variance correlation
V0    = 0.07        # 1/year, initial variance
theta = 0.35        # 1/year, long-run variance
S0    = 100.0       # spot
K     = 115.0       # strike
r     = 0.02        # 1/year, risk-free rate (zero dividends)
T     = 2.0         # years, maturity
R     = -4.0        # frozen damping parameter of the Fourier contour
M0    = 32          # time steps at level zero: dt_0 = T/32, dt_l = 2^(-l) dt_0
eps   = 2.5e-4      # total absolute error tolerance on the price
Npilot = 16         # quadrature nodes used by the pilot indicator
A0    = 12.0        # fitted algebraic prefactor, level-zero integrand (given; do not fit)
s0    = 8           # fitted algebraic smoothness index, level-zero integrand (given)
A1    = 1.4         # fitted algebraic prefactor, first level difference (given; do not fit)
s     = 7           # fitted common algebraic smoothness index of the level differences (given)

BENCHMARK CONVENTION (not paper parameters — fixed so the target is a single reproducible number): all computations in IEEE double precision; no randomness anywhere. The fractional Riccati equation is solved, per Fourier argument, with the fractional Adams predictor–corrector (PECE) scheme of Diethelm, Ford and Freed on the uniform grid t_j = j*dt_l with initial value 0: product-rectangle predictor, product-trapezoidal corrector, exactly one predictor and one corrector evaluation per time step. The damped Fourier argument is xi = u + i*R with the frozen R above entering both the characteristic function and the payoff transform; the generalized payoff transform uses the sign convention Phat(xi) = integral e^{-i xi x} P(x) dx. The total tolerance is split equally, eps_disc = eps_quad = eps/2. The pilot indicator is computed with exactly Npilot = 16 nodes of the same scaled rule used everywhere else; if the pilot level difference is exactly zero, the selected level is 1. The per-evaluation cost normalization is irrelevant (only the cost exponent enters the allocation). Report the final scalar to at least 8 significant figures and every quantity in the reporting list below to at least 6 significant figures.

Do not substitute a different time-stepping scheme for the named one. Do not replace the source's quadrature-weight adaptation with an independently chosen scale. Do not select the discretization level or the node counts by your own convergence experiments — both must come from the source's selection and allocation rules evaluated on this frozen configuration. Do not report a price computed to higher accuracy than the pipeline commits to: the requested scalar is the method's output at the frozen tolerance, not the limiting exact price. Do not hard-code any level or node count.

In your reasoning, report these quantities from your run alongside the final answer: (1) the quadrature weight scale actually used; (2) the real part of the terminal fractional-Riccati solution at xi* = 1.5 + i*R on the level-1 grid; (3) the level-1 Fourier integrand value at u = 1.5; (4) the pilot level-0 price computed with 16 nodes; (5) the pilot indicator D1; (6) the selected finest level L; (7) the level-zero node count and the per-correction-level node counts; (8) the level-zero quadrature term; (9) the total of the correction-level quadrature terms; (10) the final multilevel price.

Return the multilevel benchmark price V_(N,L) produced by this deterministic pipeline.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.

You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.

Rules:

The tags are required. Do not omit them or leave them empty.
The tagged value must be a finite decimal. Not NaN, not Inf, not a fraction string, not a vector, and not prose.
Put only that one number between the tags. No units, words, or extra lines.
Keep short (a few hundred words).
In <reasoning>, identify the source-dependent choices needed to reproduce the experiment and report only the few intermediate quantities necessary to justify the final answer.
Do not paste matrices, full state vectors, probability tables, circuit dumps, per-iteration solver paths, or other large intermediate outputs.

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

step01_laguerre_scale

Goal
----
Computes the scale parameter of the exponential quadrature weight used for the half-line Fourier integral. The scale is matched to the estimated asymptotic exponential decay rate of the damped Fourier integrand implied by the model parameters and the maturity, following the source's prescription. This step validates the closed-form scale formula only; it performs no time stepping and no quadrature. Deliberately excluded: node construction, integrand evaluation, and any error estimation.

```python
def laguerre_scale(alpha: float, gam: float, nu: float, rho: float,
                   v0: float, theta: float, big_t: float) -> float:
    """Scale of the exponential quadrature weight for the Fourier integral.

    Parameters
    ----------
    alpha : float
        Roughness index of the fractional kernel, 0.5 < alpha < 1.
    gam : float
        Mean-reversion speed of the variance process, gam > 0.
    nu : float
        Vol-of-vol scale parameter, nu > 0 (the vol-of-vol is gam * nu).
    rho : float
        Spot-variance correlation, -1 < rho < 1.
    v0 : float
        Initial variance, v0 >= 0.
    theta : float
        Long-run variance level, theta >= 0.
    big_t : float
        Maturity in years, big_t > 0.

    Returns
    -------
    float
        The exponential weight scale used by the scaled quadrature rule.

    Raises
    ------
    ValueError
        If any parameter is outside its admissible range.
    """
    return 0.0
```

### Step 2

step02_scaled_quadrature_rule

Goal
----
Applies the scaled half-line quadrature rule to the test function f(u) = exp(-a*u). The rule uses the nodes and weights of the classical rule associated with the exponential weight of scale sigma on (0, infinity), applied so that the returned value approximates the full half-line integral of f multiplied by two (the doubling that the pricing formula absorbs into the quadrature). This step validates node/weight rescaling and the doubling convention. Deliberately excluded: the Fourier integrand, any model content, and any error estimation.

```python
def scaled_laguerre_exponential(sigma: float, n_quad: int, a: float) -> float:
    """Scaled half-line quadrature applied to f(u) = exp(-a*u).

    Parameters
    ----------
    sigma : float
        Scale of the exponential quadrature weight, sigma > 0.
    n_quad : int
        Number of quadrature nodes, n_quad >= 1.
    a : float
        Decay rate of the integrand, a > 0.

    Returns
    -------
    float
        The quadrature approximation of 2 * integral_0^inf exp(-a*u) du.

    Raises
    ------
    ValueError
        If sigma <= 0, a <= 0, or n_quad < 1.
    """
    return 0.0
```

### Step 3

step03_riccati_terminal

Goal
----
Solves the fractional Riccati equation of the rough Heston characteristic function at a single Fourier argument xi = u + i*damp on the uniform grid of the discretization hierarchy (level ell has m0 * 2^ell time steps on [0, T]), using the fractional Adams predictor-corrector (PECE) scheme of Diethelm, Ford and Freed with h(0) = 0, and returns the real part of the terminal nodal value h(xi, T). This step validates the time-stepping scheme only. Deliberately excluded: the characteristic-function exponent, the payoff transform, and all quadrature.

```python
def riccati_terminal_re(u: float, damp: float, level: int, alpha: float,
                        gam: float, nu: float, rho: float, big_t: float,
                        m0: int) -> float:
    """Real part of the terminal fractional-Riccati solution h(u + i*damp, T).

    Parameters
    ----------
    u : float
        Real part of the Fourier argument, u >= 0.
    damp : float
        Imaginary part of the Fourier argument (damping), damp <= 0.
    level : int
        Discretization level, level >= 0; the grid has m0 * 2**level steps.
    alpha : float
        Roughness index, 0.5 < alpha < 1.
    gam : float
        Mean-reversion speed, gam >= 0.
    nu : float
        Vol-of-vol scale, nu > 0.
    rho : float
        Correlation, -1 < rho < 1.
    big_t : float
        Maturity, big_t > 0.
    m0 : int
        Number of time steps at level zero, m0 >= 1.

    Returns
    -------
    float
        Re h(u + i*damp, T) on the level grid.

    Raises
    ------
    ValueError
        If any parameter is outside its admissible range.
    """
    return 0.0
```

### Step 4

step04_cf_exponent

Goal
----
Assembles the fully discrete characteristic-function exponent G_level(xi) at a single Fourier argument xi = u + i*damp from the nodal fractional-Riccati values on the level grid, combining the drift contribution with the time integral of the exponent integrand evaluated on the same nodal grid, and returns Re G_level(xi). This step validates the level-defining discrete exponent. Deliberately excluded: the payoff transform, the damped integrand prefactor, and all quadrature in the Fourier variable.

```python
def cf_exponent_re(u: float, damp: float, level: int, alpha: float,
                   gam: float, nu: float, rho: float, v0: float,
                   theta: float, s0: float, r: float, big_t: float,
                   m0: int) -> float:
    """Real part of the fully discrete characteristic exponent G_level.

    Parameters
    ----------
    u : float
        Real part of the Fourier argument, u >= 0.
    damp : float
        Imaginary part of the Fourier argument (damping), damp <= 0.
    level : int
        Discretization level, level >= 0.
    alpha, gam, nu, rho : float
        Rough Heston model parameters (see step 3 for admissible ranges).
    v0 : float
        Initial variance, v0 >= 0.
    theta : float
        Long-run variance, theta >= 0.
    s0 : float
        Spot price, s0 > 0.
    r : float
        Risk-free rate.
    big_t : float
        Maturity, big_t > 0.
    m0 : int
        Level-zero step count, m0 >= 1.

    Returns
    -------
    float
        Re G_level(u + i*damp).

    Raises
    ------
    ValueError
        If any parameter is outside its admissible range.
    """
    return 0.0
```

### Step 5

step05_fourier_integrand

Goal
----
Evaluates the half-line Fourier pricing integrand g_level(u) for the European call at a single point u >= 0: the discounted real part of the product of the discrete characteristic function exp(G_level) and the generalized payoff transform of the call, both evaluated at the damped argument xi = u + i*damp. This step validates the integrand assembly (payoff transform, damping, discounting, real-part reduction). Deliberately excluded: quadrature in u, level differences, and any parameter selection.

```python
def fourier_integrand(u: float, damp: float, level: int, alpha: float,
                      gam: float, nu: float, rho: float, v0: float,
                      theta: float, s0: float, strike: float, r: float,
                      big_t: float, m0: int) -> float:
    """Half-line Fourier pricing integrand g_level(u) for the European call.

    Parameters
    ----------
    u : float
        Fourier variable, u >= 0.
    damp : float
        Damping (contour) parameter, damp < -1 for the call transform.
    level : int
        Discretization level, level >= 0.
    alpha, gam, nu, rho, v0, theta : float
        Rough Heston model parameters (admissible ranges as in steps 3-4).
    s0 : float
        Spot price, s0 > 0.
    strike : float
        Strike price, strike > 0.
    r : float
        Risk-free rate.
    big_t : float
        Maturity, big_t > 0.
    m0 : int
        Level-zero step count, m0 >= 1.

    Returns
    -------
    float
        g_level(u).

    Raises
    ------
    ValueError
        If any parameter is outside its admissible range.
    """
    return 0.0
```

### Step 6

step06_single_level_price

Goal
----
Computes the single-level price V_(N, level): the scaled half-line quadrature rule with n_quad nodes and weight scale sigma applied to the level-level Fourier integrand. This step validates the coupling of quadrature and integrand at one fixed level. Deliberately excluded: level differences, pilot indicators, level selection, and point allocation.

```python
def single_level_price(level: int, n_quad: int, sigma: float, damp: float,
                       alpha: float, gam: float, nu: float, rho: float,
                       v0: float, theta: float, s0: float, strike: float,
                       r: float, big_t: float, m0: int) -> float:
    """Single-level scaled-quadrature price V_(n_quad, level).

    Parameters
    ----------
    level : int
        Discretization level, level >= 0.
    n_quad : int
        Number of quadrature nodes, n_quad >= 1.
    sigma : float
        Scale of the exponential quadrature weight, sigma > 0.
    damp : float
        Damping parameter, damp < -1.
    alpha, gam, nu, rho, v0, theta : float
        Rough Heston model parameters (admissible ranges as in steps 3-5).
    s0, strike : float
        Spot and strike, both > 0.
    r : float
        Risk-free rate.
    big_t : float
        Maturity, big_t > 0.
    m0 : int
        Level-zero step count, m0 >= 1.

    Returns
    -------
    float
        V_(n_quad, level).

    Raises
    ------
    ValueError
        If any parameter is outside its admissible range.
    """
    return 0.0
```

### Step 7

step07_select_level

Goal
----
Selects the finest discretization level L from the prescribed discretization tolerance: computes the pilot first level difference D1 = |V_(n_pilot,1) - V_(n_pilot,0)| with the scaled rule at n_pilot nodes, extrapolates it with the supplied empirical convergence rate p, and returns the smallest admissible level (at least 1) as a float. This step validates the pilot indicator and the selection rule. Deliberately excluded: quadrature-point allocation and the telescoped price.

```python
def select_level(eps_disc: float, p: float, n_pilot: int, sigma: float,
                 damp: float, alpha: float, gam: float, nu: float,
                 rho: float, v0: float, theta: float, s0: float,
                 strike: float, r: float, big_t: float, m0: int) -> float:
    """Smallest admissible discretization level for the given tolerance.

    Parameters
    ----------
    eps_disc : float
        Discretization-error tolerance, eps_disc > 0.
    p : float
        Empirical convergence rate used by the indicator, p > 0.
    n_pilot : int
        Number of quadrature nodes for the pilot level difference,
        n_pilot >= 1.
    sigma : float
        Scale of the exponential quadrature weight, sigma > 0.
    damp : float
        Damping parameter, damp < -1.
    alpha, gam, nu, rho, v0, theta : float
        Rough Heston model parameters (admissible ranges as in steps 3-6).
    s0, strike : float
        Spot and strike, both > 0.
    r : float
        Risk-free rate.
    big_t : float
        Maturity, big_t > 0.
    m0 : int
        Level-zero step count, m0 >= 1.

    Returns
    -------
    float
        The selected level L (an integer value returned as float, >= 1).

    Raises
    ------
    ValueError
        If any parameter is outside its admissible range.
    """
    return 0.0
```

### Step 8

step08_allocate_points

Goal
----
Allocates quadrature points across the multilevel hierarchy for a given finest level big_l and quadrature tolerance eps_quad, using the algebraic quadrature-error model with the supplied fitted constants (level-zero prefactor a0 and smoothness index s0_idx; first-correction prefactor a1 and common correction index s_idx), the per-evaluation cost exponent beta, and the empirical rate p for prefactor propagation across correction levels. Returns the total number of quadrature points (level-zero plus all corrections) as a float. This step validates the closed-form work-optimal allocation and its integer rounding. Deliberately excluded: any integrand evaluation.

```python
def allocate_points_total(big_l: int, eps_quad: float, a0: float,
                          s0_idx: float, a1: float, s_idx: float,
                          beta: float, p: float, big_t: float,
                          m0: int) -> float:
    """Total quadrature points allocated across the hierarchy.

    Parameters
    ----------
    big_l : int
        Finest correction level, big_l >= 1.
    eps_quad : float
        Quadrature-error tolerance, eps_quad > 0.
    a0 : float
        Fitted algebraic prefactor of the level-zero integrand, a0 > 0.
    s0_idx : float
        Fitted smoothness index of the level-zero integrand, s0_idx >= 1.
    a1 : float
        Fitted algebraic prefactor of the first level difference, a1 > 0.
    s_idx : float
        Common fitted smoothness index of the level differences, s_idx >= 1.
    beta : float
        Cost exponent of one integrand evaluation at level ell
        (work proportional to dt_ell^(-beta)), beta > 0.
    p : float
        Empirical rate used to propagate correction prefactors, p > 0.
    big_t : float
        Maturity, big_t > 0.
    m0 : int
        Level-zero step count, m0 >= 1.

    Returns
    -------
    float
        Total node count: level-zero points plus all correction points
        (an integer value returned as float).

    Raises
    ------
    ValueError
        If any parameter is outside its admissible range.
    """
    return 0.0
```

### Step 9

step09_multilevel_price

Goal
----
Orchestrates the complete pipeline and returns the multilevel benchmark price. It derives the weight scale from the model (step 1 oracle), verifies the scaled-rule invariant (step 2 oracle), computes the pilot level prices (step 6 oracle) and the selected level (step 7 oracle, cross-checked against an internal recomputation), obtains the total allocation (step 8 oracle, cross-checked against the internally recomputed allocation vector), verifies the probe values of the Riccati solution, the discrete exponent, and the integrand (step 3-5 oracles) against its own traced evaluations, and then assembles the telescoped price: the level-zero quadrature plus the sum of correction-level quadratures of level differences, each correction evaluated at its own allocated node count with both levels of the difference sampled at the same scaled nodes. The pipeline's rate convention (the empirical rate used by the selection rule and the prefactor propagation) is fixed internally. Raises on any cross-check inconsistency. A local re-trace of nodal integrand values is used for state that earlier steps do not expose.

```python
def rough_heston_multilevel_price(eps: float, damp: float, alpha: float,
                                  gam: float, nu: float, rho: float,
                                  v0: float, theta: float, s0: float,
                                  strike: float, r: float, big_t: float,
                                  m0: int, n_pilot: int, a0: float,
                                  s0_idx: float, a1: float, s_idx: float,
                                  beta: float) -> float:
    """Multilevel scaled-quadrature benchmark price of the European call.

    Parameters
    ----------
    eps : float
        Total error tolerance, eps > 0 (split equally between
        discretization and quadrature).
    damp : float
        Damping parameter, damp < -1.
    alpha, gam, nu, rho, v0, theta : float
        Rough Heston model parameters; gam > 0 here because the weight
        scale is derived from the model.
    s0, strike : float
        Spot and strike, both > 0.
    r : float
        Risk-free rate.
    big_t : float
        Maturity, big_t > 0.
    m0 : int
        Level-zero step count, m0 >= 1.
    n_pilot : int
        Pilot node count for the level-selection indicator, n_pilot >= 1.
    a0, s0_idx, a1, s_idx : float
        Fitted algebraic quadrature-model constants (see step 8).
    beta : float
        Cost exponent of one integrand evaluation, beta > 0.

    Returns
    -------
    float
        The multilevel benchmark price V_(N, L).

    Raises
    ------
    ValueError
        If any parameter is outside its admissible range, or if an
        internal cross-check against a sub-step oracle fails.
    """
    return 0.0
```
