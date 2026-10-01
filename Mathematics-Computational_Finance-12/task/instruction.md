# Mathematics-Computational_Finance-12

## Background

Fourier-based valuation is one of the standard routes to European option prices. Rather than solving a partial integro-differential equation or simulating paths, these methods exploit the fact that for many models the characteristic function of the terminal log-price is available while the density itself is not, and recover the price as a frequency-domain integral against the transform of the payoff. The Carr-Madan FFT method, the Fourier-cosine (COS) method and the Shannon Wavelet Inverse Fourier Technique (SWIFT) are the principal members of this family. They differ in the basis used to represent the density: global cosine series in one case, and in the other dilations and translations of the cardinal sine function, whose own Fourier transforms have compact support and therefore produce structured, localized coefficient representations. Localization is attractive for densities that are asymmetric, sharply peaked, or heavy-tailed, since the magnitude of a coefficient then reflects the contribution of a specific region of the density rather than of the whole domain.

The accuracy of any such method at a finite resolution is controlled by three separate approximations: projecting the density onto finitely many basis functions, truncating the resulting series to a finite index range, and evaluating each coefficient by numerical quadrature. In practice the parameters governing these three stages have often been chosen heuristically. A line of recent work has sought to put that choice on a rigorous footing, first for the COS method and then more broadly, by relating the truncation range to the analytic structure of the characteristic function in the complex plane. The distance from the real axis to the nearest singularity of the transform governs the exponential decay rate of the coefficients, while the local nature of that singularity, whether a pole or an algebraic or logarithmic branch point, fixes the polynomial prefactor. Exponential damping of the payoff and the density is one device that makes this structure directly accessible, because it converts the coefficient computation into an evaluation of the transform along a horizontal line in the complex plane rather than on the real axis alone. Deriving sharp decay bounds in this setting, and turning them into explicit rules for selecting the damping level, the resolution and the truncation range, is an active research programme.

A largely separate literature concerns what dynamics the transform should describe. Empirical returns show both infinite-activity jump behaviour, captured by tempered-stable and CGMY-type specifications that retain a stable-like singularity at the origin while tempering large jumps exponentially, and clustering, in which large moves are followed by periods of elevated jump risk. Clustering has traditionally been modelled with Hawkes-type self-exciting point processes, in which the occurrence of a discrete event raises the intensity of future events. That construction is event-based and sits awkwardly with an infinite-activity description of returns, where the number of jumps on any interval is infinite and there are no countable events to excite anything. Bridging the two requires excitation driven by realized jump variation rather than by event counts, with the excitation bounded so that the average feedback stays finite despite the infinite activity. Models of this kind retain an affine structure, so their transforms remain computable, but only as the solution of a system of ordinary differential equations rather than in closed form, and their stability requires a subcriticality condition relating the strength of the feedback to the rate of mean reversion.

## Problem

Fourier and wavelet methods price European options directly from the characteristic function of the log-price, and their accuracy at a fixed resolution is governed by how fast the coefficients of the density and the payoff decay. A recent refinement applies an exponential damping $e^{-\alpha y}$ to the payoff and the reciprocal weight to the density, which moves the coefficient computation entirely into the frequency domain and removes the need for a physical-domain truncation parameter or for cumulants of the density. The damping level is not free: it must lie in the set of exponents $\alpha$ for which the damped payoff and the damped density are both integrable, and for a European call the upper edge of that set is the largest exponent at which the terminal stock price still has a finite moment. For a model whose transform is available only by solving a system of ordinary differential equations, that upper edge is a property of the model at the given horizon and is not read off from the parameters of the jump-size law.

Consider a risk-neutral log-price carrying a Brownian component together with an infinite-activity jump component whose jump sizes follow a normalized asymmetric tempered-stable shape, normalized so that its second moment equals one, with the positive half-line carrying a fraction $p$ of it, and whose predictable activity scale is itself excited by the asset's own realized price jumps through the bounded excitation function $g(y) = 1 - e^{-ay^{2}}$. The joint state of log-price and activity is affine, so the conditional characteristic function is exponential-affine in the state and its coefficient functions solve a generalized Riccati system; the log-price drift is fixed by the requirement that the discounted stock price be a local martingale. Price a European call on this model with the damped wavelet method above, at the fixed resolution level and translation range given below, taking the damping level to be $85\%$ of the way from the lower edge of the admissible set to its upper edge at this horizon.

Use the following configuration.

- jump-size shape: $p = 0.55$, $M = 6.5$, $G = 3.0$, stable index $\alpha_{\mathrm{TS}} = 1.1$
- excitation parameter: $a = 0.8$
- activity process: $\kappa = 3.5$, $\bar{\lambda} = 0.015$, $\eta = 2.0$, $\lambda_0 = 0.02$
- market: $r = 0.03$, $\sigma = 0.05$, $S_0 = 95$
- contract: European call, $K = 98$, $T = 0.25$
- resolution level $m = 3$; translation indices $k = 20, 21, \ldots, 101$ inclusive
- coefficient quadrature: uniform composite trapezoidal rule with $N_Q = 513$ nodes across the whole symmetric frequency interval determined by $m$, endpoints included
- accuracy: every Lévy integral to a relative accuracy of at least $10^{-9}$, using on the small-jump region the quadrature rule prescribed by the numerical protocol published with this model; the Riccati coefficient functions to a relative accuracy of at least $10^{-11}$; the upper edge of the admissible set to an absolute accuracy of at least $10^{-9}$

Your reasoning should make clear how the jump-size shape integrals, the martingale drift restriction, the upper edge of the admissible set at this horizon, the selected damping level, and the damped payoff and density coefficients each enter the result. Report the price of the European call to nine significant figures.

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

01_levy_shape_integrals

Goal
----
Compute three integrals of the normalized asymmetric tempered-stable jump-size shape that the rest of the pipeline consumes: the mean excitation of the activity process, the compensated exponential jump integral that enters the risk-neutral drift restriction, and the jump-entropy integral that enters the sufficient true-martingale condition.

The shape is parameterised by p, M, G and a_ts, where p in (0,1) is the share of the second jump moment carried by positive jumps, M and G are the positive and negative exponential tempering rates, and a_ts in (0,2) is the stable index. The shape is normalized so that its second moment is one. The excitation function is g(y) = 1 - exp(-a_exc * y**2), with a_exc > 0. In the drift-restriction integral only jumps of size below one are compensated: the integrand is exp(y) - 1 - y on the small-jump region and exp(y) - 1 on the tail, with the reflected forms on the negative half-line.

Every Levy integral in this task, in this step and in all later steps, is evaluated with the following protocol. Map the negative half-line onto the positive one by reflection. Truncate the positive half-line at y_cut = 10 and split it into the small-jump region (0, 1) and the tail (1, y_cut). On the small-jump region use 128-node Gauss-Jacobi quadrature with the endpoint weight induced by the near-origin behaviour of the compensated integrands. On the tail use 128-node Gauss-Legendre quadrature.

Return the three values in the order: mean excitation, drift-restriction integral, martingale-condition integral.

Raises ValueError if p is not in (0,1); if M <= 0 or G <= 0; if a_ts is not in (0,2); if a_exc <= 0; or if M <= 1, the positive-tail condition under which the drift-restriction integral is finite.

```python
def levy_shape_integrals(p: float, M: float, G: float, a_ts: float,
                         a_exc: float) -> 'np.ndarray':
    '''Compute the three shape integrals of the jump-size law.

    Parameters
    ----------
    p : float
        Share of the second jump moment carried by positive jumps, in (0,1).
    M : float
        Positive-jump exponential tempering rate, greater than 1.
    G : float
        Negative-jump exponential tempering rate, positive.
    a_ts : float
        Stable index of the jump-size shape, in (0,2).
    a_exc : float
        Excitation parameter, positive.

    Returns
    -------
    result : np.ndarray
        Shape (3,): mean excitation, drift-restriction integral,
        martingale-condition integral, in that order. The drift-restriction
        integral compensates only jumps of size below one, so its integrand is
        exp(y) - 1 - y on the small-jump region and exp(y) - 1 on the tail.

    Raises
    ------
    ValueError
        If p is not in (0,1); if M <= 0 or G <= 0; if a_ts is not in (0,2);
        if a_exc <= 0; or if M <= 1, the positive-tail condition under which
        the drift-restriction integral is finite.
    '''
    return result  # placeholder
```

### Step 2

02_activity_riccati_transform

Goal
----
Evaluate the conditional characteristic function of the terminal log-price of the endogenous-activity jump model, on a horizontal line in the complex frequency plane.

The model couples a log-price carrying a Brownian component and an infinite-activity jump component whose predictable activity scale is itself driven by the asset's own realized price jumps, through a bounded excitation of the realized jump variation. The joint state of log-price and activity is affine, so its conditional Fourier-Laplace transform is exponential-affine in the state, with coefficient functions solving a system of ordinary differential equations obtained from the infinitesimal generator of the joint process. The log-price drift is not free: it is fixed by the requirement that the discounted stock price be a local martingale.

The argument u_line is a strictly increasing array of n non-negative real frequencies. Write the transform in the convention in which it is the expectation of the exponential of i times the argument times the terminal log-price. In that convention this step evaluates it at the argument whose real part is the negative of the entry of u_line and whose imaginary part is the negative of the damping level alpha. Both signs are part of the contract: they are the ones under which these values feed the density coefficients of a later step without further conjugation or reflection.

Evaluate every Levy integral with the split protocol of the shape-integral step: reflection onto the positive half-line, truncation at y_cut = 10, a 128-node Gauss-Jacobi rule on (0, 1) carrying the endpoint weight induced by the near-origin behaviour of the compensated integrands, and a 128-node Gauss-Legendre rule on the tail. In this step's own integrand, as in the drift-restriction integral supplied to it, only jumps of size below one are compensated. The excitation function is g(y) = 1 - exp(-a_exc * y**2), with a_exc > 0.

Integrate the coefficient system over the interval [0, T] to a relative accuracy of at least 1e-11. The integration scheme is not constrained.

Return the transform values packed as a two-row real array: row 0 the real parts, row 1 the imaginary parts, in the order of u_line.

Raises ValueError if u_line is not a one-dimensional array of non-negative, strictly increasing values; if T <= 0; if lam0 < 0; if sigma < 0; if S0 <= 0; or if the coefficient system fails to remain bounded on [0, T], which indicates that the requested damping level lies outside the admissible strip.

```python
def activity_riccati_transform(u_line: 'np.ndarray', alpha: float, T: float,
                               chiJ: float, p: float, M: float, G: float,
                               a_ts: float, a_exc: float, kappa: float,
                               lam_bar: float, eta: float, lam0: float,
                               r: float, sigma: float,
                               S0: float) -> 'np.ndarray':
    '''Evaluate the conditional characteristic function on a damped line.

    Parameters
    ----------
    u_line : np.ndarray
        Shape (n,), strictly increasing non-negative real frequencies.
    alpha : float
        Damping level fixing the imaginary part of the transform argument.
    T : float
        Horizon, positive.
    chiJ : float
        Drift-restriction integral from the shape-integral step.
    p, M, G, a_ts, a_exc : float
        Jump-size shape and excitation parameters.
    kappa, lam_bar, eta, lam0 : float
        Activity mean-reversion rate, long-run level, feedback strength and
        initial level.
    r, sigma, S0 : float
        Risk-free rate, Brownian volatility and initial spot price.

    Returns
    -------
    result : np.ndarray
        Shape (2, n): row 0 real parts, row 1 imaginary parts.

    Raises
    ------
    ValueError
        If u_line is not a non-empty one-dimensional array of non-negative,
        strictly increasing values; if T <= 0; if lam0 < 0; if sigma < 0; if
        S0 <= 0; or if the coefficient system fails to remain bounded on
        [0, T], which indicates that the requested damping level lies outside
        the admissible strip.
    '''
    return result  # placeholder
```

### Step 3

03_moment_growth_driver

Goal
----
Evaluate the scalar autonomous driver that governs the activity coefficient function when the transform of the previous step is evaluated at a purely imaginary frequency argument.



At such an argument the coefficient system collapses to a single real autonomous ordinary differential equation in the activity coefficient alone. Its driver is the generator-induced expression already used in the full system, specialised to a real exponent, and its sign structure decides whether the exponential moment of the terminal stock price at that exponent stays finite over a given horizon.



Evaluate every Levy integral with the split protocol of the shape-integral step: reflection onto the positive half-line, truncation at y_cut = 20, a 128-node Gauss-Jacobi rule on (0, 1) carrying the endpoint weight induced by the near-origin behaviour of the compensated integrands, and a 224-node Gauss-Legendre rule on the tail. This step truncates at 20 rather than at the 10 used by the shape-integral step, because the feedback factor exp(eta * x * g(y)) multiplies the dropped tail by the same amount as the part that is kept, so y_cut = 10 does not reach the relative accuracy of 1e-9 that this step requires. In this step's own integrand, as in the drift-restriction integral supplied to it, only jumps of size below one are compensated. The excitation function is g(y) = 1 - exp(-a_exc * y**2), with a_exc > 0.



Return the driver evaluated at each point of psi_grid, in the order supplied.



Raises ValueError if psi_grid is not a non-empty one-dimensional array; if any entry of psi_grid is negative; if alpha <= 0; if kappa <= 0; or if eta < 0.

```python
def moment_growth_driver(psi_grid: 'np.ndarray', alpha: float, chiJ: float,
                         p: float, M: float, G: float, a_ts: float,
                         a_exc: float, kappa: float,
                         eta: float) -> 'np.ndarray':
    '''Evaluate the scalar autonomous driver of the activity coefficient.

    Parameters
    ----------
    psi_grid : np.ndarray
        Shape (n,), non-negative activity-coefficient values.
    alpha : float
        Real exponent at which the driver is specialised, positive.
    chiJ : float
        Drift-restriction integral from the shape-integral step.
    p, M, G, a_ts, a_exc : float
        Jump-size shape and excitation parameters.
    kappa : float
        Activity mean-reversion rate, positive.
    eta : float
        Feedback strength, non-negative.

    Returns
    -------
    result : np.ndarray
        Shape (n,), the driver at each grid point, in the order supplied.

    Raises
    ------
    ValueError
        If psi_grid is not a non-empty one-dimensional array; if any entry of
        psi_grid is negative; if alpha <= 0; if kappa <= 0; or if eta < 0.
    '''
    return result  # placeholder
```

### Step 4

04_transform_strip_boundary

Goal
----
Determine the largest exponent at which the terminal stock price of this model has a finite exponential moment at the horizon T.

This boundary is what makes a damping level admissible for the pricing method of the following steps. It is a property of the model at a finite horizon, and it is at most the positive tempering rate M, which is the boundary of the jump-size shape's own exponential moment. It is also at least the corresponding boundary for an unbounded horizon.

Evaluate every Levy integral with the split protocol of the shape-integral step: reflection onto the positive half-line, truncation at y_cut = 20, a 128-node Gauss-Jacobi rule on (0, 1) carrying the endpoint weight induced by the near-origin behaviour of the compensated integrands, and a 224-node Gauss-Legendre rule on the tail. This step truncates at 20 rather than at the 10 used by the shape-integral step, because the feedback factor exp(eta * x * g(y)) multiplies the dropped tail by the same amount as the part that is kept, so y_cut = 10 does not reach the absolute accuracy of 1e-9 that this step requires. In this step's own integrand, as in the drift-restriction integral supplied to it, only jumps of size below one are compensated. The excitation function is g(y) = 1 - exp(-a_exc * y**2), with a_exc > 0.

Determine the boundary to an absolute accuracy of at least 1e-9 and return it. Search only over exponents strictly between 1 and M. If the terminal price has a finite exponential moment at every exponent below M, return M - 1e-6, that is M minus the absolute amount 10^-6, which is the largest exponent the search considers.

Raises ValueError if T <= 0; if M <= 1; if kappa <= 0; or if eta < 0.

```python
def transform_strip_boundary(T: float, chiJ: float, p: float, M: float,
                             G: float, a_ts: float, a_exc: float,
                             kappa: float, eta: float) -> float:
    '''Largest exponent with a finite terminal exponential moment at T.

    Parameters
    ----------
    T : float
        Horizon, positive.
    chiJ : float
        Drift-restriction integral from the shape-integral step.
    p, M, G, a_ts, a_exc : float
        Jump-size shape and excitation parameters.
    kappa : float
        Activity mean-reversion rate, positive.
    eta : float
        Feedback strength, non-negative.

    Returns
    -------
    result : float
        The finite-horizon exponent boundary.

    Raises
    ------
    ValueError
        If T <= 0; if M <= 1; if kappa <= 0; or if eta < 0.
    '''
    return result  # placeholder
```

### Step 5

05_damping_parameter

Goal
----
Fix the damping level for the pricing method and report the value of the parameter-selection objective there.

The damping level is placed at the fraction frac of the way from the lower edge of the admissible set, which for a European call payoff is the exponent one, to the finite-horizon boundary supplied as alpha_expl. The objective reported alongside it is the objective of the damped wavelet method's own parameter-selection algorithm, evaluated at that damping level; it is diagnostic only and does not influence the returned damping level.

That objective is the product of two masses. One is the mass of the damped payoff, that is the integral of the damped call payoff over the real line. The other is the mass of the damped density, that is the exponential moment of the terminal stock price at the damping level, which this step obtains by evaluating the transform at zero frequency using the same coefficient system, Levy quadrature protocol and accuracy requirement as the transform step. In this step's own integrand, as in the drift-restriction integral supplied to it, only jumps of size below one are compensated. The excitation function is g(y) = 1 - exp(-a_exc * y**2), with a_exc > 0.

Return the two values in the order: damping level, objective value.

Raises ValueError if frac is not in (0,1); if alpha_expl <= 1; if K <= 0; or if the resulting damping level lies outside the admissible strip.

```python
def damping_parameter(alpha_expl: float, frac: float, K: float, T: float,
                      chiJ: float, p: float, M: float, G: float, a_ts: float,
                      a_exc: float, kappa: float, lam_bar: float, eta: float,
                      lam0: float, r: float, sigma: float,
                      S0: float) -> 'np.ndarray':
    '''Fix the damping level and report the selection objective there.

    Parameters
    ----------
    alpha_expl : float
        Finite-horizon exponent boundary, greater than 1.
    frac : float
        Fraction of the way from the lower edge to the boundary, in (0,1).
    K : float
        Strike, positive.
    T : float
        Horizon, positive.
    chiJ : float
        Drift-restriction integral from the shape-integral step.
    p, M, G, a_ts, a_exc, kappa, lam_bar, eta, lam0, r, sigma, S0 : float
        Model and market parameters.

    Returns
    -------
    result : np.ndarray
        Shape (2,): damping level, objective value, in that order.

    Raises
    ------
    ValueError
        If frac is not in (0,1); if alpha_expl <= 1; if K <= 0; or if the
        resulting damping level lies outside the admissible strip.
    '''
    return result  # placeholder
```

### Step 6

06_payoff_coefficients

Goal
----
Compute the damped payoff coefficients of a European call struck at K, at resolution level m, for every integer translation index from lam_lo to lam_hi inclusive.

The density and the payoff are represented in a basis of Shannon scaling functions at level m. Because the Fourier transform of that basis function has compact support, each coefficient is a finite-interval frequency integral of the transform of the damped payoff against a single complex exponential, with the interval determined by the resolution level and symmetric about zero. Compute each coefficient with the uniform composite trapezoidal rule, where NQ is the total number of nodes placed across that whole symmetric interval, endpoints included, so that the node spacing is the interval width divided by NQ minus one.

The damped payoff is the call payoff multiplied by a decaying exponential in the log-price at rate alpha. Its Fourier transform is available in closed form, so no numerical integration in the log-price variable is required.

Return the coefficients in increasing order of translation index.

Raises ValueError if lam_lo > lam_hi; if NQ is not an odd integer of at least 3; if alpha <= 1, the condition under which the damped call payoff is integrable; or if K <= 0.

```python
def payoff_coefficients(alpha: float, m: int, lam_lo: int, lam_hi: int,
                        NQ: int, K: float) -> 'np.ndarray':
    '''Damped payoff coefficients of a European call.

    Parameters
    ----------
    alpha : float
        Damping level, greater than 1.
    m : int
        Resolution level.
    lam_lo, lam_hi : int
        Inclusive bounds of the translation range, lam_lo <= lam_hi.
    NQ : int
        Total trapezoidal nodes across the symmetric frequency interval,
        endpoints included; odd and at least 3.
    K : float
        Strike, positive.

    Returns
    -------
    result : np.ndarray
        Shape (lam_hi - lam_lo + 1,), in increasing translation index.

    Raises
    ------
    ValueError
        If lam_lo > lam_hi; if NQ is not an odd integer of at least 3; if
        alpha <= 1, the condition under which the damped call payoff is
        integrable; or if K <= 0.
    '''
    return result  # placeholder
```

### Step 7

07_density_coefficients

Goal
----
Compute the damped density coefficients at resolution level m for every integer translation index from lam_lo to lam_hi inclusive, from the packed transform values produced earlier.

The construction mirrors the payoff coefficients of the previous step: the same symmetric finite frequency interval determined by the resolution level, the same uniform composite trapezoidal rule with NQ counted the same way, and the same complex exponential in the translation index.

The transform values supplied must have been evaluated on the non-negative half of exactly that grid, in increasing order, so their number is one more than half of NQ minus one. The values on the negative half are not supplied and must be reconstructed: the damped density is a real function of the log-price, so its transform at a frequency and at the negative of that frequency are complex conjugates of one another. Reconstruct the negative half by that conjugation rather than by re-evaluating the transform.

Return the coefficients in increasing order of translation index.

Raises ValueError if transform_packed does not have exactly two rows; if NQ is not an odd integer of at least 3; if the number of columns of transform_packed is not one more than half of NQ minus one; or if lam_lo > lam_hi.

```python
def density_coefficients(transform_packed: 'np.ndarray', alpha: float, m: int,
                         lam_lo: int, lam_hi: int, NQ: int) -> 'np.ndarray':
    '''Damped density coefficients from packed transform values.

    Parameters
    ----------
    transform_packed : np.ndarray
        Shape (2, (NQ + 1) // 2): real and imaginary parts of the transform on
        the non-negative half of the frequency grid, in increasing order.
    alpha : float
        Damping level used to produce transform_packed.
    m : int
        Resolution level.
    lam_lo, lam_hi : int
        Inclusive bounds of the translation range, lam_lo <= lam_hi.
    NQ : int
        Total trapezoidal nodes across the symmetric frequency interval,
        endpoints included; odd and at least 3.

    Returns
    -------
    result : np.ndarray
        Shape (lam_hi - lam_lo + 1,), in increasing translation index.

    Raises
    ------
    ValueError
        If transform_packed does not have exactly two rows; if NQ is not an
        odd integer of at least 3; if the number of columns of
        transform_packed is not one more than half of NQ minus one; or if
        lam_lo > lam_hi.
    '''
    return result  # placeholder
```

### Step 8

08_damped_swift_price

Goal
----
Assemble the full pipeline and return the price of the European call at the configuration supplied in cfg.



Compute the shape integrals; verify that the model's mean-subcriticality condition holds, so that the activity process has a finite stationary mean; determine the finite-horizon exponent boundary; fix the damping level and its objective; cross-check the scalar driver over the range [0, 20] at the selected damping level against that boundary, requiring the driver to change sign when the boundary equals the positive tempering rate; evaluate the transform on the non-negative half of the frequency grid used by the coefficient rule; form the payoff and density coefficients over the translation range; and combine them into the discounted price.



The boundary is treated as equal to the positive tempering rate when the two differ by less than one part in a thousand.



Return the price as a native float.



Raises ValueError if the mean-subcriticality condition fails; if the driver is strictly positive over [0, 20] at the selected damping level while the boundary equals the positive tempering rate; or if any stage of the chain rejects the configuration, for example a positive tempering rate that does not exceed one, or an even node count.

```python
def damped_swift_price(cfg: dict) -> float:
    '''Assemble the pipeline and return the European call price.

    Parameters
    ----------
    cfg : dict
        Configuration with keys p, M, G, a_ts, kappa, lam_bar, eta, a_exc,
        lam0, r, sigma, S0, K, T, m, NQ, lam_lo, lam_hi, frac.

    Returns
    -------
    result : float
        The discounted price of the European call.

    Raises
    ------
    ValueError
        If the mean-subcriticality condition fails; if the driver is
        strictly positive over [0, 20] at the selected damping level while
        the boundary equals the positive tempering rate; or if any stage of
        the chain rejects the configuration, for example a positive
        tempering rate that does not exceed one, or an even node count.
    '''
    return result  # placeholder
```
