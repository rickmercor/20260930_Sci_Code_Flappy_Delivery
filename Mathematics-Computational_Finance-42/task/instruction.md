# Mathematics-Computational_Finance-42

## Background

A credit default swap is an insurance contract on a company failing to pay. The buyer pays a fixed rate each quarter until either the contract matures or the company defaults; if default comes first the seller covers the loss. Pricing one means deciding what that fixed rate should be, and that reduces to a single question about probability: how likely is the company to still be alive at each future payment date. Everything else in the contract is bookkeeping on a discount curve.

The standard way to model the timing of default is to give the company a hazard rate, an instantaneous propensity to fail that itself moves randomly through time. The convenient choice makes that rate the exponential of a mean-reverting Gaussian process, because an exponential is always positive and mean reversion keeps it from wandering off. The inconvenience is that this choice destroys every closed form. Survival probability is the average of an exponential of an integral of an exponential of a Gaussian, and there is no formula for it. The usual remedies are a fine grid in space and time, or simulating many paths, and both are slow enough to hurt when a desk has to recalibrate across many names and many maturities each day.

The method this task implements comes from a different tradition. Instead of discretising, it borrows a variational idea from statistical physics: sort every path by its own time average, and for each possible value of that average replace the awkward exponential term by the best quadratic substitute, chosen so that the substitute and the original agree in value, slope and curvature once both are blurred over the spread of paths that share that average. A quadratic problem has an exact solution, so what remains is a one-dimensional integral over the possible averages, which a few dozen quadrature points resolve.

The subtlety, and the reason the construction is not just a saddle point, is that fixing the time average changes the fluctuations left over. The paths that share an average are more tightly bunched than free paths, and the blurring width has to reflect that, which in turn changes the quadratic substitute, which changes the width again. Solving that small self-referential loop at each average point is the whole computational content of the method, and it is what buys the accuracy: the result tracks a fine PDE solve to a few parts in ten thousand at a small fraction of the cost.

## Problem

Pricing credit derivatives under a lognormal default intensity is awkward because the intensity has no closed-form Arrow-Debreu density: the survival expectation is a Feynman-Kac functional of a mean-reverting process whose exponential appears in the exponent, so neither a moment-generating function nor an affine transform is available and practitioners fall back on a lattice, a PDE or Monte Carlo. A recent method builds a semi-analytical approximation instead, classifying paths by their own time average, replacing the true potential at each average point by the quadratic one that best matches it in a variational sense, and integrating the resulting Gaussian object over the average point with a small fixed quadrature. Your task is to run that construction on one fully specified single-name credit default swap and report the par spread it returns.

Here is the exact setup to use:
- Intensity model: $$dX_t = k(\theta - X_t)\,dt + \sigma\,dW_t, \qquad h_t = \exp(X_t),$$ with mean reversion $k = 0.35$, diffusion $\sigma = 0.62$, a mean-reversion level constant in time at $\theta = \ln(0.025)$, and initial state $x_0 = \ln(0.035)$. The intensity weight in the generalised Arrow-Debreu density is $\lambda = 1$.
- Quantities required: the survival expectation $Q(u) = \mathbb{E}\left[e^{-\int_0^u h_s\,ds}\right]$ and the intensity-weighted expectation $G(u) = \mathbb{E}\left[h_u\,e^{-\int_0^u h_s\,ds}\right]$, both obtained from the source paper's representation, with $Q(0) = 1$ and $G(0) = e^{x_0}$ taken exactly.
- Contract: a five-year single-name CDS with quarterly premium dates $T_i = i\,\Delta$, $\Delta = 0.25$, $i = 1,\dots,20$, recovery $R = 0.4$, on a flat continuously compounded discount curve $D(u) = e^{-ru}$ with $r = 0.025$.
- Legs: premium annuity $$\mathcal{A} = \sum_{i=1}^{20}\Delta\,D(T_i)\,Q(T_i),$$ protection leg $$\mathcal{P} = (1-R)\int_0^{5} D(u)\,G(u)\,du$$ evaluated by the composite trapezoidal rule on the twenty-one dates $T_0,\dots,T_{20}$, and par spread $s = 10^4\,\mathcal{P}/\mathcal{A}$ in basis points.
- Average-point quadrature: at each maturity $T$ use the same $81$-point Gauss-Legendre rule, affine-mapped onto $[m - 9\varsigma,\ m + 9\varsigma]$ with $$m = \theta + (x_0-\theta)\frac{1-e^{-kT}}{kT}, \qquad \varsigma = \sigma\sqrt{T/3},$$ the weights carrying the mapping Jacobian.
- Variational solve: at every node solve the paper's scalar self-consistency condition by fixed-point iteration started from the mean-reversion speed, to a relative tolerance of $10^{-15}$.

Run the pipeline and report the par spread $s$. The answer is graded to an absolute tolerance of $10^{-6}$. In your reasoning, report the following, every one of them taken at the average point $\bar{x} = x_0$ and the five-year horizon unless stated otherwise: the trial frequency and the fluctuation width; the endpoint displacement; the action correction; the quadratic, the linear and the constant coefficient of the analytic integration over the terminal state; the natural logarithm of the average-point weight that integration produces at that node, meaning the weight the quadrature over the average point sums, not the normalisation of the constrained propagator that precedes it; $Q(5)$ and $G(5)$; the premium annuity and the protection leg; the value the average-point integral of that weight takes when the intensity weight is set to $\lambda = 0$ instead of $1$; and the spread the same schedule would return if the terminal payoff were evaluated at the centre of the residual Gaussian instead of averaged over it.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.

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

Harmonic fluctuation width

Goal
----
Return the width of the Gaussian fluctuations that a path makes about its own time average, for a trial harmonic action of a given frequency over a given horizon. This single scalar is what the smearing operator of the variational scheme integrates against, so every quantity built later in the pipeline inherits it. It depends only on the frequency, the horizon and the diffusion coefficient, never on the state or on the default intensity.

```python
import numpy as np


def harmonic_fluctuation_width(omega, T, sigma):
    """Width of the constrained Gaussian fluctuation of the trial action.

    For a trial harmonic action of frequency omega on [0, T] with diffusion
    coefficient sigma, paths are classified by their time average and the
    residual fluctuation about that average is Gaussian.  This returns its
    variance, written through the half-product f = omega*T/2.

    Args:
        omega (float): trial harmonic frequency, strictly positive.
        T (float): horizon, strictly positive.
        sigma (float): diffusion coefficient of the state process, strictly
            positive.

    Expected return:
        float: the fluctuation width, strictly positive.  As omega*T tends to
        zero the value tends to sigma**2 * T / 12.

    Raises:
        ValueError: if omega <= 0, if T <= 0, or if sigma <= 0.
    """
    return 0.0
```

### Step 2

Variational frequency

Goal
----
Solve the variational condition that fixes the trial frequency at one average point and return that frequency. The condition matches the curvature of the trial quadratic potential to the smeared curvature of the true one, which makes the frequency and the fluctuation width mutually dependent and the problem a scalar fixed point rather than a formula.

```python
import numpy as np


def variational_frequency(xbar, T, k, sigma, lam):
    """Trial frequency at one average point.

    The frequency solves the variational curvature-matching condition of the
    source method for a lognormal intensity h = exp(x) under

        dX_t = k*(theta_t - X_t)*dt + sigma*dW_t,

    weighted by exp(-lam * integral of h).  The condition couples the
    frequency to the fluctuation width of the previous step, so it is solved
    as a scalar fixed point started from omega = k and iterated to a relative
    tolerance of 1e-15.

    Args:
        xbar (float): average point at which the condition is imposed.
        T (float): horizon, strictly positive.
        k (float): mean-reversion speed, strictly positive.
        sigma (float): diffusion coefficient, strictly positive.
        lam (float): intensity weight, non-negative.

    Expected return:
        float: the trial frequency omega.  With lam = 0 it is exactly k.  The
        fluctuation width that belongs to it is obtained by passing omega back
        through the previous step.

    Raises:
        ValueError: if T <= 0, if k <= 0, if sigma <= 0, or if lam < 0.
    """
    return 0.0
```

### Step 3

Linear coefficient integrals

Goal
----
Build the linear coefficient of the trial potential at one average point and return it together with the three weighted time integrals of it that every later stage consumes. The linear coefficient comes from matching the smeared first derivative of the true potential; the integrals weight it against the two hyperbolic solutions of the trial equation of motion, which is how the endpoints of the interval feel a force applied in its interior.

```python
import numpy as np


def linear_coefficient_integrals(xbar, omega, T, k, sigma, theta):
    """Linear trial coefficient and its reduced weighted time integrals.

    The linear coefficient of the trial quadratic potential at the average
    point xbar follows from matching the smeared first derivative of the true
    potential of the lognormal intensity model.  With a mean-reversion level
    constant in time it is itself constant in time.  The three reduced
    combinations returned are the two endpoint response integrals and the
    nested double integral, each already divided by the hyperbolic sine of
    the full frequency-horizon product so that nothing overflows.

    Args:
        xbar (float): average point.
        omega (float): trial frequency at that average point, strictly
            positive.
        T (float): horizon, strictly positive.
        k (float): mean-reversion speed.
        sigma (float): diffusion coefficient, strictly positive.
        theta (float): mean-reversion level, constant in time.

    Expected return:
        np.ndarray of shape (5,) packed as [gamma, gamma_hat, r_sum, r_zero,
        r_cross]: the linear coefficient, its plain time integral, and the
        sum, start-weighted and nested reduced combinations.

    Raises:
        ValueError: if omega <= 0, if T <= 0, or if sigma <= 0.
    """
    return np.zeros(5)
```

### Step 4

Displacement and correction

Goal
----
Turn the reduced response integrals into the two displacements and the one action correction that the linear part of the trial potential generates. A constant force applied to a harmonic path does three things: it shifts the whole path, it shifts the point about which the constrained fluctuation is centred, and it lowers the action by an amount quadratic in the force. The three outputs here are exactly those three effects.

```python
import numpy as np


def displacement_and_correction(reduced, omega, T, sigma):
    """Endpoint displacement, action correction and centre displacement.

    Consumes the five reduced quantities of the previous step and returns the
    three scalars the linear part of the trial potential contributes: the
    displacement that enters the propagator between the endpoints, the
    additive correction to the trial action, and the displacement of the
    centre of the constrained diagonal density away from the average point.

    Args:
        reduced: (5,) array [gamma, gamma_hat, r_sum, r_zero, r_cross] as
            produced by the previous step.
        omega (float): trial frequency, strictly positive.
        T (float): horizon, strictly positive.
        sigma (float): diffusion coefficient.

    Expected return:
        np.ndarray of shape (3,) packed as [delta, correction, delta_gamma].
        The third entry vanishes when the linear coefficient is constant in
        time.

    Raises:
        ValueError: if reduced does not hold exactly five entries, if
        omega <= 0, or if T <= 0.
    """
    return np.zeros(3)
```

### Step 5

Log trial normalisation

Goal
----
Assemble the logarithm of the normalisation of the constrained trial propagator at one average point. Three ingredients enter: the Gaussian prefactor of the constrained measure, the ratio of the half-product to its hyperbolic sine, and the exponential of the action correction less the time integral of the constant part of the trial potential. The logarithm is returned rather than the value because the exponent runs over many tens across the integration window and the value itself underflows long before the contribution to the integral becomes negligible.

```python
import numpy as np


def log_trial_normalisation(xbar, omega, alpha, corr, T, k, sigma, lam, theta):
    """Logarithm of the constrained trial propagator normalisation.

    Combines the Gaussian prefactor of the constrained measure, the harmonic
    fluctuation determinant and the exponential of the action correction less
    the time integral of the constant part of the trial potential.  The
    constant part is fixed by matching the smeared value of the true
    potential of the lognormal intensity model, so it carries the intensity in
    its smeared form.

    Args:
        xbar (float): average point.
        omega (float): trial frequency, strictly positive.
        alpha (float): fluctuation width, strictly positive.
        corr (float): additive action correction from the previous step.
        T (float): horizon, strictly positive.
        k (float): mean-reversion speed.
        sigma (float): diffusion coefficient, strictly positive.
        lam (float): intensity weight.
        theta (float): mean-reversion level, constant in time.

    Expected return:
        float: the natural logarithm of the normalisation.

    Raises:
        ValueError: if omega <= 0, if alpha <= 0, if T <= 0, or if sigma <= 0.
    """
    return 0.0
```

### Step 6

Endpoint integration coefficients

Goal
----
Carry out the integral over the terminal state analytically and return the quadratic, linear and constant coefficients it is performed with, the point the remaining payoff is evaluated at, and the logarithm of the resulting weight for this average point. The terminal state appears quadratically in three places at once: in the constrained propagator between the endpoints, in the propagator's dependence on the average, and in the drift-removal factor that turned the mean-reverting process into a driftless one. Collecting all three is what makes the integral Gaussian and therefore closed-form.

```python
import numpy as np


def endpoint_coefficients(xbar, x0, delta, reduced, alpha, omega, log_N, T, k,
                          sigma, theta):
    """Coefficients of the analytic terminal-state integration.

    Collects every dependence on the terminal state - from the constrained
    propagator, from its dependence on the average point, and from the
    drift-removal factor that maps the mean-reverting process onto a driftless
    one - into a quadratic, a linear and a constant coefficient, then returns
    them together with the point at which the remaining payoff is evaluated
    and the logarithm of the weight attached to this average point.

    Args:
        xbar (float): average point.
        x0 (float): initial state.
        delta (float): endpoint displacement from the previous step.
        reduced: (5,) array of reduced quantities from the third step.
        alpha (float): fluctuation width, strictly positive.
        omega (float): trial frequency, strictly positive.
        log_N (float): logarithm of the trial normalisation.
        T (float): horizon.
        k (float): mean-reversion speed.
        sigma (float): diffusion coefficient, strictly positive.
        theta (float): mean-reversion level, constant in time.

    Expected return:
        np.ndarray of shape (5,) packed as [A, B, C, centre, log_weight],
        where centre = x0 - B/(2*A).

    Raises:
        ValueError: if reduced does not hold exactly five entries, if
        alpha <= 0, if omega <= 0, or if sigma <= 0.
    """
    return np.zeros(5)
```

### Step 7

Average-point quadrature grid

Goal
----
Build the fixed quadrature grid over the average point on which the whole estimator is evaluated. The grid is a Gauss-Legendre rule mapped onto a window centred on the mean of the time average of the mean-reverting process and scaled by the standard deviation that a driftless path of the same diffusion would have for its own time average. Both the centre and the scale are closed-form, so the grid is a deterministic function of the horizon and the model and costs nothing to rebuild at each maturity.

```python
import numpy as np


def average_point_grid(T, x0, k, sigma, theta, nq, span):
    """Gauss-Legendre grid over the average point.

    The window is centred on the mean of the time average of the
    mean-reverting process over [0, T] started at x0, and its half-width is
    span times the standard deviation that the time average of a driftless
    path with the same diffusion coefficient would have.  A Gauss-Legendre
    rule of nq points is affine-mapped onto that window.

    Args:
        T (float): horizon, strictly positive.
        x0 (float): initial state.
        k (float): mean-reversion speed, strictly positive.
        sigma (float): diffusion coefficient, strictly positive.
        theta (float): mean-reversion level, constant in time.
        nq (int): number of quadrature nodes, at least 1.
        span (float): half-width of the window in units of the scale,
            strictly positive.

    Expected return:
        np.ndarray of shape (2*nq,) packed as [u_1..u_nq, w_1..w_nq] with
        nodes in increasing order and weights carrying the mapping Jacobian.

    Raises:
        ValueError: if T <= 0, if k <= 0, if sigma <= 0, if nq < 1, or if
        span <= 0.
    """
    return np.zeros(2 * int(nq))
```

### Step 8

Assemble expectation

Goal
----
Apply the smearing operator to the payoff and sum the per-node weights into the two expectations the pricing needs: the one for a payoff identically equal to one, and the one for a payoff equal to the intensity itself. Both are produced from the same per-node data, because the terminal integration has already been carried out and all that remains is a Gaussian smearing of the payoff about the point that integration left behind.

```python
import numpy as np


def assemble_expectation(log_weight, centre, A, quad_weight):
    """Smear the payoff and sum the per-node contributions.

    Each node of the average-point grid carries a log weight, the point at
    which the payoff is evaluated after the terminal state has been integrated
    out, and the quadratic coefficient of that integration.  The payoff is
    smeared over the residual Gaussian uncertainty left by the terminal
    integration and the results are summed against the quadrature weights.
    Two payoffs are evaluated: the constant payoff one, and the exponential
    payoff exp(x).

    Args:
        log_weight: (n,) array of per-node log weights.
        centre: (n,) array of payoff evaluation points.
        A: (n,) array of quadratic coefficients, all strictly positive.
        quad_weight: (n,) array of quadrature weights.

    Expected return:
        np.ndarray of shape (2,) packed as [unit payoff expectation,
        exponential payoff expectation].

    Raises:
        ValueError: if the four inputs do not all have the same length, if
        they are empty, or if any entry of A is not positive.
    """
    return np.zeros(2)
```

### Step 9

Par CDS spread

Goal
----
Chain the eight earlier steps over the whole premium schedule and return the par spread the semi-analytical method produces. For every premium date the grid over the average point is rebuilt, the variational frequency and width are solved at each node, the reduced integrals, displacements, correction, normalisation and endpoint coefficients are assembled, and the two expectations are summed; the survival values build the premium annuity and the intensity-weighted values build the protection leg. This is the orchestrator step: it chains the eight earlier reference implementations rather than reproducing any of them inline.

```python
import numpy as np


def par_cds_spread(model, contract, numerics):
    """Chain the eight earlier steps and return the par credit spread.

    For each premium date T_i = i*dtau, i = 1..nper, builds the average-point
    grid, solves the variational frequency at every node and recovers the
    fluctuation width that belongs to it, assembles
    the reduced integrals, the displacements, the action correction, the
    normalisation and the endpoint coefficients, and sums the two smeared
    expectations.  The survival values Q(T_i) form the premium annuity

        annuity = sum_i dtau * D(T_i) * Q(T_i),

    and the intensity-weighted values G(T_i) form the protection leg by the
    composite trapezoidal rule on the same dates, with G(0) = exp(x0) and
    Q(0) = 1,

        protection = (1 - rec) * trapezoid over [0, T_nper] of D(u) * G(u),

    where D(u) = exp(-rate*u).  The spread is 1e4 * protection / annuity.

    Args:
        model: (4,) array [k, sigma, theta, x0].
        contract: (5,) array [dtau, nper, rec, rate, lam].
        numerics: (3,) array [nq, span, reserved].

    Expected return:
        float: the par credit spread in basis points.

    Raises:
        ValueError: if model, contract and numerics do not have lengths
        4, 5 and 3, or if dtau <= 0, or if nper < 1.
    """
    return 0.0
```
