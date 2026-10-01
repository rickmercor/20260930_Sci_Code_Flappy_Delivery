# Mathematics-Computational_Finance-17

## Background

A forward contract fixes today the exchange rate at which currency will be swapped on a future date. A flexible forward relaxes the date: the holder still gets the agreed rate, but may call for delivery at any point inside a window. That sounds like a small concession and it is not. The right to choose when to settle is worth money, because the holder will settle at the moment the agreed rate looks best against the market, and pricing the contract means putting a number on that right.
The difficulty is that the decision is forward-looking. Taking delivery now locks in the gain available now; waiting keeps the chance of a better one, at the cost of whatever the two currencies earn in the meantime. There is therefore a moving threshold, a level of the exchange rate below which waiting is no longer worth it. That threshold is not written in the contract. It emerges from the model and has to be found together with the price, because each depends on the other. The standard way out is to write the price as the plain forward plus an integral over the window that collects, at each instant, the advantage of settling rather than waiting, weighted by the chance of being where settling is better. The rearrangement is exact, and it turns an awkward optimisation into a fixed-point problem: guess the threshold, evaluate the integral, read off a better threshold, repeat.
Many of these contracts are agreed before their rate is known. The rate is set on a fixing date from the market forward at that moment, so seen from today everything the contract will pay is proportional to a spot level that has not happened yet. That has to be dealt with before anything after the fixing date can be valued or discounted.
What makes each evaluation expensive is the model underneath. Market data will not tolerate a constant volatility, so volatility gets its own random dynamics, and those dynamics change character at scheduled dates, which is how the term structure of the option market is reproduced; the two interest rates drift in calendar time as well. The price of that realism is that the distribution of the exchange rate is known only through its Fourier transform, built by solving a small nonlinear ordinary differential equation whose coefficients jump at each of those dates.
The last ingredient is the one that makes the method fast. The quantity to be integrated is cut off sharply at the threshold, and a sharp cut-off is what a spectral method handles badly, because the series that represents it rings. The remedy is to damp the integrand by an exponential before transforming it, which makes the transform of the cut-off piece an elementary expression, and then to pair the two transforms directly.

## Problem

A flexible forward lets its holder pick the delivery date inside an agreed window, so on top of the plain forward it carries an early-exercise premium for the timing right. In the forward-starting version priced here the delivery rate is not agreed today: it is fixed at a later date to the outright forward then prevailing, so from today's standpoint the whole contract is written on a spot level that has not yet been observed. Under a time-inhomogeneous Heston model with time-dependent interest rates the premium has no closed form. The decomposition formula writes it as a time integral of expectations taken over the exercise region, and a recent method evaluates each expectation spectrally, damping the truncated integrand so that its transform is analytic and pairing it against the characteristic function. Solving the contract completely means iterating that evaluation until the exercise surface stops moving; your task is one such evaluation, on the surface iterate supplied below.

Here is the exact setup to use:
- Model: $$dX_t = \left(r_d(t) - r_f(t) - \tfrac12 v_t\right)dt + \sqrt{v_t}\,dW^S_t, \qquad dv_t = \kappa\left(\theta(t) - v_t\right)dt + \xi(t)\sqrt{v_t}\,dW^v_t, \qquad dW^S_t\,dW^v_t = \rho(t)\,dt,$$ with $X_t = \log S_t$, spot $S_0 = 1$, mean reversion $\kappa = 2.0$ and initial variance $v_0 = 0.09$.
- Term structure: $\xi$, $\rho$ and $\theta$ are constant on each of the three calendar intervals $[0,\,0.5)$, $[0.5,\,1.0)$, $[1.0,\,1.5]$, taking the values $\xi = (0.40,\ 0.50,\ 0.45)$, $\rho = (-0.60,\ -0.40,\ -0.50)$ and $\theta = (0.09,\ 0.10,\ 0.08)$ on those intervals in order.
- Rates: deterministic, $r_d(t) = 0.028 + 0.004\,e^{-1.5t}$ and $r_f(t) = 0.034 + 0.003\,e^{-0.8t}$, with $t$ in years.
- Contract: a flexible forward on one unit of foreign currency. At the fixing date $t_0 = 0.625$ the delivery rate $K$ is set equal to the outright forward rate prevailing at $t_0$ for delivery at $T_2$. The delivery window is $[T_1, T_2] = [0.75,\ 1.5]$; the holder takes delivery once, at a date $\tau$ of their choice in the window, and receives $K - S_\tau$ in domestic currency.
- Exercise surface iterate: in log-moneyness $x = \log(S/K)$, $$x^{\star}(u, v) = x^{\star}(T_2^-) - \left(A_0 + B_0\,v\right)\sqrt{T_2 - u},\qquad A_0 = 0.28,\quad B_0 = 1.60,$$ where $x^{\star}(T_2^-)$ is the terminal value of the exercise boundary for this contract. Delivery is taken at $u$ when $x_u < x^{\star}(u, v_u)$.
- Damping: apply the source paper's damped spectral evaluation with damping parameter $\alpha = 1$.
- Frequency rule: composite Simpson on $[-\Omega, \Omega]$ with $\Omega = 300$ and $N_\omega = 8192$ equal sub-intervals.
- Time rule: the premium integral is taken by the composite trapezoidal rule on a uniform grid of $N_t = 16$ sub-intervals spanning the delivery window.

Run the pipeline and report the contract value at time $0$ in domestic pips, $s = 10^{4}\,V(0)$. The answer is graded to an absolute tolerance of $10^{-6}$. In your reasoning, report the terminal boundary $x^{\star}(T_2^-)$ and the two surface coefficients at $u = T_1$; the delivery rate $K$ as a multiple of the spot at the fixing date; the premium density, meaning the integrand of the early-exercise premium in the decomposition discounted to time $0$ and expressed in pips per year, at $u = 1$, at $u = 1.25$ and at $u = T_2$; and the value the same pipeline returns with the time rule halved to eight sub-intervals, with it refined to thirty-two, with $B_0$ doubled to $3.20$, with the fixing date moved to $t_0 = 0.5$, and with $A_0$ set to zero, each change made on its own.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the Simpson nodes, the frequency-grid kernel values, or the per-node integrand table.

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

Riccati interval advance

Goal
----
Advance the variance exponent of the joint characteristic function of log-spot and variance across one interval on which the volatility of variance and the correlation are constant, and return both the value the exponent reaches and its integral over the interval. Every exponent used later in the pipeline is assembled from calls to this step, so an error here propagates everywhere downstream.

```python
import numpy as np


def riccati_interval_advance(b0_re, b0_im, u1_re, u1_im, xi, rho, kappa, s):
    """The variance exponent B of the joint characteristic function obeys a scalar
    quadratic ordinary differential equation in time-to-delivery.  Its three
    coefficients follow from the dynamics given in the problem statement and
    from the frequency argument u1, and on this interval the
    volatility-of-volatility and the correlation are constant, so the
    coefficients are constant too.  Advance that equation from B(0) = b0 across
    a span of length s.  Both B(s) and the integral of B over [0, s] are
    returned, the second because the accumulated log-spot exponent consumes it.

    The frequency argument is complex, not real: the evaluation runs on a
    contour shifted off the real axis, so an implementation checked only at real
    u1 has not been checked.

    Args:
        b0_re (float): real part of the initial condition B(0).
        b0_im (float): imaginary part of the initial condition B(0).
        u1_re (float): real part of the frequency argument u1.
        u1_im (float): imaginary part of the frequency argument u1.
        xi (float): volatility of volatility on the interval, strictly positive.
        rho (float): correlation on the interval, in [-1, 1].
        kappa (float): mean-reversion speed, strictly positive.
        s (float): interval length, non-negative.

    Expected return:
        np.ndarray of shape (4,) packed as [Re B(s), Im B(s), Re integral,
        Im integral], where integral is the integral of B over [0, s].  For
        s = 0 the result is [b0_re, b0_im, 0, 0].

    Array inputs:
        b0_re, b0_im, u1_re and u1_im may also be 1-D arrays of one common
        length m (or broadcast against scalars).  The four quantities are
        then evaluated elementwise and the result has shape (4, m).

    Raises:
    ValueError: if xi <= 0, if rho lies outside [-1, 1], if kappa <= 0, or if s < 0.
    """
    return np.zeros(4)
```

### Step 2

Characteristic exponents

Goal
----
Return the two exponents of the joint characteristic function of log-spot and variance for one step of the time-inhomogeneous model, starting from a general complex coefficient on the terminal variance. The step may span any of the parameter breakpoints, and the domestic and foreign rates move continuously in calendar time rather than being constant.

```python
import numpy as np


def characteristic_exponents(u1_re, u1_im, b0_re, b0_im, t, T, model, term):
    """Exponents A and B of the joint characteristic function for the step t -> T.

    With x the log-spot and v the variance,

        E[ exp(1j*u1*x_T + b0*v_T) | x_t = x, v_t = v ]
            = exp(1j*u1*x + A + B*v),

    under the dynamics of the problem statement, with the piecewise-constant
    term structure in term and the domestic and foreign rate curves in model.
    Each rate curve has the form r(s) = level + amplitude*exp(-decay*s) in
    calendar time s; a decay of zero means the rate is the constant
    level + amplitude.

    Args:
        u1_re (float): real part of the frequency argument u1.
        u1_im (float): imaginary part of the frequency argument u1.
        b0_re (float): real part of the variance coefficient b0.
        b0_im (float): imaginary part of the variance coefficient b0.
        t (float): calendar start of the step.
        T (float): calendar end of the step, not before t.
        model (np.ndarray): shape (8,), packed as [kappa, v0, rd_level,
            rd_amplitude, rd_decay, rf_level, rf_amplitude, rf_decay].
        term (np.ndarray): shape (13,), packed as the four interval edges
            followed by the three volatility-of-variance values, the three
            correlations and the three mean-reversion levels.

    Expected return:
        np.ndarray of shape (4,) packed as [Re A, Im A, Re B, Im B].  For
        T == t the result is [0, 0, b0_re, b0_im].

    Array inputs:
        u1_re, u1_im, b0_re and b0_im may also be 1-D arrays of one common
        length m (or broadcast against scalars).  A and B are then evaluated
        elementwise and the result has shape (4, m).

    Raises:
    ValueError: if model does not have eight entries, if term does not have thirteen entries, if T < t, if kappa <= 0, if the interval edges are not strictly increasing, or if any volatility of variance is not positive.
    """
    return np.zeros(4)
```

### Step 3

Simpson frequency rule

Goal
----
Build the quadrature rule used for every frequency integral in the pipeline: a composite Simpson rule on a symmetric interval about the origin, returned as nodes and weights together because every later step consumes them as a pair.

```python
import numpy as np


def simpson_frequency_rule(omega_max, n):
    """Composite Simpson rule on [-omega_max, omega_max].

    Args:
        omega_max (float): half-width of the frequency interval, strictly
            positive.
        n (int): number of equal sub-intervals, an even integer of at least two.

    Expected return:
        np.ndarray of shape (2*(n+1),) packed as [nodes, weights], nodes in
        increasing order from -omega_max to +omega_max and weights already
        carrying the step length, so that they sum to 2*omega_max.

    Raises:
    ValueError: if omega_max <= 0, if n < 2, or if n is odd.
    """
    return np.zeros(2 * (int(n) + 1))
```

### Step 4

Truncated payoff prefactors

Goal
----
Return the damped Fourier transform of the amount the early-exercise decomposition collects on the exercise region, cut off sharply at a given exercise level. The amount is what accrues to the holder of a contract paying K − S at delivery, per unit strike and per unit time, while the state sits where immediate delivery is optimal.

```python
import numpy as np


def truncated_payoff_transform(omega, alpha, xstar, rd, rf):
    """Damped Fourier transform of the truncated early-exercise amount.

    Let g(x) be the amount the early-exercise decomposition collects, per
    unit strike and per unit time, while the state sits in the exercise
    region of a contract that pays K - S at delivery.  It is written in
    log-moneyness x = log(S/K) and uses the instantaneous domestic rate rd
    and foreign rate rf of the date in question.  This step returns

        G(omega) = integral over x from -inf to xstar of
                   g(x) * exp((1j*omega + alpha)*x) dx

    at every node of omega.

    Args:
        omega (np.ndarray): frequency nodes, shape (m,), non-empty.
        alpha (float): damping, strictly positive.
        xstar (float): cut-off, the exercise level in log-moneyness.
        rd (float): instantaneous domestic rate.
        rf (float): instantaneous foreign rate.

    Expected return:
        np.ndarray of shape (2*m,) packed as [Re G, Im G], each block of
        length m.

    Raises:
    ValueError: if omega is empty or if alpha <= 0.
    """
    m = np.atleast_1d(np.asarray(omega, dtype=float)).reshape(-1).size
    return np.zeros(2 * m)
```

### Step 5

Exercise surface coefficients

Goal
----
Return the intercept and the slope, in the variance, of the supplied exercise-surface iterate at one date in the delivery window. The pipeline only ever needs the surface at one date at a time, and because the surface is affine in the variance those two numbers describe it completely.

```python
import numpy as np


def exercise_surface_coefficients(u, T2, A0, B0, model):
    """Intercept and slope of the affine exercise-surface iterate at time u.

    The iterate is

        xstar(u, v) = xstar_terminal - (A0 + B0*v) * sqrt(T2 - u),

    in log-moneyness, where xstar_terminal is the terminal value of the
    exercise boundary of the contract in the problem statement.  That level
    is a property of the payoff and the rate curves, not a configuration
    constant, and it is not zero.

    Args:
        u (float): time in the delivery window, not after T2.
        T2 (float): far end of the delivery window, strictly positive.
        A0 (float): intercept shape constant.
        B0 (float): slope shape constant.
        model (np.ndarray): shape (8,), packed as [kappa, v0, rd_level,
            rd_amplitude, rd_decay, rf_level, rf_amplitude, rf_decay], each
            rate curve being r(s) = level + amplitude*exp(-decay*s).

    Expected return:
        np.ndarray of shape (2,) packed as [intercept, slope], so that the
        surface at variance v is intercept + slope*v.  At u = T2 the slope
        vanishes and the intercept is the terminal boundary.

    Raises:
    ValueError: if model does not have eight entries, if T2 <= 0, if u > T2, or if either rate at T2 is not strictly positive.
    """
    return np.zeros(2)
```

### Step 6

Spot-weighted variance exponents

Goal
----
Return the exponents of the transform of the variance at a fixing date, taken in the measure that weights every path by the spot at that date, discounted on the foreign curve and normalised so the weights average to one. A contract whose strike is set at the fixing date in proportion to the spot values everything after that date per unit of the fixing-date spot, and this step supplies the average over the fixing-date variance that such a valuation needs.

```python
import numpy as np


def spot_weighted_variance_exponents(q_re, q_im, t0, model, term):
    """Exponents of the spot-weighted transform of the variance at t0.

    With S the spot, D_d(0, t) and D_f(0, t) the domestic and foreign
    discount factors built from the rate curves in model, and

        Z_t = S_t * D_d(0, t) / (S_0 * D_f(0, t)),

    this step returns Ahat and Bhat such that

        E[ Z_t0 * exp(q * v_t0) ] = exp(Ahat + Bhat * v0),

    the expectation being taken under the pricing measure of the problem
    statement from the initial state at time 0.  Each rate curve has the form
    r(s) = level + amplitude*exp(-decay*s).

    Args:
        q_re (float): real part of the variance coefficient q.
        q_im (float): imaginary part of the variance coefficient q.
        t0 (float): the date at which the variance is observed, non-negative.
        model (np.ndarray): shape (8,), packed as [kappa, v0, rd_level,
            rd_amplitude, rd_decay, rf_level, rf_amplitude, rf_decay].
        term (np.ndarray): shape (13,), packed as the four interval edges
            followed by the three volatility-of-variance values, the three
            correlations and the three mean-reversion levels.

    Expected return:
        np.ndarray of shape (4,) packed as [Re Ahat, Im Ahat, Re Bhat,
        Im Bhat].  For t0 = 0 the result is [0, 0, q_re, q_im].

    Array inputs:
        q_re and q_im may also be 1-D arrays of one common length m (or
        broadcast against a scalar).  The exponents are then evaluated
        elementwise and the result has shape (4, m).

    Raises:
    ValueError: if model does not have eight entries, if term does not have thirteen entries, if t0 < 0, if kappa <= 0, if the interval edges are not strictly increasing, or if any volatility of variance is not positive.
    """
    return np.zeros(4)
```

### Step 7

Forward-start kernel

Goal
----
Return the damped Fourier transform, in the log-moneyness at the fixing date, of the exercise-region expectation at one date in the delivery window, seen from time 0 for a contract whose strike is set at the fixing date. The inversion step turns this object back into an expectation, and this is where the transform of the truncated amount, the characteristic function after fixing and the average over the fixing-date variance come together.

```python
import numpy as np


def forward_start_kernel(omega, alpha, t0, u, a, b, model, term):
    """Damped transform, in the fixing-date log-moneyness, of the forward-start
    exercise-region expectation at date u.

    Write X = log S and Z_t0 = S_t0 * D_d(0, t0) / (S_0 * D_f(0, t0)), the
    same spot weight as in the spot-weighted variance step, and let g_u be
    the amount transformed in the truncated-transform step, evaluated with the
    instantaneous rates at date u.  For a fixing-date log-moneyness x define

        Jhat(x) = E[ Z_t0 * g_u(x + X_u - X_t0)
                     * 1{ x + X_u - X_t0 < a + b*v_u } ],

    the expectation being taken under the pricing measure from the initial
    state at time 0, with v0 from model.  This step returns

        W(omega) = integral over x in R of exp((1j*omega + alpha)*x) * Jhat(x) dx

    at every node of omega.

    Args:
        omega (np.ndarray): frequency nodes, shape (m,), non-empty.
        alpha (float): damping, strictly positive.
        t0 (float): fixing date, non-negative.
        u (float): date in the delivery window, not before t0.
        a (float): intercept of the exercise surface at u.
        b (float): slope of the exercise surface at u.
        model (np.ndarray): shape (8,), packed as [kappa, v0, rd_level,
            rd_amplitude, rd_decay, rf_level, rf_amplitude, rf_decay].
        term (np.ndarray): shape (13,), packed as the four interval edges
            followed by the three volatility-of-variance values, the three
            correlations and the three mean-reversion levels.

    Expected return:
        np.ndarray of shape (2*m,) packed as [Re W, Im W], each block of
        length m.

    Raises:
    ValueError: if omega is empty, if alpha <= 0, if t0 < 0, if u < t0, or if model does not have eight entries or term thirteen.
    """
    m = np.atleast_1d(np.asarray(omega, dtype=float)).reshape(-1).size
    return np.zeros(2 * m)
```

### Step 8

Exercise region expectation

Goal
----
Turn a damped transform back into the expectation it encodes, at one value of the log-moneyness, by summing the kernel against the frequency rule. The result is real, because the function being recovered is real.

```python
import numpy as np


def exercise_region_expectation(kernel_flat, rule_flat, alpha, x_src):
    """Recover an expectation from its damped transform at one log-moneyness.

    kernel_flat holds samples, at the nodes of rule_flat, of a damped transform

        W(omega) = integral over x in R of exp((1j*omega + alpha)*x) * f(x) dx

    of a real function f.  Using the quadrature rule for the frequency
    integral, return the value f(x_src).

    Args:
        kernel_flat (np.ndarray): shape (2*m,), packed as [Re W, Im W].
        rule_flat (np.ndarray): shape (2*m,), packed as [nodes, weights].
        alpha (float): damping, strictly positive, the same one used to build
            the kernel.
        x_src (float): log-moneyness at which f is recovered.

    Expected return:
        float: the recovered value f(x_src), a real number.

    Raises:
    ValueError: if either packed array has odd length, if the two imply a different number of nodes, or if alpha <= 0.
    """
    return 0.0
```

### Step 9

Premium time rule

Goal
----
Build the time rule for the early-exercise premium: the dates at which the exercise-region expectation is evaluated and the weights that combine them, with the domestic discounting from the fixing date already folded into the weights.

```python
import numpy as np


def premium_time_rule(T1, T2, nstep, t0, model):
    """Discounted composite trapezoidal rule over the delivery window.

    The nodes are nstep + 1 equally spaced dates from T1 to T2.  Each weight
    is the composite trapezoidal weight of its node multiplied by the
    domestic discount factor from the fixing date t0 to that node, built from
    the domestic rate curve r_d(s) = level + amplitude*exp(-decay*s) in model.

    Args:
        T1 (float): near end of the delivery window, not before t0.
        T2 (float): far end of the delivery window, strictly after T1.
        nstep (int): number of equal sub-intervals, at least one.
        t0 (float): fixing date, non-negative.
        model (np.ndarray): shape (8,), packed as [kappa, v0, rd_level,
            rd_amplitude, rd_decay, rf_level, rf_amplitude, rf_decay].

    Expected return:
        np.ndarray of shape (2*(nstep+1),) packed as [nodes, weights], nodes
        increasing from T1 to T2.

    Raises:
    ValueError: if model does not have eight entries, if T2 <= T1, if nstep < 1, if t0 < 0, or if T1 < t0.
    """
    return np.zeros(2 * (int(nstep) + 1))
```

### Step 10

Forward-start flexible forward value

Goal
----
Chain the earlier steps into one evaluation of the forward-starting flexible forward on the supplied exercise-surface iterate, and return its value at time 0 in domestic pips.

```python
import numpy as np


def forward_start_flexible_forward_value(model, term, contract, numerics):
    """Value in domestic pips of the forward-starting flexible forward.

    The contract is on one unit of foreign currency with S_0 = 1.  At the
    fixing date t0 its delivery rate K is set equal to the outright forward
    rate then prevailing for delivery at T2.  The holder then takes delivery
    once, at a single date in [T1, T2] of their choosing, and receives
    K - S at that date in domestic currency.  The exercise surface is the
    supplied iterate in log-moneyness log(S/K) with shape constants A0 and
    B0; the frequency integral uses the composite Simpson rule on
    [-omega_max, omega_max] with n_omega sub-intervals and damping alpha;
    the premium integral over the window uses the composite trapezoidal rule
    with n_step sub-intervals.

    Args:
        model (np.ndarray): shape (8,), packed as [kappa, v0, rd_level,
            rd_amplitude, rd_decay, rf_level, rf_amplitude, rf_decay], each
            rate curve being r(s) = level + amplitude*exp(-decay*s).
        term (np.ndarray): shape (13,), packed as the four interval edges
            followed by the three volatility-of-variance values, the three
            correlations and the three mean-reversion levels.
        contract (np.ndarray): shape (5,), packed as [t0, T1, T2, A0, B0].
        numerics (np.ndarray): shape (4,), packed as
            [alpha, omega_max, n_omega, n_step]; the last two are read as
            integers.

    Expected return:
        float: the contract value at time 0 in domestic pips, ten thousand
        times the value per unit of foreign notional.

    Raises:
    ValueError: if model does not have eight entries, if term does not have thirteen, if contract does not have five, if numerics does not have four, if t0 < 0, if T1 < t0, or if T2 <= T1.
    """
    return 0.0
```
