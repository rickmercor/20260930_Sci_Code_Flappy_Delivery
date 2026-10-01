# Mathematics-Computational_Finance-4

## Background

American and Bermudan options can be exercised on several dates before maturity, so their value solves an optimal stopping problem: under the risk-neutral measure it is the supremum, over admissible stopping times, of the expected discounted exercise payoff. When the payoff depends on several underlying assets, lattice and finite-difference methods lose their practicality quickly and Monte Carlo simulation becomes the method of choice.

Regression-based Monte Carlo works backward through the exercise dates. At each date the continuation value, the discounted conditional expectation of the value of continuing, is approximated by projecting simulated future values onto a finite set of basis functions of the current state. Comparing the immediate payoff with this approximation gives a stopping rule. Evaluated on independent simulated paths, such a rule is feasible but generally suboptimal, so the resulting estimate is a lower bound on the option value.

Duality supplies the complementary upper bound. For any martingale that starts at zero, the expected pathwise maximum over exercise dates of the discounted payoff minus the martingale is at least the option value, with equality for the martingale part of the discounted value process. Upper-bound methods therefore try to build a martingale close to that optimal one, traditionally with nested simulations inside each path, which are expensive. Regression-based dual methods instead reuse information from the primal calculation and orthogonal projections to build the martingale without nesting, and the gap between the two bounds measures the quality of the approximation.

Options on the maximum of several assets are a standard benchmark for these methods because the exercise region is not a simple threshold in one variable. How well the regressions capture the continuation value depends heavily on the basis: features that follow the assets currently driving the payoff, and features that already carry option-like curvature such as closed-form European prices, typically tighten both bounds considerably compared with plain polynomials.

## Problem

Bermudan options on several assets are priced by optimal stopping, and regression-based Monte Carlo gives a primal lower bound from an estimated stopping rule and a dual upper bound from a martingale built out of the same regressions. Your task is to reproduce one deterministic example of such a primal-dual calculation for a Bermudan call on the maximum of three assets, using the single-projection alpha construction built on orthogonal projections, and to report the dual upper bound.

Use three independent assets under risk-neutral geometric Brownian motion and a Bermudan call on the largest of the three with:

- spot S0 = 100 for every asset
- strike K = 100
- maturity T = 3
- risk-free rate r = 0.05
- dividend yield q = 0.10 for every asset
- volatility sigma = 0.20 for every asset, zero correlation
- number of exercise intervals = 9, exercise dates t_j = jT/9 for j = 1,...,9 (no exercise at time zero)
- payoff = max(max(S1, S2, S3) - K, 0)
- number of training paths = 50,000, training seed = 314159
- number of independent pricing/dual paths = 50,000, pricing seed = 271828
- NumPy random generator = numpy.random.default_rng, with all innovations of a sample drawn in one call as rng.standard_normal((n_paths, n_steps, 3)), entry [n, j, i] driving asset i of path n over step j
- least-squares solver = numpy.linalg.lstsq(A, y, rcond=None), float64 arithmetic

Use the 16-function regression basis of the source paper's Bermudan max-call experiment, built from the two largest of the three asset prices divided by K and including the European two-asset max-call price on those two prices, with expiry T - t_j and the same q, sigma and zero correlation, divided by K. Evaluate that European price in closed form to double precision; at maturity it equals its payoff.

At an eligible date, exercise if and only if the payoff is strictly positive and at least the fitted continuation value. Use all simulated paths in every regression. Fit the stopping rule on the training sample, freeze its coefficients, and construct the backward quantities, the single-projection alpha martingale and the pathwise dual estimator on the independent pricing sample.

In <reasoning>, report the time-zero European two-asset max-call price used by the basis (both prices 100, expiry T), the primal lower bound, the primal-dual gap and the dual upper bound. Report the dual upper bound to at least eight decimal places; it is graded to an absolute tolerance of 1e-6.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
- State the method briefly (one sentence each for the regression, the exercise rule, the martingale and the dual estimator) and show only the scalars that determine the final number.
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

01_simulate_gbm_paths

Goal
----
Simulate independent risk-neutral geometric Brownian motion paths for several assets on an equally spaced time grid. Given a common spot price, maturity, risk-free rate, dividend yield, volatility, number of time steps, number of paths, number of assets and a random seed, return the full price array including the initial prices.

```python
def simulate_gbm_paths(
    spot: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    n_steps: int,
    n_paths: int,
    n_assets: int,
    seed: int,
) -> "np.ndarray":
    """Simulate independent risk-neutral GBM paths for several assets.

    Parameters
    ----------
    spot : float
        Common initial price of every asset, strictly positive.
    maturity : float
        Horizon T in years, strictly positive.
    rate : float
        Continuously compounded risk-free rate.
    dividend_yield : float
        Continuous dividend yield q of every asset.
    volatility : float
        Common volatility sigma, non-negative.
    n_steps : int
        Number of equal time steps, at least one.
    n_paths : int
        Number of simulated paths, at least one.
    n_assets : int
        Number of independent assets, at least one.
    seed : int
        Seed for numpy.random.default_rng.

    Returns
    -------
    paths : np.ndarray
        Float64 array of shape (n_paths, n_steps + 1, n_assets). Innovations
        are drawn as default_rng(seed).standard_normal((n_paths, n_steps,
        n_assets)).

    Raises
    ------
    ValueError
        If spot or maturity is not finite and positive, rate or
        dividend_yield is not finite, volatility is not finite and
        non-negative, or n_steps, n_paths or n_assets is not an integer >= 1,
        or seed is not an integer.
    """
    return paths
```

### Step 2

02_price_two_asset_max_call

Goal
----
Price a European call on the maximum of two assets exactly. Given the two current prices, the strike, the time to expiry, the risk-free rate, each asset's dividend yield and volatility and the correlation between the assets, return the arbitrage-free price of the payoff max(max(S1, S2) - K, 0) at expiry.

```python
def price_two_asset_max_call(
    s1: "np.ndarray",
    s2: "np.ndarray",
    strike: float,
    tau: float,
    rate: float,
    q1: float,
    q2: float,
    sigma1: float,
    sigma2: float,
    rho: float,
) -> "np.ndarray":
    """European call on max(S1, S2) with strike K and time to expiry tau.

    Parameters
    ----------
    s1, s2 : np.ndarray
        Current prices of the two assets, strictly positive; scalars or
        arrays that broadcast together.
    strike : float
        Strike K, strictly positive.
    tau : float
        Time to expiry in years, non-negative.
    rate : float
        Continuously compounded risk-free rate.
    q1, q2 : float
        Continuous dividend yields of the two assets.
    sigma1, sigma2 : float
        Volatilities of the two assets, strictly positive.
    rho : float
        Correlation between the two assets' Brownian motions, in (-1, 1).

    Returns
    -------
    price : np.ndarray
        Float64 prices with the broadcast shape of s1 and s2. For tau = 0
        the price is max(max(s1, s2) - strike, 0).

    Raises
    ------
    ValueError
        If s1 and s2 do not broadcast together, any price is not finite and
        positive, strike is not finite and positive, tau is not finite and
        non-negative, rate, q1 or q2 is not finite, a volatility is not
        finite and positive, or rho is not in the open interval (-1, 1).
    """
    return price
```

### Step 3

03_build_max_call_payoffs_and_basis

Goal
----
Build the Bermudan max-call exercise payoffs and the 16-function regression basis on simulated multi-asset paths. Given the price paths of two or more assets, strike, maturity, risk-free rate, dividend yield and volatility, return the payoff of the call on the maximum of all assets at every date and the basis evaluated at every path and date.

```python
def build_max_call_payoffs_and_basis(
    paths: "np.ndarray",
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Max-call payoffs and the 16-function basis on paths of two or more assets.

    Parameters
    ----------
    paths : np.ndarray
        Prices of shape (n_paths, n_steps + 1, n_assets) with n_assets >= 2,
        on the equally spaced grid t_j = j * maturity / n_steps, all finite and
        positive, n_steps >= 1.
    strike : float
        Strike K, strictly positive.
    maturity : float
        Maturity T in years, strictly positive.
    rate : float
        Continuously compounded risk-free rate.
    dividend_yield : float
        Common continuous dividend yield of the assets.
    volatility : float
        Common volatility of the assets, strictly positive.

    Returns
    -------
    payoffs : np.ndarray
        Shape (n_paths, n_steps + 1), the largest asset price minus K, floored
        at zero.
    basis : np.ndarray
        Shape (n_paths, n_steps + 1, 16), columns in the order
        [1, x1, x1^2, x1^3, x1^4, x2, x2^2, x2^3, x2^4, x1 x2, x1^2 x2,
        x1 x2^2, x1^2 x2^2, c, c^2, c^3], where x1 >= x2 are the largest and
        second-largest prices on that path and date divided by K (equal when
        two assets tie for the maximum) and c is the European two-asset
        max-call price on those two prices with expiry maturity - t_j
        (dividend yield q for both, zero correlation, common volatility)
        divided by K.

    Raises
    ------
    ValueError
        If paths is not a finite, positive array of shape (n_paths, n_times,
        n_assets) with n_times >= 2 and n_assets >= 2, strike or maturity is
        not finite and positive, rate or dividend_yield is not finite, or
        volatility is not finite and positive.
    """
    return payoffs, basis
```

### Step 4

04_fit_backward_primal

Goal
----
Fit the backward least-squares continuation regressions on the training sample and return the fitted coefficients, the training exercise decisions and the backward value process. Given the exercise payoffs, the regression basis at every path and date, the maturity and the risk-free rate, work backward from maturity through the eligible exercise dates.

```python
def fit_backward_primal(
    payoffs: "np.ndarray",
    basis: "np.ndarray",
    maturity: float,
    rate: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Backward least-squares continuation regressions on a training sample.

    Parameters
    ----------
    payoffs : np.ndarray
        Exercise payoffs of shape (n_paths, n_times), finite and
        non-negative, with n_times = n_steps + 1 >= 2.
    basis : np.ndarray
        Regression features of shape (n_paths, n_times, p), finite, p >= 1.
    maturity : float
        Maturity T, strictly positive; dates are t_j = j T / n_steps.
    rate : float
        Continuously compounded risk-free rate.

    Returns
    -------
    coefficients : np.ndarray
        Shape (n_times, p): the least-squares coefficients of the
        continuation regression at each eligible date j = 1, ...,
        n_steps - 1; rows 0 and n_steps are zero.
    exercise_policy : np.ndarray
        Boolean shape (n_paths, n_times): the rule's exercise decisions at
        the eligible dates, exercise at maturity exactly when the payoff is
        positive, and no exercise at time zero.
    backward_values : np.ndarray
        Shape (n_paths, n_times): the realised value of the rule at every
        date, undiscounted; the time-zero column is the date-one value
        discounted over one step.

    Raises
    ------
    ValueError
        If payoffs is not a finite, non-negative two-dimensional array with at
        least two dates, basis is not a finite array of shape (n_paths,
        n_times, p) with p >= 1, maturity is not finite and positive, or rate
        is not finite.
    """
    return coefficients, exercise_policy, backward_values
```

### Step 5

05_apply_frozen_primal_policy

Goal
----
Apply the frozen training-sample stopping rule to an independent pricing sample and return each path's stopping index, its discounted cashflow and their mean, the primal lower-bound estimate. Given the pricing-sample payoffs and basis, the coefficients fitted on the training sample, the maturity and the risk-free rate, stop every path at its first eligible exercise date.

```python
def apply_frozen_primal_policy(
    payoffs: "np.ndarray",
    basis: "np.ndarray",
    coefficients: "np.ndarray",
    maturity: float,
    rate: float,
) -> "tuple[np.ndarray, np.ndarray, float]":
    """Evaluate a frozen regression stopping rule on an independent sample.

    Parameters
    ----------
    payoffs : np.ndarray
        Pricing-sample payoffs of shape (n_paths, n_times), finite and
        non-negative, n_times >= 2.
    basis : np.ndarray
        Pricing-sample features of shape (n_paths, n_times, p), finite.
    coefficients : np.ndarray
        Frozen training coefficients of shape (n_times, p), finite.
    maturity : float
        Maturity T, strictly positive; dates are t_j = j T / n_steps.
    rate : float
        Continuously compounded risk-free rate.

    Returns
    -------
    stopping_indices : np.ndarray
        Int64 array of shape (n_paths,): the stopping date index of each
        path, in 1, ..., n_steps.
    discounted_cashflows : np.ndarray
        Shape (n_paths,): each path's payoff at its stopping date,
        discounted to time zero.
    primal_lower_bound : float
        Mean of discounted_cashflows, as a Python float.

    Raises
    ------
    ValueError
        If payoffs is not a finite, non-negative two-dimensional array with at
        least two dates, basis is not a finite array of shape (n_paths,
        n_times, p) with p >= 1, coefficients is not a finite array of shape
        (n_times, p), maturity is not finite and positive, or rate is not
        finite.
    """
    return stopping_indices, discounted_cashflows, primal_lower_bound
```

### Step 6

06_build_backward_discounted_process

Goal
----
Construct the backward value process of the frozen stopping rule on the pricing sample, in undiscounted and in time-zero-discounted units. Given the pricing-sample payoffs and basis, the frozen training coefficients, the maturity and the risk-free rate, roll the realized value back from maturity to time zero.

```python
def build_backward_discounted_process(
    payoffs: "np.ndarray",
    basis: "np.ndarray",
    coefficients: "np.ndarray",
    maturity: float,
    rate: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Backward value process of the frozen policy on the pricing sample.

    Parameters
    ----------
    payoffs : np.ndarray
        Pricing-sample payoffs of shape (n_paths, n_times), finite and
        non-negative, n_times >= 2.
    basis : np.ndarray
        Pricing-sample features of shape (n_paths, n_times, p), finite.
    coefficients : np.ndarray
        Frozen training coefficients of shape (n_times, p), finite.
    maturity : float
        Maturity T, strictly positive; dates are t_j = j T / n_steps.
    rate : float
        Continuously compounded risk-free rate.

    Returns
    -------
    backward_values : np.ndarray
        Shape (n_paths, n_times): the realised value of the frozen rule at
        every date, undiscounted, with the time-zero column equal to the
        date-one value discounted over one step.
    discounted_backward : np.ndarray
        Shape (n_paths, n_times): backward_values discounted from each date
        t_j to time zero.

    Raises
    ------
    ValueError
        If payoffs is not a finite, non-negative two-dimensional array with at
        least two dates, basis is not a finite array of shape (n_paths,
        n_times, p) with p >= 1, coefficients is not a finite array of shape
        (n_times, p), maturity is not finite and positive, or rate is not
        finite.
    """
    return backward_values, discounted_backward
```

### Step 7

07_price_three_asset_max_call

Goal
----
Price a European call on the maximum of three assets exactly. Given the current prices of the three assets, the strike, the time to expiry, the risk-free rate, each asset's dividend yield and volatility and the correlation matrix of their Brownian motions, return the arbitrage-free price of the payoff max(max(S1, S2, S3) - K, 0) at expiry.

```python
def price_three_asset_max_call(
    spots: "np.ndarray",
    strike: float,
    tau: float,
    rate: float,
    dividend_yields: "np.ndarray",
    volatilities: "np.ndarray",
    correlation: "np.ndarray",
) -> "np.ndarray":
    """European call on max(S1, S2, S3) with strike K and time to expiry tau.

    Parameters
    ----------
    spots : np.ndarray
        Current prices of shape (..., 3), all finite and strictly positive;
        the last axis holds the three assets and any leading axes index
        independent price vectors.
    strike : float
        Strike K, finite and strictly positive.
    tau : float
        Time to expiry in years, finite and non-negative.
    rate : float
        Continuously compounded risk-free rate, finite.
    dividend_yields : np.ndarray
        Continuous dividend yields of the three assets, shape (3,), finite.
    volatilities : np.ndarray
        Volatilities of the three assets, shape (3,), finite and strictly
        positive.
    correlation : np.ndarray
        Correlation matrix of the three Brownian motions, shape (3, 3),
        finite, symmetric, with unit diagonal and positive definite.

    Returns
    -------
    price : np.ndarray
        Float64 prices with shape spots.shape[:-1]. For tau = 0 the price is
        max(max(spots over the last axis) - strike, 0).

    Raises
    ------
    ValueError
        If spots is not a finite, strictly positive array whose last axis has
        length 3, strike is not finite and positive, tau is not finite and
        non-negative, rate is not finite, dividend_yields is not a finite
        array of shape (3,), volatilities is not a finite, strictly positive
        array of shape (3,), or correlation is not a finite, symmetric,
        positive definite (3, 3) matrix with unit diagonal.
    """
    return price
```

### Step 8

08_compute_dual_upper_bound

Goal
----
Run the complete primal-dual pipeline for the Bermudan call on the maximum of several independent assets and return the dual upper-bound estimate. Given the common spot price, strike, maturity, rate, dividend yield, volatility, number of assets (two or three), number of exercise intervals, and the sizes and seeds of the training and pricing samples, compose the earlier steps into one evaluation and construct the dual martingale within this step.

```python
def compute_dual_upper_bound(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    n_assets: int,
    n_steps: int,
    n_training_paths: int,
    training_seed: int,
    n_pricing_paths: int,
    pricing_seed: int,
) -> "tuple[np.ndarray, np.ndarray, float, float, float]":
    """Primal-dual evaluation of a Bermudan max-call on independent assets.

    Parameters
    ----------
    spot : float
        Common initial price of every asset.
    strike : float
        Strike K.
    maturity : float
        Maturity T in years; exercise dates are t_j = j T / n_steps,
        j = 1, ..., n_steps.
    rate : float
        Continuously compounded risk-free rate.
    dividend_yield : float
        Common continuous dividend yield.
    volatility : float
        Common volatility.
    n_assets : int
        Number of independent assets, two or three.
    n_steps : int
        Number of equal exercise intervals.
    n_training_paths, n_pricing_paths : int
        Sizes of the training and the independent pricing samples.
    training_seed, pricing_seed : int
        Seeds of the two samples; each sample is simulated with the draw
        layout of simulate_gbm_paths.

    Returns
    -------
    discounted_exercise_payoffs : np.ndarray
        Shape (n_pricing_paths, n_steps + 1), each pricing path's payoff at
        every date discounted to time zero.
    dual_path_values : np.ndarray
        Shape (n_pricing_paths,), each path's largest discounted payoff minus
        martingale over the exercise dates j = 1, ..., n_steps.
    dual_upper_bound : float
        Mean of dual_path_values.
    primal_dual_gap : float
        dual_upper_bound minus the primal lower bound of the frozen policy.
    european_price : float
        Time-zero price of the European call on the maximum of all n_assets
        assets (no early exercise), each at the common spot with the common
        dividend yield and volatility, independent, with expiry maturity.

    Raises
    ------
    ValueError
        If spot, strike, maturity or volatility is not finite and positive,
        rate or dividend_yield is not finite, n_assets is not the integer 2 or
        3, n_steps, n_training_paths or n_pricing_paths is not a
        positive integer, or a seed is not a non-negative integer.
    """
    return discounted_exercise_payoffs, dual_path_values, dual_upper_bound, primal_dual_gap, european_price
```
