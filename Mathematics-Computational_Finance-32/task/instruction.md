# Mathematics-Computational_Finance-32

## Background

An exchange option gives its holder the right to swap one risky asset for another at maturity. It was priced in closed form by Fischer and by Margrabe in 1978 under lognormal dynamics. Taking the delivered asset as numeraire turns the exchange option into an ordinary call with unit strike on the price ratio, and the interest rate drops out.

Real markets break the lognormal assumptions in several ways at once. Volatility is itself random and correlated with returns, which stochastic-volatility models such as Heston's capture with a mean-reverting variance process. Liquidity varies over time. Assets that are hard to trade command a discount, and one tractable way to model this multiplies the frictionless price by a liquidity discount process driven by a market-wide liquidity factor and an asset-specific sensitivity. Economic conditions also change in discrete steps, which regime-switching models describe with a finite-state Markov chain that modulates the model coefficients.

Combining these features keeps the problem tractable when the joint dynamics stay affine. The characteristic function of the log price ratio is then exponential in the state variables, and its coefficients solve Riccati-type ordinary differential equations. The regime dependence can be absorbed into a matrix-valued linear equation driven by the chain's generator. Option values follow from Fourier inversion of the characteristic function. The recent literature this task draws on applies that programme to exchange options with regime-switching Heston variances and stochastic market liquidity, and compares the resulting prices with constant-volatility and constant-liquidity benchmarks.

## Problem

Exchange options pay the difference between two risky assets at maturity, and recent work prices them in a market where the long-run variance levels switch with a Markov chain between economic regimes, each asset carries its own Heston variance, and a stochastic market-liquidity factor discounts both prices. The liquidity channel adds a term quadratic in the liquidity factor to the variance of each asset, so the characteristic function of the log price ratio remains exponential-affine once the second asset is used as numeraire; your task is to price one such option.

Under the risk-neutral measure, for i = 1, 2, dS_i/S_i = r(X) dt + sigma_i(X) dW_i + beta_i alpha dW_{i,d} + sqrt(nu_i) dB_i, dnu_i = kappa_i (theta_i(X) - nu_i) dt + xi_i sqrt(nu_i) dB_{i,nu}, and the market-liquidity factor follows dalpha = a (b - alpha) dt + eta dW_alpha. The correlations are dW_1 dW_2 = rho dt, dB_i dB_{i,nu} = rho_i dt and dW_{i,d} dW_alpha = rho~_i dt, and every other pair of Brownian motions is independent, including W_{1,d} and W_{2,d}. X is a two-state Markov chain independent of all Brownian motions, with rate 0.5 from regime 1 to regime 2 and rate 0.45 from regime 2 to regime 1.

Use S_1(0) = 100, S_2(0) = 95, nu_1(0) = nu_2(0) = 0.1, alpha(0) = 0.3, kappa_1 = kappa_2 = 2, xi_1 = xi_2 = 0.1, rho_1 = rho_2 = -0.5, rho = -0.25, rho~_1 = rho~_2 = -0.7, beta_1 = beta_2 = 0.5, a = 0.2, b = 0.3 and eta = 0.9. In regime 1 take sigma_1 = 0.1, sigma_2 = 0.2, theta_1 = theta_2 = 0.1 and r = 0.04; in regime 2, where the economy is at time 0, take sigma_1 = 0.1, sigma_2 = 0.2, theta_1 = theta_2 = 0.3 and r = 0.02. Report the time-0 value of the European option paying (S_1(T) - S_2(T))^+ at T = 1 year, accurate to 1e-3.

In your reasoning, state which drifts change when the numeraire changes, and give numerically the two exercise probabilities that price the unit-strike call on S_1/S_2 under the second asset as numeraire, the value if the economy starts in regime 1, the value if regime switching is switched off with the economy staying in regime 2, and the value when the liquidity channel is removed (beta_1 = beta_2 = 0).

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

01_variance_riccati

Goal
----
Evaluate, in closed form, the coefficient that multiplies a Heston variance in the exponent of the log-price characteristic function.

```python
import numpy as np

def variance_riccati(delta: np.ndarray, tau: float, xi: float, c0: float, c1: float) -> np.ndarray:
    '''Closed-form variance coefficient of the affine characteristic function.

    Parameters
    ----------
    delta : np.ndarray
        One-dimensional array of n >= 1 transform values; real or complex
        (for example delta - 1j), all finite.
    tau : float
        Remaining time, tau >= 0.
    xi : float
        Volatility of the variance factor, xi > 0.
    c0 : float
        Real part of the linear coefficient of the Riccati equation.
    c1 : float
        Coefficient of i * delta in the linear coefficient.

    Returns
    -------
    y : np.ndarray
        Shape (2, n) float array: real and imaginary parts of Y(delta, tau).

    Raises
    ------
    ValueError
        If delta is not a finite one-dimensional array with at least one
        entry, if tau is negative, if xi is not positive, or if c0 or c1 is
        not finite.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs inside the function body.
    '''
    return np.zeros((2, np.atleast_1d(delta).size), dtype=float)  # placeholder
```

### Step 2

02_liquidity_coefficients

Goal
----
Evaluate the coefficients through which the stochastic market-liquidity factor enters the characteristic function, and the part of the constant term they generate.

```python
import numpy as np

def liquidity_coefficients(delta: np.ndarray, tau: float, a: float, b: float, eta: float,
                           m: float, c_x: float, beta_sq: float, n_quad: int = 64) -> np.ndarray:
    '''Liquidity coefficients C, B and their constant-term contribution.

    Parameters
    ----------
    delta : np.ndarray
        One-dimensional array of n >= 1 finite transform values (real or complex).
    tau : float
        Remaining time, tau >= 0.
    a : float
        Mean-reversion rate a > 0 in the constant drift a b of alpha.
    b : float
        Long-run level b in the constant drift a b of alpha.
    eta : float
        Volatility of alpha, eta > 0.
    m : float
        Coefficient of alpha in its drift under the measure used.
    c_x : float
        Instantaneous covariance of alpha with the log price per unit
        eta * alpha.
    beta_sq : float
        beta1^2 + beta2^2 >= 0.
    n_quad : int
        Number of Gauss-Legendre nodes for the remaining integral, >= 2.

    Returns
    -------
    coeffs : np.ndarray
        Shape (6, n) float array; see the module description.

    Raises
    ------
    ValueError
        If any argument is outside its domain.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    '''
    return np.zeros((6, np.atleast_1d(delta).size), dtype=float)  # placeholder
```

### Step 3

03_regime_factor

Goal
----
Compute the regime-switching factor of the characteristic function: the conditional expectation, given the initial regime, of the exponential of the regime-dependent part of the constant term.

```python
import numpy as np

def regime_factor(delta: np.ndarray, tau: float, generator: np.ndarray, sig2: np.ndarray,
                  kappa1: float, theta1: np.ndarray, xi1: float, d_c0: float, d_c1: float,
                  kappa2: float, theta2: np.ndarray, xi2: float, e_c0: float, e_c1: float,
                  initial_regime: int, n_steps: int = 200) -> np.ndarray:
    '''Regime-switching factor of the characteristic function.

    Parameters
    ----------
    delta : np.ndarray
        One-dimensional array of n >= 1 finite transform values.
    tau : float
        Remaining time, tau >= 0.
    generator : np.ndarray
        Shape (k, k) generator of the chain, k >= 1, non-negative off-diagonal
        entries and zero row sums.
    sig2 : np.ndarray
        Shape (k,) constant variance rate of the log price in each regime.
    kappa1, kappa2 : float
        Mean-reversion rates of the two variance factors.
    theta1, theta2 : np.ndarray
        Shape (k,) long-run variances of the two factors in each regime.
    xi1, xi2 : float
        Volatilities of the two variance factors, > 0.
    d_c0, d_c1, e_c0, e_c1 : float
        Linear coefficients (c0, c1) of the Riccati equations of D and E, as
        in ``variance_riccati``.
    initial_regime : int
        Regime at time 0, numbered from 1 to k.
    n_steps : int
        Number of exponential midpoint steps, >= 1.

    Returns
    -------
    h : np.ndarray
        Shape (2, n) float array: real and imaginary parts of h_j.

    Raises
    ------
    ValueError
        If any argument is outside its domain or the shapes are inconsistent.

    Notes
    -----
    Obtain D and E from ``variance_riccati``. Include every import your
    implementation needs inside the function body.
    '''
    return np.zeros((2, np.atleast_1d(delta).size), dtype=float)  # placeholder
```

### Step 4

04_characteristic_function

Goal
----
Assemble the characteristic function of the log price ratio ln(S1(T)/S2(T)) under the measure that takes the second asset as numeraire, at the initial state and regime.

```python
import numpy as np

def characteristic_function(delta: np.ndarray, tau: float, model: dict, n_quad: int = 64,
                            n_steps: int = 200) -> np.ndarray:
    '''Characteristic function of ln(S1(T)/S2(T)) under the S2-numeraire measure.

    Parameters
    ----------
    delta : np.ndarray
        One-dimensional array of n >= 1 finite transform values (real or complex).
    tau : float
        Time to maturity, tau >= 0.
    model : dict
        Keys: s1, s2, nu1, nu2, alpha (initial values, s1, s2 > 0, nu1, nu2 >= 0);
        kappa1, kappa2 > 0; xi1, xi2 > 0; rho1, rho2 (each asset with its own
        variance); rho (between the two diffusive Brownian motions);
        rho_tilde1, rho_tilde2 (each asset's liquidity noise with the market
        liquidity noise); beta1, beta2 >= 0; a > 0, b, eta > 0 (market
        liquidity under the pricing measure); sigma1, sigma2, theta1, theta2
        (sequences, one value per regime); generator (k x k); initial_regime
        (1 to k).
    n_quad : int
        Gauss-Legendre nodes for the liquidity constant term.
    n_steps : int
        Exponential midpoint steps for the regime factor.

    Returns
    -------
    phi : np.ndarray
        Shape (2, n) float array: real and imaginary parts.

    Raises
    ------
    ValueError
        If a key is missing or any entry is outside its domain.

    Notes
    -----
    Use ``variance_riccati``, ``liquidity_coefficients`` and
    ``regime_factor``. Include every import your implementation needs inside
    the function body.
    '''
    return np.zeros((2, np.atleast_1d(delta).size), dtype=float)  # placeholder
```

### Step 5

05_exercise_probabilities

Goal
----
Recover from the characteristic function the two probabilities that price the unit-strike call on the price ratio, together with the normalisation phi(-i).

```python
import numpy as np

def exercise_probabilities(tau: float, model: dict, delta_max: float = 40.0, n_nodes: int = 240,
                           n_quad: int = 64, n_steps: int = 200) -> np.ndarray:
    '''phi(-i) and the two Gil-Pelaez probabilities.

    Parameters
    ----------
    tau : float
        Time to maturity, tau > 0.
    model : dict
        Model dictionary as in ``characteristic_function``.
    delta_max : float
        Truncation of the transform integrals, > 0.
    n_nodes : int
        Gauss-Legendre nodes on (0, delta_max), >= 2.
    n_quad : int
        Passed to the characteristic function.
    n_steps : int
        Passed to the characteristic function.

    Returns
    -------
    out : np.ndarray
        Shape (3,) float array [phi(-i), P1, P2].

    Raises
    ------
    ValueError
        If any argument is outside its domain.

    Notes
    -----
    Use ``characteristic_function``. Include every import your implementation
    needs inside the function body.
    '''
    return np.zeros(3, dtype=float)  # placeholder
```

### Step 6

06_margrabe_price

Goal
----
Compute the Margrabe price of the option to exchange asset 2 for asset 1 when the log price ratio is Gaussian with constant variance rate.

```python
import numpy as np

def margrabe_price(s1: float, s2: float, variance_rate: float, tau: float) -> float:
    '''Margrabe value of the option to exchange asset 2 for asset 1.

    Parameters
    ----------
    s1 : float
        Price of asset 1, s1 > 0.
    s2 : float
        Price of asset 2, s2 > 0.
    variance_rate : float
        Constant variance rate of ln(S1/S2), > 0.
    tau : float
        Time to maturity, tau > 0.

    Returns
    -------
    value : float
        Exchange value, as a native Python float.

    Raises
    ------
    ValueError
        If any argument is not a finite positive number.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    '''
    return 0.0  # placeholder
```

### Step 7

07_price_comparisons

Goal
----
Price the exchange option under the full model and under three reduced versions that isolate its channels.

```python
import numpy as np

def price_comparisons(tau: float, model: dict, delta_max: float = 40.0, n_nodes: int = 240,
                      n_quad: int = 64, n_steps: int = 200) -> np.ndarray:
    '''Exchange value under the full model and three reductions.

    Parameters
    ----------
    tau : float
        Time to maturity, tau > 0.
    model : dict
        Model dictionary as in ``characteristic_function``, with two regimes.
    delta_max, n_nodes, n_quad, n_steps
        Numerical settings passed to ``exercise_probabilities``.

    Returns
    -------
    values : np.ndarray
        Shape (4,) float array; see the module description.

    Raises
    ------
    ValueError
        If the model does not have exactly two regimes or any argument is
        outside its domain.

    Notes
    -----
    Use ``exercise_probabilities``. Include every import your implementation
    needs inside the function body.
    '''
    return np.zeros(4, dtype=float)  # placeholder
```

### Step 8

08_exchange_option_price

Goal
----
Chain the sub-problem functions 01-07 and return the time-0 value of the European option to exchange asset 2 for asset 1 under the regime-switching model with Heston variances and stochastic market liquidity. Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (variance_riccati, liquidity_coefficients, regime_factor, characteristic_function, exercise_probabilities, margrabe_price, price_comparisons) rather than reimplementing them.

```python
import numpy as np

def exchange_option_price(model: dict = None, tau: float = 1.0, delta_max: float = 40.0,
                          n_nodes: int = 240, n_quad: int = 64, n_steps: int = 200) -> float:
    '''Time-0 value of the exchange option under the full model.

    Parameters
    ----------
    model : dict or None
        Model dictionary as in ``characteristic_function`` (two regimes). None
        selects the parameter set of the problem.
    tau : float
        Time to maturity, tau > 0.
    delta_max, n_nodes, n_quad, n_steps
        Numerical settings passed to the transform pricing.

    Returns
    -------
    value : float
        The exchange option value, as a native Python float.

    Raises
    ------
    ValueError
        If any argument is outside its domain or a consistency check fails.

    Notes
    -----
    This is the final, orchestrating step: call the public functions of
    sub-problems 01-07 and feed each returned value into the next, rather
    than reimplementing them. Include every import your implementation needs
    inside the function body.
    '''
    return 0.0  # placeholder
```
