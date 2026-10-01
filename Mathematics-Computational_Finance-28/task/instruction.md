# Mathematics-Computational_Finance-28

## Background

An American option can be exercised at any time before maturity, so its arbitrage-free value is the value of an optimal stopping problem: the Snell envelope of the discounted gain process, the smallest supermartingale that dominates it. In practice this value is computed backwards in time, by lattices, finite differences or least-squares Monte Carlo, and each of these methods treats the option at one maturity at a time.

In interest-rate modelling, Heath, Jarrow and Morton replaced the modelling of a single state variable by the modelling of an entire forward curve, with absence of arbitrage imposed as a restriction on the drift of that curve. Extensions of this idea to equity options have mostly targeted European objects such as implied-volatility or forward-variance surfaces. A recent line of work instead applies the forward-modelling view directly to the gain process of an American claim. The option value is written as the current gain plus the accumulated forward drift of the gain, and the continuation region of the stopping problem enters through that forward drift, so the early-exercise premium is embedded in the representation itself. A companion multiplicative form describes the whole maturity curve of option prices in the manner of a bond term structure.

For puts and calls the gain has a kink at the strike, so its dynamics contain a local-time term from the Tanaka-Meyer formula. That term is a singular measure concentrated where the price crosses the strike and cannot be evaluated pointwise. Any numerical use of the representation therefore needs a regularisation of the local time, and the resulting value inherits a bias from it that is largest near the money, where the price spends the most time close to the strike.

## Problem

American option values are usually computed by backward recursion, but a recent forward-modelling approach in the spirit of Heath, Jarrow and Morton instead writes the value as the current gain plus an integral over maturities of a forward drift of the gain process, and this forward drift carries the continuation region of the optimal stopping problem inside it. For a put the gain has a kink at the strike, so its drift contains a local-time term, and to make the representation computable that term is regularised with a Gaussian kernel at the strike. Your task is to evaluate that regularised representation exactly for one contract and report a single number.

Under the pricing measure the underlying follows dS = r S dt + b S dW with S_0 = 100, r = 0.05 and b = 0.20. The claim is an American put with strike K = 100 and maturity T = 1 year, with gain G_t = (K - S_t)^+ and exercise at the put's optimal stopping time. Regularise the local time at the strike exactly as that approach does, with a Gaussian kernel of standard deviation eps = 0.5 in price units.

Evaluate the representation, meaning the current gain plus the forward drifts integrated over maturities up to the optimal exercise time, as an exact expectation under this model rather than as a Monte Carlo estimate on a time grid. Report its value at time 0 converged to within 1e-3.

In your reasoning, give numerically the part of the value contributed by the local-time term, the value obtained when that term is dropped, the American put price itself, and the value of the same representation when exercise is possible only at maturity. Then explain the sign and size of the gap between the representation and the American price.

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

01_tanaka_drift_rate

Goal
----
Evaluate, on a grid of asset prices, the two parts of the drift rate mu - r G of the American put's gain process, with the local time at the strike regularised by a Gaussian kernel.

```python
import numpy as np

def tanaka_drift_rate(s_grid: np.ndarray, strike: float, rate: float, vol: float, eps: float) -> np.ndarray:
    '''Drift rate mu - r G of the put gain, split into its two parts.

    Parameters
    ----------
    s_grid : np.ndarray
        One-dimensional array of n >= 1 finite, non-negative asset prices.
    strike : float
        Strike K > 0.
    rate : float
        Risk-free rate r (finite).
    vol : float
        Volatility b >= 0.
    eps : float
        Standard deviation of the Gaussian kernel, in price units, eps > 0.

    Returns
    -------
    parts : np.ndarray
        Shape (2, n) float array: row 0 the part without local time, row 1
        the regularised local-time part.

    Raises
    ------
    ValueError
        If s_grid is not a finite, non-negative one-dimensional array with at
        least one entry, or if strike, rate, vol or eps is outside its domain.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs inside the function body.
    '''
    return np.zeros((2, np.atleast_1d(s_grid).size), dtype=float)  # placeholder
```

### Step 2

02_bs_operator_bands

Goal
----
Return the three-band coefficients of the discounted Black-Scholes generator on a uniform asset grid.

```python
import numpy as np

def bs_operator_bands(s_grid: np.ndarray, rate: float, vol: float) -> np.ndarray:
    '''Tridiagonal coefficients of the discounted Black-Scholes generator.

    Parameters
    ----------
    s_grid : np.ndarray
        Strictly increasing, uniformly spaced, non-negative grid of n >= 3 prices.
    rate : float
        Risk-free rate r (finite).
    vol : float
        Volatility b >= 0.

    Returns
    -------
    bands : np.ndarray
        Shape (3, n) float array: [0] sub-diagonal, [1] diagonal, [2]
        super-diagonal coefficient of each row.

    Raises
    ------
    ValueError
        If the grid is not finite, non-negative, strictly increasing and
        uniform (relative spacing tolerance 1e-9) with at least three nodes,
        or if rate or vol is outside its domain.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    '''
    return np.zeros((3, np.asarray(s_grid).size), dtype=float)  # placeholder
```

### Step 3

03_implicit_step

Goal
----
Take one fully implicit time step of a linear system defined by three-band generator coefficients, with prescribed values at both end nodes.

```python
import numpy as np

def implicit_step(bands: np.ndarray, dt: float, rhs: np.ndarray, left_value: float,
                  right_value: float) -> np.ndarray:
    '''One backward Euler step with Dirichlet end values.

    Parameters
    ----------
    bands : np.ndarray
        Shape (3, n) generator coefficients (sub-diagonal, diagonal,
        super-diagonal per row), n >= 3.
    dt : float
        Step length, dt >= 0.
    rhs : np.ndarray
        Shape (n,) right-hand side.
    left_value : float
        Value imposed on the first entry.
    right_value : float
        Value imposed on the last entry.

    Returns
    -------
    x : np.ndarray
        Shape (n,) float solution.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, n < 3, any input is not finite, or
        dt is negative.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    '''
    return np.zeros(np.asarray(rhs).size, dtype=float)  # placeholder
```

### Step 4

04_american_put_exercise

Goal
----
The put value is marched backward from its payoff at maturity with fully implicit steps of the discounted Black-Scholes generator, the first node carrying the value K and the last node the value 0 at every level. After each implicit step the early-exercise constraint is imposed by projection onto the payoff: the value at every node becomes the larger of the implicit continuation value and the payoff there. A node belongs to the exercise region of that level when the payoff there is strictly positive and strictly above the implicit continuation value. These regions define the stopping rule of the additive representation: a path is stopped the first time it stands on an exercise node.

```python
import numpy as np

def american_put_exercise(s_grid: np.ndarray, strike: float, rate: float, vol: float,
                          maturity: float, n_steps: int) -> np.ndarray:
    '''American put value at time 0 and the exercise indicators of every level.

    Parameters
    ----------
    s_grid : np.ndarray
        Uniform grid of n >= 3 prices starting at 0.
    strike : float
        Strike K > 0.
    rate : float
        Risk-free rate r (finite).
    vol : float
        Volatility b >= 0.
    maturity : float
        Maturity T > 0.
    n_steps : int
        Number of uniform time steps, n_steps >= 1.

    Returns
    -------
    out : np.ndarray
        Shape (n_steps + 1, n) float array; see the module description.

    Raises
    ------
    ValueError
        If the grid does not start at 0 or is not uniform, or if any
        argument is outside its domain.

    Notes
    -----
    Build the steps from ``bs_operator_bands`` and ``implicit_step``. Include
    every import your implementation needs inside the function body.
    '''
    return np.zeros((int(n_steps) + 1, np.asarray(s_grid).size), dtype=float)  # placeholder
```

### Step 5

05_stopped_drift_integral

Goal
----
Compute, on the asset grid at time 0, the expected discounted integral of given drift-rate functions accumulated until the stopping time defined by the exercise regions, or until maturity.

```python
import numpy as np

def stopped_drift_integral(s_grid: np.ndarray, sources: np.ndarray, exercise: np.ndarray,
                           rate: float, vol: float, maturity: float) -> np.ndarray:
    '''Expected discounted drift integral up to the stopping time, at time 0.

    Parameters
    ----------
    s_grid : np.ndarray
        Uniform grid of n >= 3 prices.
    sources : np.ndarray
        Shape (k, n) drift-rate values on the grid, k >= 1 (a 1-D array is
        treated as a single row).
    exercise : np.ndarray
        Shape (n_steps, n) exercise indicators of the levels t_0, ...,
        t_{n_steps - 1}; entries are 0 or 1.
    rate : float
        Risk-free rate r (finite).
    vol : float
        Volatility b >= 0.
    maturity : float
        Maturity T > 0; the step is maturity / n_steps.

    Returns
    -------
    values : np.ndarray
        Shape (k, n) float array.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, the indicators are not 0 or 1, any
        input is not finite, or any parameter is outside its domain.

    Notes
    -----
    Build the steps from ``bs_operator_bands`` and ``implicit_step``. Include
    every import your implementation needs inside the function body.
    '''
    return np.zeros(np.atleast_2d(sources).shape, dtype=float)  # placeholder
```

### Step 6

06_european_additive_value

Goal
----
Evaluate the additive forward representation of the put without early exercise by direct quadrature, split into its parts.

```python
import numpy as np

def european_additive_value(spot: float, strike: float, rate: float, vol: float,
                            maturity: float, eps: float) -> np.ndarray:
    '''Additive representation of the European put by quadrature.

    Parameters
    ----------
    spot : float
        Current price S_0 > 0.
    strike : float
        Strike K > 0.
    rate : float
        Risk-free rate r (finite).
    vol : float
        Volatility b > 0.
    maturity : float
        Maturity T > 0.
    eps : float
        Standard deviation of the Gaussian kernel in price units,
        0 < eps < strike / 10.

    Returns
    -------
    parts : np.ndarray
        Shape (3,) float array: [value, integral without local time,
        integral of the local-time part]. The value equals (K - S_0)^+ plus
        the two integrals.

    Raises
    ------
    ValueError
        If any argument is outside its stated domain.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    '''
    return np.zeros(3, dtype=float)  # placeholder
```

### Step 7

07_additive_decomposition

Goal
----
Evaluate the additive forward representation of the American put at one spot price on a grid, together with its parts and the American price itself.

```python
import numpy as np

def additive_decomposition(spot: float, strike: float, rate: float, vol: float, maturity: float,
                           eps: float, n_intervals: int, n_steps: int,
                           smax_factor: float) -> np.ndarray:
    '''Additive representation of the American put and its parts.

    Parameters
    ----------
    spot : float
        Current price S_0, a grid node with 0 < S_0 < smax_factor * K.
    strike : float
        Strike K > 0.
    rate : float
        Risk-free rate r (finite).
    vol : float
        Volatility b > 0.
    maturity : float
        Maturity T > 0.
    eps : float
        Standard deviation of the Gaussian kernel in price units, eps > 0.
    n_intervals : int
        Number of grid intervals on [0, smax_factor * K], >= 2.
    n_steps : int
        Number of uniform time steps, >= 1.
    smax_factor : float
        Right end of the grid in units of the strike, > 1.

    Returns
    -------
    parts : np.ndarray
        Shape (4,) float array; see the module description.

    Raises
    ------
    ValueError
        If any argument is outside its domain or the spot is not a grid node.

    Notes
    -----
    Use ``tanaka_drift_rate``, ``american_put_exercise`` and
    ``stopped_drift_integral``. Include every import your implementation
    needs inside the function body.
    '''
    return np.zeros(4, dtype=float)  # placeholder
```

### Step 8

08_additive_american_value

Goal
----
Chain the sub-problem functions 01-07 and return the value of the regularised additive forward representation of the American put at the spot. Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (tanaka_drift_rate, bs_operator_bands, implicit_step, american_put_exercise, stopped_drift_integral, european_additive_value, additive_decomposition) rather than reimplementing them.

```python
import numpy as np

def additive_american_value(spot: float = 100.0, strike: float = 100.0, rate: float = 0.05,
                            vol: float = 0.2, maturity: float = 1.0, eps: float = 0.5,
                            n_intervals: int = 4000, n_steps: int = 4000,
                            smax_factor: float = 4.0) -> float:
    '''Regularised additive representation of the American put at the spot.

    Parameters
    ----------
    spot : float
        Current price S_0, a node of the fine grid, 0 < S_0 < smax_factor * K.
    strike : float
        Strike K > 0.
    rate : float
        Risk-free rate r (finite).
    vol : float
        Volatility b > 0.
    maturity : float
        Maturity T > 0.
    eps : float
        Standard deviation of the Gaussian kernel in price units,
        0 < eps < K / 10.
    n_intervals : int
        Number of intervals of the fine grid on [0, smax_factor * K].
    n_steps : int
        Number of uniform time steps of the fine computation.
    smax_factor : float
        Right end of the grid in units of the strike, > 1.

    Returns
    -------
    value : float
        The representation value, as a native Python float.

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
