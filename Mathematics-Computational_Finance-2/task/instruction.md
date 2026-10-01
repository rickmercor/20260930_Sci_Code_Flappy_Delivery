# Mathematics-Computational_Finance-2

## Background

Portfolio decisions are re-solved as new return observations arrive, and in practice only finitely many historical observations are available. Sample average approximation replaces the true return distribution with the empirical one; although asymptotically consistent, its finite-sample optimiser amplifies estimation error and performs poorly out of sample, a longstanding concern in the portfolio literature. Distributionally robust optimisation addresses this by optimising against the least favourable distribution in a data-centred ambiguity set.

An order-one Wasserstein ball is an attractive choice of ambiguity set because its transport geometry allows probability mass to move from each observation to arbitrary points of the return support, rather than merely reweighting the observed scenarios, and it supports finite-sample guarantees and asymptotic consistency. That same flexibility creates the computational difficulty: for expected-utility maximisation with a smooth concave utility, the standard Wasserstein duality argument produces a convex programme whose constraints are indexed by the entire support, so it is semi-infinite rather than finite.

Replacing the utility by a piecewise-affine surrogate built from its tangent lines is one route to a finite programme whose size stays manageable as the number of assets grows. A concave utility never lies above its tangent lines, so the surrogate problem judges portfolios by a more optimistic criterion than the utility the investor actually holds, and the portfolio it selects must be assessed separately under that utility. How much is lost in that assessment is the practical price of the approximation. The setting is a single-period allocation with logarithmic growth utility, a common choice in sequential investment problems, at a portfolio dimension of practical size.

## Problem

Data-driven portfolio selection that replaces the unknown return distribution with the empirical distribution of a short sample is known to amplify estimation error, and one standard remedy is to optimise instead against the least favourable distribution in an order-one Wasserstein ball centred at that empirical distribution. For a smooth concave utility the resulting worst-case problem has constraints ranging over every point of the return support and is therefore not finite as written; a recent line of work obtains a finite surrogate by replacing the utility with a piecewise-affine majorant before taking the worst case. Solving the surrogate yields both a value and a portfolio, and the portfolio can then be assessed under the original utility. Your task is to quantify the discrepancy between the two for one concrete deterministic instance, and to report a single number.

The instance uses $n = 1000$ risky assets and $N = 25$ observed return vectors, with the following configuration.

- Return support: the box $\mathcal{X} = \{x \in \mathbb{R}^{1000} : x_{\min} \le x \le x_{\max}\}$ with $(x_{\min})_i = -\big(150 + (37i \bmod 121)\big)/1000$ and $(x_{\max})_i = \big(120 + (53i \bmod 101)\big)/1000$ for $i = 0, 1, \dots, 999$.
- Admissible portfolios: $\mathcal{W} = \{w \in \mathbb{R}^{1000} : w \ge 0,\ \sum_i w_i = 1,\ w_i \le 0.01\}$.
- Observed returns: $\widehat{X}_{j,i} = \big(\big((7j + 13i + 5ji) \bmod 181\big) - 90\big)\big/1000$ for $j = 0, 1, \dots, 24$ and $i = 0, 1, \dots, 999$, where $\widehat{X}_{j,\cdot}$ is the $j$-th observation and the empirical distribution places mass $1/N$ on each.
- Wasserstein radius: $\varepsilon = 10^{-2}$, with the transport cost measured in the $\ell_1$ norm on $\mathbb{R}^{1000}$.

The utility is the logarithmic growth rate of wealth $U(y) = \log(1 + y)$, with $\log$ the natural logarithm, evaluated at the portfolio return $y = \langle w, x \rangle$. Let $[\underline{y}, \overline{y}]$ denote the interval of values $\langle w, x \rangle$ attainable as $w$ ranges over $\mathcal{W}$ and $x$ ranges over $\mathcal{X}$. The surrogate utility is the pointwise minimum of the tangent lines to $U$ at nine points spaced uniformly across $[\underline{y}, \overline{y}]$, both endpoints included. Every distribution in the ambiguity set is supported on $\mathcal{X}$.

Define the following two quantities.

- The surrogate value: the maximum over $\mathcal{W}$ of the worst-case expected surrogate utility taken over the ambiguity set. For this instance the maximiser is unique; call it $w^\star$.
- The delivered value: the worst-case expected utility of $w^\star$ over the same ambiguity set, with the utility taken to be $U$ itself rather than the surrogate.

Report the surrogate value minus the delivered value, to at least twelve significant figures.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

attainable_return_range

Goal
----
Return the problem statement's interval [y_lo, y_hi] for the return box [xmin, xmax] and per-asset cap `cap`.

```python
import numpy as np


def attainable_return_range(xmin: np.ndarray, xmax: np.ndarray,
                            cap: float) -> np.ndarray:
    '''Return [y_lo, y_hi].

    Parameters
    ----------
    xmin : np.ndarray
        Shape (n,). Lower bounds of the return box.
    xmax : np.ndarray
        Shape (n,). Upper bounds of the return box.
    cap : float
        Per-asset weight cap.

    Returns
    -------
    y_range : np.ndarray
        Shape (2,), dtype float.

    Raises
    ------
    ValueError
        If xmin and xmax are not finite one-dimensional arrays of a common
        length n >= 1 with -1 < xmin < xmax elementwise, or if cap is not a
        finite scalar with cap > 0 and n * cap >= 1.
    '''
    return y_range  # placeholder
```

### Step 2

tangent_coefficients

Goal
----
Return the surrogate utility's affine pieces for M tangent points placed on y_range as in the problem statement.

```python
import numpy as np


def tangent_coefficients(y_range: np.ndarray, M: int) -> np.ndarray:
    '''Return the affine pieces of the surrogate utility.

    Parameters
    ----------
    y_range : np.ndarray
        Shape (2,). Interval carrying the tangent points.
    M : int
        Number of tangent points.

    Returns
    -------
    coeffs : np.ndarray
        Shape (2, M), dtype float.

    Raises
    ------
    ValueError
        If y_range is not a finite array of shape (2,) with
        -1 < y_range[0] < y_range[1], or if M is not an integer with M >= 2.
    '''
    return coeffs  # placeholder
```

### Step 3

surrogate_worst_case

Goal
----
Return the worst-case expected surrogate utility of the portfolio `weights`.

```python
import numpy as np


def surrogate_worst_case(weights: np.ndarray, xmin: np.ndarray, xmax: np.ndarray,
                         returns: np.ndarray, coeffs: np.ndarray,
                         radius: float) -> float:
    '''Return the worst-case expected surrogate utility of a fixed portfolio.

    Parameters
    ----------
    weights : np.ndarray
        Shape (n,). Portfolio weights.
    xmin : np.ndarray
        Shape (n,). Lower bounds of the return box.
    xmax : np.ndarray
        Shape (n,). Upper bounds of the return box.
    returns : np.ndarray
        Shape (N, n). Observed returns, one per row.
    coeffs : np.ndarray
        Shape (2, M). Affine pieces of the surrogate utility.
    radius : float
        Wasserstein radius.

    Returns
    -------
    value : float
        To floating-point accuracy.

    Raises
    ------
    ValueError
        If xmin and xmax are not finite one-dimensional arrays of a common
        length n >= 1 with -1 < xmin < xmax elementwise; if returns is not a
        finite (N, n) array with N >= 1 and every row inside the box; if
        weights is not a finite nonnegative (n,) array summing to one within
        1e-9; if coeffs is not a finite (2, M) array with M >= 1 and
        nonnegative slopes; or if radius is not a finite scalar > 0.
    '''
    return value  # placeholder
```

### Step 4

surrogate_portfolio

Goal
----
Return the admissible portfolio with per-asset cap `cap` that maximises the value returned by surrogate_worst_case.

```python
import numpy as np


def surrogate_portfolio(xmin: np.ndarray, xmax: np.ndarray, returns: np.ndarray,
                        coeffs: np.ndarray, cap: float, radius: float) -> np.ndarray:
    '''Return the surrogate-optimal admissible portfolio.

    Parameters
    ----------
    xmin : np.ndarray
        Shape (n,). Lower bounds of the return box.
    xmax : np.ndarray
        Shape (n,). Upper bounds of the return box.
    returns : np.ndarray
        Shape (N, n). Observed returns, one per row.
    coeffs : np.ndarray
        Shape (2, M). Affine pieces of the surrogate utility.
    cap : float
        Per-asset weight cap.
    radius : float
        Wasserstein radius.

    Returns
    -------
    weights : np.ndarray
        Shape (n,), dtype float, to floating-point accuracy.

    Raises
    ------
    ValueError
        If xmin and xmax are not finite one-dimensional arrays of a common
        length n >= 1 with -1 < xmin < xmax elementwise; if returns is not a
        finite (N, n) array with N >= 1 and every row inside the box; if
        coeffs is not a finite (2, M) array with M >= 1 and nonnegative
        slopes; if cap is not a finite scalar with cap > 0 and n * cap >= 1;
        or if radius is not a finite scalar > 0.
    '''
    return weights  # placeholder
```

### Step 5

inner_worst_case

Goal
----
Return each row's inner minimisation value in the strong dual of the worst-case expected utility of `weights`, at transport multiplier `lam`.

```python
import numpy as np


def inner_worst_case(weights: np.ndarray, xmin: np.ndarray, xmax: np.ndarray,
                     returns: np.ndarray, lam: float) -> np.ndarray:
    '''Return the per-observation inner values of the strong dual.

    Parameters
    ----------
    weights : np.ndarray
        Shape (n,). Portfolio weights.
    xmin : np.ndarray
        Shape (n,). Lower bounds of the return box.
    xmax : np.ndarray
        Shape (n,). Upper bounds of the return box.
    returns : np.ndarray
        Shape (N, n). Observed returns, one per row.
    lam : float
        Transport multiplier.

    Returns
    -------
    values : np.ndarray
        Shape (N,), dtype float, in row order, to floating-point accuracy.

    Raises
    ------
    ValueError
        If lam is not a finite scalar >= 0; if xmin and xmax are not finite
        one-dimensional arrays of a common length n >= 1 with
        -1 < xmin < xmax elementwise; if returns is not a finite (N, n) array
        with N >= 1 and every row inside the box; or if weights is not a
        finite nonnegative (n,) array summing to one within 1e-9.
    '''
    return values  # placeholder
```

### Step 6

delivered_value

Goal
----
Return the worst-case expected utility of the portfolio `weights`.

```python
import numpy as np


def delivered_value(weights: np.ndarray, xmin: np.ndarray, xmax: np.ndarray,
                    returns: np.ndarray, radius: float) -> float:
    '''Return the worst-case expected utility of a fixed portfolio.

    Parameters
    ----------
    weights : np.ndarray
        Shape (n,). Portfolio weights.
    xmin : np.ndarray
        Shape (n,). Lower bounds of the return box.
    xmax : np.ndarray
        Shape (n,). Upper bounds of the return box.
    returns : np.ndarray
        Shape (N, n). Observed returns, one per row.
    radius : float
        Wasserstein radius.

    Returns
    -------
    value : float
        To floating-point accuracy.

    Raises
    ------
    ValueError
        If radius is not a finite scalar > 0; if xmin and xmax are not finite
        one-dimensional arrays of a common length n >= 1 with
        -1 < xmin < xmax elementwise; if returns is not a finite (N, n) array
        with N >= 1 and every row inside the box; or if weights is not a
        finite nonnegative (n,) array summing to one within 1e-9.
    '''
    return value  # placeholder
```

### Step 7

surrogate_shortfall

Goal
----
Return the surrogate value minus the delivered value.

```python
import numpy as np


def surrogate_shortfall(xmin: np.ndarray, xmax: np.ndarray, returns: np.ndarray,
                        cap: float, radius: float, M: int) -> float:
    '''Return the surrogate value minus the delivered value.

    Parameters
    ----------
    xmin : np.ndarray
        Shape (n,). Lower bounds of the return box.
    xmax : np.ndarray
        Shape (n,). Upper bounds of the return box.
    returns : np.ndarray
        Shape (N, n). Observed returns, one per row.
    cap : float
        Per-asset weight cap.
    radius : float
        Wasserstein radius.
    M : int
        Number of tangent points.

    Returns
    -------
    shortfall : float
        To floating-point accuracy.

    Raises
    ------
    ValueError
        If xmin and xmax are not finite one-dimensional arrays of a common
        length n >= 1 with -1 < xmin < xmax elementwise; if cap is not a
        finite scalar with cap > 0 and n * cap >= 1; if M is not an integer
        with M >= 2; if returns is not a finite (N, n) array with N >= 1 and
        every row inside the box; or if radius is not a finite scalar > 0.
    '''
    return shortfall  # placeholder
```
