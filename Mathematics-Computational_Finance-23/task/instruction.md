# Mathematics-Computational_Finance-23

## Background

American and Bermudan derivatives are optimal-stopping problems. A feasible stopping policy supplies a primal Monte Carlo value that is typically interpreted as a lower estimate, while a martingale-dual representation supplies an upper estimate when an admissible martingale is inserted into a pathwise maximum. Using both sides gives a useful numerical bracket on an option value.

Regression-based stopping methods approximate conditional continuation values from cross sections of simulated state paths. In multidimensional problems, symmetry-aware state representations can improve regression signal quality because economically equivalent states need not be presented to the basis in different coordinate orders. The basis dimension nevertheless controls a bias-variance tradeoff, so enrichment can alter both the stopping policy and any dual construction built from it.

Efficient dual Monte Carlo methods attempt to approximate the Doob martingale without nested conditional simulations. Adjacent-time least-squares projections provide one route: fitted values and projection residuals can be combined into pathwise martingale increments. Different projection orderings and state representations can produce distinct finite-sample upper estimates even when they target the same Snell-envelope structure.

## Problem

A regression-based primal-dual Monte Carlo method can bracket Bermudan option values without nested simulation by extracting approximate Doob martingales from backward primal payoff paths. For a two-asset Bermudan min-put, quantify how the primal-dual geometry changes as a compact common polynomial basis is enriched, while respecting the source's distinct single-projection $\alpha$, double-projection $\beta$, and pathwise-sorted-state treatments.

Use $S_0=(100,100)$, strike $K=100$, continuously compounded $r=0.06$, common dividend yield $q=0$, common volatility $\sigma=0.30$, instantaneous Brownian correlation $\rho=0.25$, maturity $T=0.75$, eight equally spaced exercise intervals after initiation, and two independent antithetic Monte Carlo sets with $N=512$ paths each. Generate each set with `numpy.random.default_rng(seed).standard_normal((8,256,2))`; transform the second Gaussian component at every step to $\rho Z_1+\sqrt{1-\rho^2}Z_2$, concatenate these 256 correlated shock paths with their sign reversals along the path axis, and use seed 17 for policy fitting and seed 41 for evaluation and dual construction. Fit the backward primal policy using all paths, replay that fixed policy on the independent evaluation set, and do not permit exercise at initiation. At every later date take exercise only when the immediate min-put payoff is strictly positive and at least the fitted continuation value; otherwise carry the one-step-discounted realized payoff back.

Before invoking any source-specific projection formula, explicitly report in the reasoning the source-independent benchmark checks $\Delta t=T/8$, the one-step discount factor $D=e^{-r\Delta t}$, the Gaussian correlation loading $c_\rho=\sqrt{1-\rho^2}$, and the total-degree basis dimensions $m_p=(p+1)(p+2)/2$ for $p=1,2,3,4$.

For polynomial degree $p\in(1,2,3,4)$, use all normalized total-degree monomials $z_1^{a}z_2^{b}$ with $a+b\le p$, where $z_i=S_i/K-1$; order columns by increasing total degree and, within each degree, decreasing $a$. On the unsorted state, construct both source martingales exactly from the adjacent-time orthogonal projections in the source's single-projection and double-projection sections and Monte Carlo implementation (Eqs. 28--30, 35--37, and 40--46), using the same evaluation paths to fit dual coefficients and accumulate each martingale. For the source's min-put sorting treatment, order the two asset prices pathwise at every date as $S^{(1)}=\max(S_1,S_2),\;S^{(2)}=\min(S_1,S_2)$ before basis evaluation, refit the primal policy, and form the sorted $\alpha$ branch only; retain the $\beta$ branch on the unsorted state. Write $b_p=(L_{u,p},U^{\alpha}_{u,p},U^{\beta}_{u,p},L_{s,p},U^{\alpha}_{s,p})$ and
$$g_p=\left(\frac{U^{\alpha}_{u,p}-L_{u,p}}{L_{u,p}},\;\frac{U^{\beta}_{u,p}-L_{u,p}}{L_{u,p}},\;\frac{U^{\alpha}_{s,p}-L_{s,p}}{L_{s,p}},\;\frac{L_{s,p}-L_{u,p}}{L_{u,p}}\right).$$

Treat $(g_1,g_2,g_3,g_4)$ as a path in $\mathbb R^4$ and compute $\mathcal L=\|g_2-g_1\|_2+\|g_3-g_2\|_2+\|g_4-g_3\|_2$, using the source's one-step discounting, time-zero projection, martingale accumulation, and pathwise dual maximum conventions without intermediate rounding. Report $\mathcal L$ rounded to exactly 12 digits after the decimal point. Output `<final_answer>` first, followed by `<reasoning>`; the final-answer tag must contain exactly one finite decimal value, while the reasoning should identify the paper-specific projection and sorting logic; report the first evaluation path state after one step together with its antithetic partner, the sorted degree-2 basis row for that state, the Euclidean norm of the unsorted degree-4 primal coefficient row at the fourth exercise date, and the terminal Euclidean norms of the degree-4 unsorted $\alpha$ and $\beta$ martingales; list all four $b_p$ and $g_p$ vectors; and state the three Euclidean path increments.

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

01_simulate_correlated_antithetic_gbm_paths

Goal
----
Generate deterministic two-asset risk-neutral GBM paths with correlation and antithetic pairing.

```python
def simulate_correlated_antithetic_gbm_paths(s0: "np.ndarray", r: float, q: float, sigma: float, rho: float, maturity: float, steps: int, n_paths: int, seed: int) -> "np.ndarray":
    """Return correlated two-asset GBM paths with antithetic shocks.

    Parameters
    ----------
    s0 : np.ndarray
        Length-2 initial asset-price vector.
    r : float
        Continuously compounded risk-free rate.
    q : float
        Common continuously compounded dividend yield.
    sigma : float
        Common annualized volatility.
    rho : float
        Instantaneous Brownian correlation.
    maturity : float
        Option maturity.
    steps : int
        Number of equal time intervals.
    n_paths : int
        Even total number of paths. The first half uses the generated correlated
        shocks and the second half uses their sign reversals.
    seed : int
        Seed passed to ``numpy.random.default_rng``.

    Returns
    -------
    paths : np.ndarray
        Array of shape ``(steps + 1, n_paths, 2)``.
    """
    return paths
```

### Step 2

02_min_put_polynomial_basis

Goal
----
Evaluate the nested two-asset normalized polynomial basis, optionally after pathwise sorting.

```python
def min_put_polynomial_basis(states: "np.ndarray", strike: float, degree: int, sort_state: bool = False) -> "np.ndarray":
    """Return normalized total-degree monomials for two-asset states.

    Parameters
    ----------
    states : np.ndarray
        Array of shape ``(n, 2)`` containing asset prices.
    strike : float
        Strike used to normalize moneyness as ``z_i = S_i / strike - 1``.
    degree : int
        Maximum total polynomial degree.
    sort_state : bool
        If True, sort each two-asset state in descending price order (larger
        price first) before evaluating the basis.

    Returns
    -------
    Phi : np.ndarray
        Matrix with ``(degree + 1)(degree + 2)/2`` columns. Columns are ordered
        by increasing total degree and, within a total degree, decreasing
        exponent of the first state variable.
    """
    return Phi
```

### Step 3

03_fit_min_put_primal_policy

Goal
----
Fit the backward primal stopping policy on the independent training sample.

```python
def fit_min_put_primal_policy(paths: "np.ndarray", strike: float, r: float, dt: float, degree: int, sort_state: bool = False) -> "np.ndarray":
    """Fit backward continuation coefficients on training paths.

    Parameters
    ----------
    paths : np.ndarray
        Training paths of shape ``(steps + 1, n_paths, 2)``.
    strike : float
        Min-put strike.
    r : float
        Continuously compounded risk-free rate.
    dt : float
        Time increment between exercise dates.
    degree : int
        Maximum total polynomial degree.
    sort_state : bool
        Whether to sort the two asset prices pathwise before basis evaluation.

    Returns
    -------
    theta : np.ndarray
        Coefficient array of shape ``(steps + 1, p)`` with the fitted rows for
        dates 1 through ``steps - 1`` and zero rows at initiation and maturity.
    """
    return theta
```

### Step 4

04_evaluate_min_put_payoff_process

Goal
----
Replay the fixed policy on independent evaluation paths and recover the full realized-payoff process.

```python
def evaluate_min_put_payoff_process(paths: "np.ndarray", theta: "np.ndarray", strike: float, r: float, dt: float, degree: int, sort_state: bool = False) -> "np.ndarray":
    """Evaluate a fixed min-put stopping policy on new paths.

    Parameters
    ----------
    paths : np.ndarray
        Evaluation paths of shape ``(steps + 1, n_paths, 2)``.
    theta : np.ndarray
        Backward continuation coefficients from the training sample.
    strike : float
        Min-put strike.
    r : float
        Continuously compounded risk-free rate.
    dt : float
        Time increment between exercise dates.
    degree : int
        Maximum total polynomial degree used by ``theta``.
    sort_state : bool
        Whether the fixed policy evaluates the sorted-state basis.

    Returns
    -------
    H_D : np.ndarray
        Array of shape ``(steps + 1, n_paths)``. Rows 1 through maturity are
        the source-style realized payoff process in each row's time units; row
        0 is the one-period-discounted row-1 payoff used for the time-zero
        lower estimate.
    """
    return H_D
```

### Step 5

05_single_projection_alpha_martingale

Goal
----
Construct the paper-specific single-projection alpha martingale on unsorted or sorted states.

```python
def single_projection_alpha_martingale(paths: "np.ndarray", H_D: "np.ndarray", strike: float, r: float, dt: float, degree: int, sort_state: bool = False) -> "np.ndarray":
    """Return the source's alpha-algorithm martingale process.

    Parameters
    ----------
    paths : np.ndarray
        Evaluation paths of shape ``(steps + 1, n_paths, 2)``.
    H_D : np.ndarray
        Realized-payoff process from the same evaluation paths.
    strike : float
        Min-put strike used by the common basis.
    r : float
        Continuously compounded risk-free rate.
    dt : float
        Time increment between exercise dates.
    degree : int
        Maximum total polynomial degree.
    sort_state : bool
        Whether both adjacent-time basis evaluations use the pathwise-sorted state.

    Returns
    -------
    M_alpha : np.ndarray
        Martingale array of shape ``(steps + 1, n_paths)`` with a zero initial row.
    """
    return M_alpha
```

### Step 6

06_double_projection_beta_martingale

Goal
----
Construct the paper-specific unsorted double-projection beta martingale.

```python
def double_projection_beta_martingale(paths: "np.ndarray", H_D: "np.ndarray", strike: float, r: float, dt: float, degree: int) -> "np.ndarray":
    """Return the source's unsorted beta-algorithm martingale process.

    Parameters
    ----------
    paths : np.ndarray
        Unsorted evaluation paths of shape ``(steps + 1, n_paths, 2)``.
    H_D : np.ndarray
        Realized-payoff process from the same unsorted evaluation paths.
    strike : float
        Min-put strike used by the common basis.
    r : float
        Continuously compounded risk-free rate.
    dt : float
        Time increment between exercise dates.
    degree : int
        Maximum total polynomial degree.

    Returns
    -------
    M_beta : np.ndarray
        Martingale array of shape ``(steps + 1, n_paths)`` with a zero initial row.
    """
    return M_beta
```

### Step 7

07_unsorted_primal_dual_bounds

Goal
----
Compute the unsorted primal lower estimate and both paper-specific dual upper estimates for one basis degree.

```python
def unsorted_primal_dual_bounds(train_paths: "np.ndarray", eval_paths: "np.ndarray", strike: float, r: float, dt: float, degree: int) -> "np.ndarray":
    """Return unsorted primal, alpha-dual, and beta-dual prices.

    Parameters
    ----------
    train_paths : np.ndarray
        Independent policy-training paths.
    eval_paths : np.ndarray
        Evaluation paths used for the fixed policy and both dual projections.
    strike : float
        Min-put strike.
    r : float
        Continuously compounded risk-free rate.
    dt : float
        Time increment between exercise dates.
    degree : int
        Maximum total polynomial degree.

    Returns
    -------
    bounds : np.ndarray
        Length-3 vector ``[lower, alpha_upper, beta_upper]``.
    """
    return bounds
```

### Step 8

08_sorting_aware_bound_vector

Goal
----
Extend the unsorted bound vector with the source-approved sorted primal and alpha bounds.

```python
def sorting_aware_bound_vector(train_paths: "np.ndarray", eval_paths: "np.ndarray", strike: float, r: float, dt: float, degree: int) -> "np.ndarray":
    """Return unsorted bounds together with the source-approved sorted branch.

    Parameters
    ----------
    train_paths : np.ndarray
        Independent policy-training paths.
    eval_paths : np.ndarray
        Evaluation paths used for policy evaluation and dual fitting.
    strike : float
        Min-put strike.
    r : float
        Continuously compounded risk-free rate.
    dt : float
        Time increment between exercise dates.
    degree : int
        Maximum total polynomial degree.

    Returns
    -------
    bounds : np.ndarray
        Length-5 vector ``[L_u, U_alpha_u, U_beta_u, L_s, U_alpha_s]``.
    """
    return bounds
```

### Step 9

09_cumulative_sorting_basis_path

Goal
----
Accumulate the normalized sorting/projection gap path across polynomial basis enrichment.

```python
def cumulative_sorting_basis_path(s0: "np.ndarray", strike: float, r: float, q: float, sigma: float, rho: float, maturity: float, steps: int, n_paths: int, train_seed: int, eval_seed: int, degrees: tuple) -> float:
    """Return the cumulative normalized sorting/projection path length.

    Parameters
    ----------
    s0 : np.ndarray
        Length-2 initial asset-price vector.
    strike : float
        Min-put strike.
    r : float
        Continuously compounded risk-free rate.
    q : float
        Common dividend yield.
    sigma : float
        Common annualized volatility.
    rho : float
        Instantaneous Brownian correlation.
    maturity : float
        Option maturity.
    steps : int
        Number of equally spaced exercise intervals after initiation.
    n_paths : int
        Even path count for each of the independent training and evaluation sets.
    train_seed : int
        RNG seed for policy training paths.
    eval_seed : int
        RNG seed for evaluation and dual paths.
    degrees : tuple
        Increasing sequence of polynomial degrees with at least two entries.

    Returns
    -------
    path_length : float
        Sum of Euclidean distances between consecutive four-component normalized
        gap vectors built from ``[L_u,U_alpha_u,U_beta_u,L_s,U_alpha_s]``.
    """
    return path_length
```
