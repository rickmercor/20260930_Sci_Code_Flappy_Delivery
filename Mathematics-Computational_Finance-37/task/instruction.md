# Mathematics-Computational_Finance-37

## Background

Joint calibration of equity-index and volatility-index option surfaces is a
long-standing puzzle in derivatives markets: the two markets impose
apparently conflicting constraints (a pronounced negative skew in equity
options versus comparatively flat implied volatilities in volatility-index
options), and classical single-factor stochastic-volatility models
generally cannot reconcile both simultaneously. Recent work approaches this
through martingale optimal transport, treating the joint law of the index
and its volatility across multiple maturities as an entropy-regularized
constrained optimization problem, solved numerically via first-order
mirror-descent-type iterative schemes on a finite-state discretization.

A separate and much older strand of derivatives pricing concerns default or
mortality intensity models, where a stochastic process drives the rate at
which an adverse event (default, death) occurs. When the intensity process
is not analytically tractable, survival probabilities and related pricing
integrals traditionally require computationally expensive PDE or Monte
Carlo methods. A path-integral approach, borrowed from statistical physics
and applied to this class of models, offers a semi-analytical alternative:
by classifying the space of paths according to a summary statistic (their
time-average) and constructing the best-fitting tractable (quadratic)
approximation within each class, the survival probability reduces to a
single well-behaved numerical integration rather than a full space-time
discretization.

Equity-linked life insurance combines both worlds: a policyholder's fund is
linked to an equity index (introducing market and volatility risk), a
death or maturity guarantee depends on a stochastic mortality process, and
an embedded surrender option introduces an optimal-stopping decision. The
combination of a non-recombining accumulated fund, multiple stochastic
risk factors, and Bermudan-style early exercise makes this a genuinely
multi-factor dynamic programming problem, and modern approaches represent
the fund dimension adaptively rather than through a brute-force grid, while
handling the surrender decision directly as a backward obstacle condition
layered on top of the fund's own recursive valuation.

## Problem

This task chains a market-calibrated equity-index model, a stochastic default-intensity survival model, and a surrenderable life-insurance contract valuation into a single deterministic scalar: the contract's fair initial value.

Stage 1 asks for a discretized joint law over an equity index and its volatility across three dates (S1, V1, S2, V2, S3), calibrated so that each marginal matches a prescribed target distribution while the conditional forward-martingale and log-contract-dispersion constraints between adjacent dates are satisfied as closely as a finite iterative reconciliation scheme allows. Build the reference law by interleaving lognormal transition kernels (mean zero, variance V_i times the accrual fraction) with independent draws of the volatility marginals, then design and run your own iterative reconciliation scheme toward the target marginals and conditional constraints. How you weigh exact marginal matching against only approximately satisfying the conditional constraints under a finite computational budget, and the exact clipping, step-size, and any penalty-growth choices, are yours to make; state them explicitly since they affect the final answer.

Stage 2 asks for the survival probability of a stochastic default/mortality intensity h_t = exp(X_t), where X follows dX_t = k(theta - X_t)dt + sigma dW_t, evaluated at two horizons T=1 and T=2 from a given initial state. Use the path-integral (Feynman-type) semi-analytical approximation for this class of intensity models: classify paths by their time-average, approximate the resulting reduced density with a self-consistent quadratic ansatz, solve the resulting nonlinear fixed-point system for the ansatz's own parameters, and integrate the resulting closed-form expression over the average-point variable to obtain the survival probability. The nonlinear solve's initial guess, iteration scheme, and the outer quadrature's node count and integration range are yours to choose and should be verified for stability.

Stage 3 asks for the fair initial value of a 3-year surrenderable equity-linked life-insurance contract. An initial contribution is credited to a fund that earns the equity index's return from Stage 1. At each of the first two policy anniversaries, if the policyholder is alive, they choose between surrendering the contract for a fraction of the current fund value, or paying a further contribution to continue and letting the fund run onward, taking whichever of these two options is worth more to them. At maturity, the contract pays the greater of the fund value and an accumulating guaranteed minimum. If the policyholder dies during any policy year, a death benefit equal to the greater of the fund value and that year's accumulated guaranteed minimum is paid at the following anniversary instead of whatever the surviving policyholder would have received, using the survival probabilities from Stage 2 to weight this outcome against survival at each of the first two anniversaries. Use the Stage 1 calibrated joint law directly as the fund's own discretized return process (its three post-inception index dates correspond exactly to the contract's three policy years), and use the Stage 2 survival probabilities, evaluated at both a one-year and a two-year horizon, to determine the probability of reaching, and of dying before, each of the first two anniversaries. Report the contract's fair initial value as a single deterministic real number.

Use these exact inputs. Stage 1: 5-point index grid log-spaced across [-0.35, 0.35] around an index level of 1, 3-point volatility grid [0.12, 0.22, 0.32], index-marginal target weights [0.06, 0.20, 0.48, 0.20, 0.06] (normalized), volatility-marginal target weights [0.30, 0.45, 0.25], accrual fraction per step 30/365. Stage 2: k=0.3, sigma=0.35, theta=ln(0.055), initial state x0=ln(0.045). Stage 3: initial contribution and each subsequent annual contribution D=100, guarantee accumulation rate 2% per year, surrender fraction of fund value 100%, zero interest and dividend rates.

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

reference_tensor

Goal
----
Build the reference (prior) joint law over (S1, V1, S2, V2, S3) by interleaving independent lognormal transition kernels.

```python
import numpy as np


def build_reference_tensor(S_grid: np.ndarray, V_grid: np.ndarray, tau: float) -> np.ndarray:
    """Build the reference (prior) joint law over (S1, V1, S2, V2, S3) by
    interleaving independent lognormal transition kernels.

    Parameters
    ----------
    S_grid : np.ndarray
        1D array of nS candidate index levels.
    V_grid : np.ndarray
        1D array of nV candidate volatility levels.
    tau : float
        Accrual fraction per step (e.g. 30/365).

    Returns
    -------
    pi : np.ndarray
        (nS, nV, nS, nV, nS) array, a valid probability tensor (nonnegative,
        sums to 1) over (S1, V1, S2, V2, S3). S1's law is uniform over the
        grid; V1, V2 are drawn independently and uniformly; S2 | S1, V1 and
        S3 | S2, V2 each follow a discretized lognormal kernel with mean
        zero in log-return and variance V_i^2 * tau, renormalized over the
        S_grid.

    Raises
    ------
    ValueError
        If S_grid or V_grid is not a 1D array of length >= 1, or tau <= 0.
    """
    nS = len(S_grid)
    pi = np.ones((nS, len(V_grid), nS, len(V_grid), nS))
    pi = pi / pi.sum()  # placeholder
    return pi
```

### Step 2

spx_vix_calibration

Goal
----
Run the augmented-Bregman mirror-descent SPX-VIX calibration (Algorithm 1) to reconcile a reference tensor toward the given target marginals and (approximately) the conditional martingale/dispersion constraints.

```python
import numpy as np


def spx_vix_calibration(S_grid: np.ndarray, V_grid: np.ndarray, tau: float,
                         muS: np.ndarray, muV: np.ndarray,
                         Kout: int = 6, Kin: int = 40) -> np.ndarray:
    """Run the augmented-Bregman mirror-descent SPX-VIX calibration
    (Algorithm 1) to reconcile a reference tensor toward the given target
    marginals and (approximately) the conditional martingale/dispersion
    constraints.

    Parameters
    ----------
    S_grid : np.ndarray
        1D array of nS candidate index levels.
    V_grid : np.ndarray
        1D array of nV candidate volatility levels.
    tau : float
        Accrual fraction per step.
    muS : np.ndarray
        1D array of length nS, the target S1 marginal (must sum to 1).
    muV : np.ndarray
        1D array of length nV, the shared target V1 and V2 marginal (must
        sum to 1).
    Kout : int
        Number of outer sweeps (penalty-growth steps).
    Kin : int
        Number of inner sweeps per outer step.

    Returns
    -------
    pi : np.ndarray
        (nS, nV, nS, nV, nS) calibrated probability tensor.

    Raises
    ------
    ValueError
        If muS or muV does not sum to 1 (within 1e-6), or if Kout < 1 or
        Kin < 1.
    """
    nS, nV = len(S_grid), len(V_grid)
    pi = np.ones((nS, nV, nS, nV, nS))
    pi = pi / pi.sum()  # placeholder
    return pi
```

### Step 3

marginalize_to_sss

Goal
----
Marginalize the calibrated (S1,V1,S2,V2,S3) joint law over V1 and V2 to obtain the (S1,S2,S3) joint law the downstream contract valuation step consumes.

```python
import numpy as np


def marginalize_to_sss(pi: np.ndarray) -> np.ndarray:
    """Marginalize the calibrated (S1,V1,S2,V2,S3) joint law over V1 and V2
    to obtain the joint law over (S1,S2,S3) alone.

    Parameters
    ----------
    pi : np.ndarray
        (nS, nV, nS, nV, nS) calibrated probability tensor over
        (S1, V1, S2, V2, S3).

    Returns
    -------
    joint123 : np.ndarray
        (nS, nS, nS) joint law over (S1, S2, S3), summing pi over its V1
        (axis 1) and V2 (axis 3) axes.

    Raises
    ------
    ValueError
        If pi is not a 5D array, or if its S1, S2, S3 axes (0, 2, 4) do not
        all share the same length.
    """
    nS = pi.shape[0] if pi.ndim >= 1 else 0
    joint123 = np.zeros((nS, nS, nS))  # placeholder
    return joint123
```

### Step 4

gtfk_selfconsistent

Goal
----
Solve for the GTFK trial-potential's self-consistent parameters (omega, delta_gamma) at a given path-average state xbar, for a Black-Karasinski intensity process with mean-reversion k, volatility sigma, long-run log-intensity theta, over horizon T.

```python
import numpy as np


def gtfk_selfconsistent(k: float, sigma: float, theta: float, T: float, xbar: float) -> tuple:
    """Solve for the GTFK trial-potential's self-consistent parameters
    (omega, delta_gamma) at a given path-average state xbar, for a
    Black-Karasinski intensity process with mean-reversion k, volatility
    sigma, long-run log-intensity theta, over horizon T.

    Parameters
    ----------
    k : float
        Mean-reversion strength (> 0).
    sigma : float
        Volatility (> 0).
    theta : float
        Long-run mean of the log-intensity process.
    T : float
        Horizon (> 0).
    xbar : float
        Path-average state at which the trial potential is centered.

    Returns
    -------
    omega : float
        The self-consistent effective frequency.
    delta_gamma : float
        The self-consistent mean-shift correction.

    Raises
    ------
    ValueError
        If k <= 0, sigma <= 0, or T <= 0.
    """
    omega = np.sqrt(k ** 2 + sigma ** 2 * np.exp(xbar))  # placeholder
    delta_gamma = 0.0  # placeholder
    return omega, delta_gamma
```

### Step 5

survival_probability

Goal
----
Compute the survival probability Q(x0, T) = E[exp(-integral_0^T exp(X_t) dt)] for a Black-Karasinski intensity process dX_t = k(theta - X_t)dt + sigma dW_t started at X_0 = x0, using the GTFK path-integral semi-analytical approximation.

```python
import numpy as np


def survival_probability(k: float, sigma: float, theta: float, x0: float, T: float) -> float:
    """Compute the survival probability Q(x0, T) = E[exp(-integral_0^T
    exp(X_t) dt)] for a Black-Karasinski intensity process dX_t =
    k(theta - X_t)dt + sigma dW_t started at X_0 = x0, using the GTFK
    path-integral semi-analytical approximation.

    Parameters
    ----------
    k : float
        Mean-reversion strength (> 0).
    sigma : float
        Volatility (> 0).
    theta : float
        Long-run mean of the log-intensity process.
    x0 : float
        Initial state.
    T : float
        Horizon (> 0).

    Returns
    -------
    Q : float
        The survival probability, a real number in (0, 1].

    Raises
    ------
    ValueError
        If k <= 0, sigma <= 0, or T <= 0.
    """
    return float(np.exp(-np.exp(x0) * T))  # placeholder (crude, not the GTFK method)
```

### Step 6

contract_value

Goal
----
Compute the fair initial value of the 3-year surrenderable equity-linked contract via backward obstacle recursion.

```python
import numpy as np


def contract_value(joint123: np.ndarray, S_grid: np.ndarray, D: float, gm: float,
                    alpha_s: float, Q1: float, Q2: float) -> float:
    """Compute the fair initial value of the 3-year surrenderable
    equity-linked contract via backward obstacle recursion.

    Parameters
    ----------
    joint123 : np.ndarray
        (nS, nS, nS) joint law over (S1, S2, S3), already marginalized over
        the volatility coordinates.
    S_grid : np.ndarray
        1D array of nS candidate index levels (index level 1.0 at
        inception).
    D : float
        Contribution credited at inception and at each continuation
        (> 0).
    gm : float
        Guarantee accumulation rate per year.
    alpha_s : float
        Surrender fraction of the current fund value, in (0, 1].
    Q1 : float
        Survival probability to anniversary 1 (in (0, 1]).
    Q2 : float
        Survival probability to anniversary 2 (in (0, 1], and <= Q1).

    Returns
    -------
    U0 : float
        The contract's fair initial value.

    Raises
    ------
    ValueError
        If Q1 or Q2 is not in (0, 1], if Q2 > Q1, or if D <= 0.
    """
    return 0.0  # placeholder
```

### Step 7

orchestrator_contract_value

Goal
----
Orchestrates the full four-stage chain: calibrate the SPX-VIX joint law (the equity-index process driving the fund), marginalize it over the volatility coordinates to obtain the (S1,S2,S3) law the contract actually uses, compute the survival probability at one and two years (the mortality process governing the death-benefit weighting), and combine both into the surrenderable contract's backward obstacle recursion. The available step functions are spx_vix_calibration, marginalize_to_sss, survival_probability, and contract_value; determining the correct arguments and composition order for each is part of this step.

```python
import numpy as np

def compute_contract_value() -> float:
    """Run the full pipeline and return the fair initial value of the
    3-year surrenderable equity-linked contract, using the task's exact
    instance inputs (see the problem statement for all numeric values).

    Returns
    -------
    value : float
        The contract's fair initial value, a single deterministic real
        number.

    Raises
    ------
    TypeError
        If called with any argument (this function takes none; the task
        instance is fixed).
    """
    return 0.0  # placeholder
```
