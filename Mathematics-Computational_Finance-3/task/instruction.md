# Mathematics-Computational_Finance-3

## Background

Multi-period portfolio optimization allocates portfolio weights across several periods while accounting for transaction costs and diversification. With finite return scenarios, portfolio loss is evaluated using scenario probabilities.

VaR is a quantile-based risk measure that can be nonconvex as a function of portfolio weights. In finite-scenario models, VaR admits an exact representation as a difference of two CVaR terms. Since CVaR is convex and piecewise affine, this representation enables a difference-of-convex (DC) formulation.

The resulting penalized problem can be solved with a projected inertial DC method that uses first-order information from the finite-scenario CVaR terms.

## Problem

Consider a multi-period portfolio allocation problem in which a finite-scenario Value-at-Risk (VaR) constraint at each period is reformulated exactly as a difference of two Conditional Value-at-Risk (CVaR) terms, yielding a penalized difference-of-convex (DC) objective over the portfolio simplex that is minimized with a custom inertial, safeguarded, line-search-based DC descent scheme. Consider an instance with T=2 periods, n=3 assets, and S=6 scenarios per period with non-uniform probabilities p¹ = [0.05, 0.10, 0.15, 0.20, 0.22, 0.28] and p² = [0.08, 0.09, 0.12, 0.16, 0.24, 0.31]; scenario return matrices (rows = scenarios, columns = assets) ξ¹ = [[-0.15,0.05,0.03],[0.04,-0.13,0.02],[0.03,0.02,-0.12],[-0.05,-0.04,0.06],[0.08,0.07,0.09],[-0.02,0.03,-0.06]] and ξ² = [[-0.12,0.04,0.02],[0.05,-0.11,0.03],[0.02,0.03,-0.10],[-0.04,-0.03,0.05],[0.09,0.08,0.10],[-0.03,0.02,-0.05]]; previous holding w_prev = [0.40, 0.35, 0.25]; and initial portfolio x⁰ = x⁻¹ with period 1 = [0.45, 0.30, 0.25] and period 2 = [0.30, 0.45, 0.25]. Use VaR confidence level α = 0.65, VaR threshold τ = 0.02, transaction-cost coefficients c = [0.06, 0.10, 0.14], weights λ₁ = 0.07 and λ₂ = 0.02, penalty parameter ρ = 6, DC-splitting parameter γ = 0.005, and algorithm parameters θ = 0.05, ν = 0.04, σ = 0.004, η = 0.5, λ̄ = 1.2, with no early stopping criterion applied. Using the exact finite-scenario VaR–CVaR identity to build the penalized DC objective and its associated inertial descent update rule, run exactly K = 4 outer iterations starting from x⁰, and compute the period-2, asset-1 portfolio weight of the resulting iterate x⁴. Report this weight to at least six decimal places.

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

01_compute_scenario_gap

Goal
----
Compute the discrete confidence-gap quantity associated with a finite scenario probability vector.

```python
import numpy as np

def compute_scenario_gap(probabilities: np.ndarray, confidence: float) -> float:
    """Return the discrete confidence gap for a finite probability distribution.

    Parameters
    ----------
    probabilities : np.ndarray
        One-dimensional positive scenario probabilities that sum to one.
    confidence : float
        Confidence level in the open interval (0, 1).

    Returns
    -------
    gap : float
        Difference between the confidence level and the largest subset probability
        strictly below it.

    Raises
    ------
    ValueError
        If probabilities are invalid or confidence is outside (0, 1).
    """
    return gap
```

### Step 2

02_compute_loss_vector

Goal
----
Compute the scenario loss vector induced by a portfolio weight vector and a scenario-return matrix.

```python
import numpy as np

def compute_loss_vector(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return one loss value per return scenario.

    Parameters
    ----------
    returns : np.ndarray
        Array of shape (S, n) containing scenario returns.
    weights : np.ndarray
        Portfolio weights of length n.

    Returns
    -------
    losses : np.ndarray
        Vector of S scenario losses in the same row order as returns.

    Raises
    ------
    ValueError
        If the dimensions are incompatible or inputs contain non-finite values.
    """
    return losses
```

### Step 3

03_compute_cvar

Goal
----
Compute a probability-weighted upper-tail conditional risk value from finite scenario losses.

```python
import numpy as np

def compute_cvar(losses: np.ndarray, probabilities: np.ndarray, beta: float) -> float:
    """Return the finite-scenario upper-tail conditional risk value.

    Parameters
    ----------
    losses : np.ndarray
        One-dimensional scenario loss values.
    probabilities : np.ndarray
        Positive scenario probabilities summing to one.
    beta : float
        Tail confidence level in (0, 1).

    Returns
    -------
    value : float
        Minimum Rockafellar-Uryasev finite-scenario objective value.

    Raises
    ------
    ValueError
        If dimensions, probabilities, or beta are invalid.
    """
    return value
```

### Step 4

04_build_dc_risk_terms

Goal
----
Combine two conditional risk values and the supplied splitting controls into the scalar risk terms used by the penalized model.

```python
import numpy as np

def build_dc_risk_terms(cvar_lower: float, cvar_alpha: float, alpha: float, gamma: float, tau: float) -> np.ndarray:
    """Return the scalar quantities defining one period's penalized risk contribution.

    Parameters
    ----------
    cvar_lower : float
        Conditional risk value evaluated at the lower tail level.
    cvar_alpha : float
        Conditional risk value evaluated at the requested confidence level.
    alpha : float
        Requested confidence level.
    gamma : float
        Positive splitting offset.
    tau : float
        VaR threshold.

    Returns
    -------
    terms : np.ndarray
        Five-entry vector containing the two splitting coefficients, the two
        transformed branch values, and their maximum.

    Raises
    ------
    ValueError
        If alpha is outside (0, 1) or gamma is outside (0, alpha).
    """
    return terms
```

### Step 5

05_compute_cvar_subgradient

Goal
----
Compute a valid scenario-based subgradient of the finite-scenario upper-tail risk at one portfolio allocation.

```python
import numpy as np

def compute_cvar_subgradient(returns: np.ndarray, probabilities: np.ndarray, weights: np.ndarray, beta: float) -> np.ndarray:
    """Return an asset-space subgradient of finite-scenario CVaR.

    Parameters
    ----------
    returns : np.ndarray
        Scenario-return matrix with shape (S, n).
    probabilities : np.ndarray
        Scenario probabilities of length S.
    weights : np.ndarray
        Portfolio weights of length n.
    beta : float
        Confidence level in (0, 1).

    Returns
    -------
    gradient : np.ndarray
        Subgradient vector in asset coordinates. When several scenarios share
        the threshold loss, the residual tail mass is assigned to them in
        increasing scenario-index order, each up to its full probability.

    Raises
    ------
    ValueError
        If dimensions or probability inputs are invalid.
    """
    return gradient
```

### Step 6

06_solve_regularized_subproblem

Goal
----
Solve the strongly convex allocation subproblem associated with one expansion point of the penalized portfolio objective.

```python
import numpy as np

def solve_regularized_subproblem(probabilities: np.ndarray, returns: np.ndarray, w_prev: np.ndarray, c: np.ndarray, lambda1: float, lambda2: float, rho: float, alpha: float, gamma: float, tau: float, nu: float, w: np.ndarray, linearized_h_gradient: np.ndarray) -> np.ndarray:
    """Return the allocation solving the regularized convex subproblem.

    Parameters
    ----------
    probabilities : np.ndarray
        Scenario probabilities, shape (2, 6).
    returns : np.ndarray
        Scenario returns, shape (2, 6, 3).
    w_prev : np.ndarray
        Previous holding, shape (3,).
    c : np.ndarray
        Asset transaction-cost coefficients, shape (3,).
    lambda1, lambda2, rho, alpha, gamma, tau, nu : float
        Model and decomposition controls.
    w : np.ndarray
        Current expansion allocation, shape (2, 3).
    linearized_h_gradient : np.ndarray
        Asset-space linearization vector for the concave DC component, length 6.

    Returns
    -------
    y : np.ndarray
        Unique two-period allocation returned by the convex subproblem. The
        subproblem must be solved to high accuracy (objective tolerance 1e-9 or
        tighter); the returned allocation is compared at 1e-6.

    Raises
    ------
    ValueError
        If dimensions are incompatible or the supplied data violate the tested input contract.
    """
    return y
```

### Step 7

07_take_ibdca_iteration

Goal
----
Perform one safeguarded boosted portfolio-allocation iteration from two consecutive iterates.

```python
import numpy as np

def take_ibdca_iteration(probabilities: np.ndarray, returns: np.ndarray, w_prev: np.ndarray, c: np.ndarray, lambda1: float, lambda2: float, rho: float, alpha: float, gamma: float, tau: float, theta: float, nu: float, sigma: float, eta: float, bar_lambda: float, x_k: np.ndarray, x_km1: np.ndarray) -> np.ndarray:
    """Return the next allocation produced by one safeguarded boosted iteration.

    Parameters
    ----------
    probabilities : np.ndarray
        Scenario probabilities, shape (2, 6).
    returns : np.ndarray
        Scenario returns, shape (2, 6, 3).
    w_prev : np.ndarray
        Holding before period 1.
    c : np.ndarray
        Transaction-cost coefficients.
    lambda1, lambda2, rho, alpha, gamma, tau, theta, nu, sigma, eta, bar_lambda : float
        Model and algorithm controls. No arguments are optional.
    x_k : np.ndarray
        Current two-period allocation.
    x_km1 : np.ndarray
        Previous two-period allocation.

    Returns
    -------
    x_next : np.ndarray
        Next two-period allocation, shape (2, 3).

    Raises
    ------
    ValueError
        If the supplied arrays do not have the instance shapes or gamma violates
        the finite-scenario gap condition.
    """
    return x_next
```

### Step 8

08_run_ibdca_target

Goal
----
Run the full deterministic portfolio-allocation procedure for the requested number of outer iterations and extract the final target component.

```python
import numpy as np

def run_ibdca_target(probabilities: np.ndarray, returns: np.ndarray, w_prev: np.ndarray, c: np.ndarray, lambda1: float, lambda2: float, rho: float, alpha: float, gamma: float, tau: float, theta: float, nu: float, sigma: float, eta: float, bar_lambda: float, x0: np.ndarray, K: int) -> float:
    """Return the requested terminal allocation component after K outer iterations.

    Parameters
    ----------
    probabilities : np.ndarray
        Scenario probabilities, shape (2, 6).
    returns : np.ndarray
        Scenario returns, shape (2, 6, 3).
    w_prev : np.ndarray
        Previous holding before the first period.
    c : np.ndarray
        Transaction-cost coefficients.
    lambda1, lambda2, rho, alpha, gamma, tau, theta, nu, sigma, eta, bar_lambda : float
        Model, decomposition, inertia, and line-search controls.
    x0 : np.ndarray
        Initial allocation. The algorithm uses the same array as the x^{-1} state.
    K : int
        Number of outer iterations to execute; no early stopping is used.

    Returns
    -------
    value : float
        Period-2, asset-1 component of the terminal allocation.

    Raises
    ------
    ValueError
        If the supplied instance shapes are incompatible or K is negative.
    """
    return value
```
