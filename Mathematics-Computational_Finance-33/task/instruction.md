# Mathematics-Computational_Finance-33

## Background

Weather and insurance contracts often depend on the accumulated size of events over a fixed period. Event counts alone do not determine the payout: both arrival times and event magnitudes matter, and large events may be followed by unusually high activity. A model with feedback from event size to future arrivals can represent this clustering.

Pricing such contracts requires a distribution for future losses under a specified valuation measure. Market quotes can constrain that measure, but a quote for a linear payment does not determine the value of a capped or otherwise nonlinear claim directly. Numerical methods must preserve the interaction between event size, future activity, and the contractual payoff while controlling the cost of representing the accumulated-loss state.

## Problem

Price a capped claim on accumulated positive marks when each event also increases the arrival intensity, with time measured in years, marks in loss units, and one currency unit paid per loss unit.
Under the pricing law, $d\lambda_t=\kappa(\bar\lambda-\lambda_t)\,dt+\beta\,dU_t$, the event intensity is $\lambda_{t-}$, and the original mark law has independent-of-state Gamma components with shapes $(2,6)$, rates $(4,2.5)$ per loss unit, and probabilities $(0.6,0.4)$; apply a normalized exponential tilt $e^{\theta x}$ to this mark law while keeping the event intensity unchanged.
Use $(\kappa,\bar\lambda,\lambda_0,\beta,r)=(8,2,2.7,1.1,0.02)$, where $\kappa,r$ are annual rates, $\lambda$ is in events per year and $\beta$ is the intensity increment per loss unit, and determine the unique $\theta\in[0,0.35]$ that gives the time-zero price $2.2$ for a swap paying $U_{0.5}-U_0$ at time $0.5$, using the exact first-moment dynamics of the untruncated process.
At $U_0=0.15$, value the payoff $\min((U_T-1.2)^+,3)$ at $T=150/365$, discounted at $r$, under that same calibrated mark law.

The requested value is the finite discretization defined by a transform reduction in $U$ on the line $\eta=0.3+iy$, using $97$ intensity nodes $\lambda_i=i/4$ for $i=0,\ldots,96$, $256$ equal backward time steps, and $48$ positive generalized Gaussian nodes for each Gamma component with weight $z^{k-1}e^{-z}$ and change of variable $z=(b-\theta)x$.
Treat intensity drift and discount implicitly with the monotone first-order spatial discretization for reverse time, treat the entire jump gain-minus-loss explicitly at the later calendar time, and use the inward one-sided drift at each endpoint together with piecewise-linear interpolation and constant clamping for shifted intensity queries.
Recover the price from the positive-frequency integral using composite Simpson weights on $y_j=j/8$, $j=0,\ldots,192$, and interpolate linearly to $\lambda_0$; use the analytic damped transform of the capped payoff, include both frequency halves through conjugate symmetry, and retain the finite-grid value without extrapolation or clipping.
Report this single price with absolute error at most $10^{-7}$, and in the reasoning give the calibrated tilt, tilted mean mark, effective first-moment decay rate, and discrete explicit-jump stability factor, explaining how the normalized tilt and backward drift convention enter the valuation.

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

01_tilt_mark_distribution

Goal
----
Return the normalized, exponentially reweighted Gamma mixture.

```python
def tilt_mark_distribution(
    probabilities: np.ndarray,
    shapes: np.ndarray,
    rates: np.ndarray,
    theta: float,
) -> np.ndarray:
    r"""Return the normalized, exponentially reweighted Gamma mixture.

    Parameters
    ----------
    probabilities : np.ndarray
        Shape (M,), positive component probabilities.
    shapes : np.ndarray
        Shape (M,), positive Gamma shapes.
    rates : np.ndarray
        Shape (M,), positive original rates in inverse loss units.
    theta : float
        Finite tilt, strictly below every original rate.

    Returns
    -------
    mixture : np.ndarray
        Shape (M, 3), columns probability, shape, and tilted rate.

    Raises
    ------
    ValueError
        If arrays are empty, non-real, non-finite, differently shaped, nonpositive, or
        unnormalized, or theta is inadmissible.

    Notes
    -----
    Inputs are not modified. Numerical comparisons use rtol=1e-9 and atol=1e-11.
    """
    return
```

### Step 2

02_calibrate_mark_tilt

Goal
----
Determine the mark tilt from a discounted cumulative-loss swap.

```python
def calibrate_mark_tilt(
    probabilities: np.ndarray,
    shapes: np.ndarray,
    rates: np.ndarray,
    market: np.ndarray,
    swap_maturity: float,
    swap_price: float,
    bracket: np.ndarray,
) -> float:
    r"""Determine the mark tilt from a discounted cumulative-loss swap.

    Parameters
    ----------
    probabilities : np.ndarray
        Shape (M,), normalized positive probabilities.
    shapes : np.ndarray
        Shape (M,), positive Gamma shapes.
    rates : np.ndarray
        Shape (M,), positive rates.
    market : np.ndarray
        Shape (5,), ordered kappa, baseline intensity, initial intensity, excitation
        beta, interest rate; kappa > 0, others nonnegative, and baseline plus initial
        intensity positive.
    swap_maturity : float
        Positive settlement time in years.
    swap_price : float
        Nonnegative discounted price in loss-payment currency units.
    bracket : np.ndarray
        Shape (2,), strictly increasing finite tilt endpoints; every endpoint satisfies
        theta < min(rates) and kappa > beta times tilted mean.

    Returns
    -------
    theta : float
        Calibrated mark tilt in inverse loss units.

    Raises
    ------
    ValueError
        If mixture or market inputs violate their stated domains, the bracket is
        inadmissible or non-subcritical, or the quote is not bracketed.

    Notes
    -----
    Inputs are not modified. Numerical comparisons use rtol=1e-9 and atol=1e-11.
    """
    return
```

### Step 3

03_build_mark_quadrature

Goal
----
Construct positive quadrature rules for the tilted mark components.

```python
def build_mark_quadrature(
    mixture: np.ndarray,
    order: int,
) -> np.ndarray:
    r"""Construct positive quadrature rules for the tilted mark components.

    Parameters
    ----------
    mixture : np.ndarray
        Shape (M, 3), columns normalized positive probability, positive shape, positive
        tilted rate.
    order : int
        Positive quadrature order per component, at most 128.

    Returns
    -------
    rule : np.ndarray
        Shape (M, order, 2), final axis mark node in loss units and probability weight.

    Raises
    ------
    ValueError
        If mixture entries are invalid, probabilities are unnormalized, or order is
        outside the integer range 1 through 128.

    Notes
    -----
    Inputs are not modified. Numerical comparisons use rtol=1e-9 and atol=1e-11.
    """
    return
```

### Step 4

04_build_backward_drift

Goal
----
Assemble the banded implicit drift and discount system.

```python
def build_backward_drift(
    grid: np.ndarray,
    market: np.ndarray,
    dt: float,
) -> np.ndarray:
    r"""Assemble the banded implicit drift and discount system.

    Parameters
    ----------
    grid : np.ndarray
        Shape (I+1,), at least three equally spaced increasing nonnegative intensity
        nodes beginning at zero.
    market : np.ndarray
        Shape (5,), kappa, baseline, initial, beta, rate; baseline must lie within the
        grid.
    dt : float
        Positive time step in years.

    Returns
    -------
    bands : np.ndarray
        Shape (3, I+1), upper/main/lower band storage of the implicit matrix.

    Raises
    ------
    ValueError
        If grid is nonuniform or invalid, market is invalid, baseline is outside the
        grid, or dt is not positive and finite.

    Notes
    -----
    Inputs are not modified. Numerical comparisons use rtol=1e-9 and atol=1e-11.
    """
    return
```

### Step 5

05_build_jump_transfer

Goal
----
Construct the modal jump-gain matrices with shifted-intensity interpolation.

```python
def build_jump_transfer(
    grid: np.ndarray,
    rule: np.ndarray,
    beta: float,
    frequencies: np.ndarray,
    delta: float,
) -> np.ndarray:
    r"""Construct the modal jump-gain matrices with shifted-intensity interpolation.

    Parameters
    ----------
    grid : np.ndarray
        Uniform intensity grid of shape (I+1,), starting at zero.
    rule : np.ndarray
        Shape (M, Q, 2), positive mark nodes and probability weights summing to one.
    beta : float
        Nonnegative excitation coefficient in intensity per loss unit.
    frequencies : np.ndarray
        Shape (J,), at least one nonnegative increasing frequency, first entry zero, in
        inverse loss units.
    delta : float
        Nonnegative finite contour shift in inverse loss units; quadrature exponentials
        must remain finite.

    Returns
    -------
    transfer : np.ndarray
        Complex array of shape (J, I+1, I+1), the dimensionless modal jump-gain matrices.

    Raises
    ------
    ValueError
        If grid, rule, frequency ordering, beta or delta is invalid, weights are
        unnormalized, or the modal factors overflow.

    Notes
    -----
    Inputs are not modified. Numerical comparisons use rtol=1e-9 and atol=1e-11.
    """
    return
```

### Step 6

06_evolve_modal_values

Goal
----
Propagate terminal exponential modes to time zero.

```python
def evolve_modal_values(
    grid: np.ndarray,
    bands: np.ndarray,
    transfer: np.ndarray,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    r"""Propagate terminal exponential modes to time zero.

    Parameters
    ----------
    grid : np.ndarray
        Uniform intensity grid of shape (I+1,), beginning at zero.
    bands : np.ndarray
        Shape (3, I+1), real implicit matrix with nonpositive off diagonals, positive
        row-dominance margin and zero unused corners.
    transfer : np.ndarray
        Finite complex gain array of shape (J, I+1, I+1), J >= 1.
    dt : float
        Positive finite time step in years.
    n_steps : int
        Positive number of backward steps.

    Returns
    -------
    modes : np.ndarray
        Complex array of shape (I+1, J), time-zero discounted exponential modes.

    Raises
    ------
    ValueError
        If dimensions, signs, row dominance, finite data or time parameters are invalid,
        the explicit-jump factor is at least one, or evolution becomes non-finite.

    Notes
    -----
    Inputs are not modified. Numerical comparisons use rtol=1e-9 and atol=1e-11.
    """
    return
```

### Step 7

07_invert_capped_payoff

Goal
----
Recover one capped-claim value from discounted exponential modes.

```python
def invert_capped_payoff(
    grid: np.ndarray,
    modes: np.ndarray,
    frequencies: np.ndarray,
    delta: float,
    initial_intensity: float,
    initial_loss: float,
    strike: float,
    cap: float,
) -> float:
    r"""Recover one capped-claim value from discounted exponential modes.

    Parameters
    ----------
    grid : np.ndarray
        Uniform intensity grid of shape (I+1,), starting at zero.
    modes : np.ndarray
        Finite complex discounted modes of shape (I+1, J).
    frequencies : np.ndarray
        Shape (J,), odd J >= 3, uniform increasing frequencies starting at zero.
    delta : float
        Positive contour shift in inverse loss units.
    initial_intensity : float
        Initial intensity within the grid.
    initial_loss : float
        Nonnegative accumulated loss.
    strike : float
        Nonnegative loss strike.
    cap : float
        Positive payoff cap in loss units.

    Returns
    -------
    price : float
        Finite-grid capped-claim value in currency units.

    Raises
    ------
    ValueError
        If grids, modes, dimensions, parameters or frequency parity are invalid, or the
        inversion is non-finite.

    Notes
    -----
    Inputs are not modified. Numerical comparisons use rtol=1e-9 and atol=1e-11.
    """
    return
```

### Step 8

08_price_calibrated_claim

Goal
----
Calibrate the mark law and return the complete finite-grid capped-claim price.

```python
def price_calibrated_claim(
    probabilities: np.ndarray,
    shapes: np.ndarray,
    rates: np.ndarray,
    market: np.ndarray,
    swap: np.ndarray,
    contract: np.ndarray,
    resolution: np.ndarray,
) -> float:
    r"""Calibrate the mark law and return the complete finite-grid capped-claim price.

    Parameters
    ----------
    probabilities : np.ndarray
        Shape (M,), normalized positive original component probabilities.
    shapes : np.ndarray
        Shape (M,), positive original Gamma shapes.
    rates : np.ndarray
        Shape (M,), positive original Gamma rates.
    market : np.ndarray
        Shape (5,), ordered kappa, baseline intensity, initial intensity, excitation,
        interest rate; kappa > 0, others nonnegative, and baseline plus initial
        intensity positive.
    swap : np.ndarray
        Shape (4,), positive settlement time, nonnegative quote, lower tilt, upper tilt;
        admissible subcritical increasing bracket.
    contract : np.ndarray
        Shape (4,), positive claim maturity, nonnegative initial loss, nonnegative
        strike, positive cap.
    resolution : np.ndarray
        Shape (7,), intensity maximum, integer interval count >= 2, integer time-step
        count >= 1, integer quadrature order 1..128, positive contour shift, positive
        frequency maximum, even integer frequency interval count >= 2.

    Returns
    -------
    price : float
        One finite capped-claim price in currency units.

    Raises
    ------
    ValueError
        If any component contract fails, resolution counts are nonintegral or out of
        range, the calibration quote is not bracketed, initial or baseline intensity is
        outside the grid, the contour is inadmissible, or the explicit-jump stability
        bound is violated.

    Notes
    -----
    Inputs are not modified. Numerical comparisons use rtol=1e-9 and atol=1e-11.
    """
    return
```
