# Mathematics-Computational_Finance-43

## Background

Finite-interval variance sampling and truncated Euler discretization produce distinct state biases before regression-based optimal stopping. Their fixed-stream absolute-error ratio is a reproducible benchmark diagnostic, not a universal efficiency claim.

## Problem

Quantify the fixed-stream accuracy amplification of a finite-interval variance-endpoint simulation over the truncated-Euler comparator used in the source study for an at-the-money early-exercise put under a two-factor square-root stochastic-volatility model. Use $S_0=K=61.9$, $T=0.25$, $r=0.03$, initial variances $(0.2,0.49)$, mean-reversion rates $(0.9,1.2)$, long-run variances $(0.1,0.15)$, vol-of-vol values $(0.1,0.2)$, and price/variance correlations $(-0.5,-0.5)$; the factor pairs are otherwise independent.

Use twelve equal intervals, allow exercise at every positive grid date, and simulate $N=16384$ paths. For the finite-interval scheme, use dedicated `numpy.random.default_rng` streams with seeds 173 and 271 for the variance factors and seed 389 for log price, drawing one length-$N$ vector per step and drawing the first-factor price vector before the second-factor vector. For the comparator, reset those same three streams, draw each factor's Euler variance normal from its dedicated stream and the factor-one then factor-two orthogonal normals from the price stream, and apply the source study's zero-truncated variance Euler update and standard spot Euler update. Apply the source study's second-order in-the-money least-squares stopping protocol to both schemes, using NumPy ordinary least squares with `rcond=None` and retaining continuation on equality.

For this deterministic contract, treat the independently obtained reference scalar $P_{\mathrm{ref}}=9.504$ as an exact benchmark input. The retained digits characterize reproducibility of the prescribed fixed-stream calculation, not statistical uncertainty. If the two fixed-stream estimates are $P_{\mathrm{long}}$ and $P_{\mathrm{Euler}}$, report
$$A=\frac{|P_{\mathrm{Euler}}-P_{\mathrm{ref}}|}{|P_{\mathrm{long}}-P_{\mathrm{ref}}|}.$$
In the reasoning, state the two prices and their absolute errors, and explain what the ratio does and does not establish about the two simulation schemes. Report the ratio and all four intermediate scalars to at least eight significant figures.

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

01_compute_double_heston_aes_coefficients

Goal
----
Compute the seven ordered coefficients in one endpoint-conditioned double-Heston log-price update.

```python
def compute_double_heston_aes_coefficients(
    dt: float,
    rate: float,
    kappa: "np.ndarray",
    theta: "np.ndarray",
    gamma: "np.ndarray",
    rho: "np.ndarray",
) -> "np.ndarray":
    r"""Return the ordered coefficient vector for one time interval.

    Parameters
    ----------
    dt, rate : float
        Positive interval length and finite risk-free rate.
    kappa, theta, gamma, rho : numpy.ndarray
        Length-two factor parameters; scales are positive and correlations
        lie in [-1, 1].

    Returns
    -------
    coefficients : numpy.ndarray
        ``[constant, current_v1, current_v2, next_v1, next_v2,
        residual_v1, residual_v2]`` for the log update.

    Raises
    ------
    ValueError
        If ``dt`` is not positive, inputs are nonfinite or not length two,
        CIR scales are not positive, or a correlation is outside [-1, 1].
    """
    return coefficients
```

### Step 2

02_compute_cir_transition_parameters

Goal
----
Compute the ordered parameters of one finite-interval CIR transition.

```python
def compute_cir_transition_parameters(
    kappa: float, theta: float, gamma: float, dt: float
) -> "np.ndarray":
    r"""Return the scale, degrees of freedom, and noncentrality multiplier.

    Parameters
    ----------
    kappa, theta, gamma : float
        Positive mean reversion, long-run variance, and vol-of-vol.
    dt : float
        Positive transition interval.

    Returns
    -------
    parameters : numpy.ndarray
        Three finite transition parameters in source-defined order.

    Raises
    ------
    ValueError
        If any input is nonfinite or nonpositive.
    """
    return parameters
```

### Step 3

03_simulate_cir_variance_paths

Goal
----
Simulate one variance factor by iterating its finite-interval conditional transition.

```python
def simulate_cir_variance_paths(
    initial_variance: float,
    transition_parameters: "np.ndarray",
    n_steps: int,
    n_paths: int,
    seed: int,
) -> "np.ndarray":
    r"""Return variance paths including the initial row.

    Parameters
    ----------
    initial_variance : float
        Finite nonnegative initial variance.
    transition_parameters : numpy.ndarray
        Scale, degrees of freedom, and noncentrality multiplier.
    n_steps, n_paths, seed : int
        Positive counts and deterministic random seed.

    Returns
    -------
    paths : numpy.ndarray
        Nonnegative array of shape ``(n_steps + 1, n_paths)``.

    Raises
    ------
    ValueError
        If the transition state or simulation controls are invalid.
    """
    return paths
```

### Step 4

04_simulate_double_heston_aes_paths

Goal
----
Simulate asset paths from two variance paths with the endpoint-conditioned log-price update.

```python
def simulate_double_heston_aes_paths(
    initial_spot: float, coefficients: "np.ndarray",
    variance_one: "np.ndarray",
    variance_two: "np.ndarray", seed: int,
) -> "np.ndarray":
    r"""Return asset paths implied by two supplied variance-path arrays.

    Parameters
    ----------
    initial_spot : float
        Finite positive initial asset price.
    coefficients : numpy.ndarray
        Seven coefficients from the endpoint-conditioned construction.
    variance_one, variance_two : numpy.ndarray
        Equal-shaped nonnegative variance paths including time zero.
    seed : int
        Seed for factor-one then factor-two normal vectors at each step.

    Returns
    -------
    spots : numpy.ndarray
        Positive asset paths with the same shape as each variance array.
    Raises
    ------
    ValueError
        If any state, coefficient, initial value, or seed is invalid.
    """
    return spots
```

### Step 5

05_apply_double_heston_lsm_step

Goal
----
Apply one in-the-money least-squares Bermudan stopping decision.

```python
def apply_double_heston_lsm_step(
    spot: "np.ndarray", variance_one: "np.ndarray", variance_two: "np.ndarray",
    strike: float, current_step: int, cashflows: "np.ndarray",
    exercise_steps: "np.ndarray", rate: float, dt: float,
) -> "np.ndarray":
    r"""Return updated cashflows and stopping indices after one exercise date.

    Parameters
    ----------
    spot, variance_one, variance_two, cashflows, exercise_steps : numpy.ndarray
        Equal path vectors holding current and selected stopping states.
    strike, rate, dt : float
        Positive strike and interval length, and a finite interest rate.
    current_step : int
        Nonnegative current exercise index.
    Returns
    -------
    state : numpy.ndarray
        Two rows containing updated cashflows and stopping indices.

    Raises
    ------
    ValueError
        If path vectors, stopping states, or scalar date inputs are invalid.
    """
    return state
```

### Step 6

06_price_discounted_cashflows

Goal
----
Discount each selected stopping payoff to time zero and average across paths.

```python
def price_discounted_cashflows(
    cashflows: "np.ndarray", exercise_steps: "np.ndarray", rate: float, dt: float,
) -> float:
    r"""Return the time-zero Monte Carlo mean of stopping cashflows.

    Parameters
    ----------
    cashflows : numpy.ndarray
        Finite nonnegative realized payoff per path.
    exercise_steps : numpy.ndarray
        Nonnegative stopping-step index per path.
    rate : float
        Finite continuously compounded risk-free rate.
    dt : float
        Positive interval length.

    Returns
    -------
    price : float
        Discounted path mean as a native Python float.
    Raises
    ------
    ValueError
        If cashflow states or discounting inputs violate the stated contract.
    """
    return 0.0
```

### Step 7

07_summarize_bermudan_exercise_effect

Goal
----
Summarize the maturity-only value and the contribution produced by early exercise.

```python
def summarize_bermudan_exercise_effect(
    bermudan_price: float, terminal_payoffs: "np.ndarray",
    exercise_steps: "np.ndarray", rate: float, dt: float, maturity_step: int,
) -> "np.ndarray":
    r"""Return three diagnostics that reconstruct the Bermudan price.

    Parameters
    ----------
    bermudan_price : float
        Finite nonnegative time-zero stopping-policy mean.
    terminal_payoffs, exercise_steps : numpy.ndarray
        Equal nonempty vectors of maturity payoffs and selected stopping indices.
    rate, dt, maturity_step : float, float, int
        Finite rate, positive interval length, and positive terminal index.

    Returns
    -------
    diagnostics : numpy.ndarray
        ``[maturity_only_value, early_count, mean_early_increment]``.

    Raises
    ------
    ValueError
        If arrays, scalars, or stopping indices violate the stated contract.
    """
    return diagnostics
```

### Step 8

08_run_double_heston_aes_bermudan_pipeline

Goal
----
Run the complete seeded double-Heston long-step Bermudan pricing pipeline.

```python
def run_double_heston_aes_bermudan(
    n_paths: int = 4096, n_steps: int = 6, seed_v1: int = 173,
    seed_v2: int = 271, seed_price: int = 389,
) -> float:
    r"""Return the seeded Bermudan put value for the fixed task parameters.

    Parameters
    ----------
    n_paths, n_steps : int
        Positive path and interval counts.
    seed_v1, seed_v2, seed_price : int
        Dedicated deterministic random seeds.

    Returns
    -------
    price : float
        Time-zero option value.

    Raises
    ------
    ValueError
        If counts are not positive integers or seeds are not integers.

    Notes
    -----
    Call the seven preceding public functions in order.
    """
    return 0.0
```

### Step 9

09_simulate_double_heston_truncated_euler_paths

Goal
----
Simulate the paper's truncated-Euler comparator for the fixed double-Heston model.

```python
def simulate_double_heston_truncated_euler_paths(
    n_paths: int, n_steps: int, seed_v1: int, seed_v2: int, seed_price: int,
) -> "np.ndarray":
    r"""Return spot and variance paths from the source Euler comparator.

    Use the fixed double-Heston parameters in the task. At each step, draw the
    factor-one and factor-two variance normals from their dedicated streams,
    then the corresponding orthogonal normals from the price stream. Apply
    positive truncation after each variance update and standard Euler to spot,
    using current nonnegative variances and the correlated normal pairs.

    Parameters
    ----------
    n_paths, n_steps : int
        Positive path and interval counts.
    seed_v1, seed_v2, seed_price : int
        Nonnegative deterministic stream seeds.

    Returns
    -------
    paths : numpy.ndarray
        Float array of shape ``(3, n_steps + 1, n_paths)`` ordered as spot,
        first variance, and second variance.

    Raises
    ------
    ValueError
        If counts are not positive integers or seeds are not nonnegative integers.
    """
    return paths
```

### Step 10

10_run_double_heston_accuracy_benchmark

Goal
----
Compare fixed-stream early-exercise prices from the long-step and truncated-Euler schemes.

```python
def run_double_heston_accuracy_benchmark(
    n_paths: int = 16384, n_steps: int = 12, seed_v1: int = 173,
    seed_v2: int = 271, seed_price: int = 389, reference_price: float = 9.504,
) -> float:
    r"""Return the Euler-to-long-step absolute pricing-error ratio.
    Use all preceding public functions to cross-check the long-step price, price the Euler paths at strike 61.9, and form the error ratio.

    Parameters
    ----------
    n_paths, n_steps : int
        Positive counts; seeds are nonnegative integers.
    reference_price : float
        Finite positive benchmark; other parameters are deterministic seeds.
    Returns
    -------
    ratio : float
        Euler absolute error divided by long-step absolute error.
    Raises
    ------
    ValueError
        For invalid inputs, inconsistent long-step results, or a zero denominator.
    """
    return 0.0
```
