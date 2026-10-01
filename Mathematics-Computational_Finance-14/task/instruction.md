# Mathematics-Computational_Finance-14

## Background

Early-exercise derivatives couple stochastic asset dynamics to decisions made at several future dates. Their value depends on continuation values that are not known when paths are generated, so conventional simulation methods often alternate forward sampling with backward optimization.

A recent BSDE formulation represents the intervals between decision dates as linked equations and places every component inside one forward computational framework. Intermediate conditions express exercise or compounding events while trainable value and hedge processes supply the unknown conditional quantities.

The method yields prices and hedge sensitivities for compound and Bermudan contracts, including high-dimensional baskets. Its numerical reliability is assessed through terminal and intermediate residuals, comparisons with analytical prices or tree benchmarks, and an a posteriori estimate connecting those residuals and the time discretization to solution error.

## Problem

Multi-date early exercise can be represented by BSDE components on successive time intervals, but their intermediate continuation values are unknown and couple the components. Evaluate a deterministic frozen-surrogate instance of a coupled multi-interval BSDE discretization for a Bermudan geometric-basket put and report its a posteriori residual certificate.

The underlying follows the source's risk-neutral two-asset dynamics. Segment-start values and gridwise hedge controls are represented by deterministic frozen neural heads, so no fitting or backward regression is performed.

Your task is to solve one concrete deterministic example of this pipeline. Use the following configuration:

- Use the frozen Brownian stream supplied by the code task. The default configuration of the final orchestration function identifies the canonical stream for this instance; do not substitute or regenerate it from an independently chosen seed. That stream is the single tensor `numpy.random.default_rng(260118634).normal(0.0, sqrt(h), size=(10, 7, 2))`, drawn in one call and indexed time-major as (time step, path, Brownian component).
- `n_paths = 7`
- `segment_steps = [2, 2, 2, 2, 2]`
- `h = 0.125`
- `x0 = [48.0, 52.0]`
- `x_scale = [50.0, 50.0]`
- Every frozen head receives the elementwise state input `log(x / x_scale)`.
- `r = 0.06`
- `q = [0.01, 0.015]`
- `sigma = [[0.24, 0.05], [0.08, 0.20]]`
- `theta = [0.45, -0.30]`
- `strikes = [51.00, 50.00, 49.00, 48.00, 47.00]`
- `feature_matrix = [[0.70, -0.30], [-0.40, 0.60], [0.25, 0.50], [-0.55, -0.20]]`
- `feature_bias = [0.10, -0.15, 0.05, 0.20]`
- `value_weights = [[1.20, -0.40, 0.70, 0.30], [0.90, -0.60, 0.50, -0.20], [0.60, -0.30, 0.40, 0.10], [0.50, -0.25, 0.35, 0.05], [0.40, -0.20, 0.30, 0.15]]`
- `value_bias = [3.40, 3.00, 2.60, 2.20, 1.80]`
- `control_bias = [[-1.80, -1.40], [-1.70, -1.35], [-1.60, -1.25], [-1.50, -1.15], [-1.40, -1.05], [-1.30, -0.95], [-1.20, -0.85], [-1.10, -0.75], [-1.00, -0.65], [-0.90, -0.55]]`
- `control_weights = [[[0.30, -0.20, 0.15, 0.10], [-0.10, 0.25, 0.20, -0.15]], [[0.25, -0.15, 0.10, 0.12], [-0.08, 0.22, 0.18, -0.12]], [[0.22, -0.18, 0.14, 0.08], [-0.12, 0.20, 0.16, -0.10]], [[0.20, -0.14, 0.12, 0.06], [-0.10, 0.18, 0.14, -0.08]], [[0.18, -0.12, 0.10, 0.05], [-0.08, 0.16, 0.12, -0.06]], [[0.16, -0.10, 0.08, 0.04], [-0.06, 0.14, 0.10, -0.05]], [[0.14, -0.08, 0.06, 0.03], [-0.05, 0.12, 0.08, -0.04]], [[0.12, -0.06, 0.05, 0.02], [-0.04, 0.10, 0.07, -0.03]], [[0.10, -0.05, 0.04, 0.02], [-0.03, 0.09, 0.06, -0.02]], [[0.08, -0.04, 0.03, 0.01], [-0.02, 0.08, 0.05, -0.02]]]`

Use the source's coupled multi-interval BSDE discretization, intermediate Bermudan compounding conditions, simultaneous joint objective, and a posteriori estimate without its unknown multiplicative constant. Every component here carries the source's general driver in both the value and the control argument, `f_j(t, x, y, z) = -r*y + theta . z`, with `theta . z` the inner product over the Brownian dimensions. Use the stated frozen-head convention, and apply the source-defined segment indexing and finite-path reduction conventions to the listed arrays without numerical rounding. Your final answer must be a single number: the constant-free a posteriori certificate at full precision.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Background

Early-exercise derivatives couple stochastic asset dynamics to decisions made at several future dates. Their value depends on continuation values that are not known when paths are generated, so conventional simulation methods often alternate forward sampling with backward optimization.

A recent BSDE formulation represents the intervals between decision dates as linked equations and places every component inside one forward computational framework. Intermediate conditions express exercise or compounding events while trainable value and hedge processes supply the unknown conditional quantities.

The method yields prices and hedge sensitivities for compound and Bermudan contracts, including high-dimensional baskets. Its numerical reliability is assessed through terminal and intermediate residuals, comparisons with analytical prices or tree benchmarks, and an a posteriori estimate connecting those residuals and the time discretization to solution error.

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

01_generate_brownian_increments

Goal
----
Generate the Brownian increments for the fixed Monte Carlo paths.

A Brownian increment over a uniform interval of length h has independent

components distributed as N(0, h). Drawing the complete tensor in time-major

order fixes both the pseudorandom stream and the association between increments,

paths, and Brownian dimensions.

Inputs

------

seed: Nonnegative seed for the NumPy random generator.

n_steps: Positive number of uniform time increments.

n_paths: Positive number of Monte Carlo paths.

dim_w: Positive Brownian dimension.

h: Positive uniform time step.

Returns

-------

increments: Float array of shape (n_steps, n_paths, dim_w).

```python
import numpy as np


def generate_brownian_increments(
    seed: int, n_steps: int, n_paths: int, dim_w: int, h: float
) -> np.ndarray:
    """Generate time-major Brownian increments with a fixed NumPy stream.

    Parameters
    ----------
    seed : int
        Nonnegative seed for ``numpy.random.default_rng``.
    n_steps : int
        Positive number of time increments.
    n_paths : int
        Positive number of simulated paths.
    dim_w : int
        Positive Brownian dimension.
    h : float
        Positive finite uniform step size.

    Raises
    ------
    ValueError
        If the seed is not a nonnegative integer, any dimension is not a
        positive integer, or h is not finite and positive.

    Returns
    -------
    increments : np.ndarray
        Array of shape ``(n_steps, n_paths, dim_w)``.
    """
    return increments  # noqa: F821
```

### Step 2

02_simulate_gbm_paths

Goal
----
Advance the risk-neutral asset process on the uniform grid.

Under the risk-neutral measure, the Euler-Maruyama state update must retain the

multiplicative dependence of each asset on its current level and respect the

asset-by-Brownian orientation of the volatility matrix. Every grid state is

retained because later BSDE components begin and end at different dates.

Inputs

------

x0: Positive initial asset vector of shape (d,).

q: Dividend-yield vector of shape (d,).

sigma: Volatility matrix of shape (d, dim_w).

r: Risk-free rate.

h: Positive uniform time step.

increments: Brownian array of shape (n_steps, n_paths, dim_w).

Returns

-------

paths: Float array of shape (n_steps + 1, n_paths, d).

```python
import numpy as np


def simulate_gbm_paths(
    x0: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    r: float,
    h: float,
    increments: np.ndarray,
) -> np.ndarray:
    """Simulate risk-neutral geometric Brownian paths by Euler-Maruyama.

    Parameters
    ----------
    x0 : np.ndarray
        Positive initial asset vector of shape ``(d,)``.
    q : np.ndarray
        Finite dividend-yield vector of shape ``(d,)``.
    sigma : np.ndarray
        Finite volatility matrix of shape ``(d, dim_w)``.
    r : float
        Finite risk-free rate.
    h : float
        Positive finite step size.
    increments : np.ndarray
        Brownian increments of shape ``(n_steps, n_paths, dim_w)``.

    Raises
    ------
    ValueError
        If dimensions disagree, any input is nonfinite, x0 or h is not
        positive, or an Euler update produces a nonpositive asset value.

    Returns
    -------
    paths : np.ndarray
        Asset paths of shape ``(n_steps + 1, n_paths, d)``.
    """
    return paths  # noqa: F821
```

### Step 3

03_evaluate_value_heads

Goal
----
Evaluate the frozen value head at every segment start.

A fully forward compound-BSDE solver parameterizes the value at the beginning

of every segment as a function of the state there, including random intermediate

states. Each frozen value head is a linear readout of a shared tanh feature map

of normalized log prices.

Inputs

------

paths: Positive asset paths of shape (n_times, n_paths, d).

segment_starts: Increasing grid indices for the segment starts.

x_scale: Positive log-price scale of shape (d,).

feature_matrix, feature_bias: Shared tanh feature parameters.

value_weights, value_bias: Segment-specific value-head parameters.

Returns

-------

values: Float array of shape (n_segments, n_paths).

```python
import numpy as np


def evaluate_value_heads(
    paths: np.ndarray,
    segment_starts: np.ndarray,
    x_scale: np.ndarray,
    feature_matrix: np.ndarray,
    feature_bias: np.ndarray,
    value_weights: np.ndarray,
    value_bias: np.ndarray,
) -> np.ndarray:
    """Evaluate frozen segment-start value heads on their path states.

    Parameters
    ----------
    paths : np.ndarray
        Positive paths of shape ``(n_times, n_paths, d)``.
    segment_starts : np.ndarray
        Increasing segment-start indices of shape ``(m,)``.
    x_scale : np.ndarray
        Positive feature scale of shape ``(d,)``.
    feature_matrix : np.ndarray
        Feature coefficients of shape ``(n_features, d)``.
    feature_bias : np.ndarray
        Feature offsets of shape ``(n_features,)``.
    value_weights : np.ndarray
        Head weights of shape ``(m, n_features)``.
    value_bias : np.ndarray
        Head offsets of shape ``(m,)``.

    Raises
    ------
    ValueError
        If arrays have incompatible shapes, paths or x_scale are not positive,
        segment starts are invalid, or any input is nonfinite.

    Returns
    -------
    values : np.ndarray
        Segment-start values of shape ``(m, n_paths)``.
    """
    return values  # noqa: F821
```

### Step 4

04_evaluate_control_heads

Goal
----
Evaluate the frozen martingale-control heads along the path grid.

The martingale control is parameterized separately at every grid point in a

deep BSDE discretization. Each frozen head consumes the shared normalized

log-price features and returns one coefficient per Brownian dimension.

Preserving the distinct time, path, Brownian, and feature axes is essential

when evaluating all heads over the path grid.

Inputs

------

paths: Positive asset paths of shape (n_steps + 1, n_paths, d).

x_scale: Positive log-price scale of shape (d,).

feature_matrix, feature_bias: Shared tanh feature parameters.

control_weights, control_bias: Grid-specific control-head parameters.

Returns

-------

controls: Float array of shape (n_steps, n_paths, dim_w).

```python
import numpy as np


def evaluate_control_heads(
    paths: np.ndarray,
    x_scale: np.ndarray,
    feature_matrix: np.ndarray,
    feature_bias: np.ndarray,
    control_weights: np.ndarray,
    control_bias: np.ndarray,
) -> np.ndarray:
    """Evaluate frozen grid-point control heads.

    Parameters
    ----------
    paths : np.ndarray
        Positive states of shape ``(n_steps + 1, n_paths, d)``.
    x_scale : np.ndarray
        Positive feature scale of shape ``(d,)``.
    feature_matrix : np.ndarray
        Feature coefficients of shape ``(n_features, d)``.
    feature_bias : np.ndarray
        Feature offsets of shape ``(n_features,)``.
    control_weights : np.ndarray
        Head weights of shape ``(n_steps, dim_w, n_features)``.
    control_bias : np.ndarray
        Head offsets of shape ``(n_steps, dim_w)``.

    Raises
    ------
    ValueError
        If arrays have incompatible shapes, states or scales are not positive,
        the number of heads differs from the number of steps, or inputs are nonfinite.

    Returns
    -------
    controls : np.ndarray
        Control values of shape ``(n_steps, n_paths, dim_w)``.
    """
    return controls  # noqa: F821
```

### Step 5

05_propagate_bsde_segments

Goal
----
Propagate every BSDE component forward on its own interval.

Every backward component is rewritten as a forward recurrence on its own

interval. The driver depends on both the value and the control, so the

discrete drift contributes one term proportional to the current value and a

second proportional to the control contracted with the driver coefficient.

The control also enters through the Brownian contraction at each local step.

Every segment begins from its separately parameterized value head.

Inputs

------

value_starts: Segment-start values of shape (n_segments, n_paths).

controls: Grid controls of shape (n_steps, n_paths, dim_w).

increments: Brownian increments with the same shape as controls.

segment_steps: Positive grid counts for the successive segments.

r: Risk-free rate in the driver f=-rY+theta.Z.

h: Positive uniform time step.

theta: Driver control coefficient of shape (dim_w,).

Returns

-------

terminals: Float array of segment right-end values.

```python
import numpy as np


def propagate_bsde_segments(
    value_starts: np.ndarray,
    controls: np.ndarray,
    increments: np.ndarray,
    segment_steps: np.ndarray,
    r: float,
    h: float,
    theta: np.ndarray,
) -> np.ndarray:
    """Propagate every BSDE segment forward to its right endpoint.

    Parameters
    ----------
    value_starts : np.ndarray
        Segment-start values of shape ``(m, n_paths)``.
    controls : np.ndarray
        Martingale controls of shape ``(n_steps, n_paths, dim_w)``.
    increments : np.ndarray
        Brownian increments with the same shape as controls.
    segment_steps : np.ndarray
        Positive step counts of shape ``(m,)`` summing to n_steps.
    r : float
        Finite risk-free rate in the driver ``f=-rY+theta.Z``.
    h : float
        Positive finite time step.
    theta : np.ndarray
        Finite driver control coefficient of shape ``(dim_w,)``.

    Raises
    ------
    ValueError
        If shapes disagree, segment counts are not positive integers or do not
        cover the grid, h is not positive, or any numeric input is nonfinite.

    Returns
    -------
    terminals : np.ndarray
        Forward-propagated segment endpoint values of shape ``(m, n_paths)``.
    """
    return terminals  # noqa: F821
```

### Step 6

06_construct_bermudan_targets

Goal
----
Construct the intermediate reflection and final payoff targets.

At each intermediate Bermudan date, the paper's compounding condition couples

the ending component to the next segment's continuation value and the immediate

exercise value. The final boundary follows the terminal convention rather than

an intermediate coupling. Exercise depends on the geometric basket formed from

the asset state at the corresponding segment endpoint.

Inputs

------

paths: Positive asset paths of shape (n_steps + 1, n_paths, d).

value_starts: Continuation values at the segment starts.

segment_steps: Positive grid counts for the successive segments.

strikes: Put strikes for the compounding and terminal dates.

Returns

-------

targets: Float array of shape (n_segments, n_paths).

```python
import numpy as np

def construct_bermudan_targets(
    paths: np.ndarray,
    value_starts: np.ndarray,
    segment_steps: np.ndarray,
    strikes: np.ndarray,
) -> np.ndarray:
    """Construct intermediate reflection targets and the terminal payoff.

    Parameters
    ----------
    paths : np.ndarray
        Positive paths of shape ``(n_steps + 1, n_paths, d)``.
    value_starts : np.ndarray
        Segment-start continuation values of shape ``(m, n_paths)``.
    segment_steps : np.ndarray
        Positive integer grid counts of shape ``(m,)``.
    strikes : np.ndarray
        Finite strikes of shape ``(m,)``.

    Raises
    ------
    ValueError
        If shapes disagree, paths are not positive and finite, segment counts
        are invalid, or values and strikes are nonfinite.

    Returns
    -------
    targets : np.ndarray
        Compounding and terminal targets of shape ``(m, n_paths)``.
    """
    return targets  # noqa: F821
```

### Step 7

07_compute_joint_objective

Goal
----
Evaluate the simultaneous residual objective across all segments.

The compound method enforces every intermediate condition and the final

terminal condition simultaneously. Empirical endpoint residuals are reduced

over paths for each component, then combined according to the paper's joint

objective convention.

Inputs

------

terminals: Forward-propagated endpoint values by segment and path.

targets: Intermediate compounding and final payoff targets of matching shape.

Returns

-------

loss_summary: Segment mean squared losses followed by the joint objective.

```python
import numpy as np


def compute_joint_objective(terminals: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute the segment residual losses and their joint sum.

    Parameters
    ----------
    terminals : np.ndarray
        Forward segment endpoint values of shape ``(m, n_paths)``.
    targets : np.ndarray
        Compounding and terminal targets with the same shape.

    Raises
    ------
    ValueError
        If the arrays are not matching nonempty matrices or contain nonfinite values.

    Returns
    -------
    loss_summary : np.ndarray
        Vector of length ``m + 1`` containing segment MSEs followed by the
        paper-defined joint objective.
    """
    return loss_summary  # noqa: F821
```

### Step 8

08_compute_error_certificate

Goal
----
Form the constant-free a posteriori residual certificate.

The a posteriori estimate for the fully forward compound system controls the

aggregate state, value, and control error through the discretization scale and

the complete joint residual objective. Removing the unknown multiplicative

constant leaves the paper-defined computable certificate.

Inputs

------

loss_summary: Segment losses followed by their validated joint sum.

h: Positive uniform time step.

Returns

-------

certificate: Native float containing the constant-free error certificate.

```python
import numpy as np


def compute_error_certificate(loss_summary: np.ndarray, h: float) -> float:
    """Compute the constant-free certificate from the validated objective.

    Parameters
    ----------
    loss_summary : np.ndarray
        Segment losses followed by their joint sum.
    h : float
        Positive finite uniform step size.

    Raises
    ------
    ValueError
        If loss_summary is not a finite nonnegative vector with at least one
        component and a consistent final sum, or h is not finite and positive.

    Returns
    -------
    certificate : float
        Constant-free a posteriori certificate.
    """
    return certificate  # noqa: F821
```

### Step 9

09_run_full_pipeline

Goal
----
Assemble the frozen three-segment compound-BSDE calculation.

The configured surrogate evaluates independent value and control heads for the

successive BSDE components. The complete calculation couples adjacent forward

segments at the exercise dates and reduces all residual conditions to the

constant-free a posteriori certificate. This orchestrator is the final step in

Studio.

Inputs

------

seed: Nonnegative seed controlling the complete Brownian tensor.

Returns

-------

certificate: Native float reported as the final answer.

```python
def run_full_pipeline(seed: int = 260118634) -> float:
    """Run the frozen three-segment pipeline and return its certificate.

    Parameters
    ----------
    seed : int
        Nonnegative seed controlling the complete Brownian tensor.

    Raises
    ------
    ValueError
        If seed is not a nonnegative integer.

    Returns
    -------
    certificate : float
        Constant-free a posteriori certificate for the configured instance.
    """
    return certificate  # noqa: F821
```
