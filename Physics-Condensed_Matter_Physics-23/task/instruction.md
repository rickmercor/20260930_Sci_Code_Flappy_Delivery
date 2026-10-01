# Physics-Condensed_Matter_Physics-23

## Background

Fock-space variational Monte Carlo estimates variational energies and natural-gradient information from discrete many-body configurations. A concentrated Born distribution can make exploration inefficient, while good chain acceptance alone does not ensure low-variance estimates. Importance reweighting allows the sampling construction to change while preserving the Born-distribution variational objective.

This task examines the coupled sampling and optimization behavior of a finite deterministic instance with a real log-linear ansatz. The supplied Hamiltonian, initial state, optimizer settings, and draw arrays define two consecutive optimization blocks.

## Problem

A finite Fock-space variational calculation uses the importance-reweighted construction with Hamiltonian-guided auxiliary sampling, an evaluation Markov kernel, an adaptive amplitude exponent, and predictive sample-space stochastic reconfiguration. For the seven-state, nine-parameter instance below, determine the first parameter after two consecutive optimization blocks. The real log-linear ansatz is $\psi_\theta(x)=s_x\exp[b_x+O_x\theta]$, where $s_x=\operatorname{sign}(\text{base_psi}_x)$, $b_x=\log|\text{base_psi}_x|$, and `O` is the supplied feature and logarithmic-derivative table.

```text
H = [[-1.25, -0.80,  0.35,  0.00,  0.00,  0.00, -0.25],
     [-0.80, -0.70, -0.55,  0.22,  0.00, -0.18,  0.00],
     [ 0.35, -0.55,  0.15, -0.72,  0.28,  0.00,  0.16],
     [ 0.00,  0.22, -0.72,  0.65, -0.40,  0.31,  0.00],
     [ 0.00,  0.00,  0.28, -0.40, -0.35, -0.63,  0.27],
     [ 0.00, -0.18,  0.00,  0.31, -0.63,  1.10, -0.48],
     [-0.25,  0.00,  0.16,  0.00,  0.27, -0.48,  0.25]]
base_psi = [0.82, -0.37, 0.145, -0.061, 0.29, -0.013, 0.51]
O = [[ 0.15, -0.40,  0.64,  0.89,  0.99,  0.92,  0.62, -1.00, 1.00],
     [ 0.75,  0.10,  0.99,  0.81, -0.26,  0.70, -0.23, -0.67, 0.44],
     [-0.30,  0.65,  0.86, -0.16, -0.93,  0.36, -0.90, -0.33, 0.11],
     [ 1.20, -0.55,  0.33, -0.95,  0.49, -0.03, -0.90,  0.00, 0.00],
     [-0.80,  0.35, -0.35, -0.71,  0.80, -0.42, -0.21,  0.33, 0.11],
     [ 0.45,  1.10, -0.87,  0.31, -0.70, -0.74,  0.63,  0.67, 0.44],
     [-0.10, -0.85, -0.98,  0.99, -0.62, -0.94,  1.00,  1.00, 1.00]]
initial_parameters = [0.30498276274429814, 0, 0, 0, 0, 0, 0, 0, 0]
cutoff = 0.20
initial_alpha = 1.35
beta = 0.47
initial_state = 3
diagonal_shift = 0.22649441768635536
predictor_coefficient = 0.13361074072574586
learning_rate = 0.22292696898140568
alpha_damping = 0.02
proposal_draws = [[0.91, 0.05, 0.95, 0.65, 0.73, 0.93, 0.70, 0.10, 0.96, 0.80, 0.40, 0.35],
                  [0.18, 0.84, 0.33, 0.99, 0.58, 0.07, 0.76, 0.44, 0.61, 0.28, 0.92, 0.15]]
accept_draws = [[0.13, 0.90, 0.09, 0.27, 0.32, 0.88, 0.02, 0.42, 0.99, 0.006, 0.75, 0.11],
                [0.61, 0.04, 0.72, 0.15, 0.49, 0.95, 0.21, 0.36, 0.08, 0.77, 0.12, 0.54]]
kernel_draws = [[0.20, 0.75, 0.46, 0.47, 0.10, 0.90, 0.30, 0.65, 0.05, 0.49, 0.40, 0.80],
                [0.47, 0.12, 0.88, 0.39, 0.50, 0.03, 0.71, 0.22, 0.95, 0.46, 0.31, 0.66]]
```

The retained graph consists of the nonzero off-diagonal links satisfying $|H_{xy}|\geq\text{cutoff}$; local energies use the complete supplied `H`. Each block consumes one row of each draw array in order, collecting one evaluation configuration at each of its 12 sequential transitions, with no additional thermalization or discarded transitions. Order categorical outcomes by increasing state index and select the first cumulative probability strictly greater than its draw; use strict inequalities for both acceptance and evaluation-kernel Bernoulli decisions, so equality takes the rejection or stay branch respectively. Initialize the preceding unscaled direction to $\mathbf d_0=\mathbf0$, hold parameters and exponent fixed within each block, and continue the same chain and optimizer state into block 2. Use binary64 arithmetic without intermediate rounding, 0-based state indices, and 1-based transition numbers; round only the final first parameter to 12 decimal places.

In `<reasoning>`, give the identities that determine the auxiliary dynamics, importance-reweighted estimator, adaptive exponent, and predictive SR update, explaining the evaluation/chain timing, the local-energy convention, and what is carried into block 2. Report the accepted transition numbers and evaluation histogram for each block, the energy pair $(\widehat E_1,\widehat E_2)$, the block-1 exponent triple $(\kappa_1,\alpha_1^\star,\alpha_1)$, the initial predictor $\mathbf p_1$, $d_{1,0}$, $\theta_{1,0}$, $\alpha_2$, and $d_{2,0}$.

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

build_retained_kernel

Goal
----
Construct the symmetric retained-magnitude kernel and its row-normalized Hamiltonian proposal from a finite real symmetric Hamiltonian, using an inclusive off-diagonal magnitude cutoff and requiring every state to retain at least one reversible neighbor.

```python
import numpy as np

def build_retained_kernel(H: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Build the retained Hamiltonian magnitudes and proposal matrix.

    Parameters
    ----------
    H : np.ndarray
        Finite real symmetric Hamiltonian with shape ``(n_states, n_states)``.
    cutoff : float
        Strictly positive finite threshold. A nonzero off-diagonal entry is retained when
        ``abs(H[x, y]) >= cutoff``.

    Returns
    -------
    retained_magnitudes : np.ndarray
        Symmetric nonnegative matrix ``W`` of retained off-diagonal magnitudes.
    proposal : np.ndarray
        Row-stochastic Hamiltonian proposal ``Q``.

    Raises
    ------
    ValueError
        If the inputs are invalid, the Hamiltonian is not symmetric, or a state has no
        retained reversible neighbor.
    """
    return np.empty_like(np.asarray(H), dtype=float), np.empty_like(np.asarray(H), dtype=float)
```

### Step 2

compute_ir_log_fields

Goal
----
Compute the auxiliary log mass, the exactly marginalized evaluation log mass, and the first two exponent-response fields of the retained stay-or-displace mixture without forming exponentially large or small masses.

```python
import numpy as np

def _logsumexp(values: np.ndarray) -> float:
    """Return a stable log-sum-exp of a nonempty finite one-dimensional array."""
    values = np.asarray(values, dtype=float)
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum), dtype=float)))

def compute_ir_log_fields(
    retained_magnitudes: np.ndarray,
    log_abs_psi: np.ndarray,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute log-domain auxiliary and evaluation fields.

    Parameters
    ----------
    retained_magnitudes : np.ndarray
        Symmetric nonnegative retained magnitude matrix ``W`` with zero diagonal and
        strictly positive row sums.
    log_abs_psi : np.ndarray
        Finite logarithms of nonzero wave-function magnitudes.
    alpha : float
        Finite auxiliary amplitude exponent in ``[0, 2]``.
    beta : float
        Finite evaluation displacement probability in ``[0, 1)``.

    Returns
    -------
    log_auxiliary : np.ndarray
        Unnormalized auxiliary log masses.
    log_evaluation : np.ndarray
        Exactly marginalized unnormalized evaluation log masses.
    exponent_score : np.ndarray
        First derivative of ``log_evaluation`` with respect to ``alpha``.
    exponent_second_ratio : np.ndarray
        Ratio of the second derivative of the unnormalized evaluation mass to that mass.

    Raises
    ------
    ValueError
        If any input is invalid or incompatible.
    """
    return (
        np.empty_like(np.asarray(log_abs_psi), dtype=float),
        np.empty_like(np.asarray(log_abs_psi), dtype=float),
        np.empty_like(np.asarray(log_abs_psi), dtype=float),
        np.empty_like(np.asarray(log_abs_psi), dtype=float),
    )
```

### Step 3

run_log_hamiltonian_chain

Goal
----
Propagate a deterministic Hamiltonian-guided Metropolis-Hastings chain in the log domain, recording each pre-state, categorical trial, acceptance probability, strict acceptance flag, and immediately mutated post-state.

```python
from numbers import Integral

import numpy as np

def _draw_categorical_strict(probabilities: np.ndarray, draw: float) -> int:
    """Select the first index whose cumulative probability is strictly greater than draw."""
    cumulative = 0.0
    last_positive = -1
    for index, probability in enumerate(np.asarray(probabilities, dtype=float)):
        if probability > 0.0:
            last_positive = index
        cumulative += float(probability)
        if cumulative > draw:
            return int(index)
    return int(last_positive)

def run_log_hamiltonian_chain(
    proposal: np.ndarray,
    log_auxiliary: np.ndarray,
    initial_state: int,
    proposal_draws: np.ndarray,
    accept_draws: np.ndarray,
) -> np.ndarray:
    """Run a sequential log-domain Metropolis-Hastings chain.

    Parameters
    ----------
    proposal : np.ndarray
        Row-stochastic proposal matrix with reversible positive support.
    log_auxiliary : np.ndarray
        Finite unnormalized auxiliary log masses.
    initial_state : int
        Zero-based state before the first supplied move.
    proposal_draws : np.ndarray
        One-dimensional categorical draws in ``[0, 1)``.
    accept_draws : np.ndarray
        One-dimensional strict-acceptance draws in ``[0, 1)``.

    Returns
    -------
    trace : np.ndarray
        Array with columns ``pre_state``, ``trial_state``, ``acceptance_probability``,
        ``accepted_flag``, and ``post_state``.

    Raises
    ------
    ValueError
        If the proposal, log masses, state, or draws are invalid or incompatible.
    """
    return np.empty((np.asarray(proposal_draws).size, 5), dtype=float)
```

### Step 4

select_evaluation_histogram

Goal
----
Select one estimator configuration from every pre-transition chain state by either staying or reusing that move's Hamiltonian trial, then combine the realized configurations into an ascending unique-state histogram.

```python
import numpy as np

def select_evaluation_histogram(
    chain_trace: np.ndarray,
    beta: float,
    kernel_draws: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the strict stay-or-displace estimator kernel and group repeats.

    Parameters
    ----------
    chain_trace : np.ndarray
        Binary64 trace with shape ``(n_moves, 5)`` and columns ``pre_state``,
        ``trial_state``, ``acceptance_probability``, ``accepted_flag``, ``post_state``.
    beta : float
        Finite displacement probability in ``[0, 1)``.
    kernel_draws : np.ndarray
        One-dimensional draws in ``[0, 1)`` with one entry per move.

    Returns
    -------
    evaluation_states : np.ndarray
        Selected estimator states in move order.
    unique_states : np.ndarray
        Distinct estimator states in ascending order.
    counts : np.ndarray
        Positive integer multiplicities corresponding to ``unique_states``.

    Raises
    ------
    ValueError
        If the trace, probability, or draws are invalid or inconsistent.
    """
    return (
        np.empty(np.asarray(chain_trace).shape[0], dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )
```

### Step 5

compute_adaptive_reweighting

Goal
----
Compute complete-Hamiltonian local energies, stable grouped Born-to-evaluation weights, and the damped projected update of the auxiliary amplitude exponent from residual-weighted evaluation-mass derivatives.

```python
import numpy as np

def _logsumexp(values: np.ndarray) -> float:
    """Return a stable log-sum-exp."""
    values = np.asarray(values, dtype=float)
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum), dtype=float)))

def _signed_exponential_sum(log_magnitudes: np.ndarray, signs: np.ndarray) -> float:
    """Evaluate a finite signed exponential sum with cancellation-aware log arithmetic."""
    log_magnitudes = np.asarray(log_magnitudes, dtype=float)
    signs = np.asarray(signs, dtype=float)
    positive = log_magnitudes[signs > 0.0]
    negative = log_magnitudes[signs < 0.0]
    log_positive = _logsumexp(positive) if positive.size else -np.inf
    log_negative = _logsumexp(negative) if negative.size else -np.inf
    if not np.isfinite(log_negative):
        return float(np.exp(log_positive))
    if not np.isfinite(log_positive):
        return -float(np.exp(log_negative))
    if log_positive == log_negative:
        return 0.0
    if log_positive > log_negative:
        log_abs = log_positive + float(np.log(-np.expm1(log_negative - log_positive)))
        output_sign = 1.0
    else:
        log_abs = log_negative + float(np.log(-np.expm1(log_positive - log_negative)))
        output_sign = -1.0
    return output_sign * float(np.exp(log_abs))

def compute_adaptive_reweighting(
    H: np.ndarray,
    amplitude_signs: np.ndarray,
    log_abs_psi: np.ndarray,
    log_evaluation: np.ndarray,
    exponent_score: np.ndarray,
    exponent_second_ratio: np.ndarray,
    unique_states: np.ndarray,
    counts: np.ndarray,
    alpha: float,
    damping: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute grouped weights, local energies, and the next exponent.

    Parameters
    ----------
    H : np.ndarray
        Finite real symmetric complete Hamiltonian.
    amplitude_signs : np.ndarray
        Vector containing exactly ``-1`` or ``+1`` for every state.
    log_abs_psi : np.ndarray
        Finite log magnitudes for every state.
    log_evaluation : np.ndarray
        Finite unnormalized evaluation log masses.
    exponent_score : np.ndarray
        First exponent derivative of the evaluation log mass.
    exponent_second_ratio : np.ndarray
        Second derivative of the evaluation mass divided by that mass.
    unique_states : np.ndarray
        Strictly increasing sampled state indices.
    counts : np.ndarray
        Positive integer multiplicities for the sampled states.
    alpha : float
        Current exponent in ``[0, 2]``.
    damping : float, default=0.02
        Finite exponent damping factor in ``(0, 1]``.

    Returns
    -------
    grouped_weights : np.ndarray
        Normalized grouped importance weights.
    local_energies : np.ndarray
        Complete-Hamiltonian local energies for ``unique_states``.
    diagnostics : np.ndarray
        ``[weighted_energy, curvature, projected_alpha, next_alpha]``.

    Raises
    ------
    ValueError
        If inputs are invalid, incompatible, or a local energy is not finite.
    """
    return (
        np.empty_like(np.asarray(unique_states), dtype=float),
        np.empty_like(np.asarray(unique_states), dtype=float),
        np.empty(4, dtype=float),
    )
```

### Step 6

solve_predictive_sample_space_sr

Goal
----
Solve the shifted predictive stochastic-reconfiguration system in sampled-configuration space, adding the projection of the preceding unscaled direction before any learning-rate scaling and returning the new unscaled parameter direction.

```python
import numpy as np

def solve_predictive_sample_space_sr(
    log_derivatives: np.ndarray,
    unique_states: np.ndarray,
    grouped_weights: np.ndarray,
    local_energies: np.ndarray,
    previous_direction: np.ndarray,
    predictor_coefficient: float,
    diagonal_shift: float,
) -> np.ndarray:
    """Compute a predictive shifted SR direction in sample space.

    Parameters
    ----------
    log_derivatives : np.ndarray
        Finite state-by-parameter logarithmic derivative matrix.
    unique_states : np.ndarray
        Strictly increasing sampled state indices.
    grouped_weights : np.ndarray
        Strictly positive normalized weights for ``unique_states``.
    local_energies : np.ndarray
        Finite local energies for ``unique_states``.
    previous_direction : np.ndarray
        Finite preceding unscaled direction with one entry per parameter.
    predictor_coefficient : float
        Finite nonnegative multiplier of the preceding direction.
    diagonal_shift : float
        Finite strictly positive SR shift.

    Returns
    -------
    direction : np.ndarray
        New unscaled predictive SR direction.

    Raises
    ------
    ValueError
        If inputs are invalid, incompatible, or the sample-space solve is singular or non-finite.
    """
    return np.empty(np.asarray(log_derivatives).shape[1], dtype=float)
```

### Step 7

run_two_block_ir_sr_update

Goal
----
Run two sequential importance-reweighted sampling-and-optimization blocks by composing every preceding public operation, initializing block 1 with a zero preceding unscaled SR direction, carrying all mutated state into block 2, and returning the first final parameter.

```python
from numbers import Integral

import numpy as np

def run_two_block_ir_sr_update(
    H: np.ndarray,
    base_psi: np.ndarray,
    log_derivatives: np.ndarray,
    initial_parameters: np.ndarray,
    cutoff: float,
    initial_alpha: float,
    beta: float,
    initial_state: int,
    proposal_draws: np.ndarray,
    accept_draws: np.ndarray,
    kernel_draws: np.ndarray,
    diagonal_shift: float,
    predictor_coefficient: float,
    learning_rate: float,
    alpha_damping: float = 0.02,
) -> float:
    """Run two coupled sampling, reweighting, adaptation, and predictive-SR blocks.

    The implementation must call ``build_retained_kernel``, ``compute_ir_log_fields``,
    ``run_log_hamiltonian_chain``, ``select_evaluation_histogram``,
    ``compute_adaptive_reweighting``, and ``solve_predictive_sample_space_sr``. Block 1
    uses a zero preceding unscaled direction; block 2 uses the direction from block 1.

    Parameters
    ----------
    H : np.ndarray
        Finite real symmetric complete Hamiltonian.
    base_psi : np.ndarray
        Finite nonzero signed reference amplitudes.
    log_derivatives : np.ndarray
        Finite state-by-parameter matrix used by the log-linear ansatz and SR.
    initial_parameters : np.ndarray
        Finite initial parameter vector.
    cutoff : float
        Positive inclusive retained-connection threshold.
    initial_alpha : float
        Initial auxiliary exponent in ``[0, 2]``.
    beta : float
        Evaluation displacement probability in ``[0, 1)``.
    initial_state : int
        Zero-based chain state before block 1.
    proposal_draws : np.ndarray
        Array with shape ``(2, n_moves)`` containing categorical draws in ``[0, 1)``.
    accept_draws : np.ndarray
        Array with shape ``(2, n_moves)`` containing strict-acceptance draws in ``[0, 1)``.
    kernel_draws : np.ndarray
        Array with shape ``(2, n_moves)`` containing estimator-kernel draws in ``[0, 1)``.
    diagonal_shift : float
        Positive shifted-SR regularization.
    predictor_coefficient : float
        Nonnegative multiplier of the preceding unscaled direction.
    learning_rate : float
        Positive parameter-update scale applied after each SR solve.
    alpha_damping : float, default=0.02
        Exponent damping factor in ``(0, 1]``.

    Returns
    -------
    final_component : float
        First parameter component after block 2.

    Raises
    ------
    ValueError
        If any input violates this or a preceding step's contract.
    """
    return 0.0
```
