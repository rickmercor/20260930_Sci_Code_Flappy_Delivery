# Chemistry-Quantum_Chemistry-24

## Background

## Occupation-space propagation

A fixed-spin Hubbard basis consists of binary occupation patterns. Stochastic propagation remains within that sector by either retaining the current pattern or moving one electron to an empty neighboring site. State-dependent channel probabilities control variance, while signed amplitudes preserve the fermionic operator action.

The propagation convention must remain consistent along every path. Fermionic parity is especially important on edges that cross occupied orbitals in the declared ordering; inconsistent signs can corrupt individual trajectories even when some aggregate observables possess gauge-related invariances.

## Bias and matched references

The direct and midway estimators reorganize the same finite imaginary-time propagation but have different statistical behavior. One is dominated by rare-event fluctuations, while the nonlinear construction of the other creates a leading finite-sample bias. Jackknife bias correction combines estimates from related sample blocks to suppress that leading term without changing the underlying trajectory ensemble.

Sampling error must be measured against a deterministic reference representing the same finite-step approximation. This separates stochastic error from imaginary-time discretization error and makes the comparison of estimator RMSEs meaningful.

## Problem

Stochastic Hubbard-trace estimators trade rare-event variance against nonlinear finite-sample bias. Reproduce the article's direct, Sigma-MTP, and jackknife-2 comparison for the benchmark below and determine the percentage RMSE improvement obtained by the corrected estimator.

Use ring edges \([(0,1),(1,2),(2,3),(3,0)]\), \(N_\uparrow=2\), \(N_\downarrow=1\), \(t=1\), \(U=2\), and \(\Delta\beta=0.05\). Order the 24 occupations with lexicographic up-site combinations in the outer loop and lexicographic down-site combinations in the inner loop, using orbitals \((0\uparrow,\ldots,3\uparrow,0\downarrow,\ldots,3\downarrow)\). With NumPy imported as `np`, run `np.random.default_rng(20260821).random((16,24,64,12))` once on axes (replicate, start, sample, step); the full paths use all 12 steps, while the midway calculation reuses the first six draws and consecutive 32-sample halves.

Recover from the cited article and its QMC appendix the finite-step propagation rule, channel weights and signed amplitudes, the two partition estimators, the jackknife-2 correction, and the matching deterministic reference. For the seeded inverse-CDF experiment, build each occupation's channel list with three nested loops: the outer loop runs over the listed edges in their supplied row order, the middle loop over spin 0 then spin 1, and the inner loop over \(i\to j\) then \(j\to i\). The retention channel occupies the first interval and the hop channels follow it in that same order; use left-closed, right-open channel intervals and the declared spin-orbital order for fermionic parity.

Measure all three population RMSEs against the exact reference associated with the same 12-step finite propagator and report \(100(\mathrm{RMSE}_{\rm direct}-\mathrm{RMSE}_{\rm JK})/\mathrm{RMSE}_{\rm direct}\). For this task, the following compact audit is mandatory and is the complete set of determining values permitted by the output block: the basis size and its first and last occupations; the first occupation's ordered moves, channel normalization, and the parity of its up-spin closing-edge hop; the one-step symmetry residual and exact reference; the first-replicate direct estimate, Sigma-MTP triplet, and jackknife value; and the signed mean offset and population RMSE of each estimator. Briefly identify the article section or appendix supporting the recovered propagation, estimator, correction, and reference conventions. Report ordinary audit scalars to at least ten digits after the decimal point, using scientific notation for a roundoff-scale residual; report the final percentage to at least eight digits after the decimal point.

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

hubbard_hop_channels

Goal
----
Construct ordered channel data for one fixed-spin occupation.

```python
import numpy as np


def hubbard_hop_channels(
    occupation: np.ndarray,
    edges: np.ndarray,
    hopping: float,
    interaction: float,
    delta_beta: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Construct ordered channel data for one fixed-spin occupation.

    Preserve the supplied edge order and return the directed move table, its
    nonnegative source-protocol coefficients, and the scalar channel
    normalization. The input arrays must not be mutated.

    Returns
    -------
    moves : np.ndarray
        Integer array of shape (n_moves, 3). Each row is
        (spin index 0 or 1, source site, destination site), listed in the
        channel scan order defined by the task.
    coefficients : np.ndarray
        Float array of shape (n_moves,), aligned row by row with moves.
    normalization : float
        The scalar channel normalization. When the occupation admits no
        allowed hop, return arrays of shape (0, 3) and (0,) with
        normalization 1.0.
    """
    return result
```

### Step 2

sample_hubbard_step

Goal
----
Advance one Hubbard occupation using one supplied inverse-CDF draw.

```python
import numpy as np


def sample_hubbard_step(
    occupation: np.ndarray,
    moves: np.ndarray,
    coefficients: np.ndarray,
    normalization: float,
    interaction: float,
    delta_beta: float,
    uniform: float,
) -> tuple[np.ndarray, float]:
    """Advance one occupation with one supplied inverse-CDF draw.

    Apply the source single-step convention to the ordered channel data and
    return the resulting binary occupation and signed amplitude. Do not mutate
    the supplied occupation or channel arrays.

    moves is an integer array of shape (n_moves, 3) whose rows are
    (spin index 0 or 1, source site, destination site), and coefficients is
    the aligned float array of shape (n_moves,); both are in the channel scan
    order defined by the task.
    """
    return result
```

### Step 3

propagate_hubbard_walkers

Goal
----
Propagate fixed-spin Hubbard walkers through a supplied matrix of random draws.

```python
import numpy as np


def propagate_hubbard_walkers(
    initial_states: np.ndarray,
    edges: np.ndarray,
    hopping: float,
    interaction: float,
    delta_beta: float,
    uniforms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate fixed-spin walkers through a supplied draw matrix.

    Compose hubbard_hop_channels and sample_hubbard_step at every draw. Return
    final occupation rows and cumulative signed amplitudes without mutating
    any input array.
    """
    return result
```

### Step 4

direct_partition_estimate

Goal
----
Evaluate the source direct partition-function estimator for a trajectory ensemble.

```python
import numpy as np


def direct_partition_estimate(
    basis_states: np.ndarray,
    final_states: np.ndarray,
    path_amplitudes: np.ndarray,
) -> float:
    """Evaluate the source direct estimator for a trajectory ensemble.

    Axis 0 indexes starting basis configurations and axis 1 their independent
    trajectories. Return one finite scalar without mutating the trajectory
    arrays.
    """
    return result
```

### Step 5

midway_amplitude_samples

Goal
----
Assemble sample-major midway transition-amplitude data.

```python
import numpy as np


def midway_amplitude_samples(
    basis_states: np.ndarray,
    final_states: np.ndarray,
    path_amplitudes: np.ndarray,
) -> np.ndarray:
    """Assemble sample-major midway transition-amplitude data.

    Return an array with axes (sample, target, source) using the supplied basis
    ordering and signed path amplitudes. Preserve the source-target convention
    and do not mutate the input arrays.
    """
    return result
```

### Step 6

sigma_mtp_estimates

Goal
----
Evaluate the source nonlinear estimator on a complete sample block and its two consecutive halves.

```python
import numpy as np


def sigma_mtp_estimates(amplitude_samples: np.ndarray) -> np.ndarray:
    """Return the source nonlinear estimates for the full block and its two consecutive halves, in that order."""
    return result
```

### Step 7

jackknife2_mtp

Goal
----
Apply the source second-order jackknife correction to an ordered estimator triplet.

```python
import numpy as np


def jackknife2_mtp(estimates: np.ndarray) -> tuple[float, float]:
    """Return the source bias-corrected estimate and its leading-bias estimate from the ordered three-value input."""
    return result
```

### Step 8

hubbard_rmse_reduction

Goal
----
Compose the seven precursor operations and measure the corrected estimator’s population-RMSE improvement.

```python
import numpy as np


def hubbard_rmse_reduction(
    n_sites: int,
    n_up: int,
    n_down: int,
    edges: np.ndarray,
    hopping: float,
    interaction: float,
    delta_beta: float,
    uniforms: np.ndarray,
) -> float:
    """Compose all seven precursor functions and return the corrected estimator's percentage population-RMSE change."""
    return result
```
