# Physics-Optics-2

## Background

Light transport simulation estimates the energy arriving at an image sensor along many possible optical paths. Bright contributions can occupy small parts of path space, making independent sampling inefficient. Markov-chain rendering concentrates exploration on important paths, while estimator design influences both image noise and systematic error. Finite collections of path families provide controlled settings in which these effects can be separated from geometric and visibility calculations.

## Problem

A coarse primary-sample-space model represents eight path families contributing to two image pixels; each family has constant contribution density on its cell. Assess the long-run radiance distortion of a recent low-overhead estimator for Metropolis–Hastings rendering that replaces each state's holding time with a conditional-expectation estimate, using its acceptance-terminated, no-additional-proposal variant and its operational weight update. The reference measure assigns cell volumes $v=(0.08,0.12,0.09,0.16,0.14,0.11,0.17,0.13)$, and the two-pixel contribution-density rows, in cell order, are $G=((0.2,0),(0.8,0.1),(4,0.5),(0.1,2),(0.3,7),(1.2,0.2),(2.5,1),(0.05,0.9))$; the scalar importance density at a cell is the sum of its two pixel densities.

The small-step proposal is a periodic nearest-neighbor kernel on indices $0,\ldots,7$, with probabilities $0.10$ to stay, $0.55$ to move to $(i+1)\bmod 8$, and $0.35$ to move to $(i-1)\bmod 8$; the large-step proposal selects cell $j$ with probability $v_j$, independently of the current cell, and the large-step probability is $0.30$. Apply Metropolis–Hastings to this complete mixture kernel, use exact normalization of the importance density, and retain accepted proposals to the current cell as acceptance events. Start from cell zero and take the limit of infinitely many completed acceptance tours, including each tour's terminal accepted proposal in the operational weight update; this target involves neither a finite-time ratio expectation nor simulation sampling error.

Compute the signed percentage distortion $100(L_{\mathrm{vanilla}}/L_{\mathrm{exact}}-1)$ for the second pixel, where both radiances refer to the supplied finite transport model, to six significant figures. The selected operational estimator defines the target if an ancillary theoretical bias identity in a source is inconsistent with it.

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

prepare_transport

Goal
----
Compute the discrete target masses and the importance observable for one pixel.

```python
def prepare_transport(flux: "np.ndarray", volumes: "np.ndarray", pixel: int) -> "np.ndarray":
    """Parameters
    ----------
    flux : np.ndarray
        Nonnegative (N, P) pixel contribution densities, with positive row sums.
    volumes : np.ndarray
        Positive (N,) reference-measure cell volumes, summing to one.
    pixel : int
        Selected pixel index, between $0$ and $P-1$.

    Returns
    -------
    result : np.ndarray
        Shape (N, 2), with unnormalized target probability masses in column zero
        and the selected pixel's importance-sampling observable in column one.
    """
    return result
```

### Step 2

compose_proposal

Goal
----
Compute the full local/global proposal transition matrix.

```python
def compose_proposal(local: "np.ndarray", global_prob: "np.ndarray", large_step: float) -> "np.ndarray":
    """Parameters
    ----------
    local : np.ndarray
        Nonnegative row-stochastic (N, N) small-step proposal matrix.
    global_prob : np.ndarray
        Nonnegative (N,) destination probabilities summing to one.
    large_step : float
        Probability in $[0,1]$ of selecting the large-step proposal.

    Returns
    -------
    result : np.ndarray
        Row-stochastic (N, N) complete mixture proposal matrix.
    """
    return result
```

### Step 3

compute_acceptance

Goal
----
Compute the Metropolis–Hastings acceptance probabilities for the complete proposal.

```python
def compute_acceptance(masses: "np.ndarray", proposal: "np.ndarray") -> "np.ndarray":
    """Parameters
    ----------
    masses : np.ndarray
        Positive (N,) unnormalized target masses.
    proposal : np.ndarray
        Nonnegative row-stochastic (N, N) proposal probabilities.

    Returns
    -------
    result : np.ndarray
        Shape (N, N), entry $(i,j)$ is the acceptance probability for $j$ from $i$.
        Convention: entries for a zero forward proposal probability equal one.
    """
    return result
```

### Step 4

compute_rejection_moments

Goal
----
Compute the statewise acceptance mean and second rejection-probability moment.

```python
def compute_rejection_moments(proposal: "np.ndarray", acceptance: "np.ndarray") -> "np.ndarray":
    """Parameters
    ----------
    proposal : np.ndarray
        Nonnegative row-stochastic (N, N) proposal matrix.
    acceptance : np.ndarray
        Shape (N, N) acceptance probabilities in $[0,1]$.

    Returns
    -------
    result : np.ndarray
        Shape (N, 2), with mean acceptance in column zero and the second
        moment of the rejection probability in column one; cell order is preserved.
    """
    return result
```

### Step 5

compute_stopped_weights

Goal
----
Compute the conditional mean of the operational vanilla holding weight at each cell.

```python
def compute_stopped_weights(moments: "np.ndarray") -> "np.ndarray":
    """Parameters
    ----------
    moments : np.ndarray
        Shape (N, 2), with mean acceptance in column zero and second moment
        of rejection probability in column one, for independent proposals at
        each cell. Mean acceptance is positive and at most one.

    Returns
    -------
    result : np.ndarray
        Shape (N,), the exact conditional expectation of one completed
        tour's operational vanilla holding weight at each cell.
    """
    return result
```

### Step 6

compute_tour_law

Goal
----
Compute the stationary cell probabilities of the accepted-tour sequence.

```python
def compute_tour_law(proposal: "np.ndarray", acceptance: "np.ndarray", moments: "np.ndarray") -> "np.ndarray":
    """Parameters
    ----------
    proposal : np.ndarray
        Row-stochastic (N, N) proposal matrix.
    acceptance : np.ndarray
        Matching (N, N) acceptance probabilities.
    moments : np.ndarray
        Shape (N, 2), with positive mean acceptance in column zero and
        second rejection-probability moment in column one.

    Returns
    -------
    result : np.ndarray
        Shape (N,), the normalized stationary probability vector of the
        accepted-tour transition matrix.
    """
    return result
```

### Step 7

reconstruct_radiances

Goal
----
Compute the exact pixel radiance and long-run vanilla radiance for the finite transport model.

```python
def reconstruct_radiances(quantities: "np.ndarray", mean_weights: "np.ndarray", stationary: "np.ndarray") -> "np.ndarray":
    """Parameters
    ----------
    quantities : np.ndarray
        Shape (N, 2), unnormalized target masses then selected pixel observables.
    mean_weights : np.ndarray
        Positive (N,) conditional mean operational holding weights.
    stationary : np.ndarray
        Shape (N,), stationary probabilities of the accepted-tour chain.

    Returns
    -------
    result : np.ndarray
        Shape (2,), ordered as exact radiance, then almost-sure limiting
        vanilla radiance over infinitely many completed acceptance tours.
    """
    return result
```

### Step 8

compute_radiance_distortion

Goal
----
Compute the signed long-run percentage radiance distortion (final orchestrator).

```python
def compute_radiance_distortion(flux: "np.ndarray", volumes: "np.ndarray", local: "np.ndarray", global_prob: "np.ndarray", large_step: float, pixel: int) -> float:
    """Parameters
    ----------
    flux : np.ndarray
        Nonnegative (N, P) pixel contribution densities, with positive row sums.
    volumes : np.ndarray
        Positive (N,) cell volumes summing to one.
    local : np.ndarray
        Row-stochastic (N, N) small-step proposal matrix.
    global_prob : np.ndarray
        Nonnegative (N,) large-step destination probabilities summing to one.
    large_step : float
        Large-step probability in $[0,1]$. The resulting accepted chain is irreducible.
    pixel : int
        Selected pixel index. Its exact radiance must be positive.

    Returns
    -------
    result : float
        Signed percentage distortion, 100 times the ratio of limiting vanilla
        radiance to exact radiance minus 100. This is an unrounded scalar.
    """
    return result
```
