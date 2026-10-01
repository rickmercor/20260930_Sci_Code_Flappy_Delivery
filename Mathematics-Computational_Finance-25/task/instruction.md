# Mathematics-Computational_Finance-25

## Background

The joint calibration of an index smile and a volatility-index smile reduces, on a
finite state space, to finding a discrete coupling that satisfies several families of affine
constraints at once: one row per prescribed marginal, and one martingale row per state
carrying enough mass. When the prescribed marginals are incompatible with the martingale
rows, no coupling satisfies everything, and a solver working to a finite budget must decide
which family it holds tight and which absorbs the mismatch.

Two standard ingredients are relevant. Cyclically projecting onto each affine row in turn,
in the Kullback-Leibler geometry, is the classical iterative-proportional-fitting idea: for a
row whose coefficient vector is an indicator, the projection is a single multiplicative
rescaling of that row. Relaxing a family instead, and driving it with the gradient of a
quadratic term on its residuals under a multiplicative update, is the entropic mirror-descent
counterpart. A scheme that mixes the two assigns different priorities to different constraint
families, and the allocation of residual it produces is a property of the scheme rather than
of the constraint set.

Two facts about martingale couplings matter here. A martingale coupling cannot reduce
dispersion, so prescribing a target marginal more concentrated than the source makes the
system infeasible. And conditional Jensen applied to the active martingale rows turns that
observation into a computable certificate, bounding the target dispersion below by a
source-side quantity restricted to those rows, with no optimiser involved.

## Problem

A recent line of work on joint calibration of index and volatility smiles studies what a
finite-budget solver actually sacrifices when the affine system it is handed is
incompatible. The setting is a discrete coupling on a common one-dimensional grid,
constrained by two prescribed marginals together with a martingale row at each grid state
carrying enough mass, and the question is which constraint family absorbs the
incompatibility when the two families cannot be met at once. The source introduces a
priority-split sweep scheme that assigns the two constraint families different
priorities: one is held as a tight constraint and corrected by capped exponential tilts,
the other is relaxed and driven by a soft update. Your task is to run that scheme on one
prescribed instance and report how badly the marginal family ends up violated.

Use the following configuration.

- Grid: x_j = 0.25 + 0.0375 j for j = 0, 1, ..., 40, so 41 nodes spanning 0.25 to 1.75.
- Reference centre: 1.0.
- Source marginal: the Gaussian kernel exp(-(x - 1.0)^2 / (2 s^2)) with s = 0.15,
  evaluated on the grid and normalised to unit total mass.
- Target marginal: the same construction with s = 0.15 * sqrt(0.70).
- A martingale row is imposed at each grid state whose source mass is at least 1 percent
  of the largest source mass on the grid.
- Starting iterate: the product of the two prescribed marginals, normalised.
- Sweep budget: 80 sweeps.
- Cap on the cumulative log-tilt magnitude of a capped-tilt correction: 0.02.
- Cap on the log magnitude of the soft update: 1.0.
- Report the median over the final 10 percent of sweeps.

Compute the largest absolute violation of the marginal family at the end of that run,
measured as the worst deviation of either marginal of the coupling from its prescribed
value. In your reasoning, also report three scalars: the corresponding largest absolute
violation of the martingale family on the active states, the marginal-family violation
the same run gives at a budget of 160 sweeps instead of 80, and the largest absolute
entry of the soft-family gradient evaluated at the starting iterate. Quote each to at
least six significant figures. Your final answer must be a single number: the
marginal-family violation at 80 sweeps.

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
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
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

discrete_gaussian_weights

Goal
----
Return the normalised discrete weight vector that an unnormalised Gaussian kernel induces on a prescribed one-dimensional grid. This supplies the two prescribed marginals of the source's controlled feasibility instance.

```python
import numpy as np


def discrete_gaussian_weights(x, sigma, centre):
    """Return the normalised discrete weight vector that an unnormalised Gaussian kernel induces on a prescribed one-dimensional grid. This supplies the two prescribed marginals of the source's controlled feasibility instance.

    Returns
    -------
    ndarray of shape (n,), float64: the normalised weights, summing to one.
    """
    return np.zeros(np.asarray(x, dtype=float).shape, dtype=float)
```

### Step 2

discrete_second_moment

Goal
----
Return the mass-weighted mean squared deviation of a grid about a given centre, using a weight vector that need not already be normalised.

```python
import numpy as np


def discrete_second_moment(x, w, centre):
    """Return the mass-weighted mean squared deviation of a grid about a given centre, using a weight vector that need not already be normalised.

    Returns
    -------
    float: the mass-weighted mean squared deviation about centre.
    """
    return 0.0
```

### Step 3

active_state_mask

Goal
----
Return the indicator of the grid states whose marginal mass reaches a prescribed fraction of the largest mass on the grid. Only these states carry a martingale row in the source's thresholded affine system.

```python
import numpy as np


def active_state_mask(w, mass_frac):
    """Return the indicator of the grid states whose marginal mass reaches a prescribed fraction of the largest mass on the grid. Only these states carry a martingale row in the source's thresholded affine system.

    Returns
    -------
    ndarray of shape (n,), float64: 1.0 on an active state, 0.0 otherwise.
    """
    return np.zeros(np.asarray(w, dtype=float).shape, dtype=float)
```

### Step 4

jensen_feasibility_gap

Goal
----
Return the slack in the conditional-Jensen inequality that any coupling meeting both prescribed marginals and the martingale rows on the active states would have to satisfy. A negative value certifies that the thresholded affine system is infeasible.

```python
import numpy as np


def jensen_feasibility_gap(x, w_source, w_target, mask, centre):
    """Return the slack in the conditional-Jensen inequality that any coupling meeting both prescribed marginals and the martingale rows on the active states would have to satisfy. A negative value certifies that the thresholded affine system is infeasible.

    Returns
    -------
    float: target side minus source side. Negative certifies infeasibility.
    """
    return 0.0
```

### Step 5

product_coupling

Goal
----
Return the product coupling of two prescribed marginals, normalised to unit total mass. This is the starting iterate the source prescribes for every scheme it compares.

```python
import numpy as np


def product_coupling(w_source, w_target):
    """Return the product coupling of two prescribed marginals, normalised to unit total mass. This is the starting iterate the source prescribes for every scheme it compares.

    Returns
    -------
    ndarray of shape (n, m), float64: the normalised product coupling.
    """
    return np.zeros((np.asarray(w_source, dtype=float).size,
                     np.asarray(w_target, dtype=float).size), dtype=float)
```

### Step 6

capped_marginal_tilt

Goal
----
Apply the exponential tilt that drives one entry of one marginal of a coupling towards its prescribed mass, with the log-tilt magnitude limited by a cap supplied by the caller. `axis` names the axis that is SUMMED OVER to form the marginal being corrected, exactly as in numpy: axis=1 corrects an entry of P.sum(axis=1), so it rescales the row P[index, :]; axis=0 corrects an entry of P.sum(axis=0), so it rescales the column P[:, index]. A slice carrying no mass is returned unchanged.

```python
import numpy as np


def capped_marginal_tilt(P, axis, index, target, cap):
    """Apply the exponential tilt that drives one entry of one marginal of a coupling towards its prescribed mass, with the log-tilt magnitude limited by a cap supplied by the caller. `axis` names the axis that is SUMMED OVER to form the marginal being corrected, exactly as in numpy: axis=1 corrects an entry of P.sum(axis=1), so it rescales the row P[index, :]; axis=0 corrects an entry of P.sum(axis=0), so it rescales the column P[:, index]. A slice carrying no mass is returned unchanged.

    Returns
    -------
    ndarray with the same shape as P, float64: the coupling after the tilt. With axis=1 only P[index, :] changes; with axis=0 only P[:, index] changes.
    """
    return np.array(P, dtype=float, copy=True)
```

### Step 7

martingale_soft_gradient

Goal
----
Return the gradient of the quadratic penalty on the martingale rows of the active states, evaluated at the current coupling. Inactive states contribute nothing.

```python
import numpy as np


def martingale_soft_gradient(P, x, mask, penalty):
    """Return the gradient of the quadratic penalty on the martingale rows of the active states, evaluated at the current coupling. Inactive states contribute nothing.

    Returns
    -------
    ndarray with the same shape as P, float64: the penalty gradient.
    """
    return np.zeros(np.asarray(P, dtype=float).shape, dtype=float)
```

### Step 8

hybrid_sweep

Goal
----
Advance the coupling by one sweep of the priority-split calibration scheme the source introduces, in which the two marginal families and the martingale family are given different priorities. The arrangement within a sweep, and how the relaxed family is driven, are fixed by the source.

```python
import numpy as np


def hybrid_sweep(P, x, w_source, w_target, mask, hard_cap, penalty, base_step, soft_cap):
    """Advance the coupling by one sweep of the priority-split calibration scheme the source introduces, in which the two marginal families and the martingale family are given different priorities. The arrangement within a sweep, and how the relaxed family is driven, are fixed by the source.

    Returns
    -------
    ndarray with the same shape as P, float64: the coupling after one sweep, normalised to unit total mass.
    """
    return np.array(P, dtype=float, copy=True)
```

### Step 9

residual_pair

Goal
----
Return the worst absolute violation of the marginal family and the worst absolute violation of the martingale family on the active states, for a given coupling.

```python
import numpy as np


def residual_pair(P, x, w_source, w_target, mask):
    """Return the worst absolute violation of the marginal family and the worst absolute violation of the martingale family on the active states, for a given coupling.

    Returns
    -------
    ndarray of shape (2,), float64: the marginal-family violation followed by the martingale-family violation.
    """
    return np.zeros(2, dtype=float)
```

### Step 10

finite_budget_priority_audit

Goal
----
Run the whole audit on the prescribed instance for the given sweep budget by calling every earlier step function and using its output, and return the reported diagnostics. The instance is the one prescribed in the problem statement (41-node grid 0.25 + 0.0375 j, centre 1.0, source width 0.15, target width 0.15 sqrt(0.70), one percent mass threshold, capped-tilt cap 0.02, soft-update log cap 1.0, median over the final ten percent of sweeps), and the soft update uses the source's quadratic weight and base step. Entry 5, the active source dispersion, is the unnormalised sum over the active states of the source mass times the squared deviation from the centre, the right-hand side of the conditional-Jensen bound. Entry 6, the first tilt's mass shift, is measured on the coupling after one sweep from the product start: the absolute deviation of row 0's mass from its prescribed source mass before, minus after, one capped tilt of row 0 (axis=1, index 0, cap 0.02) applied to that coupling. Entry 7 is the largest absolute entry of the soft-family gradient at the product start.

```python
import numpy as np


def finite_budget_priority_audit(n_sweeps):
    """Run the whole audit on the prescribed instance for the given sweep budget by calling every earlier step function and using its output, and return the reported diagnostics.

    Returns
    -------
    ndarray of shape (8,), float64: the marginal-family violation, the martingale-family violation, the feasibility slack, the number of active states, the target dispersion, the active source dispersion (the unnormalised active-restricted sum), the first tilt's mass shift (as defined in the step description), and the largest initial soft-gradient magnitude.
    """
    return np.zeros(8, dtype=float)
```
