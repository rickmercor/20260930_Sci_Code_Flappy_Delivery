# Material_Science-Molecular_Modeling-8

## Background

Variable-weight particle simulations of rarefied gases and plasmas must periodically reduce their particle count, and naive merging distorts the velocity distribution far beyond the conserved low-order moments. Casting the reduction as a sparse nonnegative reweighting problem lets a merge conserve an arbitrary declared set of velocity and spatial moments, and even collision rates against a second species, exactly. On a small fixed particle family the whole pipeline can be audited: the conserved ledger returns at machine precision while third-order statistics reveal what merging genuinely changes.

## Problem

Variable-weight particle simulations of rarefied gases and plasmas must periodically reduce their particle count, and naive merging distorts the velocity distribution. The source paper's moment-preserving NNLS merging algorithm selects a sparse nonnegative reweighting of the existing particles through a non-negative least-squares problem, so that a declared set of velocity and spatial moments, and additionally the collision rates against a second species, are conserved exactly. Your job is to implement the merging pipeline exactly as the source specifies it and audit one merge on the fixed particle family below. The load-bearing algorithmic choices, how the particle coordinates are transformed before the moment rows are built and how the right-hand side is formed consistently with them, how the reference scales are normalized, at which stage the column scaling is computed and how it is defined, the structure of the active-set solver, in which coordinates the rate rows are evaluated and how each rate row is scaled, and how the solution vector turns back into surviving particles, are the source's; recover them from the paper. The problem is calibrated so that departing from those choices changes the reported numbers by amounts far beyond the grading tolerance.

The merged species has N = 24 particles and the fixed background species has 10 particles, built by integer constructors, all indices zero-based:
- w[i] = 0.5 + (((i+1)(i+3) + a) mod 7)/10
- v[i][c] = (((i+1)(i+2) + (c+2)(i+5) + a) mod 47)/23.5 - 1 for c = 0, 1, 2
- x[i] = (((i+2)(i+4)a + 1) mod 41)/41
- background: w2[k] = 0.8 + (((k+1)(k+2) + a) mod 5)/10 and v2[k][c] = (((k+2)(k+3) + (c+3)(k+1) + 2a) mod 31)/25 - 0.6
with a = 3 for dataset 1 and a = 5 for dataset 2. The conserved velocity moments are the ten multi-indices (0,0,0), (1,0,0), (0,1,0), (0,0,1), (2,0,0), (0,2,0), (0,0,2), (1,1,0), (1,0,1), (0,1,1) in that order, followed by the spatial powers 1 and 2. Two collision rates against the background are conserved, with total cross sections sigma_1(g) = 1/(1 + g^2) and sigma_2(g) = g/(1 + g) of the relative speed g, and reference rate coefficients equal to one. The retention threshold is eps = 1e-8. The declared deterministic solver conventions are: dual tolerance 1e-10 on the stopping test, the smallest index among gradient maximizers, least-squares subproblems by QR factorization, and a 200-iteration guard raising ValueError. The probe vector is u = (0.5, -0.25, 0.75).

Implement eight functions. bin_aggregates(w, v, x) returns the (5,) vector of the total weight, weighted mean velocity and weighted mean position, raising ValueError on shape mismatches, nonpositive weights or nonfinite input. reference_scales(w, v, x) returns the (4,) reference velocity and position scales exactly as the source defines them, raising ValueError if any scale degenerates. scaled_moment_system(w, v, x) returns the (12, N+1) scaled moment block for the declared multi-index sets in the declared order, built from the particles transformed exactly as the source prescribes, with the consistently constructed right-hand side appended as the last column. rate_rows(w, v, w2, v2) returns the (2, N+1) rate-preservation rows for the declared cross sections, evaluated in the coordinates the source requires for rate rows and scaled per row as the source prescribes, with the consistent right-hand side as the last column, expressed in the same unknown as the moment block. column_scaling(A) returns the (N,) per-column scaling factors of the stacked constraint matrix exactly as the source defines them, raising ValueError on a zero column. nnls_lawson_hanson(A, b) returns the nonnegative least-squares solution computed with the source's active-set algorithm under the declared conventions. nnls_merge(w, v, x, w2, v2, eps) performs the full merge: stack the scaled moment block and the rate rows, apply the column scaling, solve, recover the relative weights as the source prescribes, retain particles whose relative weight reaches eps, and return the (M, 5) survivors [weight, vx, vy, vz, x] in ascending original index order with the weights, velocities and positions the source assigns them. merge_audit(eps), the final step, must be assembled by calling the earlier functions: it builds both datasets from the constructors above, runs the merge on each, and returns a float64 array (2, 7) with columns: the surviving particle count; the total post-merge weight; the Euclidean norm of the pre-versus-post difference of all conserved quantities (the 12 raw moments and the 2 unscaled rates); the probe cube sum over survivors of weight times (u dot v)^3; the position cube sum of weight times x^3; and the maximum and minimum surviving weights.

Evaluate merge_audit with eps = 1e-8. All outputs are float64, finite, and deterministic: two runs on identical inputs must agree exactly. As the final answer, report the sum over the two datasets of the probe cube column (the fourth column) to six significant figures.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, -11.13, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). As numerical results, report only the merge_audit quantities for each of the two datasets — the surviving particle count, the total post-merge weight, the conserved-quantity residual norm, the probe cube sum, the position cube sum, and the maximum and minimum surviving weights — plus the final summed value. Alongside them, state in ONE sentence each the source conventions your numbers depend on - the pipeline stage order, the reference-scale and column-scaling choices, the rate-row construction, the active-set loop and survivor rule, and the conserved-versus-changed ledger - naming each choice and justifying it from the source; these single-sentence statements do not count as a pipeline summary.
Do not paste the input particle lists, full constraint matrices, per-iteration active sets, or per-particle tables.

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

bin_aggregates

Goal
----
Compute the bin aggregates of the merge group: the total weight, the weighted mean velocity and the weighted mean position, packed as a length-5 vector.

```python
import numpy as np


def bin_aggregates(w, v, x):
    """w: (N,) positive weights; v: (N, 3) velocities; x: (N,) positions.
    Returns (5,) float64: the total weight, the three components of the
    weighted mean velocity, and the weighted mean position, in that order.
    Raises ValueError on shape mismatch, nonpositive weights or nonfinite
    input."""
    return np.zeros(5)
```

### Step 2

reference_scales

Goal
----
Compute the reference velocity and position scales of the merge group exactly as the source defines them for its scaled system, packed as a length-4 vector.

```python
import numpy as np


def reference_scales(w, v, x):
    """w: (N,) weights; v: (N, 3) velocities; x: (N,) positions.
    Returns (4,) float64: the three components of the reference velocity
    scale and the reference position scale, exactly as the source defines
    them. Raises ValueError if any scale degenerates to zero."""
    return np.ones(4)
```

### Step 3

scaled_moment_system

Goal
----
Assemble the scaled moment block of the merging system for the declared multi-index sets: 10 velocity moment rows for the multi-indices (0,0,0), (1,0,0), (0,1,0), (0,0,1), (2,0,0), (0,2,0), (0,0,2), (1,1,0), (1,0,1), (0,1,1) in that order, then 2 spatial rows for powers 1 and 2, built from the particles transformed exactly as the source prescribes, with the right-hand side constructed consistently as the source's algorithm states, appended as the last column.

```python
import numpy as np


def scaled_moment_system(w, v, x):
    """w: (N,) weights; v: (N, 3) velocities; x: (N,) positions.
    Returns (12, N+1) float64: the scaled moment rows for the declared
    velocity multi-index set and spatial powers, in the declared order, with
    the consistently constructed right-hand side as the final column.
    Raises ValueError on degenerate scales."""
    return np.zeros((12, np.asarray(w).shape[0] + 1))
```

### Step 4

rate_rows

Goal
----
Assemble the two rate-preservation rows against the fixed background species for the declared cross sections sigma_1(g) = 1/(1+g^2) and sigma_2(g) = g/(1+g), with reference rate coefficients equal to one, using the particle velocities in the form the source requires for rate rows and the source's row scaling, with the consistent right-hand side as the final column, expressed in the same unknown as the scaled moment block.

```python
import numpy as np


def rate_rows(w, v, w2, v2):
    """w, v: (N,) weights and (N, 3) velocities of the merged species;
    w2, v2: (M,) weights and (M, 3) velocities of the fixed background
    species. Returns (2, N+1) float64: one row per declared cross section,
    built and scaled exactly as the source prescribes for rate preservation,
    with the consistent right-hand side as the final column. Raises
    ValueError on shape mismatch."""
    return np.zeros((2, np.asarray(v).shape[0] + 1))
```

### Step 5

column_scaling

Goal
----
Compute the per-column scaling factors of the stacked constraint matrix exactly as the source defines them, guarding against zero columns.

```python
import numpy as np


def column_scaling(A):
    """A: (m, N) stacked constraint matrix. Returns (N,) float64: the
    per-column scaling factors exactly as the source defines them.
    Raises ValueError if any column is identically zero."""
    return np.ones(np.asarray(A).shape[1])
```

### Step 6

nnls_lawson_hanson

Goal
----
Solve the nonnegative least-squares problem A y = b, y >= 0 with the source's active-set algorithm under the declared deterministic conventions: dual tolerance 1e-10 on the stopping test, the smallest index among gradient maximizers, least-squares subproblems by QR factorization, the source's step-length rule and passive-set drop rule, and a 200-iteration guard raising ValueError.

```python
import numpy as np


def nnls_lawson_hanson(A, b):
    """A: (m, N); b: (m,). Returns (N,) float64: the nonnegative
    least-squares solution computed by the source's active-set algorithm
    under the declared conventions (dual tolerance 1e-10, smallest index
    among gradient maximizers, QR-based least-squares subproblems, the
    source's step-length and drop rules, 200-iteration guard).
    Raises ValueError if an iteration limit is exceeded."""
    return np.zeros(np.asarray(A).shape[1])
```

### Step 7

nnls_merge

Goal
----
Perform the full rate-preserving merge: stack the scaled moment block and the rate rows, apply the column scaling, solve the nonnegative least-squares system, recover the relative weights as the source prescribes, retain particles whose relative weight reaches the threshold eps, and return the survivors in ascending original index order with the weights, velocities and positions the source assigns them.

```python
import numpy as np


def nnls_merge(w, v, x, w2, v2, eps):
    """w, v, x: merged-species particles; w2, v2: fixed background species;
    eps: retention threshold on relative weights. Returns (M, 5) float64:
    one row per surviving particle, [weight, vx, vy, vz, x], in ascending
    original index order, assembled exactly as the source's algorithm
    prescribes. Raises ValueError on an invalid threshold."""
    return np.zeros((1, 5))
```

### Step 8

merge_audit

Goal
----
Build the two declared datasets from the integer constructors, run the full rate-preserving merge on each with eps, and report per dataset: the surviving particle count; the total post-merge weight; the Euclidean norm of the pre-versus-post difference of all conserved quantities (the 12 raw moments and the 2 unscaled rates); the weighted cube probe sum over survivors of weight times (u dot v)^3 for the declared probe u; the weighted position cube sum of weight times x^3; and the maximum and minimum surviving weights. Assemble by calling the earlier sub-problem functions.

```python
import numpy as np


def merge_audit(eps):
    """eps: retention threshold. Builds the two declared datasets, runs the
    rate-preserving merge on each, and returns a float64 array (2, 7) with
    columns: surviving count; total post-merge weight; the norm of the
    pre-versus-post difference of the 12 raw moments and 2 unscaled rates;
    the probe cube sum of weight times (u dot v)^3; the position cube sum of
    weight times x^3; and the maximum and minimum surviving weights.
    Assembled by calling the earlier sub-problem functions. Raises
    ValueError on an invalid threshold."""
    return np.zeros((2, 7))
```
