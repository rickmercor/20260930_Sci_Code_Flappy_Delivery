# Mathematics-Numerical_Linear_Algebra-35

## Background

Multi-way measurements of workforce or industry conditions are frequently collected under tight structural assumptions, because full observation is expensive and only a fraction of cells is ever actually rated or reported. A common assumption is that the measurement factors cleanly along its axes, which turns completion into recovering a handful of per-axis scalars rather than every individual cell, but only when the observed pattern is rich enough to fix those factors uniquely.

When two such partially observed measurements share an axis, such as the same set of industries appearing in both a workforce dataset and a separate firm-level dataset, a second dataset that is better sampled along one axis can be used to fill gaps in a dataset that is poorly sampled along that same axis. This is done by learning a single linear map that connects the two measurements through their shared axis and recovering both measurements jointly with that map, rather than completing them independently and fusing the results afterward. Because the resulting objective mixes data-fit terms with rank-reducing penalties and is bilinear rather than jointly convex, tractability depends on the objective decomposing into blocks that are each convex on their own.

Downstream of any such recovered measurement, operations settings that treat certification as a decaying resource face a different kind of problem: a worker's qualification in a skill erodes without upkeep and is restored only through training that competes with other constraints for the same limited hours, so meeting a hard compliance deadline across an entire workforce becomes a multi-period resource allocation problem rather than a one-time assignment.

## Problem

A regional workforce board is building a three-part decision pipeline for one manufacturing site.

Stage one produces a task-exposure landscape: a three-way array indexed by six supplier industries (i=1..6), four job functions (f=1..4), and three AI-abstraction rungs (a=1..3, ordered from rung 1, functions with no AI-assisted tasks, to rung 3, functions whose tasks are mostly automated). The array is modeled as multiplicatively separable: each cell equals the product of one industry factor, one function factor, and one rung factor, with no interaction terms. Seventeen of the 72 cells are rated: (1,1,2)=0.4864, (1,1,3)=0.5472, (1,2,2)=0.7296, (1,3,3)=0.9576, (2,1,2)=0.3584, (2,2,2)=0.5376, (3,1,3)=0.7416, (3,2,1)=1.3596, (3,2,2)=0.9888, (3,2,3)=1.1124, (3,3,2)=1.1536, (3,4,2)=0.9064, (4,1,1)=0.8976, (4,2,2)=0.9792, (4,3,3)=1.2852, (5,2,2)=0.6816, (6,1,2)=0.5120. These values are exact and consistent with the separable model, and this sampling pattern determines all 72 cells uniquely up to two compensating scalars that cancel in every cell value. Complete the full 6×4×3 array; the sampling is sparse and some (industry, function) pairs are hit only once, so the completion must exploit the separable structure globally rather than fit axis factors slice by slice. Use the recursive completion that resolves one axis at a time, each level reducing the array's order by one.

The completed array is then fused with a second dataset: a small-business-health matrix over the same six industries and four ordered firm-size tiers (tier 1, the smallest sole-proprietor scale, through tier 4, mid-sized firms). Only 12 of 24 cells are reported, with measurement noise: (1,3)=0.5971, (1,4)=0.6371, (2,2)=0.6107, (2,4)=0.6874, (3,3)=0.5769, (3,4)=0.6285, (4,1)=0.6851, (4,3)=0.7550, (5,2)=0.6332, (5,4)=0.7840, (6,3)=0.5173, (6,4)=0.5639. The landscape and the health matrix share their industry mode. Flatten the completed landscape along that mode into a 6×12 matrix, with six industries as rows and the twelve (function, rung) pairs as columns in any fixed order (the order is immaterial), and model this flattening as approximately equal to the health matrix times the transpose of an unknown 12×4 linear coupling operator. Recover the denoised landscape, the completed health matrix, and the coupling operator jointly, minimizing one penalized objective with six terms: (i) one half the squared Frobenius misfit of the landscape to the stage-1 completed array, over all 72 cells; (ii) one half the squared Frobenius misfit of the health matrix to the 12 reported cells only; (iii) 0.20 times one half the squared Frobenius norm of the coupling-relation residual; (iv) 0.10 times one half the squared Frobenius norm of the coupling operator, as ridge regularization; (v) 0.20 times the nuclear norm of the industry-mode flattening of the landscape; (vi) 0.20 times the nuclear norm of the health matrix. The objective is bilinear in the health matrix and the coupling operator, so it is not jointly convex, but it is convex in each of the three blocks separately. Solve it by proximal block-coordinate iteration, taking each block's step size below the reciprocal of that block's Lipschitz constant; on this instance the critical point reached is insensitive to the starting point and to the step sizes, so iterate to convergence and read off the recovered 6×4 health matrix, whose entries all land strictly inside the unit interval.

That health matrix seeds the third stage. The site's shop floor has six workers, one recruited from each industry's labor pool, so worker w carries industry w's capability signature; its skill ladder has four rungs graded on the same ordinal scale as the firm-size tiers, rung 1 at sole-proprietor scale and rung 4 at enterprise scale. Reading the two ordinal ladders rank-for-rank, the health score of (sector, tier) becomes the initial capability level of the worker from that sector at that skill rung, giving a 6-worker by 4-skill matrix of levels entering shift 1.

A worker is eligible to work a skill only while their level in it is at least 0.60. Over shifts t=1..24, within each shift a worker's level in skill k is first multiplied by retention factor 1 − d(k), then raised by 0.05 per hour trained on skill k that shift, with decay rates d(1)=0.012, d(2)=0.030, d(3)=0.020, d(4)=0.008 per shift; levels are confined to [0,1]. No worker may train more than 3.0 hours total per shift across all four skills, and no skill may receive more than 3.5 total instructor-hours per shift across all six workers. Every worker must be certified in every skill at the end of every shift from shift 8 through shift 24 inclusive; training hours may be fractional and split freely across skills and shifts.

In your reasoning, report the rule that selects which axis is resolved at each level of the stage-one recursion and the resulting order of the three axes; the condition on the observed sampling pattern that makes that completion exact and unique, together with the assumptions under which the guarantee holds; the guarantee that makes the stage-two block iteration well posed, with its conditions and its conclusion; the recovered 6×4 health matrix; and which initial worker-skill levels start below the certification threshold. Report, to four decimal places, the minimum total training hours, summed over all workers, all four skills, and all 24 shifts, needed to satisfy the coverage mandate under both capacity limits.

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

01_build_axis_constraint_matrix

Goal
----
Build the pairwise homogeneous constraint matrix that a sparse, partially observed, multiplicatively separable N-way array imposes on the factor pattern of the axes complementary to one removed axis.

```python
import numpy as np

def build_axis_constraint_matrix(obs: dict, axis: int) -> np.ndarray:
    """Build the pairwise homogeneous constraint matrix for one removed axis.

    Parameters
    ----------
    obs : dict
        Mapping from N-tuple integer indices to observed float values of a
        sparsely sampled array assumed to be an outer product of one factor
        vector per axis (no interaction terms).
    axis : int
        Which index position (0 <= axis < N) to remove; two observations are
        paired when they agree on this axis.

    Returns
    -------
    B : np.ndarray
        Array of shape (n_pairs, n_complementary): n_complementary is the
        number of distinct complementary index tuples in obs, columns ordered
        by the ascending sorted complementary tuple; rows follow the pair
        enumeration order in the module docstring, each oriented so that its
        entry in the column of the later observation's complementary tuple is
        positive for positive observed values, and are not rescaled. A one-dimensional empty array of
        shape (0,) when no removed-axis value has two or more observations.

    Raises
    ------
    ValueError
        If obs is empty, index tuples differ in length, or axis is out of range.
    """
    return None
```

### Step 2

02_select_recursion_axis

Goal
----
Select which axis of a sparsely observed, multiplicatively separable N-way array is resolved next in the recursive completion, from the observed cells alone.

```python
def select_recursion_axis(obs: dict) -> int:
    """Choose the axis to resolve next from the observed cells.

    Parameters
    ----------
    obs : dict
        Mapping from N-tuple integer indices (N >= 2) to observed float values
        of a sparsely sampled, multiplicatively separable array.

    Returns
    -------
    axis : int
        Zero-based position of the axis selected for resolution, following the
        tie conventions in the module docstring.

    Raises
    ------
    ValueError
        If obs is empty, has index tuples of fewer than two entries, or no axis
        has a constraint matrix with at least one row.
    """
    return None
```

### Step 3

03_extract_complementary_pattern

Goal
----
Extract, up to scale, the factor pattern shared by the axes complementary to one removed axis of a sparsely observed, multiplicatively separable N-way array: one value per distinct complementary index tuple, in the column order of build_axis_constraint_matrix.

```python
import numpy as np

def extract_complementary_pattern(obs: dict, axis: int) -> np.ndarray:
    """Extract the complementary-axes factor pattern for one removed axis.

    Parameters
    ----------
    obs : dict
        Mapping from N-tuple integer indices to observed float values of a
        sparsely sampled, multiplicatively separable array.
    axis : int
        Zero-based position of the axis being removed.

    Returns
    -------
    x : np.ndarray
        One-dimensional array with one entry per distinct complementary index
        tuple (ascending sorted order), unit Euclidean norm, first entry of
        magnitude above 1e-12 positive.

    Raises
    ------
    ValueError
        If obs is empty, axis is out of range, or no removed-axis value has
        two or more observations (no constraint rows).
    """
    return None
```

### Step 4

04_fit_resolved_factor

Goal
----
Recover the factor vector of one resolved axis of a sparsely observed, multiplicatively separable N-way array, given the factor vectors of all the other axes, as the vector that best reproduces the observed cells.

```python
import numpy as np

def fit_resolved_factor(obs: dict, axis: int, other_factors: list, size: int) -> np.ndarray:
    """Fit the resolved axis's factor vector from the observed cells.

    Parameters
    ----------
    obs : dict
        Mapping from N-tuple integer indices to observed float values.
    axis : int
        Zero-based position of the axis whose factor is being recovered.
    other_factors : list
        The N-1 factor vectors (1-D arrays) of the other axes, in increasing
        axis position, skipping `axis`.
    size : int
        Number of indices along the resolved axis (length of the result).

    Returns
    -------
    u : np.ndarray
        One-dimensional array of length size, the factor of the resolved axis.

    Raises
    ------
    ValueError
        If obs is empty, axis is out of range, other_factors has the wrong
        length, or some index of the resolved axis is never observed.
    """
    return None
```

### Step 5

05_complete_separable_landscape

Goal
----
Complete a sparsely observed, multiplicatively separable N-way array from exact, noiseless observations, returning the full array.

```python
import numpy as np

def complete_separable_landscape(obs: dict, dims: tuple) -> np.ndarray:
    """Complete a sparse, multiplicatively separable N-way array.

    Parameters
    ----------
    obs : dict
        Mapping from N-tuple integer indices to observed float values of a
        sparsely sampled array that is an outer product of one factor vector
        per axis (no interaction terms), with exact, noiseless values.
    dims : tuple
        Length-N tuple giving the size of each axis.

    Returns
    -------
    completed : np.ndarray
        Array of shape dims, the full outer product of the recovered per-axis
        factor vectors.

    Raises
    ------
    ValueError
        If obs is empty, dims has fewer than two axes or a non-positive size,
        or an observed index is out of bounds for dims.
    """
    return None
```

### Step 6

06_recover_coupled_matrices

Goal
----
Jointly recover a partially observed matrix that shares its row axis with a completed N-way landscape, by iterating a proximal block-coordinate scheme to a critical point of one penalized objective, and return the recovered matrix.

```python
import numpy as np

def recover_coupled_matrices(
    landscape: np.ndarray,
    health_obs: dict,
    health_dims: tuple,
    lam_landscape: float,
    lam_health: float,
    lam_coupling: float,
    delta: float,
    n_iters: int,
) -> np.ndarray:
    """Recover the coupled matrix by proximal block-coordinate iteration.

    Parameters
    ----------
    landscape : np.ndarray
        The completed N-way landscape (first axis is the shared row axis); the
        fixed target of the landscape data term and the landscape iterate's
        starting point.
    health_obs : dict
        Mapping from 2-tuple (row, column) indices to observed float values of
        the partially observed matrix.
    health_dims : tuple
        (n_rows, n_cols) of the matrix; n_rows must equal landscape.shape[0].
    lam_landscape, lam_health, lam_coupling : float
        Nuclear-norm weight on the landscape flattening, nuclear-norm weight
        on the matrix, and weight on the coupling-residual term.
    delta : float
        Ridge weight on the coupling operator (must be positive).
    n_iters : int
        Number of block sweeps to perform (0 returns the initialization).

    Returns
    -------
    health_matrix : np.ndarray
        Array of shape health_dims, the matrix iterate after n_iters sweeps.

    Raises
    ------
    ValueError
        If health_dims[0] differs from landscape.shape[0], a nuclear-norm or
        coupling weight is negative, delta is not positive, n_iters is
        negative, or an observed index is out of bounds.
    """
    return None
```

### Step 7

07_solve_training_schedule

Goal
----
Compute the minimum total training hours, summed over every worker, skill and shift, that keeps every worker certified in every skill from the mandate start through the last shift, under per-shift skill decay, additive training gains, level bounds and both capacity limits.

```python
import numpy as np

def solve_training_schedule(
    health_matrix: np.ndarray,
    decay_rates: list,
    threshold: float,
    train_rate: float,
    n_shifts: int,
    mandate_start: int,
    worker_budget: float,
    skill_budget: float,
) -> float:
    """Minimum total training hours meeting the certification mandate.

    Parameters
    ----------
    health_matrix : np.ndarray
        Array of shape (n_workers, n_skills); entry [w, k] clipped to [0, 1]
        is worker w's level in skill k entering shift 1.
    decay_rates : list
        Length-n_skills list of per-skill per-shift decay rates in [0, 1).
    threshold : float
        Minimum level required for certification.
    train_rate : float
        Level gained per hour of training (positive).
    n_shifts : int
        Number of shifts (shifts are numbered 1..n_shifts).
    mandate_start : int
        First shift (1-indexed) at whose end certification is required; the
        requirement holds through shift n_shifts inclusive.
    worker_budget : float
        Maximum total training hours for one worker in one shift.
    skill_budget : float
        Maximum total instructor-hours for one skill in one shift.

    Returns
    -------
    total_hours : float
        The minimum total training hours over all workers, skills and shifts.

    Raises
    ------
    ValueError
        If health_matrix is not 2-D, decay_rates has the wrong length or an
        entry outside [0, 1), train_rate is not positive, n_shifts is not
        positive, mandate_start is outside [1, n_shifts], a budget is negative,
        or the mandate cannot be met under the limits (infeasible).
    """
    return None
```

### Step 8

08_compute_minimum_training_hours

Goal
----
Orchestrator: run the full three-stage pipeline end to end and report the minimum total training hours.

```python
def compute_minimum_training_hours(
    landscape_obs: dict,
    landscape_dims: tuple,
    health_obs: dict,
    health_dims: tuple,
    lam_landscape: float,
    lam_health: float,
    lam_coupling: float,
    delta: float,
    n_iters_palm: int,
    decay_rates: list,
    threshold: float,
    train_rate: float,
    n_shifts: int,
    mandate_start: int,
    worker_budget: float,
    skill_budget: float,
) -> float:
    """Run the full pipeline and report the minimum total training hours.

    Parameters
    ----------
    landscape_obs : dict
        Sparse, exact observations of the multiplicatively separable landscape.
    landscape_dims : tuple
        Shape of the full landscape.
    health_obs : dict
        Sparse, noisy observations of the health matrix.
    health_dims : tuple
        (n_workers, n_skills) shape of the health matrix.
    lam_landscape, lam_health, lam_coupling, delta : float
        Weights of the coupled-recovery objective.
    n_iters_palm : int
        Number of block sweeps for the coupled recovery.
    decay_rates : list
        Length-n_skills per-skill decay rates.
    threshold : float
        Minimum level required for certification.
    train_rate : float
        Level gained per hour of training.
    n_shifts : int
        Total number of shifts.
    mandate_start : int
        First shift (1-indexed) at which certification is required.
    worker_budget : float
        Maximum total training hours per worker per shift.
    skill_budget : float
        Maximum total instructor-hours per skill per shift.

    Returns
    -------
    total_hours : float
        The minimum feasible total training hours across every worker, skill
        and shift.
    """
    return None
```
