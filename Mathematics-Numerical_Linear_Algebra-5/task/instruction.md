# Mathematics-Numerical_Linear_Algebra-5

## Background

The Kaczmarz method, originally proposed by Kaczmarz in 1937 for general linear systems, solves such systems by repeated orthogonal projections onto the hyperplanes defined by individual equations; its cyclic (deterministic, sweep-order) row-selection variant is the one historically known as the algebraic reconstruction technique (ART) from tomographic imaging, while randomized row selection is a later variant that carries the method's now-standard linear convergence guarantee for consistent systems. Block and greedy variants supplanted single-row projections with row blocks and biased the selection toward rows with large residuals, trading extra per-step work for faster empirical convergence, while extended Kaczmarz methods added a second, column-oriented projection to handle inconsistent or rank-deficient systems by driving an auxiliary vector toward the optimal residual in parallel with the solution update. A recognized tension in this line of work is a seesaw effect: purely greedy selection can drive a large number of iterations before convergence and undermines convergence stability, while purely random selection avoids that instability but forgoes the guidance a residual-informed choice provides, and recent double-block extended Kaczmarz methods combine residual-informed probabilistic sampling of both column and row blocks to capture most of greedy selection's stabilizing benefit without its convergence-stability pathology. The broader research this task draws on also emphasizes an MPI-based parallel implementation that partitions rows across ranks with synchronization points between the column and row phases, since those two phases are strictly sequentially dependent within an iteration, together with GPU acceleration aimed at the per-iteration cost.

## Problem

Randomized Kaczmarz methods solve large linear least-squares systems by repeatedly projecting onto blocks of rows or columns chosen according to residual-derived probabilities, and extended double-block variants of this family target inconsistent systems by tracking an auxiliary vector alongside the solution estimate to absorb the part of the right-hand side that no solution can reach. Your task is to reconstruct one such double-block extended Kaczmarz iteration from its named components only, run it in both a serial form and a hybrid-parallel form that distributes the row dimension across several ranks, each for a fixed number of iterations on the same fully specified inconsistent least-squares instance, and report how the parallel form's relative solution error compares to the serial form's. Each iteration of either form executes two phases in a fixed order: a column phase that samples a block of column indices and updates the auxiliary vector using the sampled columns, followed by a row phase that samples a block of row indices and updates the solution estimate from the sampled rows.

Set up the instance and the march with exactly the following configuration:

- Instance: using `numpy.random.default_rng(20260920)`, draw, in this order and from the same generator, `values = rng.standard_normal((600, 150))`, `mask = rng.random((600, 150)) < 0.20`, `x_true = rng.standard_normal(150)`, and `w = rng.standard_normal(600)`. Form the 600×150 coefficient matrix `A = values * mask`.
- Noise: build `r` orthogonal to `range(A)` by computing `xi = numpy.linalg.lstsq(A, w, rcond=None)[0]` and setting `r = w - A @ xi`; rescale `r` so that its Euclidean norm equals 0.10 times the Euclidean norm of `A @ x_true`. Form the right-hand side `b = A @ x_true + r`.
- Block sizes: use block fraction η = 0.2, giving a fixed column-block size of 30 indices and a fixed row-block size of 120 indices, with no index repeated inside a block, drawn from the full index range of 150 columns and 600 rows respectively, at every iteration.
- Sampling: select each index block by sequential inverse-cumulative-distribution-function sampling, never selecting the same index twice within a block — for each of the 30 (column phase) or 120 (row phase) draws in turn, draw `u = rng.random()`, form the cumulative sums of the current weight vector, select the smallest index whose cumulative sum is greater than or equal to `u` times the total weight of the current vector, then set that selected index's weight to zero before the next draw.
- Generators and draw order: use a second, independent generator `numpy.random.default_rng(7)` created once before the march begins and consumed for every sampling draw of every iteration; within each iteration, draw the column-index block completely before drawing the row-index block.
- Initial condition and budget: initialize the solution estimate as the zero vector of length 150; run the march for a fixed budget of exactly 12 iterations, with no tolerance-based early-stopping rule of any kind — this system is built to be inconsistent, so the least-squares residual never vanishes, and any stopping test based on the size of that residual would never trigger for this instance.
- Linear algebra convention: every least-squares solve you perform, anywhere in either march, including the one used to build the noise vector above, must use minimum-norm least squares via `numpy.linalg.lstsq(..., rcond=None)` in float64 precision.
- Parallel formulation: in addition to the serial march above, also run a hybrid-parallel version of the same double-block iteration that splits the 600 rows into P = 4 ranks of contiguous rows, in index order. The row partition is fixed by the following rule, applied once, before the march begins: every row $i$ of $A$ ($i=0,\dots,599$) carries an associated nonnegative per-row quantity $c_i$, computed once, up front, over the whole matrix; this task does not name what $c_i$ measures — deriving it, from the properties of the parallel formulation described below and in the Scientific Background and from the structure of the configuration, is itself part of reconstructing this rule, the same way the sampling-weight and update formulas named later in this task are to be derived from their own names alone. Let $C=\sum_i c_i$ be the matrix's total. Take the running cumulative sum $S(i)=c_0+c_1+\cdots+c_i$, computed as a single pass over the whole matrix from row 0, monotonically, and never reset. For each of the $P-1=3$ internal boundaries, taken in increasing order, boundary $p$ (for $p=1,2,3$) is placed immediately after the first row index $i$ such that $S(i)\ge p\cdot C/P$ — i.e., the first row whose cumulative $c_i$-total, counted from row 0 and never reset at an earlier boundary, reaches or exceeds a $p/P$ share of the matrix's total. Rank 0 owns every row from row 0 up to (not including) the first boundary; rank $P-1$ owns every row from the last boundary through row 599, regardless of its own share of $C$; each of the remaining ranks owns the rows strictly between its two neighboring boundaries. If this rule would ever leave a rank with no rows at all, pull that rank's own boundary forward by one row so every rank owns at least one row. The parallel march uses the same instance, the same η = 0.2, the same fixed budget of 12 iterations, and the same zero-vector initial solution estimate as the serial march, but it draws from its own, freshly constructed generator `numpy.random.default_rng(7)` — a separate generator object from the one the serial march uses, never shared with it. Within each iteration of the parallel march, draw the column-index block, shared across all ranks, before drawing any rank's row-index block, and draw the row-index blocks rank by rank in increasing rank order (rank 0 first, then rank 1, and so on), with each rank's row-index draw restricted to that rank's own rows and independent of every other rank's draw; each rank's own row-block size is $\max(1,\operatorname{round}(\eta\cdot 600/4))=30$, the same fixed count for every rank regardless of that rank's own row share. Every least-squares solve performed anywhere within the parallel march follows the same linear algebra convention as the serial march. The prompt intentionally leaves unstated how the parallel march's updates are formed from the partition and the sampled blocks, and how the per-rank results are combined back into a single update at the end of each iteration; reconstruct these from the phase names, the partition, and the draw order given above.
- Output: run the serial march and the parallel march independently, each for its fixed budget of 12 iterations, and report as the single final numeric answer the ratio ρ formed by dividing the parallel march's relative solution error `‖x_T^{par} − x*‖₂ / ‖x*‖₂` by the serial march's relative solution error `‖x_T − x*‖₂ / ‖x*‖₂`, where `x*` is the minimum-norm least-squares solution of `A x = b`, rounded to 4 significant figures. In the reasoning block, additionally report: (i) the serial march's relative solution error alone, rounded to 4 significant figures; and (ii) a statement — not a numeric value — of the form of the method's guaranteed linear convergence factor, of the assumption that guarantee places on the sampled row blocks, of what the constant appearing in that factor is and why it is not something this instance evaluates, and of how the guarantee available for the auxiliary vector differs in structure from the guarantee available for the solution estimate.

Derive the column and row sampling-weight formulas, the auxiliary vector's initialization and its projection update, and the block least-squares solution update from the properties named above and the structure of the configuration, for both the serial and the parallel march; the prompt intentionally leaves these unstated beyond their names. In the reasoning block, explain the steps you took to get the final answer, covering how the instance and the sampling streams were set up and how each march was run, and, for each setup choice the configuration above fixes, what that choice implies for the reported quantities and what would change if it were made differently; do not report computed values beyond the three requested above; naming the configuration constants this prompt already supplies is not a reported value. Running the serial march and the parallel march once each under these fixed settings determines the requested quantities, so the result is fully deterministic.

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

01_build_kaczmarz_instance

Goal
----
Builds a fully pinned, deterministically inconsistent least-squares instance for the double-block extended Kaczmarz march: a sparse random coefficient matrix, a right-hand side whose noise component is exactly orthogonal to the matrix's column space, and the derived scalars every later step needs to sample and solve against it.

```python
def build_kaczmarz_instance(seed: int = 20260920, m: int = 600, n: int = 150, density: float = 0.20, noise_ratio: float = 0.10) -> dict:
    """Build a pinned inconsistent least-squares instance for RGDBEK.

    Parameters
    ----------
    seed : int
        Seed for `numpy.random.default_rng`, consumed in the fixed draw
        order values, mask, x_true, w (must be a non-negative integer).
    m : int
        Number of rows of the coefficient matrix (must be a positive
        integer with m >= n).
    n : int
        Number of columns of the coefficient matrix (must be a positive
        integer with n <= m).
    density : float
        Target fraction of nonzero entries of the coefficient matrix
        (must satisfy 0 < density <= 1).
    noise_ratio : float
        Ratio of the noise vector's Euclidean norm to the Euclidean norm
        of the noiseless signal A @ x_true (must be > 0).

    Returns
    -------
    instance : dict
        A : (m, n) float array, the sparse random coefficient matrix.
        b : (m,) float array, the right-hand side A @ x_star + r_opt.
        x_star : (n,) float array, the exact least-squares solution
            (equal to the ground-truth vector by construction).
        r_opt : (m,) float array, the exact optimal residual
            b - A @ x_star (orthogonal to the range of A by construction).
        col_sq_norms : (n,) float array, squared Euclidean norm of every
            column of A.
        row_sq_norms : (m,) float array, squared Euclidean norm of every
            row of A.
        b_norm : float, the Euclidean norm of b.

    Raises
    ------
    ValueError
        If seed is not a non-negative integer, if m or n is not a
        positive integer or m < n, if density is not in (0, 1], if
        noise_ratio is not a positive real number, or if the drawn noise
        vector has exactly zero norm (a degenerate instance).
    """
    return instance
```

### Step 2

02_sample_index_block

Goal
----
Draws a block of distinct indices without replacement from a probability vector, using sequential inverse-cumulative-distribution-function sampling from a supplied random number generator, and is the shared sampling primitive used for both the column block and the row block of every iteration.

```python
def sample_index_block(probabilities: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a block of distinct indices by sequential inverse-CDF sampling.

    Parameters
    ----------
    probabilities : numpy.ndarray
        (n,) nonnegative weight (or probability) vector to sample from.
    block_size : int
        Number of distinct indices to draw (must be a positive integer
        no larger than the number of strictly positive entries of
        probabilities).
    rng : numpy.random.Generator
        A live NumPy random Generator object (e.g. from
        numpy.random.default_rng), consumed one `rng.random()` call per
        drawn index, in order. This must be the SAME generator object
        threaded across every call within a march, never a freshly
        seeded generator per call.

    Returns
    -------
    indices : numpy.ndarray
        (block_size,) array of distinct integer indices into
        probabilities, in the order they were drawn.

    Raises
    ------
    ValueError
        If probabilities is not 1-D, if any entry is negative, if
        block_size is not a positive integer, or if fewer than
        block_size entries of probabilities are strictly positive.
    """
    return indices
```

### Step 3

03_column_phase_update

Goal
----
Runs one iteration's column phase of the serial march end to end: forms the residual-derived column sampling weight from the current auxiliary vector, draws a block of column indices from it, and updates the auxiliary vector using the sampled columns, returning only the updated vector.

```python
def column_phase_update(A: np.ndarray, z: np.ndarray, col_sq_norms: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Run one iteration's column-phase weight formation, sampling, and update.

    Parameters
    ----------
    A : numpy.ndarray
        (m, n) coefficient matrix.
    z : numpy.ndarray
        (m,) current auxiliary residual vector.
    col_sq_norms : numpy.ndarray
        (n,) squared Euclidean norm of every column of A, precomputed by
        the caller (must not be recomputed from A inside this function).
    block_size : int
        Number of distinct column indices to sample this iteration (must
        be a positive integer no larger than the number of columns with
        strictly positive column sampling weight).
    rng : numpy.random.Generator
        A live NumPy random Generator object, consumed for this
        iteration's column-index draw using the pinned sampling
        convention. Must be the SAME generator object threaded across
        every call within a march, never a freshly seeded generator per
        call.

    Returns
    -------
    z_next : numpy.ndarray
        (m,) the auxiliary vector after this iteration's column-phase
        weight formation, sampling, and update have all been applied.

    Raises
    ------
    ValueError
        If A is not 2-D, if z's length does not match A's number of
        rows, if col_sq_norms's length does not match A's number of
        columns, if any entry of col_sq_norms is not strictly positive,
        if block_size is not a positive integer, if every resulting
        column weight is exactly zero, or if fewer than block_size
        columns have strictly positive weight.
    """
    return z_next
```

### Step 4

04_row_phase_update

Goal
----
Runs one iteration's row phase of the serial march end to end: forms the residual-derived row sampling weight from the solution estimate and the auxiliary vector as it stands when this step is called, draws a block of row indices from it, and updates the solution estimate with a block least-squares step over the sampled rows.

```python
def row_phase_update(A: np.ndarray, b: np.ndarray, z: np.ndarray, x: np.ndarray, row_sq_norms: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Run one iteration's row-phase weight formation, sampling, and update.

    Parameters
    ----------
    A : numpy.ndarray
        (m, n) coefficient matrix.
    b : numpy.ndarray
        (m,) right-hand side vector.
    z : numpy.ndarray
        (m,) the auxiliary residual vector as it stands when the row
        phase runs (the caller decides which value of the auxiliary
        vector this is; this function does not assume or check it).
    x : numpy.ndarray
        (n,) current solution estimate.
    row_sq_norms : numpy.ndarray
        (m,) squared Euclidean norm of every row of A, precomputed by
        the caller (must not be recomputed from A inside this
        function).
    block_size : int
        Number of distinct row indices to sample this iteration (must
        be a positive integer no larger than the number of rows with
        strictly positive row sampling weight).
    rng : numpy.random.Generator
        A live NumPy random Generator object, consumed for this
        iteration's row-index draw using the pinned sampling
        convention. Must be the SAME generator object threaded across
        every call within a march, never a freshly seeded generator per
        call.

    Returns
    -------
    x_next : numpy.ndarray
        (n,) the solution estimate after this iteration's row-phase
        weight formation, sampling, and block least-squares update have
        all been applied.

    Raises
    ------
    ValueError
        If A is not 2-D, if b's or z's length does not match A's number
        of rows, if x's length does not match A's number of columns, if
        row_sq_norms's length does not match A's number of rows, if any
        entry of row_sq_norms is not strictly positive, if block_size is
        not a positive integer, if every resulting row weight is
        exactly zero, or if fewer than block_size rows have strictly
        positive weight.
    """
    return x_next
```

### Step 5

05_march_rgdbek

Goal
----
Runs the full serial RGDBEK march for a fixed number of iterations: every iteration runs the column phase and then runs the row phase against the auxiliary vector the column phase just produced, never against its pre-update value, with a single random number generator threaded through every sampling draw of the whole march.

```python
def march_rgdbek(A: np.ndarray, b: np.ndarray, eta: float, iterations: int, march_seed: int) -> dict:
    """Run the full serial RGDBEK march for a fixed iteration budget.

    Parameters
    ----------
    A : numpy.ndarray
        (m, n) coefficient matrix.
    b : numpy.ndarray
        (m,) right-hand side vector.
    eta : float
        Block-size fraction (must satisfy 0 < eta <= 1); the column block
        size is max(1, round(eta * n)) and the row block size is
        max(1, round(eta * m)).
    iterations : int
        Number of fixed iterations to run (must be a positive integer).
    march_seed : int
        Seed for a SINGLE `numpy.random.default_rng(march_seed)`
        generator constructed once before the loop and threaded through
        every sampling draw of the whole march, columns before rows
        within every iteration (must be a non-negative integer).

    Returns
    -------
    result : dict
        x : (n,) float array, the final solution estimate x_T.
        z : (m,) float array, the final auxiliary residual vector z_T.
        error_history : list of float, length `iterations`, the
            Euclidean norm of the solution-estimate update
            ||x_{k+1} - x_k|| at every iteration, in order (diagnostic
            only).

    Raises
    ------
    ValueError
        If A is not 2-D, if b's length does not match A's number of
        rows, if eta is not in (0, 1], if iterations is not a positive
        integer, or if march_seed is not a non-negative integer.
    """
    return result
```

### Step 6

06_parallel_column_phase

Goal
----
Computes one iteration's column-phase update of the auxiliary residual vector for the hybrid-parallel march, using the same residual-derived column sampling weight and the same shared draw as the single-rank formulation, then producing the updated auxiliary vector across the fixed rank partition.

```python
def parallel_column_phase(A: np.ndarray, z: np.ndarray, n_ranks: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Apply one iteration's column-phase update in the hybrid-parallel march.

    Parameters
    ----------
    A : numpy.ndarray
        (m, n) coefficient matrix.
    z : numpy.ndarray
        (m,) current auxiliary residual vector, using the same row
        ordering as A.
    n_ranks : int
        Number of ranks the row dimension is partitioned into. Must be
        a positive integer no greater than A's number of rows; ranks
        own a fixed, contiguous partition of the rows, determined by
        the rule named in the configuration.
    block_size : int
        Number of distinct column indices to sample this iteration.
        Must be a positive integer no larger than the number of
        columns with strictly positive column sampling weight.
    rng : numpy.random.Generator
        The parallel march's own shared sampling stream (the same
        sampling convention as the single-rank formulation's draws),
        consumed for this iteration's column-index draw.

    Returns
    -------
    z_next : numpy.ndarray
        (m,) the auxiliary vector after this iteration's column-phase
        update has been applied, in the same row ordering as z.

    Raises
    ------
    ValueError
        If A is not 2-D, if z's length does not match A's number of
        rows, if n_ranks is not a positive integer no greater than A's
        number of rows, if block_size is not a positive integer no
        larger than the number of columns with strictly positive
        column sampling weight, or if any column of A has a zero
        squared Euclidean norm.
    """
    return z_next
```

### Step 7

07_parallel_row_phase

Goal
----
Computes one iteration's row-phase update of the solution estimate for the hybrid-parallel march, applying the row phase separately within each rank of the fixed rank partition, each rank drawing its own row-index block from the shared sampling stream restricted to its own rows, in increasing rank order, to produce the iteration's updated solution estimate.

```python
def parallel_row_phase(A: np.ndarray, b: np.ndarray, z: np.ndarray, x: np.ndarray, n_ranks: int, block_size_per_rank: int, rng: np.random.Generator) -> np.ndarray:
    """Apply one iteration's row-phase update in the hybrid-parallel march.

    Parameters
    ----------
    A : numpy.ndarray
        (m, n) coefficient matrix.
    b : numpy.ndarray
        (m,) right-hand side vector.
    z : numpy.ndarray
        (m,) the auxiliary residual vector as it stands when the row
        phase runs, using the same row ordering as A and reflecting the
        fixed, contiguous rank partition named in the configuration.
    x : numpy.ndarray
        (n,) current solution estimate.
    n_ranks : int
        Number of ranks the row dimension is partitioned into. Must be
        a positive integer no greater than A's number of rows; ranks
        own a fixed, contiguous partition of the rows, determined by
        the rule named in the configuration.
    block_size_per_rank : int
        Number of distinct row indices each rank samples this
        iteration, from its own rows only. Must be a positive integer
        no larger than the number of rows, within any single rank's
        block, with strictly positive row sampling weight.
    rng : numpy.random.Generator
        The parallel march's own shared sampling stream (the same
        sampling convention as the single-rank formulation's draws),
        consumed for every rank's row-index draw this iteration, rank
        by rank in increasing rank order.

    Returns
    -------
    x_next : numpy.ndarray
        (n,) the solution estimate after this iteration's parallel
        row-phase update has been applied.

    Raises
    ------
    ValueError
        If A is not 2-D, if b's or z's length does not match A's number
        of rows, if x's length does not match A's number of columns, if
        n_ranks is not a positive integer no greater than A's number of
        rows, if block_size_per_rank is not a positive integer, or if
        any rank's own row sampling weights are all exactly zero or
        number fewer than block_size_per_rank strictly positive
        entries.
    """
    return x_next
```

### Step 8

08_march_parallel_rgdbek

Goal
----
Runs the full hybrid-parallel RGDBEK march for a fixed number of iterations: every iteration runs the parallel column phase and then runs the parallel row phase against the auxiliary vector the column phase just produced, never against its pre-update value, with a single random number generator threaded through every sampling draw of every rank of the whole march.

```python
def march_parallel_rgdbek(A: np.ndarray, b: np.ndarray, eta: float, iterations: int, march_seed: int, n_ranks: int = 4) -> dict:
    """Run the full hybrid-parallel RGDBEK march for a fixed iteration budget.

    Parameters
    ----------
    A : numpy.ndarray
        (m, n) coefficient matrix.
    b : numpy.ndarray
        (m,) right-hand side vector.
    eta : float
        Block-size fraction (must satisfy 0 < eta <= 1); the shared
        column block size is max(1, round(eta * n)) and each rank's own
        row block size is max(1, round(eta * m / n_ranks)), the same
        fixed count for every rank regardless of that rank's own row
        share.
    iterations : int
        Number of fixed iterations to run (must be a positive integer).
    march_seed : int
        Seed for a SINGLE `numpy.random.default_rng(march_seed)`
        generator constructed once before the loop and threaded through
        every sampling draw of every rank of the whole march, the
        shared column draw before any rank's row draw within every
        iteration, and every rank's row draw in increasing rank order
        (must be a non-negative integer). This must be a separate
        generator object from the one the serial march uses.
    n_ranks : int
        Number of ranks the row dimension is partitioned into (must be
        a positive integer no greater than A's number of rows; ranks
        own a fixed, contiguous partition of the rows, determined by
        the rule named in the configuration). Defaults to the
        benchmark's pinned rank count.

    Returns
    -------
    result : dict
        x : (n,) float array, the final solution estimate x_T^par.
        z : (m,) float array, the final auxiliary residual vector z_T^par.
        error_history : list of float, length `iterations`, the
            Euclidean norm of the solution-estimate update
            ||x_{k+1} - x_k|| at every iteration, in order (diagnostic
            only).

    Raises
    ------
    ValueError
        If A is not 2-D, if b's length does not match A's number of
        rows, if eta is not in (0, 1], if iterations is not a positive
        integer, if march_seed is not a non-negative integer, or if
        n_ranks is not a positive integer no greater than A's number of
        rows.
    """
    return result
```

### Step 9

09_convergence_diagnostics

Goal
----
Computes the convergence diagnostics that compare a finished serial march and a finished parallel march against the instance's exact targets, and against each other: the relative solution error of each march, the ratio of the parallel march's relative solution error to the serial march's, and the auxiliary-residual error ratio of each march measured against the auxiliary vector's known initial value.

```python
def convergence_diagnostics(A: np.ndarray, b: np.ndarray, x_serial: np.ndarray, x_parallel: np.ndarray, z_serial: np.ndarray, z_parallel: np.ndarray, x_star: np.ndarray, r_opt: np.ndarray) -> dict:
    """Compute the convergence diagnostics of a finished serial and parallel march.

    Parameters
    ----------
    A : numpy.ndarray
        (m, n) coefficient matrix.
    b : numpy.ndarray
        (m,) right-hand side vector, used as the shared reference point
        for both marches' auxiliary-residual-error-ratio denominators
        (both marches are initialized with the auxiliary vector equal
        to b).
    x_serial : numpy.ndarray
        (n,) final solution estimate from the serial march.
    x_parallel : numpy.ndarray
        (n,) final solution estimate from the parallel march.
    z_serial : numpy.ndarray
        (m,) final auxiliary residual vector from the serial march.
    z_parallel : numpy.ndarray
        (m,) final auxiliary residual vector from the parallel march.
    x_star : numpy.ndarray
        (n,) exact least-squares solution.
    r_opt : numpy.ndarray
        (m,) exact optimal residual.

    Returns
    -------
    diagnostics : dict
        rel_error_serial : float, ||x_serial - x_star||_2 / ||x_star||_2.
        rel_error_parallel : float, ||x_parallel - x_star||_2 / ||x_star||_2.
        error_ratio : float, rel_error_parallel / rel_error_serial.
        z_error_ratio_serial : float, ||z_serial - r_opt||_2 / ||b - r_opt||_2.
        z_error_ratio_parallel : float, ||z_parallel - r_opt||_2 / ||b - r_opt||_2.

    Raises
    ------
    ValueError
        If A is not 2-D, if b's or r_opt's length does not match A's
        number of rows, if x_serial's, x_parallel's, or x_star's length
        does not match A's number of columns, if z_serial's or
        z_parallel's length does not match A's number of rows, if
        x_star has zero norm, if b equals r_opt exactly (a degenerate
        z-error-ratio denominator), or if x_serial equals x_star exactly
        (a degenerate error_ratio denominator).
    """
    return diagnostics
```

### Step 10

10_run_rgdbek_comparison

Goal
----
Chains the instance construction, the full serial RGDBEK march, the full hybrid-parallel RGDBEK march, and the convergence diagnostics of the preceding steps, in order, at the requested configuration, and extracts the single requested observable: the ratio of the parallel march's relative solution error to the serial march's relative solution error.

```python
def run_rgdbek_comparison(seed: int = 20260920, march_seed: int = 7, m: int = 600, n: int = 150, density: float = 0.20, noise_ratio: float = 0.10, eta: float = 0.2, iterations: int = 12, n_ranks: int = 4) -> float:
    """Run the full RGDBEK comparison experiment and return the error ratio.

    Parameters
    ----------
    seed : int
        Seed for the instance construction's random number generator
        (must be a non-negative integer).
    march_seed : int
        Seed for the serial march's single, threaded random number
        generator. The parallel march is seeded with this same integer
        value but on its own, freshly constructed, separate generator
        object (must be a non-negative integer).
    m : int
        Number of rows of the coefficient matrix (must be a positive
        integer with m >= n).
    n : int
        Number of columns of the coefficient matrix (must be a positive
        integer with n <= m).
    density : float
        Target fraction of nonzero entries of the coefficient matrix
        (must satisfy 0 < density <= 1).
    noise_ratio : float
        Ratio of the noise vector's Euclidean norm to the Euclidean norm
        of the noiseless signal (must be > 0).
    eta : float
        Block-size fraction for both marches (must satisfy 0 < eta <= 1).
    iterations : int
        Number of fixed iterations each march runs (must be a positive
        integer).
    n_ranks : int
        Number of ranks the parallel march's row dimension is
        partitioned into (must be a positive integer no greater than
        m; ranks own a fixed, contiguous partition of the rows,
        determined by the rule named in the configuration).

    Returns
    -------
    error_ratio : float
        The ratio of the parallel march's relative solution error to
        the serial march's relative solution error, after each march's
        fixed-budget run:
        (||x_T^par - x_star||_2 / ||x_star||_2)
        / (||x_T - x_star||_2 / ||x_star||_2).

    Raises
    ------
    ValueError
        If any parameter violates the constraints stated above, or if
        any chained step itself raises ValueError on the given
        configuration.
    """
    return error_ratio
```
