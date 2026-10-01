# Mathematics-Numerical_Linear_Algebra-54

## Background

Row-action (Kaczmarz-type) methods solve large linear systems by repeatedly projecting the current iterate using one, or a few, equations at a time; their Bregman generalization replaces Euclidean projections by projections in the geometry of a strongly convex objective, which lets an $\ell_1$-type penalty promote sparse solutions. Most convergence analyses assume the right-hand side is known exactly, or at least fixed. A more demanding but realistic model lets every query of a measurement return a fresh, independent, zero-mean corruption. Under that model, any method with a constant step size can only reach a neighbourhood of the true solution whose radius is set by the noise, but a step size that shrinks at the right rate can average the noise away and converge exactly. A recent paper combines this adaptive stepping with batch averaging of several weighted row-action directions per iteration and with a noise-aware choice of sampling probabilities and weights, and shows that the convergence guarantee improves monotonically with the batch size and that down-weighting unreliable rows strictly reduces the noise injected per step.

## Problem

Row-action solvers reconstruct an unknown vector from a linear system by touching only a few equations per step, and their Bregman-projection generalization allows a strongly convex objective (for example an $\ell_1$-plus-squared-$\ell_2$ term) to shape the recovered solution toward sparsity. When every query of a right-hand-side entry returns that entry corrupted by a *fresh*, independent, zero-mean error, a fixed-step method stalls in a noise ball, whereas a recent paper shows that querying a batch of equations at once, aggregating their weighted row-action directions in a specific way, choosing the sampling law and weights from the per-row noise levels through a single coupling identity, and shrinking the step size along an explicitly defined auxiliary sequence drives the iterates to the exact noise-free solution. Your task is to carry out a small, fully deterministic run of that method and report the resulting squared error between the final primal iterate and the exact solution. The paper's exact sampling rule, weight normalization, aggregation of the batch, spectral convergence quantity, step-size recursion, initialization of the auxiliary sequence, and the error-bound constant it borrows from earlier work are not restated here — see the sources for their exact definitions. Use the following configuration:

- $A = \begin{bmatrix} 2 & 1 & 0 & 1 \\ 1 & 3 & 1 & 0 \\ 0 & 1 & 4 & 1 \\ 3 & 0 & 1 & 2 \\ 1 & 1 & 1 & 1 \\ 2 & -1 & 2 & 0 \end{bmatrix}$ (6×4, rows listed top to bottom, rows indexed $0,\dots,5$), exact solution $\hat x = [2, 0, -1, 0]^T$, and clean right-hand side $b = A\hat x$
- Per-row noise standard deviations $\sigma = [0.5,\ 0.5,\ 3.0,\ 0.5,\ 3.0,\ 0.5]$ (row $i$ has level $\sigma_i$); every query of entry $i$ returns $b_i$ plus a fresh $\mathcal N(0,\sigma_i^2)$ draw
- Objective $f(x) = \lambda\|x\|_1 + \tfrac12\|x\|_2^2$ with $\lambda = 0.1$; the primal iterate is always recovered from the dual iterate through $\nabla f^*$, and the dual iterate starts at $x_0^* = 0$
- Relaxation parameter $\alpha = 1$ in the paper's coupling identity between sampling probabilities and weights; use the paper's noise-aware optimal choice of that coupling
- Batch size $\tau = 3$ and $K = 6$ iterations $k = 0,\dots,5$; initialize the paper's auxiliary step-size sequence with its *exact* value computed from $\hat x$ (as in the paper's experiments)
- The error-bound constant $\gamma$ that enters the paper's step-size rule must take its *exact theoretical value* for this objective and this system: the paper defines $\gamma$ through the constant of its error-bound assumption, rescaled by $\|A\|_F^2$, and defers the explicit form of that constant — in terms of $A$, $\lambda$ and the nonzero entries of $\hat x$ — to the earlier sparse-Kaczmarz convergence analysis it cites; use that explicit constant (not restated here, and stated there with the inequality in the opposite direction)
- Randomness: a single generator `rng = np.random.default_rng(3)` supplies everything. At each iteration, first draw the batch as `batch = rng.choice(6, size=3, replace=True, p=p)`, where `p` is the paper's sampling distribution over rows $0,\dots,5$; then draw the fresh noise for the three draws in one call, `eps = rng.normal(0.0, sigma[batch])`, so the noisy value used for draw $j$ is `b[batch[j]] + eps[j]`. A row drawn twice in one batch receives two independent noise values. No other random draws are made.

Let $x_6$ denote the primal iterate after these six iterations. Compute the squared error $\|x_6 - \hat x\|_2^2$.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> concise but complete: show the sampling distribution and weights, the spectral quantity, the error-bound constant $\gamma$, the initial auxiliary value, each iteration's batch, step size and primal iterate, since these are exactly what determines the final number.
Do not re-paste the input matrix $A$; reference its entries only as needed for the computation above.

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

01_noise_aware_coupling

Goal
----
Build the paper's noise-aware row-sampling distribution and the matching per-row weights for a system whose right-hand side is observed under heterogeneous, fresh, zero-mean noise.

```python
def noise_aware_coupling(A: "np.ndarray", sigma: "np.ndarray",
                         alpha: float) -> tuple:
    '''Compute the paper's optimal noise-aware sampling probabilities and
    the coupled per-row weights.

    Parameters
    ----------
    A : np.ndarray
        (m, n) coefficient matrix with no zero row.
    sigma : np.ndarray
        (m,) per-row noise standard deviations, all strictly positive.
    alpha : float
        Relaxation parameter of the paper's coupling identity, > 0.

    Returns
    -------
    result : tuple of (np.ndarray, np.ndarray)
        (p, w): p is the (m,) sampling distribution over rows (nonnegative,
        summing to 1) and w the (m,) per-row weights, both as defined by the
        paper's optimal noise-aware scheme under its coupling identity with
        relaxation parameter alpha (not restated here).

    Raises
    ------
    ValueError
        If A is not a 2D array, if sigma does not have shape (m,) matching
        A's rows, if any entry of sigma is not strictly positive, if alpha
        is not strictly positive, or if any row of A is exactly the zero
        vector.
    '''
    return p, w  # placeholder
```

### Step 2

02_aabk_spectral_bound

Goal
----
Compute the largest eigenvalue of the paper's symmetric positive semidefinite matrix that governs the convergence of its block-averaged, weighted row-action iteration.

```python
def aabk_spectral_bound(A: "np.ndarray", w: "np.ndarray", alpha: float,
                        tau: int) -> float:
    '''Compute sigma_max(T), the largest eigenvalue of the paper's
    convergence matrix T (eq. (6)).

    Parameters
    ----------
    A : np.ndarray
        (m, n) coefficient matrix.
    w : np.ndarray
        (m,) per-row weights produced by noise_aware_coupling (the diagonal
        of the weight matrix W).
    alpha : float
        Relaxation parameter of the paper's coupling identity, > 0.
    tau : int
        Batch size (number of rows drawn per iteration), >= 1.

    Returns
    -------
    sigma_max_T : float
        The largest eigenvalue of the matrix T defined in eq. (6) of the
        paper (not restated here), as a native Python float.

    Raises
    ------
    ValueError
        If A is not a 2D array, if w does not have shape (m,) matching A's
        rows, if alpha is not strictly positive, or if tau is not a
        positive integer.
    '''
    return sigma_max_T  # placeholder
```

### Step 3

03_sparse_bregman_distance

Goal
----
Evaluate the Bregman distance, with respect to the sparsity-promoting strongly convex objective f(x) = lam*||x||_1 + (1/2)||x||_2^2, from the primal point encoded by a dual vector x_star to a target point y.

```python
def sparse_bregman_distance(x_star: "np.ndarray", y: "np.ndarray",
                            lam: float) -> float:
    '''Compute D_f^{x*}(x, y) for f(x) = lam*||x||_1 + (1/2)||x||_2^2, where
    the primal point x is the one encoded by the dual vector x_star, i.e.
    x = grad f*(x_star).

    Parameters
    ----------
    x_star : np.ndarray
        (n,) dual vector (a subgradient of f at the primal point x it
        encodes).
    y : np.ndarray
        (n,) target point.
    lam : float
        Sparsity parameter lam >= 0 of the objective f.

    Returns
    -------
    dist : float
        The Bregman distance D_f^{x*}(x, y) as a native Python float
        (nonnegative).

    Raises
    ------
    ValueError
        If x_star and y do not have the same 1D shape, or if lam is
        negative.
    '''
    return dist  # placeholder
```

### Step 4

04_aabk_beta_init

Goal
----
Compute the exact initial value of the paper's auxiliary step-size sequence from the initial Bregman distance to the exact solution and the paper's per-step noise prefactor.

```python
def aabk_beta_init(A: "np.ndarray", sigma: "np.ndarray", p: "np.ndarray",
                   w: "np.ndarray", tau: int, breg0: float) -> float:
    '''Compute the exact initial auxiliary value beta_0 of the paper's
    adaptive step-size rule (Theorem 2.1).

    Parameters
    ----------
    A : np.ndarray
        (m, n) coefficient matrix with no zero row.
    sigma : np.ndarray
        (m,) per-row noise standard deviations, all strictly positive.
    p : np.ndarray
        (m,) sampling distribution over rows, from noise_aware_coupling.
    w : np.ndarray
        (m,) per-row weights, from noise_aware_coupling.
    tau : int
        Batch size, >= 1.
    breg0 : float
        The Bregman distance D_f^{x*_0}(x_0, x_hat) from the starting point
        to the exact solution (as produced by sparse_bregman_distance),
        >= 0.

    Returns
    -------
    beta0 : float
        The exact initial value beta_0 of the paper's auxiliary sequence,
        as defined in Theorem 2.1 (not restated here), as a native Python
        float.

    Raises
    ------
    ValueError
        If A is not a 2D array, if sigma, p or w do not have shape (m,)
        matching A's rows, if any entry of sigma is not strictly positive,
        if any row of A is exactly the zero vector, if tau is not a
        positive integer, or if breg0 is negative.
    '''
    return beta0  # placeholder
```

### Step 5

05_error_bound_gamma

Goal
----
Compute the exact value of the error-bound constant gamma that enters the paper's adaptive step-size rule, for the sparsity-promoting objective, from the coefficient matrix, the exact solution and the sparsity parameter.

```python
def error_bound_gamma(A: "np.ndarray", x_hat: "np.ndarray", lam: float) -> float:
    '''Compute the paper's error-bound constant gamma (Assumption 3.2, with
    gamma := theta(x_hat) / ||A||_F^2) for f(x) = lam*||x||_1 + (1/2)||x||_2^2.

    Parameters
    ----------
    A : np.ndarray
        (m, n) coefficient matrix, not identically zero.
    x_hat : np.ndarray
        (n,) exact (minimum-f) solution of A x = b, with at least one
        nonzero entry.
    lam : float
        Sparsity parameter lam >= 0 of the objective f.

    Returns
    -------
    gamma : float
        The paper's constant gamma, as a native Python float, using the
        explicit error-bound constant of the cited reference for this
        objective (not restated here). The singular-value quantity in that
        constant is the minimum, over all nonempty column subsets J of A
        with A_J nonzero, of the smallest positive singular value of the
        column submatrix A_J.

    Raises
    ------
    ValueError
        If A is not a 2D array or is identically zero, if x_hat does not
        have shape (n,) matching A's columns or has no nonzero entry, or if
        lam is negative.
    '''
    return gamma  # placeholder
```

### Step 6

06_aabk_averaged_direction

Goal
----
Form the paper's block-averaged, weighted row-action direction for one iteration from a batch of drawn rows and the fresh noisy right-hand-side value returned for each draw.

```python
def aabk_averaged_direction(A: "np.ndarray", x: "np.ndarray", w: "np.ndarray",
                            batch: "np.ndarray",
                            b_noisy: "np.ndarray") -> "np.ndarray":
    '''Form the paper's direction d_k (eq. (4)) for one iteration.

    Parameters
    ----------
    A : np.ndarray
        (m, n) coefficient matrix with no zero row.
    x : np.ndarray
        (n,) current primal iterate.
    w : np.ndarray
        (m,) per-row weights, from noise_aware_coupling.
    batch : np.ndarray
        (tau,) integer row indices drawn for this iteration, in draw order;
        indices may repeat.
    b_noisy : np.ndarray
        (tau,) fresh noisy right-hand-side value returned for each draw, in
        the same draw order as batch (entry j belongs to row batch[j]).

    Returns
    -------
    d : np.ndarray
        (n,) the direction d_k defined in eq. (4) of the paper (not
        restated here).

    Raises
    ------
    ValueError
        If A is not a 2D array, if x does not have shape (n,) matching A's
        columns, if w does not have shape (m,) matching A's rows, if batch
        is empty, if batch and b_noisy do not have the same 1D shape, or if
        any index in batch is out of bounds for the rows of A.
    '''
    return d  # placeholder
```

### Step 7

07_aabk_adaptive_step

Goal
----
Compute this iteration's adaptive step size from the paper's auxiliary sequence and advance that sequence to the next iteration, per the paper's optimal step-size rule.

```python
def aabk_adaptive_step(beta_k: float, alpha: float, gamma: float,
                       sigma_max_T: float) -> tuple:
    '''Compute the adaptive step size eta_k and the next auxiliary value
    beta_{k+1} per eq. (7) of the paper.

    Parameters
    ----------
    beta_k : float
        Current value of the paper's auxiliary sequence, >= 0.
    alpha : float
        Relaxation parameter of the paper's coupling identity, > 0.
    gamma : float
        The paper's error-bound constant (Assumption 3.2), > 0.
    sigma_max_T : float
        Largest eigenvalue of the paper's matrix T (from
        aabk_spectral_bound), > 0.

    Returns
    -------
    result : tuple of (float, float)
        (eta_k, beta_next): the step size to use at this iteration and the
        auxiliary value for the next iteration, both as defined in eq. (7)
        of the paper (not restated here), as native Python floats.

    Raises
    ------
    ValueError
        If beta_k is negative, or if alpha, gamma or sigma_max_T is not
        strictly positive.
    '''
    return eta_k, beta_next  # placeholder
```

### Step 8

08_bregman_dual_update

Goal
----
Take one dual step along the averaged direction with the given step size and map the new dual iterate back to the primal iterate through the conjugate of the sparsity-promoting objective.

```python
def bregman_dual_update(x_star: "np.ndarray", d: "np.ndarray", eta: float,
                        lam: float) -> tuple:
    '''Perform the dual step and primal recovery of eq. (5) for
    f(x) = lam*||x||_1 + (1/2)||x||_2^2.

    Parameters
    ----------
    x_star : np.ndarray
        (n,) current dual iterate x*_k.
    d : np.ndarray
        (n,) direction d_k from aabk_averaged_direction.
    eta : float
        Step size eta_k from aabk_adaptive_step, >= 0.
    lam : float
        Sparsity parameter lam >= 0 of the objective f.

    Returns
    -------
    result : tuple of (np.ndarray, np.ndarray)
        (x_star_next, x_next): the new dual iterate x*_{k+1} and the new
        primal iterate x_{k+1} = grad f*(x*_{k+1}), both of shape (n,), as
        defined in eq. (5) of the paper (not restated here).

    Raises
    ------
    ValueError
        If x_star and d do not have the same 1D shape, if eta is negative,
        or if lam is negative.
    '''
    return x_star_next, x_next  # placeholder
```

### Step 9

09_aabk_orchestrator

Goal
----
[ORCHESTRATOR] Run K deterministic iterations of the paper's adaptive, block-averaged, noise-aware Bregman row-action method under the fresh-noise model and report the final squared error against the exact solution.

```python
def aabk_pipeline(A: "np.ndarray", x_hat: "np.ndarray", sigma: "np.ndarray",
                  lam: float, alpha: float, tau: int, K: int,
                  seed: int) -> float:
    '''Run K iterations of Algorithm 1 of the paper and report the final
    squared error against the exact solution.

    The clean right-hand side is b = A x_hat. The dual iterate starts at
    x*_0 = 0 (so the primal iterate starts at x_0 = grad f*(0)), the
    auxiliary step-size sequence is initialized exactly (using x_hat) as in
    Theorem 2.1, the error-bound constant gamma of the step-size rule takes
    its exact value for this objective (as produced by error_bound_gamma),
    and the objective is f(x) = lam*||x||_1 + (1/2)||x||_2^2.

    Random protocol (a convention of this task; use it exactly). A single
    generator rng = np.random.default_rng(seed) supplies all randomness.
    At each iteration k = 0, ..., K-1, in this order:
      1. batch = rng.choice(m, size=tau, replace=True, p=p), where p is the
         (m,) sampling distribution of Step 1 and rows are indexed 0..m-1;
      2. eps = rng.normal(0.0, sigma[batch]), a single call returning the
         tau fresh noise values in draw order;
      3. the fresh noisy value for draw j is b[batch[j]] + eps[j].
    No other random draws are made.

    Parameters
    ----------
    A : np.ndarray
        (m, n) coefficient matrix with no zero row.
    x_hat : np.ndarray
        (n,) exact solution of the clean system A x = b.
    sigma : np.ndarray
        (m,) per-row noise standard deviations, all strictly positive.
    lam : float
        Sparsity parameter lam >= 0 of the objective f.
    alpha : float
        Relaxation parameter of the paper's coupling identity, > 0.
    tau : int
        Batch size, >= 1.
    K : int
        Number of iterations to run, >= 1.
    seed : int
        Seed of the single np.random.default_rng generator.

    Returns
    -------
    error : float
        The squared Euclidean error ||x_K - x_hat||_2^2 after K iterations,
        as a native Python float.

    Raises
    ------
    ValueError
        If A is not a 2D array, if x_hat does not have shape (n,) matching
        A's columns, if sigma does not have shape (m,) matching A's rows or
        has a non-positive entry, if any row of A is exactly the zero
        vector, if x_hat has no nonzero entry, if lam is negative, if alpha
        is not strictly positive, or if tau or K is not a positive integer.
    '''
    return error  # placeholder
```
