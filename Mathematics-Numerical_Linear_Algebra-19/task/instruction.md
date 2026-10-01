# Mathematics-Numerical_Linear_Algebra-19

## Background

Large sparse linear systems are a major computational cost in coastal and estuarine simulations. The paper introduces NRSAI, an adaptive sparse approximate-inverse preconditioner designed to improve the efficiency of GMRES. It combines matrix-power-based initialization with residual-guided refinement to balance approximation quality and sparsity. The authors evaluate its usefulness through construction time, storage requirements, and solver performance.

## Problem

The paper's residual-based sparse inverse aims to speed up GMRES without making the preconditioner too dense. For the synthetic system below, find the smallest left-to-right residual ratio that can be guaranteed across three right-hand sides when both constructions use the same tolerance pair and share a storage budget.

For $B\in\mathbb R^{48\times48}$, use zero-based indices $i=0,\ldots,47$, reduce column indices of $B$ modulo $48$, set unspecified entries to zero, and take
$$
\begin{aligned}
B_{i,i}&=\frac74+\frac{i\bmod5}{16},
& B_{i,i+1}&=-\frac54,\\
B_{i,i-1}&=-\frac12,
& B_{i,i+5}&=\frac78(-1)^i,\\
d_i&=2^{((7i)\bmod9)-4},
& s_i&=2^{\lfloor(i\bmod3)/2\rfloor},\\
D_{\mathrm L}&=\operatorname{diag}(d_0,\ldots,d_{47}),
& D_{\mathrm R}&=\operatorname{diag}(s_0/d_0,\ldots,s_{47}/d_{47}),\\
A&=D_{\mathrm L}BD_{\mathrm R},
& c&=2,\qquad K=980,\\
b_i^{(0)}&=1+\frac{(-1)^i}{4}+\frac{i\bmod7}{32},\\
b_i^{(1)}&=(-1)^i\left(1+\frac{i\bmod5}{16}\right),\\
b_i^{(2)}&=\frac{((3i)\bmod11)-5}{8}+\frac{i\bmod2}{32},\\
\Omega&=\left[\frac15,\frac{\sqrt3}{5}\right]
\times\left[\frac{\sqrt2}{10},\frac{\sqrt3}{5}\right]
\quad\text{for }(\varepsilon,\delta).
\end{aligned}
$$
For each $(\varepsilon,\delta)\in\Omega$, construct $S_A$ and $S_{A^{\mathsf T}}$ independently using the supplied paper's columnwise power-initialized, residual-adaptive algorithm with selection cap $c$, interpreting its power patterns, comparisons, and unregularized Euclidean least-squares minimizers in exact real arithmetic.
Break equal residual magnitudes by increasing global row index, retain the paper's per-column used-index history, and add no coefficient dropping, equilibration, or iteration cap.

Use $M_{\mathrm R}=S_A$ for right preconditioning and $M_{\mathrm L}=S_{A^{\mathsf T}}^{\mathsf T}$ for left preconditioning. For $q\in\{\mathrm L,\mathrm R\}$ and $j\in\{0,1,2\}$, let $x_{q,j}$ be the physical solution iterate after two consecutive three-step GMRES cycles for $Ax=b^{(j)}$ in the corresponding formulation of equation (2) of the paper, starting from zero, retaining the first-cycle iterate, keeping $M_q$ fixed, and minimizing the unweighted Euclidean residual of that formulation without early exit. If $J_k^C$ is the final active support of column $k$ of $S_C$, count active positions even when their fitted coefficients vanish and define
$$
\begin{aligned}
F(\varepsilon,\delta)
&=\sum_{k=0}^{47}\left(|J_k^A|+|J_k^{A^{\mathsf T}}|\right),\\
\Omega_K&=\{(\varepsilon,\delta)\in\Omega:F(\varepsilon,\delta)\le K\},\\
R_{q,j}(\varepsilon,\delta)
&=\frac{\|b^{(j)}-Ax_{q,j}\|_2}{\|b^{(j)}\|_2}.
\end{aligned}
$$
Compute the following global minimum with absolute error at most $10^{-3}$, reporting the final answer to three decimal places:
$$
Q=\min_{(\varepsilon,\delta)\in\Omega_K}
\max_{j\in\{0,1,2\}}
\frac{R_{\mathrm L,j}(\varepsilon,\delta)}
{R_{\mathrm R,j}(\varepsilon,\delta)}.
$$
In your brief reasoning, justify the adaptive construction, storage feasibility, residual comparison, and global parameter search; report only an attaining tolerance pair, its combined active-pattern size, and the two residuals determining $Q$ as numerical checkpoints, showing decimal results to at most three places and retaining exact fractions for tolerance values when needed, without rounding internal calculations.

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

01_initial_pattern.py

Goal
----
Determine one NRSAI column's initial active rows and coefficient positions from the exact numerical patterns of the identity, C, and C squared.

```python
import numpy as np

def initial_pattern(C: np.ndarray, k: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the initial active row and coefficient indices for column k.

    Parameters
    ----------
    C : np.ndarray
        Nonempty real square matrix with nonzero diagonal, nonsingular when
        its binary64 entries are interpreted as exact rational numbers.
        Conversion to binary64 occurs before all exact support calculations.
    k : int
        Zero-based column index, 0 <= k < C.shape[0].

    Returns
    -------
    I, J : tuple[tuple[int, ...], tuple[int, ...]]
        Increasing, duplicate-free global row and coefficient indices.
        J is the union of the separate numerical supports of I_n, C, C^2.
        I contains exactly the nonzero rows of C[:, J].

    Raises
    ------
    ValueError
        If C is not a nonempty finite real square matrix after binary64
        conversion, has a zero diagonal entry, or is exactly singular;
        or if k is not a non-Boolean integer in the stated range.
    """
    return (), ()
```

### Step 2

02_fit_column.py

Goal
----
Fit one inverse column on a prescribed coefficient support and return its coefficients, full residual, and squared residual norm exactly.

```python
import numpy as np

def fit_column(C: np.ndarray, k: int, J: tuple[int, ...]) -> tuple[tuple, tuple, tuple[int, int]]:
    """Return exact coefficients, the full residual, and its squared norm.

    Parameters
    ----------
    C : np.ndarray
        Finite real nonempty square matrix with nonzero diagonal, nonsingular
        in exact arithmetic after conversion to binary64.
    k : int
        Non-Boolean integer column index in [0, n).
    J : tuple[int, ...]
        Nonempty increasing distinct coefficient indices in [0, n), containing k.

    Returns
    -------
    coefficients, residual, beta : tuple
        coefficients has len(J) entries in J order; residual has n entries.
        Each entry, and beta, is a reduced (numerator, denominator) tuple of
        native integers with positive denominator. The residual is C*m-e_k.
        beta is the squared Euclidean norm, not the norm or a rounded value.
        All coefficients in J are refitted without regularization.

    Raises
    ------
    ValueError
        If C fails the matrix conditions, k is invalid, or J is empty,
        unsorted, duplicated, out of range, or does not contain k.
    """
    return (), (), (0, 1)
```

### Step 3

03_partition_decision.py

Goal
----
Partition a squared-tolerance rectangle into the next NRSAI decisions at one fixed residual state.

```python
import numpy as np

def partition_decision(
    residual: tuple,
    I: tuple[int, ...],
    used: tuple[int, ...],
    box: tuple,
    c: int = 2,
) -> tuple:
    """Split a parameter box into exact next-decision regions.

    Parameters
    ----------
    residual : tuple
        Nonempty full residual; each entry is a reduced integer rational pair.
    I, used : tuple[int, ...]
        Increasing distinct indices in [0, n); I is nonempty and used is a
        subset of I. Residual entries outside I must be zero.
    box : tuple
        (eta_interval, xi_interval) for eta=epsilon**2 and xi=delta**2.
        An interval is (low_pair, high_pair, low_closed, high_closed).
        Pairs are reduced integer rationals; closure flags are integers 0 or 1.
        Require 0 <= eta_low <= eta_high and 0 < xi_low <= xi_high <= 1.
        A singleton is allowed only when both ends are closed.
    c : int
        Positive non-Boolean integer cap on selected rows.

    Returns
    -------
    decisions : tuple
        Entries are (subbox, selected_indices). Empty selected_indices means
        terminate. Otherwise rows are ranked by decreasing residual magnitude,
        then increasing global index. The full residual norm is used.
        Regions are disjoint and cover box exactly. Order: tolerance stop,
        no-eligible stop, then selections of sizes 1 through min(c, available),
        omitting empty regions. Equality stops in eta and is eligible in xi.

    Raises
    ------
    ValueError
        If rational encodings, indices, closure flags, box bounds, or c are
        invalid; if I is empty, used is not a subset of I, the residual is
        empty, or a residual outside I is nonzero.
    """
    return ()
```

### Step 4

04_expand_pattern.py

Goal
----
Expand the active coefficient and row sets from selected residual rows, retaining every selection in the column history.

```python
import numpy as np

def expand_pattern(
    C: np.ndarray,
    I: tuple[int, ...],
    J: tuple[int, ...],
    used: tuple[int, ...],
    selected: tuple[int, ...],
) -> tuple:
    """Return enlarged active rows, coefficients, and permanent row history.

    Parameters
    ----------
    C : np.ndarray
        Finite real nonempty square matrix with nonzero diagonal, nonsingular
        in exact arithmetic after binary64 conversion.
    I, J, used : tuple[int, ...]
        Increasing distinct indices in [0, n). I and J are nonempty;
        I must be exactly the nonzero row set of C[:, J]; used is a subset of I.
    selected : tuple[int, ...]
        Distinct indices in I excluding used, in any order. Empty is allowed.

    Returns
    -------
    I_new, J_new, used_new : tuple
        Three increasing integer tuples. J_new adds every nonzero column of
        C[selected, :]; I_new includes all rows touched by J_new; used_new
        records all selections even when J_new equals J. No position is dropped.

    Raises
    ------
    ValueError
        If C fails the matrix conditions; if an index collection is invalid;
        if I or J is empty; if I is not the row closure of J; if used is not
        contained in I; or if selected repeats or includes a used/outside row.
    """
    return (), (), ()
```

### Step 5

05_column_regions.py

Goal
----
Resolve every terminal support of one NRSAI column over a closed or half-open squared-tolerance box.

```python
import numpy as np

def column_regions(C: np.ndarray, k: int, box: tuple, c: int = 2) -> tuple:
    """Return the exact terminal-support partition for one NRSAI column.

    Parameters
    ----------
    C : np.ndarray
        Finite real nonempty square matrix with nonzero diagonal, nonsingular
        in exact arithmetic after binary64 conversion.
    k : int
        Non-Boolean integer column index in [0, n).
    box : tuple
        Squared-parameter box as in partition_decision: two intervals, each
        (low_rational, high_rational, low_closed, high_closed), with reduced
        integer rational pairs, integer 0/1 flags, eta >= 0 and 0 < xi <= 1.
        Bounds are ordered and a singleton must be closed at both ends.
    c : int
        Positive non-Boolean integer selection cap.

    Returns
    -------
    regions : tuple
        Entries (subbox, J), with numerical box encodings and increasing
        integer coefficient supports. Regions are disjoint and cover box.
        Different histories ending with the same support are not merged.
        Sort by the numerical flattened interval fields, then by J.
        Selection histories start empty; all active coefficients are refitted;
        no dropping, additional scaling, or iteration cap is applied.

    Raises
    ------
    ValueError
        If C, k, box, or c violates the stated conditions.
    """
    return ()
```

### Step 6

06_joint_patterns.py

Goal
----
Intersect the column partitions at a common parameter pair and retain the distinct combined support patterns that meet the shared storage budget.

```python
import numpy as np

def joint_patterns(partitions: tuple, budget: int) -> tuple:
    """Intersect column regions and return feasible joint support patterns.

    Parameters
    ----------
    partitions : tuple
        An even, nonzero number 2*n of column partitions. Each partition is a
        nonempty tuple of (box, J), with boxes encoded as in column_regions
        and nonempty increasing integer supports in [0, n). Its boxes must
        not overlap, including at closed boundaries. The first n partitions
        belong to C=A, in column order; the next n belong to C=A.T.
    budget : int
        Nonnegative non-Boolean integer maximum for the sum of support sizes.

    Returns
    -------
    configurations : tuple
        Entries (representative_box, supports, total_size). supports contains
        2*n increasing index tuples in input order. Only nonempty common
        parameter intersections with total_size <= budget are retained.
        Return one representative per distinct supports, choosing the first
        numerical box in lexicographic flattened-interval order. Sort output
        by supports. Return an empty tuple if no configuration is feasible.
        Zero fitted values are not dropped from the active-position count.

    Raises
    ------
    ValueError
        If partitions is empty, has odd length, contains an empty partition,
        malformed/overlapping boxes, or invalid supports; or budget is not
        a nonnegative non-Boolean integer.
    """
    return ()
```

### Step 7

07_evaluate_patterns.py

Goal
----
Assemble the two inverse approximations on fixed supports and evaluate their original-system relative residuals after the prescribed restarted GMRES solves.

```python
import numpy as np

def evaluate_patterns(
    A: np.ndarray,
    rhs: np.ndarray,
    supports: tuple,
    restart: int = 3,
    cycles: int = 2,
) -> np.ndarray:
    """Return true relative residuals for fixed left and right patterns.

    Parameters
    ----------
    A : np.ndarray
        Finite real nonsingular square matrix of size n with nonzero diagonal;
        entries are converted to binary64 before exact column fitting.
    rhs : np.ndarray
        Finite real array of shape (n, m), m >= 1, with no zero column.
        These are the original right-hand sides, with no extra diagonal scaling.
    supports : tuple
        Exactly 2*n nonempty increasing index tuples. Tuple k contains k
        for the kth column of A; tuple n+k contains k for the kth column of A.T.
    restart, cycles : int
        Positive non-Boolean integers, with restart <= n.

    Returns
    -------
    residuals : np.ndarray
        Float64 array of shape (2, m): row 0 is left, row 1 is right.
        Column fits use exact rational arithmetic; Krylov minimization is
        evaluated numerically without rounding reported intermediate values.
        Each solve starts at zero and retains its endpoint at each restart.
        No convergence tolerance is used. An invariant Krylov subspace uses
        its available basis; a zero starting residual leaves the iterate fixed.

    Raises
    ------
    ValueError
        If A fails its conditions; rhs has the wrong shape or a zero column;
        supports is malformed, missing a required diagonal index, or has the
        wrong length; restart/cycles are invalid; or binary64 evaluation cannot
        produce finite coefficients, iterates, or relative residuals.
    """
    return np.empty((0, 0), dtype=float)
```

### Step 8

08_solve_shared_budget.py

Goal
----
Resolve the complete shared-parameter domain and return the smallest worst-right-hand-side left-to-right residual ratio under the storage budget.

```python
import numpy as np

def solve_shared_budget(
    A: np.ndarray,
    rhs: np.ndarray,
    box: tuple,
    budget: int,
    c: int = 2,
    restart: int = 3,
    cycles: int = 2,
) -> float:
    """Return the global shared-budget minimax residual ratio.

    Parameters
    ----------
    A : np.ndarray
        Finite real nonempty square matrix with nonzero diagonal, nonsingular
        exactly after binary64 conversion.
    rhs : np.ndarray
        Finite real (n, m) array, m >= 1, with no zero column.
    box : tuple
        Domain for (epsilon**2, delta**2), encoded as in column_regions:
        ((eta_low_pair, eta_high_pair, eta_low_closed, eta_high_closed),
         (xi_low_pair, xi_high_pair, xi_low_closed, xi_high_closed)).
        Rational pairs must be reduced with positive denominators; closure
        flags are integer 0/1; eta >= 0 and 0 < xi <= 1. Closed singletons
        are allowed. No rounding of thresholds is permitted.
    budget : int
        Nonnegative non-Boolean combined active-position budget.
    c : int
        Positive non-Boolean selection cap.
    restart, cycles : int
        Positive non-Boolean GMRES counts, with restart <= n.

    Returns
    -------
    minimum : float
        Unrounded native Python float. The same parameter pair determines
        both preconditioners. Each feasible configuration is scored by the
        maximum same-right-hand-side left/right true residual ratio; the
        smallest score is returned. Scientific answer tolerance is 1e-3.
        The final displayed three-decimal answer is a separate formatting step.

    Raises
    ------
    ValueError
        If any input violates the stated contracts, no feasible joint pattern
        exists, any feasible right relative residual is zero, or numerical
        evaluation produces a nonfinite residual ratio.
    """
    return 0.0
```
