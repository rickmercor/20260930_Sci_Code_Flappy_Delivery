# Mathematics-Numerical_Linear_Algebra-27

## Background

Large sparse linear systems occur throughout scientific computing, and repeated movement of matrix and vector data can make them expensive to solve. This paper studies a nested Krylov approach that combines FGMRES with short Richardson solves and adaptive correction weights. Its aim is to reduce computational cost while maintaining reliable convergence despite inexact inner calculations.

## Problem

The paper uses short, nested Krylov solves with adaptive Richardson weights to exploit half precision; for the matrix and three right-hand sides below, determine the largest relative uncertainty in diagonal stabilization under which one configuration halves every true residual compared with fixed unit weights, regardless of the order in which the systems are solved.

Use zero-based indices $i=0,\ldots,47$, reduce column indices of $A$ modulo $48$, set unspecified entries to zero, let $\Pi$ contain the six permutations of $(0,1,2)$, and take
$$
\begin{gathered}
A_{i,i}=1+\frac{1+(i\bmod3)}{4096},\qquad
A_{i,i+1}=-\frac7{16},\qquad A_{i,i-1}=-\frac3{16},\\
A_{i,i+8}=-\frac14,\qquad A_{i,i-8}=-\frac18,\qquad
(m_1,m_4)=(2,2),\\
b_i^{(0)}=1+\frac{(-1)^i}{4}+\frac{i\bmod7}{32},\qquad
b_i^{(1)}=(-1)^i\left(1+\frac{i\bmod5}{16}\right),\\
b_i^{(2)}=\frac{((3i)\bmod11)-5}{8}+\frac{i\bmod2}{32},\\
I_\ell=\{4\ell,4\ell+1,4\ell+2,4\ell+3\},\qquad
\ell=0,\ldots,11,\\
T_\ell(\lambda)=A(I_\ell,I_\ell)
+(\lambda-1)\operatorname{diag}\!\left(
\operatorname{diag}A(I_\ell,I_\ell)\right),\\
\Lambda=\left[\frac{287}{256},\frac{289}{256}\right],\\
\Theta=\{(u,v,c):(u,v)\in\{(3,4),(4,3),(6,2)\},\
c\in\{5,11,17,23,31\}\}.
\end{gathered}
$$
Define $M(\lambda)$ by natural-order, unpivoted block-Jacobi ILU(0) of the $T_\ell(\lambda)$, with unit lower diagonals and exact-real factorization followed by a single fp16 rounding of the factors, applied through triangular solves without forming inverses; use Table 1's precision assignments and leave $A$ unchanged.
All subsequent scalar operations use IEEE round-to-nearest, ties-to-even with retained subnormals and no fusion, increasing-index reductions and direct sum-of-squares norms, and casts to the receiving level's vector format.
At each FGMRES level, use one classical Gram--Schmidt pass with all coefficients taken from the incoming vector and the accumulated projection subtracted once, Givens QR with positive $\rho=\sqrt{a^2+b^2}$ and $(c_g,s_g)=(a/\rho,b/\rho)$, setting the annihilated pair to $(\rho,0)$, and ordinary back substitution.
Store the Richardson weights in fp16, evaluate the local weight and stored-weight average entirely in fp32 using promoted copies of the fp16 data, and use fp16 triangular solves for the correction, with an fp32 iterate update followed by an fp16 cast on weight-update calls only.

For each fixed $\lambda\in\Lambda$, $\theta=(m_2,m_3,c)\in\Theta$, and $\pi\in\Pi$, let $x_{q,\pi,t}(\lambda;\theta)$ be the physical iterate after solving the batch's $t$th system $Ax=b^{(\pi(t))}$ using exactly two outer restart cycles of the paper's three-FGMRES-plus-Richardson hierarchy, without early exits, where $q=\mathrm{adaptive}$ uses Algorithm 1 and $q=\mathrm{fixed}$ uses weight one throughout with no updates.
Every system and inner invocation starts from zero, while an outer restart retains its solution iterate; initialize the adaptive counter to one at the start of each batch and retain its weights and counter across all calls, restarts, and changes of right-hand side within that batch, resetting them only between different batches.
Using the original $A$ and fp64 arithmetic for the final residuals and ratios, define
$$
\begin{gathered}
R_{q,\pi,t}(\lambda;\theta)
=\frac{\|b^{(\pi(t))}-Ax_{q,\pi,t}(\lambda;\theta)\|_2}
{\|b^{(\pi(t))}\|_2},\\
G(\lambda;\theta)
=\min_{\substack{\pi\in\Pi\\t\in\{0,1,2\}}}
\frac{R_{\mathrm{fixed},\pi,t}(\lambda;\theta)}
{R_{\mathrm{adaptive},\pi,t}(\lambda;\theta)},\\
I(\alpha,d)=[\alpha(1-d),\alpha(1+d)],\\
Q=10^6\sup_{\substack{
\alpha>0,\ \theta\in\Theta,\ 0\le d<1\\
I(\alpha,d)\subseteq\Lambda\\
G(\lambda;\theta)\ge2\ \text{for every }\lambda\in I(\alpha,d)
}} d,
\end{gathered}
$$
and report $Q$, the relative uncertainty half-width in parts per million, to three decimal places with absolute error at most $10^{-3}$.
In your brief reasoning, justify the precision transitions, history dependence, and global interval coverage, state whether the supremum is attained, and give only a maximizing configuration and the two boundaries of its limiting safe interval as numerical checkpoints, using exact values or certified isolating intervals where needed.

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

01_build_block_factors.py

Goal
----
Construct the exact tridiagonal block factors and round their completed entries once to binary16.

```python
def build_block_factors(diagonal: "np.ndarray", multiplier: tuple) -> tuple:
    """Return the stored factors of the benchmark's four-row blocks.

    Parameters
    ----------
    diagonal : np.ndarray
        One-dimensional real finite binary64 diagonal, with positive length
        divisible by four and entries in [1, 2]. Its binary64 values are
        interpreted as exact rationals. Each contiguous four entries defines
        one tridiagonal block with subdiagonal -3/16 and superdiagonal -7/16.
    multiplier : tuple
        Two Python integers (numerator, denominator), denominator positive,
        representing an exact value in [1, 2]. Fractions need not be reduced.
        Multiply only the block diagonals by this value. Factor exactly in
        natural order with unit lower diagonals, then round each completed
        factor entry once to IEEE binary16, nearest with ties to even.

    Returns
    -------
    lower, upper : tuple of np.ndarray
        Binary16 arrays of shape (len(diagonal)//4, 4). lower[:, 0] is zero;
        lower[:, 1:] holds the three strict lower subdiagonal entries.
        upper holds the four upper diagonal entries. Unit lower diagonals
        and the upper superdiagonal -7/16 are implicit. No input is changed.

    Raises
    ------
    ValueError
        If diagonal is not a nonempty real finite vector of length divisible
        by four with entries in [1, 2], or multiplier is not an integer pair
        with positive denominator and value in [1, 2].
    """
    return lower, upper  # placeholder
```

### Step 2

02_factor_rounding_regions.py

Goal
----
Partition the closed multiplier domain into all factor-rounding point and open-interval states.

```python
def factor_rounding_regions(diagonal: "np.ndarray", domain: tuple) -> tuple:
    """Enumerate every stored factor state, including isolated boundaries.

    Parameters
    ----------
    diagonal : np.ndarray
        Real finite vector of length 4*b, b >= 1, with entries in [1, 2].
        Block interpretation and rounding are those of build_block_factors.
    domain : tuple
        Two integer rational pairs (left, right), defining a closed interval
        1 <= left <= right <= 2. A singleton interval is permitted.

    Returns
    -------
    boundaries, strata, lower_states, upper_states : tuple
        boundaries is a tuple in increasing root order. Each entry is
        (coefficients, low, high): coefficients are the primitive integer
        coefficients in descending degree, with positive leading coefficient,
        of a polynomial specifying the boundary; low and high are integer
        rational pairs isolating its unique root. Rational boundaries have
        low == high and a linear polynomial. Each bracket has width <= 1e-24.
        Only domain endpoints and genuine internal factor transitions occur.
        strata is an int64 array of shape (2*len(boundaries)-1, 3), in spatial
        order: (left_boundary_id, right_boundary_id, state_id). Equal endpoint
        IDs denote a singleton; consecutive IDs denote an OPEN interval.
        Endpoints use their exact ties-to-even states, not one-sided limits.
        lower_states and upper_states are binary16 arrays of shape (s, b, 4).
        State IDs are assigned at first occurrence in strata, with duplicate
        full-factor states sharing an ID. All returned values are numeric;
        no symbolic expressions or strings occur in the result.

    Raises
    ------
    ValueError
        If diagonal violates its stated domain, domain is not two integer
        rational pairs with positive denominators, or its bounds do not
        satisfy 1 <= left <= right <= 2.
    """
    return boundaries, strata, lower_states, upper_states  # placeholder
```

### Step 3

03_apply_primary.py

Goal
----
Apply the stored block factors in the requested arithmetic precision, using two triangular solves.

```python
def apply_primary(lower: "np.ndarray", upper: "np.ndarray", rhs: "np.ndarray", precision: int) -> "np.ndarray":
    """Apply the benchmark's stored block-Jacobi inverse action.

    Parameters
    ----------
    lower, upper : np.ndarray
        Real finite arrays with common shape (b, 4), b >= 1, exactly
        representable in binary16. lower[:, 0] must be zero and every upper
        entry must be nonzero. Unit lower diagonals and the upper
        superdiagonal -7/16 are implicit; see build_block_factors.
    rhs : np.ndarray
        Finite real vector of length 4*b, representable as finite values in
        the requested precision. Cast before either triangular solve.
    precision : int
        16, 32, or 64. Perform each scalar product, subtraction and division
        in that format, nearest-even, with subnormals retained and no FMA.
        The lower solve proceeds in increasing row order; the upper solve
        in decreasing row order. Do not form an inverse or round just once
        at the end. The factors remain stored binary16 values in all cases.

    Returns
    -------
    result : np.ndarray
        Vector of length 4*b in the requested floating-point dtype.
        Inputs are not modified. A zero rhs returns a zero vector.

    Raises
    ------
    ValueError
        If factor shapes or values violate the stated contract, precision
        is not 16, 32, or 64, rhs has the wrong shape or non-real/nonfinite
        entries, its cast is nonfinite, or a computed result is nonfinite.
    """
    return result  # placeholder
```

### Step 4

04_richardson_step.py

Goal
----
Execute one two-correction Richardson invocation and return its persistent adaptive state.

```python
def richardson_step(A: "np.ndarray", rhs: "np.ndarray", lower: "np.ndarray", upper: "np.ndarray",
                    weights: "np.ndarray", counter: int, interval: int, adaptive: bool = True) -> tuple:
    """Run the innermost two-iteration solver from a zero iterate.

    Parameters
    ----------
    A : np.ndarray
        Finite real square matrix of dimension n = 4*b with a finite
        binary16 representation. Store A in binary16 for this invocation.
    rhs : np.ndarray
        Real finite vector of length n; cast to binary16 on entry.
    lower, upper : np.ndarray
        Stored block factors satisfying apply_primary's contract.
    weights : np.ndarray
        Two finite binary16-representable stored weights; not modified.
    counter : int
        One-based positive invocation count. If counter == 1, initialize
        both weights to one, irrespective of the supplied weights.
    interval : int
        Positive update interval. A scheduled call has counter % interval == 0.
    adaptive : bool
        If True, use Algorithm 1. Compute each scheduled local weight and
        its average in binary32 using promoted stored binary16 A, factors,
        and residual; that weight computation uses a fresh binary32 primary
        solve. The current correction always uses a binary16 primary solve.
        On a scheduled call use the fresh local weight in a binary32 iterate
        update and then cast to binary16; store the running average in
        binary16. Otherwise use the stored weight entirely in binary16.
        Average with ell = counter//interval, including the initial unit
        weight. If False, use fixed unit weights and do no weight updates.
        In both modes form the second residual from rhs - A*z, not an
        incrementally propagated residual, and increment counter once.
        All arithmetic uses separate nearest-even operations, retained
        subnormals, increasing-index sums, and no FMA.

    Returns
    -------
    z, new_weights, next_counter : tuple
        z is binary16 shape (n,), new_weights is binary16 shape (2,), and
        next_counter is a Python integer. No input is modified.

    Raises
    ------
    ValueError
        If any matrix, vector, factor, or state violates its stated domain;
        if adaptive is not Boolean; if a storage cast or arithmetic result
        is nonfinite; or if a scheduled local-weight denominator is zero.
    """
    return z, new_weights, next_counter  # placeholder
```

### Step 5

05_nested_fgmres_cycle.py

Goal
----
Run one mixed-precision flexible GMRES cycle, using the next nested solver as its changing preconditioner.

```python
def nested_fgmres_cycle(A: "np.ndarray", rhs: "np.ndarray", lower: "np.ndarray", upper: "np.ndarray",
                        iterations: tuple, interval: int, weights: "np.ndarray", counter: int,
                        level: int = 0, initial: "np.ndarray | None" = None, adaptive: bool = True) -> tuple:
    """Return one prescribed FGMRES cycle and the updated Richardson state.

    Parameters
    ----------
    A, rhs, lower, upper
        Matrix, vector and stored block factors as in richardson_step.
    iterations : tuple
        Three positive integers, each <= n, for the outer, middle and third
        FGMRES lengths. Richardson always has two corrections per invocation.
    interval, weights, counter, adaptive
        Adaptive controls as in richardson_step. State is carried through
        every child call and is returned, not reset at a level boundary.
    level : int
        0, 1, or 2. Level 0 uses binary64 matrix/vector/arithmetic. Level 1
        uses binary32 matrix/vector/arithmetic. Level 2 uses a binary16
        matrix with binary32 vectors/arithmetic and Richardson as its child.
        Higher-level children are recursively the next FGMRES level.
    initial : np.ndarray or None
        None means a zero starting iterate and residual equal to rhs. An
        explicit vector is cast to this level's vector format and uses the
        freshly computed residual rhs - A_level*initial. Inner calls always
        use None. rhs is cast to the receiving level on entry.

        Use one classical Gram-Schmidt pass: all coefficients from the same
        incoming vector, accumulate the projection by increasing basis index,
        then subtract once. Norms are increasing-index sums of separately
        rounded squares followed by a square root. Apply previous Givens
        rotations in increasing order; a new pair (a,b) uses positive
        rho=sqrt(a*a+b*b), c=a/rho, s=b/rho and becomes (rho,0). Apply each
        rotation to the reduced rhs too. Back substitution uses increasing
        column-index sums. Combine the stored child corrections in increasing
        basis order and add the accumulated correction once to initial.
        Use nearest-even separate operations, retained subnormals, and no
        fusion, reorthogonalization, library least-squares substitution or
        convergence-based early exit.

    Returns
    -------
    x, new_weights, next_counter : tuple
        x has this level's vector dtype and shape (n,). Stored weights are
        binary16 shape (2,), and next_counter is a Python integer. No input
        is modified. Only Richardson calls advance the counter.

    Raises
    ------
    ValueError
        If any supplied value violates the above shape, type or numeric
        domains; if iterations is not three integers in [1,n]; if level is
        not 0,1,2; if the initial residual norm, an Arnoldi norm, a Givens
        denominator or a reduced triangular pivot is zero; or if any child
        call has an invalid local-weight denominator or nonfinite arithmetic.
        Such breakdowns are errors for this fixed-budget interface, not
        silent early-convergence branches.
    """
    return x, new_weights, next_counter  # placeholder
```

### Step 6

06_batch_residuals.py

Goal
----
Evaluate fixed and adaptive F3R for every ordering of a batch while preserving adaptive history within each ordering.

```python
def batch_residuals(A: "np.ndarray", right_hand_sides: "np.ndarray", lower: "np.ndarray", upper: "np.ndarray",
                    theta: tuple, outer_iterations: int = 2, restarts: int = 2) -> tuple:
    """Evaluate all complete right-hand-side permutations at one factor state.

    Parameters
    ----------
    A, lower, upper
        Original matrix and stored block factors as in nested_fgmres_cycle.
    right_hand_sides : np.ndarray
        Real finite shape (r,n), with 1 <= r <= 3 and each row nonzero.
    theta : tuple
        Three positive integers (m2,m3,interval), with m2,m3 <= n.
    outer_iterations, restarts : int
        Positive integers; outer_iterations <= n. Each system uses exactly
        restarts outer cycles and starts from zero. Every outer restart
        retains its solution and Richardson state. At the start of each
        independent permutation, initialize weights=(1,1) and counter=1.
        Subsequent systems reset their solution but retain weights/counter.
        The fixed control uses unit weights and no updates. Use all six
        permutations for r=3 (all r! in general), in lexicographic order.
        Shared-prefix reuse is allowed only for identical ordered prefixes.

    Returns
    -------
    fixed, adaptive, final_weights, final_counters : tuple
        fixed and adaptive are binary64 arrays of shape (r!,r). Column t
        refers to the system at position t of that lexicographic permutation.
        Residuals are norm(b-A*x)/norm(b), using original A and binary64
        increasing-index products/sums and direct sum-of-squares norms.
        final_weights is binary16 shape (r!,2), and final_counters is int64
        shape (r!,), for the ADAPTIVE batches. Inputs are not modified.

    Raises
    ------
    ValueError
        If inputs violate the stated domains, a right-hand side is zero,
        or any child solver encounters a documented zero denominator,
        breakdown, overflowing storage cast or nonfinite arithmetic.
    """
    return fixed, adaptive, final_weights, final_counters  # placeholder
```

### Step 7

07_safe_component_margins.py

Goal
----
Find connected safe multiplier components and enclose their relative uncertainty margins.

```python
def safe_component_margins(boundaries: tuple, strata: "np.ndarray", gains: "np.ndarray",
                          threshold: float = 2.0) -> "np.ndarray":
    """Return all safe components with certified endpoint-derived margins.

    Parameters
    ----------
    boundaries : tuple
        Numeric boundary descriptors from factor_rounding_regions. Each is
        (integer_polynomial_coefficients, rational_low, rational_high).
        Brackets lie in [1,2], are in strictly increasing disjoint order,
        and have width <= 1e-24. A rational singleton bracket is allowed.
        Polynomials identify the boundaries; numerical enclosures below use
        the supplied brackets, which must isolate those boundaries.
    strata : np.ndarray
        Integer array from factor_rounding_regions, shape (2*b-1,3): all
        singleton and open-interval rows in alternating spatial order, with
        valid nonnegative stored-state IDs.
    gains : np.ndarray
        Finite nonnegative binary64 array of shape (number_of_states,k),
        k >= 1. Entry (s,j) is the worst matching fixed/adaptive residual
        ratio for state s and configuration j. Use threshold comparisons
        without preliminary rounding; equality is safe.
    threshold : float
        Positive finite real number.

    Returns
    -------
    components : np.ndarray
        Binary64 shape (number_of_components,7), sorted first by configuration
        column, then by left boundary. Row fields are configuration_id,
        left_boundary_id, right_boundary_id, left_closed, right_closed,
        lower_margin_ppm, upper_margin_ppm. Closure flags are 0 or 1.
        Margins enclose 1e6*(b-a)/(b+a) by exact rational endpoint arithmetic
        followed by outward-rounded binary64 bounds. A safe singleton has
        zero margin. A component's supremum is attained exactly when both
        endpoints are included. No safe states gives shape (0,7).

    Raises
    ------
    ValueError
        If descriptors are malformed, brackets are outside [1,2], reversed,
        overlapping, or wider than 1e-24; strata do not form the complete
        alternating partition; gains have invalid shape/entries or missing
        state IDs; or threshold is not finite and positive. Polynomial-root
        validity is a precondition provided by factor_rounding_regions.
    """
    return components  # placeholder
```

### Step 8

08_solve_order_robust_margin.py

Goal
----
Run the full exact-region, mixed-precision batch calculation and return the largest guaranteed robustness margin.

```python
def solve_order_robust_margin(domain: tuple = ((287, 256), (289, 256)),
                             configurations: tuple | None = None, threshold: float = 2.0) -> float:
    """Compute the complete benchmark margin in parts per million.

    Parameters
    ----------
    domain : tuple
        Closed rational-pair subinterval of [287/256,289/256], including a
        singleton if requested. Use the factor_rounding_regions descriptor.
    configurations : tuple or None
        Nonempty tuple of distinct (m2,m3,c) triples with (m2,m3) among
        (3,4),(4,3),(6,2) and c among 5,11,17,23,31. None means all 15 triples,
        ordered by that nesting order and increasing c.
    threshold : float
        Finite positive required fixed/adaptive true-residual ratio.
        The prompt uses 2.0. Other values are test instances of the same
        robustness calculation, not changes to the solver algorithm.

        Construct the ORIGINAL n=48 matrix using zero-based indices modulo
        48: A[i,i]=1+(1+i%3)/4096, A[i,i+1]=-7/16, A[i,i-1]=-3/16,
        A[i,i+8]=-1/4, A[i,i-8]=-1/8; unspecified entries are zero.
        The three rhs rows have entries:
        b0[i]=1+(-1)**i/4+(i%7)/32,
        b1[i]=(-1)**i*(1+(i%5)/16),
        b2[i]=(((3*i)%11)-5)/8+(i%2)/32.
        Use four-row block factors, three nested FGMRES levels, two
        Richardson corrections, and exactly two outer cycles of length two.
        For every factor state and triple evaluate all ordered batches using
        batch_residuals. The gain is the minimum matching fixed/adaptive
        ratio over all permutations and positions. Resolve every real
        multiplier through the exact factor partition, not a sampled grid.

    Returns
    -------
    margin_ppm : float
        Native Python float within 1e-6 absolute error of the supremal
        1e6*d, for some common configuration and interval
        [alpha*(1-d), alpha*(1+d)] wholly contained in the safe domain.
        Internal values are not rounded to three decimals. A safe singleton
        has margin 0.0. An empty safe set has no finite supremal margin and
        raises ValueError rather than returning an invented zero.

    Raises
    ------
    ValueError
        If domain is not a valid closed rational subinterval, configurations
        are empty, duplicated or outside the allowed set, threshold is not
        finite and positive, any child calculation has a documented
        breakdown/nonfinite result, any adaptive residual used as a ratio
        denominator is zero, or the safe set is empty.
    """
    return margin_ppm  # placeholder
```
