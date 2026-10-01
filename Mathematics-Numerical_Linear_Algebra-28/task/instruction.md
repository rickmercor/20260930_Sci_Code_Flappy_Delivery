# Mathematics-Numerical_Linear_Algebra-28

## Background

Large Hermitian eigenproblems often require only a small number of extremal states rather than the full spectrum. Block iterative eigensolvers address this setting by building a low-dimensional search space and repeatedly refining it through spectral extraction and correction steps.

The source study describes a randomized variant in which the geometry used for search-space orthogonalization is modified while the reduced spectral extraction remains an exact generalized eigenvalue computation. The problem statement above instantiates that procedure as a deterministic finite calculation.

The task focuses on the numerical realization of this procedure rather than on any particular application domain. Follow the source study for the algorithmic operations and use the problem statement's fixed parameters and deterministic random stream without substituting an alternative method.

## Problem

Compute one deterministic finite instance of a sketched block eigensolver for the sparse symmetric matrix family used in the source study. The matrix dimension is $n=96$, the number of wanted eigenpairs is $k=3$, the restart dimensions are $j_{\min}=3$ and $j_{\max}=9$, the sketch dimension is $s=5j_{\max}=45$, and the embedding sparsity is $\zeta=4$.

Define $A\in\mathbb{R}^{96\times96}$ by

$$
A_{ii}=\ln(99+i),
\qquad i=1,\ldots,96,
$$

and

$$
A_{i,i+1}=A_{i+1,i}
=
0.19\sin(0.73i)+0.004\cos(0.41i),
\qquad i=1,\ldots,95,
$$

with all other entries zero.

Initialize the three-column block $V_0$ from the mutually orthonormal vectors

$$
v^{(1)}_j=\frac{1}{\sqrt{96}},
\qquad
v^{(2)}_j=\frac{(-1)^{j-1}}{\sqrt{96}},
\qquad
v^{(3)}_j=\frac{q_j}{\sqrt{96}},
$$

where $q_j$ is the periodic sequence $1,1,-1,-1$.

Initialize one random-number generator as

$$
\texttt{rng}=\texttt{numpy.random.default\_rng(2026)}.
$$

Construct one fixed sketch $S\in\mathbb{R}^{45\times96}$ by iterating over the matrix columns. For each column, draw four distinct sketch rows using

$$
\texttt{rng.choice(45,4,replace=False)}
$$

and immediately draw four independent signs using

$$
2\,\texttt{rng.integers(0,2,4)}-1.
$$

Place the resulting signed values divided by $2$ at the selected rows. Use this single continuous random-generator stream throughout the construction, with no reseeding or resetting.

For the remaining numerical procedure, use the source study's conventions for the following operations: the sketch-orthonormalization of the initial block, the full-space generalized projected eigenproblem and Ritz-vector construction, and the first restart transformation. These conventions are intentionally not restated here and must be recovered from the source study.

Let $\widetilde V_j\in\mathbb{R}^{96\times j}$ denote the current $j$-column full-space search basis, let $Q_j=S\widetilde V_j$ denote its stored sketch, and let $W_j=A\widetilde V_j$ denote its stored operator image.

For each retained Ritz pair $(\theta_i,u_i)$, form the Davidson correction directly as

$$
t_i=(D-\theta_i I)^{-1}r_i,
$$

where

$$
D=\operatorname{diag}(A),
\qquad
r_i=Au_i-\theta_i u_i.
$$

Sketch-orthonormalize each newly generated correction block according to the source study's procedure, retain all three residuals as active by using a zero convergence threshold, do not draw or recompute a new sketch at the restart, and use IEEE-754 double precision for both full-space and sketched numerical quantities throughout.

Begin from $j=3$. Perform exactly two expansion iterations, so that the first search space reaches $j=9$. Execute exactly one restart, retaining the three lowest Ritz states according to the source procedure, and then perform exactly two further expansion iterations so that the second search space again reaches $j=9$. Stop immediately after the second-cycle Ritz extraction, before applying a second restart.

For each projected eigensolve, use the exact full-space matrices

$$
H_j=\widetilde V_j^{T}A\widetilde V_j,
\qquad
G_j=\widetilde V_j^{T}\widetilde V_j,
$$

and retain the three smallest Ritz values with the normalization specified by the source procedure. Fix the sign of each full-space Ritz vector

$$
U=\widetilde V_jY
$$

so that its first nonzero component is positive.

In the reasoning, report the three eigenvalues of $G_3$, its spectral condition number $\kappa_2(G_3)$, the lowest Ritz value at the first-cycle $j=3$, $j=6$, and $j=9$ extractions, the three diagonal entries of the restarted projected matrix

$$
H_3^{\mathrm{restart}}
=
\widetilde V_3^{T}A\widetilde V_3,
$$

the lowest Ritz value at the second-cycle $j=6$ and $j=9$ extractions, the spectral condition number of the final $G_9$, the source study's sketch-embedding condition on the eigenvalues of $G_j$, and the source study's orthogonalization-cost comparison for the deterministic and randomized procedures.

The requested quantity is the spectral condition number

$$
\kappa_2(G_9)
=
\frac{\lambda_{\max}(G_9)}
{\lambda_{\min}(G_9)},
$$

where $G_9$ is the full-space Gram matrix at the final second-cycle $j=9$ extraction.

Compute this single scalar deterministically, and report it and every other requested numerical value to at least eight significant figures.

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

01_build_instance

Goal
----
Construct the complete deterministic instance used by the benchmark computation: the sparse symmetric operator, the prescribed three-column initial block and the fixed sparse-sign sketch. The operator is real, symmetric and tridiagonal with entries fixed by the benchmark specification, the initial block contains three deterministic, mutually orthonormal directions, and the sketch is generated from the supplied seed as one continuous random stream. The three objects are returned together so that the next step can consume the complete numerical state directly.

```python
def build_instance(
    n: int = 96,
    sketch_size: int = 45,
    zeta: int = 4,
    seed: int = 2026,
) -> tuple:
    """Construct the operator, the initial block and the sparse-sign sketch.

    Parameters
    ----------
    n : int
        Matrix dimension, with n >= 4 and n divisible by 4.
    sketch_size : int
        Number of rows of the sketch.
    zeta : int
        Number of nonzero entries in each sketch column,
        1 <= zeta <= sketch_size.
    seed : int
        Seed of the single random stream used for the sketch.

    Returns
    -------
    tuple
        (A, V0, S): the n-by-n operator, the n-by-3 initial block and the
        sketch_size-by-n sketch.

    Raises
    ------
    ValueError
        If n < 4, n is not divisible by 4, or any sketch parameter is
        invalid.

    Notes
    -----
    For dimension n the three columns of V0 are the constant,
    alternating-sign and period-four vectors specified by the benchmark,
    each divided by sqrt(n). The sketch is drawn column by column from
    numpy.random.default_rng(seed): for each column the rows are drawn
    with rng.choice(sketch_size, zeta, replace=False) and then the signs
    with 2*rng.integers(0, 2, zeta) - 1, without reseeding. Each selected
    entry is the drawn sign divided by sqrt(zeta) (divided by 2 for the
    benchmark zeta = 4).
    """
    return result
```

### Step 2

02_rcgs_initial

Goal
----
Sketch-orthonormalize the prescribed initial search block for the randomized eigensolver representation used by subsequent spectral computations. The operation must preserve the dimensionality of the initial block while producing a basis together with its corresponding lower-dimensional sketch representation. The operator and sketch are carried forward unchanged so that later stages can consume the complete numerical state directly.

```python
def rcgs_initial(state: tuple) -> tuple:
    """Construct the transformed initial search representation.

    Parameters
    ----------
    state : tuple
        Numerical state produced by Step 01 containing the operator,
        initial search block, and fixed sketch.

    Returns
    -------
    tuple
        Numerical state containing the quantities required by the
        generalized Ritz stage.

    Raises
    ------
    ValueError
        If the chained state is malformed, dimensions are inconsistent,
        or the supplied search block is numerically rank deficient.

    Notes
    -----
    The returned quantities must remain consistent with the fixed
    sketch used by the preceding stage.

    Column signs follow the positive-diagonal convention: the upper
    triangular factor relating the sketch of the supplied block to the
    returned sketch has positive diagonal entries.
    """
    return result
```

### Step 3

03_generalized_ritz

Goal
----
Use the current search-space state (from Step 02 or from an expansion in Step 05) to perform the source study's exact full-space generalized Rayleigh-Ritz extraction and obtain the selected approximate spectral states of the benchmark operator. The extraction operates on the reduced representation associated with the current basis and returns the selected states together with the quantities required for the next expansion stage. The operator, sketch, basis, and auxiliary representation are carried forward unchanged as part of the numerical chain.

```python
def generalized_ritz(state: tuple, k: int = 3) -> tuple:
    """Extract the selected approximate spectral states.

    Parameters
    ----------
    state : tuple
        Numerical state (A, S, Vt, Q) produced by Step 02 or by Step 05:
        operator, sketch, current full-space basis and its stored sketch.
    k : int
        Positive number of spectral states to retain, bounded by the
        available search-space dimension.

    Returns
    -------
    tuple
        Numerical state containing the propagated search information and
        the selected Ritz quantities and residual information.

    Raises
    ------
    ValueError
        If the chained state is malformed, dimensions are inconsistent,
        k is invalid, or the computed quantities are not finite.

    Notes
    -----
    The selected states must remain compatible with the generalized
    spectral problem defined by the preceding stage.
    """
    return result
```

### Step 4

04_davidson_correction

Goal
----
Generate the numerical block of new search directions from the spectral information produced by Step 03. The new directions are constructed in the original problem space and preserve the ordering and number of retained states from the preceding stage. The resulting block is returned appended to the Step 03 state; the next search-space construction stage receives it as (A, S, Vt, Q, W, T).

```python
def davidson_correction(state: tuple) -> tuple:
    """Construct the next block of search directions.

    Parameters
    ----------
    state : tuple
        Numerical state produced by Step 03 containing the operator,
        sketch, search basis, Ritz information, and residuals.

    Returns
    -------
    tuple
        Numerical state extended with the newly constructed search-direction
        information.

    Raises
    ------
    ValueError
        If the chained state is malformed, dimensions are inconsistent,
        required quantities are non-finite, or a new numerical direction
        cannot be constructed.

    Notes
    -----
    The correction for retained pair i is t_i = (D - theta_i I)^(-1) r_i
    with D = diag(A), applied componentwise. A zero denominator gives a
    zero correction when the residual component is zero, and raises
    ValueError otherwise.

    The returned state must preserve all information required by the
    expansion stage.
    """
    return result
```

### Step 5

05_sketched_block_expansion

Goal
----
Append one block of correction directions to the first-cycle search space using the source study's randomized sketch-orthogonalization. The input state holds the operator, the fixed sketch, the current full-space basis together with its stored sketch and its operator image, and the correction block produced by Step 04. The current basis is the sketch-orthonormal basis maintained during the first cycle. The output is the expanded basis with its expanded stored sketch and operator image.

```python
def sketched_block_expansion(state: tuple) -> tuple:
    """Append a correction block to the search space by sketched orthogonalization.

    Parameters
    ----------
    state : tuple
        (A, S, Vt, Q, W, T): operator (n by n), sketch (s by n), current
        full-space basis (n by j), its stored sketch (s by j), its operator
        image A Vt (n by j) and the correction block (n by p).

    Returns
    -------
    tuple
        (A, S, Vt_new, Q_new, W_new) with j + p columns in the basis, its
        stored sketch and its operator image.

    Raises
    ------
    ValueError
        If the state is malformed, the dimensions are inconsistent, the
        sketch has fewer rows than the expanded basis has columns, or the
        correction block is numerically rank deficient after projection.

    Notes
    -----
    Use a single stage-1 pass (p_s = 1) of the source's randomized
    Gram-Schmidt. Column signs follow the positive-diagonal convention:
    the upper triangular factor relating the sketch of the projected block
    to its returned sketch has positive diagonal entries.
    """
    return result
```

### Step 6

06_restart_rotation

Goal
----
Restart the search space onto the retained Ritz states as the source study prescribes. The input is the generalized Ritz state of Step 03 at the maximal search-space dimension; the output is the restarted three-column basis together with its stored sketch and its operator image, which begin the next cycle. The fixed sketch is carried forward unchanged and no new sketch is drawn.

```python
def restart_rotation(state: tuple) -> tuple:
    """Restart the search space onto the retained Ritz states.

    Parameters
    ----------
    state : tuple
        Generalized Ritz state produced by Step 03 at the maximal
        dimension: (A, S, Vt, Q, W, theta, Y, U, R).

    Returns
    -------
    tuple
        (A, S, V_restart, Q_restart, W_restart): the restarted basis, its
        stored sketch and its operator image, with one column per retained
        Ritz state.

    Raises
    ------
    ValueError
        If the state is malformed or its dimensions are inconsistent.

    Notes
    -----
    The retained Ritz coefficients are used with the signs they carry in
    the input state.
    """
    return result
```

### Step 7

07_post_restart_expansion

Goal
----
Apply one sketched correction expansion to the second-cycle search state that begins at the restart. The input is the current second-cycle basis (the restarted three-column basis from Step 06, followed by any correction blocks already appended in this cycle), its stored sketch and operator image, together with one active correction block. The output appends that block and returns the updated basis, stored sketch and operator image for the next second-cycle extraction.

```python
def post_restart_expansion(state: tuple) -> tuple:
    """Apply the source-defined second-cycle expansion to a restarted state.

    Parameters
    ----------
    state : tuple
        (A, S, Vt, Q, W, T): operator, fixed sketch, current second-cycle
        full-space basis, stored sketch, operator image and one active
        correction block. The basis starts with the three-column output of
        the restart stage and may already contain correction blocks appended
        earlier in the second cycle.

    Returns
    -------
    tuple
        (A, S, Vt_new, Q_new, W_new) after appending the correction block.

    Raises
    ------
    ValueError
        If the state is malformed, dimensions are inconsistent, the sketch
        is too small for the expanded basis, or the correction block is
        numerically rank deficient after projection.

    Notes
    -----
    This step represents one second-cycle expansion. Carry the fixed
    sketch forward unchanged and do not restart or re-sketch-orthonormalize
    the columns already in the basis here.
    """
    return result
```

### Step 8

08_final_orchestrator

Goal
----
Complete the deterministic numerical computation and return the spectral condition number of the final full-space Gram matrix.

```python
def solve_once_restarted_ritz_entry(
    n: int = 96,
    k: int = 3,
    jmax: int = 9,
    sketch_size: int = 45,
    zeta: int = 4,
    seed: int = 2026,
) -> float:
    """Run the complete deterministic calculation and return the final Gram condition number.

    Parameters
    ----------
    n : int
        Matrix dimension.
    k : int
        Number of retained spectral states.
    jmax : int
        Maximum search-space dimension.
    sketch_size : int
        Number of rows in the sketch.
    zeta : int
        Positive sketch sparsity parameter.
    seed : int
        Seed for the deterministic sketch construction.

    Returns
    -------
    float
        Spectral condition number of the final full-space Gram matrix.

    Raises
    ------
    ValueError
        If any numerical parameter is invalid, the dimensions are
        inconsistent, or the final Gram matrix is not positive definite.

    Notes
    -----
    The returned scalar is the final spectral condition number specified by the benchmark.
    """
    return result
```
