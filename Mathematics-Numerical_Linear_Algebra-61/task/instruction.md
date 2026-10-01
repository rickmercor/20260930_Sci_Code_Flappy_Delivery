# Mathematics-Numerical_Linear_Algebra-61

## Background

Iterative methods for overdetermined linear systems often never form A explicitly. They see the residual through short random sketches, then take cheap updates that look like stochastic gradient steps on a least-squares objective. When some rows of A are much better conditioned than others, those sketched updates mix the two blocks, and the contraction tracks the scaled condition of the full matrix rather than of the well-conditioned part alone. How a solver splits rows, enforces a subset of equations, and builds the next search direction from a sketch is not unique. The leftover quality of a short run is then a single coordinate of one iterate, not a residual norm and not the dense least-squares solution.

## Problem

A tall consistent linear system Ax = b can be treated by a short sketched iteration that holds a few equations exactly and sees the remaining residual only through Gaussian sketches. Different constraint sets and different uses of those sketches produce different iterates.

Use numpy Generator default_rng(7) to form thin QR factors of standard_normal draws of shapes (12, 8) and (8, 8), and set A = U diag(s) V^T with s = (8, 6, 4.5, 1.1, 0.8, 0.55, 0.4, 0.3); draw a right-hand side that is exactly A times a standard_normal(8) vector from the same generator, so the system is consistent. The constraint rows, indexing from 0, are 2, 8, and 3. Using default_rng(11), draw three Gaussian sketches of shape (9, 2) in order and use them on the complementary nine rows. Take three steps of the paper's constrained sketched iteration for this system, using those sketches in order, with truncation length 2. Your tagged final answer must be a single number: the first coordinate of the resulting iterate, indexing from 0. In <reasoning>, report these computed scalars: x^0_0, d^0_0, the first coordinate after one step and after two steps, δ_0, δ_1, δ_2, the second-step Gram–Schmidt coefficient and p^1_0, the third-step Gram–Schmidt coefficient, S_{0,00}, S_{1,00}, S_{2,00}, and S_0^T r^0. The tagged final answer remains only the first coordinate after three steps.

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

01_construct_consistent_system

Goal
----
A two-cluster singular spectrum is the regime in which an unconstrained sketched gradient mixes a well-conditioned row block with an ill-conditioned complement. This step manufactures a reproducible overdetermined consistent instance so later constrained sketches see a known deterministic pair (A, b).

```python
import numpy as np


def construct_consistent_system(m: int, n: int, s: np.ndarray, seed: int) -> np.ndarray:
    """Pack $A$ of shape $(m, n)$ and a consistent right-hand side $b$.

    Parameters
    ----------
    m : int
        Row dimension, $m > n \ge 2$.
    n : int
        Column dimension.
    s : np.ndarray
        Positive singular values, shape $(n,)$.
    seed : int
        RNG seed for numpy Generator default_rng.

    Returns
    -------
    pack : np.ndarray
        Concatenation of $(m, n)$, $A.\mathrm{ravel}()$, and $b$.

    Raises
    ------
    ValueError
        If $m$ or $n$ is not an integer, if not $m > n \ge 2$, if $s$ does
        not have shape $(n,)$, or if $s$ is not positive and finite.
    """
    return np.zeros(1)
```

### Step 2

02_extract_row_blocks

Goal
----
Split a packed (A, b) into constraint rows A_p, b_p and complementary rows A_r, b_r for a 0-based index set I. Require 1 <= mp < m and distinct indices.

```python
import numpy as np


def extract_row_blocks(pack: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Pack A_p, b_p, A_r, b_r for 0-based constraint row indices.

    Parameters
    ----------
    pack : np.ndarray
        Packed $(A, b)$ from construct_consistent_system.
    indices : np.ndarray
        Distinct 0-based constraint rows, length $m_p$ with $1 \le m_p < m$.

    Returns
    -------
    block_pack : np.ndarray
        Concatenation of $(m_p, n, m_r)$, $A_p.\mathrm{ravel}()$, $b_p$,
        $A_r.\mathrm{ravel}()$, $b_r$.

    Raises
    ------
    ValueError
        If pack is too short or its length does not match the header, if
        packed shapes are not positive, if indices are not distinct, if
        not $1 \le m_p < m$, or if any index is out of range.
    """
    return np.zeros(1)
```

### Step 3

03_constraint_initial_iterate

Goal
----
Return the minimum-norm solution $x_0 = A_p^+ b_p$ of the packed constraint block.

```python
import numpy as np


def constraint_initial_iterate(block_pack: np.ndarray) -> np.ndarray:
    """Return $x_0 = A_p^+ b_p$ from a packed row split.

    Parameters
    ----------
    block_pack : np.ndarray
        Packed $(A_p, b_p, A_r, b_r)$ from extract_row_blocks.

    Returns
    -------
    x0 : np.ndarray
        Vector of length $n$.

    Raises
    ------
    ValueError
        If block_pack is too short, if packed shapes are not positive, if
        the pack length does not match the header, or if the constraint
        block is not a finite 2d array with a matching right-hand side.
    """
    return np.zeros(1)
```

### Step 4

04_draw_gaussian_sketches

Goal
----
Draw $n_{\mathrm{steps}}$ independent Gaussian sketches of shape $(m_r, q)$ from default_rng(seed) and pack them in order. Require $m_r$, $q$, $n_{\mathrm{steps}} \ge 1$.

```python
import numpy as np


def draw_gaussian_sketches(m_r: int, q: int, n_steps: int, seed: int) -> np.ndarray:
    """Pack $n_{\mathrm{steps}}$ independent Gaussian matrices of shape $(m_r, q)$.

    Parameters
    ----------
    m_r : int
        Complementary row count, $m_r \ge 1$.
    q : int
        Sketch width, $q \ge 1$.
    n_steps : int
        Number of sketches, $n_{\mathrm{steps}} \ge 1$.
    seed : int
        RNG seed for numpy Generator default_rng.

    Returns
    -------
    pack : np.ndarray
        Concatenation of $(n_{\mathrm{steps}}, m_r, q)$ and the sketches in order.

    Raises
    ------
    ValueError
        If $m_r$, $q$, or $n_{\mathrm{steps}}$ is not an integer, or if
        any of them is less than 1.
    """
    return np.zeros(1)
```

### Step 5

05_projected_sketched_direction

Goal
----
Return the sketched search direction at the current iterate after it has been made tangent to the affine constraint. The sketch $S$ acts on the complementary residual. Require $S$ of shape $(m_r, q)$ with $q \ge 1$.

```python
import numpy as np


def projected_sketched_direction(
    block_pack: np.ndarray, x: np.ndarray, S: np.ndarray
) -> np.ndarray:
    """Return the constraint-feasible sketched direction at x.

    Parameters
    ----------
    block_pack : np.ndarray
        Packed $(A_p, b_p, A_r, b_r)$.
    x : np.ndarray
        Current iterate, length $n$.
    S : np.ndarray
        Sketch of shape $(m_r, q)$, $q \ge 1$, acting on the complementary residual.

    Returns
    -------
    d : np.ndarray
        Constraint-feasible sketched direction of length $n$.

    Raises
    ------
    ValueError
        If block_pack is malformed, if $x$ does not have shape $(n,)$, or if
        $S$ does not have shape $(m_r, q)$ with $q \ge 1$.
    """
    return np.zeros(1)
```

### Step 6

06_window_orthogonalize

Goal
----
Return a new search direction obtained by orthogonalizing d against a window of previous directions. An empty window of shape (n, 0) returns d. Previous columns must be nonzero.

```python
import numpy as np


def window_orthogonalize(d: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Return the direction after orthogonalizing $d$ against the window.

    Parameters
    ----------
    d : np.ndarray
        New projected direction, length $n \ge 1$.
    window : np.ndarray
        Previous directions, shape $(n, w)$ with $w \ge 0$. An empty window
        ($w = 0$) returns $d$ unchanged.

    Returns
    -------
    p : np.ndarray
        Orthogonalized direction of length $n$.

    Raises
    ------
    ValueError
        If $d$ is not a nonempty finite vector, if window does not have
        shape $(n, w)$, or if a window direction is zero.
    """
    return np.zeros(1)
```

### Step 7

07_line_search_step

Goal
----
Advance $x$ by an exact line search along the nonzero direction $p$, using the complementary residual as seen through $S$. Require $S$ of shape $(m_r, q)$.

```python
import numpy as np


def line_search_step(
    A_r: np.ndarray, b_r: np.ndarray, x: np.ndarray, S: np.ndarray, p: np.ndarray
) -> np.ndarray:
    """Return the iterate after an exact line search along $p$.

    Parameters
    ----------
    A_r : np.ndarray
        Complementary rows, shape $(m_r, n)$.
    b_r : np.ndarray
        Complementary right-hand side, length $m_r$.
    x : np.ndarray
        Current iterate, length $n$.
    S : np.ndarray
        Sketch of shape $(m_r, q)$, $q \ge 1$.
    p : np.ndarray
        Nonzero search direction, length $n$.

    Returns
    -------
    x_new : np.ndarray
        Updated iterate of length $n$.

    Raises
    ------
    ValueError
        If $A_r$ is not a nonempty 2d array, if $b_r$, $x$, or $p$ has an
        incompatible shape, if $S$ does not have shape $(m_r, q)$ with
        $q \ge 1$, or if the search direction is zero.
    """
    return np.zeros(1)
```

### Step 8

08_run_constrained_sketch_entry

Goal
----
Run the constrained sketched iteration by calling the earlier sub-problem functions. Return coordinate coord of the iterate after n_steps. Require q, n_steps, ell >= 1 and 0 <= coord < n.

```python
import numpy as np


def run_constrained_sketch_entry(
    m: int,
    n: int,
    s: np.ndarray,
    data_seed: int,
    sketch_seed: int,
    indices: np.ndarray,
    q: int,
    n_steps: int,
    ell: int,
    coord: int,
) -> float:
    """Run the constrained sketched iteration; return one coordinate.

    Parameters
    ----------
    m : int
        Row dimension, $m > n \ge 2$.
    n : int
        Column dimension.
    s : np.ndarray
        Positive singular values, shape $(n,)$.
    data_seed : int
        RNG seed for the consistent system.
    sketch_seed : int
        RNG seed for the Gaussian sketches.
    indices : np.ndarray
        Distinct 0-based constraint rows, length $m_p$ with $1 \le m_p < m$.
    q : int
        Sketch width, $q \ge 1$.
    n_steps : int
        Number of sketched steps, $n_{\mathrm{steps}} \ge 1$.
    ell : int
        Orthogonalization window length, $\ell \ge 1$.
    coord : int
        0-based output coordinate, $0 \le \mathrm{coord} < n$.

    Returns
    -------
    entry : float
        Coordinate $\mathrm{coord}$ of the iterate after $n_{\mathrm{steps}}$.

    Raises
    ------
    ValueError
        If $q$, $n_{\mathrm{steps}}$, or $\ell$ is not an integer $\ge 1$,
        or if $\mathrm{coord}$ is out of range.
    """
    return 0.0
```
