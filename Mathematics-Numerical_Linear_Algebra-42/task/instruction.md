# A magnitude-ordered half-line eigenvalue from stable Laguerre collocation

## Background

# Scientific background

Pseudospectral collocation replaces differential operators by dense matrices acting on function values at carefully chosen nodes. On the half-line $[0,\infty)$, Laguerre nodes are a natural choice because their basis functions incorporate exponential decay and therefore match many boundary-value and quantum-mechanical problems better than a truncated bounded interval.

The difficulty is numerical rather than formal. Classical formulas can evaluate high-degree Laguerre polynomials and exponentially weighted factors separately, creating intermediate overflow, underflow, and cancellation even when the desired differentiation-matrix entries are moderate. The paper adapts the Glaser--Liu--Rokhlin predictor/corrector method so local Taylor propagation generates classical Laguerre roots and their weighted derivatives simultaneously, using a modified recurrence only during initialization. Those quantities then feed source-specific generalized-Laguerre coefficient and diagonal formulas for the first- and second-order matrices. Because each analytic diagonal depends on its full-grid index, the generalized parameter, and the original degree, selected principal blocks do not become smaller collocation problems. Together these choices extend the usable degree in ordinary floating-point arithmetic without symbolic algebra or arbitrary precision.

The Woods–Saxon profile provides a sensitive application because coordinate scaling affects both the differential operator and the potential sampling points. Boundary elimination, quadratic operator scaling, stable sampling of the decaying profile, removal of infinite generalized modes in homogeneous form, and one-based spectral ordering must therefore be applied consistently before a single mode magnitude can be interpreted.

## Problem

Compute one bound-state spectral value for the dimensionless half-line equation $-y''(x)+y(x)=\lambda q(x)y(x)$ by using the source-linked stable augmented Laguerre pseudospectral construction in double-precision arithmetic and without randomness. Fix $\alpha=0$, let $\widetilde{x}_1<\cdots<\widetilde{x}_{160}$ be the positive roots of $L_{160}$, and prepend the endpoint $\widetilde{x}_0=0$, so the augmented grid has $N=161$ nodes in increasing order. Use the scale $\beta=10$ and physical nodes $x_i=\widetilde{x}_i/\beta$, with first- and second-derivative operators transformed by $\beta$ and $\beta^2$, respectively. Construct the augmented differentiation matrix using the stable weighted-function coefficient ratios, the direct diagonal entries, and the source's second-order recurrence. Enforce $y(0)=0$ by deleting the endpoint row and column from the scaled second-derivative matrix, take $A=-D^{(2)}_{1:,1:}+I_{160}$, and form $Q=\operatorname{diag}(q(x_1),\ldots,q(x_{160}))$ with $q(x)=\bigl(1+\exp((x-R)/a)\bigr)^{-1}$, $R=5.08685476$, and $a=0.929852862$. Solve $Ay=\lambda Qy$, retain finite generalized eigenvalues, order their absolute values increasingly, and report the 25th value using one-based spectral ordering. All quantities are dimensionless, array storage is zero-based, and the required result is exactly one finite decimal.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.
You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you
are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that
determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration
paths, or per-fold candidate tables.

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

01_compute_glr_laguerre_state

Goal
----
Compute the classical Laguerre nodes and weighted derivatives simultaneously.



Use the source's modified Glaser--Liu--Rokhlin construction rather than a

Jacobi eigensolve followed by global polynomial evaluation.  The distinction

is consequential beyond the floating-point boundary where otherwise valid

root routines or separately evaluated exponential factors cease to return a

complete finite state.

```python
def compute_glr_laguerre_state(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the augmented GLR nodes and weighted root derivatives.

    ``n`` must be a non-boolean integer of at least one.  The returned nodes
    contain zero followed by the increasing positive roots of the degree-``n``
    classical Laguerre polynomial.  The second vector contains the derivative
    of its exponentially weighted Laguerre function at those roots, generated
    as part of the same predictor/corrector construction.  Invalid or
    nonfinite states raise ``ValueError``.

    Parameters
    ----------
    n : int
        Number of positive Laguerre nodes.

    Returns
    -------
    nodes : np.ndarray
        Float vector of shape ``(n + 1,)`` beginning with zero.
    weighted_derivatives : np.ndarray
        Finite nonzero float vector of shape ``(n,)`` in root order.
    """
    return nodes, weighted_derivatives  # noqa: F821 - required model stub
```

### Step 2

02_build_truncated_collocation_state

Goal
----
Build coupled first- and second-derivative blocks from a weighted grid state.



Implement the source's augmented generalized-Laguerre Tables 2 and 3 as one

operation.  A noncontiguous principal block retains the original polynomial

degree and parameter; it is not the differentiation matrix of the selected

nodes viewed as a smaller grid.

```python
def build_truncated_collocation_state(
    n: int,
    alpha: float,
    nodes: np.ndarray,
    weighted_derivatives: np.ndarray,
    selected_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return coupled derivative blocks for selected full-grid indices.

    ``nodes`` and ``weighted_derivatives`` must be the ordered finite state for
    degree ``n`` and parameter ``alpha > -1``.  If supplied,
    ``selected_indices`` must be a nonempty, strictly increasing integer
    vector of unique full-grid indices.  ``None`` selects every node.  Invalid
    shapes, signs, indices, or nonfinite values raise ``ValueError``.

    Parameters
    ----------
    n : int
        Number of positive nodes; the full augmented size is ``n + 1``.
    alpha : float
        Generalized-Laguerre parameter, strictly greater than ``-1``.
    nodes : np.ndarray
        Full augmented node vector of shape ``(n + 1,)``.
    weighted_derivatives : np.ndarray
        Weighted degree-``n`` derivatives at positive nodes, shape ``(n,)``.
    selected_indices : np.ndarray or None, optional
        Increasing full-grid indices for the requested principal block.

    Returns
    -------
    np.ndarray
        Float array of shape ``(2, m, m)``.  Plane zero is the first-order
        block and plane one is the second-order block.
    """
    return derivative_blocks  # noqa: F821 - required model stub
```

### Step 3

03_scale_and_reduce_second_order

Goal
----
Map a full augmented second-derivative operator to the physical half-line.



The coordinate scale acts quadratically on the second derivative.  The

homogeneous endpoint condition is then imposed by removing the endpoint row

and column, while the positive physical nodes retain their original order.

```python
def scale_and_reduce_second_order(
    nodes: np.ndarray, second_order: np.ndarray, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive physical nodes and the reduced physical operator.

    ``nodes`` must be a finite increasing vector beginning at zero,
    ``second_order`` must be a matching finite square matrix, and ``beta``
    must be a finite positive scalar.  Invalid inputs raise ``ValueError``.

    Parameters
    ----------
    nodes : np.ndarray
        Unscaled augmented nodes, shape ``(N,)``.
    second_order : np.ndarray
        Unscaled second-derivative matrix, shape ``(N, N)``.
    beta : float
        Positive coordinate scale.

    Returns
    -------
    physical_nodes : np.ndarray
        Positive physical nodes, shape ``(N - 1,)``.
    reduced_second_order : np.ndarray
        Physical second-derivative operator after endpoint deletion, shape
        ``(N - 1, N - 1)``.
    """
    return physical_nodes, reduced_second_order  # noqa: F821 - model stub
```

### Step 4

04_sample_woods_saxon_profile

Goal
----
Sample the positive Woods--Saxon weight on the reduced half-line grid.



The generalized Sturm--Liouville pencil uses the decaying profile

`$q(x) = 1 / (1 + exp((x - radius) / diffuseness))$`.  A stable logistic

evaluation is required because large Laguerre nodes can make the raw

exponential overflow even though the mathematical profile remains finite.

```python
def sample_woods_saxon_profile(
    physical_nodes: np.ndarray, radius: float, diffuseness: float
) -> np.ndarray:
    """Return Woods--Saxon samples at positive physical nodes.

    ``physical_nodes`` must be a nonempty, finite, strictly increasing vector
    of positive values.  ``radius`` and ``diffuseness`` must be finite positive
    scalars.  Invalid inputs raise ``ValueError``.

    Parameters
    ----------
    physical_nodes : np.ndarray
        Positive physical nodes, shape ``(m,)``.
    radius : float
        Positive profile radius.
    diffuseness : float
        Positive surface thickness.

    Returns
    -------
    np.ndarray
        Finite profile values in ``(0, 1)``, shape ``(m,)``.
    """
    return potential  # noqa: F821 - model stub
```

### Step 5

05_assemble_generalized_pencil

Goal
----
Assemble the reduced generalized Sturm--Liouville matrix pencil.



For the equation ``-y'' + y = lambda q(x) y``, endpoint deletion has already

been applied to the second-derivative matrix.  The left pencil matrix is

therefore `$A = -D2 + I$` and the right matrix is the diagonal sampling of the

strictly positive profile.

```python
def assemble_generalized_pencil(
    reduced_second_order: np.ndarray, potential: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the finite-dimensional pencil ``(A, Q)``.

    ``reduced_second_order`` must be a nonempty finite square matrix and
    ``potential`` a matching finite vector with strictly positive entries.
    Invalid inputs raise ``ValueError``.

    Parameters
    ----------
    reduced_second_order : np.ndarray
        Physical second-derivative matrix, shape ``(m, m)``.
    potential : np.ndarray
        Positive profile samples, shape ``(m,)``.

    Returns
    -------
    a_matrix : np.ndarray
        Left pencil matrix ``-D2 + I``, shape ``(m, m)``.
    q_matrix : np.ndarray
        Diagonal right pencil matrix, shape ``(m, m)``.
    """
    return a_matrix, q_matrix  # noqa: F821 - model stub
```

### Step 6

06_compute_finite_spectral_magnitudes

Goal
----
Extract and order the finite spectrum of a generalized matrix pencil.



The homogeneous generalized-eigenvalue representation `$alpha / beta$` is

used so infinite modes can be identified before division.  This matters for

singular right-hand matrices and avoids treating overflowed quotients as

ordinary finite eigenvalues.

```python
def compute_finite_spectral_magnitudes(
    a_matrix: np.ndarray, q_matrix: np.ndarray
) -> np.ndarray:
    """Return sorted magnitudes of the finite generalized eigenvalues.

    Both arguments must be matching, nonempty, finite square matrices.  Modes
    whose homogeneous denominator is exactly zero are excluded.  A
    ``ValueError`` is raised if no finite eigenvalue remains.

    Parameters
    ----------
    a_matrix : np.ndarray
        Left matrix of ``A y = lambda Q y``, shape ``(m, m)``.
    q_matrix : np.ndarray
        Right matrix of ``A y = lambda Q y``, shape ``(m, m)``.

    Returns
    -------
    np.ndarray
        Nondecreasing finite eigenvalue magnitudes.
    """
    return magnitudes  # noqa: F821 - model stub
```

### Step 7

07_select_one_based_mode

Goal
----
Select a one-based mode from an already ordered finite spectrum.



The scientific convention numbers modes beginning at one.  This boundary

stage enforces that convention and rejects unordered or contaminated spectra

instead of silently selecting from malformed upstream data.

```python
def select_one_based_mode(magnitudes: np.ndarray, mode_index: int) -> float:
    """Return the requested one-based entry of a sorted magnitude vector.

    ``magnitudes`` must be nonempty, finite, nonnegative, and nondecreasing.
    ``mode_index`` must be a non-boolean integer in ``[1, len(magnitudes)]``.
    Invalid inputs raise ``ValueError``.

    Parameters
    ----------
    magnitudes : np.ndarray
        Ordered finite mode magnitudes.
    mode_index : int
        One-based mode number.

    Returns
    -------
    float
        The selected mode magnitude.
    """
    return mode_magnitude  # noqa: F821 - model stub
```

### Step 8

08_compute_laguerre_woods_saxon_mode

Goal
----
Run the complete half-line collocation pipeline and report one mode magnitude.



This final stage composes the simultaneous GLR root/derivative state, the direct

first/second collocation block, physical scaling, stable Woods--Saxon sampling,

generalized-pencil assembly, homogeneous finite-spectrum extraction, and

one-based mode selection.  The physical task uses the classical Laguerre case

`$alpha = 0$` while retaining the full-grid degree convention.

```python
def compute_laguerre_woods_saxon_mode(
    n: int,
    beta: float,
    radius: float,
    diffuseness: float,
    mode_index: int,
) -> float:
    """Return one magnitude-ordered generalized eigenvalue.

    ``n`` and ``mode_index`` must be non-boolean integers with ``n >= 1`` and
    ``1 <= mode_index <= n``.  ``beta``, ``radius`` and ``diffuseness`` must be
    finite positive scalars.  Every validation rule of the earlier stages also
    applies.  A ``ValueError`` is raised if fewer than ``mode_index`` finite
    eigenvalues are available or the result is nonfinite.

    Parameters
    ----------
    n : int
        Number of positive roots and reduced pencil dimension.
    beta : float
        Positive coordinate scale.
    radius : float
        Positive potential radius.
    diffuseness : float
        Positive potential surface thickness.
    mode_index : int
        One-based index after sorting finite eigenvalue magnitudes.

    Returns
    -------
    float
        The selected finite eigenvalue magnitude.
    """
    return mode_magnitude  # noqa: F821 - required model stub
```
