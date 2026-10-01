# Squared Newton decrement after two warm-started progressive projections

## Background

# Scientific background

Implicit time integration for deformable and rigid-body simulation leads to nonlinear energy minimizations whose Newton systems can become indefinite, because individual pieces of the domain contribute negative curvature through their material or contact response. A Newton step is only usable when the assembled system is positive definite, so solvers modify the local contributions before assembly to force definiteness. Modifying every local contribution always works, but it discards curvature information that the global system did not actually need corrected.

That observation motivates on-demand alternatives, in which only a subset of the local contributions is modified and the subset grows until the assembled system factorizes. Choosing the subset cheaply is the difficulty, since recovering global eigenvalues after assembly is far too expensive; the assembled gradient turns out to be a usable proxy for which regions are responsible. Such schemes carry an adaptive tolerance that controls how aggressively the subset grows, and because that tolerance is tightened on failure and relaxed after a successful step, its value at a given iteration depends on the history of the solve rather than on that iteration alone. Very weak regularization, including large time steps and quasistatic problems, is a regime where modifying everything can still be preferable.

## Problem

Two consecutive Newton iterations of a finite-element time step are solved with an on-demand element-Hessian modification scheme; compute the squared Newton decrement of the second iteration in double-precision arithmetic with zero-based indexing throughout. The mesh has ten degrees of freedom and twelve ordered two-degree-of-freedom elements whose global index pairs are $I_0=(0,1),I_1=(1,2),\ldots,I_8=(8,9)$ in chain order, followed by $I_9=(0,6)$, $I_{10}=(3,8)$ and $I_{11}=(2,9)$; the regularizer is $M=0.4\,I_{10}$ and the first-iteration local Hessians are $H^{(1)}_e=\begin{bmatrix}p_e&q_e\\q_e&r_e\end{bmatrix}$ with

| $e$ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $p_e$ | $-3.5$ | $-1.2$ | $-4.5$ | $-0.8$ | $1.9$ | $-0.45$ | $1.6$ | $1.85$ | $-0.3$ | $1.6$ | $1.9$ | $-0.7$ |
| $q_e$ | $0.45$ | $0.7$ | $0.3$ | $0.55$ | $0.3$ | $0.45$ | $0.3$ | $0.45$ | $0.3$ | $0.55$ | $0.3$ | $0.45$ |
| $r_e$ | $1.4$ | $1.15$ | $1.9$ | $1.2$ | $1.9$ | $0.95$ | $1.65$ | $1.9$ | $0.95$ | $1.65$ | $1.9$ | $1.4$ |

At the second iteration the re-evaluated local Hessians are $H^{(2)}_e=H^{(1)}_e+s_eI_2$ with $s_1=2.6$, $s_2=2.2$ and $s_e=0$ otherwise, and the two assembled residuals are

$$g^{(1)}=(24,-12,6,-3,1.5,-0.75,0.375,-0.375,0.1875,-0.1875)^T,\qquad g^{(2)}=(36,-18,9,-5,2.25,-1.125,0.5625,-0.5625,0.28125,-0.28125)^T.$$

Assemble element contributions with real $0/1$ selectors $S_e\in\mathbb{R}^{2\times10}$, each row holding exactly one unit entry, and decide positive definiteness by Cholesky factorization. The scheme's conventions are deliberately not restated here and must match the established ones: how each element's priority is scored from the assembled residual, the sentinel value that defers the first finite selection threshold until a modification is first required, the formula that then sets it, the comparison used when a score equals the threshold, the contraction factor applied after a failed factorization and the relaxation factor applied after a completed step together with their standard default values, the cutoff below which every remaining element is taken instead, and the floor imposed on clamped local eigenvalues. The two Newton iterations form one continuous solve. Solve each accepted system $H_{\mathrm{acc}}\Delta x=-g$ and report the single finite scalar $\lambda^2=-\bigl(g^{(2)}\bigr)^T\Delta x^{(2)}$ from the second iteration.

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
Keep <reasoning> short (a few hundred words). Show only the scalars that
determine the final number.
You must also report, as a short list: the twelve first-iteration element
scores; the smallest eigenvalue of the unprojected first-iteration assembly;
the two published factors and their default values; the fallback cutoff and
what it is for; the threshold at each round of each iteration and which
elements each round admits; the accepted first-iteration threshold and the
value carried into the second; the first iteration's squared Newton decrement;
the smallest eigenvalue of the accepted second-iteration Hessian; and the
values the two rejected policies would give instead.
Do not paste the input matrices, the element Hessian table, or the assembled
Hessians.

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

01_build_selection_matrices

Goal
----
Build the local-to-global selectors used by element assembly.



For ordered element indices `$I_e = (i, j)$$, the 0/1 matrix$$S_e$`

extracts `$(x_i, x_j)$` from the global vector. Its transpose scatters local

Hessian contributions through ``S_e.T @ H_e @ S_e``.

```python
def build_selection_matrices(index_pairs: np.ndarray, n_dof: int) -> np.ndarray:
    """Construct one two-row 0/1 selection matrix per element.

    Raises ``ValueError`` unless every one of the following holds: ``n_dof`` is
    an integer of at least 2; ``index_pairs`` is an integer array of shape
    ``(n_elements, 2)`` with at least one element; every index lies in
    ``[0, n_dof)``; and the two indices of each element are distinct.

    Parameters
    ----------
    index_pairs : np.ndarray
        Integer array of shape ``(n_elements, 2)`` in local row order. The two
        entries of a row address different degrees of freedom, so an element
        whose two indices are equal is rejected rather than assembled.
    n_dof : int
        Number of global degrees of freedom.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_elements, 2, n_dof)`` whose entries are
        ``0.0`` and ``1.0``, with exactly one unit entry per row. The array is
        float-typed; a ``dtype=bool`` array is not the expected return value.
    """
    return result  # noqa: F821 - required model stub
```

### Step 2

02_compute_local_residual_scores

Goal
----
Compute the element priority scores.



Each score summarizes the assembled residual restricted to the two degrees of

freedom of one element, `$S_e g$`, as a single nonnegative number. The ordered

score vector sets the order in which elements are considered by the

factorization stage, and the summary chosen here fixes which elements tie

against a given threshold. Which summary of `$S_e g$` is meant is fixed by the

established convention and is not restated here.

```python
def compute_local_residual_scores(
    selection_matrices: np.ndarray, residual: np.ndarray
) -> np.ndarray:
    """Score every element from the restricted assembled residual.

    Raises ``ValueError`` unless every one of the following holds:
    ``selection_matrices`` is rank three with a nonempty leading axis and at
    least one row per element; ``residual`` is one-dimensional with length
    equal to the last axis of ``selection_matrices``; every entry of both
    arguments is finite; every entry of ``selection_matrices`` equals ``0`` or
    ``1``; and each of its rows sums to exactly one.

    Parameters
    ----------
    selection_matrices : np.ndarray
        Float array of shape ``(n_elements, 2, n_dof)`` with entries ``0.0``
        and ``1.0`` and exactly one unit entry per row. It is float-typed, so
        do not reject it for not being ``dtype=bool``.
    residual : np.ndarray
        Finite assembled residual of shape ``(n_dof,)``.

    Returns
    -------
    np.ndarray
        Element scores of shape ``(n_elements,)`` in input order.
    """
    return result  # noqa: F821 - required model stub
```

### Step 3

03_project_element_hessian

Goal
----
Apply the local spectral projection to one element Hessian.



For ``H_e = Q_e D_e Q_e.T``, replace each diagonal value by

`$Dhat_e[ii] = max(D_e[ii], epsilon)$` and form ``Hhat_e = Q_e Dhat_e Q_e.T``.

The positive floor differs from a zero-clamp positive-semidefinite projection.

```python
def project_element_hessian(
    hessian: np.ndarray, eigenvalue_floor: float = 1e-8
) -> np.ndarray:
    """Clamp every eigenvalue below ``eigenvalue_floor``.

    Raises ``ValueError`` unless every one of the following holds: ``hessian``
    is a nonempty square two-dimensional array; all its entries are finite;
    it equals its own transpose to within ``1e-12`` absolute, which is
    verified rather than assumed, so a nonsymmetric argument is rejected
    instead of being symmetrized or passed to the eigendecomposition; and
    ``eigenvalue_floor`` is a finite strictly positive scalar.

    Parameters
    ----------
    hessian : np.ndarray
        Finite square element Hessian. Symmetry is a checked requirement, not
        a caller guarantee.
    eigenvalue_floor : float
        Strictly positive eigenvalue floor.

    Returns
    -------
    np.ndarray
        Symmetric projected Hessian with the same shape.
    """
    return result  # noqa: F821 - required model stub
```

### Step 4

04_assemble_global_hessian

Goal
----
Assemble the global Newton Hessian from local contributions.



The finite-element relation is ``H = M + sum_e S_e.T @ H_e @ S_e``. This

unprojected assembly is the first matrix whose factorization is attempted, before any

local curvature is modified.

```python
def assemble_global_hessian(
    regularizer: np.ndarray,
    element_hessians: np.ndarray,
    selection_matrices: np.ndarray,
) -> np.ndarray:
    """Assemble ``M + sum(S_e.T @ H_e @ S_e)``.

    Raises ``ValueError`` unless every one of the following holds:
    ``regularizer`` is a nonempty square two-dimensional array;
    ``element_hessians`` and ``selection_matrices`` are both rank three with
    equal and nonempty leading axes; ``element_hessians`` is square along its
    last two axes with side equal to the middle axis of
    ``selection_matrices``; the last axis of ``selection_matrices`` equals the
    side of ``regularizer``; every entry of all three arguments is finite;
    ``regularizer`` equals its transpose to within ``1e-12`` absolute; and
    each element Hessian equals its own transpose to the same tolerance. The
    two symmetry conditions are verified rather than assumed.

    Parameters
    ----------
    regularizer : np.ndarray
        Finite symmetric matrix of shape ``(n_dof, n_dof)``.
    element_hessians : np.ndarray
        Symmetric local matrices of shape ``(E, m, m)``.
    selection_matrices : np.ndarray
        Scatter/extract matrices of shape ``(E, m, n_dof)``.

    Returns
    -------
    np.ndarray
        Symmetric global Hessian of shape ``(n_dof, n_dof)``.
    """
    return result  # noqa: F821 - required model stub
```

### Step 5

05_select_projection_mask

Goal
----
Select the next set of elements to project.



An element that has not yet been processed becomes eligible when its score

stands in the required relation to the current threshold. The boundary case, a

score exactly equal to the threshold, is decided by the established convention

rather than by an arbitrary choice, and that convention is not restated here.

When the caller signals the fallback regime, every unprocessed element is

selected regardless of score.

```python
def select_projection_mask(
    scores: np.ndarray,
    delta: float,
    projected_mask: np.ndarray,
    force_all: bool = False,
) -> np.ndarray:
    """Select unprocessed elements for the next projection round.

    Raises ``ValueError`` unless every one of the following holds: ``scores``
    is a nonempty one-dimensional array whose entries are all finite and
    nonnegative; ``projected_mask`` has ``dtype=bool`` and exactly the shape
    of ``scores``; ``delta`` is a scalar that is neither NaN nor negative
    (``inf`` is permitted); and ``force_all`` is a Boolean.

    Parameters
    ----------
    scores : np.ndarray
        Nonnegative element scores of shape ``(n_elements,)``.
    delta : float
        Current nonnegative selection threshold.
    projected_mask : np.ndarray
        Record of elements already processed, with ``dtype=bool`` and the
        shape of ``scores``.
    force_all : bool, optional
        When true, select every unprocessed element and ignore ``delta``.

    Returns
    -------
    np.ndarray
        Integer 0/1 mask of shape ``(n_elements,)``.
    """
    return result  # noqa: F821 - required model stub
```

### Step 6

06_find_accepted_hessian

Goal
----
Report the state at which the assembled system becomes factorizable.



The starting point is the unprojected assembly ``H = M + sum(S_e.T @ H_e @

S_e)``. If it already admits a Cholesky factorization, the state is reported

with nothing projected. Otherwise elements are admitted to the projected set

in successive rounds governed by the threshold, each admitted element

contributing ``S_e.T @ (Hhat_e - H_e) @ S_e`` exactly once and never twice,

until a round leaves the assembly factorizable.



How the threshold is initialized when none is yet in force, how it moves from

one round to the next, and the level at which the elements still remaining are

admitted unconditionally are all fixed by the established convention and are

not restated here.



The returned state carries both the threshold in force at acceptance and the

cumulative projected set, because a later iteration resumes from that state

rather than starting fresh.

```python
def find_accepted_state(
    regularizer: np.ndarray,
    residual: np.ndarray,
    element_hessians: np.ndarray,
    selection_matrices: np.ndarray,
    scores: np.ndarray,
    delta_init: float,
    eigenvalue_floor: float = 1e-8,
    fallback_threshold: float = 1e-12,
) -> np.ndarray:
    """Return the accepted threshold and cumulative projected set.

    Raises ``ValueError`` unless every one of the following holds:
    ``regularizer`` is square, finite, and equal to its transpose to within
    ``1e-12`` absolute; ``residual`` has shape ``(n_dof,)`` and is finite;
    ``element_hessians`` has shape ``(n_elements, 2, 2)``, is finite, and is
    symmetric along its last two axes to the same tolerance;
    ``selection_matrices`` has shape ``(n_elements, 2, n_dof)``, is finite,
    has every entry equal to ``0`` or ``1``, and has exactly one unit entry
    per row; ``scores`` has shape ``(n_elements,)`` and is finite and
    nonnegative; ``delta_init`` is a scalar that is strictly positive and not
    NaN (``inf`` is permitted); and ``eigenvalue_floor`` and
    ``fallback_threshold`` are both finite and strictly positive.
    ``ValueError`` is also raised if the loop exhausts every element and the
    fully projected assembly is still not positive definite.

    Parameters
    ----------
    regularizer : np.ndarray
        Finite symmetric matrix of shape ``(n_dof, n_dof)``.
    residual : np.ndarray
        Finite assembled residual of shape ``(n_dof,)``.
    element_hessians : np.ndarray
        Ordered symmetric matrices of shape ``(n_elements, 2, 2)``.
    selection_matrices : np.ndarray
        Ordered 0/1 selectors of shape ``(n_elements, 2, n_dof)``, float-typed
        with exactly one unit entry per row.
    scores : np.ndarray
        Ordered nonnegative element scores of shape ``(n_elements,)``.
    delta_init : float
        Positive threshold in force when the loop starts. Pass ``inf`` when no
        projection has been required yet, in which case the first finite
        threshold is derived from ``residual`` on entry to the loop.
    eigenvalue_floor : float, optional
        Positive local spectral floor.
    fallback_threshold : float, optional
        Positive convention constant governing when the round-based selection
        is abandoned in favour of admitting everything that remains.

    Returns
    -------
    np.ndarray
        Length ``n_elements + 1``. Entry 0 is the threshold in force when the
        factorization succeeded, and is returned unchanged as ``delta_init``
        when the unprojected assembly is already positive definite. Entries 1
        onward are the cumulative 0/1 projected indicator.
    """
    return result  # noqa: F821 - required model stub
```

### Step 7

07_solve_newton_step

Goal
----
Solve the accepted Newton system and preserve the descent convention.



Once Cholesky certifies `$H_acc$` as positive definite, the step solves

`$H_acc @ delta_x = -g$`. The sign is essential because the final squared

decrement is ``-g.T @ delta_x`` and a valid step satisfies ``g.T @ delta_x < 0``.

```python
def solve_newton_step(
    accepted_hessian: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    """Compute the Newton step after a Cholesky acceptance check.

    Raises ``ValueError`` unless every one of the following holds:
    ``accepted_hessian`` is a square two-dimensional array; ``residual`` is
    one-dimensional with length equal to its side; every entry of both is
    finite; ``accepted_hessian`` equals its own transpose to within ``1e-12``
    absolute; and ``accepted_hessian`` is positive definite. Positive
    definiteness is a checked requirement rather than a caller guarantee, so
    an argument whose Cholesky factorization fails is rejected rather than
    solved.

    Parameters
    ----------
    accepted_hessian : np.ndarray
        Finite square matrix of shape ``(n, n)``. Symmetry and positive
        definiteness are checked requirements, not caller guarantees.
    residual : np.ndarray
        Finite assembled residual of shape ``(n,)``.

    Returns
    -------
    np.ndarray
        Newton step of shape ``(n,)``.
    """
    return result  # noqa: F821 - required model stub
```

### Step 8

08_compute_progressive_decrement

Goal
----
Run the multi-iteration pipeline and report its final scalar.



Each iteration builds the selectors, scores the elements from that iteration's

residual, determines the accepted state starting from the threshold currently

in force, and solves the accepted system. The threshold is state that persists

across iterations rather than being restarted at each one; exactly what a

later iteration inherits from its predecessor's accepted threshold is fixed by

the established convention and is not restated here.



The returned scalar is the squared Newton decrement of the final iteration,

``lambda_squared = -g.T @ delta_x``.

```python
def compute_progressive_decrement(
    regularizer: np.ndarray,
    residuals: np.ndarray,
    index_pairs: np.ndarray,
    element_hessians: np.ndarray,
    eigenvalue_floor: float = 1e-8,
    fallback_threshold: float = 1e-12,
) -> float:
    """Compute the squared decrement of the final warm-started iteration.

    Raises ``ValueError`` unless every one of the following holds:
    ``regularizer`` is a nonempty square two-dimensional array that equals its
    transpose to within ``1e-12`` absolute; ``residuals`` has shape
    ``(n_iterations, n_dof)`` with at least one iteration; ``index_pairs`` is
    an integer array of shape ``(n_elements, 2)`` with at least one element,
    every index in ``[0, n_dof)``, and the two indices of each element
    distinct; ``element_hessians`` has shape
    ``(n_iterations, n_elements, 2, 2)`` with iteration and element counts
    matching the two preceding arguments, and is symmetric along its last two
    axes to the same tolerance; every numeric entry is finite; and
    ``eigenvalue_floor`` and ``fallback_threshold`` are both finite and
    strictly positive. ``ValueError`` is also raised if any iteration exhausts
    every element without reaching a positive-definite assembly, or if the
    resulting decrement is not finite. The three symmetry and index conditions
    are verified rather than assumed.

    Parameters
    ----------
    regularizer : np.ndarray
        Finite symmetric matrix of shape ``(n_dof, n_dof)``.
    residuals : np.ndarray
        Finite assembled residuals of shape ``(n_iterations, n_dof)``, in
        Newton-iteration order.
    index_pairs : np.ndarray
        Ordered zero-based element pairs of shape ``(n_elements, 2)``.
    element_hessians : np.ndarray
        Re-evaluated symmetric local Hessians of shape
        ``(n_iterations, n_elements, 2, 2)``.
    eigenvalue_floor : float, optional
        Positive local spectral floor.
    fallback_threshold : float, optional
        Positive convention constant governing when the round-based selection
        is abandoned in favour of admitting everything that remains.

    Returns
    -------
    float
        Finite squared Newton decrement of the last iteration.
    """
    return result  # noqa: F821 - required model stub
```
