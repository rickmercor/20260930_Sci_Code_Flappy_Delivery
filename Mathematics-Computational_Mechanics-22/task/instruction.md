# Normalized residual for a condensed contact solve

## Background

# Scientific background

Mixed finite-element discretizations of nearly incompressible solids introduce local pressure variables in addition to displacement variables. Eliminating the pressure block produces a displacement-space core that contains the material stiffness and a volumetric Schur-complement contribution. Contact adds a positive-semidefinite tangent assembled from a comparatively small collection of interaction rows, while geometric or linearization terms may leave a smaller nonsymmetric correction in the complete Newton operator.

Repeatedly solving such systems is expensive when the elastic core is ill-conditioned or the active contact set changes. A reduced interaction-space preconditioner addresses this cost by retaining only part of the contact tangent. Its accuracy depends on the metric used to compare interaction directions and on the criterion used to stop enlarging the retained space; those choices must be recovered from the governing formulation rather than inferred from row magnitudes.

The reduced correction can be applied through a low-rank inverse identity after solves with the condensed core and a small dense system in interaction space. In the symmetric positive-definite regime, neglected interaction response gives spectral and inverse-action certificates. Inside flexible right-preconditioned GMRES, each right action produces a distinct preconditioned Arnoldi column, so the least-squares basis and the Krylov basis cannot be conflated. Solver acceptance must still use a newly computed residual of the complete Newton equation.

## Problem

Consider a nondimensional pressure-condensed finite-element Newton system with five displacement degrees of freedom, four row-oriented contact modes, a local pressure block, and a nonsymmetric correction. The fixed data are

$$
K_0=
\begin{bmatrix}
40&-1&0.2&0&0\\
-1&12&-0.3&0.1&0\\
0.2&-0.3&2.5&-0.2&0.1\\
0&0.1&-0.2&1.2&-0.1\\
0&0&0.1&-0.1&0.8
\end{bmatrix},
\quad
B=
\begin{bmatrix}
1&-1&0&0&0.5\\
0&0.5&-1&1&0
\end{bmatrix},
\quad
D=\operatorname{diag}(2,1.5),
$$

$$
U=
\begin{bmatrix}
8&0&0&0&0\\
0&4&0&0&0\\
0&0&2&0.2&0\\
0&0&0.1&1.5&1.2
\end{bmatrix},
\quad
N=
\begin{bmatrix}
0.08&0.02&0&0&0\\
-0.01&-0.04&0.03&0&0\\
0&-0.02&0.05&0.01&0\\
0&0&-0.03&-0.02&0.04\\
0.01&0&0&-0.02&0.03
\end{bmatrix},
\quad
b=
\begin{bmatrix}
1\\-0.5\\0.75\\0.2\\-1.1
\end{bmatrix}.
$$

Using the primary source as the governing specification, evaluate the first accepted linear-solver update in float64 arithmetic from the zero displacement iterate, with condition budget $\kappa_\star=2.5$. Interpret the displayed blocks, the admissibility requirement, and the solver's acceptance rule under that formulation; recover every required construction and convention from the source. Compute the single dimensionless target

$$
\rho=\frac{\lVert b-Ax_1\rVert_2}{\lVert b\rVert_2}.
$$

Give a compact, checkable justification of the source-specific reduced space and the accepted update. Include only the quantitative evidence needed to establish the selection, certify the reduced construction, and verify the final complete-equation residual.

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
Keep <reasoning> short (a few hundred words). Do not paste the input matrices,
implementation traces, or candidate tables.

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

01_condense_volumetric_core

Goal
----
Factor and condense the local pressure contribution without losing its metric.

```python
import numpy as np


def condense_volumetric_core(
    material_stiffness: np.ndarray,
    pressure_coupling: np.ndarray,
    pressure_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return the compliance-scaled rows, their Gram matrix, and the core.

    Write $L$ for the lower-triangular Cholesky factor of the local block, so
    that the block equals $LL^\top$. The scaled rows are the unique solution
    $V$ of $LV=B$, the induced Gram matrix is $V^\top V$, and the condensed
    core is the material stiffness plus that Gram matrix. Symmetrize the Gram
    matrix and the core by averaging with their transposes before returning
    them.

    Raises ValueError unless every input is finite; material_stiffness is a
    nonempty symmetric square matrix of shape (n, n); pressure_coupling has
    shape (m, n) with m at least zero; pressure_block is a symmetric positive-
    definite matrix of shape (m, m); and the condensed core is symmetric
    positive definite. Positive definiteness is decided by whether a Cholesky
    factorization succeeds. With m equal to zero the scaled rows have shape
    (0, n), the Gram matrix is exactly zero, and the core is the symmetrized
    material stiffness.

    Parameters
    ----------
    material_stiffness : np.ndarray
        Symmetric displacement matrix of shape (n, n).
    pressure_coupling : np.ndarray
        Signed row-oriented local-field coupling of shape (m, n).
    pressure_block : np.ndarray
        Symmetric positive-definite local block of shape (m, m).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The scaled pressure rows of shape (m, n), their displacement Gram
        matrix of shape (n, n), and the condensed SPD core of shape (n, n).
    """
    return result  # noqa: F821 - required model stub
```

### Step 2

02_assemble_contact_operators

Goal
----
Keep the interaction energy, reference operator, and full operator distinct.

```python
import numpy as np


def assemble_contact_operators(
    condensed_core: np.ndarray,
    interaction_rows: np.ndarray,
    correction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Return interaction energy, symmetric reference, full and symmetric part.

    With core $M$, ordered rows $U$, and correction $N$, the returned objects
    are the interaction energy $U^\top U$, the symmetric reference
    $H=M+U^\top U$, the complete operator $A=H+N$, and the symmetric part
    $\tfrac{1}{2}(A+A^\top)=H+\tfrac{1}{2}(N+N^\top)$, in that order. The
    interaction energy and the reference are symmetrized by averaging with
    their transposes; the complete operator is returned unsymmetrized.

    Raises ValueError unless all inputs are finite; condensed_core is a
    nonempty symmetric positive-definite square matrix; interaction_rows has
    row-oriented shape (m, n), including (0, n) and m greater than n; and
    correction has shape (n, n). The correction need not be symmetric.

    Parameters
    ----------
    condensed_core : np.ndarray
        Symmetric positive-definite displacement core of shape (n, n).
    interaction_rows : np.ndarray
        Ordered row-oriented interaction factor of shape (m, n).
    correction : np.ndarray
        Complete-system correction of shape (n, n).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Interaction energy, symmetric reference operator, complete operator,
        and the symmetric part of the complete operator, each of shape (n, n).
    """
    return result
```

### Step 3

03_build_response_gramian

Goal
----
Rank the interaction rows by the response they drive through the core.

```python
import numpy as np


def build_response_gramian(
    condensed_core: np.ndarray,
    interaction_rows: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    r"""Return the response operator, its levels, the retained projector, ratio.

    The response operator $G$ is the symmetric interaction-space operator whose
    quadratic form $c^\top Gc$ is the core energy $d^\top M^{-1}d$ of the
    displacement load $d=U^\top c$ that the row combination $c$ applies. Its
    levels are the eigenvalues of $G$ in nonincreasing order, clipped at zero.
    The retained projector is the orthogonal projector $\Pi$ of rank $r$ on
    interaction space maximizing $\operatorname{tr}(\Pi G)$. The cutoff ratio
    is the first omitted level divided by the last retained level, and is
    exactly 0.0 when nothing is omitted or nothing is retained.

    Raises ValueError unless condensed_core is finite, symmetric positive
    definite, and nonempty; interaction_rows is finite with shape (m, n),
    including (0, n) and m greater than n; rank is a Python or NumPy integer,
    not a bool, lying in [0, m]; the response is positive semidefinite to
    tolerance; the maximizing projector is unique; and the retained rank is
    supported by the numerically positive part of the response, so a rank
    reaching into its null space is rejected.

    Parameters
    ----------
    condensed_core : np.ndarray
        Symmetric positive-definite core of shape (n, n).
    interaction_rows : np.ndarray
        Ordered row-oriented interaction factor of shape (m, n).
    rank : int
        Retained response dimension.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, float]
        Response operator of shape (m, m), nonincreasing levels of shape (m,),
        retained orthogonal projector of shape (m, m), and the cutoff ratio.
    """
    return result
```

### Step 4

04_select_response_subspace

Goal
----
Certify the spectrum a retained subspace produces and what it leaves behind.

```python
import numpy as np


def certify_preconditioned_spectrum(
    condensed_core: np.ndarray,
    interaction_rows: np.ndarray,
    response_levels: np.ndarray,
    retained_projector: np.ndarray | None,
    retained_factor: np.ndarray | None = None,
    reference_core: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, np.ndarray]:
    r"""Return measured and predicted spectra, two certificates, and two ratios.

    Write $H=M+U^\top U$ for the fully contacted symmetric operator and
    $P=M+U^\top\Pi U$ for the operator retaining only the selected part of the
    contact energy. The measured spectrum is the ascending spectrum of the
    pencil $(H,P)$. The predicted spectrum is what the deflation theorem gives
    for that pencil, ascending and of length n, from the response that $\Pi$
    actually leaves outside the retained space. The certificate is the sharpest
    constant $c$ for which the inverse action of $P$ reproduces the inverse
    action of $H$ on every right-hand side to within $c$, the discrepancy
    measured in the core energy norm and the right-hand side in its dual norm.
    The worst omitted interaction is the largest value that the squared
    Euclidean norm of the contact action $\Pi$ discards attains relative to the
    core energy of the displacement producing it, and the posterior bound is
    the ceiling on the measured conditioning that this single number
    guarantees for any admissible projector. The final value is the ratio of
    the largest measured eigenvalue to the smallest.

    Here condensed_core is the fixed SPD core $\widetilde M$ used for response
    selection and inverse application. If reference_core is supplied, it is
    the SPD core $M$ in the unchanged reference operator. Let $c_1,c_2$ be the
    extremal generalized eigenvalues of $(M,\widetilde M)$, let
    $E=U^\top(I-\Pi)U$, and let $\delta$ be the largest generalized eigenvalue
    of $(E,\widetilde M)$. The final array is

    ``[c1, c2, delta, max(c2 + delta, 1) / min(c1, 1), kappa_true]``,

    where kappa_true is the measured condition number of
    $(M+U^\top U,\widetilde M+U^\top\Pi U)$. With no reference core, take
    $M=\widetilde M$; the robust ceiling then reduces to the exact posterior
    ceiling.

    Supply exactly one retained-space representation. If retained_factor is
    None, retained_projector must be a finite symmetric idempotent matrix of
    shape (m, m). Otherwise retained_projector must be None and retained_factor
    must be a finite matrix of shape (m, k), including k=0 and k greater than
    m. Its orthogonal projector is formed from the left singular vectors whose
    singular values exceed max(m, k, 1) times machine epsilon times the largest
    singular value; an all-zero factor represents the empty space. Raises
    ValueError unless condensed_core is finite symmetric positive definite;
    reference_core, when supplied, is finite symmetric positive definite with
    shape (n, n);
    interaction_rows has shape (m, n), including (0, n) and m greater than n;
    response_levels is a finite nonnegative nonincreasing vector of shape (m,)
    reproducing the response eigenvalues to rtol 1e-9 and atol 1e-11; the
    represented subspace leaves the response invariant to rtol 1e-8 and atol
    1e-10; P is positive definite; the measured and predicted spectra agree to
    rtol 1e-8 and atol 1e-10; and the measured conditioning does not exceed the
    posterior bound by more than 1e-8; and the measured true-pair conditioning
    does not exceed its Appendix-C ceiling by more than 1e-8.

    Parameters
    ----------
    condensed_core : np.ndarray
        Symmetric positive-definite core of shape (n, n).
    interaction_rows : np.ndarray
        Ordered interaction factor of shape (m, n).
    response_levels : np.ndarray
        Nonincreasing response levels of shape (m,).
    retained_projector : np.ndarray or None
        Response-invariant orthogonal projector of shape (m, m), or None when
        retained_factor supplies the representation.
    retained_factor : np.ndarray or None
        Optional rectangular spanning factor of shape (m, k).
    reference_core : np.ndarray or None
        Optional unchanged SPD reference core M of shape (n, n).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float, float, float, float, np.ndarray]
        Measured ascending spectrum of shape (n,), predicted ascending spectrum
        of shape (n,), the omitted-response certificate, the condition number
        of the measured spectrum, the worst omitted interaction, and the
        posterior condition bound, followed by the five-entry approximate-core
        certificate described above.
    """
    return result
```

### Step 5

05_build_reduced_interaction_state

Goal
----
Build the compact retained-coordinate state used by the inverse action.

```python
import numpy as np


def build_reduced_interaction_state(
    condensed_core: np.ndarray,
    interaction_rows: np.ndarray,
    retained_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Return the reduced preconditioner and its compact Woodbury state.

    Let the columns of ``retained_basis`` be the supplied orthonormal
    interaction-space basis. Return the symmetric reduced preconditioner,
    the retained row matrix, the corresponding core-response block, and the
    reduced matrix required by the source's retained-basis inverse action.
    The basis may have zero columns; in that case the last three returns have
    shapes (0, n), (n, 0), and (0, 0), respectively.

    Raises ValueError unless condensed_core is finite symmetric positive
    definite with shape (n, n); interaction_rows is finite with shape (m, n),
    including (0, n) and m greater than n; retained_basis is finite with shape
    (m, r), 0 <= r <= m, and has orthonormal columns to rtol 1e-10 and atol
    1e-12; and both the reduced matrix and reduced preconditioner are positive
    definite when nonempty.

    Parameters
    ----------
    condensed_core : np.ndarray
        Symmetric positive-definite displacement core of shape (n, n).
    interaction_rows : np.ndarray
        Ordered interaction factor of shape (m, n).
    retained_basis : np.ndarray
        Supplied orthonormal interaction-space basis of shape (m, r).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Reduced preconditioner P of shape (n, n), retained rows R of shape
        (r, n), core responses Z of shape (n, r), and reduced matrix S of
        shape (r, r).
    """
    return result
```

### Step 6

06_apply_reduced_inverse

Goal
----
Apply a compact retained-coordinate inverse and its true adjoint.

```python
import numpy as np


def apply_reduced_inverse(
    base_operator: np.ndarray,
    retained_rows: np.ndarray,
    core_responses: np.ndarray,
    reduced_matrix: np.ndarray,
    right_hand_sides: np.ndarray,
    transposed: bool,
) -> np.ndarray:
    r"""Apply the compact Woodbury action, or its adjoint, to a vector/block.

    The three compact arrays must constitute one valid retained state for the
    supplied base operator: the response block solves the retained transpose
    loads through the base, and the reduced matrix is the identity plus their
    retained-coordinate coupling.  Apply Algorithm 1 of the source when
    ``transposed`` is false and the adjoint of that complete action otherwise.

    Raises ValueError unless all arrays are finite; base_operator is a nonempty
    nonsingular square matrix of shape (n, n); retained_rows, core_responses,
    and reduced_matrix have shapes (r, n), (n, r), and (r, r), including r=0;
    the two compact-state identities hold to rtol 1e-10 and atol 1e-12;
    reduced_matrix is nonsingular; right_hand_sides has shape (n,) or (n, k)
    with k at least one; and transposed is a Python or NumPy bool.  Preserve
    the exact vector-or-block shape of the supplied right side.

    Parameters
    ----------
    base_operator : np.ndarray
        Nonsingular base operator M of shape (n, n).
    retained_rows : np.ndarray
        Retained row matrix R of shape (r, n).
    core_responses : np.ndarray
        Core-response block Z of shape (n, r).
    reduced_matrix : np.ndarray
        Reduced matrix S of shape (r, r).
    right_hand_sides : np.ndarray
        One vector (n,) or a column block (n, k).
    transposed : bool
        Select the adjoint of the complete inverse action when True.

    Returns
    -------
    np.ndarray
        Compact inverse action with the same shape as right_hand_sides.
    """
    return result
```

### Step 7

07_certify_residual_bounds

Goal
----
Bound and evaluate the first right-preconditioned residual reduction.

```python
import numpy as np


def certify_residual_bounds(
    full_operator: np.ndarray,
    right_inverse: np.ndarray,
    right_hand_side: np.ndarray,
) -> tuple[float, float, float, float, float]:
    r"""Return the exact one-step reduction, its coefficient, and the estimate.

    Let $\tilde{A}=AW$ be the right-preconditioned operator. The exact
    reduction is $\min_{\alpha}\lVert b-\alpha\tilde{A}b\rVert_2/\lVert
    b\rVert_2$ and the coefficient is the minimizing $\alpha$; when
    $\tilde{A}b$ vanishes to working precision the coefficient is exactly 0.0
    and the reduction exactly 1.0, the threshold being
    ``np.finfo(float).eps * max(1.0, inflation) * np.linalg.norm(b)``. The
    coercivity is the least eigenvalue of the symmetric part of $\tilde{A}$
    and the inflation is the operator 2-norm of $\tilde{A}$; the estimate is
    the sharpest bound on the reduction that those two scalars supply
    simultaneously for every right-hand side.

    Raises ValueError unless every input is finite; full_operator and
    right_inverse are square with the same nonempty shape (n, n);
    right_hand_side has shape (n,) and nonzero norm; the inflation is
    positive; and the exact reduction does not exceed the estimate by more
    than 1e-9.

    Parameters
    ----------
    full_operator : np.ndarray
        Complete, potentially nonsymmetric operator A of shape (n, n).
    right_inverse : np.ndarray
        Right-preconditioning action W of shape (n, n).
    right_hand_side : np.ndarray
        Nonzero right-hand side b of shape (n,).

    Returns
    -------
    tuple[float, float, float, float, float]
        Exact one-step reduction, optimal coefficient, field-of-values
        estimate, coercivity, and inflation.
    """
    return result
```

### Step 8

08_compute_directional_residual

Goal
----
Run flexible right preconditioning from compact retained-coordinate states.

```python
import numpy as np


def compute_directional_residual(
    full_operator: np.ndarray,
    right_hand_side: np.ndarray,
    base_operator: np.ndarray,
    retained_row_sequence: tuple[np.ndarray, ...],
    core_response_sequence: tuple[np.ndarray, ...],
    reduced_matrix_sequence: tuple[np.ndarray, ...],
    max_iterations: int,
    cycle_lengths: np.ndarray | None = None,
    initial_guess: np.ndarray | None = None,
    lower_retained_rows: np.ndarray | None = None,
    lower_core_responses: np.ndarray | None = None,
    lower_reduced_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    r"""Run restarted FGMRES from ordered one- or two-level compact states.

    At column j, apply the source's compact retained-basis inverse action for
    the j-th row/response/reduced-matrix triple to the current Arnoldi vector.
    If the three lower-level arrays are supplied, first apply their fixed
    Woodbury action over base_operator and use that effective core beneath
    every top-level correction; otherwise solve base_operator directly.
    Store that preconditioned direction, use two passes of orthogonalization on
    its complete-operator image. Within each cycle, accept the
    least-Euclidean-norm coordinates minimizing the fresh residual at that
    cycle's start over the directions stored in that cycle. At a restart,
    discard the Arnoldi basis, keep the accumulated iterate, and normalize a
    freshly evaluated complete residual. Close the entire recurrence when a
    twice-orthogonalized image vanishes to working precision.

    cycle_lengths is a one-dimensional positive-integer array whose sum is
    max_iterations; None means one unrestarted cycle. initial_guess is zero
    when omitted. The residual history has length max_iterations + 1, starts at
    the fresh normalized residual of that guess, and holds
    the freshly evaluated normalized residual after every used column, with the
    terminal value repeated through unused entries.  The returned coordinate
    vector is zero-padded to max_iterations and concatenates the accepted local
    least-squares coordinates cycle by cycle.

    Raises ValueError unless full_operator and base_operator are finite
    nonempty square matrices of the same shape; base_operator is nonsingular;
    right_hand_side is a finite nonzero vector of shape (n,); max_iterations is
    a Python or NumPy integer, not a bool, in [1, n]; each supplied sequence is
    a tuple or list of exactly max_iterations arrays; and at iteration j the
    three arrays have shapes (r_j, n), (n, r_j), and (r_j, r_j), including
    r_j=0, are finite, satisfy the compact-state identities to rtol 1e-10 and
    atol 1e-12, and define a nonsingular reduced matrix.  A numerically zero
    compact inverse direction is also rejected. The three lower-level arrays
    must be supplied together or omitted together; when supplied they have
    shapes (p, n), (n, p), and (p, p), satisfy their two compact identities,
    and define a nonsingular reduced matrix. Every top-level response is
    checked against the resulting effective core. cycle_lengths must have the
    partition stated above, and initial_guess must be finite with shape (n,).

    Parameters
    ----------
    full_operator : np.ndarray
        Complete, potentially nonsymmetric operator A of shape (n, n).
    right_hand_side : np.ndarray
        Nonzero right-hand side of shape (n,).
    base_operator : np.ndarray
        Fixed nonsingular core M of shape (n, n).
    retained_row_sequence : tuple[np.ndarray, ...]
        Ordered retained row matrices R_j of shapes (r_j, n).
    core_response_sequence : tuple[np.ndarray, ...]
        Ordered core-response blocks Z_j of shapes (n, r_j).
    reduced_matrix_sequence : tuple[np.ndarray, ...]
        Ordered reduced matrices S_j of shapes (r_j, r_j).
    max_iterations : int
        Requested number of flexible Arnoldi columns.
    cycle_lengths : np.ndarray or None
        Optional positive-integer restart partition summing to max_iterations.
    initial_guess : np.ndarray or None
        Optional initial iterate of shape (n,); defaults to zero.
    lower_retained_rows, lower_core_responses, lower_reduced_matrix : np.ndarray or None
        Optional fixed lower-level compact Woodbury state.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, int]
        Final iterate of shape (n,), normalized residual history of shape
        (k + 1,), zero-padded final coordinates of shape (k,), and the number
        of iterations taken.
    """
    return result
```

### Step 9

09_compute_contact_residual

Goal
----
Evaluate the certified condensed-contact pipeline through adaptive FGMRES.

```python
import numpy as np


def compute_contact_residual(
    material_stiffness: np.ndarray,
    pressure_coupling: np.ndarray,
    pressure_block: np.ndarray,
    interaction_rows: np.ndarray,
    correction: np.ndarray,
    right_hand_side: np.ndarray,
    condition_budget: float | np.ndarray,
    pressure_condition_budget: float | None = None,
    cycle_lengths: np.ndarray | None = None,
    initial_guess: np.ndarray | None = None,
) -> float:
    r"""Chain steps 1--8 and return a nested restarted-FGMRES full residual.

    An implementation must call condense_volumetric_core,
    assemble_contact_operators, build_response_gramian,
    certify_preconditioned_spectrum, build_reduced_interaction_state,
    apply_reduced_inverse, certify_residual_bounds, and
    compute_directional_residual. Interpret a scalar condition_budget as a
    one-entry schedule and a one-dimensional array as an iteration-indexed
    schedule. For each entry independently, determine the smallest retained
    response rank whose Appendix-C robust condition ceiling is no larger than
    that entry, where a ceiling counts as no larger when it does not exceed the
    entry by more than 1e-12 times max(1, |entry|); rank zero is admissible and
    all numerically positive response modes must be retained if no smaller rank
    meets it. If
    pressure_condition_budget is supplied, first select the smallest pressure
    response rank whose exact posterior ceiling over material_stiffness meets
    that scalar budget under the same comparison tolerance, build its fixed
    compact state, and use the resulting
    pressure-surrogate core for every contact selection while retaining the
    exact condensed core as the Appendix-C reference. If the pressure budget is
    omitted, use the exact condensed core directly. For every selected contact
    projector recover an orthonormal basis of its dominant response space,
    certify that factor representation, build its compact state, and apply
    that state to the identity once forward and once as the adjoint. Confirm
    the adjoint action and preserve the compact states in schedule order.
    Certify every scheduled action with certify_residual_bounds. The first
    accepted history entry must also equal its closed-form certified reduction.
    Pass the ordered compact states, not pre-expanded dense inverses, to a
    flexible cycle whose requested length is the schedule length, together
    with the optional lower state, restart partition, and initial guess, and
    return its terminal history entry.

    Raises ValueError unless condition_budget is either a finite real scalar,
    not a bool, or a nonempty finite one-dimensional real array, not Boolean,
    with every entry at least one and length no larger than the displacement
    dimension; pressure_condition_budget is None or a finite real scalar, not
    a bool, at least one; under any preceding contract; or unless the
    scaled pressure rows reproduce the volumetric Gram matrix, the returned
    symmetric part matches the complete operator, the retained projector commutes with
    the response, the cutoff ratio lies in the unit interval, the spectral
    prediction agrees, every recovered basis spans the selected projector,
    every approximate-core certificate satisfies its five defining bounds,
    each compact state satisfies its two defining identities, each reduced
    preconditioner equals its own reassembly, every adjoint action is the
    transpose of its forward action, the first certified exact reduction matches the
    first accepted residual, and the terminal residual agrees with a separate
    fresh evaluation to rtol 1e-11 and atol 1e-13.

    Parameters
    ----------
    material_stiffness, pressure_coupling, pressure_block : np.ndarray
        Raw mixed finite-element core data.
    interaction_rows, correction : np.ndarray
        Ordered interaction factor and complete-operator correction.
    right_hand_side : np.ndarray
        Nonzero displacement-space right-hand side.
    condition_budget : float or np.ndarray
        One robust contact condition ceiling or a per-iteration schedule.
    pressure_condition_budget : float or None
        Optional exact pressure-level posterior condition ceiling.
    cycle_lengths : np.ndarray or None
        Optional positive restart partition summing to the schedule length.
    initial_guess : np.ndarray or None
        Optional finite displacement-space initial iterate.

    Returns
    -------
    float
        Finite dimensionless terminal normalized full residual.
    """
    return result
```
