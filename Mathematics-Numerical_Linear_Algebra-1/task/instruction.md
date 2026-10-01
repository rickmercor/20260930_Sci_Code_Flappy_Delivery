# Mathematics-Numerical_Linear_Algebra-1

## Background

Krylov solvers for the large nonsymmetric sparse systems produced by discretizing transport equations usually need a preconditioner, and sparse approximate inverses are attractive because they are applied through matrix-vector products and can be built column by column in parallel. Their weak point is the sparsity pattern: a fixed pattern either wastes nonzeros or misses the dominant entries of the inverse, so adaptive procedures grow the pattern of each column from information gathered while the column is being computed.

The task below builds such a preconditioner for a convection-dominated test matrix and measures how close it comes to the inverse.

## Problem

Sparse approximate inverse preconditioners minimize the Frobenius defect of a right inverse column by column, using small least-squares problems on selected supports. Evaluate the residual-driven refinement published in 2025 that obtains its a-priori supports from a truncated Cayley-Hamilton expansion and grows them using the authors' residual-admission threshold and per-pass cap, on the following conservative convection-diffusion experiment.

Use an $18 \times 18$ tensor grid of cells, with edges $x_i = (i/18)^{1.7}$ and $y_j = (j/18)^{1.3}$ for $i, j = 0, \dots, 18$, midpoint centres $X_i = (x_i + x_{i+1})/2$ and $Y_j = (y_j + y_{j+1})/2$, and zero-based unknown index $18j + i$. The PDE is $\nabla \cdot (\mathbf{v} u - D \nabla u) + \sigma u$, with periodic continuation in $y$ and homogeneous conditions on the TOTAL outward flux at the two $x$ boundaries, $J_{\mathrm{out}} = \rho\, u_{\mathrm{boundary}}$. The cell velocities are $v_x = 30 (1 + X_i) \cos(2 \pi Y_j)$ and $v_y = 60 \sin(2 \pi (X_i - 0.37))$; define the face and cell data by

$$D_x[j,i] = 0.08 \left[ 1 + 0.6 \sin^2(\pi x_i) \cos^2(2 \pi Y_j) \right], \quad i = 0, \dots, 18$$

$$D_y[j,i] = 0.12 \left[ 1 + 0.5 \cos^2(\pi X_i) \sin^2(2 \pi y_{j+1}) \right], \quad j = 0, \dots, 17$$

$$\sigma[j,i] = 0.1 + 0.05 \cos^2(\pi X_i) \sin^2(2 \pi Y_j)$$

$$\rho_{\mathrm{left}}[j] = 0.2 + 2 \sin^2(2 \pi Y_j), \quad \rho_{\mathrm{right}}[j] = 0.5 + 3 \cos^2(2 \pi Y_j)$$

At each interior face, use the exact constant flux $J = v_f u - D_f \, du/ds$ of the one-dimensional homogeneous transport equation between the adjacent cell centres, with their unknown values as endpoint data, the supplied face diffusivity, and the linear interpolant of the adjacent normal cell velocities at the face; the upper face of the last $y$ cell connects to the first by periodic continuation. At each $x$ boundary, use the adjacent cell velocity and boundary diffusivity on the half-cell segment and eliminate the boundary trace using the stated total-flux condition. Let $L$ map cell values to their integrated outward face-flux balances plus $\sigma$ times cell volume times cell value, and form $A = V^{-1/2} L V^{-1/2}$, where $V$ is the diagonal cell-volume matrix.

Build $M$ for this $A$ using residual tolerance $\epsilon = 0.1$, admission threshold $\delta = 0.25$, cap $c = 3$, and at most three nonempty admissions per column, including admissions that add no new coordinate; break equal residual magnitudes toward the smaller global row index and use the numerical nonzero support of floating-point $A^2$. Report $\lVert AM - I \rVert_F$ to ten significant figures. In the short reasoning, state the source's seed, admission, enlargement and early-stop rules; the fitted interior-flux and eliminated Robin coefficients; the weighted conservation identity implied by the integrated balance; $\mathrm{nnz}(A)$, $\mathrm{nnz}(M)$, the count reaching $\epsilon$, the count that completed all three admissions and is still above $\epsilon$, the largest residual and its one-based cell $(i+1, j+1)$, and the share of $M$ outside the seed supports; also give the norm when $\epsilon = 0.2$, when $\epsilon = 0.1$ with diagonal-only seeds, and the authors' reported norm and density ratio at $\epsilon = 0.1$ for their smallest test matrix.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal, not NaN, Inf, a fraction, a vector, or prose.
- Put only that one number between the tags, without units or extra lines.
Keep <reasoning> to a few hundred words, showing only the defining formulas and summary scalars. Do not paste matrices, coefficient vectors, or per-column trajectories.

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

01_assemble_upwind_convection_diffusion

Goal
----
Assemble a conservative exponentially fitted convection-diffusion operator on nonuniform cells with periodic and total-flux Robin boundaries.

```python
def assemble_upwind_convection_diffusion(velocity_x: "np.ndarray", velocity_y: "np.ndarray", *, fitted: dict = None) -> "np.ndarray":
    r"""Return the conservative fitted transport matrix of a cell grid.

    Assemble the conservative cell-centred operator
    $\nabla \cdot (\mathbf{v} u - D \nabla u) + \sigma u$ on a rectangular
    tensor grid, periodic in $y$ and with homogeneous total-flux Robin
    conditions at the $x$ ends. Return $A = V^{-1/2} L V^{-1/2}$, where $L$
    maps cell values to integrated outward flux balances plus
    $\sigma \, V \, u$ and $V$ is the diagonal cell volume. Unknown $(i, j)$,
    counted from zero, has index $j \cdot n_x + i$.

    ``velocity_x`` and ``velocity_y`` have shape $(n_y, n_x)$ and are
    cell-centre values. The required ``fitted`` dictionary contains:
      x_edges, y_edges: strictly increasing finite 1-D coordinate arrays
        of lengths $n_x + 1$ and $n_y + 1$, with $n_x, n_y \ge 1$;
      diffusion_x: positive finite $(n_y, n_x + 1)$ face diffusivities,
        including both $x$ boundary faces;
      diffusion_y: positive finite $(n_y, n_x)$ face diffusivities, entry
        $[j, i]$ belonging to the upper face of cell $(i, j)$, including the
        seam;
      reaction: nonnegative finite $(n_y, n_x)$ cell reaction coefficients;
      robin_left, robin_right: nonnegative length-$n_y$ arrays, allowing
        $+\infty$. An infinite value means zero Dirichlet trace. Zero means
        no flux.

    Cell centres are edge midpoints. Each face flux is the constant value
    $F = v_f u(s) - D_f \, du(s)/ds$ of the exact one-dimensional homogeneous
    transport equation on the segment joining the two adjacent centres, with
    their cell values as endpoint data. The segment crosses the $y$ seam by
    periodic continuation. Its speed $v_f$ is the linear interpolant of the
    adjacent normal velocity components at that face; $D_f$ is the supplied
    face diffusivity. Multiply $F$ by the face length and add its outward
    contribution to EACH adjoining cell. Distinct periodic faces must all be
    counted even if they join the same pair ($n_y = 2$); the two contributions
    cancel when both cells are the same ($n_y = 1$).

    At an $x$ boundary use the adjacent cell's velocity and the supplied
    boundary diffusivity on the half-cell segment from its centre to the
    boundary. Eliminate the boundary trace using outward TOTAL flux
    $F_{\mathrm{out}} = \rho \, u_{\mathrm{boundary}}$; $\rho = +\infty$
    prescribes $u_{\mathrm{boundary}} = 0$. This condition includes
    convection, so replacing it by a diffusive Robin condition defines a
    different matrix. Reaction is integrated exactly as a piecewise constant
    cell term. Both boundary faces count when $n_x = 1$.

    Use continuous zero-speed limits. Peclet numbers can have magnitude up to
    $10^{4}$ and can be as small as $10^{-14}$. Cell widths lie in
    $[10^{-6}, 10^{3}]$, diffusivities in $[10^{-100}, 10^{100}]$, and
    absolute velocities and reactions are at most $10^{100}$. Positive finite
    Robin values lie in $[10^{-250}, 10^{100}]$. All resulting matrix entries
    are representable finite floats. These numerical ranges are
    preconditions, not additional validation rules. Overflow or premature
    underflow in an intermediate exponential is not an acceptable result.
    Preserve each matrix entry to $10^{-10}$ relative to
    $\max(1, \text{that entry's absolute magnitude})$.

    Raise ValueError for a missing or nondictionary ``fitted`` value, missing
    required keys, wrong array dimensions or shapes, nonincreasing or
    nonfinite edges, nonfinite velocities, diffusion or reaction, nonpositive
    diffusion, negative reaction, or negative or NaN Robin coefficients.
    Other keys are ignored.

    Parameters
    ----------
    velocity_x : np.ndarray
        Float array of shape $(n_y, n_x)$: the $x$ velocity component at cell
        centres, indexed ``[j, i]``.
    velocity_y : np.ndarray
        Float array of shape $(n_y, n_x)$: the $y$ velocity component, same
        indexing.
    fitted : dict
        Required finite-volume data described above.

    Returns
    -------
    np.ndarray
        Dense float array of shape $(n_x n_y,\ n_x n_y)$.

    Raises
    ------
    ValueError
        If ``fitted`` is absent or is not a dictionary, if a required key is
        missing, or if any supplied array violates the shape, monotonicity,
        finiteness, positivity or sign conditions listed above.
    """
    return matrix
```

### Step 2

02_seed_power_pattern

Goal
----
Form the a-priori sparsity pattern of one column of the approximate inverse from the low powers of the matrix.

```python
def seed_power_pattern(matrix: "np.ndarray", column: int) -> "np.ndarray":
    r"""Return the a-priori sparsity pattern of one column of the approximate inverse.

    Write $A$ for ``matrix`` and $k$ for ``column``. The pattern of column $k$
    is the set of row positions at which column $k$ of the identity $I$,
    column $k$ of $A$, or column $k$ of the square $A^2$ is nonzero, that is
    $J_k^0 = \{\, i : I_{ik} \ne 0 \ \text{or}\ A_{ik} \ne 0 \ \text{or}\ (A^2)_{ik} \ne 0 \,\}$.
    The square is formed in floating point, so a position at which its
    products cancel exactly to zero does not belong to the pattern unless the
    identity or $A$ itself puts it there.

    Parameters
    ----------
    matrix : np.ndarray
        Float array of shape ``(n, n)``, ``n >= 1``, with finite entries.
    column : int
        Column index ``k`` with ``0 <= k < n``.

    Returns
    -------
    np.ndarray
        Sorted 1-D integer array of the distinct pattern positions.

    Raises
    ------
    ValueError
        If ``matrix`` is not a nonempty square 2-D array of finite numbers, or
        ``column`` is not an integer in ``[0, n)`` (booleans are rejected).
    """
    return pattern
```

### Step 3

03_initialize_adaptive_column

Goal
----
Start one column of the adaptive sparse approximate inverse from its a-priori pattern by solving the least-squares problem restricted to the rows that pattern reaches.

```python
def initialize_adaptive_column(matrix: "np.ndarray", column: int) -> tuple:
    r"""Return the a-priori pattern of one column and the least-squares column on it.

    Write $A$ for ``matrix`` and $k$ for ``column``. The a-priori pattern
    $J_k^0$ is the set of positions at which column $k$ of the identity, of
    $A$, or of $A^2$ (formed in floating point) is nonzero. The starting
    column $m$ is the vector of length $n$ whose entries vanish outside
    $J_k^0$ and which minimizes $\lVert A m - e_k \rVert_2$, with $e_k$ the
    unit vector of index $k$. ``matrix[:, pattern]`` is assumed to have full
    column rank, so $m$ is unique; its entries must be accurate to $10^{-12}$
    relative to the largest of them. Column scales may differ by up to
    $10^{198}$: after normalizing each reached column by its 2-norm, the
    reduced matrix has condition number at most $10^{5}$. The full-rank
    minimizer is required even for weakly scaled columns. Its coefficients
    multiplied by the respective column 2-norms must also be accurate to
    $10^{-8}$ relative to $\max(1, \text{the largest such coefficient})$. All
    relevant column norms and coefficients are finite. These are numerical
    accuracy conditions, not a prescribed library routine or rank-truncation
    rule.

    Parameters
    ----------
    matrix : np.ndarray
        Float array of shape ``(n, n)``, ``n >= 1``, with finite entries.
    column : int
        Column index ``k`` with ``0 <= k < n``.

    Returns
    -------
    tuple
        ``(pattern, m)``: the sorted 1-D integer array of pattern positions and
        the float column of length ``n``.

    Raises
    ------
    ValueError
        If ``matrix`` is not a nonempty square 2-D array of finite numbers, or
        ``column`` is not an integer in ``[0, n)`` (booleans are rejected).
    """
    return (pattern, column_vector)
```

### Step 4

04_select_residual_rows

Goal
----
Choose which residual entries drive one enlargement pass of a column, using a threshold relative to the residual norm and a cap on the number admitted.

```python
def select_residual_rows(residual: "np.ndarray", rows: "np.ndarray", used: "np.ndarray", threshold: float, cap: int) -> "np.ndarray":
    r"""Return the residual rows admitted in one enlargement pass.

    Write $r$ for ``residual``, $\delta$ for ``threshold`` and $c$ for ``cap``.
    The candidates are the entries of ``rows`` that do not appear in ``used``.
    A candidate $i$ qualifies when $\lvert r(i) \rvert \ge \delta \lVert r \rVert_2$,
    the norm being taken over the whole vector $r$. Of the qualifying
    candidates, at most $c$ are admitted: those with the largest
    $\lvert r(i) \rvert$. The admitted rows are returned in order of
    decreasing $\lvert r(i) \rvert$, ties going to the smaller row index
    first. The result may be empty.

    Parameters
    ----------
    residual : np.ndarray
        1-D float array of length ``n >= 1`` with finite entries.
    rows : np.ndarray
        1-D integer array of distinct row indices in ``[0, n)``; may be
        empty.
    used : np.ndarray
        1-D integer array of row indices in ``[0, n)`` already admitted in
        earlier passes; may be empty.
    threshold : float
        Finite nonnegative relative threshold.
    cap : int
        Maximum number of rows admitted, at least 1.

    Returns
    -------
    np.ndarray
        1-D integer array of admitted row indices, ordered as described.

    Raises
    ------
    ValueError
        If ``residual`` is not a nonempty 1-D array of finite numbers, ``rows``
        or ``used`` is not a 1-D integer array of indices in ``[0, n)``,
        ``rows`` repeats an index, ``threshold`` is not a finite nonnegative
        number, or ``cap`` is not an integer of at least 1 (booleans are
        rejected).
    """
    return admitted
```

### Step 5

05_enlarge_column_pattern

Goal
----
Enlarge the sparsity pattern of a column so that the least-squares solution can act on the residual entries admitted in the current pass.

```python
def enlarge_column_pattern(matrix: "np.ndarray", pattern: "np.ndarray", selected: "np.ndarray") -> "np.ndarray":
    r"""Return the column pattern enlarged toward the selected residual entries.

    Write $A$ for ``matrix``, $J$ for ``pattern`` and $\hat{D}$ for
    ``selected``. For a column $m$ of length $n$ whose entries vanish outside
    $J$, the residual of the column is $r = A m - e_k$. Return the sorted
    union of $J$ with every position $j$ such that letting $m(j)$ be nonzero
    changes at least one residual entry $r(i)$ with $i \in \hat{D}$, that is
    $J \cup \{\, j : A(i, j) \ne 0 \ \text{for some}\ i \in \hat{D} \,\}$. An
    empty $\hat{D}$ leaves the pattern unchanged.

    Parameters
    ----------
    matrix : np.ndarray
        Float array of shape ``(n, n)``, ``n >= 1``, with finite entries.
    pattern : np.ndarray
        1-D integer array of distinct positions in ``[0, n)``, in any order.
    selected : np.ndarray
        1-D integer array of residual indices in ``[0, n)``; may be empty.

    Returns
    -------
    np.ndarray
        Sorted 1-D integer array of the distinct positions of the enlarged
        pattern.

    Raises
    ------
    ValueError
        If ``matrix`` is not a nonempty square 2-D array of finite numbers, or
        ``pattern`` or ``selected`` is not a 1-D integer array of indices in
        ``[0, n)``, or ``pattern`` repeats a position.
    """
    return enlarged
```

### Step 6

06_advance_column_pattern

Goal
----
Carry out the pattern update of one enlargement pass of a column: pick the admissible residual rows of the current reduced problem, grow the pattern through them and record them as admitted.

```python
def advance_column_pattern(matrix: "np.ndarray", pattern: "np.ndarray", residual: "np.ndarray", used: "np.ndarray", threshold: float, cap: int) -> tuple:
    r"""Return the support and admission history of one adaptive transition.

    Write $A$ for ``matrix``, $J$ for ``pattern``, $r$ for ``residual``, $U$
    for ``used``, $\delta$ for ``threshold`` and $c$ for ``cap``. The residual
    is $r = A m - e_k$ for a column supported on $J$. The reduced problem's
    row set is $I = \{\, i : A(i, J) \ne 0 \,\}$, the rows on which the linear
    map $A[:, J]$ can be nonzero. Admissions are drawn from $I \setminus U$.
    Their residual magnitudes must satisfy
    $\lvert r(i) \rvert \ge \delta \lVert r \rVert_2$, where the norm includes
    every component of $r$. The admission set $\hat{D}$ contains the first $c$
    qualifying rows in descending residual magnitude, with smaller global row
    index breaking ties, or all of them if fewer qualify.

    The returned column support is the smallest superset of $J$ containing
    every coordinate of $m$ that can affect $r$ at an index in $\hat{D}$, that
    is $J \cup \{\, j : A(i, j) \ne 0 \ \text{for some}\ i \in \hat{D} \,\}$.
    The returned history is $U \cup \hat{D}$. History records admission
    independently of support growth. Thus an empty $\hat{D}$ leaves both sets
    unchanged, while a nonempty $\hat{D}$ is recorded even when the support
    stays the same. The row indices in $I$, $\hat{D}$ and $U$ are indices of
    $A$, not positions within a reduced vector. Both output sets are
    represented in increasing index order.

    Parameters
    ----------
    matrix : np.ndarray
        Float array of shape ``(n, n)``, ``n >= 1``, with finite entries.
    pattern : np.ndarray
        1-D integer array of distinct positions in ``[0, n)``, at least one.
    residual : np.ndarray
        Float array of length ``n`` with finite entries.
    used : np.ndarray
        1-D integer array of distinct row indices in ``[0, n)``; may be empty.
    threshold : float
        Finite nonnegative relative admission threshold.
    cap : int
        Maximum number of rows admitted, at least 1.

    Returns
    -------
    tuple
        ``(new_pattern, new_used)``: two sorted 1-D integer arrays of distinct
        indices.

    Raises
    ------
    ValueError
        If ``matrix`` is not a nonempty square 2-D array of finite numbers,
        ``pattern`` or ``used`` is not a 1-D integer array of distinct indices
        in ``[0, n)``, ``pattern`` is empty, ``residual`` does not have length
        ``n`` or has a non-finite entry, ``threshold`` is not a finite
        nonnegative number, or ``cap`` is not an integer of at least 1
        (booleans are rejected).
    """
    return (new_pattern, new_used)
```

### Step 7

07_extend_reduced_qr

Goal
----
Extend a reduced least-squares factorization when residual admission introduces new column coordinates and new residual rows, and recover the full-space column and residual norm.

```python
def extend_reduced_qr(state: tuple, added_rows: "np.ndarray", added_columns: "np.ndarray", coupling: "np.ndarray", column: int, size: int) -> tuple:
    r"""Extend a reduced QR state and solve its full-space column problem.

    The input state is $(I, J, Q, R)$. It represents the old reduced matrix
    $B = Q R$, with global row indices $I$ and global column indices $J$ in
    the supplied orders. $Q$ has orthonormal columns; $R$ is upper triangular
    with positive diagonal. The old columns are zero on ``added_rows``.
    ``coupling`` supplies the new columns on the concatenated row order
    $(I, \text{added\_rows})$ and in the order ``added_columns``. These data
    define the enlarged matrix completely: the old block is $B$ above zero
    rows, followed by ``coupling``. All entries outside the represented global
    rows and columns are zero.

    Return a state for this same enlarged matrix with both global index
    arrays sorted, and with the unique thin QR factorization whose $R$ has
    positive diagonal. Also return the length-``size`` vector $m$ supported on
    the returned columns minimizing
    $\lVert A_{\text{represented}} m - e_{\text{column}} \rVert_2$, and that
    FULL residual norm. An unrepresented target row contributes its unit
    residual, so
    $\lVert r \rVert_2 = \sqrt{\lVert \hat{r} \rVert_2^2 + [\, k \notin I \,]}$.
    The empty initial state has $I$ and $J$ empty and $Q$ and $R$ of shape
    $(0, 0)$. Empty additions are allowed when the resulting state has at
    least one column; a no-growth update still solves for ``column``.

    Column 2-norms may range from $10^{-100}$ to $10^{100}$. The represented
    matrix has full column rank and condition number at most $10^{5}$ after
    dividing each column by its 2-norm. This is a full-rank least-squares
    problem; its weakly scaled columns remain part of the solution. The
    entries of $Q$ must be accurate to $10^{-8}$ relative to
    $\max(1, \text{largest entry magnitude})$. The same accuracy is required
    for $R$ after dividing each column by its represented column's 2-norm,
    and for the solution coefficients after multiplying each by that norm.
    Equivalent stable factorization/update methods are accepted.

    Parameters
    ----------
    state : tuple
        (I, J, Q, R): distinct 1-D integer index arrays of lengths a and p,
        finite float Q of shape (a, p) and R of shape (p, p), with p <= a.
        Indices lie in [0, size). The factors satisfy the properties above.
    added_rows : np.ndarray
        Distinct 1-D integer global indices, disjoint from I, in any order.
    added_columns : np.ndarray
        Distinct 1-D integer global indices, disjoint from J, in any order.
    coupling : np.ndarray
        Finite float array of shape (a + len(added_rows), len(added_columns)).
    column : int
        Target unit-vector index in [0, size).
    size : int
        Positive full-space dimension. There must be at least one resulting
        column and at least as many represented rows as columns.

    Returns
    -------
    tuple
        (new_state, m, residual_norm), where new_state is (sorted_I,
        sorted_J, thin_Q, positive_diagonal_R), m is a length-size float
        array and residual_norm is a float. All state is explicit.

    Raises
    ------
    ValueError
        If state is not a four-entry tuple, size or column is not a valid
        integer (booleans excluded), an index array has wrong dimension,
        noninteger/repeated/out-of-range entries, old and added indices
        overlap, a factor/coupling has the wrong shape or nonfinite entries,
        or the resulting dimensions violate the conditions above.
        Orthogonality, triangularity and full column rank are preconditions.
    """
    return (new_state, column_vector, residual_norm)
```

### Step 8

08_compute_adaptive_column

Goal
----
Build one column of the adaptive residual-based sparse approximate inverse by solving on an initial pattern and then enlarging the pattern pass by pass under a residual tolerance, a relative admission threshold, a per-pass cap and a pass limit.

```python
def compute_adaptive_column(matrix: "np.ndarray", column: int, tolerance: float, threshold: float, cap: int, max_passes: int, *, seed_mode: str = "power") -> tuple:
    r"""Return the terminal least-squares column of the adaptive support process.

    Write $A$ for ``matrix`` and $k$ for ``column``. For any support $J$, let
    $m(J)$ be the unique minimizer of $\lVert A m - e_k \rVert_2$ among
    vectors that vanish outside $J$, where $e_k$ is the $k$-th unit vector.
    Its entries must be accurate to $10^{-12}$ relative to the largest.
    Coefficients multiplied by their matrix-column 2-norms must also be
    accurate to $10^{-8}$ relative to
    $\max(1, \text{the largest such coefficient})$. The column norms may span
    up to $10^{198}$, but each reduced matrix after column normalization has
    condition number at most $10^{5}$. All relevant norms and coefficients
    are finite. Define $r(J) = A \, m(J) - e_k$ over all $n$ rows, including
    rows not reached by $A[:, J]$.

    With ``seed_mode`` equal to ``"power"`` the initial support is the union
    of the nonzero positions of $e_k$, $A e_k$ and $A^2 e_k$, where $A^2$
    denotes the floating-point matrix square, including exact numerical
    cancellations. With ``seed_mode`` equal to ``"diagonal"`` the initial
    support is $\{k\}$ alone. The initial admission history is empty.

    At each state, the eligible rows are the current reduced row support
    excluding the admission history. A row qualifies if
    $\lvert r(J)(i) \rvert \ge \text{threshold} \cdot \lVert r(J) \rVert_2$.
    Admit at most ``cap`` qualifying rows, ranked by decreasing residual
    magnitude with smaller global row index breaking ties. The next support
    is the smallest superset of $J$ containing all column coordinates coupled
    to those admitted residual rows; the next history includes all
    admissions.

    Return the first state with residual norm at most ``tolerance``, with no
    qualifying admission, or with ``max_passes`` completed admissions. One
    completed admission means one nonempty group of admitted rows, even if it
    introduces no new column coordinate. The initial minimization consumes no
    admission, and an empty admission consumes none. The returned vector and
    norm correspond to the least-squares problem on the terminal support,
    after all counted admissions have taken effect.

    Parameters
    ----------
    matrix : np.ndarray
        Float array of shape $(n, n)$, $n \ge 1$, with finite entries, such
        that ``matrix[:, pattern]`` has full column rank for every pattern met
        (true for any nonsingular matrix).
    column : int
        Column index $k$ with $0 \le k < n$.
    tolerance : float
        Finite positive residual tolerance $\epsilon$.
    threshold : float
        Finite nonnegative relative admission threshold $\delta$.
    cap : int
        Maximum number of rows admitted per pass, at least 1.
    max_passes : int
        Maximum number of completed passes, at least 0.
    seed_mode : str
        Either ``"power"`` for the $e_k$, $A e_k$, $A^2 e_k$ support, or
        ``"diagonal"`` for the single position $k$.

    Returns
    -------
    tuple
        ``(m, residual_norm, passes)``: the final column as a float array of
        length $n$, the float $\lVert A m - e_k \rVert_2$ of that column, and
        the int number of completed passes.

    Raises
    ------
    ValueError
        If ``matrix`` is not a nonempty square 2-D array of finite numbers,
        ``column`` is not an integer in $[0, n)$, ``tolerance`` is not a
        finite positive number, ``threshold`` is not a finite nonnegative
        number, ``cap`` is not an integer of at least 1, ``max_passes`` is not
        an integer of at least 0, or ``seed_mode`` is neither ``"power"`` nor
        ``"diagonal"`` (booleans are rejected throughout).
    """
    return (column_vector, residual_norm, passes)
```

### Step 9

09_evaluate_preconditioner_benchmark

Goal
----
Assemble the benchmark convection-diffusion matrix from its velocity field, build its adaptive residual-based sparse approximate inverse column by column and return the Frobenius norm of $AM - I$.

```python
def evaluate_preconditioner_benchmark(
    n_side: int = 18,
    flow_x: float = 30.0,
    flow_y: float = 60.0,
    flow_center: float = 0.37,
    tolerance: float = 0.1,
    threshold: float = 0.25,
    cap: int = 3,
    max_passes: int = 3,
    *, seed_mode: str = "power",
) -> float:
    r"""Return $\lVert A M - I \rVert_F$ for the adaptive preconditioner of the benchmark matrix.

    The cells have edges $x_i = (i/n)^{1.7}$ and $y_j = (j/n)^{1.3}$ for
    $i, j = 0, \dots, n$ with $n$ = ``n_side``, and midpoint centres
    $(X_i, Y_j)$. Cell speeds are
    $v_x = \text{flow\_x} \, (1 + X_i) \cos(2 \pi Y_j)$ and
    $v_y = \text{flow\_y} \, \sin(2 \pi (X_i - \text{flow\_center}))$. The
    assembly data are

    $$D_x[j,i] = 0.08 \left[ 1 + 0.6 \sin^2(\pi x_i) \cos^2(2 \pi Y_j) \right]$$

    $$D_y[j,i] = 0.12 \left[ 1 + 0.5 \cos^2(\pi X_i) \sin^2(2 \pi y_{j+1}) \right]$$

    $$\text{reaction}[j,i] = 0.1 + 0.05 \cos^2(\pi X_i) \sin^2(2 \pi Y_j)$$

    $$\rho_{\mathrm{left}}[j] = 0.2 + 2 \sin^2(2 \pi Y_j), \quad
    \rho_{\mathrm{right}}[j] = 0.5 + 3 \cos^2(2 \pi Y_j)$$

    $A$ is the finite-volume, $y$-periodic, $x$-Robin operator returned by
    ``assemble_upwind_convection_diffusion``, including its volume scaling.
    $M$ is the right preconditioner whose column $k$ is the column returned by
    ``compute_adaptive_column(A, k, tolerance, threshold, cap, max_passes,
    seed_mode=seed_mode)``, every column being computed independently. Return
    $\lVert A M - I \rVert_F$.

    Parameters
    ----------
    n_side : int
        Number of cells per axis, at least 1.
    flow_x : float
        Finite amplitude of the $x$ velocity component.
    flow_y : float
        Finite amplitude of the $y$ velocity component.
    flow_center : float
        Finite $x$ phase shift in the sine defining the $y$ velocity.
    tolerance : float
        Finite positive column residual tolerance $\epsilon$.
    threshold : float
        Finite nonnegative relative admission threshold $\delta$.
    cap : int
        Maximum number of rows admitted per pass, at least 1.
    max_passes : int
        Maximum number of passes per column, at least 0.
    seed_mode : str
        ``"power"`` for the $e_k$, $A e_k$, $A^2 e_k$ seed supports, or
        ``"diagonal"`` to start every column from its diagonal position
        alone.

    Returns
    -------
    float
        The Frobenius norm $\lVert A M - I \rVert_F$.

    Raises
    ------
    ValueError
        If ``n_side`` is not an integer of at least 1, a flow parameter is not
        a finite number, ``seed_mode`` is neither ``"power"`` nor
        ``"diagonal"``, or any argument passed on to the earlier steps is
        outside their stated domains (booleans are rejected).
    """
    return 0.0
```
