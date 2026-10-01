"""
Extend a reduced least-squares factorization when residual admission introduces new column coordinates and new residual rows, and recover the full-space column and residual norm.

Equation (19) of the source method refits a column on an enlarged support, $\min \|A(I_k^{l+1}, J_k^{l+1}) m_k(J_k^{l+1}) - e_k(I_k^{l+1})\|_2$. The old columns vanish on newly reached rows, so their orthogonal factor $Q$ can be embedded in the larger row space. The old factorization $B = QR$ and the newly introduced couplings suffice to determine the enlarged problem, including a target row outside its represented rows.

Returns
-------
tuple: ((sorted row indices $I$, sorted column indices $J$, $Q$, $R$), full column $m$, full residual norm $\|r\|_2$).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_extend_reduced_qr(state: tuple, added_rows: "np.ndarray", added_columns: "np.ndarray", coupling: "np.ndarray", column: int, size: int) -> tuple:
    """Scale-aware orthogonal extension followed by a small column reorder."""

    def _qr_column_norms(block):
        """Column norms without squaring the physical column scales."""
        largest = np.max(np.abs(block), axis=0)
        return largest * np.linalg.norm(block / largest, axis=0)

    if not (isinstance(size, (int, np.integer)) and not isinstance(size, bool) and size > 0):
        raise ValueError("size must be a positive integer")
    if not (isinstance(column, (int, np.integer)) and not isinstance(column, bool) and 0 <= column < size):
        raise ValueError("column must be a full-space integer index")
    if not isinstance(state, tuple) or len(state) != 4:
        raise ValueError("state must contain I, J, Q and R")

    def _indices(values):
        values = np.asarray(values)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("indices must be one-dimensional integer arrays")
        if np.unique(values).size != values.size or np.any(values < 0) or np.any(values >= size):
            raise ValueError("indices must be distinct and in range")
        return values.astype(int)

    old_rows, old_columns = _indices(state[0]), _indices(state[1])
    new_rows, new_columns = _indices(added_rows), _indices(added_columns)
    if np.intersect1d(old_rows, new_rows).size or np.intersect1d(old_columns, new_columns).size:
        raise ValueError("old and added index sets must be disjoint")
    old_q, old_r = np.asarray(state[2], dtype=float), np.asarray(state[3], dtype=float)
    new_block = np.asarray(coupling, dtype=float)
    a, p, d, t = old_rows.size, old_columns.size, new_rows.size, new_columns.size
    if (old_q.shape != (a, p) or old_r.shape != (p, p)
            or new_block.shape != (a + d, t) or p > a
            or p + t == 0 or p + t > a + d):
        raise ValueError("inconsistent reduced dimensions")
    if not all(np.all(np.isfinite(x)) for x in (old_q, old_r, new_block)):
        raise ValueError("factors and couplings must be finite")

    embedded_q = np.zeros((a + d, p))
    embedded_q[:a] = old_q
    old_scales = _qr_column_norms(old_r) if p else np.empty(0)
    old_normalized_r = old_r / old_scales if p else old_r.copy()
    if t:
        new_scales = _qr_column_norms(new_block)
        normalized_block = new_block / new_scales
        cross = embedded_q.T @ normalized_block
        orthogonal_block = normalized_block - embedded_q @ cross
        # Reprojection preserves weak new directions in almost-aligned blocks.
        correction = embedded_q.T @ orthogonal_block
        cross += correction
        orthogonal_block -= embedded_q @ correction
        added_q, added_r = np.linalg.qr(orthogonal_block, mode="reduced")
        working_q = np.column_stack((embedded_q, added_q))
        working_r = np.zeros((p + t, p + t))
        working_r[:p, :p] = old_normalized_r
        working_r[:p, p:] = cross
        working_r[p:, p:] = added_r
    else:
        new_scales = np.empty(0)
        working_q, working_r = embedded_q, old_normalized_r

    row_ids = np.concatenate((old_rows, new_rows))
    column_ids = np.concatenate((old_columns, new_columns))
    row_order, column_order = np.argsort(row_ids), np.argsort(column_ids)
    # The factor describes sorted global columns, not append order.
    rotation, sorted_r = np.linalg.qr(working_r[:, column_order], mode="reduced")
    sorted_q = (working_q @ rotation)[row_order]
    signs = np.where(np.diag(sorted_r) < 0.0, -1.0, 1.0)
    sorted_q *= signs
    sorted_r *= signs[:, None]
    scales = np.concatenate((old_scales, new_scales))[column_order]
    rows, columns = row_ids[row_order], column_ids[column_order]
    target = (rows == column).astype(float)
    projected = sorted_q.T @ target
    scaled_solution = np.linalg.solve(sorted_r, projected)
    vector = np.zeros(size)
    vector[columns] = scaled_solution / scales
    reduced_residual = sorted_q @ projected - target
    norm = np.sqrt(reduced_residual @ reduced_residual + float(not np.any(rows == column)))
    return (rows, columns, sorted_q, sorted_r * scales), vector, float(norm)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Twelve explicit differential cases for Studio's structural parser."""
    common = '''import numpy as np
def make_state(seed, I, J, extra_I, extra_J, scales, alignment, size, target):
    rng = np.random.default_rng(seed)
    I, J, extra_I, extra_J = [np.array(x, dtype=int) for x in (I,J,extra_I,extra_J)]
    a, p, d, t = len(I), len(J), len(extra_I), len(extra_J)
    if p:
        base = rng.normal(size=(a,p))
        q, r = np.linalg.qr(base, mode='reduced')
        signs = np.where(np.diag(r) < 0, -1., 1.)
        q *= signs
        r *= signs[:,None]
        r *= np.asarray(scales[:p])
    else:
        q, r = np.empty((a,0)), np.empty((0,0))
    C = rng.normal(size=(a+d,t))
    if alignment and p and t:
        embedded = np.vstack((q,np.zeros((d,p))))
        C = embedded @ rng.normal(size=(p,t)) + alignment*C
    C *= np.asarray(scales[p:])
    B = np.column_stack((np.vstack((q@r,np.zeros((d,p)))),C))
    largest = np.max(np.abs(B),axis=0)
    norms = largest*np.linalg.norm(B/largest,axis=0)
    ordered_norms = norms[np.argsort(np.r_[J,extra_J])]
    return (I,J,q,r), extra_I, extra_J, C, target, size, ordered_norms
def pack(fn, case):
    state,di,dj,C,target,size,norms = case
    result = fn(tuple(x.copy() for x in state),di.copy(),dj.copy(),C.copy(),target,size)
    if not isinstance(result,tuple) or len(result)!=3 or not isinstance(result[0],tuple) or len(result[0])!=4:
        raise AssertionError('Expected (four-array state, vector, norm)')
    I,J,Q,R = [np.asarray(x) for x in result[0]]
    m = np.asarray(result[1])
    a,p = len(state[0])+len(di),len(state[1])+len(dj)
    if (I.shape!=(a,) or J.shape!=(p,) or Q.shape!=(a,p) or R.shape!=(p,p) or m.shape!=(size,)
            or not np.issubdtype(I.dtype,np.integer) or not np.issubdtype(J.dtype,np.integer)):
        raise AssertionError('Incorrect state dimensions or index types')
    return np.concatenate((I.astype(float),J.astype(float),Q.ravel(),(R/norms).ravel(),
                           m[J]*norms,m[np.setdiff1d(np.arange(size),J)],np.array([result[2]])))
'''
    status = common + '''case = make_state(61,[0,2,4],[1],[],[3],[1.,2.],0.,6,0)
def status(fn, mode):
    state,di,dj,C,k,n,norms = case
    if mode == 0:
        C = C[:-1]
    elif mode == 1:
        dj = state[1].copy()
    else:
        n = True
    try:
        fn(tuple(x.copy() for x in state),di.copy(),dj.copy(),C.copy(),k,n)
    except ValueError:
        return 1
    return 0
'''
    return [
        # Normal: new rows and columns with interleaved global indices.
        {
            "setup": common + 'case = make_state(19, [7, 1, 9, 4, 0], [6, 2], [5, 3], [1, 8, 0], [1.0, 2.0, 0.3, 4.0, 1.0], 0.0, 12, 3)\n',
            "call": 'pack(extend_reduced_qr, case)',
            "gold_call": 'pack(_oracle_extend_reduced_qr, case)',
            "tol": 1e-08,
        },
        # Boundary: new columns without additional residual rows.
        {
            "setup": common + 'case = make_state(23, [4, 0, 6, 2, 8, 1], [7, 3], [], [0, 5], [2.0, 0.4, 3.0, 0.2], 0.0, 10, 6)\n',
            "call": 'pack(extend_reduced_qr, case)',
            "gold_call": 'pack(_oracle_extend_reduced_qr, case)',
            "tol": 1e-08,
        },
        # Scale edge: old and added columns have very different norms.
        {
            "setup": common + 'case = make_state(31, [6, 2, 9, 0], [7, 4], [1, 8, 3], [0, 5], [1e-80, 1e+80, 1e+45, 1e-45], 0.0, 11, 8)\n',
            "call": 'pack(extend_reduced_qr, case)',
            "gold_call": 'pack(_oracle_extend_reduced_qr, case)',
            "tol": 1e-08,
        },
        # No growth: refit the unchanged scaled state for a represented target.
        {
            "setup": common + 'case = make_state(37, [4, 1, 7, 0, 3, 8], [6, 2, 5], [], [], [1e+70, 1e-70, 3.0], 0.0, 10, 1)\n',
            "call": 'pack(extend_reduced_qr, case)',
            "gold_call": 'pack(_oracle_extend_reduced_qr, case)',
            "tol": 1e-08,
        },
        # Unrepresented target: keep its full unit residual.
        {
            "setup": common + 'case = make_state(41, [3, 0, 5, 1], [4, 2], [], [], [1.0, 2.0], 0.0, 8, 7)\n',
            "call": 'pack(extend_reduced_qr, case)',
            "gold_call": 'pack(_oracle_extend_reduced_qr, case)',
            "tol": 1e-08,
        },
        # Empty state: bootstrap a factorization from added rows and columns.
        {
            "setup": common + 'case = make_state(43, [], [], [5, 0, 3, 1], [4, 2, 0], [1e-99, 1e+99, 1.0], 0.0, 7, 3)\n',
            "call": 'pack(extend_reduced_qr, case)',
            "gold_call": 'pack(_oracle_extend_reduced_qr, case)',
            "tol": 1e-08,
        },
        # Near alignment: resolve weak directions in the new column block.
        {
            "setup": common + 'case = make_state(47, [5, 2, 7, 0, 8, 3], [6, 1], [4, 9], [0, 7], [1e-90, 1e+90, 1e+60, 1e-60], 0.001, 11, 9)\n',
            "call": 'pack(extend_reduced_qr, case)',
            "gold_call": 'pack(_oracle_extend_reduced_qr, case)',
            "tol": 1e-08,
        },
        # Growth with a target outside both the old and new represented rows.
        {
            "setup": common + 'case = make_state(53, [4, 1, 7, 0], [6, 2], [3, 5], [0, 4], [1.0, 2.0, 0.5, 3.0], 0.0, 10, 9)\n',
            "call": 'pack(extend_reduced_qr, case)',
            "gold_call": 'pack(_oracle_extend_reduced_qr, case)',
            "tol": 1e-08,
        },
        # New zero rows without new columns.
        {
            "setup": common + 'case = make_state(59, [3, 1, 4], [2], [0], [], [1e-60], 0.0, 6, 0)\n',
            "call": 'pack(extend_reduced_qr, case)',
            "gold_call": 'pack(_oracle_extend_reduced_qr, case)',
            "tol": 1e-08,
        },
        # Invalid coupling shape.
        {
            "setup": status,
            "call": 'status(extend_reduced_qr, 0)',
            "gold_call": 'status(_oracle_extend_reduced_qr, 0)',
        },
        # Invalid overlap of old and added column indices.
        {
            "setup": status,
            "call": 'status(extend_reduced_qr, 1)',
            "gold_call": 'status(_oracle_extend_reduced_qr, 1)',
        },
        # Invalid Boolean full-space dimension.
        {
            "setup": status,
            "call": 'status(extend_reduced_qr, 2)',
            "gold_call": 'status(_oracle_extend_reduced_qr, 2)',
        },
    ]
