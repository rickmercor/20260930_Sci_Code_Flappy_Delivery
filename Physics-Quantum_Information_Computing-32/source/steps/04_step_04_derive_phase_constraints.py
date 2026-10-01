"""
Derive the binary sign constraints for each commuting context.

For a context $S$, form $M_S$ with columns $(\boldsymbol a_j,\boldsymbol b_j)$. A relation $\boldsymbol c\in\ker_{\mathbb F_2}M_S$ specifies a product proportional to identity. Evaluate that product in increasing measurement order, using

$$

(p,a,b)(p',a',b')=(p+p'+2b\cdot a'\bmod4,\ a+a'\bmod2,\ b+b'\bmod2).

$$

Commutativity restricts the resulting phase to $(-1)^\sigma$. An assignment $f_j=(-1)^{x_j}$ is physical precisely when $K_S\boldsymbol x=\boldsymbol\sigma$ over $\mathbb F_2$, where the rows of $K_S$ span the kernel. To make the returned basis unambiguous, reduce $M_S$ to reduced row echelon form with leftmost pivots; construct one null vector per free column, in ascending free-column order. The zero-dimensional kernel contributes no rows.

Returns
-------
Integer array of shape $(L,m+2)$; each row contains the context index, the phase bit $\sigma$, then the kernel vector padded with zeros outside its context, ordered by context and free column.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def derive_phase_constraints(
    encoded: "np.ndarray", contexts: "np.ndarray"
) -> "np.ndarray":
    r"""Derive the binary sign constraints for each commuting context.

    Parameters
    ----------
    encoded : np.ndarray
        Unsigned Hermitian Pauli encoding of shape $(m,2n+1)$.
    contexts : np.ndarray
        Nonempty binary membership array of shape $(C,m)$ whose rows are nonempty commuting sets; they need not be maximal for this step.

    Returns
    -------
    relations : np.ndarray
        Integer array of shape $(L,m+2)$; each row contains the context index, the phase bit $\sigma$, then the kernel vector padded with zeros outside its context, ordered by context and free column.

    Raises
    ------
    ValueError
        If the encoding is invalid, a context has an invalid shape or entries, is empty, or contains anticommuting observables.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _binary_rref(matrix):
    reduced = np.array(matrix, dtype=np.int64, copy=True)
    pivots = []
    row = 0
    for column in range(reduced.shape[1]):
        candidates = np.flatnonzero(reduced[row:, column])
        if not candidates.size:
            continue
        pivot = row + int(candidates[0])
        reduced[[row, pivot]] = reduced[[pivot, row]]
        for other in range(reduced.shape[0]):
            if other != row and reduced[other, column]:
                reduced[other] ^= reduced[row]
        pivots.append(column)
        row += 1
        if row == reduced.shape[0]:
            break
    return reduced, pivots


def _binary_kernel(matrix):
    reduced, pivots = _binary_rref(matrix)
    free = [j for j in range(matrix.shape[1]) if j not in pivots]
    basis = np.zeros((len(free), matrix.shape[1]), dtype=np.int64)
    for row, column in enumerate(free):
        basis[row, column] = 1
        for pivot_row, pivot_column in enumerate(pivots):
            basis[row, pivot_column] = reduced[pivot_row, column]
    return basis


def _checked_contexts(contexts, m):
    values = np.asarray(contexts)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != m:
        raise ValueError("contexts must have shape (C,m)")
    if not np.all((values == 0) | (values == 1)) or np.any(values.sum(axis=1) == 0):
        raise ValueError("each context must have nonempty binary membership")
    return values.astype(np.int64, copy=True)


def _oracle_derive_phase_constraints(
    encoded: "np.ndarray", contexts: "np.ndarray"
) -> "np.ndarray":
    values, n = _checked_encoding(encoded)
    m = len(values)
    contexts = _checked_contexts(contexts, m)
    adjacency = _oracle_build_frustration_matrix(values)
    relations = []
    for context_index, context in enumerate(contexts):
        indices = np.flatnonzero(context)
        if np.any(adjacency[np.ix_(indices, indices)]):
            raise ValueError("context contains anticommuting observables")
        kernel = _binary_kernel(values[indices, 1:].T)
        for relation in kernel:
            phase = 0
            accumulated_b = np.zeros(n, dtype=np.int64)
            for local_index in np.flatnonzero(relation):
                word = values[indices[local_index]]
                phase = (
                    phase + int(word[0]) + 2 * int(accumulated_b @ word[1 : n + 1])
                ) % 4
                accumulated_b ^= word[n + 1 :]
            padded = np.zeros(m + 2, dtype=np.int64)
            padded[:2] = context_index, phase // 2
            padded[2 + indices] = relation
            relations.append(padded)
    return np.array(relations, dtype=np.int64).reshape(-1, m + 2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical cases for this step."""
    return [
        {
            "setup": """import numpy as np
encoded = np.array([[0,1,1,0,0],[2,1,1,1,1],[0,0,0,1,1]])
contexts = np.ones((1,3), dtype=int)
""",
            "call": "derive_phase_constraints(encoded.copy(), contexts.copy())",
            "gold_call": "_oracle_derive_phase_constraints(encoded.copy(), contexts.copy())",
            "tol": 0,
        },
        {
            "setup": """import numpy as np
encoded = np.array([[0,1,0,0,0],[0,0,1,0,0]])
contexts = np.ones((1,2), dtype=int)
""",
            "call": "derive_phase_constraints(encoded.copy(), contexts.copy())",
            "gold_call": "_oracle_derive_phase_constraints(encoded.copy(), contexts.copy())",
            "tol": 0,
        },
        {
            "setup": """import numpy as np
encoded = np.array([[0,1,0,0,0],[0,0,1,0,0],[0,1,1,0,0]])
contexts = np.ones((1,3), dtype=int)
""",
            "call": "derive_phase_constraints(encoded.copy(), contexts.copy())",
            "gold_call": "_oracle_derive_phase_constraints(encoded.copy(), contexts.copy())",
            "tol": 0,
        },
        {
            "setup": """import numpy as np
encoded = np.array([[0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 1, 1], [2, 1, 1, 1, 1, 1, 0], [2, 1, 1, 1, 0, 1, 1], [2, 1, 1, 1, 1, 0, 1], [0, 0, 0, 0, 1, 0, 1]])
contexts = np.array([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0]])
""",
            "call": "derive_phase_constraints(encoded.copy(), contexts.copy())",
            "gold_call": "_oracle_derive_phase_constraints(encoded.copy(), contexts.copy())",
            "tol": 0,
        },
        {
            "setup": """import numpy as np
encoded = np.array([[0,1,0],[0,0,1]])
contexts = np.ones((1,2), dtype=int)

def _raises_value_error(fn):
    try:
        fn(encoded.copy(), contexts.copy())
    except ValueError:
        return 1
    return 0
""",
            "call": "_raises_value_error(derive_phase_constraints)",
            "gold_call": "_raises_value_error(_oracle_derive_phase_constraints)",
            "tol": 0.0,
        },
    ]
