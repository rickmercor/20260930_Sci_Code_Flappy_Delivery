"""
Generate the projected stabilizer vertices from admissible context signs.

For each maximal context $S$, the physical sign vectors are the affine binary solutions of $K_Sx=\sigma$. If $x_*$ is one solution, enumerate $x_*+\ker K_S$; the number of solutions is $2^{|S|-\operatorname{rank}K_S}$. Embed each assignment as a measured correlation vector,

$$

v_j=\begin{cases}(-1)^{x_j},\&j\in S,\\0,\&j\notin S.\end{cases}

$$

The convex hull of these vectors is the measured projection of the stabilizer polytope. Zero outside $S$ is essential: it expresses the correlations of the mixed stabilizer state associated with that commuting subgroup. Return distinct columns in lexicographic order of their full signed measurement vectors.

Returns
-------
Integer array $(m,N)$ of distinct projected vertices, lexicographically sorted by column; all entries are $-1,0,1$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def construct_projected_vertices(
    contexts: "np.ndarray", relations: "np.ndarray"
) -> "np.ndarray":
    r"""Generate the projected stabilizer vertices from admissible context signs.

    Parameters
    ----------
    contexts : np.ndarray
        Nonempty binary membership array $(C,m)$ of the maximal commuting contexts.
    relations : np.ndarray
        Integer array $(L,m+2)$ from the relation step; context index, phase bit, then padded relation bits. Empty shape $(0,m+2)$ is allowed.

    Returns
    -------
    vertices : np.ndarray
        Integer array $(m,N)$ of distinct projected vertices, lexicographically sorted by column; all entries are $-1,0,1$.

    Raises
    ------
    ValueError
        If membership or relation dimensions or bits are invalid, an index is out of range, a relation has support outside its context, or a binary constraint system is inconsistent.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _affine_binary_solutions(matrix, target):
    columns = matrix.shape[1]
    augmented, pivots = _binary_rref(np.column_stack((matrix, target)))
    if columns in pivots:
        raise ValueError("inconsistent binary constraints")
    particular = np.zeros(columns, dtype=np.int64)
    for row, column in enumerate(pivots):
        particular[column] = augmented[row, -1]
    kernel = _binary_kernel(matrix)
    solutions = np.empty((1 << len(kernel), columns), dtype=np.int64)
    for mask in range(len(solutions)):
        value = particular.copy()
        for j, direction in enumerate(kernel):
            if (mask >> j) & 1:
                value ^= direction
        solutions[mask] = value
    return solutions


def _oracle_construct_projected_vertices(
    contexts: "np.ndarray", relations: "np.ndarray"
) -> "np.ndarray":
    raw = np.asarray(contexts)
    if raw.ndim != 2:
        raise ValueError("contexts must be a matrix")
    contexts = _checked_contexts(raw, raw.shape[1])
    m = contexts.shape[1]
    relations = np.asarray(relations)
    if (
        relations.ndim != 2
        or relations.shape[1] != m + 2
        or not np.issubdtype(relations.dtype, np.integer)
    ):
        raise ValueError("relations must be an integer (L,m+2) array")
    if not np.all((relations[:, 1:] == 0) | (relations[:, 1:] == 1)):
        raise ValueError("phase and relation entries must be binary")
    if np.any((relations[:, 0] < 0) | (relations[:, 0] >= len(contexts))):
        raise ValueError("context index out of range")
    columns = []
    for index, context in enumerate(contexts):
        active = np.flatnonzero(context)
        selected = relations[relations[:, 0] == index]
        if np.any(selected[:, 2:][:, context == 0]):
            raise ValueError("relation has support outside its context")
        signs = _affine_binary_solutions(selected[:, 2:][:, active], selected[:, 1])
        block = np.zeros((len(signs), m), dtype=np.int64)
        block[:, active] = 1 - 2 * signs
        columns.extend(block)
    return np.unique(np.array(columns), axis=0).T.copy()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical cases for this step."""
    return [
        {
            "setup": """import numpy as np
contexts = np.ones((1,3), dtype=int)
relations = np.array([[0,1,1,1,1]])
""",
            "call": "construct_projected_vertices(contexts.copy(), relations.copy())",
            "gold_call": "_oracle_construct_projected_vertices(contexts.copy(), relations.copy())",
            "tol": 0,
        },
        {
            "setup": """import numpy as np
contexts = np.eye(3, dtype=int)
relations = np.empty((0,5), dtype=int)
""",
            "call": "construct_projected_vertices(contexts.copy(), relations.copy())",
            "gold_call": "_oracle_construct_projected_vertices(contexts.copy(), relations.copy())",
            "tol": 0,
        },
        {
            "setup": """import numpy as np
contexts = np.ones((1,3), dtype=int)
relations = np.array([[0,0,1,1,1]])
""",
            "call": "construct_projected_vertices(contexts.copy(), relations.copy())",
            "gold_call": "_oracle_construct_projected_vertices(contexts.copy(), relations.copy())",
            "tol": 0,
        },
        {
            "setup": """import numpy as np
contexts = np.array([[1, 1, 1, 1, 1, 1, 1]])
relations = np.array([[0, 1, 1, 1, 0, 1, 0, 0, 0], [0, 1, 1, 0, 1, 0, 1, 0, 0], [0, 1, 1, 1, 1, 0, 0, 1, 0], [0, 0, 0, 1, 1, 0, 0, 0, 1]])
""",
            "call": "construct_projected_vertices(contexts.copy(), relations.copy())",
            "gold_call": "_oracle_construct_projected_vertices(contexts.copy(), relations.copy())",
            "tol": 0,
        },
        {
            "setup": """import numpy as np
contexts = np.ones((1,2), dtype=int)
relations = np.array([[0,1,0,0]])

def _raises_value_error(fn):
    try:
        fn(contexts.copy(), relations.copy())
    except ValueError:
        return 1
    return 0
""",
            "call": "_raises_value_error(construct_projected_vertices)",
            "gold_call": "_raises_value_error(_oracle_construct_projected_vertices)",
            "tol": 0.0,
        },
    ]
