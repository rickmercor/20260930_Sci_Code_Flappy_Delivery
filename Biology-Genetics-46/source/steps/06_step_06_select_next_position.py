"""
Choose the next variant to add to the growing global assembly, or open a new phasing block when the current one is exhausted, from the state of the SNP line graph alone.

Because haplotype labels are arbitrary, a variant can only be attached to an existing assembly through vertices that already touch phased positions, and attaching the variant supported by the most such vertices keeps every decision maximally constrained. When no vertex reaches the phased set the current connected component is finished and a new, independently labelled block must be opened.

Returns
-------
np.ndarray of shape (3 + U,), int: the action code, the chosen SNP, the chosen vertex and the frontier indicator.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def select_next_position(nodes: np.ndarray, node_phased: np.ndarray,
                         position_phased: np.ndarray) -> np.ndarray:
    """Decide the next move of the greedy variant-by-variant assembly.

    Two vertices of the SNP line graph are adjacent when they share exactly one
    SNP. The frontier is the set of unprocessed vertices that are adjacent to at
    least one processed vertex and that still carry at least one unphased SNP.

    When the frontier is not empty the chosen variant is the unphased SNP that
    appears in the largest number of frontier vertices, ties broken in favour of
    the smallest SNP index. When it is empty but some unprocessed vertex still
    carries an unphased SNP, a new block is opened at the first such vertex in
    the given vertex order. Otherwise no move remains.

    Parameters
    ----------
    nodes : np.ndarray
        Integer array of shape (U, 2) holding the SNP pair of each vertex in
        topological order.
    node_phased : np.ndarray
        Array of shape (U,) of zeros and ones, one where the vertex has already
        been processed.
    position_phased : np.ndarray
        Array of shape (n_snps,) of zeros and ones, one where the SNP has
        already been assigned in the global haplotype matrix.

    Returns
    -------
    move : np.ndarray
        Integer array of shape (3 + U,). Entry 0 is the action: 0 to phase a
        variant, 1 to open a new block, 2 when nothing remains. Entry 1 is the
        chosen SNP index for action 0 and -1 otherwise. Entry 2 is the chosen
        vertex index for action 1 and -1 otherwise. Entries 3 onward are one
        for each frontier vertex and zero elsewhere, and are all zero unless
        the action is 0.

    Raises
    ------
    ValueError
        If ``nodes`` is not a two-dimensional array with two columns of finite
        integer-valued entries, if ``node_phased`` or ``position_phased`` is not
        a non-empty one-dimensional array of zeros and ones of the matching
        length, or if any entry of ``nodes`` is not a valid index into
        ``position_phased``.
    """
    return move  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_select_next_position(nodes: np.ndarray, node_phased: np.ndarray,
                                 position_phased: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    vertices = np.asarray(nodes, dtype=float)
    if vertices.size == 0:
        vertices = vertices.reshape(0, 2)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("nodes must be a two-dimensional array with two columns")
    if vertices.size and not (np.all(np.isfinite(vertices))
                              and np.allclose(vertices, np.round(vertices),
                                              rtol=0.0, atol=1e-12)):
        raise ValueError("nodes entries must be finite and integer valued")
    vertices = np.round(vertices).astype(int)
    n_vertices = int(vertices.shape[0])

    processed = np.asarray(node_phased, dtype=float)
    assigned = np.asarray(position_phased, dtype=float)
    if processed.ndim != 1 or processed.size != n_vertices:
        raise ValueError("node_phased must be one-dimensional of length U")
    if assigned.ndim != 1 or assigned.size < 1:
        raise ValueError("position_phased must be a non-empty one-dimensional array")
    if not np.all(np.isin(processed, (0.0, 1.0))) or not np.all(np.isin(assigned, (0.0, 1.0))):
        raise ValueError("node_phased and position_phased entries must be 0 or 1")
    processed = processed.astype(bool)
    assigned = assigned.astype(bool)
    if n_vertices and (np.any(vertices < 0) or np.any(vertices >= assigned.size)):
        raise ValueError("nodes entries must be valid SNP indices")

    move = np.zeros(3 + n_vertices, dtype=int)
    move[1] = -1
    move[2] = -1

    # A vertex can extend the assembly only while one of its two SNPs is unphased.
    extendable = np.array([not (assigned[vertices[t, 0]] and assigned[vertices[t, 1]])
                           for t in range(n_vertices)], dtype=bool)

    frontier = np.zeros(n_vertices, dtype=bool)
    for t in range(n_vertices):
        if processed[t] or not extendable[t]:
            continue
        pair = set(vertices[t].tolist())
        for other in range(n_vertices):
            if other != t and processed[other] and len(pair & set(vertices[other].tolist())) == 1:
                frontier[t] = True
                break

    if np.any(frontier):
        connectivity = {}
        for t in np.flatnonzero(frontier):
            for snp in vertices[t].tolist():
                if not assigned[snp]:
                    connectivity[int(snp)] = connectivity.get(int(snp), 0) + 1
        chosen = max(sorted(connectivity), key=lambda snp: connectivity[snp])
        move[0] = 0
        move[1] = int(chosen)
        move[3:] = frontier.astype(int)
        return move

    remaining = [t for t in range(n_vertices) if (not processed[t]) and extendable[t]]
    if remaining:
        move[0] = 1
        move[2] = int(remaining[0])
        return move

    move[0] = 2
    return move

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: a partially built assembly whose frontier offers two rival
        #     variants of different connectivity (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
nodes = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4]])
node_phased = np.array([1, 0, 0, 0, 0, 0, 0, 0])
position_phased = np.array([1, 1, 0, 0, 0])
""",
            "call": "sig(select_next_position(nodes, node_phased, position_phased), 1.0)",
            "gold_call": "sig(_oracle_select_next_position(nodes, node_phased, position_phased), 1.0)",
        },
        # --- Valid: a tie in connectivity, which must fall to the smaller SNP ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
nodes = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
node_phased = np.array([1, 0, 0, 0, 0, 0])
position_phased = np.array([1, 1, 0, 0])
""",
            "call": "sig(select_next_position(nodes, node_phased, position_phased), 1.0)",
            "gold_call": "sig(_oracle_select_next_position(nodes, node_phased, position_phased), 1.0)",
        },
        # --- Valid: a frontier in which the most connected unphased SNP is
        #     neither the smallest nor the least connected one on offer ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
nodes = np.array([[0, 1], [0, 3], [0, 9], [1, 9]])
node_phased = np.array([1, 0, 0, 0])
position_phased = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
""",
            "call": "sig(select_next_position(nodes, node_phased, position_phased), 1.0)",
            "gold_call": "sig(_oracle_select_next_position(nodes, node_phased, position_phased), 1.0)",
        },
        # --- Valid: an empty frontier with work left, which must open a new
        #     block at the first unprocessed extendable vertex ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
nodes = np.array([[0, 1], [0, 2], [1, 2], [6, 7], [6, 8], [7, 8]])
node_phased = np.array([1, 1, 1, 0, 0, 0])
position_phased = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
""",
            "call": "sig(select_next_position(nodes, node_phased, position_phased), 1.0)",
            "gold_call": "sig(_oracle_select_next_position(nodes, node_phased, position_phased), 1.0)",
        },
        # --- Boundary: nothing processed at all, the initial call of the
        #     assembly, which must also open a block ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
nodes = np.array([[0, 1], [0, 2], [1, 2]])
node_phased = np.array([0, 0, 0])
position_phased = np.array([0, 0, 0])
""",
            "call": "sig(select_next_position(nodes, node_phased, position_phased), 1.0)",
            "gold_call": "sig(_oracle_select_next_position(nodes, node_phased, position_phased), 1.0)",
        },
        # --- Boundary: every SNP assigned already, so no move remains even
        #     though unprocessed vertices are still present ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
nodes = np.array([[0, 1], [0, 2], [1, 2]])
node_phased = np.array([1, 0, 0])
position_phased = np.array([1, 1, 1])
""",
            "call": "sig(select_next_position(nodes, node_phased, position_phased), 1.0)",
            "gold_call": "sig(_oracle_select_next_position(nodes, node_phased, position_phased), 1.0)",
        },
        # --- Edge: an unphased SNP that belongs to no vertex at all, which the
        #     frontier cannot reach and which must be left alone ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
nodes = np.array([[0, 1], [0, 2], [1, 2]])
node_phased = np.array([1, 1, 1])
position_phased = np.array([1, 1, 1, 0])
""",
            "call": "sig(select_next_position(nodes, node_phased, position_phased), 1.0)",
            "gold_call": "sig(_oracle_select_next_position(nodes, node_phased, position_phased), 1.0)",
        },
        # --- Edge: a processed vertex that shares both SNPs with no other, so
        #     adjacency by exactly one shared SNP is what defines the frontier ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
nodes = np.array([[0, 3], [1, 2], [1, 4], [2, 4], [3, 5]])
node_phased = np.array([0, 1, 0, 0, 0])
position_phased = np.array([0, 1, 1, 0, 0, 0])
""",
            "call": "sig(select_next_position(nodes, node_phased, position_phased), 1.0)",
            "gold_call": "sig(_oracle_select_next_position(nodes, node_phased, position_phased), 1.0)",
        },
        # --- Invalid: a vertex naming a SNP outside the assignment vector ---
        {
            "setup": """import numpy as np
nodes = np.array([[0, 1], [0, 9]])
node_phased = np.array([0, 0])
position_phased = np.array([0, 0, 0])
def run_model():
    try:
        select_next_position(nodes, node_phased, position_phased)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_next_position(nodes, node_phased, position_phased)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a vertex list that is not two columns wide ---
        {
            "setup": """import numpy as np
nodes = np.array([[0, 1, 2], [0, 2, 3]])
node_phased = np.array([0, 0])
position_phased = np.array([0, 0, 0, 0])
def run_model():
    try:
        select_next_position(nodes, node_phased, position_phased)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_next_position(nodes, node_phased, position_phased)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a processed-vertex flag that is neither zero nor one ---
        {
            "setup": """import numpy as np
nodes = np.array([[0, 1], [0, 2]])
node_phased = np.array([0, 2])
position_phased = np.array([0, 0, 0])
def run_model():
    try:
        select_next_position(nodes, node_phased, position_phased)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_next_position(nodes, node_phased, position_phased)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
