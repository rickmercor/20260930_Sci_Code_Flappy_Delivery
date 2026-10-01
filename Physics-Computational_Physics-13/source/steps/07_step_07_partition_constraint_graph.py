"""
Recover the kinematic graph from the constraint Jacobian, cut its cycles, and group the constraints into small clusters for the block-diagonal Schur approximation.

The Schur complement of the saddle-point system is dense, because the inverse of the mechanical block couples any two constraints that touch a common body and, through chains of such couplings, effectively every constraint to every other. Forming and factorising it is out of the question at scale, so it is replaced by a block-diagonal surrogate whose blocks are groups of constraints. The grouping is not arbitrary: it is read off the topology of the mechanism. Two constraints interact at leading order exactly when they share a body, so the constraint adjacency graph is built with one vertex per constraint and an edge wherever two constraints have a body in common. Since each row of the Jacobian is nonzero only in the columns of the two bodies its joint connects, the whole topology, both the body graph and the constraint adjacency graph, is recoverable from the sparsity pattern of the Jacobian itself.

Cluster size is capped rather than left free, and the cap is small. The local Schur complement of a cluster of k constraints is a dense k by k matrix that must be formed and factorised, at a cost growing as the cube of k, and it must be applied at every Krylov iteration. Keeping k at or below a small constant keeps that cost negligible per cluster and linear in total, while still capturing the dominant local coupling; the standard setting in this framework caps clusters at six constraints. The clusters themselves are formed by a deterministic greedy pass: constraints are visited in ascending index, an unassigned constraint opens a new cluster, and unassigned neighbours in the adjacency graph are absorbed in ascending index until the cap is reached.

Block-diagonal grouping alone is blind to kinematic loops. A closed chain creates couplings between constraints that no size-limited clustering of a local adjacency graph can contain, and dropping them degrades the preconditioner exactly where mechanisms are hardest. The remedy is to identify a feedback edge set of the body graph, a minimal set of joints whose removal leaves a forest, and to treat their constraints separately: they are excluded from the ordinary clustering and reserved for the dense low-rank correction built in the next step. For a connected body graph the number of such joints equals the number of independent cycles, which is why the correction stays small. The feedback set is found by a single union-find pass over the joints in ascending order, a joint being cut precisely when both of its bodies already lie in the same component.

Returns
-------
np.ndarray of shape (m, 2), int: per constraint, its cluster index in column zero and its cut flag in column one.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def partition_constraint_graph(jacobian: np.ndarray, dofs_per_body: int,
                               max_cluster: int) -> np.ndarray:
    """Cluster the constraints and mark those on a feedback edge set.

    Parameters
    ----------
    jacobian : np.ndarray
        Constraint Jacobian of shape (m, n); every row must be nonzero in the
        columns of exactly two bodies.
    dofs_per_body : int
        Number of generalised coordinates per body (dofs_per_body >= 1); the
        column count must be an exact multiple of it.
    max_cluster : int
        Maximum number of constraints per cluster (max_cluster >= 1).

    Returns
    -------
    partition : np.ndarray
        Integer array of shape (m, 2). Column zero holds the cluster index of
        every constraint, consecutively numbered from zero; column one is one
        for constraints lying on a cut joint and zero otherwise.
    """
    return partition  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_partition_constraint_graph(jacobian: np.ndarray, dofs_per_body: int,
                                       max_cluster: int) -> np.ndarray:
    # Local imports and nested helpers keep the oracle self-contained when the
    # harness executes it in isolation.
    import numpy as np

    def _find_root(parent, node):
        """Return the representative of a node with path compression."""
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for name, value in (("dofs_per_body", dofs_per_body), ("max_cluster", max_cluster)):
        if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                and int(value) >= 1):
            raise ValueError(f"{name} must be an integer >= 1")
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("jacobian must be a non-empty 2D array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("jacobian must be finite")

    dofs_per_body = int(dofs_per_body)
    max_cluster = int(max_cluster)
    n_rows, n_cols = matrix.shape
    if n_cols % dofs_per_body != 0:
        raise ValueError("the column count of jacobian must be a multiple of dofs_per_body")
    n_bodies = n_cols // dofs_per_body

    # Every constraint row must touch exactly two bodies; that pair is its joint.
    touched = []
    for row in range(n_rows):
        bodies = np.unique(np.flatnonzero(matrix[row] != 0.0) // dofs_per_body)
        if bodies.size != 2:
            raise ValueError("every row of jacobian must touch exactly two bodies")
        touched.append((int(bodies[0]), int(bodies[1])))

    # Joints in order of first appearance, with the constraints they carry.
    joint_index = {}
    joint_rows = []
    for row, pair in enumerate(touched):
        if pair not in joint_index:
            joint_index[pair] = len(joint_rows)
            joint_rows.append([])
        joint_rows[joint_index[pair]].append(row)

    # Feedback edge set: a joint is cut when both its bodies already share a component.
    parent = list(range(n_bodies))
    cut = np.zeros(n_rows, dtype=int)
    for pair, index in sorted(joint_index.items(), key=lambda item: item[1]):
        root_i, root_j = _find_root(parent, pair[0]), _find_root(parent, pair[1])
        if root_i == root_j:
            for row in joint_rows[index]:
                cut[row] = 1
        else:
            parent[root_i] = root_j

    # Constraint adjacency: two constraints are tied when they share a body.
    body_sets = [set(pair) for pair in touched]
    neighbours = [[s for s in range(n_rows) if s != r and (body_sets[r] & body_sets[s])]
                  for r in range(n_rows)]

    # Greedy clustering in ascending index, uncut constraints first, then cut ones.
    labels = -np.ones(n_rows, dtype=int)
    next_id = 0
    for group in (np.flatnonzero(cut == 0), np.flatnonzero(cut == 1)):
        members = set(group.tolist())
        for row in group:
            if labels[row] >= 0:
                continue
            labels[row] = next_id
            size = 1
            for other in neighbours[row]:
                if size >= max_cluster:
                    break
                if other in members and labels[other] < 0:
                    labels[other] = next_id
                    size += 1
            next_id += 1

    return np.column_stack([labels, cut]).astype(int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: closed chain with chords (normal scenario) ---
        {
            "setup": """import numpy as np
n_bodies = 12
edges = [tuple(sorted((b, (b + 1) % n_bodies))) for b in range(n_bodies)]
edges += [(0, 5), (3, 9)]
rows = []
for k, (i, j) in enumerate(edges):
    for d in range(3):
        r = np.zeros(6 * n_bodies)
        r[6 * i + d] = 1.0
        r[6 * j + d] = -1.0
        rows.append(r)
jacobian = np.array(rows)
dofs_per_body, max_cluster = 6, 6
""",
            "call": "partition_constraint_graph(jacobian, dofs_per_body, max_cluster)",
            "gold_call": "_oracle_partition_constraint_graph(jacobian, dofs_per_body, max_cluster)",
        },
        # --- Valid: open chain, so no joint is ever cut ---
        {
            "setup": """import numpy as np
n_bodies = 8
edges = [(b, b + 1) for b in range(n_bodies - 1)]
rows = []
for k, (i, j) in enumerate(edges):
    for d in range(3):
        r = np.zeros(6 * n_bodies)
        r[6 * i + d] = 1.0
        r[6 * j + d] = -1.0
        rows.append(r)
jacobian = np.array(rows)
dofs_per_body, max_cluster = 6, 6
""",
            "call": "partition_constraint_graph(jacobian, dofs_per_body, max_cluster)",
            "gold_call": "_oracle_partition_constraint_graph(jacobian, dofs_per_body, max_cluster)",
        },
        # --- Boundary: cluster cap of one, so every constraint is its own cluster ---
        {
            "setup": """import numpy as np
n_bodies = 6
edges = [tuple(sorted((b, (b + 1) % n_bodies))) for b in range(n_bodies)]
rows = []
for k, (i, j) in enumerate(edges):
    for d in range(3):
        r = np.zeros(6 * n_bodies)
        r[6 * i + d] = 1.0
        r[6 * j + d] = -1.0
        rows.append(r)
jacobian = np.array(rows)
dofs_per_body, max_cluster = 6, 1
""",
            "call": "partition_constraint_graph(jacobian, dofs_per_body, max_cluster)",
            "gold_call": "_oracle_partition_constraint_graph(jacobian, dofs_per_body, max_cluster)",
        },
        # --- Edge: densely cycled graph where most joints end up cut ---
        {
            "setup": """import numpy as np
n_bodies = 5
edges = [(i, j) for i in range(n_bodies) for j in range(i + 1, n_bodies)]
rows = []
for k, (i, j) in enumerate(edges):
    r = np.zeros(3 * n_bodies)
    r[3 * i] = 1.0
    r[3 * j] = -1.0
    rows.append(r)
jacobian = np.array(rows)
dofs_per_body, max_cluster = 3, 4
""",
            "call": "partition_constraint_graph(jacobian, dofs_per_body, max_cluster)",
            "gold_call": "_oracle_partition_constraint_graph(jacobian, dofs_per_body, max_cluster)",
        },
        # --- Invalid: a row touching only one body is not a bilateral constraint ---
        {
            "setup": """import numpy as np
jacobian = np.zeros((2, 12))
jacobian[0, 0] = 1.0
jacobian[0, 6] = -1.0
jacobian[1, 1] = 1.0
def run_model():
    try:
        partition_constraint_graph(jacobian, 6, 6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_partition_constraint_graph(jacobian, 6, 6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: column count not a multiple of the body dimension ---
        {
            "setup": """import numpy as np
jacobian = np.ones((3, 7))
def run_model():
    try:
        partition_constraint_graph(jacobian, 6, 6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_partition_constraint_graph(jacobian, 6, 6)
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
