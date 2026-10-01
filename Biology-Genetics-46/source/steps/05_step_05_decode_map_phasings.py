"""
Decode one phasing per vertex of the SNP line graph by maximising the joint score of the directed phasing model over each connected component.

The score of a configuration is the product of every vertex potential with every transition along an edge, and components share no edge, so each is maximised on its own evidence. Maximising over one parent at a time is not the same thing once the graph carries cycles, because the parents of a vertex share upstream factors that a per-parent maximisation counts more than once.

Returns
-------
np.ndarray of shape (U,), int: the decoded phasing index of every vertex of the SNP line graph.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def decode_map_phasings(nodes: np.ndarray, node_potentials: list,
                        edges: np.ndarray, transitions: list) -> np.ndarray:
    """Decode the maximum-a-posteriori phasing index of every vertex.

    The score of an assignment of one phasing index to every vertex is the
    product of the potential each vertex gives its own index with the
    transition each edge gives the pair of indices at its endpoints. Two
    vertices lie in the same component when the undirected graph induced by
    ``edges`` connects them, components share no edge, and the score is
    therefore maximised over each component separately; a vertex belonging to
    no edge is a component on its own. Among assignments of equal score the one
    whose tuple of state indices, read in ascending vertex order, is smallest
    lexicographically is returned.

    Parameters
    ----------
    nodes : np.ndarray
        Integer array of shape (U, 2) holding the SNP pair of each vertex, in
        ascending order of the first SNP, ties broken by the second.
    node_potentials : list
        Sequence of U one-dimensional arrays of strictly positive floats;
        entry t holds the potential of every phasing of vertex t.
    edges : np.ndarray
        Integer array of shape (E, 2) of directed edges, each row holding a
        parent vertex index strictly smaller than its child vertex index.
    transitions : list
        Sequence of E two-dimensional arrays of finite non-negative floats;
        entry e has shape (Mp, Mc) for the parent and child of edge e.

    Returns
    -------
    states : np.ndarray
        Integer array of shape (U,) holding, for each vertex, the index of its
        decoded phasing within that vertex's own potential array.

    Raises
    ------
    ValueError
        If ``nodes`` is not a two-dimensional integer-valued array with two
        columns, if ``node_potentials`` does not hold one non-empty
        one-dimensional array of finite strictly positive entries per vertex,
        if ``edges`` is not a two-dimensional integer-valued array with two
        columns whose entries are valid vertex indices with the parent index
        strictly smaller than the child index, if the number or the shape of
        the transition matrices does not match the edges they belong to, if any
        transition entry is not finite and non-negative, or if any component
        admits more than 2000000 assignments.
    """
    return states  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_decode_map_phasings(nodes: np.ndarray, node_potentials: list,
                                edges: np.ndarray, transitions: list) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import itertools

    import numpy as np

    vertices = np.asarray(nodes, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("nodes must be a two-dimensional array with two columns")
    if vertices.size and not (np.all(np.isfinite(vertices))
                              and np.allclose(vertices, np.round(vertices),
                                              rtol=0.0, atol=1e-12)):
        raise ValueError("nodes entries must be finite and integer valued")
    n_vertices = int(vertices.shape[0])

    potentials = [np.asarray(p, dtype=float) for p in node_potentials]
    if len(potentials) != n_vertices:
        raise ValueError("node_potentials must hold one array per vertex")
    for p in potentials:
        if p.ndim != 1 or p.size < 1:
            raise ValueError("each node potential must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(p)) or np.any(p <= 0.0):
            raise ValueError("node potential entries must be finite and strictly positive")

    links = np.asarray(edges, dtype=float)
    if links.size == 0:
        links = links.reshape(0, 2)
    if links.ndim != 2 or links.shape[1] != 2:
        raise ValueError("edges must be a two-dimensional array with two columns")
    if links.size and not (np.all(np.isfinite(links))
                           and np.allclose(links, np.round(links), rtol=0.0, atol=1e-12)):
        raise ValueError("edges entries must be finite and integer valued")
    links = np.round(links).astype(int)
    if len(transitions) != links.shape[0]:
        raise ValueError("transitions must hold one matrix per edge")

    matrices = []
    for e in range(links.shape[0]):
        parent, child = int(links[e, 0]), int(links[e, 1])
        if not (0 <= parent < n_vertices and 0 <= child < n_vertices):
            raise ValueError("edges entries must be valid vertex indices")
        if parent >= child:
            raise ValueError("each edge must run from a smaller to a larger vertex index")
        matrix = np.asarray(transitions[e], dtype=float)
        if matrix.shape != (potentials[parent].size, potentials[child].size):
            raise ValueError("each transition matrix must match the states of its edge")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
            raise ValueError("transition entries must be finite and non-negative")
        matrices.append(matrix)

    # Connected components of the undirected graph the edges induce.
    neighbours = [[] for _ in range(n_vertices)]
    for e in range(links.shape[0]):
        neighbours[int(links[e, 0])].append(int(links[e, 1]))
        neighbours[int(links[e, 1])].append(int(links[e, 0]))
    component_of = -np.ones(n_vertices, dtype=int)
    components = []
    for seed in range(n_vertices):
        if component_of[seed] >= 0:
            continue
        stack, members = [seed], []
        component_of[seed] = len(components)
        while stack:
            v = stack.pop()
            members.append(v)
            for w in neighbours[v]:
                if component_of[w] < 0:
                    component_of[w] = len(components)
                    stack.append(w)
        components.append(sorted(members))

    states = np.zeros(n_vertices, dtype=int)
    for members in components:
        sizes = [int(potentials[t].size) for t in members]
        volume = 1
        for s in sizes:
            volume *= s
        if volume > 2000000:
            raise ValueError("a component admits more than 2000000 assignments")
        inside = [e for e in range(links.shape[0])
                  if component_of[int(links[e, 0])] == component_of[members[0]]]
        slot = {t: i for i, t in enumerate(members)}
        best_score, best_assignment = -1.0, None
        # itertools.product enumerates in ascending lexicographic order, so a
        # strict improvement test keeps the smallest maximiser.
        for assignment in itertools.product(*[range(s) for s in sizes]):
            score = 1.0
            for t, index in zip(members, assignment):
                score *= float(potentials[t][index])
            for e in inside:
                score *= float(matrices[e][assignment[slot[int(links[e, 0])]],
                                           assignment[slot[int(links[e, 1])]]])
            if score > best_score:
                best_score, best_assignment = score, assignment
        for t, index in zip(members, best_assignment):
            states[t] = int(index)

    return states

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: a three-vertex cycle over three SNPs, the smallest graph on
        #     which maximising one parent at a time is not exact (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
nodes = np.array([[0, 1], [0, 2], [1, 2]])
node_potentials = [np.array([0.4, 1.2, 0.9]),
                   np.array([1.1, 0.3, 2.2]),
                   np.array([0.7, 1.9, 0.5])]
edges = np.array([[0, 1], [0, 2], [1, 2]])
transitions = [np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.1, 0.8]]),
               np.array([[0.5, 0.4, 0.1], [0.3, 0.3, 0.4], [0.2, 0.2, 0.6]]),
               np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])]
""",
            "call": "sig(decode_map_phasings(nodes, node_potentials, edges, transitions), 1.0)",
            "gold_call": "sig(_oracle_decode_map_phasings(nodes, node_potentials, edges, transitions), 1.0)",
        },
        # --- Valid: a cycle in which the two parents of the last vertex pull it
        #     in opposite directions, so only the joint score decides ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
nodes = np.array([[0, 1], [0, 2], [1, 2]])
node_potentials = [np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0])]
edges = np.array([[0, 1], [0, 2], [1, 2]])
transitions = [np.array([[0.05, 0.95], [0.9, 0.1]]),
               np.array([[0.8, 0.2], [0.15, 0.85]]),
               np.array([[0.3, 0.7], [0.6, 0.4]])]
""",
            "call": "sig(decode_map_phasings(nodes, node_potentials, edges, transitions), 1.0)",
            "gold_call": "sig(_oracle_decode_map_phasings(nodes, node_potentials, edges, transitions), 1.0)",
        },
        # --- Valid: a chain whose potentials are flat everywhere, so the decoded
        #     configuration is fixed by the transitions alone ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
nodes = np.array([[0, 1], [1, 2], [2, 3]])
node_potentials = [np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0])]
edges = np.array([[0, 1], [1, 2]])
transitions = [np.array([[0.5, 0.5], [0.5, 0.5]]),
               np.array([[0.1, 0.9], [0.5, 0.5]])]
""",
            "call": "sig(decode_map_phasings(nodes, node_potentials, edges, transitions), 1.0)",
            "gold_call": "sig(_oracle_decode_map_phasings(nodes, node_potentials, edges, transitions), 1.0)",
        },
        # --- Valid: two disconnected components, each of which must be maximised
        #     on its own evidence rather than jointly ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
nodes = np.array([[0, 1], [0, 2], [1, 2], [5, 6], [5, 7], [6, 7]])
node_potentials = [np.array([0.4, 1.2]), np.array([1.1, 0.3]), np.array([0.7, 1.9]),
                   np.array([2.0, 0.5]), np.array([0.6, 0.6]), np.array([0.2, 3.0])]
edges = np.array([[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5]])
transitions = [np.array([[0.7, 0.3], [0.4, 0.6]]),
               np.array([[0.55, 0.45], [0.25, 0.75]]),
               np.array([[0.9, 0.1], [0.2, 0.8]]),
               np.array([[0.35, 0.65], [0.8, 0.2]]),
               np.array([[0.5, 0.5], [0.5, 0.5]]),
               np.array([[0.15, 0.85], [0.6, 0.4]])]
""",
            "call": "sig(decode_map_phasings(nodes, node_potentials, edges, transitions), 1.0)",
            "gold_call": "sig(_oracle_decode_map_phasings(nodes, node_potentials, edges, transitions), 1.0)",
        },
        # --- Boundary: a graph with no edges at all, where every vertex is its
        #     own component and falls back to the arg max of its potential ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
nodes = np.array([[0, 1], [4, 5], [8, 9]])
node_potentials = [np.array([0.4, 1.2, 0.9]),
                   np.array([2.5, 0.3]),
                   np.array([0.1, 0.1, 0.1, 4.0])]
edges = np.zeros((0, 2), dtype=int)
transitions = []
""",
            "call": "sig(decode_map_phasings(nodes, node_potentials, edges, transitions), 1.0)",
            "gold_call": "sig(_oracle_decode_map_phasings(nodes, node_potentials, edges, transitions), 1.0)",
        },
        # --- Edge: a component every one of whose assignments scores zero, which
        #     must still return the smallest state tuple rather than fail ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
nodes = np.array([[0, 1], [0, 2], [1, 2]])
node_potentials = [np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0])]
edges = np.array([[0, 1], [0, 2], [1, 2]])
transitions = [np.array([[0.0, 1.0], [1.0, 0.0]]),
               np.array([[1.0, 0.0], [0.0, 1.0]]),
               np.array([[1.0, 0.0], [0.0, 1.0]])]
""",
            "call": "sig(decode_map_phasings(nodes, node_potentials, edges, transitions), 1.0)",
            "gold_call": "sig(_oracle_decode_map_phasings(nodes, node_potentials, edges, transitions), 1.0)",
        },
        # --- Edge: a transition carrying exact zeros that forbid some pairings
        #     without emptying the component ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
nodes = np.array([[0, 1], [0, 2], [1, 2]])
node_potentials = [np.array([1.0, 3.0]), np.array([2.0, 1.0]), np.array([1.0, 1.0])]
edges = np.array([[0, 1], [1, 2]])
transitions = [np.array([[0.0, 1.0], [1.0, 0.0]]),
               np.array([[1.0, 0.0], [0.0, 1.0]])]
""",
            "call": "sig(decode_map_phasings(nodes, node_potentials, edges, transitions), 1.0)",
            "gold_call": "sig(_oracle_decode_map_phasings(nodes, node_potentials, edges, transitions), 1.0)",
        },
        # --- Invalid: an edge running backwards along the vertex order ---
        {
            "setup": """import numpy as np
nodes = np.array([[0, 1], [0, 2]])
node_potentials = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
edges = np.array([[1, 0]])
transitions = [np.array([[0.5, 0.5], [0.5, 0.5]])]
def run_model():
    try:
        decode_map_phasings(nodes, node_potentials, edges, transitions)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_decode_map_phasings(nodes, node_potentials, edges, transitions)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a transition matrix whose shape does not match its edge ---
        {
            "setup": """import numpy as np
nodes = np.array([[0, 1], [0, 2]])
node_potentials = [np.array([1.0, 1.0]), np.array([1.0, 1.0, 1.0])]
edges = np.array([[0, 1]])
transitions = [np.array([[0.5, 0.5], [0.5, 0.5]])]
def run_model():
    try:
        decode_map_phasings(nodes, node_potentials, edges, transitions)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_decode_map_phasings(nodes, node_potentials, edges, transitions)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a vanishing node potential, which no vertex of a graph
        #     built from covering fragments can carry ---
        {
            "setup": """import numpy as np
nodes = np.array([[0, 1], [0, 2]])
node_potentials = [np.array([1.0, 0.0]), np.array([1.0, 1.0])]
edges = np.array([[0, 1]])
transitions = [np.array([[0.5, 0.5], [0.5, 0.5]])]
def run_model():
    try:
        decode_map_phasings(nodes, node_potentials, edges, transitions)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_decode_map_phasings(nodes, node_potentials, edges, transitions)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative transition entry ---
        {
            "setup": """import numpy as np
nodes = np.array([[0, 1], [0, 2]])
node_potentials = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
edges = np.array([[0, 1]])
transitions = [np.array([[0.5, -0.5], [0.5, 0.5]])]
def run_model():
    try:
        decode_map_phasings(nodes, node_potentials, edges, transitions)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_decode_map_phasings(nodes, node_potentials, edges, transitions)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: an edge array that is not two columns wide ---
        {
            "setup": """import numpy as np
nodes = np.array([[0, 1], [0, 2]])
node_potentials = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
edges = np.array([[0, 1, 1]])
transitions = [np.array([[0.5, 0.5], [0.5, 0.5]])]
def run_model():
    try:
        decode_map_phasings(nodes, node_potentials, edges, transitions)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_decode_map_phasings(nodes, node_potentials, edges, transitions)
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
