"""
Reduce the variable-sized node embedding to a graph-level vector.

After message passing the graph still has one row per atom. The
result is a fixed-length vector with one entry per hidden channel.

Returns
-------
return graph_embedding
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def pool_graph_embedding(node_embeddings: np.ndarray) -> np.ndarray:
    """
    Reduce the node states to one graph-level vector.

    Parameters
    ----------
    node_embeddings : np.ndarray
        GraphSAGE node states with shape (n_nodes, 8).

    Returns
    -------
    np.ndarray
        Pooled embedding with shape (8,).

    Raises
    ------
    ValueError
        If node embeddings are empty, are not (n_nodes, 8), or contain
        non-finite values.
    """
    return graph_embedding

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pool_graph_embedding(node_embeddings: np.ndarray) -> np.ndarray:
    hidden = 8
    h = np.asarray(node_embeddings, dtype=float)
    if h.ndim != 2 or h.shape[1] != hidden:
        raise ValueError("node_embeddings must have shape (n_nodes, 8)")
    if h.shape[0] == 0:
        raise ValueError("node_embeddings is empty")
    if not np.all(np.isfinite(h)):
        raise ValueError("node_embeddings contain non-finite values")
    return np.max(h, axis=0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np

node_embeddings = np.array([
    [0.2, -0.4, 0.1, 0.0, 0.3, -1.0, 0.5, 0.2],
    [0.1,  0.3, 0.4, -0.2, 0.0,  0.2, 0.1, 0.7],
])

def run_model():
    return pool_graph_embedding(np.array(node_embeddings, copy=True))

def run_gold():
    return _oracle_pool_graph_embedding(np.array(node_embeddings, copy=True))
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

node_embeddings = np.array([
    [-0.4, -0.9, -0.2, -0.3, -0.5, -0.1, -0.8, -0.6],
    [-0.2, -0.1, -0.7, -0.4, -0.6, -0.3, -0.2, -0.5],
])

def run_model():
    return pool_graph_embedding(np.array(node_embeddings, copy=True))

def run_gold():
    return _oracle_pool_graph_embedding(np.array(node_embeddings, copy=True))
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

node_embeddings = np.array([[
    0.11, -0.22, 0.33, -0.44, 0.55, -0.66, 0.77, -0.88
]])

def run_model():
    return pool_graph_embedding(np.array(node_embeddings, copy=True))

def run_gold():
    return _oracle_pool_graph_embedding(np.array(node_embeddings, copy=True))
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

node_embeddings = np.ones((2, 8))
node_embeddings[0, 3] = np.nan

def run_model():
    try:
        pool_graph_embedding(np.array(node_embeddings, copy=True))
        return 0
    except ValueError:
        return 1

def run_gold():
    try:
        _oracle_pool_graph_embedding(np.array(node_embeddings, copy=True))
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
