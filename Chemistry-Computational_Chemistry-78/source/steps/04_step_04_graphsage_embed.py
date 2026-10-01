"""
Evaluate the Graph Embedding Layers of the source paper.

This fixture uses hidden width 8. sage_weights stores three
layers numbered 1, 2, and 3. Each layer k provides
W_self_k and W_neigh_k with shape (8, in_features), oriented as
PyTorch Linear.weight so the maps are x @ W.T + b, and b_k with
shape (8,). in_features is 8 at every layer. Dropout and
batch-normalization from the source-paper figure are not applied
on this single-graph evaluation.

Returns
-------
return node_embeddings
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def graphsage_embed(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    sage_weights: dict,
) -> np.ndarray:
    """
    Embed the local-site nodes.

    Parameters
    ----------
    node_features : np.ndarray
        Scaled node attributes with shape (n_nodes, 8).
    edge_index : np.ndarray
        Bidirectional first-neighbor edges with shape (2, n_edges).
    sage_weights : dict
        Frozen tensors keyed W_self_k, W_neigh_k, and b_k for
        layers k = 1, 2, 3. Each W has shape (8, 8); each b has
        shape (8,).

    Returns
    -------
    np.ndarray
        Embedded node states with shape (n_nodes, 8).

    Raises
    ------
    ValueError
        If node features or edges are invalid, or weights are
        incompatible.
    """
    return node_embeddings

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_graphsage_embed(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    sage_weights: dict,
) -> np.ndarray:
    x = np.asarray(node_features, dtype=float)
    edges = np.asarray(edge_index, dtype=int)
    if x.ndim != 2 or x.shape[1] != 8:
        raise ValueError("node_features must have shape (n_nodes, 8)")
    if not np.all(np.isfinite(x)):
        raise ValueError("node_features contain non-finite values")
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] == 0:
        raise ValueError("edge_index must have shape (2, n_edges)")
    src = np.asarray(edges[0], dtype=int)
    dst = np.asarray(edges[1], dtype=int)
    n = x.shape[0]
    if np.any(src < 0) or np.any(dst < 0) or np.any(src >= n) or np.any(dst >= n):
        raise ValueError("edge indices are out of range")
    if np.any(src == dst):
        raise ValueError("self-loops are not allowed")
    for layer in range(1, 4):
        w_self = np.asarray(sage_weights[f"W_self_{layer}"], dtype=float)
        w_neigh = np.asarray(sage_weights[f"W_neigh_{layer}"], dtype=float)
        bias = np.asarray(sage_weights[f"b_{layer}"], dtype=float).reshape(-1)
        if w_self.shape != (8, x.shape[1]) or w_neigh.shape != (8, x.shape[1]):
            raise ValueError(f"GraphSAGE layer {layer} has incompatible weights")
        if bias.shape != (8,):
            raise ValueError(f"GraphSAGE layer {layer} has incompatible bias")
        agg = np.zeros_like(x)
        deg = np.zeros(n, dtype=float)
        np.add.at(agg, dst, x[src])
        np.add.at(deg, dst, 1.0)
        mean_n = np.zeros_like(x)
        active = deg > 0
        mean_n[active] = agg[active] / deg[active, None]
        x = x @ w_self.T + mean_n @ w_neigh.T + bias
        if layer < 3:
            x = np.maximum(x, 0.0)
    return x

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np

rng = np.random.default_rng(3)
sage_weights = {}
for layer in range(1, 4):
    sage_weights[f"W_self_{layer}"] = rng.normal(0.0, 0.35, size=(8, 8))
    sage_weights[f"W_neigh_{layer}"] = rng.normal(0.0, 0.35, size=(8, 8))
    sage_weights[f"b_{layer}"] = rng.normal(0.0, 0.08, size=(8,))
node_features = np.eye(8)
edge_index = np.array([[0, 1], [1, 0]])

def _copy_w(d):
    return {k: np.array(v, copy=True) for k, v in d.items()}

def run_model():
    return graphsage_embed(np.array(node_features, copy=True), np.array(edge_index, copy=True), _copy_w(sage_weights))

def run_gold():
    return _oracle_graphsage_embed(np.array(node_features, copy=True), np.array(edge_index, copy=True), _copy_w(sage_weights))
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

rng = np.random.default_rng(5)
sage_weights = {}
for layer in range(1, 4):
    sage_weights[f"W_self_{layer}"] = rng.normal(0.0, 0.35, size=(8, 8))
    sage_weights[f"W_neigh_{layer}"] = rng.normal(0.0, 0.35, size=(8, 8))
    sage_weights[f"b_{layer}"] = rng.normal(0.0, 0.08, size=(8,))
node_features = np.zeros((3, 8))
node_features[0, 0] = 1.0
node_features[2, 1] = 1.0
edge_index = np.array([[0, 1], [1, 0]])

def _copy_w(d):
    return {k: np.array(v, copy=True) for k, v in d.items()}

def run_model():
    return graphsage_embed(np.array(node_features, copy=True), np.array(edge_index, copy=True), _copy_w(sage_weights))

def run_gold():
    return _oracle_graphsage_embed(np.array(node_features, copy=True), np.array(edge_index, copy=True), _copy_w(sage_weights))
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

rng = np.random.default_rng(11)
sage_weights = {}
for layer in range(1, 4):
    sage_weights[f"W_self_{layer}"] = rng.normal(0.0, 0.35, size=(8, 8))
    sage_weights[f"W_neigh_{layer}"] = rng.normal(0.0, 0.35, size=(8, 8))
    sage_weights[f"b_{layer}"] = rng.normal(0.0, 0.08, size=(8,))
node_features = -0.4 * np.eye(8)
edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])

def _copy_w(d):
    return {k: np.array(v, copy=True) for k, v in d.items()}

def run_model():
    return graphsage_embed(np.array(node_features, copy=True), np.array(edge_index, copy=True), _copy_w(sage_weights))

def run_gold():
    return _oracle_graphsage_embed(np.array(node_features, copy=True), np.array(edge_index, copy=True), _copy_w(sage_weights))
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

rng = np.random.default_rng(1)
sage_weights = {}
for layer in range(1, 4):
    sage_weights[f"W_self_{layer}"] = rng.normal(0.0, 0.35, size=(8, 8))
    sage_weights[f"W_neigh_{layer}"] = rng.normal(0.0, 0.35, size=(8, 8))
    sage_weights[f"b_{layer}"] = rng.normal(0.0, 0.08, size=(8,))
sage_weights["W_self_1"] = sage_weights["W_self_1"][:, :4]
node_features = np.eye(8)
edge_index = np.array([[0, 1], [1, 0]])

def _copy_w(d):
    return {k: np.array(v, copy=True) for k, v in d.items()}

def run_model():
    try:
        graphsage_embed(np.array(node_features, copy=True), np.array(edge_index, copy=True), _copy_w(sage_weights))
        return 0
    except ValueError:
        return 1

def run_gold():
    try:
        _oracle_graphsage_embed(np.array(node_features, copy=True), np.array(edge_index, copy=True), _copy_w(sage_weights))
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
