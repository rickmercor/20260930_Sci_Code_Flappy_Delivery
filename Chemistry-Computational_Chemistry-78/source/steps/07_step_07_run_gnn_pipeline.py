"""
Evaluate the predicted reaction energy on the
supplied local adsorption environment.

This is a paper-inspired synthetic GraphSAGE fixture, not the
published checkpoint. Do not run a new DFT job. Do not train or
refit a model.

sage_weights uses the same keys as graphsage_embed: W_self_k,
W_neigh_k, and b_k for layers k = 1, 2, 3. Compose the earlier
stages of this problem. Each stage's return is an input to a
later stage.

Returns
-------
return energy
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_gnn_pipeline(
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    bonded_adsorbate: np.ndarray,
    is_surface: np.ndarray,
    occupancies: np.ndarray,
    work_function: float,
    homo_lumo: float,
    step: float,
    feature_scales: np.ndarray,
    sage_weights: dict,
    linear_weights: dict,
) -> float:
    """
    Execute the complete localized-site GNN pipeline.

    Returns
    -------
    float
        Predicted reaction energy in eV.

    Raises
    ------
    ValueError
        If any input array is invalid or a pipeline stage returns a
        non-finite energy.
    """
    return energy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_gnn_pipeline(
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    bonded_adsorbate: np.ndarray,
    is_surface: np.ndarray,
    occupancies: np.ndarray,
    work_function: float,
    homo_lumo: float,
    step: float,
    feature_scales: np.ndarray,
    sage_weights: dict,
    linear_weights: dict,
) -> float:
    """Deterministic reference implementation for the complete GNN pipeline."""

    # ============================================================
    # Step 01: Local adsorption site
    # ============================================================
    node_index = _oracle_select_local_adsorption_site(
        atomic_numbers,
        coordinates,
        bonded_adsorbate,
    )
    node_index = np.asarray(node_index, dtype=int).reshape(-1)
    n_site = node_index.size
    n_atoms = np.asarray(atomic_numbers).reshape(-1).size
    if n_site < 2 or n_site > n_atoms:
        raise ValueError("local site must be a proper subset or the full atom list")
    if not np.all(np.isin(np.asarray(bonded_adsorbate, dtype=int), node_index)):
        raise ValueError("bonded adsorbate atoms must remain in the local site")

    # ============================================================
    # Step 02: Eight-channel node attributes
    # ============================================================
    node_features = _oracle_construct_node_attributes(
        atomic_numbers,
        bonded_adsorbate,
        is_surface,
        occupancies,
        work_function,
        homo_lumo,
        step,
        feature_scales,
        node_index,
    )
    node_features = np.asarray(node_features, dtype=float)
    if node_features.shape != (n_site, 8):
        raise ValueError("node features must have shape (n_site, 8)")
    if not np.all(np.isfinite(node_features)):
        raise ValueError("node features contain non-finite values")

    # ============================================================
    # Step 03: ASE first-neighbor graph
    # ============================================================
    edge_index = _oracle_build_ase_connectivity(
        atomic_numbers,
        coordinates,
        node_index,
    )
    edge_index = np.asarray(edge_index, dtype=int)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.shape[1] == 0:
        raise ValueError("edge_index must have shape (2, n_edges)")
    if np.any(edge_index < 0) or np.any(edge_index >= n_site):
        raise ValueError("edge endpoints must use local-site numbering")
    if np.any(edge_index[0] == edge_index[1]):
        raise ValueError("self-loops are not allowed")

    # ============================================================
    # Step 04: GraphSAGE embedding
    # ============================================================
    node_embeddings = _oracle_graphsage_embed(
        node_features,
        edge_index,
        sage_weights,
    )
    node_embeddings = np.asarray(node_embeddings, dtype=float)
    if node_embeddings.shape != (n_site, 8):
        raise ValueError("GraphSAGE embeddings must have shape (n_site, 8)")
    if not np.all(np.isfinite(node_embeddings)):
        raise ValueError("GraphSAGE embeddings contain non-finite values")

    # ============================================================
    # Step 05: Graph-level pooling
    # ============================================================
    graph_embedding = _oracle_pool_graph_embedding(node_embeddings)
    graph_embedding = np.asarray(graph_embedding, dtype=float).reshape(-1)
    if graph_embedding.shape != (8,):
        raise ValueError("pooled embedding must have length 8")
    if not np.all(np.isfinite(graph_embedding)):
        raise ValueError("pooled embedding contains non-finite values")
    if np.any(graph_embedding < np.max(node_embeddings, axis=0) - 1e-12):
        raise ValueError("pooled embedding must be the per-channel node maximum")

    # ============================================================
    # Step 06: Linear reaction-energy head
    # ============================================================
    energy = _oracle_linear_energy_head(graph_embedding, linear_weights)
    energy = float(np.asarray(energy).reshape(-1)[0])
    if not np.isfinite(energy):
        raise ValueError("predicted reaction energy must be finite")

    return energy

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np

atomic_numbers = np.array([28, 28, 31, 28, 8, 6, 8, 1, 28], dtype=int)
coordinates = np.array([
    [0.00, 0.00, 0.00],
    [2.50, 0.00, 0.00],
    [1.25, 2.16, 0.00],
    [1.25, 0.72, -2.05],
    [0.15, 0.10, 1.82],
    [1.25, 0.10, 2.48],
    [2.35, 0.10, 1.82],
    [1.25, 0.10, 3.56],
    [8.50, 8.50, 8.50],
], dtype=float)
bonded_adsorbate = np.array([4, 6], dtype=int)
is_surface = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1], dtype=int)
occupancies = np.array([
    [0.62, 0.18, 8.41, 0.00],
    [0.58, 0.22, 8.38, 0.00],
    [1.82, 0.95, 10.00, 0.00],
    [0.71, 0.08, 8.55, 0.00],
    [1.84, 4.21, 0.00, 0.00],
    [1.58, 2.36, 0.00, 0.00],
    [1.79, 4.28, 0.00, 0.00],
    [0.98, 0.00, 0.00, 0.00],
    [0.65, 0.15, 8.40, 0.00],
], dtype=float)
work_function = 5.14
homo_lumo = 8.37
step = 1.0
feature_scales = np.array([31.0, 1.32, 3.44, 2.00, 4.50, 10.00, 8.37, 1.00])
rng = np.random.default_rng(20260309)
sage_weights = {}
for layer in range(1, 4):
    sage_weights[f"W_self_{layer}"] = rng.normal(0.0, 0.35, size=(8, 8))
    sage_weights[f"W_neigh_{layer}"] = rng.normal(0.0, 0.35, size=(8, 8))
    sage_weights[f"b_{layer}"] = rng.normal(0.0, 0.08, size=(8,))
rng_lin = np.random.default_rng(20260309 + 17)
linear_weights = {
    "W": rng_lin.normal(0.0, 0.35, size=(1, 8)),
    "b": rng_lin.normal(0.0, 0.08, size=(1,)),
}

def _copy_w(d):
    return {k: np.array(v, copy=True) for k, v in d.items()}

def run_model():
    return run_gnn_pipeline(
        atomic_numbers.copy(), coordinates.copy(), bonded_adsorbate.copy(),
        is_surface.copy(), occupancies.copy(), work_function, homo_lumo, step,
        feature_scales.copy(), _copy_w(sage_weights), _copy_w(linear_weights),
    )

def run_gold():
    return _oracle_run_gnn_pipeline(
        atomic_numbers.copy(), coordinates.copy(), bonded_adsorbate.copy(),
        is_surface.copy(), occupancies.copy(), work_function, homo_lumo, step,
        feature_scales.copy(), _copy_w(sage_weights), _copy_w(linear_weights),
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([28, 28, 31, 28, 8, 6, 8, 1, 28], dtype=int)
coordinates = np.array([
    [0.00, 0.00, 0.00],
    [2.50, 0.00, 0.00],
    [1.25, 2.16, 0.00],
    [1.25, 0.72, -2.05],
    [0.15, 0.10, 1.82],
    [1.25, 0.10, 2.48],
    [2.35, 0.10, 1.82],
    [1.25, 0.10, 3.56],
    [8.50, 8.50, 8.50],
], dtype=float)
bonded_adsorbate = np.array([4, 6], dtype=int)
is_surface = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1], dtype=int)
occupancies = np.array([
    [0.62, 0.18, 8.41, 0.00],
    [0.58, 0.22, 8.38, 0.00],
    [1.82, 0.95, 10.00, 0.00],
    [0.71, 0.08, 8.55, 0.00],
    [1.84, 4.21, 0.00, 0.00],
    [1.58, 2.36, 0.00, 0.00],
    [1.79, 4.28, 0.00, 0.00],
    [0.98, 0.00, 0.00, 0.00],
    [0.65, 0.15, 8.40, 0.00],
], dtype=float)
work_function = 5.14
homo_lumo = 8.37
step = 1.0
feature_scales = np.array([31.0, 1.32, 3.44, 2.00, 4.50, 10.00, 8.37, 1.00])
rng = np.random.default_rng(20260309)
sage_weights = {}
for layer in range(1, 4):
    sage_weights[f"W_self_{layer}"] = rng.normal(0.0, 0.35, size=(8, 8))
    sage_weights[f"W_neigh_{layer}"] = rng.normal(0.0, 0.35, size=(8, 8))
    sage_weights[f"b_{layer}"] = rng.normal(0.0, 0.08, size=(8,))
rng_lin = np.random.default_rng(20260309 + 17)
linear_weights = {
    "W": rng_lin.normal(0.0, 0.35, size=(1, 8)),
    "b": rng_lin.normal(0.0, 0.08, size=(1,)),
}
occupancies = occupancies.copy()
occupancies[2, 2] -= 1.5

def _copy_w(d):
    return {k: np.array(v, copy=True) for k, v in d.items()}

def run_model():
    return run_gnn_pipeline(
        atomic_numbers.copy(), coordinates.copy(), bonded_adsorbate.copy(),
        is_surface.copy(), occupancies.copy(), work_function, homo_lumo, step,
        feature_scales.copy(), _copy_w(sage_weights), _copy_w(linear_weights),
    )

def run_gold():
    return _oracle_run_gnn_pipeline(
        atomic_numbers.copy(), coordinates.copy(), bonded_adsorbate.copy(),
        is_surface.copy(), occupancies.copy(), work_function, homo_lumo, step,
        feature_scales.copy(), _copy_w(sage_weights), _copy_w(linear_weights),
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([28, 28, 31, 28, 8, 6, 8, 1, 28], dtype=int)
coordinates = np.array([
    [0.00, 0.00, 0.00],
    [2.50, 0.00, 0.00],
    [1.25, 2.16, 0.00],
    [1.25, 0.72, -2.05],
    [0.15, 0.10, 1.82],
    [1.25, 0.10, 2.48],
    [2.35, 0.10, 1.82],
    [1.25, 0.10, 3.56],
    [8.50, 8.50, 8.50],
], dtype=float)
bonded_adsorbate = np.array([4, 6], dtype=int)
is_surface = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1], dtype=int)
occupancies = np.array([
    [0.62, 0.18, 8.41, 0.00],
    [0.58, 0.22, 8.38, 0.00],
    [1.82, 0.95, 10.00, 0.00],
    [0.71, 0.08, 8.55, 0.00],
    [1.84, 4.21, 0.00, 0.00],
    [1.58, 2.36, 0.00, 0.00],
    [1.79, 4.28, 0.00, 0.00],
    [0.98, 0.00, 0.00, 0.00],
    [0.65, 0.15, 8.40, 0.00],
], dtype=float)
work_function = 5.14
homo_lumo = 8.37
step = 1.0
feature_scales = np.array([31.0, 1.32, 3.44, 2.00, 4.50, 10.00, 8.37, 1.00])
rng = np.random.default_rng(20260309)
sage_weights = {}
for layer in range(1, 4):
    sage_weights[f"W_self_{layer}"] = rng.normal(0.0, 0.35, size=(8, 8))
    sage_weights[f"W_neigh_{layer}"] = rng.normal(0.0, 0.35, size=(8, 8))
    sage_weights[f"b_{layer}"] = rng.normal(0.0, 0.08, size=(8,))
rng_lin = np.random.default_rng(20260309 + 17)
linear_weights = {
    "W": rng_lin.normal(0.0, 0.35, size=(1, 8)),
    "b": rng_lin.normal(0.0, 0.08, size=(1,)),
}
coordinates = coordinates.copy()
coordinates[8] = np.array([12.0, -9.0, 7.0])
occupancies = occupancies.copy()
occupancies[8, 0] = 1.7

def _copy_w(d):
    return {k: np.array(v, copy=True) for k, v in d.items()}

def run_model():
    return run_gnn_pipeline(
        atomic_numbers.copy(), coordinates.copy(), bonded_adsorbate.copy(),
        is_surface.copy(), occupancies.copy(), work_function, homo_lumo, step,
        feature_scales.copy(), _copy_w(sage_weights), _copy_w(linear_weights),
    )

def run_gold():
    return _oracle_run_gnn_pipeline(
        atomic_numbers.copy(), coordinates.copy(), bonded_adsorbate.copy(),
        is_surface.copy(), occupancies.copy(), work_function, homo_lumo, step,
        feature_scales.copy(), _copy_w(sage_weights), _copy_w(linear_weights),
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np

atomic_numbers = np.array([28, 28, 31, 28, 8, 6, 8, 1, 28], dtype=int)
coordinates = np.array([
    [0.00, 0.00, 0.00],
    [2.50, 0.00, 0.00],
    [1.25, 2.16, 0.00],
    [1.25, 0.72, -2.05],
    [0.15, 0.10, 1.82],
    [1.25, 0.10, 2.48],
    [2.35, 0.10, 1.82],
    [1.25, 0.10, 3.56],
    [8.50, 8.50, 8.50],
], dtype=float)
bonded_adsorbate = np.array([4, 6], dtype=int)
is_surface = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1], dtype=int)
occupancies = np.array([
    [0.62, 0.18, 8.41, 0.00],
    [0.58, 0.22, 8.38, 0.00],
    [1.82, 0.95, 10.00, 0.00],
    [0.71, 0.08, 8.55, 0.00],
    [1.84, 4.21, 0.00, 0.00],
    [1.58, 2.36, 0.00, 0.00],
    [1.79, 4.28, 0.00, 0.00],
    [0.98, 0.00, 0.00, 0.00],
    [0.65, 0.15, 8.40, 0.00],
], dtype=float)
work_function = 5.14
homo_lumo = 8.37
step = 1.0
feature_scales = np.array([31.0, 1.32, 3.44, 2.00, 4.50, 10.00, 8.37, 1.00])
rng = np.random.default_rng(20260309)
sage_weights = {}
for layer in range(1, 4):
    sage_weights[f"W_self_{layer}"] = rng.normal(0.0, 0.35, size=(8, 8))
    sage_weights[f"W_neigh_{layer}"] = rng.normal(0.0, 0.35, size=(8, 8))
    sage_weights[f"b_{layer}"] = rng.normal(0.0, 0.08, size=(8,))
rng_lin = np.random.default_rng(20260309 + 17)
linear_weights = {
    "W": rng_lin.normal(0.0, 0.35, size=(1, 8)),
    "b": rng_lin.normal(0.0, 0.08, size=(1,)),
}
coords = coordinates.copy()
coords[0] = coords[1]

def run_model():
    try:
        run_gnn_pipeline(
            atomic_numbers.copy(), coords.copy(), bonded_adsorbate.copy(), is_surface.copy(),
            occupancies.copy(), work_function, homo_lumo, step, feature_scales.copy(),
            {k: np.array(v, copy=True) for k, v in sage_weights.items()},
            {k: np.array(v, copy=True) for k, v in linear_weights.items()},
        )
        return 0
    except ValueError:
        return 1

def run_gold():
    try:
        _oracle_run_gnn_pipeline(
            atomic_numbers.copy(), coords.copy(), bonded_adsorbate.copy(), is_surface.copy(),
            occupancies.copy(), work_function, homo_lumo, step, feature_scales.copy(),
            {k: np.array(v, copy=True) for k, v in sage_weights.items()},
            {k: np.array(v, copy=True) for k, v in linear_weights.items()},
        )
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
