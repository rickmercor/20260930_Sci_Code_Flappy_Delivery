#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def select_local_adsorption_site(
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    bonded_adsorbate: np.ndarray,
) -> np.ndarray:
    cutoff = 4.5
    allowed_z = np.array(
        [1, 6, 7, 8, 13, 26, 28, 29, 30, 31, 47, 78, 79], dtype=int
    )
    z = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    r = np.asarray(coordinates, dtype=float)
    bonded = np.asarray(bonded_adsorbate, dtype=int).reshape(-1)
    if z.size == 0:
        raise ValueError("at least one atom is required")
    if r.ndim != 2 or r.shape[1] != 3 or r.shape[0] != z.size:
        raise ValueError("coordinates must have shape (n_atoms, 3)")
    if not np.all(np.isfinite(r)):
        raise ValueError("coordinates contain non-finite values")
    if not np.all(np.isin(z, allowed_z)):
        raise ValueError("atomic numbers must belong to the paper element set")
    if bonded.size == 0:
        raise ValueError("at least one bonded adsorbate atom is required")
    if np.any(bonded < 0) or np.any(bonded >= z.size):
        raise ValueError("bonded adsorbate indices are out of range")
    if np.unique(bonded).size != bonded.size:
        raise ValueError("bonded adsorbate indices must be unique")
    pair = np.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
    if np.any(pair[np.triu_indices(z.size, 1)] < 1e-8):
        raise ValueError("distinct atoms must not coincide")
    keep = np.zeros(z.size, dtype=bool)
    keep[bonded] = True
    for idx in bonded:
        keep |= pair[:, idx] <= cutoff
    selected = np.flatnonzero(keep)
    if selected.size < 2:
        raise ValueError("local adsorption site must contain at least two atoms")
    return selected

import numpy as np


def construct_node_attributes(
    atomic_numbers: np.ndarray,
    bonded_adsorbate: np.ndarray,
    is_surface: np.ndarray,
    occupancies: np.ndarray,
    work_function: float,
    homo_lumo: float,
    step: float,
    feature_scales: np.ndarray,
    node_index: np.ndarray,
) -> np.ndarray:
    n_features = 8
    covalent_radii = {
        1: 0.31,
        6: 0.76,
        7: 0.71,
        8: 0.66,
        13: 1.21,
        26: 1.32,
        28: 1.24,
        29: 1.32,
        30: 1.22,
        31: 1.22,
        47: 1.45,
        78: 1.36,
        79: 1.36,
    }
    pauling_en = {
        1: 2.20,
        6: 2.55,
        7: 3.04,
        8: 3.44,
        13: 1.61,
        26: 1.83,
        28: 1.91,
        29: 1.90,
        30: 1.65,
        31: 1.81,
        47: 1.93,
        78: 2.28,
        79: 2.54,
    }
    z = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    bonded = np.asarray(bonded_adsorbate, dtype=int).reshape(-1)
    surface = np.asarray(is_surface, dtype=int).reshape(-1)
    occ = np.asarray(occupancies, dtype=float)
    node_index = np.asarray(node_index, dtype=int).reshape(-1)
    scales = np.asarray(feature_scales, dtype=float).reshape(-1)
    if z.size == 0:
        raise ValueError("at least one atom is required")
    if occ.ndim != 2 or occ.shape != (z.size, 4):
        raise ValueError("occupancies must have shape (n_atoms, 4)")
    if surface.size != z.size:
        raise ValueError("is_surface must have one entry per atom")
    if not np.all(np.isin(surface, (0, 1))):
        raise ValueError("is_surface must be a 0/1 mask")
    if node_index.size == 0:
        raise ValueError("node_index is empty")
    if np.any(node_index < 0) or np.any(node_index >= z.size):
        raise ValueError("node_index is out of range")
    if np.unique(node_index).size != node_index.size:
        raise ValueError("node_index must be unique")
    if not np.all(np.isin(bonded, node_index)):
        raise ValueError("bonded adsorbate atoms must remain in the local site")
    if np.any(surface[bonded] != 0):
        raise ValueError("bonded adsorbate atoms must not be surface atoms")
    if not np.isfinite(work_function) or not np.isfinite(homo_lumo) or not np.isfinite(step):
        raise ValueError("scalar surface/adsorbate descriptors must be finite")
    if scales.shape != (n_features,):
        raise ValueError("feature_scales must have length 8")
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("feature_scales must be positive and finite")
    if not np.all(np.isfinite(occ)):
        raise ValueError("occupancies contain non-finite values")
    if np.any(occ[node_index] < 0):
        raise ValueError("orbital occupancies must be non-negative")
    for zi in z[node_index]:
        if int(zi) not in covalent_radii:
            raise ValueError("atomic numbers must belong to the paper element set")

    sel = node_index
    z_sel = z[sel]
    radius = np.array([covalent_radii[int(zi)] for zi in z_sel], dtype=float)
    chi = np.array([pauling_en[int(zi)] for zi in z_sel], dtype=float)
    site_electronic = np.where(surface[sel] == 1, float(work_function), float(homo_lumo))
    raw = np.column_stack(
        [
            z_sel.astype(float),
            radius,
            chi,
            occ[sel, 0],
            occ[sel, 1],
            occ[sel, 2],
            site_electronic,
            np.full(sel.size, float(step)),
        ]
    )
    return raw / scales

import numpy as np


def build_ase_connectivity(
    atomic_numbers: np.ndarray,
    coordinates: np.ndarray,
    node_index: np.ndarray,
) -> np.ndarray:
    ase_mult = 1.1
    covalent_radii = {
        1: 0.31,
        6: 0.76,
        7: 0.71,
        8: 0.66,
        13: 1.21,
        26: 1.32,
        28: 1.24,
        29: 1.32,
        30: 1.22,
        31: 1.22,
        47: 1.45,
        78: 1.36,
        79: 1.36,
    }
    z = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    r = np.asarray(coordinates, dtype=float)
    node_index = np.asarray(node_index, dtype=int).reshape(-1)
    if r.ndim != 2 or r.shape[1] != 3 or r.shape[0] != z.size:
        raise ValueError("coordinates must have shape (n_atoms, 3)")
    if not np.all(np.isfinite(r)):
        raise ValueError("coordinates contain non-finite values")
    if node_index.size < 2:
        raise ValueError("at least two local-site atoms are required")
    if np.any(node_index < 0) or np.any(node_index >= z.size):
        raise ValueError("node_index is out of range")
    if np.unique(node_index).size != node_index.size:
        raise ValueError("node_index must be unique")
    z_sel = z[node_index]
    r_sel = r[node_index]
    for zi in z_sel:
        if int(zi) not in covalent_radii:
            raise ValueError("atomic numbers must belong to the paper element set")
    radii = np.array([covalent_radii[int(zi)] for zi in z_sel], dtype=float)
    dist = np.linalg.norm(r_sel[:, None, :] - r_sel[None, :, :], axis=-1)
    undirected = []
    n = z_sel.size
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] < ase_mult * (radii[i] + radii[j]):
                undirected.append((i, j))
    if not undirected:
        raise ValueError("ASE neighbor list is empty")
    src = []
    dst = []
    for i, j in undirected:
        src.extend([i, j])
        dst.extend([j, i])
    return np.vstack([np.array(src, dtype=int), np.array(dst, dtype=int)])

import numpy as np


def graphsage_embed(
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

import numpy as np


def pool_graph_embedding(node_embeddings: np.ndarray) -> np.ndarray:
    hidden = 8
    h = np.asarray(node_embeddings, dtype=float)
    if h.ndim != 2 or h.shape[1] != hidden:
        raise ValueError("node_embeddings must have shape (n_nodes, 8)")
    if h.shape[0] == 0:
        raise ValueError("node_embeddings is empty")
    if not np.all(np.isfinite(h)):
        raise ValueError("node_embeddings contain non-finite values")
    return np.max(h, axis=0)

import numpy as np


def linear_energy_head(
    graph_embedding: np.ndarray,
    linear_weights: dict,
) -> float:
    hidden = 8
    z = np.asarray(graph_embedding, dtype=float).reshape(-1)
    if z.shape[0] != hidden:
        raise ValueError("graph_embedding must have length 8")
    if not np.all(np.isfinite(z)):
        raise ValueError("graph_embedding contains non-finite values")
    w = np.asarray(linear_weights["W"], dtype=float)
    b = np.asarray(linear_weights["b"], dtype=float).reshape(-1)
    if w.shape != (1, hidden) or b.shape != (1,):
        raise ValueError("linear head has incompatible weights")
    return float((z @ w.T + b).reshape(-1)[0])

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
    """Deterministic reference implementation for the complete GNN pipeline."""

    # Step 01: Local adsorption site
    node_index = select_local_adsorption_site(
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

    # Step 02: Eight-channel node attributes
    node_features = construct_node_attributes(
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

    # Step 03: ASE first-neighbor graph
    edge_index = build_ase_connectivity(
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

    # Step 04: GraphSAGE embedding
    node_embeddings = graphsage_embed(
        node_features,
        edge_index,
        sage_weights,
    )
    node_embeddings = np.asarray(node_embeddings, dtype=float)
    if node_embeddings.shape != (n_site, 8):
        raise ValueError("GraphSAGE embeddings must have shape (n_site, 8)")
    if not np.all(np.isfinite(node_embeddings)):
        raise ValueError("GraphSAGE embeddings contain non-finite values")

    # Step 05: Graph-level pooling
    graph_embedding = pool_graph_embedding(node_embeddings)
    graph_embedding = np.asarray(graph_embedding, dtype=float).reshape(-1)
    if graph_embedding.shape != (8,):
        raise ValueError("pooled embedding must have length 8")
    if not np.all(np.isfinite(graph_embedding)):
        raise ValueError("pooled embedding contains non-finite values")
    if np.any(graph_embedding < np.max(node_embeddings, axis=0) - 1e-12):
        raise ValueError("pooled embedding must be the per-channel node maximum")

    # Step 06: Linear reaction-energy head
    energy = linear_energy_head(graph_embedding, linear_weights)
    energy = float(np.asarray(energy).reshape(-1)[0])
    if not np.isfinite(energy):
        raise ValueError("predicted reaction energy must be finite")

    return energy
SCICODE_GOLD_EOF
