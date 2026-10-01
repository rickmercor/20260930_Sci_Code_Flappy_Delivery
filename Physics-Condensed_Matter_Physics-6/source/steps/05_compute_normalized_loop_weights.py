"""
Compute the normalized excitation weight for each supplied connected-loop mask.

These weights measure correlated bond fluctuations relative to the BP
baseline.  Individual corrections can have either sign even when the complete
Ising partition function is positive.

Loop rows follow the canonical ordering from the preceding enumeration step.
Edge matrices use the documented lower-endpoint row and higher-endpoint column
orientation.  The returned one-dimensional array preserves loop-row order and
may contain either sign.

Parameters
----------
edges : np.ndarray
    Integer array of shape ``(E,2)`` in canonical endpoint orientation.
h : np.ndarray
    Finite rectangular site-field array; row-major flattening defines site
    labels used by ``edges``.
bond_factors : np.ndarray
    Float array of shape ``(E,2,2)`` returned by the network builder.
projectors0 : np.ndarray
    Float array of shape ``(E,2,2)`` returned by ``compute_bp_vacuum``.
projectors_perp : np.ndarray
    Float array of shape ``(E,2,2)`` returned by ``compute_bp_vacuum``.
site_factors : np.ndarray
    Positive float array of shape ``(h.size,)`` returned by
    ``compute_bp_vacuum``.
loops : np.ndarray
    Integer binary array of shape ``(L,E)`` in canonical loop order.

Returns
-------
loop_weights : np.ndarray
    Float array of shape ``(L,)`` preserving the input loop order.

Raises
------
ValueError
    If array shapes are inconsistent, loop entries are not binary, or the
    vacuum normalization is nonpositive or nonfinite.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_normalized_loop_weights(
    edges: "np.ndarray", h: "np.ndarray", bond_factors: "np.ndarray",
    projectors0: "np.ndarray", projectors_perp: "np.ndarray",
    site_factors: "np.ndarray", loops: "np.ndarray",
) -> "np.ndarray":
    '''Evaluate normalized weights for the supplied connected-loop masks.

    Parameters
    ----------
    edges : np.ndarray
        Integer array of shape ``(E,2)`` in canonical endpoint orientation.
    h : np.ndarray
        Finite rectangular site-field array; row-major flattening defines site
        labels used by ``edges``.
    bond_factors : np.ndarray
        Float array of shape ``(E,2,2)`` returned by the network builder.
    projectors0 : np.ndarray
        Float array of shape ``(E,2,2)`` returned by ``compute_bp_vacuum``.
    projectors_perp : np.ndarray
        Float array of shape ``(E,2,2)`` returned by ``compute_bp_vacuum``.
    site_factors : np.ndarray
        Positive float array of shape ``(h.size,)`` returned by
        ``compute_bp_vacuum``.
    loops : np.ndarray
        Integer binary array of shape ``(L,E)`` in canonical loop order.

    Returns
    -------
    loop_weights : np.ndarray
        Float array of shape ``(L,)`` preserving the input loop order.

    Raises
    ------
    ValueError
        If array shapes are inconsistent, loop entries are not binary, or the
        vacuum normalization is nonpositive or nonfinite.
    '''
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_normalized_loop_weights(
    edges: "np.ndarray", h: "np.ndarray", bond_factors: "np.ndarray",
    projectors0: "np.ndarray", projectors_perp: "np.ndarray",
    site_factors: "np.ndarray", loops: "np.ndarray",
) -> "np.ndarray":
    edges = np.asarray(edges, dtype=np.int64)
    h = np.asarray(h, dtype=float)
    bond_factors = np.asarray(bond_factors, dtype=float)
    projectors0 = np.asarray(projectors0, dtype=float)
    projectors_perp = np.asarray(projectors_perp, dtype=float)
    site_factors = np.asarray(site_factors, dtype=float)
    loops = np.asarray(loops, dtype=np.int64)
    E = edges.shape[0]
    N = h.size
    if edges.ndim != 2 or edges.shape[1] != 2 or h.ndim != 2:
        raise ValueError("edges and h have invalid shapes")
    if bond_factors.shape != (E, 2, 2) or projectors0.shape != (E, 2, 2) or projectors_perp.shape != (E, 2, 2):
        raise ValueError("edge-factor arrays have inconsistent shapes")
    if site_factors.shape != (N,) or not np.all(np.isfinite(site_factors)) or np.any(site_factors <= 0.0):
        raise ValueError("site_factors must be finite and positive")
    if loops.ndim != 2 or loops.shape[1] != E or np.any((loops != 0) & (loops != 1)):
        raise ValueError("loops must be a binary array with shape (L,E)")
    L = loops.shape[0]
    if L == 0:
        return np.empty(0, dtype=float)

    state_ids = np.arange(1 << N, dtype=np.uint64)[:, None]
    bit_positions = np.arange(N, dtype=np.uint64)
    spin_indices = ((state_ids >> bit_positions) & 1).astype(np.int8)
    spins = 2.0 * spin_indices - 1.0
    field_values = np.exp(spins @ h.reshape(-1))

    vacuum_edge_values = np.empty((E, field_values.size), dtype=float)
    excited_edge_values = np.empty_like(vacuum_edge_values)
    for e, (u0, v0) in enumerate(edges):
        u = int(u0)
        v = int(v0)
        left = bond_factors[e, spin_indices[:, u], :]
        right = bond_factors[e, spin_indices[:, v], :]
        vacuum_edge_values[e] = np.einsum("ni,ij,nj->n", left, projectors0[e], right)
        excited_edge_values[e] = np.einsum("ni,ij,nj->n", left, projectors_perp[e], right)

    Z0 = float(np.prod(site_factors))
    if not np.isfinite(Z0) or Z0 <= 0.0:
        raise ValueError("vacuum normalization must be finite and positive")
    result = np.empty(L, dtype=float)
    for ell, mask in enumerate(loops):
        values = field_values.copy()
        for e in range(E):
            values *= excited_edge_values[e] if mask[e] else vacuum_edge_values[e]
        result[ell] = float(np.sum(values) / Z0)
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return single-loop, multi-loop, and empty-loop contraction cases."""
    helper = """import numpy as np

def run_weights(build_fn, solve_fn, vacuum_fn, loop_fn, weight_fn, Kh, Kv, h, cutoff):
    e,n,d,t,b = build_fn(Kh.copy(),Kv.copy(),h.copy())
    m = solve_fn(e.copy(),n.copy(),d.copy(),t.copy(),1e-13)
    sf,_,p0,pp,_ = vacuum_fn(e.copy(),n.copy(),d.copy(),t.copy(),m.copy())
    ls = loop_fn(e.copy(),cutoff)
    return weight_fn(e.copy(),h.copy(),b.copy(),p0.copy(),pp.copy(),sf.copy(),ls.copy())
"""
    return [
        {
            "setup": helper + "\nKh=np.array([[0.40],[0.33]])\nKv=np.array([[0.28,0.45]])\nh=np.array([[0.04,-0.07],[0.02,0.05]])",
            "call": "run_weights(build_ising_tensor_network,solve_bp_messages,compute_bp_vacuum,enumerate_connected_generalized_loops,compute_normalized_loop_weights,Kh,Kv,h,4)",
            "gold_call": "run_weights(_oracle_build_ising_tensor_network,_oracle_solve_bp_messages,_oracle_compute_bp_vacuum,_oracle_enumerate_connected_generalized_loops,_oracle_compute_normalized_loop_weights,Kh,Kv,h,4)",
        },
        {
            "setup": helper + "\nKh=np.array([[0.31,0.47],[0.38,0.29]])\nKv=np.array([[0.42,0.35,0.51]])\nh=np.array([[0.03,-0.06,0.08],[-0.04,0.05,-0.02]])",
            "call": "run_weights(build_ising_tensor_network,solve_bp_messages,compute_bp_vacuum,enumerate_connected_generalized_loops,compute_normalized_loop_weights,Kh,Kv,h,6)",
            "gold_call": "run_weights(_oracle_build_ising_tensor_network,_oracle_solve_bp_messages,_oracle_compute_bp_vacuum,_oracle_enumerate_connected_generalized_loops,_oracle_compute_normalized_loop_weights,Kh,Kv,h,6)",
        },
        {
            "setup": helper + "\nKh=np.array([[0.72]])\nKv=np.empty((0,2))\nh=np.array([[0.21,-0.16]])",
            "call": "run_weights(build_ising_tensor_network,solve_bp_messages,compute_bp_vacuum,enumerate_connected_generalized_loops,compute_normalized_loop_weights,Kh,Kv,h,2)",
            "gold_call": "run_weights(_oracle_build_ising_tensor_network,_oracle_solve_bp_messages,_oracle_compute_bp_vacuum,_oracle_enumerate_connected_generalized_loops,_oracle_compute_normalized_loop_weights,Kh,Kv,h,2)",
        },
    ]
