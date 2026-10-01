"""
Compute the BP-vacuum normalization data and oriented edge decomposition arrays.

The BP reference supplies a baseline contribution to the logarithm of the
partition function.  Bond-level residual data make it possible to quantify
the effect of correlations that this local baseline does not capture.

The edge orientation is inherited from the network builder.  Returned edge
matrices use the lower-labeled endpoint as row index and the higher-labeled
endpoint as column index.  ``site_factors`` are ordered by site label and
``F0`` is the scalar log-vacuum contribution used by the final pipeline.

Parameters
----------
edges : np.ndarray
    Integer array of shape ``(E,2)`` in canonical orientation.
neighbors : np.ndarray
    Integer array of shape ``(N,4)`` in increasing active-neighbor order.
degrees : np.ndarray
    Integer array of shape ``(N,)``.
tensors : np.ndarray
    Float array of shape ``(N,16)``.
messages : np.ndarray
    Float array of shape ``(E,2,2)`` returned by ``solve_bp_messages``.

Returns
-------
site_factors : np.ndarray
    Positive float array of shape ``(N,)``.
overlaps : np.ndarray
    Float array of shape ``(E,)`` in edge-list order.
projectors0 : np.ndarray
    Float array of shape ``(E,2,2)`` in the documented endpoint orientation.
projectors_perp : np.ndarray
    Float array of shape ``(E,2,2)`` in the same orientation.
F0 : float
    BP-vacuum contribution to ``log Z``.

Raises
------
ValueError
    If shapes are inconsistent or a required normalization is nonpositive
    or nonfinite.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_bp_vacuum(
    edges: "np.ndarray", neighbors: "np.ndarray", degrees: "np.ndarray",
    tensors: "np.ndarray", messages: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]":
    '''Return site factors, edge overlaps, edge matrices, and the BP log term.

    Parameters
    ----------
    edges : np.ndarray
        Integer array of shape ``(E,2)`` in canonical orientation.
    neighbors : np.ndarray
        Integer array of shape ``(N,4)`` in increasing active-neighbor order.
    degrees : np.ndarray
        Integer array of shape ``(N,)``.
    tensors : np.ndarray
        Float array of shape ``(N,16)``.
    messages : np.ndarray
        Float array of shape ``(E,2,2)`` returned by ``solve_bp_messages``.

    Returns
    -------
    site_factors : np.ndarray
        Positive float array of shape ``(N,)``.
    overlaps : np.ndarray
        Float array of shape ``(E,)`` in edge-list order.
    projectors0 : np.ndarray
        Float array of shape ``(E,2,2)`` in the documented endpoint orientation.
    projectors_perp : np.ndarray
        Float array of shape ``(E,2,2)`` in the same orientation.
    F0 : float
        BP-vacuum contribution to ``log Z``.

    Raises
    ------
    ValueError
        If shapes are inconsistent or a required normalization is nonpositive
        or nonfinite.
    '''
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np


def _oracle_compute_bp_vacuum(
    edges: "np.ndarray", neighbors: "np.ndarray", degrees: "np.ndarray",
    tensors: "np.ndarray", messages: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]":
    edges = np.asarray(edges, dtype=np.int64)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    degrees = np.asarray(degrees, dtype=np.int64)
    tensors = np.asarray(tensors, dtype=float)
    messages = np.asarray(messages, dtype=float)
    E = edges.shape[0]
    N = degrees.size
    if edges.ndim != 2 or edges.shape[1] != 2 or neighbors.shape != (N, 4) or tensors.shape != (N, 16):
        raise ValueError("network arrays have inconsistent shapes")
    if messages.shape != (E, 2, 2):
        raise ValueError("messages must have shape (E,2,2)")

    edge_index = {tuple(row): i for i, row in enumerate(edges.tolist())}
    overlaps = np.empty(E, dtype=float)
    projectors0 = np.empty((E, 2, 2), dtype=float)
    projectors_perp = np.empty((E, 2, 2), dtype=float)
    for e in range(E):
        overlap = float(np.dot(messages[e, 0], messages[e, 1]))
        if not np.isfinite(overlap) or overlap <= 0.0:
            raise ValueError("edge overlap must be finite and positive")
        overlaps[e] = overlap
        projectors0[e] = np.outer(messages[e, 1], messages[e, 0]) / overlap
        projectors_perp[e] = np.eye(2, dtype=float) - projectors0[e]

    site_factors = np.empty(N, dtype=float)
    for v in range(N):
        d = int(degrees[v])
        nbrs = neighbors[v, :d]
        value = 0.0
        for bits in itertools.product((0, 1), repeat=d):
            flat = 0
            for b in bits:
                flat = 2 * flat + b
            term = tensors[v, flat]
            for j, nbr0 in enumerate(nbrs):
                nbr = int(nbr0)
                ee = edge_index[(min(v, nbr), max(v, nbr))]
                incoming = messages[ee, 0] if nbr < v else messages[ee, 1]
                term *= incoming[bits[j]] / np.sqrt(overlaps[ee])
            value += term
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("site factor must be finite and positive")
        site_factors[v] = value
    F0 = float(np.sum(np.log(site_factors)))
    return site_factors, overlaps, projectors0, projectors_perp, F0

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return loopy, isolated-site, and degree-one vacuum cases."""
    helper = """import numpy as np

def run_vacuum(build_fn, solve_fn, vacuum_fn, Kh, Kv, h):
    e,n,d,t,_ = build_fn(Kh.copy(),Kv.copy(),h.copy())
    m = solve_fn(e.copy(),n.copy(),d.copy(),t.copy(),1e-13)
    sf,ov,p0,pp,f0 = vacuum_fn(e.copy(),n.copy(),d.copy(),t.copy(),m.copy())
    return (sf,ov,p0,pp,float(f0))
"""
    return [
        {
            "setup": helper + "\nKh=np.array([[0.31],[0.47]])\nKv=np.array([[0.42,0.35]])\nh=np.array([[0.03,-0.06],[0.08,-0.04]])",
            "call": "run_vacuum(build_ising_tensor_network,solve_bp_messages,compute_bp_vacuum,Kh,Kv,h)",
            "gold_call": "run_vacuum(_oracle_build_ising_tensor_network,_oracle_solve_bp_messages,_oracle_compute_bp_vacuum,Kh,Kv,h)",
        },
        {
            "setup": helper + "\nKh=np.empty((1,0))\nKv=np.empty((0,1))\nh=np.array([[0.17]])",
            "call": "run_vacuum(build_ising_tensor_network,solve_bp_messages,compute_bp_vacuum,Kh,Kv,h)",
            "gold_call": "run_vacuum(_oracle_build_ising_tensor_network,_oracle_solve_bp_messages,_oracle_compute_bp_vacuum,Kh,Kv,h)",
        },
        {
            "setup": helper + "\nKh=np.array([[0.72]])\nKv=np.empty((0,2))\nh=np.array([[0.21,-0.16]])",
            "call": "run_vacuum(build_ising_tensor_network,solve_bp_messages,compute_bp_vacuum,Kh,Kv,h)",
            "gold_call": "run_vacuum(_oracle_build_ising_tensor_network,_oracle_solve_bp_messages,_oracle_compute_bp_vacuum,Kh,Kv,h)",
        },
    ]
