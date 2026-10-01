"""
Compute the directed tensor-BP fixed-point messages for a supplied grid network.

Belief propagation summarizes the environment of each site through directed
edge messages.  On a graph with cycles, its fixed point provides a local
reference against which nonlocal corrections can be measured.

Directed-message storage follows the edge orientation returned by the preceding
step: for edge ``(u,v)`` with ``u<v``, row 0 stores ``u -> v`` and row 1 stores
``v -> u``.  The task instance uses simultaneous sweeps, unit Euclidean
normalization, positive zeroth component, and the supplied Euclidean stopping
tolerance.

Parameters
----------
edges : np.ndarray
    Integer array of shape ``(E,2)`` returned by
    ``build_ising_tensor_network``.
neighbors : np.ndarray
    Integer array of shape ``(N,4)`` returned by the preceding step.
degrees : np.ndarray
    Integer array of shape ``(N,)`` returned by the preceding step.
tensors : np.ndarray
    Float array of shape ``(N,16)`` returned by the preceding step.
tol : float
    Finite positive threshold for the maximum Euclidean message change
    between consecutive simultaneous sweeps.

Returns
-------
messages : np.ndarray
    Float array of shape ``(E,2,2)``.  For edge ``(u,v)`` with ``u<v``,
    ``messages[e,0]`` is ``u -> v`` and ``messages[e,1]`` is ``v -> u``.
    Every returned message has unit Euclidean norm and positive zeroth
    component.

Raises
------
ValueError
    If the supplied arrays are structurally inconsistent or ``tol`` is not
    finite and positive.
RuntimeError
    If the prescribed fixed point is not reached within the internal
    safety limit.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_bp_messages(
    edges: "np.ndarray", neighbors: "np.ndarray", degrees: "np.ndarray",
    tensors: "np.ndarray", tol: float,
) -> "np.ndarray":
    '''Return converged directed BP messages in the canonical edge orientation.

    Parameters
    ----------
    edges : np.ndarray
        Integer array of shape ``(E,2)`` returned by
        ``build_ising_tensor_network``.
    neighbors : np.ndarray
        Integer array of shape ``(N,4)`` returned by the preceding step.
    degrees : np.ndarray
        Integer array of shape ``(N,)`` returned by the preceding step.
    tensors : np.ndarray
        Float array of shape ``(N,16)`` returned by the preceding step.
    tol : float
        Finite positive threshold for the maximum Euclidean message change
        between consecutive simultaneous sweeps.

    Returns
    -------
    messages : np.ndarray
        Float array of shape ``(E,2,2)``.  For edge ``(u,v)`` with ``u<v``,
        ``messages[e,0]`` is ``u -> v`` and ``messages[e,1]`` is ``v -> u``.
        Every returned message has unit Euclidean norm and positive zeroth
        component.

    Raises
    ------
    ValueError
        If the supplied arrays are structurally inconsistent or ``tol`` is not
        finite and positive.
    RuntimeError
        If the prescribed fixed point is not reached within the internal
        safety limit.
    '''
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np


def _oracle_solve_bp_messages(
    edges: "np.ndarray", neighbors: "np.ndarray", degrees: "np.ndarray",
    tensors: "np.ndarray", tol: float,
) -> "np.ndarray":
    edges = np.asarray(edges, dtype=np.int64)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    degrees = np.asarray(degrees, dtype=np.int64)
    tensors = np.asarray(tensors, dtype=float)
    tol = float(tol)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (E,2)")
    N = degrees.size
    if neighbors.shape != (N, 4) or tensors.shape != (N, 16):
        raise ValueError("network arrays have inconsistent shapes")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    E = edges.shape[0]
    if E == 0:
        return np.empty((0, 2, 2), dtype=float)

    edge_index = {tuple(row): i for i, row in enumerate(edges.tolist())}
    messages = np.zeros((E, 2, 2), dtype=float)
    messages[:, :, 0] = 1.0
    for _ in range(100000):
        updated = np.empty_like(messages)
        for e, (u0, v0) in enumerate(edges):
            u = int(u0)
            v = int(v0)
            for direction, src, dst in ((0, u, v), (1, v, u)):
                d = int(degrees[src])
                nbrs = neighbors[src, :d]
                matches = np.flatnonzero(nbrs == dst)
                if matches.size != 1:
                    raise ValueError("edge and neighbor tables are inconsistent")
                outgoing_axis = int(matches[0])
                vector = np.zeros(2, dtype=float)
                for bits in itertools.product((0, 1), repeat=d):
                    flat = 0
                    for b in bits:
                        flat = 2 * flat + b
                    term = tensors[src, flat]
                    for j, nbr0 in enumerate(nbrs):
                        nbr = int(nbr0)
                        if nbr == dst:
                            continue
                        ee = edge_index[(min(src, nbr), max(src, nbr))]
                        incoming = messages[ee, 0] if nbr < src else messages[ee, 1]
                        term *= incoming[bits[j]]
                    vector[bits[outgoing_axis]] += term
                norm = float(np.linalg.norm(vector))
                if not np.isfinite(norm) or norm <= 0.0:
                    raise ValueError("message update has zero or nonfinite norm")
                vector /= norm
                if vector[0] < 0.0:
                    vector = -vector
                updated[e, direction] = vector
        change = float(np.max(np.linalg.norm(updated - messages, axis=2)))
        messages = updated
        if change < tol:
            return messages
    raise RuntimeError("BP did not converge within the safety limit")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal loopy, isolated, degree-one, and tolerance-variation cases."""
    helper = """import numpy as np

def run_bp(build_fn, solve_fn, Kh, Kv, h, tol):
    e,n,d,t,_ = build_fn(Kh.copy(),Kv.copy(),h.copy())
    return solve_fn(e.copy(),n.copy(),d.copy(),t.copy(),tol)
"""
    return [
        {
            "setup": helper + "\nKh=np.array([[0.31],[0.47]])\nKv=np.array([[0.42,0.35]])\nh=np.array([[0.03,-0.06],[0.08,-0.04]])",
            "call": "run_bp(build_ising_tensor_network,solve_bp_messages,Kh,Kv,h,1e-13)",
            "gold_call": "run_bp(_oracle_build_ising_tensor_network,_oracle_solve_bp_messages,Kh,Kv,h,1e-13)",
        },
        {
            "setup": helper + "\nKh=np.empty((1,0))\nKv=np.empty((0,1))\nh=np.array([[0.17]])",
            "call": "run_bp(build_ising_tensor_network,solve_bp_messages,Kh,Kv,h,1e-13)",
            "gold_call": "run_bp(_oracle_build_ising_tensor_network,_oracle_solve_bp_messages,Kh,Kv,h,1e-13)",
        },
        {
            "setup": helper + "\nKh=np.array([[0.72]])\nKv=np.empty((0,2))\nh=np.array([[0.21,-0.16]])",
            "call": "run_bp(build_ising_tensor_network,solve_bp_messages,Kh,Kv,h,1e-13)",
            "gold_call": "run_bp(_oracle_build_ising_tensor_network,_oracle_solve_bp_messages,Kh,Kv,h,1e-13)",
        },
        {
            "setup": helper + "\nKh=np.array([[0.28,0.51],[0.44,0.33]])\nKv=np.array([[0.37,0.46,0.31]])\nh=np.array([[0.04,-0.05,0.09],[-0.02,0.07,-0.03]])",
            "call": "run_bp(build_ising_tensor_network,solve_bp_messages,Kh,Kv,h,1e-10)",
            "gold_call": "run_bp(_oracle_build_ising_tensor_network,_oracle_solve_bp_messages,Kh,Kv,h,1e-10)",
        },
    ]
