"""
Construct the numerical data for an open rectangular Ising tensor network.

An Ising partition function can be represented using local site tensors joined
by two-state bond indices.  This local representation makes both the graph
geometry and its contraction available to later BP and loop corrections.

Sites are numbered in row-major order.  The returned edge list uses lower site
label first and is sorted lexicographically.  For each site, neighbor labels are
stored in increasing order.  The flattened tensor entries use those neighbor
axes in that order; only the first ``2**degree`` entries of each tensor row are
active.  The spin index of ``bond_factors`` follows the ordered basis
``(-1,+1)`` and the bond index follows ``(0,1)``.

Parameters
----------
K_h : np.ndarray
    Finite nonnegative array with shape ``(R,C-1)`` containing horizontal
    couplings for an ``R`` by ``C`` open grid.
K_v : np.ndarray
    Finite nonnegative array with shape ``(R-1,C)`` containing vertical
    couplings for the same grid.
h : np.ndarray
    Finite array with shape ``(R,C)`` containing site fields.  ``R`` and
    ``C`` must both be positive.

Returns
-------
edges : np.ndarray
    Integer array of shape ``(E,2)``.  Each row is ``(u,v)`` with ``u<v``;
    rows are sorted lexicographically.
neighbors : np.ndarray
    Integer array of shape ``(R*C,4)``.  Active neighbor labels appear in
    increasing order and unused entries are ``-1``.
degrees : np.ndarray
    Integer array of shape ``(R*C,)`` giving the number of active neighbors.
tensors : np.ndarray
    Float array of shape ``(R*C,16)``.  For site ``v`` of degree ``d``, the
    first ``2**d`` entries are the C-order flattening of its local tensor in
    the neighbor-axis order above; remaining entries are zero.
bond_factors : np.ndarray
    Float array of shape ``(E,2,2)``.  Axis 1 uses spin order ``(-1,+1)``
    and axis 2 uses bond-index order ``(0,1)``.

Raises
------
ValueError
    If the array dimensions are inconsistent, a required value is
    nonfinite, or a coupling is negative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_ising_tensor_network(
    K_h: "np.ndarray", K_v: "np.ndarray", h: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    '''Build topology, site tensors, and edge factors for a rectangular grid.

    Parameters
    ----------
    K_h : np.ndarray
        Finite nonnegative array with shape ``(R,C-1)`` containing horizontal
        couplings for an ``R`` by ``C`` open grid.
    K_v : np.ndarray
        Finite nonnegative array with shape ``(R-1,C)`` containing vertical
        couplings for the same grid.
    h : np.ndarray
        Finite array with shape ``(R,C)`` containing site fields.  ``R`` and
        ``C`` must both be positive.

    Returns
    -------
    edges : np.ndarray
        Integer array of shape ``(E,2)``.  Each row is ``(u,v)`` with ``u<v``;
        rows are sorted lexicographically.
    neighbors : np.ndarray
        Integer array of shape ``(R*C,4)``.  Active neighbor labels appear in
        increasing order and unused entries are ``-1``.
    degrees : np.ndarray
        Integer array of shape ``(R*C,)`` giving the number of active neighbors.
    tensors : np.ndarray
        Float array of shape ``(R*C,16)``.  For site ``v`` of degree ``d``, the
        first ``2**d`` entries are the C-order flattening of its local tensor in
        the neighbor-axis order above; remaining entries are zero.
    bond_factors : np.ndarray
        Float array of shape ``(E,2,2)``.  Axis 1 uses spin order ``(-1,+1)``
        and axis 2 uses bond-index order ``(0,1)``.

    Raises
    ------
    ValueError
        If the array dimensions are inconsistent, a required value is
        nonfinite, or a coupling is negative.
    '''
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np


def _oracle_build_ising_tensor_network(
    K_h: "np.ndarray", K_v: "np.ndarray", h: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    K_h = np.asarray(K_h, dtype=float)
    K_v = np.asarray(K_v, dtype=float)
    h = np.asarray(h, dtype=float)
    if h.ndim != 2 or h.shape[0] < 1 or h.shape[1] < 1:
        raise ValueError("h must have shape (R,C) with R,C >= 1")
    R, C = h.shape
    if K_h.shape != (R, max(C - 1, 0)):
        raise ValueError("K_h must have shape (R,C-1)")
    if K_v.shape != (max(R - 1, 0), C):
        raise ValueError("K_v must have shape (R-1,C)")
    if not (np.all(np.isfinite(K_h)) and np.all(np.isfinite(K_v)) and np.all(np.isfinite(h))):
        raise ValueError("all inputs must be finite")
    if np.any(K_h < 0.0) or np.any(K_v < 0.0):
        raise ValueError("couplings must be nonnegative")

    records = []
    for r in range(R):
        for c in range(C - 1):
            u = r * C + c
            records.append((u, u + 1, float(K_h[r, c])))
    for r in range(R - 1):
        for c in range(C):
            u = r * C + c
            records.append((u, (r + 1) * C + c, float(K_v[r, c])))
    records.sort(key=lambda x: (x[0], x[1]))

    E = len(records)
    N = R * C
    edges = np.asarray([(u, v) for u, v, _ in records], dtype=np.int64).reshape(E, 2)
    spins = np.array([-1.0, 1.0], dtype=float)
    bond_factors = np.empty((E, 2, 2), dtype=float)
    for e, (_, _, K) in enumerate(records):
        bond_factors[e, :, 0] = np.sqrt(np.cosh(K))
        bond_factors[e, :, 1] = spins * np.sqrt(np.sinh(K))

    neighbor_lists = [[] for _ in range(N)]
    for u, v in edges:
        neighbor_lists[int(u)].append(int(v))
        neighbor_lists[int(v)].append(int(u))
    for lst in neighbor_lists:
        lst.sort()
    degrees = np.asarray([len(lst) for lst in neighbor_lists], dtype=np.int64)
    neighbors = np.full((N, 4), -1, dtype=np.int64)
    for v, lst in enumerate(neighbor_lists):
        neighbors[v, : len(lst)] = lst

    edge_index = {tuple(row): i for i, row in enumerate(edges.tolist())}
    tensors = np.zeros((N, 16), dtype=float)
    for v in range(N):
        d = int(degrees[v])
        for bits in itertools.product((0, 1), repeat=d):
            flat = 0
            for b in bits:
                flat = 2 * flat + b
            value = 0.0
            for spin_index, spin in enumerate(spins):
                term = np.exp(h.flat[v] * spin)
                for j, nbr in enumerate(neighbor_lists[v]):
                    e = edge_index[(min(v, nbr), max(v, nbr))]
                    term *= bond_factors[e, spin_index, bits[j]]
                value += term
            tensors[v, flat] = value
    return edges, neighbors, degrees, tensors, bond_factors

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, degree-one, isolated-site, and invalid-shape cases."""
    helper = """import numpy as np

def pack_network(fn, Kh, Kv, h):
    e, n, d, t, b = fn(Kh.copy(), Kv.copy(), h.copy())
    return (e, n, d, t, b)

def expect_value_error(fn, Kh, Kv, h):
    try:
        fn(Kh.copy(), Kv.copy(), h.copy())
    except ValueError:
        return 1.0
    return 0.0
"""
    return [
        {
            "setup": helper + "\nKh=np.array([[0.31],[0.47]])\nKv=np.array([[0.42,0.35]])\nh=np.array([[0.03,-0.06],[0.08,-0.04]])",
            "call": "pack_network(build_ising_tensor_network,Kh,Kv,h)",
            "gold_call": "pack_network(_oracle_build_ising_tensor_network,Kh,Kv,h)",
        },
        {
            "setup": helper + "\nKh=np.array([[0.52]])\nKv=np.empty((0,2))\nh=np.array([[0.09,-0.11]])",
            "call": "pack_network(build_ising_tensor_network,Kh,Kv,h)",
            "gold_call": "pack_network(_oracle_build_ising_tensor_network,Kh,Kv,h)",
        },
        {
            "setup": helper + "\nKh=np.empty((1,0))\nKv=np.empty((0,1))\nh=np.array([[0.17]])",
            "call": "pack_network(build_ising_tensor_network,Kh,Kv,h)",
            "gold_call": "pack_network(_oracle_build_ising_tensor_network,Kh,Kv,h)",
        },
        {
            "setup": helper + "\nKh=np.array([[0.2,0.3]])\nKv=np.empty((0,2))\nh=np.array([[0.0,0.0]])",
            "call": "expect_value_error(build_ising_tensor_network,Kh,Kv,h)",
            "gold_call": "expect_value_error(_oracle_build_ising_tensor_network,Kh,Kv,h)",
        },
    ]
