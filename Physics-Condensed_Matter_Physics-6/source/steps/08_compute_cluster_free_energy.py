"""
Compute the finite-weight connected-cluster physical free-energy density.

The finite-order logarithmic estimate becomes a physical free-energy density
after applying the thermodynamic sign and dividing by the number of spins.
The cutoff determines which nonlocal corrections enter this approximation.

The input arrays define an open rectangular Ising instance in the same layout as
the problem statement.  ``max_weight`` is the total cluster edge-weight cutoff
and ``bp_tol`` is the Euclidean stopping tolerance for simultaneous directed
message sweeps.  The returned scalar is per spin and uses the physical free-
energy sign convention at beta equal to one.

Parameters
----------
K_h : np.ndarray
    Finite nonnegative array of shape ``(R,C-1)``.
K_v : np.ndarray
    Finite nonnegative array of shape ``(R-1,C)``.
h : np.ndarray
    Finite field array of shape ``(R,C)`` with ``R,C >= 1``.
max_weight : int
    Nonnegative total cluster edge-weight cutoff.
bp_tol : float
    Finite positive Euclidean stopping tolerance for the prescribed BP
    fixed point.

Returns
-------
free_energy_density : float
    Physical free-energy density per spin at beta equal to one.

Raises
------
ValueError
    If the supplied grid or scalar settings violate the documented domain.
RuntimeError
    If the prescribed BP fixed point is not reached within the internal
    safety limit.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_cluster_free_energy(
    K_h: "np.ndarray", K_v: "np.ndarray", h: "np.ndarray",
    max_weight: int, bp_tol: float,
) -> float:
    '''Return the connected-cluster physical free-energy density for the grid.

    Parameters
    ----------
    K_h : np.ndarray
        Finite nonnegative array of shape ``(R,C-1)``.
    K_v : np.ndarray
        Finite nonnegative array of shape ``(R-1,C)``.
    h : np.ndarray
        Finite field array of shape ``(R,C)`` with ``R,C >= 1``.
    max_weight : int
        Nonnegative total cluster edge-weight cutoff.
    bp_tol : float
        Finite positive Euclidean stopping tolerance for the prescribed BP
        fixed point.

    Returns
    -------
    free_energy_density : float
        Physical free-energy density per spin at beta equal to one.

    Raises
    ------
    ValueError
        If the supplied grid or scalar settings violate the documented domain.
    RuntimeError
        If the prescribed BP fixed point is not reached within the internal
        safety limit.
    '''
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_cluster_free_energy(
    K_h: "np.ndarray", K_v: "np.ndarray", h: "np.ndarray",
    max_weight: int, bp_tol: float,
) -> float:
    h_array = np.asarray(h, dtype=float)
    if h_array.ndim != 2 or h_array.size < 1:
        raise ValueError("h must be a nonempty rectangular field array")
    max_weight = int(max_weight)
    if max_weight < 0:
        raise ValueError("max_weight must be nonnegative")
    bp_tol = float(bp_tol)
    if not np.isfinite(bp_tol) or bp_tol <= 0.0:
        raise ValueError("bp_tol must be finite and positive")

    edges, neighbors, degrees, tensors, bond_factors = _oracle_build_ising_tensor_network(
        K_h, K_v, h_array
    )
    messages = _oracle_solve_bp_messages(
        edges, neighbors, degrees, tensors, bp_tol
    )
    site_factors, _, projectors0, projectors_perp, F0 = _oracle_compute_bp_vacuum(
        edges, neighbors, degrees, tensors, messages
    )
    loops = _oracle_enumerate_connected_generalized_loops(edges, max_weight)
    loop_weights = _oracle_compute_normalized_loop_weights(
        edges, h_array, bond_factors, projectors0, projectors_perp,
        site_factors, loops
    )
    multiplicities, coefficients, _ = _oracle_enumerate_connected_loop_clusters(
        edges, loops, max_weight
    )
    _, delta = _oracle_evaluate_cluster_correction(
        loop_weights, multiplicities, coefficients
    )
    return float(-(F0 + delta) / h_array.size)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return canonical, no-loop, and smaller loopy whole-pipeline cases."""
    return [
        {
            "setup": "import numpy as np\nKh=np.array([[0.368,0.464,0.304],[0.400,0.336,0.496],[0.432,0.288,0.384]])\nKv=np.array([[0.352,0.448,0.416,0.320],[0.480,0.384,0.304,0.432]])\nh=np.array([[0.07,-0.04,0.02,0.09],[-0.05,0.06,-0.08,0.03],[0.04,-0.06,0.05,-0.02]])",
            "call": "compute_cluster_free_energy(Kh.copy(),Kv.copy(),h.copy(),10,1e-13)",
            "gold_call": "_oracle_compute_cluster_free_energy(Kh.copy(),Kv.copy(),h.copy(),10,1e-13)",
            "tol": 1e-11,
        },
        {
            "setup": "import numpy as np\nKh=np.empty((1,0))\nKv=np.empty((0,1))\nh=np.array([[0.11]])",
            "call": "compute_cluster_free_energy(Kh.copy(),Kv.copy(),h.copy(),0,1e-13)",
            "gold_call": "_oracle_compute_cluster_free_energy(Kh.copy(),Kv.copy(),h.copy(),0,1e-13)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nKh=np.array([[0.31,0.47],[0.38,0.29]])\nKv=np.array([[0.42,0.35,0.51]])\nh=np.array([[0.03,-0.06,0.08],[-0.04,0.05,-0.02]])",
            "call": "compute_cluster_free_energy(Kh.copy(),Kv.copy(),h.copy(),6,1e-13)",
            "gold_call": "_oracle_compute_cluster_free_energy(Kh.copy(),Kv.copy(),h.copy(),6,1e-13)",
            "tol": 1e-11,
        },
    ]
