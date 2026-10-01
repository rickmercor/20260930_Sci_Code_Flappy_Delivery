"""
Orchestrator. Build the nx-by-ny uniform lattice of spacing dx with its origin node at index 0 and the horizon set to m times dx. For every node of the lattice build its family, evaluate the scaled basis at its neighbours and at the node, assemble the derivative and energy blocks with exponent q, and obtain its coupling row against the target vector with parameter lam; assemble those rows into the global linear system of Eq (69) that determines all nodal weights at once, solve it directly, and return the optimized influence weight of the origin corner node. Call the earlier step functions rather than reimplementing any of them. Raise ValueError if nx or ny is not an integer of at least 2, if dx or q is not positive, if lam is negative, if m is not a positive integer, or if the horizon exceeds what the grid extent can hold.

A corner node has lost roughly three quarters of its neighbourhood, so it is where the surface effect is largest and where the correction is most visible. Its weight is a single number that depends on the lattice geometry, on the coupling to its neighbours and on every choice the source makes in the fit.

Returns
-------
float, the optimized influence weight of the corner node.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def corner_influence_weight(nx, ny, dx, m, q, lam):
    """float, the optimized influence weight of the corner node."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_corner_influence_weight(nx, ny, dx, m, q, lam):
    """ORCHESTRATOR. Optimized influence weight at the corner node (0, 0) of an nx-by-ny grid.

    Calls every earlier step directly and uses each result. For EVERY node of the lattice:
      1 family with midpoint volumes, 2 scaled basis at the neighbours and at the node,
      3 derivative block, 4 energy block, 5 full-ball targets (once), 6 the normalized local
      fit, read off as w_i - sum_j gamma_ij w_j = f_i. Those N rows form the global sparse
      system K_w w = F_w of the source's eq (69); it is solved directly and w[0] returned.
    """
    import numpy as np
    if int(nx) != nx or int(ny) != ny or nx < 2 or ny < 2:
        raise ValueError("nx and ny must be integers >= 2")
    if dx <= 0.0 or q <= 0.0 or lam < 0.0:
        raise ValueError("dx and q must be positive, lam nonnegative")
    if int(m) != m or m < 1:
        raise ValueError("m must be a positive integer")
    nx, ny, m = int(nx), int(ny), int(m)
    delta = m * float(dx)
    if delta > (min(nx, ny) - 1) * dx:
        raise ValueError("horizon exceeds the grid; the corner family would be clipped by extent, not by the boundary")
    X, Y = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dx, indexing="ij")
    coords = np.column_stack([X.ravel(), Y.ravel()])
    N = coords.shape[0]

    p0 = _oracle_scaled_monomial_basis(np.zeros((1, 2)), delta)[0]
    tgt = _oracle_full_ball_targets(delta, q)
    K = np.eye(N)
    F = np.zeros(N)
    for i in range(N):
        fam = _oracle_neighbor_family(coords, i, delta, dx)
        P = _oracle_scaled_monomial_basis(fam[:, 1:3], delta)
        MD = _oracle_derivative_block(fam, P, p0, delta, q)
        ME = _oracle_energy_block(fam, P, p0, delta, q)
        row = _oracle_local_coupling_row(P, MD, ME, tgt, lam)
        idx = fam[:, 0].astype(int)
        K[i, idx] -= row[1:]
        F[i] = row[0]
    w = np.linalg.solve(K, F)
    return float(w[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "nx, ny, dx, m, q, lam = 21, 11, 0.0625, 4, 3, 0.0",
            "call": "corner_influence_weight(nx, ny, dx, m, q, lam)",
            "gold_call": "_oracle_corner_influence_weight(nx, ny, dx, m, q, lam)",
        },
        {
            "setup": "nx, ny, dx, m, q, lam = 9, 9, 0.125, 2, 3, 0.0",
            "call": "corner_influence_weight(nx, ny, dx, m, q, lam)",
            "gold_call": "_oracle_corner_influence_weight(nx, ny, dx, m, q, lam)",
        },
        {
            "setup": "nx, ny, dx, m, q, lam = 15, 15, 0.03125, 6, 3, 1e-8",
            "call": "corner_influence_weight(nx, ny, dx, m, q, lam)",
            "gold_call": "_oracle_corner_influence_weight(nx, ny, dx, m, q, lam)",
        },
    ]
