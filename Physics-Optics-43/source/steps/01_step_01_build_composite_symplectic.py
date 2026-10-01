"""
Build the composite complex symplectic matrix of the Gaussian stage of a unified boson sampler.

Every Gaussian unitary on M modes acts on the vector of canonical operators a = [a_1..a_M, a_1^dag..a_M^dag] as an element of the complex symplectic group, written in blocks as T = [[U, V], [conj(V), conj(U)]] subject to U U^{*T} - V V^{*T} = 1 and U V^T = V U^T. A passive interferometer is the V = 0 case; single-mode squeezing supplies the off-diagonal blocks.

The interferometer is specified as an ordered list of two-mode gates. A gate (p, q, theta, phi) acts as the identity except on the block

    G[p, p] = exp(i*phi) * cos(theta),    G[p, q] = -sin(theta),

    G[q, p] = exp(i*phi) * sin(theta),    G[q, q] =  cos(theta),

and the gates are applied in list order, each new gate left-multiplying the running product, so W = G_last @ ... @ G_first. With C = diag(cosh r_k) and S = diag(sinh r_k) the composite blocks are U = W C and V = W S, and both symplectic constraints follow from unitarity of W together with cosh^2 - sinh^2 = 1.

The block convention matters downstream, because the generating-function blocks of the next step are written in terms of U^{-1}, U^{-*}, U^{-T} and U^{-*T}. Note in particular that U = W C and U = C W are both valid symplectics but describe different physical orderings; this task uses U = W C, squeezing applied before the interferometer.

Returns
-------
np.ndarray of shape (2M, 2M), complex: the matrix [[U, V], [conj(V), conj(U)]] with U = W @ diag(cosh r) and V = W @ diag(sinh r).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def build_composite_symplectic(gates: list, r_list: list) -> np.ndarray:
    """Build the 2M x 2M complex symplectic matrix of the Gaussian stage.
 
    Parameters
    ----------
    gates : list
        Ordered list of (p, q, theta, phi) tuples. p and q are integer mode
        indices with p != q, theta and phi are floats in radians. Gates are
        applied in list order, each left-multiplying the running product.
    r_list : list
        List of M real squeezing parameters, one per mode.
 
    Returns
    -------
    T : np.ndarray
        Complex array of shape (2M, 2M) equal to [[U, V], [conj(V), conj(U)]]
        with U = W @ diag(cosh r), V = W @ diag(sinh r), and W the
        interferometer built from ``gates``.
 
    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    M = len(r_list)
    return np.zeros((2 * M, 2 * M), dtype=complex)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_composite_symplectic(gates: list, r_list: list) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
 
    if not isinstance(r_list, (list, tuple, np.ndarray)):
        raise ValueError("r_list must be a sequence of squeezing parameters")
    M = len(r_list)
    if M == 0:
        raise ValueError("r_list must contain at least one mode")
    for value in r_list:
        if not (isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(float(value))):
            raise ValueError("squeezing parameters must be finite real numbers")
    for gate in gates:
        if len(gate) != 4:
            raise ValueError("each gate must be a 4-tuple (p, q, theta, phi)")
        p, q = int(gate[0]), int(gate[1])
        if p == q:
            raise ValueError("gate modes p and q must differ")
        if not (0 <= p < M and 0 <= q < M):
            raise ValueError("gate mode index out of range")
        if not (np.isfinite(float(gate[2])) and np.isfinite(float(gate[3]))):
            raise ValueError("gate angles must be finite")
 
    W = np.eye(M, dtype=complex)
    for (p, q, theta, phi) in gates:
        p, q = int(p), int(q)
        G = np.eye(M, dtype=complex)
        G[p, p] = np.exp(1j * float(phi)) * np.cos(float(theta))
        G[p, q] = -np.sin(float(theta))
        G[q, p] = np.exp(1j * float(phi)) * np.sin(float(theta))
        G[q, q] = np.cos(float(theta))
        W = G @ W
 
    r = np.asarray(r_list, dtype=float)
    U = W @ np.diag(np.cosh(r)).astype(complex)
    V = W @ np.diag(np.sinh(r)).astype(complex)
    return np.block([[U, V], [V.conj(), U.conj()]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: six-mode benchmark configuration (normal scenario) ---
        {
            "setup": """import numpy as np
gates = [(0, 1, 0.50, 0.30), (2, 3, 0.90, 1.40), (4, 5, 1.20, 0.60),
         (1, 2, 0.70, 1.90), (3, 4, 1.10, 0.80), (0, 5, 0.40, 2.20)]
r_list = [0.30, 0.45, 0.60, 0.35, 0.50, 0.40]
""",
            "call": "build_composite_symplectic(gates, r_list)",
            "gold_call": "_oracle_build_composite_symplectic(gates, r_list)",
        },
        # --- Valid: empty gate list leaves a pure squeezing symplectic ---
        {
            "setup": """import numpy as np
gates = []
r_list = [0.4, 0.7]
""",
            "call": "build_composite_symplectic(gates, r_list)",
            "gold_call": "_oracle_build_composite_symplectic(gates, r_list)",
        },
        # --- Boundary: zero squeezing leaves a purely passive device (V = 0) ---
        {
            "setup": """import numpy as np
gates = [(0, 1, 0.8, 1.2), (1, 2, 0.3, 0.5)]
r_list = [0.0, 0.0, 0.0]
""",
            "call": "build_composite_symplectic(gates, r_list)",
            "gold_call": "_oracle_build_composite_symplectic(gates, r_list)",
        },
        # --- Edge: both symplectic constraints must hold exactly (4 modes) ---
        {
            "setup": """import numpy as np
gates = [(0, 1, 0.6, 0.4), (2, 3, 1.0, 1.3), (1, 2, 0.8, 0.9), (0, 3, 1.1, 0.5)]
r_list = [0.5, 0.35, 0.6, 0.45]
M = 4
""",
            "call": (
                "(lambda T: (bool(np.allclose(T[:M, :M] @ T[:M, :M].conj().T "
                "- T[:M, M:] @ T[:M, M:].conj().T, np.eye(M))), "
                "bool(np.allclose(T[:M, :M] @ T[:M, M:].T, T[:M, M:] @ T[:M, :M].T))))"
                "(build_composite_symplectic(gates, r_list))"
            ),
            "gold_call": '(lambda T: (bool(np.allclose(T[:M, :M] @ T[:M, :M].conj().T - T[:M, M:] @ T[:M, M:].conj().T, np.eye(M))), bool(np.allclose(T[:M, :M] @ T[:M, M:].T, T[:M, M:] @ T[:M, :M].T))))(_oracle_build_composite_symplectic(gates, r_list))',
        },
    ]
