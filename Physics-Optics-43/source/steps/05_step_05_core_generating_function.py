"""
Assemble the normalized generating function of the sampler at a fixed value of the generating variables.

Combine the generating-function blocks with the Fock-projection sum and the scalar normalization to form

    F(x) = Wsum(x) / ( |det U| * sqrt( prod_i D_i(x) ) ),

where Wsum(x) is the weighted Hafnian sum for the detection pattern evaluated with the blocks A(x) and B(x), the D_i(x) are the diagonal entries of D, and U is the upper-left block of the composite symplectic matrix. The photon-number probability is obtained in a later step by differentiating F(x) with respect to the generating variables.

The determinant factor comes from reordering the number-operator exponentials and the square root of the product of the D_i from reordering the squeezing generators. The square root is essential: the normalization is not prod_i D_i.

Two limits serve as checks. At x = 0 every D_i equals 1, so F(0) = Wsum(0) / |det U|; with no input photons this is already the full probability and it reproduces the standard Gaussian sampling result. For a passive device with all r_k = 0 one has V = 0, |det U| = 1 and D = 1, so F reduces to a permanent-type expression. Because the interferometer is unitary, |det U| equals the product of cosh(r_k) whatever the interferometer, which gives a cheap independent check on the normalization.

Returns
-------
complex: the value of F(x) defined above.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def core_generating_function(T: np.ndarray, x: list, m_pattern: list) -> complex:
    """Evaluate the normalized generating function at a given x.
 
    Parameters
    ----------
    T : np.ndarray
        Complex array of shape (2M, 2M) in the block layout
        [[U, V], [conj(V), conj(U)]].
    x : list
        Sequence of M generating variables.
    m_pattern : list
        Sequence of M non-negative integers, the detected photon number in each
        mode.
 
    Returns
    -------
    value : complex
        The value of F(x) = Wsum(x) / (|det U| sqrt(prod_i D_i(x))).
 
    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0j  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_core_generating_function(T: np.ndarray, x: list, m_pattern: list) -> complex:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
 
    T = np.asarray(T, dtype=complex)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("T must be a square matrix")
    if T.shape[0] == 0 or T.shape[0] % 2 != 0:
        raise ValueError("T must have even dimension 2M with M >= 1")
    M = T.shape[0] // 2
    if len(x) != M:
        raise ValueError("x must have length M")
    if len(m_pattern) != M:
        raise ValueError("m_pattern must have length M")
 
    D_diag, A, B = _oracle_generating_blocks(T, x)
    Wsum = _oracle_fock_projection_sum(A, B, m_pattern)
    U = T[:M, :M]
    norm = abs(np.linalg.det(U)) * np.sqrt(np.prod(D_diag))
    if norm == 0:
        raise ValueError("normalization vanishes; the configuration is singular")
    return complex(Wsum / norm)

# =============================================================================
# TEST CASES
# =============================================================================

_BENCH_T = """import numpy as np
M = 6
W = np.eye(M, dtype=complex)
for (p, q, t, f) in [(0, 1, 0.50, 0.30), (2, 3, 0.90, 1.40), (4, 5, 1.20, 0.60),
                     (1, 2, 0.70, 1.90), (3, 4, 1.10, 0.80), (0, 5, 0.40, 2.20)]:
    G0 = np.eye(M, dtype=complex)
    G0[p, p] = np.exp(1j * f) * np.cos(t); G0[p, q] = -np.sin(t)
    G0[q, p] = np.exp(1j * f) * np.sin(t); G0[q, q] = np.cos(t)
    W = G0 @ W
r = np.array([0.30, 0.45, 0.60, 0.35, 0.50, 0.40])
U = W @ np.diag(np.cosh(r)); V = W @ np.diag(np.sinh(r))
T = np.block([[U, V], [V.conj(), U.conj()]])
"""
 
 
def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark pattern evaluated off the origin (normal scenario) ---
        {
            "setup": _BENCH_T + "x = [0.0, 0.0, 0.0, 1e-3, 0.0, 0.0]\nm = [0, 1, 1, 0, 0, 0]\n",
            "call": '(lambda z: np.array([z.real, z.imag], dtype=float))(core_generating_function(T, x, m))',
            "gold_call": '(lambda z: np.array([z.real, z.imag], dtype=float))(_oracle_core_generating_function(T, x, m))',
        },
        # --- Valid: at the origin with no input photons this is already a probability ---
        {
            "setup": _BENCH_T + "x = [0.0] * 6\nm = [0, 1, 1, 0, 0, 0]\n",
            "call": '(lambda z: np.array([z.real, z.imag], dtype=float))(core_generating_function(T, x, m))',
            "gold_call": '(lambda z: np.array([z.real, z.imag], dtype=float))(_oracle_core_generating_function(T, x, m))',
        },
        # --- Boundary: vacuum detection at the origin returns 1 / prod cosh(r) ---
        {
            "setup": _BENCH_T + "x = [0.0] * 6\nm = [0, 0, 0, 0, 0, 0]\nEXPECTED = 1.0 / np.prod(np.cosh(r))\n",
            "call": "bool(abs(core_generating_function(T, x, m).real - EXPECTED) < 1e-12)",
            "gold_call": 'bool(abs(_oracle_core_generating_function(T, x, m).real - EXPECTED) < 1e-12)',
        },
        # --- Edge: passive limit, zero squeezing gives |det U| = 1 and D = 1 ---
        {
            "setup": """import numpy as np
M = 3
W = np.eye(M, dtype=complex)
for (p, q, t, f) in [(0, 1, 0.6, 0.4), (1, 2, 1.0, 1.3)]:
    G0 = np.eye(M, dtype=complex)
    G0[p, p] = np.exp(1j * f) * np.cos(t); G0[p, q] = -np.sin(t)
    G0[q, p] = np.exp(1j * f) * np.sin(t); G0[q, q] = np.cos(t)
    W = G0 @ W
U = W.copy(); V = np.zeros((M, M), dtype=complex)
T = np.block([[U, V], [V.conj(), U.conj()]])
x = [0.0] * 3
m = [0, 0, 0]
""",
            "call": "bool(abs(core_generating_function(T, x, m).real - 1.0) < 1e-12)",
            "gold_call": 'bool(abs(_oracle_core_generating_function(T, x, m).real - 1.0) < 1e-12)',
        },
    ]
