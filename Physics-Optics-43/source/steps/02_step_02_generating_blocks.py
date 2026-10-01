"""
Evaluate the generating-function blocks D, A and B at a given value of the generating variables.

A photon-number projector can be written through a generating function, |n><n| = (1/n!) d^n/dx^n E(x) at x = 0, with E(x) = x^{n_hat} = :exp[-(1-x) n_hat]: in normal order. Conjugating E(x) by the Gaussian unitary and restoring normal order with the Lie-algebraic reordering identities of the source work yields a normally ordered Gaussian form whose exponent is controlled by three blocks,

    D = diag( 1 - X U^{-*} V^* X V^T U^{-T} ),

    A = -V^* U^{-1} + U^{-T} X D^{-1} V^{*T} U^{-*T} X U^{-1},

    B = -( 1 - U^{-*T} X D^{-1} U^{-1} ),

with X = diag(x_1, ..., x_M). Here U^{-*} is the inverse of the complex conjugate of U, U^{-T} the inverse of its transpose, and U^{-*T} the inverse of its conjugate transpose.

Several structural facts are worth using as self-checks. D and A depend on x only at second order, since each contains X twice, so at x = 0 every D_i equals 1 and A(0) = -V^* U^{-1}. B is the only block linear in x, with B(0) = -1 and first-order response dB/dx_i = U^{-*T} E_ii U^{-1}, which is what carries input-photon information. A must come out symmetric for every x, since it is the coefficient matrix of a pair of identical ladder operators, and a transposition slip in its second term breaks that.

The conjugation pattern is not cosmetic: A(0) = -V^* U^{-1} is not the same object as -V U^{-1}, and the two coincide only when the interferometer is real orthogonal.

Returns
-------
tuple (D_diag, A, B): D_diag is a complex np.ndarray of shape (M,) holding the diagonal entries of D, and A and B are complex np.ndarrays of shape (M, M).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def generating_blocks(T: np.ndarray, x: list) -> tuple:
    """Evaluate the generating-function blocks at a given x.
 
    Parameters
    ----------
    T : np.ndarray
        Complex array of shape (2M, 2M) in the block layout
        [[U, V], [conj(V), conj(U)]].
    x : list
        Sequence of M generating variables.
 
    Returns
    -------
    blocks : tuple
        (D_diag, A, B) where D_diag is a complex ndarray of shape (M,) holding
        the diagonal entries of D, and A and B are complex ndarrays of shape
        (M, M).
 
    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    M = T.shape[0] // 2
    return (np.ones(M, dtype=complex),
            np.zeros((M, M), dtype=complex),
            np.zeros((M, M), dtype=complex))  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_generating_blocks(T: np.ndarray, x: list) -> tuple:
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
    for value in x:
        if not np.isfinite(complex(value).real) or not np.isfinite(complex(value).imag):
            raise ValueError("generating variables must be finite")
 
    U = T[:M, :M]
    V = T[:M, M:]
    X = np.diag(np.asarray(x, dtype=complex))
 
    Ui = np.linalg.inv(U)
    Uci = np.linalg.inv(U.conj())
    Uti = np.linalg.inv(U.T)
    Udi = np.linalg.inv(U.conj().T)
 
    Dm = np.eye(M, dtype=complex) - X @ Uci @ V.conj() @ X @ V.T @ Uti
    D_diag = np.diag(Dm).copy()
    if np.any(D_diag == 0):
        raise ValueError("D has a vanishing diagonal entry; blocks are singular here")
    Dinv = np.diag(1.0 / D_diag)
 
    A = -V.conj() @ Ui + Uti @ X @ Dinv @ V.conj().T @ Udi @ X @ Ui
    B = -(np.eye(M, dtype=complex) - Udi @ X @ Dinv @ Ui)
    return D_diag, A, B

# =============================================================================
# TEST CASES
# =============================================================================

_BENCH_6 = """import numpy as np
M = 6
W = np.eye(M, dtype=complex)
for (p, q, t, f) in [(0, 1, 0.50, 0.30), (2, 3, 0.90, 1.40), (4, 5, 1.20, 0.60),
                     (1, 2, 0.70, 1.90), (3, 4, 1.10, 0.80), (0, 5, 0.40, 2.20)]:
    G = np.eye(M, dtype=complex)
    G[p, p] = np.exp(1j * f) * np.cos(t); G[p, q] = -np.sin(t)
    G[q, p] = np.exp(1j * f) * np.sin(t); G[q, q] = np.cos(t)
    W = G @ W
r = np.array([0.30, 0.45, 0.60, 0.35, 0.50, 0.40])
U = W @ np.diag(np.cosh(r)); V = W @ np.diag(np.sinh(r))
T = np.block([[U, V], [V.conj(), U.conj()]])
"""
 
 
def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark configuration evaluated away from the origin ---
        {
            "setup": _BENCH_6 + "x = [0.0, 0.0, 0.0, 1e-3, 0.0, 0.0]\n",
            "call": "generating_blocks(T, x)",
            "gold_call": "_oracle_generating_blocks(T, x)",
        },
        # --- Valid: at x = 0 the blocks collapse to D = 1, B = -1, A = -V* U^-1 ---
        {
            "setup": """import numpy as np
M = 3
W = np.eye(M, dtype=complex)
for (p, q, t, f) in [(0, 1, 0.6, 0.4), (1, 2, 1.0, 1.3), (0, 2, 0.8, 0.9)]:
    G = np.eye(M, dtype=complex)
    G[p, p] = np.exp(1j * f) * np.cos(t); G[p, q] = -np.sin(t)
    G[q, p] = np.exp(1j * f) * np.sin(t); G[q, q] = np.cos(t)
    W = G @ W
r = np.array([0.5, 0.35, 0.6])
U = W @ np.diag(np.cosh(r)); V = W @ np.diag(np.sinh(r))
T = np.block([[U, V], [V.conj(), U.conj()]])
x = [0.0, 0.0, 0.0]
""",
            "call": (
                "(lambda o: (bool(np.allclose(o[0], np.ones(3))), "
                "bool(np.allclose(o[2], -np.eye(3))), "
                "bool(np.allclose(o[1], -T[:3, 3:].conj() @ np.linalg.inv(T[:3, :3])))))"
                "(generating_blocks(T, x))"
            ),
            "gold_call": '(lambda o: (bool(np.allclose(o[0], np.ones(3))), bool(np.allclose(o[2], -np.eye(3))), bool(np.allclose(o[1], -T[:3, 3:].conj() @ np.linalg.inv(T[:3, :3])))))(_oracle_generating_blocks(T, x))',
        },
        # --- Boundary: A must stay symmetric at a generic nonzero x (4 modes) ---
        {
            "setup": """import numpy as np
M = 4
W = np.eye(M, dtype=complex)
for (p, q, t, f) in [(0, 1, 0.6, 0.4), (2, 3, 1.0, 1.3), (1, 2, 0.8, 0.9), (0, 3, 1.1, 0.5)]:
    G = np.eye(M, dtype=complex)
    G[p, p] = np.exp(1j * f) * np.cos(t); G[p, q] = -np.sin(t)
    G[q, p] = np.exp(1j * f) * np.sin(t); G[q, q] = np.cos(t)
    W = G @ W
r = np.array([0.5, 0.35, 0.6, 0.45])
U = W @ np.diag(np.cosh(r)); V = W @ np.diag(np.sinh(r))
T = np.block([[U, V], [V.conj(), U.conj()]])
x = [0.05, 0.02, 0.01, 0.04]
""",
            "call": '(lambda o: bool(np.allclose(o[1], o[1].T)))(generating_blocks(T, x))',
            "gold_call": '(lambda o: bool(np.allclose(o[1], o[1].T)))(_oracle_generating_blocks(T, x))',
        },
        # --- Edge: single mode with no interferometer, A(0) reduces to -tanh(r) ---
        {
            "setup": """import numpy as np
r0 = 0.55
U = np.array([[np.cosh(r0)]], dtype=complex)
V = np.array([[np.sinh(r0)]], dtype=complex)
T = np.block([[U, V], [V.conj(), U.conj()]])
x = [0.0]
EXPECTED = -np.tanh(0.55)
""",
            "call": "bool(abs((lambda o: o[1][0, 0])(generating_blocks(T, x)) - EXPECTED) < 1e-12)",
            "gold_call": 'bool(abs((lambda o: o[1][0, 0])(_oracle_generating_blocks(T, x)) - EXPECTED) < 1e-12)',
        },
    ]
