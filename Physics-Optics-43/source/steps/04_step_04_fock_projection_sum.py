"""
Evaluate the Fock-projection sum over coherent-amplitude derivatives for a detection pattern.

The Glauber-Sudarshan representation of a Fock projector is a finite sum of delta-function derivatives,

    |m><m| = integral d^2 alpha  sum_{k=0}^{m} (1/k!) binom(m, k)

             d^k/d alpha^k d^k/d alpha^{*k} [ delta^(2)(alpha) ] |alpha><alpha|,

and integrating by parts transfers those derivatives onto the coherent symbol with an overall plus sign, because the derivatives always come in alpha, alpha^* pairs. Applying this mode by mode gives

    Wsum = sum over k of [ prod_j binom(m_j, k_j) / k_j! ] * Haf( G_sub(k) ),

with each k_j running from 0 to m_j and G_sub(k) the submatrix of [[A, B], [B^T, A*]] selected by k as defined in the previous step. The Hafnian of the empty matrix is 1 and the Hafnian of any odd-dimensional matrix is 0.

Each term is a derivative of a Gaussian exponential at the origin, hence a Hafnian. The sum over k is the sum over subsets that appears in the probability formula of the source work, and the terms with k_j = 0 are not spurious: they supply the alternating structure that makes the vacuum expectation come out right. For a collision-free pattern on two modes at x = 0, where B = -1, the four terms combine as 1 - 1 - 1 + (|A_12|^2 + 1) = |A_12|^2, recovering the pure Gaussian result; dropping the k_j = 0 terms would leave a spurious additive 1.

Returns
-------
complex: the value of the weighted sum of Hafnians defined above.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def fock_projection_sum(A: np.ndarray, B: np.ndarray, m_pattern: list) -> complex:
    """Evaluate the weighted Hafnian sum for a detection pattern.
 
    Parameters
    ----------
    A : np.ndarray
        Complex array of shape (M, M), the symmetric block of G.
    B : np.ndarray
        Complex array of shape (M, M), the mixed block of G.
    m_pattern : list
        Sequence of M non-negative integers, the detected photon number in each
        mode.
 
    Returns
    -------
    value : complex
        The weighted sum of Hafnians over all k with 0 <= k_j <= m_j.
 
    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example ``import numpy as np``,
    ``import itertools`` and ``from math import comb, factorial``) inside the
    function body.
    """
    return 0j  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _hafnian(mat) -> complex:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
 
    mat = np.asarray(mat, dtype=complex)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Hafnian requires a square matrix")
    n = mat.shape[0]
    if n == 0:
        return 1.0 + 0j
    if n % 2 == 1:
        return 0.0 + 0j
 
    def matchings(rem):
        if not rem:
            yield []
            return
        first = rem[0]
        for i in range(1, len(rem)):
            for rest in matchings(rem[1:i] + rem[i + 1:]):
                yield [(first, rem[i])] + rest
 
    total = 0.0 + 0j
    for pairing in matchings(list(range(n))):
        prod = 1.0 + 0j
        for a, b in pairing:
            prod *= mat[a, b]
        total += prod
    return complex(total)
 
 
def _oracle_fock_projection_sum(A: np.ndarray, B: np.ndarray, m_pattern: list) -> complex:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import itertools
    import numpy as np
    from math import comb, factorial
 
    A = np.asarray(A, dtype=complex)
    B = np.asarray(B, dtype=complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if B.shape != A.shape:
        raise ValueError("B must have the same shape as A")
    M = A.shape[0]
    if len(m_pattern) != M:
        raise ValueError("m_pattern must have length M")
    for mj in m_pattern:
        if int(mj) != mj or int(mj) < 0:
            raise ValueError("m_pattern entries must be non-negative integers")
 
    total = 0.0 + 0j
    for k_vec in itertools.product(*[range(int(mj) + 1) for mj in m_pattern]):
        coef = 1.0
        for mj, kj in zip(m_pattern, k_vec):
            coef *= comb(int(mj), int(kj)) / factorial(int(kj))
        # -- Sub-problem 03: assemble G and select the submatrix for this k.
        G_sub = _oracle_assemble_g_submatrix(A, B, list(k_vec))
        total += coef * _hafnian(G_sub)
    return complex(total)

# =============================================================================
# TEST CASES
# =============================================================================

_BENCH_BLOCKS = """import numpy as np
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
A = -V.conj() @ np.linalg.inv(U)
B = -np.eye(M, dtype=complex)
"""
 
 
def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark detection pattern at the origin (normal scenario) ---
        {
            "setup": _BENCH_BLOCKS + "m = [0, 1, 1, 0, 0, 0]\n",
            "call": '(lambda z: np.array([z.real, z.imag], dtype=float))(fock_projection_sum(A, B, m))',
            "gold_call": '(lambda z: np.array([z.real, z.imag], dtype=float))(_oracle_fock_projection_sum(A, B, m))',
        },
        # --- Valid: collision pattern with two photons in one mode ---
        {
            "setup": _BENCH_BLOCKS + "m = [2, 0, 0, 0, 0, 0]\n",
            "call": '(lambda z: np.array([z.real, z.imag], dtype=float))(fock_projection_sum(A, B, m))',
            "gold_call": '(lambda z: np.array([z.real, z.imag], dtype=float))(_oracle_fock_projection_sum(A, B, m))',
        },
        # --- Edge: vacuum detection leaves a single term equal to 1 ---
        {
            "setup": _BENCH_BLOCKS + "m = [0, 0, 0, 0, 0, 0]\n",
            "call": '(lambda z: np.array([z.real, z.imag], dtype=float))(fock_projection_sum(A, B, m))',
            "gold_call": '(lambda z: np.array([z.real, z.imag], dtype=float))(_oracle_fock_projection_sum(A, B, m))',
        },
        # --- Boundary: the alternating structure must reproduce |A_12|^2 ---
        {
            "setup": _BENCH_BLOCKS + "m = [0, 1, 1, 0, 0, 0]\nEXPECTED = abs(A[1, 2]) ** 2\n",
            "call": "bool(abs(fock_projection_sum(A, B, m).real - EXPECTED) < 1e-12)",
            "gold_call": 'bool(abs(_oracle_fock_projection_sum(A, B, m).real - EXPECTED) < 1e-12)',
        },
    ]
