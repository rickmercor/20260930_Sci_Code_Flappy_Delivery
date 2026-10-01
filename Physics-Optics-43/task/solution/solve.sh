#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def build_composite_symplectic(gates: list, r_list: list) -> np.ndarray:
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

def generating_blocks(T: np.ndarray, x: list) -> tuple:
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

def assemble_g_submatrix(A: np.ndarray, B: np.ndarray, k_vector: list) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
 
    A = np.asarray(A, dtype=complex)
    B = np.asarray(B, dtype=complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if B.shape != A.shape:
        raise ValueError("B must have the same shape as A")
    M = A.shape[0]
    if len(k_vector) != M:
        raise ValueError("k_vector must have length M")
    for k in k_vector:
        if int(k) != k or int(k) < 0:
            raise ValueError("k_vector entries must be non-negative integers")
 
    G = np.block([[A, B], [B.T, A.conj()]])
    ids = [j for j, k in enumerate(k_vector) for _ in range(int(k))]
    ids += [j + M for j, k in enumerate(k_vector) for _ in range(int(k))]
    if not ids:
        return np.zeros((0, 0), dtype=complex)
    return G[np.ix_(ids, ids)]

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
 
 
def fock_projection_sum(A: np.ndarray, B: np.ndarray, m_pattern: list) -> complex:
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
        G_sub = assemble_g_submatrix(A, B, list(k_vec))
        total += coef * _hafnian(G_sub)
    return complex(total)

def core_generating_function(T: np.ndarray, x: list, m_pattern: list) -> complex:
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
 
    D_diag, A, B = generating_blocks(T, x)
    Wsum = fock_projection_sum(A, B, m_pattern)
    U = T[:M, :M]
    norm = abs(np.linalg.det(U)) * np.sqrt(np.prod(D_diag))
    if norm == 0:
        raise ValueError("normalization vanishes; the configuration is singular")
    return complex(Wsum / norm)

def richardson_x_derivative(func, n_pattern: list, h: float = 1e-3) -> complex:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import itertools
    from math import factorial
 
    if not callable(func):
        raise ValueError("func must be callable")
    if not (isinstance(h, (int, float)) and float(h) > 0.0):
        raise ValueError("h must be a positive number")
    for n in n_pattern:
        if int(n) != n or int(n) not in (0, 1, 2):
            raise ValueError("derivative orders must be 0, 1 or 2")
 
    def stencil_value(step):
        stencils = []
        for n in n_pattern:
            n = int(n)
            if n == 0:
                stencils.append([(0, 1.0)])
            elif n == 1:
                stencils.append([(1, 0.5 / step), (-1, -0.5 / step)])
            else:
                stencils.append([(1, 1.0 / step ** 2),
                                 (0, -2.0 / step ** 2),
                                 (-1, 1.0 / step ** 2)])
        total = 0.0 + 0j
        for combo in itertools.product(*stencils):
            weight = 1.0
            for _, w in combo:
                weight *= w
            total += weight * func([offset * step for offset, _ in combo])
        return total
 
    denom = 1
    for n in n_pattern:
        denom *= factorial(int(n))
    raw = (4.0 * stencil_value(float(h)) - stencil_value(2.0 * float(h))) / 3.0
    return complex(raw / denom)

def ubs_probability(T: np.ndarray, n_pattern: list, m_pattern: list, h: float = 1e-3) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
 
    T = np.asarray(T, dtype=complex)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("T must be a square matrix")
    if T.shape[0] == 0 or T.shape[0] % 2 != 0:
        raise ValueError("T must have even dimension 2M with M >= 1")
    M = T.shape[0] // 2
    if len(n_pattern) != M:
        raise ValueError("n_pattern must have length M")
    if len(m_pattern) != M:
        raise ValueError("m_pattern must have length M")
 
    def f(x):
        return core_generating_function(T, x, m_pattern)
 
    return float(richardson_x_derivative(f, n_pattern, h).real)

def compute_ubs_probability(gates: list, r_list: list, n_pattern: list, m_pattern: list) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
 
    M = len(r_list)
    if M == 0:
        raise ValueError("r_list must contain at least one mode")
    if len(n_pattern) != M:
        raise ValueError("n_pattern must have length M")
    if len(m_pattern) != M:
        raise ValueError("m_pattern must have length M")
 
    for n in n_pattern:
        if int(n) != n or int(n) not in (0, 1, 2):
            raise ValueError("n_pattern entries must be 0, 1 or 2")
 
    # -- Sub-problem 01: composite symplectic matrix of the Gaussian stage.
    T = build_composite_symplectic(gates, r_list)
 
    def F(x):
        # -- Sub-problem 02: generating-function blocks at this x.
        D_diag, A, B = generating_blocks(T, x)
        # -- Sub-problems 03-04: submatrix selection and weighted Hafnian sum.
        Wsum = fock_projection_sum(A, B, m_pattern)
        # -- Sub-problem 05: normalization by |det U| and sqrt(prod D_i).
        norm = abs(np.linalg.det(T[:M, :M])) * np.sqrt(np.prod(D_diag))
        return Wsum / norm
 
    # -- Sub-problem 06: Richardson-refined derivative in the generating
    #    variables, divided by n!.  Sub-problem 07 wraps exactly this
    #    combination, so the two must agree.
    probability = float(richardson_x_derivative(F, n_pattern, 1e-3).real)
 
    # -- Sub-problem 07: same quantity through the single-call interface.
    checked = ubs_probability(T, n_pattern, m_pattern)
    if not np.isclose(probability, checked, rtol=1e-9, atol=1e-12):
        raise ValueError("chained pipeline disagrees with the step-07 interface")
    return probability
SCICODE_GOLD_EOF
