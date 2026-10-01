#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def lfxxz_terms(N, J, Jz, hz):
    """Pauli-string terms of H = J sum_i (X_i X_{i+1} + Y_i Y_{i+1}) + Jz sum_i Z_i Z_{i+1}
    + hz sum_i Z_i (open chain) as an (n_terms, 2) float64 array [code, coefficient], ordered
    bond by bond (XX, YY, ZZ for i = 1..N-1) and then the N field terms. Site 1 is s = 0;
    X = 1, Z = 2, Y = 3 in the base-4 digit of site s."""
    if int(N) != N or N < 2:
        raise ValueError("N must be an integer >= 2")
    for v in (J, Jz, hz):
        if not np.isfinite(v):
            raise ValueError("couplings must be finite")
    N = int(N)
    rows = []
    for s in range(N - 1):
        rows.append([1 * 4 ** s + 1 * 4 ** (s + 1), float(J)])
        rows.append([3 * 4 ** s + 3 * 4 ** (s + 1), float(J)])
        rows.append([2 * 4 ** s + 2 * 4 ** (s + 1), float(Jz)])
    for s in range(N):
        rows.append([2 * 4 ** s, float(hz)])
    return np.array(rows, dtype=np.float64)

import numpy as np


def doubled_terms(terms, N):
    """Terms of the doubled Hamiltonian H_D = H (x) 1 - 1 (x) H^* on 2N sites (Methods B of the
    source): every term of H acts unchanged on the first copy (sites 1..N) and, complex
    conjugated and with a minus sign, on the second copy (sites N+1..2N). A Pauli string with
    n_Y factors of Y conjugates to (-1)^{n_Y} times itself."""
    terms = np.asarray(terms, dtype=np.float64)
    if terms.ndim != 2 or terms.shape[1] != 2 or int(N) != N or N < 1:
        raise ValueError("terms must be (n_terms, 2) and N a positive integer")
    N = int(N)
    rows = []
    for code_f, coef in terms:
        T = int(code_f)
        if T >= 4 ** N:
            raise ValueError("a term acts outside the N-site chain")
        n_y = sum(1 for s in range(N) if (T >> (2 * s)) & 3 == 3)
        rows.append([float(T), float(coef)])
        rows.append([float(T << (2 * N)), -float(coef) * (-1.0) ** n_y])
    return np.array(rows, dtype=np.float64)

import numpy as np


def liouvillian_apply(terms, M, v):
    """L[O] = i [H, O] on the Pauli-basis coefficient vector v (length 4^M). For a Hermitian
    term T and string P that anticommute, i[T, P] = 2 i T P and T P = phase * (T xor P) with
    phase in {+i, -i}, so every entry of L v is real. Commuting pairs contribute nothing."""
    terms = np.asarray(terms, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    M = int(M)
    D = 4 ** M
    if terms.ndim != 2 or terms.shape[1] != 2 or v.shape != (D,):
        raise ValueError("terms must be (n_terms, 2) and v must have length 4^M")
    codes = np.arange(D, dtype=np.int64)
    # single-site product table: O(f1) O(f2) = i^K[f1,f2] O(f1 xor f2), f = x + 2z, O = i^{xz} X^x Z^z
    K = np.zeros((4, 4), dtype=np.int64)
    ops = {0: np.eye(2), 1: np.array([[0, 1], [1, 0]]), 2: np.diag([1.0, -1.0]), 3: np.array([[0, -1j], [1j, 0]])}
    for a in range(4):
        for b in range(4):
            ph = np.trace(ops[a ^ b].conj().T @ (ops[a] @ ops[b])) / 2.0
            K[a, b] = int(round(np.angle(ph) / (np.pi / 2))) % 4
    out = np.zeros(D, dtype=np.float64)
    for code_f, coef in terms:
        T = int(code_f)
        kexp = np.zeros(D, dtype=np.int64)
        anti = np.zeros(D, dtype=np.int64)
        for s in range(M):
            t_s = (T >> (2 * s)) & 3
            if t_s == 0:
                continue
            p_s = (codes >> (2 * s)) & 3
            kexp += K[t_s, p_s]
            anti += ((p_s != 0) & (p_s != t_s)).astype(np.int64)
        mask = (anti % 2) == 1
        sign = np.real(1j ** ((kexp[mask] + 1) % 4))      # i [T, P] = 2 i T P, real
        np.add.at(out, codes[mask] ^ T, 2.0 * coef * sign * v[mask])
    return out

import numpy as np


def graph_pruning(terms, M, root, eps):
    """Algorithm 1 of the source on the Pauli basis. Edge weight of O -> O' is |c_O'|/2 where
    [H, O] = sum c_O' O' (declared convention). Accumulated weight Omega(O') is the MAXIMUM over
    paths of the product of edge weights from the root; a string is kept when Omega >= eps and
    is (re)recorded when it is new or its Omega improves. Layers are expanded until one adds
    no new string. Returns (2, |S|) array: row 0 the codes ascending, row 1 their Omega."""
    terms = np.asarray(terms, dtype=np.float64)
    M = int(M)
    D = 4 ** M
    if int(root) != root or not (0 <= root < D):
        raise ValueError("root must be a Pauli code in [0, 4^M)")
    if not (np.isfinite(eps) and eps > 0.0):
        raise ValueError("eps must be finite and positive")
    Ktab = np.zeros((4, 4), dtype=np.int64)
    ops = {0: np.eye(2), 1: np.array([[0, 1], [1, 0]]), 2: np.diag([1.0, -1.0]), 3: np.array([[0, -1j], [1j, 0]])}
    for a in range(4):
        for b in range(4):
            ph = np.trace(ops[a ^ b].conj().T @ (ops[a] @ ops[b])) / 2.0
            Ktab[a, b] = int(round(np.angle(ph) / (np.pi / 2))) % 4
    root = int(root)
    omega = {root: 1.0}
    layer = np.array([root], dtype=np.int64)
    while True:
        # coefficients of [H, O] for every O in the layer, accumulated over the terms
        acc = {}
        for code_f, coef in terms:
            T = int(code_f)
            kexp = np.zeros(layer.size, dtype=np.int64)
            anti = np.zeros(layer.size, dtype=np.int64)
            for s in range(M):
                t_s = (T >> (2 * s)) & 3
                if t_s == 0:
                    continue
                p_s = (layer >> (2 * s)) & 3
                kexp += Ktab[t_s, p_s]
                anti += ((p_s != 0) & (p_s != t_s)).astype(np.int64)
            mask = (anti % 2) == 1
            phase = 1j ** (kexp[mask] % 4)                   # [T, P] = 2 T P = 2 i^k (T xor P)
            for src, tgt, val in zip(layer[mask].tolist(), (layer[mask] ^ T).tolist(), (2.0 * coef * phase).tolist()):
                acc[(src, tgt)] = acc.get((src, tgt), 0.0) + val
        new = {}
        for (src, tgt), c in acc.items():
            if abs(c) < 1e-14:                              # contributions cancelled
                continue
            om = omega[src] * abs(c) / 2.0                   # edge weight |c_O'| / 2
            if om < eps:
                continue
            if tgt not in omega or om > omega[tgt]:
                if om > new.get(tgt, -1.0):
                    new[tgt] = om
        if not new:
            break
        omega.update(new)
        layer = np.array(sorted(new.keys()), dtype=np.int64)
    codes = np.array(sorted(omega.keys()), dtype=np.float64)
    vals = np.array([omega[int(c)] for c in codes], dtype=np.float64)
    return np.vstack([codes, vals])

import numpy as np


def hybrid_krylov(terms, M, root, S_codes, eps2, tol, kmax):
    """Algorithm 2 (hybrid), using the task's +i Liouvillian basis convention. K_0 = root string. For m = 0, 1, ...:
    O~ = i[H, K_m] projected onto span(S); n_pre = ||O~||; orthogonalise against ALL K_r;
    n_post = ||O_perp||; stop if n_post < tol; w <- w * n_post / n_pre; stop if w < eps2;
    K_{m+1} = O_perp / n_post. At most kmax vectors. Returns (K, 4^M) float64 array."""
    terms = np.asarray(terms, dtype=np.float64)
    M = int(M)
    D = 4 ** M
    if int(root) != root or not (0 <= root < D):
        raise ValueError("root must be a Pauli code in [0, 4^M)")
    if not (0.0 < eps2 < 1.0) or not (0.0 < tol < 1.0) or int(kmax) != kmax or kmax < 1:
        raise ValueError("need 0 < eps2 < 1, 0 < tol < 1 and kmax a positive integer")
    S = np.asarray(S_codes, dtype=np.float64).reshape(-1)
    mask = np.zeros(D, dtype=bool)
    mask[S.astype(np.int64)] = True
    if not mask[root]:
        raise ValueError("the root string must belong to S")
    K = [np.zeros(D)]
    K[0][root] = 1.0
    w = 1.0
    for m in range(int(kmax) - 1):
        Ot = liouvillian_apply(terms, M, K[m])              # i[H, K_m], following the step 3 generator
        Ot = np.where(mask, Ot, 0.0)                                  # projection onto span(S)
        npre = float(np.sqrt(Ot @ Ot))
        Op = Ot.copy()
        for r in range(len(K)):                                       # full Gram-Schmidt
            Op = Op - (K[r] @ Op) * K[r]
        npost = float(np.sqrt(Op @ Op))
        if npost < tol:
            break
        w = w * npost / npre
        if w < eps2:
            break
        K.append(Op / npost)
    return np.vstack(K)

import numpy as np


def shadow_matrix(terms, M, K):
    """Shadow matrix A_ij = <K_i, L K_j> with L[O] = i[H, O] (Methods E of the source);
    A is real antisymmetric for an orthonormal basis."""
    K = np.asarray(K, dtype=np.float64)
    M = int(M)
    if K.ndim != 2 or K.shape[1] != 4 ** M:
        raise ValueError("K must be (n_vectors, 4^M)")
    n = K.shape[0]
    A = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        LK = liouvillian_apply(terms, M, K[j])
        A[:, j] = K @ LK
    return A

import numpy as np


def thermofield_state(terms, N, beta):
    """Thermofield double state of Eq (18) of the source on 2N sites (first copy = sites 1..N,
    leftmost tensor factors): |TFD> = Z^{-1/2} sum_k e^{-beta E_k / 2} |psi_k> (x) |psi_k^*>,
    with E_k, |psi_k> the eigenpairs of the N-site H and Z = sum_k e^{-beta E_k}."""
    terms = np.asarray(terms, dtype=np.float64)
    N = int(N)
    if terms.ndim != 2 or terms.shape[1] != 2 or N < 1 or not (np.isfinite(beta) and beta >= 0.0):
        raise ValueError("terms must be (n_terms, 2), N positive and beta finite non-negative")
    ops = {0: np.eye(2, dtype=complex), 1: np.array([[0, 1], [1, 0]], dtype=complex),
           2: np.diag([1.0 + 0j, -1.0]), 3: np.array([[0, -1j], [1j, 0]])}
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for code_f, coef in terms:
        Mx = np.array([[1.0 + 0j]])
        for s in range(N):                            # site 1 (s = 0) is the leftmost factor
            Mx = np.kron(Mx, ops[(int(code_f) >> (2 * s)) & 3])
        H += coef * Mx
    w, V = np.linalg.eigh(H)
    psi = np.zeros(4 ** N, dtype=complex)
    for k in range(2 ** N):
        psi += np.exp(-beta * w[k] / 2.0) * np.kron(V[:, k], V[:, k].conj())
    psi /= np.sqrt(np.sum(np.exp(-beta * w)))
    return psi

import numpy as np


def pauli_expectations(psi, M):
    """<psi| P |psi> for every Pauli code P on M sites (site 1 the leftmost tensor factor),
    as a real vector of length 4^M, obtained by the site-by-site Pauli transform of
    rho = |psi><psi|."""
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    M = int(M)
    if psi.shape != (2 ** M,):
        raise ValueError("psi must have length 2^M")
    rho = np.outer(psi, psi.conj())
    out = rho.reshape([2] * M + [2] * M)
    for s in range(M):
        A = np.moveaxis(out, [0, M - s], [0, 1])       # row axis of site s+1, its column axis
        r00, r01, r10, r11 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
        # Tr(rho_s I), Tr(rho_s X), Tr(rho_s Z), Tr(rho_s Y) appended as a trailing axis
        out = np.stack([r00 + r11, r01 + r10, r00 - r11, 1j * r01 - 1j * r10], axis=-1)
    out = np.moveaxis(out, list(range(M)), list(range(M))[::-1])   # site 1 -> last (least significant) axis
    return np.real(out.reshape(-1))

import numpy as np


def exact_observable_series(terms, M, psi0, obs, ts):
    """Exact <psi(t)| P_obs |psi(t)> for the state psi0 evolved under the M-site H (given by
    its terms), by dense diagonalisation; site 1 is the leftmost tensor factor."""
    terms = np.asarray(terms, dtype=np.float64)
    M = int(M)
    psi0 = np.asarray(psi0, dtype=complex).reshape(-1)
    ts = np.asarray(ts, dtype=np.float64).reshape(-1)
    if psi0.shape != (2 ** M,) or int(obs) != obs or not (0 <= obs < 4 ** M):
        raise ValueError("psi0 must have length 2^M and obs must be a Pauli code")
    ops = {0: np.eye(2, dtype=complex), 1: np.array([[0, 1], [1, 0]], dtype=complex),
           2: np.diag([1.0 + 0j, -1.0]), 3: np.array([[0, -1j], [1j, 0]])}

    def matrix(code):
        Mx = np.array([[1.0 + 0j]])
        for s in range(M):
            Mx = np.kron(Mx, ops[(int(code) >> (2 * s)) & 3])
        return Mx

    H = np.zeros((2 ** M, 2 ** M), dtype=complex)
    for code_f, coef in terms:
        H += coef * matrix(int(code_f))
    w, V = np.linalg.eigh(H)
    c = V.conj().T @ psi0
    P = matrix(obs)
    out = np.empty(ts.size, dtype=np.float64)
    for k, t in enumerate(ts):
        psi = V @ (np.exp(-1j * w * t) * c)
        out[k] = float(np.real(psi.conj() @ (P @ psi)))
    return out

import numpy as np


def pruned_otoc(N, J, Jz, hz, beta, w_site, v_site, eps1, eps2, tol, kmax, t_final, dt):
    """ORCHESTRATOR. Thermal OTOC of Eq (16) with W = Z on w_site and V = X on v_site of the
    N-site chain at inverse temperature beta, computed with the source's Methods B protocol in
    the shadow picture: doubled Hamiltonian, root observable V^dagger (x) V^T, initial state
    W |TFD>; hybrid pruning (Alg 1 with eps1, Alg 2 with eps2, tol, kmax); shadow-matrix
    propagation on the grid t_k = k dt up to t_final; exact series for reference. Returns the
    pruned OTOC at t_final."""
    if int(N) != N or N < 2 or int(w_site) != w_site or int(v_site) != v_site:
        raise ValueError("N, w_site and v_site must be integers with N >= 2")
    if not (1 <= w_site <= N) or not (1 <= v_site <= N):
        raise ValueError("w_site and v_site must lie between 1 and N")
    if not (t_final > 0.0) or not (dt > 0.0) or dt > t_final:
        raise ValueError("need 0 < dt <= t_final")
    N = int(N)
    M = 2 * N
    terms = lfxxz_terms(N, J, Jz, hz)
    termsD = doubled_terms(terms, N)
    root = 1 * 4 ** (int(v_site) - 1) + 1 * 4 ** (N + int(v_site) - 1)     # V^dagger (x) V^T, V = X
    e_root = np.zeros(4 ** M)
    e_root[root] = 1.0
    l_root = liouvillian_apply(termsD, M, e_root)
    if float(l_root @ l_root) == 0.0:
        raise ValueError("the root observable commutes with H_D, its dynamics is trivial")
    S = graph_pruning(termsD, M, root, eps1)
    K = hybrid_krylov(termsD, M, root, S[0], eps2, tol, kmax)
    A = shadow_matrix(termsD, M, K)
    tfd = thermofield_state(terms, N, beta)
    # |Psi_0> = (W (x) 1) |TFD>, W = Z on w_site of the first copy: a sign on the amplitudes
    # whose bit for that site (site 1 the most significant bit) is 1
    idx = np.arange(4 ** N)
    bit = (idx >> (M - int(w_site))) & 1
    psi0 = tfd * np.where(bit == 1, -1.0, 1.0)
    ev = pauli_expectations(psi0, M)
    v0 = K @ ev
    nsteps = int(round(t_final / dt))
    ts = dt * np.arange(nsteps + 1)
    # shadow propagation v(t) = exp(-A t) v0 through the Hermitian matrix i A
    B = 0.5j * (A - A.T)
    lam, Wm = np.linalg.eigh(B)
    c = Wm.conj().T @ v0.astype(complex)
    series = np.array([float(np.real((Wm @ (np.exp(1j * lam * t) * c))[0])) for t in ts])
    exact = exact_observable_series(termsD, M, psi0, root, ts)
    if not np.all(np.isfinite(series)) or not np.all(np.isfinite(exact)):
        raise ValueError("non-finite series")
    return float(series[-1])
SCICODE_GOLD_EOF
