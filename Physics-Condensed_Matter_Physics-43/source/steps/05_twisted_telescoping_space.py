"""
Construct the momentum-dependent telescoping density space with the exact

basis convention used by the benchmark.

Flatten operators row-major and use deterministic complex RREF to obtain a

canonical basis. Return telescopic_basis as a two-dimensional array of shape

(telescopic_dim, d**(2*k)).

Returns
-------
# tuple, (telescopic_dim, telescopic_basis)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def twisted_telescopic_space(d: int, k: int, omega: complex, tol: float = 1e-10) -> tuple:
    """Return the canonical basis of T_{k,omega} defined above.

    Parameters
    ----------
    d : int
        Positive local physical dimension.
    k : int
        Local operator range; must be 1 or 2.
    omega : complex
        Unit-modulus sector phase.
    tol : float
        Non-negative tolerance used in canonical row reduction.

    Returns
    -------
    result : tuple
        (telescopic_dim, telescopic_basis): the dimension and canonical complex
        row basis of the twisted telescoping space.

    Raises
    ------
    ValueError
        If d, k, omega, or tol is invalid.
    """

    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np


def _oracle_twisted_telescopic_space(d: int, k: int, omega: complex, tol: float = 1e-10) -> tuple:

    def _mps_rref(a, tol=1e-10):
        try:
            tol = float(tol)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("tol must be a finite non-negative real number")
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol must be a finite non-negative real number")
        a = np.array(a, dtype=complex, copy=True)
        if a.ndim != 2:
            raise ValueError("matrix input must be two-dimensional")
        if not np.all(np.isfinite(a.real)) or not np.all(np.isfinite(a.imag)):
            raise ValueError("matrix input must contain only finite entries")
        m, n = a.shape
        pivots = []
        row = 0
        for col in range(n):
            if row >= m:
                break
            p = row + int(np.argmax(np.abs(a[row:, col])))
            if abs(a[p, col]) <= tol:
                continue
            if p != row:
                a[[row, p]] = a[[p, row]]
            a[row] /= a[row, col]
            for r in range(m):
                if r == row:
                    continue
                c = a[r, col]
                if abs(c) > tol:
                    a[r] -= c * a[row]
            a[np.abs(a) < tol] = 0.0
            pivots.append(col)
            row += 1
        return a, pivots

    def _mps_rank(a, tol=1e-10):
        a = np.asarray(a, dtype=complex)
        if a.size == 0:
            return 0
        return len(_mps_rref(a, tol)[1])

    def _mps_nullspace(a, tol=1e-10):
        R, pivots = _mps_rref(a, tol)
        n = R.shape[1]
        free = [j for j in range(n) if j not in pivots]
        K = np.zeros((n, len(free)), dtype=complex)
        for c, f in enumerate(free):
            K[f, c] = 1.0
            for r, p in enumerate(pivots):
                K[p, c] = -R[r, f]
        K[np.abs(K) < tol] = 0.0
        return K

    def _mps_row_basis(rows, tol=1e-10):
        rows = np.asarray(rows, dtype=complex)
        if rows.ndim == 1:
            rows = rows[None, :]
        if rows.size == 0:
            width = rows.shape[1] if rows.ndim == 2 else 0
            return np.zeros((0, width), dtype=complex)
        R, pivots = _mps_rref(rows, tol)
        B = R[:len(pivots)].copy()
        B[np.abs(B) < tol] = 0.0
        return B

    def _mps_words(d, L):
        return list(itertools.product(range(d), repeat=L))

    def _mps_validate_A(A):
        A = np.asarray(A, dtype=complex)
        if A.ndim != 3 or A.shape[0] < 1 or A.shape[1] < 1 or A.shape[1] != A.shape[2]:
            raise ValueError("A must have shape (d,D,D)")
        if not np.all(np.isfinite(A.real)) or not np.all(np.isfinite(A.imag)):
            raise ValueError("A must contain only finite entries")
        return A

    def _mps_validate_omega(omega, tol):
        try:
            w = complex(omega)
            tol = float(tol)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("omega and tol must be finite scalars")
        if (not np.isfinite(w.real) or not np.isfinite(w.imag) or
                not np.isfinite(tol) or tol < 0 or abs(abs(w) - 1.0) > tol):
            raise ValueError("omega must be finite with unit modulus and tol must be finite and non-negative")
        return w

    def _mps_products(A, L):
        A = _mps_validate_A(A)
        d, D, _ = A.shape
        out = []
        for word in _mps_words(d, L):
            P = np.eye(D, dtype=complex)
            for i in word:
                P = P @ A[i]
            out.append(P)
        return out

    def _mps_injectivity(A, max_L, tol):
        A = _mps_validate_A(A)
        if not isinstance(max_L, (int, np.integer)) or max_L < 1 or tol < 0:
            raise ValueError("invalid max_L or tol")
        d, D, _ = A.shape
        ranks = []
        for L in range(1, max_L + 1):
            M = np.stack([P.reshape(-1) for P in _mps_products(A, L)], axis=1)
            ranks.append(_mps_rank(M, tol))
            if ranks[-1] == D * D:
                return L, np.asarray(ranks, dtype=int)
        raise ValueError("tensor is not injective up to max_L")

    def _mps_algebra(A, tol):
        # row basis (flattened matrices) of the unital algebra generated by the A^i
        A = _mps_validate_A(A)
        d, D, _ = A.shape
        basis = _mps_row_basis(np.eye(D, dtype=complex).reshape(1, -1), tol)
        current = [np.eye(D, dtype=complex)]
        for _ in range(D * D):
            new = [A[i] @ P for P in current for i in range(d)]
            cand = np.vstack([basis] + [P.reshape(1, -1) for P in new])
            nb = _mps_row_basis(cand, tol)
            if nb.shape[0] == basis.shape[0]:
                break
            basis = nb
            current = new
        return basis

    def _mps_radical(A, tol):
        # basis (list of DxD matrices) of J = {x in Alg : Tr(x y) = 0 for all y in Alg}
        A = _mps_validate_A(A)
        D = A.shape[1]
        basis = _mps_algebra(A, tol)
        mats = [b.reshape(D, D) for b in basis]
        G = np.array([[np.trace(x @ y) for y in mats] for x in mats], dtype=complex)
        K = _mps_nullspace(G, tol)
        return [sum(K[i, c] * mats[i] for i in range(len(mats))) for c in range(K.shape[1])], basis.shape[0]

    def _mps_factors(A, max_L, tol):
        # returns (factors, factor_dims, factor_L, algebra_dim, radical_dim)
        A = _mps_validate_A(A)
        d, D, _ = A.shape
        J, algebra_dim = _mps_radical(A, tol)
        if len(J) == 0:
            L, _ = _mps_injectivity(A, max_L, tol)
            return [A], [D], [int(L)], int(algebra_dim), 0
        cols = np.hstack([x for x in J])
        Vb = _mps_row_basis(cols.T, tol)            # rows span V = sum of images
        r = Vb.shape[0]
        if r == 0 or r == D:
            raise ValueError("invariant subspace has an invalid dimension")
        piv = _mps_rref(Vb, tol)[1]
        comp = [j for j in range(D) if j not in piv]
        P = np.zeros((D, D), dtype=complex)
        P[:, :r] = Vb.T
        for c, j in enumerate(comp):
            P[j, r + c] = 1.0
        Pinv = np.linalg.inv(P)
        At = np.array([Pinv @ A[i] @ P for i in range(d)])
        if np.max(np.abs(At[:, r:, :r])) > 1e-8:
            raise ValueError("computed subspace is not invariant")
        F1 = At[:, :r, :r].copy()
        F2 = At[:, r:, r:].copy()
        F1[np.abs(F1) < tol] = 0.0
        F2[np.abs(F2) < tol] = 0.0
        L1, _ = _mps_injectivity(F1, max_L, tol)
        L2, _ = _mps_injectivity(F2, max_L, tol)
        return [F1, F2], [int(r), int(D - r)], [int(L1), int(L2)], int(algebra_dim), int(len(J))

    def _mps_local_system_one(F, k, omega, tol):
        # ordered local system of a single factor F
        F = _mps_validate_A(F)
        w = _mps_validate_omega(omega, tol)
        if k not in (1, 2):
            raise ValueError("k must be 1 or 2")
        d, D, _ = F.shape
        words = _mps_words(d, k)
        bwords = _mps_words(d, k - 1)
        nloc = len(words)
        nO = nloc * nloc
        trivial = abs(w - 1.0) <= tol
        neps = 1 if trivial else 0
        nB = len(bwords) * D * D
        nvar = nO + neps + nB
        iw = {u: i for i, u in enumerate(words)}
        ib = {u: i for i, u in enumerate(bwords)}
        plist = _mps_products(F, k)
        prod = {u: plist[iw[u]] for u in words}

        def _oi(I, J):
            return iw[I] * nloc + iw[J]

        def _bi(u, a, b):
            return nO + neps + ib[u] * D * D + a * D + b

        rows = []
        for I in words:
            for a in range(D):
                for b in range(D):
                    r = np.zeros(nvar, dtype=complex)
                    for J in words:
                        r[_oi(I, J)] += prod[J][a, b]
                    if trivial:
                        r[nO] -= prod[I][a, b]
                    for g in range(D):
                        r[_bi(I[1:], g, b)] -= F[I[0], a, g]
                        r[_bi(I[:-1], a, g)] += w * F[I[-1], g, b]
                    rows.append(r)
        M = np.stack(rows)
        matrix_rank = _mps_rank(M, tol)
        return M, matrix_rank, nvar - matrix_rank, nO, neps

    def _mps_local_systems(A, k, omega, max_L, tol):
        factors, factor_dims, factor_L, algebra_dim, radical_dim = _mps_factors(A, max_L, tol)
        mats, ranks, nulls = [], [], []
        for F in factors:
            M, rk, nul, nO, neps = _mps_local_system_one(F, k, omega, tol)
            mats.append(M)
            ranks.append(int(rk))
            nulls.append(int(nul))
        N_ref = 2 * max(factor_L) + 2 * k - 1
        return factor_L, int(N_ref), mats, ranks, nulls

    def _mps_intersect_rows(S1, S2, tol):
        if S1.shape[0] == 0 or S2.shape[0] == 0:
            return np.zeros((0, S1.shape[1]), dtype=complex)
        K = _mps_nullspace(np.hstack([S1.T, -S2.T]), tol)
        if K.shape[1] == 0:
            return np.zeros((0, S1.shape[1]), dtype=complex)
        rows = (K[:S1.shape[0], :].T) @ S1
        return _mps_row_basis(rows, tol)

    def _mps_projected(A, k, omega, max_L, tol):
        # returns (raw_nullities, factor_OE_dims, operator_dim, zero_fibre_dims, operator_basis)
        A = _mps_validate_A(A)
        factors, factor_dims, factor_L, algebra_dim, radical_dim = _mps_factors(A, max_L, tol)
        d = A.shape[0]
        nO = d ** (2 * k)
        nulls, oe_dims, fibres = [], [], []
        St = None
        for F in factors:
            M, rk, nul, nO_, neps = _mps_local_system_one(F, k, omega, tol)
            K = _mps_nullspace(M, tol)
            proj = K[:nO + neps, :]
            S_oe = _mps_row_basis(proj.T, tol)
            fib = _mps_nullspace(proj, tol).shape[1] if proj.shape[1] > 0 else 0
            nulls.append(int(nul))
            oe_dims.append(int(S_oe.shape[0]))
            fibres.append(int(fib))
            St = S_oe if St is None else _mps_intersect_rows(St, S_oe, tol)
        S = _mps_row_basis(St[:, :nO], tol) if St.shape[0] > 0 else np.zeros((0, nO), dtype=complex)
        return nulls, oe_dims, int(S.shape[0]), fibres, S

    def _mps_state(A, N):
        A = _mps_validate_A(A)
        d, D, _ = A.shape
        psi = np.zeros(d ** N, dtype=complex)
        for z, word in enumerate(_mps_words(d, N)):
            P = np.eye(D, dtype=complex)
            for i in word:
                P = P @ A[i]
            psi[z] = np.trace(P)
        return psi

    def _mps_apply_local(O, x, N, d, k, psi):
        sites = [(x + t) % N for t in range(k)]
        t = psi.reshape([d] * N)
        t = np.moveaxis(t, sites, list(range(k)))
        shape = t.shape
        t = (O @ t.reshape(d ** k, -1)).reshape(shape)
        t = np.moveaxis(t, list(range(k)), sites)
        return t.reshape(-1)

    def _mps_zero_sum_space(d, k, N, omega, tol):
        # canonical row basis of Z_{N,omega} = {O : sum_x omega^{-x} O_x = 0 as an operator}
        w = _mps_validate_omega(omega, tol)
        nloc = d ** k
        nO = nloc * nloc
        dN = d ** N
        u = np.arange(dN, dtype=np.int64)
        weights = np.array([d ** (N - 1 - j) for j in range(N)], dtype=np.int64)
        digits = (u[:, None] // weights[None, :]) % d
        wk = np.array([d ** (k - 1 - t) for t in range(k)], dtype=np.int64)
        keys, cols, coefs = [], [], []
        for x in range(N):
            sites = [(x + t) % N for t in range(k)]
            J = (digits[:, sites] * wk[None, :]).sum(axis=1)
            base = u - (digits[:, sites] * weights[sites][None, :]).sum(axis=1)
            for I in range(nloc):
                Idig = np.array([(I // wk[t]) % d for t in range(k)], dtype=np.int64)
                v = base + (Idig * weights[sites]).sum()
                keys.append(u * dN + v)
                cols.append(I * nloc + J)
                coefs.append(np.full(dN, w ** (-x), dtype=complex))
        keys = np.concatenate(keys)
        cols = np.concatenate(cols)
        coefs = np.concatenate(coefs)
        uniq, inv = np.unique(keys, return_inverse=True)
        K = np.eye(nO, dtype=complex)
        batch = 20000
        for start in range(0, uniq.size, batch):
            sel = (inv >= start) & (inv < start + batch)
            rows = np.zeros((min(batch, uniq.size - start), nO), dtype=complex)
            np.add.at(rows, (inv[sel] - start, cols[sel]), coefs[sel])
            RK = rows @ K
            NK = _mps_nullspace(RK, tol)
            K = K @ NK
            if K.shape[1] == 0:
                break
        return _mps_row_basis(K.T, tol)

    def _mps_sector_map(A, k, N, omega, tol):
        # psi and E whose columns are O_{N,omega}(E_IJ) psi
        A = _mps_validate_A(A)
        w = _mps_validate_omega(omega, tol)
        d, D, _ = A.shape
        if k not in (1, 2) or N < k or abs(w ** N - 1.0) > 10.0 * tol:
            raise ValueError("invalid or non-resonant periodic size")
        psi = _mps_state(A, N)
        nloc = d ** k
        E = np.zeros((d ** N, nloc * nloc), dtype=complex)
        for col in range(nloc * nloc):
            O = np.zeros((nloc, nloc), dtype=complex)
            O[col // nloc, col % nloc] = 1.0
            E[:, col] = sum(w ** (-x) * _mps_apply_local(O, x, N, d, k, psi) for x in range(N))
        return psi, E

    def _mps_periodic(A, k, sizes, omega, max_L, tol):
        A = _mps_validate_A(A)
        w = _mps_validate_omega(omega, tol)
        try:
            raw_sizes = np.asarray(sizes, dtype=float)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("sizes must be a finite integer sequence")
        if (raw_sizes.ndim != 1 or raw_sizes.size < 1 or
                not np.all(np.isfinite(raw_sizes)) or
                not np.all(raw_sizes == np.floor(raw_sizes))):
            raise ValueError("sizes must be a nonempty finite integer sequence")
        sizes = raw_sizes.astype(int)
        d = A.shape[0]
        _, N_ref, _, _, _ = _mps_local_systems(A, k, w, max_L, tol)
        _, _, local_dim, _, _ = _mps_projected(A, k, w, max_L, tol)
        sector_dims, zero_dims, distinct_dims, norms = [], [], [], []
        for N in sizes.tolist():
            psi, E = _mps_sector_map(A, k, int(N), w, tol)
            if abs(w - 1.0) <= tol:
                K = _mps_nullspace(np.column_stack([E, -psi]), tol)
                sector_basis = _mps_row_basis(K[:-1, :].T, tol)
            else:
                sector_basis = _mps_row_basis(_mps_nullspace(E, tol).T, tol)
            z = _mps_zero_sum_space(d, k, int(N), w, tol).shape[0]
            s = sector_basis.shape[0]
            sector_dims.append(s)
            zero_dims.append(z)
            distinct_dims.append(s - z)
            norms.append(float(np.vdot(psi, psi).real))
        return (sizes, np.asarray(sector_dims, dtype=int), np.asarray(zero_dims, dtype=int),
                np.asarray(distinct_dims, dtype=int), np.asarray(norms, dtype=float), int(local_dim), int(N_ref))

    def _mps_telescopic(d, k, omega, tol):
        w = _mps_validate_omega(omega, tol)
        if not isinstance(d, (int, np.integer)) or d < 1 or k not in (1, 2):
            raise ValueError("invalid d or k")
        n = d ** k
        if k == 1:
            return np.zeros((0, n * n), dtype=complex)
        nq = d ** (k - 1)
        Id = np.eye(d, dtype=complex)
        rows = []
        for a in range(nq):
            for b in range(nq):
                Q = np.zeros((nq, nq), dtype=complex)
                Q[a, b] = 1.0
                rows.append((np.kron(Id, Q) - w * np.kron(Q, Id)).reshape(-1))
        return _mps_row_basis(np.asarray(rows, dtype=complex), tol)

    T = _mps_telescopic(d, k, omega, tol)
    return int(T.shape[0]), T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [   {   'setup': 'def _pack_test_result(z):\n'
                     '    import numpy as np\n'
                     '    b = np.asarray(z[-1])\n'
                     '    shape = np.asarray((b.ndim, *b.shape), dtype=float)\n'
                     '    arrays = [np.asarray(x) for x in z]\n'
                     '    return np.concatenate([shape] + [x.reshape(-1).real.astype(float) for x in '
                     'arrays] + [x.reshape(-1).imag.astype(float) for x in arrays])\n'
                     'import numpy as np\n'
                     'A=np.array([[[1,-1,1,0],[0,-1,0,0],[0,0,2,1],[0,1,0,1]],[[-1,0,3,3],[0,0,2,2],[1,-1,1,0],[-1,1,1,2]],[[1,-1,0,-2],[0,1,-1,-1],[0,-1,0,1],[0,0,0,-1]]],dtype=complex); '
                     'k=2; omega=1j',
            'call': '_pack_test_result(twisted_telescopic_space(A.shape[0],k,omega))',
            'gold_call': '_pack_test_result(_oracle_twisted_telescopic_space(A.shape[0],k,omega))',
            'tol': 1e-08},
        {   'setup': 'def _pack_test_result(z):\n'
                     '    import numpy as np\n'
                     '    b = np.asarray(z[-1])\n'
                     '    shape = np.asarray((b.ndim, *b.shape), dtype=float)\n'
                     '    arrays = [np.asarray(x) for x in z]\n'
                     '    return np.concatenate([shape] + [x.reshape(-1).real.astype(float) for x in '
                     'arrays] + [x.reshape(-1).imag.astype(float) for x in arrays])\n'
                     'import numpy as np\n'
                     'A=np.array([[[-2.,3.],[-5.,6.]],[[2.,0.],[3.,-1.]]],dtype=complex); k=2; '
                     'omega=1+0j',
            'call': '_pack_test_result(twisted_telescopic_space(A.shape[0],k,omega))',
            'gold_call': '_pack_test_result(_oracle_twisted_telescopic_space(A.shape[0],k,omega))',
            'tol': 1e-08},
        {   'setup': 'def _pack_test_result(z):\n'
                     '    import numpy as np\n'
                     '    b = np.asarray(z[-1])\n'
                     '    shape = np.asarray((b.ndim, *b.shape), dtype=float)\n'
                     '    arrays = [np.asarray(x) for x in z]\n'
                     '    return np.concatenate([shape] + [x.reshape(-1).real.astype(float) for x in '
                     'arrays] + [x.reshape(-1).imag.astype(float) for x in arrays])\n'
                     'import numpy as np\n'
                     'A=np.array([[[2.,-1.],[0.,-1.]],[[2.,0.],[-2.,0.]]],dtype=complex); k=2; '
                     'omega=-1+0j',
            'call': '_pack_test_result(twisted_telescopic_space(A.shape[0],k,omega))',
            'gold_call': '_pack_test_result(_oracle_twisted_telescopic_space(A.shape[0],k,omega))',
            'tol': 1e-08},
        {   'setup': 'def _pack_test_result(z):\n'
                     '    import numpy as np\n'
                     '    b = np.asarray(z[-1])\n'
                     '    shape = np.asarray((b.ndim, *b.shape), dtype=float)\n'
                     '    arrays = [np.asarray(x) for x in z]\n'
                     '    return np.concatenate([shape] + [x.reshape(-1).real.astype(float) for x in '
                     'arrays] + [x.reshape(-1).imag.astype(float) for x in arrays])\n'
                     'import numpy as np\n'
                     'd=3; k=2; omega=complex(np.nan,0.0)\n'
                     'def refused(f):\n'
                     '    try: f(d,k,omega); return 0\n'
                     '    except ValueError: return 1',
            'call': 'refused(twisted_telescopic_space)',
            'gold_call': 'refused(_oracle_twisted_telescopic_space)',
            'tol': 1e-12},
        {   'setup': 'def _pack_test_result(z):\n'
                     '    import numpy as np\n'
                     '    b = np.asarray(z[-1])\n'
                     '    shape = np.asarray((b.ndim, *b.shape), dtype=float)\n'
                     '    arrays = [np.asarray(x) for x in z]\n'
                     '    return np.concatenate([shape] + [x.reshape(-1).real.astype(float) for x in '
                     'arrays] + [x.reshape(-1).imag.astype(float) for x in arrays])\n'
                     'import numpy as np\n'
                     'd=3; k=2; omega=1j; tol=float("nan")\n'
                     'def refused(f):\n'
                     '    try: f(d,k,omega,tol=tol); return 0\n'
                     '    except ValueError: return 1',
            'call': 'refused(twisted_telescopic_space)',
            'gold_call': 'refused(_oracle_twisted_telescopic_space)',
            'tol': 1e-12},
        {   'setup': 'def _pack_test_result(z):\n'
                     '    import numpy as np\n'
                     '    b = np.asarray(z[-1])\n'
                     '    shape = np.asarray((b.ndim, *b.shape), dtype=float)\n'
                     '    arrays = [np.asarray(x) for x in z]\n'
                     '    return np.concatenate([shape] + [x.reshape(-1).real.astype(float) for x in '
                     'arrays] + [x.reshape(-1).imag.astype(float) for x in arrays])\n'
                     'import numpy as np\n'
                     'd=3; k=1; omega=1+0j',
            'call': '_pack_test_result(twisted_telescopic_space(d,k,omega))',
            'gold_call': '_pack_test_result(_oracle_twisted_telescopic_space(d,k,omega))',
            'tol': 1e-08}]
