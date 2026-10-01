"""
Step 02: Lowest singlet and natural occupations of a half-filled Hubbard model. Lowest singlet state of a Hubbard-type pi-electron model at half filling, with its natural orbital occupations.

Bond breaking and forbidden pericyclic transition states are exactly the situations in which a single closed-shell determinant fails: when two frontier orbitals become nearly degenerate, the ground state becomes a mixture of the doubly occupied and the doubly excited configurations and acquires open-shell singlet character. The simplest many-body model that captures this is a tight-binding pi Hamiltonian with an on-site repulsion between electrons of opposite spin, solved exactly in the space of all determinants. Its lowest singlet is characterised by the natural orbitals, the eigenvectors of the spin-summed one-particle density matrix; their occupations lie between 0 and 2 and move away from 2 and 0 as static correlation grows. Low-lying triplet states of the same model can come close to or below the singlet, so the spin state has to be checked rather than assumed.

Returns
-------
numpy.ndarray of length n + 1: [lowest singlet energy in eV, natural orbital occupations in descending order]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def singlet_ground_state(h: "np.ndarray", U: float) -> "np.ndarray":
    '''Energy and natural orbital occupations of the lowest singlet of a half-filled Hubbard model.

    Parameters
    ----------
    h : np.ndarray
        Real symmetric (n, n) one-electron matrix in eV, n even with 2 <= n <= 10, finite.
    U : float
        On-site repulsion in eV, finite and >= 0.

    Returns
    -------
    res : np.ndarray
        Real array of length n + 1: res[0] is the energy in eV of the lowest-energy eigenstate with total spin S = 0 of
        H = sum_{i,j,sigma} h_ij c+_{i sigma} c_{j sigma} + U sum_i n_{i up} n_{i down} with n electrons (n/2 of each
        spin), and res[1:] are the eigenvalues of its spin-summed one-particle density matrix
        gamma_ij = sum_sigma <c+_{i sigma} c_{j sigma}> in descending order. Inputs are such that this singlet is
        nondegenerate.

    Raises
    ------
    ValueError
        If h is not a finite symmetric square matrix of even size between 2 and 10, U is not finite or U < 0, or no
        singlet is found among the six lowest states with n/2 electrons of each spin.
    '''
    return res

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _string_tables(n, k):
    import numpy as np
    from itertools import combinations
    cache = _string_tables.__dict__.setdefault("cache", {})
    if (n, k) not in cache:
        strs = [sum(1 << i for i in c) for c in combinations(range(n), k)]
        pos = {x: i for i, x in enumerate(strs)}
        rows = []
        for i, x in enumerate(strs):
            for p in range(n):
                if not x >> p & 1:
                    continue
                for q in range(n):
                    if q == p or x >> q & 1:
                        continue
                    y = x ^ (1 << p) ^ (1 << q)
                    lo, hi = min(p, q), max(p, q)
                    sign = -1.0 if bin(x & ((1 << hi) - (1 << (lo + 1)))).count("1") % 2 else 1.0
                    rows.append((i, pos[y], q, p, sign))
        occ = np.array([[x >> p & 1 for p in range(n)] for x in strs], dtype=float)
        cache[(n, k)] = (strs, pos, occ, np.array(rows, dtype=float).reshape(-1, 5))
    return cache[(n, k)]


def _check_h_U(h, U):
    import numpy as np
    h = np.asarray(h, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] < 2 or h.shape[0] % 2 or h.shape[0] > 10:
        raise ValueError("h must be a square matrix of even size between 2 and 10")
    if not np.all(np.isfinite(h)) or not np.allclose(h, h.T, rtol=0.0, atol=1e-12):
        raise ValueError("h must be finite and symmetric")
    if not np.isfinite(U) or U < 0.0:
        raise ValueError("U must be finite and >= 0")
    return h, float(U)


def _fci_states(h, U, nroots):
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    n = h.shape[0]
    k = n // 2
    strs, pos, occ, rows = _string_tables(n, k)
    m = len(strs)
    src, tgt, q, p, sgn = rows[:, 0].astype(int), rows[:, 1].astype(int), rows[:, 2].astype(int), rows[:, 3].astype(int), rows[:, 4]
    A = sp.coo_matrix((sgn * h[q, p], (tgt, src)), shape=(m, m)).tocsr() + sp.diags(occ @ np.diag(h))
    I = sp.identity(m, format="csr")
    H = sp.kron(A, I, format="csr") + sp.kron(I, A, format="csr") + sp.diags((U * occ @ occ.T).ravel())
    if m * m <= 100:
        E, V = np.linalg.eigh(H.toarray())
        return E[:nroots], V[:, :nroots]
    E, V = spl.eigsh(H, k=nroots, which="SA", tol=1e-14, v0=np.ones(m * m))
    o = np.argsort(E)
    return E[o], V[:, o]


def _spin_squared(c, n):
    import numpy as np
    k = n // 2
    strs, pos, occ, _ = _string_tables(n, k)
    up_strs, up_pos, _, _ = _string_tables(n, k + 1)
    dn_strs, dn_pos, _, _ = _string_tables(n, k - 1)
    m = len(strs)
    C = c.reshape(m, m)
    out = np.zeros((len(up_strs), len(dn_strs)))
    for ia, a in enumerate(strs):
        for ib, b in enumerate(strs):
            if C[ia, ib] == 0.0:
                continue
            for p in range(n):
                if (b >> p & 1) and not (a >> p & 1):
                    na, nb = a | (1 << p), b ^ (1 << p)
                    sign = (-1.0) ** (bin(a & ((1 << p) - 1)).count("1") + bin(b & ((1 << p) - 1)).count("1") + bin(a).count("1"))
                    out[up_pos[na], dn_pos[nb]] += sign * C[ia, ib]
    return float(np.sum(out * out))


def _one_rdm(c, n):
    import numpy as np
    k = n // 2
    strs, pos, occ, rows = _string_tables(n, k)
    m = len(strs)
    C = c.reshape(m, m)
    Pa = C @ C.T
    Pb = C.T @ C
    g = np.diag(occ.T @ np.diag(Pa) + occ.T @ np.diag(Pb))
    src, tgt, q, p, sgn = rows[:, 0].astype(int), rows[:, 1].astype(int), rows[:, 2].astype(int), rows[:, 3].astype(int), rows[:, 4]
    np.add.at(g, (q, p), sgn * (Pa[tgt, src] + Pb[tgt, src]))
    return 0.5 * (g + g.T)


def _oracle_singlet_ground_state(h: "np.ndarray", U: float) -> "np.ndarray":
    import numpy as np
    h, U = _check_h_U(h, U)
    n = h.shape[0]
    nroots = min(6, (len(_string_tables(n, n // 2)[0])) ** 2)
    E, V = _fci_states(h, U, nroots)
    for i in range(len(E)):
        if _spin_squared(V[:, i], n) < 0.5:
            occ = np.sort(np.linalg.eigvalsh(_one_rdm(V[:, i], n)))[::-1]
            return np.concatenate([[E[i]], occ])
    raise ValueError("no singlet among the lowest states")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = ("import numpy as np\n"
            "def _ring(n, t, tn):\n    h = np.zeros((n, n))\n    for i in range(n - 1):\n        h[i, i + 1] = h[i + 1, i] = t[i % len(t)]\n"
            "    h[0, n - 1] = h[n - 1, 0] = tn\n    return h\n")
    return [
        # --- Normal: hexatriene-like ring with a weak antibonding end coupling (Moebius-like) ---
        {"setup": base, "call": "np.asarray(singlet_ground_state(_ring(6, [-1.9, -2.2], 0.9), 4.0), dtype=float)",
         "gold_call": "np.asarray(_oracle_singlet_ground_state(_ring(6, [-1.9, -2.2], 0.9), 4.0), dtype=float)", "tol": 1e-9},
        # --- Normal: same ring with a bonding end coupling (Hueckel-like) and site energies ---
        {"setup": base + "H = _ring(6, [-1.9, -2.2], -0.9)\nH[0, 0] = 0.4\nH[5, 5] = -0.3\n",
         "call": "np.asarray(singlet_ground_state(H, 3.0), dtype=float)",
         "gold_call": "np.asarray(_oracle_singlet_ground_state(H, 3.0), dtype=float)", "tol": 1e-9},
        # --- Normal: four-site chain, strong repulsion ---
        {"setup": base, "call": "np.asarray(singlet_ground_state(_ring(4, [-2.8, -2.2], -0.3), 7.5), dtype=float)",
         "gold_call": "np.asarray(_oracle_singlet_ground_state(_ring(4, [-2.8, -2.2], -0.3), 7.5), dtype=float)", "tol": 1e-9},
        # --- Boundary: no repulsion, a single determinant with occupations 2 and 0 ---
        {"setup": base, "call": "np.asarray(singlet_ground_state(_ring(6, [-2.4, -2.0], -0.5), 0.0), dtype=float)",
         "gold_call": "np.asarray(_oracle_singlet_ground_state(_ring(6, [-2.4, -2.0], -0.5), 0.0), dtype=float)", "tol": 1e-9},
        # --- Boundary: six-site ring with nonuniform site energies and a frustrating next-nearest coupling ---
        {"setup": base + "H = _ring(6, [-2.5, -2.1], 0.4)\nH[0, 2] = H[2, 0] = 0.35\nH[1, 1] = 0.6\nH[4, 4] = -0.2\n",
         "call": "np.asarray(singlet_ground_state(H, 3.5), dtype=float)",
         "gold_call": "np.asarray(_oracle_singlet_ground_state(H, 3.5), dtype=float)", "tol": 1e-8},
        # --- Edge: two-site hydrogen-molecule model ---
        {"setup": base, "call": "np.asarray(singlet_ground_state(np.array([[0.0, -1.5], [-1.5, 0.0]]), 5.0), dtype=float)",
         "gold_call": "np.asarray(_oracle_singlet_ground_state(np.array([[0.0, -1.5], [-1.5, 0.0]]), 5.0), dtype=float)", "tol": 1e-10},
        # --- Edge: trimethylenemethane-like star, whose lowest state is a triplet ---
        {"setup": base + "H = np.zeros((4, 4))\nfor j, t in ((1, -2.4), (2, -2.1), (3, -2.7)):\n    H[0, j] = H[j, 0] = t\n",
         "call": "np.asarray(singlet_ground_state(H, 3.0), dtype=float)",
         "gold_call": "np.asarray(_oracle_singlet_ground_state(H, 3.0), dtype=float)", "tol": 1e-9},
        # --- Error: odd number of sites ---
        {"setup": base + "def _probe(fn):\n    try:\n        fn(np.zeros((3, 3)), 1.0)\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(singlet_ground_state)", "gold_call": "_probe(_oracle_singlet_ground_state)"},
    ]
