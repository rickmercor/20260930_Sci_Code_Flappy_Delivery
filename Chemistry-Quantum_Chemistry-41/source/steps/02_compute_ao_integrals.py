"""
Implement compute_ao_integrals, which evaluates the overlap, core-Hamiltonian and
electron-repulsion integrals and the nuclear-repulsion energy of a molecule over a basis of
contracted real solid-harmonic Gaussian functions with s, p and d angular momentum.

Electronic-structure calculations in a Gaussian basis require the one- and two-electron
integrals over the basis functions.

Returns
-------
tuple (overlap (n_bf, n_bf), core_hamiltonian (n_bf, n_bf), eri (n_bf, n_bf, n_bf, n_bf) in chemists' notation, e_nuc float) over solid-harmonic s/p/d functions, in hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_ao_integrals(charges: "np.ndarray", coords: "np.ndarray", shells: list) -> tuple:
    '''Atomic-orbital integrals of a molecule in a contracted solid-harmonic Gaussian basis.

    Parameters
    ----------
    charges : np.ndarray
        Nuclear charges Z_A, shape (n_atoms,), all positive.
    coords : np.ndarray
        Nuclear positions in bohr, shape (n_atoms, 3).
    shells : list
        Basis shells in the order that defines the atomic-orbital ordering. Each entry is a
        tuple (atom, l, exponents, coefficients): atom is the row of coords the shell is
        centred on, l its angular momentum (0, 1 or 2), exponents and coefficients the
        primitive exponents and tabulated contraction coefficients in the convention of
        normalize_contraction. A shell contributes 2l + 1 functions. Each function is the
        product of an angular polynomial in the displacement (x, y, z) from its atom, the
        radial sum sum_k w_k exp(-beta_k r^2) with the weights w_k of normalize_contraction,
        and a positive factor giving unit self-overlap. The polynomials, in this order, are
        1 for s; x, y, z for p; and xy, yz, 2z^2 - x^2 - y^2, xz, x^2 - y^2 for d.

    Returns
    -------
    integrals : tuple
        (overlap, core_hamiltonian, eri, e_nuc):
        overlap : np.ndarray, shape (n_bf, n_bf), <mu|nu>.
        core_hamiltonian : np.ndarray, shape (n_bf, n_bf), electron kinetic energy plus the
            attraction to all nuclei, in hartree.
        eri : np.ndarray, shape (n_bf, n_bf, n_bf, n_bf), electron-repulsion integrals in
            chemists' notation, eri[p, q, r, s] = (pq|rs), in hartree.
        e_nuc : float, nuclear-repulsion energy in hartree.

    Raises
    ------
    ValueError
        If charges and coords describe different numbers of atoms, two nuclei coincide, a
        shell has an angular momentum other than 0, 1 or 2 or refers to a missing atom, or a
        shell's exponents and coefficients are invalid as in normalize_contraction.
    '''
    return (overlap, core_hamiltonian, eri, e_nuc)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import gamma, gammainc

def _solid_harmonics(l: int) -> "np.ndarray":
    """Solid-harmonic polynomials (rows) on the Cartesian monomials of _cartesian_list(l)."""
    if l == 0:
        return np.array([[1.0]])
    if l == 1:
        return np.eye(3)
    return np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, -1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
    ])


def _cartesian_list(l: int) -> list:
    """Cartesian exponent triples (xx, xy, xz, yy, yz, zz order for l = 2)."""
    return [(lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)]


def _hermite_indices(L: int) -> list:
    return [(t, u, v) for t in range(L + 1) for u in range(L + 1 - t) for v in range(L + 1 - t - u)]


def _boys(n_max: int, t: "np.ndarray") -> "np.ndarray":
    """Boys functions F_0..F_n_max at the points t."""
    t = np.asarray(t, dtype=float)
    small = t < 1e-12
    t_safe = np.where(small, 1.0, t)
    out = np.empty((n_max + 1,) + t.shape)
    for n in range(n_max + 1):
        exact = gamma(n + 0.5) * gammainc(n + 0.5, t_safe) / (2.0 * t_safe ** (n + 0.5))
        out[n] = np.where(small, 1.0 / (2 * n + 1) - t / (2 * n + 3), exact)
    return out


def _hermite_1d(la: int, lb: int, x_ab: float, a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    """E[i, j, t] (i <= la, j <= lb, t <= i + j) of a 1-D Gaussian product, per primitive pair."""
    p = a + b
    x_pa = -b / p * x_ab
    x_pb = a / p * x_ab
    e = np.zeros((la + 1, lb + 1, la + lb + 2) + a.shape)
    e[0, 0, 0] = np.exp(-a * b / p * x_ab * x_ab)
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == 0 and j == 0:
                continue
            for t in range(i + j + 1):
                if i > 0:
                    val = x_pa * e[i - 1, j, t] + (t + 1) * e[i - 1, j, t + 1]
                    if t > 0:
                        val = val + e[i - 1, j, t - 1] / (2.0 * p)
                else:
                    val = x_pb * e[i, j - 1, t] + (t + 1) * e[i, j - 1, t + 1]
                    if t > 0:
                        val = val + e[i, j - 1, t - 1] / (2.0 * p)
                e[i, j, t] = val
    return e[:, :, : la + lb + 1]


def _hermite_coulomb(L: int, alpha: "np.ndarray", x: "np.ndarray", y: "np.ndarray", z: "np.ndarray") -> dict:
    """Hermite Coulomb integrals R_{tuv}, t + u + v <= L, broadcast over the batch."""
    boys = _boys(L, alpha * (x * x + y * y + z * z))
    r = {(0, 0, 0, n): (-2.0 * alpha) ** n * boys[n] for n in range(L + 1)}
    for total in range(1, L + 1):
        for (t, u, v) in _hermite_indices(total):
            if t + u + v != total:
                continue
            for n in range(L - total + 1):
                if t > 0:
                    val = x * r[(t - 1, u, v, n + 1)]
                    if t > 1:
                        val = val + (t - 1) * r[(t - 2, u, v, n + 1)]
                elif u > 0:
                    val = y * r[(t, u - 1, v, n + 1)]
                    if u > 1:
                        val = val + (u - 1) * r[(t, u - 2, v, n + 1)]
                else:
                    val = z * r[(t, u, v - 1, n + 1)]
                    if v > 1:
                        val = val + (v - 1) * r[(t, u, v - 2, n + 1)]
                r[(t, u, v, n)] = val
    return {key[:3]: val for key, val in r.items() if key[3] == 0}


def _prepare_shells(coords: "np.ndarray", shells: list) -> list:
    prepared = []
    for shell in shells:
        atom, l, exponents, coefficients = shell
        if l not in (0, 1, 2):
            raise ValueError("only s, p and d shells (l = 0, 1, 2) are supported")
        if not (0 <= int(atom) < len(coords)) or int(atom) != atom:
            raise ValueError("shell refers to a missing atom")
        weights = _oracle_normalize_contraction(l, exponents, coefficients)
        exponents = np.asarray(exponents, dtype=float).ravel()
        keep = weights != 0.0
        prepared.append({"centre": coords[int(atom)], "l": int(l), "exps": exponents[keep],
                         "w": weights[keep], "cart": _cartesian_list(int(l))})
    return prepared


def _pair_data(sa: dict, sb: dict) -> dict:
    a = np.repeat(sa["exps"], sb["exps"].size)
    b = np.tile(sb["exps"], sa["exps"].size)
    w = np.repeat(sa["w"], sb["exps"].size) * np.tile(sb["w"], sa["exps"].size)
    p = a + b
    centre_p = (a[:, None] * sa["centre"] + b[:, None] * sb["centre"]) / p[:, None]
    ab = sa["centre"] - sb["centre"]
    herm = [_hermite_1d(sa["l"], sb["l"] + 2, ab[k], a, b) for k in range(3)]
    L = sa["l"] + sb["l"]
    hidx = _hermite_indices(L)
    dens = np.zeros((len(sa["cart"]), len(sb["cart"]), len(hidx), p.size))
    for ci, (ix, iy, iz) in enumerate(sa["cart"]):
        for cj, (jx, jy, jz) in enumerate(sb["cart"]):
            for h, (t, u, v) in enumerate(hidx):
                if t <= ix + jx and u <= iy + jy and v <= iz + jz:
                    dens[ci, cj, h] = w * herm[0][ix, jx, t] * herm[1][iy, jy, u] * herm[2][iz, jz, v]
    return {"b": b, "w": w, "p": p, "P": centre_p, "herm": herm, "dens": dens, "hidx": hidx, "L": L}


def _one_electron_block(sa: dict, sb: dict, pair: dict, charges: "np.ndarray", coords: "np.ndarray") -> tuple:
    b, w, p, herm = pair["b"], pair["w"], pair["p"], pair["herm"]
    pref = (np.pi / p) ** 1.5

    def _s1(k, i, j):
        return herm[k][i, j, 0] if j >= 0 else np.zeros_like(p)

    def _t1(k, i, j):
        val = b * (2 * j + 1) * _s1(k, i, j) - 2.0 * b * b * _s1(k, i, j + 2)
        if j >= 2:
            val = val - 0.5 * j * (j - 1) * _s1(k, i, j - 2)
        return val

    na, nb = len(sa["cart"]), len(sb["cart"])
    s_blk = np.zeros((na, nb))
    t_blk = np.zeros((na, nb))
    for ci, lmn_a in enumerate(sa["cart"]):
        for cj, lmn_b in enumerate(sb["cart"]):
            sx, sy, sz = (_s1(k, lmn_a[k], lmn_b[k]) for k in range(3))
            s_blk[ci, cj] = np.sum(w * pref * sx * sy * sz)
            t_blk[ci, cj] = np.sum(w * pref * (_t1(0, lmn_a[0], lmn_b[0]) * sy * sz
                                                + sx * _t1(1, lmn_a[1], lmn_b[1]) * sz
                                                + sx * sy * _t1(2, lmn_a[2], lmn_b[2])))
    v_blk = np.zeros((na, nb))
    for z_c, centre_c in zip(charges, coords):
        pc = pair["P"] - centre_c
        r = _hermite_coulomb(pair["L"], p, pc[:, 0], pc[:, 1], pc[:, 2])
        r_mat = np.array([r[key] for key in pair["hidx"]])
        v_blk -= z_c * np.einsum("abhp,hp,p->ab", pair["dens"], r_mat, 2.0 * np.pi / p)
    return s_blk, t_blk + v_blk


def _eri_block(bra: dict, ket: dict) -> "np.ndarray":
    p = bra["p"][:, None]
    q = ket["p"][None, :]
    alpha = p * q / (p + q)
    pq = bra["P"][:, None, :] - ket["P"][None, :, :]
    r = _hermite_coulomb(bra["L"] + ket["L"], alpha, pq[..., 0], pq[..., 1], pq[..., 2])
    coupling = np.empty((len(bra["hidx"]), len(ket["hidx"])) + alpha.shape)
    for x, (t, u, v) in enumerate(bra["hidx"]):
        for y, (tt, uu, vv) in enumerate(ket["hidx"]):
            coupling[x, y] = (-1) ** (tt + uu + vv) * r[(t + tt, u + uu, v + vv)]
    coupling *= 2.0 * np.pi ** 2.5 / (p * q * np.sqrt(p + q))
    return np.einsum("abhp,hgpq,cdgq->abcd", bra["dens"], coupling, ket["dens"], optimize=True)


def _oracle_compute_ao_integrals(charges: "np.ndarray", coords: "np.ndarray", shells: list) -> tuple:
    charges = np.asarray(charges, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    if charges.size != coords.shape[0]:
        raise ValueError("charges and coords must describe the same atoms")
    e_nuc = 0.0
    for i in range(charges.size):
        for j in range(i):
            distance = np.linalg.norm(coords[i] - coords[j])
            if distance == 0.0:
                raise ValueError("two nuclei coincide")
            e_nuc += charges[i] * charges[j] / distance
    prepared = _prepare_shells(coords, shells)
    offsets = np.cumsum([0] + [len(sh["cart"]) for sh in prepared])
    n_cart = offsets[-1]
    overlap = np.zeros((n_cart, n_cart))
    core = np.zeros((n_cart, n_cart))
    pairs = {}
    for i in range(len(prepared)):
        for j in range(i + 1):
            pair = _pair_data(prepared[i], prepared[j])
            pairs[(i, j)] = pair
            s_blk, h_blk = _one_electron_block(prepared[i], prepared[j], pair, charges, coords)
            rows, cols = slice(offsets[i], offsets[i + 1]), slice(offsets[j], offsets[j + 1])
            overlap[rows, cols], overlap[cols, rows] = s_blk, s_blk.T
            core[rows, cols], core[cols, rows] = h_blk, h_blk.T
    eri = np.zeros((n_cart, n_cart, n_cart, n_cart))
    keys = sorted(pairs)
    for n1, (i, j) in enumerate(keys):
        for (k, l) in keys[: n1 + 1]:
            blk = _eri_block(pairs[(i, j)], pairs[(k, l)])
            si, sj, sk, sl = (slice(offsets[x], offsets[x + 1]) for x in (i, j, k, l))
            eri[si, sj, sk, sl] = blk
            eri[sj, si, sk, sl] = blk.transpose(1, 0, 2, 3)
            eri[si, sj, sl, sk] = blk.transpose(0, 1, 3, 2)
            eri[sj, si, sl, sk] = blk.transpose(1, 0, 3, 2)
            eri[sk, sl, si, sj] = blk.transpose(2, 3, 0, 1)
            eri[sl, sk, si, sj] = blk.transpose(3, 2, 0, 1)
            eri[sk, sl, sj, si] = blk.transpose(2, 3, 1, 0)
            eri[sl, sk, sj, si] = blk.transpose(3, 2, 1, 0)
    n_sph = sum(2 * sh["l"] + 1 for sh in prepared)
    to_sph = np.zeros((n_cart, n_sph))
    col = 0
    for x, sh in enumerate(prepared):
        block = _solid_harmonics(sh["l"])
        to_sph[offsets[x]:offsets[x + 1], col:col + block.shape[0]] = block.T
        col += block.shape[0]
    to_sph = to_sph / np.sqrt(np.diag(to_sph.T @ overlap @ to_sph))[None, :]
    overlap_sph = to_sph.T @ overlap @ to_sph
    core_sph = to_sph.T @ core @ to_sph
    eri_sph = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, to_sph, to_sph, to_sph, to_sph, optimize=True)
    return overlap_sph, core_sph, eri_sph, float(e_nuc)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    basis = """import numpy as np
from copy import deepcopy as dc
def flatten(result):
    overlap, core, eri, e_nuc = result
    return np.concatenate([np.ravel(overlap), np.ravel(core), np.ravel(eri), [float(e_nuc)]])
N_AUG = [
    (0, [1027.828458, 188.4512226, 52.72186097, 18.11138217, 7.033179691, 2.896651794],
        [0.0091635963, 0.0493614929, 0.1685383049, 0.3705627997, 0.4164915298, 0.1303340841]),
    (0, [0.1846836552, 0.402911141, 0.9277239437, 2.411325783, 7.758467071, 39.19880787],
        [0.2407061763, 0.5951172526, 0.2502417861, -0.0337853715, -0.0469917101, -0.0132527881]),
    (0, [0.09, 0.1846836552, 0.402911141, 0.9277239437],
        [-155.7296559966, 61.8126687524, -69.8698010234, -5.5545900093]),
    (1, [0.1846836552, 0.402911141, 0.9277239437, 2.411325783, 7.758467071, 39.19880787],
        [0.1017082955, 0.4258595477, 0.4180364347, 0.1738967435, 0.0376793698, 0.0037596966]),
    (1, [0.09, 0.1846836552, 0.402911141, 0.9277239437],
        [-0.0840929047, -0.0859046913, 0.0449394729, 0.150796203]),
    (2, [0.09, 0.1846836552, 0.402911141, 0.9277239437],
        [-0.0966051188, -0.131498833, 0.0681453767, -0.1191458075]),
]
H_AUG = [
    (0, [0.100112428, 0.2430767471, 0.6259552659, 1.822142904, 6.513143725, 35.52322122],
        [0.1303340841, 0.4164915298, 0.3705627997, 0.1685383049, 0.0493614929, 0.0091635963]),
    (0, [0.0654284286, 0.100112428, 0.2430767471, 0.6259552659],
        [4.8518572723, 0.4054347007, 0.0547707665, 0.1436562367]),
    (0, [0.0654284286, 0.100112428, 0.2430767471, 0.6259552659],
        [1.746076269, -5.7363906541, -0.3296656157, 0.1098876093]),
    (1, [0.0654284286, 0.100112428, 0.2430767471, 0.6259552659],
        [0.4762021876, -0.1110297174, 5.00489e-05, 0.0083992741]),
]
def on_atom(atom, entries):
    return [(atom, l, list(e), list(c)) for (l, e, c) in entries]
"""
    invalid = """
def run_model():
    try:
        compute_ao_integrals(dc(charges), dc(coords), dc(shells))
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_compute_ao_integrals(dc(charges), dc(coords), dc(shells))
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Typical: an N-H pair off the coordinate axes: contracted p and d shells sharing their
        #     exponents on N (d-d, d-p and p-p integrals), a contracted s shell on H ---
        {
            "setup": basis + """
charges = np.array([7.0, 1.0])
coords = np.array([[0.3, -0.2, 0.1], [-0.6, 0.9, -1.4]])
shells = on_atom(0, [N_AUG[4], N_AUG[5]]) + on_atom(1, [H_AUG[1]])
""",
            "call": "flatten(compute_ao_integrals(dc(charges), dc(coords), dc(shells)))",
            "gold_call": "flatten(_oracle_compute_ao_integrals(dc(charges), dc(coords), dc(shells)))",
            "tol": 1e-9,
        },
        # --- Boundary: a single nitrogen atom away from the origin (no nuclear repulsion): a
        #     six-primitive core s shell, a diffuse s shell with strongly cancelling coefficients
        #     and a d shell ---
        {
            "setup": basis + """
charges = np.array([7.0])
coords = np.array([[0.4, -0.7, 1.3]])
shells = on_atom(0, [N_AUG[0], N_AUG[2], N_AUG[5]])
""",
            "call": "flatten(compute_ao_integrals(dc(charges), dc(coords), dc(shells)))",
            "gold_call": "flatten(_oracle_compute_ao_integrals(dc(charges), dc(coords), dc(shells)))",
            "tol": 1e-9,
        },
        # --- Typical: bent H-N-H fragment (d functions in the field of three nuclei) with shells
        #     listed out of atomic order ---
        {
            "setup": basis + """
charges = np.array([1.0, 7.0, 1.0])
coords = np.array([[1.5, 1.1, 0.2], [0.0, 0.0, 0.0], [-1.6, 0.9, -0.3]])
shells = on_atom(1, [N_AUG[2], N_AUG[5]]) + on_atom(2, [H_AUG[2]]) + on_atom(0, [H_AUG[2]])
""",
            "call": "flatten(compute_ao_integrals(dc(charges), dc(coords), dc(shells)))",
            "gold_call": "flatten(_oracle_compute_ao_integrals(dc(charges), dc(coords), dc(shells)))",
            "tol": 1e-9,
        },
        # --- Edge: uncontracted d shells on two centres 4.8 bohr apart (one with a negative
        #     coefficient), a very diffuse s function and a two-primitive p shell, nuclear charges
        #     2 and 3 ---
        {
            "setup": basis + """
charges = np.array([2.0, 3.0])
coords = np.array([[0.0, 0.0, 0.0], [1.2, -2.5, 3.9]])
shells = [(0, 2, [0.35], [1.0]), (1, 2, [1.4], [-1.0]), (1, 0, [0.03], [1.0]), (0, 1, [2.2, 0.4], [0.3, 0.8])]
""",
            "call": "flatten(compute_ao_integrals(dc(charges), dc(coords), dc(shells)))",
            "gold_call": "flatten(_oracle_compute_ao_integrals(dc(charges), dc(coords), dc(shells)))",
            "tol": 1e-9,
        },
        # --- Invalid: an f shell is outside the supported angular momenta ---
        {
            "setup": basis + """
charges = np.array([7.0])
coords = np.array([[0.0, 0.0, 0.0]])
shells = on_atom(0, [N_AUG[2]]) + [(0, 3, [0.8], [1.0])]
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: two nuclei at the same position ---
        {
            "setup": basis + """
charges = np.array([1.0, 1.0])
coords = np.array([[0.0, 0.0, 0.7], [0.0, 0.0, 0.7]])
shells = on_atom(0, [H_AUG[0]]) + on_atom(1, [H_AUG[0]])
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
