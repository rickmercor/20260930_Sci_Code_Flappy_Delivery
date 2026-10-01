"""
Compute the overlap, kinetic-energy and nuclear-attraction matrices of a molecule over contracted Cartesian s- and p-type Gaussian basis functions.

Electronic-structure methods built on an atom-centred Gaussian basis start from the
one-electron integrals. The overlap matrix defines the metric of the non-orthogonal basis, and
the core Hamiltonian (kinetic energy plus attraction to all nuclei) is the one-electron part of
every Fock matrix in the later steps. Basis-set contraction coefficients are tabulated for
normalised primitives, and the contracted functions are renormalised to unit self-overlap.

Returns
-------
np.ndarray of shape (3, nbf, nbf): the overlap, kinetic-energy and nuclear-attraction matrices stacked in that order (kinetic and nuclear attraction in hartree)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_one_electron_integrals(coords: "np.ndarray", nuclear_charges: "np.ndarray", shells: list) -> "np.ndarray":
    '''Overlap, kinetic-energy and nuclear-attraction matrices over contracted s/p Gaussians.

    Parameters
    ----------
    coords : np.ndarray
        Nuclear positions in bohr, shape (n_atoms, 3).
    nuclear_charges : np.ndarray
        Nuclear charges Z_A (atomic units), shape (n_atoms,). They enter only the
        nuclear-attraction matrix.
    shells : list of dict
        Basis shells in basis-function order. Each dict has the keys
        ``"center"`` (int, row index into ``coords``), ``"l"`` (int, 0 for s or 1 for p),
        ``"exponents"`` (sequence of positive floats, bohr^-2) and ``"coefficients"``
        (sequence of floats of the same length). The coefficients multiply normalised
        primitive Gaussians; every contracted function is then renormalised to unit
        self-overlap. An s shell contributes one basis function and a p shell three,
        ordered x, y, z. An SP shell of a basis-set library is supplied as a separate
        s shell and p shell with their own coefficients.

    Returns
    -------
    ints : np.ndarray
        Array of shape (3, nbf, nbf) with ints[0] the overlap matrix S, ints[1] the
        kinetic-energy matrix T and ints[2] the nuclear-attraction matrix V (hartree),
        which contains the attraction of an electron to every nucleus. All three are real
        symmetric; nbf is the
        total number of basis functions.

    Raises
    ------
    ValueError
        If ``coords`` is not of shape (n_atoms, 3), the charges do not match the atoms,
        a shell has l other than 0 or 1, an invalid centre index, or empty or
        mismatched exponent and coefficient lists, or a non-positive exponent.
    '''
    return ints

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import gamma, gammainc


def _cartesian_components(l):
    """Cartesian exponent triples of a shell; a p shell is ordered x, y, z."""
    return [(lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)]


def _boys(n_max, t):
    """Boys functions F_n(t) for n = 0..n_max; result has shape (n_max + 1,) + t.shape."""
    t = np.asarray(t, dtype=float)
    out = np.empty((n_max + 1,) + t.shape)
    small = t < 1e-8
    t_safe = np.where(small, 1.0, t)
    for n in range(n_max + 1):
        a = n + 0.5
        series = 1.0 / (2 * n + 1) - t / (2 * n + 3) + t * t / (2.0 * (2 * n + 5))
        out[n] = np.where(small, series, gamma(a) * gammainc(a, t_safe) / (2.0 * t_safe ** a))
    return out


def _normalized_shells(coords, shells):
    """Validate the shells and fold primitive and contraction normalisation into the coefficients."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    out = []
    for shell in shells:
        center, l = int(shell["center"]), int(shell["l"])
        exps = np.asarray(shell["exponents"], dtype=float)
        coefs = np.asarray(shell["coefficients"], dtype=float)
        if l not in (0, 1):
            raise ValueError("only s (l = 0) and p (l = 1) shells are supported")
        if not 0 <= center < coords.shape[0]:
            raise ValueError("shell centre index out of range")
        if exps.ndim != 1 or exps.size == 0 or exps.shape != coefs.shape or np.any(exps <= 0.0):
            raise ValueError("exponents and coefficients must be matching non-empty lists, exponents > 0")
        d = coefs * (2.0 * exps / np.pi) ** 0.75 * (4.0 * exps) ** (0.5 * l)
        p = exps[:, None] + exps[None, :]
        self_overlap = np.sum(d[:, None] * d[None, :] * (np.pi / p) ** 1.5 / (2.0 * p) ** l)
        out.append((coords[center].copy(), l, exps, d / np.sqrt(self_overlap)))
    return out


def _hermite_expansion(la, lb, a, b, q):
    """1D Hermite expansion coefficients E[i][j][t] for all primitive pairs (arrays a, b), q = A - B."""
    p = a + b
    xpa, xpb = -b / p * q, a / p * q
    e = [[None] * (lb + 1) for _ in range(la + 1)]
    e[0][0] = np.exp(-a * b / p * q * q)[None, :]
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == 0 and j == 0:
                continue
            prev, xp = (e[i - 1][j], xpa) if i > 0 else (e[i][j - 1], xpb)
            n = i + j
            cur = np.zeros((n + 1, p.size))
            for t in range(n + 1):
                val = np.zeros(p.size)
                if t >= 1:
                    val = val + 0.5 / p * prev[t - 1]
                if t <= n - 1:
                    val = val + xp * prev[t]
                if t + 1 <= n - 1:
                    val = val + (t + 1) * prev[t + 1]
                cur[t] = val
            e[i][j] = cur
    return e


def _hermite_coulomb(l_max, alpha, x, y, z):
    """Hermite Coulomb integrals R_tuv(alpha, (x, y, z)) for t + u + v <= l_max; shape (L+1, L+1, L+1, ...)."""
    shape = np.broadcast(alpha, x, y, z).shape
    boys = _boys(l_max, alpha * (x * x + y * y + z * z))
    r = np.zeros((l_max + 1,) * 4 + shape)
    factor = np.ones(shape)
    for n in range(l_max + 1):
        r[n, 0, 0, 0] = factor * boys[n]
        factor = factor * (-2.0 * alpha)
    for order in range(1, l_max + 1):
        for t in range(order + 1):
            for u in range(order - t + 1):
                v = order - t - u
                for n in range(l_max - order + 1):
                    if t > 0:
                        val = x * r[n + 1, t - 1, u, v]
                        if t > 1:
                            val = val + (t - 1) * r[n + 1, t - 2, u, v]
                    elif u > 0:
                        val = y * r[n + 1, t, u - 1, v]
                        if u > 1:
                            val = val + (u - 1) * r[n + 1, t, u - 2, v]
                    else:
                        val = z * r[n + 1, t, u, v - 1]
                        if v > 1:
                            val = val + (v - 1) * r[n + 1, t, u, v - 2]
                    r[n, t, u, v] = val
    return r[0]


def _shell_pair(shell_a, shell_b, extra=0):
    """Primitive-pair data and Hermite expansion of the product of two shells."""
    center_a, la, exps_a, coefs_a = shell_a
    center_b, lb, exps_b, coefs_b = shell_b
    a = np.repeat(exps_a, exps_b.size)
    b = np.tile(exps_b, exps_a.size)
    p = a + b
    q = center_a - center_b
    e = [_hermite_expansion(la, lb + extra, a, b, np.full_like(a, q[k])) for k in range(3)]
    comps_a, comps_b = _cartesian_components(la), _cartesian_components(lb)
    l_sum = la + lb
    tuv = [(t, u, v) for t in range(l_sum + 1) for u in range(l_sum + 1 - t)
           for v in range(l_sum + 1 - t - u)]
    eh = np.zeros((len(comps_a) * len(comps_b), len(tuv), a.size))
    k = 0
    for ax, ay, az in comps_a:
        for bx, by, bz in comps_b:
            for m, (t, u, v) in enumerate(tuv):
                if t <= ax + bx and u <= ay + by and v <= az + bz:
                    eh[k, m] = e[0][ax][bx][t] * e[1][ay][by][u] * e[2][az][bz][v]
            k += 1
    return {"a": a, "b": b, "p": p, "lb": lb, "l_sum": l_sum, "tuv": tuv, "e": e, "eh": eh,
            "center": (a[:, None] * center_a[None, :] + b[:, None] * center_b[None, :]) / p[:, None],
            "cc": np.repeat(coefs_a, exps_b.size) * np.tile(coefs_b, exps_a.size),
            "comps_a": comps_a, "comps_b": comps_b}


def _shell_offsets(norm_shells):
    offsets = [0]
    for _, l, _, _ in norm_shells:
        offsets.append(offsets[-1] + (l + 1) * (l + 2) // 2)
    return offsets


def _oracle_compute_one_electron_integrals(coords: "np.ndarray", nuclear_charges: "np.ndarray", shells: list) -> "np.ndarray":
    coords = np.asarray(coords, dtype=float)
    charges = np.asarray(nuclear_charges, dtype=float).ravel()
    norm_shells = _normalized_shells(coords, shells)
    if charges.size != coords.shape[0]:
        raise ValueError("one nuclear charge per atom is required")
    offsets = _shell_offsets(norm_shells)
    nbf = offsets[-1]
    s_mat, t_mat, v_mat = np.zeros((nbf, nbf)), np.zeros((nbf, nbf)), np.zeros((nbf, nbf))
    for i, shell_a in enumerate(norm_shells):
        for j, shell_b in enumerate(norm_shells[: i + 1]):
            pair = _shell_pair(shell_a, shell_b, extra=2)
            p, b, e = pair["p"], pair["b"], pair["e"]
            pref = pair["cc"] * (np.pi / p) ** 1.5
            for ia, (ax, ay, az) in enumerate(pair["comps_a"]):
                for ib, (bx, by, bz) in enumerate(pair["comps_b"]):
                    ov, kin = [], []
                    for d, (la_d, lb_d) in enumerate(((ax, bx), (ay, by), (az, bz))):
                        ov.append(e[d][la_d][lb_d][0])
                        kd = -2.0 * b * b * e[d][la_d][lb_d + 2][0] + b * (2 * lb_d + 1) * e[d][la_d][lb_d][0]
                        if lb_d >= 2:
                            kd = kd - 0.5 * lb_d * (lb_d - 1) * e[d][la_d][lb_d - 2][0]
                        kin.append(kd)
                    s_mat[offsets[i] + ia, offsets[j] + ib] = np.sum(pref * ov[0] * ov[1] * ov[2])
                    t_mat[offsets[i] + ia, offsets[j] + ib] = np.sum(
                        pref * (kin[0] * ov[1] * ov[2] + ov[0] * kin[1] * ov[2] + ov[0] * ov[1] * kin[2]))
            vals = np.zeros(pair["eh"].shape[0])
            for nucleus, z in zip(coords, charges):
                pc = pair["center"] - nucleus[None, :]
                r = _hermite_coulomb(pair["l_sum"], p, pc[:, 0], pc[:, 1], pc[:, 2])
                rv = np.array([r[t, u, v] for (t, u, v) in pair["tuv"]])
                vals -= z * np.einsum("ktp,tp,p->k", pair["eh"], rv, pair["cc"] * 2.0 * np.pi / p)
            nb = len(pair["comps_b"])
            for ia in range(len(pair["comps_a"])):
                for ib in range(nb):
                    v_mat[offsets[i] + ia, offsets[j] + ib] = vals[ia * nb + ib]
    lower = np.tril_indices(nbf, -1)
    for m in (s_mat, t_mat, v_mat):
        m[lower[1], lower[0]] = m[lower]
    return np.stack([s_mat, t_mat, v_mat])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    sto3g = """import copy
import numpy as np
H_S = ([3.425250914, 0.6239137298, 0.1688554040], [0.1543289673, 0.5353281423, 0.4446345422])
O_S = ([130.7093214, 23.80886605, 6.443608313], [0.1543289673, 0.5353281423, 0.4446345422])
O_SP = [5.033151319, 1.169596125, 0.3803889600]
N_S = ([99.10616896, 18.05231239, 4.885660238], [0.1543289673, 0.5353281423, 0.4446345422])
N_SP = [3.780455879, 0.8784966449, 0.2857143744]
SP_S = [-0.09996722919, 0.3995128261, 0.7001154689]
SP_P = [0.1559162750, 0.6076837186, 0.3919573931]
def shell(center, l, exps, coefs):
    return {"center": center, "l": l, "exponents": list(exps), "coefficients": list(coefs)}
def heavy_shells(center, core, sp):
    return [shell(center, 0, *core), shell(center, 0, sp, SP_S), shell(center, 1, sp, SP_P)]
"""
    return [
        # --- Normal: H2O / STO-3G (s and p shells, 7 functions), equilibrium-like geometry ---
        {
            "setup": sto3g + """
coords = np.array([[0.0, -0.143225816552, 0.0],
                   [1.638036840407, 1.136548822547, 0.0],
                   [-1.638036840407, 1.136548822547, 0.0]])
charges = np.array([8.0, 1.0, 1.0])
shells = heavy_shells(0, O_S, O_SP) + [shell(1, 0, *H_S), shell(2, 0, *H_S)]
""",
            "call": "compute_one_electron_integrals(coords.copy(), charges.copy(), copy.deepcopy(shells))",
            "gold_call": "_oracle_compute_one_electron_integrals(coords.copy(), charges.copy(), copy.deepcopy(shells))",
            "tol": 1e-8,
        },
        # --- Normal: NH3 / STO-3G, non-planar geometry with no atom on the axes ---
        {
            "setup": sto3g + """
coords = np.array([[0.10, -0.05, 0.12],
                   [1.87, 0.02, -0.55],
                   [-0.86, 1.60, -0.50],
                   [-0.90, -1.55, -0.62]])
charges = np.array([7.0, 1.0, 1.0, 1.0])
shells = heavy_shells(0, N_S, N_SP) + [shell(k, 0, *H_S) for k in (1, 2, 3)]
""",
            "call": "compute_one_electron_integrals(coords.copy(), charges.copy(), copy.deepcopy(shells))",
            "gold_call": "_oracle_compute_one_electron_integrals(coords.copy(), charges.copy(), copy.deepcopy(shells))",
            "tol": 1e-8,
        },
        # --- Boundary: a single contracted s function on one nucleus (1 x 1 matrices) ---
        {
            "setup": sto3g + """
coords = np.array([[0.3, -0.2, 0.7]])
charges = np.array([1.0])
shells = [shell(0, 0, *H_S)]
""",
            "call": "compute_one_electron_integrals(coords.copy(), charges.copy(), copy.deepcopy(shells))",
            "gold_call": "_oracle_compute_one_electron_integrals(coords.copy(), charges.copy(), copy.deepcopy(shells))",
            "tol": 1e-8,
        },
        # --- Edge: uncontracted p shells on two off-axis centres, an extra point charge,
        #     and a ghost-like centre carrying a basis function but zero charge ---
        {
            "setup": sto3g + """
coords = np.array([[0.0, 0.0, 0.0],
                   [0.7, -1.1, 0.9],
                   [-1.3, 0.4, 0.5]])
charges = np.array([3.0, 1.5, 0.0])
shells = [shell(0, 1, [0.8], [1.0]), shell(1, 1, [0.35], [1.0]),
          shell(2, 0, [0.5, 0.12], [0.6, 0.5]), shell(1, 0, [1.7], [1.0])]
""",
            "call": "compute_one_electron_integrals(coords.copy(), charges.copy(), copy.deepcopy(shells))",
            "gold_call": "_oracle_compute_one_electron_integrals(coords.copy(), charges.copy(), copy.deepcopy(shells))",
            "tol": 1e-8,
        },
    ]
