"""
Implement compute_ao_integrals, which evaluates the one- and two-electron integrals of a
molecule over a basis of contracted Cartesian Gaussian functions.

Every correlated molecular energy in this task is built from the same four integral
quantities in the atomic-orbital basis: the overlap matrix, the core Hamiltonian (electron
kinetic energy plus electron-nucleus attraction), the electron-repulsion integrals and the
nuclear-repulsion energy. The basis is supplied shell by shell; each shell sits on one
nucleus, carries one angular momentum and one contraction of primitive Gaussians, and
expands into its Cartesian components.

Returns
-------
tuple (overlap (n_bf, n_bf), core_hamiltonian (n_bf, n_bf), eri (n_bf, n_bf, n_bf, n_bf), e_nuc float), all energies in hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_ao_integrals(charges: "np.ndarray", coords: "np.ndarray", shells: list) -> tuple:
    '''Atomic-orbital integrals of a molecule in a contracted Cartesian Gaussian basis.

    Parameters
    ----------
    charges : np.ndarray
        Nuclear charges Z_A, shape (n_atoms,), all positive.
    coords : np.ndarray
        Nuclear positions in bohr, shape (n_atoms, 3); no two nuclei coincide.
    shells : list
        Basis shells in the order that defines the atomic-orbital ordering. Each entry is a
        tuple (atom, l, exponents, coefficients): atom is the row of coords the shell is
        centred on, l is the angular momentum (0 for s, 1 for p, 2 for d), exponents is a
        sequence of positive primitive exponents and coefficients an equal-length sequence
        of contraction coefficients of either sign. The coefficients multiply normalized
        primitive Gaussians, and every contracted Cartesian function is normalized to unit
        self-overlap. An s shell contributes one function; a p shell contributes three,
        ordered x, y, z; a d shell contributes six Cartesian functions, ordered xx, xy, xz,
        yy, yz, zz.

    Returns
    -------
    integrals : tuple
        (overlap, core_hamiltonian, eri, e_nuc):
        overlap : np.ndarray, shape (n_bf, n_bf), <mu|nu>.
        core_hamiltonian : np.ndarray, shape (n_bf, n_bf), kinetic energy plus attraction
            to all nuclei, in hartree.
        eri : np.ndarray, shape (n_bf, n_bf, n_bf, n_bf), electron-repulsion integrals in
            chemists' notation, eri[p, q, r, s] = (pq|rs), in hartree.
        e_nuc : float, nuclear-repulsion energy in hartree (0.0 for a single atom).

    Raises
    ------
    ValueError
        If a shell has an angular momentum other than 0, 1 or 2, refers to a missing atom,
        has exponents and coefficients of different lengths, or has a non-positive
        exponent.
    '''
    return (overlap, core_hamiltonian, eri, e_nuc)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import gammainc, gamma


def _cartesian_components(l: int) -> list:
    """Cartesian exponent triples of angular momentum l (x, y, z order for l = 1)."""
    return [(lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)]


def _double_factorial(n: int) -> float:
    return 1.0 if n <= 0 else float(np.prod(np.arange(n, 0, -2)))


def _boys(n_max: int, t: "np.ndarray") -> "np.ndarray":
    """Boys functions F_0..F_n_max at the points t (shape (n_max + 1,) + t.shape)."""
    t = np.asarray(t, dtype=float)
    small = t < 1e-10
    t_safe = np.where(small, 1.0, t)
    out = np.empty((n_max + 1,) + t.shape)
    for n in range(n_max + 1):
        exact = gamma(n + 0.5) * gammainc(n + 0.5, t_safe) / (2.0 * t_safe ** (n + 0.5))
        out[n] = np.where(small, 1.0 / (2 * n + 1) - t / (2 * n + 3), exact)
    return out


def _hermite_expansion(i: int, j: int, x_ab: float, a: "np.ndarray", b: "np.ndarray") -> list:
    """Hermite expansion coefficients E^{ij}_t, t = 0..i+j, of a 1-D Gaussian product."""
    p = a + b
    x_pa = -b / p * x_ab
    x_pb = a / p * x_ab
    cache = {(0, 0, 0): np.exp(-a * b / p * x_ab * x_ab)}

    def _coef(ii, jj, t):
        if ii < 0 or jj < 0 or t < 0 or t > ii + jj:
            return 0.0
        key = (ii, jj, t)
        if key not in cache:
            if ii > 0:
                cache[key] = _coef(ii - 1, jj, t - 1) / (2 * p) + x_pa * _coef(ii - 1, jj, t) + (t + 1) * _coef(ii - 1, jj, t + 1)
            else:
                cache[key] = _coef(ii, jj - 1, t - 1) / (2 * p) + x_pb * _coef(ii, jj - 1, t) + (t + 1) * _coef(ii, jj - 1, t + 1)
        return cache[key]

    return [_coef(i, j, t) * np.ones_like(p) for t in range(i + j + 1)]


def _hermite_coulomb(l_max: int, alpha: "np.ndarray", r: "np.ndarray") -> dict:
    """Hermite Coulomb integrals R_{tuv}, t + u + v <= l_max, for separations r (shape (3,) + batch)."""
    x, y, z = r
    boys = _boys(l_max, alpha * (x * x + y * y + z * z))
    cache = {}

    def _rec(t, u, v, n):
        if t < 0 or u < 0 or v < 0:
            return 0.0
        key = (t, u, v, n)
        if key not in cache:
            if t == u == v == 0:
                cache[key] = (-2.0 * alpha) ** n * boys[n]
            elif t > 0:
                cache[key] = (t - 1) * _rec(t - 2, u, v, n + 1) + x * _rec(t - 1, u, v, n + 1)
            elif u > 0:
                cache[key] = (u - 1) * _rec(t, u - 2, v, n + 1) + y * _rec(t, u - 1, v, n + 1)
            else:
                cache[key] = (v - 1) * _rec(t, u, v - 2, n + 1) + z * _rec(t, u, v - 1, n + 1)
        return cache[key]

    return {(t, u, v): _rec(t, u, v, 0)
            for t in range(l_max + 1) for u in range(l_max + 1 - t) for v in range(l_max + 1 - t - u)}


def _basis_functions(coords: "np.ndarray", shells: list) -> list:
    """Contracted Cartesian functions (centre, exponent triple, exponents, normalized weights)."""
    functions = []
    for shell in shells:
        atom, l, exponents, coefficients = shell
        exponents = np.asarray(exponents, dtype=float).ravel()
        coefficients = np.asarray(coefficients, dtype=float).ravel()
        if l not in (0, 1, 2):
            raise ValueError("only s, p and d shells (l = 0, 1, 2) are supported")
        if not (0 <= atom < len(coords)):
            raise ValueError("shell refers to a missing atom")
        if exponents.size == 0 or exponents.size != coefficients.size:
            raise ValueError("exponents and coefficients must be non-empty and of equal length")
        if np.any(exponents <= 0.0):
            raise ValueError("exponents must be positive")
        for lmn in _cartesian_components(l):
            dfac = np.prod([_double_factorial(2 * k - 1) for k in lmn])
            prim_norm = (2 * exponents / np.pi) ** 0.75 * (4 * exponents) ** (l / 2) / np.sqrt(dfac)
            weights = coefficients * prim_norm
            pair = exponents[:, None] + exponents[None, :]
            self_overlap = np.sum(weights[:, None] * weights[None, :] * dfac * (np.pi / pair) ** 1.5 / (2 * pair) ** l)
            functions.append((coords[atom], lmn, exponents, weights / np.sqrt(self_overlap)))
    return functions


def _primitive_pairs(f_a: tuple, f_b: tuple) -> tuple:
    """Flattened primitive-pair data and Hermite coefficients for two contracted functions."""
    centre_a, lmn_a, exp_a, w_a = f_a
    centre_b, lmn_b, exp_b, w_b = f_b
    a = np.repeat(exp_a, exp_b.size)
    b = np.tile(exp_b, exp_a.size)
    weight = np.repeat(w_a, exp_b.size) * np.tile(w_b, exp_a.size)
    p = a + b
    centre_p = (a * centre_a[:, None] + b * centre_b[:, None]) / p
    hermite = [{jj: _hermite_expansion(lmn_a[k], jj, centre_a[k] - centre_b[k], a, b)
                for jj in range(max(0, lmn_b[k] - 2), lmn_b[k] + 3)} for k in range(3)]
    return b, weight, p, centre_p, hermite


def _oracle_compute_ao_integrals(charges: "np.ndarray", coords: "np.ndarray", shells: list) -> tuple:
    charges = np.asarray(charges, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    if charges.size != coords.shape[0]:
        raise ValueError("charges and coords must describe the same atoms")
    functions = _basis_functions(coords, shells)
    n_bf = len(functions)
    overlap = np.zeros((n_bf, n_bf))
    kinetic = np.zeros((n_bf, n_bf))
    attraction = np.zeros((n_bf, n_bf))
    densities = {}
    for i in range(n_bf):
        for j in range(i + 1):
            b, weight, p, centre_p, hermite = _primitive_pairs(functions[i], functions[j])
            lmn_i, lmn_j = functions[i][1], functions[j][1]

            def _shifted_overlap(shift):
                value = (np.pi / p) ** 1.5
                for k in range(3):
                    jj = lmn_j[k] + shift[k]
                    if jj < 0:
                        return np.zeros_like(p)
                    value = value * hermite[k][jj][0]
                return value

            s_ij = _shifted_overlap((0, 0, 0))
            overlap[i, j] = overlap[j, i] = np.sum(weight * s_ij)
            t_ij = b * (2 * sum(lmn_j) + 3) * s_ij
            for k in range(3):
                up = [0, 0, 0]
                up[k] = 2
                down = [0, 0, 0]
                down[k] = -2
                t_ij = t_ij - 2 * b * b * _shifted_overlap(up) - 0.5 * lmn_j[k] * (lmn_j[k] - 1) * _shifted_overlap(down)
            kinetic[i, j] = kinetic[j, i] = np.sum(weight * t_ij)
            terms = []
            for t in range(lmn_i[0] + lmn_j[0] + 1):
                for u in range(lmn_i[1] + lmn_j[1] + 1):
                    for v in range(lmn_i[2] + lmn_j[2] + 1):
                        terms.append(((t, u, v), weight * hermite[0][lmn_j[0]][t] * hermite[1][lmn_j[1]][u] * hermite[2][lmn_j[2]][v]))
            l_pair = sum(lmn_i) + sum(lmn_j)
            v_ij = 0.0
            for z_c, centre_c in zip(charges, coords):
                r_tuv = _hermite_coulomb(l_pair, p, centre_p - centre_c[:, None])
                v_ij -= z_c * sum(np.sum(c * 2 * np.pi / p * r_tuv[key]) for key, c in terms)
            attraction[i, j] = attraction[j, i] = v_ij
            densities[(i, j)] = (p, centre_p, terms, l_pair)
    eri = np.zeros((n_bf, n_bf, n_bf, n_bf))
    keys = sorted(densities)
    for n1, bra in enumerate(keys):
        p, centre_p, terms_p, l_p = densities[bra]
        for ket in keys[:n1 + 1]:
            q, centre_q, terms_q, l_q = densities[ket]
            pp, qq = p[:, None], q[None, :]
            alpha = pp * qq / (pp + qq)
            r_tuv = _hermite_coulomb(l_p + l_q, alpha, centre_p[:, :, None] - centre_q[:, None, :])
            prefactor = 2 * np.pi ** 2.5 / (pp * qq * np.sqrt(pp + qq))
            value = 0.0
            for (t1, u1, v1), c1 in terms_p:
                for (t2, u2, v2), c2 in terms_q:
                    value += (-1) ** (t2 + u2 + v2) * np.sum(c1[:, None] * c2[None, :] * prefactor * r_tuv[(t1 + t2, u1 + u2, v1 + v2)])
            (i, j), (k, l) = bra, ket
            for idx in ((i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                        (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i)):
                eri[idx] = value
    e_nuc = 0.0
    for a_idx in range(charges.size):
        for b_idx in range(a_idx):
            distance = np.linalg.norm(coords[a_idx] - coords[b_idx])
            if distance == 0.0:
                raise ValueError("two nuclei coincide")
            e_nuc += charges[a_idx] * charges[b_idx] / distance
    return overlap, kinetic + attraction, eri, float(e_nuc)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    helper = """import numpy as np
from copy import deepcopy as dc
def flatten(result):
    overlap, core, eri, e_nuc = result
    return np.concatenate([np.ravel(overlap), np.ravel(core), np.ravel(eri), [float(e_nuc)]])
o_631g = [
    (0, [5484.671660, 825.2349460, 188.0469580, 52.96450000, 16.89757040, 5.799635340],
     [0.001831074430, 0.01395017220, 0.06844507810, 0.2327143360, 0.4701928980, 0.3585208530]),
    (0, [15.53961625, 3.599933586, 1.013761750], [-0.1107775495, -0.1480262627, 1.130767015]),
    (1, [15.53961625, 3.599933586, 1.013761750], [0.07087426823, 0.3397528391, 0.7271585773]),
    (0, [0.2700058226], [1.0]),
    (1, [0.2700058226], [1.0]),
]
h_631g = [
    (0, [18.73113696, 2.825394365, 0.6401216923], [0.03349460434, 0.2347269535, 0.8137573261]),
    (0, [0.1612777588], [1.0]),
]
def on_atom(atom, entries):
    return [(atom, l, e, c) for (l, e, c) in entries]
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
        # --- Typical: water in 6-31G at a bent, off-axis geometry (13 functions) ---
        {
            "setup": helper + """
charges = np.array([8.0, 1.0, 1.0])
coords = np.array([[0.1, -0.2, 0.05], [0.0, 1.5, 1.1], [-0.3, -1.4, 1.0]])
shells = on_atom(0, o_631g) + on_atom(1, h_631g) + on_atom(2, h_631g)
""",
            "call": "compute_ao_integrals(dc(charges), dc(coords), dc(shells))",
            "gold_call": "_oracle_compute_ao_integrals(dc(charges), dc(coords), dc(shells))",
            "tol": 1e-9,
        },
        # --- Boundary: a single atom with s and p shells (no nuclear repulsion, off-origin centre) ---
        {
            "setup": helper + """
charges = np.array([8.0])
coords = np.array([[0.4, -0.7, 1.3]])
shells = on_atom(0, o_631g)
""",
            "call": "compute_ao_integrals(dc(charges), dc(coords), dc(shells))",
            "gold_call": "_oracle_compute_ao_integrals(dc(charges), dc(coords), dc(shells))",
            "tol": 1e-9,
        },
        # --- Edge: stretched H2 with a diffuse s function, contracted and uncontracted p
        #     functions, and shells listed out of atomic order ---
        {
            "setup": helper + """
charges = np.array([1.0, 1.0])
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.2]])
shells = [(1, 0, [0.05], [1.0]), (0, 0, [18.73113696, 2.825394365, 0.6401216923], [0.03349460434, 0.2347269535, 0.8137573261]),
          (1, 1, [0.8], [1.0]), (0, 1, [2.5, 0.35], [0.4, 0.7])]
""",
            "call": "compute_ao_integrals(dc(charges), dc(coords), dc(shells))",
            "gold_call": "_oracle_compute_ao_integrals(dc(charges), dc(coords), dc(shells))",
            "tol": 1e-9,
        },
        # --- Typical: a polarized O-H pair off the axes, with a contracted Cartesian d shell
        #     on O and a p shell on H (d-d, d-p and d-s integrals) ---
        {
            "setup": helper + """
charges = np.array([8.0, 1.0])
coords = np.array([[0.1, -0.2, 0.05], [0.4, 1.3, 1.2]])
shells = [(0, 0, [15.53961625, 3.599933586, 1.013761750], [-0.1107775495, -0.1480262627, 1.130767015]),
          (0, 1, [15.53961625, 3.599933586, 1.013761750], [0.07087426823, 0.3397528391, 0.7271585773]),
          (0, 2, [2.1, 0.45], [0.35, 0.8]),
          (1, 0, [18.73113696, 2.825394365, 0.6401216923], [0.03349460434, 0.2347269535, 0.8137573261]),
          (1, 1, [0.75], [1.0])]
""",
            "call": "compute_ao_integrals(dc(charges), dc(coords), dc(shells))",
            "gold_call": "_oracle_compute_ao_integrals(dc(charges), dc(coords), dc(shells))",
            "tol": 1e-9,
        },
        # --- Invalid: an f shell is outside the supported angular momenta ---
        {
            "setup": helper + """
charges = np.array([8.0])
coords = np.array([[0.0, 0.0, 0.0]])
shells = on_atom(0, o_631g) + [(0, 3, [0.8], [1.0])]
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: exponents and coefficients of different lengths ---
        {
            "setup": helper + """
charges = np.array([1.0, 1.0])
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
shells = [(0, 0, [1.0, 0.2], [1.0]), (1, 0, [1.0], [1.0])]
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
