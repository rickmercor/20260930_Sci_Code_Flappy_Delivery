"""
Compute the full four-index tensor of electron-repulsion integrals, in chemists' notation, over contracted Cartesian s- and p-type Gaussian basis functions.

The two-electron part of the molecular Hamiltonian enters every mean-field and correlated
method through the electron-repulsion integrals over real basis functions, which have eightfold
permutational symmetry. The same basis conventions as for the one-electron integrals apply:
coefficients refer to normalised primitives, and contracted functions have unit self-overlap.

Returns
-------
np.ndarray of shape (nbf, nbf, nbf, nbf): the electron-repulsion integrals in chemists' notation, in hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_electron_repulsion_integrals(coords: "np.ndarray", shells: list) -> "np.ndarray":
    '''Electron-repulsion integrals (mn|ls) over contracted s/p Gaussians, chemists' notation.

    Parameters
    ----------
    coords : np.ndarray
        Nuclear positions (shell centres) in bohr, shape (n_atoms, 3).
    shells : list of dict
        Basis shells in basis-function order, in the same format as for the one-electron
        integrals: keys ``"center"`` (int), ``"l"`` (0 or 1), ``"exponents"`` and
        ``"coefficients"`` (equal-length sequences). The coefficients multiply normalised
        primitives, and each contracted function is renormalised to unit self-overlap.
        An s shell gives one function and a p shell three, ordered x, y, z.

    Returns
    -------
    eri : np.ndarray
        Array of shape (nbf, nbf, nbf, nbf) in hartree, with
        eri[m, n, l, s] holding (mn|ls), the Coulomb repulsion between the charge
        distribution of the basis-function pair (m, n) and that of the pair (l, s).

    Raises
    ------
    ValueError
        If ``coords`` is not of shape (n_atoms, 3), or a shell has l other than 0 or 1,
        an invalid centre index, empty or mismatched exponent and coefficient lists, or a
        non-positive exponent.
    '''
    return eri

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_electron_repulsion_integrals(coords: "np.ndarray", shells: list) -> "np.ndarray":
    norm_shells = _normalized_shells(coords, shells)
    offsets = _shell_offsets(norm_shells)
    nbf = offsets[-1]
    eri = np.zeros((nbf, nbf, nbf, nbf))
    pairs = {}
    for i in range(len(norm_shells)):
        for j in range(i + 1):
            pairs[(i, j)] = _shell_pair(norm_shells[i], norm_shells[j])
    keys = sorted(pairs)
    for x, (i, j) in enumerate(keys):
        bra = pairs[(i, j)]
        for k, l in keys[: x + 1]:
            ket = pairs[(k, l)]
            p, q = bra["p"][:, None], ket["p"][None, :]
            alpha = p * q / (p + q)
            pq = bra["center"][:, None, :] - ket["center"][None, :, :]
            r = _hermite_coulomb(bra["l_sum"] + ket["l_sum"], alpha, pq[..., 0], pq[..., 1], pq[..., 2])
            rm = np.empty((len(bra["tuv"]), len(ket["tuv"])) + alpha.shape)
            for m, (t, u, v) in enumerate(bra["tuv"]):
                for mm, (tt, uu, vv) in enumerate(ket["tuv"]):
                    rm[m, mm] = (-1.0) ** (tt + uu + vv) * r[t + tt, u + uu, v + vv]
            weight = 2.0 * np.pi ** 2.5 / (p * q * np.sqrt(p + q)) * bra["cc"][:, None] * ket["cc"][None, :]
            block = np.einsum("atx,tsxy,bsy->ab", bra["eh"], rm * weight[None, None], ket["eh"], optimize=True)
            na, nb = len(bra["comps_a"]), len(bra["comps_b"])
            nc, nd = len(ket["comps_a"]), len(ket["comps_b"])
            block = block.reshape(na, nb, nc, nd)
            si = slice(offsets[i], offsets[i] + na)
            sj = slice(offsets[j], offsets[j] + nb)
            sk = slice(offsets[k], offsets[k] + nc)
            sl = slice(offsets[l], offsets[l] + nd)
            for b1, (s1, s2, s3, s4) in ((block, (si, sj, sk, sl)),
                                         (block.transpose(2, 3, 0, 1), (sk, sl, si, sj))):
                eri[s1, s2, s3, s4] = b1
                eri[s2, s1, s3, s4] = b1.transpose(1, 0, 2, 3)
                eri[s1, s2, s4, s3] = b1.transpose(0, 1, 3, 2)
                eri[s2, s1, s4, s3] = b1.transpose(1, 0, 3, 2)
    return eri

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    basis = """import copy
import numpy as np
H_S = ([3.425250914, 0.6239137298, 0.1688554040], [0.1543289673, 0.5353281423, 0.4446345422])
O_S = ([130.7093214, 23.80886605, 6.443608313], [0.1543289673, 0.5353281423, 0.4446345422])
O_SP = [5.033151319, 1.169596125, 0.3803889600]
SP_S = [-0.09996722919, 0.3995128261, 0.7001154689]
SP_P = [0.1559162750, 0.6076837186, 0.3919573931]
H_631G = [([18.73113696, 2.825394365, 0.6401216923], [0.03349460434, 0.2347269535, 0.8137573261]),
          ([0.1612777588], [1.0])]
HE_631G = [([38.421634, 5.77803, 1.241774], [0.04013973935, 0.261246097, 0.7931846246]),
           ([0.297964], [1.0])]
def shell(center, l, exps, coefs):
    return {"center": center, "l": l, "exponents": list(exps), "coefficients": list(coefs)}
"""
    return [
        # --- Normal: H2O / STO-3G (7 functions, s and p shells) ---
        {
            "setup": basis + """
coords = np.array([[0.0, -0.143225816552, 0.0],
                   [1.638036840407, 1.136548822547, 0.0],
                   [-1.638036840407, 1.136548822547, 0.0]])
shells = [shell(0, 0, *O_S), shell(0, 0, O_SP, SP_S), shell(0, 1, O_SP, SP_P),
          shell(1, 0, *H_S), shell(2, 0, *H_S)]
""",
            "call": "compute_electron_repulsion_integrals(coords.copy(), copy.deepcopy(shells))",
            "gold_call": "_oracle_compute_electron_repulsion_integrals(coords.copy(), copy.deepcopy(shells))",
            "tol": 1e-8,
        },
        # --- Normal: HeH+ / 6-31G (s-only, split-valence, 4 functions) ---
        {
            "setup": basis + """
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4632]])
shells = [shell(0, 0, *HE_631G[0]), shell(0, 0, *HE_631G[1]),
          shell(1, 0, *H_631G[0]), shell(1, 0, *H_631G[1])]
""",
            "call": "compute_electron_repulsion_integrals(coords.copy(), copy.deepcopy(shells))",
            "gold_call": "_oracle_compute_electron_repulsion_integrals(coords.copy(), copy.deepcopy(shells))",
            "tol": 1e-8,
        },
        # --- Boundary: a single contracted s function, (11|11) only ---
        {
            "setup": basis + """
coords = np.array([[0.2, 0.1, -0.4]])
shells = [shell(0, 0, *H_S)]
""",
            "call": "compute_electron_repulsion_integrals(coords.copy(), copy.deepcopy(shells))",
            "gold_call": "_oracle_compute_electron_repulsion_integrals(coords.copy(), copy.deepcopy(shells))",
            "tol": 1e-8,
        },
        # --- Edge: uncontracted and contracted p shells on off-axis centres plus a diffuse s ---
        {
            "setup": basis + """
coords = np.array([[0.0, 0.0, 0.0], [0.7, -1.1, 0.9], [-1.3, 0.4, 0.5]])
shells = [shell(0, 1, [0.8], [1.0]), shell(1, 1, [1.9, 0.35], [0.4, 0.7]),
          shell(2, 0, [0.5, 0.12], [0.6, 0.5])]
""",
            "call": "compute_electron_repulsion_integrals(coords.copy(), copy.deepcopy(shells))",
            "gold_call": "_oracle_compute_electron_repulsion_integrals(coords.copy(), copy.deepcopy(shells))",
            "tol": 1e-8,
        },
    ]
