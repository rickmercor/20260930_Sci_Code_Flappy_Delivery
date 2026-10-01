"""
Implement run_rhf, which solves the closed-shell restricted Hartree-Fock equations for given
atomic-orbital integrals and returns the total energy, the canonical orbital energies and the
molecular-orbital coefficients.

Restricted Hartree-Fock theory describes a closed-shell molecule by a single determinant of
doubly occupied spatial orbitals.

Returns
-------
tuple (e_total float in hartree, orbital_energies np.ndarray (n_bf,) ascending, coefficients np.ndarray (n_bf, n_bf) with C^T S C = 1) of closed-shell RHF
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_rhf(overlap: "np.ndarray", core_hamiltonian: "np.ndarray", eri: "np.ndarray", e_nuc: float, n_occ: int) -> tuple:
    '''Closed-shell restricted Hartree-Fock solution.

    Parameters
    ----------
    overlap : np.ndarray
        Atomic-orbital overlap matrix S, shape (n_bf, n_bf), symmetric positive definite.
    core_hamiltonian : np.ndarray
        Core Hamiltonian h (kinetic energy plus nuclear attraction), shape (n_bf, n_bf), in
        hartree.
    eri : np.ndarray
        Electron-repulsion integrals (pq|rs) in chemists' notation, shape
        (n_bf, n_bf, n_bf, n_bf), in hartree.
    e_nuc : float
        Nuclear-repulsion energy in hartree.
    n_occ : int
        Number of doubly occupied spatial orbitals, 1 <= n_occ <= n_bf.

    Returns
    -------
    result : tuple
        (e_total, orbital_energies, coefficients):
        e_total : float, electronic energy plus e_nuc, in hartree.
        orbital_energies : np.ndarray, shape (n_bf,), canonical orbital energies in ascending
            order, in hartree.
        coefficients : np.ndarray, shape (n_bf, n_bf), orbital coefficients as columns in the
            same order, with C^T S C = 1; the first n_occ columns are occupied.
        The solution is the aufbau solution, converged until the orbital-gradient matrix
        F P S - S P F has no element larger than 1e-10 (P the density matrix, F the Fock
        matrix).

    Raises
    ------
    ValueError
        If the matrices have inconsistent shapes, n_occ is outside 1..n_bf, or the iterations
        do not converge within 200 cycles.
    '''
    return (e_total, orbital_energies, coefficients)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import eigh


def _fock_matrix(core_hamiltonian: "np.ndarray", eri: "np.ndarray", density: "np.ndarray") -> "np.ndarray":
    """Closed-shell Fock matrix h + J[P] - K[P]/2 for the total density matrix P."""
    coulomb = np.einsum("pqrs,rs->pq", eri, density, optimize=True)
    exchange = np.einsum("prqs,rs->pq", eri, density, optimize=True)
    return core_hamiltonian + coulomb - 0.5 * exchange


def _diis_extrapolate(history: list) -> "np.ndarray":
    """Pulay DIIS combination of stored (Fock, error) pairs."""
    m = len(history)
    b = -np.ones((m + 1, m + 1))
    b[m, m] = 0.0
    for i in range(m):
        for j in range(m):
            b[i, j] = np.sum(history[i][1] * history[j][1])
    rhs = np.zeros(m + 1)
    rhs[m] = -1.0
    weights = np.linalg.lstsq(b, rhs, rcond=None)[0][:m]
    return sum(w * f for w, (f, _) in zip(weights, history))


def _oracle_run_rhf(overlap: "np.ndarray", core_hamiltonian: "np.ndarray", eri: "np.ndarray", e_nuc: float, n_occ: int) -> tuple:
    overlap = np.asarray(overlap, dtype=float)
    core_hamiltonian = np.asarray(core_hamiltonian, dtype=float)
    eri = np.asarray(eri, dtype=float)
    n_bf = overlap.shape[0]
    if overlap.shape != (n_bf, n_bf) or core_hamiltonian.shape != (n_bf, n_bf) or eri.shape != (n_bf,) * 4:
        raise ValueError("inconsistent integral shapes")
    if int(n_occ) != n_occ or not 1 <= n_occ <= n_bf:
        raise ValueError("n_occ must be an integer between 1 and n_bf")
    n_occ = int(n_occ)
    energies, coefficients = eigh(core_hamiltonian, overlap)
    history = []
    e_old = None
    for _ in range(200):
        density = 2.0 * coefficients[:, :n_occ] @ coefficients[:, :n_occ].T
        fock = _fock_matrix(core_hamiltonian, eri, density)
        e_elec = 0.5 * np.sum(density * (core_hamiltonian + fock))
        error = fock @ density @ overlap - overlap @ density @ fock
        if np.abs(error).max() < 1e-10 and e_old is not None and abs(e_elec - e_old) < 1e-12:
            return float(e_elec + e_nuc), energies, coefficients
        e_old = e_elec
        history = (history + [(fock, error)])[-8:]
        energies, coefficients = eigh(_diis_extrapolate(history), overlap)
    raise ValueError("Hartree-Fock iterations did not converge")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    helper = """import numpy as np
from scipy.special import erf
def s_integrals(charges, coords, shells):
    # closed-form integrals over contracted s Gaussians; coefficients multiply normalized primitives
    charges = np.asarray(charges, float); coords = np.asarray(coords, float)
    fns = []
    for atom, exps, coefs in shells:
        a = np.asarray(exps, float); c = np.asarray(coefs, float) * (2 * a / np.pi) ** 0.75
        s = np.sum(c[:, None] * c[None, :] * (np.pi / (a[:, None] + a[None, :])) ** 1.5)
        fns.append((coords[atom], a, c / np.sqrt(s)))
    def f0(t):
        t = np.asarray(t, float)
        return np.where(t < 1e-12, 1.0 - t / 3.0, 0.5 * np.sqrt(np.pi / np.maximum(t, 1e-300)) * erf(np.sqrt(t)))
    def pair(i, j):
        A, a, ca = fns[i]; B, b, cb = fns[j]
        p = a[:, None] + b[None, :]; mu = a[:, None] * b[None, :] / p
        K = np.exp(-mu * np.sum((A - B) ** 2)); P = (a[:, None, None] * A + b[None, :, None] * B) / p[..., None]
        return p, mu, K * ca[:, None] * cb[None, :], P, np.sum((A - B) ** 2)
    n = len(fns); S = np.zeros((n, n)); H = np.zeros((n, n)); prs = {}
    for i in range(n):
        for j in range(n):
            p, mu, w, P, r2 = pair(i, j); prs[i, j] = (p, w, P)
            S[i, j] = np.sum(w * (np.pi / p) ** 1.5)
            H[i, j] = np.sum(w * mu * (3 - 2 * mu * r2) * (np.pi / p) ** 1.5)
            for Z, C in zip(charges, coords):
                H[i, j] -= Z * np.sum(w * 2 * np.pi / p * f0(p * np.sum((P - C) ** 2, axis=-1)))
    eri = np.zeros((n, n, n, n))
    for (i, j), (p, w1, P) in prs.items():
        for (k, l), (q, w2, Q) in prs.items():
            pp = p.reshape(-1, 1); qq = q.reshape(1, -1)
            d2 = np.sum((P.reshape(-1, 1, 3) - Q.reshape(1, -1, 3)) ** 2, axis=-1)
            eri[i, j, k, l] = np.sum(np.outer(w1, w2) * 2 * np.pi ** 2.5 / (pp * qq * np.sqrt(pp + qq)) * f0(pp * qq / (pp + qq) * d2))
    en = sum(charges[x] * charges[y] / np.linalg.norm(coords[x] - coords[y]) for x in range(len(charges)) for y in range(x))
    return S, H, eri, float(en)
H_S = [([0.100112428, 0.2430767471, 0.6259552659, 1.822142904, 6.513143725, 35.52322122],
        [0.1303340841, 0.4164915298, 0.3705627997, 0.1685383049, 0.0493614929, 0.0091635963]),
       ([0.0654284286, 0.100112428, 0.2430767471, 0.6259552659], [4.8518572723, 0.4054347007, 0.0547707665, 0.1436562367]),
       ([0.0654284286, 0.100112428, 0.2430767471, 0.6259552659], [1.746076269, -5.7363906541, -0.3296656157, 0.1098876093])]
def hydrogens(n_atoms):
    return [(a, e, c) for a in range(n_atoms) for (e, c) in H_S]
def invariants(result, n_occ):
    e_total, eps, C = result
    return np.concatenate([[e_total], eps, np.ravel(C[:, :n_occ] @ C[:, :n_occ].T)])
"""
    invalid = """
def run_model():
    try:
        run_rhf(S.copy(), H.copy(), eri.copy(), en, n_occ)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_run_rhf(S.copy(), H.copy(), eri.copy(), en, n_occ)
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Typical: H2 near equilibrium (1.4 bohr) in three contracted s functions per atom ---
        {
            "setup": helper + """
S, H, eri, en = s_integrals([1.0, 1.0], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], hydrogens(2))
n_occ = 1
""",
            "call": "invariants(run_rhf(S.copy(), H.copy(), eri.copy(), en, n_occ), n_occ)",
            "gold_call": "invariants(_oracle_run_rhf(S.copy(), H.copy(), eri.copy(), en, n_occ), n_occ)",
            "tol": 1e-8,
        },
        # --- Typical: linear H4 chain with unequal spacings (two occupied orbitals) ---
        {
            "setup": helper + """
S, H, eri, en = s_integrals([1.0] * 4, [[0, 0, 0.0], [0, 0, 1.5], [0, 0, 3.4], [0, 0, 4.9]], hydrogens(4))
n_occ = 2
""",
            "call": "invariants(run_rhf(S.copy(), H.copy(), eri.copy(), en, n_occ), n_occ)",
            "gold_call": "invariants(_oracle_run_rhf(S.copy(), H.copy(), eri.copy(), en, n_occ), n_occ)",
            "tol": 1e-8,
        },
        # --- Edge: HeH+ (unequal charges, 2 electrons) with a one-atom s basis on each centre ---
        {
            "setup": helper + """
shells = [(0, [38.421634, 5.77803, 1.241774, 0.297964], [0.023766, 0.154679, 0.469630, 0.513572]),
          (0, [0.5], [1.0])] + hydrogens(2)[3:]
S, H, eri, en = s_integrals([2.0, 1.0], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.46]], shells)
n_occ = 1
""",
            "call": "invariants(run_rhf(S.copy(), H.copy(), eri.copy(), en, n_occ), n_occ)",
            "gold_call": "invariants(_oracle_run_rhf(S.copy(), H.copy(), eri.copy(), en, n_occ), n_occ)",
            "tol": 1e-8,
        },
        # --- Boundary: every orbital occupied (n_occ = n_bf): density fixed by the basis ---
        {
            "setup": helper + """
shells = [(0, [1.3], [1.0]), (1, [0.9], [1.0])]
S, H, eri, en = s_integrals([1.0, 1.0], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]], shells)
n_occ = 2
""",
            "call": "invariants(run_rhf(S.copy(), H.copy(), eri.copy(), en, n_occ), n_occ)",
            "gold_call": "invariants(_oracle_run_rhf(S.copy(), H.copy(), eri.copy(), en, n_occ), n_occ)",
            "tol": 1e-8,
        },
        # --- Invalid: more occupied orbitals than basis functions ---
        {
            "setup": helper + """
S, H, eri, en = s_integrals([1.0, 1.0], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], hydrogens(2))
n_occ = 7
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
