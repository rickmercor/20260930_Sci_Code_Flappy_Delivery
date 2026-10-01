"""
Excitonic coupling of the antiparallel stacked homodimer obtained from a CCSD and EOM-CCSD treatment of the dimer as a whole.

The supersystem route treats the two molecules as one closed-shell system. Its Hamiltonian is H(lambda) = H_A + H_B + lambda V_AB, where H_A and H_B are the PPP operators of the two molecules, each with its own core charges and without any integral that connects them, and V_AB = sum_{mu in A, nu in B} gamma_AB[mu, nu] (n_mu - Z_mu)(n_nu - Z_nu) is the intermolecular Coulomb operator of the antiparallel stack. At lambda = 0 the dimer has two degenerate singlet excited states at the excitation energy of the selected monomer state, one with the excitation on A and one with it on B. The dimer also has charge-transfer states and states of other character, possibly below this pair, and they are not part of it.

For lambda different from zero the ground state of H(lambda) is treated at the CCSD level and the pair of excited states at the EOM-CCSD level, in the full singles and doubles space of the eight-electron dimer reference built from the RHF determinants of the two molecules. The pair splits. The supersystem coupling V^s is half of that splitting to first order in the intermolecular interaction,

  |V^s| = lim_{lambda -> 0} [ E_+(lambda) - E_-(lambda) ] / (2 lambda)

where E_+ and E_- are the two EOM-CCSD excitation energies of the pair. Because the stack is symmetric under the two-fold rotation that exchanges the molecules, no first-order shift separates the two members, and the whole first-order splitting is set by the coupling. The first-order value does not depend on whether the reference orbitals are re-optimised for the interacting dimer.

Nothing about the result may be taken from the isolated molecules unless it is exactly what the CCSD and EOM-CCSD equations of the dimer produce: the dimer calculation is truncated at doubles relative to the eight-electron reference, and that truncation is not the same thing as combining two exact monomer descriptions. The coupling is asked in eV to a relative accuracy of 1e-9, so it has to be evaluated in the limit itself rather than read off one small but finite lambda.

Returns
-------
float: |V^s|, the magnitude of the supersystem excitonic coupling in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def supersystem_excitonic_coupling(bonds: "np.ndarray", betas: "np.ndarray", alphas: "np.ndarray",
                                   hubbard_u: "np.ndarray", core_charges: "np.ndarray", n_occ: int,
                                   root: int, separation: float) -> float:
    '''Half of the first-order EOM-CCSD splitting of the dimer's locally excited singlet pair.

    Parameters
    ----------
    bonds : numpy.ndarray
        Bond lengths of the chain in angstrom, length n_sites - 1.
    betas : numpy.ndarray
        Resonance integral of each bond in eV.
    alphas : numpy.ndarray
        Site energies in eV, length n_sites.
    hubbard_u : numpy.ndarray
        One-centre repulsions in eV, length n_sites.
    core_charges : numpy.ndarray
        Core charge Z of each site, length n_sites.
    n_occ : int
        Number of doubly occupied orbitals of one molecule.
    root : int
        Index, counting from zero, of the monomer singlet excited state whose dimer
        pair is split.
    separation : float
        Stacking distance between the molecular planes in angstrom.

    Returns
    -------
    coupling : float
        |V^s| in eV, the limit of half the splitting divided by lambda.

    Raises
    ------
    ValueError
        If any argument violates the constraints of the monomer matrices, the
        antiparallel stack geometry, the RHF, CCSD or EOM-CCSD steps it feeds, or
        if root does not select an available singlet excited state.
    '''
    return coupling

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _embed_manifold(man_mono, man_dimer, shift):
    import numpy as np
    key = {m: k for k, m in enumerate(man_dimer)}
    return np.array([key[(tuple((p + shift, s) for p, s in c), tuple((p + shift, s) for p, s in a))]
                     for c, a in man_mono], dtype=int)


def _intermolecular_operator(space, G, ZA, ZB, nA):
    import scipy.sparse as sp
    I = sp.identity(space["dim"], format="csr")
    W = sp.csr_matrix((space["dim"], space["dim"]))
    for mu in range(nA):
        for nu in range(G.shape[1]):
            W = W + G[mu, nu] * ((_site_occupation(space, mu) - ZA[mu] * I) @ (_site_occupation(space, nA + nu) - ZB[nu] * I))
    return W.tocsr()


def _supersystem_pieces(bonds, betas, alphas, hubbard_u, core_charges, n_occ, root, separation):
    import numpy as np
    M = _oracle_ppp_monomer_matrices(bonds, betas, alphas, hubbard_u)
    G = _oracle_antiparallel_stack_interactions(bonds, hubbard_u, separation)
    H1, G1, Z, nocc = _check_ppp_system(M[0], M[1], core_charges, n_occ)
    n = H1.shape[0]
    st = _monomer_transition(H1, G1, Z, nocc, root)
    # non-interacting dimer on the product of the monomer RHF determinants
    Hd = np.zeros((2 * n, 2 * n)); Hd[:n, :n] = H1; Hd[n:, n:] = H1
    Gd = np.zeros((2 * n, 2 * n)); Gd[:n, :n] = G1; Gd[n:, n:] = G1
    Cd = np.zeros((2 * n, 2 * n)); Cd[:n, :n] = st["C"]; Cd[n:, n:] = st["C"]
    Zd = np.concatenate([Z, Z])
    E0_dimer = _oracle_ccsd_energy(Hd, Gd, Zd, 2 * nocc)
    occ = list(range(nocc)) + list(range(n, n + nocc))
    cc = _ccsd_state(Hd, Gd, Zd, Cd, occ)
    if abs(cc["E0"] - E0_dimer) > 1e-8:
        raise ValueError("dimer reference is not size-consistent")
    Mb = _cc_block(cc, cc["H"])
    W = _intermolecular_operator(cc["space"], G, Z, Z, n)
    mapA = _embed_manifold(st["cc"]["man"], cc["man"], 0)
    mapB = _embed_manifold(st["cc"]["man"], cc["man"], n)
    dim = Mb.shape[0]
    isA = np.zeros(dim, bool); isA[1 + mapA] = True
    isB = np.zeros(dim, bool); isB[1 + mapB] = True
    isX = ~(isA | isB); isX[0] = False
    E = E0_dimer + st["omega"]
    RB = np.zeros(dim); RB[0] = st["r"][0]; RB[1 + mapB] = st["r"][1:]
    LA = np.zeros(dim); LA[1 + mapA] = st["l"][1:]
    LX = np.linalg.solve((E * np.eye(int(isX.sum())) - Mb[np.ix_(isX, isX)]).T, LA[isA] @ Mb[np.ix_(isA, isX)])
    LAX = LA.copy(); LAX[isX] = LX
    Wb = _cc_block(cc, W)
    t1 = np.linalg.solve(_cc_jacobian(cc, cc["T"]), -Wb[1:, 0])
    T1 = _cc_tmat(cc, t1)
    Wrel = _cc_block(cc, W + cc["H"] @ T1 - T1 @ cc["H"])
    return dict(Vs=float(abs(LAX @ Wrel @ RB)), LX_weight=float(np.linalg.norm(LX) / np.linalg.norm(st["l"][1:])),
                Vs_frozen=float(abs(LAX @ Wb @ RB)), Vs_local=float(abs(LA @ Wrel @ RB)), omega=st["omega"],
                E0_dimer=E0_dimer)


def _oracle_supersystem_excitonic_coupling(bonds: "np.ndarray", betas: "np.ndarray", alphas: "np.ndarray",
                                           hubbard_u: "np.ndarray", core_charges: "np.ndarray", n_occ: int,
                                           root: int, separation: float) -> float:
    return _supersystem_pieces(bonds, betas, alphas, hubbard_u, core_charges, n_occ, root, separation)["Vs"]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: the benchmark C=C-N stack at 7 angstrom ---
        {
            "setup": """import numpy as np
""",
            "call": "supersystem_excitonic_coupling(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]), np.array([1.0, 1.0, 2.0]), 2, 0, 7.0)",
            "gold_call": "_oracle_supersystem_excitonic_coupling(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]), np.array([1.0, 1.0, 2.0]), 2, 0, 7.0)",
            "tol": 1e-8,
        },
        # --- Normal: the same stack at 4 angstrom ---
        {
            "setup": """import numpy as np
""",
            "call": "supersystem_excitonic_coupling(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]), np.array([1.0, 1.0, 2.0]), 2, 0, 4.0)",
            "gold_call": "_oracle_supersystem_excitonic_coupling(np.array([1.34, 1.40]), np.array([-2.4, -2.0]), np.array([0.0, 0.0, -3.0]), np.array([11.13, 11.13, 16.76]), np.array([1.0, 1.0, 2.0]), 2, 0, 4.0)",
            "tol": 1e-8,
        },
        # --- Boundary: a two-electron polar unit, four electrons in the dimer ---
        {
            "setup": """import numpy as np
""",
            "call": "supersystem_excitonic_coupling(np.array([1.34]), np.array([-2.4]), np.array([0.0, -2.0]), np.array([11.13, 12.34]), np.array([1.0, 1.0]), 1, 0, 5.0)",
            "gold_call": "_oracle_supersystem_excitonic_coupling(np.array([1.34]), np.array([-2.4]), np.array([0.0, -2.0]), np.array([11.13, 12.34]), np.array([1.0, 1.0]), 1, 0, 5.0)",
            "tol": 1e-8,
        },
        # --- Edge: a symmetric two-carbon unit, where the monomer singles vanish by symmetry ---
        {
            "setup": """import numpy as np
""",
            "call": "supersystem_excitonic_coupling(np.array([1.34]), np.array([-2.4]), np.array([0.0, 0.0]), np.array([11.13, 11.13]), np.array([1.0, 1.0]), 1, 0, 6.0)",
            "gold_call": "_oracle_supersystem_excitonic_coupling(np.array([1.34]), np.array([-2.4]), np.array([0.0, 0.0]), np.array([11.13, 11.13]), np.array([1.0, 1.0]), 1, 0, 6.0)",
            "tol": 1e-8,
        },
        # --- Edge: an allyl cation stack ---
        {
            "setup": """import numpy as np
""",
            "call": "supersystem_excitonic_coupling(np.array([1.40, 1.40]), np.array([-2.4, -2.4]), np.zeros(3), np.full(3, 11.13), np.ones(3), 1, 0, 6.0)",
            "gold_call": "_oracle_supersystem_excitonic_coupling(np.array([1.40, 1.40]), np.array([-2.4, -2.4]), np.zeros(3), np.full(3, 11.13), np.ones(3), 1, 0, 6.0)",
            "tol": 1e-8,
        },
        # --- Edge: a strongly donating C=C-N unit at long range ---
        {
            "setup": """import numpy as np
""",
            "call": "supersystem_excitonic_coupling(np.array([1.34, 1.40]), np.array([-2.4, -1.6]), np.array([0.0, 0.0, -5.0]), np.array([11.13, 11.13, 16.76]), np.array([1.0, 1.0, 2.0]), 2, 0, 12.0)",
            "gold_call": "_oracle_supersystem_excitonic_coupling(np.array([1.34, 1.40]), np.array([-2.4, -1.6]), np.array([0.0, 0.0, -5.0]), np.array([11.13, 11.13, 16.76]), np.array([1.0, 1.0, 2.0]), 2, 0, 12.0)",
            "tol": 1e-8,
        },
        # --- Invalid: a root index beyond the singlet states of a two-electron unit ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        supersystem_excitonic_coupling(np.array([1.34]), np.array([-2.4]), np.array([0.0, -2.0]), np.array([11.13, 12.34]), np.array([1.0, 1.0]), 1, 7, 5.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_supersystem_excitonic_coupling(np.array([1.34]), np.array([-2.4]), np.array([0.0, -2.0]), np.array([11.13, 12.34]), np.array([1.0, 1.0]), 1, 7, 5.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
