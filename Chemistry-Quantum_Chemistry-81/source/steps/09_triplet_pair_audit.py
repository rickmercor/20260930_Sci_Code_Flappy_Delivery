"""
Return the (len(U_list), 47) audit table with one row per Coulomb parameter of U_list: columns 0 to 17 the populations [exact spin-adapted, T0T0 prescription] for the bases (covalent, family, paper) in that order, for 2^1Ag+ (columns 0 to 5), 1^1Bu+ (6 to 11) and 1^5Ag+ (12 to 17); columns 18 to 23 the six excitation energies of dark_state_energies; 24 V_01; 25 V_{0,N-1}; 26 the mean diagonal element of the half-filled Hamiltonian; 27 the number of covalent triplets below 2^1Ag+; 28 and 29 the first and last family energies of the (N_d - 1)-dimer subchain; 30 <S^z_0> of its T_1; 31 to 34 the entries [|<T0T0|2^1Ag+>|^2, |<1TT|2^1Ag+>|^2, |<1TT|1^1Ag+>|^2, <1TT|H|1TT> - E_0] of the product m = 1, j = 0, k = 0; 35 to 46 for the bases (covalent, family, paper) the four numbers [n_b, largest and smallest singlet-coupled Gram eigenvalue, rank, the rank being the number of singlet-coupled Gram eigenvalues above 1e-8 of the largest]. The head element [0, 0], the exact spin-adapted covalent-basis population of 2^1Ag+ at U_list[0], is the final answer. Model and conventions: an all-trans polyene of N carbon atoms (N even), sites i = 0, ..., N-1, N_d = N/2 ethylene dimers with dimer n formed by the sites 2n and 2n+1, one pi electron per site, described by the Pariser-Parr-Pople Hamiltonian H = -sum_{i,sigma} t_i (c+_{i,sigma} c_{i+1,sigma} + h.c.) + U sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2) + sum_{i<j} V_ij (n_i - 1)(n_j - 1), energies in eV; bond-alternated hopping t_i = t0 (1 + delta) for even i (the double bonds 0-1, 2-3, ...) and t_i = t0 (1 - delta) for odd i; Ohno potential V_ij = U / sqrt(1 + (U eps r_ij / 14.397)^2) with U in eV, r_ij in angstrom, 14.397 eV angstrom = e^2/(4 pi eps_0) and eps the relative permittivity. Geometry: site 0 at the origin; bond i (from site i to site i+1) has length r_double for even i and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg; r_ij are the resulting planar distances. Determinant basis of the (n_up, n_down) sector: an up-spin configuration is the bit mask sum over occupied sites of 2^i, likewise a down-spin configuration; the C(N, n_up) up configurations and the C(N, n_down) down configurations are each listed in increasing integer order and the basis index of a determinant is i_up D_down + i_down; a determinant is the string of all up creation operators (ascending site) followed by all down creation operators (ascending site) acting on the vacuum, so that nearest-neighbour hopping matrix elements carry no fermionic sign. Symmetries of the half-filled S_z = 0 sector (n_up = n_down = N/2): S(S+1) from the total spin S^2 = S- S+ + S_z (S_z + 1) with S+ = sum_i c+_{i,up} c_{i,down} and S- its adjoint; the site reversal P maps a determinant to the determinant with site i replaced by N-1-i; the occupation complement J maps a determinant to the determinant with every occupation inverted (empty <-> occupied for each spin); both permutations, with the constant sign of the operator reordering, commute with H. Labels of an eigenstate k are S(S+1) and the products p_k p_0 and j_k j_0 of its P and J eigenvalues with those of the ground state (the ground state reads +1, +1); inside a cluster of eigenvalues closer than 1e-7 (relative to the largest computed energy magnitude) the eigenvectors are rotated so that S^2, P and J are simultaneously diagonal and the members are ordered by (S(S+1), p, j) ascending. Covalent sector: singlets and quintets with j_k j_0 = +1, triplets with j_k j_0 = -1 (on singly occupied configurations the complement acts as the global spin flip, whose M = 0 eigenvalue alternates with S). State names: 1^1Ag+ = ground state; 2^1Ag+ = the second singlet with (p, j) = (+1, +1) (the dark state); 1^1Bu- = the lowest singlet with (-1, -1) (the optically bright state); 1^1Bu+ = the lowest singlet with (-1, +1); 1^3Bu = the lowest triplet (-1, -1); 1^5Ag+ = the lowest quintet with (+1, +1). Subchains and triplet families: a partition of the N_d dimers into a left subchain of m dimers (sites 0 to 2m-1) and a right subchain of N_d - m dimers (sites 2m to N-1); each subchain is treated as an isolated open PPP chain with the same parameters and the distances of its own segment of the geometry; its covalent triplets are its triplet eigenstates with j opposite to its own ground state (labels resolved as above); its triplet family T_j(m), j = 1, ..., m, consists of the m lowest covalent triplets; the spin components of a triplet are related by the ladder operators, T_{+1} = S+ T_0 / sqrt2 and T_{-1} = S- T_0 / sqrt2. Triplet-pair products: |A x B> is the site-ordered (Jordan-Wigner) tensor product, the creation-operator string of the left state in site order (site 0 up, site 0 down, site 1 up, ...) followed by that of the right state with its sites shifted by 2m, re-expressed in the full-chain determinant basis with the sign of the reordering; T0T0 = T_0(left) x T_0(right); the singlet-coupled pair is 1TT = (T_{+1} x T_{-1} - T_0 x T_0 + T_{-1} x T_{+1})/sqrt3 and the quintet-coupled (M = 0) pair is 2TT = (T_{+1} x T_{-1} + 2 T_0 x T_0 + T_{-1} x T_{+1})/sqrt6 (Clebsch-Gordan coupling of two spin-1 objects). Pair bases: 'paper' = {T_j(m) x T_1(N_d - m): 1 <= m <= N_d - 1, 1 <= j <= m}; 'family' = {T_j(m) x T_k(N_d - m): all family members on both subchains}; 'covalent' = every pair of covalent triplets of the two subchains (linearly dependent: the span is defined by the singular values of the product matrix above 1e-8 of the largest). Populations of a state Psi in a pair basis: the T0T0 prescription is the squared norm of the projection of Psi onto the span of the T0T0 products multiplied by 3 for a singlet and by 3/2 for a quintet (the factor that would convert a single M = 0 product into its spin-coupled counterpart); the exact spin-adapted population is the squared norm of the projection of Psi onto the span of the singlet-coupled products (singlet states) or of the quintet-coupled products (quintet state). Reference parameters: N = 8 (octatetraene), U = 8 eV (audit set 8, 4, 14 eV), eps = 2, t0 = 2.4 eV, delta = 1/12, r_double = 1.35 angstrom, r_single = 1.45 angstrom, angle_deg = 120, n_states = 100. Raise ValueError for invalid parameters, if N < 4 or if U_list is empty.

The audit places the source's T0T0 prescription next to the exact spin-adapted projection for three nested bases and three interaction strengths, exposing that the prescription overestimates the singlet triplet-pair weight, underestimates the quintet one and exceeds unity in the complete covalent basis at strong coupling, while the exact populations grow monotonically with the basis.

Returns
-------
A (len(U_list), 47) float64 array; entry [0, 0] is the final answer.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def triplet_pair_audit(N: int, U_list: list, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, n_states: int) -> np.ndarray:
    """Return the (len(U_list), 47) audit table with one row per Coulomb parameter of U_list:
    columns 0 to 17 the populations [exact spin-adapted, T0T0 prescription] for the bases
    (covalent, family, paper) in that order, for 2^1Ag+ (columns 0 to 5), 1^1Bu+ (6 to 11)
    and 1^5Ag+ (12 to 17); columns 18 to 23 the six excitation energies of
    dark_state_energies; 24 V_01; 25 V_{0,N-1}; 26 the mean diagonal element of the half-
    filled Hamiltonian; 27 the number of covalent triplets below 2^1Ag+; 28 and 29 the first
    and last family energies of the (N_d - 1)-dimer subchain; 30 <S^z_0> of its T_1; 31 to
    34 the entries [|<T0T0|2^1Ag+>|^2, |<1TT|2^1Ag+>|^2, |<1TT|1^1Ag+>|^2, <1TT|H|1TT> -
    E_0] of the product m = 1, j = 0, k = 0; 35 to 46 for the bases (covalent, family,
    paper) the four numbers [n_b, largest and smallest singlet-coupled Gram eigenvalue,
    rank, the rank being the number of singlet-coupled Gram eigenvalues above 1e-8 of
    the largest].

    Parameters
    ----------
    N : int
        Number of carbon atoms (even, >= 2).
    U_list : list
        Coulomb parameters in eV, one audit row each (non-empty).
    eps : float
        Relative permittivity of the Ohno potential (> 0).
    t0 : float
        Mean hopping integral in eV (> 0).
    delta : float
        Bond-alternation parameter, t_i = t0 (1 +- delta) (in (-1, 1)).
    r_double : float
        Double-bond length in angstrom (> 0).
    r_single : float
        Single-bond length in angstrom (> 0).
    angle_deg : float
        C-C-C bond angle in degrees (in (0, 180]; 180 gives a linear chain).
    n_states : int
        Number of lowest eigenstates of the half-filled S_z = 0 sector to compute (1 <= n_states <= sector dimension) and to search for the named states.

    Returns
    -------
    A (len(U_list), 47) float64 array; entry [0, 0] is the final answer.

    Raises
    ------
    ValueError
        For invalid parameters, if N < 4 or if U_list is empty.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import sqrt, pi, cos, sin
from scipy.linalg import eigh


def _EPS_RANK():
    return 1e-8

def _check_model(N, U, eps, t0, delta, r_double, r_single, angle_deg):
    if not (isinstance(N, (int, np.integer)) and not isinstance(N, bool)) or N < 2 or N % 2:
        raise ValueError("N must be an even integer >= 2")
    for x in (U, eps, t0, delta, r_double, r_single, angle_deg):
        if not np.isfinite(float(x)):
            raise ValueError("model parameters must be finite")
    if U < 0.0 or eps <= 0.0 or t0 <= 0.0 or r_double <= 0.0 or r_single <= 0.0:
        raise ValueError("U >= 0, eps > 0, t0 > 0 and positive bond lengths are required")
    if not (0.0 < angle_deg <= 180.0):
        raise ValueError("bond angle must lie in (0, 180] degrees")
    if not (-1.0 < delta < 1.0):
        raise ValueError("bond alternation delta must lie in (-1, 1)")

def _oracle_triplet_pair_audit(N: int, U_list: list, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, n_states: int) -> np.ndarray:
    _check_model(N, 0.0, eps, t0, delta, r_double, r_single, angle_deg)
    if not isinstance(U_list, (list, tuple, np.ndarray)) or len(U_list) < 1:
        raise ValueError("U_list must contain at least one Coulomb parameter")
    if N < 4:
        raise ValueError("a triplet pair needs at least two dimers")
    Nd = N // 2
    rows = []
    for U in U_list:
        U = float(U)
        V = _oracle_coulomb_matrix(N, U, eps, r_double, r_single, angle_deg)
        Hs = _oracle_ppp_hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, Nd, Nd)
        lab = _oracle_symmetry_labels(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states)
        en = _oracle_dark_state_energies(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states)
        fam = _oracle_triplet_family(Nd - 1, U, eps, t0, delta, r_double, r_single, angle_deg)
        prod = _oracle_triplet_pair_product(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states, 1, 0, 0)
        pops = []
        grams = []
        for kind in ("covalent", "family", "paper"):
            pk = _oracle_triplet_pair_populations(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_states, kind)
            gk = _oracle_pair_basis_gram(N, U, eps, t0, delta, r_double, r_single, angle_deg, kind)
            pops.append(pk)
            grams.append([gk.shape[1], gk[1, 0], gk[1, -1], float(np.sum(gk[1] > _EPS_RANK() * gk[1, 0]))])
        pops = np.array(pops)
        ncov = float(np.sum((np.abs(lab[:, 1] - 2.0) < 1e-6) & (np.abs(lab[:, 3] + 1.0) < 1e-6) & (lab[:, 0] < en[0])))
        row = [pops[0, 0, 1], pops[0, 0, 0], pops[1, 0, 1], pops[1, 0, 0], pops[2, 0, 1], pops[2, 0, 0],
               pops[0, 1, 1], pops[0, 1, 0], pops[1, 1, 1], pops[1, 1, 0], pops[2, 1, 1], pops[2, 1, 0],
               pops[0, 2, 1], pops[0, 2, 0], pops[1, 2, 1], pops[1, 2, 0], pops[2, 2, 1], pops[2, 2, 0]]
        row += list(en)
        row += [V[0, 1], V[0, N - 1], float(np.trace(Hs)) / Hs.shape[0], ncov, fam[0, 0], fam[-1, 0], fam[0, 2]]
        row += [prod[0], prod[1], prod[5], prod[6]]
        for g in grams:
            row += g
        rows.append(row)
    return np.array(rows, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nN = 8\nU_list = [8.0, 4.0, 14.0]\neps = 2.0\nt0 = 2.4\ndelta = 1.0 / 12.0\nr_double = 1.35\nr_single = 1.45\nangle_deg = 120.0\nn_states = 100\nU_model = list(U_list)\nU_gold = list(U_list)\n',
         'call': 'triplet_pair_audit(N, U_model, eps, t0, delta, r_double, r_single, angle_deg, n_states)',
         'gold_call': '_oracle_triplet_pair_audit(N, U_gold, eps, t0, delta, r_double, r_single, angle_deg, n_states)'},
        {'setup': 'import numpy as np\nN = 6\nU_list = [4.0]\neps = 1.0\nt0 = 2.5\ndelta = 0.1\nr_double = 1.4\nr_single = 1.4\nangle_deg = 180.0\nn_states = 40\nU_model = list(U_list)\nU_gold = list(U_list)\n',
         'call': 'triplet_pair_audit(N, U_model, eps, t0, delta, r_double, r_single, angle_deg, n_states)',
         'gold_call': '_oracle_triplet_pair_audit(N, U_gold, eps, t0, delta, r_double, r_single, angle_deg, n_states)'},
        {'setup': 'import numpy as np\nN = 4\nU_list = [14.0]\neps = 2.0\nt0 = 2.4\ndelta = 1.0 / 12.0\nr_double = 1.34\nr_single = 1.46\nangle_deg = 125.0\nn_states = 36\nU_model = list(U_list)\nU_gold = list(U_list)\n',
         'call': 'float(np.asarray(triplet_pair_audit(N, U_model, eps, t0, delta, r_double, r_single, angle_deg, n_states))[0, 0])',
         'gold_call': 'float(np.asarray(_oracle_triplet_pair_audit(N, U_gold, eps, t0, delta, r_double, r_single, angle_deg, n_states))[0, 0])'},
        {"setup": "import numpy as np\n# invalid input: an empty list of Coulomb parameters must raise ValueError\ndef _probe(fn):\n    try:\n        fn(8, [], 2.0, 2.4, 1.0 / 12.0, 1.35, 1.45, 120.0, 100)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n", "call": "_probe(triplet_pair_audit)", "gold_call": "_probe(_oracle_triplet_pair_audit)"}
    ]
