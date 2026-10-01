"""
Return the dense (D, D) matrix of the PPP Hamiltonian in the ordered determinant basis of the (n_up, n_down) sector, D = C(N, n_up) C(N, n_down), in eV. Model and conventions: an all-trans polyene of N carbon atoms (N even), sites i = 0, ..., N-1, N_d = N/2 ethylene dimers with dimer n formed by the sites 2n and 2n+1, one pi electron per site, described by the Pariser-Parr-Pople Hamiltonian H = -sum_{i,sigma} t_i (c+_{i,sigma} c_{i+1,sigma} + h.c.) + U sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2) + sum_{i<j} V_ij (n_i - 1)(n_j - 1), energies in eV; bond-alternated hopping t_i = t0 (1 + delta) for even i (the double bonds 0-1, 2-3, ...) and t_i = t0 (1 - delta) for odd i; Ohno potential V_ij = U / sqrt(1 + (U eps r_ij / 14.397)^2) with U in eV, r_ij in angstrom, 14.397 eV angstrom = e^2/(4 pi eps_0) and eps the relative permittivity. Geometry: site 0 at the origin; bond i (from site i to site i+1) has length r_double for even i and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg; r_ij are the resulting planar distances. Determinant basis of the (n_up, n_down) sector: an up-spin configuration is the bit mask sum over occupied sites of 2^i, likewise a down-spin configuration; the C(N, n_up) up configurations and the C(N, n_down) down configurations are each listed in increasing integer order and the basis index of a determinant is i_up D_down + i_down; a determinant is the string of all up creation operators (ascending site) followed by all down creation operators (ascending site) acting on the vacuum, so that nearest-neighbour hopping matrix elements carry no fermionic sign. Symmetries of the half-filled S_z = 0 sector (n_up = n_down = N/2): S(S+1) from the total spin S^2 = S- S+ + S_z (S_z + 1) with S+ = sum_i c+_{i,up} c_{i,down} and S- its adjoint; the site reversal P maps a determinant to the determinant with site i replaced by N-1-i; the occupation complement J maps a determinant to the determinant with every occupation inverted (empty <-> occupied for each spin); both permutations, with the constant sign of the operator reordering, commute with H. Labels of an eigenstate k are S(S+1) and the products p_k p_0 and j_k j_0 of its P and J eigenvalues with those of the ground state (the ground state reads +1, +1); inside a cluster of eigenvalues closer than 1e-7 (relative to the largest computed energy magnitude) the eigenvectors are rotated so that S^2, P and J are simultaneously diagonal and the members are ordered by (S(S+1), p, j) ascending. Covalent sector: singlets and quintets with j_k j_0 = +1, triplets with j_k j_0 = -1 (on singly occupied configurations the complement acts as the global spin flip, whose M = 0 eigenvalue alternates with S). State names: 1^1Ag+ = ground state; 2^1Ag+ = the second singlet with (p, j) = (+1, +1) (the dark state); 1^1Bu- = the lowest singlet with (-1, -1) (the optically bright state); 1^1Bu+ = the lowest singlet with (-1, +1); 1^3Bu = the lowest triplet (-1, -1); 1^5Ag+ = the lowest quintet with (+1, +1). Subchains and triplet families: a partition of the N_d dimers into a left subchain of m dimers (sites 0 to 2m-1) and a right subchain of N_d - m dimers (sites 2m to N-1); each subchain is treated as an isolated open PPP chain with the same parameters and the distances of its own segment of the geometry; its covalent triplets are its triplet eigenstates with j opposite to its own ground state (labels resolved as above); its triplet family T_j(m), j = 1, ..., m, consists of the m lowest covalent triplets; the spin components of a triplet are related by the ladder operators, T_{+1} = S+ T_0 / sqrt2 and T_{-1} = S- T_0 / sqrt2. Triplet-pair products: |A x B> is the site-ordered (Jordan-Wigner) tensor product, the creation-operator string of the left state in site order (site 0 up, site 0 down, site 1 up, ...) followed by that of the right state with its sites shifted by 2m, re-expressed in the full-chain determinant basis with the sign of the reordering; T0T0 = T_0(left) x T_0(right); the singlet-coupled pair is 1TT = (T_{+1} x T_{-1} - T_0 x T_0 + T_{-1} x T_{+1})/sqrt3 and the quintet-coupled (M = 0) pair is 2TT = (T_{+1} x T_{-1} + 2 T_0 x T_0 + T_{-1} x T_{+1})/sqrt6 (Clebsch-Gordan coupling of two spin-1 objects). Pair bases: 'paper' = {T_j(m) x T_1(N_d - m): 1 <= m <= N_d - 1, 1 <= j <= m}; 'family' = {T_j(m) x T_k(N_d - m): all family members on both subchains}; 'covalent' = every pair of covalent triplets of the two subchains (linearly dependent: the span is defined by the singular values of the product matrix above 1e-8 of the largest). Populations of a state Psi in a pair basis: the T0T0 prescription is the squared norm of the projection of Psi onto the span of the T0T0 products multiplied by 3 for a singlet and by 3/2 for a quintet (the factor that would convert a single M = 0 product into its spin-coupled counterpart); the exact spin-adapted population is the squared norm of the projection of Psi onto the span of the singlet-coupled products (singlet states) or of the quintet-coupled products (quintet state). Reference parameters: N = 8 (octatetraene), U = 8 eV (audit set 8, 4, 14 eV), eps = 2, t0 = 2.4 eV, delta = 1/12, r_double = 1.35 angstrom, r_single = 1.45 angstrom, angle_deg = 120, n_states = 100. Raise ValueError for an invalid geometry or model parameter, or if n_up or n_down is not an integer between 0 and N.

Exact diagonalization of the pi-electron Hamiltonian is what makes the correlated dark state of a short polyene available without approximation; the determinant ordering and the operator ordering fixed here are what every later projection relies on.

Returns
-------
A (D, D) float64 symmetric array in eV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ppp_hamiltonian(N: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, n_up: int, n_down: int) -> np.ndarray:
    """Return the dense (D, D) matrix of the PPP Hamiltonian in the ordered determinant basis
    of the (n_up, n_down) sector, D = C(N, n_up) C(N, n_down), in eV. Model and conventions:
    an all-trans polyene of N carbon atoms (N even), sites i = 0, ..., N-1, N_d = N/2
    ethylene dimers with dimer n formed by the sites 2n and 2n+1, one pi electron per site,
    described by the Pariser-Parr-Pople Hamiltonian H = -sum_{i,sigma} t_i (c+_{i,sigma}
    c_{i+1,sigma} + h.c.) + U sum_i (n_{i,up} - 1/2)(n_{i,down} - 1/2) + sum_{i<j} V_ij (n_i
    - 1)(n_j - 1), energies in eV; bond-alternated hopping t_i = t0 (1 + delta) for even i
    (the double bonds 0-1, 2-3, ...) and t_i = t0 (1 - delta) for odd i; Ohno potential V_ij
    = U / sqrt(1 + (U eps r_ij / 14.397)^2) with U in eV, r_ij in angstrom, 14.397 eV
    angstrom = e^2/(4 pi eps_0) and eps the relative permittivity.

    Parameters
    ----------
    N : int
        Number of carbon atoms (even, >= 2).
    U : float
        On-site Coulomb parameter in eV (>= 0).
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
    n_up : int
        Number of up-spin electrons (0 <= n_up <= N).
    n_down : int
        Number of down-spin electrons (0 <= n_down <= N).

    Returns
    -------
    A (D, D) float64 symmetric array in eV.

    Raises
    ------
    ValueError
        For an invalid geometry or model parameter, or if n_up or n_down is not an
        integer between 0 and N.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import sqrt, pi, cos, sin
from scipy.linalg import eigh


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

def _positions(N, r_double, r_single, angle_deg):
    """All-trans zigzag geometry: bond i (atom i to atom i + 1) has length r_double for even i
    and r_single for odd i and makes the angle +phi (even i) or -phi (odd i) with the chain
    axis, phi = (180 - angle_deg)/2 degrees, so that every C-C-C angle equals angle_deg."""
    phi = (180.0 - angle_deg) * pi / 360.0
    pos = np.zeros((N, 2))
    for i in range(N - 1):
        ell = r_double if i % 2 == 0 else r_single
        ang = phi if i % 2 == 0 else -phi
        pos[i + 1] = pos[i] + ell * np.array([cos(ang), sin(ang)])
    return pos

def _ohno(U, eps, r):
    return U / np.sqrt(1.0 + (U * eps * r / 14.397) ** 2)

def _popcount_table(N):
    t = np.zeros(1 << N, dtype=np.int64)
    for b in range(N):
        t[(np.arange(1 << N) >> b) & 1 == 1] += 1
    return t

def _configs(N, k):
    """All bit masks of N bits with exactly k bits set, ascending as integers."""
    allm = np.arange(1 << N, dtype=np.int64)
    return allm[_popcount_table(N)[allm] == k]

def _sector(N, n_up, n_down):
    """Determinant basis of the (n_up, n_down) sector: index = i_up * D_down + i_down, up and
    down configurations ascending as integers; creation operators ordered all-up (ascending
    site) then all-down (ascending site)."""
    cu = _configs(N, n_up)
    cd = _configs(N, n_down)
    u = np.repeat(cu, len(cd))
    d = np.tile(cd, len(cu))
    return cu, cd, u, d

def _index(cu, cd, u, d):
    iu = np.searchsorted(cu, u)
    idn = np.searchsorted(cd, d)
    return iu * len(cd) + idn

def _hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_up, n_down):
    cu, cd, u, d = _sector(N, n_up, n_down)
    D = len(u)
    pos = _positions(N, r_double, r_single, angle_deg)
    r = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
    V = _ohno(U, eps, r)
    np.fill_diagonal(V, 0.0)
    occ_u = ((u[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
    occ_d = ((d[:, None] >> np.arange(N)[None, :]) & 1).astype(float)
    q = occ_u + occ_d - 1.0
    diag = U * ((occ_u - 0.5) * (occ_d - 0.5)).sum(axis=1) + 0.5 * np.einsum("ki,ij,kj->k", q, V, q)
    H = np.zeros((D, D))
    H[np.arange(D), np.arange(D)] = diag
    cols = np.arange(D)
    for i in range(N - 1):
        t = t0 * (1.0 + delta) if i % 2 == 0 else t0 * (1.0 - delta)
        j = i + 1
        for a, b in ((i, j), (j, i)):
            ma, mb = 1 << a, 1 << b
            for spin in (0, 1):
                occ = u if spin == 0 else d
                sel = ((occ & ma) != 0) & ((occ & mb) == 0)
                new = occ[sel] ^ ma ^ mb
                if spin == 0:
                    rows = _index(cu, cd, new, d[sel])
                else:
                    rows = _index(cu, cd, u[sel], new)
                H[rows, cols[sel]] += -t
    return H

def _oracle_ppp_hamiltonian(N: int, U: float, eps: float, t0: float, delta: float, r_double: float, r_single: float, angle_deg: float, n_up: int, n_down: int) -> np.ndarray:
    _check_model(N, U, eps, t0, delta, r_double, r_single, angle_deg)
    for x in (n_up, n_down):
        if not (isinstance(x, (int, np.integer)) and not isinstance(x, bool)) or x < 0 or x > N:
            raise ValueError("electron numbers must be integers between 0 and N")
    return _hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_up, n_down)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nN = 4\nU = 14.0\neps = 2.0\nt0 = 2.4\ndelta = 1.0 / 12.0\nr_double = 1.34\nr_single = 1.46\nangle_deg = 125.0\nn_states = 36\nn_up = 2\nn_down = 2\n',
         'call': 'ppp_hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_up, n_down)',
         'gold_call': '_oracle_ppp_hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_up, n_down)'},
        {'setup': 'import numpy as np\nN = 6\nU = 4.0\neps = 1.0\nt0 = 2.5\ndelta = 0.1\nr_double = 1.4\nr_single = 1.4\nangle_deg = 180.0\nn_states = 40\nn_up = 3\nn_down = 3\n',
         'call': 'ppp_hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_up, n_down)',
         'gold_call': '_oracle_ppp_hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_up, n_down)'},
        {'setup': 'import numpy as np\nN = 4\nU = 14.0\neps = 2.0\nt0 = 2.4\ndelta = 1.0 / 12.0\nr_double = 1.34\nr_single = 1.46\nangle_deg = 125.0\nn_states = 36\nn_up = 3\nn_down = 1\n',
         'call': 'ppp_hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_up, n_down)',
         'gold_call': '_oracle_ppp_hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_up, n_down)'},
        {'setup': 'import numpy as np\nN = 6\nU = 8.0\neps = 2.0\nt0 = 2.4\ndelta = 1.0 / 12.0\nr_double = 1.35\nr_single = 1.45\nangle_deg = 120.0\nn_up = 4\nn_down = 2\n',
         'call': 'ppp_hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_up, n_down)',
         'gold_call': '_oracle_ppp_hamiltonian(N, U, eps, t0, delta, r_double, r_single, angle_deg, n_up, n_down)'},
        {"setup": "import numpy as np\n# invalid input: more up electrons than sites must raise ValueError\ndef _probe(fn):\n    try:\n        fn(8, 8.0, 2.0, 2.4, 1.0 / 12.0, 1.35, 1.45, 120.0, 9, 4)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n", "call": "_probe(ppp_hamiltonian)", "gold_call": "_probe(_oracle_ppp_hamiltonian)"}
    ]
