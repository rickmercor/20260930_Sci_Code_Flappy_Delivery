"""
Compute the Fock matrix F_kl = dE/d rho_lk of the Jordan-Wigner-mapped spin Hamiltonian,
longitudinal and string (transverse) parts together, for a Slater determinant with density
matrix rho.

cfg describes an open nx x ny cluster of spin-1/2 sites (x, y), 0 <= x < nx, 0 <= y < ny, with N
= nx ny sites numbered p = (x if y is even else nx - 1 - x) + y nx (a snake along x; ny = 1 is a
chain). The Hamiltonian is H = sum over bonds of J (s^x_p s^x_q + s^y_p s^y_q + Delta s^z_p
s^z_q), with spin operators s = sigma/2 and every bond counted once, open boundaries: nearest
neighbours (x, y)-(x+1, y) and (x, y)-(x, y+1) with J = J1, and the plaquette diagonals (x,
y)-(x+1, y+1) and (x, y)-(x+1, y-1) with J = J2.

The spins are mapped to spinless fermions, one orbital per site with orbital index p, by the
Jordan-Wigner transformation s^+_p = c^dag_p exp(i pi sum_{q<p} n_q), s^z_p = n_p - 1/2, n_p =
c^dag_p c_p. The sector is M = 0: N is even and every determinant holds N_f = N/2 fermions.

A Slater determinant |Phi> of N_f fermions enters only through its one-body density matrix
rho_kl = <Phi| c^dag_l c_k |Phi> (note the index order): an N x N Hermitian idempotent matrix of
trace N_f, complex in general.

In a transverse coupling s^x_m s^x_n + s^y_m s^y_n = (s^+_m s^-_n + s^+_n s^-_m)/2 the strings
of the two sites combine: s^+_m s^-_n = c^dag_m c_n exp(i sum_{q not in {m, n}} alpha[m, n, q]
n_q), with the pair phases alpha[m, n, q] = theta_mq - theta_nq of jw_pair_phases (theta_pq = pi
for q < p, 0 otherwise).

E[rho] = <Phi|H|Phi> is taken as the function of rho that Wick's theorem gives: every
expectation value of a product of creation and annihilation operators on distinct sites is
written as a determinant of a submatrix of rho, without ever using rho^2 = rho. This makes E a
fixed polynomial in the N^2 entries of rho, and all derivatives treat those entries as
independent variables (no Hermiticity constraint).

The Fock matrix is F_kl = dE/d rho_lk, so that dE = Tr(F d rho); it is Hermitian at Hermitian
rho.

The comparison needs about 1e-10 relative accuracy.

Returns
-------
F : np.ndarray -- Complex array of shape (N, N).
"""

import numpy as np
from scipy.linalg import expm

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fock_matrix(rho: "np.ndarray", cfg: dict) -> "np.ndarray":
    '''Compute the Fock matrix F_kl = dE/d rho_lk of the Jordan-Wigner-mapped spin Hamiltonian,
    longitudinal and string (transverse) parts together, for a Slater determinant with
    density matrix rho.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix of shape (N, N) of a Slater determinant, rho_kl = <c^dag_l c_k>.
    cfg : dict
        Cluster parameters. Keys read:
            nx : number of columns, x = 0, ..., nx-1
            ny : number of rows, y = 0, ..., ny-1 (ny = 1 is a chain)
            J1 : nearest-neighbour coupling
            J2 : coupling on both plaquette diagonals
            Delta : anisotropy of the s^z s^z coupling

    Returns
    -------
    F : np.ndarray
        Complex array of shape (N, N).
    '''
    return F

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _adj(N, second=False):
    """adj(N) with adj_ji = d det N / dN_ij and, if second, d2 det N / dN_ij dN_kl; finite for singular N."""
    U, s, Vh = np.linalg.svd(N)
    m = len(s)
    i = np.arange(m)
    M = np.broadcast_to(s, (m, m, m)).copy()
    M[i, :, i] = 1.0
    M[:, i, i] = 1.0
    W = np.linalg.det(U) * np.linalg.det(Vh) * M.prod(axis=2)
    V = Vh.conj().T
    adj = (V * np.diag(W)) @ U.conj().T
    if not second:
        return adj
    Y = (U.conj()[:, None, :] * V[None, :, :]).reshape(m * m, m)
    T = (Y @ W @ Y.T).reshape(m, m, m, m)
    return adj, T - T.transpose(0, 3, 2, 1)


def _oracle_fock_matrix(rho: "np.ndarray", cfg: dict) -> "np.ndarray":
    rho = np.asarray(rho, dtype=complex)
    F = _oracle_z_fock(rho, cfg)
    for c, rows, cols, D, N in _pairs(rho, cfg):
        F[np.ix_(cols, rows)] += c * _adj(N) * D
    return F

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # normal: a real rotated Neel determinant on a 5x2 ladder
        {
            "setup": 'import numpy as np\nfrom scipy.linalg import expm\ncfg = dict(nx=5, ny=2, J1=1.0, J2=0.6, Delta=1.0)\nN = 10\nocc = sorted((x if y % 2 == 0 else 4 - x) + y * 5 for x in range(5) for y in range(2) if (x + y) % 2 == 0)\nC = np.eye(N)[:, occ + [q for q in range(N) if q not in occ]]\nk = np.arange(N)\nG = 0.4 * np.sin(np.add.outer(k, 2 * k) + 1.0) + 0j * np.cos(np.add.outer(2 * k, k))\nC = expm(G - G.conj().T) @ C\nrho = C[:, :N // 2] @ C[:, :N // 2].conj().T\ncfg_ref = dict(cfg)\nrho_ref = rho.copy()',
            "call": 'fock_matrix(rho, cfg) + 4.0',
            "gold_call": '_oracle_fock_matrix(rho_ref, cfg_ref) + 4.0',
        },
        # normal: a complex determinant on a 4x3 cluster
        {
            "setup": 'import numpy as np\nfrom scipy.linalg import expm\ncfg = dict(nx=4, ny=3, J1=1.0, J2=0.4, Delta=1.0)\nN = 12\nocc = sorted((x if y % 2 == 0 else 3 - x) + y * 4 for x in range(4) for y in range(3) if (x + y) % 2 == 0)\nC = np.eye(N)[:, occ + [q for q in range(N) if q not in occ]]\nk = np.arange(N)\nG = 0.5 * np.sin(np.add.outer(k, 2 * k) + 1.0) + 0.25j * np.cos(np.add.outer(2 * k, k))\nC = expm(G - G.conj().T) @ C\nrho = C[:, :N // 2] @ C[:, :N // 2].conj().T\ncfg_ref = dict(cfg)\nrho_ref = rho.copy()',
            "call": 'fock_matrix(rho, cfg) + 4.0',
            "gold_call": '_oracle_fock_matrix(rho_ref, cfg_ref) + 4.0',
        },
        # normal: a 6x2 ladder with Delta = 1.3
        {
            "setup": 'import numpy as np\nfrom scipy.linalg import expm\ncfg = dict(nx=6, ny=2, J1=1.0, J2=0.6, Delta=1.3)\nN = 12\nocc = sorted((x if y % 2 == 0 else 5 - x) + y * 6 for x in range(6) for y in range(2) if (x + y) % 2 == 0)\nC = np.eye(N)[:, occ + [q for q in range(N) if q not in occ]]\nk = np.arange(N)\nG = 0.6 * np.sin(np.add.outer(k, 2 * k) + 1.0) + 0j * np.cos(np.add.outer(2 * k, k))\nC = expm(G - G.conj().T) @ C\nrho = C[:, :N // 2] @ C[:, :N // 2].conj().T\ncfg_ref = dict(cfg)\nrho_ref = rho.copy()',
            "call": 'fock_matrix(rho, cfg) + 4.0',
            "gold_call": '_oracle_fock_matrix(rho_ref, cfg_ref) + 4.0',
        },
        # edge: a complex determinant of a six-site chain, where no string survives
        {
            "setup": 'import numpy as np\nfrom scipy.linalg import expm\ncfg = dict(nx=6, ny=1, J1=1.0, J2=0.0, Delta=1.0)\nN = 6\nocc = sorted((x if y % 2 == 0 else 5 - x) + y * 6 for x in range(6) for y in range(1) if (x + y) % 2 == 0)\nC = np.eye(N)[:, occ + [q for q in range(N) if q not in occ]]\nk = np.arange(N)\nG = 0.5 * np.sin(np.add.outer(k, 2 * k) + 1.0) + 0.3j * np.cos(np.add.outer(2 * k, k))\nC = expm(G - G.conj().T) @ C\nrho = C[:, :N // 2] @ C[:, :N // 2].conj().T\ncfg_ref = dict(cfg)\nrho_ref = rho.copy()',
            "call": 'fock_matrix(rho, cfg) + 4.0',
            "gold_call": '_oracle_fock_matrix(rho_ref, cfg_ref) + 4.0',
        },
        # normal: a complex determinant on the 12x2 ladder, strings crossing up to 22 sites
        {
            "setup": 'import numpy as np\nfrom scipy.linalg import expm\ncfg = dict(nx=12, ny=2, J1=1.0, J2=0.6, Delta=1.0)\nN = 24\nocc = sorted((x if y % 2 == 0 else 11 - x) + y * 12 for x in range(12) for y in range(2) if (x + y) % 2 == 0)\nC = np.eye(N)[:, occ + [q for q in range(N) if q not in occ]]\nk = np.arange(N)\nG = 0.45 * np.sin(np.add.outer(k, 2 * k) + 1.0) + 0.1j * np.cos(np.add.outer(2 * k, k))\nC = expm(G - G.conj().T) @ C\nrho = C[:, :N // 2] @ C[:, :N // 2].conj().T\ncfg_ref = dict(cfg)\nrho_ref = rho.copy()',
            "call": 'fock_matrix(rho, cfg) + 4.0',
            "gold_call": '_oracle_fock_matrix(rho_ref, cfg_ref) + 4.0',
        },
    ]
