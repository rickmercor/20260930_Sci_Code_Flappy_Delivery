"""
Evaluate the total energy E[rho] = <Phi|H|Phi> of a Slater determinant of the
Jordan-Wigner-mapped spin Hamiltonian from its density matrix rho, including the strings of the
transverse couplings.

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

Return the real part as a float (the imaginary part vanishes at Hermitian rho). The comparison
needs about 1e-10 relative accuracy.

Returns
-------
E : float -- Energy <Phi|H|Phi>.
"""

import numpy as np
from scipy.linalg import expm

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hf_energy(rho: "np.ndarray", cfg: dict) -> float:
    '''Evaluate the total energy E[rho] = <Phi|H|Phi> of a Slater determinant of the
    Jordan-Wigner-mapped spin Hamiltonian from its density matrix rho, including the strings
    of the transverse couplings.

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
    E : float
        Energy <Phi|H|Phi>.
    '''
    return E

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _pairs(rho, cfg):
    """Per-pair intermediates (J/2, rows, cols, D, N) for every ordered bond pair (m, n)."""
    J, al = _coupling(cfg), _oracle_jw_pair_phases(cfg)
    out = []
    for m, n in zip(*np.nonzero(J)):
        S = np.nonzero(al[m, n])[0]
        rows, cols = np.r_[S, n], np.r_[S, m]
        D = np.r_[np.exp(1j * al[m, n, S]) - 1.0, 1.0]
        N = D[:, None] * rho[np.ix_(rows, cols)]
        N[np.arange(len(S)), np.arange(len(S))] += 1.0
        out.append((0.5 * J[m, n], rows, cols, D, N))
    return out


def _oracle_hf_energy(rho: "np.ndarray", cfg: dict) -> float:
    rho = np.asarray(rho, dtype=complex)
    Jz = cfg["Delta"] * _coupling(cfg)
    d = np.diag(rho)
    E = 0.5 * (d @ Jz @ d - np.sum(Jz * rho * rho.T)) - 0.5 * np.sum(Jz @ d) + 0.125 * np.sum(Jz)
    for c, rows, cols, D, N in _pairs(rho, cfg):
        E += c * np.linalg.det(N)
    return float(np.real(E))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # normal: a real rotated Neel determinant on a 5x2 ladder
        {
            "setup": 'import numpy as np\nfrom scipy.linalg import expm\ncfg = dict(nx=5, ny=2, J1=1.0, J2=0.6, Delta=1.0)\nN = 10\nocc = sorted((x if y % 2 == 0 else 4 - x) + y * 5 for x in range(5) for y in range(2) if (x + y) % 2 == 0)\nC = np.eye(N)[:, occ + [q for q in range(N) if q not in occ]]\nk = np.arange(N)\nG = 0.4 * np.sin(np.add.outer(k, 2 * k) + 1.0) + 0j * np.cos(np.add.outer(2 * k, k))\nC = expm(G - G.conj().T) @ C\nrho = C[:, :N // 2] @ C[:, :N // 2].conj().T\ncfg_ref = dict(cfg)\nrho_ref = rho.copy()',
            "call": 'hf_energy(rho, cfg)',
            "gold_call": '_oracle_hf_energy(rho_ref, cfg_ref)',
        },
        # normal: a complex determinant on a 6x2 ladder
        {
            "setup": 'import numpy as np\nfrom scipy.linalg import expm\ncfg = dict(nx=6, ny=2, J1=1.0, J2=0.4, Delta=1.0)\nN = 12\nocc = sorted((x if y % 2 == 0 else 5 - x) + y * 6 for x in range(6) for y in range(2) if (x + y) % 2 == 0)\nC = np.eye(N)[:, occ + [q for q in range(N) if q not in occ]]\nk = np.arange(N)\nG = 0.5 * np.sin(np.add.outer(k, 2 * k) + 1.0) + 0.2j * np.cos(np.add.outer(2 * k, k))\nC = expm(G - G.conj().T) @ C\nrho = C[:, :N // 2] @ C[:, :N // 2].conj().T\ncfg_ref = dict(cfg)\nrho_ref = rho.copy()',
            "call": 'hf_energy(rho, cfg)',
            "gold_call": '_oracle_hf_energy(rho_ref, cfg_ref)',
        },
        # normal: a complex determinant on a 4x3 cluster with Delta = 0.5
        {
            "setup": 'import numpy as np\nfrom scipy.linalg import expm\ncfg = dict(nx=4, ny=3, J1=1.0, J2=0.4, Delta=0.5)\nN = 12\nocc = sorted((x if y % 2 == 0 else 3 - x) + y * 4 for x in range(4) for y in range(3) if (x + y) % 2 == 0)\nC = np.eye(N)[:, occ + [q for q in range(N) if q not in occ]]\nk = np.arange(N)\nG = 0.7 * np.sin(np.add.outer(k, 2 * k) + 1.0) + 0.3j * np.cos(np.add.outer(2 * k, k))\nC = expm(G - G.conj().T) @ C\nrho = C[:, :N // 2] @ C[:, :N // 2].conj().T\ncfg_ref = dict(cfg)\nrho_ref = rho.copy()',
            "call": 'hf_energy(rho, cfg)',
            "gold_call": '_oracle_hf_energy(rho_ref, cfg_ref)',
        },
        # boundary: J2 = 0, strings only on the rungs
        {
            "setup": 'import numpy as np\nfrom scipy.linalg import expm\ncfg = dict(nx=6, ny=2, J1=1.0, J2=0.0, Delta=1.0)\nN = 12\nocc = sorted((x if y % 2 == 0 else 5 - x) + y * 6 for x in range(6) for y in range(2) if (x + y) % 2 == 0)\nC = np.eye(N)[:, occ + [q for q in range(N) if q not in occ]]\nk = np.arange(N)\nG = 0.6 * np.sin(np.add.outer(k, 2 * k) + 1.0) + 0.15j * np.cos(np.add.outer(2 * k, k))\nC = expm(G - G.conj().T) @ C\nrho = C[:, :N // 2] @ C[:, :N // 2].conj().T\ncfg_ref = dict(cfg)\nrho_ref = rho.copy()',
            "call": 'hf_energy(rho, cfg)',
            "gold_call": '_oracle_hf_energy(rho_ref, cfg_ref)',
        },
        # edge: the unrotated Neel determinant of a six-site chain, E = -(N - 1)/4
        {
            "setup": 'import numpy as np\nfrom scipy.linalg import expm\ncfg = dict(nx=6, ny=1, J1=1.0, J2=0.0, Delta=1.0)\nN = 6\nocc = sorted((x if y % 2 == 0 else 5 - x) + y * 6 for x in range(6) for y in range(1) if (x + y) % 2 == 0)\nC = np.eye(N)[:, occ + [q for q in range(N) if q not in occ]]\nk = np.arange(N)\nG = 0.0 * np.sin(np.add.outer(k, 2 * k) + 1.0) + 0j * np.cos(np.add.outer(2 * k, k))\nC = expm(G - G.conj().T) @ C\nrho = C[:, :N // 2] @ C[:, :N // 2].conj().T\ncfg_ref = dict(cfg)\nrho_ref = rho.copy()',
            "call": 'hf_energy(rho, cfg)',
            "gold_call": '_oracle_hf_energy(rho_ref, cfg_ref)',
        },
        # normal: a complex determinant on the 12x2 ladder, strings crossing up to 22 sites
        {
            "setup": 'import numpy as np\nfrom scipy.linalg import expm\ncfg = dict(nx=12, ny=2, J1=1.0, J2=0.6, Delta=1.0)\nN = 24\nocc = sorted((x if y % 2 == 0 else 11 - x) + y * 12 for x in range(12) for y in range(2) if (x + y) % 2 == 0)\nC = np.eye(N)[:, occ + [q for q in range(N) if q not in occ]]\nk = np.arange(N)\nG = 0.45 * np.sin(np.add.outer(k, 2 * k) + 1.0) + 0.1j * np.cos(np.add.outer(2 * k, k))\nC = expm(G - G.conj().T) @ C\nrho = C[:, :N // 2] @ C[:, :N // 2].conj().T\ncfg_ref = dict(cfg)\nrho_ref = rho.copy()',
            "call": 'hf_energy(rho, cfg)',
            "gold_call": '_oracle_hf_energy(rho_ref, cfg_ref)',
        },
    ]
