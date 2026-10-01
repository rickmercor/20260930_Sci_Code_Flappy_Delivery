"""
Compute the Fock matrix F^z of the longitudinal part sum over bonds of J Delta s^z_p s^z_q of
the Hamiltonian for a Slater determinant with density matrix rho.

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

E^z[rho] = <Phi| sum over bonds of J Delta s^z_p s^z_q |Phi> is taken as the polynomial in the
entries of rho that Wick's theorem gives (expectation values of operator products on distinct
sites written as determinants of submatrices of rho, never using rho^2 = rho), constants from
s^z = n - 1/2 included. F^z_kl = dE^z/d rho_lk with all N^2 entries of rho independent, so that
dE^z = Tr(F^z d rho).

Returns
-------
Fz : np.ndarray -- Complex array of shape (N, N).
"""

import numpy as np
from scipy.linalg import expm

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def z_fock(rho: "np.ndarray", cfg: dict) -> "np.ndarray":
    '''Compute the Fock matrix F^z of the longitudinal part sum over bonds of J Delta s^z_p
    s^z_q of the Hamiltonian for a Slater determinant with density matrix rho.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix of shape (N, N), rho_kl = <c^dag_l c_k>.
    cfg : dict
        Cluster parameters. Keys read:
            nx : number of columns, x = 0, ..., nx-1
            ny : number of rows, y = 0, ..., ny-1 (ny = 1 is a chain)
            J1 : nearest-neighbour coupling
            J2 : coupling on both plaquette diagonals
            Delta : anisotropy of the s^z s^z coupling

    Returns
    -------
    Fz : np.ndarray
        Complex array of shape (N, N).
    '''
    return Fz

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _site(x, y, nx):
    """Snake site index of (x, y)."""
    return (x if y % 2 == 0 else nx - 1 - x) + y * nx


def _coupling(cfg):
    """Symmetric N x N matrix of bond couplings J (zero on non-bonds)."""
    nx, ny = cfg["nx"], cfg["ny"]
    J = np.zeros((nx * ny, nx * ny))
    for x in range(nx):
        for y in range(ny):
            for dx, dy, c in ((1, 0, cfg["J1"]), (0, 1, cfg["J1"]), (1, 1, cfg["J2"]), (1, -1, cfg["J2"])):
                if x + dx < nx and 0 <= y + dy < ny:
                    p, q = _site(x, y, nx), _site(x + dx, y + dy, nx)
                    J[p, q] = J[q, p] = c
    return J


def _oracle_z_fock(rho: "np.ndarray", cfg: dict) -> "np.ndarray":
    rho = np.asarray(rho, dtype=complex)
    Jz = cfg["Delta"] * _coupling(cfg)
    return np.diag(Jz @ (np.diag(rho) - 0.5)) - Jz * rho

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # normal: a real rotated Neel determinant on a 5x2 ladder
        {
            "setup": 'import numpy as np\nfrom scipy.linalg import expm\ncfg = dict(nx=5, ny=2, J1=1.0, J2=0.6, Delta=1.0)\nN = 10\nocc = sorted((x if y % 2 == 0 else 4 - x) + y * 5 for x in range(5) for y in range(2) if (x + y) % 2 == 0)\nC = np.eye(N)[:, occ + [q for q in range(N) if q not in occ]]\nk = np.arange(N)\nG = 0.4 * np.sin(np.add.outer(k, 2 * k) + 1.0) + 0j * np.cos(np.add.outer(2 * k, k))\nC = expm(G - G.conj().T) @ C\nrho = C[:, :N // 2] @ C[:, :N // 2].conj().T\ncfg_ref = dict(cfg)\nrho_ref = rho.copy()',
            "call": 'z_fock(rho, cfg) + 4.0',
            "gold_call": '_oracle_z_fock(rho_ref, cfg_ref) + 4.0',
        },
        # normal: a complex determinant on a 4x3 cluster with Delta = 0.5
        {
            "setup": 'import numpy as np\nfrom scipy.linalg import expm\ncfg = dict(nx=4, ny=3, J1=1.0, J2=0.4, Delta=0.5)\nN = 12\nocc = sorted((x if y % 2 == 0 else 3 - x) + y * 4 for x in range(4) for y in range(3) if (x + y) % 2 == 0)\nC = np.eye(N)[:, occ + [q for q in range(N) if q not in occ]]\nk = np.arange(N)\nG = 0.7 * np.sin(np.add.outer(k, 2 * k) + 1.0) + 0.3j * np.cos(np.add.outer(2 * k, k))\nC = expm(G - G.conj().T) @ C\nrho = C[:, :N // 2] @ C[:, :N // 2].conj().T\ncfg_ref = dict(cfg)\nrho_ref = rho.copy()',
            "call": 'z_fock(rho, cfg) + 4.0',
            "gold_call": '_oracle_z_fock(rho_ref, cfg_ref) + 4.0',
        },
        # boundary: ferromagnetic longitudinal coupling, Delta = -0.5
        {
            "setup": 'import numpy as np\nfrom scipy.linalg import expm\ncfg = dict(nx=6, ny=2, J1=1.0, J2=0.4, Delta=-0.5)\nN = 12\nocc = sorted((x if y % 2 == 0 else 5 - x) + y * 6 for x in range(6) for y in range(2) if (x + y) % 2 == 0)\nC = np.eye(N)[:, occ + [q for q in range(N) if q not in occ]]\nk = np.arange(N)\nG = 0.5 * np.sin(np.add.outer(k, 2 * k) + 1.0) + 0.2j * np.cos(np.add.outer(2 * k, k))\nC = expm(G - G.conj().T) @ C\nrho = C[:, :N // 2] @ C[:, :N // 2].conj().T\ncfg_ref = dict(cfg)\nrho_ref = rho.copy()',
            "call": 'z_fock(rho, cfg) + 4.0',
            "gold_call": '_oracle_z_fock(rho_ref, cfg_ref) + 4.0',
        },
        # edge: the unrotated Neel determinant of a six-site chain
        {
            "setup": 'import numpy as np\nfrom scipy.linalg import expm\ncfg = dict(nx=6, ny=1, J1=1.0, J2=0.0, Delta=1.0)\nN = 6\nocc = sorted((x if y % 2 == 0 else 5 - x) + y * 6 for x in range(6) for y in range(1) if (x + y) % 2 == 0)\nC = np.eye(N)[:, occ + [q for q in range(N) if q not in occ]]\nk = np.arange(N)\nG = 0.0 * np.sin(np.add.outer(k, 2 * k) + 1.0) + 0j * np.cos(np.add.outer(2 * k, k))\nC = expm(G - G.conj().T) @ C\nrho = C[:, :N // 2] @ C[:, :N // 2].conj().T\ncfg_ref = dict(cfg)\nrho_ref = rho.copy()',
            "call": 'z_fock(rho, cfg) + 4.0',
            "gold_call": '_oracle_z_fock(rho_ref, cfg_ref) + 4.0',
        },
    ]
