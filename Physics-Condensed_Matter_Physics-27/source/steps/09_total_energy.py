"""
Compute E_HF + E_c(RPA) for the lowest-energy Hartree-Fock-stable Jordan-Wigner determinant of
the spin cluster described by cfg.

cfg describes an open nx x ny cluster of spin-1/2 sites (x, y), 0 <= x < nx, 0 <= y < ny, with N
= nx ny sites numbered p = (x if y is even else nx - 1 - x) + y nx (a snake along x; ny = 1 is a
chain). The Hamiltonian is H = sum over bonds of J (s^x_p s^x_q + s^y_p s^y_q + Delta s^z_p
s^z_q), with spin operators s = sigma/2 and every bond counted once, open boundaries: nearest
neighbours (x, y)-(x+1, y) and (x, y)-(x, y+1) with J = J1, and the plaquette diagonals (x,
y)-(x+1, y+1) and (x, y)-(x+1, y-1) with J = J2.

The spins are mapped to spinless fermions, one orbital per site with orbital index p, by the
Jordan-Wigner transformation s^+_p = c^dag_p exp(i pi sum_{q<p} n_q), s^z_p = n_p - 1/2, n_p =
c^dag_p c_p. The sector is M = 0: N is even and every determinant holds N_f = N/2 fermions.

E_HF = <Phi|H|Phi> of the lowest-energy Hartree-Fock-stable determinant in the M = 0 sector (a
stationary rho that projects onto the N_f lowest eigenvectors of its Fock matrix F_kl = dE/d
rho_lk, with a positive definite orbital Hessian). E_c is the RPA correlation energy from that
determinant's orbital Hessian blocks in its canonical orbital basis, (1/4) [sum over the
positive RPA branch of omega - Tr A].

Converge max_kl |(F rho - rho F)_kl| below 1e-12: looser convergence moves E_c at the 1e-10
level. The comparison needs about 1e-10 relative accuracy.

Returns
-------
E_total : float -- E_HF + E_c.
"""

import numpy as np
from scipy.linalg import expm

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def total_energy(cfg: dict) -> float:
    '''Compute E_HF + E_c(RPA) for the lowest-energy Hartree-Fock-stable Jordan-Wigner
    determinant of the spin cluster described by cfg.

    Parameters
    ----------
    cfg : dict
        Cluster parameters. Keys read:
            nx : number of columns, x = 0, ..., nx-1
            ny : number of rows, y = 0, ..., ny-1 (ny = 1 is a chain)
            J1 : nearest-neighbour coupling
            J2 : coupling on both plaquette diagonals
            Delta : anisotropy of the s^z s^z coupling

    Returns
    -------
    E_total : float
        E_HF + E_c.
    '''
    return E_total

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_total_energy(cfg: dict) -> float:
    rho = _oracle_hf_solution(cfg)
    A, B = _oracle_orbital_hessian(rho, cfg)
    return float(_oracle_hf_energy(rho, cfg) + _oracle_rpa_correlation(A, B))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # normal: 5x2 ladder, J2 = 0.6
        {
            "setup": 'cfg = dict(nx=5, ny=2, J1=1.0, J2=0.6, Delta=1.0)\ncfg_ref = dict(cfg)',
            "call": 'total_energy(cfg)',
            "gold_call": '_oracle_total_energy(cfg_ref)',
        },
        # normal: 6x2 ladder, J2 = 0.4
        {
            "setup": 'cfg = dict(nx=6, ny=2, J1=1.0, J2=0.4, Delta=1.0)\ncfg_ref = dict(cfg)',
            "call": 'total_energy(cfg)',
            "gold_call": '_oracle_total_energy(cfg_ref)',
        },
        # normal: 4x3 cluster, J2 = 0.4
        {
            "setup": 'cfg = dict(nx=4, ny=3, J1=1.0, J2=0.4, Delta=1.0)\ncfg_ref = dict(cfg)',
            "call": 'total_energy(cfg)',
            "gold_call": '_oracle_total_energy(cfg_ref)',
        },
        # edge: 6x2 ladder at J2 = 0.6, the frustrated side
        {
            "setup": 'cfg = dict(nx=6, ny=2, J1=1.0, J2=0.6, Delta=1.0)\ncfg_ref = dict(cfg)',
            "call": 'total_energy(cfg)',
            "gold_call": '_oracle_total_energy(cfg_ref)',
        },
        # boundary: open XX chain; exact free fermions and zero RPA correction
        {
            "setup": "cfg = dict(nx=8, ny=1, J1=1.0, J2=0.0, Delta=0.0)\ncfg_ref = dict(cfg)",
            "call": "total_energy(cfg)",
            "gold_call": "_oracle_total_energy(cfg_ref)",
        },
    ]
