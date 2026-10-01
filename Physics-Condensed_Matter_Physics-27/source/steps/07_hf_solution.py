"""
Find the density matrix of the lowest-energy Hartree-Fock-stable Slater determinant of the
Jordan-Wigner-mapped spin Hamiltonian in the M = 0 sector.

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

A determinant is stationary when rho is the projector onto the N_f lowest eigenvectors of its
own Fock matrix F[rho], and Hartree-Fock stable when its orbital Hessian H = [[A, B], [B*, A*]]
(as defined in orbital_hessian) is positive definite. Return the density matrix of the stable
stationary determinant with the lowest E[rho]; several stationary determinants typically exist,
some of them saddle points close in energy. The returned rho must be idempotent and satisfy
max_kl |(F rho - rho F)_kl| < 1e-12.

H is invariant under a global spin flip, which maps a solution rho to the equal-energy
stationary partner 1 - L rho^T L with L = diag((-1)^p); complex conjugation of rho gives
another, and reversing the site order p -> N - 1 - p (a symmetry of the cluster and of its
numbering) gives a third. Any of these equivalent solutions is acceptable: the entrywise
magnitudes |rho_kl - delta_kl/2| are the same for all of them.

Returns
-------
rho : np.ndarray -- Complex array of shape (N, N), the density matrix rho_kl = <c^dag_l c_k>.
"""

import numpy as np
from scipy.linalg import expm

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hf_solution(cfg: dict) -> "np.ndarray":
    '''Find the density matrix of the lowest-energy Hartree-Fock-stable Slater determinant of
    the Jordan-Wigner-mapped spin Hamiltonian in the M = 0 sector.

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
    rho : np.ndarray
        Complex array of shape (N, N), the density matrix rho_kl = <c^dag_l c_k>.
    '''
    return rho

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm


def _scf(rho, cfg):
    """Aufbau SCF with DIIS extrapolation of F, converged to max|[F, rho]| < 1e-12."""
    Nf = len(rho) // 2
    Fs, Es = [], []
    for it in range(500):
        F = _oracle_fock_matrix(rho, cfg)
        F = 0.5 * (F + F.conj().T)
        e = F @ rho - rho @ F
        if np.abs(e).max() < 1e-12:
            break
        Fs, Es = (Fs + [F])[-8:], (Es + [e])[-8:]
        n = len(Fs)
        if n >= 3 and np.abs(e).max() < 0.1:
            M = -np.ones((n + 1, n + 1))
            M[n, n] = 0.0
            M[:n, :n] = [[np.vdot(a, b).real for b in Es] for a in Es]
            c = np.linalg.solve(M, np.r_[np.zeros(n), -1.0])
            F = sum(ci * Fi for ci, Fi in zip(c, Fs))
        v = np.linalg.eigh(F)[1][:, :Nf]
        rho = v @ v.conj().T
    return rho


def _oracle_hf_solution(cfg: dict) -> "np.ndarray":
    nx, ny = cfg["nx"], cfg["ny"]
    N = nx * ny
    Nf = N // 2
    occ = [_site(x, y, nx) for x in range(nx) for y in range(ny) if (x + y) % 2 == 0]
    rho = np.zeros((N, N), dtype=complex)
    rho[occ, occ] = 1.0
    for cycle in range(20):
        rho = _scf(rho, cfg)
        A, B = _oracle_orbital_hessian(rho, cfg)
        n = len(A)
        lam, U = np.linalg.eigh(np.block([[A, B], [B.conj(), A.conj()]]))
        if lam[0] > 0.0:
            break
        # unstable: rotate the canonical orbitals along the lowest Hessian mode u = (z, z*) and re-converge
        x, y = U[:n, 0], U[n:, 0].conj()
        z = x + y if np.linalg.norm(x + y) > np.linalg.norm(x - y) else 1j * (x - y)
        F = _oracle_fock_matrix(rho, cfg)
        C = np.linalg.eigh(0.5 * (F + F.conj().T))[1]
        kap = np.zeros((N, N), dtype=complex)
        kap[Nf:, :Nf] = 0.3 * z.reshape(N - Nf, Nf) / np.linalg.norm(z)
        C = C @ expm(kap - kap.conj().T)
        rho = C[:, :Nf] @ C[:, :Nf].conj().T
    return rho

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # normal: 5x2 ladder, J2 = 0.6, full site-resolved magnetisation matrix compared entry by entry
        {
            "setup": 'import numpy as np\ncfg = dict(nx=5, ny=2, J1=1.0, J2=0.6, Delta=1.0)\ncfg_ref = dict(cfg)',
            "call": 'np.abs(hf_solution(cfg) - 0.5 * np.eye(10))',
            "gold_call": 'np.abs(_oracle_hf_solution(cfg_ref) - 0.5 * np.eye(10))',
        },
        # normal: 6x2 ladder, J2 = 0.4
        {
            "setup": 'import numpy as np\ncfg = dict(nx=6, ny=2, J1=1.0, J2=0.4, Delta=1.0)\ndef probe(r):\n    D = np.abs(r - 0.5 * np.eye(len(r)))\n    p = np.arange(len(r))\n    return np.array([np.sum(D * (1.0 + 0.5 * np.cos(j * (1.3 * p[:, None] + 0.7 * p[None, :]) + 0.4 * j))) for j in range(1, 5)] + [np.sum(D ** 2)])\ncfg_ref = dict(cfg)',
            "call": 'probe(hf_solution(cfg))',
            "gold_call": 'probe(_oracle_hf_solution(cfg_ref))',
        },
        # normal: 4x3 cluster, J2 = 0.4
        {
            "setup": 'import numpy as np\ncfg = dict(nx=4, ny=3, J1=1.0, J2=0.4, Delta=1.0)\ndef probe(r):\n    D = np.abs(r - 0.5 * np.eye(len(r)))\n    p = np.arange(len(r))\n    return np.array([np.sum(D * (1.0 + 0.5 * np.cos(j * (1.3 * p[:, None] + 0.7 * p[None, :]) + 0.4 * j))) for j in range(1, 5)] + [np.sum(D ** 2)])\ncfg_ref = dict(cfg)',
            "call": 'probe(hf_solution(cfg))',
            "gold_call": 'probe(_oracle_hf_solution(cfg_ref))',
        },
        # edge: open XX chain of eight sites, free fermions without strings
        {
            "setup": 'import numpy as np\ncfg = dict(nx=8, ny=1, J1=1.0, J2=0.0, Delta=0.0)\ndef probe(r):\n    D = np.abs(r - 0.5 * np.eye(len(r)))\n    p = np.arange(len(r))\n    return np.array([np.sum(D * (1.0 + 0.5 * np.cos(j * (1.3 * p[:, None] + 0.7 * p[None, :]) + 0.4 * j))) for j in range(1, 5)] + [np.sum(D ** 2)])\ncfg_ref = dict(cfg)',
            "call": 'probe(hf_solution(cfg))',
            "gold_call": 'probe(_oracle_hf_solution(cfg_ref))',
        },
    ]
