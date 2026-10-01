"""
Build the orbital Hessian blocks A and B of the Jordan-Wigner Hartree-Fock energy in the
canonical orbital basis of the Fock matrix F[rho].

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

rho is stationary: it is the projector onto the N_f lowest eigenvectors of F[rho]. The canonical
orbitals are those eigenvectors, the columns of the unitary C in ascending order of eigenvalue;
orbitals i < N_f are occupied and a >= N_f are virtual, N_v = N - N_f, n = N_v N_f.

For complex amplitudes z_ai (a virtual, i occupied) let kappa be the anti-Hermitian matrix with
kappa_ai = z_ai, kappa_ia = -conj(z_ai) and all other entries zero in the canonical basis, and
rotate the determinant to rho(z) = e^K rho e^-K with K = C kappa C^dag in the site basis.
Collect the amplitudes in vec z, with z_ai at position (a - N_f) N_f + i, and set u = (vec z,
conj(vec z)). The orbital Hessian H = [[A, B], [B*, A*]], with A Hermitian and B complex
symmetric (both n x n), is defined by E[rho(z)] = E[rho] + (1/2) u^dag H u + O(|z|^3).

Canonical orbitals are fixed only up to a phase per orbital and a unitary mixing inside
degenerate levels. A and B change with that choice, but the spectrum of H, the spectrum of A, Tr
A and the Frobenius norm of B do not. The comparison needs about 1e-10 relative accuracy.

Returns
-------
AB : np.ndarray -- Complex array of shape (2, n, n), n = N_v N_f: AB[0] = A, AB[1] = B.
"""

import numpy as np
from scipy.linalg import expm

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def orbital_hessian(rho: "np.ndarray", cfg: dict) -> "np.ndarray":
    '''Build the orbital Hessian blocks A and B of the Jordan-Wigner Hartree-Fock energy in the
    canonical orbital basis of the Fock matrix F[rho].

    Parameters
    ----------
    rho : np.ndarray
        Density matrix of shape (N, N) of a stationary Slater determinant, rho_kl = <c^dag_l c_k>.
    cfg : dict
        Cluster parameters. Keys read:
            nx : number of columns, x = 0, ..., nx-1
            ny : number of rows, y = 0, ..., ny-1 (ny = 1 is a chain)
            J1 : nearest-neighbour coupling
            J2 : coupling on both plaquette diagonals
            Delta : anisotropy of the s^z s^z coupling

    Returns
    -------
    AB : np.ndarray
        Complex array of shape (2, n, n), n = N_v N_f: AB[0] = A, AB[1] = B.
    '''
    return AB

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_orbital_hessian(rho: "np.ndarray", cfg: dict) -> "np.ndarray":
    rho = np.asarray(rho, dtype=complex)
    N = len(rho)
    Nf = N // 2
    F = _oracle_fock_matrix(rho, cfg)
    eps, C = np.linalg.eigh(0.5 * (F + F.conj().T))
    K = _oracle_fock_kernel(rho, cfg)
    o, v = C[:, :Nf], C[:, Nf:]
    A = np.einsum("mnls,ma,ni,lb,sj->aibj", K, v.conj(), o, v, o.conj(), optimize=True)
    B = np.einsum("mnls,ma,ni,lj,sb->aibj", K, v.conj(), o, o, v.conj(), optimize=True)
    n = (N - Nf) * Nf
    A = A.reshape(n, n) + np.diag((eps[Nf:, None] - eps[None, :Nf]).ravel())
    return np.array([A, B.reshape(n, n)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # normal: stationary stable determinant of the 5x2 cluster, J2 = 0.6
        {
            "setup": 'import numpy as np\nN = 10\nu = [0.6370834394947129, -0.2786315566083346, -0.0399051877836496, 0.0399933969761149, 0.009811365220657599, -0.01993260587554402, 0.004580204304931379, 0.05464534406803473, -0.0248646059226992, -0.3824418626047825, 0.3384383963845566, -0.2230321541893325, 0.02998823522953448, 0.03999339697530768, -0.004580204304969405, -0.02884980563041355, -0.003630991847943778, 0.3041964394222973, 0.02486460592269858, 0.6890138887251021, -0.2230321541932449, -0.03990518778373257, 0.05464534406890464, 0.003630991847901598, -0.3249671103022208, 0.003630991847943896, 0.0546453440680349, 0.338438396384106, -0.2786315566052393, 0.02486460592306677, 0.3041964394218896, -0.003630991847902217, -0.02884980563041353, -0.004580204304931404, 0.6370834394943725, -0.3824418626070856, -0.02486460592306771, 0.05464534406890453, 0.004580204304969387, -0.01993260587554398, 0.3629165605056262, -0.2786315566052394, 0.03990518778373243, 0.03999339697530758, -0.009811365220657758, 0.6615616036158941, -0.2230321541932447, -0.02998823522953465, 0.03999339697611519, 0.3109861112748982, -0.2230321541893328, 0.03990518778364985, 0.6615616036154436, -0.2786315566083349, 0.3629165605052872]\nrho = np.zeros((N, N))\nrho[np.triu_indices(N)] = u\nrho = rho + np.triu(rho, 1).T\ncfg = dict(nx=5, ny=2, J1=1.0, J2=0.6, Delta=1.0)\ndef invariants(H):\n    A, B = H\n    M = np.block([[A, B], [B.conj(), A.conj()]])\n    return np.concatenate([np.linalg.eigvalsh(M), np.linalg.eigvalsh(A), [np.trace(A).real, np.linalg.norm(B)]])\ncfg_ref = dict(cfg)\nrho_ref = rho.copy()',
            "call": 'invariants(orbital_hessian(rho, cfg))',
            "gold_call": 'invariants(_oracle_orbital_hessian(rho_ref, cfg_ref))',
        },
        # normal: stationary stable determinant of the 6x2 cluster, J2 = 0.4
        {
            "setup": 'import numpy as np\nN = 12\nu = [0.6614434790500133, -0.2854790653034515, -0.0369429317442741, 0.04245308169922492, 0.006854485660899579, -0.01093951323248264, -4.165071065820314e-15, 0.01290135475465922, -0.005477293104719144, -0.05590483369850661, 0.02550120634782027, 0.3676029072522697, 0.3122163945281266, -0.2194650723736831, 0.03275923277679266, 0.02790826715173123, -0.006854485660881087, -0.01290135475470788, -4.643768009016114e-14, 0.03921458153619064, 0.008019002614904711, -0.2841764167353731, -0.02550120634781966, 0.7198927231950373, -0.2508650431660847, -0.03275923277691854, 0.0424530816993544, 0.00547729310474801, -0.03921458153611904, 5.914019274300131e-14, 0.2855719729749278, -0.008019002614904711, -0.05590483369850661, 0.2801072768049782, -0.219465072373854, 0.03694293174433366, 0.05590483369859747, -0.008019002614789246, -0.2855719729748107, -5.908641631524603e-14, 0.03921458153619085, 0.005477293104719339, 0.6877836054722355, -0.2854790653036768, -0.02550120634785551, 0.2841764167347724, 0.008019002614789186, -0.03921458153611912, 4.624469210345872e-14, 0.01290135475465888, 0.3385565209494686, -0.3676029072518279, 0.02550120634785542, 0.05590483369859731, -0.005477293104748799, -0.0129013547547075, 4.139050213680662e-15, 0.6614434790505302, -0.2854790653036776, -0.03694293174433398, 0.04245308169935454, 0.006854485660881269, -0.01093951323248284, 0.3122163945277651, -0.2194650723738542, 0.03275923277691854, 0.02790826715173183, -0.006854485660899577, 0.7198927231950213, -0.2508650431660844, -0.03275923277679227, 0.04245308169922481, 0.2801072768049626, -0.2194650723736832, 0.03694293174427444, 0.6877836054718727, -0.2854790653034518, 0.3385565209499871]\nrho = np.zeros((N, N))\nrho[np.triu_indices(N)] = u\nrho = rho + np.triu(rho, 1).T\ncfg = dict(nx=6, ny=2, J1=1.0, J2=0.4, Delta=1.0)\ndef invariants(H):\n    A, B = H\n    M = np.block([[A, B], [B.conj(), A.conj()]])\n    return np.concatenate([np.linalg.eigvalsh(M), np.linalg.eigvalsh(A), [np.trace(A).real, np.linalg.norm(B)]])\ncfg_ref = dict(cfg)\nrho_ref = rho.copy()',
            "call": 'invariants(orbital_hessian(rho, cfg))',
            "gold_call": 'invariants(_oracle_orbital_hessian(rho_ref, cfg_ref))',
        },
        # normal: stationary stable determinant of the 4x3 cluster, J2 = 0.4
        {
            "setup": 'import numpy as np\nN = 12\nu = [0.7676241837694604, -0.3715835946117982, -0.05931285092968444, 0.07780293125555424, 0.01253274137530076, -0.03131367413612595, -0.01062610448280676, 0.1529102785540548, 0.03365980428579168, -0.05637073289056369, -0.009776282030037359, 0.04114993633034332, 0.2722557004516715, -0.1867419757453862, 0.0442914785840441, 0.02837669375630039, -0.008306244266074576, -0.1471449998750976, -0.01776141489514994, -0.008769149360022613, -0.01130588081121271, -0.009067511056037349, 0.009776282030037275, 0.7426416145806496, -0.3275097437356681, -0.03811256181586585, 0.1996663337533578, 0.01221090323598654, -0.02494398983839274, -0.006683403902356624, 0.005582427461579241, 0.01130588081121302, -0.05637073289056379, 0.2633067380991015, -0.2771123674600073, 0.01113493793310733, 0.02269302306573881, 0.001651394632611901, 0.003337155503540336, 0.006683403902356826, -0.008769149360022786, -0.03365980428579168, 0.7558517885939147, -0.2784383594652014, -0.03974210717135609, 0.04363964214997533, -0.001651394632612286, -0.02494398983839268, 0.01776141489515013, 0.1529102785540535, 0.2345595924416148, -0.1922634100231521, 0.03974210717135589, 0.0226930230657384, -0.01221090323598621, -0.1471449998750968, 0.01062610448280673, 0.7654404075583844, -0.2784383594652018, -0.01113493793310742, 0.1996663337533572, 0.008306244266074736, -0.03131367413612531, 0.2441482114060837, -0.2771123674600061, 0.03811256181586564, 0.02837669375630028, -0.01253274137530089, 0.7366932619008978, -0.3275097437356686, -0.04429147858404413, 0.07780293125555444, 0.2573583854193495, -0.186741975745386, 0.05931285092968457, 0.7277442995483274, -0.3715835946117986, 0.2323758162305392]\nrho = np.zeros((N, N))\nrho[np.triu_indices(N)] = u\nrho = rho + np.triu(rho, 1).T\ncfg = dict(nx=4, ny=3, J1=1.0, J2=0.4, Delta=1.0)\ndef invariants(H):\n    A, B = H\n    M = np.block([[A, B], [B.conj(), A.conj()]])\n    return np.concatenate([np.linalg.eigvalsh(M), np.linalg.eigvalsh(A), [np.trace(A).real, np.linalg.norm(B)]])\ncfg_ref = dict(cfg)\nrho_ref = rho.copy()',
            "call": 'invariants(orbital_hessian(rho, cfg))',
            "gold_call": 'invariants(_oracle_orbital_hessian(rho_ref, cfg_ref))',
        },
        # boundary: stationary stable determinant of the six-site Heisenberg chain, where no string survives
        {
            "setup": 'import numpy as np\nN = 6\nu = [0.4999999999999877, -0.4678140057688586, 4.440892098500626e-15, 0.1556045802024087, -8.049116928532385e-16, -0.08329027930393063, 0.5000000000000102, -0.1746875543456864, -2.525757381022231e-15, 0.02518559832955128, 8.743006318923108e-16, 0.4999999999999856, -0.4418953190256205, 2.3037127760972e-15, 0.1556045802024088, 0.5000000000000142, -0.1746875543456861, -4.468647674116255e-15, 0.4999999999999897, -0.4678140057688582, 0.5000000000000127]\nrho = np.zeros((N, N))\nrho[np.triu_indices(N)] = u\nrho = rho + np.triu(rho, 1).T\ncfg = dict(nx=6, ny=1, J1=1.0, J2=0.0, Delta=1.0)\ndef invariants(H):\n    A, B = H\n    M = np.block([[A, B], [B.conj(), A.conj()]])\n    return np.concatenate([np.linalg.eigvalsh(M), np.linalg.eigvalsh(A), [np.trace(A).real, np.linalg.norm(B)]])\ncfg_ref = dict(cfg)\nrho_ref = rho.copy()',
            "call": 'invariants(orbital_hessian(rho, cfg))',
            "gold_call": 'invariants(_oracle_orbital_hessian(rho_ref, cfg_ref))',
        },
    ]
