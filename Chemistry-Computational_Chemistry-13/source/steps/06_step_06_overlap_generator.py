"""
Construct the Gaussian generating function for overlaps of two Hagedorn bases.

Different Gaussian centres define nonorthogonal Hagedorn bases. The overlap is a complex Gaussian integral, including the phases of both normalization determinants.

Returns
-------
B, b, G
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def overlap_generator(q_bra, p_bra, Q_bra, P_bra, S_bra, logdet_bra,
                              q_ket, p_ket, Q_ket, P_ket, S_ket, logdet_ket, hbar):
    r"""Return B, b, G such that
    F(u,v)=sum_{k,l} <phi_bra,k|phi_ket,l>*u**k*v**l/sqrt(k!*l!)
          =G*exp(z.T B z/2+b.T z), z=(u,v), with bra variables first.
    Variables u,v are formal and are not conjugated. B is complex symmetric.
    This definition fully fixes the signs, ordering and factorial convention.

    For either frame define y=x-q, A=P@inv(Q), and
    phi_0(x)=(pi*hbar)^(-D/4)*exp(-logdetQ/2)
             *exp(i*(y.T A y/2+p.T y+S)/hbar).
    Its basis generating function is
    sum_k phi_k(x)*v**k/sqrt(k!)
     =phi_0(x)*exp(sqrt(2/hbar)*v.T inv(Q)y-v.T inv(Q)conj(Q)v/2).
    Conjugate the bra coefficients before carrying out the integral over real x.
    In the Gaussian integral use the analytic determinant square root for complex
    symmetric precision W with positive-definite real part: its log determinant
    is the sum of principal logarithms of W's eigenvalues, not necessarily the
    principal scalar logarithm of det(W). The supplied lifted logdetQ values
    are part of the state and may differ by multiples of 2*pi*i for identical Q.

    Parameters
    ----------
    q_bra, p_bra, q_ket, p_ket : finite real arrays, shape (D,), D>=1
    Q_bra, P_bra, Q_ket, P_ket : finite complex arrays, shape (D,D)
        Each frame obeys Q.T P-P.T Q=0 and Q.conj().T P-P.conj().T Q=2i I.
    S_bra, S_ket : finite real scalars
    logdet_bra, logdet_ket : finite complex scalars, exp(logdet)=det(Q)
    hbar : finite positive float

    Returns
    -------
    B : complex symmetric array, shape (2D,2D)
    b : complex array, shape (2D,)
    G : complex scalar, the vacuum overlap

    Raises
    ------
    ValueError
        For invalid or nonfinite shapes/data, nonpositive hbar, canonical defects
        above 1e-8, or determinant mismatch exceeding 1e-8*max(1,abs(det(Q))).
    """
    return B, b, G

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_overlap_generator(q_bra, p_bra, Q_bra, P_bra, S_bra, logdet_bra,
                              q_ket, p_ket, Q_ket, P_ket, S_ket, logdet_ket, hbar):
    import numpy as np
    q_bra, p_bra, q_ket, p_ket = [np.asarray(a, dtype=float) for a in (q_bra, p_bra, q_ket, p_ket)]
    Q_bra, P_bra, Q_ket, P_ket = [np.asarray(a, dtype=complex) for a in (Q_bra, P_bra, Q_ket, P_ket)]
    d = q_bra.size
    if (q_bra.shape != (d,) or d == 0
            or any(a.shape != (d,) for a in (p_bra, q_ket, p_ket))
            or any(a.shape != (d, d) for a in (Q_bra, P_bra, Q_ket, P_ket))
            or not all(np.all(np.isfinite(a)) for a in
                       (q_bra, p_bra, q_ket, p_ket, Q_bra, P_bra, Q_ket, P_ket,
                        S_bra, S_ket, logdet_bra, logdet_ket, hbar))
            or hbar <= 0):
        raise ValueError("Invalid Gaussian inputs")
    for Q, P, logdet in ((Q_bra, P_bra, logdet_bra), (Q_ket, P_ket, logdet_ket)):
        if (not np.allclose(Q.T @ P - P.T @ Q, 0, atol=1e-8, rtol=0)
                or not np.allclose(Q.conj().T @ P - P.conj().T @ Q,
                                   2j * np.eye(d), atol=1e-8, rtol=0)
                or abs(np.exp(logdet) - np.linalg.det(Q)) > 1e-8 * max(1.0, abs(np.linalg.det(Q)))):
            raise ValueError("Noncanonical frame or inconsistent determinant lift")
    inverse_bra = np.linalg.inv(Q_bra).conj()
    inverse_ket = np.linalg.inv(Q_ket)
    A_bra = (P_bra @ np.linalg.inv(Q_bra)).conj()
    A_ket = P_ket @ inverse_ket
    precision = -1j * (A_ket - A_bra) / hbar
    linear = 1j * (-A_ket @ q_ket + A_bra @ q_bra + p_ket - p_bra) / hbar
    constant = 1j * (0.5 * q_ket @ A_ket @ q_ket - 0.5 * q_bra @ A_bra @ q_bra
                     - p_ket @ q_ket + p_bra @ q_bra + S_ket - S_bra) / hbar
    rows = np.sqrt(2.0 / hbar) * np.vstack((inverse_bra, inverse_ket))
    shift = -np.concatenate((rows[:d] @ q_bra, rows[d:] @ q_ket))
    B = rows @ np.linalg.solve(precision, rows.T)
    B[:d, :d] -= inverse_bra @ Q_bra
    B[d:, d:] -= inverse_ket @ Q_ket.conj()
    b = shift + rows @ np.linalg.solve(precision, linear)
    # Each eigenvalue of this accretive matrix has positive real part.
    # Summing their principal logarithms keeps the matrix square-root branch.
    logdet_precision = np.sum(np.log(np.linalg.eigvals(precision)))
    log_prefactor = (0.5 * d * np.log(2.0 / hbar)
                     - 0.5 * (np.conj(logdet_bra) + logdet_ket + logdet_precision)
                     + constant + 0.5 * linear @ np.linalg.solve(precision, linear))
    overlap = complex(np.exp(log_prefactor))
    return 0.5 * (B + B.T), b, overlap

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "name": 'case_1',
            "setup": """import numpy as np
from scipy.linalg import expm
Q=np.array([[0.9,0.12],[0.12,1.15]])
P=1j*np.linalg.inv(Q)
M=np.array([[1.2,0.13],[0.13,0.85]])
K=np.array([[1.7,-0.21],[-0.21,0.8]])
qb=np.array([-0.2,0.1]);pb=np.array([0.04,-0.06])
sb=0.0;lb=np.log(np.linalg.det(Q))+0j
W=np.block([[np.zeros((2,2)),np.linalg.inv(M)],[-K,np.zeros((2,2))]])
frame=expm(0.35*W)@np.vstack((Q,P))
Qt=frame[:2];Pt=frame[2:]
qk=np.array([0.12,-0.3]);pk=np.array([-0.2,0.07]);sk=-0.17
lk=np.log(np.linalg.det(Qt))
""",
            "call": 'overlap_generator(qb, pb, Q, P, sb, lb, qk, pk, Qt, Pt, sk, lk, 0.7)',
            "gold_call": '_oracle_overlap_generator(qb, pb, Q, P, sb, lb, qk, pk, Qt, Pt, sk, lk, 0.7)',
        },
        {
            "name": 'case_2',
            "setup": """import numpy as np
from scipy.linalg import expm
Q=np.array([[0.9,0.12],[0.12,1.15]])
P=1j*np.linalg.inv(Q)
M=np.array([[1.2,0.13],[0.13,0.85]])
K=np.array([[1.7,-0.21],[-0.21,0.8]])
qb=np.array([-0.2,0.1]);pb=np.array([0.04,-0.06])
sb=0.0;lb=np.log(np.linalg.det(Q))+0j
W=np.block([[np.zeros((2,2)),np.linalg.inv(M)],[-K,np.zeros((2,2))]])
frame=expm(0.35*W)@np.vstack((Q,P))
Qt=frame[:2];Pt=frame[2:]
qk=np.array([0.12,-0.3]);pk=np.array([-0.2,0.07]);sk=-0.17
lk=np.log(np.linalg.det(Qt))

qk=qb.copy();pk=pb.copy();Qt=Q.copy();Pt=P.copy();sk=sb;lk=lb
""",
            "call": 'overlap_generator(qb, pb, Q, P, sb, lb, qk, pk, Qt, Pt, sk, lk, 0.7)',
            "gold_call": '_oracle_overlap_generator(qb, pb, Q, P, sb, lb, qk, pk, Qt, Pt, sk, lk, 0.7)',
        },
        {
            "name": 'case_3',
            "setup": """import numpy as np
from scipy.linalg import expm
Q=np.array([[0.9,0.12],[0.12,1.15]])
P=1j*np.linalg.inv(Q)
M=np.array([[1.2,0.13],[0.13,0.85]])
K=np.array([[1.7,-0.21],[-0.21,0.8]])
qb=np.array([-0.2,0.1]);pb=np.array([0.04,-0.06])
sb=0.0;lb=np.log(np.linalg.det(Q))+0j
W=np.block([[np.zeros((2,2)),np.linalg.inv(M)],[-K,np.zeros((2,2))]])
frame=expm(0.35*W)@np.vstack((Q,P))
Qt=frame[:2];Pt=frame[2:]
qk=np.array([0.12,-0.3]);pk=np.array([-0.2,0.07]);sk=-0.17
lk=np.log(np.linalg.det(Qt))

qk=qb.copy();pk=pb.copy();Qt=Q.copy();Pt=P.copy();sk=sb;lk=lb+2j*np.pi
""",
            "call": 'overlap_generator(qb, pb, Q, P, sb, lb, qk, pk, Qt, Pt, sk, lk, 0.7)',
            "gold_call": '_oracle_overlap_generator(qb, pb, Q, P, sb, lb, qk, pk, Qt, Pt, sk, lk, 0.7)',
        },
        {
            "name": 'case_4',
            "setup": """import numpy as np
from scipy.linalg import expm
Q=np.array([[0.9,0.12],[0.12,1.15]])
P=1j*np.linalg.inv(Q)
M=np.array([[1.2,0.13],[0.13,0.85]])
K=np.array([[1.7,-0.21],[-0.21,0.8]])
qb=np.array([-0.2,0.1]);pb=np.array([0.04,-0.06])
sb=0.0;lb=np.log(np.linalg.det(Q))+0j
W=np.block([[np.zeros((2,2)),np.linalg.inv(M)],[-K,np.zeros((2,2))]])
frame=expm(0.35*W)@np.vstack((Q,P))
Qt=frame[:2];Pt=frame[2:]
qk=np.array([0.12,-0.3]);pk=np.array([-0.2,0.07]);sk=-0.17
lk=np.log(np.linalg.det(Qt))
import numpy as np
qb=np.zeros(3);pb=qb.copy();qk=qb.copy();pk=qb.copy();Q=np.eye(3,dtype=complex);P=1j*Q;Qt=Q.copy();Pt=(4.+1j)*Q;sb=0.;sk=.3;lb=0j;lk=0j
""",
            "call": 'overlap_generator(qb, pb, Q, P, sb, lb, qk, pk, Qt, Pt, sk, lk, 0.7)',
            "gold_call": '_oracle_overlap_generator(qb, pb, Q, P, sb, lb, qk, pk, Qt, Pt, sk, lk, 0.7)',
        },
        {
            "name": 'case_5',
            "setup": """import numpy as np
from scipy.linalg import expm
Q=np.array([[0.9,0.12],[0.12,1.15]])
P=1j*np.linalg.inv(Q)
M=np.array([[1.2,0.13],[0.13,0.85]])
K=np.array([[1.7,-0.21],[-0.21,0.8]])
qb=np.array([-0.2,0.1]);pb=np.array([0.04,-0.06])
sb=0.0;lb=np.log(np.linalg.det(Q))+0j
W=np.block([[np.zeros((2,2)),np.linalg.inv(M)],[-K,np.zeros((2,2))]])
frame=expm(0.35*W)@np.vstack((Q,P))
Qt=frame[:2];Pt=frame[2:]
qk=np.array([0.12,-0.3]);pk=np.array([-0.2,0.07]);sk=-0.17
lk=np.log(np.linalg.det(Qt))

lk+=.2

def _status(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    raise AssertionError("Expected ValueError")
""",
            "call": '_status(overlap_generator, qb, pb, Q, P, sb, lb, qk, pk, Qt, Pt, sk, lk, 0.7)',
            "gold_call": '_status(_oracle_overlap_generator, qb, pb, Q, P, sb, lb, qk, pk, Qt, Pt, sk, lk, 0.7)',
        },
    ]
