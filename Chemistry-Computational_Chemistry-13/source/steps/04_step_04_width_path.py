"""
Evolve the width with the constant reference Hessian and retain the determinant phase.

The width obeys an autonomous linear Hamiltonian system. Its complex determinant must be followed continuously because its inverse square root sets the Gaussian phase.

Returns
-------
Q_path, P_path, logdetQ
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def width_path(mass, reference_hessian, q0, p0, times):
    r"""Solve dQ/dt=M^{-1}P and dP/dt=-K_ref Q exactly at the supplied times.
    K_ref is constant; it may be singular or indefinite. Q0 is real SPD and P0
    is complex. Both initial canonical identities must hold:
    Q0.T P0-P0.T Q0=0 and Q0.conj().T P0-P0.conj().T Q0=2i I.
    Return logdetQ=log(abs(det(Q)))+i*theta. Set theta[0]=0 and choose at each
    successive sample the 2*pi lift of arg(det(Q)) nearest the preceding theta.
    All supported grids resolve the true phase: each true increment has magnitude
    less than pi, with no ties. Do not reset to the principal determinant phase.

    Parameters
    ----------
    mass : real SPD (D,D) array
    reference_hessian : real symmetric (D,D) array
    q0 : real SPD (D,D) initial Q matrix
    p0 : complex (D,D) initial P matrix
    times : finite (N,) array, N>=1, starting at zero
        If N>1, entries are strictly increasing or strictly decreasing.

    Returns
    -------
    Q_path, P_path : complex arrays, shape (N,D,D)
    logdetQ : complex array, shape (N,)

    Raises
    ------
    ValueError
        For invalid or nonfinite shapes; mass not SPD; symmetry errors above
        1e-12 in mass, reference_hessian or q0; q0 imaginary part above 1e-12
        or q0 not SPD; initial canonical errors above 1e-9; or invalid times.
    """
    return Q_path, P_path, logdetQ

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_width_path(mass, reference_hessian, q0, p0, times):
    import numpy as np
    from scipy.linalg import expm
    mass = np.asarray(mass, dtype=float)
    reference_hessian = np.asarray(reference_hessian, dtype=float)
    q0 = np.asarray(q0, dtype=complex)
    p0 = np.asarray(p0, dtype=complex)
    times = np.asarray(times, dtype=float)
    if (mass.ndim != 2 or mass.shape[0] == 0 or mass.shape[0] != mass.shape[1]
            or any(a.shape != mass.shape for a in (reference_hessian, q0, p0))
            or times.ndim != 1 or times.size == 0 or times[0] != 0
            or (times.size > 1 and not (np.all(np.diff(times) > 0) or np.all(np.diff(times) < 0)))
            or not all(np.all(np.isfinite(a)) for a in (mass, reference_hessian, q0, p0, times))
            or not np.allclose(mass, mass.T, atol=1e-12, rtol=0)
            or not np.allclose(reference_hessian, reference_hessian.T, atol=1e-12, rtol=0)
            or np.min(np.linalg.eigvalsh(mass)) <= 0
            or not np.allclose(q0.imag, 0, atol=1e-12, rtol=0)
            or not np.allclose(q0, q0.T, atol=1e-12, rtol=0)
            or np.min(np.linalg.eigvalsh(q0.real)) <= 0):
        raise ValueError("Invalid width inputs")
    d = mass.shape[0]
    if (not np.allclose(q0.T @ p0 - p0.T @ q0, 0, atol=1e-9, rtol=0)
            or not np.allclose(q0.conj().T @ p0 - p0.conj().T @ q0,
                               2j * np.eye(d), atol=1e-9, rtol=0)):
        raise ValueError("Initial frame is not canonical")
    generator = np.block([[np.zeros((d, d)), np.linalg.inv(mass)],
                          [-reference_hessian, np.zeros((d, d))]])
    initial = np.vstack((q0, p0))
    frames = np.stack([expm(t * generator) @ initial for t in times])
    qs, ps = frames[:, :d], frames[:, d:]
    signs, logabs = np.linalg.slogdet(qs)
    phases = np.unwrap(np.angle(signs))
    phases -= phases[0]
    logs = logabs + 1j * phases
    return qs, ps, logs

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "name": 'case_1',
            "setup": """import numpy as np
M=np.array([[1.2,.1],[.1,.9]]);K=np.array([[1.7,.2],[.2,2.1]]);Q=np.array([[.9,.07],[.07,1.1]]);P=1j*np.linalg.inv(Q);times=np.linspace(0,13,131)
""",
            "call": 'width_path(M, K, Q, P, times)',
            "gold_call": '_oracle_width_path(M, K, Q, P, times)',
        },
        {
            "name": 'case_2',
            "setup": """import numpy as np
M=np.eye(2);K=np.eye(2);Q=np.eye(2);P=1j*Q;times=np.array([0.])
""",
            "call": 'width_path(M, K, Q, P, times)',
            "gold_call": '_oracle_width_path(M, K, Q, P, times)',
        },
        {
            "name": 'case_3',
            "setup": """import numpy as np
M=np.array([[1.,.2],[.2,1.5]]);K=np.zeros((2,2));Q=np.eye(2);P=1j*Q;times=np.linspace(0,1,11)
""",
            "call": 'width_path(M, K, Q, P, times)',
            "gold_call": '_oracle_width_path(M, K, Q, P, times)',
        },
        {
            "name": 'case_4',
            "setup": """import numpy as np
M=np.eye(2);K=np.diag([-.4,1.2]);Q=np.array([[1.,.1],[.1,.9]]);P=(np.array([[.2,.07],[.07,-.1]])+1j*np.linalg.inv(Q@Q))@Q;times=np.linspace(0,-1.5,31)
""",
            "call": 'width_path(M, K, Q, P, times)',
            "gold_call": '_oracle_width_path(M, K, Q, P, times)',
        },
        {
            "name": 'case_5',
            "setup": """import numpy as np
M=np.eye(1);K=M.copy();Q=M.copy();P=2j*M;times=np.array([0.,.1])

def _status(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    raise AssertionError("Expected ValueError")
""",
            "call": '_status(width_path, M, K, Q, P, times)',
            "gold_call": '_status(_oracle_width_path, M, K, Q, P, times)',
        },
    ]
