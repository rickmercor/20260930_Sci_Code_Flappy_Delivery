"""
Construct a fixed gauge for the ground-state Gaussian with a full mass matrix.

A full mass matrix generally does not commute with the force matrix. The ground-state Gaussian precision is the positive solution of A M^{-1} A = K_g.

Returns
-------
Q0, P0, E0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def initial_frame(mass, ground_hessian, hbar):
    r"""Use the ground vibrational state of H_g=p.T M^{-1} p/2 + y.T K_g y/2.
    Let A be the symmetric positive-definite solution of A M^{-1} A=K_g.
    Fix the frame gauge Q0=A^{-1/2}, P0=i A^{1/2}; use principal SPD matrix powers.
    The initial Gaussian is (pi*hbar)^(-D/4)*det(Q0)^(-1/2)
    times exp(-y.T A y/(2*hbar)). Return its zero-point energy E0.

    Parameters
    ----------
    mass, ground_hessian : real symmetric positive-definite arrays, shape (D,D)
    hbar : finite positive float

    Returns
    -------
    Q0 : real SPD array, shape (D,D)
    P0 : complex array, shape (D,D)
    E0 : float
        hbar/2 times the sum of the positive generalized normal-mode frequencies.

    Raises
    ------
    ValueError
        For non-square, empty, unequal or nonfinite matrices; lack of symmetry
        within absolute tolerance 1e-12; non-positive-definite matrices; or hbar<=0
        or nonfinite hbar.
    """
    return Q0, P0, E0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_initial_frame(mass, ground_hessian, hbar):
    import numpy as np

    mass = np.asarray(mass, dtype=float)
    ground_hessian = np.asarray(ground_hessian, dtype=float)
    if (mass.ndim != 2 or mass.shape[0] == 0 or mass.shape[0] != mass.shape[1]
            or ground_hessian.shape != mass.shape
            or not all(np.all(np.isfinite(a)) for a in (mass, ground_hessian))
            or not np.allclose(mass, mass.T, atol=1e-12, rtol=0)
            or not np.allclose(ground_hessian, ground_hessian.T, atol=1e-12, rtol=0)
            or not np.isfinite(hbar) or hbar <= 0):
        raise ValueError("Invalid mass, Hessian, or hbar")

    values, vectors = np.linalg.eigh(mass)
    if np.min(values) <= 0:
        raise ValueError("Mass must be positive definite")
    root_mass = (vectors * values ** 0.5) @ vectors.T
    inverse_root_mass = (vectors * values ** -0.5) @ vectors.T

    dynamical = inverse_root_mass @ ground_hessian @ inverse_root_mass
    values, vectors = np.linalg.eigh(dynamical)
    if np.min(values) <= 0:
        raise ValueError("Hessian must be positive definite")
    frequencies = (vectors * values ** 0.5) @ vectors.T

    precision = root_mass @ frequencies @ root_mass
    values, vectors = np.linalg.eigh(precision)
    if np.min(values) <= 0:
        raise ValueError("Precision must be positive definite")
    Q0 = (vectors * values ** -0.5) @ vectors.T
    P0 = 1j * ((vectors * values ** 0.5) @ vectors.T)
    E0 = 0.5 * hbar * np.trace(frequencies)
    return Q0, P0, float(E0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "name": 'case_1',
            "setup": """import numpy as np
data={'mass': [[1.2, 0.12, -0.08], [0.12, 1.5, 0.1], [-0.08, 0.1, 0.9]], 'ground_hessian': [[1.4, 0.23, -0.14], [0.23, 2.1, 0.17], [-0.14, 0.17, 0.95]], 'hbar': 0.7}
""",
            "call": "initial_frame(data['mass'], data['ground_hessian'], data['hbar'])",
            "gold_call": "_oracle_initial_frame(data['mass'], data['ground_hessian'], data['hbar'])",
        },
        {
            "name": 'case_2',
            "setup": """import numpy as np
M=np.array([[4.]]);K=np.array([[9.]])
""",
            "call": 'initial_frame(M, K, 0.5)',
            "gold_call": '_oracle_initial_frame(M, K, 0.5)',
        },
        {
            "name": 'case_3',
            "setup": """import numpy as np
M=np.array([[1.,.2],[.2,2.]]);K=2.25*M
""",
            "call": 'initial_frame(M, K, 1.0)',
            "gold_call": '_oracle_initial_frame(M, K, 1.0)',
        },
        {
            "name": 'case_4',
            "setup": """import numpy as np
M=np.diag([1.,2.]);K=np.array([[1e-6,0.],[0.,3.]])
""",
            "call": 'initial_frame(M, K, 0.8)',
            "gold_call": '_oracle_initial_frame(M, K, 0.8)',
        },
        {
            "name": 'case_5',
            "setup": """import numpy as np
M=np.eye(2);K=np.diag([1.,0.])

def _status(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    raise AssertionError("Expected ValueError")
""",
            "call": '_status(initial_frame, M, K, 1.0)',
            "gold_call": '_status(_oracle_initial_frame, M, K, 1.0)',
        },
    ]
