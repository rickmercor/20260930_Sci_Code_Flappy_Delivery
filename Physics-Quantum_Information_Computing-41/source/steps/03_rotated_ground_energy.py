"""
Recover the lowest physical energy of a rotated, thresholded Hermitian pencil.

Projective transformations preserve the untruncated generalized eigenvectors, whereas thresholding changes the retained space. The physical ground energy must be selected after undoing the pencil rotation, especially when roots lie on different sides of the inverse-map pole.

Returns
-------
float, the lowest finite physical energy obtained by inverse-rotating the generalized eigenvalues in the retained subspace, returned as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh

def rotated_ground_energy(hamiltonian: 'np.ndarray', overlap: 'np.ndarray', theta: float, cutoff: float) -> float:
    """Recover the lowest physical energy of a rotated, thresholded Hermitian pencil.
    
    Parameters
    ----------
    hamiltonian, overlap : np.ndarray, shape (n, n)
        Matching nonempty finite Hermitian matrices, absolute Hermiticity tolerance 1e-10.
    theta : float
        Finite rotation angle in radians.
    cutoff : float
        Finite nonnegative absolute eigenvalue threshold.
    
    Returns
    -------
    result : float
        Smallest finite back-transformed generalized eigenvalue in the retained space.
    
    Raises
    ------
    ValueError
        If shapes, finite values, Hermiticity or scalar domains fail, if no denominator eigenvalue exceeds the cutoff, or if no finite physical root remains.
    
    Notes
    -----
    Write c=cos(theta), s=sin(theta), A=c*hamiltonian-s*overlap and B=c*overlap+s*hamiltonian. Retain the eigenvectors of B with eigenvalues strictly greater than cutoff; compress both A and B to that subspace, solve the Hermitian generalized eigenproblem there and undo the rotation of each eigenray. Exclude roots whose inverse-map denominator has absolute value <=1e-12. Select the minimum recovered physical energy, rather than relying on the ordering of rotated roots. The input overlap need not be positive definite; the retained denominator is positive definite by construction. Do not regularize discarded modes or normalize the projected Hamiltonian separately.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_rotated_ground_energy(hamiltonian: 'np.ndarray', overlap: 'np.ndarray', theta: float, cutoff: float) -> float:
    import numpy as np
    from numbers import Real
    from scipy.linalg import eigh
    try:
        h = np.asarray(hamiltonian, dtype=complex)
        s = np.asarray(overlap, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError('Numerical matrices are required.') from exc
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] < 1 or s.shape != h.shape:
        raise ValueError('The pencil matrices must be nonempty and equally sized square matrices.')
    for a in (h, s):
        if not np.all(np.isfinite(a)) or not np.allclose(a, a.conj().T, atol=1e-10, rtol=0):
            raise ValueError('The pencil matrices must be finite and Hermitian.')
    for a in (theta, cutoff):
        if isinstance(a, bool) or not isinstance(a, Real) or not np.isfinite(a):
            raise ValueError('Angle and cutoff must be finite real scalars.')
    if cutoff < 0:
        raise ValueError('The cutoff must be nonnegative.')
    c, t = np.cos(theta), np.sin(theta)
    a = c * h - t * s
    b = c * s + t * h
    beta, u = eigh(b)
    u = u[:, beta > cutoff]
    if u.shape[1] == 0:
        raise ValueError('The retained overlap subspace is empty.')
    ac, bc = u.conj().T @ a @ u, u.conj().T @ b @ u
    lam = eigh((ac + ac.conj().T) / 2, (bc + bc.conj().T) / 2, eigvals_only=True)
    denominator = c - t * lam
    good = np.abs(denominator) > 1e-12
    if not np.any(good):
        raise ValueError('Every retained root lies at the excluded inverse-rotation pole.')
    physical = (c * lam[good] + t) / denominator[good]
    if not np.all(np.isfinite(physical)):
        raise ValueError('The recovered energies are nonfinite.')
    return float(np.min(physical))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic cases with oracle-independent input setup."""
    exception_setup = """def _exception_code(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""

    setup_1 = """from copy import deepcopy
import numpy as np
h = np.array([[0.6, 0.12j], [-0.12j, 1.3]])
s = np.array([[1.0, 0.2], [0.2, 0.8]])
"""

    setup_2 = """from copy import deepcopy
import numpy as np
h = np.diag([-2.0, 3.0])
s = np.eye(2)
"""

    setup_3 = """from copy import deepcopy
import numpy as np
h = np.diag([-10.0, 1.0])
s = np.diag([0.5, 1.0])
"""

    setup_4 = """from copy import deepcopy
import numpy as np
h = np.diag([1.0, 1.0])
s = np.diag([-1.0, 1.0])
"""

    setup_5 = """from copy import deepcopy
import numpy as np
h = np.array([[0.9]])
s = np.array([[0.3]])
"""

    setup_6 = """from copy import deepcopy
import numpy as np
h = np.eye(2)
s = np.eye(2)
"""

    setup_7 = """from copy import deepcopy
import numpy as np
h = np.eye(2)
s = np.zeros((2, 2))
"""

    setup_8 = """from copy import deepcopy
import numpy as np
h = np.array([[1.0, 1.0], [0.0, 1.0]])
s = np.eye(2)
"""

    return [
        # Case 1
        {
            'setup': setup_1,
            'call': 'rotated_ground_energy(*deepcopy((h, s, 0.6, 0.2)))',
            'gold_call': '_oracle_rotated_ground_energy(*deepcopy((h, s, 0.6, 0.2)))',
        },
        # Case 2
        {
            'setup': setup_2,
            'call': 'rotated_ground_energy(*deepcopy((h, s, 0.0, 0.0)))',
            'gold_call': '_oracle_rotated_ground_energy(*deepcopy((h, s, 0.0, 0.0)))',
        },
        # Case 3
        {
            'setup': setup_3,
            'call': 'rotated_ground_energy(*deepcopy((h, s, 0.0, 0.5)))',
            'gold_call': '_oracle_rotated_ground_energy(*deepcopy((h, s, 0.0, 0.5)))',
        },
        # Case 4
        {
            'setup': setup_4,
            'call': 'rotated_ground_energy(*deepcopy((h, s, 1.2, 0.1)))',
            'gold_call': '_oracle_rotated_ground_energy(*deepcopy((h, s, 1.2, 0.1)))',
        },
        # Case 5
        {
            'setup': setup_5,
            'call': 'rotated_ground_energy(*deepcopy((h, s, 0.2, 0.01)))',
            'gold_call': '_oracle_rotated_ground_energy(*deepcopy((h, s, 0.2, 0.01)))',
        },
        # Case 6
        {
            'setup': setup_6 + exception_setup,
            'call': '_exception_code(rotated_ground_energy, *deepcopy((h, s, 0.0, 1.0)))',
            'gold_call': '_exception_code(_oracle_rotated_ground_energy, *deepcopy((h, s, 0.0, 1.0)))',
        },
        # Case 7
        {
            'setup': setup_7 + exception_setup,
            'call': '_exception_code(rotated_ground_energy, *deepcopy((h, s, np.pi / 2, 0.0)))',
            'gold_call': '_exception_code(_oracle_rotated_ground_energy, *deepcopy((h, s, np.pi / 2, 0.0)))',
        },
        # Case 8
        {
            'setup': setup_8 + exception_setup,
            'call': '_exception_code(rotated_ground_energy, *deepcopy((h, s, 0.0, 0.1)))',
            'gold_call': '_exception_code(_oracle_rotated_ground_energy, *deepcopy((h, s, 0.0, 0.1)))',
        },
        # Case 9
        {
            'setup': setup_6 + exception_setup,
            'call': '_exception_code(rotated_ground_energy, *deepcopy((h, s, 0.0, -0.1)))',
            'gold_call': '_exception_code(_oracle_rotated_ground_energy, *deepcopy((h, s, 0.0, -0.1)))',
        },
    ]
