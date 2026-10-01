"""
Differentiate the retained spectral projector through second order.

The regularized subspace moves when the rotated denominator changes. Differentiate both $P^2=P$ and $[B,P]=0$ at fixed cutoff; only gaps between retained and discarded clusters are relevant. Degeneracy within either cluster must not require eigenvector phase conventions or division by a zero internal gap.

Returns
-------
np.ndarray, complex, shape (3, n, n), containing P(0), P'(0), and P''(0) in that order, where P is the Hermitian orthogonal projector onto denominator eigenvalues strictly above the fixed cutoff. All matrices are in the original coordinates, and derivatives are not divided by factorials.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh

def cutoff_projector_jet(denominator_jet: 'np.ndarray', cutoff: float, gap_tolerance: float = 1e-10) -> 'np.ndarray':
    """Differentiate the retained spectral projector through second order.
    
    Parameters
    ----------
    denominator_jet : np.ndarray, complex, shape (3, n, n)
        Value, first derivative and second derivative of a Hermitian denominator at zero.
        Every slice is finite and Hermitian to absolute tolerance 1e-10.
    cutoff : float
        Finite nonnegative fixed spectral cutoff; retain eigenvalues strictly above it.
    gap_tolerance : float, optional
        Finite nonnegative exclusion distance from the cutoff, default 1e-10.
    
    Returns
    -------
    result : np.ndarray, complex, shape (3, n, n)
        P(0), P'(0), P''(0), the Hermitian orthogonal-projector derivatives in original coordinates.
    
    Raises
    ------
    ValueError
        If shapes, Hermiticity, finite values or scalar domains fail, or any base eigenvalue lies within gap_tolerance of the cutoff.
    
    Notes
    -----
    The local denominator is B(x)=B0+x*B1+(x*x/2)*B2+O(x**3); coefficients are actual derivatives. All-retained and all-discarded clusters have constant projectors and zero derivatives. Internal eigenvalue multiplicity is valid. Return the projector itself, not a choice of retained eigenvectors. The result must be independent of eigenbasis phases and of unitary changes within degenerate clusters.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_cutoff_projector_jet(denominator_jet: 'np.ndarray', cutoff: float, gap_tolerance: float = 1e-10) -> 'np.ndarray':
    import numpy as np
    from numbers import Real
    try:
        b = np.asarray(denominator_jet, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError('A numerical Hermitian matrix jet is required.') from exc
    if b.ndim != 3 or b.shape[0] != 3 or b.shape[1] != b.shape[2] or b.shape[1] < 1 or not np.all(np.isfinite(b)) or not np.allclose(b, b.conj().transpose(0, 2, 1), atol=1e-10, rtol=0):
        raise ValueError('The denominator jet must have shape (3, n, n), with finite Hermitian slices.')
    for x in (cutoff, gap_tolerance):
        if isinstance(x, bool) or not isinstance(x, Real) or not np.isfinite(x) or x < 0:
            raise ValueError('Cutoff and gap tolerance must be finite and nonnegative.')
    beta, u = np.linalg.eigh(b[0])
    if np.min(np.abs(beta - cutoff)) <= gap_tolerance:
        raise ValueError('A denominator eigenvalue is too close to the cutoff for this derivative contract.')
    retained = np.flatnonzero(beta > cutoff)
    discarded = np.flatnonzero(beta < cutoff)
    e = u.conj().T @ b[1] @ u
    f = u.conj().T @ b[2] @ u
    p0 = np.diag((beta > cutoff).astype(float)).astype(complex)
    p1 = np.zeros_like(p0)
    for i in discarded:
        for j in retained:
            p1[i, j] = e[i, j] / (beta[j] - beta[i])
            p1[j, i] = p1[i, j].conjugate()
    square = p1 @ p1
    p2 = np.zeros_like(p0)
    p2[np.ix_(retained, retained)] = -2 * square[np.ix_(retained, retained)]
    p2[np.ix_(discarded, discarded)] = 2 * square[np.ix_(discarded, discarded)]
    rhs = f + 2 * (e @ p1 - p1 @ e)
    for i in discarded:
        for j in retained:
            p2[i, j] = rhs[i, j] / (beta[j] - beta[i])
            p2[j, i] = p2[i, j].conjugate()
    return np.stack([u @ p @ u.conj().T for p in (p0, p1, p2)])

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
b0 = np.diag([0.1, 0.1, 0.8, 0.8]).astype(complex)
z = np.array([[0, 0.2j, 0.3, 0.1j], [-0.2j, 0.1, -0.25j, 0.4], [0.3, 0.25j, -0.2, 0.15], [-0.1j, 0.4, 0.15, 0.3]], complex)
r = np.arange(16).reshape(4, 4) / 30
b2 = r + r.T
b = np.stack([b0, z, b2])
"""

    setup_2 = """from copy import deepcopy
import numpy as np
b0 = np.diag([0.1, 0.1, 0.8, 0.8]).astype(complex)
z = np.array([[0, 0.2j, 0.3, 0.1j], [-0.2j, 0.1, -0.25j, 0.4], [0.3, 0.25j, -0.2, 0.15], [-0.1j, 0.4, 0.15, 0.3]], complex)
r = np.arange(16).reshape(4, 4) / 30
b2 = r + r.T
b = np.stack([b0, z, b2])
u, _ = np.linalg.qr(np.array([[1, 1j, 2, 0], [2, -1, 1j, 1], [0, 2, 1, 1j], [1j, 1, 0, 2]], complex))
b = np.array([u @ x @ u.conj().T for x in b])
"""

    setup_3 = """from copy import deepcopy
import numpy as np
b = np.stack([np.diag([1.0, 2.0]), np.ones((2, 2)), np.eye(2)])
"""

    setup_4 = """from copy import deepcopy
import numpy as np
b = np.stack([np.diag([0.1, 0.2]), np.ones((2, 2)), np.eye(2)])
"""

    setup_5 = """from copy import deepcopy
import numpy as np
b = np.stack([np.diag([0.499, 0.501, 1.0]), np.array([[0.0, 0.0002, 0.1], [0.0002, 0.0, 0.03], [0.1, 0.03, 0.0]]), np.eye(3) * 0.01])
"""

    setup_6 = """from copy import deepcopy
import numpy as np
b = np.stack([np.diag([0.2, 0.8]), np.zeros((2, 2)), np.array([[0.0, 0.3j], [-0.3j, 0.0]])])
"""

    setup_7 = """from copy import deepcopy
import numpy as np
b = np.stack([np.diag([0.5, 0.8]), np.zeros((2, 2)), np.zeros((2, 2))])
"""

    setup_8 = """from copy import deepcopy
import numpy as np
b = np.zeros((2, 3, 3))
"""

    setup_9 = """from copy import deepcopy
import numpy as np
b = np.zeros((3, 2, 2))
b[0] = np.eye(2)
b[1, 0, 1] = 1.0
"""

    return [
        # Case 1
        {
            'setup': setup_1,
            'call': 'cutoff_projector_jet(*deepcopy((b, 0.4)))',
            'gold_call': '_oracle_cutoff_projector_jet(*deepcopy((b, 0.4)))',
        },
        # Case 2
        {
            'setup': setup_2,
            'call': 'cutoff_projector_jet(*deepcopy((b, 0.4)))',
            'gold_call': '_oracle_cutoff_projector_jet(*deepcopy((b, 0.4)))',
        },
        # Case 3
        {
            'setup': setup_3,
            'call': 'cutoff_projector_jet(*deepcopy((b, 0.1)))',
            'gold_call': '_oracle_cutoff_projector_jet(*deepcopy((b, 0.1)))',
        },
        # Case 4
        {
            'setup': setup_4,
            'call': 'cutoff_projector_jet(*deepcopy((b, 0.9)))',
            'gold_call': '_oracle_cutoff_projector_jet(*deepcopy((b, 0.9)))',
        },
        # Case 5
        {
            'setup': setup_5,
            'call': 'cutoff_projector_jet(*deepcopy((b, 0.5)))',
            'gold_call': '_oracle_cutoff_projector_jet(*deepcopy((b, 0.5)))',
        },
        # Case 6
        {
            'setup': setup_6,
            'call': 'cutoff_projector_jet(*deepcopy((b, 0.5)))',
            'gold_call': '_oracle_cutoff_projector_jet(*deepcopy((b, 0.5)))',
        },
        # Case 7
        {
            'setup': setup_7 + exception_setup,
            'call': '_exception_code(cutoff_projector_jet, *deepcopy((b, 0.5)))',
            'gold_call': '_exception_code(_oracle_cutoff_projector_jet, *deepcopy((b, 0.5)))',
        },
        # Case 8
        {
            'setup': setup_8 + exception_setup,
            'call': '_exception_code(cutoff_projector_jet, *deepcopy((b, 0.5)))',
            'gold_call': '_exception_code(_oracle_cutoff_projector_jet, *deepcopy((b, 0.5)))',
        },
        # Case 9
        {
            'setup': setup_9 + exception_setup,
            'call': '_exception_code(cutoff_projector_jet, *deepcopy((b, 0.5)))',
            'gold_call': '_exception_code(_oracle_cutoff_projector_jet, *deepcopy((b, 0.5)))',
        },
    ]
