"""
Compute the second-order physical ground-energy response in a moving retained subspace.

Compression and differentiation do not commute when the projector moves. A smooth orthonormal frame for its range, transported by $[P^{\prime},P]$, gives frame-independent energy derivatives after generalized metric normalization and inverse eigenray rotation. Derive the response from the moving projected generalized eigenproblem, including first-order eigenvector mixing.

Returns
-------
np.ndarray, float, shape (3,), containing the physical ground energy E(0), its first derivative E'(0), and its second derivative E''(0), in that order, for the locally simple branch in the supplied moving retained subspace after inverse rotation; derivatives are not divided by factorials.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh

def projected_energy_jet(numerator_jet: 'np.ndarray', denominator_jet: 'np.ndarray', projector_jet: 'np.ndarray', theta: float, gap_tolerance: float = 1e-10) -> 'np.ndarray':
    """Compute the second-order physical ground-energy response in a moving retained subspace.
    
    Parameters
    ----------
    numerator_jet, denominator_jet : np.ndarray, complex, shape (3, n, n)
        Rotated pencil value, first derivative and second derivative at zero.
    projector_jet : np.ndarray, complex, shape (3, n, n)
        Value, first derivative and second derivative of the orthogonal retained projector.
        Slices obey differentiated idempotency to absolute Frobenius tolerance 1e-7.
    theta : float
        Finite fixed rotation angle in radians.
    gap_tolerance : float, optional
        Finite nonnegative exclusion gap for the selected generalized eigenvalue, default 1e-10.
    
    Returns
    -------
    result : np.ndarray, float, shape (3,)
        Physical ground energy, its first derivative and its second derivative at zero.
    
    Raises
    ------
    ValueError
        If jets have unequal or invalid shapes, nonfinite or non-Hermitian slices, invalid projector identities, an empty range, a non-positive compressed denominator, an excluded inverse-map pole with no usable root, a nonisolated selected root, or invalid scalar parameters.
    
    Notes
    -----
    Hermiticity is checked at absolute tolerance 1e-10. The supplied projector jet defines the moving subspace; the range of P0 is identified by projector eigenvalues >0.5. For a smooth orthonormal frame U(x) of that range, solve the pencil U(x)^dagger A(x) U(x), U(x)^dagger B(x) U(x). Select the lowest physical energy after inverse rotation, excluding inverse-map denominators of absolute value <=1e-12, and follow its locally simple branch. Other generalized eigenvalues may be degenerate with one another. Metric normalization, frame acceleration, second-order mixing and the curvature of the inverse map must all be retained. Differentiating unnormalized coordinate components or holding the projector fixed defines a different observable.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_projected_energy_jet(numerator_jet: 'np.ndarray', denominator_jet: 'np.ndarray', projector_jet: 'np.ndarray', theta: float, gap_tolerance: float = 1e-10) -> 'np.ndarray':
    import numpy as np
    from numbers import Real
    from scipy.linalg import eigh
    try:
        a = np.asarray(numerator_jet, dtype=complex)
        b = np.asarray(denominator_jet, dtype=complex)
        p = np.asarray(projector_jet, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError('Numerical matrix jets are required.') from exc
    if a.ndim != 3 or a.shape[0] != 3 or a.shape[1] != a.shape[2] or a.shape[1] < 1 or b.shape != a.shape or p.shape != a.shape:
        raise ValueError('All three jets must have the same shape (3, n, n).')
    for x in (a, b, p):
        if not np.all(np.isfinite(x)) or not np.allclose(x, x.conj().transpose(0, 2, 1), atol=1e-10, rtol=0):
            raise ValueError('Jet slices must be finite and Hermitian.')
    if isinstance(theta, bool) or not isinstance(theta, Real) or not np.isfinite(theta):
        raise ValueError('The angle must be a finite real scalar.')
    if isinstance(gap_tolerance, bool) or not isinstance(gap_tolerance, Real) or not np.isfinite(gap_tolerance) or gap_tolerance < 0:
        raise ValueError('The gap tolerance must be finite and nonnegative.')
    identities = (p[0] @ p[0] - p[0], p[0] @ p[1] + p[1] @ p[0] - p[1], p[0] @ p[2] + p[2] @ p[0] + 2 * p[1] @ p[1] - p[2])
    if any(np.linalg.norm(x, 'fro') > 1e-7 for x in identities):
        raise ValueError('The supplied projector jet violates differentiated idempotency.')
    values, vectors = np.linalg.eigh(p[0])
    u0 = vectors[:, values > 0.5]
    if u0.shape[1] == 0:
        raise ValueError('The retained subspace is empty.')
    k = p[1] @ p[0] - p[0] @ p[1]
    dk = p[2] @ p[0] - p[0] @ p[2]
    u1 = k @ u0
    u2 = (dk + k @ k) @ u0
    compressed = []
    for m in (a, b):
        m0 = u0.conj().T @ m[0] @ u0
        m1 = u1.conj().T @ m[0] @ u0 + u0.conj().T @ m[1] @ u0 + u0.conj().T @ m[0] @ u1
        m2 = u2.conj().T @ m[0] @ u0 + u0.conj().T @ m[0] @ u2 + 2 * u1.conj().T @ m[0] @ u1 + 2 * u1.conj().T @ m[1] @ u0 + 2 * u0.conj().T @ m[1] @ u1 + u0.conj().T @ m[2] @ u0
        compressed.append([(x + x.conj().T) / 2 for x in (m0, m1, m2)])
    aa, bb = compressed
    if np.min(np.linalg.eigvalsh(bb[0])) <= 0:
        raise ValueError('The compressed denominator must be positive definite.')
    lam, z = eigh(aa[0], bb[0])
    c, s = np.cos(theta), np.sin(theta)
    den = c - s * lam
    good = np.flatnonzero(np.abs(den) > 1e-12)
    if good.size == 0:
        raise ValueError('Every retained root is at the excluded inverse-rotation pole.')
    physical = (c * lam[good] + s) / den[good]
    g = int(good[np.argmin(physical)])
    other = [j for j in range(len(lam)) if j != g]
    if other and np.min(np.abs(lam[g] - lam[other])) <= gap_tolerance:
        raise ValueError('The selected generalized eigenvalue is not sufficiently isolated.')
    target = z[:, g]
    residual = aa[1] - lam[g] * bb[1]
    first = float(np.real(target.conj() @ residual @ target))
    second = float(np.real(target.conj() @ (aa[2] - lam[g] * bb[2]) @ target))
    second -= 2 * first * float(np.real(target.conj() @ bb[1] @ target))
    for j in other:
        coupling = z[:, j].conj() @ residual @ target
        second += 2 * abs(coupling) ** 2 / (lam[g] - lam[j])
    result = np.array([(c * lam[g] + s) / den[g], first / den[g] ** 2, second / den[g] ** 2 + 2 * s * first ** 2 / den[g] ** 3], dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError('The physical energy jet is nonfinite.')
    return result

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
a0 = np.array([[0.4, 0.1j, 0.2, 0.05], [-0.1j, 0.6, 0.1, 0.12j], [0.2, 0.1, 0.7, 0.2], [0.05, -0.12j, 0.2, 1.2]], complex)
a1 = 0.11 * np.array([[1, 0.2, 0.1j, 0], [0.2, -1, 0.3, 0.2], [-0.1j, 0.3, 0.5, 0.4j], [0, 0.2, -0.4j, 0.7]], complex)
a2 = -0.3 * a0
aj = np.stack([a0, a1, a2])
pj = np.array([
    [
        [0j, 0j, 0j, 0j],
        [0j, 0j, 0j, 0j],
        [0j, 0j, (1+0j), 0j],
        [0j, 0j, 0j, (1+0j)],
    ],
    [
        [0j, 0j, (0.4285714285714285+0j), 0.14285714285714285j],
        [0j, 0j, -0.3571428571428571j, (0.5714285714285714+0j)],
        [(0.4285714285714285+0j), 0.3571428571428571j, 0j, 0j],
        [-0.14285714285714285j, (0.5714285714285714+0j), 0j, 0j],
    ],
    [
        [(0.4081632653061223+0j), 0.46938775510204067j, (0.9251700680272107-0.06122448979591835j), (0.5306122448979591+0.20408163265306117j)],
        [-0.46938775510204067j, (0.9081632653061223+0j), (0.46938775510204084-0.5510204081632651j), (0.707482993197279+0.15306122448979587j)],
        [(0.9251700680272107+0.06122448979591835j), (0.46938775510204084+0.5510204081632651j), (-0.6224489795918365+0j), -0.5306122448979591j],
        [(0.5306122448979591-0.20408163265306117j), (0.707482993197279-0.15306122448979587j), 0.5306122448979591j, (-0.6938775510204082+0j)],
    ],
], dtype=np.complex128)
"""

    setup_2 = """from copy import deepcopy
import numpy as np
aj = np.array([[[2.0]], [[0.3]], [[0.1]]])
b = np.array([[[1.0]], [[0.2]], [[0.4]]])
pj = np.array([[[1.0]], [[0.0]], [[0.0]]])
"""

    setup_3 = """from copy import deepcopy
import numpy as np
aj = np.stack([np.diag([0.2, 1.0, 1.0]), np.ones((3, 3)) * 0.1, np.eye(3) * 0.03])
b = np.stack([np.eye(3), np.eye(3) * 0.2, np.eye(3) * 0.1])
pj = np.stack([np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))])
"""

    setup_4 = """from copy import deepcopy
import numpy as np
aj = np.stack([np.diag([0.0, 2.0]), np.diag([0.1, 0.2]), np.diag([0.3, 0.4])])
b = np.stack([np.eye(2), np.diag([0.2, 0.3]), np.eye(2) * 0.1])
pj = np.stack([np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))])
"""

    setup_5 = """from copy import deepcopy
import numpy as np
aj = np.stack([np.eye(2), np.eye(2), np.zeros((2, 2))])
b = np.stack([np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))])
pj = b.copy()
"""

    return [
        # Case 1
        {
            'setup': setup_1,
            'call': 'projected_energy_jet(*deepcopy((aj, b, pj, 0.3)))',
            'gold_call': '_oracle_projected_energy_jet(*deepcopy((aj, b, pj, 0.3)))',
        },
        # Case 2
        {
            'setup': setup_1 + 'u, _ = np.linalg.qr(np.array([[1, 1j, 2, 0], [2, -1, 1j, 1], [0, 2, 1, 1j], [1j, 1, 0, 2]], complex))\naj = np.array([u @ x @ u.conj().T for x in aj])\nb = np.array([u @ x @ u.conj().T for x in b])\npj = np.array([u @ x @ u.conj().T for x in pj])\n',
            'call': 'projected_energy_jet(*deepcopy((aj, b, pj, 0.3)))',
            'gold_call': '_oracle_projected_energy_jet(*deepcopy((aj, b, pj, 0.3)))',
        },
        # Case 3
        {
            'setup': setup_2,
            'call': 'projected_energy_jet(*deepcopy((aj, b, pj, 0.0)))',
            'gold_call': '_oracle_projected_energy_jet(*deepcopy((aj, b, pj, 0.0)))',
        },
        # Case 4
        {
            'setup': setup_3,
            'call': 'projected_energy_jet(*deepcopy((aj, b, pj, 0.2)))',
            'gold_call': '_oracle_projected_energy_jet(*deepcopy((aj, b, pj, 0.2)))',
        },
        # Case 5
        {
            'setup': setup_4,
            'call': 'projected_energy_jet(*deepcopy((aj, b, pj, 0.7)))',
            'gold_call': '_oracle_projected_energy_jet(*deepcopy((aj, b, pj, 0.7)))',
        },
        # Case 6
        {
            'setup': setup_1 + 'aj[1:] = 0\nb[1:] = 0\npj[1:] = 0\n',
            'call': 'projected_energy_jet(*deepcopy((aj, b, pj, 0.3)))',
            'gold_call': '_oracle_projected_energy_jet(*deepcopy((aj, b, pj, 0.3)))',
        },
        # Case 7
        {
            'setup': setup_1 + 'pj[0] = 0\npj[1:] = 0\n' + exception_setup,
            'call': '_exception_code(projected_energy_jet, *deepcopy((aj, b, pj, 0.3)))',
            'gold_call': '_exception_code(_oracle_projected_energy_jet, *deepcopy((aj, b, pj, 0.3)))',
        },
        # Case 8
        {
            'setup': setup_5 + exception_setup,
            'call': '_exception_code(projected_energy_jet, *deepcopy((aj, b, pj, 0.0)))',
            'gold_call': '_exception_code(_oracle_projected_energy_jet, *deepcopy((aj, b, pj, 0.0)))',
        },
        # Case 9
        {
            'setup': setup_1 + 'pj[1] = np.eye(4)\n' + exception_setup,
            'call': '_exception_code(projected_energy_jet, *deepcopy((aj, b, pj, 0.3)))',
            'gold_call': '_exception_code(_oracle_projected_energy_jet, *deepcopy((aj, b, pj, 0.3)))',
        },
        # Case 10
        {
            'setup': setup_1 + 'b[0] = -np.eye(4)\n' + exception_setup,
            'call': '_exception_code(projected_energy_jet, *deepcopy((aj, b, pj, 0.3)))',
            'gold_call': '_exception_code(_oracle_projected_energy_jet, *deepcopy((aj, b, pj, 0.3)))',
        },
        # Case 11
        {
            'setup': setup_1 + 'aj = aj[:2]\n' + exception_setup,
            'call': '_exception_code(projected_energy_jet, *deepcopy((aj, b, pj, 0.3)))',
            'gold_call': '_exception_code(_oracle_projected_energy_jet, *deepcopy((aj, b, pj, 0.3)))',
        },
    ]
