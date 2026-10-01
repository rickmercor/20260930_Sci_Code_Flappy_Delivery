"""
Evaluate the source's smooth, pressure-sensitive strength criterion, Eq. (18): its single eigenstrain direction, the gradient of its strength potential with respect to the fracture eigenstrain, and the potential's own value. The criterion is written as two pieces. Which piece applies is decided by one specific scalar of the current state, and that scalar is not the one the potential is differentiated against; the source says explicitly why it has to be that one.

Eq. (18) combines a tensile surface with a cone that stiffens under pressure, the stiffening measured against a reference strain supplied as data. The source notes that the eigenstrain's volumetric part is sign restricted by construction, so it cannot be the quantity that detects compression, and it also notes that the direction chosen in the compressive piece is what removes the need for a separate non-penetration potential.

Returns
-------
A (3, 6) float64 array: row 0 the eigenstrain direction, row 1 the potential gradient, row 2 is [potential value, branch flag, 0, 0, 0, 0] with branch flag +1 on the first piece of Eq. (18) and -1 on the second.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def smooth_direction_gradient(eps: "np.ndarray", eta: "np.ndarray", ft: float, fs: float, eps_ref: float) -> "np.ndarray":
    """Evaluate the source's smooth, pressure-sensitive strength criterion, Eq. (18): its
    single eigenstrain direction, the gradient of its strength potential with respect to the
    fracture eigenstrain, and the potential's own value, on the piece of Eq. (18) that the
    source's own switch selects.

    Args:
        eps: Total strain as a (6,) array in the source's six-component tensor layout.
        eta: Fracture eigenstrain as a (6,) array in the same layout.
        ft: Tensile strength, in GPa.
        fs: Shear strength, in GPa.
        eps_ref: Reference strain of the pressure-sensitive criterion.

    Returns:
        A (3, 6) float64 array: row 0 the eigenstrain direction, row 1 the potential
        gradient, row 2 is [potential value, branch flag, 0, 0, 0, 0] with branch flag +1 on
        the first piece of Eq. (18) and -1 on the second. Where the potential value is zero
        its gradient is taken as the zero vector, and a direction whose defining strain
        vanishes is the zero vector.

    Raises:
        ValueError: If eps or eta is not a finite (6,) array, or if ft, fs or eps_ref is not
            positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _im() -> "np.ndarray":
    """Six-component column of the identity tensor."""
    return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def _tr(v: "np.ndarray") -> float:
    return float(v[0] + v[1] + v[2])


def _dev(v: "np.ndarray") -> "np.ndarray":
    return v - (_tr(v) / 3.0) * _im()


def _unit(v: "np.ndarray") -> "np.ndarray":
    n = float(np.linalg.norm(v))
    return (v / n) if n > 0.0 else np.zeros(6)


def _oracle_smooth_direction_gradient(eps: "np.ndarray", eta: "np.ndarray", ft: float, fs: float, eps_ref: float) -> "np.ndarray":
    eps = np.asarray(eps, dtype=float)
    eta = np.asarray(eta, dtype=float)
    if eps.shape != (6,) or eta.shape != (6,):
        raise ValueError("eps and eta must be six-component arrays")
    if not (np.all(np.isfinite(eps)) and np.all(np.isfinite(eta))):
        raise ValueError("eps and eta must be finite")
    if ft <= 0.0 or fs <= 0.0 or eps_ref <= 0.0:
        raise ValueError("ft, fs and eps_ref must be positive")
    de_e = _dev(eps)
    de_n = _dev(eta)
    nd = float(np.linalg.norm(de_n))
    if _tr(eps) >= 0.0:
        G = _unit(eps)
        den = np.sqrt(ft ** 2 * _tr(eta) ** 2 + fs ** 2 * nd ** 2)
        dFd = np.zeros(6) if den == 0.0 else (ft ** 2 * _tr(eta) * _im() + fs ** 2 * de_n) / den
        Fd = float(den)
        branch = 1.0
    else:
        G = _unit(de_e)
        fac = fs * (1.0 - _tr(eps) / eps_ref)
        dFd = (fac * de_n / nd) if nd > 0.0 else np.zeros(6)
        Fd = float(fac * nd)
        branch = -1.0
    return np.vstack([G, dFd, np.array([Fd, branch, 0.0, 0.0, 0.0, 0.0])])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\neps = np.array([6.0e-4, -2.0e-4, 0.0, 0.0, 0.0, 9.0e-4])\neta = np.array([2.0e-4, 5.0e-5, 0.0, 0.0, 0.0, 3.0e-4])\nft = 0.15\nfs = 0.15\neps_ref = 0.01\n',
         'call': 'smooth_direction_gradient(eps, eta, ft, fs, eps_ref)',
         'gold_call': '_oracle_smooth_direction_gradient(eps, eta, ft, fs, eps_ref)'},
        {'setup': 'import numpy as np\neps = np.array([-1.2e-3, -1.3e-3, 0.0, 4.0e-4, 0.0, 2.0e-4])\neta = np.array([1.0e-4, -3.0e-4, 0.0, 0.0, 0.0, 2.5e-4])\nft = 0.15\nfs = 0.15\neps_ref = 0.01\n',
         'call': 'smooth_direction_gradient(eps, eta, ft, fs, eps_ref)',
         'gold_call': '_oracle_smooth_direction_gradient(eps, eta, ft, fs, eps_ref)'},
        {'setup': 'import numpy as np\neps = np.array([-8.0e-4, -5.0e-4, 0.0, 0.0, 0.0, 1.1e-3])\neta = np.zeros(6)\nft = 0.2\nfs = 0.1\neps_ref = 0.004\n',
         'call': 'smooth_direction_gradient(eps, eta, ft, fs, eps_ref)',
         'gold_call': '_oracle_smooth_direction_gradient(eps, eta, ft, fs, eps_ref)'},
        {'setup': 'import numpy as np\n# invalid input: a zero reference strain is not positive and must raise ValueError\neps = np.array([6.0e-4, -2.0e-4, 0.0, 0.0, 0.0, 9.0e-4])\neta = np.zeros(6)\nft = 0.15\nfs = 0.15\neps_ref = 0.0\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: smooth_direction_gradient(eps, eta, ft, fs, eps_ref))',
         'gold_call': '_catches_value_error(lambda: _oracle_smooth_direction_gradient(eps, eta, ft, fs, eps_ref))'},
    ]
