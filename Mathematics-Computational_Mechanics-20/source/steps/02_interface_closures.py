"""
Return the source's two interface quantities for a two-phase mixture, the interface pressure first and the interface velocity component second. Both are averages of the corresponding phasic quantities over the two phases, but they are NOT averaged with the same weights: one uses a weight built from the volume fractions alone and the other a weight built from the phasic masses. Which weight belongs to which quantity is the source's closure; recover it from the paper. The volume fraction given is that of phase one, so phase two carries its complement.

Averaging the two-phase equations leaves interface quantities that the phasic equations exchange momentum and work through; the model is closed only once they are prescribed, and different prescriptions give genuinely different models.

Returns
-------
A float64 array of shape (2,) + phi.shape: the interface pressure first, the interface velocity component second.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def interface_closures(phi: "np.ndarray", p1: "np.ndarray", p2: "np.ndarray", rho1: "np.ndarray", rho2: "np.ndarray", u1: "np.ndarray", u2: "np.ndarray") -> "np.ndarray":
    """Return the source's two interface quantities for a two-phase mixture, the interface
    pressure first and one component of the interface velocity second, each an average of
    the phasic quantities with the source's own weights.

    Args:
        phi: Volume fraction of phase one, any array shape; phase two carries 1 - phi.
        p1: Pressure of phase one, shaped like phi.
        p2: Pressure of phase two, shaped like phi.
        rho1: Density of phase one, shaped like phi.
        rho2: Density of phase two, shaped like phi.
        u1: One velocity component of phase one, shaped like phi.
        u2: The same velocity component of phase two, shaped like phi.

    Returns:
        A float64 array of shape (2,) + phi.shape: the interface pressure first, the interface
        velocity component second.

    Raises:
        ValueError: If any volume fraction lies outside [0, 1] or any mixture density is not
            positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_interface_closures(phi: "np.ndarray", p1: "np.ndarray", p2: "np.ndarray", rho1: "np.ndarray", rho2: "np.ndarray", u1: "np.ndarray", u2: "np.ndarray") -> "np.ndarray":
    phi = np.asarray(phi, dtype=np.float64)
    if np.any(phi < 0.0) or np.any(phi > 1.0):
        raise ValueError("volume fraction must lie in [0, 1]")
    m1 = phi * np.asarray(rho1, dtype=np.float64)
    m2 = (1.0 - phi) * np.asarray(rho2, dtype=np.float64)
    if np.any(m1 + m2 <= 0.0):
        raise ValueError("mixture density must be positive")
    pI = phi * np.asarray(p1, dtype=np.float64) + (1.0 - phi) * np.asarray(p2, dtype=np.float64)
    uI = (m1 * np.asarray(u1, dtype=np.float64) + m2 * np.asarray(u2, dtype=np.float64)) / (m1 + m2)
    return np.stack([pI, uI])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nphi = np.array([0.2, 0.5, 0.9])\np1 = np.array([1.0e5, 2.0e5, 1.5e5])\np2 = np.array([1.1e5, 1.0e5, 1.2e5])\nrho1 = np.array([1.0e3, 9.0e2, 1.1e3])\nrho2 = np.array([1.2, 1.0, 1.4])\nu1 = np.array([1.0, -0.5, 2.0])\nu2 = np.array([0.4, 1.3, -0.7])\n',
         'call': 'interface_closures(phi, p1, p2, rho1, rho2, u1, u2)',
         'gold_call': '_oracle_interface_closures(phi, p1, p2, rho1, rho2, u1, u2)', 'tol': 1e-8},
        {'setup': 'import numpy as np\nphi = np.array([[1.0e-2, 0.99], [0.5, 0.25]])\np1 = np.full((2, 2), 1.0e5)\np2 = np.full((2, 2), 1.0e5)\nrho1 = np.full((2, 2), 1.0e3)\nrho2 = np.full((2, 2), 1.2)\nu1 = np.array([[3.0, 3.0], [0.4, -0.2]])\nu2 = np.array([[3.0, 3.0], [1.0, 1.5]])\n',
         'call': 'interface_closures(phi, p1, p2, rho1, rho2, u1, u2)',
         'gold_call': '_oracle_interface_closures(phi, p1, p2, rho1, rho2, u1, u2)', 'tol': 1e-8},
        {'setup': 'import numpy as np\nphi = np.array([0.5])\np1 = np.array([2.0e5])\np2 = np.array([0.0])\nrho1 = np.array([1.0])\nrho2 = np.array([1.0])\nu1 = np.array([1.0])\nu2 = np.array([-1.0])\n',
         'call': 'interface_closures(phi, p1, p2, rho1, rho2, u1, u2)',
         'gold_call': '_oracle_interface_closures(phi, p1, p2, rho1, rho2, u1, u2)', 'tol': 1e-8},
        {'setup': 'import numpy as np\n# invalid input: a volume fraction above one lies outside [0, 1] and must raise ValueError\nphi = np.array([1.2])\np1 = np.array([1.0e5])\np2 = np.array([1.0e5])\nrho1 = np.array([1.0e3])\nrho2 = np.array([1.2])\nu1 = np.array([1.0])\nu2 = np.array([0.4])\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: interface_closures(phi, p1, p2, rho1, rho2, u1, u2))',
         'gold_call': '_catches_value_error(lambda: _oracle_interface_closures(phi, p1, p2, rho1, rho2, u1, u2))'},
    ]
