"""
Return the source's interface distance-like function for the phase field with FINITE NON-ZERO bounds, in which the volume fraction asymptotes to a small positive value and its complement rather than to zero and one. It is the interface thickness times the logarithm of a ratio formed from the volume fraction, and that ratio is NOT the one used for the unbounded phase field: the bound enters it. The exact ratio is derived in the source's appendix; recover it from there rather than from the main text. Add the very small constant 1e-100 to numerator and denominator to control the limit, and clamp each of them at zero first so the function stays finite in floating point at and beyond the bounds. Form each clamped quantity so that a volume fraction sitting exactly on either bound gives exactly zero before the constant is added, rather than a rounding residue of order 1e-16.

Reformulating a phase-field sharpening term through a distance-like variable makes the equilibrium interface profile an exact fixed point of the regularisation rather than an approximate one, which is what stops the profile from drifting as it is advected.

Returns
-------
A float64 array shaped like phi holding the distance-like function.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def interface_distance(phi: "np.ndarray", eps: float, delta: float) -> "np.ndarray":
    """Return the source's interface distance-like function for the phase field with finite
    non-zero bounds, evaluated elementwise with the 1e-100 safeguard and the zero clamps
    described in the step text.

    Args:
        phi: Volume fraction of phase one, any array shape.
        eps: Interface thickness, positive.
        delta: The finite bound, in [0, 0.5); the volume fraction asymptotes to delta and
            1 - delta.

    Returns:
        A float64 array shaped like phi holding the distance-like function.

    Raises:
        ValueError: If eps is not positive or delta does not lie in [0, 0.5).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_interface_distance(phi: "np.ndarray", eps: float, delta: float) -> "np.ndarray":
    phi = np.asarray(phi, dtype=np.float64)
    if eps <= 0.0 or not (0.0 <= delta < 0.5):
        raise ValueError("need eps > 0 and delta in [0, 0.5)")
    lo = np.maximum(phi - delta, 0.0) + 1.0e-100
    hi = np.maximum((1.0 - delta) - phi, 0.0) + 1.0e-100
    return eps * np.log(lo / hi)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nN = 200\ndx = 1.0 / N\nx = (np.arange(N) + 0.5) * dx\ndelta = 1e-2\neps = 2.0 * dx\nd = 0.25 - np.abs(x - 0.5)\nphi = delta + (1 - 2 * delta) * 0.5 * (1 + np.tanh(d / (2 * eps)))\n',
         'call': 'interface_distance(phi, eps, delta)',
         'gold_call': '_oracle_interface_distance(phi, eps, delta)', 'tol': 1e-10},
        {'setup': 'import numpy as np\nN = 64\nh = 1.0 / N\nx = (np.arange(N) + 0.5) * h\nX, Y = np.meshgrid(x, x, indexing="ij")\ndelta = 5e-2\neps = 2.0 * h\nd = 0.2 - np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)\nphi = delta + (1 - 2 * delta) * 0.5 * (1 + np.tanh(d / (2 * eps)))\n',
         'call': 'interface_distance(phi, eps, delta)',
         'gold_call': '_oracle_interface_distance(phi, eps, delta)', 'tol': 1e-10},
        {'setup': 'import numpy as np\ndelta = 0.25\neps = 0.01\nphi = np.array([0.25, 0.375, 0.5, 0.625, 0.75, 0.2])\n',
         'call': 'interface_distance(phi, eps, delta)',
         'gold_call': '_oracle_interface_distance(phi, eps, delta)', 'tol': 1e-10},
        {'setup': 'import numpy as np\n# invalid input: a bound of one half is not in [0, 0.5) and must raise ValueError\nphi = np.array([0.5])\neps = 0.01\ndelta = 0.5\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: interface_distance(phi, eps, delta))',
         'gold_call': '_catches_value_error(lambda: _oracle_interface_distance(phi, eps, delta))'},
    ]
