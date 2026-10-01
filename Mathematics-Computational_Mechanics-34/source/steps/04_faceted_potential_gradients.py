"""
Differentiate both strength potentials of the non-smooth criterion, Eq. (17a), with respect to the fracture eigenstrain. The first potential is the degradable one that sets the tensile and shear capacity; the second is the potential the source adds to hold the non-penetration condition once the point has lost its cohesion. Reproduce the second potential's magnitude and sign exactly as the source states them, and note which of the two the degradation function is allowed to touch.

The two potentials of Eq. (17a) are switched by the sign of the eigenstrain's volumetric part through the source's positive and negative part brackets, so only one of them acts at a time. The source explains in the text following Eq. (17) why the second potential's coefficient is set the way it is and why it is kept out of the degradation, and that explanation is what fixes both its scale and its sign.

Returns
-------
A (2, 6) float64 array: row 0 is the gradient of the degradable potential, row 1 the gradient of the non-degradable one.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def faceted_potential_gradients(eta: "np.ndarray", ft: float, fs: float) -> "np.ndarray":
    """Differentiate both strength potentials of the non-smooth criterion, Eq. (17a), with
    respect to the fracture eigenstrain: the degradable potential that sets the tensile and
    shear capacity, and the non-degradable potential that holds the non-penetration condition.

    Args:
        eta: Fracture eigenstrain as a (6,) array in the source's six-component tensor layout.
        ft: Tensile strength, in GPa.
        fs: Shear strength, in GPa.

    Returns:
        A (2, 6) float64 array: row 0 is the gradient of the degradable potential, row 1 the
        gradient of the non-degradable one. At a vanishing trace of eta the positive-part
        bracket of the degradable potential is inactive and the negative-part bracket of the
        non-degradable potential is active, and at a vanishing deviatoric part of eta the
        deviatoric term of the degradable gradient is zero.

    Raises:
        ValueError: If eta is not a finite (6,) array, or if ft or fs is not positive.
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


def _oracle_faceted_potential_gradients(eta: "np.ndarray", ft: float, fs: float) -> "np.ndarray":
    eta = np.asarray(eta, dtype=float)
    if eta.shape != (6,) or not np.all(np.isfinite(eta)):
        raise ValueError("eta must be a finite six-component array")
    if ft <= 0.0 or fs <= 0.0:
        raise ValueError("ft and fs must be positive strengths")
    im = _im()
    de = _dev(eta)
    nd = float(np.linalg.norm(de))
    dFd = ft * (1.0 if _tr(eta) > 0.0 else 0.0) * im
    if nd > 0.0:
        dFd = dFd + fs * de / nd
    dFi = (-1.0e6) * ft * (1.0 if _tr(eta) <= 0.0 else 0.0) * im
    return np.vstack([dFd, dFi])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\neta = np.array([3.0e-4, 1.0e-4, 0.0, 0.0, 0.0, 2.0e-4])\nft = 0.15\nfs = 0.15\n',
         'call': 'faceted_potential_gradients(eta, ft, fs)',
         'gold_call': '_oracle_faceted_potential_gradients(eta, ft, fs)'},
        {'setup': 'import numpy as np\neta = np.array([-4.0e-4, -2.0e-4, 0.0, 0.0, 0.0, 1.0e-4])\nft = 0.15\nfs = 0.15\n',
         'call': 'faceted_potential_gradients(eta, ft, fs)',
         'gold_call': '_oracle_faceted_potential_gradients(eta, ft, fs)'},
        {'setup': 'import numpy as np\neta = np.zeros(6)\nft = 0.2\nfs = 0.09\n',
         'call': 'faceted_potential_gradients(eta, ft, fs)',
         'gold_call': '_oracle_faceted_potential_gradients(eta, ft, fs)'},
        {'setup': 'import numpy as np\n# invalid input: a zero tensile strength is not positive and must raise ValueError\neta = np.array([3.0e-4, 1.0e-4, 0.0, 0.0, 0.0, 2.0e-4])\nft = 0.0\nfs = 0.15\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: faceted_potential_gradients(eta, ft, fs))',
         'gold_call': '_catches_value_error(lambda: _oracle_faceted_potential_gradients(eta, ft, fs))'},
    ]
