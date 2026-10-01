"""
Assemble the isotropic linear-elastic operator that the source uses for the bulk response, in the six-component tensor layout the source fixes in its discretisation section (Eq. 20). Return it together with the two moduli that Eq. (2) is written in. The layout the source fixes there is load bearing: it decides what the tensor norms appearing in the strength potentials evaluate to.

The source stores the bulk behaviour as linear elastic under small strains, Eq. (2), parameterised by a bulk and a shear modulus obtained from Young's modulus and Poisson ratio. Its computational section then rewrites every tensor as a six-component column so that a double contraction becomes a plain inner product; the scaling that makes that identity hold is stated there and must be reproduced exactly.

Returns
-------
A (7, 6) float64 array: rows 0 to 5 are the 6x6 elastic operator, row 6 is [bulk modulus, shear modulus, 0, 0, 0, 0].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def elastic_operator(E: float, nu: float) -> "np.ndarray":
    """Assemble the isotropic linear-elastic operator that the source uses for the bulk
    response, in the six-component tensor layout the source fixes in its discretisation
    section (Eq. 20), together with the two moduli that Eq. (2) is written in.

    Args:
        E: Young's modulus, in GPa.
        nu: Poisson ratio.

    Returns:
        A (7, 6) float64 array: rows 0 to 5 are the 6x6 elastic operator, row 6 is
        [bulk modulus, shear modulus, 0, 0, 0, 0].

    Raises:
        ValueError: If E is not a positive finite number or nu does not lie in (-1, 0.5).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _im() -> "np.ndarray":
    """Six-component column of the identity tensor."""
    return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def _oracle_elastic_operator(E: float, nu: float) -> "np.ndarray":
    if not np.isfinite(E) or E <= 0.0:
        raise ValueError("E must be a positive finite modulus")
    if not np.isfinite(nu) or nu <= -1.0 or nu >= 0.5:
        raise ValueError("nu must lie in (-1, 0.5)")
    im = _im()
    K = E / (3.0 * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    D = K * np.outer(im, im) + 2.0 * mu * (np.eye(6) - np.outer(im, im) / 3.0)
    out = np.zeros((7, 6))
    out[:6, :] = D
    out[6, 0] = K
    out[6, 1] = mu
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nE = 200.0\nnu = 0.3\n',
         'call': 'elastic_operator(E, nu)',
         'gold_call': '_oracle_elastic_operator(E, nu)'},
        {'setup': 'import numpy as np\nE = 70.0\nnu = 0.22\n',
         'call': 'elastic_operator(E, nu)',
         'gold_call': '_oracle_elastic_operator(E, nu)'},
        {'setup': 'import numpy as np\nE = 350.0\nnu = 0.42\n',
         'call': 'elastic_operator(E, nu)',
         'gold_call': '_oracle_elastic_operator(E, nu)'},
        {'setup': 'import numpy as np\n# invalid input: a Poisson ratio of 0.5 lies outside (-1, 0.5) and must raise ValueError\nE = 200.0\nnu = 0.5\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: elastic_operator(E, nu))',
         'gold_call': '_catches_value_error(lambda: _oracle_elastic_operator(E, nu))'},
    ]
