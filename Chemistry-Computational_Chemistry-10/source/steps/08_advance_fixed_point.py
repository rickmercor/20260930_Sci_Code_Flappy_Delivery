"""
One iteration of the fixed-point map.

Each iteration takes the current pair of derived arguments, reconstructs the pair of coexisting fractions they encode, re-evaluates the two scalar combinations of the coexistence conditions on that reconstructed pair, and returns the arguments implied in turn by those combinations. One iteration therefore gives exactly the arguments that the previous step would return for the reconstructed pair.

Because the fractions are reconstructed from the hyperbolic-sine factor and an exponential of the second argument, the whole update can be written directly in terms of the current arguments and the factor, with no explicit reference to the fractions. Every appearance of a fraction raised to a power becomes a power of that factor multiplied by a hyperbolic sine of a correspondingly scaled second argument.

The physical-branch check applies to the incoming arguments only: the hyperbolic-sine factor they imply must be finite and strictly positive, and an input whose factor is zero, negative or not finite is rejected. No further range check is made on the fractions that an accepted input encodes. A negative first argument is likewise accepted whenever the incoming factor is positive. 

The iteration converges when the spectral radius of the map's Jacobian at the fixed point is below one.

Return the updated pair of arguments together with the hyperbolic-sine factor they imply.

Returns
-------
np.ndarray of shape (3,), dtype float: [updated first derived argument, updated second derived argument, updated hyperbolic-sine factor]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def advance_fixed_point(a: float, b: float, chi: float, gamma: float,
                       r_eff: float) -> np.ndarray:
    """Advance the coexistence fixed-point map by one iteration.

    Parameters
    ----------
    a : float
        Current first derived argument.
    b : float
        Current second derived argument.
    chi : float
        Dimensionless interaction parameter, >= 0.
    gamma : float
        Length-asymmetry coefficient of the reduced model.
    r_eff : float
        Effective chain length, > 0.

    Returns
    -------
    out : np.ndarray
        Three floats: the updated first derived argument, the updated second derived
        argument, and the ratio-of-hyperbolic-sines factor built from the updated pair.

    Raises
    ------
    ValueError
       If any argument is not a finite real scalar, if r_eff is not positive or chi is negative, if the incoming arguments are degenerate, if the factor they imply is not finite and strictly positive, or if the update is not finite.
    """
    return out  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _real_scalar(name, value):
    """Reject arrays, complex values, bools and non-finite input."""
    if isinstance(value, bool) or isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(name + " must be a real scalar")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + " must be a real scalar")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(name + " must be finite")
    return v


def _positive(name, value):
    v = _real_scalar(name, value)
    if v <= 0.0:
        raise ValueError(name + " must be strictly positive")
    return v


def _fraction(name, value):
    v = _real_scalar(name, value)
    if not (0.0 < v < 1.0):
        raise ValueError(name + " must lie strictly inside (0, 1)")
    return v


def _H(a, b):
    a = _real_scalar("a", a)
    b = _real_scalar("b", b)
    den = np.sinh(a + b)
    if den == 0.0:
        raise ValueError("degenerate arguments: sinh(a + b) vanishes")
    return np.sinh(a) / den


def _xy(phi_I, phi_II, chi, gamma):
    x = 1.5 * chi * (np.sqrt(phi_I) - np.sqrt(phi_II))
    y = 0.5 * chi * (phi_I ** 1.5 - phi_II ** 1.5) + gamma * (phi_I - phi_II)
    return x, y


def _ab_seed(phic, chi, gamma, r_eff, D):
    a0 = (0.75 * chi * np.sqrt(phic) + gamma) * D / 2.0
    b0 = r_eff * (0.75 * chi * (1.0 - phic) / np.sqrt(phic) - gamma) * D / 2.0
    return a0, b0


def _ab_step(a, b, chi, gamma, r_eff):
    Hp = _H(a, b)
    if not np.isfinite(Hp) or Hp <= 0.0:
        raise ValueError("iterate has left the physical branch (H <= 0)")
    a_new = 0.5 * chi * Hp ** 1.5 * np.sinh(1.5 * b) + gamma * Hp * np.sinh(b)
    b_new = r_eff * (0.5 * chi * (3.0 * np.sqrt(Hp) * np.sinh(0.5 * b)
                                  - Hp ** 1.5 * np.sinh(1.5 * b))
                     - gamma * Hp * np.sinh(b))
    if not (np.isfinite(a_new) and np.isfinite(b_new)):
        raise ValueError("iteration produced non-finite arguments")
    return a_new, b_new


def _oracle_advance_fixed_point(a: float, b: float, chi: float, gamma: float,
                       r_eff: float) -> np.ndarray:
    """Reference implementation."""
    aa = _real_scalar("a", a)
    bb = _real_scalar("b", b)
    c = _real_scalar("chi", chi)
    g = _real_scalar("gamma", gamma)
    re = _positive("r_eff", r_eff)
    if c < 0.0:
        raise ValueError("chi must be non-negative")
    a_new, b_new = _ab_step(aa, bb, c, g, re)
    return np.array([a_new, b_new, _H(a_new, b_new)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # reference system, seed arguments to first iteration
            "setup": 'import numpy as np\na, b = 1.5891142476084645, 8.4806966297935098\nchi, gamma, r_eff = 2.7350787009788555, 0.6287878787878788, 2.693877551020408\n',
            "call": 'advance_fixed_point(a, b, chi, gamma, r_eff)',
            "gold_call": '_oracle_advance_fixed_point(a, b, chi, gamma, r_eff)',
        },
        {
            # reference system, second iteration
            "setup": 'import numpy as np\na, b = 0.94278614953349282, 2.868773935924875\nchi, gamma, r_eff = 2.7350787009788555, 0.6287878787878788, 2.693877551020408\n',
            "call": 'advance_fixed_point(a, b, chi, gamma, r_eff)',
            "gold_call": '_oracle_advance_fixed_point(a, b, chi, gamma, r_eff)',
        },
        {
            # monomeric case with vanishing length-asymmetry coefficient
            "setup": 'import numpy as np\na, b = 0.70653975, 1.41307950\nchi, gamma, r_eff = 4.0, 0.0, 1.0\n',
            "call": 'advance_fixed_point(a, b, chi, gamma, r_eff)',
            "gold_call": '_oracle_advance_fixed_point(a, b, chi, gamma, r_eff)',
        },
        {
            # small arguments near the critical point
            "setup": 'import numpy as np\na, b = 1e-4, 3e-4\nchi, gamma, r_eff = 1.9, 0.6287878787878788, 2.693877551020408\n',
            "call": 'advance_fixed_point(a, b, chi, gamma, r_eff)',
            "gold_call": '_oracle_advance_fixed_point(a, b, chi, gamma, r_eff)',
        },
        {
            # uncharged limit
            "setup": 'import numpy as np\na, b = 0.5, 1.2\nchi, gamma, r_eff = 0.0, 0.5, 2.0\n',
            "call": 'advance_fixed_point(a, b, chi, gamma, r_eff)',
            "gold_call": '_oracle_advance_fixed_point(a, b, chi, gamma, r_eff)',
        },
        {
            # rejects arguments summing to zero
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        advance_fixed_point(1.0, -1.0, 2.7, 0.6, 2.7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_advance_fixed_point(1.0, -1.0, 2.7, 0.6, 2.7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects both arguments zero
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        advance_fixed_point(0.0, 0.0, 2.7, 0.6, 2.7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_advance_fixed_point(0.0, 0.0, 2.7, 0.6, 2.7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # accepts a negative first argument, since only a non-positive incoming factor is rejected
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        advance_fixed_point(-1.0, 0.4, 2.7, 0.6, 2.7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_advance_fixed_point(-1.0, 0.4, 2.7, 0.6, 2.7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects non-finite argument
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        advance_fixed_point(np.inf, 1.0, 2.7, 0.6, 2.7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_advance_fixed_point(np.inf, 1.0, 2.7, 0.6, 2.7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects negative effective length
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        advance_fixed_point(1.0, 2.0, 2.7, 0.6, -2.7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_advance_fixed_point(1.0, 2.0, 2.7, 0.6, -2.7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
