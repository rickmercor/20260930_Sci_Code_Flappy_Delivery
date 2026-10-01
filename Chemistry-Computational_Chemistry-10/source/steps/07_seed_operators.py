"""
Derived arguments evaluated on the near-critical seed.

The near-critical estimate is the natural starting point for the iteration, but it cannot be fed to the scalar combinations of the previous step directly: away from the critical point its dilute root falls below zero, and those combinations require square roots of both fractions.

The resolution is to evaluate the combinations for the seed at linear order instead. Each power of a fraction in the two scalar combinations of the previous step is replaced by its first-order behaviour about the critical fraction, with the two fractions separated by the gap width, so that the seed enters only through the width of the coexistence gap rather than through the individual fractions. The result stays real no matter how far outside the physical range the seed has strayed. This linearisation applies to the seed alone; every later iteration uses the exact combinations.

Return the two derived arguments obtained this way together with the corresponding hyperbolic-sine factor.

Returns
-------
np.ndarray of shape (3,), dtype float: [first derived argument, second derived argument, hyperbolic-sine factor]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def seed_operators(phi_c: float, chi: float, gamma: float, r_eff: float,
                  D: float) -> np.ndarray:
    """Derived arguments of the fixed-point map for the near-critical seed.

    Parameters
    ----------
    phi_c : float
        Critical total polymer volume fraction, strictly inside (0, 1).
    chi : float
        Dimensionless interaction parameter, >= 0.
    gamma : float
        Length-asymmetry coefficient of the reduced model.
    r_eff : float
        Effective chain length, > 0.
    D : float
        Width of the coexistence gap implied by the near-critical expansion, > 0.

    Returns
    -------
    out : np.ndarray
        Three floats: the first derived argument, the second derived argument, and the
        ratio-of-hyperbolic-sines factor built from them.

    Raises
    ------
    ValueError
        If the arguments are out of range, or if the derived arguments are degenerate.
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


def _oracle_seed_operators(phi_c: float, chi: float, gamma: float, r_eff: float,
                  D: float) -> np.ndarray:
    """Reference implementation."""
    pc = _fraction("phi_c", phi_c)
    c = _real_scalar("chi", chi)
    g = _real_scalar("gamma", gamma)
    re = _positive("r_eff", r_eff)
    d = _real_scalar("D", D)
    if c < 0.0:
        raise ValueError("chi must be non-negative")
    if d < 0.0:
        raise ValueError("D must be non-negative")
    a0, b0 = _ab_seed(pc, c, g, re, d)
    return np.array([a0, b0, _H(a0, b0)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # reference system: the seed has a negative dilute root yet this stays real
            "setup": 'import numpy as np\nphi_c, chi, gamma, r_eff, D = (0.1987837280267722, 2.7350787009788555,\n                              0.6287878787878788, 2.693877551020408,\n                              2.0592817681715845)\n',
            "call": 'seed_operators(phi_c, chi, gamma, r_eff, D)',
            "gold_call": '_oracle_seed_operators(phi_c, chi, gamma, r_eff, D)',
        },
        {
            # monomeric case
            "setup": 'import numpy as np\nphi_c, chi, gamma, r_eff, D = 1.0/3.0, 4.0, 0.0, 1.0, 0.8158418251\n',
            "call": 'seed_operators(phi_c, chi, gamma, r_eff, D)',
            "gold_call": '_oracle_seed_operators(phi_c, chi, gamma, r_eff, D)',
        },
        {
            # long chains
            "setup": 'import numpy as np\nphi_c, chi, gamma, r_eff, D = 0.0097123, 0.3, 0.99, 100.0, 0.02\n',
            "call": 'seed_operators(phi_c, chi, gamma, r_eff, D)',
            "gold_call": '_oracle_seed_operators(phi_c, chi, gamma, r_eff, D)',
        },
        {
            # tiny gap width
            "setup": 'import numpy as np\nphi_c, chi, gamma, r_eff, D = 0.2, 2.0, 0.6, 2.7, 1e-6\n',
            "call": 'seed_operators(phi_c, chi, gamma, r_eff, D)',
            "gold_call": '_oracle_seed_operators(phi_c, chi, gamma, r_eff, D)',
        },
        {
            # rejects a closed gap, which makes both arguments vanish
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        seed_operators(0.2, 2.0, 0.6, 2.7, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_seed_operators(0.2, 2.0, 0.6, 2.7, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects negative gap width
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        seed_operators(0.2, 2.0, 0.6, 2.7, -1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_seed_operators(0.2, 2.0, 0.6, 2.7, -1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects critical fraction at zero
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        seed_operators(0.0, 2.0, 0.6, 2.7, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_seed_operators(0.0, 2.0, 0.6, 2.7, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects negative chi
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        seed_operators(0.2, -2.0, 0.6, 2.7, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_seed_operators(0.2, -2.0, 0.6, 2.7, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
