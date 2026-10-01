"""
Scalar combinations that carry the coexistence conditions.

The route to an analytic solution is to rearrange the two coexistence conditions so that the pair of coexisting fractions appears on both sides, turning the problem into a fixed-point problem that can be iterated.

That rearrangement works through two scalar combinations of the trial pair. The first collects the electrostatic contribution to the equality of chemical potentials, x = (3/2) chi (sqrt(phi_I) - sqrt(phi_II)); the second collects the electrostatic and length-asymmetry contributions to the equality of osmotic pressures, y = (1/2) chi (phi_I^(3/2) - phi_II^(3/2)) + gamma (phi_I - phi_II). The fixed-point map is expressed through a pair of derived arguments a and b, built from x, y and the effective chain length, and the factor H = sinh(a) / sinh(a + b). The new dense and dilute fractions are phi_I = exp(b) H and phi_II = exp(-b) H, and a and b are the unique arguments for which the pair rebuilt this way satisfies both coexistence conditions exactly when x and y are held at the values computed from the trial pair.

Both trial fractions must be genuine volume fractions for these combinations to be real, since they enter through square roots and through logarithms of the solvent fraction. The dense fraction must exceed the dilute one. That requirement is on the inputs. The hyperbolic-sine factor returned is reported as computed and may be zero or negative, which happens in the uncharged limit where the first combination vanishes, so a non-positive factor is a valid return value rather than an error.

Return the two scalar combinations, the two derived arguments, and the hyperbolic-sine factor.

Returns
-------
np.ndarray of shape (5,), dtype float: [chemical-potential combination, osmotic-pressure combination, first derived argument, second derived argument, hyperbolic-sine factor]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fixed_point_operators(phi_I: float, phi_II: float, chi: float, gamma: float,
                         r_eff: float) -> np.ndarray:
    """Scalar combinations and derived arguments of the coexistence fixed-point map.

    Parameters
    ----------
    phi_I : float
        Trial dense-phase volume fraction, strictly inside (0, 1).
    phi_II : float
        Trial dilute-phase volume fraction, strictly inside (0, 1) and below phi_I.
    chi : float
        Dimensionless interaction parameter, >= 0.
    gamma : float
        Length-asymmetry coefficient of the reduced model.
    r_eff : float
        Effective chain length, > 0.

    Returns
    -------
    out : np.ndarray
        Five floats, in order: the scalar combination carrying the equality of
        chemical potentials; the scalar combination carrying the equality of osmotic
        pressures; the first derived argument; the second derived argument; and the
        ratio-of-hyperbolic-sines factor through which the map is expressed.

    Raises
    ------
    ValueError
        If either fraction lies outside (0, 1), if phi_I does not exceed phi_II, or
        if the derived arguments are degenerate.
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


def _oracle_fixed_point_operators(phi_I: float, phi_II: float, chi: float, gamma: float,
                         r_eff: float) -> np.ndarray:
    """Reference implementation."""
    pI = _fraction("phi_I", phi_I)
    pII = _fraction("phi_II", phi_II)
    c = _real_scalar("chi", chi)
    g = _real_scalar("gamma", gamma)
    re = _positive("r_eff", r_eff)
    if c < 0.0:
        raise ValueError("chi must be non-negative")
    if not (pI > pII):
        raise ValueError("phi_I must be strictly greater than phi_II")
    x, y = _xy(pI, pII, c, g)
    a = y / 2.0
    b = re * (x - y) / 2.0
    return np.array([x, y, a, b, _H(a, b)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # second-order pair of the reference system
            "setup": 'import numpy as np\npI, pII = 0.8486728199269923, 0.002734984774850931\nchi, gamma, r_eff = 2.7350787009788555, 0.6287878787878788, 2.693877551020408\n',
            "call": 'fixed_point_operators(pI, pII, chi, gamma, r_eff)',
            "gold_call": '_oracle_fixed_point_operators(pI, pII, chi, gamma, r_eff)',
        },
        {
            # first-order pair of the reference system
            "setup": 'import numpy as np\npI, pII = 0.9583406121185476, 4.1236366875509676e-08\nchi, gamma, r_eff = 2.7350787009788555, 0.6287878787878788, 2.693877551020408\n',
            "call": 'fixed_point_operators(pI, pII, chi, gamma, r_eff)',
            "gold_call": '_oracle_fixed_point_operators(pI, pII, chi, gamma, r_eff)',
        },
        {
            # monomeric case, length-asymmetry coefficient exactly zero
            "setup": 'import numpy as np\npI, pII = 0.7039443662951035, 0.07158084343152542\nchi, gamma, r_eff = 4.0, 0.0, 1.0\n',
            "call": 'fixed_point_operators(pI, pII, chi, gamma, r_eff)',
            "gold_call": '_oracle_fixed_point_operators(pI, pII, chi, gamma, r_eff)',
        },
        {
            # narrow gap just above the critical point
            "setup": 'import numpy as np\npI, pII = 0.21, 0.19\nchi, gamma, r_eff = 1.9, 0.6287878787878788, 2.693877551020408\n',
            "call": 'fixed_point_operators(pI, pII, chi, gamma, r_eff)',
            "gold_call": '_oracle_fixed_point_operators(pI, pII, chi, gamma, r_eff)',
        },
        {
            # uncharged limit leaves only the length-asymmetry contribution
            "setup": 'import numpy as np\npI, pII = 0.6, 0.05\nchi, gamma, r_eff = 0.0, 0.5, 2.0\n',
            "call": 'fixed_point_operators(pI, pII, chi, gamma, r_eff)',
            "gold_call": '_oracle_fixed_point_operators(pI, pII, chi, gamma, r_eff)',
        },
        {
            # rejects negative dilute fraction, as produced by the near-critical seed
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        fixed_point_operators(0.7226368214303909, -1.3366449467411936, 2.7350787009788555, 0.6287878787878788, 2.693877551020408)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_fixed_point_operators(0.7226368214303909, -1.3366449467411936, 2.7350787009788555, 0.6287878787878788, 2.693877551020408)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects equal fractions
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        fixed_point_operators(0.3, 0.3, 2.7, 0.6, 2.7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_fixed_point_operators(0.3, 0.3, 2.7, 0.6, 2.7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects reversed ordering
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        fixed_point_operators(0.01, 0.8, 2.7, 0.6, 2.7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_fixed_point_operators(0.01, 0.8, 2.7, 0.6, 2.7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects dilute fraction at zero
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        fixed_point_operators(0.8, 0.0, 2.7, 0.6, 2.7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_fixed_point_operators(0.8, 0.0, 2.7, 0.6, 2.7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects dense fraction at one
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        fixed_point_operators(1.0, 0.01, 2.7, 0.6, 2.7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_fixed_point_operators(1.0, 0.01, 2.7, 0.6, 2.7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
