"""
Near-critical expansion and its estimate of the coexisting fractions.

The coexistence conditions have no closed-form solution, so the analysis starts from an approximation valid close to the critical point and then improves it. The starting approximation comes from expanding the exchange chemical potential in powers of the deviation of the volume fraction from its critical value, keeping terms through the cubic order.

Imposing equality of the chemical potential in the two phases and cancelling the trivial solution in which both fractions coincide leaves a quadratic condition on the deviation. Its two roots are the near-critical estimates of the dense and dilute fractions, and the distance between them is the width of the coexistence gap implied by the expansion.

This estimate is only reliable near the critical point. Further away it overshoots, and the dilute root routinely falls below zero, which is unphysical but still usable as a starting point. Where the quadratic condition has no real root at all the expansion has left its range of validity entirely, and that situation must be reported rather than papered over.

Return the linear, quadratic and cubic expansion coefficients, the gap width, and the two estimated fractions with the dense one first.

Returns
-------
np.ndarray of shape (6,), dtype float: [linear coefficient, quadratic coefficient, cubic coefficient, gap width, dense fraction estimate, dilute fraction estimate]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def landau_seed(phi_c: float, chi: float, r_eff: float) -> np.ndarray:
    """Near-critical expansion coefficients and the seed pair of fractions.

    Parameters
    ----------
    phi_c : float
        Critical total polymer volume fraction, strictly inside (0, 1).
    chi : float
        Dimensionless interaction parameter, >= 0.
    r_eff : float
        Effective chain length, > 0.

    Returns
    -------
    out : np.ndarray
        Six floats, in order: the linear, quadratic and cubic coefficients of the
        expansion of the exchange chemical potential about phi_c; the width of the
        coexistence gap the expansion implies; and the estimated dense and dilute
        volume fractions, dense first. The dilute estimate may be negative.

    Raises
    ------
    ValueError
        If the arguments are out of range, or if the expansion admits no real
        coexistence gap at this interaction parameter.
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


def _landau(phic, chi, r_eff):
    A = 1.0 / (1.0 - phic) + 1.0 / (r_eff * phic) - 0.75 * chi / np.sqrt(phic)
    B = (1.0 / (2.0 * (1.0 - phic) ** 2) - 1.0 / (2.0 * r_eff * phic ** 2)
         + (3.0 / 16.0) * chi / phic ** 1.5)
    C = (1.0 / (3.0 * (1.0 - phic) ** 3) + 1.0 / (3.0 * r_eff * phic ** 3)
         - (3.0 / 32.0) * chi / phic ** 2.5)
    return A, B, C


def _seed(phic, chi, r_eff):
    A, B, C = _landau(phic, chi, r_eff)
    if C == 0.0:
        raise ValueError("cubic coefficient vanishes; no near-critical estimate")
    disc = B * B - 4.0 * A * C
    if not np.isfinite(disc) or disc < 0.0:
        raise ValueError("no real coexistence gap: the near-critical expansion "
                         "is outside its range of validity at this chi")
    root = np.sqrt(disc)
    D = root / C
    phi_I = phic + (-B + root) / (2.0 * C)
    phi_II = phic + (-B - root) / (2.0 * C)
    return A, B, C, D, phi_I, phi_II


def _oracle_landau_seed(phi_c: float, chi: float, r_eff: float) -> np.ndarray:
    """Reference implementation."""
    pc = _fraction("phi_c", phi_c)
    c = _real_scalar("chi", chi)
    re = _positive("r_eff", r_eff)
    if c < 0.0:
        raise ValueError("chi must be non-negative")
    A, Bc, Cc, D, phi_I, phi_II = _seed(pc, c, re)
    return np.array([A, Bc, Cc, D, phi_I, phi_II], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # reference system
            "setup": 'import numpy as np\nphi_c, chi, r_eff = 0.1987837280267722, 2.7350787009788555, 2.693877551020408\n',
            "call": 'landau_seed(phi_c, chi, r_eff)',
            "gold_call": '_oracle_landau_seed(phi_c, chi, r_eff)',
        },
        {
            # just above the critical interaction parameter the gap is narrow
            "setup": 'import numpy as np\nphi_c, chi, r_eff = 0.1987837280267722, 1.8522647142314626, 2.693877551020408\n',
            "call": 'landau_seed(phi_c, chi, r_eff)',
            "gold_call": '_oracle_landau_seed(phi_c, chi, r_eff)',
        },
        {
            # monomeric case just above threshold
            "setup": 'import numpy as np\nphi_c, chi, r_eff = 1.0 / 3.0, 3.7, 1.0\n',
            "call": 'landau_seed(phi_c, chi, r_eff)',
            "gold_call": '_oracle_landau_seed(phi_c, chi, r_eff)',
        },
        {
            # monomeric case further from threshold, dilute root goes negative
            "setup": 'import numpy as np\nphi_c, chi, r_eff = 1.0 / 3.0, 5.5, 1.0\n',
            "call": 'landau_seed(phi_c, chi, r_eff)',
            "gold_call": '_oracle_landau_seed(phi_c, chi, r_eff)',
        },
        {
            # long chains
            "setup": 'import numpy as np\nphi_c, chi, r_eff = 0.0097123, 0.3, 100.0\n',
            "call": 'landau_seed(phi_c, chi, r_eff)',
            "gold_call": '_oracle_landau_seed(phi_c, chi, r_eff)',
        },
        {
            # rejects interaction parameter beyond the validity window of the expansion
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        landau_seed(1.0 / 3.0, 9.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_landau_seed(1.0 / 3.0, 9.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects far beyond the window
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        landau_seed(1.0 / 3.0, 40.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_landau_seed(1.0 / 3.0, 40.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects critical fraction at zero
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        landau_seed(0.0, 3.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_landau_seed(0.0, 3.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects critical fraction at one
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        landau_seed(1.0, 3.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_landau_seed(1.0, 3.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects negative chi
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        landau_seed(1.0 / 3.0, -1.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_landau_seed(1.0 / 3.0, -1.0, 1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
