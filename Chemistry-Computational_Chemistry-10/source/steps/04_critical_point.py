"""
Critical fraction and critical interaction parameter.

Phase separation in this model sets in only once the interaction parameter exceeds a threshold. The threshold and the volume fraction at which the two branches emerge are found from the stationarity conditions that make the second and third derivatives of the free-energy density with respect to the volume fraction vanish simultaneously.

Solving those two conditions gives the critical volume fraction as a closed-form function of the effective chain length alone, and the critical interaction parameter as a closed-form function of that critical fraction. For the monomeric case, where the effective chain length is one, the critical fraction takes the value one third.

Both quantities anchor everything downstream: the near-critical expansion is built around this point, and the distance of the interaction parameter above its critical value sets how wide the coexistence gap is.

Returns
-------
np.ndarray of shape (2,), dtype float: [critical volume fraction, critical interaction parameter]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def critical_point(r_eff: float) -> np.ndarray:
    """Critical volume fraction and critical interaction parameter.

    Parameters
    ----------
    r_eff : float
        Effective chain length, > 0.

    Returns
    -------
    out : np.ndarray
        Two floats: the critical total polymer volume fraction, and the value of the
        dimensionless interaction parameter at which phase separation begins.

    Raises
    ------
    ValueError
        If r_eff is not a finite, strictly positive real scalar.
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


def _critical(r_eff):
    re = _positive("r_eff", r_eff)
    phic = 2.0 / (2.0 + re + np.sqrt(re * (re + 8.0)))
    chic = (8.0 / 3.0) * np.sqrt(phic) / (1.0 - phic) ** 2
    return phic, chic


def _oracle_critical_point(r_eff: float) -> np.ndarray:
    """Reference implementation."""
    phic, chic = _critical(r_eff)
    return np.array([phic, chic], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # reference system
            "setup": 'import numpy as np\nr_eff = 2.693877551020408\n',
            "call": 'critical_point(r_eff)',
            "gold_call": '_oracle_critical_point(r_eff)',
        },
        {
            # monomeric case, critical fraction is exactly one third
            "setup": 'import numpy as np\nr_eff = 1.0\n',
            "call": 'critical_point(r_eff)',
            "gold_call": '_oracle_critical_point(r_eff)',
        },
        {
            # long chains push the critical point to low fraction
            "setup": 'import numpy as np\nr_eff = 250.0\n',
            "call": 'critical_point(r_eff)',
            "gold_call": '_oracle_critical_point(r_eff)',
        },
        {
            # short chains
            "setup": 'import numpy as np\nr_eff = 0.4\n',
            "call": 'critical_point(r_eff)',
            "gold_call": '_oracle_critical_point(r_eff)',
        },
        {
            # rejects zero effective length
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        critical_point(0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_critical_point(0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects negative effective length
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        critical_point(-3.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_critical_point(-3.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects non-finite
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        critical_point(np.nan)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_critical_point(np.nan)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects array argument
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        critical_point(np.array([2.0, 3.0]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_critical_point(np.array([2.0, 3.0]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
