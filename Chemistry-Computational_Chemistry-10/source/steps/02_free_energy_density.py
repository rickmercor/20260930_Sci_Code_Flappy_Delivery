"""
One-component free-energy density.

After the electroneutrality reduction the model has a free-energy density that depends only on the total polymer volume fraction phi. It carries three contributions: the translational entropy of the polymer, controlled by the effective chain length; the translational entropy of the solvent, which occupies the remaining volume; and an attractive electrostatic term.

The electrostatic contribution is the Debye-Huckel result evaluated in the dilute-electrolyte limit, so it scales as a non-integer power of the volume fraction rather than quadratically as a Flory-Huggins contact term would. That non-integer power is the signature of correlated screening and is what makes the coexistence problem resist a closed-form solution.

Return the free-energy density in units of thermal energy per reference volume. Terms that are constant in phi and terms that are linear in phi are dropped, since both cancel from the conditions that fix phase coexistence.

Returns
-------
float, the free-energy density in units of thermal energy per reference volume, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def free_energy_density(phi: float, r_eff: float, chi: float) -> float:
    """Free-energy density of the reduced one-component model.

    Parameters
    ----------
    phi : float
        Total polymer volume fraction, strictly inside (0, 1).
    r_eff : float
        Effective chain length, > 0.
    chi : float
        Dimensionless interaction parameter, >= 0.

    Returns
    -------
    f : float
        Free-energy density in units of thermal energy per reference volume, with
        phi-constant and phi-linear terms omitted.

    Raises
    ------
    ValueError
        If phi is outside (0, 1), r_eff is not positive, or chi is negative.
    """
    return f  # placeholder

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


def _oracle_free_energy_density(phi: float, r_eff: float, chi: float) -> float:
    """Reference implementation."""
    p = _fraction("phi", phi)
    re = _positive("r_eff", r_eff)
    c = _real_scalar("chi", chi)
    if c < 0.0:
        raise ValueError("chi must be non-negative")
    return float(p / re * np.log(p) + (1.0 - p) * np.log(1.0 - p) - c * p ** 1.5)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # dense branch of the reference system
            "setup": 'import numpy as np\nphi, r_eff, chi = 0.8486728199269923, 2.693877551020408, 2.7350787009788555\n',
            "call": 'free_energy_density(phi, r_eff, chi)',
            "gold_call": '_oracle_free_energy_density(phi, r_eff, chi)',
        },
        {
            # dilute branch of the reference system
            "setup": 'import numpy as np\nphi, r_eff, chi = 0.002734984774850931, 2.693877551020408, 2.7350787009788555\n',
            "call": 'free_energy_density(phi, r_eff, chi)',
            "gold_call": '_oracle_free_energy_density(phi, r_eff, chi)',
        },
        {
            # critical fraction
            "setup": 'import numpy as np\nphi, r_eff, chi = 0.1987837280267722, 2.693877551020408, 1.8520795062808344\n',
            "call": 'free_energy_density(phi, r_eff, chi)',
            "gold_call": '_oracle_free_energy_density(phi, r_eff, chi)',
        },
        {
            # uncharged limit switches off the electrostatic term
            "setup": 'import numpy as np\nphi, r_eff, chi = 0.35, 1.0, 0.0\n',
            "call": 'free_energy_density(phi, r_eff, chi)',
            "gold_call": '_oracle_free_energy_density(phi, r_eff, chi)',
        },
        {
            # very dilute
            "setup": 'import numpy as np\nphi, r_eff, chi = 1e-9, 4.0, 3.0\n',
            "call": 'free_energy_density(phi, r_eff, chi)',
            "gold_call": '_oracle_free_energy_density(phi, r_eff, chi)',
        },
        {
            # close to full packing
            "setup": 'import numpy as np\nphi, r_eff, chi = 1.0 - 1e-9, 2.0, 2.5\n',
            "call": 'free_energy_density(phi, r_eff, chi)',
            "gold_call": '_oracle_free_energy_density(phi, r_eff, chi)',
        },
        {
            # rejects phi at zero
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        free_energy_density(0.0, 2.0, 2.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_free_energy_density(0.0, 2.0, 2.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects phi at one
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        free_energy_density(1.0, 2.0, 2.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_free_energy_density(1.0, 2.0, 2.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects phi above one
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        free_energy_density(1.4, 2.0, 2.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_free_energy_density(1.4, 2.0, 2.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects negative phi
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        free_energy_density(-0.2, 2.0, 2.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_free_energy_density(-0.2, 2.0, 2.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects negative chi
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        free_energy_density(0.3, 2.0, -1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_free_energy_density(0.3, 2.0, -1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects zero effective length
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        free_energy_density(0.3, 0.0, 2.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_free_energy_density(0.3, 0.0, 2.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
