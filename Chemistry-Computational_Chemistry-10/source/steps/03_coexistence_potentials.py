"""
Exchange chemical potential and osmotic pressure.

Two coexisting phases in this model are fixed by requiring that both the exchange chemical potential and the osmotic pressure take the same value in each phase. The chemical potential is the derivative of the free-energy density with respect to the total polymer volume fraction; the osmotic pressure is the corresponding Legendre combination that subtracts the free energy from the fraction times its derivative.

Both quantities follow the convention of the free-energy density, with its phi-constant and phi-linear terms dropped, but that convention does not act on them in the same way. The osmotic pressure is unchanged by it. The chemical potential is reported with its own phi-constant term dropped as well, so it equals the derivative of the free-energy density of the previous step plus the length-asymmetry coefficient gamma = 1 - 1/r_eff, and the pair satisfies Pi = phi (mu - gamma) - f rather than the plain Legendre relation. The shift is a constant, so it cancels from the equality of chemical potentials and neither coexistence condition is affected.

Return both quantities in units of thermal energy, the pressure per reference volume.

Returns
-------
np.ndarray of shape (2,), dtype float: [exchange chemical potential, osmotic pressure]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coexistence_potentials(phi: float, r_eff: float, chi: float) -> np.ndarray:
    """Exchange chemical potential and osmotic pressure of the reduced model.

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
    out : np.ndarray
        Two floats: the exchange chemical potential in units of thermal energy, and
        the osmotic pressure in units of thermal energy per reference volume. The
        chemical potential is shifted by +gamma = 1 - 1/r_eff relative to the
        derivative of the free-energy density returned by free_energy_density (its
        own phi-constant term is dropped too); the pressure carries no such shift.

    Raises
    ------
    ValueError
        If phi is outside (0, 1), r_eff is not positive, or chi is negative.
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


def _oracle_coexistence_potentials(phi: float, r_eff: float, chi: float) -> np.ndarray:
    """Reference implementation."""
    p = _fraction("phi", phi)
    re = _positive("r_eff", r_eff)
    c = _real_scalar("chi", chi)
    if c < 0.0:
        raise ValueError("chi must be non-negative")
    mu = np.log(p) / re - np.log(1.0 - p) - 1.5 * c * np.sqrt(p)
    Pi = -(1.0 - 1.0 / re) * p - np.log(1.0 - p) - 0.5 * c * p ** 1.5
    return np.array([mu, Pi], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # dense branch of the reference system
            "setup": 'import numpy as np\nphi, r_eff, chi = 0.8486728199269923, 2.693877551020408, 2.7350787009788555\n',
            "call": 'coexistence_potentials(phi, r_eff, chi)',
            "gold_call": '_oracle_coexistence_potentials(phi, r_eff, chi)',
        },
        {
            # dilute branch of the reference system
            "setup": 'import numpy as np\nphi, r_eff, chi = 0.002734984774850931, 2.693877551020408, 2.7350787009788555\n',
            "call": 'coexistence_potentials(phi, r_eff, chi)',
            "gold_call": '_oracle_coexistence_potentials(phi, r_eff, chi)',
        },
        {
            # monomeric limit where the length-asymmetry coefficient vanishes
            "setup": 'import numpy as np\nphi, r_eff, chi = 0.4, 1.0, 4.0\n',
            "call": 'coexistence_potentials(phi, r_eff, chi)',
            "gold_call": '_oracle_coexistence_potentials(phi, r_eff, chi)',
        },
        {
            # uncharged limit
            "setup": 'import numpy as np\nphi, r_eff, chi = 0.25, 3.0, 0.0\n',
            "call": 'coexistence_potentials(phi, r_eff, chi)',
            "gold_call": '_oracle_coexistence_potentials(phi, r_eff, chi)',
        },
        {
            # very dilute
            "setup": 'import numpy as np\nphi, r_eff, chi = 1e-8, 2.693877551020408, 2.7350787009788555\n',
            "call": 'coexistence_potentials(phi, r_eff, chi)',
            "gold_call": '_oracle_coexistence_potentials(phi, r_eff, chi)',
        },
        {
            # rejects phi at zero
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        coexistence_potentials(0.0, 2.0, 2.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_coexistence_potentials(0.0, 2.0, 2.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects phi at one
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        coexistence_potentials(1.0, 2.0, 2.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_coexistence_potentials(1.0, 2.0, 2.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects negative chi
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        coexistence_potentials(0.3, 2.0, -0.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_coexistence_potentials(0.3, 2.0, -0.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects negative effective length
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        coexistence_potentials(0.3, -2.0, 2.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_coexistence_potentials(0.3, -2.0, 2.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
