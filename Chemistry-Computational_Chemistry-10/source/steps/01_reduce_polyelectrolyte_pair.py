"""
Reduce the two-species mixture to its one-component description.

Complex coacervation is modelled here as two oppositely charged polymer species, labelled 1 and 2, dispersed in a neutral solvent. Species 1 is the positively charged polymer and species 2 the negatively charged one. Species i is characterised by a chain length r_i, measured in units of the reference volume v used to define volume fractions, and by a linear charge density sigma_i. The mixture is incompressible, so the solvent volume fraction is whatever the two polymers leave over.

Local charge neutrality ties the two polymer volume fractions to one another, which means the free energy of the mixture can be rewritten as a function of the total polymer volume fraction alone. That reduction introduces four derived quantities: the ratio of the two charge densities sigma1/sigma2, a single charge-density parameter S defined by requiring that the total polymer charge density sigma1 phi1 + sigma2 phi2 equal S phi for a neutral mixture of total fraction phi, the dimensionless interaction parameter chi = alpha S^(3/2), and an effective chain length that absorbs both chain lengths and the charge-density ratio. A fifth quantity, the deviation of the reciprocal effective chain length from unity, appears throughout the coexistence analysis.

Note that the effective chain length becomes independent of the charge-density ratio when the two chains are equally long.

Returns
-------
np.ndarray of shape (5,), dtype float: [charge-density ratio, charge-density parameter, interaction parameter, effective chain length, length-asymmetry coefficient]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reduce_polyelectrolyte_pair(r1: float, r2: float, sigma1: float, sigma2: float,
                                alpha: float = 3.655) -> np.ndarray:
    """Collapse the two-species Voorn-Overbeek mixture to one component.

    Parameters
    ----------
    r1, r2 : float
        Chain lengths of species 1 and 2 in units of the reference volume, both > 0.
    sigma1, sigma2 : float
        Linear charge densities of species 1 and 2, both > 0.
    alpha : float
        Dimensionless electrostatic prefactor of the model.

    Returns
    -------
    out : np.ndarray
        Array of five floats, in order: the charge-density ratio of species 1 to
        species 2; the single charge-density parameter of the reduced model; the
        dimensionless interaction parameter; the effective chain length; and the
        length-asymmetry coefficient equal to one minus the reciprocal effective
        chain length.

    Raises
    ------
    ValueError
        If any argument is not a finite, strictly positive real scalar.
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


def _reduce(r1, r2, sigma1, sigma2, alpha):
    r1 = _positive("r1", r1)
    r2 = _positive("r2", r2)
    s1 = _positive("sigma1", sigma1)
    s2 = _positive("sigma2", sigma2)
    al = _positive("alpha", alpha)
    u = s1 / s2
    S = 2.0 * s1 * s2 / (s1 + s2)
    chi = al * S ** 1.5
    r_eff = r1 * r2 * (1.0 + u) / (r2 + u * r1)
    gamma = 1.0 - 1.0 / r_eff
    return u, S, chi, r_eff, gamma


def _oracle_reduce_polyelectrolyte_pair(r1: float, r2: float, sigma1: float, sigma2: float,
                                alpha: float = 3.655) -> np.ndarray:
    """Reference implementation."""
    u, S, chi, r_eff, gamma = _reduce(r1, r2, sigma1, sigma2, alpha)
    return np.array([u, S, chi, r_eff, gamma], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # reference system
            "setup": 'import numpy as np\nr1, r2, s1, s2 = 4.0, 2.0, 0.80, 0.85\n',
            "call": 'reduce_polyelectrolyte_pair(r1, r2, s1, s2, 3.655)',
            "gold_call": '_oracle_reduce_polyelectrolyte_pair(r1, r2, s1, s2, 3.655)',
        },
        {
            # equal charge densities collapse the ratio to one
            "setup": 'import numpy as np\nr1, r2, s1, s2 = 3.0, 1.5, 0.7, 0.7\n',
            "call": 'reduce_polyelectrolyte_pair(r1, r2, s1, s2, 3.655)',
            "gold_call": '_oracle_reduce_polyelectrolyte_pair(r1, r2, s1, s2, 3.655)',
        },
        {
            # equal chain lengths: effective length must not depend on the ratio
            "setup": 'import numpy as np\nr1, r2, s1, s2 = 2.5, 2.5, 1.4, 0.6\n',
            "call": 'reduce_polyelectrolyte_pair(r1, r2, s1, s2, 3.655)',
            "gold_call": '_oracle_reduce_polyelectrolyte_pair(r1, r2, s1, s2, 3.655)',
        },
        {
            # strongly asymmetric pair
            "setup": 'import numpy as np\nr1, r2, s1, s2 = 0.5, 6.0, 1.4, 0.6\n',
            "call": 'reduce_polyelectrolyte_pair(r1, r2, s1, s2, 3.655)',
            "gold_call": '_oracle_reduce_polyelectrolyte_pair(r1, r2, s1, s2, 3.655)',
        },
        {
            # non-default alpha
            "setup": 'import numpy as np\nr1, r2, s1, s2 = 4.0, 2.0, 0.80, 0.85\n',
            "call": 'reduce_polyelectrolyte_pair(r1, r2, s1, s2, 2.0)',
            "gold_call": '_oracle_reduce_polyelectrolyte_pair(r1, r2, s1, s2, 2.0)',
        },
        {
            # rejects zero charge density
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        reduce_polyelectrolyte_pair(4.0, 2.0, 0.8, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_reduce_polyelectrolyte_pair(4.0, 2.0, 0.8, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects negative chain length
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        reduce_polyelectrolyte_pair(-4.0, 2.0, 0.8, 0.85)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_reduce_polyelectrolyte_pair(-4.0, 2.0, 0.8, 0.85)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects zero alpha
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        reduce_polyelectrolyte_pair(4.0, 2.0, 0.8, 0.85, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_reduce_polyelectrolyte_pair(4.0, 2.0, 0.8, 0.85, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects array argument
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        reduce_polyelectrolyte_pair(np.array([4.0]), 2.0, 0.8, 0.85)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_reduce_polyelectrolyte_pair(np.array([4.0]), 2.0, 0.8, 0.85)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects non-finite argument
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        reduce_polyelectrolyte_pair(4.0, np.inf, 0.8, 0.85)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_reduce_polyelectrolyte_pair(4.0, np.inf, 0.8, 0.85)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
