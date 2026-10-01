"""
Electric potential difference between the coexisting phases.

Phase separation of two oppositely charged polymers leaves a measurable electrostatic potential step between the dense and the dilute phase. It arises because the two species partition unequally whenever they differ in chain length or in net charge, and its sign decides whether cationic or anionic client molecules are drawn into the dense phase.

Equating the electrochemical potential of each species across the interface and eliminating the volume fractions of the individual species gives the potential difference as a sum of two contributions. One is proportional to the difference in the net charges carried by the two chains and involves the difference of the square roots of the two volume fractions; the other is proportional to the difference in chain lengths and involves the logarithm of the ratio of the two solvent fractions. Both are normalised by the total net charge of the pair.

Consequently the potential difference vanishes identically for a symmetric pair, meaning equal chain lengths and equal net charges; and the first contribution alone disappears when the two chains happen to carry equal net charge even though their lengths differ.

Species 1 is the positively charged polymer and species 2 the negatively charged one. Return the potential difference in units of thermal energy per elementary charge, taken as the dense phase minus the dilute phase.

Returns
-------
float, the potential difference in units of thermal energy per elementary charge, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def interphase_potential(phi_I: float, phi_II: float, r1: float, r2: float,
                        sigma1: float, sigma2: float, alpha: float = 3.655) -> float:
    """Electrostatic potential difference between coexisting coacervate phases.

    Parameters
    ----------
    phi_I : float
        Dense-phase total polymer volume fraction, strictly inside (0, 1).
    phi_II : float
        Dilute-phase total polymer volume fraction, strictly inside (0, 1) and below phi_I.
    r1, r2 : float
        Chain lengths of species 1 and 2, both > 0.
    sigma1, sigma2 : float
        Linear charge densities of species 1 and 2, both > 0.
    alpha : float
        Dimensionless electrostatic prefactor of the model.

    Returns
    -------
    dpsi : float
        Potential difference, dense phase minus dilute phase, in units of thermal
        energy per elementary charge. Zero for a symmetric pair.

    Raises
    ------
    ValueError
        If either fraction lies outside (0, 1), if phi_I does not exceed phi_II, or if
        any other argument is not a finite, strictly positive real scalar.
    """
    return dpsi  # placeholder

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


def _xi_zeta(r1, r2, u, alpha):
    a23 = alpha ** (2.0 / 3.0)
    xi = (u * r1 - r2) / (u * r1 + r2) * a23
    zeta = u * (r1 - r2) / ((1.0 + u) * (u * r1 + r2)) * a23
    return xi, zeta


def _oracle_interphase_potential(phi_I: float, phi_II: float, r1: float, r2: float,
                        sigma1: float, sigma2: float, alpha: float = 3.655) -> float:
    """Reference implementation."""
    pI = _fraction("phi_I", phi_I)
    pII = _fraction("phi_II", phi_II)
    if not (pI > pII):
        raise ValueError("phi_I must be strictly greater than phi_II")
    u, S, chi, r_eff, gamma = _reduce(r1, r2, sigma1, sigma2, alpha)
    xi, zeta = _xi_zeta(float(r1), float(r2), u, float(alpha))
    return float(1.5 * xi * chi ** (1.0 / 3.0) * (np.sqrt(pI) - np.sqrt(pII))
                 + 2.0 * zeta * chi ** (-2.0 / 3.0) * np.log((1.0 - pI) / (1.0 - pII)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # second-order pair of the reference system
            "setup": 'import numpy as np\npI, pII = 0.8486728199269923, 0.002734984774850931\nr1, r2, s1, s2 = 4.0, 2.0, 0.80, 0.85\n',
            "call": 'interphase_potential(pI, pII, r1, r2, s1, s2, 3.655)',
            "gold_call": '_oracle_interphase_potential(pI, pII, r1, r2, s1, s2, 3.655)',
        },
        {
            # symmetric pair gives exactly zero
            "setup": 'import numpy as np\npI, pII = 0.8, 0.01\nr1, r2, s1, s2 = 2.0, 2.0, 0.9, 0.9\n',
            "call": 'interphase_potential(pI, pII, r1, r2, s1, s2, 3.655)',
            "gold_call": '_oracle_interphase_potential(pI, pII, r1, r2, s1, s2, 3.655)',
        },
        {
            # equal net charge on unequal chains removes the charge contribution
            "setup": 'import numpy as np\npI, pII = 0.8, 0.01\nr1, r2, s1, s2 = 1.5, 2.5, 1.25, 0.75\n',
            "call": 'interphase_potential(pI, pII, r1, r2, s1, s2, 3.655)',
            "gold_call": '_oracle_interphase_potential(pI, pII, r1, r2, s1, s2, 3.655)',
        },
        {
            # equal chain lengths on unequal charges removes the length contribution
            "setup": 'import numpy as np\npI, pII = 0.8, 0.01\nr1, r2, s1, s2 = 2.0, 2.0, 1.2, 0.6\n',
            "call": 'interphase_potential(pI, pII, r1, r2, s1, s2, 3.655)',
            "gold_call": '_oracle_interphase_potential(pI, pII, r1, r2, s1, s2, 3.655)',
        },
        {
            # sign reverses when the two species are swapped
            "setup": 'import numpy as np\npI, pII = 0.8486728199269923, 0.002734984774850931\nr1, r2, s1, s2 = 2.0, 4.0, 0.85, 0.80\n',
            "call": 'interphase_potential(pI, pII, r1, r2, s1, s2, 3.655)',
            "gold_call": '_oracle_interphase_potential(pI, pII, r1, r2, s1, s2, 3.655)',
        },
        {
            # first-order pair of the reference system
            "setup": 'import numpy as np\npI, pII = 0.9583406121185476, 4.1236366875509676e-08\nr1, r2, s1, s2 = 4.0, 2.0, 0.80, 0.85\n',
            "call": 'interphase_potential(pI, pII, r1, r2, s1, s2, 3.655)',
            "gold_call": '_oracle_interphase_potential(pI, pII, r1, r2, s1, s2, 3.655)',
        },
        {
            # rejects reversed ordering
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        interphase_potential(0.01, 0.8, 4.0, 2.0, 0.8, 0.85)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_interphase_potential(0.01, 0.8, 4.0, 2.0, 0.8, 0.85)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects equal fractions
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        interphase_potential(0.4, 0.4, 4.0, 2.0, 0.8, 0.85)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_interphase_potential(0.4, 0.4, 4.0, 2.0, 0.8, 0.85)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects negative dilute fraction
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        interphase_potential(0.8, -0.1, 4.0, 2.0, 0.8, 0.85)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_interphase_potential(0.8, -0.1, 4.0, 2.0, 0.8, 0.85)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects zero charge density
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        interphase_potential(0.8, 0.01, 4.0, 2.0, 0.8, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_interphase_potential(0.8, 0.01, 4.0, 2.0, 0.8, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            # rejects dense fraction at one
            "setup": 'import numpy as np\ndef run_model():\n    try:\n        interphase_potential(1.0, 0.01, 4.0, 2.0, 0.8, 0.85)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_interphase_potential(1.0, 0.01, 4.0, 2.0, 0.8, 0.85)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n',
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
