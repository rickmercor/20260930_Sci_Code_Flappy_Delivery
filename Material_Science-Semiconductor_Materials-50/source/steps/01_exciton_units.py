"""
Effective-mass description of a Wannier-Mott exciton in a one-dimensional semiconductor.

The relative electron-hole motion along the chain axis has the kinetic energy

-(hbar^2 / 2 mu) d^2/dz^2 with the interband reduced mass 1 / mu = 1 / m_c + 1 / m_v.

With mu in units of the free-electron mass m, the natural excitonic units are the

exciton Rydberg and the exciton Bohr radius,

Effective-mass description of a Wannier-Mott exciton in a one-dimensional semiconductor.
The relative electron-hole motion along the chain axis has the kinetic energy
-(hbar^2 / 2 mu) d^2/dz^2 with the interband reduced mass 1 / mu = 1 / m_c + 1 / m_v.
With mu in units of the free-electron mass m, the natural excitonic units are the
exciton Rydberg and the exciton Bohr radius,

  R_exc = R_H mu / m,     a_exc = a_B m / mu,

so that R_exc = e^2 / (2 a_exc) = hbar^2 / (2 mu a_exc^2). For the bare one-dimensional
Coulomb potential -e^2 / |z| the ground state has binding energy R_exc and average
electron-hole separation 3 a_exc / 2. Constants: R_H = 13.605693 eV, a_B = 0.529177 A,
e^2 = 14.399645 eV A. A non-positive or non-finite mass is rejected.

Returns
-------
np.ndarray of float with shape (3,): [R_exc in eV, a_exc in Angstrom, 3 a_exc / 2 in Angstrom].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def exciton_units(mu: float) -> np.ndarray:
    '''Exciton Rydberg, exciton Bohr radius and bare 1D ground-state separation.

    Parameters
    ----------
    mu : float
        Interband reduced mass in units of the free-electron mass, > 0.

    Returns
    -------
    out : np.ndarray of float, shape (3,)
        [R_exc, a_exc, r_B0]: exciton Rydberg in eV, exciton Bohr radius in Angstrom,
        and the average separation 3 a_exc / 2 of the unscreened 1D ground state in Angstrom.
        Raises ValueError if mu is not a positive finite number.
    '''
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy import special as sp
from scipy import integrate, optimize


def _exciton_units(mu):
    mu = float(mu)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be a positive finite reduced mass")
    R_exc = 13.605693 * mu
    a_exc = 0.529177 / mu
    return np.array([R_exc, a_exc, 1.5 * a_exc])


def _oracle_exciton_units(mu: float) -> np.ndarray:
    return _exciton_units(mu)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np
from scipy import special as sp
from scipy import integrate, optimize


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
""",
            "call": 'np.round(exciton_units(0.39), 10)',
            "gold_call": 'np.round(_oracle_exciton_units(0.39), 10)',
        },
        {
            "setup": """import numpy as np
""",
            "call": 'np.round(exciton_units(1.0), 10)',
            "gold_call": 'np.round(_oracle_exciton_units(1.0), 10)',
        },
        {
            "setup": """import numpy as np
""",
            "call": 'np.round(exciton_units(0.11), 10)',
            "gold_call": 'np.round(_oracle_exciton_units(0.11), 10)',
        },
        {
            "setup": """import numpy as np
""",
            "call": 'np.round(exciton_units(1e-3), 10)',
            "gold_call": 'np.round(_oracle_exciton_units(1e-3), 10)',
        },
        {
            "setup": """import numpy as np


def run_model():
    try:
        exciton_units(0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2


def run_gold():
    try:
        _oracle_exciton_units(0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            "setup": """import numpy as np


def run_model():
    try:
        exciton_units(-0.3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2


def run_gold():
    try:
        _oracle_exciton_units(-0.3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
