"""
Free electron density in the neutral on-state drift layer under incomplete donor ionisation.

Nitrogen in 4H-SiC occupies two inequivalent lattice sites in equal numbers, one shallow and one roughly twice as deep, and at room temperature neither is fully ionised. Inside the depleted blocking layer the donors are emptied, so the space charge there is the full chemical doping. The neutral layer that carries current in the on state is different: it holds fewer free electrons than there are donors, and the shortfall grows quickly with doping. Charge neutrality summed over both sites closes on the free electron density and has to be solved rather than assumed. Energies are in electronvolts and densities in reciprocal cubic centimetres.

Returns
-------
float, the free electron density in cm^-3
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def free_electron_density(doping: float, n_c: float, degeneracy: float, ea_shallow: float, ea_deep: float, kt: float) -> float:
    '''Free electron density under incomplete ionisation of two donor sites.

    Parameters
    ----------
    doping : float
        Total chemical donor density in cm^-3.
    n_c : float
        Conduction band effective density of states in cm^-3.
    degeneracy : float
        Donor degeneracy factor.
    ea_shallow, ea_deep : float
        Activation energies of the two donor sites in eV.
    kt : float
        Thermal energy in eV.

    Returns
    -------
    float
        Free electron density in cm^-3.

    Raises
    ------
    ValueError
        If any of doping, n_c, degeneracy, kt is not finite or is not strictly positive.
        Activation energies must be finite and non-negative.
    '''
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _oracle_free_electron_density(doping: float, n_c: float, degeneracy: float, ea_shallow: float, ea_deep: float, kt: float) -> float:
    for _v in (doping, n_c, degeneracy, kt):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('doping, n_c, degeneracy, kt must be finite and strictly positive')
    for _v in (ea_shallow, ea_deep):
        if not np.isfinite(_v) or _v < 0.0:
            raise ValueError('activation energies must be finite and non-negative')
    bs = (degeneracy/n_c)*np.exp(ea_shallow/kt)
    bd = (degeneracy/n_c)*np.exp(ea_deep/kt)
    f = lambda n: doping*(0.5/(1.0 + bs*n) + 0.5/(1.0 + bd*n)) - n
    return float(brentq(f, 1.0e5, doping, xtol=1e-8, rtol=8.9e-16, maxiter=300))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": "free_electron_density(3.170246194e16, 1.7e19, 2.0, 0.060, 0.120, 0.025852)",
            "gold_call": "_oracle_free_electron_density(3.170246194e16, 1.7e19, 2.0, 0.060, 0.120, 0.025852)",
        },
        {
            "setup": "import numpy as np",
            "call": "free_electron_density(1.0e15, 1.7e19, 2.0, 0.060, 0.120, 0.025852)",
            "gold_call": "_oracle_free_electron_density(1.0e15, 1.7e19, 2.0, 0.060, 0.120, 0.025852)",
        },
        {
            "setup": "import numpy as np",
            "call": "free_electron_density(1.0e17, 1.7e19, 2.0, 0.060, 0.120, 0.025852)",
            "gold_call": "_oracle_free_electron_density(1.0e17, 1.7e19, 2.0, 0.060, 0.120, 0.025852)",
        },
        {
            "setup": "import numpy as np",
            "call": "free_electron_density(1.0e18, 1.7e19, 2.0, 0.060, 0.120, 0.025852)",
            "gold_call": "_oracle_free_electron_density(1.0e18, 1.7e19, 2.0, 0.060, 0.120, 0.025852)",
        },
        {
            "setup": "import numpy as np",
            "call": "free_electron_density(4.442828e15, 1.7e19, 2.0, 0.060, 0.060, 0.025852)",
            "gold_call": "_oracle_free_electron_density(4.442828e15, 1.7e19, 2.0, 0.060, 0.060, 0.025852)",
        },
        {
            "setup": "import numpy as np",
            "call": "free_electron_density(3.0e16, 1.7e19, 2.0, 0.045, 0.100, 0.025852)",
            "gold_call": "_oracle_free_electron_density(3.0e16, 1.7e19, 2.0, 0.045, 0.100, 0.025852)",
        },
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        free_electron_density(3.17e16, 1.7e19, 2, -0.06, 0.12, 0.025852)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_free_electron_density(3.17e16, 1.7e19, 2, -0.06, 0.12, 0.025852)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
