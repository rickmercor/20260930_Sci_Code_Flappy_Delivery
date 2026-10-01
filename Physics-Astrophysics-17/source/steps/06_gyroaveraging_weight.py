"""
Gyro-averaging of the energy change a fluctuation can impart.

Gyro-averaging of the energy change a fluctuation can impart.



A proton whose gyroradius is comparable to or larger than the perpendicular scale of a

fluctuation samples the fluctuation at many phases during one gyration, which reduces the

change of perpendicular energy it receives. This step returns the weight that multiplies

the mean-square perpendicular-energy change imparted by the electrostatic perpendicular

electric field of step 04 at perpendicular wavenumber k_perp to a proton of gyroradius

rho, as a function of k_perp rho, normalised so that it tends to 1/4 as k_perp rho tends

to zero.



Returns

-------

The dimensionless gyro-averaging weight at each k_perp rho.

Returns
-------
The dimensionless gyro-averaging weight at each k_perp rho.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gyroaveraging_weight(k_rho: "ArrayLike") -> "np.ndarray":
    """Return the gyro-averaging weight of the mean-square perpendicular-energy change.

    Parameters
    ----------
    k_rho : float or array_like
        Perpendicular wavenumber times the proton gyroradius (dimensionless). Non-empty,
        every entry finite and strictly positive.

    Returns
    -------
    numpy.ndarray
        Array with the shape of ``k_rho`` holding the weight by which gyrophase averaging
        multiplies the mean-square perpendicular-energy change produced by the electrostatic
        perpendicular electric field of step 04 at that k_perp rho; it tends to 1/4 as
        k_rho -> 0.

    Raises
    ------
    ValueError
        If ``k_rho`` is empty or has a non-finite or non-positive entry.
    """
    return weight  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import j1


def _oracle_gyroaveraging_weight(k_rho: "ArrayLike") -> "np.ndarray":
    x = _kmb_array("k_rho", k_rho)
    return (j1(x) / x) ** 2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic normal, boundary, edge and invalid-input cases."""
    imports = "import numpy as np\nimport math\nfrom scipy.special import i0e, j1\n"
    probe = (
        imports
        + "def _probe_call():\n"
        "    try:\n"
        "        gyroaveraging_weight(np.array([1.0, np.nan]))\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
        "def _probe_gold():\n"
        "    try:\n"
        "        _oracle_gyroaveraging_weight(np.array([1.0, np.nan]))\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
    )
    return [
        # the long-wavelength plateau down through the proton gyroscale
        {
            "setup": imports + "x = np.array([1.0e-3, 0.3, 1.0, 1.494035, 2.5])\n",
            "call": "gyroaveraging_weight(x.copy())",
            "gold_call": "_oracle_gyroaveraging_weight(x.copy())",
        },
        # around the first zero of the weight, where it vanishes and turns back up
        {
            "setup": imports + "x = np.array([[3.6, 3.8317], [4.0, 5.1356]])\n",
            "call": "gyroaveraging_weight(x.copy())",
            "gold_call": "_oracle_gyroaveraging_weight(x.copy())",
        },
        # far beyond the gyroscale, on the decaying oscillating tail
        {
            "setup": imports + "x = np.array([7.0156, 12.0, 30.0])\n",
            "call": "gyroaveraging_weight(x.copy())",
            "gold_call": "_oracle_gyroaveraging_weight(x.copy())",
        },
        # a non-finite entry is rejected
        {
            "setup": probe,
            "call": "_probe_call()",
            "gold_call": "_probe_gold()",
        },
    ]
