"""
Return the mean extension in nm along the force of a rigid rod of the given length (nm) under a constant force f (pN) at thermal energy k_B T (pN nm): the rotational (Langevin) average L [coth(f L / k_B T) - k_B T / (f L)]; a rod of zero length has zero extension.

A duplex much shorter than its persistence length behaves as a rigid rod whose orientation fluctuates thermally; its projection on the pulling direction is the classical Langevin function of the ratio of the mechanical torque scale f L to the thermal energy.

Returns
-------
float, the rod extension in nm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rod_extension(length: float, force: float, k_bt: float) -> float:
    """Return the mean extension in nm along the force of a rigid rod of the given length (nm) under a constant force f (pN) at thermal energy k_B T (pN nm): the rotational (Langevin) average L [coth(f L / k_B T) - k_B T / (f L)]; a rod of zero length has zero extension.

    Parameters
    ----------
    length : float
        Rod length in nm (>= 0).
    force : float
        Force in pN (> 0).
    k_bt : float
        Thermal energy in pN nm (> 0).

    Returns
    -------
    extension : float
        Extension in nm.

    Raises
    ------
    ValueError
        If length is negative or not finite, or force or k_bt is not finite and positive.
    """
    return extension

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, curve_fit


def _check_pos(x, name):
    if isinstance(x, bool) or not np.isfinite(x) or float(x) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(x)


def _oracle_rod_extension(length: float, force: float, k_bt: float) -> float:
    """Mean extension (nm) of a rigid rod of the given length under force: L [coth(fL/kT) - kT/(fL)], eq. (5)."""
    if isinstance(length, bool) or not np.isfinite(length) or float(length) < 0.0:
        raise ValueError("length must be finite and nonnegative")
    f = _check_pos(force, "force")
    kt = _check_pos(k_bt, "k_bt")
    L = float(length)
    if L == 0.0:
        return 0.0
    x = f * L / kt
    return float(L * (1.0 / np.tanh(x) - 1.0 / x))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nlength, force, k_bt = 3.0, 5.0, 4.1\n",
            "call": "rod_extension(length, force, k_bt)",
            "gold_call": "_oracle_rod_extension(length, force, k_bt)",
        },
        {
            "setup": "import numpy as np\nlength, force, k_bt = 0.0, 5.0, 4.1\n",
            "call": "rod_extension(length, force, k_bt)",
            "gold_call": "_oracle_rod_extension(length, force, k_bt)",
        },
        {
            "setup": "import numpy as np\nlength, force, k_bt = 1.5, 0.01, 4.2\n",
            "call": "rod_extension(length, force, k_bt)",
            "gold_call": "_oracle_rod_extension(length, force, k_bt)",
        },
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        rod_extension(1.0, 0.0, 4.1)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_rod_extension(1.0, 0.0, 4.1)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
