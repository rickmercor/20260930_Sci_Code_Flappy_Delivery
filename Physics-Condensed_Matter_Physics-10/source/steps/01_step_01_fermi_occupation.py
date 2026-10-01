"""
Evaluate the Fermi-Dirac occupation for an array of quasiparticle energies at a given temperature. The routine takes energies in millielectronvolts and a temperature in kelvin and returns occupations in the closed interval from zero to one, with the exponent clipped so that large ratios saturate instead of overflowing. Invalid input raises ValueError: the temperature must be a finite non-negative scalar and the energies must all be finite.

Every thermal average in this calculation reduces to an integral weighted by this occupation, so its conventions propagate into the gap equation, the inertia coefficient and hence the final answer. Temperature enters only through the ratio of energy to the thermal scale, with Boltzmann's constant taken as 0.08617333262 millielectronvolts per kelvin so that energies and temperatures share one unit system. At exactly zero temperature the function is the step that gives one below the Fermi level, zero above it and one half at it, which is the limit the finite-temperature expression approaches but cannot evaluate directly because the ratio diverges.

Returns
-------
numpy.ndarray of Fermi-Dirac occupations in [0, 1], same shape as the input energies
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def fermi_occupation(energies, temperature):
    """Return the Fermi-Dirac occupation of the given quasiparticle energies.

    Parameters
    ----------
    energies : array_like
        Quasiparticle energies in millielectronvolts.
    temperature : float
        Temperature in kelvin. Zero selects the step-function limit.

    Returns
    -------
    numpy.ndarray
        Occupations in [0, 1] with the same shape as ``energies``.

    Raises
    ------
    ValueError
        If ``temperature`` is not a finite non-negative scalar, or if any entry
        of ``energies`` is not finite.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_fermi_occupation(energies, temperature):
    # A string passes np.isscalar but has no float value, and asking numpy
    # whether it is finite raises TypeError rather than the ValueError this
    # function documents. Convert first and turn every failure into that
    # documented error, for the temperature and for the energies alike.
    if not np.isscalar(temperature) or isinstance(temperature, (str, bytes)):
        raise ValueError("temperature must be a finite non-negative scalar")
    try:
        temperature = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("temperature must be a finite non-negative scalar") from exc
    if not np.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be a finite non-negative scalar")
    try:
        e = np.asarray(energies, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("energies must all be finite") from exc
    if not np.all(np.isfinite(e)):
        raise ValueError("energies must all be finite")
    if temperature == 0.0:
        return np.where(e < 0.0, 1.0, np.where(e > 0.0, 0.0, 0.5))
    kb_mev_per_k = 0.08617333262
    x = np.clip(e / (kb_mev_per_k * temperature), -500.0, 500.0)
    return 1.0 / (np.exp(x) + 1.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\n",
            "call": "round(float(np.asarray(fermi_occupation(np.array([0.0]), 5.0))[0]), 12)",
            "gold_call": "round(float(np.asarray(_oracle_fermi_occupation(np.array([0.0]), 5.0))[0]), 12)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "[round(float(v), 10) for v in np.asarray(fermi_occupation(np.array([-2.0, 0.0, 2.0]), 0.0))]",
            "gold_call": "[round(float(v), 10) for v in np.asarray(_oracle_fermi_occupation(np.array([-2.0, 0.0, 2.0]), 0.0))]",
        },
        {
            "setup": "import numpy as np\n",
            "call": "int(bool(np.all(np.asarray(fermi_occupation(np.array([1e6, -1e6]), 1e-6)) >= 0.0)))",
            "gold_call": "int(bool(np.all(np.asarray(_oracle_fermi_occupation(np.array([1e6, -1e6]), 1e-6)) >= 0.0)))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "np.asarray(fermi_occupation(np.array([-1.5, 0.0, 1.5]), 12.0))",
            "gold_call": "np.asarray(_oracle_fermi_occupation(np.array([-1.5, 0.0, 1.5]), 12.0))",
        },
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        fermi_occupation(np.array([0.0]), -1.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n\ndef run_gold():\n    try:\n        _oracle_fermi_occupation(np.array([0.0]), -1.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
