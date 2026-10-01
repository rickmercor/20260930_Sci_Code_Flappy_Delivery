"""
Step 1: the temperature-dependent development rate of the embryo.

The rate curve is a four-parameter temperature reaction norm of the kind fitted to incubation durations of marine turtle eggs. It rises steeply through the viable range and falls again at high temperature, so each temperature of a nest record carries its own rate per minute of development.

The curve is written in kelvin. The parameter Rho25 scales the rate at the reference temperature of 298 K, DHA is the enthalpy of activation, and the pair DHH and T12H describes the high-temperature inactivation. Every nest record is held at its logged temperature until the next reading, so this rate is the local clock of the growth model of the following steps.

Returns
-------
rate : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def development_rate(
    temperature: "numpy.typing.ArrayLike",
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
) -> "numpy.ndarray":
    """Evaluate the development rate curve at each temperature, per minute.

    Parameters
    ----------
    temperature : array-like
        Temperatures in degrees Celsius. Every entry must be finite and above
        -273.15.
    DHA : float, optional
        Enthalpy of activation, in kJ/mol (547.922961 in the task data).
    DHH : float, optional
        Enthalpy of high-temperature inactivation, in kJ/mol (576.474056).
    T12H : float, optional
        Temperature of half high-temperature inactivation, in kelvin (300.631447).
    Rho25 : float, optional
        Rate at the reference temperature of 298 K, in units of 1e-7 per minute
        (88.421611).

    Returns
    -------
    rate : numpy.ndarray
        Development rate per minute, with the shape of `temperature`, with
        temperatures converted to kelvin by adding 273.15, the reference
        temperature taken as exactly 298 K and the gas constant as 8.314472
        J/(mol K).

    Raises
    ------
    ValueError
        If an entry of `temperature` is not finite or is at or below -273.15
        degrees Celsius.
    """
    return rate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_development_rate(
    temperature: "numpy.typing.ArrayLike",
    DHA: float = 547.922961,
    DHH: float = 576.474056,
    T12H: float = 300.631447,
    Rho25: float = 88.421611,
) -> "numpy.ndarray":
    """Reference implementation for development_rate."""
    T = np.asarray(temperature, dtype=float)
    if not np.all(np.isfinite(T)):
        raise ValueError("temperature entries must be finite")
    Tk = T + 273.15
    if np.any(Tk <= 0.0):
        raise ValueError("temperature must be above -273.15 degrees Celsius")
    rho25 = float(Rho25) / 1e7
    dha = float(DHA) * 1e3
    dhh = float(DHH) * 1e3
    return (rho25 * (Tk / 298.0) * np.exp((dha / 8.314472) * (1.0 / 298.0 - 1.0 / Tk))) / (
        1.0 + np.exp((dhh / 8.314472) * (1.0 / np.abs(float(T12H)) - 1.0 / Tk))
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return list of test case specifications."""
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        development_rate(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_development_rate(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: the temperature range of the supplied nest records.
        {"setup": "import numpy as np\n"
                  "TEMP = np.array([28.5, 29.0, 29.5, 30.0, 30.5, 31.0, 31.5, 32.0], dtype=float)\n",
         "call": "development_rate(np.array(TEMP, dtype=float))",
         "gold_call": "_oracle_development_rate(np.array(TEMP, dtype=float))"},
        # Boundary: the reference temperature 298 K, where the rate is set by Rho25.
        {"setup": "import numpy as np\n"
                  "TEMP = np.array([25.0, 25.0], dtype=float)\n",
         "call": "development_rate(np.array(TEMP, dtype=float))",
         "gold_call": "_oracle_development_rate(np.array(TEMP, dtype=float))"},
        # Edge: a cold and a hot temperature, where the two exponential arms dominate.
        {"setup": "import numpy as np\n"
                  "TEMP = np.array([18.0, 20.0, 33.0, 35.0, 36.0], dtype=float)\n",
         "call": "development_rate(np.array(TEMP, dtype=float))",
         "gold_call": "_oracle_development_rate(np.array(TEMP, dtype=float))"},
        # Edge: the two parameter blocks of the task, whose DHA differs in sign.
        {"setup": "import numpy as np\n"
                  "TEMP = np.array([29.0, 30.0, 31.0], dtype=float)\n",
         "call": "development_rate(np.array(TEMP, dtype=float), DHA=-719.576256, DHH=685.203574, "
                 "T12H=552.204465, Rho25=100.0)",
         "gold_call": "_oracle_development_rate(np.array(TEMP, dtype=float), DHA=-719.576256, "
                      "DHH=685.203574, T12H=552.204465, Rho25=100.0)"},
        # Invalid: a temperature below absolute zero.
        {"setup": "import numpy as np\n"
                  "TEMP = np.array([25.0, -300.0], dtype=float)\n" + invalid,
         "call": "run_model(temperature=TEMP)",
         "gold_call": "run_gold(temperature=TEMP)"},
        # Invalid: a non-finite temperature.
        {"setup": "import numpy as np\n"
                  "TEMP = np.array([25.0, np.nan], dtype=float)\n" + invalid,
         "call": "run_model(temperature=TEMP)",
         "gold_call": "run_gold(temperature=TEMP)"},
    ]
