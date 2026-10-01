"""
Return the ratio measuring how the interface divides its throughput between the two competing channels, and the apparent power law an experiment would report for that interface. The second follows from the first in closed form and is bounded between the two single-channel limits. The source fixes a convention here that the natural reading does not; follow the source.

When two channels of different order share an interface, no fixed algebraic interface law describes it: the effective power relating throughput to donor loading sits between the values the two channels would give alone, and drifts as the loading changes. The ratio that parameterises the drift compares the two channels on a common footing, and which footing that is is part of its definition.

Returns
-------
ndarray of shape (2,): the branching ratio, then the apparent exponent.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def branching_ratio_and_exponent(recombination_rate: float, fluorination_rate: float) -> "np.ndarray":
    """Return the ratio measuring how the interface divides its throughput between the two competing channels, and the apparent power law an experiment would report for that interface. The second follows from the first in closed form and is bounded between the two single-channel limits. The source fixes a convention here that the natural reading does not; follow the source.

    Parameters
    ----------
    recombination_rate : float
        Net rate of the recombination channel. Positive.
    fluorination_rate : float
        Net rate of the oxidation channel. Not negative.

    Returns
    -------
    ndarray of shape (2,): the branching ratio, then the apparent exponent.

    Raises
    ------
    ValueError: if recombination_rate is not positive, or fluorination_rate is negative.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_branching_ratio_and_exponent(recombination_rate: float, fluorination_rate: float) -> "np.ndarray":
    if recombination_rate <= 0:
        raise ValueError("recombination rate must be positive")
    if fluorination_rate < 0:
        raise ValueError("fluorination rate must not be negative")
    # CONVENTION (paper, eq. 15): the branching ratio compares the two channels' shares of the ATOMIC flux, not their raw rates, so the recombination rate is doubled before the comparison.
    # The golden solution carries the full argument and the natural wrong answer.
    branching = fluorination_rate / (2.0 * recombination_rate)
    # CONVENTION (paper, eq. B.7): the apparent interfacial exponent is the
    # logarithmic slope of the ATOMIC FLUX against the interfacial concentration, and
    # in that variable it closes to (2+B)/(1+B), bounded in [1, 2]. The same slope
    # read off the salt-side INVENTORY is a different number whenever the carriers
    # have different diffusivities, which is exactly why the paper gives them
    # different ones.
    return np.array([branching, (2.0 + branching) / (1.0 + branching)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "branching_ratio_and_exponent(0.7, 0.3)",
         "gold_call": "_oracle_branching_ratio_and_exponent(0.7, 0.3)"},   # normal
        {"setup": "import numpy as np",
         "call": "branching_ratio_and_exponent(1.0, 0.0)",
         "gold_call": "_oracle_branching_ratio_and_exponent(1.0, 0.0)"},   # boundary
        {"setup": "import numpy as np",
         "call": "branching_ratio_and_exponent(2.4e-9, 7.1e-4)",
         "gold_call": "_oracle_branching_ratio_and_exponent(2.4e-9, 7.1e-4)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        branching_ratio_and_exponent(0.0, 0.3)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_branching_ratio_and_exponent(0.0, 0.3)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
