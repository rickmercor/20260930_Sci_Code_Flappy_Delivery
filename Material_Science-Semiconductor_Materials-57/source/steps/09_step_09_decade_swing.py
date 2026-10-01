"""
Evaluate the positive subthreshold-swing magnitude in mV/dec between two transfer points.

Gate control of a nearly exponential current flank is summarized by the millivolts of gate needed for one decade of current change.

Returns
-------
float — positive SS magnitude in mV/dec
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def decade_swing(V_g1: float, V_g2: float, I1: float, I2: float) -> float:
    """Return the positive subthreshold-swing magnitude in mV/dec.

    The result is unchanged when the two transfer points are exchanged and
    is positive for either increasing or decreasing current magnitudes.

    Signed currents enter through their magnitudes |I1| and |I2|.

    Raises
    ------
    ValueError
        If inputs are non-finite, gate voltages are equal, currents are zero,
        or log10 magnitudes are equal.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_decade_swing(V_g1: float, V_g2: float, I1: float, I2: float) -> float:
    V_g1 = float(V_g1)
    V_g2 = float(V_g2)
    I1 = float(I1)
    I2 = float(I2)
    if not all(np.isfinite(v) for v in (V_g1, V_g2, I1, I2)):
        raise ValueError("All inputs must be finite.")
    if V_g1 == V_g2:
        raise ValueError("V_g1 and V_g2 must differ.")
    if I1 == 0.0 or I2 == 0.0:
        raise ValueError("Currents must be nonzero.")
    dlog = np.log10(abs(I1)) - np.log10(abs(I2))
    if dlog == 0.0:
        raise ValueError("log10|I1| and log10|I2| must differ.")
    return float(abs(V_g2 - V_g1) / abs(dlog) * 1000.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": "decade_swing(0.20, 0.32, -230.14140927397327, -7.206820836770518)",
            "gold_call": "_oracle_decade_swing(0.20, 0.32, -230.14140927397327, -7.206820836770518)",
        },
        {
            "setup": "import numpy as np",
            "call": "decade_swing(0.0, 0.060, 1.0, 0.1)",
            "gold_call": "_oracle_decade_swing(0.0, 0.060, 1.0, 0.1)",
        },
        {
            "setup": "import numpy as np",
            "call": "decade_swing(0.0, 0.060, 0.1, 1.0)",
            "gold_call": "_oracle_decade_swing(0.0, 0.060, 0.1, 1.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "decade_swing(0.060, 0.0, 1.0, 0.1)",
            "gold_call": "_oracle_decade_swing(0.060, 0.0, 1.0, 0.1)",
        },
    ]
