"""
Convert spin-resolved currents into signed spin filtering efficiency.

The signed polarization of the Landauer currents measures how completely one spin dominates transport under a given bias/gate setting.

Returns
-------
float — SFE in percent
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spin_filtering_efficiency(I_up: float, I_down: float) -> float:
    """Return the spin filtering efficiency in percent.

    Raises
    ------
    ValueError
        If either current is non-finite or I_up + I_down is zero.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_spin_filtering_efficiency(I_up: float, I_down: float) -> float:
    I_up = float(I_up)
    I_down = float(I_down)
    if not np.isfinite(I_up) or not np.isfinite(I_down):
        raise ValueError("Currents must be finite.")
    denom = I_up + I_down
    if denom == 0.0:
        raise ValueError("Total I_up + I_down must be nonzero.")
    return float((I_up - I_down) / denom * 100.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": "spin_filtering_efficiency(230.0, 1e-4)",
            "gold_call": "_oracle_spin_filtering_efficiency(230.0, 1e-4)",
        },
        {
            "setup": "import numpy as np",
            "call": "spin_filtering_efficiency(10.0, -2.0)",
            "gold_call": "_oracle_spin_filtering_efficiency(10.0, -2.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "spin_filtering_efficiency(1.0, 1.0)",
            "gold_call": "_oracle_spin_filtering_efficiency(1.0, 1.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "spin_filtering_efficiency(0.004833260464177144, 7.206820836771062)",
            "gold_call": "_oracle_spin_filtering_efficiency(0.004833260464177144, 7.206820836771062)",
        },
    ]
