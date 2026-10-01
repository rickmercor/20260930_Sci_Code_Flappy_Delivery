"""
Predict a freshly generated pair's initial fidelity from a calibrated link-quality parameter.

Use the hardware-quality model introduced in the source paper for heterogeneous links. Quality is finite and strictly positive; do not add the simulation's generation noise to this planning prediction.

Returns
-------
return float(fidelity)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

def hardware_initial_fidelity(quality):
    """Return the paper's noise-free predicted initial fidelity.

    Returns:
        float: A fidelity in (0.25, 1).
    """
    return float(fidelity)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_hardware_initial_fidelity(quality: float) -> float:
    import math
    quality = float(quality)
    if not math.isfinite(quality) or quality <= 0.0:
        raise ValueError("quality must be finite and positive")
    return 0.25 + 0.75 * math.exp(-1.0 / quality)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "", "call": "hardware_initial_fidelity(0.2)", "gold_call": "_oracle_hardware_initial_fidelity(0.2)"},
        {"setup": "", "call": "hardware_initial_fidelity(0.5)", "gold_call": "_oracle_hardware_initial_fidelity(0.5)"},
        {"setup": "", "call": "hardware_initial_fidelity(1.0)", "gold_call": "_oracle_hardware_initial_fidelity(1.0)"},
        {"setup": "", "call": "hardware_initial_fidelity(4.0)", "gold_call": "_oracle_hardware_initial_fidelity(4.0)"},
        {"setup": "", "call": "hardware_initial_fidelity(8.0)", "gold_call": "_oracle_hardware_initial_fidelity(8.0)"},
        {"setup": "", "call": "hardware_initial_fidelity(20.0)", "gold_call": "_oracle_hardware_initial_fidelity(20.0)"},
        {"setup": "", "call": "hardware_initial_fidelity(100.0)", "gold_call": "_oracle_hardware_initial_fidelity(100.0)"}
    ]
