"""
Convert one Werner-state fidelity to the parameter that composes multiplicatively under swapping.

Reconstruct the Werner-state convention in the source paper's network and noise model. Enforce the physical interval used there and preserve full precision.

Returns
-------
return float(w)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

def werner_parameter(fidelity):
    """Return the Werner parameter of one Bell-pair fidelity.

    Returns:
        float: A finite scalar.
    """
    return float(w)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_werner_parameter(fidelity: float) -> float:
    import math
    fidelity = float(fidelity)
    if not math.isfinite(fidelity) or fidelity < 0.25 or fidelity > 1.0:
        raise ValueError("fidelity must be finite and in [0.25, 1]")
    return (4.0 * fidelity - 1.0) / 3.0

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "", "call": "werner_parameter(0.25)", "gold_call": "_oracle_werner_parameter(0.25)"},
        {"setup": "", "call": "werner_parameter(0.5)", "gold_call": "_oracle_werner_parameter(0.5)"},
        {"setup": "", "call": "werner_parameter(0.68)", "gold_call": "_oracle_werner_parameter(0.68)"},
        {"setup": "", "call": "werner_parameter(0.8)", "gold_call": "_oracle_werner_parameter(0.8)"},
        {"setup": "", "call": "werner_parameter(0.917)", "gold_call": "_oracle_werner_parameter(0.917)"},
        {"setup": "", "call": "werner_parameter(0.999)", "gold_call": "_oracle_werner_parameter(0.999)"},
        {"setup": "", "call": "werner_parameter(1.0)", "gold_call": "_oracle_werner_parameter(1.0)"}
    ]
