"""
Convert the summed interface stress ratios of the two arms into the penalty stiffness of the structural cohesive element between them.

The source construction replaces the buried resin-rich layers by one equivalent structural interface. Apply its compliance aggregation rule using the supplied layer thickness and mode-appropriate resin modulus.

Returns
-------
float: the penalty stiffness of the structural cohesive element, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def resin_rich_penalty_stiffness(ratio_sum_total: float, h_rr: float, modulus: float) -> float:
    """Evaluate the source-defined equivalent penalty stiffness.

    Parameters
    ----------
    ratio_sum_total : float
        Combined dimensionless arm result from step 01
        (ratio_sum_total > 0).
    h_rr : float
        Thickness of one resin-rich layer (h_rr > 0).
    modulus : float
        Mode-appropriate elastic modulus of the resin-rich layer
        (modulus > 0).

    Returns
    -------
    penalty_stiffness : float
        Equivalent penalty stiffness prescribed by the source construction,
        as a native Python float.

    Raises ValueError if any argument is not a finite real number greater than
    zero.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_resin_rich_penalty_stiffness(ratio_sum_total: float, h_rr: float,
                                         modulus: float) -> float:

    values = {"ratio_sum_total": ratio_sum_total, "h_rr": h_rr, "modulus": modulus}
    for name, val in values.items():
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(val)) or float(val) <= 0.0:
            raise ValueError(f"{name} must be a finite number > 0")

    ratio_sum_total = float(ratio_sum_total)
    h_rr = float(h_rr)
    modulus = float(modulus)

    # The compliance of the interface is the stacked resin-layer compliance.
    interface_compliance = ratio_sum_total * h_rr / modulus
    return float(1.0 / interface_compliance)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned value: a 24-ply symmetric stack with 12 unreinforced plies
        # per arm sums to 13, so a resin layer of 0.02286 mm and modulus
        # 3400 N/mm2 gives 3400/(13*0.02286) = 11440.86... N/mm3, computed here
        # from the closed form rather than from the oracle.
        {
            "setup": """import numpy as np
EXPECTED = 3400.0 / (13.0 * 0.02286)
""",
            "call": "resin_rich_penalty_stiffness(13.0, 0.02286, 3400.0)",
            "gold_call": "EXPECTED",
        },
        # --- Valid: an opening-mode ratio sum of a thick stack (normal scenario) ---
        {
            "setup": """import numpy as np
ratio_sum_total = 16.402518
h_rr = 0.0254
modulus = 4100.0
""",
            "call": "resin_rich_penalty_stiffness(ratio_sum_total, h_rr, modulus)",
            "gold_call": "_oracle_resin_rich_penalty_stiffness(ratio_sum_total, h_rr, modulus)",
        },
        # --- Valid: shear mode, a much smaller ratio sum and a stiffer result ---
        {
            "setup": """import numpy as np
ratio_sum_total = 4.517903
h_rr = 0.0254
modulus = 1490.0
""",
            "call": "resin_rich_penalty_stiffness(ratio_sum_total, h_rr, modulus)",
            "gold_call": "_oracle_resin_rich_penalty_stiffness(ratio_sum_total, h_rr, modulus)",
        },
        # --- Boundary: a single resin layer per arm, ratio sum exactly two ---
        {
            "setup": """import numpy as np
""",
            "call": "resin_rich_penalty_stiffness(2.0, 0.02286, 1850.0)",
            "gold_call": "_oracle_resin_rich_penalty_stiffness(2.0, 0.02286, 1850.0)",
        },
        # --- Edge: a very small ratio sum drives the stiffness up sharply ---
        {
            "setup": """import numpy as np
""",
            "call": "resin_rich_penalty_stiffness(0.004, 0.02286, 560.0)",
            "gold_call": "_oracle_resin_rich_penalty_stiffness(0.004, 0.02286, 560.0)",
        },
        # --- Invalid: a non-positive ratio sum has no physical meaning ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        resin_rich_penalty_stiffness(0.0, 0.02286, 4700.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_resin_rich_penalty_stiffness(0.0, 0.02286, 4700.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative resin-rich layer thickness ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        resin_rich_penalty_stiffness(13.0, -0.02286, 4700.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_resin_rich_penalty_stiffness(13.0, -0.02286, 4700.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
