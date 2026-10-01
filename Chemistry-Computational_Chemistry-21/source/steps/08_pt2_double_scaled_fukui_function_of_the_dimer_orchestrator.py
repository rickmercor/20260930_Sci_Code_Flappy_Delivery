"""
Final orchestrator. Evaluate a site-0 Fukui function of the

two-electron dimer from the zero-weight N-centered ensemble relation, with the

weight derivative of the ensemble Hxc potential taken from the PT2-level

double-scaling approximation.

In a scaled ensemble density-functional approximation,

the regular ground-state ingredients are exact and only the weight dependence

of the ensemble Hxc potential is approximated. The resulting Fukui function

therefore differs from the exact one only through that approximation, which

isolates the weight-derivative (discontinuity) error of the approximation.

Returns
-------
a single float rounded to three decimal places. Raise     ``ValueError`` for invalid inputs, including ``branch`` not in {+1, -1}.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pt2_double_scaled_fukui(t: float, U: float, dv: float, branch: int) -> float:
    """Return the PT2 double-scaled site-0 Fukui function, rounded to three decimals.

    Expected return: a single float rounded to three decimal places. Raise
    ``ValueError`` for invalid inputs, including ``branch`` not in {+1, -1}.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_pt2_double_scaled_fukui(t: float, U: float, dv: float, branch: int) -> float:
    if branch != 1 and branch != -1:
        raise ValueError("branch must be +1 (affinity) or -1 (ionization).")
    w_plus, w_minus = _oracle_pt2_double_scaled_weight_derivatives(t, U, dv)
    if branch == 1:
        fukui = _oracle_nc_zero_weight_fukui(t, U, dv, w_plus, 1)
    else:
        fukui = _oracle_nc_zero_weight_fukui(t, U, dv, w_minus, -1)
    return round(fukui, 3)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return the canonical fixture plus further dimers on both branches."""
    return [
        # Normal: canonical fixture, ionization branch.
        {
            "setup": "",
            "call": "pt2_double_scaled_fukui(1.0, 2.5, 1.5, -1)",
            "gold_call": "_oracle_pt2_double_scaled_fukui(1.0, 2.5, 1.5, -1)",
        },
        # Normal: canonical fixture, affinity branch.
        {
            "setup": "",
            "call": "pt2_double_scaled_fukui(1.0, 2.5, 1.5, 1)",
            "gold_call": "_oracle_pt2_double_scaled_fukui(1.0, 2.5, 1.5, 1)",
        },
        # Normal: moderate correlation, strong asymmetry, affinity branch.
        {
            "setup": "",
            "call": "pt2_double_scaled_fukui(1.0, 1.5, 3.0, 1)",
            "gold_call": "_oracle_pt2_double_scaled_fukui(1.0, 1.5, 3.0, 1)",
        },
        # Normal: intermediate correlation, ionization branch.
        {
            "setup": "",
            "call": "pt2_double_scaled_fukui(1.0, 2.0, 2.4, -1)",
            "gold_call": "_oracle_pt2_double_scaled_fukui(1.0, 2.0, 2.4, -1)",
        },
        # Edge: strong correlation near the symmetric region, ionization branch.
        {
            "setup": "",
            "call": "pt2_double_scaled_fukui(1.0, 3.0, 1.0, -1)",
            "gold_call": "_oracle_pt2_double_scaled_fukui(1.0, 3.0, 1.0, -1)",
        },
    ]
