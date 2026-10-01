"""
Convert the selected continued VA mode into the single dimensionless scalar.

The final observable compares the oscillatory and exponential scales of the selected complex mode. Because both frequency components carry the same horizon-radius normalization, the requested ratio can be formed directly from the dimensionless root after branch selection.

Returns
-------
return float(answer)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def final_answer(
    d: int = 250,
    ell: int = 2,
    solver_tol: float = 1e-12,
    residual_tol: float = 1e-10,
    merge_tol: float = 1e-7,
    max_seed_distance: float = 1.0,
) -> float:
    """Return the final finite scalar for the complete VA calculation.

    Parameters
    ----------
    d, ell : int
        Spacetime dimension and scalar angular number.
    solver_tol, residual_tol, merge_tol, max_seed_distance : float
        Numerical and branch-selection controls forwarded to the prior step.

    Returns
    -------
    answer : float
        Finite positive native float equal to the requested mode ratio.

    Raises
    ------
    ValueError
        If branch selection fails, the selected root has no positive imaginary
        part, or the final scalar is nonfinite or nonpositive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_final_answer(
    d: int = 250,
    ell: int = 2,
    solver_tol: float = 1e-12,
    residual_tol: float = 1e-10,
    merge_tol: float = 1e-7,
    max_seed_distance: float = 1.0,
) -> float:
    """Reference final scalar using only the selected-branch oracle."""
    import math

    root = _oracle_select_va_branch(
        d=d,
        ell=ell,
        solver_tol=solver_tol,
        residual_tol=residual_tol,
        merge_tol=merge_tol,
        max_seed_distance=max_seed_distance,
    )
    if root.imag <= 0.0:
        raise ValueError("selected root must have a positive imaginary part")

    answer = root.real / root.imag
    if not math.isfinite(answer) or answer <= 0.0:
        raise ValueError("final mode ratio must be finite and positive")
    return float(answer)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return final-step normal, nearby, edge, and invalid cases."""
    return [
        {
            "setup": "case_name = 'normal: final d=250 challenge scalar'\n",
            "call": "final_answer()",
            "gold_call": "_oracle_final_answer()",
        },
        {
            "setup": "case_name = 'nearby: final d=240 scalar'\n",
            "call": "final_answer(d=240)",
            "gold_call": "_oracle_final_answer(d=240)",
        },
        {
            "setup": "case_name = 'edge: final d=300 scalar'\n",
            "call": "final_answer(d=300)",
            "gold_call": "_oracle_final_answer(d=300)",
        },
        {
            "setup": (
                "case_name = 'invalid: delegated branch gate failure'\n\n"
                "def _value_error_status(function, *args, **kwargs):\n"
                "    try:\n"
                "        function(*args, **kwargs)\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "    return 0\n"
            ),
            "call": "_value_error_status(final_answer, max_seed_distance=1e-4)",
            "gold_call": "_value_error_status(_oracle_final_answer, max_seed_distance=1e-4)",
        },
    ]
