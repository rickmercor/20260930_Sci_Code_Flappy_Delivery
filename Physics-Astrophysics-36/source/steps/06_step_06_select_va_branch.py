"""
Select the complex VA root that is consistent with continuation of the Table II scalar branch.

The multi-start search may recover more than one mathematically valid zero, so residual size alone is not enough to identify the requested mode. Use the local continuation predictor as a branch label, then require the selected root to remain within a finite continuation neighborhood while still satisfying the numerical matching tolerance.

Returns
-------
return complex(root)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_va_branch(
    d: int = 250,
    ell: int = 2,
    solver_tol: float = 1e-12,
    residual_tol: float = 1e-10,
    merge_tol: float = 1e-7,
    max_seed_distance: float = 1.0,
) -> complex:
    """Return the root belonging to the requested continued scalar VA branch.

    Parameters
    ----------
    d, ell : int
        Spacetime dimension and scalar angular number.
    solver_tol, residual_tol, merge_tol : float
        Numerical controls forwarded to the candidate-root search.
    max_seed_distance : float
        Positive maximum allowed complex-plane distance from the continuation
        predictor to the selected root.

    Returns
    -------
    root : complex
        Finite selected VA frequency in the positive quadrant.

    Raises
    ------
    ValueError
        If no candidate is consistent with the continuation branch or if the
        selected candidate fails an independent residual check.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_select_va_branch(
    d: int = 250,
    ell: int = 2,
    solver_tol: float = 1e-12,
    residual_tol: float = 1e-10,
    merge_tol: float = 1e-7,
    max_seed_distance: float = 1.0,
) -> complex:
    """Reference branch selection using only earlier-step oracles."""
    import math
    from numbers import Real

    def _validate_positive_real(value, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a real numeric scalar, not a boolean")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be representable as a float") from error
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return number

    distance_limit = _validate_positive_real(max_seed_distance, "max_seed_distance")
    residual_limit = _validate_positive_real(residual_tol, "residual_tol")
    seed = _oracle_continuation_seed(d=d)
    roots = _oracle_candidate_roots(
        d=d,
        ell=ell,
        solver_tol=solver_tol,
        residual_tol=residual_tol,
        merge_tol=merge_tol,
    )

    eligible = [root for root in roots if abs(root - seed) <= distance_limit]
    if not eligible:
        raise ValueError("no accepted root lies on the requested continuation branch")

    root = min(eligible, key=lambda value: (abs(value - seed), value.real, value.imag))
    residual_norm = abs(_oracle_matching_residual(d=d, ell=ell, z=root))
    if not math.isfinite(residual_norm) or residual_norm > residual_limit:
        raise ValueError("selected branch root fails the independent residual check")
    if root.real <= 0.0 or root.imag <= 0.0:
        raise ValueError("selected branch root must lie in the positive quadrant")
    return complex(root)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid branch-selection cases."""
    return [
        {
            "setup": "case_name = 'normal: select d=250 continued branch'\n",
            "call": "select_va_branch()",
            "gold_call": "_oracle_select_va_branch()",
        },
        {
            "setup": "case_name = 'boundary: select d=240 continued branch'\n",
            "call": "select_va_branch(d=240)",
            "gold_call": "_oracle_select_va_branch(d=240)",
        },
        {
            "setup": "case_name = 'edge: select upper-anchor branch'\n",
            "call": "select_va_branch(d=300)",
            "gold_call": "_oracle_select_va_branch(d=300)",
        },
        {
            "setup": (
                "case_name = 'invalid: continuation gate too tight'\n\n"
                "def _value_error_status(function, *args, **kwargs):\n"
                "    try:\n"
                "        function(*args, **kwargs)\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "    return 0\n"
            ),
            "call": "_value_error_status(select_va_branch, d=250, max_seed_distance=1e-4)",
            "gold_call": "_value_error_status(_oracle_select_va_branch, d=250, max_seed_distance=1e-4)",
        },
    ]
