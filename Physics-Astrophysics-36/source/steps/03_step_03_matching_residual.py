"""
Assemble a scale-invariant complex compatibility residual for matching the two large-d radial bases.

A virtual-absorption mode exists only when one relative amplitude can make both the field and its first derivative continuous at the interface. Eliminate that arbitrary amplitude and use the basis data from the previous step to construct a numerically stable complex residual whose zero is the matching condition.

Returns
-------
return complex(residual)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def matching_residual(
    d: int = 250,
    ell: int = 2,
    z: complex = 120.0 + 9.0j,
) -> complex:
    """Return a scale-invariant complex residual for interface compatibility.

    Parameters
    ----------
    d : int
        Spacetime dimension.
    ell : int
        Scalar angular number.
    z : complex
        Finite nonzero dimensionless frequency.

    Returns
    -------
    residual : complex
        Dimensionless outer-minus-inner logarithmic-derivative residual:
        dpsi_out / psi_out - dpsi_in / psi_in, using the basis values
        and dimensionless derivatives returned by mode_basis_at_match.
        Return this normalization and sign, without an additional multiplier.

    Raises
    ------
    ValueError
        If delegated validation fails, either basis value needed for
        normalization vanishes, or the residual is nonfinite.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_matching_residual(
    d: int = 250,
    ell: int = 2,
    z: complex = 120.0 + 9.0j,
) -> complex:
    """Reference scale-invariant matching residual from earlier-step basis data."""
    import math

    psi_in, dpsi_in, psi_out, dpsi_out = _oracle_mode_basis_at_match(d=d, ell=ell, z=z)
    if psi_in == 0.0j or psi_out == 0.0j:
        raise ValueError("basis normalization requires nonzero interface values")

    residual = dpsi_out / psi_out - dpsi_in / psi_in
    if not (math.isfinite(residual.real) and math.isfinite(residual.imag)):
        raise ValueError("matching residual must be finite")
    return complex(residual)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid cases."""
    return [
        {
            "setup": "case_name = 'normal: compatibility residual near target branch'\n",
            "call": "matching_residual()",
            "gold_call": "_oracle_matching_residual()",
        },
        {
            "setup": "case_name = 'boundary: low-dimensional valid residual'\n",
            "call": "matching_residual(d=10, ell=0, z=2.0 + 1.0j)",
            "gold_call": "_oracle_matching_residual(d=10, ell=0, z=2.0 + 1.0j)",
        },
        {
            "setup": "case_name = 'edge: large-d residual evaluation'\n",
            "call": "matching_residual(d=400, ell=2, z=195.0 + 10.0j)",
            "gold_call": "_oracle_matching_residual(d=400, ell=2, z=195.0 + 10.0j)",
        },
        {
            "setup": (
                "case_name = 'invalid: delegated singular frequency'\n\n"
                "def _value_error_status(function, *args, **kwargs):\n"
                "    try:\n"
                "        function(*args, **kwargs)\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "    return 0\n"
            ),
            "call": "_value_error_status(matching_residual, z=0.0j)",
            "gold_call": "_value_error_status(_oracle_matching_residual, z=0.0j)",
        },
    ]
