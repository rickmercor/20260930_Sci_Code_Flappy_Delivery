"""
Evaluate the inner and zero-reflection outer basis functions and their dimensionless derivatives at the large-d matching point.

The paper gives different radial basis functions on the two sides of the transition point, with the virtual-absorption exterior branch obtained by removing the reflected Hankel component. Work with the dimensionless coordinate measured in units of the matching scale and return the two basis values and first derivatives at the interface without fixing their relative amplitude.

Returns
-------
return (complex(psi_in), complex(dpsi_in), complex(psi_out), complex(dpsi_out))
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mode_basis_at_match(
    d: int = 250,
    ell: int = 2,
    z: complex = 120.0 + 9.0j,
) -> tuple[complex, complex, complex, complex]:
    """Return inner/outer basis values and derivatives at the interface.

    Parameters
    ----------
    d : int
        Spacetime dimension.
    ell : int
        Scalar angular number.
    z : complex
        Finite nonzero dimensionless frequency measured using the matching scale.

    Returns
    -------
    basis : tuple[complex, complex, complex, complex]
        Native complex values (psi_in, dpsi_in, psi_out, dpsi_out), where the
        derivatives are with respect to the dimensionless matching coordinate.

    Raises
    ------
    ValueError
        If inputs are invalid, z is zero, or a required Hankel evaluation is
        singular or nonfinite.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_mode_basis_at_match(
    d: int = 250,
    ell: int = 2,
    z: complex = 120.0 + 9.0j,
) -> tuple[complex, complex, complex, complex]:
    """Reference interface basis built from the paper's large-d solutions."""
    import math
    from scipy import special
    
    def _coerce_finite_complex(value, name: str) -> complex:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be numeric, not a boolean")
        try:
            z = complex(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be representable as a complex scalar") from error
        if not (math.isfinite(z.real) and math.isfinite(z.imag)):
            raise ValueError(f"{name} must be finite")
        return z

    nu = _oracle_order_parameter(d=d, ell=ell)
    zz = _coerce_finite_complex(z, "z")
    if zz == 0.0j:
        raise ValueError("z must be nonzero")

    h = complex(special.hankel2(nu, zz))
    h_minus = complex(special.hankel2(nu - 1.0, zz))
    h_plus = complex(special.hankel2(nu + 1.0, zz))
    if not all(math.isfinite(v.real) and math.isfinite(v.imag) for v in (h, h_minus, h_plus)):
        raise ValueError("Hankel evaluations must be finite")
    if h == 0.0j:
        raise ValueError("outer basis vanishes at the matching point")

    h_prime = 0.5 * (h_minus - h_plus)
    psi_in = 1.0 + 0.0j
    dpsi_in = -1j * zz
    psi_out = h
    dpsi_out = 0.5 * h + zz * h_prime

    values = (psi_in, dpsi_in, psi_out, dpsi_out)
    if not all(math.isfinite(v.real) and math.isfinite(v.imag) for v in values):
        raise ValueError("basis values and derivatives must be finite")
    return tuple(complex(v) for v in values)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid cases."""
    return [
        {
            "setup": "case_name = 'normal: challenge interface basis'\n",
            "call": "mode_basis_at_match()",
            "gold_call": "_oracle_mode_basis_at_match()",
        },
        {
            "setup": "case_name = 'boundary: minimum dimension with complex frequency'\n",
            "call": "mode_basis_at_match(d=10, ell=0, z=2.0 + 1.0j)",
            "gold_call": "_oracle_mode_basis_at_match(d=10, ell=0, z=2.0 + 1.0j)",
        },
        {
            "setup": "case_name = 'edge: high-dimensional basis evaluation'\n",
            "call": "mode_basis_at_match(d=400, ell=2, z=195.0 + 10.0j)",
            "gold_call": "_oracle_mode_basis_at_match(d=400, ell=2, z=195.0 + 10.0j)",
        },
        {
            "setup": (
                "case_name = 'invalid: zero frequency argument'\n\n"
                "def _value_error_status(function, *args, **kwargs):\n"
                "    try:\n"
                "        function(*args, **kwargs)\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "    return 0\n"
            ),
            "call": "_value_error_status(mode_basis_at_match, z=0.0j)",
            "gold_call": "_value_error_status(_oracle_mode_basis_at_match, z=0.0j)",
        },
    ]
