"""
Compute the large-d Hankel order parameter used by the scalar Tangherlini virtual-absorption approximation.

In the large-d scalar construction, the exterior radial solution is a Hankel function whose order is fixed by the spacetime dimension and angular index. This step isolates that paper-specific parameter so later steps can focus on the matching problem rather than repeat dimensional bookkeeping.

Returns
-------
return float(nu)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def order_parameter(d: int = 250, ell: int = 2) -> float:
    """Return the scalar large-d Hankel order.

    Parameters
    ----------
    d : int
        Spacetime dimension, required to be an integer >= 10 and not a boolean.
    ell : int
        Scalar angular number, required to be an integer >= 0 and not a boolean.

    Returns
    -------
    nu : float
        Finite positive Hankel order used by the large-d scalar solution.

    Raises
    ------
    ValueError
        If d or ell violates the stated requirements, or if the resulting order
        is not a finite positive float.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_order_parameter(d: int = 250, ell: int = 2) -> float:
    """Reference implementation of the large-d scalar Hankel order."""
    import math
    from numbers import Integral

    if isinstance(d, bool) or not isinstance(d, Integral):
        raise ValueError("d must be an integer, not a boolean")
    if isinstance(ell, bool) or not isinstance(ell, Integral):
        raise ValueError("ell must be an integer, not a boolean")

    d_i = int(d)
    ell_i = int(ell)
    if d_i < 10:
        raise ValueError("d must be >= 10")
    if ell_i < 0:
        raise ValueError("ell must be >= 0")

    try:
        nu = float(ell_i) + (float(d_i) - 3.0) / 2.0
    except (OverflowError, ValueError) as error:
        raise ValueError("d and ell must produce a representable Hankel order") from error

    if not math.isfinite(nu) or nu <= 0.0:
        raise ValueError("Hankel order must be finite and positive")
    return float(nu)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid cases."""
    return [
        {
            "setup": "case_name = 'normal: challenge order parameter'\n",
            "call": "order_parameter()",
            "gold_call": "_oracle_order_parameter()",
        },
        {
            "setup": "case_name = 'boundary: minimum allowed dimension'\n",
            "call": "order_parameter(d=10, ell=0)",
            "gold_call": "_oracle_order_parameter(d=10, ell=0)",
        },
        {
            "setup": "case_name = 'edge: high dimension and angular number'\n",
            "call": "order_parameter(d=400, ell=10)",
            "gold_call": "_oracle_order_parameter(d=400, ell=10)",
        },
        {
            "setup": (
                "case_name = 'invalid: negative angular index'\n\n"
                "def _value_error_status(function, *args, **kwargs):\n"
                "    try:\n"
                "        function(*args, **kwargs)\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "    return 0\n"
            ),
            "call": "_value_error_status(order_parameter, ell=-1)",
            "gold_call": "_value_error_status(_oracle_order_parameter, ell=-1)",
        },
    ]
