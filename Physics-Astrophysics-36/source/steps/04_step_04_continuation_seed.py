"""
Build a local continuation predictor for the intended scalar VA branch from neighboring large-d results in the paper.

The matching residual has multiple complex zeros, so a numerical solve requires a branch-identification strategy rather than an arbitrary starting point. Use the neighboring large-d scalar entries in Table II only as continuation anchors for the search; the predictor is not the final mode frequency and must later be corrected by solving the matching problem.

For this fixed continuation predictor, the printed scalar large-$d$ anchor data are $z_{200}=96.020+8.251i$ and $z_{300}=145.426+9.358i$, where $z_d=\omega r_h$ at dimension $d$. Use these printed values as the exact operands of the affine predictor specified in the return contract; they are search inputs, not the corrected mode frequencies.

Returns
-------
return complex(seed)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def continuation_seed(d: int = 250) -> complex:
    """Return a complex continuation predictor for 200 <= d <= 300.

    Parameters
    ----------
    d : int
        Integer spacetime dimension in the closed interval [200, 300].

    Returns
    -------
    seed : complex
        Affine predictor (1-t)*z_200 + t*z_300, with t=(d-200)/100.
        Use the printed scalar large-d entries of Table II for z_200 and
        z_300, at their tabulated precision. This interpolation supplies
        only the initial predictor; later steps solve the matching equation.

    Raises
    ------
    ValueError
        If d is not an integer in [200, 300] or the predictor is nonfinite.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_continuation_seed(d: int = 250) -> complex:
    """Reference linear continuation predictor from the paper's Table II anchors."""
    import math
    from numbers import Integral

    if isinstance(d, bool) or not isinstance(d, Integral):
        raise ValueError("d must be an integer, not a boolean")
    d_i = int(d)
    if d_i < 200 or d_i > 300:
        raise ValueError("d must lie in [200, 300] for this local continuation step")

    d_lo = 200.0
    d_hi = 300.0
    z_lo = complex(96.020, 8.251)
    z_hi = complex(145.426, 9.358)
    fraction = (float(d_i) - d_lo) / (d_hi - d_lo)
    seed = z_lo + fraction * (z_hi - z_lo)

    if not (math.isfinite(seed.real) and math.isfinite(seed.imag)):
        raise ValueError("continuation predictor must be finite")
    if seed.real <= 0.0 or seed.imag <= 0.0:
        raise ValueError("continuation predictor must lie in the positive quadrant")
    return complex(seed)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid cases."""
    return [
        {
            "setup": "case_name = 'normal: d=250 continuation predictor'\n",
            "call": "continuation_seed()",
            "gold_call": "_oracle_continuation_seed()",
        },
        {
            "setup": "case_name = 'boundary: lower Table II anchor'\n",
            "call": "continuation_seed(d=200)",
            "gold_call": "_oracle_continuation_seed(d=200)",
        },
        {
            "setup": "case_name = 'edge: upper Table II anchor'\n",
            "call": "continuation_seed(d=300)",
            "gold_call": "_oracle_continuation_seed(d=300)",
        },
        {
            "setup": (
                "case_name = 'invalid: dimension outside local continuation interval'\n\n"
                "def _value_error_status(function, *args, **kwargs):\n"
                "    try:\n"
                "        function(*args, **kwargs)\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "    return 0\n"
            ),
            "call": "_value_error_status(continuation_seed, d=199)",
            "gold_call": "_value_error_status(_oracle_continuation_seed, d=199)",
        },
    ]
