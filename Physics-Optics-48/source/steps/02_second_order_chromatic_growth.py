"""
Evaluate the paper's achromatic-lattice second-order chromatic emittance increment.

For the mirror-symmetric thin-lens specialization of Eq. (26), compute g_ch=2(1+L/l) sigma_delta^2 sqrt[2 L^2/beta^2 + 3(l/L+1)^2]. The rms spread sigma_delta and g_ch are dimensionless; L, l and beta are positive and must share one length unit. Evaluate the square root stably, for example with hypot(sqrt(2)L/beta,sqrt(3)(l/L+1)); inputs whose finite result cannot be represented as an IEEE-754 double are outside the supported domain.

Returns
-------
Return one nonnegative dimensionless float g_ch.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

def second_order_chromatic_growth(sigma_delta, length_m, half_gap_m, beta_m):
    """Return the dimensionless second-order chromatic emittance increment.

    Args:
        sigma_delta: Nonnegative dimensionless rms relative energy spread.
        length_m: Positive lattice length L in metres.
        half_gap_m: Positive half-cell gap l in metres.
        beta_m: Positive beta function in metres.

    Returns:
        float: One finite nonnegative dimensionless scalar.

    Raises:
        ValueError: If any input is nonfinite, sigma_delta is negative, a
            length is nonpositive, or the result is not a finite float.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_second_order_chromatic_growth(sigma_delta, length_m, half_gap_m, beta_m):
    import math
    sigma_delta = float(sigma_delta)
    length_m = float(length_m)
    half_gap_m = float(half_gap_m)
    beta_m = float(beta_m)
    if not all(math.isfinite(x) for x in (sigma_delta, length_m, half_gap_m, beta_m)):
        raise ValueError("inputs must be finite")
    if sigma_delta < 0 or min(length_m, half_gap_m, beta_m) <= 0:
        raise ValueError("spread is nonnegative and lengths are positive")
    try:
        shape_root = math.hypot(math.sqrt(2.0) * length_m / beta_m,
                                math.sqrt(3.0) * (half_gap_m / length_m + 1.0))
        result = 2.0 * (1.0 + length_m / half_gap_m) * sigma_delta ** 2 * shape_root
    except (OverflowError, ZeroDivisionError):
        raise ValueError("result exceeds the supported finite domain")
    if not math.isfinite(result):
        raise ValueError("result exceeds the supported finite domain")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"", "call":"second_order_chromatic_growth(0.,2.,4.,.03)", "gold_call":"_oracle_second_order_chromatic_growth(0.,2.,4.,.03)"},
        {"setup":"", "call":"second_order_chromatic_growth(.032,5**.5,2*5**.5,.015*5**.5)", "gold_call":"_oracle_second_order_chromatic_growth(.032,5**.5,2*5**.5,.015*5**.5)"},
        {"setup":"", "call":"second_order_chromatic_growth(.021,1.7,1.1,.028)", "gold_call":"_oracle_second_order_chromatic_growth(.021,1.7,1.1,.028)"},
        {"setup":"", "call":"second_order_chromatic_growth(.006,.8,9.5,.12)", "gold_call":"_oracle_second_order_chromatic_growth(.006,.8,9.5,.12)"},
        {"setup":"", "call":"second_order_chromatic_growth(.075,12.,.35,.018)", "gold_call":"_oracle_second_order_chromatic_growth(.075,12.,.35,.018)"},
        {"setup":"", "call":"second_order_chromatic_growth(.019,3.4,3.4,3.4)", "gold_call":"_oracle_second_order_chromatic_growth(.019,3.4,3.4,3.4)"},
        {"setup":"", "call":"second_order_chromatic_growth(.028,17.,34.,.255)", "gold_call":"_oracle_second_order_chromatic_growth(.028,17.,34.,.255)"},
        {"setup":"def _ve(fn):\n try: fn()\n except ValueError: return 1.0\n except Exception: return -1.0\n return 0.0", "call":"_ve(lambda: second_order_chromatic_growth(-.01,2.,4.,.03))", "gold_call":"_ve(lambda: _oracle_second_order_chromatic_growth(-.01,2.,4.,.03))"}
    ]
