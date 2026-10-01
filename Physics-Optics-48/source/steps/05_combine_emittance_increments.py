"""
Combine independent chromatic, geometric and ISR increments into outgoing-emittance ratios.

Add the supplied relative increments in quadrature with the unit incoming emittance. ISR is horizontal only. Preserve the [x,y] ordering. The supported numerical domain contains finite nonnegative increments only when both quadrature norms are representable as finite IEEE-754 double-precision values; refuse an unrepresentable result with ValueError rather than returning infinity.

Returns
-------
Return a length-2 NumPy float array [R_x,R_y], both dimensionless.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def combine_emittance_increments(chromatic_growth, geometric_growth, isr_growth):
    """Return outgoing-to-incoming emittance ratios [R_x,R_y].

    Args:
        chromatic_growth: Nonnegative dimensionless chromatic increment.
        geometric_growth: Length-2 nonnegative array [g_x,g_y].
        isr_growth: Nonnegative dimensionless horizontal ISR increment.

    Returns:
        numpy.ndarray: Length-2 float array [R_x,R_y], dimensionless.

    Raises:
        ValueError: If any increment is nonfinite or negative, or if
            geometric_growth is not a finite length-2 array, or either
            quadrature result cannot be represented as a finite float.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_combine_emittance_increments(chromatic_growth, geometric_growth, isr_growth):
    import math
    import numpy as np
    chromatic_growth = float(chromatic_growth)
    geometric_growth = np.asarray(geometric_growth, dtype=float)
    isr_growth = float(isr_growth)
    if geometric_growth.shape != (2,) or not np.all(np.isfinite(geometric_growth)):
        raise ValueError("geometric_growth must be a finite length-2 array")
    if not all(math.isfinite(x) for x in (chromatic_growth, isr_growth)):
        raise ValueError("increments must be finite")
    if chromatic_growth < 0 or isr_growth < 0 or np.any(geometric_growth < 0):
        raise ValueError("increments must be nonnegative")
    rx = math.hypot(1.0, chromatic_growth, geometric_growth[0], isr_growth)
    ry = math.hypot(1.0, chromatic_growth, geometric_growth[1])
    if not all(math.isfinite(x) for x in (rx, ry)):
        raise ValueError("quadrature result exceeds the supported finite domain")
    return np.array([rx, ry], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np", "call":"combine_emittance_increments(0.,np.array([0.,0.]),0.)", "gold_call":"_oracle_combine_emittance_increments(0.,np.array([0.,0.]),0.)"},
        {"setup":"import numpy as np", "call":"combine_emittance_increments(.2,np.array([.1,.3]),.4)", "gold_call":"_oracle_combine_emittance_increments(.2,np.array([.1,.3]),.4)"},
        {"setup":"import numpy as np", "call":"combine_emittance_increments(.05,np.array([.8,.2]),0.)", "gold_call":"_oracle_combine_emittance_increments(.05,np.array([.8,.2]),0.)"},
        {"setup":"import numpy as np", "call":"combine_emittance_increments(0.,np.array([0.,0.]),.75)", "gold_call":"_oracle_combine_emittance_increments(0.,np.array([0.,0.]),.75)"},
        {"setup":"import numpy as np", "call":"combine_emittance_increments(1e200,np.array([0.,0.]),0.)", "gold_call":"_oracle_combine_emittance_increments(1e200,np.array([0.,0.]),0.)"},
        {"setup":"import numpy as np", "call":"combine_emittance_increments(.17,np.array([.91,0.]),.02)", "gold_call":"_oracle_combine_emittance_increments(.17,np.array([.91,0.]),.02)"},
        {"setup":"import numpy as np", "call":"combine_emittance_increments(1e-8,np.array([3e-8,4e-8]),5e-8)", "gold_call":"_oracle_combine_emittance_increments(1e-8,np.array([3e-8,4e-8]),5e-8)"},
        {"setup":"import numpy as np\ndef _ve(fn):\n try: fn()\n except ValueError: return 1.0\n except Exception: return -1.0\n return 0.0", "call":"_ve(lambda: combine_emittance_increments(1.7e308,np.array([1.7e308,0.]),0.))", "gold_call":"_ve(lambda: _oracle_combine_emittance_increments(1.7e308,np.array([1.7e308,0.]),0.))"}
    ]
