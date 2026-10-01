"""
Return the mean turnover time for the inhibited static type-A reference.

The static reference provides the comparison timescale for the same inhibited chemistry without conformational switching.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def static_turnover(u0: float, u1: float, u2: float, w0: float, w1: float) -> float:
    """Return the mean product-formation time for the inhibited static type-A catalyst.

    All rates are positive finite values in inverse seconds.

    Returns
    -------
    float
        Mean static turnover time in seconds.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_static_turnover(u0: float, u1: float, u2: float, w0: float, w1: float) -> float:
    return float((1.0/u0 + w0/(u0*u2)) * (1.0 + u1/w1) + 1.0/u2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary and edge cases with isolated inputs."""
    return [
        {
            'setup': '',
            'call': 'static_turnover(10.,3.,1.,4.,1/3)',
            'gold_call': '_oracle_static_turnover(10.,3.,1.,4.,1/3)',
            'tol': 1e-12,
        },
        {
            'setup': '',
            'call': 'static_turnover(4.,2.,.5,1.2,.25)',
            'gold_call': '_oracle_static_turnover(4.,2.,.5,1.2,.25)',
            'tol': 1e-12,
        },
        {
            'setup': '',
            'call': 'static_turnover(8.,.7,2.4,.2,.9)',
            'gold_call': '_oracle_static_turnover(8.,.7,2.4,.2,.9)',
            'tol': 1e-12,
        },
        {
            'setup': '',
            'call': 'static_turnover(.8,6.,.2,5.,2.)',
            'gold_call': '_oracle_static_turnover(.8,6.,.2,5.,2.)',
            'tol': 1e-12,
        },
        {
            'setup': '',
            'call': 'static_turnover(10., 1e-12, 1., 4., 1/3)',
            'gold_call': '_oracle_static_turnover(10., 1e-12, 1., 4., 1/3)',
            'tol': 1e-12,
        },
    ]
