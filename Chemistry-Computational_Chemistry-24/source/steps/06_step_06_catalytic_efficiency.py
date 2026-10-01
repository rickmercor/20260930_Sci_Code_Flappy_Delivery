"""
Return the source-defined catalytic-efficiency scalar from static and dynamic mean turnover times.

The benchmark compares catalytic performance of the dynamic and static regimes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def catalytic_efficiency(static_time: float, dynamic_time: float) -> float:
    """Return the source-defined catalytic-efficiency quantity.

    ``static_time`` and ``dynamic_time`` are positive finite mean turnover times
    in seconds.

    Returns
    -------
    float
        Dimensionless catalytic-efficiency quantity.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_catalytic_efficiency(static_time: float, dynamic_time: float) -> float:
    return float(static_time / dynamic_time)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': '', 'call': 'catalytic_efficiency(6.,1.3122806007045038)', 'gold_call': '_oracle_catalytic_efficiency(6.,1.3122806007045038)', 'tol': 1e-12},
        {'setup': '', 'call': 'catalytic_efficiency(2.,2.)', 'gold_call': '_oracle_catalytic_efficiency(2.,2.)', 'tol': 1e-12},
        {'setup': '', 'call': 'catalytic_efficiency(.1,5.)', 'gold_call': '_oracle_catalytic_efficiency(.1,5.)', 'tol': 1e-12},
        {'setup': '', 'call': 'catalytic_efficiency(100.,.07)', 'gold_call': '_oracle_catalytic_efficiency(100.,.07)', 'tol': 1e-12},
    ]
