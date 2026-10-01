"""
Return the source-defined energy-efficiency scalar from the supplied energetic and catalytic quantities.

The energetic benchmark relates dissipation per product to the catalytic performance gain.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def energy_efficiency(delta_w: float, efficiency: float) -> float:
    """Return the source-defined energy-efficiency quantity.

    ``delta_w`` is positive dissipation per product in units of ``k_B T`` and
    ``efficiency`` is greater than one for every tested input.

    Returns
    -------
    float
        Finite source-defined energy-efficiency quantity.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_energy_efficiency(delta_w: float, efficiency: float) -> float:
    return float(delta_w / (efficiency - 1.0))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': '', 'call': 'energy_efficiency(1.7147634614896477,4.572192865442705)', 'gold_call': '_oracle_energy_efficiency(1.7147634614896477,4.572192865442705)', 'tol': 1e-12},
        {'setup': '', 'call': 'energy_efficiency(.5,2.)', 'gold_call': '_oracle_energy_efficiency(.5,2.)', 'tol': 1e-12},
        {'setup': '', 'call': 'energy_efficiency(10.,1.1)', 'gold_call': '_oracle_energy_efficiency(10.,1.1)', 'tol': 1e-12},
        {'setup': '', 'call': 'energy_efficiency(.01,50.)', 'gold_call': '_oracle_energy_efficiency(.01,50.)', 'tol': 1e-12},
    ]
