"""
Compose the canonical surface descriptor, nonseparating and separating cut enumeration, multiplicity aggregation, reduced surface recursion and first-cut normalization to return the probability that two independent first-cut draws have the same unlabelled component collection. Accept genus 0 or 1, one through three positive integer boundary counts and E=sum(marks)+3*b+6*g-6 from 1 through 11. Reject booleans, floating-point counts and other inputs with ValueError. The terminal triangle has no first-cut distribution and is rejected here.

The reduced recursion divides a connected cut-weight sum by d=E-L with L=2*g+b-1, uses reduced triangle weight 1 and multiplies reduced weights across disconnected components. A boundary merge carries ni*nj; ordered genus and separating splits carry n/2, with unstable disks excluded. Aggregate multiplicities before normalization and before squaring. Return the sum of squared normalized outcome probabilities as a native float to absolute error at most 1e-11, without rounding. This is a topology collision, not equality of labeled curves.

Returns
-------
Expected return: float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from math import fsum


def surface_collision(genus, marks):
    """Return the aggregated first-cut topology collision probability."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from math import fsum


def _oracle_surface_collision(genus, marks):
    _, probabilities = _oracle_first_cut_distribution(genus, marks)
    return float(fsum(p * p for p in probabilities))

# =============================================================================
# TEST CASES
# =============================================================================

# Differential gold_call cases below cover normal, boundary, and edge inputs (Appendix A.2).
def test_cases():
    cases = []
    for name, g, marks in [('target', 1, (2, 3)), ('one_outcome', 0, (1, 1)), ('one_boundary_torus', 1, (3,)), ('repeated', 0, (2, 2, 1))]:
        cases.append(dict(name=name, setup=f'g={g}; marks={marks!r}', call='surface_collision(g,marks)', gold_call='_oracle_surface_collision(g,marks)', expected=None, comparator='allclose', atol=1e-11, rtol=0))
    for name, g, marks in [('terminal', 0, (3,)), ('oversized', 1, (99,)), ('boolean', False, (4,))]:
        setup=f'g={g!r}; marks={marks!r}\ndef status(fn):\n    try: fn(g,marks)\n    except ValueError: return 1\n    except Exception: return 2\n    return 0\ndef run_model(): return status(surface_collision)\ndef run_gold(): return status(_oracle_surface_collision)'
        cases.append(dict(name=name, setup=setup, call='run_model()', gold_call='run_gold()', expected=None, comparator='allclose', atol=0, rtol=0))
    return cases
