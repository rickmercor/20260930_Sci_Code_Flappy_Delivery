"""
Return (total_weight, probabilities) for the aggregated first-cut topology distribution, with probabilities in lexicographic component-descriptor order. Accept genus 0 or 1, one through three positive integer boundary mark counts and E=sum(marks)+3*b+6*g-6 from 1 through 11. Booleans, floating-point counts and other inputs raise ValueError; the E=0 triangle is not a first-cut query.

For each outcome from the multiplicity-preserving cut enumeration, its weight is its multiplicity times the product of the connected component reduced weights. Sum these weights to obtain the parent normalizer and divide each weight by that sum. The reduced recursion uses degree d=E-(2*g+b-1), divides each connected nonterminal sum by d and assigns reduced weight 1 to the triangle. The product convention on disconnected outcomes is essential. Preserve all outcomes, including those with identical boundary counts on separate components, and return native floats with absolute or relative error at most 1e-11.

Returns
-------
Expected return: (float, tuple[float, ...]).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from math import fsum


def first_cut_distribution(genus, marks):
    """Return the total first-cut weight and its normalized probability tuple."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from math import fsum


def _oracle_first_cut_distribution(genus, marks):
    g, b, edges, _, _ = _oracle_surface_descriptor(genus, marks)
    if edges == 0:
        raise ValueError('terminal surface has no first cut')
    weights = []
    for parts, mult in _oracle_aggregate_cuts(g, b):
        weight = float(mult)
        for child_g, child_b in parts:
            weight *= _oracle_reduced_surface_weight(child_g, child_b)
        weights.append(weight)
    total = fsum(weights)
    return total, tuple(weight / total for weight in weights)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    cases = [
        dict(name='two_boundaries', setup='g=1; marks=(2,3)', call='(lambda r:(r[0],*r[1]))(first_cut_distribution(g,marks))', gold_call='(lambda r:(r[0],*r[1]))(_oracle_first_cut_distribution(g,marks))', expected=None, comparator='allclose', atol=1e-11, rtol=1e-11),
        dict(name='one_cut_annulus', setup='g=0; marks=(1,1)', call='(lambda r:(r[0],*r[1]))(first_cut_distribution(g,marks))', gold_call='(lambda r:(r[0],*r[1]))(_oracle_first_cut_distribution(g,marks))', expected=None, comparator='allclose', atol=1e-11, rtol=1e-11),
        dict(name='one_boundary', setup='g=1; marks=(3,)', call='(lambda r:(r[0],*r[1]))(first_cut_distribution(g,marks))', gold_call='(lambda r:(r[0],*r[1]))(_oracle_first_cut_distribution(g,marks))', expected=None, comparator='allclose', atol=1e-11, rtol=1e-11),
        dict(name='repeated', setup='g=0; marks=(2,2,1)', call='(lambda r:(r[0],*r[1]))(first_cut_distribution(g,marks))', gold_call='(lambda r:(r[0],*r[1]))(_oracle_first_cut_distribution(g,marks))', expected=None, comparator='allclose', atol=1e-11, rtol=1e-11),
    ]
    setup="def status(fn):\n    try: fn(0,(3,))\n    except ValueError: return 1\n    except Exception: return 2\n    return 0\ndef run_model(): return status(first_cut_distribution)\ndef run_gold(): return status(_oracle_first_cut_distribution)"
    cases.append(dict(name='terminal_rejected', setup=setup, call='run_model()', gold_call='run_gold()', expected=None, comparator='allclose', atol=0, rtol=0))
    return cases
