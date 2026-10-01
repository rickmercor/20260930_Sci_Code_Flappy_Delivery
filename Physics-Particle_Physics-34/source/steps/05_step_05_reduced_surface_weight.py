"""
Evaluate the reduced tropical weight h for a connected surface in spacetime dimension 2. Accept genus 0 or 1, one through three positive integer boundary counts and E=sum(marks)+3*b+6*g-6 from 0 through 11, with only the triangle allowed at E=0. Invalid counts, booleans and floating-point counts raise ValueError. The loop count is L=2*g+b-1 and the degree is d=E-L. The triangle (0,(3,)) has h=1 even though d=0.

For every other surface let h(S) be the sum over aggregated one-cut outcomes of their multiplicity times the PRODUCT of component reduced weights, divided by d(S). Use the complete multiplicity-preserving cut families: boundary merge ni*nj and ordered genus-lowering or separating splits n/2, discarding any outcome containing an unstable one-mark or two-mark disk. Each child has fewer E, so memoization by canonical connected descriptors terminates at triangles. Return a native float, with absolute or relative numerical error at most 1e-11. Do not replace a disconnected product by a division using its total degree.

Returns
-------
Expected return: float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from fractions import Fraction
from functools import lru_cache


def reduced_surface_weight(genus, marks):
    """Return the connected reduced weight, including the triangle convention."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from fractions import Fraction
from functools import lru_cache


def _oracle_reduced_surface_weight(genus, marks):
    g, b, _, _, _ = _oracle_surface_descriptor(genus, marks)

    @lru_cache(None)
    def visit(g, b):
        if (g, b) == (0, (3,)):
            return Fraction(1)
        degree = sum(b) + 2 * len(b) + 4 * g - 5
        total = Fraction(0)
        for parts, mult in _oracle_aggregate_cuts(g, b):
            term = Fraction(mult)
            for child_g, child_b in parts:
                term *= visit(child_g, child_b)
            total += term
        return total / degree

    return float(visit(g, b))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    cases = [
        dict(name='torus_two_boundaries', setup='g=1; marks=(2,3)', call='reduced_surface_weight(g,marks)', gold_call='_oracle_reduced_surface_weight(g,marks)', expected=None, comparator='allclose', atol=1e-11, rtol=1e-11),
        dict(name='triangle', setup='g=0; marks=(3,)', call='reduced_surface_weight(g,marks)', gold_call='_oracle_reduced_surface_weight(g,marks)', expected=None, comparator='allclose', atol=1e-11, rtol=1e-11),
        dict(name='annulus', setup='g=0; marks=(1,1)', call='reduced_surface_weight(g,marks)', gold_call='_oracle_reduced_surface_weight(g,marks)', expected=None, comparator='allclose', atol=1e-11, rtol=1e-11),
        dict(name='repeated', setup='g=0; marks=(2,2,1)', call='reduced_surface_weight(g,marks)', gold_call='_oracle_reduced_surface_weight(g,marks)', expected=None, comparator='allclose', atol=1e-11, rtol=1e-11),
    ]
    setup="def status(fn):\n    try: fn(0,(0,4))\n    except ValueError: return 1\n    except Exception: return 2\n    return 0\ndef run_model(): return status(reduced_surface_weight)\ndef run_gold(): return status(_oracle_reduced_surface_weight)"
    cases.append(dict(name='invalid', setup=setup, call='run_model()', gold_call='run_gold()', expected=None, comparator='allclose', atol=0, rtol=0))
    return cases
