"""
Combine the raw nonseparating and separating curve families into unlabelled cut outcomes. Accept genus 0 or 1, one through three positive integer boundary counts and E=sum(marks)+3*b+6*g-6 from 0 through 11, with only the triangle allowed at E=0. Invalid counts, booleans and floating-point counts raise ValueError. Each output record is (components,multiplicity), sorted lexicographically by components. Each component is (genus,sorted_marks), and the components form an unordered multiset represented by a sorted tuple.

Boundary merging has multiplicity ni*nj. Genus lowering and separating cuts use ordered mark splits with multiplicity n/2. Add multiplicities of EVERY raw record with the same canonical component collection, including records from different indexed boundary choices. Do not identify repeated boundaries before enumeration and do not divide by a component automorphism after aggregation. The source cut families discard outcomes containing unstable one-mark or two-mark disks. Return an empty tuple for the terminal triangle.

Returns
-------
Expected return: tuple of (tuple of numeric component descriptors, numeric multiplicity).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def aggregate_cuts(genus, marks):
    """Return sorted unlabelled cut records with summed multiplicities."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_aggregate_cuts(genus, marks):
    totals = {}
    for parts, weight in _oracle_nonseparating_cuts(genus, marks) + _oracle_separating_cuts(genus, marks):
        key = tuple(sorted((g, tuple(sorted(b))) for g, b in parts))
        totals[key] = totals.get(key, 0) + weight
    return tuple(sorted(totals.items()))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    encode = "\ndef encode(rows):\n    out=[len(rows)]\n    for parts,w in rows:\n        out.extend([w,len(parts)])\n        for g,b in parts: out.extend([g,len(b),*b])\n    return out"
    cases = [
        dict(name='two_boundaries', setup='g=1; marks=(2,3)'+encode, call='encode(aggregate_cuts(g,marks))', gold_call='encode(_oracle_aggregate_cuts(g,marks))', expected=None, comparator='allclose', atol=0, rtol=0),
        dict(name='triangle', setup='g=0; marks=(3,)'+encode, call='encode(aggregate_cuts(g,marks))', gold_call='encode(_oracle_aggregate_cuts(g,marks))', expected=None, comparator='allclose', atol=0, rtol=0),
        dict(name='repeated', setup='g=0; marks=(2,2,1)'+encode, call='encode(aggregate_cuts(g,marks))', gold_call='encode(_oracle_aggregate_cuts(g,marks))', expected=None, comparator='allclose', atol=0, rtol=0),
        dict(name='one_boundary', setup='g=1; marks=(3,)'+encode, call='encode(aggregate_cuts(g,marks))', gold_call='encode(_oracle_aggregate_cuts(g,marks))', expected=None, comparator='allclose', atol=0, rtol=0),
    ]
    setup="def status(fn):\n    try: fn(0,())\n    except ValueError: return 1\n    except Exception: return 2\n    return 0\ndef run_model(): return status(aggregate_cuts)\ndef run_gold(): return status(_oracle_aggregate_cuts)"
    cases.append(dict(name='invalid', setup=setup, call='run_model()', gold_call='run_gold()', expected=None, comparator='allclose', atol=0, rtol=0))
    return cases
