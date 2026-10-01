"""
Enumerate nonseparating cuts on a connected unpunctured surface. Accept genus 0 or 1, one through three positive integer boundary counts and E=sum(marks)+3*b+6*g-6 from 0 through 11, with only the triangle allowed at E=0. Reject booleans, floating-point counts and inputs outside this domain with ValueError. Return raw records (components, multiplicity), where components is a tuple containing one (genus, sorted_marks) descriptor. Do not aggregate records.

First enumerate unordered distinct boundary index pairs i<j in lexicographic order. Merge ni and nj into ni+nj+2 marks at unchanged genus with multiplicity ni*nj. Then, if g=1, enumerate boundary indices of the sorted parent followed by a=0 through n; replace that boundary by a+1 and n-a+1 on ONE connected surface of genus g-1, with multiplicity n/2. Repeated equal boundary counts remain distinct indexed choices. The half factor compensates ordered reversal and must not be applied again.

Returns
-------
Expected return: tuple of (tuple of component descriptors, numeric multiplicity).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nonseparating_cuts(genus, marks):
    """Return raw numeric records for boundary merging and genus lowering."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_nonseparating_cuts(genus, marks):
    g, b, _, _, _ = _oracle_surface_descriptor(genus, marks)
    out = []
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            rest = tuple(n for k, n in enumerate(b) if k not in (i, j))
            child = (g, tuple(sorted(rest + (b[i] + b[j] + 2,))))
            out.append(((child,), b[i] * b[j]))
    if g:
        for i, n in enumerate(b):
            rest = b[:i] + b[i + 1:]
            for a in range(n + 1):
                child = (g - 1, tuple(sorted(rest + (a + 1, n - a + 1))))
                out.append(((child,), n / 2))
    return tuple(out)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    encode = "\ndef encode(rows):\n    out=[len(rows)]\n    for parts,w in rows:\n        out.extend([w,len(parts)])\n        for g,b in parts: out.extend([g,len(b),*b])\n    return out"
    cases = [
        dict(name='genus_and_merge', setup='g=1; marks=(2,3)'+encode, call='encode(nonseparating_cuts(g,marks))', gold_call='encode(_oracle_nonseparating_cuts(g,marks))', expected=None, comparator='allclose', atol=0, rtol=0),
        dict(name='triangle', setup='g=0; marks=(3,)'+encode, call='encode(nonseparating_cuts(g,marks))', gold_call='encode(_oracle_nonseparating_cuts(g,marks))', expected=None, comparator='allclose', atol=0, rtol=0),
        dict(name='repeated', setup='g=0; marks=(1,1,2)'+encode, call='encode(nonseparating_cuts(g,marks))', gold_call='encode(_oracle_nonseparating_cuts(g,marks))', expected=None, comparator='allclose', atol=0, rtol=0),
    ]
    setup="def status(fn):\n    try: fn(0,(True,))\n    except ValueError: return 1\n    except Exception: return 2\n    return 0\ndef run_model(): return status(nonseparating_cuts)\ndef run_gold(): return status(_oracle_nonseparating_cuts)"
    cases.append(dict(name='invalid', setup=setup, call='run_model()', gold_call='run_gold()', expected=None, comparator='allclose', atol=0, rtol=0))
    return cases
