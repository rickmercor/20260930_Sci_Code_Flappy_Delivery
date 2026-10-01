"""
Enumerate raw separating cuts on a connected orientable unpunctured surface. Accept genus 0 or 1, one through three positive integer boundary counts and E=sum(marks)+3*b+6*g-6 from 0 through 11, with only the triangle allowed at E=0. Reject booleans, floating-point counts and out-of-domain inputs with ValueError. Sort parent marks, retaining repeated entries as distinct indexed boundaries.

For each boundary index i with n marks, enumerate genus h=0 through g, mark split a=0 through n and masks from 0 through 2**(b-1)-1 in that order. A set bit assigns the corresponding remaining boundary to the left component. Append a+1 marks to the left at genus h and n-a+1 to the right at genus g-h. Sort marks within each component and sort the pair of components. Discard the ENTIRE outcome if either component is a genus-zero disk with one or two marks. Append (component_pair,n/2) for every other choice without aggregating duplicate pairs. This ordered enumeration already includes reversal with its half multiplicity; no additional symmetry division is made.

Returns
-------
Expected return: tuple of (pair of component descriptors, numeric multiplicity).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def separating_cuts(genus, marks):
    """Return raw numeric records for the separating-curve family."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_separating_cuts(genus, marks):
    g, b, _, _, _ = _oracle_surface_descriptor(genus, marks)
    out = []
    for i, n in enumerate(b):
        rest = b[:i] + b[i + 1:]
        for h in range(g + 1):
            for a in range(n + 1):
                for mask in range(1 << len(rest)):
                    left = tuple(x for k, x in enumerate(rest) if mask & (1 << k))
                    right = tuple(x for k, x in enumerate(rest) if not mask & (1 << k))
                    pair = tuple(sorted(((h, tuple(sorted(left + (a + 1,)))), (g - h, tuple(sorted(right + (n - a + 1,)))))))
                    if any(part in ((0, (1,)), (0, (2,))) for part in pair):
                        continue
                    out.append((pair, n / 2))
    return tuple(out)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    encode = "\ndef encode(rows):\n    out=[len(rows)]\n    for parts,w in rows:\n        out.extend([w,len(parts)])\n        for g,b in parts: out.extend([g,len(b),*b])\n    return out"
    cases = [
        dict(name='torus_two_boundaries', setup='g=1; marks=(2,3)'+encode, call='encode(separating_cuts(g,marks))', gold_call='encode(_oracle_separating_cuts(g,marks))', expected=None, comparator='allclose', atol=0, rtol=0),
        dict(name='triangle', setup='g=0; marks=(3,)'+encode, call='encode(separating_cuts(g,marks))', gold_call='encode(_oracle_separating_cuts(g,marks))', expected=None, comparator='allclose', atol=0, rtol=0),
        dict(name='square_disk', setup='g=0; marks=(4,)'+encode, call='encode(separating_cuts(g,marks))', gold_call='encode(_oracle_separating_cuts(g,marks))', expected=None, comparator='allclose', atol=0, rtol=0),
        dict(name='repeated_boundaries', setup='g=0; marks=(2,2,1)'+encode, call='encode(separating_cuts(g,marks))', gold_call='encode(_oracle_separating_cuts(g,marks))', expected=None, comparator='allclose', atol=0, rtol=0),
    ]
    setup="def status(fn):\n    try: fn(2,(3,))\n    except ValueError: return 1\n    except Exception: return 2\n    return 0\ndef run_model(): return status(separating_cuts)\ndef run_gold(): return status(_oracle_separating_cuts)"
    cases.append(dict(name='invalid', setup=setup, call='run_model()', gold_call='run_gold()', expected=None, comparator='allclose', atol=0, rtol=0))
    return cases
