"""
A connected orientable unpunctured surface is specified by genus g and positive boundary mark counts. For cubic scalar ribbon graphs, its triangulation has E=sum(marks)+3*b+6*g-6 arcs and L=2*g+b-1 loops, where b counts boundaries. At spacetime dimension 2 its degree is d=E-L. Return (g, sorted_marks, E, L, d), preserving repeated boundary counts. The finite domain has g in {0,1}, one through three boundaries and E from 0 through 11. The only allowed E=0 surface is the triangle (0,(3,)); disks with one or two marks are unstable. Accept integer scalars but not booleans or floating-point values, and raise ValueError outside the domain.

Canonical descriptors identify unlabelled topology without collapsing repeated boundary components. The triangle represents an empty triangulation and supplies the terminal reduced weight rather than a first-cut draw.

Returns
-------
Expected return: (int, tuple[int, ...], int, int, int).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral


def surface_descriptor(genus, marks):
    """Return the canonical numeric surface descriptor for the stated finite domain."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from numbers import Integral


def _oracle_surface_descriptor(genus, marks):
    if isinstance(genus, bool) or not isinstance(genus, Integral) or genus not in (0, 1):
        raise ValueError('genus outside domain')
    try:
        values = tuple(marks)
    except (TypeError, ValueError):
        raise ValueError('marks must be an integer sequence') from None
    if not 1 <= len(values) <= 3:
        raise ValueError('boundary count outside domain')
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in values):
        raise ValueError('invalid mark count')
    g = int(genus)
    b = tuple(sorted(int(n) for n in values))
    edges = sum(b) + 3 * len(b) + 6 * g - 6
    loops = 2 * g + len(b) - 1
    degree = edges - loops
    if not 0 <= edges <= 11 or (edges == 0 and (g, b) != (0, (3,))) or (edges > 0 and degree <= 0):
        raise ValueError('surface outside domain')
    return g, b, edges, loops, degree

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    cases = [
        dict(name='two_boundaries', setup='g=1; marks=(3,2)', call='(lambda r:(r[0],*r[1],*r[2:]))(surface_descriptor(g,marks))', gold_call='(lambda r:(r[0],*r[1],*r[2:]))(_oracle_surface_descriptor(g,marks))', expected=None, comparator='allclose', atol=0, rtol=0),
        dict(name='triangle', setup='g=0; marks=(3,)', call='(lambda r:(r[0],*r[1],*r[2:]))(surface_descriptor(g,marks))', gold_call='(lambda r:(r[0],*r[1],*r[2:]))(_oracle_surface_descriptor(g,marks))', expected=None, comparator='allclose', atol=0, rtol=0),
        dict(name='repeated_boundaries', setup='g=0; marks=(2,2,1)', call='(lambda r:(r[0],*r[1],*r[2:]))(surface_descriptor(g,marks))', gold_call='(lambda r:(r[0],*r[1],*r[2:]))(_oracle_surface_descriptor(g,marks))', expected=None, comparator='allclose', atol=0, rtol=0),
    ]
    for name, g, marks in [('boolean_genus', True, (3,)), ('unstable', 0, (2,)), ('huge_mark', 1, (10**100,)), ('float_mark', 0, (4.0,)), ('empty', 0, ())]:
        setup = f'g={g!r}; marks={marks!r}\ndef status(fn):\n    try: fn(g,marks)\n    except ValueError: return 1\n    except Exception: return 2\n    return 0\ndef run_model(): return status(surface_descriptor)\ndef run_gold(): return status(_oracle_surface_descriptor)'
        cases.append(dict(name=name, setup=setup, call='run_model()', gold_call='run_gold()', expected=None, comparator='allclose', atol=0, rtol=0))
    return cases
