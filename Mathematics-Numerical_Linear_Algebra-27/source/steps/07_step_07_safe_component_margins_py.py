"""
Find connected safe multiplier components and enclose their relative uncertainty margins.

Safety must hold at every multiplier, including isolated rounding boundaries. Consecutive safe open intervals remain disconnected when the separating point is unsafe. Component boundaries determine the supremal margin; endpoint closure separately determines whether that limiting margin is attained.



For safe-component limiting endpoints a and b, margin_ppm = 1e6*(b-a)/(b+a). Endpoint inclusion is recorded separately. A singleton component has zero margin. An unsafe singleton prevents joining its adjacent open intervals.

Returns
-------
np.ndarray: component descriptors with binary64 lower and upper bounds on each margin in parts per million.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def safe_component_margins(boundaries: tuple, strata: "np.ndarray", gains: "np.ndarray",
                          threshold: float = 2.0) -> "np.ndarray":
    """Return all safe components with certified endpoint-derived margins.

    Parameters
    ----------
    boundaries : tuple
        Numeric boundary descriptors from factor_rounding_regions. Each is
        (integer_polynomial_coefficients, rational_low, rational_high).
        Brackets lie in [1,2], are in strictly increasing disjoint order,
        and have width <= 1e-24. A rational singleton bracket is allowed.
        Polynomials identify the boundaries; numerical enclosures below use
        the supplied brackets, which must isolate those boundaries.
    strata : np.ndarray
        Integer array from factor_rounding_regions, shape (2*b-1,3): all
        singleton and open-interval rows in alternating spatial order, with
        valid nonnegative stored-state IDs.
    gains : np.ndarray
        Finite nonnegative binary64 array of shape (number_of_states,k),
        k >= 1. Entry (s,j) is the worst matching fixed/adaptive residual
        ratio for state s and configuration j. Use threshold comparisons
        without preliminary rounding; equality is safe.
    threshold : float
        Positive finite real number.

    Returns
    -------
    components : np.ndarray
        Binary64 shape (number_of_components,7), sorted first by configuration
        column, then by left boundary. Row fields are configuration_id,
        left_boundary_id, right_boundary_id, left_closed, right_closed,
        lower_margin_ppm, upper_margin_ppm. Closure flags are 0 or 1.
        Margins enclose 1e6*(b-a)/(b+a) by exact rational endpoint arithmetic
        followed by outward-rounded binary64 bounds. A safe singleton has
        zero margin. A component's supremum is attained exactly when both
        endpoints are included. No safe states gives shape (0,7).

    Raises
    ------
    ValueError
        If descriptors are malformed, brackets are outside [1,2], reversed,
        overlapping, or wider than 1e-24; strata do not form the complete
        alternating partition; gains have invalid shape/entries or missing
        state IDs; or threshold is not finite and positive. Polynomial-root
        validity is a precondition provided by factor_rounding_regions.
    """
    return components  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _f3r_outward(value, direction):
    v = float(value)
    rational_v = _Fraction(v)
    if direction < 0 and rational_v > value:
        v = float(np.nextafter(v, -np.inf))
    elif direction > 0 and rational_v < value:
        v = float(np.nextafter(v, np.inf))
    return v

def _oracle_safe_component_margins(boundaries: tuple, strata: "np.ndarray", gains: "np.ndarray",
                                  threshold: float = 2.0) -> "np.ndarray":
    if not isinstance(boundaries, (tuple, list)) or len(boundaries) == 0:
        raise ValueError("At least one boundary is required")
    brackets = []
    for desc in boundaries:
        if not isinstance(desc, (tuple, list)) or len(desc) != 3:
            raise ValueError("Invalid boundary descriptor")
        coeff, lp, hp = desc
        if (not isinstance(coeff, (tuple, list)) or len(coeff) < 2
                or any(isinstance(c, bool) or not isinstance(c, (int, np.integer)) for c in coeff)
                or coeff[0] <= 0):
            raise ValueError("Invalid integer boundary polynomial")
        lo, hi = _f3r_fraction(lp), _f3r_fraction(hp)
        if not 1 <= lo <= hi <= 2 or hi - lo > _Fraction(1, 10**24):
            raise ValueError("Invalid boundary bracket")
        if brackets and brackets[-1][1] >= lo:
            raise ValueError("Boundary brackets must be disjoint and ordered")
        brackets.append((lo, hi))
    st = np.asarray(strata)
    g = _f3r_real_array(gains, 'gains')
    nb = len(brackets)
    if st.dtype.kind not in 'iu' or st.shape != (2*nb-1,3):
        raise ValueError("Invalid stratum structure")
    expected = [(i,i) if j % 2 == 0 else (i,i+1) for j in range(2*nb-1) for i in [j//2]]
    if any(tuple(row[:2]) != exp for row, exp in zip(st, expected)):
        raise ValueError("Strata must alternate all points and open intervals")
    if g.ndim != 2 or g.shape[0] < 1 or g.shape[1] < 1 or np.any(g < 0) or np.any(st[:,2] < 0) or np.any(st[:,2] >= g.shape[0]):
        raise ValueError("Invalid gain table or state IDs")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("threshold must be positive and finite") from exc
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    result = []
    def _append(configuration, start, stop):
        li, ri = int(st[start,0]), int(st[stop,1])
        lc, rc = int(start % 2 == 0), int(stop % 2 == 0)
        if li == ri:
            qlow = qhigh = _Fraction(0)
        else:
            al, au = brackets[li]
            bl, bu = brackets[ri]
            qlow = 10**6 * (bl-au)/(bl+au)
            qhigh = 10**6 * (bu-al)/(bu+al)
        result.append((configuration,li,ri,lc,rc,_f3r_outward(qlow,-1),_f3r_outward(qhigh,1)))
    for j in range(g.shape[1]):
        first = None
        for i, row in enumerate(st):
            safe = g[int(row[2]),j] >= threshold
            if safe and first is None:
                first = i
            elif not safe and first is not None:
                _append(j,first,i-1)
                first = None
        if first is not None:
            _append(j,first,len(st)-1)
    return np.asarray(result, dtype=np.float64).reshape(-1,7)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and declared-invalid test specifications."""
    return [
        # normal: multiple configurations and endpoint closure
        {'setup': '\n'
                  'boundaries=(((1, -1), (1, 1), (1, 1)), ((4, -5), (5, 4), (5, 4)), ((2, -3), (3, 2), (3, 2)), ((1, '
                  '-2), (2, 1), (2, 1)))\n'
                  'strata=np.array([[0, 0, 0], [0, 1, 1], [1, 1, 2], [1, 2, 3], [2, 2, 4], [2, 3, 5], [3, 3, 6]], '
                  'dtype=np.int64).reshape((7, 3))\n'
                  'gains=np.array([[0.0, 3.0], [3.0, 3.0], [3.0, 0.0], [3.0, 3.0], [3.0, 3.0], [0.0, 3.0], [0.0, '
                  '3.0]], dtype=np.float64).reshape((7, 2))\n'
                  'expected=np.array([[0.0, 0.0, 2.0, 0.0, 1.0, 200000.0, 200000.0], [1.0, 0.0, 1.0, 1.0, 0.0, '
                  '111111.11111111111, 111111.11111111112], [1.0, 1.0, 3.0, 0.0, 1.0, 230769.23076923075, '
                  '230769.23076923078]], dtype=np.float64).reshape((3, 7))\n'
                  'import numpy as np\n'
                  '\n'
                  'def _case_equal(actual, expected):\n'
                  '    if isinstance(expected, np.ndarray):\n'
                  '        if not isinstance(actual, np.ndarray) or actual.shape != expected.shape or actual.dtype '
                  '!= expected.dtype:\n'
                  '            return False\n'
                  "        if expected.dtype in (np.dtype('float16'),np.dtype('float32')) or expected.dtype.kind in "
                  "'iu':\n"
                  '            return np.array_equal(actual,expected)\n'
                  '        return bool(np.isfinite(actual).all() and '
                  'np.allclose(actual,expected,rtol=2e-12,atol=1e-15))\n'
                  '    if isinstance(expected,tuple):\n'
                  '        return isinstance(actual,tuple) and len(actual)==len(expected) and all(_case_equal(a,e) '
                  'for a,e in zip(actual,expected))\n'
                  '    if isinstance(expected,(int,np.integer)):\n'
                  '        return isinstance(actual,(int,np.integer)) and not isinstance(actual,(bool,np.bool_)) and '
                  'int(actual)==int(expected)\n'
                  '    return actual==expected\n'
                  '\n'
                  'def _case_check(actual):\n'
                  '    try:\n'
                  '        return int(_case_equal(actual,expected))\n'
                  '    except (TypeError,ValueError,OverflowError,IndexError):\n'
                  '        return 0',
         'call': '_case_check(safe_component_margins(boundaries,strata,gains))',
         'gold_call': '_case_check(_oracle_safe_component_margins(boundaries,strata,gains))'},
        # edge: isolated unsafe point disconnects passing intervals
        {'setup': '\n'
                  'boundaries=(((1, -1), (1, 1), (1, 1)), ((4, -5), (5, 4), (5, 4)), ((2, -3), (3, 2), (3, 2)), ((1, '
                  '-2), (2, 1), (2, 1)))\n'
                  'strata=np.array([[0, 0, 0], [0, 1, 1], [1, 1, 2], [1, 2, 3], [2, 2, 4], [2, 3, 5], [3, 3, 6]], '
                  'dtype=np.int64).reshape((7, 3))\n'
                  'gains=np.array([[3.0], [3.0], [0.0], [3.0], [3.0], [3.0], [3.0]], dtype=np.float64).reshape((7, '
                  '1))\n'
                  'expected=np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 111111.11111111111, 111111.11111111112], [0.0, 1.0, '
                  '3.0, 0.0, 1.0, 230769.23076923075, 230769.23076923078]], dtype=np.float64).reshape((2, 7))\n'
                  'import numpy as np\n'
                  '\n'
                  'def _case_equal(actual, expected):\n'
                  '    if isinstance(expected, np.ndarray):\n'
                  '        if not isinstance(actual, np.ndarray) or actual.shape != expected.shape or actual.dtype '
                  '!= expected.dtype:\n'
                  '            return False\n'
                  "        if expected.dtype in (np.dtype('float16'),np.dtype('float32')) or expected.dtype.kind in "
                  "'iu':\n"
                  '            return np.array_equal(actual,expected)\n'
                  '        return bool(np.isfinite(actual).all() and '
                  'np.allclose(actual,expected,rtol=2e-12,atol=1e-15))\n'
                  '    if isinstance(expected,tuple):\n'
                  '        return isinstance(actual,tuple) and len(actual)==len(expected) and all(_case_equal(a,e) '
                  'for a,e in zip(actual,expected))\n'
                  '    if isinstance(expected,(int,np.integer)):\n'
                  '        return isinstance(actual,(int,np.integer)) and not isinstance(actual,(bool,np.bool_)) and '
                  'int(actual)==int(expected)\n'
                  '    return actual==expected\n'
                  '\n'
                  'def _case_check(actual):\n'
                  '    try:\n'
                  '        return int(_case_equal(actual,expected))\n'
                  '    except (TypeError,ValueError,OverflowError,IndexError):\n'
                  '        return 0',
         'call': '_case_check(safe_component_margins(boundaries,strata,gains))',
         'gold_call': '_case_check(_oracle_safe_component_margins(boundaries,strata,gains))'},
        # boundary: equality-safe singleton has zero margin
        {'setup': '\n'
                  'boundaries=(((1, -1), (1, 1), (1, 1)), ((4, -5), (5, 4), (5, 4)), ((2, -3), (3, 2), (3, 2)), ((1, '
                  '-2), (2, 1), (2, 1)))\n'
                  'strata=np.array([[0, 0, 0], [0, 1, 1], [1, 1, 2], [1, 2, 3], [2, 2, 4], [2, 3, 5], [3, 3, 6]], '
                  'dtype=np.int64).reshape((7, 3))\n'
                  'gains=np.array([[0.0], [0.0], [2.0], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64).reshape((7, '
                  '1))\n'
                  'expected=np.array([[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]], dtype=np.float64).reshape((1, 7))\n'
                  'import numpy as np\n'
                  '\n'
                  'def _case_equal(actual, expected):\n'
                  '    if isinstance(expected, np.ndarray):\n'
                  '        if not isinstance(actual, np.ndarray) or actual.shape != expected.shape or actual.dtype '
                  '!= expected.dtype:\n'
                  '            return False\n'
                  "        if expected.dtype in (np.dtype('float16'),np.dtype('float32')) or expected.dtype.kind in "
                  "'iu':\n"
                  '            return np.array_equal(actual,expected)\n'
                  '        return bool(np.isfinite(actual).all() and '
                  'np.allclose(actual,expected,rtol=2e-12,atol=1e-15))\n'
                  '    if isinstance(expected,tuple):\n'
                  '        return isinstance(actual,tuple) and len(actual)==len(expected) and all(_case_equal(a,e) '
                  'for a,e in zip(actual,expected))\n'
                  '    if isinstance(expected,(int,np.integer)):\n'
                  '        return isinstance(actual,(int,np.integer)) and not isinstance(actual,(bool,np.bool_)) and '
                  'int(actual)==int(expected)\n'
                  '    return actual==expected\n'
                  '\n'
                  'def _case_check(actual):\n'
                  '    try:\n'
                  '        return int(_case_equal(actual,expected))\n'
                  '    except (TypeError,ValueError,OverflowError,IndexError):\n'
                  '        return 0',
         'call': '_case_check(safe_component_margins(boundaries,strata,gains))',
         'gold_call': '_case_check(_oracle_safe_component_margins(boundaries,strata,gains))'},
        # boundary: empty safe set
        {'setup': '\n'
                  'boundaries=(((1, -1), (1, 1), (1, 1)), ((4, -5), (5, 4), (5, 4)), ((2, -3), (3, 2), (3, 2)), ((1, '
                  '-2), (2, 1), (2, 1)))\n'
                  'strata=np.array([[0, 0, 0], [0, 1, 1], [1, 1, 2], [1, 2, 3], [2, 2, 4], [2, 3, 5], [3, 3, 6]], '
                  'dtype=np.int64).reshape((7, 3))\n'
                  'gains=np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64).reshape((7, '
                  '1))\n'
                  'expected=np.array([], dtype=np.float64).reshape((0, 7))\n'
                  'import numpy as np\n'
                  '\n'
                  'def _case_equal(actual, expected):\n'
                  '    if isinstance(expected, np.ndarray):\n'
                  '        if not isinstance(actual, np.ndarray) or actual.shape != expected.shape or actual.dtype '
                  '!= expected.dtype:\n'
                  '            return False\n'
                  "        if expected.dtype in (np.dtype('float16'),np.dtype('float32')) or expected.dtype.kind in "
                  "'iu':\n"
                  '            return np.array_equal(actual,expected)\n'
                  '        return bool(np.isfinite(actual).all() and '
                  'np.allclose(actual,expected,rtol=2e-12,atol=1e-15))\n'
                  '    if isinstance(expected,tuple):\n'
                  '        return isinstance(actual,tuple) and len(actual)==len(expected) and all(_case_equal(a,e) '
                  'for a,e in zip(actual,expected))\n'
                  '    if isinstance(expected,(int,np.integer)):\n'
                  '        return isinstance(actual,(int,np.integer)) and not isinstance(actual,(bool,np.bool_)) and '
                  'int(actual)==int(expected)\n'
                  '    return actual==expected\n'
                  '\n'
                  'def _case_check(actual):\n'
                  '    try:\n'
                  '        return int(_case_equal(actual,expected))\n'
                  '    except (TypeError,ValueError,OverflowError,IndexError):\n'
                  '        return 0',
         'call': '_case_check(safe_component_margins(boundaries,strata,gains))',
         'gold_call': '_case_check(_oracle_safe_component_margins(boundaries,strata,gains))'},
        # invalid: missing endpoint stratum
        {'setup': '\n'
                  'boundaries=(((1, -1), (1, 1), (1, 1)), ((4, -5), (5, 4), (5, 4)), ((2, -3), (3, 2), (3, 2)), ((1, '
                  '-2), (2, 1), (2, 1)))\n'
                  'strata=np.array([[0, 0, 0], [0, 1, 1], [1, 1, 2], [1, 2, 3], [2, 2, 4], [2, 3, 5]], '
                  'dtype=np.int64).reshape((6, 3))\n'
                  'gains=np.array([[3.0], [3.0], [3.0], [3.0], [3.0], [3.0], [3.0]], dtype=np.float64).reshape((7, '
                  '1))\n'
                  'def _case_raise(fn):\n'
                  '    try:\n'
                  '        fn()\n'
                  '        return 0\n'
                  '    except ValueError:\n'
                  '        return 1\n'
                  '    except Exception:\n'
                  '        return 2',
         'call': '_case_raise(lambda: safe_component_margins(boundaries,strata,gains))',
         'gold_call': '_case_raise(lambda: _oracle_safe_component_margins(boundaries,strata,gains))'},
    ]
