"""
Construct the exact tridiagonal block factors and round their completed entries once to binary16.

The diagonal multiplier is used only in constructing the primary preconditioner. Each four-row block is tridiagonal, so natural-order ILU(0) introduces no discarded fill. Its pivots are not rounded during elimination.



For each block, u[0] = lambda*a[0]; l[k] = (-3/16)/u[k-1]; u[k] = lambda*a[k] - 21/(256*u[k-1]), k = 1, 2, 3. Evaluate these recurrences exactly, then round each completed entry once to fp16.

Returns
-------
tuple: (lower, upper), two binary16 arrays of shape (number_of_blocks, 4).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_block_factors(diagonal: "np.ndarray", multiplier: tuple) -> tuple:
    """Return the stored factors of the benchmark's four-row blocks.

    Parameters
    ----------
    diagonal : np.ndarray
        One-dimensional real finite binary64 diagonal, with positive length
        divisible by four and entries in [1, 2]. Its binary64 values are
        interpreted as exact rationals. Each contiguous four entries defines
        one tridiagonal block with subdiagonal -3/16 and superdiagonal -7/16.
    multiplier : tuple
        Two Python integers (numerator, denominator), denominator positive,
        representing an exact value in [1, 2]. Fractions need not be reduced.
        Multiply only the block diagonals by this value. Factor exactly in
        natural order with unit lower diagonals, then round each completed
        factor entry once to IEEE binary16, nearest with ties to even.

    Returns
    -------
    lower, upper : tuple of np.ndarray
        Binary16 arrays of shape (len(diagonal)//4, 4). lower[:, 0] is zero;
        lower[:, 1:] holds the three strict lower subdiagonal entries.
        upper holds the four upper diagonal entries. Unit lower diagonals
        and the upper superdiagonal -7/16 are implicit. No input is changed.

    Raises
    ------
    ValueError
        If diagonal is not a nonempty real finite vector of length divisible
        by four with entries in [1, 2], or multiplier is not an integer pair
        with positive denominator and value in [1, 2].
    """
    return lower, upper  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from fractions import Fraction as _Fraction
import numpy as np

def _f3r_fraction(pair):
    if (not isinstance(pair, (tuple, list)) or len(pair) != 2
            or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in pair)
            or int(pair[1]) <= 0):
        raise ValueError("Expected an integer numerator and positive integer denominator")
    return _Fraction(int(pair[0]), int(pair[1]))

def _f3r_pair(value):
    value = _Fraction(value)
    return (value.numerator, value.denominator)

def _f3r_diagonal(diagonal):
    try:
        raw = np.asarray(diagonal)
        if raw.dtype.kind not in 'fiu':
            raise ValueError("diagonal must be real numeric data")
        a = np.asarray(raw, dtype=np.float64)
    except (TypeError, OverflowError) as exc:
        raise ValueError("Invalid diagonal") from exc
    if a.ndim != 1 or a.size == 0 or a.size % 4 or not np.isfinite(a).all() or np.any(a < 1) or np.any(a > 2):
        raise ValueError("diagonal must have length 4*b and entries in [1,2]")
    return a

def _f3r_half_exact(value):
    # The seed conversion locates neighbours; exact distances decide rounding.
    value = _Fraction(value)
    seed = np.float16(float(value))
    candidates = (np.nextafter(seed, np.float16(-np.inf), dtype=np.float16),
                  seed, np.nextafter(seed, np.float16(np.inf), dtype=np.float16))
    distances = [abs(value - _Fraction(float(x))) for x in candidates]
    best = min(distances)
    tied = [x for x, dist in zip(candidates, distances) if dist == best]
    return next((x for x in tied if int(np.asarray(x).view(np.uint16)) % 2 == 0), tied[0])

def _oracle_build_block_factors(diagonal: "np.ndarray", multiplier: tuple) -> tuple:
    a = _f3r_diagonal(diagonal)
    lam = _f3r_fraction(multiplier)
    if not 1 <= lam <= 2:
        raise ValueError("multiplier must lie in [1,2]")
    lower = np.zeros((a.size // 4, 4), dtype=np.float16)
    upper = np.empty_like(lower)
    for block in range(a.size // 4):
        previous = None
        for k in range(4):
            pivot = lam * _Fraction(float(a[4 * block + k]))
            if k:
                sub = _Fraction(-3, 16) / previous
                lower[block, k] = _f3r_half_exact(sub)
                pivot -= sub * _Fraction(-7, 16)
            upper[block, k] = _f3r_half_exact(pivot)
            previous = pivot
    return lower, upper

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and declared-invalid test specifications."""
    return [
        # normal: distinct block types
        {'setup': '\n'
                  'diagonal=np.array([1.000244140625, 1.00048828125, 1.000732421875, 1.000244140625, 1.00048828125, '
                  '1.000732421875, 1.000244140625, 1.00048828125], dtype=np.float64).reshape((8,))\n'
                  'multiplier=(9, 8)\n'
                  'expected=(np.array([[0.0, -0.1666259765625, -0.1781005859375, -0.178955078125], [0.0, '
                  '-0.1666259765625, -0.1781005859375, -0.1790771484375]], dtype=np.float16).reshape((2, 4)), '
                  'np.array([[1.125, 1.052734375, 1.0478515625, 1.046875], [1.1259765625, 1.052734375, 1.0478515625, '
                  '1.046875]], dtype=np.float16).reshape((2, 4)))\n'
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
         'call': '_case_check(build_block_factors(diagonal,multiplier))',
         'gold_call': '_case_check(_oracle_build_block_factors(diagonal,multiplier))'},
        # boundary: lower domain endpoint
        {'setup': '\n'
                  'diagonal=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64).reshape((4,))\n'
                  'multiplier=(1, 1)\n'
                  'expected=(np.array([[0.0, -0.1875, -0.2042236328125, -0.2059326171875]], '
                  'dtype=np.float16).reshape((1, 4)), np.array([[1.0, 0.91796875, 0.91064453125, 0.91015625]], '
                  'dtype=np.float16).reshape((1, 4)))\n'
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
         'call': '_case_check(build_block_factors(diagonal,multiplier))',
         'gold_call': '_case_check(_oracle_build_block_factors(diagonal,multiplier))'},
        # edge: binary16 ties-to-even diagonal
        {'setup': '\n'
                  'diagonal=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64).reshape((4,))\n'
                  'multiplier=(2049, 2048)\n'
                  'expected=(np.array([[0.0, -0.1873779296875, -0.2041015625, -0.205810546875]], '
                  'dtype=np.float16).reshape((1, 4)), np.array([[1.0, 0.91845703125, 0.9111328125, 0.91064453125]], '
                  'dtype=np.float16).reshape((1, 4)))\n'
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
         'call': '_case_check(build_block_factors(diagonal,multiplier))',
         'gold_call': '_case_check(_oracle_build_block_factors(diagonal,multiplier))'},
        # boundary: upper input endpoints
        {'setup': '\n'
                  'diagonal=np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64).reshape((4,))\n'
                  'multiplier=(2, 1)\n'
                  'expected=(np.array([[0.0, -0.046875, -0.047119140625, -0.047119140625]], '
                  'dtype=np.float16).reshape((1, 4)), np.array([[4.0, 3.98046875, 3.978515625, 3.978515625]], '
                  'dtype=np.float16).reshape((1, 4)))\n'
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
         'call': '_case_check(build_block_factors(diagonal,multiplier))',
         'gold_call': '_case_check(_oracle_build_block_factors(diagonal,multiplier))'},
        # invalid: zero denominator
        {'setup': '\n'
                  'diagonal=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64).reshape((4,))\n'
                  'multiplier=(1, 0)\n'
                  'def _case_raise(fn):\n'
                  '    try:\n'
                  '        fn()\n'
                  '        return 0\n'
                  '    except ValueError:\n'
                  '        return 1\n'
                  '    except Exception:\n'
                  '        return 2',
         'call': '_case_raise(lambda: build_block_factors(diagonal,multiplier))',
         'gold_call': '_case_raise(lambda: _oracle_build_block_factors(diagonal,multiplier))'},
    ]
