"""
Apply the stored block factors in the requested arithmetic precision, using two triangular solves.

The same stored binary16 factors are used in both ordinary Richardson corrections and local-weight calculations. In the latter, the factors and residual are promoted before solving in binary32. Casting an already completed binary16 solve to binary32 is not the same operation.



The primary action solves L*y = rhs followed by U*z = y. The unit diagonal of L and the superdiagonal -7/16 of U are implicit.

Returns
-------
np.ndarray: primary-preconditioner action in binary16, binary32, or binary64 as requested.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def apply_primary(lower: "np.ndarray", upper: "np.ndarray", rhs: "np.ndarray", precision: int) -> "np.ndarray":
    """Apply the benchmark's stored block-Jacobi inverse action.

    Parameters
    ----------
    lower, upper : np.ndarray
        Real finite arrays with common shape (b, 4), b >= 1, exactly
        representable in binary16. lower[:, 0] must be zero and every upper
        entry must be nonzero. Unit lower diagonals and the upper
        superdiagonal -7/16 are implicit; see build_block_factors.
    rhs : np.ndarray
        Finite real vector of length 4*b, representable as finite values in
        the requested precision. Cast before either triangular solve.
    precision : int
        16, 32, or 64. Perform each scalar product, subtraction and division
        in that format, nearest-even, with subnormals retained and no FMA.
        The lower solve proceeds in increasing row order; the upper solve
        in decreasing row order. Do not form an inverse or round just once
        at the end. The factors remain stored binary16 values in all cases.

    Returns
    -------
    result : np.ndarray
        Vector of length 4*b in the requested floating-point dtype.
        Inputs are not modified. A zero rhs returns a zero vector.

    Raises
    ------
    ValueError
        If factor shapes or values violate the stated contract, precision
        is not 16, 32, or 64, rhs has the wrong shape or non-real/nonfinite
        entries, its cast is nonfinite, or a computed result is nonfinite.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from functools import lru_cache as _lru_cache
import numpy as np

def _f3r_real_array(value, name):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in 'fiu':
            raise ValueError(name + " must contain real numbers")
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, OverflowError) as exc:
        raise ValueError(name + " must contain real numbers") from exc
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite")
    return result

def _f3r_factor_arrays(lower, upper):
    l, u = _f3r_real_array(lower, 'lower'), _f3r_real_array(upper, 'upper')
    if l.ndim != 2 or l.shape[1] != 4 or l.shape[0] == 0 or l.shape != u.shape:
        raise ValueError("Factor arrays must have matching shape (b,4)")
    with np.errstate(over='ignore'):
        l16, u16 = l.astype(np.float16), u.astype(np.float16)
    if (not np.array_equal(l, l16.astype(np.float64)) or not np.array_equal(u, u16.astype(np.float64))
            or np.any(l[:, 0] != 0) or np.any(u == 0)):
        raise ValueError("Factors must be finite binary16 values with valid diagonals")
    return l16, u16

def _f3r_vector(value, size, dtype, name):
    arr = _f3r_real_array(value, name)
    if arr.shape != (size,):
        raise ValueError(name + " has an incompatible shape")
    with np.errstate(over='ignore'):
        arr = arr.astype(dtype)
    if not np.isfinite(arr).all():
        raise ValueError(name + " overflows its storage format")
    return arr

def _f3r_integer(value, name, low=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < low:
        raise ValueError(name + " is outside its integer domain")
    return int(value)

def _f3r_finite(value):
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite arithmetic result")
    return value

def _f3r_dot(a, b, dtype):
    return np.cumsum(np.multiply(a, b, dtype=dtype), dtype=dtype)[-1]

def _f3r_norm(a, dtype):
    return dtype(np.sqrt(_f3r_dot(a, a, dtype)))

@_lru_cache(maxsize=16)
def _f3r_sparse_data(shape, raw):
    A = np.frombuffer(raw, dtype=np.float64).reshape(shape)
    n = A.shape[0]
    widths = np.count_nonzero(A, axis=1)
    width = max(1, int(widths.max()))
    cols = np.full((n, width), n, dtype=np.int64)
    vals = np.zeros((n, width), dtype=np.float64)
    for i in range(n):
        js = np.flatnonzero(A[i])
        cols[i, :len(js)] = js
        vals[i, :len(js)] = A[i, js]
    return cols, (vals, vals.astype(np.float32), vals.astype(np.float16))

def _f3r_matrix(A, n):
    a = _f3r_real_array(A, 'A')
    if a.shape != (n, n):
        raise ValueError("A has an incompatible shape")
    with np.errstate(over='ignore'):
        if not np.isfinite(a.astype(np.float16)).all():
            raise ValueError("A must have a finite binary16 representation")
    return a

def _f3r_mv(A, v, level, dtype):
    cols, vals = _f3r_sparse_data(A.shape, A.tobytes())
    vpad = np.append(np.asarray(v, dtype=dtype), dtype(0))
    pro = np.multiply(vals[level], vpad[cols], dtype=dtype)
    return np.cumsum(pro, axis=1, dtype=dtype)[:, -1]

def _f3r_dot_many(a, b, dtype):
    # The last axis is one vector; never reduce across independent solves.
    return np.cumsum(np.multiply(a, b, dtype=dtype), axis=-1, dtype=dtype)[..., -1]


def _f3r_norm_many(a, dtype):
    return np.sqrt(_f3r_dot_many(a, a, dtype)).astype(dtype, copy=False)


def _f3r_mv_many(A, v, level, dtype):
    cols, vals = _f3r_sparse_data(A.shape, A.tobytes())
    vpad = np.concatenate((np.asarray(v, dtype=dtype), np.zeros((len(v), 1), dtype=dtype)), axis=1)
    products = np.multiply(vals[level][None, :, :], vpad[:, cols], dtype=dtype)
    return np.cumsum(products, axis=2, dtype=dtype)[:, :, -1]


def _f3r_gold_apply_primary_many(lower, upper, rhs, precision):
    """Internal numerical kernel for already validated independent factor/vector rows."""
    dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    r = _f3r_finite(np.asarray(rhs, dtype=dtype)).reshape(lower.shape)
    l, u = lower.astype(dtype, copy=False), upper.astype(dtype, copy=False)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        y = r.copy()
        for k in range(1, 4):
            y[:, :, k] = np.subtract(r[:, :, k], np.multiply(l[:, :, k], y[:, :, k-1], dtype=dtype), dtype=dtype)
        z = np.zeros_like(y)
        z[:, :, 3] = np.divide(y[:, :, 3], u[:, :, 3], dtype=dtype)
        for k in range(2, -1, -1):
            term = np.multiply(dtype(-7/16), z[:, :, k+1], dtype=dtype)
            z[:, :, k] = np.divide(np.subtract(y[:, :, k], term, dtype=dtype), u[:, :, k], dtype=dtype)
    return _f3r_finite(z.reshape((len(lower), -1)))


def _oracle_apply_primary(lower: "np.ndarray", upper: "np.ndarray", rhs: "np.ndarray", precision: int) -> "np.ndarray":
    if precision not in (16, 32, 64) or isinstance(precision, (bool, np.bool_)):
        raise ValueError("precision must be 16, 32, or 64")
    dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    lower, upper = _f3r_factor_arrays(lower, upper)
    r = _f3r_vector(rhs, lower.size, dtype, 'rhs')
    return _f3r_gold_apply_primary_many(lower[None, :, :], upper[None, :, :], r[None, :], precision)[0]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and declared-invalid test specifications."""
    return [
        # normal: binary16 triangular arithmetic
        {'setup': '\n'
                  'lower=np.array([[0.0, -0.13330078125, -0.1151123046875, -0.154296875]], '
                  'dtype=np.float16).reshape((1, 4))\n'
                  'upper=np.array([[1.40625, 1.62890625, 1.21484375, 1.9013671875]], dtype=np.float16).reshape((1, '
                  '4))\n'
                  'rhs=np.array([0.2, -0.3, 1.0, 0.7], dtype=np.float64).reshape((4,))\n'
                  'precision=16\n'
                  'expected=np.array([0.1700439453125, 0.0894775390625, 0.9580078125, 0.44677734375], '
                  'dtype=np.float16).reshape((4,))\n'
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
         'call': '_case_check(apply_primary(lower,upper,rhs,precision))',
         'gold_call': '_case_check(_oracle_apply_primary(lower,upper,rhs,precision))'},
        # normal: binary32 triangular arithmetic
        {'setup': '\n'
                  'lower=np.array([[0.0, -0.13330078125, -0.1151123046875, -0.154296875]], '
                  'dtype=np.float16).reshape((1, 4))\n'
                  'upper=np.array([[1.40625, 1.62890625, 1.21484375, 1.9013671875]], dtype=np.float16).reshape((1, '
                  '4))\n'
                  'rhs=np.array([0.2, -0.3, 1.0, 0.7], dtype=np.float64).reshape((4,))\n'
                  'precision=32\n'
                  'expected=np.array([0.17007794976234436, 0.08953624963760376, 0.9581394791603088, '
                  '0.44675323367118835], dtype=np.float32).reshape((4,))\n'
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
         'call': '_case_check(apply_primary(lower,upper,rhs,precision))',
         'gold_call': '_case_check(_oracle_apply_primary(lower,upper,rhs,precision))'},
        # normal: binary64 triangular arithmetic
        {'setup': '\n'
                  'lower=np.array([[0.0, -0.13330078125, -0.1151123046875, -0.154296875]], '
                  'dtype=np.float16).reshape((1, 4))\n'
                  'upper=np.array([[1.40625, 1.62890625, 1.21484375, 1.9013671875]], dtype=np.float16).reshape((1, '
                  '4))\n'
                  'rhs=np.array([0.2, -0.3, 1.0, 0.7], dtype=np.float64).reshape((4,))\n'
                  'precision=64\n'
                  'expected=np.array([0.1700779489470027, 0.0895362644725087, 0.9581394846878225, '
                  '0.44675324337875527], dtype=np.float64).reshape((4,))\n'
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
         'call': '_case_check(apply_primary(lower,upper,rhs,precision))',
         'gold_call': '_case_check(_oracle_apply_primary(lower,upper,rhs,precision))'},
        # boundary: zero rhs
        {'setup': '\n'
                  'lower=np.array([[0.0, -0.13330078125, -0.1151123046875, -0.154296875]], '
                  'dtype=np.float16).reshape((1, 4))\n'
                  'upper=np.array([[1.40625, 1.62890625, 1.21484375, 1.9013671875]], dtype=np.float16).reshape((1, '
                  '4))\n'
                  'rhs=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape((4,))\n'
                  'precision=16\n'
                  'expected=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float16).reshape((4,))\n'
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
         'call': '_case_check(apply_primary(lower,upper,rhs,precision))',
         'gold_call': '_case_check(_oracle_apply_primary(lower,upper,rhs,precision))'},
        # edge: retained binary16 subnormals
        {'setup': '\n'
                  'lower=np.array([[0.0, -0.13330078125, -0.1151123046875, -0.154296875]], '
                  'dtype=np.float16).reshape((1, 4))\n'
                  'upper=np.array([[1.40625, 1.62890625, 1.21484375, 1.9013671875]], dtype=np.float16).reshape((1, '
                  '4))\n'
                  'rhs=np.array([5.960464477539063e-08, -5.960464477539063e-08, 1.1920928955078125e-07, '
                  '5.960464477539063e-08], dtype=np.float64).reshape((4,))\n'
                  'precision=16\n'
                  'expected=np.array([5.960464477539063e-08, 0.0, 1.1920928955078125e-07, 5.960464477539063e-08], '
                  'dtype=np.float16).reshape((4,))\n'
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
         'call': '_case_check(apply_primary(lower,upper,rhs,precision))',
         'gold_call': '_case_check(_oracle_apply_primary(lower,upper,rhs,precision))'},
        # invalid: zero upper pivot
        {'setup': '\n'
                  'lower=np.array([[0.0, -0.13330078125, -0.1151123046875, -0.154296875]], '
                  'dtype=np.float16).reshape((1, 4))\n'
                  'upper=np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float16).reshape((1, 4))\n'
                  'rhs=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64).reshape((4,))\n'
                  'precision=16\n'
                  'def _case_raise(fn):\n'
                  '    try:\n'
                  '        fn()\n'
                  '        return 0\n'
                  '    except ValueError:\n'
                  '        return 1\n'
                  '    except Exception:\n'
                  '        return 2',
         'call': '_case_raise(lambda: apply_primary(lower,upper,rhs,precision))',
         'gold_call': '_case_raise(lambda: _oracle_apply_primary(lower,upper,rhs,precision))'},
    ]
