"""
Execute one two-correction Richardson invocation and return its persistent adaptive state.

Algorithm 1 counts complete Richardson invocations. Scheduled calls use a locally residual-minimizing weight for each current correction while separately averaging it into the weight stored for later calls. The initial unit weight participates in that average. State is explicit in this interface so independent batches cannot contaminate one another.



At Richardson position k, r = rhs - A*z and q = A*(M*r). On an update call, gamma = dot(r,q)/dot(q,q); the current correction uses gamma. With ell = counter//interval, the stored average is (ell*omega[k] + gamma)/(ell+1). The signature fixes the precision of each of these operations.

Returns
-------
tuple: binary16 correction vector, two binary16 stored weights, and the incremented integer call count.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def richardson_step(A: "np.ndarray", rhs: "np.ndarray", lower: "np.ndarray", upper: "np.ndarray",
                    weights: "np.ndarray", counter: int, interval: int, adaptive: bool = True) -> tuple:
    """Run the innermost two-iteration solver from a zero iterate.

    Parameters
    ----------
    A : np.ndarray
        Finite real square matrix of dimension n = 4*b with a finite
        binary16 representation. Store A in binary16 for this invocation.
    rhs : np.ndarray
        Real finite vector of length n; cast to binary16 on entry.
    lower, upper : np.ndarray
        Stored block factors satisfying apply_primary's contract.
    weights : np.ndarray
        Two finite binary16-representable stored weights; not modified.
    counter : int
        One-based positive invocation count. If counter == 1, initialize
        both weights to one, irrespective of the supplied weights.
    interval : int
        Positive update interval. A scheduled call has counter % interval == 0.
    adaptive : bool
        If True, use Algorithm 1. Compute each scheduled local weight and
        its average in binary32 using promoted stored binary16 A, factors,
        and residual; that weight computation uses a fresh binary32 primary
        solve. The current correction always uses a binary16 primary solve.
        On a scheduled call use the fresh local weight in a binary32 iterate
        update and then cast to binary16; store the running average in
        binary16. Otherwise use the stored weight entirely in binary16.
        Average with ell = counter//interval, including the initial unit
        weight. If False, use fixed unit weights and do no weight updates.
        In both modes form the second residual from rhs - A*z, not an
        incrementally propagated residual, and increment counter once.
        All arithmetic uses separate nearest-even operations, retained
        subnormals, increasing-index sums, and no FMA.

    Returns
    -------
    z, new_weights, next_counter : tuple
        z is binary16 shape (n,), new_weights is binary16 shape (2,), and
        next_counter is a Python integer. No input is modified.

    Raises
    ------
    ValueError
        If any matrix, vector, factor, or state violates its stated domain;
        if adaptive is not Boolean; if a storage cast or arithmetic result
        is nonfinite; or if a scheduled local-weight denominator is zero.
    """
    return z, new_weights, next_counter  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _f3r_state(weights, counter, interval, adaptive):
    w = _f3r_vector(weights, 2, np.float16, 'weights')
    if not np.array_equal(_f3r_real_array(weights, 'weights'), w.astype(np.float64)):
        raise ValueError("weights must be stored binary16 values")
    counter = _f3r_integer(counter, 'counter')
    interval = _f3r_integer(interval, 'interval')
    if not isinstance(adaptive, (bool, np.bool_)):
        raise ValueError("adaptive must be Boolean")
    if counter == 1:
        w = np.ones(2, dtype=np.float16)
    return w.copy(), counter, interval

def _f3r_gold_richardson_many(A, rhs, lower, upper, weights, counter, interval, adaptive=True):
    """Independent Richardson invocations with a common call number, never shared weights."""
    v = _f3r_finite(np.asarray(rhs, dtype=np.float16)).copy()
    omega = np.asarray(weights, dtype=np.float16).copy()
    if counter == 1:
        omega.fill(np.float16(1))
    z = np.zeros_like(v)
    scheduled = bool(adaptive) and counter % interval == 0
    for k in range(2):
        r = v.copy() if k == 0 else np.subtract(v, _f3r_mv_many(A, z, 2, np.float16), dtype=np.float16)
        correction = _f3r_gold_apply_primary_many(lower, upper, r, 16)
        if scheduled:
            r32 = r.astype(np.float32)
            q = _f3r_mv_many(A, _f3r_gold_apply_primary_many(lower, upper, r32, 32), 2, np.float32)
            denominator = _f3r_dot_many(q, q, np.float32)
            if np.any(denominator == 0) or not np.isfinite(denominator).all():
                raise ValueError("Invalid local-weight denominator")
            gamma = np.divide(_f3r_dot_many(r32, q, np.float32), denominator, dtype=np.float32)
            ell = counter // interval
            weighted = np.multiply(np.float32(ell), omega[:, k].astype(np.float32), dtype=np.float32)
            avg = np.divide(np.add(weighted, gamma, dtype=np.float32), np.float32(ell + 1), dtype=np.float32)
            omega[:, k] = avg.astype(np.float16)
            z = np.add(z.astype(np.float32), np.multiply(gamma[:, None], correction.astype(np.float32), dtype=np.float32), dtype=np.float32).astype(np.float16)
        else:
            weight = omega[:, k, None] if adaptive else np.float16(1)
            z = np.add(z, np.multiply(weight, correction, dtype=np.float16), dtype=np.float16)
        _f3r_finite(z)
        _f3r_finite(omega)
    return z, omega, counter + 1


def _oracle_richardson_step(A: "np.ndarray", rhs: "np.ndarray", lower: "np.ndarray", upper: "np.ndarray",
                           weights: "np.ndarray", counter: int, interval: int, adaptive: bool = True) -> tuple:
    lower, upper = _f3r_factor_arrays(lower, upper)
    A = _f3r_matrix(A, lower.size)
    v = _f3r_vector(rhs, lower.size, np.float16, 'rhs')
    omega, counter, interval = _f3r_state(weights, counter, interval, adaptive)
    z, omega, counter = _f3r_gold_richardson_many(
        A, v[None, :], lower[None, :, :], upper[None, :, :], omega[None, :], counter, interval, adaptive)
    return z[0], omega[0], counter

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and declared-invalid test specifications."""
    return [
        # normal: retained non-update weights
        {'setup': '\n'
                  'A=np.array([[1.25, -0.4375, 0.125, 0.0], [-0.1875, 1.5, -0.4375, 0.0], [0.0, -0.1875, 1.125, '
                  '-0.4375], [-0.125, 0.0, -0.1875, 1.75]], dtype=np.float64).reshape((4, 4))\n'
                  'rhs=np.array([0.2, -0.3, 1.0, 0.7], dtype=np.float64).reshape((4,))\n'
                  'lower=np.array([[0.0, -0.13330078125, -0.1151123046875, -0.154296875]], '
                  'dtype=np.float16).reshape((1, 4))\n'
                  'upper=np.array([[1.40625, 1.62890625, 1.21484375, 1.9013671875]], dtype=np.float16).reshape((1, '
                  '4))\n'
                  'weights=np.array([1.25, 0.875], dtype=np.float16).reshape((2,))\n'
                  'counter=7\n'
                  'interval=5\n'
                  'adaptive=True\n'
                  'expected=(np.array([0.116455078125, 0.135498046875, 1.1376953125, 0.541015625], '
                  'dtype=np.float16).reshape((4,)), np.array([1.25, 0.875], dtype=np.float16).reshape((2,)), 8)\n'
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
         'call': '_case_check(richardson_step(A,rhs,lower,upper,weights,counter,interval,adaptive))',
         'gold_call': '_case_check(_oracle_richardson_step(A,rhs,lower,upper,weights,counter,interval,adaptive))'},
        # normal: scheduled update uses fresh correction
        {'setup': '\n'
                  'A=np.array([[1.25, -0.4375, 0.125, 0.0], [-0.1875, 1.5, -0.4375, 0.0], [0.0, -0.1875, 1.125, '
                  '-0.4375], [-0.125, 0.0, -0.1875, 1.75]], dtype=np.float64).reshape((4, 4))\n'
                  'rhs=np.array([0.2, -0.3, 1.0, 0.7], dtype=np.float64).reshape((4,))\n'
                  'lower=np.array([[0.0, -0.13330078125, -0.1151123046875, -0.154296875]], '
                  'dtype=np.float16).reshape((1, 4))\n'
                  'upper=np.array([[1.40625, 1.62890625, 1.21484375, 1.9013671875]], dtype=np.float16).reshape((1, '
                  '4))\n'
                  'weights=np.array([1.25, 0.875], dtype=np.float16).reshape((2,))\n'
                  'counter=5\n'
                  'interval=5\n'
                  'adaptive=True\n'
                  'expected=(np.array([0.0997314453125, 0.1378173828125, 1.1171875, 0.53271484375], '
                  'dtype=np.float16).reshape((4,)), np.array([1.1845703125, 0.99755859375], '
                  'dtype=np.float16).reshape((2,)), 6)\n'
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
         'call': '_case_check(richardson_step(A,rhs,lower,upper,weights,counter,interval,adaptive))',
         'gold_call': '_case_check(_oracle_richardson_step(A,rhs,lower,upper,weights,counter,interval,adaptive))'},
        # boundary: first-call unit initialization
        {'setup': '\n'
                  'A=np.array([[1.25, -0.4375, 0.125, 0.0], [-0.1875, 1.5, -0.4375, 0.0], [0.0, -0.1875, 1.125, '
                  '-0.4375], [-0.125, 0.0, -0.1875, 1.75]], dtype=np.float64).reshape((4, 4))\n'
                  'rhs=np.array([0.2, -0.3, 1.0, 0.7], dtype=np.float64).reshape((4,))\n'
                  'lower=np.array([[0.0, -0.13330078125, -0.1151123046875, -0.154296875]], '
                  'dtype=np.float16).reshape((1, 4))\n'
                  'upper=np.array([[1.40625, 1.62890625, 1.21484375, 1.9013671875]], dtype=np.float16).reshape((1, '
                  '4))\n'
                  'weights=np.array([1.25, 0.875], dtype=np.float16).reshape((2,))\n'
                  'counter=1\n'
                  'interval=5\n'
                  'adaptive=True\n'
                  'expected=(np.array([0.1162109375, 0.1290283203125, 1.0947265625, 0.5205078125], '
                  'dtype=np.float16).reshape((4,)), np.array([1.0, 1.0], dtype=np.float16).reshape((2,)), 2)\n'
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
         'call': '_case_check(richardson_step(A,rhs,lower,upper,weights,counter,interval,adaptive))',
         'gold_call': '_case_check(_oracle_richardson_step(A,rhs,lower,upper,weights,counter,interval,adaptive))'},
        # edge: second averaging index
        {'setup': '\n'
                  'A=np.array([[1.25, -0.4375, 0.125, 0.0], [-0.1875, 1.5, -0.4375, 0.0], [0.0, -0.1875, 1.125, '
                  '-0.4375], [-0.125, 0.0, -0.1875, 1.75]], dtype=np.float64).reshape((4, 4))\n'
                  'rhs=np.array([0.2, -0.3, 1.0, 0.7], dtype=np.float64).reshape((4,))\n'
                  'lower=np.array([[0.0, -0.13330078125, -0.1151123046875, -0.154296875]], '
                  'dtype=np.float16).reshape((1, 4))\n'
                  'upper=np.array([[1.40625, 1.62890625, 1.21484375, 1.9013671875]], dtype=np.float16).reshape((1, '
                  '4))\n'
                  'weights=np.array([1.25, 0.875], dtype=np.float16).reshape((2,))\n'
                  'counter=10\n'
                  'interval=5\n'
                  'adaptive=True\n'
                  'expected=(np.array([0.0997314453125, 0.1378173828125, 1.1171875, 0.53271484375], '
                  'dtype=np.float16).reshape((4,)), np.array([1.20703125, 0.95654296875], '
                  'dtype=np.float16).reshape((2,)), 11)\n'
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
         'call': '_case_check(richardson_step(A,rhs,lower,upper,weights,counter,interval,adaptive))',
         'gold_call': '_case_check(_oracle_richardson_step(A,rhs,lower,upper,weights,counter,interval,adaptive))'},
        # control: fixed weight ignores adaptive state
        {'setup': '\n'
                  'A=np.array([[1.25, -0.4375, 0.125, 0.0], [-0.1875, 1.5, -0.4375, 0.0], [0.0, -0.1875, 1.125, '
                  '-0.4375], [-0.125, 0.0, -0.1875, 1.75]], dtype=np.float64).reshape((4, 4))\n'
                  'rhs=np.array([0.2, -0.3, 1.0, 0.7], dtype=np.float64).reshape((4,))\n'
                  'lower=np.array([[0.0, -0.13330078125, -0.1151123046875, -0.154296875]], '
                  'dtype=np.float16).reshape((1, 4))\n'
                  'upper=np.array([[1.40625, 1.62890625, 1.21484375, 1.9013671875]], dtype=np.float16).reshape((1, '
                  '4))\n'
                  'weights=np.array([1.25, 0.875], dtype=np.float16).reshape((2,))\n'
                  'counter=5\n'
                  'interval=5\n'
                  'adaptive=False\n'
                  'expected=(np.array([0.1162109375, 0.1290283203125, 1.0947265625, 0.5205078125], '
                  'dtype=np.float16).reshape((4,)), np.array([1.25, 0.875], dtype=np.float16).reshape((2,)), 6)\n'
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
         'call': '_case_check(richardson_step(A,rhs,lower,upper,weights,counter,interval,adaptive))',
         'gold_call': '_case_check(_oracle_richardson_step(A,rhs,lower,upper,weights,counter,interval,adaptive))'},
        # boundary: zero non-update residual
        {'setup': '\n'
                  'A=np.array([[1.25, -0.4375, 0.125, 0.0], [-0.1875, 1.5, -0.4375, 0.0], [0.0, -0.1875, 1.125, '
                  '-0.4375], [-0.125, 0.0, -0.1875, 1.75]], dtype=np.float64).reshape((4, 4))\n'
                  'rhs=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape((4,))\n'
                  'lower=np.array([[0.0, -0.13330078125, -0.1151123046875, -0.154296875]], '
                  'dtype=np.float16).reshape((1, 4))\n'
                  'upper=np.array([[1.40625, 1.62890625, 1.21484375, 1.9013671875]], dtype=np.float16).reshape((1, '
                  '4))\n'
                  'weights=np.array([1.25, 0.875], dtype=np.float16).reshape((2,))\n'
                  'counter=3\n'
                  'interval=5\n'
                  'expected=(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float16).reshape((4,)), np.array([1.25, 0.875], '
                  'dtype=np.float16).reshape((2,)), 4)\n'
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
         'call': '_case_check(richardson_step(A,rhs,lower,upper,weights,counter,interval))',
         'gold_call': '_case_check(_oracle_richardson_step(A,rhs,lower,upper,weights,counter,interval))'},
        # invalid: zero local-weight denominator
        {'setup': '\n'
                  'A=np.array([[1.25, -0.4375, 0.125, 0.0], [-0.1875, 1.5, -0.4375, 0.0], [0.0, -0.1875, 1.125, '
                  '-0.4375], [-0.125, 0.0, -0.1875, 1.75]], dtype=np.float64).reshape((4, 4))\n'
                  'rhs=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape((4,))\n'
                  'lower=np.array([[0.0, -0.13330078125, -0.1151123046875, -0.154296875]], '
                  'dtype=np.float16).reshape((1, 4))\n'
                  'upper=np.array([[1.40625, 1.62890625, 1.21484375, 1.9013671875]], dtype=np.float16).reshape((1, '
                  '4))\n'
                  'weights=np.array([1.0, 1.0], dtype=np.float16).reshape((2,))\n'
                  'counter=5\n'
                  'interval=5\n'
                  'def _case_raise(fn):\n'
                  '    try:\n'
                  '        fn()\n'
                  '        return 0\n'
                  '    except ValueError:\n'
                  '        return 1\n'
                  '    except Exception:\n'
                  '        return 2',
         'call': '_case_raise(lambda: richardson_step(A,rhs,lower,upper,weights,counter,interval))',
         'gold_call': '_case_raise(lambda: _oracle_richardson_step(A,rhs,lower,upper,weights,counter,interval))'},
    ]
