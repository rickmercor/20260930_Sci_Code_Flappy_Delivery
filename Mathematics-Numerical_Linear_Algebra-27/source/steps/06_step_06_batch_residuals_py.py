"""
Evaluate fixed and adaptive F3R for every ordering of a batch while preserving adaptive history within each ordering.

Different right-hand-side orders can create different stored Richardson weights. Every new system starts from zero, but the adaptive state is retained within its batch. Identical ordered prefixes may be shared because their starting state is identical. Unordered prefixes cannot be shared. Both variants are evaluated by the original-system relative residual in binary64.



For each order pi and position t, R[q,pi,t] = norm(b[pi[t]] - A*x[q,pi,t])/norm(b[pi[t]]). The fixed and adaptive residual arrays use the same order and position. Each independent ordered batch begins with its own weights and counter.

Returns
-------
tuple: two binary64 residual tables, final binary16 adaptive weights, and integer batch counters.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def batch_residuals(A: "np.ndarray", right_hand_sides: "np.ndarray", lower: "np.ndarray", upper: "np.ndarray",
                    theta: tuple, outer_iterations: int = 2, restarts: int = 2) -> tuple:
    """Evaluate all complete right-hand-side permutations at one factor state.

    Parameters
    ----------
    A, lower, upper
        Original matrix and stored block factors as in nested_fgmres_cycle.
    right_hand_sides : np.ndarray
        Real finite shape (r,n), with 1 <= r <= 3 and each row nonzero.
    theta : tuple
        Three positive integers (m2,m3,interval), with m2,m3 <= n.
    outer_iterations, restarts : int
        Positive integers; outer_iterations <= n. Each system uses exactly
        restarts outer cycles and starts from zero. Every outer restart
        retains its solution and Richardson state. At the start of each
        independent permutation, initialize weights=(1,1) and counter=1.
        Subsequent systems reset their solution but retain weights/counter.
        The fixed control uses unit weights and no updates. Use all six
        permutations for r=3 (all r! in general), in lexicographic order.
        Shared-prefix reuse is allowed only for identical ordered prefixes.

    Returns
    -------
    fixed, adaptive, final_weights, final_counters : tuple
        fixed and adaptive are binary64 arrays of shape (r!,r). Column t
        refers to the system at position t of that lexicographic permutation.
        Residuals are norm(b-A*x)/norm(b), using original A and binary64
        increasing-index products/sums and direct sum-of-squares norms.
        final_weights is binary16 shape (r!,2), and final_counters is int64
        shape (r!,), for the ADAPTIVE batches. Inputs are not modified.

    Raises
    ------
    ValueError
        If inputs violate the stated domains, a right-hand side is zero,
        or any child solver encounters a documented zero denominator,
        breakdown, overflowing storage cast or nonfinite arithmetic.
    """
    return fixed, adaptive, final_weights, final_counters  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools as _itertools
import numpy as np


def _f3r_gold_batch_residuals_many(A, B, lower, upper, theta, outer_iterations=2, restarts=2, fixed_by_rhs=None):
    """Internal batch kernel: independent factor states and ordered prefixes, no shared adaptive state."""
    m2, m3, interval = theta
    ms = (outer_iterations, m2, m3)
    nstates, nrhs = len(lower), len(B)
    norms = _f3r_norm_many(B, np.float64)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("Invalid right-hand-side norm")
    def _solve(j, w, count, adaptive):
        width = len(j)
        rhs = np.tile(B[j], (nstates, 1))
        ll = np.repeat(lower, width, axis=0)
        uu = np.repeat(upper, width, axis=0)
        x = None
        for cycle in range(restarts):
            x, w, count = _f3r_gold_nested_fgmres_many(A, rhs, ll, uu, ms, interval, w, count, 0, x, adaptive)
        residual = np.subtract(rhs, _f3r_mv_many(A, x, 0, np.float64), dtype=np.float64)
        ratios = np.divide(_f3r_norm_many(residual, np.float64), np.tile(norms[j], nstates), dtype=np.float64)
        if not np.isfinite(ratios).all():
            raise ValueError("Nonfinite true residual")
        return ratios.reshape((nstates, width)), w.reshape((nstates, width, 2)), count
    if fixed_by_rhs is None:
        fixed_by_rhs = _solve(list(range(nrhs)), np.ones((nstates * nrhs, 2), dtype=np.float16), 1, False)[0]
    # Distinct ordered prefixes retain distinct state rows. Only identical prefixes are reused.
    previous = {(): 0}
    previous_weights = np.ones((nstates, 1, 2), dtype=np.float16)
    count = 1
    residuals = {}
    for length in range(1, nrhs + 1):
        prefixes = tuple(_itertools.permutations(range(nrhs), length))
        parent_indices = [previous[prefix[:-1]] for prefix in prefixes]
        w = previous_weights[:, parent_indices, :].reshape((-1, 2)).copy()
        rr, ww, count = _solve([prefix[-1] for prefix in prefixes], w, count, True)
        for i, prefix in enumerate(prefixes):
            residuals[prefix] = rr[:, i].copy()
        previous, previous_weights = {prefix: i for i, prefix in enumerate(prefixes)}, ww
    permutations = tuple(_itertools.permutations(range(nrhs)))
    orders = np.asarray(permutations, dtype=np.int64)
    fixed = np.asarray(fixed_by_rhs[:, orders], dtype=np.float64)
    adaptive = np.stack([np.stack([residuals[order[:t+1]] for t in range(nrhs)], axis=1) for order in permutations], axis=1)
    return fixed, adaptive, previous_weights.copy(), np.full((nstates, len(permutations)), count, dtype=np.int64)


def _oracle_batch_residuals(A: "np.ndarray", right_hand_sides: "np.ndarray", lower: "np.ndarray", upper: "np.ndarray",
                            theta: tuple, outer_iterations: int = 2, restarts: int = 2) -> tuple:
    lower, upper = _f3r_factor_arrays(lower, upper)
    n = lower.size
    A = _f3r_matrix(A, n)
    B = _f3r_real_array(right_hand_sides, 'right_hand_sides')
    if B.ndim != 2 or B.shape[1] != n or not 1 <= B.shape[0] <= 3 or np.any(np.all(B == 0, axis=1)):
        raise ValueError("right_hand_sides must have 1 to 3 nonzero rows of length n")
    if not isinstance(theta, (tuple, list)) or len(theta) != 3:
        raise ValueError("theta must contain three positive integers")
    m2, m3, interval = (_f3r_integer(v, 'theta entry') for v in theta)
    m1 = _f3r_integer(outer_iterations, 'outer_iterations')
    restarts = _f3r_integer(restarts, 'restarts')
    if max(m1, m2, m3) > n:
        raise ValueError("FGMRES lengths cannot exceed n")
    result = _f3r_gold_batch_residuals_many(A, B, lower[None, :, :], upper[None, :, :], (m2, m3, interval), m1, restarts)
    return tuple(value[0] for value in result)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and declared-invalid test specifications."""
    return [
        # normal: all six retained-state orders
        {'setup': 'import numpy as np\n'
                  'n=48\n'
                  'i=np.arange(n)\n'
                  'A=np.zeros((n,n),dtype=np.float64)\n'
                  'A[i,i]=1+(1+i%3)/4096\n'
                  'for offset,value in ((1,-7/16),(-1,-3/16),(8,-1/4),(-8,-1/8)):\n'
                  '    A[i,(i+offset)%n]=value\n'
                  'B=np.stack((1+(-1.0)**i/4+(i%7)/32,(-1.0)**i*(1+(i%5)/16),(((3*i)%11)-5)/8+(i%2)/32))\n'
                  'lower=np.array([[0.0, -0.1666259765625, -0.1781005859375, -0.178955078125], [0.0, '
                  '-0.1666259765625, -0.1781005859375, -0.1790771484375], [0.0, -0.16650390625, -0.17822265625, '
                  '-0.178955078125], [0.0, -0.1666259765625, -0.1781005859375, -0.178955078125], [0.0, '
                  '-0.1666259765625, -0.1781005859375, -0.1790771484375], [0.0, -0.16650390625, -0.17822265625, '
                  '-0.178955078125], [0.0, -0.1666259765625, -0.1781005859375, -0.178955078125], [0.0, '
                  '-0.1666259765625, -0.1781005859375, -0.1790771484375], [0.0, -0.16650390625, -0.17822265625, '
                  '-0.178955078125], [0.0, -0.1666259765625, -0.1781005859375, -0.178955078125], [0.0, '
                  '-0.1666259765625, -0.1781005859375, -0.1790771484375], [0.0, -0.16650390625, -0.17822265625, '
                  '-0.178955078125]], dtype=np.float16).reshape((12, 4))\n'
                  'upper=np.array([[1.125, 1.052734375, 1.0478515625, 1.046875], [1.1259765625, 1.052734375, '
                  '1.0478515625, 1.046875], [1.1259765625, 1.052734375, 1.0478515625, 1.0478515625], [1.125, '
                  '1.052734375, 1.0478515625, 1.046875], [1.1259765625, 1.052734375, 1.0478515625, 1.046875], '
                  '[1.1259765625, 1.052734375, 1.0478515625, 1.0478515625], [1.125, 1.052734375, 1.0478515625, '
                  '1.046875], [1.1259765625, 1.052734375, 1.0478515625, 1.046875], [1.1259765625, 1.052734375, '
                  '1.0478515625, 1.0478515625], [1.125, 1.052734375, 1.0478515625, 1.046875], [1.1259765625, '
                  '1.052734375, 1.0478515625, 1.046875], [1.1259765625, 1.052734375, 1.0478515625, 1.0478515625]], '
                  'dtype=np.float16).reshape((12, 4))\n'
                  '\n'
                  '\n'
                  'expected=(np.array([[1.2378456287264246e-08, 2.310737541780634e-08, 5.091518572673492e-08], '
                  '[1.2378456287264246e-08, 5.091518572673492e-08, 2.310737541780634e-08], [2.310737541780634e-08, '
                  '1.2378456287264246e-08, 5.091518572673492e-08], [2.310737541780634e-08, 5.091518572673492e-08, '
                  '1.2378456287264246e-08], [5.091518572673492e-08, 1.2378456287264246e-08, 2.310737541780634e-08], '
                  '[5.091518572673492e-08, 2.310737541780634e-08, 1.2378456287264246e-08]], '
                  'dtype=np.float64).reshape((6, 3)), np.array([[4.268961237458433e-11, 1.5539802455289731e-09, '
                  '1.7060465792320113e-08], [4.268961237458433e-11, 1.1993281113187228e-08, 2.0503211069274465e-09], '
                  '[9.958816759304467e-10, 1.0932891082043156e-09, 2.0522043686651322e-08], [9.958816759304467e-10, '
                  '2.2767168631229495e-08, 1.2693352341855175e-09], [1.3824121108249541e-08, 3.5291479414482066e-09, '
                  '1.538152308445747e-09], [1.3824121108249541e-08, 8.903491510664077e-10, 1.1485513699908384e-09]], '
                  'dtype=np.float64).reshape((6, 3)), np.array([[1.1005859375, 1.1083984375], [1.0849609375, '
                  '1.1533203125], [1.1220703125, 1.109375], [1.126953125, 1.0986328125], [1.083984375, '
                  '1.1396484375], [1.0927734375, 1.1416015625]], dtype=np.float16).reshape((6, 2)), np.array([145, '
                  '145, 145, 145, 145, 145], dtype=np.int64).reshape((6,)))\n'
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
         'call': '_case_check(batch_residuals(A,B,lower,upper,(4,3,5)))',
         'gold_call': '_case_check(_oracle_batch_residuals(A,B,lower,upper,(4,3,5)))'},
        # boundary: one system
        {'setup': 'import numpy as np\n'
                  'n=48\n'
                  'i=np.arange(n)\n'
                  'A=np.zeros((n,n),dtype=np.float64)\n'
                  'A[i,i]=1+(1+i%3)/4096\n'
                  'for offset,value in ((1,-7/16),(-1,-3/16),(8,-1/4),(-8,-1/8)):\n'
                  '    A[i,(i+offset)%n]=value\n'
                  'B=np.stack((1+(-1.0)**i/4+(i%7)/32,(-1.0)**i*(1+(i%5)/16),(((3*i)%11)-5)/8+(i%2)/32))\n'
                  'lower=np.array([[0.0, -0.1666259765625, -0.1781005859375, -0.178955078125], [0.0, '
                  '-0.1666259765625, -0.1781005859375, -0.1790771484375], [0.0, -0.16650390625, -0.17822265625, '
                  '-0.178955078125], [0.0, -0.1666259765625, -0.1781005859375, -0.178955078125], [0.0, '
                  '-0.1666259765625, -0.1781005859375, -0.1790771484375], [0.0, -0.16650390625, -0.17822265625, '
                  '-0.178955078125], [0.0, -0.1666259765625, -0.1781005859375, -0.178955078125], [0.0, '
                  '-0.1666259765625, -0.1781005859375, -0.1790771484375], [0.0, -0.16650390625, -0.17822265625, '
                  '-0.178955078125], [0.0, -0.1666259765625, -0.1781005859375, -0.178955078125], [0.0, '
                  '-0.1666259765625, -0.1781005859375, -0.1790771484375], [0.0, -0.16650390625, -0.17822265625, '
                  '-0.178955078125]], dtype=np.float16).reshape((12, 4))\n'
                  'upper=np.array([[1.125, 1.052734375, 1.0478515625, 1.046875], [1.1259765625, 1.052734375, '
                  '1.0478515625, 1.046875], [1.1259765625, 1.052734375, 1.0478515625, 1.0478515625], [1.125, '
                  '1.052734375, 1.0478515625, 1.046875], [1.1259765625, 1.052734375, 1.0478515625, 1.046875], '
                  '[1.1259765625, 1.052734375, 1.0478515625, 1.0478515625], [1.125, 1.052734375, 1.0478515625, '
                  '1.046875], [1.1259765625, 1.052734375, 1.0478515625, 1.046875], [1.1259765625, 1.052734375, '
                  '1.0478515625, 1.0478515625], [1.125, 1.052734375, 1.0478515625, 1.046875], [1.1259765625, '
                  '1.052734375, 1.0478515625, 1.046875], [1.1259765625, 1.052734375, 1.0478515625, 1.0478515625]], '
                  'dtype=np.float16).reshape((12, 4))\n'
                  '\n'
                  '\n'
                  'expected=(np.array([[1.2378456287264246e-08]], dtype=np.float64).reshape((1, 1)), '
                  'np.array([[4.268961237458433e-11]], dtype=np.float64).reshape((1, 1)), np.array([[1.033203125, '
                  '1.1376953125]], dtype=np.float16).reshape((1, 2)), np.array([49], dtype=np.int64).reshape((1,)))\n'
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
         'call': '_case_check(batch_residuals(A,B[:1],lower,upper,(4,3,5)))',
         'gold_call': '_case_check(_oracle_batch_residuals(A,B[:1],lower,upper,(4,3,5)))'},
        # edge: reordered two-system batch and shifted update schedule
        {'setup': 'import numpy as np\n'
                  'n=48\n'
                  'i=np.arange(n)\n'
                  'A=np.zeros((n,n),dtype=np.float64)\n'
                  'A[i,i]=1+(1+i%3)/4096\n'
                  'for offset,value in ((1,-7/16),(-1,-3/16),(8,-1/4),(-8,-1/8)):\n'
                  '    A[i,(i+offset)%n]=value\n'
                  'B=np.stack((1+(-1.0)**i/4+(i%7)/32,(-1.0)**i*(1+(i%5)/16),(((3*i)%11)-5)/8+(i%2)/32))\n'
                  'lower=np.array([[0.0, -0.1666259765625, -0.1781005859375, -0.178955078125], [0.0, '
                  '-0.1666259765625, -0.1781005859375, -0.1790771484375], [0.0, -0.16650390625, -0.17822265625, '
                  '-0.178955078125], [0.0, -0.1666259765625, -0.1781005859375, -0.178955078125], [0.0, '
                  '-0.1666259765625, -0.1781005859375, -0.1790771484375], [0.0, -0.16650390625, -0.17822265625, '
                  '-0.178955078125], [0.0, -0.1666259765625, -0.1781005859375, -0.178955078125], [0.0, '
                  '-0.1666259765625, -0.1781005859375, -0.1790771484375], [0.0, -0.16650390625, -0.17822265625, '
                  '-0.178955078125], [0.0, -0.1666259765625, -0.1781005859375, -0.178955078125], [0.0, '
                  '-0.1666259765625, -0.1781005859375, -0.1790771484375], [0.0, -0.16650390625, -0.17822265625, '
                  '-0.178955078125]], dtype=np.float16).reshape((12, 4))\n'
                  'upper=np.array([[1.125, 1.052734375, 1.0478515625, 1.046875], [1.1259765625, 1.052734375, '
                  '1.0478515625, 1.046875], [1.1259765625, 1.052734375, 1.0478515625, 1.0478515625], [1.125, '
                  '1.052734375, 1.0478515625, 1.046875], [1.1259765625, 1.052734375, 1.0478515625, 1.046875], '
                  '[1.1259765625, 1.052734375, 1.0478515625, 1.0478515625], [1.125, 1.052734375, 1.0478515625, '
                  '1.046875], [1.1259765625, 1.052734375, 1.0478515625, 1.046875], [1.1259765625, 1.052734375, '
                  '1.0478515625, 1.0478515625], [1.125, 1.052734375, 1.0478515625, 1.046875], [1.1259765625, '
                  '1.052734375, 1.0478515625, 1.046875], [1.1259765625, 1.052734375, 1.0478515625, 1.0478515625]], '
                  'dtype=np.float16).reshape((12, 4))\n'
                  '\n'
                  '\n'
                  'expected=(np.array([[2.271603843026005e-09, 4.82932842564751e-10], [4.82932842564751e-10, '
                  '2.271603843026005e-09]], dtype=np.float64).reshape((2, 2)), np.array([[5.1315600498954796e-08, '
                  '4.864633444467275e-09], [1.1913630317739123e-08, 4.1981124048736465e-09]], '
                  'dtype=np.float64).reshape((2, 2)), np.array([[1.0927734375, 1.099609375], [1.0322265625, '
                  '1.13671875]], dtype=np.float16).reshape((2, 2)), np.array([97, 97], '
                  'dtype=np.int64).reshape((2,)))\n'
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
         'call': '_case_check(batch_residuals(A,B[[2,0]],lower,upper,(3,4,11)))',
         'gold_call': '_case_check(_oracle_batch_residuals(A,B[[2,0]],lower,upper,(3,4,11)))'},
        # invalid: zero right-hand side
        {'setup': 'import numpy as np\n'
                  'n=48\n'
                  'i=np.arange(n)\n'
                  'A=np.zeros((n,n),dtype=np.float64)\n'
                  'A[i,i]=1+(1+i%3)/4096\n'
                  'for offset,value in ((1,-7/16),(-1,-3/16),(8,-1/4),(-8,-1/8)):\n'
                  '    A[i,(i+offset)%n]=value\n'
                  'B=np.stack((1+(-1.0)**i/4+(i%7)/32,(-1.0)**i*(1+(i%5)/16),(((3*i)%11)-5)/8+(i%2)/32))\n'
                  'lower=np.array([[0.0, -0.1666259765625, -0.1781005859375, -0.178955078125], [0.0, '
                  '-0.1666259765625, -0.1781005859375, -0.1790771484375], [0.0, -0.16650390625, -0.17822265625, '
                  '-0.178955078125], [0.0, -0.1666259765625, -0.1781005859375, -0.178955078125], [0.0, '
                  '-0.1666259765625, -0.1781005859375, -0.1790771484375], [0.0, -0.16650390625, -0.17822265625, '
                  '-0.178955078125], [0.0, -0.1666259765625, -0.1781005859375, -0.178955078125], [0.0, '
                  '-0.1666259765625, -0.1781005859375, -0.1790771484375], [0.0, -0.16650390625, -0.17822265625, '
                  '-0.178955078125], [0.0, -0.1666259765625, -0.1781005859375, -0.178955078125], [0.0, '
                  '-0.1666259765625, -0.1781005859375, -0.1790771484375], [0.0, -0.16650390625, -0.17822265625, '
                  '-0.178955078125]], dtype=np.float16).reshape((12, 4))\n'
                  'upper=np.array([[1.125, 1.052734375, 1.0478515625, 1.046875], [1.1259765625, 1.052734375, '
                  '1.0478515625, 1.046875], [1.1259765625, 1.052734375, 1.0478515625, 1.0478515625], [1.125, '
                  '1.052734375, 1.0478515625, 1.046875], [1.1259765625, 1.052734375, 1.0478515625, 1.046875], '
                  '[1.1259765625, 1.052734375, 1.0478515625, 1.0478515625], [1.125, 1.052734375, 1.0478515625, '
                  '1.046875], [1.1259765625, 1.052734375, 1.0478515625, 1.046875], [1.1259765625, 1.052734375, '
                  '1.0478515625, 1.0478515625], [1.125, 1.052734375, 1.0478515625, 1.046875], [1.1259765625, '
                  '1.052734375, 1.0478515625, 1.046875], [1.1259765625, 1.052734375, 1.0478515625, 1.0478515625]], '
                  'dtype=np.float16).reshape((12, 4))\n'
                  '\n'
                  '\n'
                  'def _case_raise(fn):\n'
                  '    try:\n'
                  '        fn()\n'
                  '        return 0\n'
                  '    except ValueError:\n'
                  '        return 1\n'
                  '    except Exception:\n'
                  '        return 2',
         'call': '_case_raise(lambda: batch_residuals(A,np.zeros((1,48)),lower,upper,(4,3,5)))',
         'gold_call': '_case_raise(lambda: _oracle_batch_residuals(A,np.zeros((1,48)),lower,upper,(4,3,5)))'},
    ]
