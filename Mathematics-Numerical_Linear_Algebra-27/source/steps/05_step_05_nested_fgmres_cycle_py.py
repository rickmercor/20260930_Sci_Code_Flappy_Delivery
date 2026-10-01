"""
Run one mixed-precision flexible GMRES cycle, using the next nested solver as its changing preconditioner.

F3R uses three flexible GMRES levels above Richardson. A parent retains each correction returned by its child, because the child is state-dependent rather than a fixed linear operator. Each child invocation starts from zero; the shared Richardson state survives every return. A second outer cycle is invoked with the first cycle endpoint as its initial iterate.



A cycle minimizes norm(beta*e1 - Hbar*y) over the computed reduced system and updates x_new = x_old + sum_j(y[j]*z[j]), where z[j] are the actual returned child corrections. The signature fixes classical Gram-Schmidt, Givens rotations, rounding and restart conventions.

Returns
-------
tuple: one cycle endpoint, two stored binary16 weights, and the next Richardson invocation count.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nested_fgmres_cycle(A: "np.ndarray", rhs: "np.ndarray", lower: "np.ndarray", upper: "np.ndarray",
                        iterations: tuple, interval: int, weights: "np.ndarray", counter: int,
                        level: int = 0, initial: "np.ndarray | None" = None, adaptive: bool = True) -> tuple:
    """Return one prescribed FGMRES cycle and the updated Richardson state.

    Parameters
    ----------
    A, rhs, lower, upper
        Matrix, vector and stored block factors as in richardson_step.
    iterations : tuple
        Three positive integers, each <= n, for the outer, middle and third
        FGMRES lengths. Richardson always has two corrections per invocation.
    interval, weights, counter, adaptive
        Adaptive controls as in richardson_step. State is carried through
        every child call and is returned, not reset at a level boundary.
    level : int
        0, 1, or 2. Level 0 uses binary64 matrix/vector/arithmetic. Level 1
        uses binary32 matrix/vector/arithmetic. Level 2 uses a binary16
        matrix with binary32 vectors/arithmetic and Richardson as its child.
        Higher-level children are recursively the next FGMRES level.
    initial : np.ndarray or None
        None means a zero starting iterate and residual equal to rhs. An
        explicit vector is cast to this level's vector format and uses the
        freshly computed residual rhs - A_level*initial. Inner calls always
        use None. rhs is cast to the receiving level on entry.

        Use one classical Gram-Schmidt pass: all coefficients from the same
        incoming vector, accumulate the projection by increasing basis index,
        then subtract once. Norms are increasing-index sums of separately
        rounded squares followed by a square root. Apply previous Givens
        rotations in increasing order; a new pair (a,b) uses positive
        rho=sqrt(a*a+b*b), c=a/rho, s=b/rho and becomes (rho,0). Apply each
        rotation to the reduced rhs too. Back substitution uses increasing
        column-index sums. Combine the stored child corrections in increasing
        basis order and add the accumulated correction once to initial.
        Use nearest-even separate operations, retained subnormals, and no
        fusion, reorthogonalization, library least-squares substitution or
        convergence-based early exit.

    Returns
    -------
    x, new_weights, next_counter : tuple
        x has this level's vector dtype and shape (n,). Stored weights are
        binary16 shape (2,), and next_counter is a Python integer. No input
        is modified. Only Richardson calls advance the counter.

    Raises
    ------
    ValueError
        If any supplied value violates the above shape, type or numeric
        domains; if iterations is not three integers in [1,n]; if level is
        not 0,1,2; if the initial residual norm, an Arnoldi norm, a Givens
        denominator or a reduced triangular pivot is zero; or if any child
        call has an invalid local-weight denominator or nonfinite arithmetic.
        Such breakdowns are errors for this fixed-budget interface, not
        silent early-convergence branches.
    """
    return x, new_weights, next_counter  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _f3r_gold_nested_fgmres_many(A, rhs, lower, upper, iterations, interval, weights, counter,
                              level=0, initial=None, adaptive=True):
    """Vectorize independent solves, keeping every within-solve operation in its original order."""
    dtype = np.float64 if level == 0 else np.float32
    b = _f3r_finite(np.asarray(rhs, dtype=dtype)).copy()
    omega = np.asarray(weights, dtype=np.float16).copy()
    if counter == 1:
        omega.fill(np.float16(1))
    if initial is None:
        x, r = np.zeros_like(b), b.copy()
    else:
        x = _f3r_finite(np.asarray(initial, dtype=dtype)).copy()
        r = np.subtract(b, _f3r_mv_many(A, x, level, dtype), dtype=dtype)
    beta = _f3r_norm_many(r, dtype)
    if np.any(beta == 0) or not np.isfinite(beta).all():
        raise ValueError("Invalid initial residual norm")
    m = iterations[level]
    size, n = b.shape
    V = np.zeros((m + 1, size, n), dtype=dtype)
    Z = np.zeros((m, size, n), dtype=dtype)
    H = np.zeros((size, m + 1, m), dtype=dtype)
    g = np.zeros((size, m + 1), dtype=dtype)
    cs, sn = np.zeros((size, m), dtype=dtype), np.zeros((size, m), dtype=dtype)
    g[:, 0], V[0] = beta, np.divide(r, beta[:, None], dtype=dtype)
    for j in range(m):
        if level == 2:
            z, omega, counter = _f3r_gold_richardson_many(A, V[j], lower, upper, omega, counter, interval, adaptive)
        else:
            z, omega, counter = _f3r_gold_nested_fgmres_many(A, V[j], lower, upper, iterations, interval, omega, counter, level + 1, None, adaptive)
        Z[j] = np.asarray(z, dtype=dtype)
        w = _f3r_mv_many(A, Z[j], level, dtype)
        for i in range(j + 1):
            H[:, i, j] = _f3r_dot_many(V[i], w, dtype)
        projection = np.zeros((size, n), dtype=dtype)
        for i in range(j + 1):
            projection = np.add(projection, np.multiply(H[:, i, j, None], V[i], dtype=dtype), dtype=dtype)
        w = np.subtract(w, projection, dtype=dtype)
        norm = _f3r_norm_many(w, dtype)
        if np.any(norm == 0) or not np.isfinite(norm).all():
            raise ValueError("Arnoldi breakdown")
        H[:, j + 1, j], V[j + 1] = norm, np.divide(w, norm[:, None], dtype=dtype)
        for i in range(j):
            aa, bb = H[:, i, j].copy(), H[:, i + 1, j].copy()
            H[:, i, j] = np.add(np.multiply(cs[:, i], aa, dtype=dtype), np.multiply(sn[:, i], bb, dtype=dtype), dtype=dtype)
            H[:, i + 1, j] = np.add(np.multiply(-sn[:, i], aa, dtype=dtype), np.multiply(cs[:, i], bb, dtype=dtype), dtype=dtype)
        aa, bb = H[:, j, j].copy(), H[:, j + 1, j].copy()
        rho = np.sqrt(np.add(np.multiply(aa, aa, dtype=dtype), np.multiply(bb, bb, dtype=dtype), dtype=dtype)).astype(dtype, copy=False)
        if np.any(rho == 0) or not np.isfinite(rho).all():
            raise ValueError("Invalid Givens denominator")
        cs[:, j], sn[:, j] = np.divide(aa, rho, dtype=dtype), np.divide(bb, rho, dtype=dtype)
        H[:, j, j], H[:, j + 1, j] = rho, dtype(0)
        aa, bb = g[:, j].copy(), g[:, j + 1].copy()
        g[:, j] = np.add(np.multiply(cs[:, j], aa, dtype=dtype), np.multiply(sn[:, j], bb, dtype=dtype), dtype=dtype)
        g[:, j + 1] = np.add(np.multiply(-sn[:, j], aa, dtype=dtype), np.multiply(cs[:, j], bb, dtype=dtype), dtype=dtype)
    y = np.zeros((size, m), dtype=dtype)
    for i in range(m - 1, -1, -1):
        if np.any(H[:, i, i] == 0):
            raise ValueError("Zero reduced triangular pivot")
        total = np.zeros(size, dtype=dtype)
        for j in range(i + 1, m):
            total = np.add(total, np.multiply(H[:, i, j], y[:, j], dtype=dtype), dtype=dtype)
        y[:, i] = np.divide(np.subtract(g[:, i], total, dtype=dtype), H[:, i, i], dtype=dtype)
    correction = np.zeros((size, n), dtype=dtype)
    for j in range(m):
        correction = np.add(correction, np.multiply(Z[j], y[:, j, None], dtype=dtype), dtype=dtype)
    return _f3r_finite(np.add(x, correction, dtype=dtype)), omega, counter


def _oracle_nested_fgmres_cycle(A: "np.ndarray", rhs: "np.ndarray", lower: "np.ndarray", upper: "np.ndarray",
                                iterations: tuple, interval: int, weights: "np.ndarray", counter: int,
                                level: int = 0, initial: "np.ndarray | None" = None, adaptive: bool = True) -> tuple:
    lower, upper = _f3r_factor_arrays(lower, upper)
    n = lower.size
    A = _f3r_matrix(A, n)
    if not isinstance(iterations, (tuple, list)) or len(iterations) != 3:
        raise ValueError("iterations must contain three integers")
    ms = tuple(_f3r_integer(m, 'iteration count') for m in iterations)
    if any(m > n for m in ms):
        raise ValueError("Iteration counts cannot exceed n")
    level = _f3r_integer(level, 'level', 0)
    if level not in (0, 1, 2):
        raise ValueError("level must be 0,1,2")
    omega, counter, interval = _f3r_state(weights, counter, interval, adaptive)
    dtype = np.float64 if level == 0 else np.float32
    b = _f3r_vector(rhs, n, dtype, 'rhs')
    x = None if initial is None else _f3r_vector(initial, n, dtype, 'initial')[None, :]
    x, omega, counter = _f3r_gold_nested_fgmres_many(
        A, b[None, :], lower[None, :, :], upper[None, :, :], ms, interval,
        omega[None, :], counter, level, x, adaptive)
    return x[0], omega[0], counter

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and declared-invalid test specifications."""
    return [
        # normal: complete nested outer cycle
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
                  'weights=np.array([1.0, 1.0], dtype=np.float16).reshape((2,))\n'
                  'counter=1\n'
                  'expected=(np.array([20.630642678381985, 21.005114804102014, 21.462937859212943, '
                  '21.42768809987234, 20.706303088626917, 21.44114860430254, 21.989732765990478, 21.944263569293934, '
                  '21.05570522641226, 21.502146461606, 21.544920611158044, 20.576289725639914, 21.022775677537712, '
                  '21.613954504494217, 21.583661112327373, 20.92473456690218, 21.382279970794638, '
                  '21.906627287938225, 21.85516038962556, 21.078041315932083, 21.501857406866097, '
                  '21.614037617204193, 20.57435561993304, 21.015613237456854, 21.512922813999463, 21.53133860143212, '
                  '20.852933623751444, 21.42057540377115, 21.925965037993066, 21.913856446060855, '
                  '21.058024885285857, 21.45833880492367, 21.440055009543016, 20.461715910911863, '
                  '20.910677454528162, 21.529093666045792, 21.581099226233835, 20.991685794370994, '
                  '21.478082117609194, 21.928905738240186, 21.66480700907204, 20.802397812673583, 21.23482168789637, '
                  '21.39424440232752, 20.500231625292884, 21.153400713197797, 21.826918521378687, '
                  '21.860861194726215], dtype=np.float64).reshape((48,)), np.array([1.048828125, 1.1142578125], '
                  'dtype=np.float16).reshape((2,)), 25)\n'
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
         'call': '_case_check(nested_fgmres_cycle(A,B[2],lower,upper,(2,4,3),5,weights,counter))',
         'gold_call': '_case_check(_oracle_nested_fgmres_cycle(A,B[2],lower,upper,(2,4,3),5,weights,counter))'},
        # edge: continued outer restart
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
                  'weights=np.array([1.048828125, 1.1142578125], dtype=np.float16).reshape((2,))\n'
                  'counter=25\n'
                  'initial=np.array([20.630642678381985, 21.005114804102014, 21.462937859212943, 21.42768809987234, '
                  '20.706303088626917, 21.44114860430254, 21.989732765990478, 21.944263569293934, 21.05570522641226, '
                  '21.502146461606, 21.544920611158044, 20.576289725639914, 21.022775677537712, 21.613954504494217, '
                  '21.583661112327373, 20.92473456690218, 21.382279970794638, 21.906627287938225, 21.85516038962556, '
                  '21.078041315932083, 21.501857406866097, 21.614037617204193, 20.57435561993304, '
                  '21.015613237456854, 21.512922813999463, 21.53133860143212, 20.852933623751444, 21.42057540377115, '
                  '21.925965037993066, 21.913856446060855, 21.058024885285857, 21.45833880492367, '
                  '21.440055009543016, 20.461715910911863, 20.910677454528162, 21.529093666045792, '
                  '21.581099226233835, 20.991685794370994, 21.478082117609194, 21.928905738240186, '
                  '21.66480700907204, 20.802397812673583, 21.23482168789637, 21.39424440232752, 20.500231625292884, '
                  '21.153400713197797, 21.826918521378687, 21.860861194726215], dtype=np.float64).reshape((48,))\n'
                  'expected=(np.array([20.63905302255082, 21.01352072970473, 21.471360152842223, 21.43613539611702, '
                  '20.714777921617195, 21.44962941427042, 21.998212304096143, 21.95271842025222, 21.064134398108905, '
                  '21.51055980687134, 21.55335015015638, 20.584752837549896, 21.03127385702855, 21.62247159317744, '
                  '21.592166676299954, 20.933208692740404, 21.390703949499663, 21.91502171610756, '
                  '21.863579598317564, 21.086508668873627, 21.51035493998129, 21.622555984016397, '
                  '20.582849949102847, 21.02406124015891, 21.5213327883374, 21.539718884880166, 20.861338035329048, '
                  '21.429037041873606, 21.934446959289577, 21.92235029140491, 21.066493169795095, '
                  '21.466764692961245, 21.448442138354718, 20.470093296291225, 20.919075555037562, '
                  '21.53753489617051, 21.589560324647984, 21.000150559107734, 21.486519535089524, '
                  '21.937311829594464, 21.67320260462886, 20.81079025733848, 21.24323475217115, 21.402682136522028, '
                  '20.50868671835768, 21.16186358734169, 21.835371725785862, 21.869293923926417], '
                  'dtype=np.float64).reshape((48,)), np.array([1.0537109375, 1.1591796875], '
                  'dtype=np.float16).reshape((2,)), 49)\n'
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
         'call': '_case_check(nested_fgmres_cycle(A,B[2],lower,upper,(2,4,3),5,weights,counter,0,initial))',
         'gold_call': '_case_check(_oracle_nested_fgmres_cycle(A,B[2],lower,upper,(2,4,3),5,weights,counter,0,initial))'},
        # boundary: single third-level iteration
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
                  'weights=np.array([1.0, 1.0], dtype=np.float16).reshape((2,))\n'
                  'counter=1\n'
                  'expected=(np.array([1.7580511569976807, 1.7513190507888794, 1.8138318061828613, '
                  '1.3079593181610107, 1.8734593391418457, 1.8657654523849487, 1.8676890134811401, '
                  '1.2406378984451294, 1.7686302661895752, 1.785941481590271, 1.8590333461761475, '
                  '1.3425817489624023, 1.875382900238037, 1.8253726959228516, 1.767668604850769, 1.244484782218933, '
                  '1.8138318061828613, 1.8349900245666504, 1.901349663734436, 1.3637399673461914, '
                  '1.8349900245666504, 1.7272757291793823, 1.775362491607666, 1.2733368873596191, 1.854224681854248, '
                  '1.8686506748199463, 1.9100053310394287, 1.3329644203186035, 1.754204273223877, '
                  '1.7407399415969849, 1.8138318061828613, 1.3098827600479126, 1.877306342124939, '
                  '1.8715358972549438, 1.8724976778030396, 1.243523120880127, 1.770553708076477, 1.7869032621383667, '
                  '1.8580716848373413, 1.3425817489624023, 1.8580716848373413, 1.8272961378097534, '
                  '1.7772859334945679, 1.2492934465408325, 1.814793586730957, 1.8282577991485596, 1.875382900238037, '
                  '1.3079593181610107], dtype=np.float32).reshape((48,)), np.array([1.0, 1.0], '
                  'dtype=np.float16).reshape((2,)), 2)\n'
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
         'call': '_case_check(nested_fgmres_cycle(A,B[0],lower,upper,(1,1,1),5,weights,counter,2))',
         'gold_call': '_case_check(_oracle_nested_fgmres_cycle(A,B[0],lower,upper,(1,1,1),5,weights,counter,2))'},
        # normal: middle-level inherited state
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
                  'weights=np.array([1.125, 1.25], dtype=np.float16).reshape((2,))\n'
                  'counter=49\n'
                  'expected=(np.array([1.9004337787628174, 0.2744016945362091, 2.089944839477539, '
                  '0.3706052601337433, 2.375558614730835, 0.525653064250946, 2.126788377761841, 0.31259045004844666, '
                  '2.049190044403076, 0.08267826586961746, 1.9135172367095947, 0.2841133773326874, '
                  '2.1055731773376465, 0.35229820013046265, 2.316304922103882, 0.4915144443511963, '
                  '2.1294424533843994, 0.31180036067962646, 2.055900812149048, 0.0822875052690506, '
                  '1.9046019315719604, 0.2666774094104767, 2.0887367725372314, 0.353543221950531, 2.336190938949585, '
                  '0.5154308080673218, 2.1442344188690186, 0.3132142424583435, 2.0442326068878174, '
                  '0.06074545532464981, 1.8864121437072754, 0.26952892541885376, 2.1197690963745117, '
                  '0.40637537837028503, 2.3745851516723633, 0.5289776921272278, 2.140017509460449, '
                  '0.27747026085853577, 1.9888250827789307, 0.031606532633304596, 1.909305453300476, '
                  '0.37747466564178467, 2.2045047283172607, 0.43766751885414124, 2.408626079559326, '
                  '0.49551090598106384, 2.0059938430786133, 0.163178950548172], dtype=np.float32).reshape((48,)), '
                  'np.array([1.056640625, 1.2255859375], dtype=np.float16).reshape((2,)), 61)\n'
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
         'call': '_case_check(nested_fgmres_cycle(A,B[1],lower,upper,(2,4,3),5,weights,counter,1))',
         'gold_call': '_case_check(_oracle_nested_fgmres_cycle(A,B[1],lower,upper,(2,4,3),5,weights,counter,1))'},
        # invalid: zero starting residual
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
                  'weights=np.array([1.0, 1.0], dtype=np.float16).reshape((2,))\n'
                  'counter=1\n'
                  'def _case_raise(fn):\n'
                  '    try:\n'
                  '        fn()\n'
                  '        return 0\n'
                  '    except ValueError:\n'
                  '        return 1\n'
                  '    except Exception:\n'
                  '        return 2',
         'call': '_case_raise(lambda: nested_fgmres_cycle(A,np.zeros(48),lower,upper,(2,4,3),5,weights,counter))',
         'gold_call': '_case_raise(lambda: '
                      '_oracle_nested_fgmres_cycle(A,np.zeros(48),lower,upper,(2,4,3),5,weights,counter))'},
    ]
