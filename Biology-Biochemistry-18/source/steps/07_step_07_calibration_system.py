"""
Construct the joint calibration half-spaces with shared intrinsic factors and condition-specific concentrations.

Activity is condition-specific. Each active reaction contributes a driving-force constraint and two absolute-mismatch inequalities. The same intrinsic factors couple all conditions of one design.

Returns
-------
A finite ndarray with P*M+K+2 columns, including the right-hand side.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def calibration_system(operator: "np.ndarray", net: "np.ndarray",
                       energy: "np.ndarray", assay: "np.ndarray",
                       intervals: "np.ndarray", x_bounds: "np.ndarray",
                       activity: float, drive: float) -> "np.ndarray":
    """
    operator has shape (P,N,1+M+K), packed as step 4; net and energy
    are (P,N) for one design. assay is (H,M); intervals is (P,H,2);
    x_bounds is (P,M,2), with ordered lower/upper bounds. activity is
    nonnegative and drive positive. Return augmented half-spaces [D|d]
    for D*y<=d, variables y=(x_0,...,x_(P-1),m,delta); each x has M
    entries and the single shared m has K. Columns number P*M+K+2.
    For each condition in order, visit active reactions in ascending
    index: driving-force row, upper mismatch row, lower mismatch row;
    then assay upper/lower pairs in supplied order; then concentration
    upper/lower pairs in reactant order. Append -delta<=0 last. Only
    abs(net)>activity is active; equality is inactive. Do not encode
    the intrinsic uncertainty ball here. H and K may be zero. Invalid
    alignment, nonfinite data, reversed bounds or invalid thresholds
    raise ValueError. This explicit row order is the numerical contract.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_calibration_system(operator: "np.ndarray", net: "np.ndarray",
                       energy: "np.ndarray", assay: "np.ndarray",
                       intervals: "np.ndarray", x_bounds: "np.ndarray",
                       activity: float, drive: float) -> "np.ndarray":
    import numpy as np
    op, n, est, h, bands, xb = map(lambda x: np.asarray(x, dtype=float),
                                  (operator, net, energy, assay, intervals, x_bounds))
    if (op.ndim != 3 or n.shape != op.shape[:2] or est.shape != n.shape
            or h.ndim != 2 or op.shape[2] < 1+h.shape[1]
            or bands.shape != (n.shape[0], h.shape[0], 2)
            or xb.shape != (n.shape[0], h.shape[1], 2)
            or np.any(bands[..., 0] > bands[..., 1]) or np.any(xb[..., 0] > xb[..., 1])
            or any(not np.isfinite(x).all() for x in (op, n, est, h, bands, xb))
            or not np.isfinite(activity) or activity < 0
            or not np.isfinite(drive) or drive <= 0):
        raise ValueError('Aligned finite calibration data and ordered bounds required')
    p_count, r_count = n.shape
    m, k = h.shape[1], op.shape[2]-1-h.shape[1]
    width = p_count*m+k+1
    rows = []
    def _append(coeff, bound):
        rows.append(np.r_[coeff, bound])
    for p in range(p_count):
        for j in range(r_count):
            if abs(n[p, j]) <= activity:
                continue
            row = np.zeros(width)
            row[p*m:(p+1)*m] = op[p, j, 1:1+m]
            row[p_count*m:p_count*m+k] = op[p, j, 1+m:]
            sign, offset = np.sign(n[p, j]), op[p, j, 0]
            _append(sign*row, -drive-sign*offset)
            upper, lower = row.copy(), -row
            upper[-1] = lower[-1] = -1.
            _append(upper, est[p, j]-offset)
            _append(lower, offset-est[p, j])
        for row_id in range(h.shape[0]):
            row = np.zeros(width)
            row[p*m:(p+1)*m] = h[row_id]
            _append(row, bands[p, row_id, 1])
            _append(-row, -bands[p, row_id, 0])
        for i in range(m):
            row = np.zeros(width)
            row[p*m+i] = 1.
            _append(row, xb[p, i, 1])
            _append(-row, -xb[p, i, 0])
    row = np.zeros(width)
    row[-1] = -1.
    _append(row, 0.)
    return np.asarray(rows)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'op=np.array([[[-1.,1.,.4],[-2.,-1.,-.4]],[[-.5,1.,.4],[-1.5,-1.,-.4]]])\n'
               'n=np.array([[2.,1.],[1.,-1.]]); e=np.array([[-2.,-1.],[-1.,1.]])\n'
               'h=np.array([[1.]]); bands=np.array([[[-1.,1.]],[[-.5,.5]]])\n'
               'xb=np.tile([-2.,2.],(2,1,1))\n',
      'call': 'calibration_system(op,n,e,h,bands,xb,1e-7,.001)',
      'gold_call': '_oracle_calibration_system(op,n,e,h,bands,xb,1e-7,.001)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'op=np.array([[[-1.,1.,.4],[-2.,-1.,-.4]],[[-.5,1.,.4],[-1.5,-1.,-.4]]])\n'
               'n=np.array([[2.,1.],[1.,-1.]]); e=np.array([[-2.,-1.],[-1.,1.]])\n'
               'h=np.array([[1.]]); bands=np.array([[[-1.,1.]],[[-.5,.5]]])\n'
               'xb=np.tile([-2.,2.],(2,1,1))\n'
               'n=np.array([[1e-7,0.],[-1e-7,0.]])',
      'call': 'calibration_system(op,n,e,h,bands,xb,1e-7,.001)',
      'gold_call': '_oracle_calibration_system(op,n,e,h,bands,xb,1e-7,.001)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'op=np.array([[[-1.,1.,.4],[-2.,-1.,-.4]],[[-.5,1.,.4],[-1.5,-1.,-.4]]])\n'
               'n=np.array([[2.,1.],[1.,-1.]]); e=np.array([[-2.,-1.],[-1.,1.]])\n'
               'h=np.array([[1.]]); bands=np.array([[[-1.,1.]],[[-.5,.5]]])\n'
               'xb=np.tile([-2.,2.],(2,1,1))\n'
               'h=np.empty((0,1)); bands=np.empty((2,0,2)); op=op[:,:,:2]',
      'call': 'calibration_system(op,n,e,h,bands,xb,1e-7,.001)',
      'gold_call': '_oracle_calibration_system(op,n,e,h,bands,xb,1e-7,.001)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'op=np.array([[[-1.,1.,.4],[-2.,-1.,-.4]],[[-.5,1.,.4],[-1.5,-1.,-.4]]])\n'
               'n=np.array([[2.,1.],[1.,-1.]]); e=np.array([[-2.,-1.],[-1.,1.]])\n'
               'h=np.array([[1.]]); bands=np.array([[[-1.,1.]],[[-.5,.5]]])\n'
               'xb=np.tile([-2.,2.],(2,1,1))\n'
               'bands[0,0]=[2.,1.]\n'
               '\n'
               'def _raises_value_error(fn, arguments):\n'
               '    try:\n'
               '        fn(*arguments)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_raises_value_error(calibration_system, (op,n,e,h,bands,xb,1e-7,.001,))',
      'gold_call': '_raises_value_error(_oracle_calibration_system, '
                   '(op,n,e,h,bands,xb,1e-7,.001,))',
      'tol': 0.0}]
